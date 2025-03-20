use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    str,
    time::Instant,
};

use clap::Parser;
use indexed_deflate::{AccessPointSpan, GzDecoder, GzIndexBuilder, Result};

#[derive(Parser, Debug)]
struct Cli {
    /// Whether to build a new index (will be saved to --gz path with added ".index" suffix)
    #[arg(long)]
    build: bool,

    /// .tar.gz file
    input: std::path::PathBuf,

    /// A file to extract from the tarball
    extract: Option<String>,

    /// Output filename for extracted file
    #[arg(long)]
    output: Option<std::path::PathBuf>,
}

fn build_tar_index(gz_path: &Path, index_path: &Path) -> Result<()> {
    let gz = File::open(gz_path)?;
    let mut index = File::create(index_path)?;

    // GzIndexBuilder supports Read and Seek
    let mut builder = GzIndexBuilder::new(gz, &index, AccessPointSpan::default())?;

    // Extract the tar file listing, while decompressing
    let mut archive = tar::Archive::new(&mut builder);
    let files: HashMap<String, (u64, u64)> = archive
        .entries_with_seek()?
        .map(|file| {
            let file = file.unwrap();
            let path = str::from_utf8(&file.path_bytes()).unwrap().to_owned();
            (path, (file.raw_file_position(), file.size()))
        })
        .collect();

    // Finish writing the index to disk
    builder.finish()?;

    // Append our serialized file listing to the index file
    index.write_all(&postcard::to_stdvec(&files).unwrap())?;

    Ok(())
}

fn use_tar_index(gz_path: &Path, index_path: &Path, target: &str) -> Result<Vec<u8>> {
    let gz = File::open(gz_path)?;
    let index = File::open(index_path)?;

    // GzDecoder supports Read and Seek
    let mut stream = GzDecoder::new(gz, index)?;

    // Load the tar file listing from the end of the index file
    let files: HashMap<String, (u64, u64)> = stream.with_index(|index| {
        let mut buf = Vec::new();
        index.read_to_end(&mut buf)?;
        Ok(postcard::from_bytes(&buf).unwrap())
    })?;

    let (file_pos, file_size) = files.get(target).unwrap();

    // Seek in the decompressed stream to read the file
    stream.seek(SeekFrom::Start(*file_pos))?;
    let mut buf = vec![0; *file_size as usize];
    stream.read_exact(&mut buf)?;

    Ok(buf)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let index_path = cli.input.with_extension({
        let mut ext = cli.input.extension().unwrap().to_owned();
        ext.push(".index");
        ext
    });

    if cli.build {
        println!("Building index from {}", cli.input.to_string_lossy());

        let start = Instant::now();
        build_tar_index(&cli.input, &index_path)?;
        println!("Built index in {:.3} secs", start.elapsed().as_secs_f64());
    }

    if let Some(extract) = cli.extract {
        let start = Instant::now();
        let file = use_tar_index(&cli.input, &index_path, &extract)?;
        println!(
            "Read {} in {:.3} secs",
            extract,
            start.elapsed().as_secs_f64()
        );

        if let Some(output) = cli.output {
            println!("Writing {} to {}", extract, output.to_string_lossy());
            let mut output = File::create(output)?;
            output.write_all(&file)?;
        } else {
            std::io::stdout().write_all(&file)?;
        }
    }

    Ok(())
}
