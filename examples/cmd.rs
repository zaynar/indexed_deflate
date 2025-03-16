use core::str;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    let index_path = cli.input.with_extension({
        let mut ext = cli.input.extension().unwrap().to_owned();
        ext.push(".index");
        ext
    });

    if cli.build {
        println!("Building index from {}", cli.input.to_string_lossy());

        let tar_gz = File::open(&cli.input)?;
        let mut index = File::create(&index_path)?;

        let mut builder = GzIndexBuilder::new(tar_gz, &index, AccessPointSpan::default())?;

        let mut archive = tar::Archive::new(&mut builder);

        let files: HashMap<String, (u64, u64)> = archive
            .entries()?
            .map(|file| {
                let file = file.unwrap();

                let path = str::from_utf8(&file.path_bytes())
                    .expect("Paths must be valid UTF-8")
                    .to_owned();
                (path, (file.raw_file_position(), file.size()))
            })
            .collect();

        builder.finish()?;

        let files_buf = postcard::to_stdvec(&files).unwrap();
        index.write_u32::<LittleEndian>(files_buf.len() as u32)?;
        index.write_all(&files_buf)?;
    }

    if let Some(extract) = cli.extract {
        let tar_gz = File::open(&cli.input)?;
        let index = File::open(&index_path)?;

        let mut stream = GzDecoder::new(tar_gz, index)?;

        let files: HashMap<String, (u64, u64)> = stream.with_index(|index| {
            let len = index.read_u32::<LittleEndian>()?;
            let mut buf = vec![0; len as usize];
            index.read_exact(&mut buf)?;
            Ok(postcard::from_bytes(&buf).unwrap())
        })?;

        let (file_pos, file_size) = files.get(&extract).unwrap();

        stream.seek(SeekFrom::Start(*file_pos))?;
        let mut buf = vec![0; *file_size as usize];
        stream.read_exact(&mut buf)?;

        if let Some(output) = cli.output {
            println!("Writing {} to {}", extract, output.to_string_lossy());
            let mut output = File::create(output)?;
            output.write_all(&buf)?;
        } else {
            std::io::stdout().write_all(&buf)?;
        }
    }

    Ok(())
}
