# `indexed_deflate`

Gzip/Zlib/DEFLATE decoder with efficient random access.

As DEFLATE does not normally support random access, we build an index while decompressing the
entire input. This contains a set of access points, typically one per 1MB of input.
We can restart decompression from any access point, letting us seek to any byte for the
cost of decompressing at most 1MB of discarded data (a few milliseconds on a desktop CPU).

The index is saved to disk and can be reused for any subsequent processing of the same file.

Decompression is implemented with the pure-Rust [`miniz_oxide`](https://crates.io/crates/miniz_oxide).

# Memory

With the default configuration, the index file will be up to 3% of the size of the input file.
Only a small map of file offsets is stored in RAM, roughly 0.003% of the size of the input file.
This minimises the startup cost when a process only wants to use a small part of the index.

# Usage

An example implementing random access to `.tar.gz` files:

```rust
use std::{collections::HashMap, fs::File, io::{Read, Seek, SeekFrom, Write}, str};
use indexed_deflate::{AccessPointSpan, GzDecoder, GzIndexBuilder, Result};

fn build_tar_index() -> Result<()> {
    let gz = File::open("example.tar.gz")?;
    let mut index = File::create("example.tar.gz.index")?;

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

fn use_tar_index() -> Result<()> {
    let gz = File::open("example.tar.gz")?;
    let index = File::open("example.tar.gz.index")?;

    // GzDecoder supports Read and Seek
    let mut stream = GzDecoder::new(gz, index)?;

    // Load the tar file listing from the end of the index file
    let files: HashMap<String, (u64, u64)> = stream.with_index(|index| {
        let mut buf = Vec::new();
        index.read_to_end(&mut buf)?;
        Ok(postcard::from_bytes(&buf).unwrap())
    })?;

    let (file_pos, file_size) = files.get("example.txt").unwrap();

    // Seek in the decompressed stream to read the file
    stream.seek(SeekFrom::Start(*file_pos))?;
    let mut buf = vec![0; *file_size as usize];
    stream.read_exact(&mut buf)?;

    println!("{}", str::from_utf8(&buf).unwrap());

    Ok(())
}
```
