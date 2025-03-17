# `indexed_deflate`

Gzip decoder with efficient* support for random access with `Seek`.

As gzip does not normally support random access, we build an index while decompressing the
entire input. This contains a set of access points, typically one per 1MB of input.
We can restart decompression from any access point, letting us seek to any byte for the
cost of decompressing at most 1MB of discarded data (a few milliseconds on a desktop CPU).

The index can be saved to disk and reused for any subsequent processing of the same file.

Each access point takes up to 32KB of storage (less if the compression ratio is good).
If we have one access point for every 1MB of input, the index will be up to 3% of
the size of the input file.

When building or using an index file, only a small map of file offsets must be stored
in RAM. The bulky decompressor state will be kept on disk. This allows indexes to be larger
than RAM, and minimises the startup cost when a process only wants to use a small part of the
index.

(If you prefer to keep the index in memory, you can use a `Cursor<Vec<u8>>` instead of a `File`.)

This also allows a multi-threaded application to create multiple readers over the same file,
allowing parallel decompression of different sections of the file, with only a small RAM cost.

# Usage example

See `examples/cmd.rs` in the source code.

## Building the index

```rust
fn build_index() -> Result<()> {
    let mut gz = File::open("example.tar.gz")?;
    let mut index = File::create("example.tar.gz.index")?;

    let mut builder = GzIndexBuilder::new(gz, &index)?;

    // You must read the whole file through the builder to trigger creation of
    // the index file. You can immediately discard the data, or use it to perform
    // any other preprocessing you need.
    //
    // Here we use it to extract the file listing from a .tar.gz
    let mut archive = tar::Archive::new(&mut builder);
    let files: HashMap<String, (u64, u64)> =
        archive.entries()?.map(|file| {
            let file = file.unwrap();
            let path = str::from_utf8(&file.path_bytes()).unwrap().to_owned();
            (path, (file.raw_file_position(), file.size()))
        }).collect();

    // Finish writing the index to disk
    builder.finish()?;

    // Append our serialized file listing to the index file
    let files_buf = postcard::to_stdvec(&files).unwrap();
    index.write_u32::<LittleEndian>(files_buf.len() as u32)?;
    index.write_all(&files_buf)?;
}
```

## Using the index

```rust
fn use_index() -> Result<()> {
    let mut gz = File::open("example.tar.gz")?;
    let mut index = File::open("example.tar.gz.index")?;

    let mut stream = GzDecoder::new(gz, index)?;

    // Load the tar file listing from the end of the index file
    let files: HashMap<String, (u64, u64)> =
        stream.with_index(|index| {
            let len = index.read_u32::<LittleEndian>()?;
            let mut buf = vec![0; len as usize];
            index.read_exact(&mut buf)?;
            Ok(postcard::from_bytes(&buf).unwrap())
        })?;

    let (file_pos, file_size) = files.get("example.txt").unwrap();

    // Seek in the decompressed stream to read the file
    stream.seek(SeekFrom::Start(*file_pos))?;
    let mut buf = vec![0; file_size];
    stream.read_exact(&mut buf)?;
    println!("{}", str::from_utf8(&buf).unwrap());
}
```