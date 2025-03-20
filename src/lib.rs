//! Gzip/Zlib/DEFLATE decoder with efficient random access.
//!
//! As DEFLATE does not normally support random access, we build an index while decompressing the
//! entire input. This contains a set of access points, typically one per 1MB of input.
//! We can restart decompression from any access point, letting us seek to any byte for the
//! cost of decompressing at most 1MB of discarded data (a few milliseconds on a desktop CPU).
//!
//! The index is saved to disk and can be reused for any subsequent processing of the same file.
//!
//! Decompression is implemented with the pure-Rust [`miniz_oxide`](https://crates.io/crates/miniz_oxide).
//!
//! # Memory
//!
//! Each access point takes up to 32KB of storage (less if the compression ratio is good).
//! If we have one access point for every 1MB of input, the index will be up to 3% of
//! the size of the input file.
//!
//! For both building and using an index file, only a small map of file offsets (about 32 bytes per
//! access point, or 0.003% the size of the input file) is stored in RAM.
//! The bulky decompressor state will be kept on disk. This allows indexes to be larger
//! than RAM, and minimises the startup cost when a process only wants to use a small part of the
//! index.
//!
//! This also allows a multi-threaded application to create multiple readers over the same file,
//! allowing parallel decompression of different sections of the file, with only a small RAM cost.
//!
//! # Backward compatibility
//!
//! Currently there are no compatibility guarantees for the index file format.
//! If the format changes incompatibly, the decoder will reject it with
//! [`Error::IndexIncompatibleVersion`] and you must rebuild the index.
//!
//! # DEFLATE block sizes
//!
//! To avoid having to store the entire decompressor state in the index file (including Huffman
//! trees etc), we only create access points at the boundaries between DEFLATE blocks.
//! If the blocks are large relative to the requested `AccessPointSpan` (default 1MB), this may
//! significantly increase the spacing of access points.
//!
//! Experiments indicate that GNU Gzip (the standard command-line `gzip`) and
//! `miniz_oxide` have a maximum block size of roughly 64KB compressed, so they should not
//! be a problem.
//! (Uncompressed blocks may be several MB, but access points are based on compressed size.)
//!
//! `libdeflate` has a maximum block size of roughly 300KB uncompressed
//! (under its default configuration).
//! `zopfli` has a maximum block size of roughly 1MB uncompressed (default).
//! That will make our index less efficient; but these implementations are explicitly not designed for
//! compressing very large files, so you are less likely to encounter them in this context.
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```
//! # use std::{fs::File, io::{Read, Seek, SeekFrom, Write}};
//! # use indexed_deflate::{AccessPointSpan, GzDecoder, GzIndexBuilder, Result};
//! #
//! fn build_index() -> Result<()> {
//!     let gz = File::open("example.gz")?;
//!
//!     // If you seek backwards while building, the index must be opened in read-write mode.
//!     // Otherwise you can use the write-only `File::create()` instead.
//!     let index = File::options()
//!         .create(true)
//!         .truncate(true)
//!         .read(true)
//!         .write(true)
//!         .open("example.gz.index")?;
//!
//!     let mut builder = GzIndexBuilder::new(gz, index, AccessPointSpan::default())?;
//!
//!     // Trigger decompression of the entire file. This may take a long time
//!     builder.seek(SeekFrom::End(0))?;
//!
//!     // Finish writing the index to disk
//!     builder.finish()?;
//!
//!     Ok(())
//! }
//!
//! fn use_index() -> Result<()> {
//!     let gz = File::open("example.gz")?;
//!     let index = File::open("example.gz.index")?;
//!
//!     let mut decoder = GzDecoder::new(gz, index)?;
//!
//!     // This seek should only take a few milliseconds
//!     decoder.seek(SeekFrom::Start(100_000_000))?;
//!
//!     let mut buf = vec![0u8; 1024];
//!     decoder.read_exact(&mut buf)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## .tar.gz random access
//!
//! ```
//! # use std::{collections::HashMap, fs::File, io::{Read, Seek, SeekFrom, Write}, str};
//! # use indexed_deflate::{AccessPointSpan, GzDecoder, GzIndexBuilder, Result};
//! #
//! fn build_tar_index() -> Result<()> {
//!     let gz = File::open("example.tar.gz")?;
//!     let mut index = File::create("example.tar.gz.index")?;
//!
//!     // GzIndexBuilder supports Read and Seek
//!     let mut builder = GzIndexBuilder::new(gz, &index, AccessPointSpan::default())?;
//!
//!     // Extract the tar file listing, while decompressing
//!     let mut archive = tar::Archive::new(&mut builder);
//!     let files: HashMap<String, (u64, u64)> = archive
//!         .entries_with_seek()?
//!         .map(|file| {
//!             let file = file.unwrap();
//!             let path = str::from_utf8(&file.path_bytes()).unwrap().to_owned();
//!             (path, (file.raw_file_position(), file.size()))
//!         })
//!         .collect();
//!
//!     // Finish writing the index to disk
//!     builder.finish()?;
//!
//!     // Append our serialized file listing to the index file
//!     index.write_all(&postcard::to_stdvec(&files).unwrap())?;
//!
//!     Ok(())
//! }
//!
//! fn use_tar_index() -> Result<()> {
//!     let gz = File::open("example.tar.gz")?;
//!     let index = File::open("example.tar.gz.index")?;
//!
//!     // GzDecoder supports Read and Seek
//!     let mut stream = GzDecoder::new(gz, index)?;
//!
//!     // Load the tar file listing from the end of the index file
//!     let files: HashMap<String, (u64, u64)> = stream.with_index(|index| {
//!         let mut buf = Vec::new();
//!         index.read_to_end(&mut buf)?;
//!         Ok(postcard::from_bytes(&buf).unwrap())
//!     })?;
//!
//!     let (file_pos, file_size) = files.get("example.txt").unwrap();
//!
//!     // Seek in the decompressed stream to read the file
//!     stream.seek(SeekFrom::Start(*file_pos))?;
//!     let mut buf = vec![0; *file_size as usize];
//!     stream.read_exact(&mut buf)?;
//!
//!     println!("{}", str::from_utf8(&buf).unwrap());
//!
//!     Ok(())
//! }
//! ```

use std::io::{Read, Seek, Write};

use base::{BaseDecoder, BaseIndexBuilder, ReadDecoder, SeekDecoder, Wrapper};

mod base;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("index file did not start with magic string")]
    IndexBadMagic,
    #[error("index file was not closed with `finish()`")]
    IndexUnfinished,
    #[error("index file version is incompatible")]
    IndexIncompatibleVersion,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Approximate number of bytes of compressed input between access points.
/// Larger spans will result in smaller indexes, but slower seek times.
#[derive(Copy, Clone)]
pub struct AccessPointSpan(u64);

impl AccessPointSpan {
    pub const fn new(span_bytes: u64) -> Self {
        Self(span_bytes)
    }
}

impl Default for AccessPointSpan {
    fn default() -> Self {
        Self(1024 * 1024)
    }
}

macro_rules! create_interface {
    ($decoder:ident, $builder:ident, $wrapper:expr) => {
        /// Decompresses the input file, using the previously-built index to allow fast seeking.
        pub struct $decoder<C, I> {
            base: BaseDecoder<C, I>,
        }

        impl<C, I> $decoder<C, I>
        where
            // C:Seek is not technically required for these methods, but there's no point using
            // this library without it, so require it here as a kind of documentation
            C: Read + Seek,
            I: Read + Seek,
        {
            /// Creates a new decoder, and parses the headers from both streams.
            ///
            /// # Errors
            ///
            /// - [`Error::IndexBadMagic`] if `index` is not recognized as an index file.
            /// - [`Error::IndexUnfinished`] if you failed to call `finish()` when building the index.
            /// - [`Error::IndexIncompatibleVersion`] if `index` is not compatible with this
            ///   version of the crate.
            ///
            /// In any of the above cases, you should rebuild the index and try again.
            ///
            /// - [`Error::Io`] on IO error.
            pub fn new(compressed: C, index: I) -> Result<Self> {
                Ok(Self {
                    base: BaseDecoder::new(compressed, index, $wrapper)?,
                })
            }

            /// Runs a callback with the index stream. The stream cursor will initially be at
            /// the end of the index data.
            ///
            /// You can append custom data to the end of the index file after calling
            /// the index builder's `finish()`, and then read it back using this function.
            /// E.g. when handling a `.tar.gz` file, you can construct a map of file names/offsets/sizes
            /// and store it at the end of the index, without the hassle of maintaining a second
            /// metadata file.
            pub fn with_index<F, T>(&mut self, f: F) -> std::io::Result<T>
            where
                F: FnOnce(&mut I) -> std::io::Result<T>,
            {
                self.base.with_index(f)
            }
        }

        impl<C, I> Read for $decoder<C, I>
        where
            C: Read,
            I: Read + Seek,
        {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.base.read(buf)
            }
        }

        impl<C, I> Seek for $decoder<C, I>
        where
            C: Read + Seek,
            I: Read + Seek,
        {
            fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
                self.base.seek(pos)
            }
        }

        /// Decompresses the input file and builds an index to allow fast seeking with the
        /// corresponding `Decoder` type.
        ///
        /// This implements the `Read` trait to return the decompressed data.
        ///
        /// Optionally, this type implements `Seek` if you provide `C: Read + Write` and
        /// `I: Read + Write + Seek`. Seeking to a previously-decompressed location will
        /// use the partially-built index.
        ///
        /// (Note that [`File::create()`](std::fs::File::create) returns a write-only file:
        /// it does implement `Read`, but may fail at runtime when read from.
        /// Use [`File::options()`](std::fs::File::options) instead for the index file,
        /// if you need seeking.)
        ///
        /// The index will only extend up to the furthest byte of `compressed` that was read or
        /// seeked to. If you do not access the whole file, the end will not be indexed.
        /// You can use `seek(SeekFrom::End(0))` to build the index over the entire file.
        pub struct $builder<C, I> {
            base: BaseIndexBuilder<C, I>,
        }

        impl<C, I> $builder<C, I>
        where
            C: Read,
            I: Write + Seek,
        {
            /// Creates a new index builder, which reads `compressed` and writes to `index`.
            pub fn new(compressed: C, index: I, span: AccessPointSpan) -> Result<Self> {
                Ok(Self {
                    base: BaseIndexBuilder::new(compressed, index, span, $wrapper)?,
                })
            }

            /// Finishes writing the index. If you don't call this, the index will be corrupted
            /// and cannot be read by the `Decoder`.
            pub fn finish(self) -> Result<()> {
                self.base.finish()
            }
        }

        impl<C, I> Read for $builder<C, I>
        where
            C: Read,
            I: Write + Seek,
        {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.base.read(buf)
            }
        }

        impl<C, I> Seek for $builder<C, I>
        where
            C: Read + Seek,
            I: Read + Write + Seek,
        {
            fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
                self.base.seek(pos)
            }
        }
    };
}

create_interface!(DeflateDecoder, DeflateIndexBuilder, Wrapper::Deflate);
create_interface!(GzDecoder, GzIndexBuilder, Wrapper::Gzip);
create_interface!(ZlibDecoder, ZlibIndexBuilder, Wrapper::Zlib);

impl<C, I> GzDecoder<C, I>
where
    C: Read,
    I: Read + Seek,
{
    /// Returns the gzip header associated with this stream.
    pub fn header(&self) -> Option<gzip_header::GzHeader> {
        self.base.header()
    }
}

impl<C, I> GzIndexBuilder<C, I>
where
    C: Read,
    I: Write + Seek,
{
    /// Returns the gzip header associated with this stream.
    pub fn header(&self) -> Option<gzip_header::GzHeader> {
        self.base.header()
    }
}
