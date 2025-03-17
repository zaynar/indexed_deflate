//! Gzip decoder with efficient* support for random access with `Seek`.
//!
//! As gzip does not normally support random access, we build an index while decompressing the
//! entire input. This contains a set of access points, typically one per 1MB of input.
//! We can restart decompression from any access point, letting us seek to any byte for the
//! cost of decompressing at most 1MB of discarded data (a few milliseconds on a desktop CPU).
//!
//! The index can be saved to disk and reused for any subsequent processing of the same file.
//!
//! Each access point takes up to 32KB of storage (less if the compression ratio is good).
//! If we have one access point for every 1MB of input, the index will be up to 3% of
//! the size of the input file.
//!
//! When building or using an index file, only a small map of file offsets must be stored
//! in RAM. The bulky decompressor state will be kept on disk. This allows indexes to be larger
//! than RAM, and minimises the startup cost when a process only wants to use a small part of the
//! index.
//!
//! (If you prefer to keep the index in memory, you can use a `Cursor<Vec<u8>>` instead of a `File`.)
//!
//! This also allows a multi-threaded application to create multiple readers over the same file,
//! allowing parallel decompression of different sections of the file, with only a small RAM cost.
//!
//! # Usage example
//!
//! See `examples/cmd.rs` in the source code.
//!
//! ## Building the index
//!
//! ```
//! fn build_index() -> Result<()> {
//!     let mut gz = File::open("example.tar.gz")?;
//!     let mut index = File::create("example.tar.gz.index")?;
//!
//!     let mut builder = GzIndexBuilder::new(gz, &index)?;
//!
//!     // You must read the whole file through the builder to trigger creation of
//!     // the index file. You can immediately discard the data, or use it to perform
//!     // any other preprocessing you need.
//!     //
//!     // Here we use it to extract the file listing from a .tar.gz
//!     let mut archive = tar::Archive::new(&mut builder);
//!     let files: HashMap<String, (u64, u64)> =
//!         archive.entries()?.map(|file| {
//!             let file = file.unwrap();
//!             let path = str::from_utf8(&file.path_bytes()).unwrap().to_owned();
//!             (path, (file.raw_file_position(), file.size()))
//!         }).collect();
//!
//!     // Finish writing the index to disk
//!     builder.finish()?;
//!
//!     // Append our serialized file listing to the index file
//!     let files_buf = postcard::to_stdvec(&files).unwrap();
//!     index.write_u32::<LittleEndian>(files_buf.len() as u32)?;
//!     index.write_all(&files_buf)?;
//! }
//! ```
//!
//! ## Using the index
//!
//! ```
//! fn use_index() -> Result<()> {
//!     let mut gz = File::open("example.tar.gz")?;
//!     let mut index = File::open("example.tar.gz.index")?;
//!
//!     let mut stream = GzDecoder::new(gz, index)?;
//!
//!     // Load the tar file listing from the end of the index file
//!     let files: HashMap<String, (u64, u64)> =
//!         stream.with_index(|index| {
//!             let len = index.read_u32::<LittleEndian>()?;
//!             let mut buf = vec![0; len as usize];
//!             index.read_exact(&mut buf)?;
//!             Ok(postcard::from_bytes(&buf).unwrap())
//!         })?;
//!
//!     let (file_pos, file_size) = files.get("example.txt").unwrap();
//!
//!     // Seek in the decompressed stream to read the file
//!     stream.seek(SeekFrom::Start(*file_pos))?;
//!     let mut buf = vec![0; file_size];
//!     stream.read_exact(&mut buf)?;
//!     println!("{}", str::from_utf8(&buf).unwrap());
//! }
//! ```

use std::io::{Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use miniz_oxide::{
    deflate::CompressionLevel,
    inflate::{
        core::{decompress, inflate_flags, BlockBoundaryState, DecompressorOxide},
        TINFLStatus,
    },
};
use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("postcard serialization/deserialization error")]
    Postcard(#[from] postcard::Error),
    #[error("index file did not start with magic string")]
    IndexBadMagic,
    #[error("index file was not closed with `finish()`")]
    IndexUnfinished,
    #[error("index file version is incompatible")]
    IndexIncompatibleVersion,
}

pub type Result<T> = std::result::Result<T, Error>;

// DEFLATE max window size
const WINDOW_SIZE: u64 = 32768;

// Should be large enough to get reasonably efficient reads from the input file.
const INPUT_BUF_SIZE: usize = 32768;

// Must be at least the window size (32KB), and a power of two. Should be larger than input buffer
// by at least the typical compression ratio, so `decompress()` can process the whole lot in one go.
const OUTPUT_BUF_SIZE: u64 = 65536;

/// Target number of bytes of compressed input between access points.
/// Larger spans will result in smaller indexes, but slower seek times.
#[derive(Copy, Clone)]
pub struct AccessPointSpan(u64);

impl AccessPointSpan {
    pub const fn new(span: u64) -> Self {
        Self(span)
    }
}

impl Default for AccessPointSpan {
    fn default() -> Self {
        Self(1024 * 1024)
    }
}

const HEADER_MAGIC: [u8; 4] = *b"GzIx";

/// Indicates the user started writing the index file but didn't call `finish()`,
/// so it is corrupted
const VERSION_UNFINISHED: u32 = 0xffff_ffff;

const VERSION_LATEST: u32 = 2;

struct Header {
    magic: [u8; 4], // HEADER_MAGIC
    version: u32,   // VERSION_LATEST
    windows_size: u64,
    points_size: u32,
}

// Index file format:
//
// struct {
//   userdata_pre: [u8; ...],
//
//   header: Header,
//   windows: [u8; header.windows_size],
//   points: [u8; header.points_size],
//
//   userdata_post: [u8; ...],
// }
//
// `windows` is a non-delimited sequence of compressed 32KB windows.
// AccessPoint contains a pointer intto this region.
//
// `points` is Vec<AccessPoint> serialized with postcard.
//
// The user can store arbitrary data before and after the index.

/// Index file header
impl Header {
    fn new(windows_size: u64, points_size: u32) -> Self {
        Self {
            magic: HEADER_MAGIC,
            version: VERSION_LATEST,
            windows_size,
            points_size,
        }
    }

    fn unfinished() -> Self {
        Self {
            magic: HEADER_MAGIC,
            version: VERSION_UNFINISHED,
            windows_size: 0,
            points_size: 0,
        }
    }

    fn read<R: Read>(mut r: R) -> std::io::Result<Self> {
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;

        let version = r.read_u32::<LittleEndian>()?;
        let windows_size = r.read_u64::<LittleEndian>()?;
        let points_size = r.read_u32::<LittleEndian>()?;

        Ok(Header {
            magic,
            version,
            windows_size,
            points_size,
        })
    }

    fn write<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;
        w.write_u64::<LittleEndian>(self.windows_size)?;
        w.write_u32::<LittleEndian>(self.points_size)?;
        Ok(())
    }

    fn size() -> u64 {
        20
    }

    fn validate(&self) -> Result<()> {
        if self.magic != HEADER_MAGIC {
            return Err(Error::IndexBadMagic);
        }
        if self.version == VERSION_UNFINISHED {
            return Err(Error::IndexUnfinished);
        }
        if self.version != VERSION_LATEST {
            return Err(Error::IndexIncompatibleVersion);
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct AccessPoint {
    out_pos: u64,
    in_pos: u64,

    /// Offset into the index file's `windows` data
    window_offset: u64,
    window_compressed_size: u16,

    /// Number of bits from the `in_pos` byte, that are part of the new deflate block
    num_bits: u8,
}

/// Shared code for building and reading indexes
struct GzCommon<GzStream> {
    gz_stream: GzStream,
    gz_header: gzip_header::GzHeader,
    done: bool,

    decomp: Box<DecompressorOxide>,

    // Input buffer, and the slice currently containing valid data
    input: Vec<u8>, // INPUT_BUF_SIZE
    input_offset: usize,
    input_size: usize,

    /// Position in file that corresponds to `input[input_offset]`
    input_pos: u64,

    /// Circular buffer for output. Size is `OUTPUT_BUF_SIZE`
    output: Vec<u8>,
    /// Number of bytes written to output buffer (not wrapped to buffer size)
    output_pos: u64,
    /// Number of bytes returned from output to consumer, now available for reuse
    output_ret: u64,
}

// TODO: Separate out the header parts so we can have GzDecoder, ZlibDecoder, DeflateDecoder

impl<GzStream> GzCommon<GzStream>
where
    GzStream: Read,
{
    fn new(gz_stream: GzStream, gz_header: gzip_header::GzHeader) -> Self {
        Self {
            gz_stream,
            gz_header,
            done: false,

            decomp: Box::new(DecompressorOxide::new()),

            input: vec![0; INPUT_BUF_SIZE],
            input_offset: 0,
            input_size: 0,
            input_pos: 0,

            output: vec![0; OUTPUT_BUF_SIZE as usize],
            output_pos: 0,
            output_ret: 0,
        }
    }

    fn has_output(&self) -> bool {
        self.output_pos != self.output_ret
    }

    fn flush_output(&mut self, buf: &mut [u8]) -> usize {
        let copied = (self.output_pos - self.output_ret).min(buf.len() as u64);
        // TODO: get rid of the explicit loop
        for i in 0..copied {
            buf[i as usize] = self.output[((self.output_ret + i) % OUTPUT_BUF_SIZE) as usize];
        }
        self.output_ret += copied;
        copied as usize
    }

    fn make_progress(&mut self, flags: u32) -> std::io::Result<TINFLStatus> {
        // Output buffer must be fully consumed, because we're going to overwrite it
        assert!(!self.has_output());

        // Refill input buffer if empty
        if self.input_offset >= self.input_size {
            self.input_offset = 0;
            self.input_size = self.gz_stream.read(&mut self.input)?;
        }

        let flags = flags
            | if self.input_size > 0 {
                inflate_flags::TINFL_FLAG_HAS_MORE_INPUT
            } else {
                0
            };

        let (status, in_consumed, out_produced) = decompress(
            &mut self.decomp,
            &self.input[self.input_offset..self.input_size],
            &mut self.output,
            (self.output_pos % OUTPUT_BUF_SIZE) as usize,
            flags,
        );

        self.input_offset += in_consumed;
        self.input_pos += in_consumed as u64;
        self.output_pos += out_produced as u64;

        if status == TINFLStatus::Done {
            self.done = true;
        }

        Ok(status)
    }
}

pub struct GzDecoder<GzStream, IndexStream> {
    common: GzCommon<GzStream>,

    index_stream: IndexStream,

    gz_start: u64,

    /// Offset of header in `index_stream`, to support absolute seeking
    header_pos: u64,
    /// Deserialized from index file
    access_points: Vec<AccessPoint>,

    userdata_pos: u64,
}

impl<GzStream, IndexStream> GzDecoder<GzStream, IndexStream>
where
    GzStream: Read + Seek,
    IndexStream: Read + Seek,
{
    pub fn new(mut gz_stream: GzStream, mut index_stream: IndexStream) -> Result<Self> {
        let header_pos = index_stream.stream_position()?;

        let header = Header::read(&mut index_stream)?;
        header.validate()?;

        index_stream.seek(SeekFrom::Start(
            header_pos + Header::size() + header.windows_size,
        ))?;
        let mut points = vec![0; header.points_size as usize];
        index_stream.read_exact(&mut points)?;
        let access_points = postcard::from_bytes(&points)?;
        let userdata_pos = index_stream.stream_position()?;

        // println!("{:?}", access_points);

        let gz_header = gzip_header::read_gz_header(&mut gz_stream)?;
        let gz_start = gz_stream.stream_position()?;

        Ok(Self {
            common: GzCommon::new(gz_stream, gz_header),
            index_stream,
            gz_start,
            header_pos,
            userdata_pos,
            access_points,
        })
    }

    /// Runs a callback with the index stream. The stream cursor will be at the end of the
    /// index data.
    ///
    /// You can append custom data to the end of the index file after calling
    /// `GzIndexBuilder::finish()`, and then read it back using this function.
    /// E.g. when handling a `.tar.gz`, you can store a map of the tar file listing
    /// (name, offset, size) at the end of the index, and use that to implement random access
    /// to named files, without the hassle of maintaining a second metadata file.
    pub fn with_index<F, T>(&mut self, f: F) -> std::io::Result<T>
    where
        F: FnOnce(&mut IndexStream) -> std::io::Result<T>,
    {
        let pos = self.index_stream.stream_position()?;
        self.index_stream.seek(SeekFrom::Start(self.userdata_pos))?;
        let r = f(&mut self.index_stream);
        self.index_stream.seek(SeekFrom::Start(pos))?;
        r
    }

    /// Returns the gzip header associated with this stream
    pub fn header(&self) -> Option<gzip_header::GzHeader> {
        Some(self.common.gz_header.clone())
    }
}

impl<GzStream, IndexStream> GzDecoder<GzStream, IndexStream>
where
    GzStream: Read,
{
    fn make_progress(&mut self) -> std::io::Result<()> {
        let flags = 0;
        let status = self.common.make_progress(flags)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }

    fn skip(&mut self, amount: u64) -> std::io::Result<()> {
        let mut remaining = amount;
        while remaining > 0 {
            // Generate some output
            while !self.common.has_output() && !self.common.done {
                self.make_progress()?;
            }

            let skipped = (self.common.output_pos - self.common.output_ret).min(remaining);
            self.common.output_ret += skipped;
            remaining -= skipped;

            if skipped == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "seek past EOF",
                ));
            }
        }

        Ok(())
    }
}

impl<GzStream, IndexStream> Read for GzDecoder<GzStream, IndexStream>
where
    GzStream: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while !self.common.has_output() && !self.common.done {
            self.make_progress()?;
        }

        Ok(self.common.flush_output(buf))
    }
}

impl<GzStream, IndexStream> Seek for GzDecoder<GzStream, IndexStream>
where
    GzStream: Read + Seek,
    IndexStream: Read + Seek,
{
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        // TODO: refactor this to support seeking in `GzIndexBuilder`, so we can use the
        // partial index when seeking backwards, and keep building the index when
        // seeking forwards. Users can simply call SeekEnd to index the whole thing.

        let target_out_pos = match pos {
            SeekFrom::Start(n) => n,
            SeekFrom::End(_) => {
                // TODO: Support this by decompressing until EOF and then caching the size
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "SeekFrom::End not supported",
                ));
            }
            SeekFrom::Current(n) => (self.common.output_pos as i64 + n) as u64,
        };

        let partition = self
            .access_points
            .partition_point(|c| c.out_pos <= target_out_pos);

        // TODO: if we're already between the access point and target, don't seek, just skip from here

        // TODO: if target is within output buf, don't seek, just update output_ret appropriately

        // TODO: if the start is <32KB before the start of a partition, we can load that
        // partition and then return directly from the window

        if partition == 0 {
            // Jump back to the start of the file

            self.common.gz_stream.seek(SeekFrom::Start(self.gz_start))?;
            self.common.input_offset = 0;
            self.common.input_size = 0;
            self.common.input_pos = 0;
            self.common.output_pos = 0;
            self.common.output_ret = 0;
            *self.common.decomp = DecompressorOxide::new();
            self.common.done = false;

            self.skip(target_out_pos)?;
        } else {
            let point = &self.access_points[partition - 1];

            self.common
                .gz_stream
                .seek(SeekFrom::Start(self.gz_start + point.in_pos))?;
            self.common.input_offset = 0;
            self.common.input_size = 0;
            self.common.input_pos = point.in_pos;
            self.common.output_pos = point.out_pos;
            self.common.output_ret = point.out_pos;
            self.common.done = false;

            // If the block was not byte-aligned, extract the bits from the first byte
            let bit_buf = if point.num_bits != 0 {
                let mut buf = [0];
                self.common.gz_stream.read_exact(&mut buf)?;
                buf[0] >> (8 - point.num_bits)
            } else {
                0
            };

            let decomp_state = BlockBoundaryState {
                num_bits: point.num_bits,
                bit_buf,
                ..Default::default()
            };
            *self.common.decomp = DecompressorOxide::from_block_boundary_state(&decomp_state);

            // Read the compressed window from the index
            let mut window = vec![0; point.window_compressed_size as usize];
            self.index_stream.seek(SeekFrom::Start(
                self.header_pos + Header::size() + point.window_offset,
            ))?;
            self.index_stream.read_exact(&mut window)?;

            let window = miniz_oxide::inflate::decompress_to_vec(&window)
                .map_err(|_| std::io::Error::other("error decompressing index window"))?;

            assert_eq!(window.len(), WINDOW_SIZE as usize);

            // Copy into `output`, just before `output_pos`
            // TODO: get rid of the explicit loop
            for i in 0..WINDOW_SIZE {
                self.common.output[((self.common.output_pos + OUTPUT_BUF_SIZE - WINDOW_SIZE + i)
                    % OUTPUT_BUF_SIZE) as usize] = window[i as usize];
            }

            self.skip(target_out_pos - point.out_pos)?;
        };

        Ok(target_out_pos)
    }
}

pub struct GzIndexBuilder<GzStream, IndexStream> {
    common: GzCommon<GzStream>,

    index_stream: IndexStream,
    span: AccessPointSpan,

    header_pos: u64,
    /// Size of window data in index file
    windows_size: u64,
    access_points: Vec<AccessPoint>,

    last_block_start: u64,
    last_access_point: u64,
}

impl<GzStream, IndexStream> GzIndexBuilder<GzStream, IndexStream>
where
    GzStream: Read,
    IndexStream: Write + Seek,
{
    pub fn new(
        mut gz_stream: GzStream,
        mut index_stream: IndexStream,
        span: AccessPointSpan,
    ) -> Result<Self> {
        let header_pos = index_stream.stream_position()?;

        Header::unfinished().write(&mut index_stream)?;

        let gz_header = gzip_header::read_gz_header(&mut gz_stream)?;

        Ok(Self {
            common: GzCommon::new(gz_stream, gz_header),

            index_stream,
            span,

            header_pos,
            windows_size: 0,
            access_points: Vec::new(),

            last_block_start: 0,
            last_access_point: 0,
        })
    }

    /// Returns the gzip header associated with this stream
    pub fn header(&self) -> Option<gzip_header::GzHeader> {
        Some(self.common.gz_header.clone())
    }

    /// Finishes writing the index. If you don't call this, the index will be corrupted
    /// and cannot be read by `GzDecoder`.
    pub fn finish(mut self) -> Result<()> {
        // Serialize `access_points` onto the end of the index
        // TODO: get rid of the postcard dependency, just serialize it manually and compress
        let points_pos = self.index_stream.stream_position()?;
        postcard::to_io(&self.access_points, &mut self.index_stream)?;

        let end_pos = self.index_stream.stream_position()?;
        let points_size = end_pos - points_pos;

        // Construct the proper header, and overwrite the temporary one at the start of the index
        let header = Header::new(self.windows_size, points_size as u32);
        self.index_stream.seek(SeekFrom::Start(self.header_pos))?;
        header.write(&mut self.index_stream)?;

        // Return to the end, so the user can append their own data
        self.index_stream.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }

    fn create_access_point(&mut self) -> std::io::Result<()> {
        // TODO: If the file was compressed with `gzip --rsyncable`, there is a periodic full flush
        // that means subsequent blocks won't read from the old window. If we could detect that
        // (how?), we could create very cheap access points.

        // Copy the last 32KB from output buffer
        let window: Vec<u8> = (0..WINDOW_SIZE)
            .map(|i| {
                self.common.output[((self.common.output_pos + OUTPUT_BUF_SIZE - WINDOW_SIZE + i)
                    % OUTPUT_BUF_SIZE) as usize]
            })
            .collect();

        let window_compressed =
            miniz_oxide::deflate::compress_to_vec(&window, CompressionLevel::DefaultLevel as u8);

        // Deflate should never expand the 32KB input by 2x, so it's safe to store it as u16
        let window_compressed_size =
            u16::try_from(window_compressed.len()).expect("compressed window too large");

        let num_bits = self
            .common
            .decomp
            .get_block_boundary_state()
            .unwrap()
            .num_bits;

        // If the next block depends on some buffered bits from the previous input
        // byte, we'll re-read that byte when resuming, to avoid having to store buf_bit
        let in_pos_offset = if num_bits == 0 { 0 } else { 1 };

        self.access_points.push(AccessPoint {
            out_pos: self.common.output_pos,
            in_pos: self.common.input_pos - in_pos_offset,
            window_offset: self.windows_size,
            window_compressed_size,
            num_bits,
        });
        self.index_stream.write_all(&window_compressed)?;

        self.windows_size += window_compressed_size as u64;

        Ok(())
    }

    fn make_progress(&mut self) -> std::io::Result<()> {
        let flags = inflate_flags::TINFL_FLAG_STOP_ON_BLOCK_BOUNDARY;
        let status = self.common.make_progress(flags)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            TINFLStatus::BlockBoundary => {
                self.last_block_start = self.common.input_pos;

                if self.common.input_pos - self.last_access_point >= self.span.0 {
                    self.last_access_point = self.common.input_pos;

                    self.create_access_point()?;
                }

                Ok(())
            }
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }
}

impl<GzStream, IndexStream> Read for GzIndexBuilder<GzStream, IndexStream>
where
    GzStream: Read,
    IndexStream: Write + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while !self.common.has_output() && !self.common.done {
            self.make_progress()?;
        }

        Ok(self.common.flush_output(buf))
    }
}
