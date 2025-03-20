use std::io::{Read, Seek, SeekFrom, Write};

use indexed_deflate::{AccessPointSpan, Result};
use paste::paste;
use rand::{seq::SliceRandom, SeedableRng};
use sha2::{Digest, Sha256};

fn data_random(i: u64) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(i.to_le_bytes());
    hasher.finalize().to_vec()
}

fn data_compressible(i: u64) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(i.to_le_bytes());
    hasher.finalize().iter().map(|n| n % 16).collect()
}

// Create some pseudorandom data, compress it, build the index,
// then decode and seek randomly to check it reads the correct data
macro_rules! basic_test_data {
    ($name:ident, $encoder:ident, $decoder:ident, $builder:ident, $data:ident) => {
        #[test]
        fn $name() -> Result<()> {
            let file_size = 4 * 1024 * 1024;
            let chunk_size = Sha256::output_size() as u64;
            let num_chunks = file_size / chunk_size;

            let gz = tempfile::NamedTempFile::new()?;
            let index = tempfile::NamedTempFile::new()?;

            let mut encoder = flate2::write::$encoder::new(&gz, flate2::Compression::default());
            for i in 0..num_chunks {
                encoder.write_all(&$data(i))?;
            }
            encoder.finish()?;

            let mut builder = indexed_deflate::$builder::new(
                gz.reopen()?,
                index.reopen()?,
                AccessPointSpan::new(128 * 1024),
            )?;
            builder.seek(SeekFrom::End(0))?;
            builder.finish()?;

            let mut decoder = indexed_deflate::$decoder::new(gz.reopen()?, index.reopen()?)?;

            let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
            let mut chunks: Vec<_> = (0..num_chunks).collect();
            chunks.shuffle(&mut rng);
            for &c in &chunks[0..1024] {
                decoder.seek(SeekFrom::Start(c * chunk_size))?;
                let mut buf = vec![0; chunk_size as usize];
                decoder.read_exact(&mut buf)?;
                assert_eq!(buf, $data(c));
            }

            Ok(())
        }
    };
}

macro_rules! basic_test {
    ($name:ident, $encoder:ident, $decoder:ident, $builder:ident) => {
        paste! {
            basic_test_data!([<basic_ $name _r>], $encoder, $decoder, $builder, data_random);
            basic_test_data!([<basic_ $name _c>], $encoder, $decoder, $builder, data_compressible);
        }
    };
}

basic_test!(gz, GzEncoder, GzDecoder, GzIndexBuilder);
basic_test!(deflate, DeflateEncoder, DeflateDecoder, DeflateIndexBuilder);
basic_test!(zlib, ZlibEncoder, ZlibDecoder, ZlibIndexBuilder);

// Write some data both before and after the index, and check we can still read it
#[test]
fn test_index_userdata() -> Result<()> {
    let file_size = 1024 * 1024;
    let chunk_size = Sha256::output_size() as u64;
    let num_chunks = file_size / chunk_size;

    let gz = tempfile::NamedTempFile::new()?;
    let index = tempfile::NamedTempFile::new()?;

    let mut encoder = flate2::write::GzEncoder::new(&gz, flate2::Compression::default());
    for i in 0..num_chunks {
        encoder.write_all(&data_random(i))?;
    }
    encoder.finish()?;

    {
        let mut index_build = index.reopen()?;
        index_build.write_all(&data_random(1000))?;

        let mut builder = indexed_deflate::GzIndexBuilder::new(
            gz.reopen()?,
            &index_build,
            AccessPointSpan::new(128 * 1024),
        )?;
        builder.seek(SeekFrom::End(0))?;
        builder.finish()?;
        index_build.write_all(&data_random(2000))?;
    }

    {
        let mut index_decode = index.reopen()?;

        let mut buf = vec![0; chunk_size as usize];
        index_decode.read_exact(&mut buf)?;
        assert_eq!(buf, data_random(1000));

        let mut decoder = indexed_deflate::GzDecoder::new(gz.reopen()?, index_decode)?;

        decoder.with_index(|index_callback| {
            let mut buf = Vec::new();
            index_callback.read_to_end(&mut buf)?;
            assert_eq!(buf, data_random(2000));
            Ok(())
        })?;

        let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
        let mut chunks: Vec<_> = (0..num_chunks).collect();
        chunks.shuffle(&mut rng);
        for &c in &chunks[0..1024] {
            decoder.seek(SeekFrom::Start(c * chunk_size))?;
            let mut buf = vec![0; chunk_size as usize];
            decoder.read_exact(&mut buf)?;
            assert_eq!(buf, data_random(c));
        }
    }

    Ok(())
}
