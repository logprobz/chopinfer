use std::io::Read;

use crate::{ArtifactError, Result};

const INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

pub fn bytes(input: &[u8]) -> String {
    let mut hasher = Sha256State::new();
    hasher.update(input).expect("slice length fits u64");
    hasher.finalize()
}

pub fn reader(reader: &mut impl Read) -> Result<String> {
    let mut buffer = [0_u8; 1024 * 1024];
    let mut hasher = Sha256State::new();
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| ArtifactError::new(format!("cannot hash artifact stream: {error}")))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count])?;
    }
    Ok(hasher.finalize())
}

pub(crate) fn hex(digest: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

/// SHA-256 over a byte stream a caller feeds in pieces.
///
/// A caller that already holds every byte hashes with [`crate::sha256_bytes`],
/// and one reading a stream to its end hashes with [`crate::sha256_reader`].
/// This is for the third case: bytes that pass through a caller once, in
/// arbitrary pieces, and are hashed as they go rather than read a second time.
/// The pieces do not have to align with anything; the same bytes in any
/// division produce the same digest.
///
/// The state carries no notion of what the bytes mean. It is a digest and
/// nothing else.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sha256State {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    byte_len: u64,
}

impl Default for Sha256State {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256State {
    /// Begins a digest over an empty stream.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: INITIAL,
            buffer: [0; 64],
            buffered: 0,
            byte_len: 0,
        }
    }

    /// Adds the next piece of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the stream grows past the length SHA-256 can
    /// describe, which is the only way this can fail.
    pub fn update(&mut self, mut input: &[u8]) -> Result<()> {
        self.byte_len = self
            .byte_len
            .checked_add(u64::try_from(input.len()).map_err(|_| {
                ArtifactError::new("SHA-256 input length does not fit u64".to_owned())
            })?)
            .ok_or_else(|| ArtifactError::new("SHA-256 input length overflow".to_owned()))?;
        if self.byte_len > u64::MAX / 8 {
            return Err(ArtifactError::new(
                "SHA-256 bit length exceeds the format limit".to_owned(),
            ));
        }
        if self.buffered != 0 {
            let needed = 64 - self.buffered;
            let count = needed.min(input.len());
            self.buffer[self.buffered..self.buffered + count].copy_from_slice(&input[..count]);
            self.buffered += count;
            input = &input[count..];
            if self.buffered != 64 {
                // This piece ended inside the block being filled. The tail
                // below writes what is left of the input over the start of that
                // block, so returning here is what keeps the partial block for
                // the next piece instead of dropping it.
                return Ok(());
            }
            compress(&mut self.state, &self.buffer);
            self.buffered = 0;
        }
        // Let sha2 borrow all whole blocks in one call, so hardware backends
        // can process a long input without per-block dispatch overhead.
        use sha2::digest::{
            block_buffer::{BlockBuffer, Eager},
            consts::U64,
        };
        let mut blocks = BlockBuffer::<U64, Eager>::default();
        blocks.digest_blocks(input, |blocks| sha2::compress256(&mut self.state, blocks));
        let tail = blocks.get_data();
        self.buffer[..tail.len()].copy_from_slice(tail);
        self.buffered = tail.len();
        Ok(())
    }

    /// Ends the stream and returns its digest as 64 lowercase hexadecimal
    /// characters, the form every pinned identity in this crate is written in.
    #[must_use]
    pub fn finalize(self) -> String {
        hex(&self.finish())
    }

    pub(crate) fn finish(mut self) -> [u8; 32] {
        let bit_len = self.byte_len * 8;
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            compress(&mut self.state, &self.buffer);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffered..56].fill(0);
        }
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut self.state, &self.buffer);
        let mut digest = [0_u8; 32];
        for (word, target) in self.state.iter().zip(digest.chunks_exact_mut(4)) {
            target.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

fn compress(state: &mut [u32; 8], block: &[u8]) {
    sha2::compress256(
        state,
        std::slice::from_ref(sha2::digest::generic_array::GenericArray::from_slice(block)),
    );
}

#[cfg(test)]
mod tests {
    use super::bytes;

    #[test]
    fn standard_vectors() {
        assert_eq!(
            bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

#[cfg(test)]
mod backend_regression {
    use super::*;
    const EMPTY: Sha256State = Sha256State::new();

    #[test]
    fn state_api_and_clone_continuations() {
        fn traits<T: Clone + Eq + std::fmt::Debug>() {}
        traits::<Sha256State>();
        assert_eq!(EMPTY, Sha256State::default());
        let mut state = EMPTY;
        state.update(b"a").unwrap();
        let mut same = state.clone();
        state.update(b"bc").unwrap();
        same.update(b"bc").unwrap();
        assert_eq!(state, same);
        assert_eq!(
            state.finalize(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut different = EMPTY;
        different.update(b"abd").unwrap();
        assert_ne!(same, different);
        assert_ne!(same.finalize(), different.finalize());
    }

    #[test]
    fn length_limits_remain_enforced() {
        let mut state = EMPTY;
        state.byte_len = u64::MAX / 8;
        assert!(state.update(b"x").is_err());
        let mut state = EMPTY;
        state.byte_len = u64::MAX;
        assert!(state.update(b"x").is_err());
    }

    #[test]
    fn reader_propagates_errors() {
        struct Broken;
        impl std::io::Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("fixture read failure"))
            }
        }
        assert!(reader(&mut Broken).is_err());
    }

    #[test]
    fn published_long_vector() {
        assert_eq!(
            bytes(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            bytes(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
