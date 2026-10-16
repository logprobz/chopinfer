#![forbid(unsafe_code)]

//! What the public digest surface promises.
//!
//! Three entry points hash bytes here: one over a slice, one over a reader read
//! to its end, and one incremental state a caller feeds in pieces. They are one
//! implementation, so what these tests hold is that the standard vectors come
//! out right and that no division of the same bytes changes the answer.

use std::io::Cursor;

use runtime_weights::{Sha256State, sha256_bytes, sha256_reader};

/// FIPS 180-4 vectors, taken through the incremental state itself.
#[test]
fn the_incremental_state_produces_the_standard_vectors() {
    for (input, expected) in [
        (
            b"".as_slice(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc".as_slice(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_slice(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ] {
        let mut state = Sha256State::new();
        state.update(input).expect("the vector length fits u64");
        assert_eq!(state.finalize(), expected);
    }

    // The long vector, which is the one that exercises many blocks rather than
    // the padding of a single one.
    let mut state = Sha256State::new();
    for _ in 0..1_000 {
        state
            .update(&[b'a'; 1_000])
            .expect("the vector length fits u64");
    }
    assert_eq!(
        state.finalize(),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

/// The lengths where SHA-256 padding changes shape, hashed in one piece.
///
/// 55 is the last length that pads inside its own block, 56 is the first that
/// needs a second one, and 64 is a whole block with nothing left over. The
/// megabyte lengths are where [`sha256_reader`] fills, empties, and refills its
/// own buffer.
#[test]
fn the_padding_and_buffer_boundaries_hash_correctly() {
    for (length, expected) in boundary_vectors() {
        let input = pattern(length);
        let mut state = Sha256State::new();
        state.update(&input).expect("the fixture fits u64");
        assert_eq!(state.finalize(), expected, "length {length}");
        assert_eq!(sha256_bytes(&input), expected, "length {length}");
        assert_eq!(
            sha256_reader(&mut Cursor::new(input)).expect("a cursor never fails to read"),
            expected,
            "length {length}"
        );
    }
}

/// The same bytes in any division are the same digest.
///
/// A caller that hashes a file while it writes it chooses its pieces from
/// whatever the reads returned, so no piece size may be privileged: one byte at
/// a time, a piece that straddles a block boundary, and an empty piece in the
/// middle all have to agree with the one-shot digest.
#[test]
fn any_division_of_the_same_bytes_hashes_alike() {
    for (length, expected) in boundary_vectors() {
        let input = pattern(length);
        for piece in [1_usize, 7, 63, 64, 65, 100, 4_096, 1_048_576] {
            let mut state = Sha256State::new();
            for chunk in input.chunks(piece) {
                state.update(chunk).expect("the fixture fits u64");
                // An empty piece is a legal thing for a reader to hand back,
                // and it must not disturb the state it passes through.
                state.update(&[]).expect("an empty piece is legal");
            }
            assert_eq!(
                state.finalize(),
                expected,
                "length {length} in pieces of {piece}"
            );
        }
    }
}

/// A piece boundary placed at every offset of one multi-block message.
#[test]
fn every_split_point_of_one_message_hashes_alike() {
    let input = pattern(200);
    let expected = sha256_bytes(&input);
    for split in 0..=input.len() {
        let mut state = Sha256State::new();
        state.update(&input[..split]).expect("the fixture fits u64");
        state.update(&input[split..]).expect("the fixture fits u64");
        assert_eq!(state.finalize(), expected, "split at {split}");
    }
}

/// A fresh state is an empty stream, and `Default` is that same state.
#[test]
fn a_new_state_is_the_empty_digest() {
    assert_eq!(Sha256State::new().finalize(), sha256_bytes(b""));
    assert_eq!(Sha256State::default().finalize(), sha256_bytes(b""));
}

/// The lengths above, with the digest each one is known to have.
fn boundary_vectors() -> [(usize, &'static str); 15] {
    [
        (
            1,
            "084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5",
        ),
        (
            55,
            "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b",
        ),
        (
            56,
            "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27",
        ),
        (
            57,
            "35df609437dcfea3279283ab79fd554e2bf78f8f7ae2de532d8ee300b09e8f73",
        ),
        (
            63,
            "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055",
        ),
        (
            64,
            "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241",
        ),
        (
            65,
            "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e",
        ),
        (
            119,
            "9ce7368e4daf32341631b492e80359dc9f594b48453cd0dd5bf0b19279cc177e",
        ),
        (
            120,
            "7836b787757e95e58b3ca5aec90b1b004e8deba1e50e9675af9cabf1a13a04b5",
        ),
        (
            127,
            "a8d23e75d936f303d248888d9b165ee543f4cbafcad3c9dd2a79bd84faa11d07",
        ),
        (
            128,
            "d2742f1f4ac6bb7ca2b239ee18402ba8b3f9f8e652d2a72973c2b9ba11c08cf6",
        ),
        (
            129,
            "307f8fc2c1622b92762e818d39a185d4d667ad49a4b07ceae1f4afa008a93ec4",
        ),
        (
            1_048_575,
            "9487450cca5e9cca35dbef99a45b4c727652508e60210c77c3f41a527328e02a",
        ),
        (
            1_048_576,
            "172c15dc2e12b50e523d8e657cbe7fbb11c1053252bbf1e1431077d57d8128fd",
        ),
        (
            1_048_577,
            "75059752ce0b546127d584a3992627babb59950933c6d82d10ad46fec48e7c96",
        ),
    ]
}

/// The fixture bytes the vectors above were taken over.
fn pattern(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| ((index * 7 + 3) % 256) as u8)
        .collect()
}
