//! `decode_working_bytes`, derived from headers before any allocation, is
//! an upper bound on what decoding really holds at its peak, checked
//! against the allocator for every format: in-tree fixtures, generated
//! images, and VP8L streams that declare many Huffman groups, whose tables
//! scale with the bytes that declare them rather than with the image. One
//! test, since the live-byte count is global.
// SAFETY-REVIEWED: the only unsafe code is the counting allocator below,
// which forwards to the system allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use runtime_media::{
    Alpha, ImagePolicy, MediaLimits, Orientation, decode_image, decode_working_bytes,
};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY-REVIEWED: every method forwards to the system allocator with the
// caller's pointer and layout unchanged and only updates counters.
unsafe impl GlobalAlloc for Counting {
    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc(layout) };
        grow(pointer, layout.size())
    }

    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        grow(pointer, layout.size())
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        let next = unsafe { System.realloc(pointer, layout, size) };
        // A move holds both blocks at once, so the new one counts before
        // the old one goes.
        if !grow(next, size).is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        }
        next
    }
}

fn grow(pointer: *mut u8, bytes: usize) -> *mut u8 {
    if !pointer.is_null() {
        let live = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
        PEAK.fetch_max(live, Ordering::SeqCst);
    }
    pointer
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The peak live bytes `run` adds to what was live when it started.
fn peak_of<T>(run: impl FnOnce() -> T) -> (T, usize) {
    let start = LIVE.load(Ordering::SeqCst);
    PEAK.store(start, Ordering::SeqCst);
    let result = run();
    (result, PEAK.load(Ordering::SeqCst) - start)
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels: 1 << 26,
        maximum_audio_frames: 0,
        maximum_frames: 1,
        maximum_tokens: 0,
        maximum_aspect_ratio: 1e6,
    }
}

fn noise(length: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn png(width: u32, height: u32, color: png::ColorType, depth: png::BitDepth) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    if color == png::ColorType::Indexed {
        encoder.set_palette(noise(768, 3));
        encoder.set_trns(noise(200, 5));
    }
    let row = (width as usize * color.samples() * depth as usize).div_ceil(8);
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&noise(row * height as usize, width ^ height))
        .unwrap();
    writer.finish().unwrap();
    out
}

/// An Adam7-interlaced 8-bit RGB PNG of noise, stored uncompressed.
fn adam7_png(width: u32, height: u32) -> Vec<u8> {
    const PASSES: [(u32, u32, u32, u32); 7] = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];
    let count = |size: u32, start: u32, step: u32| size.saturating_sub(start).div_ceil(step);
    let mut raw = Vec::new();
    for (x0, y0, dx, dy) in PASSES {
        let (columns, rows) = (count(width, x0, dx), count(height, y0, dy));
        if columns == 0 {
            continue;
        }
        for row in 0..rows {
            raw.push(0);
            raw.extend(noise(columns as usize * 3, row ^ (x0 << 8) ^ (y0 << 16)));
        }
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let chunk = |kind: &[u8; 4], data: &[u8], out: &mut Vec<u8>| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    let mut ihdr = width.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 1]);
    chunk(b"IHDR", &ihdr, &mut out);
    chunk(
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 1),
        &mut out,
    );
    chunk(b"IEND", &[], &mut out);
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0_u32, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg())
        })
    })
}

/// A GIF whose 800 x 600 first frame sits at (300, 200) on a 1000 x 700
/// screen, so the canvas grows to 1100 x 800, and a second frame follows.
fn gif() -> Vec<u8> {
    let mut encoder = gif::Encoder::new(Vec::new(), 1000, 700, &noise(768, 7)).unwrap();
    for (left, top, seed) in [(300, 200, 11), (0, 0, 13)] {
        let frame = gif::Frame {
            left,
            top,
            width: 800,
            height: 600,
            transparent: Some(4),
            buffer: noise(800 * 600, seed).into(),
            ..Default::default()
        };
        encoder.write_frame(&frame).unwrap();
    }
    encoder.into_inner().unwrap()
}

/// The lossy fixture with `extra` zero bytes appended to its VP8 chunk,
/// which image-webp reads into the frame's last partition and ignores.
fn long_last_partition(fixtures: &Path, extra: usize) -> Vec<u8> {
    let file = std::fs::read(fixtures.join("centered-pad-webp-lossy-301x199.webp")).unwrap();
    assert_eq!(&file[12..16], b"VP8 ");
    let size = u32::from_le_bytes(file[16..20].try_into().unwrap()) as usize;
    let mut vp8 = file[20..20 + size].to_vec();
    vp8.resize(size + extra, 0);
    let padded = vp8.len() + (vp8.len() & 1);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(4 + 8 + padded as u32).to_le_bytes());
    out.extend_from_slice(b"WEBPVP8 ");
    out.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
    out.extend_from_slice(&vp8);
    out.resize(8 + 4 + 8 + padded, 0);
    out
}

fn lossless_webp(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    image_webp::WebPEncoder::new(&mut out)
        .encode(
            &noise((width * height * 4) as usize, 17),
            width,
            height,
            image_webp::ColorType::Rgba8,
        )
        .unwrap();
    out
}

/// A PNG chunk: length, type, data and CRC.
fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = (data.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&out[4..]).to_be_bytes());
    out
}

/// `png` with `chunks` inserted after its IHDR.
fn with_chunks(png: &[u8], chunks: &[Vec<u8>]) -> Vec<u8> {
    [&png[..33], &chunks.concat(), &png[33..]].concat()
}

/// A little-endian TIFF stream whose IFD0 holds orientation 6, padded with
/// `padding` zero bytes.
fn exif6(padding: usize) -> Vec<u8> {
    let mut out = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
    out.resize(out.len() + padding, 0);
    out
}

/// A zTXt chunk holding `text` under `key`.
fn ztxt(key: &[u8], text: &[u8]) -> Vec<u8> {
    let data = [
        key,
        b"\0\0",
        &miniz_oxide::deflate::compress_to_vec_zlib(text, 6),
    ]
    .concat();
    png_chunk(b"zTXt", &data)
}

/// An XMP packet with orientation 6, padded with spaces to `length` bytes.
fn xmp6(length: usize) -> Vec<u8> {
    let mut out = br#"<x:xmpmeta><rdf:Description tiff:Orientation="6"/></x:xmpmeta>"#.to_vec();
    out.resize(length, b' ');
    out
}

/// An animated WebP whose one frame is `vp8l` (a VP8L chunk's data) on a
/// `width` x `height` canvas, with an EXIF orientation of 6.
fn animated_oriented_webp(width: u32, height: u32, vp8l: &[u8]) -> Vec<u8> {
    let le24 = |value: u32| value.to_le_bytes()[..3].to_vec();
    let sub = |fourcc: &[u8; 4], data: &[u8]| {
        let mut out = fourcc.to_vec();
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(0);
        }
        out
    };
    let vp8x = [vec![0x0a, 0, 0, 0], le24(width - 1), le24(height - 1)].concat();
    let anmf = [
        le24(0),
        le24(0),
        le24(width - 1),
        le24(height - 1),
        le24(100),
        vec![0],
        sub(b"VP8L", vp8l),
    ]
    .concat();
    let body = [
        b"WEBP".to_vec(),
        sub(b"VP8X", &vp8x),
        sub(b"ANIM", &[0; 6]),
        sub(b"ANMF", &anmf),
        sub(b"EXIF", &exif6(0)),
    ]
    .concat();
    [
        b"RIFF".as_slice(),
        &(body.len() as u32).to_le_bytes(),
        &body,
    ]
    .concat()
}

/// Bits written least significant first, as VP8L reads them.
#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    used: usize,
}

impl Bits {
    fn put(&mut self, value: u32, count: u32) {
        for bit in 0..count {
            if self.used % 8 == 0 {
                self.bytes.push(0);
            }
            *self.bytes.last_mut().unwrap() |= (((value >> bit) & 1) as u8) << (self.used % 8);
            self.used += 1;
        }
    }

    /// A prefix code, its most significant bit first.
    fn code(&mut self, code: u32, length: u32) {
        for bit in (0..length).rev() {
            self.put(code >> bit, 1);
        }
    }

    /// A simple prefix code of one symbol, which reads no bits.
    fn single(&mut self, symbol: u32) {
        self.put(1, 1);
        self.put(0, 1);
        if symbol < 2 {
            self.put(0, 1);
            self.put(symbol, 1);
        } else {
            self.put(1, 1);
            self.put(symbol, 8);
        }
    }

    /// Symbols 0 to 10 with lengths 1 to 10 and 10: a complete code whose
    /// lookup table has all 1024 entries, from 93 bits.
    fn skewed(&mut self) {
        self.put(0, 1);
        // Code-length code lengths in VP8L's order 17, 18, 0, 1, 2, 3, 4,
        // 5, 16, 6, 7, 8, 9, 10: lengths 1 to 6 take 3 bits, 7 to 10 take 4.
        self.put(14 - 4, 4);
        for length in [0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 4, 4, 4, 4] {
            self.put(length, 3);
        }
        // Eleven code lengths follow.
        self.put(1, 1);
        self.put(1, 3);
        self.put(11 - 2, 4);
        for length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10] {
            if length <= 6 {
                self.code(length - 1, 3);
            } else {
                self.code(length - 7 + 12, 4);
            }
        }
    }

    /// Symbols 0 to 2047 all of length 11, from 1089 bits: the full lookup
    /// table and a 4096-node tree for the codes past it.
    fn long(&mut self) {
        self.put(0, 1);
        // Lengths 11 and 16 (repeat the previous length) take one bit each.
        self.put(15 - 4, 4);
        for length in [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] {
            self.put(length, 3);
        }
        self.put(1, 1);
        self.put(4, 3);
        self.put(343 - 2, 10);
        self.code(0, 1);
        self.code(0, 1);
        for _ in 0..341 {
            self.code(1, 1);
            self.put(3, 2);
        }
    }
}

/// A 4 x 4 lossless WebP whose entropy image names `groups` meta-code
/// groups, each `group` writes, and whose pixels all use the last one,
/// which `pixel` writes one pixel of. Returns the file and the VP8L bytes.
fn huffman_groups(
    groups: u32,
    cache: bool,
    group: impl Fn(&mut Bits),
    pixel: impl Fn(&mut Bits),
) -> (Vec<u8>, usize) {
    let mut bits = Bits::default();
    bits.put(0x2f, 8);
    bits.put(3, 14);
    bits.put(3, 14);
    bits.put(0, 4);
    // No transforms; the color cache; meta codes over 4 x 4 blocks.
    bits.put(0, 1);
    if cache {
        bits.put(1, 1);
        bits.put(11, 4);
    } else {
        bits.put(0, 1);
    }
    bits.put(1, 1);
    bits.put(0, 3);
    // The 1 x 1 entropy image: its one pixel's red and green name the
    // last group.
    bits.put(0, 1);
    for symbol in [(groups - 1) & 0xff, (groups - 1) >> 8, 0, 0, 0] {
        bits.single(symbol);
    }
    for _ in 0..groups {
        group(&mut bits);
    }
    for _ in 0..16 {
        pixel(&mut bits);
    }
    let vp8l = bits.bytes;
    let mut file = b"RIFF".to_vec();
    let padded = vp8l.len() + (vp8l.len() & 1);
    file.extend_from_slice(&(4 + 8 + padded as u32).to_le_bytes());
    file.extend_from_slice(b"WEBPVP8L");
    file.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
    file.extend_from_slice(&vp8l);
    file.resize(8 + 4 + 8 + padded, 0);
    (file, vp8l.len())
}

fn check(name: &str, bytes: &[u8], policy: ImagePolicy) -> (usize, u64) {
    let bound = decode_working_bytes(bytes, limits(), policy.orientation).unwrap();
    let (decoded, peak) = peak_of(|| decode_image(bytes, limits(), policy));
    decoded.unwrap_or_else(|error| panic!("{name}: {error}"));
    assert!(
        peak as u64 <= bound,
        "{name}: peak {peak} above bound {bound}"
    );
    (peak, bound)
}

#[test]
fn decode_peaks_stay_within_their_header_bounds() {
    let stored = ImagePolicy {
        alpha: Alpha::Discard,
        orientation: Orientation::Stored,
    };
    let exif = ImagePolicy {
        alpha: Alpha::CompositeWhite,
        orientation: Orientation::Exif,
    };
    let fixtures = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));
    let mut files = std::fs::read_dir(fixtures)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("png" | "jpg" | "gif" | "webp")
            )
        })
        .collect::<Vec<_>>();
    files.sort();
    assert!(files.len() >= 34, "{} fixtures", files.len());
    for path in &files {
        let bytes = std::fs::read(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        check(&name, &bytes, stored);
        check(&name, &bytes, exif);
    }
    // The decode corpus, under both orientations where it decodes.
    for entry in std::fs::read_dir(fixtures.join("image-decode")).unwrap() {
        let path = entry.unwrap().path();
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        for policy in [stored, exif] {
            if decode_image(&bytes, limits(), policy).is_ok() {
                check(&name, &bytes, policy);
            }
        }
    }

    use png::{BitDepth, ColorType};
    for (name, bytes) in [
        ("png rgb", png(1500, 1000, ColorType::Rgb, BitDepth::Eight)),
        (
            "png wide rgba",
            png(65_536, 3, ColorType::Rgba, BitDepth::Eight),
        ),
        (
            "png gray 1-bit",
            png(999, 777, ColorType::Grayscale, BitDepth::One),
        ),
        (
            "png palette",
            png(640, 480, ColorType::Indexed, BitDepth::Eight),
        ),
        ("png adam7", adam7_png(997, 601)),
        ("gif offset frame", gif()),
        ("webp lossless", lossless_webp(800, 600)),
        (
            "webp lossy long last partition",
            long_last_partition(fixtures, 8 << 20),
        ),
    ] {
        check(name, &bytes, stored);
        check(
            name,
            &bytes,
            ImagePolicy {
                alpha: Alpha::CompositeWhite,
                ..stored
            },
        );
    }

    // Orientation metadata: a wide image rotated onto its side (the
    // decoder's rows stay at the stored width), a 1 MiB compressed raw
    // profile, compressed XMP up to the 4 MiB budget, and an animated WebP
    // whose canvas is transposed.
    let wide = png(65_536, 3, ColorType::Rgb, BitDepth::Eight);
    let rotated = with_chunks(&wide, &[png_chunk(b"eXIf", &exif6(0))]);
    let hex = exif6(500 << 10)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let profile = format!("\nexif\n{:8}\n{hex}\n", hex.len() / 2);
    assert!(profile.len() <= 1 << 20);
    let small = png(640, 480, ColorType::Rgb, BitDepth::Eight);
    let raw_profile = with_chunks(
        &small,
        &[ztxt(b"Raw profile type exif", profile.as_bytes())],
    );
    let budget = with_chunks(&small, &vec![ztxt(b"XML:com.adobe.xmp", &xmp6(1 << 20)); 4]);
    let over = with_chunks(&small, &vec![ztxt(b"XML:com.adobe.xmp", &xmp6(1 << 20)); 5]);
    let file = lossless_webp(800, 600);
    let webp = animated_oriented_webp(800, 600, &file[20..]);
    for (name, bytes, size) in [
        ("png wide rotated", &rotated, (3, 65_536)),
        ("png raw profile", &raw_profile, (480, 640)),
        ("png metadata at budget", &budget, (480, 640)),
        ("webp animated rotated", &webp, (600, 800)),
    ] {
        check(name, bytes, exif);
        let image = decode_image(bytes, limits(), exif).unwrap();
        assert_eq!((image.width, image.height), size, "{name}");
    }
    assert!(
        decode_image(&over, limits(), exif)
            .unwrap_err()
            .to_string()
            .contains("inflates past")
    );
    check("png metadata over budget, stored", &over, stored);

    // Huffman groups cost far more than the bytes that declare them, but
    // the bound's allowance per lossless byte covers the densest forms:
    // full lookup tables with long-code trees, and many one-symbol groups.
    let (dense, dense_bytes) = huffman_groups(
        64,
        true,
        |bits| {
            bits.long();
            for _ in 0..4 {
                bits.skewed();
            }
        },
        |bits| bits.put(0, 14),
    );
    let (peak, _) = check("vp8l dense groups", &dense, stored);
    assert!(peak > 256 * dense_bytes, "{peak} from {dense_bytes} bytes");
    let (many, many_bytes) = huffman_groups(
        65_536,
        false,
        |bits| {
            for _ in 0..5 {
                bits.single(0);
            }
        },
        |_| {},
    );
    let (peak, _) = check("vp8l many groups", &many, stored);
    assert!(peak > 64 * many_bytes, "{peak} from {many_bytes} bytes");
}
