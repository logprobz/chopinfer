use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MediaError(pub String);

impl MediaError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MediaError {}

pub type MediaResult<T> = Result<T, MediaError>;

/// `len` copies of `value`, or an error instead of an abort when the host
/// cannot allocate them.
pub(crate) fn try_filled<T: Clone>(len: usize, value: T) -> MediaResult<Vec<T>> {
    let mut out = try_with_capacity(len)?;
    out.resize(len, value);
    Ok(out)
}

/// An empty vector with room for `len` elements, or an error instead of an
/// abort when the host cannot allocate it.
pub(crate) fn try_with_capacity<T>(len: usize) -> MediaResult<Vec<T>> {
    let mut out = Vec::new();
    try_reserve(&mut out, len)?;
    Ok(out)
}

pub(crate) fn try_reserve<T>(out: &mut Vec<T>, additional: usize) -> MediaResult<()> {
    #[cfg(test)]
    if out.capacity() - out.len() < additional
        && fail_allocation(additional.saturating_mul(size_of::<T>()))
    {
        return Err(MediaError::new("media buffer allocation failed"));
    }
    out.try_reserve_exact(additional)
        .map_err(|_| MediaError::new("media buffer allocation failed"))
}

#[cfg(test)]
thread_local! {
    static ALLOCATION_FAILURE: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn fail_allocation(bytes: usize) -> bool {
    ALLOCATION_FAILURE.with(|failure| match failure.get() {
        Some((size, skip)) if size == bytes => {
            failure.set(skip.checked_sub(1).map(|skip| (size, skip)));
            skip == 0
        }
        _ => false,
    })
}

/// Fail one reservation of `bytes` on this test thread after `skip` matches.
#[cfg(test)]
pub(crate) fn with_allocation_failure<T>(bytes: usize, skip: usize, run: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ALLOCATION_FAILURE.set(None);
        }
    }
    ALLOCATION_FAILURE.set(Some((bytes, skip)));
    let _reset = Reset;
    run()
}

/// How a decoder handles transparency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Alpha {
    /// Composite each pixel onto white by its alpha.
    #[default]
    CompositeWhite,
    /// Keep the color channels and ignore alpha, as PIL's `convert("RGB")`
    /// does.
    Discard,
}

/// How a JPEG decoder treats an EXIF orientation tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orientation {
    /// Transpose the pixels into the tagged orientation.
    #[default]
    Exif,
    /// Keep the stored pixel order and size, as PIL's `Image.open` followed
    /// by `convert("RGB")` does without `ImageOps.exif_transpose`.
    Stored,
}

#[derive(Clone, Copy, Debug)]
pub struct MediaLimits {
    pub maximum_bytes: usize,
    pub maximum_pixels: u64,
    pub maximum_audio_frames: u64,
    pub maximum_frames: usize,
    pub maximum_tokens: u64,
    pub maximum_aspect_ratio: f64,
}

impl MediaLimits {
    pub fn check_bytes(self, length: usize) -> MediaResult<()> {
        if length > self.maximum_bytes {
            return Err(MediaError::new("compressed media byte limit exceeded"));
        }
        Ok(())
    }

    pub fn check_image(self, width: u32, height: u32) -> MediaResult<()> {
        self.check_named_image("decoded", width, height)
    }

    /// [`Self::check_image`] for an image whose refusal names its format
    /// (`name`), its size and the limit that applied.
    pub(crate) fn check_named_image(self, name: &str, width: u32, height: u32) -> MediaResult<()> {
        if width == 0 || height == 0 {
            return Err(MediaError::new(format!(
                "{name} image {width} x {height} has no pixels"
            )));
        }
        let pixels = u64::from(width) * u64::from(height);
        if pixels > self.maximum_pixels {
            return Err(MediaError::new(format!(
                "{name} image {width} x {height} has {pixels} pixels, above the {}-pixel limit",
                self.maximum_pixels
            )));
        }
        let ratio = f64::from(width.max(height)) / f64::from(width.min(height));
        if !self.maximum_aspect_ratio.is_finite() || ratio > self.maximum_aspect_ratio {
            return Err(MediaError::new(format!(
                "{name} image {width} x {height} has an aspect ratio above {}",
                self.maximum_aspect_ratio
            )));
        }
        Ok(())
    }

    pub fn check_audio(self, frames: u64) -> MediaResult<()> {
        if frames > self.maximum_audio_frames {
            return Err(MediaError::new("decoded audio frame limit exceeded"));
        }
        Ok(())
    }

    pub fn check_tokens(self, tokens: u64) -> MediaResult<()> {
        if tokens > self.maximum_tokens {
            return Err(MediaError::new("media token limit exceeded"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaSource {
    pub mime: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Probe {
    Png {
        width: u32,
        height: u32,
    },
    Jpeg {
        width: u32,
        height: u32,
    },
    Wav {
        sample_rate: u32,
        channels: u16,
        frames: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>, limits: MediaLimits) -> MediaResult<Self> {
        limits.check_image(width, height)?;
        let expected = u64::from(width) * u64::from(height) * 3;
        if u64::try_from(data.len()).ok() != Some(expected) {
            return Err(MediaError::new(
                "RGB byte count differs from image geometry",
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn decode_base64(input: &str, maximum_bytes: usize) -> MediaResult<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 || bytes.len() / 4 > maximum_bytes / 3 + 1 {
        return Err(MediaError::new("invalid or oversized base64 payload"));
    }
    let mut out = Vec::with_capacity((bytes.len() / 4 * 3).min(maximum_bytes));
    for (index, group) in bytes.chunks_exact(4).enumerate() {
        let last = index + 1 == bytes.len() / 4;
        let pad = group.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 2 || (!last && pad != 0) || group[..4 - pad].contains(&b'=') {
            return Err(MediaError::new("invalid base64 padding"));
        }
        let mut sextets = [0_u8; 4];
        for (i, &byte) in group[..4 - pad].iter().enumerate() {
            sextets[i] = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => return Err(MediaError::new("invalid base64 alphabet")),
            };
        }
        if (pad == 1 && sextets[2] & 3 != 0) || (pad == 2 && sextets[1] & 15 != 0) {
            return Err(MediaError::new("noncanonical base64 padding bits"));
        }
        let word = (u32::from(sextets[0]) << 18)
            | (u32::from(sextets[1]) << 12)
            | (u32::from(sextets[2]) << 6)
            | u32::from(sextets[3]);
        if out.len() + 3 - pad > maximum_bytes {
            return Err(MediaError::new("compressed media byte limit exceeded"));
        }
        out.push((word >> 16) as u8);
        if pad < 2 {
            out.push((word >> 8) as u8);
        }
        if pad == 0 {
            out.push(word as u8);
        }
    }
    Ok(out)
}

pub fn decode_data_url(input: &str, limits: MediaLimits) -> MediaResult<MediaSource> {
    let rest = input
        .strip_prefix("data:")
        .ok_or_else(|| MediaError::new("media URL must be data:"))?;
    let (header, payload) = rest
        .split_once(',')
        .ok_or_else(|| MediaError::new("data URL lacks comma"))?;
    let mime = header
        .strip_suffix(";base64")
        .ok_or_else(|| MediaError::new("data URL requires base64"))?;
    if !matches!(
        mime,
        "image/png" | "image/jpeg" | "audio/wav" | "audio/x-wav" | "video/mp4"
    ) {
        return Err(MediaError::new("unsupported media MIME type"));
    }
    let bytes = decode_base64(payload, limits.maximum_bytes)?;
    limits.check_bytes(bytes.len())?;
    Ok(MediaSource {
        mime: mime.into(),
        bytes,
    })
}

fn percent_decode_path(value: &str) -> MediaResult<PathBuf> {
    let mut bytes = Vec::with_capacity(value.len());
    let mut iter = value.as_bytes().iter().copied();
    while let Some(byte) = iter.next() {
        let decoded = if byte == b'%' {
            let hi = iter.next().and_then(|b| (b as char).to_digit(16));
            let lo = iter.next().and_then(|b| (b as char).to_digit(16));
            match (hi, lo) {
                (Some(hi), Some(lo)) => ((hi << 4) | lo) as u8,
                _ => return Err(MediaError::new("invalid file URL escape")),
            }
        } else {
            byte
        };
        if decoded == 0 {
            return Err(MediaError::new("NUL in file URL"));
        }
        bytes.push(decoded);
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Ok(std::ffi::OsString::from_vec(bytes).into())
    }
    #[cfg(not(unix))]
    {
        Ok(PathBuf::from(
            String::from_utf8(bytes).map_err(|_| MediaError::new("invalid file URL UTF-8"))?,
        ))
    }
}

pub fn read_source(
    url: &str,
    allowed_root: Option<&Path>,
    limits: MediaLimits,
) -> MediaResult<MediaSource> {
    if url.starts_with("data:") {
        return decode_data_url(url, limits);
    }
    let path = url
        .strip_prefix("file://")
        .ok_or_else(|| MediaError::new("network media URLs are refused"))?;
    if !path.starts_with('/') {
        return Err(MediaError::new("file URL must have an absolute path"));
    }
    let root = allowed_root.ok_or_else(|| MediaError::new("local media files are disabled"))?;
    let root =
        fs::canonicalize(root).map_err(|e| MediaError::new(format!("allowlist root: {e}")))?;
    let path = fs::canonicalize(percent_decode_path(path)?)
        .map_err(|e| MediaError::new(format!("media path: {e}")))?;
    if !path.starts_with(&root) {
        return Err(MediaError::new("media path is outside allowlist"));
    }
    read_local_source(&path, &root, limits)
}

fn read_local_source(path: &Path, root: &Path, limits: MediaLimits) -> MediaResult<MediaSource> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A resolved path can be replaced before open. Do not wait on a FIFO
        // or follow a replacement symlink; validate the descriptor below.
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .map_err(|e| MediaError::new(format!("open media: {e}")))?;
    let metadata = file
        .metadata()
        .map_err(|e| MediaError::new(format!("stat media: {e}")))?;
    if !metadata.is_file() {
        return Err(MediaError::new("media path is not a regular file"));
    }
    if metadata.len() > limits.maximum_bytes as u64 {
        return Err(MediaError::new("compressed media byte limit exceeded"));
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let opened = fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .map_err(|e| MediaError::new(format!("inspect opened media: {e}")))?;
        if !opened.starts_with(root) {
            return Err(MediaError::new("opened media is outside allowlist"));
        }
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.by_ref()
        .take(limits.maximum_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| MediaError::new(format!("read media: {e}")))?;
    limits.check_bytes(bytes.len())?;
    let mime = if bytes.get(4..8) == Some(b"ftyp".as_slice()) {
        "video/mp4"
    } else {
        match probe(&bytes, limits)? {
            Probe::Png { .. } => "image/png",
            Probe::Jpeg { .. } => "image/jpeg",
            Probe::Wav { .. } => "audio/wav",
        }
    };
    Ok(MediaSource {
        mime: mime.into(),
        bytes,
    })
}

pub fn probe(bytes: &[u8], limits: MediaLimits) -> MediaResult<Probe> {
    probe_oriented(bytes, limits, Orientation::Exif)
}

/// [`probe`], with a JPEG's size taken under `orientation`.
pub fn probe_oriented(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<Probe> {
    limits.check_bytes(bytes.len())?;
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let (width, height) = crate::png::probe(bytes, limits)?;
        Ok(Probe::Png { width, height })
    } else if bytes.starts_with(b"\xff\xd8") {
        let (width, height) = crate::jpeg::probe_oriented(bytes, limits, orientation)?;
        limits.check_image(width, height)?;
        Ok(Probe::Jpeg { width, height })
    } else if bytes.starts_with(b"RIFF") {
        let info = crate::wav::probe(bytes)?;
        limits.check_audio(info.frames)?;
        Ok(Probe::Wav {
            sample_rate: info.sample_rate,
            channels: info.channels,
            frames: info.frames,
        })
    } else {
        Err(MediaError::new("unsupported media header"))
    }
}

/// The decoder's host bytes for the image `probe` read from `bytes`, from
/// its header alone: a JPEG's coefficients and planes per padded block and
/// the RGB output ([`crate::jpeg::decode_storage_bytes`]); a PNG's RGB
/// output, the png crate's rows and the stream it inflates
/// ([`crate::png::decode_storage_bytes`]). The encoded source itself is
/// not counted.
pub fn image_decode_bytes(bytes: &[u8], probe: Probe, limits: MediaLimits) -> MediaResult<u64> {
    match probe {
        Probe::Jpeg { .. } => crate::jpeg::decode_storage_bytes(bytes, limits),
        Probe::Png { .. } => crate::png::decode_storage_bytes(bytes, limits),
        Probe::Wav { .. } => Err(MediaError::new("not an image")),
    }
}

/// [`content_key`] of a payload of `length` bytes that `payload` passes to
/// its sink in pieces, so the payload is never held whole.
pub fn content_key_streamed(
    recipe_tag: &str,
    parameters: &[u8],
    length: u64,
    payload: impl FnOnce(&mut dyn FnMut(&[u8])),
) -> [u8; 32] {
    let mut sha = Sha256::new();
    for part in [recipe_tag.as_bytes(), parameters] {
        sha.update((part.len() as u64).to_le_bytes());
        sha.update(part);
    }
    sha.update(length.to_le_bytes());
    let mut written = 0_u64;
    payload(&mut |chunk| {
        written += chunk.len() as u64;
        sha.update(chunk);
    });
    debug_assert_eq!(written, length, "a streamed payload is its declared length");
    sha.finalize().into()
}

pub fn content_key(recipe_tag: &str, parameters: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut sha = Sha256::new();
    for part in [recipe_tag.as_bytes(), parameters, payload] {
        sha.update((part.len() as u64).to_le_bytes());
        sha.update(part);
    }
    sha.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_requires_canonical_padding_and_a_bound() {
        assert_eq!(decode_base64("TWFu", 3).unwrap(), b"Man");
        assert_eq!(decode_base64("TWE=", 2).unwrap(), b"Ma");
        for bad in ["TWE", "TW=E", "TWF=", "TWE=\n", "TWE=="] {
            assert!(decode_base64(bad, 8).is_err(), "{bad}");
        }
        assert!(decode_base64("TWFu", 2).is_err());
    }

    #[test]
    fn content_key_separates_recipe_parameters_and_payload() {
        assert_ne!(content_key("a", b"bc", b"d"), content_key("ab", b"c", b"d"));
        assert_ne!(content_key("a", b"bc", b"d"), content_key("a", b"bc", b"e"));
    }

    #[test]
    fn local_file_allowlist_rejects_escape_and_network_urls() {
        let root =
            std::env::temp_dir().join(format!("runtime-media-allowlist-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("allowed")).unwrap();
        fs::write(root.join("outside.png"), b"not media").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("outside.png"), root.join("allowed/escape.png"))
            .unwrap();
        let limits = MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1024,
            maximum_audio_frames: 1024,
            maximum_frames: 1,
            maximum_tokens: 10,
            maximum_aspect_ratio: 200.0,
        };
        assert!(read_source("https://example.com/a.png", Some(&root), limits).is_err());
        assert!(
            read_source(
                &format!("file://{}", root.join("allowed/escape.png").display()),
                Some(&root.join("allowed")),
                limits
            )
            .is_err()
        );
        assert!(read_source("file:///etc/passwd", None, limits).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn local_files_preserve_bytes_and_refuse_special_files_and_replacement_symlinks() {
        use std::os::unix::{fs::symlink, net::UnixListener};
        let root =
            std::env::temp_dir().join(format!("runtime-media-regular-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let limits = MediaLimits {
            maximum_bytes: 4096,
            maximum_pixels: 1024,
            maximum_audio_frames: 1024,
            maximum_frames: 1,
            maximum_tokens: 10,
            maximum_aspect_ratio: 200.0,
        };
        let bytes = include_bytes!("../tests/fixtures/red-then-blue.mp4");
        let path = root.join("video.mp4");
        fs::write(&path, bytes).unwrap();
        let link = root.join("link.mp4");
        symlink(&path, &link).unwrap();
        for input in [&path, &link] {
            let source =
                read_source(&format!("file://{}", input.display()), Some(&root), limits).unwrap();
            assert_eq!(source.mime, "video/mp4");
            assert_eq!(source.bytes, bytes);
            assert_eq!(
                content_key("video", b"parameters", &source.bytes),
                content_key("video", b"parameters", bytes)
            );
        }
        assert_eq!(
            read_local_source(&root, &root, limits)
                .unwrap_err()
                .to_string(),
            "media path is not a regular file"
        );
        let socket = root.join("socket");
        let listener = UnixListener::bind(&socket).unwrap();
        assert!(read_local_source(&socket, &root, limits).is_err());
        // The link is accepted through canonicalization, but never when it
        // appears at the resolved path between canonicalization and open.
        assert!(read_local_source(&link, &root, limits).is_err());
        drop(listener);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    fn check_special_file_open(mode: &str) {
        use std::process::Command;
        use std::time::{Duration, Instant};
        let root = std::env::temp_dir().join(format!(
            "runtime-media-special-{}-{mode}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        assert!(
            Command::new("mkfifo")
                .arg(root.join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "core::tests::local_file_open_child",
                "--nocapture",
            ])
            .env("CHOPINFER_MEDIA_OPEN_TEST_ROOT", &root)
            .env("CHOPINFER_MEDIA_OPEN_TEST_MODE", mode)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut timed_out = false;
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                timed_out = true;
                child.kill().unwrap();
                break child.wait().unwrap();
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        fs::remove_dir_all(root).unwrap();
        assert!(
            !timed_out,
            "{mode}: local-file open blocked for two seconds"
        );
        assert!(status.success(), "{mode}: child failed: {status}");
    }

    #[cfg(unix)]
    #[test]
    fn local_fifo_open_never_waits_for_a_writer() {
        check_special_file_open("fifo");
    }

    #[cfg(unix)]
    #[test]
    fn local_file_replaced_after_resolution_never_blocks() {
        check_special_file_open("replacement");
    }

    #[cfg(unix)]
    #[test]
    fn local_file_open_child() {
        let Some(root) = std::env::var_os("CHOPINFER_MEDIA_OPEN_TEST_ROOT") else {
            return;
        };
        let root = fs::canonicalize(root).unwrap();
        let limits = MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1024,
            maximum_audio_frames: 1024,
            maximum_frames: 1,
            maximum_tokens: 10,
            maximum_aspect_ratio: 200.0,
        };
        let mode = std::env::var("CHOPINFER_MEDIA_OPEN_TEST_MODE").unwrap();
        let result = if mode == "replacement" {
            let path = root.join("input.wav");
            fs::write(&path, b"regular file before resolution").unwrap();
            let resolved = fs::canonicalize(path).unwrap();
            assert!(resolved.starts_with(&root));
            fs::rename(root.join("fifo"), &resolved).unwrap();
            read_local_source(&resolved, &root, limits)
        } else {
            read_source(
                &format!("file://{}", root.join("fifo").display()),
                Some(&root),
                limits,
            )
        };
        assert_eq!(
            result.unwrap_err().to_string(),
            "media path is not a regular file"
        );
    }
}
