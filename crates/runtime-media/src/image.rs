//! Image formats behind one interface: sniffing by signature, probing the
//! size from headers, the decoder's working set before it allocates, and
//! decoding to RGB under a transparency and orientation policy.

use crate::{Alpha, MediaError, MediaLimits, MediaResult, Orientation, RgbImage};

/// An encoded image format a modality may accept.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Webp,
}

impl ImageFormat {
    pub const ALL: [Self; 4] = [Self::Png, Self::Jpeg, Self::Gif, Self::Webp];

    /// The name a modality declares the format by.
    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.name() == name)
    }

    pub fn mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }

    /// The format a media type names: type and subtype compare without
    /// regard to case, parameters are ignored and `image/jpg` is JPEG.
    pub fn from_mime(mime: &str) -> Option<Self> {
        match crate::core::media_type(mime).as_str() {
            "image/jpg" => Some(Self::Jpeg),
            essence => Self::ALL
                .into_iter()
                .find(|format| format.mime() == essence),
        }
    }

    /// The format an encoded image's signature names.
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            Some(Self::Png)
        } else if bytes.starts_with(b"\xff\xd8") {
            Some(Self::Jpeg)
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            Some(Self::Gif)
        } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
            Some(Self::Webp)
        } else {
            None
        }
    }
}

/// How a decoder treats transparency and an EXIF orientation tag.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImagePolicy {
    pub alpha: Alpha,
    pub orientation: Orientation,
}

fn sniffed(bytes: &[u8]) -> MediaResult<ImageFormat> {
    ImageFormat::sniff(bytes).ok_or_else(|| MediaError::new("unsupported image header"))
}

/// The format and size of an encoded image under `orientation`, from its
/// headers, with the size checked against `limits`.
pub fn probe_image(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<(ImageFormat, u32, u32)> {
    limits.check_bytes(bytes.len())?;
    let format = sniffed(bytes)?;
    let (width, height) = match format {
        ImageFormat::Png => crate::png::probe_oriented(bytes, limits, orientation)?,
        ImageFormat::Jpeg => crate::jpeg::probe_oriented(bytes, limits, orientation)?,
        ImageFormat::Gif => crate::gif::probe(bytes, limits)?,
        ImageFormat::Webp => crate::webp::probe(bytes, limits, orientation)?,
    };
    Ok((format, width, height))
}

/// The host bytes an image's decode holds at its peak, from its headers
/// before anything is allocated: the decoder's buffers and the RGB image,
/// but nothing of the recipe that follows.
pub fn decode_working_bytes(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<u64> {
    limits.check_bytes(bytes.len())?;
    match sniffed(bytes)? {
        ImageFormat::Png => crate::png::decode_storage_bytes(bytes, limits, orientation),
        ImageFormat::Jpeg => crate::jpeg::decode_storage_bytes(bytes, limits),
        ImageFormat::Gif => crate::gif::decode_storage_bytes(bytes, limits),
        ImageFormat::Webp => crate::webp::decode_storage_bytes(bytes, limits, orientation),
    }
}

/// Decodes an image by the format its signature names.
pub fn decode_image(
    bytes: &[u8],
    limits: MediaLimits,
    policy: ImagePolicy,
) -> MediaResult<RgbImage> {
    match sniffed(bytes)? {
        ImageFormat::Png => {
            crate::png::decode_oriented(bytes, limits, policy.alpha, policy.orientation)
        }
        ImageFormat::Jpeg => crate::jpeg::decode_oriented(bytes, limits, policy.orientation),
        ImageFormat::Gif => crate::gif::decode(bytes, limits, policy.alpha),
        ImageFormat::Webp => crate::webp::decode(bytes, limits, policy.alpha, policy.orientation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_types_and_signatures_name_formats() {
        for (mime, format) in [
            ("image/png", Some(ImageFormat::Png)),
            ("IMAGE/JPG", Some(ImageFormat::Jpeg)),
            ("image/Jpeg; q=1", Some(ImageFormat::Jpeg)),
            ("image/gif", Some(ImageFormat::Gif)),
            ("image/webp", Some(ImageFormat::Webp)),
            ("image/svg+xml", None),
            ("audio/wav", None),
        ] {
            assert_eq!(ImageFormat::from_mime(mime), format, "{mime}");
        }
        for format in ImageFormat::ALL {
            assert_eq!(ImageFormat::from_name(format.name()), Some(format));
            assert_eq!(ImageFormat::from_mime(format.mime()), Some(format));
        }
        assert_eq!(
            ImageFormat::sniff(b"RIFF\x04\0\0\0WEBPVP8 "),
            Some(ImageFormat::Webp)
        );
        assert_eq!(ImageFormat::sniff(b"RIFF\x04\0\0\0WAVEfmt "), None);
        assert_eq!(ImageFormat::sniff(b"GIF89a"), Some(ImageFormat::Gif));
        assert_eq!(ImageFormat::sniff(b"GIF88a"), None);
    }
}
