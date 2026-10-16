//! Bounded precision metadata carried through existing authenticated component
//! services. Repeated operation bindings compress without changing their schema.
use super::{
    PrecisionError,
    execution::{MAX_PRECISION_DOCUMENT_BYTES, PrecisionDocument},
};

const MAGIC: &[u8; 8] = b"CHOPPRC1";
/// The existing V1 authenticated component bound, independent of serving code.
pub const MAX_COMPONENT_BYTES: usize = 1 << 20;
pub const COMPONENT_FORMAT: &str = "chop.precision.zlib.v1";

fn error(reason: impl std::fmt::Display) -> PrecisionError {
    PrecisionError::new("precision component", "encoding", reason.to_string())
}

pub fn encode(document: &PrecisionDocument) -> Result<Vec<u8>, PrecisionError> {
    document.validate()?;
    let json = serde_json::to_vec(document).map_err(error)?;
    if json.len() > MAX_PRECISION_DOCUMENT_BYTES {
        return Err(error("decoded precision component exceeds its byte bound"));
    }
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&json, 6);
    if compressed.len() + 16 > MAX_COMPONENT_BYTES {
        return Err(error(
            "encoded precision component exceeds the authenticated metadata bound",
        ));
    }
    let mut result = Vec::with_capacity(compressed.len() + 16);
    result.extend_from_slice(MAGIC);
    result.extend_from_slice(&(json.len() as u32).to_le_bytes());
    result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    Ok(result)
}

pub fn decode(bytes: &[u8]) -> Result<PrecisionDocument, PrecisionError> {
    use miniz_oxide::inflate::{
        TINFLStatus,
        core::{DecompressorOxide, decompress, inflate_flags},
    };
    if bytes.len() < 16 || bytes.len() > MAX_COMPONENT_BYTES || &bytes[..8] != MAGIC {
        return Err(error(
            "unsupported or oversized precision component envelope",
        ));
    }
    let decoded = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let encoded = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if decoded == 0 || decoded > MAX_PRECISION_DOCUMENT_BYTES || encoded != bytes.len() - 16 {
        return Err(error("precision component lengths exceed their bounds"));
    }
    let mut json = vec![0; decoded];
    let mut state = Box::<DecompressorOxide>::default();
    let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
        | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let (status, input, output) = decompress(&mut state, &bytes[16..], &mut json, 0, flags);
    if status != TINFLStatus::Done || input != encoded || output != decoded {
        return Err(error(
            "invalid, truncated or trailing compressed precision data",
        ));
    }
    let document: PrecisionDocument = serde_json::from_slice(&json).map_err(error)?;
    document.validate()?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(json: &[u8], decoded: usize) -> Vec<u8> {
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(json, 6);
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(decoded as u32).to_le_bytes());
        bytes.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&compressed);
        bytes
    }

    #[test]
    fn compressed_schema_is_strict() {
        for (json, reason) in [
            (
                br#"{"version":1,"version":2,"operations":[]}"#.as_slice(),
                "duplicate field",
            ),
            (
                br#"{"version":1,"operations":[],"unknown_policy":"fp4"}"#.as_slice(),
                "unknown field",
            ),
        ] {
            let error = decode(&envelope(json, json.len())).unwrap_err().to_string();
            assert!(error.contains(reason), "{error}");
        }
    }

    #[test]
    fn decompression_cannot_exceed_the_declared_bound() {
        let json = vec![b' '; 4096];
        let error = decode(&envelope(&json, 64)).unwrap_err().to_string();
        assert!(error.contains("compressed precision data"), "{error}");
        let error = decode(&envelope(&json, MAX_PRECISION_DOCUMENT_BYTES + 1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("lengths exceed"), "{error}");
    }
}
