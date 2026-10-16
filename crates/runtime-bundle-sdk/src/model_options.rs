//! The capability-gated model options create tail.
use std::collections::BTreeMap;

use runtime_bundle_abi::{BundleStatusV1, MODEL_OPTIONS_TAG_V1};
use serde_json::Value;

use crate::BatchReaderV1;

/// Reads the tagged JSON object at the current create-record cursor. The
/// bundle must check its complete declaration and compiled limits before use.
pub fn read(reader: &mut BatchReaderV1<'_>) -> Result<BTreeMap<String, Value>, BundleStatusV1> {
    reader
        .expect_magic(MODEL_OPTIONS_TAG_V1)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let count = reader.u32().map_err(|_| BundleStatusV1::InvalidArgument)? as usize;
    if count > 64 * 1024 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let bytes = reader
        .take(count)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let values: BTreeMap<String, Value> =
        serde_json::from_slice(bytes).map_err(|_| BundleStatusV1::InvalidArgument)?;
    if values.is_empty()
        || values.len() > 64
        || values.iter().any(|(name, value)| {
            name.len() > 128 || value.is_null() || value.is_array() || value.is_object()
        })
    {
        return Err(BundleStatusV1::InvalidArgument);
    }
    Ok(values)
}
