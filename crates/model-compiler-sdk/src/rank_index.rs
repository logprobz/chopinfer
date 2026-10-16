//! Model-neutral rank-image indexes. Layout and schema IDs remain model-owned.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ViewIndex {
    pub name: String,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout_id: String,
    pub source_tensor_digests: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RankIndex {
    pub schema: String,
    pub rank: usize,
    pub image: String,
    pub image_bytes: u64,
    pub sha256: String,
    pub views: Vec<ViewIndex>,
}

/// Read bounded metadata from a regular file without following its final symlink.
pub fn read(path: &std::path::Path, limit: u64) -> Result<Vec<u8>, crate::CompilerError> {
    use std::io::Read;
    let run = || -> std::io::Result<Vec<u8>> {
        let file = runtime_weights::open_nofollow(path)?;
        if file.metadata()?.len() > limit {
            return Err(std::io::Error::other("index is not a bounded regular file"));
        }
        let mut bytes = Vec::new();
        file.take(limit + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit {
            return Err(std::io::Error::other("index exceeds limit"));
        }
        Ok(bytes)
    };
    run().map_err(|error| {
        crate::CompilerError::validation(format!("read {}: {error}", path.display()))
    })
}

pub fn compact(bytes: &[u8], limit: usize) -> Result<Vec<u8>, crate::CompilerError> {
    let mut index: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| crate::CompilerError::validation(e.to_string()))?;
    for view in index["views"]
        .as_array_mut()
        .ok_or_else(|| crate::CompilerError::validation("index needs views"))?
    {
        view.as_object_mut()
            .ok_or_else(|| crate::CompilerError::validation("view needs object"))?
            .remove("source_tensor_digests");
    }
    let bytes =
        serde_json::to_vec(&index).map_err(|e| crate::CompilerError::validation(e.to_string()))?;
    if bytes.len() > limit {
        return Err(crate::CompilerError::validation(
            "runtime index exceeds metadata limit",
        ));
    }
    Ok(bytes)
}
