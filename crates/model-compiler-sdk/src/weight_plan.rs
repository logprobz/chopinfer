//! Scaffolding of a bundle's weight plan (`chop_bundle_weight_plan_v1`): the
//! cache of what it reads from a request, the record sources of the request's
//! files and a plan's identity. What a plan places where stays the model's.
use runtime_bundle_abi::weight_plan::{RecordSourceV1, WeightPlanRequestV1, WeightPlanV1};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A digest of what a plan reads from its request: hashing these instead of
/// the whole request (the tokenizer and megabytes of headers) keeps a cache
/// hit cheap.
pub struct RequestKey(Sha256);

impl RequestKey {
    /// Starts with `purpose`, the checkpoint identity (the SHA-256 of every
    /// file's name, length and header) and the file count.
    pub fn new(purpose: &[u8], request: &WeightPlanRequestV1) -> Self {
        let mut digest = Sha256::new();
        digest.update(purpose);
        digest.update(request.checkpoint_identity);
        digest.update((request.files.len() as u64).to_le_bytes());
        Self(digest)
    }

    /// Adds `bytes` with their length.
    pub fn bytes(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    /// Adds `bytes` when present, with their length, and their absence
    /// otherwise.
    pub fn optional(&mut self, bytes: Option<&[u8]>) {
        match bytes {
            Some(bytes) => {
                self.0.update([1]);
                self.bytes(bytes);
            }
            None => self.0.update([0]),
        }
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

/// The value built for the last request, by its [`RequestKey`]: the host
/// asks once per image and rank of the same checkpoint.
pub struct RequestCache<T> {
    last: Mutex<Option<([u8; 32], Arc<T>)>>,
}

impl<T> Default for RequestCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RequestCache<T> {
    pub const fn new() -> Self {
        Self {
            last: Mutex::new(None),
        }
    }

    /// The cached value of `key`, or the one `build` makes, which is kept.
    /// Concurrent requests build one at a time.
    pub fn get_or_build<E: From<&'static str>>(
        &self,
        key: [u8; 32],
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<Arc<T>, E> {
        let mut last = self.last.lock().map_err(|_| "weight plan cache poisoned")?;
        if let Some((cached, value)) = last.as_ref()
            && *cached == key
        {
            return Ok(value.clone());
        }
        let value = Arc::new(build()?);
        *last = Some((key, value.clone()));
        Ok(value)
    }
}

/// The request's checkpoint files by name, as plan records cite them.
pub struct RequestFiles<'a> {
    positions: BTreeMap<&'a str, u32>,
}

impl<'a> RequestFiles<'a> {
    pub fn new(request: &'a WeightPlanRequestV1) -> Self {
        Self {
            positions: request
                .files
                .iter()
                .enumerate()
                .map(|(position, file)| (file.name.as_str(), position as u32))
                .collect(),
        }
    }

    /// The record source of `tensor`, stored in the checkpoint file `file`.
    pub fn source(&self, tensor: &str, file: &str) -> Result<RecordSourceV1, String> {
        Ok(RecordSourceV1 {
            file: *self
                .positions
                .get(file)
                .ok_or_else(|| format!("tensor {tensor}: file {file} is not in the checkpoint"))?,
            tensor: tensor.into(),
        })
    }
}

/// What `plan` loads for `request`, not the bytes of any image: the host
/// serves the same identity for the planned rank, and the plan's rank index
/// names it in place of an image digest.
pub fn plan_identity(plan: &WeightPlanV1, request: &WeightPlanRequestV1) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            plan.identity_document(&request.checkpoint_identity, &request.implementation)
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::weight_plan::CheckpointFileV1;

    /// One build per key, and a file outside the request is refused.
    #[test]
    fn the_cache_builds_once_per_key_and_sources_name_request_files() {
        let request = WeightPlanRequestV1 {
            files: ["a.safetensors", "b.safetensors"]
                .map(|name| CheckpointFileV1 {
                    name: name.into(),
                    ..Default::default()
                })
                .to_vec(),
            ..Default::default()
        };
        let cache = RequestCache::new();
        let mut builds = 0;
        let mut get = |purpose: &[u8]| {
            cache
                .get_or_build(RequestKey::new(purpose, &request).finish(), || {
                    builds += 1;
                    Ok::<_, String>(purpose.to_vec())
                })
                .unwrap()
        };
        let first = get(b"x");
        assert!(Arc::ptr_eq(&first, &get(b"x")));
        assert_eq!(*get(b"y"), b"y");
        assert_eq!(builds, 2);
        let files = RequestFiles::new(&request);
        assert_eq!(files.source("t", "b.safetensors").unwrap().file, 1);
        assert!(
            files
                .source("t", "c.safetensors")
                .unwrap_err()
                .contains("tensor t: file c.safetensors is not in the checkpoint")
        );
    }
}
