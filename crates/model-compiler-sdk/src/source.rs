//! Measured local source provenance shared by preparation and explicit evaluation.
use crate::files::{Result, SourceStamp, stamp};
use runtime_weights::checkpoint::CheckpointFile;
use runtime_weights::{hash_range, sha256_bytes};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshFile {
    pub stamp: SourceStamp,
    pub resolved: PathBuf,
    pub link: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freshness {
    pub files: BTreeMap<String, Option<FreshFile>>,
    pub metadata: BTreeMap<String, Option<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpan {
    pub file: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub metadata: BTreeMap<String, Option<String>>,
    pub tensors: BTreeMap<String, String>,
}

impl SourceIdentity {
    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_bytes(&serde_json::to_vec(self)?))
    }
}

/// Retains all descriptors until publication. Small inputs can be stored in an
/// imported snapshot; raw weight files are never assumed to be immutable.
pub struct SourceFiles {
    root: PathBuf,
    pub freshness: Freshness,
    pub inputs: BTreeMap<String, Vec<u8>>,
    pub metadata_bytes: u64,
    pub metadata_bytes_hashed: u64,
    files: BTreeMap<String, CheckpointFile>,
    pub headers: BTreeMap<String, runtime_weights::ShardHeader>,
    absent: Vec<String>,
}

impl SourceFiles {
    pub fn capture(
        root: &Path,
        required: &[(&str, u64)],
        optional: &[(&str, u64)],
        index: &str,
    ) -> Result<Self> {
        let mut source = Self {
            root: root.into(),
            freshness: Freshness {
                files: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            inputs: BTreeMap::new(),
            metadata_bytes: 0,
            metadata_bytes_hashed: 0,
            files: BTreeMap::new(),
            headers: BTreeMap::new(),
            absent: Vec::new(),
        };
        for (name, limit) in required {
            source.metadata(name, *limit)?;
        }
        for (name, limit) in optional {
            match std::fs::symlink_metadata(root.join(name)) {
                Ok(_) => source.metadata(name, *limit)?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    source.absent.push((*name).into());
                    source.freshness.files.insert((*name).into(), None);
                    source.freshness.metadata.insert((*name).into(), None);
                }
                Err(e) => return Err(e.into()),
            }
        }
        let bytes = source.inputs.get(index).ok_or("missing source index")?;
        let value = crate::metadata::parse(bytes, bytes.len())?;
        let shards: BTreeSet<_> = value["weight_map"]
            .as_object()
            .ok_or("missing source weight_map")?
            .values()
            .map(|v| v.as_str().ok_or("invalid shard filename"))
            .collect::<std::result::Result<_, _>>()?;
        // Each shard opens, reads and hashes its header on its own; several
        // threads share the shards and the results keep the shard order.
        let shards: Vec<&str> = shards.into_iter().collect();
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .clamp(1, 8);
        let chunk = shards.len().div_ceil(threads).max(1);
        let read: Vec<Result<_>> = std::thread::scope(|scope| {
            let workers: Vec<_> = shards
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|name| -> Result<_> {
                                let file = CheckpointFile::open(root, name)?;
                                let header = runtime_weights::read_safetensors_header(&file)?;
                                let mut bytes = vec![0; header.data_offset() as usize];
                                runtime_weights::read_exact_at(&file, &mut bytes, 0)?;
                                Ok((file, header, bytes.len() as u64, sha256_bytes(&bytes)))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("a header worker panicked"))
                .collect()
        });
        for (name, read) in shards.into_iter().zip(read) {
            let (file, header, bytes, digest) = read?;
            source.metadata_bytes += 2 * bytes;
            source.metadata_bytes_hashed += bytes;
            source.headers.insert(name.into(), header);
            source.freshness.metadata.insert(name.into(), Some(digest));
            source.insert(name, file)?;
        }
        source.verify()?;
        Ok(source)
    }

    /// Sources a caller already holds in memory: small documents by name and
    /// the safetensors headers of the shards `index` maps tensors to, each as
    /// `(file length, header bytes)`. Nothing is opened, so `file` and
    /// `measure` have no files to read; header metadata digests are recorded.
    pub fn from_memory(
        inputs: BTreeMap<String, Vec<u8>>,
        headers: &BTreeMap<String, (u64, Vec<u8>)>,
        index: &str,
    ) -> Result<Self> {
        let mut source = Self {
            root: PathBuf::new(),
            freshness: Freshness {
                files: BTreeMap::new(),
                metadata: BTreeMap::new(),
            },
            inputs,
            metadata_bytes: 0,
            metadata_bytes_hashed: 0,
            files: BTreeMap::new(),
            headers: BTreeMap::new(),
            absent: Vec::new(),
        };
        for (name, bytes) in &source.inputs {
            source.metadata_bytes += bytes.len() as u64;
            source
                .freshness
                .metadata
                .insert(name.clone(), Some(sha256_bytes(bytes)));
        }
        let bytes = source.inputs.get(index).ok_or("missing source index")?;
        let value = crate::metadata::parse(bytes, bytes.len())?;
        let shards: BTreeSet<String> = value["weight_map"]
            .as_object()
            .ok_or("missing source weight_map")?
            .values()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or("invalid shard filename")
            })
            .collect::<std::result::Result<_, _>>()?;
        // Each shard's header parses and hashes on its own; several threads
        // share the shards and the results keep the shard order.
        let shards: Vec<String> = shards.into_iter().collect();
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .clamp(1, 8);
        let chunk = shards.len().div_ceil(threads).max(1);
        let parsed: Vec<Result<(u64, String, _)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = shards
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|name| -> Result<_> {
                                let (length, header) = headers.get(name).ok_or_else(|| {
                                    format!("missing safetensors header for {name}")
                                })?;
                                let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
                                prefix.extend_from_slice(header);
                                let parsed = runtime_weights::parse_safetensors_header(
                                    &mut prefix.as_slice(),
                                    *length,
                                )?;
                                Ok((prefix.len() as u64, sha256_bytes(&prefix), parsed))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("a header worker panicked"))
                .collect()
        });
        for (name, parsed) in shards.into_iter().zip(parsed) {
            let (bytes, digest, parsed) = parsed?;
            source.metadata_bytes += bytes;
            source.freshness.metadata.insert(name.clone(), Some(digest));
            source.headers.insert(name, parsed);
        }
        Ok(source)
    }

    fn insert(&mut self, name: &str, file: CheckpointFile) -> Result<()> {
        file.verify()?;
        let identity = FreshFile {
            stamp: stamp(&file)?,
            resolved: file.resolved_path().into(),
            link: file.link_target().map(Path::to_path_buf),
        };
        if self.files.insert(name.into(), file).is_some() {
            return Err("duplicate source input".into());
        }
        self.freshness.files.insert(name.into(), Some(identity));
        Ok(())
    }

    fn metadata(&mut self, name: &str, limit: u64) -> Result<()> {
        let file = CheckpointFile::open(&self.root, name)?;
        let bytes = file.read_bounded(limit)?;
        self.metadata_bytes += bytes.len() as u64;
        self.metadata_bytes_hashed += bytes.len() as u64;
        self.freshness
            .metadata
            .insert(name.into(), Some(sha256_bytes(&bytes)));
        self.inputs.insert(name.into(), bytes);
        self.insert(name, file)
    }

    pub fn file(&self, name: &str) -> Result<&CheckpointFile> {
        let file = self
            .files
            .get(name)
            .ok_or("source file outside captured set")?;
        file.verify()?;
        Ok(file)
    }

    pub fn verify(&self) -> Result<()> {
        for file in self.files.values() {
            file.verify()?;
        }
        for name in &self.absent {
            match std::fs::symlink_metadata(self.root.join(name)) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                _ => return Err(format!("optional source input changed: {name}").into()),
            }
        }
        Ok(())
    }

    /// Reuse digests from this version's packing receipts, then measure excluded
    /// tensors too. Full provenance includes payloads not used by the runtime.
    pub fn measure(
        &self,
        spans: &BTreeMap<String, TensorSpan>,
        packed: &BTreeMap<String, String>,
    ) -> Result<(SourceIdentity, u64)> {
        self.verify()?;
        if packed.keys().any(|name| !spans.contains_key(name)) {
            return Err("unexpected packed source receipt".into());
        }
        let mut tensors = BTreeMap::new();
        let mut measured = 0;
        for (name, span) in spans {
            let digest = if let Some(digest) = packed.get(name) {
                if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("invalid packed source digest".into());
                }
                digest.clone()
            } else {
                measured += span.bytes;
                hash_range(self.file(&span.file)?, span.offset, span.bytes)?
            };
            tensors.insert(name.clone(), digest);
        }
        self.verify()?;
        Ok((
            SourceIdentity {
                metadata: self.freshness.metadata.clone(),
                tensors,
            },
            measured,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// More shards than header threads: both readers keep every shard's own
    /// header and digest, in shard order, and report the first missing shard.
    #[test]
    fn many_shards_read_in_order_by_both_readers() {
        let root = std::env::temp_dir().join(format!("source-shards-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let mut weight_map = serde_json::Map::new();
        let mut headers = BTreeMap::new();
        for shard in 0..23_u64 {
            let name = format!("shard-{shard:02}.safetensors");
            let tensor = format!("t{shard}");
            let header = format!(
                r#"{{"{tensor}":{{"dtype":"U8","shape":[{n}],"data_offsets":[0,{n}]}}}}"#,
                n = shard + 1
            );
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(std::iter::repeat_n(shard as u8, shard as usize + 1));
            fs::write(root.join(&name), &bytes).unwrap();
            headers.insert(name.clone(), (bytes.len() as u64, header.into_bytes()));
            weight_map.insert(tensor, name.into());
        }
        let index = serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap();
        fs::write(root.join("index"), &index).unwrap();
        let captured = SourceFiles::capture(&root, &[("index", 1 << 20)], &[], "index").unwrap();
        let inputs = BTreeMap::from([("index".to_owned(), index)]);
        let memory = SourceFiles::from_memory(inputs.clone(), &headers, "index").unwrap();
        for source in [&captured, &memory] {
            assert_eq!(
                source.headers.keys().collect::<Vec<_>>(),
                headers.keys().collect::<Vec<_>>()
            );
            for (shard, (name, (_, header))) in headers.iter().enumerate() {
                let tensors = source.headers[name].tensors();
                assert_eq!(tensors.keys().collect::<Vec<_>>(), [&format!("t{shard}")]);
                let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
                prefix.extend_from_slice(header);
                assert_eq!(
                    source.freshness.metadata[name].as_deref(),
                    Some(sha256_bytes(&prefix).as_str())
                );
            }
        }
        assert_eq!(captured.freshness.metadata, memory.freshness.metadata);
        // Two shards lack headers: the earlier one in shard order is named.
        let mut missing = headers.clone();
        missing.remove("shard-05.safetensors");
        missing.remove("shard-19.safetensors");
        let error = SourceFiles::from_memory(inputs, &missing, "index")
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("shard-05.safetensors"), "{error}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn packing_receipts_avoid_a_second_pass_but_excluded_payloads_keep_provenance() {
        let root = std::env::temp_dir().join(format!("source-provenance-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let header = br#"{"used":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"excluded":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        fs::write(root.join("weights"), &bytes).unwrap();
        fs::write(
            root.join("index"),
            br#"{"weight_map":{"used":"weights","excluded":"weights"}}"#,
        )
        .unwrap();
        let source =
            SourceFiles::capture(&root, &[("index", 1024)], &[("optional", 1024)], "index")
                .unwrap();
        let spans = BTreeMap::from([
            (
                "used".into(),
                TensorSpan {
                    file: "weights".into(),
                    offset: 8 + header.len() as u64,
                    bytes: 2,
                },
            ),
            (
                "excluded".into(),
                TensorSpan {
                    file: "weights".into(),
                    offset: 10 + header.len() as u64,
                    bytes: 2,
                },
            ),
        ]);
        let packed = BTreeMap::from([("used".into(), sha256_bytes(&[1, 2]))]);
        let (first, count) = source.measure(&spans, &packed).unwrap();
        assert_eq!(count, 2);
        let (full, count) = source.measure(&spans, &BTreeMap::new()).unwrap();
        assert_eq!(count, 4);
        assert_eq!(first, full);
        *bytes.last_mut().unwrap() = 5;
        fs::write(root.join("replacement"), &bytes).unwrap();
        fs::rename(root.join("replacement"), root.join("weights")).unwrap();
        assert!(source.verify().is_err());
        let changed =
            SourceFiles::capture(&root, &[("index", 1024)], &[("optional", 1024)], "index")
                .unwrap();
        assert_ne!(
            first.sha256().unwrap(),
            changed
                .measure(&spans, &packed)
                .unwrap()
                .0
                .sha256()
                .unwrap()
        );
        fs::write(root.join("optional"), b"added").unwrap();
        assert!(changed.verify().is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
