//! The checkpoint as a weight plan request sees it, read at preparation:
//! every safetensors file's name, length and header, and the documents the
//! encoder plan reads. A package checks the vision tower from the same
//! headers the host's plan request later carries before it plans the
//! encoder image. Nothing here reads a tensor payload. Every file is
//! opened as direct preparation and the host open it
//! (`runtime_weights::checkpoint::CheckpointFile`): a regular file in the
//! checkpoint directory, or a confined HF snapshot link to its blob.

use model_compiler_sdk::files::Result;
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use runtime_weights::checkpoint::CheckpointFile;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// The documents the encoder plan reads: the config and the weight map.
const DOCUMENTS: [&str; 2] = ["config.json", "model.safetensors.index.json"];

/// The largest safetensors header read, as the host bounds it.
const MAX_HEADER_BYTES: u64 = 256 << 20;

/// The largest document read.
const MAX_DOCUMENT_BYTES: u64 = 64 << 20;

/// The checkpoint file `name` (`file` or `dir/file`) of `root`, opened as
/// direct preparation opens it.
pub(crate) fn open(root: &Path, name: &str) -> Result<CheckpointFile> {
    let (parent, file) = match name.split_once('/') {
        Some((directory, file)) => (root.join(directory), file),
        None => (root.to_path_buf(), name),
    };
    Ok(CheckpointFile::open(&parent, file).map_err(|e| format!("{name}: {e}"))?)
}

/// The safetensors files a direct load opens: those of `root` and of its
/// immediate subdirectories other than hidden ones, named `dir/file`, in
/// name order.
fn safetensors_names(root: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "checkpoint name is not UTF-8")?;
        if !entry.file_type()?.is_dir() {
            names.push(name);
        } else if !name.starts_with('.') {
            for inner in std::fs::read_dir(entry.path())? {
                let inner = inner?;
                if !inner.file_type()?.is_dir() {
                    let file = inner
                        .file_name()
                        .into_string()
                        .map_err(|_| "checkpoint name is not UTF-8")?;
                    names.push(format!("{name}/{file}"));
                }
            }
        }
    }
    names.retain(|name| name.ends_with(".safetensors"));
    names.sort();
    Ok(names)
}

/// The file `name` of `root` with its header, checked as the host's
/// checkpoint opener checks it (`runtime_weights::parse_safetensors_header`
/// over the file's length): every tensor's offsets inside the file, the
/// spans covering its data without a gap or an overlap.
fn file(root: &Path, name: &str) -> Result<CheckpointFileV1> {
    let file = open(root, name)?;
    let byte_length = file.metadata()?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| format!("{name}: {e}"))?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0 || header_length > MAX_HEADER_BYTES || 8 + header_length > byte_length {
        return Err(format!("{name}: safetensors header length is outside the file").into());
    }
    let mut header = vec![0; header_length as usize];
    file.read_exact_at(&mut header, 8)?;
    let mut bytes = prefix.to_vec();
    bytes.extend_from_slice(&header);
    runtime_weights::parse_safetensors_header(&mut bytes.as_slice(), byte_length)
        .map_err(|e| format!("{name}: {e}"))?;
    Ok(CheckpointFileV1 {
        name: name.into(),
        byte_length,
        data_offset: 8 + header_length,
        header,
    })
}

/// The checkpoint document `name` of `root`, read within `limit` bytes.
pub(crate) fn document(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    Ok(open(root, name)?
        .read_bounded(limit)
        .map_err(|e| format!("{name}: {e}"))?)
}

/// The plan request for rank 0 of `ranks` of the checkpoint at `root`: its
/// safetensors headers and the documents the encoder plan reads, those the
/// checkpoint has.
pub fn request(root: &Path, ranks: u32) -> Result<WeightPlanRequestV1> {
    let files = safetensors_names(root)?
        .iter()
        .map(|name| file(root, name))
        .collect::<Result<Vec<_>>>()?;
    let mut documents = Vec::new();
    for name in DOCUMENTS {
        if std::fs::symlink_metadata(root.join(name))
            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        {
            continue;
        }
        let bytes = document(root, name, MAX_DOCUMENT_BYTES)?;
        documents.push(NamedBytesV1 {
            name: name.into(),
            bytes,
        });
    }
    Ok(WeightPlanRequestV1 {
        ranks,
        files,
        documents,
        ..Default::default()
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The length of a safetensors file holding `header` whose data ends
    /// where its last tensor does.
    pub(crate) fn logical_length(header: &[u8]) -> u64 {
        let parsed: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header).unwrap();
        let end = parsed
            .values()
            .filter_map(|tensor| tensor["data_offsets"][1].as_u64())
            .max()
            .unwrap_or(0);
        8 + header.len() as u64 + end
    }

    /// Writes `header` as the safetensors file `path`, sparse at `length`
    /// bytes: no tensor bytes are written.
    pub(crate) fn write_sparse(path: &Path, header: &[u8], length: u64) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(path, bytes).unwrap();
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(length)
            .unwrap();
    }

    /// Writes the headers of `request`'s files under `root`, each a sparse
    /// safetensors file ending where its last tensor does, and its
    /// documents.
    pub(crate) fn write_headers(root: &Path, request: &WeightPlanRequestV1) {
        for file in &request.files {
            write_sparse(
                &root.join(&file.name),
                &file.header,
                logical_length(&file.header),
            );
        }
        for document in &request.documents {
            std::fs::write(root.join(&document.name), &document.bytes).unwrap();
        }
    }

    /// Rewrites `request`'s weight map to assign exactly the tensors its
    /// files' headers hold, each to its file.
    pub(crate) fn index_headers(request: &mut WeightPlanRequestV1) {
        let mut map = serde_json::Map::new();
        for file in &request.files {
            let header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&file.header).unwrap();
            for name in header.keys().filter(|name| *name != "__metadata__") {
                map.insert(name.clone(), file.name.clone().into());
            }
        }
        let bytes = serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap();
        request
            .documents
            .retain(|document| document.name != DOCUMENTS[1]);
        request.documents.push(NamedBytesV1 {
            name: DOCUMENTS[1].into(),
            bytes,
        });
    }

    /// A FIFO in place of a shard and a link out of the checkpoint
    /// directory are refused as direct preparation refuses them, before
    /// either is opened.
    #[test]
    fn refuses_what_direct_preparation_refuses() {
        let root = std::env::temp_dir().join(format!("glm5next-unsafe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let fifo = root.join("weights.safetensors");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        let error = request(&root, 4).unwrap_err().to_string();
        assert!(
            error.contains("weights.safetensors: source is not a regular file"),
            "{error}"
        );
        std::fs::remove_file(&fifo).unwrap();
        let outside = std::env::temp_dir().join(format!(
            "glm5next-outside-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&outside, [2, 0, 0, 0, 0, 0, 0, 0, b'{', b'}']).unwrap();
        std::os::unix::fs::symlink(&outside, &fifo).unwrap();
        let error = request(&root, 4).unwrap_err().to_string();
        assert!(
            error.contains("links are permitted only in a confined HF snapshot"),
            "{error}"
        );
        std::fs::remove_file(&outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A tensor whose offsets leave the file, or overlap another tensor's, is
    /// refused when the header is read, as the host's checkpoint opener
    /// refuses it, though its dtype, shape and span length are right.
    #[test]
    fn refuses_offsets_outside_the_file_or_overlapping() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let request = model_glm5next_contract::direct::test_support::request(0, |_| {});
        let length = logical_length(&request.files[0].header);
        let payload = length - 8 - request.files[0].header.len() as u64;
        for (case, expected) in [("outside", "gap"), ("overlap", "overlap")] {
            let root = std::env::temp_dir()
                .join(format!("glm5next-offsets-{case}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let mut header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&request.files[0].header).unwrap();
            let offsets = &header[KV_B]["data_offsets"];
            let span = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
            let start = if case == "outside" { payload + 128 } else { 0 };
            header[KV_B]["data_offsets"] = serde_json::json!([start, start + span]);
            let bytes = serde_json::to_vec(&header).unwrap();
            // The file keeps the payload length the valid header covers.
            let path = root.join(&request.files[0].name);
            write_sparse(&path, &bytes, 8 + bytes.len() as u64 + payload);
            let error = super::request(&root, 4).unwrap_err().to_string();
            assert!(
                error.starts_with("model.safetensors: ") && error.contains(expected),
                "{case}: {error}"
            );
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// The headers and documents read back are the ones written, files in
    /// subdirectories included; a header longer than its file is refused.
    #[test]
    fn reads_every_header_and_the_plan_documents() {
        let root = std::env::temp_dir().join(format!("glm5next-headers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        let mut written = model_glm5next_contract::direct::test_support::request(0, |_| {});
        let mut nested = written.files[0].clone();
        nested.name = "part/b.safetensors".into();
        nested.header = br#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        written.files.push(nested);
        write_headers(&root, &written);
        std::fs::write(root.join(".hidden/c.safetensors"), b"ignored").unwrap();
        std::fs::write(root.join("notes.txt"), b"ignored").unwrap();
        let read = request(&root, 4).unwrap();
        assert_eq!(read.ranks, 4);
        assert_eq!(read.files.len(), 2);
        for (read, written) in read.files.iter().zip(&written.files) {
            assert_eq!((&read.name, &read.header), (&written.name, &written.header));
            assert_eq!(read.data_offset, 8 + written.header.len() as u64);
            assert_eq!(read.byte_length, logical_length(&written.header));
        }
        assert_eq!(read.documents, written.documents);
        std::fs::write(root.join("short.safetensors"), 64_u64.to_le_bytes()).unwrap();
        let error = request(&root, 4).unwrap_err().to_string();
        assert!(
            error.contains("short.safetensors: safetensors header length"),
            "{error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
