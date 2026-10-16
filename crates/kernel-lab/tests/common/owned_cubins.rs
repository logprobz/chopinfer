//! Load production artifacts without linking a bundle's exported C ABI.
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

fn read(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|e| e.to_string())?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("artifact is not a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("artifact exceeds size limit".into());
    }
    Ok(bytes)
}

pub fn load(name: &str) -> Result<Vec<u8>, String> {
    let path = std::env::var_os("CHOPINFER_KERNEL_OWNED_CUBINS").ok_or_else(|| {
        format!(
            "this GPU test needs CHOPINFER_KERNEL_OWNED_CUBINS: a JSON manifest \
             {{\"cubins\": {{\"<name>\": {{\"path\": \"<cubin path>\", \
             \"sha256\": \"<64 hex digest>\"}}}}}} of a pinned bundle build's cubins, here \
             {name}; no in-repo script writes it, so a GPU run fails without it"
        )
    })?;
    load_manifest(Path::new(&path), name)
}

fn load_manifest(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&read(path, 1 << 20)?).map_err(|e| e.to_string())?;
    let artifact = &manifest["cubins"][name];
    let path = artifact["path"].as_str().ok_or("missing artifact path")?;
    let expected = artifact["sha256"]
        .as_str()
        .ok_or("missing artifact digest")?;
    let bytes = read(Path::new(path), 32 << 20)?;
    if kernel_build::digest(&bytes) != expected {
        return Err(format!("production artifact digest mismatch: {name}"));
    }
    Ok(bytes)
}

#[test]
fn artifact_digest_and_no_follow() {
    let dir = std::env::temp_dir().join(format!("owned-cubins-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let file = dir.join("artifact.cubin");
    let manifest = dir.join("manifest.json");
    std::fs::write(&file, b"production bytes").unwrap();
    let value = serde_json::json!({"cubins": {"artifact": {
        "path": file, "sha256": kernel_build::digest(b"production bytes")
    }}});
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(
        load_manifest(&manifest, "artifact").unwrap(),
        b"production bytes"
    );
    std::fs::write(&file, b"different bytes").unwrap();
    assert!(
        load_manifest(&manifest, "artifact")
            .unwrap_err()
            .contains("digest mismatch")
    );
    assert!(load_manifest(&manifest, "missing").is_err());
    std::fs::remove_file(&file).unwrap();
    std::os::unix::fs::symlink(&manifest, &file).unwrap();
    assert!(load_manifest(&manifest, "artifact").is_err());
    assert!(load_manifest(&file, "artifact").is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
