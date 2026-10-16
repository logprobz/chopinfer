#![forbid(unsafe_code)]

//! `chopinfer verify --model DIR`, run as a program over small synthetic
//! checkpoints in the three layouts whose recorded digests it reads offline:
//! a manifest, a hub cache snapshot and `hf download --local-dir` metadata.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;

use runtime_weights::sha256_bytes;

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp(name: &str) -> Temp {
    let root =
        std::env::temp_dir().join(format!("checkpoint-verify-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    Temp(root)
}

fn verify(arguments: &[&str]) -> (bool, String) {
    let output: Output = std::process::Command::new(env!("CARGO_BIN_EXE_chopinfer"))
        .arg("verify")
        .args(arguments)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

/// A one-tensor safetensors file whose payload is `fill`.
fn safetensors(fill: u8, length: usize) -> Vec<u8> {
    let header = format!(
        "{{\"w\":{{\"dtype\":\"U8\",\"shape\":[{length}],\"data_offsets\":[0,{length}]}}}}"
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(std::iter::repeat_n(fill, length));
    bytes
}

/// Two shards, an index naming both, a config and a README.
fn files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("config.json", br#"{"architectures":["SyntheticForCausalLM"]}"#.to_vec()),
        ("README.md", b"synthetic\n".to_vec()),
        ("model-00001-of-00002.safetensors", safetensors(1, 9000)),
        ("model-00002-of-00002.safetensors", safetensors(2, 20000)),
        (
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
                .to_vec(),
        ),
    ]
}

fn manifest(files: &[(&str, Vec<u8>)]) -> serde_json::Value {
    files
        .iter()
        .map(|(name, bytes)| {
            (
                (*name).to_owned(),
                serde_json::json!({"sha256": sha256_bytes(bytes), "byte_length": bytes.len()}),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn write(root: &Path, files: &[(&str, Vec<u8>)]) {
    for (name, bytes) in files {
        fs::write(root.join(name), bytes).unwrap();
    }
}

/// `git hash-object`'s name for `bytes`, from the system's own SHA-1 when the
/// test machine has one; the unit tests pin the host's SHA-1 separately.
fn git_blob(bytes: &[u8]) -> Option<String> {
    let mut child = std::process::Command::new("sha1sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    use std::io::Write;
    let mut input = child.stdin.take()?;
    input
        .write_all(format!("blob {}\0", bytes.len()).as_bytes())
        .ok()?;
    input.write_all(bytes).ok()?;
    drop(input);
    let output = child.wait_with_output().ok()?;
    Some(
        String::from_utf8(output.stdout)
            .ok()?
            .split_whitespace()
            .next()?
            .to_owned(),
    )
}

#[test]
fn a_manifest_verifies_every_file_and_reports_each_mismatch() {
    let dir = temp("manifest");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir(&checkpoint).unwrap();
    let files = files();
    write(&checkpoint, &files);
    let listed = dir.0.join("files.json");
    fs::write(&listed, manifest(&files).to_string()).unwrap();
    let model = checkpoint.to_str().unwrap();
    let (passed, text) = verify(&["--model", model, "--manifest", listed.to_str().unwrap()]);
    assert!(passed, "{text}");
    assert!(
        text.contains("5 verified, 0 failed, 0 missing, 0 unrecorded"),
        "{text}"
    );
    assert!(
        text.contains("ok model-00002-of-00002.safetensors (manifest)"),
        "{text}"
    );

    // One payload byte rewritten in place: same length and header, which a
    // direct load's identity accepts, and a content mismatch here.
    let mut shard = files[3].1.clone();
    let last = shard.len() - 1;
    shard[last] ^= 1;
    fs::write(checkpoint.join(files[3].0), &shard).unwrap();
    let (passed, text) = verify(&["--model", model, "--manifest", listed.to_str().unwrap()]);
    assert!(!passed, "{text}");
    assert!(
        text.contains(&format!(
            "MISMATCH model-00002-of-00002.safetensors: recorded {}",
            sha256_bytes(&files[3].1)
        )),
        "{text}"
    );
    assert!(text.contains("4 verified, 1 failed"), "{text}");
}

#[test]
fn missing_files_and_unrecorded_weights_fail() {
    let dir = temp("missing");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir(&checkpoint).unwrap();
    let files = files();
    // The first shard is absent: the index and the manifest both list it.
    write(
        &checkpoint,
        &files
            .iter()
            .filter(|(name, _)| !name.starts_with("model-00001"))
            .cloned()
            .collect::<Vec<_>>(),
    );
    let listed = dir.0.join("files.json");
    fs::write(&listed, manifest(&files).to_string()).unwrap();
    let model = checkpoint.to_str().unwrap();
    let (passed, text) = verify(&["--model", model, "--manifest", listed.to_str().unwrap()]);
    assert!(!passed, "{text}");
    assert!(
        text.contains("MISSING model-00001-of-00002.safetensors: listed in the manifest"),
        "{text}"
    );

    // Without the manifest nothing is recorded: the index still names the
    // absent shard, and weight files fail even with --allow-unrecorded.
    let (passed, text) = verify(&["--model", model, "--allow-unrecorded"]);
    assert!(!passed, "{text}");
    assert!(
        text.contains(
            "MISSING model-00001-of-00002.safetensors: listed in model.safetensors.index.json"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "UNRECORDED model-00002-of-00002.safetensors: no recorded SHA-256 for a weight file"
        ),
        "{text}"
    );
    assert!(text.contains("unrecorded README.md"), "{text}");

    // A README missing from an otherwise complete manifest fails unless
    // --allow-unrecorded accepts it.
    write(&checkpoint, &files);
    let partial = manifest(
        &files
            .iter()
            .filter(|(name, _)| *name != "README.md")
            .cloned()
            .collect::<Vec<_>>(),
    );
    fs::write(&listed, serde_json::json!({"rev": partial}).to_string()).unwrap();
    let spec = format!("{}#rev", listed.display());
    let (passed, text) = verify(&["--model", model, "--manifest", &spec]);
    assert!(!passed, "{text}");
    assert!(text.contains("--allow-unrecorded"), "{text}");
    let (passed, text) = verify(&["--model", model, "--manifest", &spec, "--allow-unrecorded"]);
    assert!(passed, "{text}");
    assert!(
        text.contains("4 verified, 0 failed, 0 missing, 1 unrecorded"),
        "{text}"
    );
    let (passed, text) = verify(&[
        "--model",
        model,
        "--manifest",
        &format!("{}#other", listed.display()),
    ]);
    assert!(!passed && text.contains("has no entry other"), "{text}");
}

#[test]
fn a_hub_snapshot_is_checked_against_its_blob_names() {
    let dir = temp("hub");
    let repository = dir.0.join("models--org--synthetic");
    let blobs = repository.join("blobs");
    let snapshot = repository.join("snapshots/0123456789abcdef0123456789abcdef01234567");
    fs::create_dir_all(&blobs).unwrap();
    fs::create_dir_all(&snapshot).unwrap();
    let files = files();
    let mut git_named = 0;
    for (name, bytes) in &files {
        // Small git files are named by their git blob SHA-1, LFS files by
        // their SHA-256.
        let blob = match (name.ends_with(".safetensors"), git_blob(bytes)) {
            (false, Some(sha1)) => {
                git_named += 1;
                sha1
            }
            _ => sha256_bytes(bytes),
        };
        fs::write(blobs.join(&blob), bytes).unwrap();
        std::os::unix::fs::symlink(format!("../../blobs/{blob}"), snapshot.join(name)).unwrap();
    }
    let model = snapshot.to_str().unwrap();
    let (passed, text) = verify(&["--model", model]);
    assert!(passed, "{text}");
    assert!(
        text.contains("5 verified, 0 failed, 0 missing, 0 unrecorded"),
        "{text}"
    );
    assert!(text.contains("hub blob name 2"), "{text}");
    if git_named != 0 {
        assert!(
            text.contains(&format!("hub git blob name {git_named}")),
            "{text}"
        );
    }

    // A blob whose content no longer matches its name.
    let shard = sha256_bytes(&files[2].1);
    let mut bytes = files[2].1.clone();
    bytes[100] ^= 0x10;
    fs::write(blobs.join(&shard), &bytes).unwrap();
    let (passed, text) = verify(&["--model", model]);
    assert!(!passed, "{text}");
    assert!(
        text.contains(&format!(
            "MISMATCH model-00001-of-00002.safetensors: recorded {shard} (hub blob name)"
        )),
        "{text}"
    );
}

#[test]
fn hf_download_local_dir_metadata_records_etags() {
    let dir = temp("local-dir");
    let checkpoint = dir.0.join("checkpoint");
    let metadata = checkpoint.join(".cache/huggingface/download");
    fs::create_dir_all(&metadata).unwrap();
    let files = files();
    write(&checkpoint, &files);
    for (name, bytes) in &files {
        fs::write(
            metadata.join(format!("{name}.metadata")),
            format!(
                "0123456789abcdef0123456789abcdef01234567\n{}\n1700000000.0\n",
                sha256_bytes(bytes)
            ),
        )
        .unwrap();
    }
    let model = checkpoint.to_str().unwrap();
    let (passed, text) = verify(&["--model", model]);
    assert!(passed, "{text}");
    assert!(text.contains("hf download metadata 5"), "{text}");
    fs::write(checkpoint.join("config.json"), b"{}").unwrap();
    let (passed, text) = verify(&["--model", model]);
    assert!(!passed && text.contains("MISMATCH config.json"), "{text}");
}

#[test]
fn verify_refuses_a_missing_directory_and_malformed_arguments() {
    let (passed, text) = verify(&["--model", "/nonexistent/checkpoint"]);
    assert!(
        !passed && text.contains("/nonexistent/checkpoint"),
        "{text}"
    );
    for (arguments, reason) in [
        (&[][..], "verify requires --model DIR"),
        (&["--model", "x", "--manifest", "#entry"], "FILE#ENTRY"),
        (&["--model", "x", "--jobs", "0"], "--jobs takes a count"),
        (&["--model", "x", "--refresh"], "does not take --refresh"),
    ] {
        let (passed, text) = verify(arguments);
        assert!(!passed && text.contains(reason), "{arguments:?}: {text}");
    }
}

#[test]
fn a_link_out_of_the_directory_is_refused_unread() {
    let dir = temp("link");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir(&checkpoint).unwrap();
    let files = files();
    write(&checkpoint, &files);
    let outside = dir.0.join("secret.txt");
    fs::write(&outside, b"not part of the checkpoint").unwrap();
    std::os::unix::fs::symlink(&outside, checkpoint.join("linked.txt")).unwrap();
    let mut listed = manifest(&files);
    listed["linked.txt"] =
        serde_json::json!({"sha256": sha256_bytes(b"not part of the checkpoint")});
    let manifest_path = dir.0.join("files.json");
    fs::write(&manifest_path, listed.to_string()).unwrap();
    let model = checkpoint.to_str().unwrap();
    let (passed, text) = verify(&[
        "--model",
        model,
        "--manifest",
        manifest_path.to_str().unwrap(),
    ]);
    assert!(!passed, "{text}");
    assert!(!text.contains("ok linked.txt"), "{text}");
    assert!(text.contains("linked.txt: a link"), "{text}");
    // A hub snapshot accepts only links to its own blobs.
    let repository = dir.0.join("models--org--linked");
    let snapshot = repository.join("snapshots/0123456789abcdef0123456789abcdef01234567");
    fs::create_dir_all(repository.join("blobs")).unwrap();
    fs::create_dir_all(&snapshot).unwrap();
    let blob = sha256_bytes(&files[2].1);
    fs::write(repository.join("blobs").join(&blob), &files[2].1).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/{blob}"), snapshot.join(files[2].0)).unwrap();
    fs::create_dir_all(dir.0.join("elsewhere")).unwrap();
    fs::write(dir.0.join("elsewhere").join(&blob), &files[2].1).unwrap();
    std::os::unix::fs::symlink(
        format!("../../../elsewhere/{blob}"),
        snapshot.join("other.safetensors"),
    )
    .unwrap();
    let (passed, text) = verify(&["--model", snapshot.to_str().unwrap()]);
    assert!(!passed, "{text}");
    assert!(
        text.contains(&format!("ok {} (hub blob name)", files[2].0)),
        "{text}"
    );
    assert!(text.contains("other.safetensors: a link"), "{text}");
}

#[test]
fn download_metadata_outside_the_directory_records_nothing() {
    let dir = temp("metadata-link");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir_all(checkpoint.join(".cache/huggingface")).unwrap();
    let files = files();
    write(&checkpoint, &files);
    let elsewhere = dir.0.join("download");
    fs::create_dir(&elsewhere).unwrap();
    for (name, bytes) in &files {
        fs::write(
            elsewhere.join(format!("{name}.metadata")),
            format!("0\n{}\n0\n", sha256_bytes(bytes)),
        )
        .unwrap();
    }
    std::os::unix::fs::symlink(&elsewhere, checkpoint.join(".cache/huggingface/download")).unwrap();
    let (passed, text) = verify(&["--model", checkpoint.to_str().unwrap()]);
    assert!(!passed, "{text}");
    assert!(text.contains("0 verified"), "{text}");
}

#[test]
fn a_backslash_name_is_its_own_file_and_fails() {
    let dir = temp("backslash");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir_all(checkpoint.join("a")).unwrap();
    let shard = safetensors(3, 100);
    fs::write(checkpoint.join("a/model.safetensors"), &shard).unwrap();
    // A distinct file whose name holds a literal backslash.
    fs::write(checkpoint.join("a\\model.safetensors"), safetensors(4, 100)).unwrap();
    let listed = dir.0.join("files.json");
    fs::write(
        &listed,
        serde_json::json!({"a/model.safetensors": {"sha256": sha256_bytes(&shard)}}).to_string(),
    )
    .unwrap();
    let (passed, text) = verify(&[
        "--model",
        checkpoint.to_str().unwrap(),
        "--manifest",
        listed.to_str().unwrap(),
    ]);
    assert!(!passed, "{text}");
    assert!(text.contains("1 verified"), "{text}");
    assert!(text.contains("REFUSED a\\model.safetensors"), "{text}");
}

#[test]
fn a_fifo_fails_without_blocking() {
    let dir = temp("fifo");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir(&checkpoint).unwrap();
    write(&checkpoint, &files()[..4]);
    let status = std::process::Command::new("mkfifo")
        .arg(checkpoint.join("model.safetensors.index.json"))
        .status()
        .unwrap();
    assert!(status.success());
    let manifest_fifo = dir.0.join("manifest.fifo");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&manifest_fifo)
            .status()
            .unwrap()
            .success()
    );
    for arguments in [
        vec![
            "--model",
            checkpoint.to_str().unwrap(),
            "--allow-unrecorded",
        ],
        vec![
            "--model",
            checkpoint.to_str().unwrap(),
            "--manifest",
            manifest_fifo.to_str().unwrap(),
        ],
    ] {
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_chopinfer"))
            .arg("verify")
            .args(&arguments)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if std::time::Instant::now() > deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        let output = child.wait_with_output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let status = status.unwrap_or_else(|| panic!("verify blocked on a FIFO: {arguments:?}"));
        assert!(!status.success(), "{text}");
        assert!(text.contains("not a regular file"), "{text}");
    }
}

#[test]
fn a_file_added_during_hashing_fails_the_run() {
    let dir = temp("late");
    let checkpoint = dir.0.join("checkpoint");
    fs::create_dir(&checkpoint).unwrap();
    // A sparse gibibyte takes the hasher long enough to add a file meanwhile.
    let length = 1_u64 << 30;
    fs::File::create(checkpoint.join("big.bin"))
        .unwrap()
        .set_len(length)
        .unwrap();
    let mut state = runtime_weights::Sha256State::new();
    let zeros = vec![0_u8; 8 << 20];
    for _ in 0..length / zeros.len() as u64 {
        state.update(&zeros).unwrap();
    }
    let listed = dir.0.join("files.json");
    fs::write(
        &listed,
        serde_json::json!({"big.bin": {"sha256": state.finalize()}}).to_string(),
    )
    .unwrap();
    let arguments = [
        "verify",
        "--model",
        checkpoint.to_str().unwrap(),
        "--manifest",
        listed.to_str().unwrap(),
        "--allow-unrecorded",
        "--jobs",
        "1",
    ];
    let before = std::process::Command::new(env!("CARGO_BIN_EXE_chopinfer"))
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        before.status.success(),
        "{}",
        String::from_utf8_lossy(&before.stdout)
    );
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_chopinfer"))
        .args(arguments)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(150));
    fs::write(checkpoint.join("late.txt"), b"added while verify hashed").unwrap();
    let output = child.wait_with_output().unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("changed during verification"), "{text}");
}
