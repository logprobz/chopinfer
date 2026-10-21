//! Metadata reuse retains the original file identities through preparation.
use model_deepseek41::checkpoint::Snapshot;
use std::path::PathBuf;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "checkpoint-snapshot-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir(&root).unwrap();
        let header = br#"{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0xa5, 0x5a]);
        std::fs::write(root.join("model.safetensors"), bytes).unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            br#"{"weight_map":{"t":"model.safetensors"}}"#,
        )
        .unwrap();
        Self(root)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn ranks_reuse_the_same_headers_and_refuse_source_replacement() {
    let fixture = Fixture::new();
    let mut snapshot = Snapshot::open(&fixture.0).unwrap();
    let first = snapshot.request(2, 0);
    let header = first.files[0].header.as_ptr();
    let documents = first.documents.as_ptr();
    let identity = first.checkpoint_identity;
    for (ranks, rank) in [(2, 1), (4, 0), (4, 3)] {
        let next = snapshot.request(ranks, rank);
        assert_eq!(next.files[0].header.as_ptr(), header);
        assert_eq!(next.documents.as_ptr(), documents);
        assert_eq!(next.checkpoint_identity, identity);
    }
    snapshot.verify().unwrap();
    let path = fixture.0.join("model.safetensors");
    let replacement = fixture.0.join("replacement");
    std::fs::copy(&path, &replacement).unwrap();
    std::fs::rename(&replacement, &path).unwrap();
    assert!(
        snapshot.verify().is_err(),
        "even identical replacement bytes change the source"
    );
}

#[test]
fn changed_documents_cannot_publish_a_prepared_snapshot() {
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ] {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join("tokenizer.json"), b"{}").unwrap();
        let mut snapshot = Snapshot::open(&fixture.0).unwrap();
        snapshot
            .track_documents(&fixture.0, &["tokenizer.json"])
            .unwrap();
        std::fs::write(fixture.0.join(name), b"changed").unwrap();
        assert!(snapshot.verify().is_err(), "{name}");
    }
}

#[test]
fn snapshot_refuses_a_header_not_owned_by_the_weight_map() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.0.join("model.safetensors.index.json"),
        br#"{"weight_map":{"missing":"model.safetensors"}}"#,
    )
    .unwrap();
    assert!(Snapshot::open(&fixture.0).is_err());
}

#[test]
fn standalone_planning_reads_only_the_indexed_checkpoint_inputs() {
    let fixture = Fixture::new();
    let expected = model_deepseek41::checkpoint::request(&fixture.0, 4, 0).unwrap();
    let extra = fixture.0.join("unrelated.json");
    std::fs::File::create(extra)
        .unwrap()
        .set_len(65 << 20)
        .unwrap();
    std::fs::write(fixture.0.join("unindexed.safetensors"), b"bad header").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("missing-unrelated-file", fixture.0.join("unrelated-link")).unwrap();
    let actual = model_deepseek41::checkpoint::request(&fixture.0, 4, 0).unwrap();
    assert_eq!(actual.checkpoint_identity, expected.checkpoint_identity);
    assert_eq!(actual.files, expected.files);
    assert_eq!(actual.documents, expected.documents);
    assert_eq!(
        model_deepseek41::checkpoint::metadata(&fixture.0)
            .unwrap()
            .tensors
            .len(),
        1
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_shared_mapping_cannot_change_the_owned_preparation_metadata() {
    use model_compiler_sdk::preparation::direct::CheckpointSnapshot;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::fs::MetadataExt;
    use std::process::{Child, Command, Stdio};
    use std::sync::Arc;
    struct Mapping(Child);
    impl Drop for Mapping {
        fn drop(&mut self) {
            // Closing stdin lets the tiny mapping owner release its mapping.
            self.0.stdin.take();
            let _ = self.0.wait();
        }
    }
    let fixture = Fixture::new();
    let config = fixture.0.join("config.json");
    let original = br#"{"option":"A"}"#;
    std::fs::write(&config, original).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&config)
        .unwrap();
    // Keep the mmap operation in a subprocess: this compiler crate admits
    // no unsafe Rust, including tests. Python is a CPU-gate prerequisite.
    let mut mapping = Mapping(
        Command::new("python3")
            .args([
                "-B",
                "-u",
                "-c",
                r#"
import mmap,sys
with open(sys.argv[1], 'r+b') as file, mmap.mmap(file.fileno(), 0) as page:
    page[11:12] = b'A'
    print('ready', flush=True)
    for line in sys.stdin:
        page[11:12] = line.strip().encode('ascii')
        print('written', flush=True)
"#,
            ])
            .arg(&config)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut replies = BufReader::new(mapping.0.stdout.take().unwrap());
    let mut reply = String::new();
    replies.read_line(&mut reply).unwrap();
    assert_eq!(reply.trim(), "ready");
    let mut write = |value: u8| {
        let stdin = mapping.0.stdin.as_mut().unwrap();
        stdin.write_all(&[value, b'\n']).unwrap();
        stdin.flush().unwrap();
        reply.clear();
        replies.read_line(&mut reply).unwrap();
        assert_eq!(reply.trim(), "written");
    };
    let stamp = || {
        let m = file.metadata().unwrap();
        (m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec())
    };
    let captured = Arc::new(CheckpointSnapshot::capture(&fixture.0).unwrap());
    let before = stamp();
    write(b'B');
    assert_eq!(std::fs::read(&config).unwrap(), br#"{"option":"B"}"#);
    assert_eq!(
        stamp(),
        before,
        "the fixture must exercise a timestamp-invisible write"
    );
    let mut snapshot = Snapshot::from_checkpoint(captured.clone()).unwrap();
    assert_eq!(snapshot.config()["option"], "A");
    assert_eq!(captured.document("config.json").unwrap(), original);
    assert_eq!(snapshot.request(4, 0).documents[0].bytes, original);
    write(b'A');
    snapshot.verify().unwrap();
    assert_eq!(stamp(), before);
}
