#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_weights::{ArtifactContract, FileIdentity, ShardTopology, sha256_bytes};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

pub struct TestDirectory(PathBuf);

impl TestDirectory {
    pub fn new() -> Self {
        let serial = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("runtime-weights-{}-{serial}", std::process::id()));
        fs::create_dir(&path).expect("create test directory");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn write(&self, name: &str, bytes: &[u8]) {
        fs::write(self.0.join(name), bytes).expect("write fixture");
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

pub fn safetensors(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut json = header.as_bytes().to_vec();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(payload);
    bytes
}

pub fn identity(name: &str, bytes: &[u8]) -> FileIdentity {
    FileIdentity::new(name, bytes.len() as u64, &sha256_bytes(bytes)).expect("valid identity")
}

pub fn two_shard_fixture() -> (TestDirectory, ArtifactContract) {
    let root = TestDirectory::new();
    let config = br#"{"fixture":"example-weights"}"#;
    let shard_a = safetensors(
        r#"{"tensor.a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    let shard_b = safetensors(
        r#"{"tensor.b":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[0,3]}}"#,
        &[0; 3],
    );
    let index = br#"{"metadata":{"total_size":7},"weight_map":{"tensor.a":"model-00001-of-00002.safetensors","tensor.b":"model-00002-of-00002.safetensors"}}"#;
    root.write("config.json", config);
    root.write("model.safetensors.index.json", index);
    root.write("model-00001-of-00002.safetensors", &shard_a);
    root.write("model-00002-of-00002.safetensors", &shard_b);
    let contract = ArtifactContract::new(
        "fixture/example-weights",
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        identity("config.json", config),
        identity("model.safetensors.index.json", index),
        vec![
            identity("model-00001-of-00002.safetensors", &shard_a),
            identity("model-00002-of-00002.safetensors", &shard_b),
        ],
        7,
        2,
        ShardTopology::new(4).expect("valid topology"),
    )
    .expect("valid contract");
    (root, contract)
}

pub fn contract_from_disk(
    root: &TestDirectory,
    index: &[u8],
    total_tensor_bytes: u64,
    tensor_count: usize,
) -> ArtifactContract {
    let config = fs::read(root.path().join("config.json")).expect("read config fixture");
    let shard_a = fs::read(root.path().join("model-00001-of-00002.safetensors"))
        .expect("read first shard fixture");
    let shard_b = fs::read(root.path().join("model-00002-of-00002.safetensors"))
        .expect("read second shard fixture");
    ArtifactContract::new(
        "fixture/example-weights",
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        identity("config.json", &config),
        identity("model.safetensors.index.json", index),
        vec![
            identity("model-00001-of-00002.safetensors", &shard_a),
            identity("model-00002-of-00002.safetensors", &shard_b),
        ],
        total_tensor_bytes,
        tensor_count,
        ShardTopology::new(4).expect("valid topology"),
    )
    .expect("valid fixture contract")
}
