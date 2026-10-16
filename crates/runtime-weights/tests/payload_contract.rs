mod support;

use std::fs;
use std::sync::Arc;
use std::thread;

use runtime_weights::{
    ArtifactContract, ArtifactSource, PayloadReadError, ShardTopology, open_artifacts,
};
use support::{TestDirectory, identity, safetensors};

const REPOSITORY: &str = "fixture/example-weights";
const REVISION: &str = "fixture-revision";
const SHARD: &str = "model-00001-of-00001.safetensors";

fn payload_fixture() -> (TestDirectory, ArtifactContract) {
    let root = TestDirectory::new();
    let config = br#"{"fixture":"payload-read"}"#;
    let shard = safetensors(
        r#"{"tensor.bytes":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        &[10, 11, 12, 13, 14, 15, 16, 17],
    );
    let index = br#"{"metadata":{"total_size":8},"weight_map":{"tensor.bytes":"model-00001-of-00001.safetensors"}}"#;
    root.write("config.json", config);
    root.write("model.safetensors.index.json", index);
    root.write(SHARD, &shard);
    let contract = ArtifactContract::new(
        REPOSITORY,
        REVISION,
        identity("config.json", config),
        identity("model.safetensors.index.json", index),
        vec![identity(SHARD, &shard)],
        8,
        1,
        ShardTopology::new(4).unwrap(),
    )
    .unwrap();
    (root, contract)
}

fn open_fixture(
    root: &TestDirectory,
    contract: &ArtifactContract,
) -> runtime_weights::VerifiedArtifacts {
    open_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        contract,
    )
    .unwrap()
}

#[test]
fn reads_exact_partial_and_final_tensor_bytes() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    assert_eq!(artifacts.tensor_byte_len("tensor.bytes").unwrap(), 8);
    assert_eq!(artifacts.tensor("tensor.bytes").unwrap().shape(), &[8]);

    let mut exact = [0_u8; 8];
    artifacts
        .read_tensor("tensor.bytes", 0, &mut exact)
        .unwrap();
    assert_eq!(exact, [10, 11, 12, 13, 14, 15, 16, 17]);

    let mut partial = [0_u8; 3];
    artifacts
        .read_tensor("tensor.bytes", 2, &mut partial)
        .unwrap();
    assert_eq!(partial, [12, 13, 14]);

    let mut final_byte = [0_u8; 1];
    artifacts
        .read_tensor("tensor.bytes", 7, &mut final_byte)
        .unwrap();
    assert_eq!(final_byte, [17]);
}

#[test]
fn reauthenticates_unchanged_shard_descriptors() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);

    artifacts.reauthenticate_sources().unwrap();
}

#[test]
fn reauthentication_rejects_same_size_mutation_after_payload_reads() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    let mut payload = [0_u8; 8];
    artifacts
        .read_tensor("tensor.bytes", 0, &mut payload)
        .unwrap();

    let shard_path = root.path().join(SHARD);
    let mut shard = fs::read(&shard_path).unwrap();
    let last = shard.last_mut().unwrap();
    *last ^= 0xff;
    fs::write(shard_path, shard).unwrap();

    let error = artifacts.reauthenticate_sources().unwrap_err();
    assert!(error.to_string().contains("SHA-256"));
}

#[test]
fn reauthentication_rejects_same_content_path_replacement() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    let shard_path = root.path().join(SHARD);
    let displaced_path = root.path().join("authenticated-shard");
    let replacement = fs::read(&shard_path).unwrap();
    fs::rename(&shard_path, displaced_path).unwrap();
    fs::write(&shard_path, replacement).unwrap();

    let error = artifacts.reauthenticate_sources().unwrap_err();
    assert!(error.to_string().contains("path no longer names"));
}

#[test]
fn rejects_unknown_empty_out_of_range_and_overflow_reads() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    let mut byte = [0_u8; 1];
    let mut too_many = [0_u8; 2];
    let mut empty = [];

    assert!(matches!(
        artifacts.read_tensor("missing", 0, &mut byte),
        Err(PayloadReadError::UnknownTensor)
    ));
    assert!(matches!(
        artifacts.read_tensor("tensor.bytes", 0, &mut empty),
        Err(PayloadReadError::EmptyRead)
    ));
    assert!(matches!(
        artifacts.read_tensor("tensor.bytes", 7, &mut too_many),
        Err(PayloadReadError::RangeOutOfBounds { .. })
    ));
    assert!(matches!(
        artifacts.read_tensor("tensor.bytes", u64::MAX, &mut byte),
        Err(PayloadReadError::RangeEndOverflow { .. })
    ));
}

#[test]
fn detects_truncation_before_a_payload_read() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    let shard_path = root.path().join(SHARD);
    let original_len = fs::metadata(&shard_path).unwrap().len();
    fs::OpenOptions::new()
        .write(true)
        .open(shard_path)
        .unwrap()
        .set_len(original_len - 1)
        .unwrap();

    let mut byte = [0_u8; 1];
    assert!(matches!(
        artifacts.read_tensor("tensor.bytes", 0, &mut byte),
        Err(PayloadReadError::ShardLengthChanged { .. })
    ));
}

#[test]
fn path_replacement_does_not_redirect_authenticated_reads() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    let shard_path = root.path().join(SHARD);
    let authenticated_path = root.path().join("authenticated-shard");
    let original_len = fs::metadata(&shard_path).unwrap().len();
    fs::rename(&shard_path, authenticated_path).unwrap();
    fs::write(&shard_path, vec![0xff; original_len as usize]).unwrap();

    let mut payload = [0_u8; 8];
    artifacts
        .read_tensor("tensor.bytes", 0, &mut payload)
        .unwrap();
    assert_eq!(payload, [10, 11, 12, 13, 14, 15, 16, 17]);
}

#[test]
fn descriptor_outlives_the_shard_directory_entry() {
    let (root, contract) = payload_fixture();
    let artifacts = open_fixture(&root, &contract);
    fs::remove_file(root.path().join(SHARD)).unwrap();

    let mut payload = [0_u8; 2];
    artifacts
        .read_tensor("tensor.bytes", 3, &mut payload)
        .unwrap();
    assert_eq!(payload, [13, 14]);
}

#[test]
fn supports_concurrent_positional_reads_from_one_shard() {
    let (root, contract) = payload_fixture();
    let artifacts = Arc::new(open_fixture(&root, &contract));
    let readers: Vec<_> = (0..8)
        .map(|index| {
            let artifacts = Arc::clone(&artifacts);
            thread::spawn(move || {
                for _ in 0..32 {
                    let mut byte = [0_u8; 1];
                    artifacts
                        .read_tensor("tensor.bytes", index, &mut byte)
                        .unwrap();
                    assert_eq!(byte[0], 10 + index as u8);
                }
            })
        })
        .collect();
    for reader in readers {
        reader.join().unwrap();
    }
}

#[cfg(unix)]
#[test]
fn rejects_root_and_shard_symlinks() {
    use std::os::unix::fs::symlink;

    let (root, contract) = payload_fixture();
    let root_link = root.path().with_extension("root-link");
    symlink(root.path(), &root_link).unwrap();
    let error = open_artifacts(
        &root_link,
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("real directory"));
    fs::remove_file(root_link).unwrap();

    let shard_path = root.path().join(SHARD);
    let target_path = root.path().join("shard-target");
    fs::rename(&shard_path, &target_path).unwrap();
    symlink(&target_path, &shard_path).unwrap();
    let error = open_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("symlink"));
}
