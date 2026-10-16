mod support;

use std::fs;

use runtime_weights::{ArtifactSource, inspect_artifacts, plan_shard_ownership};
use support::{contract_from_disk, safetensors, two_shard_fixture};

const REPOSITORY: &str = "fixture/example-weights";
const REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";

#[test]
fn builds_an_immutable_complete_manifest() {
    let (root, contract) = two_shard_fixture();
    let source = ArtifactSource::new(REPOSITORY, REVISION);
    let manifest = inspect_artifacts(root.path(), &source, &contract).unwrap();
    assert_eq!(manifest.repository(), REPOSITORY);
    assert_eq!(manifest.revision(), REVISION);
    assert_eq!(manifest.tensors().len(), 2);
    assert_eq!(manifest.shards().len(), 2);
    assert_eq!(manifest.total_tensor_bytes(), 7);
}

#[test]
fn rejects_wrong_revision_and_config_hash() {
    let (root, contract) = two_shard_fixture();
    let wrong = ArtifactSource::new(REPOSITORY, "wrong-revision");
    let error = inspect_artifacts(root.path(), &wrong, &contract).unwrap_err();
    assert!(error.to_string().contains("revision"));

    fs::write(root.path().join("config.json"), b"changed").unwrap();
    let source = ArtifactSource::new(REPOSITORY, REVISION);
    let error = inspect_artifacts(root.path(), &source, &contract).unwrap_err();
    assert!(error.to_string().contains("config.json"));
}

#[test]
fn rejects_missing_shards_and_missing_tensor_entries() {
    let (root, contract) = two_shard_fixture();
    fs::remove_file(root.path().join("model-00002-of-00002.safetensors")).unwrap();
    let error = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("model-00002-of-00002.safetensors")
    );
}

#[test]
fn rejects_a_header_tensor_missing_from_the_index() {
    let (root, _) = two_shard_fixture();
    let index = br#"{"metadata":{"total_size":7},"weight_map":{"tensor.a":"model-00001-of-00002.safetensors","tensor.missing":"model-00002-of-00002.safetensors"}}"#;
    root.write("model.safetensors.index.json", index);
    let contract = contract_from_disk(&root, index, 7, 2);
    let error = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("tensor.b"));
    assert!(error.to_string().contains("missing from the index"));
}

#[test]
fn rejects_duplicate_tensor_storage_across_shards() {
    let (root, _) = two_shard_fixture();
    let duplicate = safetensors(
        r#"{"tensor.a":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[0,3]}}"#,
        &[0; 3],
    );
    root.write("model-00002-of-00002.safetensors", &duplicate);
    let index = fs::read(root.path().join("model.safetensors.index.json")).unwrap();
    let contract = contract_from_disk(&root, &index, 7, 2);
    let error = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("more than one shard"));
}

#[test]
fn rejects_same_size_shard_hash_mismatch() {
    let (root, contract) = two_shard_fixture();
    let shard = root.path().join("model-00001-of-00002.safetensors");
    let mut bytes = fs::read(&shard).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(shard, bytes).unwrap();
    let error = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256"));
}

#[cfg(unix)]
#[test]
fn rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let (root, contract) = two_shard_fixture();
    let shard = root.path().join("model-00002-of-00002.safetensors");
    let outside = root.path().with_extension("outside");
    fs::rename(&shard, &outside).unwrap();
    symlink(&outside, &shard).unwrap();
    let error = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap_err();
    assert!(error.to_string().contains("symlink"));
    fs::remove_file(&outside).unwrap();
}

#[test]
fn tp4_assignment_is_sorted_round_robin_and_repeatable() {
    let (root, contract) = two_shard_fixture();
    let manifest = inspect_artifacts(
        root.path(),
        &ArtifactSource::new(REPOSITORY, REVISION),
        &contract,
    )
    .unwrap();
    let first = plan_shard_ownership(&manifest).unwrap();
    let second = plan_shard_ownership(&manifest).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first
            .owner("model-00001-of-00002.safetensors")
            .map(|rank| rank.ordinal()),
        Some(0)
    );
    assert_eq!(
        first
            .owner("model-00002-of-00002.safetensors")
            .map(|rank| rank.ordinal()),
        Some(1)
    );
}

#[test]
fn topology_accepts_only_supported_local_counts_and_bounded_ranks() {
    use runtime_weights::ShardTopology;

    for count in [1, 2, 4, 8] {
        let topology = ShardTopology::new(count).unwrap();
        assert_eq!(topology.rank_count(), count);
        assert_eq!(topology.rank(count - 1).unwrap().ordinal(), count - 1);
        assert!(topology.rank(count).is_err());
    }
    for count in [0, 3, 5, 6, 7, 9, u32::MAX] {
        assert!(ShardTopology::new(count).is_err());
    }
}
