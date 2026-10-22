use std::os::fd::AsFd;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::*;

/// A private directory of its own per test, removed on drop.
struct Directory(PathBuf);

impl Directory {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("keeper-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        prepare_directory_v1(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn image(rank: u32) -> RankImageV1 {
    RankImageV1 {
        image_role: "text".to_owned(),
        rank,
        identity: format!("{rank:064x}"),
        device: format!("device-{rank}"),
        byte_length: 100 + u64::from(rank),
    }
}

fn bytes(rank: u32) -> Vec<u8> {
    (0..100 + rank)
        .map(|index| (index * 7 + rank) as u8)
        .collect()
}

fn checkpoint() -> CheckpointV1 {
    CheckpointV1 {
        identity: "c".repeat(64),
        stamps: vec!["model.safetensors 1 2 3 4.000000000 5.000000000".to_owned()],
    }
}

/// A keeper of two text ranks whose stamps the test controls.
fn keeper(directory: &Path, stamps: Arc<Mutex<Result<Vec<String>, String>>>) -> KeeperV1 {
    KeeperV1 {
        directory: directory.to_path_buf(),
        checkpoint_identity: checkpoint().identity,
        stamps: Box::new(move || stamps.lock().unwrap().clone()),
        images: (0..2)
            .map(|rank| KeptImageV1 {
                image: image(rank),
                fd: fds::anonymous_file(&bytes(rank), 4096).unwrap(),
                mapped_bytes: 4096,
                held: None,
            })
            .collect(),
        read_only: true,
        about: serde_json::Map::new(),
    }
}

fn start(
    directory: &Path,
    stamps: Arc<Mutex<Result<Vec<String>, String>>>,
) -> std::thread::JoinHandle<Result<KeeperExitedV1, String>> {
    let keeper = keeper(directory, stamps);
    let handle = std::thread::spawn(move || serve_v1(keeper));
    let socket = directory.join(SOCKET_NAME_V1);
    for _ in 0..500 {
        if read_registry(directory).is_ok() && UnixStream::connect(&socket).is_ok() {
            return handle;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the keeper did not start");
}

fn leases(directory: &Path) -> u64 {
    status_v1(directory).unwrap()["leases"].as_u64().unwrap()
}

/// Waits until the keeper's registry shows `count` leases.
fn wait_for_leases(directory: &Path, count: u64) {
    for _ in 0..500 {
        if leases(directory) == count {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the keeper never held {count} leases");
}

#[test]
fn a_keeper_lends_what_matches_and_says_why_it_refuses_the_rest() {
    let directory = Directory::new("lend");
    let stamps = Arc::new(Mutex::new(Ok(checkpoint().stamps)));
    let keeper = start(&directory.0, Arc::clone(&stamps));
    let mut other = image(1);
    other.identity = "f".repeat(64);
    let mut moved = image(0);
    moved.device = "device-9".to_owned();
    let unknown = RankImageV1 {
        image_role: "vision".to_owned(),
        ..image(0)
    };
    let lease = lease_v1(
        &directory.0,
        checkpoint(),
        vec![image(0), other, moved, unknown, image(1)],
    )
    .unwrap();
    assert_eq!(
        lease
            .lent
            .iter()
            .map(|lent| lent.image.clone())
            .collect::<Vec<_>>(),
        [image(0), image(1)]
    );
    for lent in &lease.lent {
        assert_eq!(lent.mapped_bytes, 4096);
        assert!(lent.read_only);
        let read = fds::read_anonymous_file(lent.fd.as_fd(), 4096, lent.image.byte_length).unwrap();
        assert_eq!(read, bytes(lent.image.rank));
    }
    let reasons: Vec<&str> = lease
        .refused
        .iter()
        .map(|(_, reason)| reason.as_str())
        .collect();
    assert_eq!(
        reasons,
        [
            "the image identity differs",
            "the device differs",
            "the keeper does not hold this image"
        ]
    );
    wait_for_leases(&directory.0, 1);
    // A lease open blocks a release; closing it lets the release free the
    // keeper's memory.
    let refused = release_v1(&directory.0).unwrap_err();
    assert!(refused.contains("1 leases are open"), "{refused}");
    drop(lease);
    wait_for_leases(&directory.0, 0);
    let registry = status_v1(&directory.0).unwrap();
    assert!(registry["last_lease_end_unix"].as_u64().unwrap() > 0);
    let release = std::thread::spawn({
        let directory = directory.0.clone();
        move || release_v1(&directory)
    });
    let exited = keeper.join().unwrap().unwrap();
    assert_eq!(exited.exit, KeeperExitV1::Released);
    assert!(!directory.0.join(SOCKET_NAME_V1).exists());
    assert!(!directory.0.join(REGISTRY_NAME_V1).exists());
    // The releasing server waits until the keeper's memory is freed, which
    // the caller of `serve_v1` signals by closing this connection.
    std::thread::sleep(Duration::from_millis(50));
    assert!(!release.is_finished());
    drop(exited.waiting);
    release.join().unwrap().unwrap();
}

#[test]
fn a_different_checkpoint_borrows_nothing() {
    let directory = Directory::new("checkpoint");
    let stamps = Arc::new(Mutex::new(Ok(checkpoint().stamps)));
    let keeper = start(&directory.0, Arc::clone(&stamps));
    let mut renamed = checkpoint();
    renamed.identity = "d".repeat(64);
    let lease = lease_v1(&directory.0, renamed, vec![image(0)]).unwrap();
    assert!(lease.lent.is_empty() && lease.connection.is_none());
    assert_eq!(lease.refused[0].1, "the checkpoint identity differs");
    let mut touched = checkpoint();
    touched.stamps[0] = "model.safetensors 1 2 3 4.000000000 6.000000000".to_owned();
    let lease = lease_v1(&directory.0, touched, vec![image(0)]).unwrap();
    assert_eq!(lease.refused[0].1, "the checkpoint files differ");
    // The keeper's own files changed: it lends nothing again and says so.
    *stamps.lock().unwrap() = Err("checkpoint file model.safetensors changed".to_owned());
    let lease = lease_v1(&directory.0, checkpoint(), vec![image(0)]).unwrap();
    assert!(
        lease.refused[0]
            .1
            .contains("the keeper's checkpoint changed")
    );
    *stamps.lock().unwrap() = Ok(checkpoint().stamps);
    let lease = lease_v1(&directory.0, checkpoint(), vec![image(0)]).unwrap();
    assert!(lease.lent.is_empty());
    assert_eq!(status_v1(&directory.0).unwrap()["state"], "stale");
    stop_v1_after_close(&directory.0, keeper);
}

/// Stops the keeper and closes the connection the stop waits on.
fn stop_v1_after_close(
    directory: &Path,
    keeper: std::thread::JoinHandle<Result<KeeperExitedV1, String>>,
) {
    let stop = std::thread::spawn({
        let directory = directory.to_path_buf();
        move || stop_v1(&directory)
    });
    let exited = keeper.join().unwrap().unwrap();
    assert_eq!(exited.exit, KeeperExitV1::Stopped);
    drop(exited.waiting);
    stop.join().unwrap().unwrap();
}

#[test]
fn a_stop_ends_the_keeper_while_a_lease_keeps_its_descriptors() {
    let directory = Directory::new("stop");
    let stamps = Arc::new(Mutex::new(Ok(checkpoint().stamps)));
    let keeper = start(&directory.0, stamps);
    let lease = lease_v1(&directory.0, checkpoint(), vec![image(1)]).unwrap();
    stop_v1_after_close(&directory.0, keeper);
    let lent = &lease.lent[0];
    let read = fds::read_anonymous_file(lent.fd.as_fd(), 4096, lent.image.byte_length).unwrap();
    assert_eq!(read, bytes(1));
    assert!(lease_v1(&directory.0, checkpoint(), vec![image(1)]).is_err());
}

#[test]
fn a_directory_open_to_others_or_reached_through_a_link_is_refused() {
    let directory = Directory::new("private");
    let open = directory.0.join("open");
    std::fs::create_dir(&open).unwrap();
    std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o750)).unwrap();
    let error = prepare_directory_v1(&open).unwrap_err();
    assert!(error.contains("closed to group and others"), "{error}");
    assert!(lease_v1(&open, checkpoint(), vec![image(0)]).is_err());
    let link = directory.0.join("link");
    let target = directory.0.join("target");
    prepare_directory_v1(&target).unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let error = prepare_directory_v1(&link).unwrap_err();
    assert!(error.contains("is not a directory"), "{error}");
}

#[test]
fn a_second_keeper_does_not_take_a_live_keepers_directory() {
    let directory = Directory::new("second");
    let stamps = Arc::new(Mutex::new(Ok(checkpoint().stamps)));
    let first = start(&directory.0, Arc::clone(&stamps));
    let error = serve_v1(keeper(&directory.0, stamps)).unwrap_err();
    assert!(error.contains("another keeper serves"), "{error}");
    stop_v1_after_close(&directory.0, first);
}

#[test]
fn a_registry_naming_another_process_is_not_trusted() {
    let directory = Directory::new("registry");
    let stamps = Arc::new(Mutex::new(Ok(checkpoint().stamps)));
    let keeper = start(&directory.0, stamps);
    let path = directory.0.join(REGISTRY_NAME_V1);
    let mut registry = read_registry(&directory.0).unwrap();
    let real = registry["pid"].clone();
    registry["pid"] = serde_json::json!(1);
    std::fs::write(&path, serde_json::to_vec(&registry).unwrap()).unwrap();
    let error = lease_v1(&directory.0, checkpoint(), vec![image(0)]).unwrap_err();
    assert!(
        error.contains("not the process its registry names"),
        "{error}"
    );
    registry["pid"] = real;
    std::fs::write(&path, serde_json::to_vec(&registry).unwrap()).unwrap();
    stop_v1_after_close(&directory.0, keeper);
}

#[test]
fn the_policy_and_source_selections_parse_and_compare() {
    assert_eq!(KeeperPolicyV1::parse("prefer"), Ok(KeeperPolicyV1::Prefer));
    assert_eq!(
        KeeperPolicyV1::parse("require"),
        Ok(KeeperPolicyV1::Require)
    );
    assert!(KeeperPolicyV1::parse("always").is_err());
    assert_eq!(ArenaSourceV1::default(), ArenaSourceV1::Load);
}
