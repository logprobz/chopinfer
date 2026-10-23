//! A keeper's rank images lent to a borrowing host: the same bytes, mapped
//! instead of loaded, and loaded where the keeper does not lend them.

use super::*;
use crate::fake::FakeHostDeviceV1;
use crate::weight_keeper::{self, ArenaSourceV1, KeeperExitV1, KeeperPolicyV1, KeeperV1};
use crate::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
use runtime_bundle_abi::{
    BundleRankArenaHandleV1, BundleStatusV1, HostServicesBackendV1, RankArenaUploadV1,
};

const ROLE: &str = "fixture.rank-images.v1";

/// A private keeper directory, removed on drop.
struct KeeperDirectory(PathBuf);

impl KeeperDirectory {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("keeper-direct-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        weight_keeper::prepare_directory_v1(&path).unwrap();
        Self(path)
    }
}

impl Drop for KeeperDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn host(
    image: Arc<PlannedImageV1>,
    source: ArenaSourceV1,
) -> HostServicesV1Backend<FakeHostDeviceV1> {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
        .with_planned_compiled_image(image)
        .unwrap()
        .with_arena_source(source);
    HostServicesV1Backend::new(FakeHostDeviceV1::new(), config)
}

/// Uploads every rank's arena in one call, as a bundle does.
fn upload(
    host: &HostServicesV1Backend<FakeHostDeviceV1>,
) -> Result<Vec<BundleRankArenaHandleV1>, BundleStatusV1> {
    let image = host.open_compiled_image(ROLE.as_bytes())?;
    let uploads: Vec<_> = (0..4)
        .map(|rank| {
            let context = host.select_context(rank, rank as i32).unwrap();
            RankArenaUploadV1 {
                context,
                pool: host.select_memory_pool(context).unwrap(),
                stream: host.create_stream(context, 0).unwrap(),
                image,
                rank,
            }
        })
        .collect();
    host.upload_rank_arenas(&uploads)
}

/// A four-rank checkpoint image fixture: the checkpoint, its plans and
/// each rank's expected arena.
fn fixture() -> (Temp, Vec<WeightPlanV1>, Vec<Vec<u8>>) {
    let (rows, row_bytes) = (64_u64, 256_u64);
    let matrix = bytes((rows * row_bytes) as usize, 3);
    let root = checkpoint(&[("matrix", "U8", vec![rows, row_bytes], matrix.clone())]);
    let (plans, expected) = k_slice_plans(&[("matrix", rows, row_bytes, &matrix)], 4);
    (root, plans, expected)
}

/// A keeper serving on a thread, and the count of the memory it holds.
struct KeepingV1 {
    thread: std::thread::JoinHandle<Result<KeeperExitV1, String>>,
    held: Arc<std::sync::atomic::AtomicUsize>,
}

impl KeepingV1 {
    /// Waits for the keeper to exit, which frees what it held.
    fn join(self) -> Result<KeeperExitV1, String> {
        let exit = self.thread.join().unwrap();
        assert_eq!(self.held.load(std::sync::atomic::Ordering::SeqCst), 0);
        exit
    }
}

/// Loads the fixture's images as a keeper would, holding their memory and
/// dropping its backend, lets `edit` change what the keeper claims to hold,
/// and serves them from `directory` on a thread.
fn keep(
    root: &Path,
    plans: &[WeightPlanV1],
    directory: &Path,
    edit: impl FnOnce(&mut [weight_keeper::KeptImageV1]),
) -> KeepingV1 {
    let opened = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let keeper = host(
        Arc::new(sharing_image(&opened, plans, 2, 128 << 10)),
        ArenaSourceV1::Lend,
    );
    let mut images = weight_keeper::load::load_v1(&keeper, &[0, 1, 2, 3]).unwrap();
    // Every arena is held, not live, so the backend can go.
    assert_eq!(
        keeper.with_device(FakeHostDeviceV1::lent_allocation_counts),
        (0, 0)
    );
    let held = keeper.with_device(FakeHostDeviceV1::held_allocations);
    assert_eq!(held.load(std::sync::atomic::Ordering::SeqCst), 4);
    drop(keeper);
    assert!(images.iter().all(|kept| kept.held.is_some()));
    edit(&mut images);
    let stamps = Arc::clone(&opened);
    let lending = KeeperV1 {
        directory: directory.to_path_buf(),
        checkpoint_identity: opened.identity_hex(),
        stamps: Box::new(move || {
            stamps.verify_unchanged()?;
            stamps.stamps()
        }),
        images,
        read_only: true,
        about: serde_json::Map::new(),
    };
    let thread = std::thread::spawn(move || {
        let exited = weight_keeper::serve_v1(lending)?;
        Ok(exited.exit)
    });
    for _ in 0..500 {
        if weight_keeper::status_v1(directory).is_ok() {
            return KeepingV1 { thread, held };
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the keeper did not start");
}

fn wait_for_leases(directory: &Path, count: u64) {
    for _ in 0..500 {
        if weight_keeper::status_v1(directory).unwrap()["leases"] == count {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the keeper never held {count} leases");
}

fn stop(directory: &Path, keeper: KeepingV1) {
    weight_keeper::stop_v1(directory).unwrap();
    assert_eq!(keeper.join(), Ok(KeeperExitV1::Stopped));
}

#[test]
fn a_borrowing_host_maps_every_lent_arena_and_reads_no_checkpoint_byte() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("borrow");
    let keeper = keep(&root.0, &plans, &directory.0, |images| {
        assert_eq!(images.len(), 4);
        for (rank, kept) in images.iter().enumerate() {
            assert_eq!(kept.image.image_role, ROLE);
            assert_eq!(kept.image.rank, rank as u32);
            assert_eq!(kept.image.byte_length, expected[rank].len() as u64);
            assert_eq!(kept.image.device, format!("fake-device-{rank}"));
        }
    });
    // A second process opens the checkpoint and plans the image itself.
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let borrower = host(
        Arc::clone(&image),
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Require,
        },
    );
    let arenas = upload(&borrower).unwrap();
    for (rank, arena) in arenas.iter().enumerate() {
        let (bytes, lent, object) = borrower.rank_arena_v1(*arena);
        assert!(lent, "rank {rank} was loaded");
        assert_eq!(bytes, expected[rank], "rank {rank}");
        let access = borrower.with_device(|device| device.imported_access(object));
        assert_eq!(access, Some((vec![0, 1, 2, 3], true)));
    }
    assert_eq!(image.stats().source_bytes, 0);
    assert_eq!(image.stats().produced_bytes, 0);
    wait_for_leases(&directory.0, 1);
    // Host resources go back in the reverse of the order they came in.
    for arena in arenas.into_iter().rev() {
        borrower.release_rank_arena(arena).unwrap();
    }
    assert_eq!(
        borrower.with_device(FakeHostDeviceV1::lent_allocation_counts),
        (0, 0)
    );
    drop(borrower);
    wait_for_leases(&directory.0, 0);
    stop(&directory.0, keeper);
}

#[test]
fn a_borrower_loads_what_is_not_lent_unless_it_requires_every_image() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("partial");
    // A keeper whose rank 2 was planned by other sources.
    let keeper = keep(&root.0, &plans, &directory.0, |images| {
        images[2].image.identity = "0".repeat(64);
    });
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let required = host(
        Arc::clone(&image),
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Require,
        },
    );
    assert_eq!(upload(&required), Err(BundleStatusV1::HostError));
    assert_eq!(
        required.with_device(FakeHostDeviceV1::lent_allocation_counts),
        (0, 0)
    );
    drop(required);
    let preferred = host(
        Arc::clone(&image),
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Prefer,
        },
    );
    let arenas = upload(&preferred).unwrap();
    for (rank, arena) in arenas.iter().enumerate() {
        let (bytes, lent, _) = preferred.rank_arena_v1(*arena);
        assert_eq!(lent, rank != 2, "rank {rank}");
        assert_eq!(bytes, expected[rank], "rank {rank}");
    }
    assert!(image.stats().source_bytes > 0);
    // It lent something, so it keeps its memory for the next borrower; the
    // refused start's lease ended with its connection.
    wait_for_leases(&directory.0, 1);
    drop(preferred);
    wait_for_leases(&directory.0, 0);
    stop(&directory.0, keeper);
}

#[test]
fn a_keeper_that_lends_nothing_is_released_before_the_borrower_loads() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("release");
    let keeper = keep(&root.0, &plans, &directory.0, |images| {
        for kept in images {
            kept.image.identity = "0".repeat(64);
        }
    });
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let borrower = host(
        image,
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Prefer,
        },
    );
    let arenas = upload(&borrower).unwrap();
    assert_eq!(keeper.join(), Ok(KeeperExitV1::Released));
    for (rank, arena) in arenas.iter().enumerate() {
        let (bytes, lent, _) = borrower.rank_arena_v1(*arena);
        assert!(!lent);
        assert_eq!(bytes, expected[rank]);
    }
}

#[test]
fn a_missing_keeper_fails_only_a_start_that_requires_it() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("missing");
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let source = |policy| ArenaSourceV1::Borrow {
        directory: directory.0.clone(),
        policy,
    };
    let required = host(Arc::clone(&image), source(KeeperPolicyV1::Require));
    assert_eq!(upload(&required), Err(BundleStatusV1::HostError));
    let preferred = host(image, source(KeeperPolicyV1::Prefer));
    let arenas = upload(&preferred).unwrap();
    assert_eq!(preferred.rank_arena_v1(arenas[1]).0, expected[1]);
}
