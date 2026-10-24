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
    keep_mode(root, plans, directory, false, edit)
}

fn content_config(image: Arc<PlannedImageV1>, source: ArenaSourceV1) -> HostInstanceConfigV1 {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
        .with_planned_compiled_image(image)
        .unwrap()
        .with_arena_source(source);
    let manifest = serde_json::from_value(config.weight_identity(true).unwrap()).unwrap();
    config.with_weight_content(Arc::new(manifest)).unwrap()
}

fn keep_mode(
    root: &Path,
    plans: &[WeightPlanV1],
    directory: &Path,
    content: bool,
    edit: impl FnOnce(&mut [weight_keeper::KeptImageV1]),
) -> KeepingV1 {
    let opened = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let image = Arc::new(sharing_image(&opened, plans, 2, 128 << 10));
    let keeper = if content {
        HostServicesV1Backend::new(
            FakeHostDeviceV1::new(),
            content_config(image, ArenaSourceV1::Lend),
        )
    } else {
        host(image, ArenaSourceV1::Lend)
    };
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

/// A keeper that holds a bank as row-major groups lends nothing to a server
/// that plans the same checkpoint's bank in row tiles: the record
/// parameters, and so every rank's identity, differ. The server refuses to
/// start when it requires the keeper's images, and otherwise loads its own
/// tiled bytes.
#[test]
fn a_keeper_of_row_major_banks_lends_nothing_to_a_row_tiled_plan() {
    lends_nothing(
        "row-tiles",
        "row-major-groups",
        kernel_core::precision::layout::ROW_TILES_64X128,
    );
}

/// Nor does a keeper of the row-major or row-tiled bank lend to a server
/// that plans it in quad-lane row tiles.
#[test]
fn a_keeper_of_row_tiled_banks_lends_nothing_to_a_quad_lane_plan() {
    use kernel_core::precision::layout::{ROW_TILES_64X128, ROW_TILES_64X128_LANES};
    for (name, kept) in [
        ("lanes-rows", "row-major-groups"),
        ("lanes-tiles", ROW_TILES_64X128),
    ] {
        lends_nothing(name, kept, ROW_TILES_64X128_LANES);
    }
}

/// A keeper of the bank planned in `kept`, in keeper directory `name`, lends
/// nothing to a server that plans it in `planned`.
fn lends_nothing(name: &str, kept: &str, planned: &str) {
    lends_nothing_formats(name, kept, planned, None);
}

fn lends_nothing_formats(
    name: &str,
    kept: &str,
    planned: &str,
    formats: Option<(
        kernel_core::weight_transform::ScaleFormat,
        kernel_core::weight_transform::ScaleFormat,
    )>,
) {
    use super::packed::{bank_checkpoint, bank_plans, encoded_as};
    let (root, tensors) = bank_checkpoint();
    let (keeps, kept_bytes) = bank_plans(ROLE, kept, &tensors);
    let (plans, planned_bytes) = bank_plans(ROLE, planned, &tensors);
    let ((keeps, kept_bytes), (plans, planned_bytes)) = if let Some((a, b)) = formats {
        (
            encoded_as(&keeps, &kept_bytes, 1, a),
            encoded_as(&plans, &planned_bytes, 1, b),
        )
    } else {
        ((keeps, kept_bytes), (plans, planned_bytes))
    };
    let directory = KeeperDirectory::new(name);
    let keeper = keep(&root.0, &keeps, &directory.0, |_| {});
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
    drop(required);
    let preferred = host(
        image,
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Prefer,
        },
    );
    let arenas = upload(&preferred).unwrap();
    for (rank, arena) in arenas.iter().enumerate() {
        let (bytes, lent, _) = preferred.rank_arena_v1(*arena);
        assert!(!lent, "{kept} -> {planned} rank {rank}");
        assert_eq!(
            bytes, planned_bytes[rank],
            "{kept} -> {planned} rank {rank}"
        );
        assert_ne!(bytes, kept_bytes[rank], "{kept} -> {planned} rank {rank}");
    }
    assert_eq!(keeper.join(), Ok(KeeperExitV1::Released));
}

#[test]
fn a_keeper_never_lends_between_row_and_tile_scale_encodings() {
    use kernel_core::precision::layout::ROW_TILES_64X128_LANES;
    use kernel_core::weight_transform::ScaleFormat::{RowOffsetV1, RowTilesOffsetV1};
    for (name, formats) in [
        ("scales-to-tiles", (RowOffsetV1, RowTilesOffsetV1)),
        ("scales-to-rows", (RowTilesOffsetV1, RowOffsetV1)),
    ] {
        lends_nothing_formats(
            name,
            ROW_TILES_64X128_LANES,
            ROW_TILES_64X128_LANES,
            Some(formats),
        );
    }
}

#[test]
fn content_lending_crosses_producer_builds_without_loading_or_writable_maps() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("content-builds");
    let keeper = keep_mode(&root.0, &plans, &directory.0, true, |images| {
        assert!(
            images
                .iter()
                .all(|image| image.content.as_ref().is_some_and(|proof| proof.valid()))
        );
    });
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(
        PlannedImageV1::new_with_range_tracking(
            opened,
            &plans,
            None,
            WeightPlanImplementationV1 {
                shared_transforms: [17; 32],
                bundle_module: [0; 32],
            },
            DirectLoadOptionsV1 {
                threads_per_rank: 2,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: 128 << 10,
            },
            true,
        )
        .unwrap(),
    );
    let config = content_config(
        Arc::clone(&image),
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Require,
        },
    );
    let before = image.stats();
    let borrower = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
    let arenas = upload(&borrower).unwrap();
    for (rank, arena) in arenas.iter().enumerate() {
        let (bytes, lent, object) = borrower.rank_arena_v1(*arena);
        assert!(lent);
        assert_eq!(bytes, expected[rank]);
        assert_eq!(
            borrower.with_device(|device| device.imported_access(object)),
            Some((vec![0, 1, 2, 3], true))
        );
    }
    assert_eq!(image.stats().source_bytes, before.source_bytes);
    assert_eq!(image.stats().produced_bytes, before.produced_bytes);
    drop(borrower);
    wait_for_leases(&directory.0, 0);
    stop(&directory.0, keeper);
}

#[test]
fn keeper_checks_actual_uploaded_bytes_instead_of_trusting_a_cpu_proof() {
    let (root, plans, _) = fixture();
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
        .with_planned_compiled_image(image)
        .unwrap()
        .with_arena_source(ArenaSourceV1::Lend);
    let mut manifest: weight_keeper::content::Manifest =
        serde_json::from_value(config.weight_identity(true).unwrap()).unwrap();
    for image in &mut manifest.images {
        image.identity.content_sha256 = "f".repeat(64);
    }
    let config = config.with_weight_content(Arc::new(manifest)).unwrap();
    let keeper = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
    assert!(
        weight_keeper::load::load_v1(&keeper, &[0, 1, 2, 3])
            .unwrap_err()
            .contains("hold")
    );
    assert_eq!(
        keeper
            .with_device(FakeHostDeviceV1::held_allocations)
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

#[test]
fn content_mismatch_refuses_required_ranks_and_prefer_loads_them() {
    let (root, plans, expected) = fixture();
    let directory = KeeperDirectory::new("content-fallback");
    let keeper = keep_mode(&root.0, &plans, &directory.0, true, |images| {
        images[2].content.as_mut().unwrap().content_sha256 = "f".repeat(64);
    });
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let config = content_config(
        Arc::clone(&image),
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Require,
        },
    );
    let required = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
    assert_eq!(upload(&required), Err(BundleStatusV1::HostError));
    drop(required);
    wait_for_leases(&directory.0, 0);
    let config = content_config(
        image,
        ArenaSourceV1::Borrow {
            directory: directory.0.clone(),
            policy: KeeperPolicyV1::Prefer,
        },
    );
    let preferred = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
    for (rank, arena) in upload(&preferred).unwrap().iter().enumerate() {
        let (bytes, lent, _) = preferred.rank_arena_v1(*arena);
        assert_eq!(bytes, expected[rank]);
        assert_eq!(lent, rank != 2);
    }
    drop(preferred);
    wait_for_leases(&directory.0, 0);
    stop(&directory.0, keeper);
}

#[test]
fn content_proof_covers_compacted_bytes_zero_padding_and_component_only_layout_changes() {
    let (root, mut plans, expected) = fixture();
    let opened = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    for plan in &mut plans {
        let end = plan.arena_bytes;
        plan.arena_bytes += 512;
        plan.views.push(PlannedViewV1 {
            id: "host-padding".to_owned(),
            alignment: 256,
            placement: ViewPlacementV1::Host,
            offset: end,
            byte_length: 128,
        });
        let identity = runtime_weights::sha256_bytes(
            &plan.identity_document(&opened.identity(), &implementation_v1([0; 32])),
        );
        plan.components.push(NamedBytesV1{name:format!("chop.bundle-metadata.fixture.index.r{}",plan.rank),
            bytes:serde_json::to_vec(&serde_json::json!({"schema":"fixture.rank.v1","rank":plan.rank,
                "image_bytes":plan.arena_bytes,"direct_identity":identity,"views":[{"layout_id":"layout-a","dtype":"u8","shape":[4096]}]})).unwrap()});
    }
    let image = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    let config = content_config(Arc::clone(&image), ArenaSourceV1::Lend);
    let manifest: weight_keeper::content::Manifest =
        serde_json::from_value(config.weight_identity(true).unwrap()).unwrap();
    for (rank, record) in manifest.images.iter().enumerate() {
        let mut bytes = expected[rank].clone();
        bytes.extend([0; 256]);
        assert_eq!(record.byte_length, bytes.len() as u64);
        assert_eq!(
            record.identity.content_sha256,
            runtime_weights::sha256_bytes(&bytes)
        );
    }
    let keeper = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
    let held = weight_keeper::load::load_v1(&keeper, &[0, 1, 2, 3]).unwrap();
    assert!(held.iter().all(|item| item.content.is_some()));
    drop(held);
    drop(keeper);
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&plans[0].components[0].bytes).unwrap();
    metadata["views"][0]["layout_id"] = "layout-b".into();
    plans[0].components[0].bytes = serde_json::to_vec(&metadata).unwrap();
    let changed = Arc::new(sharing_image(&opened, &plans, 2, 128 << 10));
    assert_eq!(
        image.identity(0),
        changed.identity(0),
        "the strict recipe excludes components"
    );
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
        .with_planned_compiled_image(changed)
        .unwrap();
    assert!(
        config
            .with_weight_content(Arc::new(manifest))
            .unwrap_err()
            .contains("fresh recipe and layout")
    );
}
