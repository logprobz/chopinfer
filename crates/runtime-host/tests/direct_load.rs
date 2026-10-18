//! A planned compiled image uploads, resolves views and serves its planned
//! components through the same host services as a packaged one.
#![allow(clippy::single_range_in_vec_init)]

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use kernel_core::weight_transform::WeightTransform;
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanV1, WeightRecordV1,
};
use runtime_bundle_abi::{HostServicesTableV1, RankArenaUploadV1};
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

const ROLE: &str = "fixture.rank-images.v1";

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn planned_ranks_upload_with_a_host_view_and_planned_components() {
    let root = Temp(std::env::temp_dir().join(format!("direct-load-host-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&root.0);
    std::fs::create_dir_all(&root.0).unwrap();
    let source: Vec<u8> = (0..16000_u64).map(|i| (i * 13 + 1) as u8).collect();
    let header = serde_json::to_vec(&serde_json::json!({
        "table": {"dtype": "U8", "shape": [2000, 8], "data_offsets": [0, 16000]}
    }))
    .unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(&source);
    std::fs::write(root.0.join("model.safetensors"), file).unwrap();
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let plans: Vec<WeightPlanV1> = (0..2_u64)
        .map(|rank| WeightPlanV1 {
            ranks: 2,
            rank: rank as u32,
            image_role: ROLE.into(),
            alignment: 64,
            arena_bytes: 64 * 200,
            views: vec![
                PlannedViewV1 {
                    id: "table".into(),
                    alignment: 64,
                    placement: ViewPlacementV1::Host,
                    offset: 64 * 10,
                    byte_length: 8000,
                },
                // Device views after the cut move down by it; one reaching
                // into the cut has no arena address.
                PlannedViewV1 {
                    id: "tail".into(),
                    alignment: 64,
                    placement: ViewPlacementV1::Device,
                    offset: 64 * 10 + 8000,
                    byte_length: 64,
                },
                PlannedViewV1 {
                    id: "torn".into(),
                    alignment: 1,
                    placement: ViewPlacementV1::Device,
                    offset: 8600,
                    byte_length: 100,
                },
            ],
            components: vec![NamedBytesV1 {
                name: format!("chop.bundle-metadata.fixture.index.r{rank}"),
                bytes: format!("{{\"rank\":{rank}}}").into_bytes(),
            }],
            records: vec![
                WeightRecordV1 {
                    arena_offset: 0,
                    byte_length: 80,
                    transform: WeightTransform::Copy.id(),
                    params: vec![],
                    sources: vec![RecordSourceV1 {
                        file: 0,
                        tensor: "table".into(),
                    }],
                    axis: Some(0),
                    ranges: vec![rank * 10..rank * 10 + 10],
                    stored_shape: vec![10, 8],
                },
                WeightRecordV1 {
                    arena_offset: 64 * 10,
                    byte_length: 8000,
                    transform: WeightTransform::Copy.id(),
                    params: vec![],
                    sources: vec![RecordSourceV1 {
                        file: 0,
                        tensor: "table".into(),
                    }],
                    axis: Some(0),
                    ranges: vec![rank * 1000..rank * 1000 + 1000],
                    stored_shape: vec![1000, 8],
                },
            ],
        })
        .collect();
    let identities: Vec<String> = plans
        .iter()
        .map(|plan| {
            runtime_weights::sha256_bytes(
                &plan.identity_document(&checkpoint.identity(), &implementation_v1([0; 32])),
            )
        })
        .collect();
    let image = PlannedImageV1::new(
        checkpoint,
        plans,
        None,
        implementation_v1([0; 32]),
        DirectLoadOptionsV1 {
            threads_per_rank: 2,
            read_ahead: true,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )
    .unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table_service = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table_service.caller();
    let mut ranks = Vec::new();
    for rank in 0..2 {
        let context = host.select_context(rank, rank as i32).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        ranks.push((context, pool, stream));
    }
    let compiled = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let uploads: Vec<_> = ranks
        .iter()
        .enumerate()
        .map(|(rank, &(context, pool, stream))| RankArenaUploadV1 {
            context,
            pool,
            stream,
            image: compiled,
            rank: rank as u32,
        })
        .collect();
    let arenas = host.upload_rank_arenas(&uploads).unwrap();
    for (rank, arena) in arenas.iter().enumerate() {
        let stream = ranks[rank].2;
        let read = |span: runtime_bundle_abi::DeviceBufferSpanV1| {
            let mut bytes = vec![0_u8; span.length as usize];
            host.copy_device_to_host(stream, &mut bytes, span).unwrap();
            bytes
        };
        let whole = host
            .resolve_compiled_image_view(compiled, *arena, rank as u32, b"")
            .unwrap()
            .buffer;
        // 8000 host bytes, rounded to 64, left the arena.
        assert_eq!(whole.length, 64 * 200 - 8000_u64.div_ceil(64) * 64);
        let mut expected = source[rank * 80..rank * 80 + 80].to_vec();
        expected.resize(whole.length as usize, 0);
        assert_eq!(read(whole), expected);
        let host_table = host
            .resolve_compiled_image_view(compiled, *arena, rank as u32, b"table")
            .unwrap()
            .buffer;
        assert_eq!(read(host_table), source[rank * 8000..rank * 8000 + 8000]);
        let tail = host
            .resolve_compiled_image_view(compiled, *arena, rank as u32, b"tail")
            .unwrap()
            .buffer;
        assert_eq!(
            (tail.buffer, tail.offset),
            (whole.buffer, whole.offset + 64 * 10)
        );
        assert_eq!(read(tail), vec![0; 64]);
        assert!(
            host.resolve_compiled_image_view(compiled, *arena, rank as u32, b"torn")
                .is_err()
        );
        let role = format!("chop.bundle-metadata.fixture.index.r{rank}");
        let component = host.open_authenticated_component(role.as_bytes()).unwrap();
        let metadata = host.inspect_authenticated_component(component).unwrap();
        let mut role_bytes = vec![0; metadata.role_length as usize];
        let mut bytes = vec![0; metadata.byte_length as usize];
        host.read_authenticated_component(component, &mut role_bytes, &mut bytes)
            .unwrap();
        assert_eq!(bytes, format!("{{\"rank\":{rank}}}").into_bytes());
        host.release_authenticated_component(component).unwrap();
        // The rank identity a bundle checks its index against is the plan's,
        // never an image digest.
        let role = format!("chop.compiled-image-identity.v1/{ROLE}/{rank}");
        let component = host.open_authenticated_component(role.as_bytes()).unwrap();
        let metadata = host.inspect_authenticated_component(component).unwrap();
        let mut role_bytes = vec![0; metadata.role_length as usize];
        let mut bytes = vec![0; metadata.byte_length as usize];
        host.read_authenticated_component(component, &mut role_bytes, &mut bytes)
            .unwrap();
        let identity: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(identity["byte_length"], 64 * 200);
        assert_eq!(identity["direct_identity"], identities[rank]);
        assert!(identity.get("sha256").is_none());
        host.release_authenticated_component(component).unwrap();
    }
    for arena in arenas.into_iter().rev() {
        host.release_rank_arena(arena).unwrap();
    }
    host.release_compiled_image(compiled).unwrap();
    for (context, pool, stream) in ranks.into_iter().rev() {
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
    assert_eq!(backend.live_resource_count(), 0);
}

/// A one-rank plan of a copied tensor over a fresh checkpoint.
fn one_rank(root: &std::path::Path) -> (Arc<DirectCheckpointV1>, Vec<WeightPlanV1>) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root).unwrap();
    let header = serde_json::to_vec(&serde_json::json!({
        "table": {"dtype": "U8", "shape": [64, 8], "data_offsets": [0, 512]}
    }))
    .unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend((0..512_u32).map(|i| i as u8));
    std::fs::write(root.join("model.safetensors"), file).unwrap();
    let checkpoint = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let plan = WeightPlanV1 {
        ranks: 1,
        rank: 0,
        image_role: ROLE.into(),
        alignment: 64,
        arena_bytes: 512,
        views: vec![],
        components: vec![],
        records: vec![WeightRecordV1 {
            arena_offset: 0,
            byte_length: 512,
            transform: WeightTransform::Copy.id(),
            params: vec![],
            sources: vec![RecordSourceV1 {
                file: 0,
                tensor: "table".into(),
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![64, 8],
        }],
    };
    (checkpoint, vec![plan])
}

/// Host views the arena cannot give up are refused when the rank uploads,
/// before anything is allocated or mapped: a host span off the image
/// alignment, two host spans that overlap, a view aligned above the image
/// alignment (the cut could move it off its own), and cuts that leave the
/// arena no device bytes.
#[test]
fn planned_host_views_that_cannot_be_cut_out_are_refused_at_upload() {
    use runtime_bundle_abi::HostServicesBackendV1;
    let view = |id: &str, alignment, offset, byte_length, host: bool| PlannedViewV1 {
        id: id.into(),
        alignment,
        placement: if host {
            ViewPlacementV1::Host
        } else {
            ViewPlacementV1::Device
        },
        offset,
        byte_length,
    };
    let root =
        Temp(std::env::temp_dir().join(format!("direct-load-host-views-{}", std::process::id())));
    for (label, views) in [
        ("unaligned", vec![view("table", 1, 100, 64, true)]),
        (
            "overlapping",
            vec![
                view("table", 64, 128, 128, true),
                view("second", 64, 192, 64, true),
            ],
        ),
        (
            "misaligned after the cut",
            vec![
                view("table", 64, 0, 64, true),
                view("tail", 128, 128, 128, false),
            ],
        ),
        (
            "one cut covers the image",
            vec![view("table", 64, 0, 512, true)],
        ),
        (
            "several cuts cover the image",
            vec![
                view("head", 64, 0, 128, true),
                view("table", 64, 128, 384, true),
            ],
        ),
    ] {
        let (checkpoint, mut plans) = one_rank(&root.0);
        plans[0].views = views;
        let image = PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap();
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_planned_compiled_image(Arc::new(image))
            .unwrap();
        let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let host = table.caller();
        let context = host.select_context(0, 0).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        let compiled = host.open_compiled_image(ROLE.as_bytes()).unwrap();
        let live = backend.live_resource_count();
        assert_eq!(
            backend
                .upload_rank_arena(context, pool, stream, compiled, 0)
                .map(|_| ()),
            Err(runtime_bundle_abi::BundleStatusV1::InvalidArgument),
            "{label}"
        );
        assert_eq!(backend.live_resource_count(), live, "{label}");
        assert!(
            backend.with_device(|device| device.mapped_host_objects().is_empty()),
            "{label}"
        );
        host.release_compiled_image(compiled).unwrap();
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
        assert_eq!(backend.live_resource_count(), 0, "{label}");
    }
}

/// Every path that completes a direct upload refuses a checkpoint that
/// changed after it was planned, and leaves no arena behind.
#[test]
fn every_upload_path_refuses_a_checkpoint_changed_after_planning() {
    use runtime_bundle_abi::HostServicesBackendV1;
    for batched in [false, true] {
        let root = Temp(std::env::temp_dir().join(format!(
            "direct-load-changed-{}-{batched}",
            std::process::id()
        )));
        let (checkpoint, plans) = one_rank(&root.0);
        let image = PlannedImageV1::new(
            checkpoint,
            plans,
            None,
            implementation_v1([0; 32]),
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap();
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
            .with_planned_compiled_image(Arc::new(image))
            .unwrap();
        let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let host = table.caller();
        let context = host.select_context(0, 0).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        let compiled = host.open_compiled_image(ROLE.as_bytes()).unwrap();
        // The checkpoint grows after planning: its identity no longer holds.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(root.0.join("model.safetensors"))
            .unwrap();
        std::io::Write::write_all(&mut file, &[0; 16]).unwrap();
        let live = backend.live_resource_count();
        let result = if batched {
            host.upload_rank_arenas(&[RankArenaUploadV1 {
                context,
                pool,
                stream,
                image: compiled,
                rank: 0,
            }])
            .map(|_| ())
            .map_err(|error| error.to_string())
        } else {
            backend
                .upload_rank_arena(context, pool, stream, compiled, 0)
                .map(|_| ())
                .map_err(|status| format!("{status:?}"))
        };
        assert!(result.is_err(), "batched {batched}: {result:?}");
        assert_eq!(backend.live_resource_count(), live, "batched {batched}");
        host.release_compiled_image(compiled).unwrap();
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
}

/// A fixture manifest with one ordinary component, that does or does not
/// require the weight plan.
fn manifest(weight_plan: bool) -> runtime_bundle::BundleManifestV1 {
    use runtime_bundle::{
        BundleComponentV1, BundleGraphBucketV1, BundleKvLayoutV1, BundleManifestV1, BundleOpaqueId,
        BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1,
        BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
    };
    let id = |value: &str| BundleOpaqueId::new(value).unwrap();
    let capabilities = if weight_plan {
        vec![id(
            runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
        )]
    } else {
        Vec::new()
    };
    BundleManifestV1::new(
        id("fixture.direct.v1"),
        BundleTargetV1::new(
            vec![id("sm_120f")],
            vec![BundleTopologyV1::new(id("tp1"), 1).unwrap()],
        )
        .unwrap(),
        vec![
            BundleComponentV1::new(
                id("tokenizer"),
                id("fixture.tokenizer.v1"),
                id("fixture.format.v1"),
                BundleRelativePathV1::new("components/file.bin").unwrap(),
                512,
                "a".repeat(64),
            )
            .unwrap(),
        ],
        BundleResourceRequirementsV1::new(1, capabilities).unwrap(),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1024,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .unwrap(),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .unwrap(),
        vec![BundleGraphBucketV1::new(id("decode-1"), vec![id("tokenizer")], Vec::new()).unwrap()],
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap()
}

/// A package that requires the weight plan loads its weights only from a
/// checkpoint; a package that neither requires it nor plans a component
/// serves without one and uploads no rank image.
#[test]
fn only_a_weight_plan_package_needs_a_checkpoint() {
    use runtime_host::{check_planned_components_v1, requires_weight_plan_v1};
    let direct_only = manifest(true);
    assert!(requires_weight_plan_v1(&direct_only));
    assert!(check_planned_components_v1(&direct_only, Some(&[ROLE])).is_ok());
    let refusal = check_planned_components_v1(&direct_only, None).unwrap_err();
    assert!(refusal.contains("--checkpoint DIR"), "{refusal}");
    let plain = manifest(false);
    assert!(!requires_weight_plan_v1(&plain));
    assert!(check_planned_components_v1(&plain, None).is_ok());
    assert!(check_planned_components_v1(&plain, Some(&[ROLE])).is_ok());
}

/// A planned component (a drafter image declared without a file) serves
/// only from a checkpoint whose plans supply it; without one it is refused.
#[test]
fn a_planned_component_serves_only_when_the_plan_supplies_it() {
    use runtime_bundle::{
        BundleComponentV1, BundleGraphBucketV1, BundleManifestV1, BundleOpaqueId,
        BundlePlannedComponentV1, BundleRelativePathV1, BundleResourceRequirementsV1,
    };
    use runtime_host::check_planned_components_v1;
    assert_eq!(
        runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1,
        runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1
    );
    let id = |value: &str| BundleOpaqueId::new(value).unwrap();
    let base = manifest(true);
    let drafter = "fixture.drafter-image.v1";
    let planned = BundleManifestV1::new_with_planned_v1(
        base.id().clone(),
        base.target().clone(),
        vec![
            BundleComponentV1::new(
                id("tokenizer"),
                id("fixture.tokenizer.v1"),
                id("fixture.format.v1"),
                BundleRelativePathV1::new("components/file.bin").unwrap(),
                512,
                "a".repeat(64),
            )
            .unwrap(),
        ],
        vec![
            BundlePlannedComponentV1::new(
                id("drafter-images"),
                id(drafter),
                id("fixture.image.v1"),
            )
            .unwrap(),
        ],
        BundleResourceRequirementsV1::new(1, vec![id(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1)])
            .unwrap(),
        base.kv_layout().clone(),
        base.state_layout().clone(),
        vec![
            BundleGraphBucketV1::new(
                id("decode-1"),
                vec![id("drafter-images"), id("tokenizer")],
                Vec::new(),
            )
            .unwrap(),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    let refusal = check_planned_components_v1(&planned, None).unwrap_err();
    assert!(refusal.contains("'drafter-images'"), "{refusal}");
    assert!(refusal.contains("--checkpoint DIR"), "{refusal}");
    assert!(check_planned_components_v1(&planned, Some(&[ROLE, drafter])).is_ok());
    let declined = check_planned_components_v1(&planned, Some(&[ROLE])).unwrap_err();
    assert!(
        declined.contains("declined planned compiled image"),
        "{declined}"
    );
}

/// Exercise the real upload entry point and parse its phase report. The child
/// selects exactly the existing host-view fixture, without a CUDA device.
#[test]
fn host_view_preparation_is_excluded_from_upload_counters() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "planned_ranks_upload_with_a_host_view_and_planned_components",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log = String::from_utf8(output.stderr).unwrap();
    let reports: Vec<serde_json::Value> = log
        .lines()
        .filter_map(|line| line.strip_prefix("startup direct_load: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(reports.len(), 1, "{log}");
    let report = &reports[0];
    assert_eq!(report["counter_scope"], "cumulative");
    assert_eq!(report["preparation"]["counters"]["source_bytes"], 16000);
    assert_eq!(report["upload"]["counters"]["source_bytes"], 160);
    assert_eq!(report["cumulative"]["source_bytes"], 16160);
    assert_eq!(
        report["preparation"]["counters"]["unique_source_bytes"],
        16000
    );
    assert_eq!(report["upload"]["counters"]["unique_source_bytes"], 0);
}
