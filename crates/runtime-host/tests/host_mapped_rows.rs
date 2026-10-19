//! Host-mapped views (`chop.host-mapped-gather.v1`): a planned view placed
//! `Mapped` is the checkpoint's own bytes, mapped read-only once per
//! process, whose rows the host gathers into a device buffer through
//! page-locked staging of its own. The mapped bytes reach neither the device
//! arena nor any page-locked budget.
#![allow(clippy::single_range_in_vec_init)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use kernel_core::weight_transform::{FRAGMENT, FRAGMENT_BF16_16X16, WeightTransform};
use runtime_bundle_abi::weight_plan::{
    PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanV1, WeightRecordV1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, DeviceBufferSpanV1, HostServicesTableV1, MappedRowGatherV1,
    RankArenaUploadV1,
};
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{HostInstanceConfigV1, HostMemoryV1, HostServicesV1Backend, HostTopologyV1};

const ROLE: &str = "fixture.rank-images.v1";
/// Rows of the two tables: 8-byte codes and 1-byte scales per row.
const ROWS: u64 = 1024;

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp(label: &str) -> Temp {
    let root = std::env::temp_dir().join(format!("host-mapped-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    Temp(root)
}

fn codes() -> Vec<u8> {
    (0..ROWS * 8).map(|i| (i * 31 + 7) as u8).collect()
}

fn scales() -> Vec<u8> {
    (0..ROWS).map(|i| (i * 5 + 1) as u8).collect()
}

fn dense() -> Vec<u8> {
    (0..256_u32).map(|i| (255 - i) as u8).collect()
}

/// A checkpoint of a row table's codes and scales and one dense tensor.
fn write_checkpoint(root: &Path) {
    let header = serde_json::to_vec(&serde_json::json!({
        "codes": {"dtype": "F8_E4M3", "shape": [ROWS, 8], "data_offsets": [0, ROWS * 8]},
        "dense": {"dtype": "U8", "shape": [16, 16], "data_offsets": [ROWS * 8, ROWS * 8 + 256]},
        "scales": {"dtype": "F8_E8M0", "shape": [ROWS, 1],
                   "data_offsets": [ROWS * 8 + 256, ROWS * 9 + 256]},
    }))
    .unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(codes());
    file.extend(dense());
    file.extend(scales());
    std::fs::write(root.join("model.safetensors"), file).unwrap();
}

fn view(id: &str, placement: ViewPlacementV1, offset: u64, byte_length: u64) -> PlannedViewV1 {
    PlannedViewV1 {
        id: id.into(),
        alignment: 64,
        placement,
        offset,
        byte_length,
    }
}

fn copy(
    arena_offset: u64,
    tensor: &str,
    rows: Option<std::ops::Range<u64>>,
    stored_shape: Vec<u64>,
) -> WeightRecordV1 {
    let byte_length = stored_shape.iter().product();
    WeightRecordV1 {
        arena_offset,
        byte_length,
        transform: WeightTransform::Copy.id(),
        params: vec![],
        sources: vec![RecordSourceV1 {
            file: 0,
            tensor: tensor.into(),
        }],
        axis: rows.as_ref().map(|_| 0),
        ranges: rows.into_iter().collect(),
        stored_shape,
    }
}

/// Rank `rank` of `ranks`: the dense tensor on the device at 0, its half of
/// the dense rows in host memory at 4096, the whole codes table mapped at
/// 8192, and its own share of the scale rows mapped at 16384.
fn plan(ranks: u32, rank: u32) -> WeightPlanV1 {
    let share = ROWS / u64::from(ranks);
    let first = share * u64::from(rank);
    WeightPlanV1 {
        ranks,
        rank,
        image_role: ROLE.into(),
        alignment: 64,
        arena_bytes: 16384 + 1024,
        views: vec![
            view("dense", ViewPlacementV1::Device, 0, 256),
            view("half", ViewPlacementV1::Host, 4096, 128),
            view(
                "codes",
                ViewPlacementV1::Mapped { prewarm: false },
                8192,
                ROWS * 8,
            ),
            view(
                "scales",
                ViewPlacementV1::Mapped { prewarm: false },
                16384,
                share,
            ),
        ],
        components: vec![],
        records: vec![
            copy(0, "dense", None, vec![16, 16]),
            copy(
                4096,
                "dense",
                Some(8 * u64::from(rank % 2)..8 * u64::from(rank % 2) + 8),
                vec![8, 16],
            ),
            copy(8192, "codes", None, vec![ROWS, 8]),
            copy(16384, "scales", Some(first..first + share), vec![share, 1]),
        ],
    }
}

fn options() -> DirectLoadOptionsV1 {
    DirectLoadOptionsV1 {
        threads_per_rank: 1,
        read_ahead: false,
        direct_reads: false,
        shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
    }
}

fn planned(root: &Path, plans: Vec<WeightPlanV1>) -> Result<PlannedImageV1, String> {
    let checkpoint = Arc::new(DirectCheckpointV1::open(root)?);
    PlannedImageV1::new(
        checkpoint,
        plans,
        None,
        implementation_v1([0; 32]),
        options(),
    )
}

#[test]
fn mapped_views_map_the_checkpoint_bytes_and_leave_the_arena() {
    let root = temp("serve");
    write_checkpoint(&root.0);
    const RANKS: u32 = 4;
    let image = planned(&root.0, (0..RANKS).map(|rank| plan(RANKS, rank)).collect()).unwrap();
    // Every rank names the whole codes table: one mapping serves them all.
    let first = image.mapped_view(0, b"codes").unwrap();
    for rank in 1..RANKS {
        assert!(Arc::ptr_eq(
            first,
            image.mapped_view(rank, b"codes").unwrap()
        ));
    }
    assert!(image.mapped_view(0, b"half").is_none());
    // No view asked to be prewarmed.
    assert_eq!(image.prewarm(), None);
    assert_eq!(image.mapped_bytes(), ROWS * 8 + ROWS);
    let config =
        HostInstanceConfigV1::new(HostTopologyV1::new((0..RANKS as i32).collect()).unwrap())
            .with_planned_compiled_image(Arc::new(image))
            .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let ranks: Vec<_> = (0..RANKS)
        .map(|rank| {
            let context = host.select_context(rank, rank as i32).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            (context, pool, stream)
        })
        .collect();
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
    let share = ROWS / u64::from(RANKS);
    for (rank, arena) in arenas.iter().enumerate() {
        let rank_u32 = rank as u32;
        let stream = ranks[rank].2;
        let read = |span: DeviceBufferSpanV1| {
            let mut bytes = vec![0_u8; span.length as usize];
            host.copy_device_to_host(stream, &mut bytes, span).unwrap();
            bytes
        };
        let whole = host
            .resolve_compiled_image_view(compiled, *arena, rank_u32, b"")
            .unwrap()
            .buffer;
        // The host view (128 bytes, rounded to 64) and both mapped views
        // leave the arena; the device keeps the rest.
        assert_eq!(whole.length, 16384 + 1024 - 128 - ROWS * 8 - share);
        let dense_view = host
            .resolve_compiled_image_view(compiled, *arena, rank_u32, b"dense")
            .unwrap()
            .buffer;
        assert_eq!(read(dense_view), dense());
        let half = host
            .resolve_compiled_image_view(compiled, *arena, rank_u32, b"half")
            .unwrap()
            .buffer;
        let row = 8 * (rank % 2) * 16;
        assert_eq!(read(half), dense()[row..row + 128]);
        // A mapped view has no device address.
        for id in [&b"codes"[..], b"scales"] {
            assert_eq!(
                host.resolve_compiled_image_view(compiled, *arena, rank_u32, id),
                Err(BundleAbiError::Status(BundleStatusV1::NotFound))
            );
        }
    }
    // Page-locked memory holds the host views only; the mapped views' bytes
    // are the page cache's.
    let held = |gather_staging_bytes| HostMemoryV1 {
        pinned_bytes: u64::from(RANKS) * 128 + gather_staging_bytes,
        gather_staging_bytes,
        mapped_view_bytes: u64::from(RANKS) * (ROWS * 8 + share),
    };
    assert_eq!(backend.host_memory_v1(), held(0));

    // Gather rows of rank 1's views into a device buffer: the host reads them
    // in 12-byte staging chunks, which split rows, and copies each chunk on
    // the rank's stream.
    backend.with_device_mut(|device| device.set_fill_chunk_bytes(12));
    let (_, pool, stream) = ranks[1];
    let device = host.allocate_device_buffer(pool, stream, 64, 64).unwrap();
    let read = |span: DeviceBufferSpanV1| {
        let mut bytes = vec![0_u8; span.length as usize];
        host.copy_device_to_host(stream, &mut bytes, span).unwrap();
        bytes
    };
    let gather = |view_id: &'static [u8], row_bytes: u64, rows: &'static [u32]| MappedRowGatherV1 {
        image: compiled,
        rank: 1,
        view_id,
        row_bytes,
        rows,
    };
    let picked: &[u32] = &[3, 900, 3, 17, 1023];
    let codes_span = DeviceBufferSpanV1::new(device, 8, 40);
    host.gather_mapped_rows(stream, codes_span, gather(b"codes", 8, picked))
        .unwrap();
    let expected: Vec<u8> = picked
        .iter()
        .flat_map(|&row| codes()[row as usize * 8..row as usize * 8 + 8].to_vec())
        .collect();
    assert_eq!(read(codes_span), expected);
    // Row indices count from the start of the view: rank 1's scale rows.
    let scales_span = DeviceBufferSpanV1::new(device, 48, 3);
    host.gather_mapped_rows(stream, scales_span, gather(b"scales", 1, &[0, 255, 7]))
        .unwrap();
    let first = share as usize;
    assert_eq!(
        read(scales_span),
        [scales()[first], scales()[first + 255], scales()[first + 7]]
    );
    // The stream's staging chunk is page-locked memory the host owns.
    assert_eq!(backend.host_memory_v1(), held(12));
    let refused = |status| Err(BundleAbiError::Status(status));
    let invalid = refused(BundleStatusV1::InvalidArgument);
    // Only a view the image maps for that rank is gathered from.
    for id in [&b"half"[..], b"dense", b"absent"] {
        let request = MappedRowGatherV1 {
            view_id: id,
            ..gather(b"codes", 8, &[0])
        };
        assert_eq!(
            host.gather_mapped_rows(stream, DeviceBufferSpanV1::new(device, 0, 8), request),
            refused(BundleStatusV1::NotFound)
        );
    }
    for (request, length) in [
        // A row past the view's end.
        (gather(b"codes", 8, &[1024]), 8),
        (gather(b"scales", 1, &[256]), 1),
        // No rows, a row length that does not divide the view, and a
        // destination of another length.
        (gather(b"codes", 8, &[]), 0),
        (gather(b"codes", 0, &[1]), 0),
        (gather(b"codes", 24, &[1]), 24),
        (gather(b"codes", 8, &[1, 2]), 8),
        (
            MappedRowGatherV1 {
                rank: RANKS,
                ..gather(b"codes", 8, &[1])
            },
            8,
        ),
    ] {
        assert_eq!(
            host.gather_mapped_rows(stream, DeviceBufferSpanV1::new(device, 0, length), request),
            invalid
        );
    }
    // A graph would replay the copy of rows the host gathered once.
    host.begin_graph_capture(stream).unwrap();
    assert_eq!(
        host.gather_mapped_rows(stream, codes_span, gather(b"codes", 8, picked)),
        refused(BundleStatusV1::InvalidState)
    );
    let graph = host.end_graph_capture(stream).unwrap();
    host.release_graph(graph).unwrap();
    host.release_device_buffer(device).unwrap();

    // A gather of megabytes is read on several threads, chunk by chunk.
    let chunk = (1 << 20) + 4;
    backend.with_device_mut(|device| device.set_fill_chunk_bytes(chunk));
    let rows: Vec<u32> = (0..3 * chunk as u32 / 8)
        .map(|i| (i * 7 + 1) % 1024)
        .collect();
    let bytes = rows.len() as u64 * 8;
    let large = host
        .allocate_device_buffer(pool, stream, bytes, 64)
        .unwrap();
    let span = DeviceBufferSpanV1::new(large, 0, bytes);
    host.gather_mapped_rows(
        stream,
        span,
        MappedRowGatherV1 {
            rows: &rows,
            ..gather(b"codes", 8, &[])
        },
    )
    .unwrap();
    let table = codes();
    let expected: Vec<u8> = rows
        .iter()
        .flat_map(|&row| table[row as usize * 8..row as usize * 8 + 8].to_vec())
        .collect();
    assert!(read(span) == expected);
    host.release_device_buffer(large).unwrap();
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
    // A released stream's staging goes with it.
    assert_eq!(backend.host_memory_v1(), HostMemoryV1::default());
}

/// A mapped view that asks to be prewarmed is read into the page cache and
/// mapped while the image loads, each shared mapping once; the ranks agree
/// on the placement, prewarm included.
#[test]
fn prewarmed_views_are_resident_after_load() {
    let root = temp("prewarm");
    write_checkpoint(&root.0);
    let prewarmed = |ranks: u32, rank: u32, prewarm: bool| {
        let mut plan = plan(ranks, rank);
        for view in &mut plan.views {
            if view.id == "codes" {
                view.placement = ViewPlacementV1::Mapped { prewarm };
            }
        }
        plan
    };
    let image = planned(
        &root.0,
        (0..2).map(|rank| prewarmed(2, rank, true)).collect(),
    )
    .unwrap();
    let prewarm = image.prewarm().unwrap();
    // The codes table alone, once for both ranks.
    assert_eq!(prewarm.bytes, ROWS * 8);
    let (resident, pages) = image
        .mapped_view(1, b"codes")
        .unwrap()
        .resident_pages()
        .unwrap();
    assert_eq!(resident, pages);
    assert!(prewarm.resident_bytes >= prewarm.bytes);
    assert_eq!(
        prewarm.margin_bytes,
        runtime_host::direct::PREWARM_MARGIN_BYTES_DEFAULT
    );
    assert!(prewarm.available_bytes.is_some_and(|bytes| bytes > 0));
    let refused = planned(&root.0, vec![prewarmed(2, 0, true), prewarmed(2, 1, false)])
        .err()
        .unwrap();
    assert!(
        refused.contains("view codes differs between ranks"),
        "{refused}"
    );
}

/// A mapped view takes exactly one plain copy of tensor bytes, whole or one
/// leading-axis range, stored unchanged; anything else is refused by name
/// before a byte is mapped.
#[test]
fn mapped_views_that_are_not_plain_tensor_rows_are_refused() {
    let root = temp("refuse");
    write_checkpoint(&root.0);
    let mapped = |record: WeightRecordV1, byte_length| WeightPlanV1 {
        ranks: 1,
        rank: 0,
        image_role: ROLE.into(),
        alignment: 64,
        arena_bytes: 16384,
        views: vec![view(
            "rows",
            ViewPlacementV1::Mapped { prewarm: false },
            0,
            byte_length,
        )],
        components: vec![],
        records: vec![record],
    };
    let fragment = WeightRecordV1 {
        transform: FRAGMENT,
        params: FRAGMENT_BF16_16X16.to_le_bytes().to_vec(),
        ..copy(0, "dense", None, vec![16, 16])
    };
    let scales_checked = WeightRecordV1 {
        params: kernel_core::weight_transform::COPY_SCALES
            .to_le_bytes()
            .to_vec(),
        ..copy(0, "codes", None, vec![ROWS, 8])
    };
    for (label, plan, reason) in [
        (
            "no record",
            WeightPlanV1 {
                records: vec![],
                ..mapped(copy(0, "dense", None, vec![16, 16]), 256)
            },
            "exactly one record",
        ),
        (
            "two records",
            WeightPlanV1 {
                records: vec![
                    copy(0, "dense", Some(0..8), vec![8, 16]),
                    copy(128, "dense", Some(8..16), vec![8, 16]),
                ],
                ..mapped(copy(0, "dense", None, vec![16, 16]), 256)
            },
            "exactly one record",
        ),
        (
            "shorter record",
            mapped(copy(0, "dense", Some(0..8), vec![8, 16]), 256),
            "cover exactly",
        ),
        ("transform", mapped(fragment, 256), "plain copy"),
        (
            "checked scales",
            mapped(scales_checked, ROWS * 8),
            "plain copy",
        ),
        (
            "padded",
            mapped(copy(0, "dense", Some(0..8), vec![8, 32]), 256),
            "unchanged",
        ),
        (
            "inner axis",
            mapped(
                WeightRecordV1 {
                    axis: Some(1),
                    ranges: vec![0..8],
                    ..copy(0, "dense", None, vec![16, 8])
                },
                128,
            ),
            "leading axis",
        ),
        (
            "two ranges",
            mapped(
                WeightRecordV1 {
                    ranges: vec![0..4, 8..12],
                    ..copy(0, "dense", Some(0..8), vec![8, 16])
                },
                128,
            ),
            "leading axis",
        ),
        (
            "rows outside",
            mapped(copy(0, "dense", Some(8..17), vec![9, 16]), 144),
            "outside the tensor",
        ),
        (
            "empty rows",
            mapped(copy(0, "dense", Some(4..4), vec![0, 16]), 64),
            "exactly one record",
        ),
        (
            "unknown tensor",
            mapped(copy(0, "absent", None, vec![16, 16]), 256),
            "absent",
        ),
    ] {
        let error = planned(&root.0, vec![plan]).expect_err(label);
        assert!(error.contains(reason), "{label}: {error}");
    }
    // The same plan with the view on the device loads.
    let mut device = mapped(copy(0, "dense", Some(0..8), vec![8, 16]), 128);
    device.views[0].placement = ViewPlacementV1::Device;
    planned(&root.0, vec![device]).unwrap();
}

/// Mapping reads the inode the checkpoint opened, checked against its header
/// first: a file replaced after the checkpoint was opened is refused, and a
/// link outside a confined snapshot is never followed.
#[test]
fn mapping_refuses_replaced_files_and_links() {
    let root = temp("replaced");
    write_checkpoint(&root.0);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let plans = vec![plan(1, 0)];
    let replacement = root.0.join("replacement");
    std::fs::copy(root.0.join("model.safetensors"), &replacement).unwrap();
    std::fs::rename(&replacement, root.0.join("model.safetensors")).unwrap();
    let error = PlannedImageV1::new(
        checkpoint,
        plans,
        None,
        implementation_v1([0; 32]),
        options(),
    )
    .unwrap_err();
    assert!(error.contains("changed"), "{error}");

    let linked = temp("linked");
    std::os::unix::fs::symlink(
        root.0.join("model.safetensors"),
        linked.0.join("model.safetensors"),
    )
    .unwrap();
    let error = DirectCheckpointV1::open(&linked.0).unwrap_err();
    assert!(error.contains("link"), "{error}");
}

#[path = "host_mapped_rows/async_gather.rs"]
mod async_gather;

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
#[path = "host_mapped_rows/peer_results.rs"]
mod peer_results;
