//! The vision encoder on the fake host: create with and without its roles,
//! image records through the tower into the arena, prefill injection, and
//! text steps that launch exactly what they did before.

use runtime_bundle_sdk::{BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3};

use super::*;
use bundle_runtime::encoder::{Mode, side};
use encoder_spec::{Patchify, geometry::grid_positions};

// The stems of the tower's generated entries (`Entry::stem`).
const PATCH: &str = "chop_enc_linear_pairs_n1152_k1536_bias_f32_v1";
const POSITION: &str =
    "chop_enc_position_learned_grid_bilinear_align_corners_f32_w1152_table2304_v1";
const LAYERNORM: &str = "chop_enc_norm_layer_f32_w1152_v1";
const QKV: &str = "chop_enc_linear_pairs_n3456_k1152_bias_f32_v1";
const ROPE: &str = "chop_enc_rotary_axial_2d_q16_kv16_d72_r72_neox_v1";
const ATTENTION: &str = "chop_enc_attention_varlen_pairs_q16_kv16_d72_full_v1";
const PROJ: &str = "chop_enc_linear_pairs_n1152_k1152_bias_residual_f32_v1";
const FC1: &str = "chop_enc_linear_pairs_n4352_k1152_bias_gelu_tanh_f32_v1";
const FC2: &str = "chop_enc_linear_pairs_n1152_k4352_bias_residual_f32_v1";
const MERGER_FC1: &str = "chop_enc_linear_pairs_n4608_k4608_bias_gelu_erf_f32_v1";
const MERGER_FC2: &str = "chop_enc_linear_pairs_n2560_k4608_bias_bf16_v1";
const OVERRIDE: &str = "chop_enc_layout_tokens_w2560_v1";

const IMAGE_PAD: u32 = 248_056;
const ROW_BYTES: u64 = 2560 * 2;

/// The vision encoder's image role and the first `indexes` ranks' index
/// roles.
pub(super) fn encoder_roles(indexes: u32) -> Vec<Vec<u8>> {
    let mut roles = vec![crate::encoder::views::IMAGE_ROLE.to_vec()];
    roles.extend((0..indexes).map(|rank| crate::encoder::views::index_role(rank).into_bytes()));
    roles
}

fn create_with(host: &Rc<FakeHost>, roles: &[Vec<u8>]) -> Result<BundleInstanceV1, BundleAbiError> {
    api().create(
        HostServicesTableV1::new(Rc::clone(host)),
        &configuration_with(false, roles),
    )
}

fn execute(instance: &mut BundleInstanceV1, request: BatchRequestV1) -> Result<(), BundleAbiError> {
    instance.execute(&request.encode().unwrap(), &mut ViewSink::default())
}

/// Two text rows, 24 pads of an 8 x 12 image, one text row, 4 pads of a
/// 4 x 4 image and one text row.
fn media_prompt() -> Vec<u32> {
    let mut prompt = vec![1, 2];
    prompt.extend([IMAGE_PAD; 24]);
    prompt.push(3);
    prompt.extend([IMAGE_PAD; 4]);
    prompt.push(4);
    prompt
}

fn record(geometry: [u32; 4], page: u64, offset: u64) -> BatchMediaEncodeV3 {
    let patches = u64::from(geometry[1] * geometry[2]);
    BatchMediaEncodeV3 {
        encoder: 0,
        geometry,
        output_rows: (patches / 4) as u32,
        pages: vec![page],
        payload_offset: offset,
        payload_bytes: patches * 6144,
    }
}

/// The two images into arena pages 5 and 9, injected at prompt rows 2 and
/// 27 of item 0.
fn media() -> BatchMediaSectionV3 {
    let first = record([1, 8, 12, 0], 5, 0);
    let second = record([1, 4, 4, 0], 9, first.payload_bytes);
    let bytes = first.payload_bytes + second.payload_bytes;
    BatchMediaSectionV3 {
        encodes: vec![first, second],
        injects: vec![
            BatchMediaInjectV3 {
                item_index: 0,
                item_row: 2,
                rows: 24,
                arena_row: 5 * 64,
            },
            BatchMediaInjectV3 {
                item_index: 0,
                item_row: 27,
                rows: 4,
                arena_row: 9 * 64,
            },
        ],
        payload: (0..bytes).map(|byte| byte as u8).collect(),
    }
}

/// Launch names per stream, in issue order, from launch `from` on.
fn by_stream(host: &FakeHost, from: usize) -> BTreeMap<u32, Vec<String>> {
    let mut streams = BTreeMap::<u32, Vec<String>>::new();
    for (stream, name, _) in &host.launches.borrow()[from..] {
        streams.entry(*stream).or_default().push(name.clone());
    }
    streams
}

/// The full symbol of the entry whose symbol reads `stem`.
fn full(stem: &str) -> &'static str {
    super::sm120::encoder_symbol(stem)
}

fn tower_names() -> Vec<String> {
    let mut names = vec![PATCH, POSITION];
    for _ in 0..27 {
        names.extend([LAYERNORM, QKV, ROPE, ATTENTION, PROJ, LAYERNORM, FC1, FC2]);
    }
    names.extend([LAYERNORM, MERGER_FC1, MERGER_FC2]);
    names
        .into_iter()
        .map(|stem| full(stem).to_owned())
        .collect()
}

fn bytes_of<T: Copy, const N: usize>(values: &[T], bytes: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&value| bytes(value)).collect()
}

fn uploads_of(host: &FakeHost, from: usize, bytes: &[u8]) -> usize {
    host.uploads.borrow()[from..]
        .iter()
        .filter(|(_, uploaded)| uploaded == bytes)
        .count()
}

#[test]
fn create_takes_every_encoder_role_or_none() {
    for indexes in [0, 3] {
        let host = Rc::new(FakeHost::default());
        let refused = create_with(&host, &encoder_roles(indexes)).err();
        assert!(
            matches!(
                refused,
                Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
            ),
            "{indexes} indexes"
        );
    }
    let host = Rc::new(FakeHost::default());
    let only_indexes = encoder_roles(RANKS)[1..].to_vec();
    assert!(create_with(&host, &only_indexes).is_err());

    // Each rank holds the encoder image and its buffers beyond the text
    // program's.
    let mut bytes = Vec::new();
    for roles in [Vec::new(), encoder_roles(RANKS)] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_with(&host, &roles).unwrap();
        bytes.push(host.extents.borrow().values().sum::<u64>());
        assert_eq!(host.encoder_image.get().is_some(), !roles.is_empty());
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
    assert_eq!(
        bytes[1] - bytes[0],
        u64::from(RANKS)
            * (2_101_411_856 + crate::encoder::views::image_bytes_for(Shape::TP4.geometry))
    );
}

/// Each encoder index names the identity of the image the host serves for
/// its rank: a file index names the file's digest, a planned one the plan's
/// identity. Any other pairing is refused before the image is opened.
#[test]
fn create_refuses_an_encoder_index_that_names_another_image() {
    let file = serde_json::json!({"byte_length": crate::encoder::views::image_bytes_for(Shape::TP4.geometry),
        "sha256": "0".repeat(64)});
    let planned = serde_json::json!({"byte_length": crate::encoder::views::image_bytes_for(Shape::TP4.geometry),
        "direct_identity": "b".repeat(64)});
    for (index, served, accepted) in [
        (None, file.clone(), true),
        (
            Some(("direct_identity", "b".repeat(64))),
            planned.clone(),
            true,
        ),
        // A file image with an index that claims a planned identity.
        (
            Some(("direct_identity", "b".repeat(64))),
            file.clone(),
            false,
        ),
        // A file image whose digest differs from the index's.
        (Some(("sha256", "c".repeat(64))), file.clone(), false),
        // A planned image whose identity differs from the index's.
        (
            Some(("direct_identity", "d".repeat(64))),
            planned.clone(),
            false,
        ),
        // A planned image with an index that names a file digest.
        (None, planned.clone(), false),
    ] {
        let host = Rc::new(FakeHost::default());
        *host.encoder_index_identity.borrow_mut() = index.clone();
        *host.encoder_served_identity.borrow_mut() = Some(served.clone());
        let result = create_with(&host, &encoder_roles(RANKS));
        assert_eq!(result.is_ok(), accepted, "{index:?} {served}");
        match result {
            Ok(mut instance) => instance.destroy().unwrap(),
            Err(_) => assert_eq!(host.encoder_image.get(), None, "{index:?} {served}"),
        }
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

#[test]
fn media_is_refused_without_the_encoder() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let prompt = media_prompt();
    let journal = host.journal.borrow().len();
    let refused = execute(
        &mut instance,
        step(&[(10, 0, 0, &prompt)]).with_media(media()),
    );
    assert!(matches!(
        refused,
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert_eq!(host.journal.borrow().len(), journal);
}

#[test]
fn invalid_records_are_refused_before_device_work() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let prompt = media_prompt();
    let mut other_encoder = media();
    other_encoder.encodes[1].encoder = 1;
    let mut wrong_rows = media();
    wrong_rows.encodes[0].output_rows = 23;
    let mut video = media();
    video.encodes[1].geometry[0] = 2;
    let mut outside = media();
    outside.injects[0].item_row = 10;
    for section in [other_encoder, wrong_rows, video, outside] {
        let journal = host.journal.borrow().len();
        let refused = execute(
            &mut instance,
            step(&[(10, 0, 0, &prompt)]).with_media(section),
        );
        assert!(matches!(
            refused,
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        ));
        assert_eq!(host.journal.borrow().len(), journal);
    }
    // No transaction was opened: the next step runs.
    run(&host, &mut instance, "text", step(&[(10, 0, 0, &PROMPT)]));
    instance.commit().unwrap();
}

#[test]
fn image_records_run_the_tower_into_their_pages_and_prefill_injects_them() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let prompt = media_prompt();
    let (launches, uploads, copies) = (
        host.launches.borrow().len(),
        host.uploads.borrow().len(),
        host.copies.borrow().len(),
    );
    run(
        &host,
        &mut instance,
        "media",
        step(&[(10, 0, 0, &prompt)]).with_media(media()),
    );

    // Each rank runs both towers, one after the other, before its prefill,
    // and the prefill overrides the embedding once, before the streams
    // expand.
    let tower = tower_names();
    let streams = by_stream(&host, launches);
    assert_eq!(streams.len(), RANKS as usize);
    for names in streams.values() {
        assert_eq!(names[..tower.len()], tower[..]);
        assert_eq!(names[tower.len()..2 * tower.len()], tower[..]);
        let prefill = &names[2 * tower.len()..];
        assert_eq!(prefill[0], "chop_qwen38_ple_stage_v1");
        assert_eq!(count(prefill, full(OVERRIDE)), 1);
        let at = prefill
            .iter()
            .position(|name| name == full(OVERRIDE))
            .unwrap();
        assert_eq!(prefill[at - 1], crate::sm120::entries::EMBEDDING);
        assert_eq!(prefill[at + 1], "chop_row_copy_broadcast_streams_v1");
    }

    // Every encoder launch matches its compiled ABI; attention has no sinks.
    let compiled = super::sm120::encoder_cubins();
    let arguments = host.launch_arguments.borrow().clone();
    for (index, (_, name, _)) in host.launches.borrow().iter().enumerate().skip(launches) {
        let Some(entry) = compiled
            .iter()
            .flat_map(|cubin| &cubin.entries)
            .find(|entry| entry.symbol == *name)
        else {
            continue;
        };
        let arguments = &arguments[index];
        assert_eq!(entry.params.len(), arguments.len(), "{name}");
        for (param, argument) in entry.params.iter().zip(arguments) {
            let null = matches!(argument, KernelArgumentV1::NullDevicePointer);
            assert!(
                matches!(
                    (param, argument),
                    (
                        kernel_build::EntryParam::DevicePointer { .. },
                        KernelArgumentV1::DeviceBuffer(_) | KernelArgumentV1::NullDevicePointer
                    ) | (
                        kernel_build::EntryParam::U32 { .. },
                        KernelArgumentV1::ScalarU32(_)
                    ) | (
                        kernel_build::EntryParam::U64 { .. },
                        KernelArgumentV1::ScalarU64(_)
                    )
                ),
                "{name}"
            );
            let kernel_build::EntryParam::DevicePointer { name: param, .. } = param else {
                continue;
            };
            assert_eq!(null, param == "sinks", "{name} {param}");
        }
        if name == full(OVERRIDE) {
            assert_eq!(arguments[4], KernelArgumentV1::ScalarU64(65_536));
        }
    }

    // Counts, bounds, rotary rows and position taps of each image reach
    // every rank, as HF computes them.
    for (h, w) in [(8_u64, 12_u64), (4, 4)] {
        let patches = (h * w) as u32;
        let positions = grid_positions(Patchify::MergeBlocks, 2, h, w);
        let (cos, sin) = side::rotary(&positions, 10_000.0, 36);
        let (indices, weights) = side::position_taps(&positions, [h, w], [48, 48]);
        for bytes in [
            bytes_of(&[patches, patches / 4], u32::to_le_bytes),
            bytes_of(&[0, patches as i32], i32::to_le_bytes),
            bytes_of(&cos, f32::to_le_bytes),
            bytes_of(&sin, f32::to_le_bytes),
            bytes_of(&indices, i32::to_le_bytes),
            bytes_of(&weights, f32::to_le_bytes),
        ] {
            assert_eq!(uploads_of(&host, uploads, &bytes), RANKS as usize);
        }
    }
    // By hand for the 8 x 12 image: patch 3 is (1, 1), at table row
    // 47/7 and column 47/11; patch 2 is (1, 0), whose first height angle is 1.
    // Uploads of one length come in order, each on every rank before the
    // next: cos before sin, indices before weights.
    let upload = |bytes: u64, nth: usize| {
        host.uploads.borrow()[uploads..]
            .iter()
            .filter(|(span, _)| span.length == bytes)
            .nth(nth)
            .map(|(_, bytes)| bytes.clone())
            .unwrap()
    };
    let (dr, dc) = (47.0_f32 / 7.0 - 6.0, 47.0_f32 / 11.0 - 4.0);
    let (weights, cos) = (upload(96 * 4 * 4, RANKS as usize), upload(96 * 36 * 4, 0));
    let value = |bytes: &[u8], index: usize| {
        f32::from_le_bytes(bytes[4 * index..4 * index + 4].try_into().unwrap())
    };
    assert_eq!(value(&weights, 12), (1.0 - dr) * (1.0 - dc));
    assert_eq!(
        value(&weights, 15),
        (1.0 - (dr - 1.0).abs()) * (1.0 - (dc - 1.0).abs())
    );
    assert!((value(&weights, 15) - dr * dc).abs() < 1e-6);
    assert_eq!(value(&cos, 2 * 36), 1.0_f64.cos() as f32);
    assert_eq!(value(&cos, 2 * 36 + 18), 1.0);

    // Each image's merged rows land in exactly its page, on every rank,
    // before the prefill saves the ring.
    let copies = host.copies.borrow()[copies..].to_vec();
    assert_eq!(copies.len(), 3 * RANKS as usize);
    let arena = copies[..2 * RANKS as usize]
        .iter()
        .map(|(source, destination)| (source.offset, destination.offset, destination.length))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        arena,
        BTreeSet::from([
            (0, 5 * 64 * ROW_BYTES, 24 * ROW_BYTES),
            (0, 9 * 64 * ROW_BYTES, 4 * ROW_BYTES)
        ])
    );
    assert!(
        copies[2 * RANKS as usize..]
            .iter()
            .all(|(_, destination)| destination.length == 32 * RING_STATE_SLOT_BYTES / 2)
    );

    // The source rows name the arena row of every placeholder.
    let mut sources = vec![-1_i64; prompt.len()];
    for (row, source) in sources[2..26].iter_mut().zip(320..) {
        *row = source;
    }
    for (row, source) in sources[27..31].iter_mut().zip(576..) {
        *row = source;
    }
    let source_bytes = bytes_of(&sources, i64::to_le_bytes);
    assert_eq!(uploads_of(&host, uploads, &source_bytes), RANKS as usize);
    instance.commit().unwrap();
}

/// The journal of an encoder instance's create, a step that encodes the two
/// images of `media` and a 32 x 32 image (a payload above one 4 MiB upload
/// window) and injects the first two, and its destroy.
///
/// The shared encoder runtime (2026-10-01) replaced the hand-written
/// tower and its per-rank eager launches. Against the tower's journal
/// (12,074 lines) every stream's lines are equal in order, with buffer
/// handles renamed per stream (a per-stream journal diff at the time):
/// the same uploads, launches, arguments, arena copies and
/// synchronizations. What moved: the ranks' work interleaves (each upload
/// window, then each command, on every rank before the next); the 32 x 32
/// payload uploads in two 4 MiB windows (4 more lines); each rank's regions
/// allocate in the buffer plan's order, after every rank's image view.
// Decode v3 adds four resolves and four releases in each prefill mode;
// this prefill-only journal has no decode launch to rename.
// Prefill v2 adds 16 resolve/release lines and extends flag allocation, zero and
// argument spans from 384 to 1408 bytes. After handle normalization,
// symbols, grids and snapshot copies match v1.
// Plain decode v1 leaving the shared comm cubin and the Qwen decode v1
// entry leaving the decode cubin (2026-10-05) each drop one resolve and one
// release per rank. The prefill v1 pair leaving comm_qwen38 the same day
// drops 16 lines against the v2 pin; every other line matches it after
// renaming handles.
// The text image identity check (2026-10-07) adds one release line per
// rank before any other line, as for the text journals.
// The shared stream broadcast changes the expansion's resolve and launch
// lines, as for the text journals.
const ENCODER_JOURNAL: (usize, u64) = (12094, 0xD5FC3295FA1EB1A4);

#[test]
fn encoder_journal_is_unchanged() {
    let _default_overlap = legacy_overlap_off();
    let host = Rc::new(FakeHost::default());
    crate::pipeline::take_recorded();
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let mut section = media();
    let mut large = record([1, 32, 32, 0], 12, section.payload.len() as u64);
    large.pages = vec![12, 13, 14, 15];
    section
        .payload
        .extend((0..large.payload_bytes).map(|byte| (byte / 7) as u8));
    section.encodes.push(large);
    run(
        &host,
        &mut instance,
        "media",
        step(&[(10, 0, 0, &media_prompt())]).with_media(section),
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
    super::export_rows("encoder", &[&host]);
    let lines = host.journal.borrow().clone();
    let journal = lines.join("\n");
    if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_JOURNAL_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("encoder.txt"), &journal).unwrap();
    }
    assert_eq!(journal_digest(&lines), ENCODER_JOURNAL);
}

/// A launch the host refuses fails the step after every rank that received
/// work drains, and the next step runs. A stream that does not drain
/// poisons the instance: its regions may still be in use.
#[test]
fn encoder_failures_drain_every_stream_and_poison_when_one_does_not() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let launches = host.launches.borrow().len();
    *host.refuse_launch.borrow_mut() = Some((full(QKV).to_owned(), launches + 4 * 4));
    let journal = host.journal.borrow().len();
    let refused = execute(
        &mut instance,
        step(&[(10, 0, 0, &media_prompt())]).with_media(media()),
    );
    assert!(matches!(
        refused,
        Err(BundleAbiError::Status(BundleStatusV1::InternalError))
    ));
    let synchronized = host.journal.borrow()[journal..]
        .iter()
        .filter(|line| line.starts_with("synchronize"))
        .count();
    assert_eq!(synchronized, RANKS as usize);
    assert!(!instance.is_poisoned());
    *host.refuse_launch.borrow_mut() = None;
    run(&host, &mut instance, "text", step(&[(10, 0, 0, &PROMPT)]));
    instance.commit().unwrap();

    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    host.refuse_synchronize.set(true);
    let refused = execute(
        &mut instance,
        step(&[(10, 0, 0, &media_prompt())]).with_media(media()),
    );
    host.refuse_synchronize.set(false);
    assert!(matches!(
        refused,
        Err(BundleAbiError::Status(BundleStatusV1::InternalError))
    ));
    // The poisoned instance refuses every later call and keeps its
    // resources.
    let released = host.released.get();
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
}

/// The streams synchronized after the last journal line that starts with
/// `kind`, from line `from` on.
fn synchronized_after(host: &FakeHost, from: usize, kind: &str) -> BTreeSet<String> {
    let journal = host.journal.borrow();
    let last = from
        + journal[from..]
            .iter()
            .rposition(|line| line.starts_with(kind))
            .unwrap();
    journal[last..]
        .iter()
        .filter_map(|line| line.strip_prefix("synchronize "))
        .map(str::to_owned)
        .collect()
}

/// A device call outside the recorded submission that fails (a later
/// rank's source-row upload, an arena capture's readback) drains every
/// stream before the step fails, and the next step runs.
#[test]
fn device_failures_outside_submission_drain_every_stream() {
    let prompt = media_prompt();
    for fault in ["source rows", "arena readback"] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
        let root = std::env::temp_dir().join(format!("qwen-arena-fault-{}", std::process::id()));
        let kind = match fault {
            "source rows" => {
                // Rank 1's upload of the step's 32 source rows, after every
                // rank's 256-byte position taps of the 4 x 4 image.
                host.fail_upload
                    .set(Some((prompt.len() as u64 * 8, 2 * RANKS + 1)));
                "upload"
            }
            _ => {
                crate::instance::with_encoder(instance.handle(), |encoder| {
                    encoder.arena_capture = Some(root.clone())
                });
                host.fail_download.set(true);
                "download"
            }
        };
        let journal = host.journal.borrow().len();
        let refused = execute(
            &mut instance,
            step(&[(10, 0, 0, &prompt)]).with_media(media()),
        );
        host.fail_download.set(false);
        assert!(
            matches!(
                refused,
                Err(BundleAbiError::Status(BundleStatusV1::InternalError))
            ),
            "{fault}"
        );
        assert_eq!(
            synchronized_after(&host, journal, kind).len(),
            RANKS as usize,
            "{fault}"
        );
        assert!(!instance.is_poisoned(), "{fault}");
        crate::instance::with_encoder(instance.handle(), |encoder| encoder.arena_capture = None);
        run(&host, &mut instance, "text", step(&[(10, 0, 0, &PROMPT)]));
        instance.commit().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// A capture of a step whose rows are all resident reads the arena and
/// synchronizes outside any submission: when that stream does not drain,
/// the instance is poisoned and keeps its resources.
#[test]
fn a_capture_that_cannot_drain_poisons_the_instance() {
    let root = std::env::temp_dir().join(format!("qwen-capture-fault-{}", std::process::id()));
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    crate::instance::with_encoder(instance.handle(), |encoder| {
        encoder.capture = Some(root.clone())
    });
    let mut resident = media();
    resident.encodes.clear();
    resident.payload.clear();
    host.refuse_synchronize.set(true);
    let refused = execute(
        &mut instance,
        step(&[(10, 0, 0, &media_prompt())]).with_media(resident),
    );
    host.refuse_synchronize.set(false);
    let _ = std::fs::remove_dir_all(&root);
    assert!(matches!(
        refused,
        Err(BundleAbiError::Status(BundleStatusV1::InternalError))
    ));
    let released = host.released.get();
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
}

/// `CHOPINFER_ENCODER_SERIAL=1` runs each rank's own program, one rank after
/// another: every stream issues the same lines as under the relocated
/// program, buffer handles renamed per stream.
#[test]
fn serial_ranks_issue_what_the_relocated_program_issues() {
    let mut journals = Vec::new();
    for mode in [Mode::Relocated, Mode::Serial] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
        crate::instance::with_encoder(instance.handle(), |encoder| encoder.mode = mode);
        let from = host.journal.borrow().len();
        run(
            &host,
            &mut instance,
            "media",
            step(&[(10, 0, 0, &media_prompt())]).with_media(media()),
        );
        instance.commit().unwrap();
        journals.push(host.journal.borrow()[from..].to_vec());
    }
    // Rank by rank, not interleaved.
    let stream = |line: &String| line.split(' ').nth(1).map(str::to_owned);
    let issued = |journal: &[String]| -> Vec<String> {
        let mut streams: Vec<String> = Vec::new();
        for line in journal.iter().filter(|line| line.starts_with("launch")) {
            let at = stream(line).unwrap();
            if streams.last() != Some(&at) {
                streams.push(at);
            }
        }
        streams
    };
    assert!(issued(&journals[0]).len() > issued(&journals[1]).len());
    let per_stream = |journal: &[String]| {
        let mut streams = BTreeMap::<String, Vec<String>>::new();
        for line in journal {
            if let Some(at) = stream(line).filter(|_| line.starts_with("launch")) {
                streams.entry(at).or_default().push(line.clone());
            }
        }
        streams
    };
    assert_eq!(per_stream(&journals[0]), per_stream(&journals[1]));
}

/// The run `encoder-spec` derives for each image against what every rank
/// executes: its live row counts, its sequence bounds and the merged rows
/// it writes to its arena page.
#[test]
fn image_records_run_their_derived_geometry() {
    use encoder_spec::geometry::Media;

    let vision = encoder_spec::parse(
        include_str!("../../../contract/encoders/qwen38-vision.json"),
        &encoder_spec::Environment::new("qwen38"),
    )
    .unwrap();
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let prompt = media_prompt();
    let (uploads, copies) = (host.uploads.borrow().len(), host.copies.borrow().len());
    run(
        &host,
        &mut instance,
        "media",
        step(&[(10, 0, 0, &prompt)]).with_media(media()),
    );
    let copies = host.copies.borrow()[copies..]
        .iter()
        .map(|(source, destination)| (source.offset, destination.offset, destination.length))
        .collect::<Vec<_>>();
    for record in media().encodes {
        let [t, h, w, _] = record.geometry.map(u64::from);
        let geometry = vision
            .geometry("image.patch-grid.v1", Media::Grid([t, h, w]))
            .unwrap();
        let [derived] = geometry.runs.as_slice() else {
            panic!("one image is one run");
        };
        let mut counts: Vec<u32> = vec![derived.steps[0].rows[0] as u32];
        for step in &derived.steps {
            if counts.last() != Some(&(step.rows[1] as u32)) {
                counts.push(step.rows[1] as u32);
            }
        }
        assert_eq!(
            uploads_of(&host, uploads, &bytes_of(&counts, u32::to_le_bytes)),
            RANKS as usize
        );
        let bounds: Vec<i32> = derived.steps[2]
            .sequences
            .as_ref()
            .unwrap()
            .iter()
            .map(|&bound| bound as i32)
            .collect();
        assert_eq!(
            uploads_of(&host, uploads, &bytes_of(&bounds, i32::to_le_bytes)),
            RANKS as usize
        );
        let rows = derived.output.end - derived.output.start;
        let page = record.pages[0] * 64 * ROW_BYTES;
        let written = copies
            .iter()
            .filter(|&&copy| copy == (derived.output.start * ROW_BYTES, page, rows * ROW_BYTES))
            .count();
        assert_eq!(written, RANKS as usize);
    }
    instance.commit().unwrap();
}

#[test]
fn text_steps_launch_the_same_kernels_with_the_encoder() {
    let prompt = media_prompt();
    let mut launched_by = Vec::new();
    for roles in [Vec::new(), encoder_roles(RANKS)] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_with(&host, &roles).unwrap();
        run(
            &host,
            &mut instance,
            "prefill",
            step(&[(10, 0, 0, &prompt)]),
        );
        instance.commit().unwrap();
        run(&host, &mut instance, "decode", step(&[(10, 0, 32, &[5])]));
        instance.commit().unwrap();
        launched_by.push(launched(&host, 0));
    }
    assert_eq!(launched_by[0], launched_by[1]);
    assert_eq!(count(&launched_by[1], full(OVERRIDE)), 0);

    // An encode-only step (an image whose rows a later chunk injects) runs
    // the tower and leaves the prefill without an override.
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let mut section = media();
    section.injects.clear();
    run(
        &host,
        &mut instance,
        "encode",
        step(&[(10, 0, 0, &prompt[..2])]).with_media(section),
    );
    let names = launched(&host, 0);
    assert_eq!(count(&names, full(OVERRIDE)), 0);
    assert_eq!(count(&names, full(PATCH)), 2 * RANKS as usize);
}

/// A tapped prefill writes the merged rows of the step's images first, on
/// the encoder row axis, indexed for the request whose item injects them.
#[test]
fn tapped_prefills_write_merged_rows_on_the_encoder_axis() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    let root = std::env::temp_dir().join(format!("qwen-encoder-taps-{}", std::process::id()));
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let filter = TapFilter {
        sites: Some(vec!["enc.vision.out".into(), "embed.out".into()]),
        ..TapFilter::default()
    };
    let taps = Taps::new(root.clone(), filter).with_model("qwen38");
    crate::instance::set_taps(instance.handle(), taps);
    let prompt = media_prompt();
    execute(
        &mut instance,
        step(&[(10, 0, 0, &prompt)]).with_media(media()),
    )
    .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let dir = root.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let index = std::fs::read_to_string(dir.join("index.jsonl")).unwrap();
    let names = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names[0],
        serde_json::json!({"name": "enc.vision.out", "rank": 0, "first_row": 0, "rows": 28,
            "cols": 2560, "dtype": "bf16", "axis": "encoder"})
    );
    assert_eq!(names[1]["name"], "embed.out");
    assert_eq!(names.len(), 2);
    assert_eq!(
        read("items.json")["encoder"]["enc.vision.out"],
        serde_json::json!([{"request": 10, "start": 0, "kind": "image", "item": 0, "index": 0,
            "rows": 24},
            {"request": 10, "start": 24, "kind": "image", "item": 1, "index": 0, "rows": 4}])
    );
    assert_eq!(
        std::fs::metadata(dir.join("enc.vision.out.r0.bin"))
            .unwrap()
            .len(),
        28 * ROW_BYTES
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// The four grids of the encoder reference, an image the size of the
/// first, then the first again from its resident pages, each split inside
/// a page across two prefill steps. Every step's source rows name the arena
/// rows of its image's pages, in prompt order, on every rank, whether the
/// step encoded them or an earlier step left them resident; steps without
/// an encode run no tower and write no arena row.
#[test]
fn injects_read_the_pages_of_encoded_and_resident_images() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    let tower = tower_names();
    let mut previous = 0;
    let mut first_pages = Vec::new();
    for (image, (height, width)) in [(16, 16), (20, 32), (14, 22), (8, 40), (16, 16), (16, 16)]
        .into_iter()
        .enumerate()
    {
        let rows = height * width / 4;
        let pages: Vec<u64> = if image == 5 {
            first_pages.clone()
        } else {
            (0..u64::from(rows).div_ceil(64))
                .map(|page| 5 + 8 * image as u64 + 3 * page)
                .collect()
        };
        if image == 0 {
            first_pages = pages.clone();
        }
        let arena_row = |row: u32| pages[row as usize / 64] * 64 + u64::from(row % 64);
        for (part, (first, chunk)) in [(0, 17), (17, rows - 17)].into_iter().enumerate() {
            let name = format!("image {image} part {part}");
            let encodes = part == 0 && image != 5;
            let mut section = BatchMediaSectionV3::default();
            if encodes {
                let mut encode = record([1, height, width, 0], pages[0], 0);
                encode.pages = pages.clone();
                section.payload = vec![image as u8; encode.payload_bytes as usize];
                section.encodes.push(encode);
            }
            let mut row = first;
            while row < first + chunk {
                let run = (64 - row % 64).min(first + chunk - row);
                section.injects.push(BatchMediaInjectV3 {
                    item_index: 0,
                    item_row: row - first,
                    rows: run,
                    arena_row: arena_row(row),
                });
                row += run;
            }
            let (launches, uploads, copies) = (
                host.launches.borrow().len(),
                host.uploads.borrow().len(),
                host.copies.borrow().len(),
            );
            run(
                &host,
                &mut instance,
                &name,
                step(&[(10, 0, previous, &vec![IMAGE_PAD; chunk as usize])]).with_media(section),
            );
            instance.commit().unwrap();
            previous += u64::from(chunk);

            for (stream, names) in by_stream(&host, launches) {
                let towers = names.iter().filter(|name| tower.contains(name)).count();
                if encodes {
                    assert_eq!(names[..tower.len()], tower[..], "{name} stream {stream}");
                    assert_eq!(towers, tower.len(), "{name} stream {stream}");
                } else {
                    assert_eq!(towers, 0, "{name} stream {stream}");
                }
                assert_eq!(count(&names, full(OVERRIDE)), 1, "{name} stream {stream}");
            }

            // Each rank's override reads the arena and the source rows that
            // rank's upload wrote.
            let arguments = host.launch_arguments.borrow();
            let overrides = host.launches.borrow()[launches..]
                .iter()
                .zip(&arguments[launches..])
                .filter(|((_, symbol, _), _)| symbol == full(OVERRIDE))
                .map(|((stream, _, _), arguments)| {
                    let span = |index: usize| match arguments[index] {
                        KernelArgumentV1::DeviceBuffer(span) => span,
                        ref other => panic!("{name} override argument {index} is {other:?}"),
                    };
                    (*stream, span(1), span(2))
                })
                .collect::<Vec<_>>();
            assert_eq!(
                overrides
                    .iter()
                    .map(|&(stream, ..)| stream)
                    .collect::<BTreeSet<_>>()
                    .len(),
                RANKS as usize,
                "{name}"
            );
            let sources = (first..first + chunk)
                .map(|row| arena_row(row) as i64)
                .collect::<Vec<_>>();
            let sources = bytes_of(&sources, i64::to_le_bytes);
            assert_eq!(
                uploads_of(&host, uploads, &sources),
                RANKS as usize,
                "{name}"
            );
            let uploaded = host.uploads.borrow()[uploads..]
                .iter()
                .filter(|(_, bytes)| *bytes == sources)
                .map(|(span, _)| (span.buffer.slot(), span.offset))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                uploaded,
                overrides
                    .iter()
                    .map(|(_, _, span)| (span.buffer.slot(), span.offset))
                    .collect(),
                "{name}"
            );

            // Only the encoding step writes the image's pages.
            let arenas = overrides
                .iter()
                .map(|(_, arena, _)| (arena.buffer.slot(), arena.offset))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(arenas.len(), RANKS as usize, "{name}");
            let written = host.copies.borrow()[copies..]
                .iter()
                .filter_map(|(_, destination)| {
                    let base = arenas.get(&destination.buffer.slot())?;
                    Some((
                        destination.buffer.slot(),
                        destination.offset - base,
                        destination.length,
                    ))
                })
                .collect::<BTreeSet<_>>();
            let expected = if encodes {
                arenas
                    .keys()
                    .flat_map(|&slot| {
                        pages.iter().enumerate().map(move |(index, &page)| {
                            let rows = (u64::from(rows) - 64 * index as u64).min(64);
                            (slot, page * 64 * ROW_BYTES, rows * ROW_BYTES)
                        })
                    })
                    .collect()
            } else {
                BTreeSet::new()
            };
            assert_eq!(written, expected, "{name}");
        }
    }
    instance.destroy().unwrap();
}

/// Grid positions of `media_prompt`: text, the 4 x 6 merged image from
/// position 2, text, the 2 x 2 image, text (HF `get_rope_index`).
fn media_positions() -> (Vec<[i32; 3]>, i32) {
    let mut rows = vec![[0, 0, 0], [1, 1, 1]];
    let grid = |rows: &mut Vec<[i32; 3]>, start: i32, h: i32, w: i32| {
        for r in 0..h {
            for c in 0..w {
                rows.push([start, start + r, start + c]);
            }
        }
    };
    grid(&mut rows, 2, 4, 6);
    rows.push([8, 8, 8]);
    grid(&mut rows, 9, 2, 2);
    rows.push([11, 11, 11]);
    // Row 29 sits at position 11, so later rows add -18.
    (rows, 11 - 29)
}

#[test]
fn an_image_prompt_serves_with_the_mtp_drafter() {
    use super::speculation::{SpeculativeSink, batch, drafted, execute as speculate, kept, plain};
    let host = Rc::new(FakeHost::default());
    host.drafter.set(true);
    host.speculation.set((3, MAX_VERIFY_TOKENS));
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with(true, &encoder_roles(RANKS)),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    host.model(instance.handle());
    // Sixteen more text rows keep the fresh prefill above the 9 to 39 rows
    // whose drafter catch-up graph main currently refuses ("graph bucket",
    // c4ee3452), which is unrelated to media.
    let mut prompt = media_prompt();
    let (mut positions, _) = media_positions();
    for extra in 0..16 {
        prompt.push(5 + extra);
        positions.push([12 + extra as i32; 3]);
    }
    let delta = positions.last().unwrap()[0] - (prompt.len() as i32 - 1);
    let (item, append, slot) = plain(10, 0, 0, &prompt);
    let prefill = batch(vec![(item.with_position_axes(positions), append, slot)]);
    let launches = host.launches.borrow().len();
    speculate(&host, &mut instance, &prefill.with_media(media()));
    instance.commit().unwrap();
    let names = launched(&host, launches);
    assert_eq!(count(&names, full(OVERRIDE)), RANKS as usize);
    // A drafted verify step continues from the prompt with its rows at the
    // position delta; the drafter drafts and the kept rows commit.
    let rows = prompt.len() as i32;
    let (item, append, slot) = drafted(10, 0, prompt.len() as u64, 7, 3);
    let axes = (0..4).map(|row| [rows + row + delta; 3]).collect();
    let sink: SpeculativeSink = speculate(
        &host,
        &mut instance,
        &batch(vec![(item.with_position_axes(axes), append, slot)]),
    );
    assert_eq!(sink.drafts.len(), 1);
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

#[test]
fn cancelled_continuation_retries_the_same_encoder_occurrence() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    let root = std::env::temp_dir().join(format!("qwen-taps-cancel-{}", std::process::id()));
    let host = Rc::new(FakeHost::default());
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    crate::instance::set_taps(
        instance.handle(),
        Taps::new(
            root.clone(),
            TapFilter {
                sites: Some(vec!["enc.vision.out".into()]),
                ..TapFilter::default()
            },
        ),
    );
    host.backing.set(true);
    let mut first = media();
    first.encodes.truncate(1);
    first.injects.truncate(1);
    first.injects[0].item_row = 0;
    first.injects[0].rows = 12;
    execute(
        &mut instance,
        step(&[(10, 0, 0, &[IMAGE_PAD; 12])]).with_media(first),
    )
    .unwrap();
    instance.commit().unwrap();
    let pages = snapshot_pages(&host, 100);
    instance.capture_prefix_state(10, 12, &pages).unwrap();
    let continuation = || {
        step(&[(10, 0, 12, &[IMAGE_PAD; 12])]).with_media(BatchMediaSectionV3 {
            injects: vec![BatchMediaInjectV3 {
                item_index: 0,
                item_row: 0,
                rows: 12,
                arena_row: 5 * 64 + 12,
            }],
            ..BatchMediaSectionV3::default()
        })
    };
    execute(&mut instance, continuation()).unwrap();
    instance.cancel().unwrap();
    // Cancellation invalidates the linear pools; restore the committed prefix.
    instance
        .restore_prefix_state_in_slot(10, &[IMAGE_PAD; 12], &pages, 0)
        .unwrap();
    execute(&mut instance, continuation()).unwrap();
    instance.commit().unwrap();
    let items: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("step0002/items.json")).unwrap()).unwrap();
    assert_eq!(
        items["encoder"]["enc.vision.out"],
        serde_json::json!([{"request": 10, "start": 0, "kind": "image", "item": 0,
            "index": 12, "rows": 12}])
    );
    assert!(
        !root.join("step0001").exists(),
        "cancelled rows remain published"
    );
    instance.destroy().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn retired_prompts_forget_labels_but_keep_resident_encoder_rows() {
    use runtime_bundle_sdk::taps::{TapFilter, Taps};
    for text_only in [false, true] {
        let root = std::env::temp_dir().join(format!(
            "qwen-taps-retire-{text_only}-{}",
            std::process::id()
        ));
        let host = Rc::new(FakeHost::default());
        let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
        crate::instance::set_taps(
            instance.handle(),
            Taps::new(
                root.clone(),
                TapFilter {
                    sites: Some(vec!["enc.vision.out".into()]),
                    ..TapFilter::default()
                },
            ),
        );
        host.backing.set(true);
        let prompt = if text_only {
            vec![11; 32]
        } else {
            media_prompt()
        };
        let first = if text_only {
            BatchMediaSectionV3::default()
        } else {
            media()
        };
        execute(
            &mut instance,
            step(&[(10, 0, 0, &prompt)]).with_media(first),
        )
        .unwrap();
        instance.commit().unwrap();
        let previous = prompt.len() as u64;
        let pages = snapshot_pages(&host, 100);
        instance.capture_prefix_state(10, previous, &pages).unwrap();
        execute(&mut instance, step(&[(10, 0, previous, &[11, 12])])).unwrap();
        assert!(
            instance.retire_request(10).is_err(),
            "cannot retire an open request"
        );
        instance.cancel().unwrap();
        instance.retire_request(10).unwrap();
        instance.retire_request(10).unwrap();
        instance
            .restore_prefix_state_in_slot(10, &prompt, &pages, 0)
            .unwrap();
        let mut resident = media();
        if !text_only {
            resident.encodes.clear();
            resident.payload.clear();
        }
        execute(
            &mut instance,
            step(&[(10, 0, previous, &media_prompt())]).with_media(resident.clone()),
        )
        .unwrap();
        instance.commit().unwrap();
        let read = |step: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(root.join(step).join("items.json")).unwrap())
                .unwrap()
        };
        assert_eq!(
            read("step0002")["encoder"]["enc.vision.out"][0]["request"],
            serde_json::Value::Null
        );
        instance.retire_request(10).unwrap();
        resident.encodes.clear();
        resident.payload.clear();
        execute(
            &mut instance,
            step(&[(11, 0, 0, &media_prompt())]).with_media(resident),
        )
        .unwrap();
        instance.commit().unwrap();
        assert_eq!(
            read("step0003")["encoder"]["enc.vision.out"][0]["request"],
            11
        );
        assert_eq!(read("step0003")["encoder"]["enc.vision.out"][0]["item"], 0);
        instance.retire_request(11).unwrap();
        instance.destroy().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn overlapped_media_inject_uses_half_local_source_rows() {
    let host = Rc::new(FakeHost::default());
    let enabled = crate::instance::FORCE_OVERLAP.with(|value| value.replace(Some(true)));
    let minimum = crate::instance::FORCE_OVERLAP_MIN.with(|value| value.replace(Some(128)));
    let mut instance = create_with(&host, &encoder_roles(RANKS)).unwrap();
    crate::instance::FORCE_OVERLAP.with(|value| value.set(enabled));
    crate::instance::FORCE_OVERLAP_MIN.with(|value| value.set(minimum));
    let mut section = media();
    // Keep the existing encoded image but put its injected rows in half 1.
    for inject in &mut section.injects {
        inject.item_row += 160;
    }
    let before = host.launches.borrow().len();
    run(
        &host,
        &mut instance,
        "overlapped media",
        step(&[(10, 0, 0, &vec![IMAGE_PAD; 257])]).with_media(section),
    );
    let launches = host.launches.borrow();
    let arguments = host.launch_arguments.borrow();
    let sources = launches[before..]
        .iter()
        .zip(&arguments[before..])
        .filter(|((_, symbol, _), _)| symbol == full(OVERRIDE))
        .map(|(_, args)| match args[2] {
            KernelArgumentV1::DeviceBuffer(source) => source,
            _ => panic!("source-row pointer"),
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 2 * RANKS as usize);
    for rank in 0..RANKS as usize {
        assert_eq!(sources[rank].buffer, sources[rank + RANKS as usize].buffer);
        assert_eq!(
            sources[rank + RANKS as usize].offset - sources[rank].offset,
            128 * 8
        );
        let uploads = host.uploads.borrow();
        let (span, bytes) = uploads
            .iter()
            .find(|(span, _)| {
                span.buffer == sources[rank].buffer
                    && span.offset <= sources[rank].offset
                    && span.offset + span.length >= sources[rank].offset + 257 * 8
            })
            .unwrap();
        let start = (sources[rank].offset - span.offset) as usize;
        let values = bytes[start..start + 257 * 8]
            .chunks_exact(8)
            .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values[..160].iter().all(|&row| row == -1));
        assert!(values[160..].iter().any(|&row| row >= 0));
    }
    drop(arguments);
    drop(launches);
    instance.commit().unwrap();
    instance.destroy().unwrap();
}
