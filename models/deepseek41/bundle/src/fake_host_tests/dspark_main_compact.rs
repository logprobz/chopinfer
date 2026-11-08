use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

// Cold prompt lengths whose whole retained dispatches pack to each main width.
// Every prompt exceeds the 134-row history capacity and projects at commit.
pub(super) const BOUNDARIES: &[(u32, &[u32])] = &[
    (128, &[256]),
    (129, &[257]),
    (130, &[258]),
    (192, &[192]),
    (255, &[255]),
    (256, &[256, 256]),
    (257, &[256, 257]),
    (383, &[255, 256]),
    (384, &[192, 192]),
    (385, &[192, 193]),
    (510, &[255, 255]),
    (511, &[255, 256, 256]),
    (512, &[192, 192, 256]),
];

/// A served TP4 DSpark instance for `items` sequences, on the prefill API
/// with upper skip and step logs. With `anchors` it also takes the served
/// cuts, device results and prefill anchors APIs, so an item may keep a
/// prompt-end anchor.
fn served(items: u32, anchors: bool) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(32 * 4200);
    host.drafter.set(true);
    host.speculation.set(Some((5, 6)));
    let mut record = crate::config::tests::record(&shape, host.kv, items, false);
    for (name, value) in [
        ("prefill_chunk_tokens", serde_json::json!(8192)),
        ("prefill_upper_skip", serde_json::json!(true)),
        ("diagnostic_step_logs", true.into()),
    ] {
        record = crate::config::tests::with_option(record, name, value);
    }
    let services = HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(
        Rc::clone(&host),
    )));
    let mut api = api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap();
    if anchors {
        api = api
            .with_prefill_cuts_api(crate::instance::prefill::PREFILL_CUTS_API)
            .unwrap()
            .with_device_results_api(crate::api::BUNDLE_DEEPSEEK41_DEVICE_RESULTS_API_V1)
            .unwrap()
            .with_prefill_anchors_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_ANCHORS_API_V2)
            .unwrap();
    }
    let instance = api
        .create(services, &record)
        .unwrap_or_else(|error| panic!("create: {error:?} {:?}", host.logs.borrow()));
    (host, instance)
}

/// Commit passes of 65 to 511 inner rows run the compact pair.
#[test]
fn served_main_projection_uses_every_qualified_inner_boundary() {
    for &(inner_rows, lengths) in BOUNDARIES {
        let (host, mut instance) = served(lengths.len() as u32, false);
        let first = host.device.journal.borrow().len();
        let tokens: Vec<Vec<u32>> = lengths.iter().map(|&n| vec![7; n as usize]).collect();
        let items: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(index, values)| {
                (
                    index as u64 + 1,
                    index as u64,
                    0,
                    values.as_slice(),
                    bundle_runtime::fake_host::conformance::Phase::Prefill,
                )
            })
            .collect();
        let boundaries: Vec<_> = lengths
            .iter()
            .enumerate()
            .map(|(index, &rows)| Boundary {
                request_id: index as u64 + 1,
                prompt_tokens: rows.into(),
                chunk_end: rows.into(),
                chunk_tokens: 8192,
            })
            .collect();
        let request = encode(&super::conformance::step(&items, false), &boundaries).unwrap();
        instance
            .execute_prefill(&request, &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        let lines = host.device.journal.borrow();
        let main: Vec<_> = lines[first..]
            .iter()
            .filter(|line| {
                line.starts_with("launch ") && line.contains("_m8192_n5120_k15360_g1_f32_v2")
            })
            .collect();
        assert!(
            !main.is_empty(),
            "the served step must execute main projections"
        );
        let selected = (65..=511).contains(&inner_rows);
        assert!(
            main.iter()
                .all(|line| line.contains("_compact_") == selected),
            "{main:#?}"
        );
        assert_eq!(main.len(), if selected { 24 } else { 12 });
        drop(lines);
        let records: Vec<serde_json::Value> = host
            .logs
            .borrow()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|row| row["schema"] == "chopinfer.deepseek41.commit-shape.v1")
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["projection_passes"], 3);
        assert_eq!(records[0]["projection_rows"], 3 * inner_rows);
        for request in 1..=lengths.len() as u64 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The main projection's launches in `lines`, by kind: each line's grid
/// x (an admission's is its row count) and grid y.
#[derive(Debug, Default, PartialEq)]
struct MainLaunches {
    /// The stage-3 compact pair: the admission, then the m32 compute.
    admissions: Vec<(u32, u32)>,
    computes: Vec<(u32, u32)>,
    generic: Vec<(u32, u32)>,
    /// The exact-row small path's admission and compute.
    small_admissions: Vec<(u32, u32)>,
    small_computes: Vec<(u32, u32)>,
    /// The stream entry of a batched projection of deferred rows.
    streams: Vec<(u32, u32)>,
}

fn main_launches(lines: &[String]) -> MainLaunches {
    let grid = |line: &str, axis: &str| -> u32 {
        let value = line.split(&format!("{axis}: ")).nth(1).unwrap();
        value[..value.find(',').unwrap()].parse().unwrap()
    };
    let mut launches = MainLaunches::default();
    for line in lines.iter().filter(|line| line.starts_with("launch ")) {
        let symbol = line.split_whitespace().nth(2).unwrap();
        let kind = if !symbol.contains("_n5120_k15360_") {
            continue;
        } else if symbol.ends_with("_compact_m64_n128_s3_v1_admit_v1") {
            &mut launches.admissions
        } else if symbol.ends_with("_compact_m32_n128_s3_v1_packed_v1") {
            &mut launches.computes
        } else if symbol.contains("_m8192_n5120_k15360_g1_f32_v2") {
            &mut launches.generic
        } else if symbol.starts_with("chop_deepseek41_dense_decode_") {
            if symbol.ends_with("_admit_v1") {
                &mut launches.small_admissions
            } else {
                &mut launches.small_computes
            }
        } else if symbol.starts_with("chop_deepseek41_mxfp8_stream_") {
            &mut launches.streams
        } else {
            panic!("unexpected main projection launch {line}");
        };
        kind.push((grid(line, "grid_x"), grid(line, "grid_y")));
    }
    for kind in launches.kinds() {
        kind.sort_unstable();
    }
    launches
}

impl MainLaunches {
    fn kinds(&mut self) -> [&mut Vec<(u32, u32)>; 6] {
        [
            &mut self.admissions,
            &mut self.computes,
            &mut self.generic,
            &mut self.small_admissions,
            &mut self.small_computes,
            &mut self.streams,
        ]
    }
}

/// The launches of `repeats` passes of each of `chunks` rows: the small
/// path at 64 rows or fewer, and at 65 to 128 the compact pair.
fn expected(chunks: &[u32], repeats: usize) -> MainLaunches {
    let mut launches = MainLaunches::default();
    for &rows in chunks {
        assert!(rows <= 128);
        if rows <= 64 {
            launches.small_admissions.push((rows, 1));
            launches.small_computes.push((80, rows.div_ceil(16)));
        } else {
            launches.admissions.push((rows, 1));
            launches.computes.push((40, rows.div_ceil(32)));
        }
    }
    for kind in launches.kinds() {
        *kind = kind.repeat(repeats);
        kind.sort_unstable();
    }
    launches
}

/// A restored window and a short prompt committed whole, with no prefix
/// capture after it, reach the main projection through the bulk catch-up
/// of their next step, one pass per stage on each rank, which the route
/// sweep never runs: it retires each prefill case and starts decode and
/// verify from five-token prefixes. Each pass of 64 rows or fewer runs the
/// small path, and each of 65 to 128 rows the stage-3 compact pair.
/// Served with the prefix cache, the restored window catches up here as
/// it does; a cold prompt keeps an anchor or catches up at its capture
/// instead (`served_cold_prompts_catch_up_at_commit_and_capture`), so the
/// cold cases here stand for a prompt served without one.
#[test]
fn bulk_catch_up_projects_short_commits_and_restored_windows() {
    use bundle_runtime::fake_host::conformance::Phase;
    let token = |request: u64, position: u64| ((position * 7 + request) % 1000) as u32;
    // (request, restored prefix, prompt rows, bulk chunks): cold prompts,
    // each committed as one item, project their rows at their next step,
    // here their first drafting step; a 93,184-token prefix restores its
    // 128-row window, which the 134-row history splits into 74 and 54 rows,
    // at the step that continues its prompt.
    let cases = [
        (1, 0, 64, &[64][..]),
        (2, 0, 65, &[65][..]),
        (3, 0, 66, &[66][..]),
        (4, 0, 127, &[127][..]),
        (5, 0, 128, &[128][..]),
        (6, 93_184, 3, &[74, 54][..]),
    ];
    for (request, prefix, rows, chunks) in cases {
        let (host, mut instance) = served(1, false);
        if prefix > 0 {
            let tokens: Vec<u32> = (0..prefix).map(|at| token(request, at)).collect();
            crate::instance::execute::plan_timing::restore_without_snapshot(
                instance.handle(),
                request,
                0,
                &tokens,
            )
            .unwrap();
        }
        let end = prefix + rows;
        let prompt: Vec<u32> = (prefix..end).map(|at| token(request, at)).collect();
        let boundary = Boundary {
            request_id: request,
            prompt_tokens: end,
            chunk_end: end,
            chunk_tokens: 8192,
        };
        let prefill = encode(
            &super::conformance::step(&[(request, 0, prefix, &prompt, Phase::Prefill)], false),
            &[boundary],
        )
        .unwrap();
        let begin = host.device.journal.borrow().len();
        instance
            .execute_prefill(&prefill, &mut Sink::default())
            .unwrap_or_else(|error| panic!("prefill: {error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        let verify = encode(
            &super::conformance::step(&[(request, 0, end, &[7; 6], Phase::Verify)], true),
            &[],
        )
        .unwrap();
        let middle = host.device.journal.borrow().len();
        instance
            .execute_prefill(&verify, &mut Sink::default())
            .unwrap_or_else(|error| panic!("verify: {error:?} {:?}", host.logs.borrow()));
        instance
            .commit_accepted(&[BundleKeptRowsV1 {
                struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                reserved: 0,
                request_id: request,
                kept_rows: 1,
                reserved_tail: 0,
            }])
            .unwrap();
        // A restored window catches up in the step that continues its
        // prompt, a cold prompt in its first drafting step; the other
        // step projects no main rows.
        let lines = host.device.journal.borrow();
        let (bulk, other) = if prefix > 0 {
            (&lines[begin..middle], &lines[middle..])
        } else {
            (&lines[middle..], &lines[begin..middle])
        };
        // The bulk passes run once per stage on each of the four ranks; a
        // restored request's continued prompt rows project once per rank
        // in one batched launch at its drafting step.
        assert_eq!(
            main_launches(bulk),
            expected(chunks, 12),
            "prompt rows {rows} after {prefix}"
        );
        let batched = if prefix > 0 { &[rows as u32][..] } else { &[] };
        assert_eq!(main_launches(other), expected(batched, 4));
        drop(lines);
        let passes: Vec<_> = host
            .logs
            .borrow()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|row| row["schema"] == "chopinfer.deepseek41.step-shape.v1")
            .filter_map(|row| row["bulk_passes"].as_u64())
            .collect();
        assert_eq!(passes, if prefix > 0 { [3, 0] } else { [0, 3] });
        instance.retire_request(request).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Served cold prompts with the prefix cache. The scheduler captures each
/// prompt's state at its last multiple of 64 tokens, and the capture
/// catches the drafter up (`capture_prefix_state`), where no step-shape
/// record or route counter sees it. A prompt of 65 to 127 tokens keeps an
/// anchor there inside its one item: the step's commit catches up the
/// 64-row head on the small path and the capture the tail, on the small
/// path or, for a tail of a row or two, the batched stream entry. A
/// 64-token or 128-token prompt ends on the granule and keeps no anchor,
/// so its capture runs all its rows: three passes of 128 rows for the
/// latter, through the compact pair. The first verify step catches up
/// nothing.
#[test]
fn served_cold_prompts_catch_up_at_commit_and_capture() {
    use bundle_runtime::fake_host::conformance::Phase;
    use runtime_bundle_sdk::prefill::encode_with_cuts;
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let small = |rows| expected(&[rows], 12);
    // (prompt rows, main launches of the prefill step, of the capture)
    let streamed = MainLaunches {
        streams: vec![(80, 1); 4],
        ..MainLaunches::default()
    };
    let cases = [
        (64, MainLaunches::default(), small(64)),
        (65, small(64), streamed),
        (97, small(64), small(33)),
        (127, small(64), small(63)),
        (128, MainLaunches::default(), expected(&[128], 12)),
    ];
    for (rows, at_commit, at_capture) in cases {
        let (host, mut instance) = served(1, true);
        let tokens: Vec<u32> = (0..rows).map(|at| (at * 7 + rows) % 1000).collect();
        let capture = u64::from(rows / 64 * 64);
        let batch = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(1, false, rows, true, tokens)
                    .with_anchor(capture < u64::from(rows)),
            ],
            vec![BatchAppendV1::new(
                0,
                u64::from(rows),
                u64::from(rows),
                (0..u64::from(rows).div_ceil(64)).collect(),
                None,
            )],
        )
        .with_state_slots(vec![0]);
        let prefill = encode_with_cuts(
            &batch.encode().unwrap(),
            &[Boundary {
                request_id: 1,
                prompt_tokens: u64::from(rows),
                chunk_end: u64::from(rows),
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        let begin = host.device.journal.borrow().len();
        instance
            .execute_prefill_cuts(&prefill, &mut Sink::default())
            .unwrap_or_else(|error| panic!("prefill: {error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        let committed = host.device.journal.borrow().len();
        let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
        instance.capture_prefix_state(1, capture, &pages).unwrap();
        let captured = host.device.journal.borrow().len();
        let verify = encode(
            &super::conformance::step(&[(1, 0, u64::from(rows), &[7; 6], Phase::Verify)], true),
            &[],
        )
        .unwrap();
        instance
            .execute_prefill(&verify, &mut Sink::default())
            .unwrap_or_else(|error| panic!("verify: {error:?} {:?}", host.logs.borrow()));
        instance
            .commit_accepted(&[BundleKeptRowsV1 {
                struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                reserved: 0,
                request_id: 1,
                kept_rows: 1,
                reserved_tail: 0,
            }])
            .unwrap();
        let lines = host.device.journal.borrow();
        let case = format!("prompt rows {rows}");
        assert_eq!(main_launches(&lines[begin..committed]), at_commit, "{case}");
        assert_eq!(
            main_launches(&lines[committed..captured]),
            at_capture,
            "{case}"
        );
        assert_eq!(
            main_launches(&lines[captured..]),
            MainLaunches::default(),
            "{case}"
        );
        drop(lines);
        let passes: Vec<_> = host
            .logs
            .borrow()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|row| row["schema"] == "chopinfer.deepseek41.step-shape.v1")
            .filter_map(|row| row["bulk_passes"].as_u64())
            .collect();
        assert_eq!(passes, [0, 0], "{case}");
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
