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
/// with upper skip and step logs, and the `dspark_main_short_rows`
/// candidate on when `short`.
fn served(items: u32, short: bool) -> (Rc<Host>, BundleInstanceV1) {
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
        ("dspark_main_short_rows", short.into()),
    ] {
        record = crate::config::tests::with_option(record, name, value);
    }
    let services = HostServicesTableV1::new(Rc::new(super::dspark::host::MappedHost::new(
        Rc::clone(&host),
    )));
    let instance = api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .create(services, &record)
        .unwrap_or_else(|error| panic!("create: {error:?} {:?}", host.logs.borrow()));
    (host, instance)
}

/// Commit passes of 129 to 511 inner rows run the compact pair, and with
/// the `dspark_main_short_rows` candidate on so do those of 128.
#[test]
fn served_main_projection_uses_every_qualified_inner_boundary() {
    let cases = [false, true].into_iter().flat_map(|short| {
        BOUNDARIES
            .iter()
            .map(move |&(rows, lengths)| (short, rows, lengths))
    });
    for (short, inner_rows, lengths) in cases {
        let (host, mut instance) = served(lengths.len() as u32, short);
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
        let first_compact = if short { 65 } else { 129 };
        let selected = (first_compact..=511).contains(&inner_rows);
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
    fn kinds(&mut self) -> [&mut Vec<(u32, u32)>; 5] {
        [
            &mut self.admissions,
            &mut self.computes,
            &mut self.generic,
            &mut self.small_admissions,
            &mut self.small_computes,
        ]
    }
}

/// The launches of `repeats` passes of each of `chunks` rows: the small
/// path at 64 rows or fewer, and at 65 to 128 the compact pair with the
/// `dspark_main_short_rows` candidate on (`short`), else the generic fold.
fn expected(chunks: &[u32], repeats: usize, short: bool) -> MainLaunches {
    let mut launches = MainLaunches::default();
    for &rows in chunks {
        assert!(rows <= 128);
        if rows <= 64 {
            launches.small_admissions.push((rows, 1));
            launches.small_computes.push((80, rows.div_ceil(16)));
        } else if short {
            launches.admissions.push((rows, 1));
            launches.computes.push((40, rows.div_ceil(32)));
        } else {
            launches.generic.push((160, rows.div_ceil(16)));
        }
    }
    for kind in launches.kinds() {
        *kind = kind.repeat(repeats);
        kind.sort_unstable();
    }
    launches
}

/// Short cold prompts and a restored window reach the main projection
/// through the bulk catch-up of their next step, one pass per stage on
/// each rank, which the route sweep never runs: it retires each prefill
/// case and starts decode and verify from five-token prefixes. Each pass of
/// 64 rows or fewer runs the small path, and each of 65 to 128 rows the
/// stage-3 compact pair with the `dspark_main_short_rows` candidate on, or
/// the generic fold with it off.
#[test]
fn served_bulk_catch_up_projects_short_rows_through_the_compact_pair() {
    use bundle_runtime::fake_host::conformance::Phase;
    let token = |request: u64, position: u64| ((position * 7 + request) % 1000) as u32;
    // (request, restored prefix, prompt rows, bulk chunks): cold prompts
    // project their rows at their first drafting step; a 93,184-token
    // prefix restores its 128-row window, which the 134-row history splits
    // into 74 and 54 rows, at the step that continues its prompt.
    let cases = [
        (1, 0, 64, &[64][..]),
        (2, 0, 65, &[65][..]),
        (3, 0, 66, &[66][..]),
        (4, 0, 127, &[127][..]),
        (5, 0, 128, &[128][..]),
        (6, 93_184, 3, &[74, 54][..]),
    ];
    for (short, (request, prefix, rows, chunks)) in [false, true]
        .into_iter()
        .flat_map(|short| cases.map(|case| (short, case)))
    {
        let (host, mut instance) = served(1, short);
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
            expected(chunks, 12, short),
            "prompt rows {rows} after {prefix}, short rows {short}"
        );
        let batched = if prefix > 0 { &[rows as u32][..] } else { &[] };
        assert_eq!(main_launches(other), expected(batched, 4, short));
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
