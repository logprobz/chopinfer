//! Check adopted tail folds' fake-host outputs and committed request state.
//! Fake kernels do not perform GEMM arithmetic; the lab probe checks bits.
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};

#[test]
fn tail_folds_preserve_prefill_outputs_and_committed_state() {
    for dspark in [false, true] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.drafter.set(dspark);
        if dspark {
            host.speculation.set(Some((5, 32)));
        }
        let mut instance = prefill_skip::create_profile(&host, 8192, true);
        let mut first_outputs = None;
        for rows in [85, 127, 129, 191, 789, 831, 853, 1557, 1599] {
            let request = encode(
                &prompt(1, 0, 0, rows, true),
                &[Boundary {
                    request_id: 1,
                    prompt_tokens: rows,
                    chunk_end: rows,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap();
            let mut sink = RecordingSink::<conformance::Limits>::default();
            instance
                .execute_prefill(&request, &mut sink)
                .unwrap_or_else(|e| {
                    panic!(
                        "dspark={dspark}, rows={rows}: {e:?}: {:?}",
                        host.logs.borrow()
                    )
                });
            let outputs = sink
                .views
                .iter()
                .map(|view| {
                    let v = view.raw();
                    (
                        v.rank,
                        v.element_type,
                        v.row_stride_elements,
                        v.row_count,
                        v.live_width,
                        v.first_item_index,
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(outputs.iter().map(|v| v.3).sum::<u32>(), 1);
            instance.commit().unwrap();
            let state = crate::instance::ledger_state(instance.handle(), 0);
            assert_eq!(state, (Some((1, rows)), false));
            if let Some(first) = &first_outputs {
                assert_eq!(&outputs, first, "dspark={dspark}, rows={rows}");
            } else {
                first_outputs = Some(outputs);
            }
            instance.retire_request(1).unwrap();
            assert_eq!(
                crate::instance::ledger_state(instance.handle(), 0),
                (None, false)
            );
            host.device.journal.borrow_mut().clear();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The four TP4 projections the `prefill_compact_dense_rows` candidate
/// table admits, as their fold symbols spell them.
const COMPACT_RANGE_SHAPES: [&str; 4] = [
    "_n512_k5120_g1_",
    "_n576_k5120_g1_",
    "_n1280_k5120_g1_",
    "_n1024_k4096_g2_",
];

/// The launches of a prompt of `rows` rows and of one decode step after
/// it, with the compact range candidate `enabled`.
fn compact_range_launches(rows: u64, enabled: bool) -> [Vec<String>; 2] {
    use model_deepseek41_contract::execution_options::PREFILL_COMPACT_DENSE_ROWS;
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance =
        create_with_options(&host, 4, &[(PREFILL_COMPACT_DENSE_ROWS, enabled.into())]);
    let since = |from: usize| -> Vec<String> {
        host.device.journal.borrow()[from..]
            .iter()
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect()
    };
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &prompt(7, 0, 0, rows, true),
        &mut Views::default(),
    );
    let prefill = since(from);
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, rows, 9)]),
        &mut Views::default(),
    );
    let decode = since(from);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    [prefill, decode]
}

/// A launch line's stream and symbol.
fn stream_symbol(line: &str) -> (&str, &str) {
    let mut words = line
        .strip_prefix("launch ")
        .expect("a launch line")
        .split(' ');
    (words.next().unwrap(), words.next().unwrap())
}

/// `lines` with each of the four projections' calls over all `rows` (its
/// admit launch of `rows` CTAs, then its tiles and any tail folds on the
/// same stream) replaced by one marker, and each replaced call's symbols.
fn projection_calls(lines: &[String], rows: u64) -> (Vec<String>, Vec<Vec<String>>) {
    let mut out = Vec::new();
    let mut calls = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        let (stream, symbol) = stream_symbol(&lines[at]);
        let shape = COMPACT_RANGE_SHAPES.iter().find(|s| symbol.contains(*s));
        let whole = lines[at].contains(&format!("grid_x: {rows},"));
        let (Some(shape), true, true) = (shape, symbol.ends_with("_admit_v1"), whole) else {
            out.push(lines[at].clone());
            at += 1;
            continue;
        };
        let mut call = Vec::new();
        while at < lines.len() {
            let (next_stream, next) = stream_symbol(&lines[at]);
            if next_stream != stream || !next.contains(shape) {
                break;
            }
            call.push(next.to_owned());
            at += 1;
        }
        out.push(format!("launch {stream} projection{shape}rows{rows} "));
        calls.push(call);
    }
    (out, calls)
}

/// `lines` with each launch stream and host handle renumbered by its first
/// use.
fn renumbered(lines: &[String]) -> Vec<String> {
    let mut streams = std::collections::BTreeMap::<String, usize>::new();
    let mut handles = std::collections::BTreeMap::<String, usize>::new();
    lines
        .iter()
        .map(|line| {
            let (stream, mut rest) = line
                .strip_prefix("launch ")
                .and_then(|rest| rest.split_once(' '))
                .expect("a launch line");
            let next = streams.len();
            let stream = *streams.entry(stream.to_owned()).or_insert(next);
            let mut out = format!("launch #{stream} ");
            while let Some(at) = rest.find("raw: ") {
                let (head, tail) = rest.split_at(at + "raw: ".len());
                let digits = tail
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(tail.len());
                let next = handles.len();
                let id = *handles.entry(tail[..digits].to_owned()).or_insert(next);
                out.push_str(head);
                out.push_str(&format!("#{id}"));
                rest = &tail[digits..];
            }
            out.push_str(rest);
            out
        })
        .collect()
}

/// Fails at the first launch where `candidate` and `default` differ.
fn same_launches(what: &str, candidate: &[String], default: &[String]) {
    if let Some(at) =
        (0..candidate.len().max(default.len())).find(|&at| candidate.get(at) != default.get(at))
    {
        panic!(
            "{what}: launch {at} differs\ncandidate: {:?}\ndefault: {:?}",
            candidate.get(at),
            default.get(at)
        );
    }
}

/// Inside the candidate's range (1409 to 3071 rows) each call of the four
/// projections over the step's rows runs the compact entry in place of the
/// full 128-row tiles and their exact-row tail folds; every other launch,
/// every count outside the range, the measured exact compact row 2048 and
/// the decode step after are unchanged.
#[test]
fn compact_range_replaces_only_the_admitted_projection_calls() {
    let mut ranged = 0;
    for rows in [1408, 1409, 1500, 2048, 2304, 3071, 3072] {
        let [base_prefill, base_decode] = compact_range_launches(rows, false);
        let [prefill, decode] = compact_range_launches(rows, true);
        same_launches(
            &format!("decode after {rows} rows"),
            &renumbered(&decode),
            &renumbered(&base_decode),
        );
        if !(1409..=3071).contains(&rows) || rows == 2048 {
            same_launches(
                &format!("prefill of {rows} rows"),
                &renumbered(&prefill),
                &renumbered(&base_prefill),
            );
            continue;
        }
        let (candidate, compact) = projection_calls(&prefill, rows);
        let (default, partition) = projection_calls(&base_prefill, rows);
        same_launches(
            &format!("prefill of {rows} rows without the projection calls"),
            &renumbered(&candidate),
            &renumbered(&default),
        );
        // The shared gate and up launch back to back on one stream, so one
        // marker can hold several calls: two launches (admit and tiles) per
        // compact call, and per default call two full-tile launches and two
        // per 64-row tail fold.
        assert!(!compact.is_empty(), "rows={rows}");
        assert_eq!(compact.len(), partition.len(), "rows={rows}");
        let folds = 2 * rows.rem_euclid(128).div_ceil(64) as usize;
        for (call, base) in compact.iter().zip(&partition) {
            assert!(call.len() % 2 == 0, "rows={rows}: {call:?}");
            assert!(
                call.iter()
                    .all(|symbol| symbol.contains("_compact_m64_n128_s2_v1_")),
                "rows={rows}: {call:?}"
            );
            let calls = call.len() / 2;
            let full = base
                .iter()
                .filter(|symbol| symbol.contains("_prefill_tma_v1_"))
                .count();
            assert_eq!(full, 2 * calls, "rows={rows}: {base:?}");
            assert_eq!(base.len(), calls * (2 + folds), "rows={rows}: {base:?}");
        }
        ranged += 1;
    }
    assert_eq!(ranged, 4);
}
