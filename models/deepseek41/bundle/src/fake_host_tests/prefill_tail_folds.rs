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
/// table admits, as their fold symbols spell them, each with its row ranges
/// (the `dense_prefill.compact_range.prefill` rows).
const COMPACT_RANGES: [(&str, &[std::ops::RangeInclusive<u64>]); 4] = [
    ("_n512_k5120_g1_", &[1153..=3071, 3073..=3199]),
    ("_n576_k5120_g1_", &[1153..=2943, 2945..=3200]),
    ("_n1280_k5120_g1_", &[1153..=1279, 1281..=2368, 2401..=3200]),
    ("_n1024_k4096_g2_", &[1153..=3200]),
];

/// The launches of a prompt of `rows` rows and of one decode step after
/// it, with the compact range option `enabled`.
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

/// `lines` with each call of the `shapes` projections over all `rows` (its
/// admit launch of `rows` CTAs, then its tiles and any tail folds on the
/// same stream) replaced by one marker, and each replaced call's shape and
/// symbols.
fn projection_calls(
    lines: &[String],
    rows: u64,
    shapes: &[&'static str],
) -> (Vec<String>, Vec<(&'static str, Vec<String>)>) {
    let mut out = Vec::new();
    let mut calls = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        let (stream, symbol) = stream_symbol(&lines[at]);
        let shape = shapes.iter().find(|s| symbol.contains(*s));
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
        calls.push((*shape, call));
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

/// Fails at the first launch where `option` and `default` differ.
fn same_launches(what: &str, option: &[String], default: &[String]) {
    if let Some(at) =
        (0..option.len().max(default.len())).find(|&at| option.get(at) != default.get(at))
    {
        panic!(
            "{what}: launch {at} differs\noption: {:?}\ndefault: {:?}",
            option.get(at),
            default.get(at)
        );
    }
}

/// At a count inside a projection's ranges each call of that projection
/// over the step's rows runs the compact entry, an admit and a tiles launch,
/// in place of the full 128-row tiles and their exact-row tail folds. The
/// counts hit each range end and each count a range leaves out. The
/// `prefill_compact_dense_rows` option, adopted, changes no prefill or
/// decode launch.
#[test]
fn compact_ranges_serve_the_admitted_projection_calls() {
    let mut ranged = 0;
    for rows in [
        1152, 1153, 1279, 1280, 1281, 1500, 2048, 2368, 2369, 2400, 2401, 2943, 2944, 2945, 3071,
        3072, 3073, 3199, 3200, 3201,
    ] {
        let [prefill, decode] = compact_range_launches(rows, false);
        let [option_prefill, option_decode] = compact_range_launches(rows, true);
        same_launches(
            &format!("prefill of {rows} rows"),
            &renumbered(&option_prefill),
            &renumbered(&prefill),
        );
        same_launches(
            &format!("decode after {rows} rows"),
            &renumbered(&option_decode),
            &renumbered(&decode),
        );
        let shapes: Vec<&str> = COMPACT_RANGES
            .iter()
            .filter(|(_, ranges)| rows != 2048 && ranges.iter().any(|r| r.contains(&rows)))
            .map(|(shape, _)| *shape)
            .collect();
        if shapes.is_empty() {
            continue;
        }
        let (_, compact) = projection_calls(&prefill, rows, &shapes);
        let replaced: std::collections::BTreeSet<&str> =
            compact.iter().map(|(shape, _)| *shape).collect();
        assert_eq!(replaced, shapes.iter().copied().collect(), "rows={rows}");
        for (_, call) in &compact {
            assert!(call.len() % 2 == 0, "rows={rows}: {call:?}");
            assert!(
                call.iter()
                    .all(|symbol| symbol.contains("_compact_m64_n128_s2_v1_")),
                "rows={rows}: {call:?}"
            );
        }
        ranged += 1;
    }
    assert_eq!(ranged, 17);
}
