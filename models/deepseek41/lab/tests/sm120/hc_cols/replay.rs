//! Bitwise comparison of every candidate with the served boundary on every
//! 16- and 32-row case of real captured rows, with position, live-row,
//! concurrency, graph, special-value and mutation controls.
use super::{
    artifacts,
    cases::{self, Composite},
    corpus,
    device::{self, Candidate, Input, Kernels, Output},
};
use kernel_harness::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{finish_split, round_cols};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
type Values = BTreeMap<String, Vec<u8>>;
const POISON: u8 = 0xa5;
pub const SPECIALS: [&str; 8] = [
    "sum_nan",
    "sum_inf",
    "sum_subnormal",
    "negative_zero",
    "weight_nan",
    "gamma_inf",
    "gamma_last_nan",
    "random",
];
/// Mutations of the boundary's inputs or launch that must change its
/// outputs.
pub const MUTATIONS: [&str; 5] = [
    "epsilon_one",
    "live_short",
    "gamma_sign",
    "fn_sign",
    "sum_sign",
];
/// The candidate the input mutations run on.
pub const MUTATED: &str = "cols_r4_w32";

pub fn mismatch(a: &Values, b: &Values) -> Vec<String> {
    if a.keys().ne(b.keys()) {
        return vec!["output role set differs".into()];
    }
    b.iter()
        .filter_map(|(role, expected)| {
            let actual = a.get(role)?;
            (actual != expected).then(|| {
                format!(
                    "{role}: first differing byte {:?}, lengths {}/{}",
                    actual.iter().zip(expected).position(|(a, b)| a != b),
                    actual.len(),
                    expected.len()
                )
            })
        })
        .collect()
}
/// `actual` (of `candidate`, or the served boundary) against the served
/// `expected`. A candidate that leaves the RMS lane words to its finish must
/// leave them untouched; every other role, chain sums included, must hold
/// the served bits.
pub fn compare(actual: &Values, expected: &Values, candidate: Option<&Candidate>) -> Vec<String> {
    if candidate.is_none_or(Candidate::lanes) {
        return mismatch(actual, expected);
    }
    let strip = |v: &Values| {
        let mut v = v.clone();
        v.remove("lanes");
        v
    };
    let mut failures = mismatch(&strip(actual), &strip(expected));
    match actual.get("lanes") {
        Some(lanes) if lanes.iter().all(|&b| b == POISON) => {}
        _ => failures.push("lanes: written by a producer that leaves them to its finish".into()),
    }
    failures
}
fn same_rows(a: &Values, b: &Values, map: &[usize], rows: usize) -> bool {
    b.iter().all(|(role, data)| {
        let stride = data.len() / rows;
        a[role]
            == map
                .iter()
                .flat_map(|&r| data[r * stride..(r + 1) * stride].iter().copied())
                .collect::<Vec<_>>()
    })
}
/// The first `live` of `rows` rows of every output role.
fn live_rows(values: &Values, live: usize, rows: usize) -> Values {
    values
        .iter()
        .map(|(role, data)| (role.clone(), data[..data.len() / rows * live].to_vec()))
        .collect()
}
fn digest(values: &Values) -> String {
    kernel_build::digest(&values.values().flatten().copied().collect::<Vec<_>>())
}
fn edit(
    i: &Input,
    s: &runtime_cuda::Stream,
    role: &str,
    f: impl FnOnce(&mut Vec<u8>),
) -> LabResult<()> {
    let buffer = &i.data[role];
    let mut data = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut data, s)
        .map_err(|e| e.to_string())?;
    f(&mut data);
    buffer.copy_from_bytes(&data, s).map_err(|e| e.to_string())
}
fn alter(i: &Input, s: &runtime_cuda::Stream, pattern: &str) -> LabResult<()> {
    let (role, bits, size, index) = match pattern {
        "sum_nan" => ("sum", 0x7fc00000_u32, 4, 0),
        "sum_inf" => ("sum", 0x7f800000, 4, 0),
        "sum_subnormal" => ("sum", 1, 4, 0),
        "negative_zero" => ("sum", 0x80000000, 4, 0),
        "weight_nan" => ("fn", 0x7fc00000, 4, 0),
        "gamma_inf" => ("norm_weight", 0x7f80, 2, 0),
        "gamma_last_nan" => ("norm_weight", 0x7fc0, 2, corpus::H - 1),
        "random" => {
            let mut state = 0x51af39d2_u32;
            for (name, buffer) in &i.data {
                let bf = name == "streams" || name == "norm_weight";
                let size = if bf { 2 } else { 4 };
                let mut data = Vec::with_capacity(buffer.byte_len());
                for _ in 0..buffer.byte_len() / size {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let value = ((state >> 8) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.125;
                    if bf {
                        data.extend(kernel_core::codec::f32_to_bf16(value).to_le_bytes());
                    } else {
                        data.extend(value.to_le_bytes());
                    }
                }
                buffer
                    .copy_from_bytes(&data, s)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        _ => return Err("unknown special pattern".into()),
    };
    edit(i, s, role, |data| {
        data[index * size..(index + 1) * size].copy_from_slice(&bits.to_le_bytes()[..size]);
    })
}
/// Flip the sign of element `index` of a `size`-byte little-endian role
/// (a zero becomes 1.0), which changes that element's every use.
fn flip(
    i: &Input,
    s: &runtime_cuda::Stream,
    role: &str,
    size: usize,
    index: usize,
) -> LabResult<()> {
    edit(i, s, role, |data| {
        let at = &mut data[index * size..(index + 1) * size];
        if at.iter().all(|&b| b == 0) {
            let one: &[u8] = if size == 2 {
                &[0x80, 0x3f]
            } else {
                &[0, 0, 0x80, 0x3f]
            };
            at.copy_from_slice(one);
        } else {
            at[size - 1] ^= 0x80;
        }
    })
}
fn set_live(i: &Input, s: &Session, live: usize) -> LabResult<()> {
    i.live
        .copy_from_bytes(&(live as u32).to_ne_bytes(), &s.stream)
        .map_err(|e| e.to_string())
}
/// The prior row status under which a control can show. A sticky prior
/// zeroes the codes and saturates the scales, which hides a changed ceil
/// floor; clearing the status shows only over a sticky prior.
fn control_prior(c: &Candidate) -> u32 {
    match c {
        Candidate::Cols { finish, .. }
            if finish.mutation == finish_split::Mutation::ClearStatus =>
        {
            0x80
        }
        _ => 0,
    }
}
/// The rows of a partial batch in a `rows` bucket: its first part rows
/// live, the rest padding.
fn parts(rows: usize) -> [(&'static str, Vec<usize>); 3] {
    let half = rows / 2 + 1;
    [
        ("part_head", (0..half).collect()),
        ("part_tail", (rows - half..rows).collect()),
        ("part_strided", (0..rows).step_by(2).chain([1]).collect()),
    ]
}

struct Run<'a> {
    s: &'a Session,
    k: &'a Kernels,
    checks: Vec<Value>,
}
impl Run<'_> {
    fn check(&mut self, label: &str, name: &str, passed: bool) {
        self.checks
            .push(json!({"case":label,"name":name,"passed":passed}));
    }
    /// One eager launch from poisoned outputs.
    #[allow(clippy::too_many_arguments)]
    fn eager(
        &self,
        i: &Input,
        o: &Output,
        candidate: Option<&Candidate>,
        widen: bool,
        alias: bool,
        prior: u32,
        epsilon: Option<u32>,
    ) -> LabResult<Values> {
        device::reset(&self.s.stream, i, o, prior, alias)?;
        device::launch(
            &self.s.stream,
            self.k,
            i,
            o,
            candidate,
            widen,
            alias,
            epsilon,
        )?;
        device::read(&self.s.stream, o)
    }
    /// The candidate over `candidate_input` against the served boundary
    /// over `reference`; passes when equality is `equal`.
    #[allow(clippy::too_many_arguments)]
    fn paired(
        &mut self,
        label: &str,
        name: &str,
        reference: &Input,
        candidate_input: &Input,
        candidate: &Candidate,
        widen: bool,
        alias: bool,
        prior: u32,
        epsilon: Option<u32>,
        equal: bool,
    ) -> LabResult<()> {
        let old = Output::new(self.s, reference.rows)?;
        let new = Output::new(self.s, candidate_input.rows)?;
        let expected = self.eager(reference, &old, None, widen, alias, prior, None)?;
        let actual = self.eager(
            candidate_input,
            &new,
            Some(candidate),
            widen,
            alias,
            prior,
            epsilon,
        )?;
        self.check(
            label,
            name,
            compare(&actual, &expected, Some(candidate)).is_empty() == equal,
        );
        Ok(())
    }
}

/// Invariances and controls of one candidate, once per case kind and
/// configuration.
#[allow(clippy::too_many_arguments)]
fn controls(
    run: &mut Run,
    corpus: &[corpus::Case],
    case: &Composite,
    label: &str,
    candidate: &Candidate,
    widen: bool,
    alias: bool,
    actual: &Values,
) -> LabResult<()> {
    let s = run.s;
    let rows = case.size();
    let map = (0..rows).collect::<Vec<_>>();
    // A partial batch: its rows first and live, padded with the case's
    // other rows, in the same bucket.
    for (name, part) in parts(rows) {
        let mut padded = part.clone();
        padded.extend((0..rows).filter(|r| !part.contains(r)));
        let moved = cases::input(s, corpus, case, &padded)?;
        set_live(&moved, s, part.len())?;
        let out = Output::new(s, rows)?;
        let values = run.eager(&moved, &out, Some(candidate), widen, alias, 0, None)?;
        run.check(
            label,
            name,
            same_rows(&live_rows(&values, part.len(), rows), actual, &part, rows),
        );
        run.paired(
            label,
            &format!("{name}_served"),
            &moved,
            &moved,
            candidate,
            widen,
            alias,
            0,
            None,
            true,
        )?;
    }
    let reversed = map.iter().copied().rev().collect::<Vec<_>>();
    let moved = cases::input(s, corpus, case, &reversed)?;
    let other = Output::new(s, rows)?;
    let values = run.eager(&moved, &other, Some(candidate), widen, alias, 0, None)?;
    run.check(
        label,
        "reverse_position",
        same_rows(&values, actual, &reversed, rows),
    );
    // Two private output sets on two streams at once.
    let input = cases::input(s, corpus, case, &map)?;
    let first = Output::new(s, rows)?;
    first.assert_disjoint(&other)?;
    device::reset(&s.stream, &input, &first, 0, alias)?;
    device::reset(&s.stream, &moved, &other, 0, alias)?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let side = s.context.stream().map_err(|e| e.to_string())?;
    for _ in 0..3 {
        device::launch(
            &s.stream,
            run.k,
            &input,
            &first,
            Some(candidate),
            widen,
            alias,
            None,
        )?;
        device::launch(
            &side,
            run.k,
            &moved,
            &other,
            Some(candidate),
            widen,
            alias,
            None,
        )?;
    }
    side.synchronize().map_err(|e| e.to_string())?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let a = device::read(&s.stream, &first)?;
    let b = device::read(&s.stream, &other)?;
    // Repeated launches are idempotent only when post/comb are not aliased.
    run.check(
        label,
        "concurrent_scratch",
        alias || (a == *actual && same_rows(&b, actual, &reversed, rows)),
    );
    // Live counts the bucket serves and the edges.
    for live in BTreeSet::from([0, 1, rows / 2 + 1, rows - 1]) {
        set_live(&input, s, live)?;
        run.paired(
            label,
            &format!("live{live}"),
            &input,
            &input,
            candidate,
            widen,
            alias,
            0,
            None,
            true,
        )?;
    }
    set_live(&input, s, rows)?;
    run.paired(
        label, "sticky", &input, &input, candidate, widen, alias, 0x80, None, true,
    )?;
    for pattern in SPECIALS {
        let altered = cases::input(s, corpus, case, &map)?;
        alter(&altered, &s.stream, pattern)?;
        run.paired(
            label, pattern, &altered, &altered, candidate, widen, alias, 0, None, true,
        )?;
    }
    // Two replays of one graph.
    let out = Output::new(s, rows)?;
    let graph = s.capture(|stream| {
        device::launch(
            stream,
            run.k,
            &input,
            &out,
            Some(candidate),
            widen,
            alias,
            None,
        )
        .map(|_| ())
    })?;
    for replay in 0..2 {
        device::reset(&s.stream, &input, &out, 0, alias)?;
        graph.launch(&s.stream).map_err(|e| e.to_string())?;
        let values = device::read(&s.stream, &out)?;
        run.check(label, &format!("graph_replay{replay}"), values == *actual);
    }
    for role in actual.keys() {
        let mut changed = actual.clone();
        changed.get_mut(role).unwrap()[0] ^= 1;
        run.check(
            label,
            &format!("negative_byte_{role}"),
            !mismatch(&changed, actual).is_empty(),
        );
    }
    Ok(())
}

/// Mutations of the inputs and of the column pair that the comparison
/// must detect, on the case's site configuration.
fn mutations(
    run: &mut Run,
    corpus: &[corpus::Case],
    case: &Composite,
    widen: bool,
) -> LabResult<usize> {
    let s = run.s;
    let rows = case.size();
    let map = (0..rows).collect::<Vec<_>>();
    let reference = cases::input(s, corpus, case, &map)?;
    let mutated = device::variant(MUTATED)?;
    for name in MUTATIONS {
        let candidate = cases::input(s, corpus, case, &map)?;
        let mut epsilon = None;
        match name {
            "epsilon_one" => epsilon = Some(1.0_f32.to_bits()),
            "live_short" => set_live(&candidate, s, rows - 1)?,
            "gamma_sign" => flip(&candidate, &s.stream, "norm_weight", 2, 0)?,
            "fn_sign" => flip(&candidate, &s.stream, "fn", 4, 0)?,
            "sum_sign" => flip(&candidate, &s.stream, "sum", 4, 7)?,
            _ => unreachable!(),
        }
        let label = format!("{}:mutation", case.id);
        run.paired(
            &label, name, &reference, &candidate, &mutated, widen, true, 0, epsilon, false,
        )?;
    }
    for control in device::controls() {
        let label = format!("{}:{}", case.id, control.symbols().join("+"));
        run.paired(
            &label,
            "mutation_detected",
            &reference,
            &reference,
            &control,
            widen,
            true,
            control_prior(&control),
            None,
            false,
        )?;
    }
    Ok(MUTATIONS.len() + device::controls().len())
}

/// Every candidate on every case (or, for the sanitizers, the first case
/// of each kind in the served configuration).
pub fn matrix(s: &Session, sanitizer: bool) -> LabResult<Value> {
    let (corpus, all, sha) = artifacts::cases()?;
    let composites = if sanitizer {
        artifacts::sanitizer_cases(&all, &artifacts::sanitizer_kinds()?)
    } else {
        all.iter().collect()
    };
    let k = Kernels::load(s)?;
    let mut run = Run {
        s,
        k: &k,
        checks: Vec::new(),
    };
    let names = device::variants();
    let candidates = names
        .iter()
        .map(|n| device::variant(n))
        .collect::<LabResult<Vec<_>>>()?;
    let mut reports = Vec::new();
    let mut served = Vec::new();
    let mut captured = Vec::new();
    let mut seen = BTreeSet::new();
    let mut negatives = 0;
    let start = std::time::Instant::now();
    for case in &composites {
        let rows = case.size();
        let map = (0..rows).collect::<Vec<_>>();
        let input = cases::input(s, &corpus, case, &map)?;
        let old = Output::new(s, rows)?;
        let out = Output::new(s, rows)?;
        old.assert_disjoint(&out)?;
        let site = cases::widening(&corpus, case);
        let configurations = if sanitizer {
            vec![(site, true)]
        } else {
            vec![(site, true), (!site, false)]
        };
        for (widen, alias) in configurations {
            let configuration = if alias { "served" } else { "other" };
            let expected = run.eager(&input, &old, None, widen, alias, 0, None)?;
            served.push(json!({"case":case.id,"configuration":configuration,"output_sha256":digest(&expected)}));
            if alias && !sanitizer {
                // The served boundary reproduces the captured forwards'
                // outputs row by row (their workspace layout aside).
                let rows_by_role = cases::captured(&corpus, case)?;
                let failures = rows_by_role
                    .iter()
                    .filter(|(role, _)| role.as_str() != "partials")
                    .filter(|(role, captured)| {
                        let data = &expected[role.as_str()];
                        let stride = data.len() / rows;
                        captured
                            .iter()
                            .enumerate()
                            .any(|(r, row)| data[r * stride..(r + 1) * stride] != row[..])
                    })
                    .map(|(role, _)| role.clone())
                    .collect::<Vec<_>>();
                // Decode forwards captured the rounded and expanded rows
                // too; widening sites the widened row.
                let outputs = rows_by_role
                    .keys()
                    .filter(|r| r.as_str() != "partials")
                    .count();
                let recorded = if case.kind == "decode8x2" { 10 } else { 8 } + usize::from(site);
                captured.push(json!({"case":case.id,"outputs":outputs,"failures":failures,
                    "passed":failures.is_empty() && outputs == recorded}));
            }
            for (name, candidate) in names.iter().zip(&candidates) {
                let actual = run.eager(&input, &out, Some(candidate), widen, alias, 0, None)?;
                let failures = compare(&actual, &expected, Some(candidate));
                let mut report = json!({"case":case.id,"kind":case.kind,"rows":rows,"variant":name,
                    "configuration":configuration,"widen":widen,"alias":alias,
                    "passed":failures.is_empty(),"failures":failures});
                if !failures.is_empty() {
                    report["output_sha256"] = json!(digest(&actual));
                }
                reports.push(report);
                if sanitizer {
                    continue;
                }
                if seen.insert((case.kind, name.clone(), configuration)) {
                    let label = format!("{}:{name}:{configuration}", case.id);
                    controls(
                        &mut run, &corpus, case, &label, candidate, widen, alias, &actual,
                    )?;
                }
            }
            if alias && !sanitizer && seen.insert((case.kind, "mutations".into(), configuration)) {
                negatives += mutations(&mut run, &corpus, case, widen)?;
            }
        }
        println!(
            "HC_COLS_CASE {}",
            json!({"case":case.id,"kind":case.kind,"rows":rows,"reports":reports.len(),
                "checks":run.checks.len(),"elapsed_s":start.elapsed().as_secs_f64()})
        );
    }
    let passed = [&reports, &run.checks, &captured]
        .iter()
        .all(|v| v.iter().all(|r| r["passed"] == true))
        && !reports.is_empty();
    println!(
        "HC_COLS_RESULT sanitizer={sanitizer} cases={} reports={} checks={} negatives={negatives} captured={} passed={passed}",
        composites.len(),
        reports.len(),
        run.checks.len(),
        captured.len()
    );
    Ok(
        json!({"schema":"chopinfer.hc-cols-matrix.v1","corpus_sha256":sha,"cases":composites.len(),
        "sanitizer":sanitizer,"variants":names,"reports":reports,"served":served,"checks":run.checks,
        "captured":captured,"negatives":negatives,"passed":passed}),
    )
}

fn write(name: &str, value: &Value) -> LabResult<()> {
    std::fs::write(
        std::env::var_os(name).ok_or("report path required")?,
        serde_json::to_vec(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

#[test]
fn comparison_rejects_changed_values_row_positions_and_written_lanes() {
    let a = BTreeMap::from([("value".into(), vec![1, 2, 3, 4])]);
    let b = BTreeMap::from([("value".into(), vec![1, 2, 3, 5])]);
    assert!(!mismatch(&a, &b).is_empty());
    assert!(mismatch(&a, &a).is_empty());
    assert!(!same_rows(&a, &a, &[1, 0], 2));
    assert!(same_rows(&a, &a, &[0, 1], 2));
    let values = BTreeMap::from([("value".into(), (0..16).collect::<Vec<u8>>())]);
    assert_eq!(live_rows(&values, 3, 8)["value"], [0, 1, 2, 3, 4, 5]);
    // The served producer's lanes against a column pair's untouched words.
    let served = BTreeMap::from([
        ("lanes".into(), vec![1, 2, 3, 4]),
        ("partials".into(), vec![5, 6]),
    ]);
    let mut kept = served.clone();
    kept.insert("lanes".into(), vec![POISON; 4]);
    let cols = device::variant("cols_r2_w8").unwrap();
    let twin = device::variant("twin32").unwrap();
    assert!(compare(&kept, &served, Some(&cols)).is_empty());
    assert!(!compare(&served, &served, Some(&cols)).is_empty());
    assert!(!compare(&kept, &served, Some(&twin)).is_empty());
    assert!(compare(&served, &served, Some(&twin)).is_empty());
    assert!(compare(&served, &served, None).is_empty());
    let mut changed = kept.clone();
    changed.get_mut("partials").unwrap()[1] ^= 1;
    assert!(!compare(&changed, &served, Some(&cols)).is_empty());
    for rows in [16, 32] {
        for (_, part) in parts(rows) {
            assert!(part.len() > rows / 2 && part.len() < rows);
            assert_eq!(part.iter().collect::<BTreeSet<_>>().len(), part.len());
            assert!(part.iter().all(|&r| r < rows));
        }
    }
}

#[test]
fn every_control_runs_under_a_prior_that_shows_it() {
    let controls = device::controls();
    assert_eq!(controls.len(), 5);
    for c in &controls {
        let Candidate::Cols {
            producer, finish, ..
        } = c
        else {
            panic!("controls are column pairs");
        };
        // Each control mutates exactly one half.
        assert_ne!(
            producer.mutation == round_cols::Mutation::None,
            finish.mutation == finish_split::Mutation::None
        );
        assert_eq!(
            control_prior(c) == 0x80,
            finish.mutation == finish_split::Mutation::ClearStatus
        );
    }
    assert!(device::variants().contains(&MUTATED.to_owned()));
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COLS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn real_rows_bitwise_invariances_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, false)?;
        write("CHOPINFER_HC_COLS_REPORT", &value)?;
        if value["passed"] != true {
            return Err("bitwise matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "matrix cannot skip");
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COLS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn sanitized_real_rows_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, true)?;
        write("CHOPINFER_HC_COLS_SANITIZER_REPORT", &value)?;
        if value["passed"] != true {
            return Err("sanitizer matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "sanitizer matrix cannot skip");
}

/// Wide rows of the same authenticated sites, checked against both the
/// ordered-chain pair and the captured per-row outputs. The reference pair
/// is an arithmetic control, not the serving latency baseline above32 rows.
#[test]
#[ignore = "requires frozen HC artifacts, the complete corpus and an allocated GPU pair"]
fn wide_verify_bitwise_graphs_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(artifacts::timing_cubins().unwrap(), 1 << 30, |s| {
        let (corpus, cases, sha) = cases::load_wide()?;
        let k = Kernels::load(s)?;
        let candidate = device::variant("cols_r4tx_w32")?;
        let mut run = Run { s, k: &k, checks: Vec::new() };
        let mut reports = 0;
        let mut negatives = 0;
        let mut timings = 0;
        for case in &cases {
            let rows = case.size();
            let map = (0..rows).collect::<Vec<_>>();
            let input = cases::input(s, &corpus, case, &map)?;
            let old = Output::new(s, rows)?;
            let new = Output::new(s, rows)?;
            old.assert_disjoint(&new)?;
            let site_widen = cases::widening(&corpus, case);
            for (widen, alias) in [(site_widen, true), (!site_widen, false)] {
                let expected = run.eager(&input, &old, None, widen, alias, 0, None)?;
                let actual = run.eager(&input, &new, Some(&candidate), widen, alias, 0, None)?;
                let failures = compare(&actual, &expected, Some(&candidate));
                if !failures.is_empty() { return Err(format!("{}: {failures:?}", case.id)); }
                if alias {
                    let captured = cases::captured(&corpus, case)?;
                    let mut compared = 0;
                    for (role, values) in captured.iter().filter(|(r, _)| r.as_str() != "partials") {
                        let bytes = &actual[role];
                        let stride = bytes.len() / rows;
                        if values.iter().enumerate().any(|(row, value)| bytes[row*stride..(row+1)*stride] != value[..]) {
                            return Err(format!("{}: captured {role} differs", case.id));
                        }
                        compared += 1;
                    }
                    if compared != 8 + usize::from(site_widen) { return Err("missing captured HC outputs".into()); }
                }
                if case.site == 0 {
                    controls(&mut run, &corpus, case, &case.id, &candidate, widen, alias, &actual)?;
                    // Invalid inputs, sticky status and inactive rows also
                    // replay through captured graphs with unchanged pointers.
                    let varied = cases::input(s, &corpus, case, &map)?;
                    let outputs = [Output::new(s, rows)?, Output::new(s, rows)?];
                    let graphs = (0..2).map(|arm| s.capture(|stream| {
                        device::launch(stream, &k, &varied, &outputs[arm],
                            (arm == 1).then_some(&candidate), widen, alias, None).map(|_| ())
                    })).collect::<LabResult<Vec<_>>>()?;
                    for pattern in std::iter::once("finite").chain(SPECIALS) {
                        // Restore all operands before the next mutation.
                        for (role, data) in &input.data {
                            let mut bytes = vec![0; data.byte_len()];
                            data.copy_to_bytes(&mut bytes, &s.stream).map_err(|e| e.to_string())?;
                            varied.data[role].copy_from_bytes(&bytes, &s.stream).map_err(|e| e.to_string())?;
                        }
                        if pattern != "finite" { alter(&varied, &s.stream, pattern)?; }
                        for live in [rows, 0, rows - 1, rows] {
                            set_live(&varied, s, live)?;
                            for prior in [0, 0x80] {
                                for arm in 0..2 {
                                    device::reset(&s.stream, &varied, &outputs[arm], prior, alias)?;
                                    graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                                }
                                let reference = device::read(&s.stream, &outputs[0])?;
                                let observed = device::read(&s.stream, &outputs[1])?;
                                if !compare(&observed, &reference, Some(&candidate)).is_empty() {
                                    return Err(format!("{} graph pattern={pattern} live={live} prior={prior}", case.id));
                                }
                            }
                        }
                    }
                    if alias { negatives += mutations(&mut run, &corpus, case, widen)?; }
                }
                reports += 1;
            }
            if case.site.is_multiple_of(artifacts::TIMING_STRIDE) {
                // Independent coefficient spans make repeated graphs idempotent.
                for output in [&old, &new] { device::reset(&s.stream, &input, output, 0, false)?; }
                let reference = s.capture(|stream| device::launch(stream, &k, &input, &old, None,
                    site_widen, false, None).map(|_| ()))?;
                for (name, selected) in [("AA", None), ("AB", Some(&candidate))] {
                    let graph = s.capture(|stream| device::launch(stream, &k, &input, &new, selected,
                        site_widen, false, None).map(|_| ()))?;
                    for hot in [false, true] {
                        let graphs = [&reference, &graph];
                        let time = kernel_harness::time_ab_prepared(s, 2, 7,
                            [kernel_harness::eviction::EvictionMode::Read; 2],
                            |arm, stream| {
                                if hot { graphs[arm].launch(stream).map_err(|e| e.to_string())?; }
                                Ok(())
                            },
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()))?;
                        println!("WIDE_HC_COLS_TIMING {}", json!({"case":case.id,"rows":rows,
                            "arm":name,"hot":hot,"old_us":time.arms[0].median_ms*1000.,
                            "new_us":time.arms[1].median_ms*1000.,"ratio":time.paired_ratio,
                            "reference":"ordered-chain arithmetic control"}));
                        timings += 1;
                    }
                }
            }
            println!("WIDE_HC_COLS_CASE {}", json!({"case":case.id,"rows":rows,"reports":reports}));
        }
        assert_eq!(cases.len(), cases::SITES * cases::WIDE_ROWS.len());
        assert_eq!(reports, 2 * cases.len());
        assert_eq!(timings, cases::SITES.div_ceil(artifacts::TIMING_STRIDE) * cases::WIDE_ROWS.len() * 4);
        assert_eq!(negatives, cases::WIDE_ROWS.len() * (MUTATIONS.len() + device::controls().len()));
        assert!(!run.checks.is_empty() && run.checks.iter().all(|r| r["passed"] == true));
        s.receipt("wide-hc-cols", BTreeMap::from([
            ("corpus_sha256".into(), json!(sha)), ("cases".into(), json!(cases.len())),
            ("reports".into(), json!(reports)), ("checks".into(), json!(run.checks)),
            ("negatives".into(), json!(negatives)), ("timings".into(), json!(timings)),
        ]))?;
        println!("WIDE_HC_COLS_COMPLETE cases={} reports={reports} negatives={negatives} timings={timings}", cases.len());
        Ok(())
    }).unwrap();
    assert!(result.is_some(), "an idle GPU is required");
}
