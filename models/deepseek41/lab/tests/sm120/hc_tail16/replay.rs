//! Bitwise comparison of the finish/norm tail with the served Decode-16
//! boundary on every 16-row case of real captured rows, with position,
//! live-row, concurrency, special-value and mutation controls.
use super::{
    cases::{self, Composite, ROWS},
    corpus,
    device::{self, Arm, Input, Kernels, Output},
};
use kernel_harness::{LabResult, Session};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
type Values = BTreeMap<String, Vec<u8>>;
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
/// Mutations of the tail's inputs or launch that must change its outputs.
pub const MUTATIONS: [&str; 5] = [
    "epsilon_one",
    "live_short",
    "gamma_sign",
    "fn_sign",
    "sum_sign",
];

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
    /// One eager launch of `arm` from poisoned outputs.
    #[allow(clippy::too_many_arguments)]
    fn eager(
        &self,
        i: &Input,
        o: &Output,
        arm: Arm,
        widen: bool,
        alias: bool,
        prior: u32,
        epsilon: Option<u32>,
    ) -> LabResult<Values> {
        device::reset(&self.s.stream, i, o, prior, alias)?;
        device::launch(&self.s.stream, self.k, i, o, arm, widen, alias, epsilon)?;
        device::read(&self.s.stream, o)
    }
    /// The tail over `candidate` against the served boundary over
    /// `reference`; passes when equality is `equal`.
    #[allow(clippy::too_many_arguments)]
    fn paired(
        &mut self,
        label: &str,
        name: &str,
        reference: &Input,
        candidate: &Input,
        widen: bool,
        alias: bool,
        prior: u32,
        epsilon: Option<u32>,
        equal: bool,
    ) -> LabResult<()> {
        let old = Output::new(self.s, ROWS)?;
        let new = Output::new(self.s, ROWS)?;
        let expected = self.eager(reference, &old, Arm::Served, widen, alias, prior, None)?;
        let actual = self.eager(candidate, &new, Arm::Tail, widen, alias, prior, epsilon)?;
        self.check(
            label,
            name,
            mismatch(&actual, &expected).is_empty() == equal,
        );
        Ok(())
    }
}

/// Invariances and controls, once per case kind and configuration.
#[allow(clippy::too_many_arguments)]
fn controls(
    run: &mut Run,
    corpus: &[corpus::Case],
    case: &Composite,
    label: &str,
    widen: bool,
    alias: bool,
    actual: &Values,
) -> LabResult<()> {
    let s = run.s;
    let map = (0..ROWS).collect::<Vec<_>>();
    // A partial batch: its rows first and live, padded with the case's
    // other rows, in the same 16-row bucket.
    let strided = (0..ROWS).step_by(2).chain([1]).collect::<Vec<_>>();
    for (name, part) in [
        ("part_head", (0..9).collect::<Vec<_>>()),
        ("part_tail", (7..ROWS).collect()),
        ("part_strided", strided),
    ] {
        let mut padded = part.clone();
        padded.extend((0..ROWS).filter(|r| !part.contains(r)));
        let moved = cases::input(s, corpus, case, &padded)?;
        set_live(&moved, s, part.len())?;
        let out = Output::new(s, ROWS)?;
        let values = run.eager(&moved, &out, Arm::Tail, widen, alias, 0, None)?;
        run.check(
            label,
            name,
            same_rows(&live_rows(&values, part.len(), ROWS), actual, &part, ROWS),
        );
        run.paired(
            label,
            &format!("{name}_served"),
            &moved,
            &moved,
            widen,
            alias,
            0,
            None,
            true,
        )?;
    }
    let reversed = map.iter().copied().rev().collect::<Vec<_>>();
    let moved = cases::input(s, corpus, case, &reversed)?;
    let other = Output::new(s, ROWS)?;
    let values = run.eager(&moved, &other, Arm::Tail, widen, alias, 0, None)?;
    run.check(
        label,
        "reverse_position",
        same_rows(&values, actual, &reversed, ROWS),
    );
    // Two private output sets on two streams at once.
    let input = cases::input(s, corpus, case, &map)?;
    let first = Output::new(s, ROWS)?;
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
            Arm::Tail,
            widen,
            alias,
            None,
        )?;
        device::launch(&side, run.k, &moved, &other, Arm::Tail, widen, alias, None)?;
    }
    side.synchronize().map_err(|e| e.to_string())?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let a = device::read(&s.stream, &first)?;
    let b = device::read(&s.stream, &other)?;
    // Repeated launches are idempotent only when post/comb are not aliased.
    run.check(
        label,
        "concurrent_scratch",
        alias || (a == *actual && same_rows(&b, actual, &reversed, ROWS)),
    );
    // Live counts a Decode-16 step serves (9..16) and the edges.
    for live in BTreeSet::from([0, 1, 9, ROWS - 1]) {
        set_live(&input, s, live)?;
        run.paired(
            label,
            &format!("live{live}"),
            &input,
            &input,
            widen,
            alias,
            0,
            None,
            true,
        )?;
    }
    set_live(&input, s, ROWS)?;
    run.paired(
        label, "sticky", &input, &input, widen, alias, 0x80, None, true,
    )?;
    for pattern in SPECIALS {
        let altered = cases::input(s, corpus, case, &map)?;
        alter(&altered, &s.stream, pattern)?;
        run.paired(
            label, pattern, &altered, &altered, widen, alias, 0, None, true,
        )?;
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

/// Mutations of the tail that the comparison must detect.
fn mutations(
    run: &mut Run,
    corpus: &[corpus::Case],
    case: &Composite,
    widen: bool,
) -> LabResult<usize> {
    let s = run.s;
    let map = (0..ROWS).collect::<Vec<_>>();
    let reference = cases::input(s, corpus, case, &map)?;
    for name in MUTATIONS {
        let candidate = cases::input(s, corpus, case, &map)?;
        let mut epsilon = None;
        match name {
            "epsilon_one" => epsilon = Some(1.0_f32.to_bits()),
            "live_short" => set_live(&candidate, s, ROWS - 1)?,
            "gamma_sign" => flip(&candidate, &s.stream, "norm_weight", 2, 0)?,
            "fn_sign" => flip(&candidate, &s.stream, "fn", 4, 0)?,
            "sum_sign" => flip(&candidate, &s.stream, "sum", 4, 7)?,
            _ => unreachable!(),
        }
        let label = format!("{}:mutation", case.id);
        run.paired(
            &label, name, &reference, &candidate, widen, true, 0, epsilon, false,
        )?;
    }
    Ok(MUTATIONS.len())
}

pub fn matrix(s: &Session, memcheck: bool) -> LabResult<Value> {
    let (corpus, composites, sha) = cases::load()?;
    let composites = if memcheck {
        super::artifacts::memcheck_cases(composites)
    } else {
        composites
    };
    let k = Kernels::load(s)?;
    let mut run = Run {
        s,
        k: &k,
        checks: Vec::new(),
    };
    let mut reports = Vec::new();
    let mut captured = Vec::new();
    let mut seen = BTreeSet::new();
    let mut negatives = 0;
    let map = (0..ROWS).collect::<Vec<_>>();
    for case in &composites {
        let input = cases::input(s, &corpus, case, &map)?;
        let old = Output::new(s, ROWS)?;
        let out = Output::new(s, ROWS)?;
        old.assert_disjoint(&out)?;
        let site = cases::widening(&corpus, case);
        let configurations = if memcheck {
            vec![(site, true)]
        } else {
            vec![(site, true), (!site, false)]
        };
        for (widen, alias) in configurations {
            let configuration = if alias { "served" } else { "other" };
            let label = format!("{}:{configuration}", case.id);
            let expected = run.eager(&input, &old, Arm::Served, widen, alias, 0, None)?;
            if alias && !memcheck {
                // The served boundary reproduces the captured forwards'
                // outputs row by row (report only: those forwards ran the
                // C8 decode and Verify-32 boundaries of their build).
                let rows = cases::captured(&corpus, case)?;
                let failures = rows
                    .iter()
                    .filter(|(role, rows)| {
                        let data = &expected[role.as_str()];
                        let stride = data.len() / ROWS;
                        rows.iter()
                            .enumerate()
                            .any(|(r, row)| data[r * stride..(r + 1) * stride] != row[..])
                    })
                    .map(|(role, _)| role.clone())
                    .collect::<Vec<_>>();
                captured.push(json!({"case":case.id,"outputs":rows.len(),"failures":failures,"passed":failures.is_empty()}));
            }
            let actual = run.eager(&input, &out, Arm::Tail, widen, alias, 0, None)?;
            let failures = mismatch(&actual, &expected);
            reports.push(json!({"case":case.id,"kind":case.kind,"configuration":configuration,"widen":widen,"alias":alias,
                "passed":failures.is_empty(),"failures":failures,"output_sha256":digest(&actual)}));
            if memcheck {
                continue;
            }
            let graph = s.capture(|stream| {
                device::launch(stream, &k, &input, &out, Arm::Tail, widen, alias, None).map(|_| ())
            })?;
            for replay in 0..2 {
                device::reset(&s.stream, &input, &out, 0, alias)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                let values = device::read(&s.stream, &out)?;
                run.check(&label, &format!("graph_replay{replay}"), values == actual);
            }
            if seen.insert((case.kind, configuration)) {
                controls(&mut run, &corpus, case, &label, widen, alias, &actual)?;
                if alias {
                    negatives += mutations(&mut run, &corpus, case, widen)?;
                }
            }
        }
        println!(
            "HC_TAIL16_CASE {}",
            json!({"case":case.id,"kind":case.kind,"reports":reports.len()})
        );
    }
    let passed = [&reports, &run.checks]
        .iter()
        .all(|v| v.iter().all(|r| r["passed"] == true))
        && !reports.is_empty();
    let captured_passed = captured.iter().all(|r| r["passed"] == true);
    println!(
        "HC_TAIL16_RESULT memcheck={memcheck} cases={} reports={} checks={} negatives={negatives} captured={} captured_passed={captured_passed} passed={passed}",
        composites.len(),
        reports.len(),
        run.checks.len(),
        captured.len()
    );
    Ok(
        json!({"schema":"chopinfer.hc-tail16-matrix.v1","corpus_sha256":sha,"cases":composites.len(),
        "memcheck":memcheck,"reports":reports,"checks":run.checks,"captured":captured,"captured_passed":captured_passed,
        "negatives":negatives,"passed":passed}),
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
fn comparison_rejects_changed_values_and_row_positions() {
    let a = BTreeMap::from([("value".into(), vec![1, 2, 3, 4])]);
    let b = BTreeMap::from([("value".into(), vec![1, 2, 3, 5])]);
    assert!(!mismatch(&a, &b).is_empty());
    assert!(mismatch(&a, &a).is_empty());
    assert!(!same_rows(&a, &a, &[1, 0], 2));
    assert!(same_rows(&a, &a, &[0, 1], 2));
    let values = BTreeMap::from([("value".into(), (0..16).collect::<Vec<u8>>())]);
    assert_eq!(live_rows(&values, 3, 8)["value"], [0, 1, 2, 3, 4, 5]);
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_TAIL16_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn real_rows16_bitwise_invariances_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, false)?;
        write("CHOPINFER_HC_TAIL16_REPORT", &value)?;
        if value["passed"] != true {
            return Err("bitwise matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "matrix cannot skip");
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_TAIL16_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn memcheck_real_rows16_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, true)?;
        write("CHOPINFER_HC_TAIL16_MEMCHECK_REPORT", &value)?;
        if value["passed"] != true {
            return Err("memcheck matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "memcheck cannot skip");
}
