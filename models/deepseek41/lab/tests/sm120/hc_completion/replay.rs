//! Bitwise comparison of every candidate with the served pair on every real
//! C1 and C8 boundary, with invariance, ticket and mutation controls.
use super::{
    artifacts, corpus,
    device::{self, Input, Kernels, Output, Tickets, VARIANTS},
    registration,
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

fn captured_matches(role: &str, actual: &[u8], captured: &[u8], rows: usize) -> bool {
    if role == "partials" && captured.len() == rows * 3840 * 4 {
        return actual.len() == rows * 4096 * 4
            && actual
                .chunks_exact(4096 * 4)
                .zip(captured.chunks_exact(3840 * 4))
                .all(|(a, b)| &a[..3840 * 4] == b);
    }
    actual == captured
}
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
fn digest(values: &Values) -> String {
    kernel_build::digest(&values.values().flatten().copied().collect::<Vec<_>>())
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
    let buffer = &i.data[role];
    let mut data = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut data, s)
        .map_err(|e| e.to_string())?;
    data[index * size..(index + 1) * size].copy_from_slice(&bits.to_le_bytes()[..size]);
    buffer.copy_from_bytes(&data, s).map_err(|e| e.to_string())
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
    /// One eager launch from poisoned outputs; the tickets must return to zero.
    #[allow(clippy::too_many_arguments)]
    fn eager(
        &self,
        i: &Input,
        o: &Output,
        t: &Tickets,
        variant: &str,
        widen: bool,
        alias: bool,
        prior: u32,
    ) -> LabResult<(Values, bool)> {
        device::reset(&self.s.stream, i, o, prior, alias)?;
        device::launch(&self.s.stream, self.k, i, o, t, variant, widen, alias)?;
        let values = device::read(&self.s.stream, o)?;
        Ok((values, t.check(&self.s.stream).is_ok()))
    }
    /// Candidate equals the served pair under the same inputs and state.
    #[allow(clippy::too_many_arguments)]
    fn paired(
        &mut self,
        label: &str,
        name: &str,
        i: &Input,
        variant: &str,
        widen: bool,
        alias: bool,
        prior: u32,
    ) -> LabResult<()> {
        let old = Output::new(self.s, i.rows)?;
        let new = Output::new(self.s, i.rows)?;
        let tickets = Tickets::new(self.s, i.rows)?;
        let (expected, _) = self.eager(i, &old, &tickets, "served", widen, alias, prior)?;
        let (actual, zero) = self.eager(i, &new, &tickets, variant, widen, alias, prior)?;
        self.check(label, name, zero && mismatch(&actual, &expected).is_empty());
        Ok(())
    }
}

/// Invariances and controls, once per row count, variant and configuration.
#[allow(clippy::too_many_arguments)]
fn controls(
    run: &mut Run,
    case: &corpus::Case,
    label: &str,
    variant: &str,
    widen: bool,
    alias: bool,
    actual: &Values,
) -> LabResult<()> {
    let s = run.s;
    let rows = case.rows;
    let map = (0..rows).collect::<Vec<_>>();
    for chunk in [1, 3] {
        for (index, part) in map.chunks(chunk).enumerate() {
            let moved = Input::new(s, case, part)?;
            let out = Output::new(s, part.len())?;
            let t = Tickets::new(s, part.len())?;
            let (values, zero) = run.eager(&moved, &out, &t, variant, widen, alias, 0)?;
            run.check(
                label,
                &format!("chunk{chunk}:{index}"),
                zero && same_rows(&values, actual, part, rows),
            );
        }
    }
    let reversed = map.iter().copied().rev().collect::<Vec<_>>();
    let moved = Input::new(s, case, &reversed)?;
    let other = Output::new(s, rows)?;
    let t = Tickets::new(s, rows)?;
    let (values, zero) = run.eager(&moved, &other, &t, variant, widen, alias, 0)?;
    run.check(
        label,
        "reverse_position",
        zero && same_rows(&values, actual, &reversed, rows),
    );
    // Two private output sets and ticket arrays on two streams at once.
    let input = Input::new(s, case, &map)?;
    let first = Output::new(s, rows)?;
    let first_tickets = Tickets::new(s, rows)?;
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
            &first_tickets,
            variant,
            widen,
            alias,
        )?;
        device::launch(&side, run.k, &moved, &other, &t, variant, widen, alias)?;
    }
    side.synchronize().map_err(|e| e.to_string())?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let repeated = !alias;
    let a = device::read(&s.stream, &first)?;
    let b = device::read(&s.stream, &other)?;
    // Repeated launches are idempotent only when post/comb are not aliased.
    run.check(
        label,
        "concurrent_scratch",
        first_tickets.check(&s.stream).is_ok()
            && t.check(&s.stream).is_ok()
            && (!repeated || (a == *actual && same_rows(&b, actual, &reversed, rows))),
    );
    for live in BTreeSet::from([0, rows - 1]) {
        input
            .live
            .copy_from_bytes(&(live as u32).to_ne_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        run.paired(
            label,
            &format!("live{live}"),
            &input,
            variant,
            widen,
            alias,
            0,
        )?;
    }
    input
        .live
        .copy_from_bytes(&(rows as u32).to_ne_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    run.paired(label, "sticky", &input, variant, widen, alias, 0x80)?;
    for pattern in SPECIALS {
        let altered = Input::new(s, case, &map)?;
        alter(&altered, &s.stream, pattern)?;
        run.paired(label, pattern, &altered, variant, widen, alias, 0)?;
    }
    let mut changed = actual.clone();
    changed.get_mut("normed").unwrap()[0] ^= 1;
    run.check(
        label,
        "negative_byte",
        !mismatch(&changed, actual).is_empty(),
    );
    Ok(())
}

pub fn matrix(s: &Session, memcheck: bool) -> LabResult<Value> {
    let (cases, sha) = if memcheck {
        (artifacts::memcheck_cases()?, String::new())
    } else {
        artifacts::cases()?
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
    for case in &cases {
        let rows = case.rows;
        let input = Input::new(s, case, &(0..rows).collect::<Vec<_>>())?;
        let old = Output::new(s, rows)?;
        let out = Output::new(s, rows)?;
        old.assert_disjoint(&out)?;
        let tickets = Tickets::new(s, rows)?;
        let mut count = 0;
        let configurations = if memcheck {
            vec![(case.widening(), true)]
        } else {
            vec![(case.widening(), true), (!case.widening(), false)]
        };
        for (widen, alias) in configurations {
            let configuration = if alias { "served" } else { "other" };
            let (expected, zero) = run.eager(&input, &old, &tickets, "served", widen, alias, 0)?;
            if !zero {
                return Err("served pair changed the ticket buffer".into());
            }
            if alias && !memcheck {
                let mut errors = Vec::new();
                let mut outputs = 0;
                for (role, values) in &expected {
                    if case.record["outputs"][role]["file"].is_string() {
                        outputs += 1;
                        if !captured_matches(role, values, &case.operand("outputs", role)?, rows) {
                            errors.push(role.clone());
                        }
                    }
                }
                if outputs < 6 {
                    return Err("captured outputs are incomplete".into());
                }
                captured.push(json!({"case":case.id,"outputs":outputs,"failures":errors,"passed":errors.is_empty()}));
            }
            for variant in VARIANTS {
                let label = format!("{}:{variant}:{configuration}", case.id);
                let (actual, zero) = run.eager(&input, &out, &tickets, variant, widen, alias, 0)?;
                let mut failures = mismatch(&actual, &expected);
                if !zero {
                    failures.push("tickets".into());
                }
                reports.push(json!({"case":case.id,"rows":rows,"variant":variant,"configuration":configuration,"widen":widen,"alias":alias,"passed":failures.is_empty(),"failures":failures,"output_sha256":digest(&actual)}));
                count += 1;
                if memcheck {
                    continue;
                }
                // Two replays of one graph reuse the same tickets.
                let graph = s.capture(|stream| {
                    device::launch(stream, &k, &input, &out, &tickets, variant, widen, alias)
                        .map(|_| ())
                })?;
                for replay in 0..2 {
                    device::reset(&s.stream, &input, &out, 0, alias)?;
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                    let values = device::read(&s.stream, &out)?;
                    let zero = tickets.check(&s.stream).is_ok();
                    run.check(
                        &label,
                        &format!("graph_replay{replay}"),
                        zero && values == actual,
                    );
                }
                if seen.insert((rows, variant, configuration)) {
                    controls(&mut run, case, &label, variant, widen, alias, &actual)?;
                }
            }
        }
        println!(
            "HC_COMPLETION_CASE {}",
            json!({"case":case.id,"rows":rows,"reports":count})
        );
    }
    let mut negatives = 0;
    if !memcheck {
        for rows in [1, 8] {
            let case = cases.iter().find(|c| c.rows == rows).unwrap();
            let input = Input::new(s, case, &(0..rows).collect::<Vec<_>>())?;
            for entry in registration::controls() {
                let old = Output::new(s, rows)?;
                let new = Output::new(s, rows)?;
                let tickets = Tickets::new(s, rows)?;
                let widen = case.widening();
                let prior = 0x80;
                let (expected, _) =
                    run.eager(&input, &old, &tickets, "served", widen, true, prior)?;
                device::reset(&s.stream, &input, &new, prior, true)?;
                device::launch_with(
                    &s.stream,
                    &k,
                    &input,
                    &new,
                    &tickets,
                    Some((&entry, 8)),
                    widen,
                    true,
                )?;
                let actual = device::read(&s.stream, &new)?;
                let label = format!("{}:{}", case.id, entry.symbol(8));
                run.check(
                    &label,
                    "mutation_detected",
                    !mismatch(&actual, &expected).is_empty(),
                );
                negatives += 1;
            }
        }
    }
    let passed = [&reports, &run.checks, &captured]
        .iter()
        .all(|v| v.iter().all(|r| r["passed"] == true));
    println!(
        "HC_COMPLETION_RESULT memcheck={memcheck} cases={} reports={} checks={} negatives={negatives} passed={passed}",
        cases.len(),
        reports.len(),
        run.checks.len()
    );
    Ok(
        json!({"schema":"chopinfer.hc-completion-matrix.v1","corpus_sha256":sha,"cases":cases.len(),
        "memcheck":memcheck,"reports":reports,"checks":run.checks,"captured":captured,"negatives":negatives,"passed":passed}),
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
    let mut full = vec![0x11; 2 * 4096 * 4];
    full[4096 * 4..].fill(0x22);
    let captured = full
        .chunks_exact(4096 * 4)
        .flat_map(|row| row[..3840 * 4].iter().copied())
        .collect::<Vec<_>>();
    assert!(captured_matches("partials", &full, &captured, 2));
    full[4096 * 4] ^= 1;
    assert!(!captured_matches("partials", &full, &captured, 2));
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COMPLETION_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn real_c1_c8_bitwise_invariances_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, false)?;
        write("CHOPINFER_HC_COMPLETION_REPORT", &value)?;
        if value["passed"] != true {
            return Err("bitwise matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "matrix cannot skip");
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU, CHOPINFER_HC_COMPLETION_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn memcheck_real_c1_c8_on_gpu() {
    let result = kernel_harness::gpu_test(1 << 30, |s| -> LabResult<()> {
        let value = matrix(s, true)?;
        write("CHOPINFER_HC_COMPLETION_MEMCHECK_REPORT", &value)?;
        if value["passed"] != true {
            return Err("memcheck matrix failed; see the report".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "memcheck cannot skip");
}
