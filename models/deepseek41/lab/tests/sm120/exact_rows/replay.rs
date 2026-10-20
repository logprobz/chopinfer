use super::{
    artifacts, corpus,
    device::{self, Input, Kernels, Output, VARIANTS},
};
use kernel_harness::LabResult;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
type Values = BTreeMap<String, Vec<u8>>;
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

fn mismatch(a: &Values, b: &Values) -> Vec<String> {
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
fn control(checks: &mut Vec<Value>, label: &str, name: &str, passed: bool) {
    checks.push(json!({"case":label,"name":name,"passed":passed}));
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
#[test]
fn comparison_rejects_changed_values_and_row_positions() {
    let a = BTreeMap::from([("value".into(), vec![1, 2, 3, 4])]);
    let b = BTreeMap::from([("value".into(), vec![1, 2, 3, 5])]);
    assert!(!mismatch(&a, &b).is_empty());
    assert!(!same_rows(&a, &a, &[1, 0], 2));
}

#[test]
fn captured_partials_preserve_row_strides_and_full_scratch_checks() {
    let mut full = vec![0x11; 2 * 4096 * 4];
    full[4096 * 4..].fill(0x22);
    let captured = full
        .chunks_exact(4096 * 4)
        .flat_map(|row| row[..3840 * 4].iter().copied())
        .collect::<Vec<_>>();
    assert!(captured_matches("partials", &full, &captured, 2));
    assert!(!captured_matches(
        "partials",
        &full,
        &captured[..captured.len() - 1],
        2
    ));
    let original = full.clone();
    full[4096 * 4] ^= 1;
    assert!(!captured_matches("partials", &full, &captured, 2));
    full[4096 * 4] ^= 1;
    full[3840 * 4] ^= 1;
    assert!(captured_matches("partials", &full, &captured, 2));
    assert!(
        !mismatch(
            &BTreeMap::from([("partials".into(), original)]),
            &BTreeMap::from([("partials".into(), full)]),
        )
        .is_empty()
    );
}
#[test]
#[ignore = "requires frozen modules and real captured operands on an idle GPU"]
fn real_inputs_bitwise_invariances_and_controls_on_gpu() {
    run(&VARIANTS);
}

#[test]
#[ignore = "requires authenticated real operands and an idle GPU"]
fn served_kernels_bitwise_invariances_and_controls_on_gpu() {
    run(&["served"]);
}

fn run(variants: &[&str]) {
    let result = kernel_harness::gpu_test(512 * 1024 * 1024, |s| -> LabResult<()> {
        let (cases, sha) = artifacts::cases()?;
        let kernels = Kernels::load(s)?;
        let mut reports = Vec::new();
        let mut checks = Vec::new();
        let mut captured = Vec::new();
        let mut seen = BTreeSet::new();
        for case in &cases {
            let map = (0..case.rows).collect::<Vec<_>>();
            let input = Input::new(s, case, &map)?;
            let old = Output::new(s, case.rows)?;
            let output = Output::new(s, case.rows)?;
            old.assert_disjoint(&output)?;
            for widen in [false, true] {
                old.poison(&s.stream, 0)?;
                device::launch(&s.stream, &kernels, &input, &old, "reference", widen)?;
                let expected = device::read(&s.stream, &old)?;
                if widen == case.widening() {
                    let mut errors = Vec::new();
                    let mut count = 0;
                    for (role, actual) in &expected {
                        if case.record["outputs"][role]["file"].is_string() {
                            count += 1;
                            if !captured_matches(
                                role,
                                actual,
                                &case.operand("outputs", role)?,
                                case.rows,
                            ) {
                                errors.push(role.clone());
                            }
                        }
                    }
                    if count < 6 {
                        return Err("captured outputs are incomplete".into());
                    }
                    captured.push(json!({"case":case.id,"outputs":count,"failures":errors,"passed":errors.is_empty()}));
                }
                for &variant in variants {
                    let label = format!("{}:{variant}:widen={widen}", case.id);
                    output.poison(&s.stream, 0)?;
                    device::launch(&s.stream, &kernels, &input, &output, variant, widen)?;
                    let actual = device::read(&s.stream, &output)?;
                    let failures = mismatch(&actual, &expected);
                    let hash = kernel_build::digest(
                        &actual
                            .values()
                            .flat_map(|v| v.iter().copied())
                            .collect::<Vec<_>>(),
                    );
                    reports.push(json!({"case":case.id,"rows":case.rows,"variant":variant,"widen":widen,"passed":failures.is_empty(),"failures":failures,"output_sha256":hash}));
                    output.poison(&s.stream, 0)?;
                    let graph = s.capture(|stream| {
                        device::launch(stream, &kernels, &input, &output, variant, widen)
                    })?;
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                    control(
                        &mut checks,
                        &label,
                        "poisoned_graph",
                        device::read(&s.stream, &output)? == actual,
                    );
                    if !seen.insert((case.rows, variant, widen)) {
                        continue;
                    }
                    for chunk in [1, 3] {
                        for (index, part) in map.chunks(chunk).enumerate() {
                            let moved = Input::new(s, case, part)?;
                            let out = Output::new(s, part.len())?;
                            device::launch(&s.stream, &kernels, &moved, &out, variant, widen)?;
                            control(
                                &mut checks,
                                &label,
                                &format!("chunk{chunk}:{index}"),
                                same_rows(
                                    &device::read(&s.stream, &out)?,
                                    &actual,
                                    part,
                                    case.rows,
                                ),
                            );
                        }
                    }
                    let reversed = map.iter().copied().rev().collect::<Vec<_>>();
                    let moved = Input::new(s, case, &reversed)?;
                    let other = Output::new(s, case.rows)?;
                    device::launch(&s.stream, &kernels, &moved, &other, variant, widen)?;
                    control(
                        &mut checks,
                        &label,
                        "reverse_position",
                        same_rows(
                            &device::read(&s.stream, &other)?,
                            &actual,
                            &reversed,
                            case.rows,
                        ),
                    );
                    for ranks in [1, 2, 4] {
                        for rank in 0..ranks {
                            other.poison(&s.stream, 0)?;
                            device::launch(&s.stream, &kernels, &moved, &other, variant, widen)?;
                            control(
                                &mut checks,
                                &label,
                                &format!("replicated_ranks{ranks}:{rank}"),
                                same_rows(
                                    &device::read(&s.stream, &other)?,
                                    &actual,
                                    &reversed,
                                    case.rows,
                                ),
                            );
                        }
                    }
                    output.assert_disjoint(&other)?;
                    output.poison(&s.stream, 0)?;
                    other.poison(&s.stream, 0)?;
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let stream = s.context.stream().map_err(|e| e.to_string())?;
                    device::launch(&stream, &kernels, &moved, &other, variant, widen)?;
                    device::launch(&s.stream, &kernels, &input, &output, variant, widen)?;
                    stream.synchronize().map_err(|e| e.to_string())?;
                    control(
                        &mut checks,
                        &label,
                        "concurrent_scratch",
                        device::read(&s.stream, &output)? == actual
                            && same_rows(
                                &device::read(&s.stream, &other)?,
                                &actual,
                                &reversed,
                                case.rows,
                            ),
                    );
                    for live in [0, case.rows - 1] {
                        input
                            .live
                            .copy_from_bytes(&(live as u32).to_le_bytes(), &s.stream)
                            .map_err(|e| e.to_string())?;
                        old.poison(&s.stream, 0)?;
                        output.poison(&s.stream, 0)?;
                        device::launch(&s.stream, &kernels, &input, &old, "reference", widen)?;
                        device::launch(&s.stream, &kernels, &input, &output, variant, widen)?;
                        control(
                            &mut checks,
                            &label,
                            &format!("live{live}"),
                            device::read(&s.stream, &old)? == device::read(&s.stream, &output)?,
                        );
                    }
                    input
                        .live
                        .copy_from_bytes(&(case.rows as u32).to_le_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                    old.poison(&s.stream, 1)?;
                    output.poison(&s.stream, 1)?;
                    device::launch(&s.stream, &kernels, &input, &old, "reference", widen)?;
                    device::launch(&s.stream, &kernels, &input, &output, variant, widen)?;
                    let sticky = device::read(&s.stream, &output)?;
                    control(
                        &mut checks,
                        &label,
                        "sticky",
                        device::read(&s.stream, &old)? == sticky
                            && sticky["codes"].iter().all(|&x| x == 0)
                            && sticky["scales"].iter().all(|&x| x == 127),
                    );
                    let mut changed = actual.clone();
                    changed.get_mut("normed").unwrap()[0] ^= 1;
                    control(
                        &mut checks,
                        &label,
                        "negative_byte",
                        !mismatch(&changed, &actual).is_empty(),
                    );
                    for pattern in [
                        "sum_nan",
                        "sum_inf",
                        "sum_subnormal",
                        "negative_zero",
                        "weight_nan",
                        "gamma_inf",
                        "gamma_last_nan",
                        "random",
                    ] {
                        let special = Input::new(s, case, &map)?;
                        alter(&special, &s.stream, pattern)?;
                        old.poison(&s.stream, 0)?;
                        output.poison(&s.stream, 0)?;
                        device::launch(&s.stream, &kernels, &special, &old, "reference", widen)?;
                        device::launch(&s.stream, &kernels, &special, &output, variant, widen)?;
                        control(
                            &mut checks,
                            &label,
                            pattern,
                            device::read(&s.stream, &old)? == device::read(&s.stream, &output)?,
                        );
                    }
                }
            }
            println!(
                "HC_ROWS_CASE {}",
                json!({"case":case.id,"rows":case.rows,"reports":variants.len()*2})
            );
        }
        let passed = reports
            .iter()
            .chain(&checks)
            .chain(&captured)
            .all(|r| r["passed"] == true);
        let value = json!({"schema":"chopinfer.hc-exact-rows.v1","corpus_sha256":sha,"cases":cases.len(),"reports":reports,"checks":checks,"captured":captured,"passed":passed});
        std::fs::write(
            std::env::var_os("CHOPINFER_HC_ROWS_REPORT").ok_or("report path required")?,
            serde_json::to_vec(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "HC_ROWS_RESULT cases={} reports={} checks={} passed={passed}",
            cases.len(),
            value["reports"].as_array().unwrap().len(),
            value["checks"].as_array().unwrap().len()
        );
        if !passed {
            return Err("exact row qualification failed; all mismatches retained".into());
        }
        Ok(())
    });
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().is_some(), "matrix cannot skip");
}
