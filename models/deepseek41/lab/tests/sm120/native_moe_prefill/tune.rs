//! The native MoE tuning pair (../../../native_tune.md): per 8192-row case,
//! the served gate/up and down tiles against each inner-loop variant of
//! `native_grouped::Tune` and the two-CTA tiles, as paired CUDA graphs. The
//! bitwise arms (lean, the paired tile and its lean loop) must write the
//! served output bytes; the ablations only time. Worker 0 takes the
//! balanced synthetic case and captured layer 14, worker 1 the skewed
//! synthetic case and captured layer 39, each on its own rank.
use super::captured::{BANKS, Banks, READY, WORKER, captures, case_inputs, json};
use super::pipeline::{Inputs, Pipeline, build, native_buffers, queue};
use super::*;
use kernel_core::precision::buffers::QuantizedWeights;
use kernels_sm120::moe::native_grouped::Tune;
use std::path::{Path, PathBuf};

/// Captured layers by worker.
pub const LAYERS: [u32; 2] = [14, 39];

/// One timed arm of a projection.
struct Arm {
    label: String,
    ops: Vec<Launch>,
    output: View,
    bitwise: bool,
}

/// The served tile's ops of `kind` and every variant arm, each writing its
/// own output (and queue).
fn arms(
    a: &mut Arena<'_>,
    p: &Pipeline,
    i: &Inputs,
    kind: GatedMlpProjection,
) -> LabResult<(Vec<Launch>, View, Vec<Arm>)> {
    let gate = kind == GatedMlpProjection::GateUp;
    let shape = family::shape(p.set, kind);
    let bytes = if gate {
        p.native.out.intermediate.bytes
    } else {
        p.native.out.down.bytes
    };
    let ops =
        |a: &mut Arena<'_>, tile: &Tile, tune: Option<Tune>| -> LabResult<(Vec<Launch>, View)> {
            let q = queue(a, &shape, tile)?;
            let out = a.filled(bytes, 0xa5)?;
            let n = p.native.out;
            let buffers = if gate {
                native_buffers(i, true, i.codes, i.scales, out, q, None)
            } else {
                native_buffers(i, false, n.codes, n.scales, out, q, None)
            };
            let launches = match tune {
                None => native::launches(&shape, tile, p.rows, &buffers, Variant::Serving),
                Some(t) => native::tune_launches(&shape, tile, p.rows, &buffers, t),
            }
            .map_err(|e| e.to_string())?;
            Ok((launches, out))
        };
    let served = super::tune_family::served(kind);
    let paired = super::tune_family::paired(kind);
    let (reference, reference_out) = ops(a, &served, None)?;
    let mut out = Vec::new();
    for tune in Tune::ALL {
        let (o, v) = ops(a, &served, Some(tune))?;
        out.push(Arm {
            label: tune.name().into(),
            ops: o,
            output: v,
            bitwise: tune == Tune::Lean,
        });
    }
    let (o, v) = ops(a, &paired, None)?;
    out.push(Arm {
        label: "paired".into(),
        ops: o,
        output: v,
        bitwise: true,
    });
    let (o, v) = ops(a, &paired, Some(Tune::Lean))?;
    out.push(Arm {
        label: "paired_lean".into(),
        ops: o,
        output: v,
        bitwise: true,
    });
    Ok((reference, reference_out, out))
}

/// Every arm of both projections on one prepared case.
fn case(
    runner: &mut Runner<'_>,
    a: &mut Arena<'_>,
    p: &Pipeline,
    i: &Inputs,
    label: &str,
) -> LabResult<(serde_json::Value, usize)> {
    // The served native gate/up and its quantizer give down its input.
    runner.run(a, &p.native.gate)?;
    let mut failures = 0;
    let mut kinds = Vec::new();
    for kind in family::KINDS {
        let (reference, reference_out, arms) = arms(a, p, i, kind)?;
        runner.run(a, &reference)?;
        let expected = a.read(reference_out)?;
        let mut records = Vec::new();
        for arm in &arms {
            let bitwise = if arm.bitwise {
                runner.run(a, &arm.ops)?;
                let same = a.read(arm.output)? == expected;
                failures += usize::from(!same);
                serde_json::json!(same)
            } else {
                serde_json::Value::Null
            };
            records.push(serde_json::json!({
                "arm": arm.label,
                "bitwise": bitwise,
                "timing": super::timing::graphs(runner, a, &reference, &arm.ops)?,
            }));
        }
        kinds.push(serde_json::json!({"kind": format!("{kind:?}"), "arms": records}));
    }
    Ok((
        serde_json::json!({"case": label, "rows": p.rows, "kinds": kinds}),
        failures,
    ))
}

pub fn run(session: &Session) -> LabResult<()> {
    let worker: u32 = std::env::var(WORKER)
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|_| "worker must be 0 or 1")?;
    if worker > 1 {
        return Err("worker must be 0 or 1".into());
    }
    let mut runner = Runner::new(session)?;
    let set = family::serving::target_set();
    let capacity = super::synthetic::CAPACITY;
    let mut executed = 0;
    let mut failed = 0;
    // The synthetic case of this worker.
    {
        let synthetic = super::synthetic::cases()
            .into_iter()
            .find(|c| c.set == set && c.skewed == (worker == 1))
            .ok_or("missing synthetic case")?;
        let mut a = Arena::new(Some(session));
        let host_w = [
            super::synthetic::weights(set.count as usize, 1152, 5120),
            super::synthetic::weights(set.count as usize, 5120, 576),
        ];
        let banks = [
            super::synthetic::bank(&mut a, &host_w[0])?,
            super::synthetic::bank(&mut a, &host_w[1])?,
        ];
        let s = super::synthetic::setup(&mut a, synthetic, &host_w, banks)?;
        let p = build(&mut a, set, capacity, capacity, &s.inputs)?;
        runner.run(&a, &s.prefix)?;
        runner.run(&a, &s.route)?;
        let (record, failures) = case(&mut runner, &mut a, &p, &s.inputs, &synthetic.label())?;
        println!("native_moe_tune {record}");
        executed += 1;
        failed += usize::from(failures > 0);
    }
    // The captured 8192-row chunk of this worker's layer and rank.
    let ready = json(Path::new(&std::env::var(READY).map_err(|e| e.to_string())?))?;
    let manifest_path = PathBuf::from(ready["path"].as_str().ok_or("missing manifest path")?);
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| e.to_string())?;
    if ready["sha256"] != kernel_build::digest(&manifest_bytes) {
        return Err("capture manifest digest mismatch".into());
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let banks = Banks::open(Path::new(&std::env::var(BANKS).map_err(|e| e.to_string())?))?;
    let layer = LAYERS[worker as usize];
    let c = captures(&manifest, worker)?
        .into_iter()
        .find(|c| c.layer() == layer && c.rank() == worker && c.rows() == capacity as usize)
        .ok_or("missing captured 8192-row case")?;
    let mut a = Arena::new(Some(session));
    let host_gate = banks.read(layer, worker, true)?;
    let host_down = banks.read(layer, worker, false)?;
    let bank = |a: &mut Arena<'_>, w: &[Vec<u8>; 2]| -> LabResult<QuantizedWeights<View>> {
        Ok(QuantizedWeights {
            codes: a.bytes(&w[0])?,
            local_scales: Some(a.bytes(&w[1])?),
            global_scales: None,
        })
    };
    let gate = bank(&mut a, &host_gate)?;
    let down = bank(&mut a, &host_down)?;
    let inputs = case_inputs(&mut a, &c, gate, down)?;
    let p = build(&mut a, set, capacity, capacity, &inputs)?;
    let (record, failures) = case(
        &mut runner,
        &mut a,
        &p,
        &inputs,
        &format!("captured-l{layer}-r{worker}"),
    )?;
    println!("native_moe_tune {record}");
    executed += 1;
    failed += usize::from(failures > 0);
    println!(
        "native_moe_tune_done {}",
        serde_json::json!({"worker": worker, "cases": executed, "failed": failed})
    );
    if executed != 2 || failed > 0 {
        return Err(format!("{failed} of {executed} tuning cases failed"));
    }
    Ok(())
}

#[test]
fn tune_cpu_plan() {
    // Every arm's launches name an entry of the tuning cubin or the served
    // bundle, keep the served compute arguments, and the paired tiles fit
    // two CTAs per SM: twice the served grid, with register bounds.
    let entries: Vec<_> = super::tune_family::cubins()
        .into_iter()
        .flat_map(|s| s.entries)
        .collect();
    let symbols: std::collections::BTreeSet<String> =
        entries.iter().map(|e| e.symbol.clone()).collect();
    assert_eq!(symbols.len(), entries.len());
    let set = family::serving::target_set();
    for kind in family::KINDS {
        let shape = family::shape(set, kind);
        let served = super::tune_family::served(kind);
        let paired = super::tune_family::paired(kind);
        assert_eq!(
            native::grid(&shape, &paired),
            2 * native::grid(&shape, &served)
        );
        let b = native::Buffers {
            input: 1,
            weights: 2,
            weight_scales: 3,
            activation_scales: 4,
            offsets: 5,
            sorted_rows: 6,
            padded_rows: 7,
            live_tokens: 8,
            route_weights: 9,
            output: 10,
            counts: 11,
            active: 12,
            scratch: 13,
            raw: 0,
        };
        let reference = native::launches(&shape, &served, 8192, &b, Variant::Serving).unwrap();
        for tune in Tune::ALL {
            let ops = native::tune_launches(&shape, &served, 8192, &b, tune).unwrap();
            assert!(symbols.contains(&ops[1].symbol), "{}", ops[1].symbol);
            assert_eq!(ops[0], reference[0]);
            assert_eq!(ops[1].arguments, reference[1].arguments);
            assert_eq!(ops[1].plan, reference[1].plan);
        }
        for ops in [
            native::launches(&shape, &paired, 8192, &b, Variant::Serving).unwrap(),
            native::tune_launches(&shape, &paired, 8192, &b, Tune::Lean).unwrap(),
        ] {
            for op in &ops {
                assert!(symbols.contains(&op.symbol), "{}", op.symbol);
            }
            let compute = entries.iter().find(|e| e.symbol == ops[1].symbol).unwrap();
            assert_eq!(compute.min_blocks, 2);
        }
    }
    println!(
        "native_moe_tune_plan {}",
        serde_json::json!({"entries": symbols.len()})
    );
}

#[test]
#[ignore = "requires the staged cubins, the capture named by CHOPINFER_NATIVE_MOE_READY and banks named by CHOPINFER_NATIVE_MOE_BANKS"]
fn native_moe_tune_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 16 << 30, run).unwrap();
}
