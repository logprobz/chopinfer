//! Synthetic operands through the serving input quantizer and router:
//! criteria 2 to 5 and 7, and with CHOPINFER_NATIVE_MOE_TIMING=1 paired
//! timing. CHOPINFER_NATIVE_MOE_MEMCHECK=1 runs criterion 6's reduced
//! matrix (no oracle, no timing) under compute-sanitizer.
use super::checks::{Host, qualify};
use super::pipeline::{Inputs, build, first_difference, snapshot};
use super::*;
use kernel_core::contract::MoeQuery;
use kernel_core::moe::{RouteBuffers, RouteMetadata};
use kernel_core::precision::buffers::QuantizedWeights;
use kernel_core::precision::quantizer::BlockQuantizeBuffers;
use kernel_core::router::RowBiasRouteBuffers;

pub const CAPACITY: u32 = 8192;

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub set: moe::Experts,
    pub skewed: bool,
}
impl Case {
    pub fn label(self) -> String {
        format!(
            "e{}-{}",
            self.set.count,
            if self.skewed { "skewed" } else { "balanced" }
        )
    }
    fn logits(self, rows: u32) -> Vec<u8> {
        (0..rows)
            .flat_map(|row| {
                let first = if self.skewed {
                    0
                } else {
                    row * self.set.top_k % self.set.count
                };
                (0..self.set.count).flat_map(move |e| {
                    (-(((e + self.set.count - first) % self.set.count) as f32) / 32.).to_le_bytes()
                })
            })
            .collect()
    }
}

/// E384 balanced and skewed, and the drafter's E128 bank skewed.
pub fn cases() -> Vec<Case> {
    let sets = moe::expert_sets(&FLASH);
    let target = family::serving::target_set();
    let mut cases = vec![
        Case {
            set: target,
            skewed: false,
        },
        Case {
            set: target,
            skewed: true,
        },
    ];
    cases.extend(
        sets.into_iter()
            .filter(|s| *s != target)
            .map(|set| Case { set, skewed: true }),
    );
    cases
}

pub fn memcheck() -> bool {
    std::env::var("CHOPINFER_NATIVE_MOE_MEMCHECK").as_deref() == Ok("1")
}
pub fn timing() -> bool {
    std::env::var("CHOPINFER_NATIVE_MOE_TIMING").as_deref() == Ok("1")
}
/// The oracle and invariance checks; the timing phase turns them off.
pub fn oracle_enabled() -> bool {
    !memcheck() && std::env::var("CHOPINFER_NATIVE_MOE_ORACLE").as_deref() != Ok("0")
}

/// Deterministic MXFP4 codes and scales 120..124, as the exact lab uses.
pub fn weights(experts: usize, rows: usize, k: usize) -> [Vec<u8>; 2] {
    let codes = (0..experts * rows * k / 2)
        .map(|i| {
            let x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
            (x ^ (x >> 16)) as u8
        })
        .collect();
    let scales = (0..experts * rows * k / 32)
        .map(|i| 120 + (i % 5) as u8)
        .collect();
    [codes, scales]
}

fn values(rows: u32) -> Vec<u8> {
    let mut seed = 0xc837_6251_aa19_46d3u64;
    let mut uniform = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 32) as u32 as f64 + 0.5) / (u32::MAX as f64 + 1.)
    };
    (0..u64::from(rows) * 5120)
        .flat_map(|i| {
            let normal = (-2. * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
            let value =
                normal * (0.5 + (i / 32 % 5) as f64 * 0.25) * if i % 997 == 0 { 6. } else { 1. };
            kernel_core::codec::f32_to_bf16(value as f32).to_le_bytes()
        })
        .collect()
}

/// The quantized input, the routes and both banks of `case`.
pub struct Setup {
    pub inputs: Inputs,
    pub prefix: Vec<Launch>,
    pub route: Vec<Launch>,
    pub ids: View,
    pub poison: Vec<View>,
    pub host: Host,
}

pub fn setup(
    a: &mut Arena<'_>,
    case: Case,
    weights_host: &[[Vec<u8>; 2]; 2],
    banks: [QuantizedWeights<View>; 2],
) -> LabResult<Setup> {
    let rows = CAPACITY;
    let set = case.set;
    let live = a.bytes(&rows.to_le_bytes())?;
    let bf16 = if a.session.is_some() {
        a.bytes(&values(rows))?
    } else {
        a.filled(u64::from(rows) * 5120 * 2, 0)?
    };
    let codes = a.filled(u64::from(rows) * 5120, 0)?;
    let scales = a.filled(u64::from(rows) * 160, 0)?;
    let status = a.filled(u64::from(rows) * 4, 0)?;
    let q = registration::quantizers(&FLASH, 4)?
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize)
        .block_quantize(&moe::quantizer(5120), Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let prefix = q
        .launches(
            &BlockQuantizeBuffers {
                input: bf16,
                output: codes,
                scales: Some(scales),
                live_rows: live,
                status,
                status_row_offset: 0,
            },
            View::EMPTY,
            Counts::rows(rows),
        )
        .map_err(|e| e.to_string())?;
    let logits = if a.session.is_some() {
        a.bytes(&case.logits(rows))?
    } else {
        a.filled(u64::from(rows) * u64::from(set.count) * 4, 0)?
    };
    let backend = registration::register(Sm120::new(), &FLASH, 4)?;
    let tile = serving_op(set, GatedMlpProjection::GateUp, rows, rows, false)?.route_tile();
    let route = backend
        .moe_row_bias_route(
            &moe::router(&FLASH, set),
            &MoeQuery::LocalRoutes {
                experts: set.count,
                local_experts: set.count,
                top_k: set.top_k,
                capacity: rows,
            },
            0,
            tile,
            usage(rows),
            Need::rows(rows),
        )
        .map_err(|e| e.to_string())?;
    let mut alloc = |name: &str| {
        a.filled(
            route
                .info()
                .bytes_of(name)
                .ok_or_else(|| format!("missing {name}"))?,
            0,
        )
    };
    let metadata = RouteMetadata {
        counts: Some(alloc("route.routes.counts")?),
        offsets: alloc("route.routes.offsets")?,
        sorted_rows: alloc("route.routes.sorted_rows")?,
        padded_rows: alloc("route.routes.padded_rows")?,
        active_experts: Some(alloc("route.routes.active_experts")?),
    };
    let rb = RowBiasRouteBuffers {
        route: RouteBuffers {
            logits,
            bias: Some(alloc("route.bias")?),
            ids: alloc("route.ids")?,
            weights: alloc("route.weights")?,
            live_rows: live,
            routes: metadata,
            tile_counts: None,
            tile_offsets: None,
            ticket: None,
        },
        alternate_bias: alloc("alternate_bias")?,
        bias_select: alloc("bias_select")?,
        scores: alloc("scores")?,
    };
    let route_ops = route
        .launches(&rb, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    let poison = vec![
        rb.route.ids,
        rb.route.weights,
        metadata.counts.unwrap(),
        metadata.offsets,
        metadata.sorted_rows,
        metadata.padded_rows,
        metadata.active_experts.unwrap(),
    ];
    Ok(Setup {
        inputs: Inputs {
            codes,
            scales,
            live,
            route_weights: rb.route.weights,
            routes: metadata,
            gate: banks[0],
            down: banks[1],
        },
        prefix,
        route: route_ops,
        ids: rb.route.ids,
        poison,
        host: Host {
            codes: Vec::new(),
            scales: Vec::new(),
            ids: Vec::new(),
            gate: weights_host[0].clone(),
            down: weights_host[1].clone(),
            shared: Vec::new(),
        },
    })
}

pub fn bank(a: &mut Arena<'_>, w: &[Vec<u8>; 2]) -> LabResult<QuantizedWeights<View>> {
    Ok(QuantizedWeights {
        codes: a.bytes(&w[0])?,
        local_scales: Some(a.bytes(&w[1])?),
        global_scales: None,
    })
}

/// The synthetic shared addend of a token: deterministic, of the routed
/// rows' order of magnitude.
fn shared_row(token: usize, width: usize, scale: f32) -> Vec<f32> {
    (0..width)
        .map(|c| {
            let x = ((token * width + c) as u32).wrapping_mul(2654435761);
            (((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

pub fn run(session: &Session) -> LabResult<()> {
    let mut runner = Runner::new(session)?;
    if memcheck() || std::env::var("CHOPINFER_NATIVE_MOE_PROBE").as_deref() == Ok("1") {
        let probe = super::probe::gpu(&mut runner)?;
        println!("native_moe_probe {probe}");
        if probe["failures"] != 0 {
            return Err("instruction probe failed".into());
        }
    }
    let mut executed = 0;
    let mut failed = 0;
    for case in cases() {
        let mut a = Arena::new(Some(session));
        let set = case.set;
        let host_w = [
            weights(set.count as usize, 1152, 5120),
            weights(set.count as usize, 5120, 576),
        ];
        let banks = [bank(&mut a, &host_w[0])?, bank(&mut a, &host_w[1])?];
        let mut s = setup(&mut a, case, &host_w, banks)?;
        let p = build(&mut a, set, CAPACITY, CAPACITY, &s.inputs)?;
        runner.run(&a, &s.prefix)?;
        let lives: &[u32] = if memcheck() {
            &[1, 65, 513, 2048]
        } else {
            &[CAPACITY]
        };
        for &live in lives {
            a.write(s.inputs.live, &live.to_le_bytes())?;
            runner.run(&a, &s.route)?;
            s.host.codes = a.read(s.inputs.codes)?;
            s.host.scales = a.read(s.inputs.scales)?;
            s.host.ids = ints(&a.read(s.ids)?);
            let width = family::shape(set, GatedMlpProjection::Down).n as usize;
            s.host.shared = samples(live as usize)
                .iter()
                .map(|&t| shared_row(t, width, 1.0 / 64.0))
                .collect();
            let report = qualify(
                &mut runner,
                &mut a,
                &p,
                &s.inputs,
                &s.host,
                live as usize,
                oracle_enabled(),
            )?;
            let mut json = report.json;
            let mut passed = report.passed;
            if oracle_enabled() {
                let (inv, ok) = invariance(&mut runner, &a, &p, &s)?;
                json["invariance"] = inv;
                passed &= ok;
            }
            if !memcheck() && timing() {
                json["timing"] = serde_json::json!({
                    "gate_up": super::timing::graphs(&mut runner, &a, &p.exact.gate, &p.native.gate)?,
                    "down": super::timing::graphs(&mut runner, &a, &p.exact.down, &p.native.down)?,
                    "both": super::timing::graphs(&mut runner, &a, &p.exact.launches(), &p.native.launches())?,
                });
            }
            json["case"] = serde_json::json!(case.label());
            json["experts"] = serde_json::json!(set.count);
            json["top_k"] = serde_json::json!(set.top_k);
            json["memcheck"] = serde_json::json!(memcheck());
            json["passed"] = serde_json::json!(passed);
            println!("native_moe_synthetic {json}");
            executed += 1;
            failed += usize::from(!passed);
        }
    }
    println!(
        "native_moe_synthetic_done {}",
        serde_json::json!({"cases": executed, "failed": failed, "memcheck": memcheck()})
    );
    if failed > 0 {
        return Err(format!("{failed} synthetic cases failed"));
    }
    Ok(())
}

/// Criterion 4: live-row prefixes, graph replay and poisoned repeats of the
/// native arm (route included) give the full step's bits.
fn invariance(
    runner: &mut Runner<'_>,
    a: &Arena<'_>,
    p: &super::pipeline::Pipeline,
    s: &Setup,
) -> LabResult<(serde_json::Value, bool)> {
    let top = p.set.top_k as usize;
    let out = p.native.out;
    let views = out.all();
    let mut ops = s.route.clone();
    ops.extend(p.native.launches());
    let poison = |a: &Arena<'_>| -> LabResult<()> {
        a.poison(&views)?;
        a.poison(&s.poison)?;
        a.poison(&p.native.workspaces)
    };
    let set_live = |a: &Arena<'_>, live: u32| a.write(s.inputs.live, &live.to_le_bytes());
    set_live(a, CAPACITY)?;
    poison(a)?;
    runner.run(a, &ops)?;
    let full = snapshot(a, &views)?;
    let shape_g = family::shape(p.set, GatedMlpProjection::GateUp);
    let shape_d = family::shape(p.set, GatedMlpProjection::Down);
    let mut results = Vec::new();
    let mut ok = true;
    for live in [4097_u32, 65, 1] {
        set_live(a, live)?;
        poison(a)?;
        runner.run(a, &ops)?;
        let got = snapshot(a, &views)?;
        let rows = live as usize * top;
        let differences: Vec<Option<usize>> = (0..4)
            .map(|v| first_difference(&got[v], &full[v], rows * row_bytes(&shape_g, &shape_d, v)))
            .collect();
        let equal = differences.iter().all(Option::is_none);
        ok &= equal;
        results.push(serde_json::json!({"live": live, "bitwise_prefix": equal, "first_difference": differences}));
    }
    set_live(a, CAPACITY)?;
    // Graph replay and a poisoned eager repeat.
    runner.ensure(&ops)?;
    let graph = runner
        .session
        .capture(|stream| runner.issue(a, &ops, stream))?;
    poison(a)?;
    graph
        .launch(&runner.session.stream)
        .map_err(|e| e.to_string())?;
    runner
        .session
        .stream
        .synchronize()
        .map_err(|e| e.to_string())?;
    let graphed = snapshot(a, &views)?;
    poison(a)?;
    runner.run(a, &ops)?;
    let repeated = snapshot(a, &views)?;
    let graph_equal = graphed == full;
    let repeat_equal = repeated == full;
    ok &= graph_equal && repeat_equal;
    Ok((
        serde_json::json!({"prefixes": results, "graph_bitwise": graph_equal, "repeat_bitwise": repeat_equal}),
        ok,
    ))
}

fn row_bytes(g: &Shape, d: &Shape, view: usize) -> usize {
    match view {
        0 => g.n as usize * 2,
        1 => g.n as usize,
        2 => g.n as usize / 32,
        _ => d.n as usize * 4,
    }
}

#[test]
fn cpu_plan() {
    // Every synthetic case builds both arms and the lab launches name entries
    // the lab cubin declares; the target bank's wrapper matches the lab's.
    let symbols: std::collections::BTreeSet<String> = family::cubins()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    for case in cases() {
        let mut a = Arena::new(None);
        let set = case.set;
        let host_w = [vec![], vec![]];
        let banks = [
            QuantizedWeights {
                codes: a.filled(u64::from(set.count) * 1152 * 2560, 0).unwrap(),
                local_scales: Some(a.filled(u64::from(set.count) * 1152 * 160, 0).unwrap()),
                global_scales: None,
            },
            QuantizedWeights {
                codes: a.filled(u64::from(set.count) * 5120 * 288, 0).unwrap(),
                local_scales: Some(a.filled(u64::from(set.count) * 5120 * 18, 0).unwrap()),
                global_scales: None,
            },
        ];
        let s = setup(&mut a, case, &[host_w.clone(), host_w], banks).unwrap();
        for (capacity, rows) in [(CAPACITY, CAPACITY), (CAPACITY, 65), (4096, 4096)] {
            let p = build(&mut a, set, capacity, rows, &s.inputs).unwrap();
            for op in p.native.launches() {
                op.plan.validate_geometry().unwrap();
                assert!(op.plan.dynamic_shared_bytes <= native::MAX_SHARED_BYTES);
                if op.symbol.starts_with("chop_moe_native_") {
                    assert!(symbols.contains(&op.symbol), "{}", op.symbol);
                    // Target-bank serving tiles run from the bundle.
                    assert_eq!(
                        Runner::serving(&op.symbol),
                        set == family::serving::target_set()
                    );
                }
            }
            assert!(
                p.native
                    .gate
                    .iter()
                    .any(|o| o.symbol.contains("_moe_ceil_"))
            );
            assert_eq!(p.native.gate.len(), 3);
            assert_eq!(p.native.down.len(), 2);
        }
    }
    println!(
        "native_moe_dry_run {}",
        serde_json::json!({"cases": cases().iter().map(|c| c.label()).collect::<Vec<_>>(), "lab_symbols": symbols.len(), "cubin": family::CUBIN})
    );
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_NATIVE_MOE_CUBINS"]
fn native_moe_synthetic_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 16 << 30, run).unwrap();
}
