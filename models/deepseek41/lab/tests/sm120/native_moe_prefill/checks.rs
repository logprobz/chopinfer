//! Criteria 2, 3, 5 and 7 on one pipeline: the raw gate/up sums and the
//! down outputs against the f64 oracle on sampled tokens, the serving
//! entries against the lab's raw entry and every tile variant, the
//! omit-first control, and the report-only comparison with the exact path.
use super::oracle::{self, Stage};
use super::pipeline::{Inputs, Pipeline, first_difference, native_buffers, queue};
use super::*;

/// Host copies the oracle reads.
pub struct Host {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub ids: Vec<i32>,
    pub gate: [Vec<u8>; 2],
    pub down: [Vec<u8>; 2],
    /// The shared expert's F32 addend of each sampled token, for the
    /// report-only routed sum.
    pub shared: Vec<Vec<f32>>,
}

/// The oracle rows of `routes` against packed weights: per route, every
/// output column's (value, sum of absolute products).
fn oracle_rows(
    routes: &[(usize, usize)],
    input: impl Fn(usize) -> Vec<f64> + Sync,
    weights: &[Vec<u8>; 2],
    columns: usize,
    k: usize,
) -> Vec<Vec<(f64, f64)>> {
    let e2m1 = oracle::e2m1_table();
    let threads = 4;
    let chunk = routes.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = routes
            .chunks(chunk)
            .map(|part| {
                let input = &input;
                scope.spawn(move || {
                    part.iter()
                        .map(|&(row, expert)| {
                            let x = input(row);
                            (0..columns)
                                .map(|c| {
                                    let w = expert * columns + c;
                                    oracle::dot(
                                        &x,
                                        &weights[0][w * k / 2..(w + 1) * k / 2],
                                        &weights[1][w * k / 32..(w + 1) * k / 32],
                                        &e2m1,
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
}

fn stage(rows: &[Vec<(f64, f64)>], outputs: &[f32], routes: &[(usize, usize)], k: usize) -> Stage {
    let width = rows[0].len();
    let mut s = Stage::default();
    for (r, &(route, _)) in rows.iter().zip(routes) {
        s.add(oracle::check_row(
            &outputs[route * width..(route + 1) * width],
            r,
            k,
        ));
    }
    s
}

pub struct Report {
    pub json: serde_json::Value,
    pub passed: bool,
}

/// Every check of `p` at its live rows. `variants` adds the other tiles.
pub fn qualify(
    runner: &mut Runner<'_>,
    a: &mut Arena<'_>,
    p: &Pipeline,
    i: &Inputs,
    host: &Host,
    live: usize,
    oracle_checks: bool,
) -> LabResult<Report> {
    let set = p.set;
    let top = set.top_k as usize;
    let gate_shape = family::shape(set, GatedMlpProjection::GateUp);
    let down_shape = family::shape(set, GatedMlpProjection::Down);
    let (n, kg) = (gate_shape.n as usize, gate_shape.k as usize);
    let (nd, kd) = (down_shape.n as usize, down_shape.k as usize);
    let serving_tiles = family::KINDS.map(family::serving::tile);
    let mut failures: Vec<String> = Vec::new();
    // Exact and native arms, then the lab's raw gate/up and native down on
    // the exact codes, which share the exact down's inputs.
    runner.run(a, &p.exact.launches())?;
    runner.run(a, &p.native.launches())?;
    let routes = (live * top) as u64;
    let raw = a.filled(routes.max(1) * 2 * n as u64 * 4, 255)?;
    let raw_inter = a.filled(p.native.out.intermediate.bytes, 255)?;
    let down_x = a.filled(p.exact.out.down.bytes, 255)?;
    let q = queue(a, &gate_shape, &serving_tiles[0])?;
    let qd = queue(a, &down_shape, &serving_tiles[1])?;
    let ex = p.exact.out;
    let raw_ops = native_ops(
        &gate_shape,
        &serving_tiles[0],
        p.rows,
        &native_buffers(i, true, i.codes, i.scales, raw_inter, q, Some(raw)),
        Variant::Raw,
    )?;
    let down_ops = native_ops(
        &down_shape,
        &serving_tiles[1],
        p.rows,
        &native_buffers(i, false, ex.codes, ex.scales, down_x, qd, None),
        Variant::Serving,
    )?;
    runner.run(a, &raw_ops)?;
    runner.run(a, &down_ops)?;
    // Criterion 3: the serving gate/up entry and the raw entry agree.
    let inter_bytes = live * top * n * 2;
    let native_inter = a.read(p.native.out.intermediate)?;
    let raw_inter_host = a.read(raw_inter)?;
    if let Some(at) = first_difference(&native_inter, &raw_inter_host, inter_bytes) {
        failures.push(format!(
            "serving gate/up differs from the raw entry at byte {at}"
        ));
    }
    let raw_host = floats(&a.read(raw)?);
    let down_x_host = a.read(down_x)?;
    // Every other tile gives the same bits.
    let mut variants = Vec::new();
    for kind in family::KINDS {
        for tile in family::tiles(kind) {
            if *tile == family::serving::tile(kind) {
                continue;
            }
            let gate = kind == GatedMlpProjection::GateUp;
            let shape = if gate { gate_shape } else { down_shape };
            let vq = queue(a, &shape, tile)?;
            let (ops, view, reference): (_, View, &[u8]) = if gate {
                let vraw = a.filled(raw.bytes, 255)?;
                let vinter = a.filled(raw_inter.bytes, 255)?;
                (
                    native_ops(
                        &shape,
                        tile,
                        p.rows,
                        &native_buffers(i, true, i.codes, i.scales, vinter, vq, Some(vraw)),
                        Variant::Raw,
                    )?,
                    vraw,
                    &[],
                )
            } else {
                let vdown = a.filled(down_x.bytes, 255)?;
                (
                    native_ops(
                        &shape,
                        tile,
                        p.rows,
                        &native_buffers(i, false, ex.codes, ex.scales, vdown, vq, None),
                        Variant::Serving,
                    )?,
                    vdown,
                    &down_x_host,
                )
            };
            runner.run(a, &ops)?;
            let got = a.read(view)?;
            let (expected, bytes): (Vec<u8>, usize) = if gate {
                (
                    raw_host.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    live * top * 2 * n * 4,
                )
            } else {
                (reference.to_vec(), live * top * nd * 4)
            };
            let difference = first_difference(&got, &expected, bytes);
            if let Some(at) = difference {
                failures.push(format!(
                    "{kind:?} tile {tile:?} differs from the serving tile at byte {at}"
                ));
            }
            variants.push(serde_json::json!({"kind":format!("{kind:?}"),"tile":format!("{tile:?}"),"bitwise":difference.is_none()}));
        }
    }
    // Criterion 5's control output: native down without its first chain.
    let omit = a.filled(down_x.bytes, 255)?;
    let omit_ops = native_ops(
        &down_shape,
        &serving_tiles[1],
        p.rows,
        &native_buffers(i, false, ex.codes, ex.scales, omit, qd, None),
        Variant::OmitFirst,
    )?;
    runner.run(a, &omit_ops)?;
    let mut json = serde_json::json!({
        "live": live,
        "variants": variants,
        "serving_gate_matches_raw": first_difference(&native_inter, &raw_inter_host, inter_bytes).is_none(),
    });
    if oracle_checks {
        let sampled = samples(live);
        let routes: Vec<(usize, usize)> = sampled
            .iter()
            .flat_map(|&t| (0..top).map(move |k| t * top + k))
            .map(|r| (r, host.ids[r] as usize))
            .collect();
        let e4m3 = oracle::e4m3_table();
        // Gate/up: the token's input row against gate and up weight rows.
        let gate_rows = oracle_rows(
            &routes,
            |r| {
                let t = r / top;
                oracle::activation(
                    &host.codes[t * kg..(t + 1) * kg],
                    &host.scales[t * kg / 32..(t + 1) * kg / 32],
                    &e4m3,
                )
            },
            &host.gate,
            2 * n,
            kg,
        );
        let gate_stage = stage(&gate_rows, &raw_host, &routes, kg);
        let ex_codes = a.read(ex.codes)?;
        let ex_scales = a.read(ex.scales)?;
        let down_rows = oracle_rows(
            &routes,
            |r| {
                oracle::activation(
                    &ex_codes[r * kd..(r + 1) * kd],
                    &ex_scales[r * kd / 32..(r + 1) * kd / 32],
                    &e4m3,
                )
            },
            &host.down,
            nd,
            kd,
        );
        let down_native = floats(&down_x_host);
        let down_exact = floats(&a.read(ex.down)?);
        let down_omit = floats(&a.read(omit)?);
        let native_stage = stage(&down_rows, &down_native, &routes, kd);
        let exact_stage = stage(&down_rows, &down_exact, &routes, kd);
        let omit_stage = stage(&down_rows, &down_omit, &routes, kd);
        // The full native chain's down, against the oracle of its own codes.
        let nat_codes = a.read(p.native.out.codes)?;
        let nat_scales = a.read(p.native.out.scales)?;
        let chain_rows = oracle_rows(
            &routes,
            |r| {
                oracle::activation(
                    &nat_codes[r * kd..(r + 1) * kd],
                    &nat_scales[r * kd / 32..(r + 1) * kd / 32],
                    &e4m3,
                )
            },
            &host.down,
            nd,
            kd,
        );
        let chain_native = floats(&a.read(p.native.out.down)?);
        let chain_stage = stage(&chain_rows, &chain_native, &routes, kd);
        // Report only: the ordered top-k sum with the shared addend.
        let mut sum_native = Stage::default();
        let mut sum_exact = Stage::default();
        for (s, &t) in sampled.iter().enumerate() {
            let shared = &host.shared[s];
            let row = |v: &[f32], k: usize| -> Vec<f32> {
                v[(t * top + k) * nd..(t * top + k + 1) * nd].to_vec()
            };
            let native_rows: Vec<Vec<f32>> = (0..top).map(|k| row(&down_native, k)).collect();
            let exact_rows: Vec<Vec<f32>> = (0..top).map(|k| row(&down_exact, k)).collect();
            let reference: Vec<(f64, f64)> = (0..nd)
                .map(|c| {
                    let (mut v, mut abs) = (f64::from(shared[c]), f64::from(shared[c]).abs());
                    for k in 0..top {
                        v += down_rows[s * top + k][c].0;
                        abs += down_rows[s * top + k][c].1;
                    }
                    (v, abs)
                })
                .collect();
            fn refs(rows: &[Vec<f32>]) -> Vec<&[f32]> {
                rows.iter().map(|r| r.as_slice()).collect()
            }
            sum_native.add(oracle::check_row(
                &oracle::ordered_sum(&refs(&native_rows), shared),
                &reference,
                kd,
            ));
            sum_exact.add(oracle::check_row(
                &oracle::ordered_sum(&refs(&exact_rows), shared),
                &reference,
                kd,
            ));
        }
        for (name, s) in [
            ("gate_up", &gate_stage),
            ("down", &native_stage),
            ("chain_down", &chain_stage),
        ] {
            if !s.passes() {
                failures.push(format!(
                    "{name}: {} bound failures, {} cosine failures",
                    s.failures, s.cosine_failures
                ));
            }
        }
        if exact_stage.failures > 0 {
            failures.push("the exact down fails its own bound: the oracle is wrong".into());
        }
        if omit_stage.failures == 0 {
            failures.push("omit-first control escaped the bound".into());
        }
        json["oracle"] = serde_json::json!({
            "sampled_tokens": sampled.len(),
            "routes": routes.len(),
            "gate_up": gate_stage.json(),
            "down": native_stage.json(),
            "chain_down": chain_stage.json(),
            "exact_down": exact_stage.json(),
            "omit_first_rejected": omit_stage.failures > 0,
            "omit_first": omit_stage.json(),
            "report_only": {
                "down_vs_exact": oracle::versus(&native_stage, &exact_stage),
                "ordered_sum_vs_exact": oracle::versus(&sum_native, &sum_exact),
            },
        });
    }
    json["failures"] = serde_json::json!(failures);
    Ok(Report {
        passed: failures.is_empty(),
        json,
    })
}
