//! Compare the active schedule with the locked column schedule, including
//! routing, clamped gate/up, BF16 rounding, ceil quantization and down.
use super::moe::registration;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::MoeQuery;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe::{RouteBuffers, RouteMetadata};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::router::RowBiasRouteBuffers;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{geometry::FLASH, moe};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use std::collections::BTreeMap;

struct Arena<'a> {
    session: Option<&'a Session>,
    next: u64,
    buffers: BTreeMap<u64, DeviceBuffer>,
}
impl Arena<'_> {
    fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        self.next += 1;
        if let Some(session) = self.session {
            let b = session.allocate(bytes.len())?;
            b.copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            self.buffers.insert(self.next, b);
        }
        Ok(View {
            handle: self.next,
            bytes: bytes.len() as u64,
        })
    }
    fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        if self.session.is_none() {
            self.next += 1;
            return Ok(View {
                handle: self.next,
                bytes,
            });
        }
        self.bytes(&vec![value; bytes as usize])
    }
    fn read(&self, v: View) -> LabResult<Vec<u8>> {
        let mut out = vec![0; v.bytes as usize];
        self.buffers[&v.handle]
            .copy_to_bytes(&mut out, &self.session.unwrap().stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
}

struct Arm {
    launches: Vec<Launch>,
    outputs: Vec<View>,
    poison: Vec<View>,
    status: View,
}

#[derive(Clone, Copy)]
struct Inputs {
    input: View,
    scales: View,
    live: View,
    logits: View,
    weights: [QuantizedWeights<View>; 2],
}

fn arm(a: &mut Arena<'_>, rows: u32, active: bool, inputs: Inputs) -> LabResult<Arm> {
    let Inputs {
        input,
        scales,
        live,
        logits,
        weights: [gate_weights, down_weights],
    } = inputs;
    let set = moe::expert_sets(&FLASH)[0];
    let mut backend = registration::quantizers(&FLASH, 4)?
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize);
    let entries = if active {
        registration::active_projections(&FLASH, 4)?
    } else {
        registration::projections(&FLASH, 4)?
    };
    // Hold exactly the entry under test. The prefill control keeps 64-row
    // tiles while the candidate uses the decode tile at every comparison.
    let tile = if active || rows <= 32 { 16 } else { 64 };
    let capacity = if active && rows > 32 {
        192
    } else if tile == 64 {
        8192
    } else {
        32
    };
    for e in entries.into_iter().filter(|e| {
        e.row_tile == tile
            && e.tokens_capacity == capacity
            && e.query.projection.gemm.grouped.unwrap().experts == set.count
    }) {
        backend = backend.with_quantized_chained_projection(e);
    }
    for e in registration::routes(&FLASH)
        .into_iter()
        .chain(registration::active_routes(&FLASH))
    {
        backend = backend.with_row_bias_route(e);
    }
    let usage = OperationUse {
        phase: if rows > 32 {
            ExecutionPhase::Verify
        } else {
            ExecutionPhase::Decode
        },
        capacity: rows,
        graph: true,
    };
    let need = Need::rows(rows);
    let gate = backend
        .chained_mlp_projection(
            &moe::projection(&FLASH, 4, set, GatedMlpProjection::GateUp, rows)?,
            usage,
            need,
        )
        .map_err(|e| e.to_string())?;
    let down = backend
        .chained_mlp_projection(
            &moe::projection(&FLASH, 4, set, GatedMlpProjection::Down, rows)?,
            usage,
            need,
        )
        .map_err(|e| e.to_string())?;
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
            usage,
            need,
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
    let mut alloc = |name: &str| {
        a.filled(
            gate.info()
                .bytes_of(name)
                .ok_or_else(|| format!("missing {name}"))?,
            0,
        )
    };
    let intermediate = alloc("output")?;
    let codes = alloc("scratch.output_codes")?;
    let local_scales = alloc("scratch.output_scales")?;
    let status = alloc("scratch.quantizer_status")?;
    let output = a.filled(down.info().bytes_of("output").unwrap(), 0)?;
    let routing = Some(ProjectionRouting {
        weights: rb.route.weights,
        routes: metadata,
    });
    let scratch = GatedMlpScratch {
        quantizer_status: Some(status),
        quantizer_status_row_offset: 0,
        partial: None,
        output_codes: Some(codes),
        output_scales: Some(local_scales),
        unused: live,
    };
    let gate_b = GatedMlpBuffers {
        activation: input,
        activation_global: None,
        activation_scales: Some(scales),
        live_tokens: live,
        output: intermediate,
        weights: gate_weights,
        routing,
        scratch,
    };
    let down_b = GatedMlpBuffers {
        activation: codes,
        activation_scales: Some(local_scales),
        output,
        weights: down_weights,
        ..gate_b
    };
    let mut launches = route
        .launches(&rb, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    launches.extend(
        gate.launches(&gate_b, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?,
    );
    launches.extend(
        down.launches(&down_b, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?,
    );
    let outputs = vec![
        rb.route.ids,
        rb.route.weights,
        intermediate,
        codes,
        local_scales,
        output,
    ];
    let mut poison = outputs.clone();
    poison.extend([
        metadata.counts.unwrap(),
        metadata.offsets,
        metadata.sorted_rows,
        metadata.padded_rows,
        metadata.active_experts.unwrap(),
    ]);
    Ok(Arm {
        launches,
        outputs,
        poison,
        status,
    })
}

fn weights(a: &mut Arena<'_>, n: usize, k: usize) -> LabResult<QuantizedWeights<View>> {
    let count = 384 * n * k / 2;
    let codes = (0..count)
        .map(|i| {
            let x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
            (x ^ (x >> 16)) as u8
        })
        .collect::<Vec<_>>();
    let codes = a.bytes(&codes)?;
    let scales = (0..384 * n * k / 32)
        .map(|i| 120 + (i % 5) as u8)
        .collect::<Vec<_>>();
    Ok(QuantizedWeights {
        codes,
        local_scales: Some(a.bytes(&scales)?),
        global_scales: None,
    })
}

fn compare(session: &Session, rows: u32) -> LabResult<()> {
    let mut a = Arena {
        session: Some(session),
        next: 0,
        buffers: BTreeMap::new(),
    };
    let live = a.bytes(&rows.to_le_bytes())?;
    let input = a.bytes(
        &(0..rows as usize * 5120)
            .map(|i| ((i * 13 % 112) as u8) | if i % 3 == 0 { 128 } else { 0 })
            .collect::<Vec<_>>(),
    )?;
    let scales = a.bytes(
        &(0..rows as usize * 160)
            .map(|i| 122 + (i % 5) as u8)
            .collect::<Vec<_>>(),
    )?;
    let logits = a.bytes(
        &(0..rows * 384)
            .flat_map(|i| {
                // Repeat six experts in some rows and spread the others. Repeated
                // experts cross the 16-row tile boundary at the larger buckets.
                let row = i / 384;
                let selected = if row % 3 == 0 { 0 } else { row * 37 % 384 };
                let score = ((i % 384 + 384 - selected) % 384) as f32;
                (-score / 32.0).to_le_bytes()
            })
            .collect::<Vec<_>>(),
    )?;
    let gw = weights(&mut a, 1152, 5120)?;
    let dw = weights(&mut a, 5120, 576)?;
    let arms = [
        arm(
            &mut a,
            rows,
            false,
            Inputs {
                input,
                scales,
                live,
                logits,
                weights: [gw, dw],
            },
        )?,
        arm(
            &mut a,
            rows,
            true,
            Inputs {
                input,
                scales,
                live,
                logits,
                weights: [gw, dw],
            },
        )?,
    ];
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    let mut functions = BTreeMap::<String, Function>::new();
    for op in arms.iter().flat_map(|arm| &arm.launches) {
        if !functions.contains_key(&op.symbol) {
            let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                .ok_or_else(|| format!("missing {}", op.symbol))?;
            functions.insert(
                op.symbol.clone(),
                session
                    .load_cubin(bytes)?
                    .function(&op.symbol)
                    .map_err(|e| e.to_string())?,
            );
        }
    }
    let run = |arm: usize, stream: &Stream| -> LabResult<()> {
        for op in &arms[arm].launches {
            launch(
                &session.context,
                &functions[&op.symbol],
                stream,
                op.plan,
                &op.arguments,
                &bindings,
            )?;
        }
        Ok(())
    };
    // Capture once at the bucket size. Replay with changing live rows and
    // poisoned metadata to catch stale active lists and out-of-range writes.
    let graphs = [
        session.capture(|s| run(0, s))?,
        session.capture(|s| run(1, s))?,
    ];
    for live_rows in [rows, 1, 0, rows] {
        a.buffers[&live.handle]
            .copy_from_bytes(&live_rows.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        for (index, arm) in arms.iter().enumerate() {
            for &v in &arm.poison {
                a.buffers[&v.handle]
                    .copy_from_bytes(&vec![0xa5; v.bytes as usize], &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            graphs[index]
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        for (index, (&old, &new)) in arms[0].outputs.iter().zip(&arms[1].outputs).enumerate() {
            let left = a.read(old)?;
            let right = a.read(new)?;
            if left != right {
                let at = left.iter().zip(&right).position(|(x, y)| x != y).unwrap();
                return Err(format!(
                    "rows {rows}, live {live_rows}, output {index}, byte {at}: old {} new {}",
                    left[at], right[at]
                ));
            }
            let tail = live_rows as usize * left.len() / rows as usize;
            if left[tail..].iter().any(|&b| b != 0xa5) {
                return Err("inactive output rows were written".into());
            }
        }
        for arm in &arms {
            if a.read(arm.status)?.iter().any(|&b| b != 0) {
                return Err("quantizer status is nonzero".into());
            }
        }
    }
    let timing = timed_pair(session, |arm, stream| {
        graphs[arm].launch(stream).map_err(|e| e.to_string())
    })?;
    let mut kernel_times = BTreeMap::new();
    for name in ["gateup", "down"] {
        let selected = arms
            .iter()
            .map(|arm| {
                arm.launches
                    .iter()
                    .find(|op| op.symbol.contains(&format!("before_{name}_")))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let kernels = [0, 1]
            .map(|arm| {
                session.capture(|stream| {
                    let op = selected[arm];
                    launch(
                        &session.context,
                        &functions[&op.symbol],
                        stream,
                        op.plan,
                        &op.arguments,
                        &bindings,
                    )
                })
            })
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let times = timed_pair(session, |arm, stream| {
            kernels[arm].launch(stream).map_err(|e| e.to_string())
        })?;
        kernel_times.insert(name, times);
    }
    println!(
        "rows={rows} bitwise=passed warm_old_ms={} warm_active_ms={} kernels={kernel_times:?}",
        timing[0], timing[1]
    );
    session.receipt(
        "moe-active-bitwise",
        [
            ("rows".into(), serde_json::json!(rows)),
            ("cache".into(), serde_json::json!("warm")),
            ("pipeline_ms".into(), serde_json::json!(timing)),
            ("kernel_ms".into(), serde_json::json!(kernel_times)),
            ("bitwise".into(), serde_json::json!(true)),
        ]
        .into(),
    )?;
    Ok(())
}

/// Alternate old/new and new/old after two warmups. CUDA events time
/// captured work; no receipt implies cold-cache or end-to-end serving time.
fn timed_pair(
    session: &Session,
    work: impl Fn(usize, &Stream) -> LabResult<()>,
) -> LabResult<[f32; 2]> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    let mut samples = [Vec::new(), Vec::new()];
    for round in 0..10 {
        for step in 0..2 {
            let arm = (round + step) % 2;
            start.record(&session.stream).map_err(|e| e.to_string())?;
            work(arm, &session.stream)?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let elapsed = start.elapsed_ms(&end).map_err(|e| e.to_string())?;
            if round >= 2 {
                samples[arm].push(elapsed);
            }
        }
    }
    Ok(samples.map(|mut s| {
        s.sort_by(f32::total_cmp);
        (s[3] + s[4]) * 0.5
    }))
}

fn gpu(rows: u32) {
    kernel_harness::gpu_test(6 << 30, |s| compare(s, rows)).unwrap();
}
#[test]
fn active_moe_rows_1_gpu() {
    gpu(1);
}
#[test]
fn active_moe_rows_8_gpu() {
    gpu(8);
}
#[test]
fn active_moe_rows_32_gpu() {
    gpu(32);
}
#[test]
fn active_moe_prefill_128_gpu() {
    gpu(128);
}

#[test]
fn active_moe_cpu_dry_run() {
    for rows in [1, 8, 32, 128] {
        let mut a = Arena {
            session: None,
            next: 0,
            buffers: BTreeMap::new(),
        };
        let input = a.filled(u64::from(rows) * 5120, 0).unwrap();
        let scales = a.filled(u64::from(rows) * 160, 0).unwrap();
        let live = a.filled(4, 0).unwrap();
        let logits = a.filled(u64::from(rows) * 384 * 4, 0).unwrap();
        let mut weight = |n: u64, k: u64| QuantizedWeights {
            codes: a.filled(384 * n * k / 2, 0).unwrap(),
            local_scales: Some(a.filled(384 * n * k / 32, 0).unwrap()),
            global_scales: None,
        };
        let inputs = Inputs {
            input,
            scales,
            live,
            logits,
            weights: [weight(1152, 5120), weight(5120, 576)],
        };
        for active in [false, true] {
            let arm = arm(&mut a, rows, active, inputs).unwrap();
            assert!(arm.launches.len() >= 4);
            for op in arm.launches {
                assert!(
                    bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol).is_some(),
                    "{}",
                    op.symbol
                );
                op.plan.validate_geometry().unwrap();
            }
        }
    }
}
