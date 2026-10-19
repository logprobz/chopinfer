//! Compare independent exact pipelines with the pinned serving row selection.
#[path = "../../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;
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

mod compare;
#[path = "../../../sm120/exact_prefill.rs"]
mod family;
mod tests;
mod timing;
mod vector;
use compare::*;
#[derive(Clone, Copy, Debug)]
struct Case {
    rows: u32,
    set: moe::Experts,
    skewed: bool,
}
impl Case {
    fn label(self) -> String {
        format!(
            "e{}-r{}-{}",
            self.set.count,
            self.rows,
            if self.skewed { "skewed" } else { "balanced" }
        )
    }
    fn logits(self) -> Vec<u8> {
        (0..self.rows)
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
fn cases() -> Vec<Case> {
    moe::expert_sets(&FLASH)
        .into_iter()
        .flat_map(|set| {
            [512, 2048, 4096, 8192]
                .into_iter()
                .flat_map(move |rows| [false, true].map(|skewed| Case { rows, set, skewed }))
        })
        .collect()
}
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
    sum: Vec<Launch>,
    active_list: View,
    route_counts: View,
    outputs: Vec<View>,
    poison: Vec<View>,
    status: View,
    pointers: [[u64; 10]; 2],
    clocks: Vec<View>,
    clock_launches: Vec<Launch>,
}
#[derive(Clone, Copy)]
struct Inputs {
    input: View,
    scales: View,
    live: View,
    logits: View,
    weights: [QuantizedWeights<View>; 2],
}

fn arm(a: &mut Arena<'_>, case: Case, inputs: Inputs) -> LabResult<Arm> {
    let rows = case.rows;
    let Inputs {
        input,
        scales,
        live,
        logits,
        weights: [gate_weights, down_weights],
    } = inputs;
    let set = case.set;
    let backend = registration::register(Sm120::new(), &FLASH, 4)?;
    let tile = 64;
    // Use the exact serving bucket, rather than the old fixture's blanket
    // prefill capacity of 8192, when selecting measured grouped winners.
    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: rows,
        graph: false,
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
    let bytes = gate.info().workspace.max(down.info().workspace);
    let workspace = if bytes == 0 {
        View::EMPTY
    } else {
        a.filled(bytes, 0)?
    };
    let mut launches = route
        .launches(&rb, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    launches.extend(
        gate.launches(&gate_b, workspace, Counts::rows(rows))
            .map_err(|e| format!("{} serving gate: {e}", case.label()))?,
    );
    launches.extend(
        down.launches(&down_b, workspace, Counts::rows(rows))
            .map_err(|e| format!("{} serving down: {e}", case.label()))?,
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
    if workspace != View::EMPTY {
        poison.push(workspace);
    }
    poison.extend([
        metadata.counts.unwrap(),
        metadata.offsets,
        metadata.sorted_rows,
        metadata.padded_rows,
        metadata.active_experts.unwrap(),
    ]);
    let mut arm = Arm {
        launches,
        sum: Vec::new(),
        outputs,
        poison,
        status,
        active_list: metadata.active_experts.unwrap(),
        route_counts: metadata.counts.unwrap(),
        pointers: [
            [
                input.handle,
                gate_weights.codes.handle,
                gate_weights.local_scales.unwrap().handle,
                scales.handle,
                metadata.offsets.handle,
                metadata.sorted_rows.handle,
                metadata.padded_rows.handle,
                live.handle,
                rb.route.weights.handle,
                intermediate.handle,
            ],
            [
                codes.handle,
                down_weights.codes.handle,
                down_weights.local_scales.unwrap().handle,
                local_scales.handle,
                metadata.offsets.handle,
                metadata.sorted_rows.handle,
                metadata.padded_rows.handle,
                live.handle,
                0,
                output.handle,
            ],
        ],
        clocks: Vec::new(),
        clock_launches: Vec::new(),
    };
    append_sum(a, case, inputs, &mut arm)?;
    Ok(arm)
}

fn weights(
    a: &mut Arena<'_>,
    experts: usize,
    n: usize,
    k: usize,
) -> LabResult<QuantizedWeights<View>> {
    let count = experts * n * k / 2;
    if a.session.is_none() {
        return Ok(QuantizedWeights {
            codes: a.filled(count as u64, 0)?,
            local_scales: Some(a.filled((experts * n * k / 32) as u64, 0)?),
            global_scales: None,
        });
    }
    let codes = (0..count)
        .map(|i| {
            let x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
            (x ^ (x >> 16)) as u8
        })
        .collect::<Vec<_>>();
    let codes = a.bytes(&codes)?;
    let scales = (0..experts * n * k / 32)
        .map(|i| 120 + (i % 5) as u8)
        .collect::<Vec<_>>();
    Ok(QuantizedWeights {
        codes,
        local_scales: Some(a.bytes(&scales)?),
        global_scales: None,
    })
}
fn append_sum(a: &mut Arena<'_>, case: Case, inputs: Inputs, arm: &mut Arm) -> LabResult<()> {
    use kernel_core::moe::RoutedSumBuffers;
    use kernels_sm120::moe::chained::{RoutedSum, RoutedSumOutput};
    let entry = RoutedSum {
        top_k: case.set.top_k,
        width: 5120,
        local_experts: case.set.count,
        capacity: 8192,
        output: RoutedSumOutput::F32,
    };
    let prefix = if case.set.count == FLASH.shape.moe.experts {
        format!("chop_moe_{}", FLASH.kernel_tag)
    } else {
        format!("chop_{}_dspark_moe", FLASH.kernel_tag)
    };
    let backend = Sm120::new()
        .with_routed_sum(entry, prefix.clone())
        .with_routed_sum_add(entry, prefix);
    let op = backend
        .moe_routed_sum(
            &MoeQuery::LocalRoutedSum {
                experts: case.set.count,
                local_experts: case.set.count,
                top_k: case.set.top_k,
                width: 5120,
                capacity: case.rows,
                output: kernel_core::contract::DType::F32,
            },
            0,
            Need::rows(case.rows),
        )
        .map_err(|e| e.to_string())?;
    let addend = a.filled(u64::from(case.rows) * 5120 * 4, 0)?;
    let output = a.filled(addend.bytes, 0)?;
    arm.sum = op
        .launches(
            &RoutedSumBuffers {
                routed: arm.outputs[5],
                ids: arm.outputs[0],
                addend: Some(addend),
                output,
                live_rows: inputs.live,
            },
            View::EMPTY,
            Counts::rows(case.rows),
        )
        .map_err(|e| e.to_string())?;
    arm.outputs.push(output);
    arm.poison.push(output);
    Ok(())
}
fn component(arm: &Arm, part: &str) -> Vec<Launch> {
    if part == "full" {
        return arm.launches.iter().chain(&arm.sum).cloned().collect();
    }
    arm.launches
        .iter()
        .filter(|op| {
            if part == "projections" {
                op.symbol.contains("before_gateup_") || op.symbol.contains("before_down_")
            } else {
                op.symbol.contains(&format!("before_{part}_"))
            }
        })
        .cloned()
        .collect()
}
fn candidate(a: &mut Arena<'_>, case: Case, arm: &mut Arm, v: family::Variant) -> LabResult<()> {
    let mut replacement = Vec::new();
    let mut clocks = Vec::new();
    for (i, gate) in [true, false].into_iter().enumerate() {
        let p = family::projection(case.set.count, gate);
        let scratch = a.filled(family::workspace_bytes(&p, v), 0)?;
        let decoded = a.filled(
            u64::from(case.rows)
                * u64::from(if gate { 1 } else { case.set.top_k })
                * u64::from(p.query.projection.gemm.k)
                * 2,
            0,
        )?;
        let stamp = a.filled(family::STAMP_BYTES, 0)?;
        arm.poison.extend([scratch, decoded]);
        arm.clocks.push(stamp);
        let buffers = family::Buffers {
            h: arm.pointers[i],
            counts: arm.route_counts.handle,
            active: arm.active_list.handle,
            scratch: scratch.handle,
            decoded: decoded.handle,
            stamps: stamp.handle,
        };
        replacement.push(family::launches(&p, v, case.rows, buffers, false, false)?);
        clocks.push(family::launches(&p, v, case.rows, buffers, true, false)?);
    }
    let mut seen = [false; 2];
    let mut ops = Vec::new();
    let mut clock_ops = Vec::new();
    for op in &arm.launches {
        if let Some(i) = if op.symbol.contains("before_gateup_") {
            Some(0)
        } else if op.symbol.contains("before_down_") {
            Some(1)
        } else {
            None
        } {
            if !seen[i] {
                ops.extend(replacement[i].clone());
                clock_ops.extend(clocks[i].clone());
                seen[i] = true;
            }
        } else {
            ops.push(op.clone());
            clock_ops.push(op.clone());
        }
    }
    assert!(seen.into_iter().all(|x| x));
    arm.launches = ops;
    arm.clock_launches = clock_ops;
    Ok(())
}
