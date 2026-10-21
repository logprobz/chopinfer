//! The exact and native arms of one prefill step over shared inputs: the
//! serving ops (`serving_op`), the lab's native launches, and the views
//! their outputs land in.
use super::*;
use kernel_core::moe::RouteMetadata;
use kernel_core::moe_experts::GatedMlpScratch;
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};

/// Inputs both arms read.
#[derive(Clone, Copy)]
pub struct Inputs {
    pub codes: View,
    pub scales: View,
    pub live: View,
    pub route_weights: View,
    pub routes: RouteMetadata<View>,
    pub gate: QuantizedWeights<View>,
    pub down: QuantizedWeights<View>,
}

/// One arm's outputs.
#[derive(Clone, Copy)]
pub struct Outputs {
    pub intermediate: View,
    pub codes: View,
    pub scales: View,
    pub status: View,
    pub down: View,
}
impl Outputs {
    pub fn all(&self) -> [View; 4] {
        [self.intermediate, self.codes, self.scales, self.down]
    }
}

pub struct Arm {
    pub gate: Vec<Launch>,
    pub down: Vec<Launch>,
    pub out: Outputs,
    pub workspaces: Vec<View>,
}
impl Arm {
    pub fn launches(&self) -> Vec<Launch> {
        self.gate.iter().chain(&self.down).cloned().collect()
    }
}

pub struct Pipeline {
    pub set: moe::Experts,
    pub rows: u32,
    pub exact: Arm,
    pub native: Arm,
    /// The native arm's separate intermediate quantizer, which its gate/up
    /// launches end with unless they fuse it ([`fuses_ceil`]).
    pub quantizer: Vec<Launch>,
}

/// Whether the bundle's wrapper fuses the intermediate quantizer into a
/// step's gate/up tiles (the target bank at most `FUSED_CEIL_ROWS` rows).
pub fn fuses_ceil(set: moe::Experts, rows: u32) -> bool {
    set == family::serving::target_set()
        && rows <= bundle_deepseek41::sm120::moe_native::FUSED_CEIL_ROWS
}

fn outputs(
    a: &mut Arena<'_>,
    gate: &dyn PreparedExperts,
    down: &dyn PreparedExperts,
) -> LabResult<Outputs> {
    let mut sized = |op: &dyn PreparedExperts, name: &str| -> LabResult<View> {
        a.filled(
            op.info()
                .bytes_of(name)
                .ok_or_else(|| format!("missing {name}"))?,
            0,
        )
    };
    Ok(Outputs {
        intermediate: sized(gate, "output")?,
        codes: sized(gate, "scratch.output_codes")?,
        scales: sized(gate, "scratch.output_scales")?,
        status: sized(gate, "scratch.quantizer_status")?,
        down: sized(down, "output")?,
    })
}

pub fn buffers(i: &Inputs, out: &Outputs, gate: bool) -> GatedMlpBuffers<View> {
    let gate_b = GatedMlpBuffers {
        activation: i.codes,
        activation_global: None,
        activation_scales: Some(i.scales),
        live_tokens: i.live,
        output: out.intermediate,
        weights: i.gate,
        routing: Some(ProjectionRouting {
            weights: i.route_weights,
            routes: i.routes,
        }),
        scratch: GatedMlpScratch {
            quantizer_status: Some(out.status),
            quantizer_status_row_offset: 0,
            partial: None,
            output_codes: Some(out.codes),
            output_scales: Some(out.scales),
            unused: i.live,
        },
    };
    if gate {
        gate_b
    } else {
        GatedMlpBuffers {
            activation: out.codes,
            activation_scales: Some(out.scales),
            output: out.down,
            weights: i.down,
            ..gate_b
        }
    }
}

/// The lab's native handles of one projection reading `input`/`scales`.
pub fn native_buffers(
    i: &Inputs,
    gate: bool,
    input: View,
    scales: View,
    output: View,
    scratch: View,
    raw: Option<View>,
) -> native::Buffers {
    let w = if gate { i.gate } else { i.down };
    native::Buffers {
        input: input.handle,
        weights: w.codes.handle,
        weight_scales: w.local_scales.unwrap().handle,
        activation_scales: scales.handle,
        offsets: i.routes.offsets.handle,
        sorted_rows: i.routes.sorted_rows.handle,
        padded_rows: i.routes.padded_rows.handle,
        live_tokens: i.live.handle,
        route_weights: if gate { i.route_weights.handle } else { 0 },
        output: output.handle,
        counts: i.routes.counts.unwrap().handle,
        active: i.routes.active_experts.unwrap().handle,
        scratch: scratch.handle,
        raw: raw.map_or(0, |v| v.handle),
    }
}

pub fn queue(a: &mut Arena<'_>, shape: &Shape, tile: &Tile) -> LabResult<View> {
    a.filled(native::workspace_bytes(shape, tile), 0)
}

/// Both arms of a step of `rows` at `capacity`. The native arm is the lab's
/// launches of the serving tiles, with the bundle's fused ceil tiles and
/// fixup in place of the tiles and quantizer where the wrapper fuses them;
/// for the target bank they must equal the wrapper's launches.
pub fn build(
    a: &mut Arena<'_>,
    set: moe::Experts,
    capacity: u32,
    rows: u32,
    i: &Inputs,
) -> LabResult<Pipeline> {
    let gate_op = serving_op(set, GatedMlpProjection::GateUp, capacity, rows, false)?;
    let down_op = serving_op(set, GatedMlpProjection::Down, capacity, rows, false)?;
    let arm = |a: &mut Arena<'_>| -> LabResult<(Arm, Vec<Launch>)> {
        let out = outputs(a, gate_op.as_ref(), down_op.as_ref())?;
        let (gate, w0) = op_launches(a, gate_op.as_ref(), &buffers(i, &out, true), rows)?;
        let (down, w1) = op_launches(a, down_op.as_ref(), &buffers(i, &out, false), rows)?;
        let q = quantizer(&gate)?;
        Ok((
            Arm {
                gate,
                down,
                out,
                workspaces: vec![w0, w1],
            },
            q,
        ))
    };
    let (exact, _) = arm(a)?;
    let (mut native_arm, q) = arm(a)?;
    let shapes = family::KINDS.map(|kind| family::shape(set, kind));
    // Fused steps keep the gate/up tile the fused ceil was qualified on.
    let fused = fuses_ceil(set, rows);
    let mut tiles = family::KINDS.map(family::serving::tile);
    if fused {
        tiles[0] = family::serving::GATE_UP_TILE;
    }
    let queues = [
        queue(a, &shapes[0], &tiles[0])?,
        queue(a, &shapes[1], &tiles[1])?,
    ];
    let out = native_arm.out;
    let gate_buffers = native_buffers(
        i,
        true,
        i.codes,
        i.scales,
        out.intermediate,
        queues[0],
        None,
    );
    native_arm.gate = if fused {
        bundle_deepseek41::sm120::moe_native_entries::ceil()
            .launches(
                &shapes[0],
                &tiles[0],
                rows,
                &gate_buffers,
                (out.codes.handle, out.scales.handle, out.status.handle),
            )
            .map_err(|e| e.to_string())?
    } else {
        let mut ops = native_ops(&shapes[0], &tiles[0], rows, &gate_buffers, Variant::Serving)?;
        ops.extend(q.iter().cloned());
        ops
    };
    native_arm.down = native_ops(
        &shapes[1],
        &tiles[1],
        rows,
        &native_buffers(i, false, out.codes, out.scales, out.down, queues[1], None),
        Variant::Serving,
    )?;
    native_arm.workspaces = queues.to_vec();
    if set == family::serving::target_set() {
        // The bundle's wrapper must launch exactly these entries and
        // operands (its workspace handles differ).
        for (kind, lab) in family::KINDS
            .into_iter()
            .zip([&native_arm.gate, &native_arm.down])
        {
            let op = serving_op(set, kind, capacity, rows, true)?;
            let (ops, w) = op_launches(
                a,
                op.as_ref(),
                &buffers(i, &out, kind == GatedMlpProjection::GateUp),
                rows,
            )?;
            if ops.len() != lab.len() {
                return Err(format!(
                    "{kind:?}: bundle wrapper launches {} entries, lab {}",
                    ops.len(),
                    lab.len()
                ));
            }
            use kernel_core::contract::Argument::DeviceBuffer;
            let queue = queues[usize::from(kind == GatedMlpProjection::Down)];
            for (x, y) in ops.iter().zip(lab.iter()) {
                let renamed: Vec<_> = x
                    .arguments
                    .0
                    .iter()
                    .map(|arg| {
                        if *arg == DeviceBuffer(w.handle) {
                            DeviceBuffer(queue.handle)
                        } else {
                            arg.clone()
                        }
                    })
                    .collect();
                if x.symbol != y.symbol
                    || x.plan.grid != y.plan.grid
                    || x.plan.block != y.plan.block
                    || x.plan.dynamic_shared_bytes != y.plan.dynamic_shared_bytes
                    || renamed != y.arguments.0
                {
                    return Err(format!(
                        "{kind:?}: bundle wrapper launch {} differs from the lab's",
                        x.symbol
                    ));
                }
            }
        }
    }
    Ok(Pipeline {
        set,
        rows,
        exact,
        native: native_arm,
        quantizer: q,
    })
}

/// Host copies of views.
pub fn snapshot(a: &Arena<'_>, views: &[View]) -> LabResult<Vec<Vec<u8>>> {
    views.iter().map(|v| a.read(*v)).collect()
}

/// The first byte where two equal-length buffers' first `prefix` bytes differ.
pub fn first_difference(x: &[u8], y: &[u8], prefix: usize) -> Option<usize> {
    x[..prefix]
        .iter()
        .zip(&y[..prefix])
        .position(|(a, b)| a != b)
}
