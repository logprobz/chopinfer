//! Compare the active schedule with the locked column schedule, including
//! routing, clamped gate/up, BF16 rounding, ceil quantization and down.
use super::moe::registration;
use super::row_tiled_bank;

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

mod cases;
mod prefill;
mod profile;
mod shared;
use cases::{Case, Routing};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Old,
    Selected,
    Compact,
    BlockWarps,
    Streamed,
    Previous,
    PrefillGrouped,
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
    side: Vec<Launch>,
    sum: Vec<Launch>,
    active_list: View,
    route_counts: View,
    outputs: Vec<View>,
    poison: Vec<View>,
    status: View,
}

impl Arm {
    fn uses_compact_decode_one(&self, case: Case) -> bool {
        case.phase == ExecutionPhase::Decode
            && case.rows == 1
            && [("gateup", 9), ("down", 80)]
                .into_iter()
                .all(|(part, columns)| {
                    let ops = self
                        .launches
                        .iter()
                        .filter(|op| op.symbol.contains(&format!("before_{part}_")))
                        .collect::<Vec<_>>();
                    ops.len() == 1
                        && ops[0].symbol.contains("_r16_t32_")
                        && ops[0].plan.grid == [case.set.top_k, columns, 1]
                })
    }

    fn uses_streamed(&self) -> bool {
        ["gateup", "down"].into_iter().all(|part| {
            ["pack", "small", "merge"].into_iter().all(|stage| {
                self.launches.iter().any(|op| {
                    op.symbol.contains(&format!("before_{part}_"))
                        && (op.symbol.ends_with(&format!("_stream_{stage}_v1"))
                            || op.symbol.ends_with(&format!("_stream_{stage}_tiles_v1")))
                })
            })
        })
    }
}

#[derive(Clone, Copy)]
struct Inputs {
    input: View,
    scales: View,
    live: View,
    logits: View,
    weights: Banks,
}

/// A routed set's gate/up and down banks, row-major, with the gate/up bank
/// in the row tiles the bundle stores it in at Flash TP4 when it does
/// (`moe_row_tiles`, the backbone set).
#[derive(Clone, Copy)]
struct Banks {
    gate: QuantizedWeights<View>,
    down: QuantizedWeights<View>,
    gate_tiles: Option<QuantizedWeights<View>>,
}

fn arm(
    a: &mut Arena<'_>,
    case: Case,
    mode: Mode,
    inputs: Inputs,
    shared_weights: [QuantizedWeights<View>; 3],
) -> LabResult<Arm> {
    let rows = case.rows;
    let Inputs {
        input,
        scales,
        live,
        logits,
        weights:
            Banks {
                gate: gate_weights,
                down: down_weights,
                gate_tiles,
            },
    } = inputs;
    let set = case.set;
    let mut backend = if mode == Mode::Selected {
        Sm120::new()
    } else {
        registration::quantizers(&FLASH, 4)?
            .into_iter()
            .fold(Sm120::new(), Sm120::with_block_quantize)
    };
    let streamed = (mode == Mode::Streamed || (mode == Mode::Selected && !case.compact_selected()))
        && rows <= 64
        && case.phase != ExecutionPhase::Prefill;
    if mode == Mode::Selected {
        backend = registration::register(backend, &FLASH, 4)?;
    }
    let tile = if streamed {
        64
    } else if mode == Mode::BlockWarps || case.phase == ExecutionPhase::Decode {
        16
    } else {
        64
    };
    let entries = if mode == Mode::Selected {
        Vec::new()
    } else if mode == Mode::PrefillGrouped {
        registration::prefill_projections(&FLASH)?
    } else if streamed {
        registration::streamed_projections(&FLASH)?
            .into_iter()
            .filter(|e| e.tokens_capacity == if rows <= 8 { 8 } else { 64 })
            .collect()
    } else if mode == Mode::BlockWarps {
        registration::active_projections(&FLASH, 4)?
            .into_iter()
            .filter(|e| e.tokens_capacity == if rows <= 32 { 32 } else { 192 })
            .collect()
    } else {
        registration::projections(&FLASH, 4)?
    };
    for entry in entries.into_iter().filter(|e| e.row_tile == tile) {
        let entry = if mode == Mode::Compact {
            entry.with_compact_columns().map_err(|e| e.to_string())?
        } else {
            entry
        };
        backend = backend.with_quantized_chained_projection(entry);
    }
    if matches!(mode, Mode::Previous | Mode::Compact)
        && case.phase == ExecutionPhase::Decode
        && rows == 1
    {
        for entry in registration::serving_active_projections(&FLASH, 4)? {
            backend = backend.with_quantized_chained_projection(entry);
        }
    }
    for entry in if mode == Mode::Selected {
        Vec::new()
    } else {
        registration::routes(&FLASH)
            .into_iter()
            .chain(registration::active_routes(&FLASH))
            .chain(if streamed {
                registration::streamed_routes(&FLASH)
            } else {
                Vec::new()
            })
            .collect()
    } {
        backend = backend.with_row_bias_route(entry);
    }
    let usage = OperationUse {
        phase: if mode == Mode::BlockWarps {
            ExecutionPhase::Verify
        } else {
            case.phase
        },
        capacity: case.capacity(),
        graph: case.phase != ExecutionPhase::Prefill,
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
            .map_err(|e| format!("{} {mode:?} gate: {e}", case.label()))?,
    );
    launches.extend(
        down.launches(&down_b, workspace, Counts::rows(rows))
            .map_err(|e| format!("{} {mode:?} down: {e}", case.label()))?,
    );
    let launches = row_tiled_bank::launches(launches, gate_weights, gate_tiles)?;
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
        side: Vec::new(),
        sum: Vec::new(),
        outputs,
        poison,
        status,
        active_list: metadata.active_experts.unwrap(),
        route_counts: metadata.counts.unwrap(),
    };
    shared::append(a, case, inputs, &mut arm, shared_weights)?;
    Ok(arm)
}

/// The synthetic routed banks of `set` (gate/up 1152 by 5120, down 5120 by
/// 576 per expert), with the gate/up bank's row tiles where the bundle
/// stores them.
fn banks(a: &mut Arena<'_>, set: moe::Experts) -> LabResult<Banks> {
    let experts = set.count as usize;
    let (gate, gate_tiles) = weights(a, experts, 1152, 5120, row_tiled_bank::tiled(4, set))?;
    Ok(Banks {
        gate,
        down: weights(a, experts, 5120, 576, false)?.0,
        gate_tiles,
    })
}

/// A synthetic row-major bank, and with `tiled` its copy in row tiles.
fn weights(
    a: &mut Arena<'_>,
    experts: usize,
    n: usize,
    k: usize,
    tiled: bool,
) -> LabResult<(QuantizedWeights<View>, Option<QuantizedWeights<View>>)> {
    let count = experts * n * k / 2;
    if a.session.is_none() {
        let mut bank = || -> LabResult<_> {
            Ok(QuantizedWeights {
                codes: a.filled(count as u64, 0)?,
                local_scales: Some(a.filled((experts * n * k / 32) as u64, 0)?),
                global_scales: None,
            })
        };
        return Ok((bank()?, if tiled { Some(bank()?) } else { None }));
    }
    let mut codes = (0..count)
        .map(|i| {
            let x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
            (x ^ (x >> 16)) as u8
        })
        .collect::<Vec<_>>();
    let mut scales = (0..experts * n * k / 32)
        .map(|i| 120 + (i % 5) as u8)
        .collect::<Vec<_>>();
    let mut upload = |codes: &[u8], scales: &[u8]| -> LabResult<_> {
        Ok(QuantizedWeights {
            codes: a.bytes(codes)?,
            local_scales: Some(a.bytes(scales)?),
            global_scales: None,
        })
    };
    let rows = upload(&codes, &scales)?;
    if !tiled {
        return Ok((rows, None));
    }
    row_tiled_bank::pack(&mut codes, &mut scales, experts * n, k)?;
    Ok((rows, Some(upload(&codes, &scales)?)))
}

fn compare(
    a: &mut Arena<'_>,
    case: Case,
    routed_weights: Banks,
    shared_weights: [QuantizedWeights<View>; 3],
) -> LabResult<()> {
    let session = a.session.ok_or("GPU case requires a session")?;
    let rows = case.rows;
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
    let logits = a.bytes(&case.logits())?;
    let inputs = Inputs {
        input,
        scales,
        live,
        logits,
        weights: routed_weights,
    };
    let arms = [
        Mode::Old,
        Mode::Selected,
        Mode::Compact,
        Mode::BlockWarps,
        Mode::Streamed,
        Mode::Previous,
    ]
    .map(|mode| arm(a, case, mode, inputs, shared_weights))
    .into_iter()
    .collect::<LabResult<Vec<_>>>()?;
    let dummy = a.filled(4, 0)?;
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    let mut functions = BTreeMap::<String, Function>::new();
    for op in arms
        .iter()
        .flat_map(|arm| arm.launches.iter().chain(&arm.side).chain(&arm.sum))
    {
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
    let (bytes, _) = bundle_deepseek41::sm120::dependency_probes();
    let delay = session
        .load_cubin(bytes)?
        .function(kernels_sm120::probes::COPY)
        .map_err(|e| e.to_string())?;
    let side = session.context.stream().map_err(|e| e.to_string())?;
    let fork = session.context.event().map_err(|e| e.to_string())?;
    let join = session.context.event().map_err(|e| e.to_string())?;
    let consumed = session.context.event().map_err(|e| e.to_string())?;
    let issue = |ops: &[Launch], stream: &Stream| -> LabResult<()> {
        for op in ops {
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
    let capture = |index: usize, stress: bool, dropped_join: bool| {
        session.capture(|compute| {
            let arm = &arms[index];
            if index == 0 || case.phase == ExecutionPhase::Prefill {
                issue(&arm.launches, compute)?;
                issue(&arm.side, compute)?;
                return issue(&arm.sum, compute);
            }
            fork.record(compute).map_err(|e| e.to_string())?;
            side.wait_event(&fork).map_err(|e| e.to_string())?;
            if dropped_join {
                // Force one legal interleaving after dropping the join: the
                // consumer finishes before the shared producer runs. The
                // producer still executes and the graph still rejoins.
                issue(&arm.launches, compute)?;
                issue(&arm.sum, compute)?;
                consumed.record(compute).map_err(|e| e.to_string())?;
                side.wait_event(&consumed).map_err(|e| e.to_string())?;
                issue(&arm.side, &side)?;
                join.record(&side).map_err(|e| e.to_string())?;
                compute.wait_event(&join).map_err(|e| e.to_string())?;
                return Ok(());
            }
            if stress {
                // One delayed CTA leaves the remaining SMs free for routed
                // work. The joined control must still wait for the shared partial.
                delay
                    .launch(
                        &side,
                        kernel_harness::launch::simple_config(
                            (1, 1, 1),
                            (32, 1, 1),
                            0,
                            (1, 1, 1),
                            false,
                        )?,
                        &[
                            runtime_cuda::KernelArgument::Buffer(&a.buffers[&live.handle]),
                            runtime_cuda::KernelArgument::Buffer(&a.buffers[&dummy.handle]),
                            runtime_cuda::KernelArgument::U64(1),
                            runtime_cuda::KernelArgument::U64(250_000_000),
                        ],
                    )
                    .map_err(|e| e.to_string())?;
            }
            issue(&arm.side, &side)?;
            issue(&arm.launches, compute)?;
            join.record(&side).map_err(|e| e.to_string())?;
            compute.wait_event(&join).map_err(|e| e.to_string())?;
            issue(&arm.sum, compute)?;
            Ok(())
        })
    };
    let graphs = (0..arms.len())
        .map(|i| capture(i, false, false))
        .collect::<LabResult<Vec<_>>>()?;
    let poison = |arm: &Arm| -> LabResult<()> {
        for &v in &arm.poison {
            a.buffers[&v.handle]
                .copy_from_bytes(&vec![0xff; v.bytes as usize], &session.stream)
                .map_err(|e| e.to_string())?;
        }
        a.buffers[&arm.status.handle]
            .copy_from_bytes(&vec![0; arm.status.bytes as usize], &session.stream)
            .map_err(|e| e.to_string())
    };
    let equal = |live_rows: u32| -> LabResult<()> {
        for (which, arm) in arms.iter().enumerate().skip(1) {
            for (index, (&old, &new)) in arms[0].outputs.iter().zip(&arm.outputs).enumerate() {
                let left = a.read(old)?;
                let right = a.read(new)?;
                if left != right {
                    return Err(format!(
                        "{} arm {which}, live {live_rows}, output {index}: bitwise mismatch",
                        case.label()
                    ));
                }
                let tail = live_rows as usize * left.len() / rows as usize;
                let written = &left[..tail];
                let valid = match index {
                    0 => written.chunks_exact(4).all(|v| {
                        (0..case.set.count as i32)
                            .contains(&i32::from_le_bytes(v.try_into().unwrap()))
                    }),
                    1 | 5 | 6 | 7 | 10 | 11 => written
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                    2 => written
                        .chunks_exact(2)
                        .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80),
                    3 | 8 => written.iter().all(|v| v & 0x7f != 0x7f),
                    4 | 9 => !written.contains(&255),
                    _ => unreachable!("declared output"),
                };
                if !valid {
                    return Err(format!(
                        "output {index} retains poison or invalid live values"
                    ));
                }
                if left[tail..].iter().any(|&b| b != 0xff) {
                    return Err("inactive output rows were written".into());
                }
            }
        }
        for arm in &arms {
            if a.read(arm.status)?.iter().any(|&b| b != 0) {
                return Err("quantizer status is nonzero".into());
            }
        }
        Ok(())
    };
    for live_rows in [rows, 1, 0, rows] {
        a.buffers[&live.handle]
            .copy_from_bytes(&live_rows.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        for (index, arm) in arms.iter().enumerate() {
            poison(arm)?;
            graphs[index]
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        equal(live_rows)?;
        if case.routing == Routing::All && live_rows == rows {
            for arm in &arms {
                let active = a
                    .read(arm.active_list)?
                    .chunks_exact(4)
                    .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    active[0], case.set.count as i32,
                    "all experts must be active"
                );
                assert_eq!(active[1..], (0..case.set.count as i32).collect::<Vec<_>>());
            }
        }
    }
    let mut dropped_join_detected = false;
    if case.phase != ExecutionPhase::Prefill {
        let good = capture(1, true, false)?;
        let bad = capture(1, true, true)?;
        poison(&arms[1])?;
        good.launch(&session.stream).map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        equal(rows)?;
        poison(&arms[1])?;
        bad.launch(&session.stream).map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        // Judge the final MoE result, not an event count or an intermediate.
        dropped_join_detected = a.read(*arms[0].outputs.last().unwrap())?
            != a.read(*arms[1].outputs.last().unwrap())?;
        if !dropped_join_detected {
            return Err("dropped-join negative control escaped the final-output comparison".into());
        }
        poison(&arms[1])?;
        graphs[1]
            .launch(&session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        equal(rows)?;
    }
    let selected = timed_pair(session, |i, s| {
        graphs[i].launch(s).map_err(|e| e.to_string())
    })?;
    let compact = timed_pair(session, |i, s| {
        graphs[if i == 0 { 0 } else { 2 }]
            .launch(s)
            .map_err(|e| e.to_string())
    })?;
    let block_warps = timed_pair(session, |i, s| {
        graphs[if i == 0 { 0 } else { 3 }]
            .launch(s)
            .map_err(|e| e.to_string())
    })?;
    let streamed = timed_pair(session, |i, s| {
        graphs[if i == 0 { 0 } else { 4 }]
            .launch(s)
            .map_err(|e| e.to_string())
    })?;
    let previous = timed_pair(session, |i, s| {
        graphs[if i == 0 { 5 } else { 1 }]
            .launch(s)
            .map_err(|e| e.to_string())
    })?;
    let expected_streamed = rows <= 64 && case.phase != ExecutionPhase::Prefill;
    let selected_streamed = arms[1].uses_streamed();
    let selected_compact = arms[1].uses_compact_decode_one(case);
    assert_eq!(
        selected_compact,
        case.compact_selected(),
        "compact selection coverage"
    );
    let streamed_compared = arms[4].uses_streamed();
    assert_eq!(
        selected_streamed,
        expected_streamed && !case.compact_selected(),
        "selected entry coverage"
    );
    assert_eq!(
        streamed_compared, expected_streamed,
        "streamed parity coverage"
    );
    let mut kernel_times = BTreeMap::new();
    let mut floor_ms = BTreeMap::new();
    let mut floor_ratio = BTreeMap::new();
    let active = a.read(arms[0].active_list)?;
    let active_experts = u32::from_le_bytes(active[..4].try_into().unwrap());
    for name in ["gateup", "down"] {
        let kernels = arms
            .iter()
            .map(|arm| {
                let ops = arm
                    .launches
                    .iter()
                    .filter(|op| op.symbol.contains(&format!("before_{name}_")))
                    .cloned()
                    .collect::<Vec<_>>();
                assert!(!ops.is_empty());
                session.capture(|s| issue(&ops, s))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut times = Vec::new();
        for candidate in 1..arms.len() {
            let pair = timed_pair(session, |i, s| {
                kernels[if i == 0 { 0 } else { candidate }]
                    .launch(s)
                    .map_err(|e| e.to_string())
            })?;
            if candidate == 1 {
                times.push(pair[0]);
            }
            times.push(pair[1]);
        }
        // Weight-only lower bound, including UE8M0/32 bytes. This omits
        // activation, partial and output traffic and assumes 1.6 TB/s.
        let elements = if name == "gateup" {
            1152.0 * 5120.0
        } else {
            5120.0 * 576.0
        };
        let floor = f64::from(active_experts) * elements * (0.5 + 1.0 / 32.0) / 1.6e9;
        floor_ms.insert(name, floor);
        floor_ratio.insert(
            name,
            times
                .iter()
                .map(|&t| f64::from(t) / floor)
                .collect::<Vec<_>>(),
        );
        kernel_times.insert(name, times);
    }
    let gate = cases::speed_gate(selected[0], selected[1])
        .map_err(|e| format!("serial baseline: {e}"))
        .and_then(|()| {
            cases::speed_gate(previous[0], previous[1])
                .map_err(|e| format!("previous serving selection: {e}"))
        });
    let record = serde_json::json!({"case":case.label(),"rows":rows,"experts":case.set.count,"top_k":case.set.top_k,
        "phase":format!("{:?}",case.phase),"routing":format!("{:?}",case.routing),"cache":"warm",
        "old_ms":selected[0],"active_ms":selected[1],"compact_ms":compact[1],"block_warps_ms":block_warps[1],"ratio":selected[1]/selected[0],
        "streamed_ms":streamed[1],"streamed_old_ms":streamed[0],
        "streamed_candidate":streamed_compared,"selected_streamed":selected_streamed,"selected_compact":selected_compact,
        "streamed_bitwise":streamed_compared,"previous_ms":previous[0],"selected_paired_ms":previous[1],
        "compared_modes":["old","selected","compact","block_warps","streamed","previous"],
        "compared_outputs":12,"live_replays":[rows,1,0,rows],
        "active_experts":active_experts,"bandwidth_tb_s":1.6,"weight_floor_ms":floor_ms,"floor_ratio":floor_ratio,
        "kernel_order":["old","selected","compact","block_warps","streamed","previous"],
        "kernel_ms":kernel_times,"bitwise":true,"dropped_join_detected":dropped_join_detected,
        "speed_passed":gate.is_ok(),"selection":if selected_streamed {"streamed gate/up and down"} else if selected_compact {"compact column gate/up and down"} else {"old routed kernels"}});
    println!("MOE_RESULT {record}");
    session.receipt(
        "moe-active-bitwise",
        serde_json::from_value(record).map_err(|e| e.to_string())?,
    )?;
    gate
}

/// Alternating A/B pairs, four warmups and twenty samples per arm.
fn timed_pair(
    session: &Session,
    work: impl Fn(usize, &Stream) -> LabResult<()>,
) -> LabResult<[f32; 2]> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    let mut samples = [Vec::new(), Vec::new()];
    for round in 0..24 {
        for step in 0..2 {
            let arm = (round + step) % 2;
            start.record(&session.stream).map_err(|e| e.to_string())?;
            work(arm, &session.stream)?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let elapsed = start.elapsed_ms(&end).map_err(|e| e.to_string())?;
            if round >= 4 {
                samples[arm].push(elapsed);
            }
        }
    }
    Ok(samples.map(|mut s| {
        s.sort_by(f32::total_cmp);
        (s[9] + s[10]) * 0.5
    }))
}

fn suite(bank: usize) {
    kernel_harness::gpu_test(8 << 30, |session| {
        let set = moe::expert_sets(&FLASH)[bank];
        let mut a = Arena {
            session: Some(session),
            next: 0,
            buffers: BTreeMap::new(),
        };
        let routed = banks(&mut a, set)?;
        let shared = shared::weights(&mut a)?;
        let mut errors = Vec::new();
        let weights_end = a.next;
        for case in cases::cases(set) {
            if let Err(error) = compare(&mut a, case, routed, shared) {
                errors.push(format!("{}: {error}", case.label()));
            }
            a.buffers.retain(|&handle, _| handle <= weights_end);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    })
    .unwrap();
}
#[test]
fn active_moe_backbone_gpu() {
    suite(0);
}
#[test]
fn active_moe_dspark_gpu() {
    suite(1);
}

#[test]
fn active_moe_cpu_dry_run() {
    for set in moe::expert_sets(&FLASH) {
        let mut a = Arena {
            session: None,
            next: 0,
            buffers: BTreeMap::new(),
        };
        let routed = banks(&mut a, set).unwrap();
        let shared = shared::weights(&mut a).unwrap();
        for case in cases::cases(set) {
            let input = a.filled(u64::from(case.rows) * 5120, 0).unwrap();
            let scales = a.filled(u64::from(case.rows) * 160, 0).unwrap();
            let live = a.filled(4, 0).unwrap();
            let logits = a
                .filled(u64::from(case.rows) * u64::from(set.count) * 4, 0)
                .unwrap();
            for mode in [
                Mode::Old,
                Mode::Selected,
                Mode::Compact,
                Mode::BlockWarps,
                Mode::Streamed,
                Mode::Previous,
            ] {
                let arm = arm(
                    &mut a,
                    case,
                    mode,
                    Inputs {
                        input,
                        scales,
                        live,
                        logits,
                        weights: routed,
                    },
                    shared,
                )
                .unwrap();
                if mode == Mode::Selected || mode == Mode::Streamed {
                    assert_eq!(
                        arm.uses_streamed(),
                        case.rows <= 64
                            && case.phase != ExecutionPhase::Prefill
                            && (mode != Mode::Selected || !case.compact_selected())
                    );
                }
                if mode == Mode::Selected {
                    assert_eq!(arm.uses_compact_decode_one(case), case.compact_selected());
                }
                assert_eq!(arm.side.len(), 4);
                assert_eq!(arm.sum.len(), 1);
                assert_eq!(arm.outputs.len(), 12);
                for op in arm.launches.iter().chain(&arm.side).chain(&arm.sum) {
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
}

#[test]
fn profile_moe_gpu() {
    kernel_harness::gpu_test(8 << 30, profile::run).unwrap();
}
