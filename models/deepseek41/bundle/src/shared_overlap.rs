//! Run the shared expert beside the routed experts inside decode and verify
//! graphs. The input quantization precedes the fork, and the routed sum
//! follows the join; after an r2 router finish the join precedes the gate,
//! so only the router runs beside the shared expert. That gate launches
//! without programmatic serialization: launched programmatically in capture,
//! it started beside the shared down kernel (which triggers its dependents
//! at entry) in most layers of the round 10 decode profile, as if the join
//! were a programmatic edge. Each stream retains its arithmetic and stage
//! order.
//! The shared expert owns separate backend workspace and row status.
//! Capture pieces contain whole fork/join regions. Eager runs remain serial.
//!
//! The attention sublayer's window KV projection reads only the layer's
//! quantized input, so it runs on the side stream beside the query chain
//! (the query-down projection through the query RoPE) and joins before the
//! window KV stage that reads it. Like the shared expert, it binds the
//! independent workspace and row status in decode and verify phases of at
//! most `KV_ROWS` rows ([`kv_relocated`]).

use std::collections::BTreeMap;
use std::ops::Range;

use bundle_runtime::dispatch::{self, DispatchError};
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::{Argument, LaunchPlan};
use runtime_bundle_abi::{
    BundleCudaSymbolHandleV1, BundleEventHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    HostServicesV1,
};

use crate::program::Stage;
use bundle_runtime::pipeline::launch_all;

/// Set to zero to retain serial graph captures. Positive values limit rows.
#[cfg_attr(test, allow(dead_code))]
pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_SHARED_OVERLAP";

/// Set to zero to keep the window KV projection on the compute stream in
/// every overlapped bucket.
#[cfg_attr(test, allow(dead_code))]
pub const KV_SWITCH: &str = "CHOPINFER_DEEPSEEK41_KV_OVERLAP";

/// The largest decode or verify phase whose window KV projection binds the
/// shared expert's workspace and row status: the dense decode buckets.
pub const KV_ROWS: u32 = crate::sm120::mxfp8_fold::SMALL_CAPACITY;

/// Whether `phase`'s window KV projection binds the independent workspace
/// and row status, which lets it run beside the query chain.
pub(crate) fn kv_relocated(phase: crate::program::Phase) -> bool {
    #[cfg(test)]
    if KV_REFERENCE.get() {
        return false;
    }
    !phase.prefill() && phase.rows() <= KV_ROWS
}

#[cfg(test)]
thread_local! {
    static KV_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `run` with every window KV projection bound to the compute stream's
/// workspace and row status, as before the overlap.
#[cfg(test)]
pub(crate) fn with_kv_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            KV_REFERENCE.set(self.0);
        }
    }
    let _restore = Restore(KV_REFERENCE.replace(true));
    run()
}

/// Which graph buckets overlap, read at create (which creates each rank's
/// side stream and events when any does).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Settings {
    /// The largest bucket's rows that overlap; 0 for none.
    pub max_rows: u32,
    /// Whether the window KV projection runs beside the query chain in the
    /// overlapped buckets of at most [`KV_ROWS`] rows.
    pub kv: bool,
}

impl Settings {
    /// Every bucket on one stream.
    pub const OFF: Self = Self {
        max_rows: 0,
        kv: false,
    };
    /// Every bucket overlapped, the default.
    pub const ALL: Self = Self {
        max_rows: u32::MAX,
        kv: true,
    };

    /// The settings [`SWITCH`]'s `value` and [`KV_SWITCH`]'s `kv` name.
    pub fn read(value: Option<&str>, kv: Option<&str>) -> Self {
        let kv = kv != Some("0");
        match value.map(str::parse::<u32>) {
            Some(Ok(0)) => Self::OFF,
            Some(Ok(max_rows)) => Self { max_rows, kv },
            None | Some(Err(_)) => Self { kv, ..Self::ALL },
        }
    }

    /// The process environment's settings; a test's instances take the
    /// default (not the process environment) unless it sets theirs within
    /// [`with_settings_in_tests`].
    pub fn from_env() -> Self {
        #[cfg(test)]
        return IN_TESTS.get().unwrap_or_else(|| Self::read(None, None));
        #[cfg(not(test))]
        Self::read(
            std::env::var(SWITCH).ok().as_deref(),
            std::env::var(KV_SWITCH).ok().as_deref(),
        )
    }

    /// Whether any bucket overlaps.
    pub fn on(self) -> bool {
        self.max_rows > 0
    }

    /// Whether a graph bucket of `rows` rows overlaps.
    pub fn serves(self, rows: u32) -> bool {
        self.on() && rows <= self.max_rows
    }

    /// Whether a graph bucket of `rows` rows runs its window KV projections
    /// beside the query chains.
    pub fn serves_kv(self, rows: u32) -> bool {
        self.kv && self.serves(rows) && rows <= KV_ROWS
    }
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: std::cell::Cell<Option<Settings>> = const { std::cell::Cell::new(None) };
}

/// Runs `create` with `settings`, as [`SWITCH`] would set them.
#[cfg(test)]
pub(crate) fn with_settings_in_tests<T>(settings: Settings, create: impl FnOnce() -> T) -> T {
    IN_TESTS.set(Some(settings));
    let created = create();
    IN_TESTS.set(None);
    created
}

/// Events each rank creates beside its side stream: the fork and the join.
pub const EVENTS: usize = 2;

/// One rank's streams and events of the overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lanes {
    /// The rank's stream, which captures and replays its graphs.
    pub compute: BundleStreamHandleV1,
    /// The stream the shared expert records on inside a capture.
    pub side: BundleStreamHandleV1,
    pub fork: BundleEventHandleV1,
    pub join: BundleEventHandleV1,
}

/// One submission of a capture, over launch indices of the rank's program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    /// These launches on the compute stream.
    Compute(Range<usize>),
    /// The side stream waits for the compute stream's work so far.
    Fork,
    /// These launches on the side stream.
    Side(Range<usize>),
    /// The compute stream waits for the side stream's work so far.
    Join,
}

fn end(span: DeviceBufferSpanV1) -> Result<u64, String> {
    span.offset
        .checked_add(span.length)
        .ok_or_else(|| "overlap span overflows".into())
}

fn contains(outer: DeviceBufferSpanV1, inner: DeviceBufferSpanV1) -> Result<bool, String> {
    Ok(outer.buffer == inner.buffer && outer.offset <= inner.offset && end(inner)? <= end(outer)?)
}

fn intersects(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> Result<bool, String> {
    Ok(a.length != 0
        && b.length != 0
        && a.buffer == b.buffer
        && a.offset < end(b)?
        && b.offset < end(a)?)
}

/// Bind the shared stage's workspace and status to their independent rank
/// regions. Bind narrows workspace views to OpInfo::workspace first, so an
/// undersized destination is rejected rather than silently truncated.
pub(crate) fn bind_shared(
    launch: &mut PreparedLaunch,
    buffers: &crate::buffers::RankBuffers,
) -> Result<(), String> {
    let mut relocate = |id: &mut u64| -> Result<(), String> {
        let view = launch
            .bindings
            .get(*id)
            .map_err(|e| format!("shared binding: {e:?}"))?;
        for (source, target) in [
            (buffers.workspace, buffers.shared_workspace),
            (
                buffers.scratch.row_status,
                buffers.scratch.shared_row_status,
            ),
        ] {
            if !intersects(source, view)? {
                continue;
            }
            if !contains(source, view)? {
                return Err("shared scratch straddles its source region".into());
            }
            let offset = target
                .offset
                .checked_add(view.offset - source.offset)
                .ok_or("shared scratch offset overflows")?;
            let relocated = DeviceBufferSpanV1::new(target.buffer, offset, view.length);
            if !contains(target, relocated)? {
                return Err(format!(
                    "shared scratch exceeds its prepared workspace: {} bytes",
                    view.length
                ));
            }
            *id = launch.bindings.bind(relocated);
            break;
        }
        Ok(())
    };
    for argument in &mut launch.arguments.0 {
        match argument {
            Argument::DeviceBuffer(id) => relocate(id)?,
            Argument::TensorMap(map) => relocate(&mut map.base_buffer)?,
            Argument::DevicePointerArray(ids) => {
                for id in ids {
                    relocate(id)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn visit_spans(
    launch: &PreparedLaunch,
    mut visit: impl FnMut(DeviceBufferSpanV1, usize),
) -> Result<(), String> {
    let mut add = |id, index| -> Result<(), String> {
        let span = launch
            .bindings
            .get(id)
            .map_err(|e| format!("overlap binding: {e:?}"))?;
        end(span)?;
        visit(span, index);
        Ok(())
    };
    for (index, argument) in launch.arguments.0.iter().enumerate() {
        match argument {
            Argument::DeviceBuffer(id) => add(*id, index)?,
            Argument::TensorMap(map) => add(map.base_buffer, index)?,
            Argument::DevicePointerArray(ids) => {
                for id in ids {
                    add(*id, index)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn spans(launch: &PreparedLaunch) -> Result<Vec<(DeviceBufferSpanV1, bool)>, String> {
    let mut spans = Vec::new();
    visit_spans(launch, |span, index| {
        spans.push((
            span,
            crate::Kernels::const_input_argument(&launch.symbol, index),
        ));
    })?;
    Ok(spans)
}

fn strict_spans(launch: &PreparedLaunch) -> Result<Vec<DeviceBufferSpanV1>, String> {
    let mut spans = Vec::new();
    // Prefill branches may share no bytes, even through const arguments.
    // Their audit needs resolved views but no argument-access classification.
    visit_spans(launch, |span, _| spans.push(span))?;
    Ok(spans)
}

/// Reject every shared byte between two concurrent branches. This includes
/// tensor-map bases and peer pointer arrays, as in the graph scratch audit.
pub(crate) fn disjoint_branches(
    side: &[PreparedLaunch],
    main: &[PreparedLaunch],
) -> Result<(), String> {
    // Resolve each launch once. Repeating this inside the pairwise span
    // comparison adds hundreds of milliseconds before the first GPU launch.
    let main = main
        .iter()
        .map(|launch| Ok((&launch.symbol, strict_spans(launch)?)))
        .collect::<Result<Vec<_>, String>>()?;
    for side in side {
        let side_spans = strict_spans(side)?;
        for (symbol, main_spans) in &main {
            for &a in &side_spans {
                for &b in main_spans {
                    if intersects(a, b)? {
                        return Err(format!(
                            "unordered scratch alias: {} {a:?} and {} {b:?}",
                            side.symbol, symbol,
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Copies carry spans directly, so audit their full transfer footprints as
/// well as the arithmetic launch that consumes a local owner interval.
pub(crate) fn disjoint_copy_spans(
    copies: &[DeviceBufferSpanV1],
    main: &[PreparedLaunch],
) -> Result<(), String> {
    for &a in copies {
        end(a)?;
    }
    for launch in main {
        for b in strict_spans(launch)? {
            for &a in copies {
                if intersects(a, b)? {
                    return Err(format!(
                        "unordered DMA scratch alias: {a:?} and {} {b:?}",
                        launch.symbol
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Whether a router finish ends before the shared expert, so that its
/// sublayer's gate waits for the shared expert: the lower-latency r2
/// finishes (`*_router_finish_*_finish_r2v*`). A gate started beside the
/// shared expert's last kernel loses more than such a finish saves.
pub(crate) fn joins_before_gate(symbol: &str) -> bool {
    symbol.starts_with("chop_")
        && symbol.contains("_moe_router_finish_")
        && symbol.contains("_finish_r2v")
}

/// Check actual pointer arguments before capture. Only the input codes,
/// input scales and live-row count may be shared across the two branches;
/// their producers finish before the fork, and both entry arguments must
/// declare const input pointers. Every other intersecting range
/// is rejected, including scratch introduced by a newly selected backend.
///
/// A sublayer whose router ends with the r2 finish joins before its gate:
/// `[Fork, Side, Compute(router), Join, Compute(gate onward)]`. That finish
/// ends before the shared expert, and a gate started beside the shared
/// expert loses more than the finish saves; only the router is then
/// concurrent with the shared expert and audited against it.
///
/// With `kv`, each attention sublayer's window KV projection runs beside its
/// query chain ([`units`]) and is audited against it alike.
pub fn checked_units(
    stages: &[Stage],
    program: &[PreparedLaunch],
    immutable: &[DeviceBufferSpanV1],
    kv: bool,
) -> Result<Vec<Vec<Op>>, String> {
    if stages.len() != program.len() {
        return Err("overlap stage and launch counts differ".into());
    }
    let mut units = units(stages, kv)?;
    for unit in &mut units {
        let [Op::Fork, Op::Side(side), Op::Compute(main), Op::Join] = &unit[..] else {
            continue;
        };
        let (side, mut main) = (side.clone(), main.clone());
        let kv_side = stages[side.clone()]
            .iter()
            .all(|s| matches!(s, Stage::Project(_, "attention.kv")));
        let gate = main
            .clone()
            .find(|&i| matches!(stages[i], Stage::ExpertsGateUp(_)));
        if gate.is_none() && !kv_side {
            return Err("overlapped sublayer has no gate".into());
        }
        if let Some(gate) = gate
            && program[main.start..gate]
                .iter()
                .any(|l| joins_before_gate(&l.symbol))
        {
            *unit = vec![
                Op::Fork,
                Op::Side(side.clone()),
                Op::Compute(main.start..gate),
                Op::Join,
                Op::Compute(gate..main.end),
            ];
            main = main.start..gate;
        }
        let bound = |range: Range<usize>| -> Result<Vec<_>, String> {
            range.map(|i| Ok((i, spans(&program[i])?))).collect()
        };
        let main = bound(main)?;
        for (side_index, side_spans) in bound(side)? {
            for (main_index, main_spans) in &main {
                for &(a, a_const) in &side_spans {
                    for &(b, b_const) in main_spans {
                        if !intersects(a, b)? {
                            continue;
                        }
                        let mut read_only = false;
                        for &input in immutable {
                            read_only |=
                                a_const && b_const && contains(input, a)? && contains(input, b)?;
                        }
                        if !read_only {
                            return Err(format!(
                                "unordered scratch alias: {} {a:?} and {} {b:?}",
                                program[side_index].symbol, program[*main_index].symbol,
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(units)
}

/// The stages of `layer`'s query chain, which run beside its window KV
/// projection: the query-down projection through the query RoPE.
fn query_chain(stage: Stage, layer: u32) -> bool {
    matches!(stage,
        Stage::Project(l, "attention.q-a" | "attention.q-b")
        | Stage::QueryNorm(l)
        | Stage::QueryRope(l) if l == layer)
}

/// A program's submissions in units the capture's graph pieces never split:
/// each launch outside an overlapped sublayer alone, and each such
/// sublayer's concurrent work between its fork and join as one: an MoE
/// sublayer's routed work beside its shared expert and, with `kv`, an
/// attention sublayer's query chain beside its window KV projection, which
/// lowering places right after the chain. `stages` names each launch's
/// stage.
pub fn units(stages: &[Stage], kv: bool) -> Result<Vec<Vec<Op>>, String> {
    let mut units = Vec::new();
    let mut at = 0;
    while at < stages.len() {
        if kv && let Stage::Project(layer, "attention.q-a") = stages[at] {
            let chain = at
                + stages[at..]
                    .iter()
                    .take_while(|&&s| query_chain(s, layer))
                    .count();
            let side = chain
                + stages[chain..]
                    .iter()
                    .take_while(|&&s| s == Stage::Project(layer, "attention.kv"))
                    .count();
            if side > chain {
                units.push(vec![
                    Op::Fork,
                    Op::Side(chain..side),
                    Op::Compute(at..chain),
                    Op::Join,
                ]);
                at = side;
                continue;
            }
        }
        let Stage::Project(layer, "moe.router") = stages[at] else {
            units.push(vec![Op::Compute(at..at + 1)]);
            at += 1;
            continue;
        };
        let start = at;
        let mut shared = None;
        for stage in crate::program::moe(layer).into_iter().take(7) {
            let count = stages[at..].iter().take_while(|&&s| s == stage).count();
            if count == 0 {
                return Err(format!(
                    "layer {layer} is missing {stage:?} before its routed sum"
                ));
            }
            if stage == Stage::SharedGateUp(layer) {
                shared = Some(at);
            }
            at += count;
        }
        if stages.get(at) != Some(&Stage::RoutedSum(layer)) {
            return Err(format!(
                "layer {layer} has no routed sum after its shared down"
            ));
        }
        let shared = shared.expect("shared gate/up was visited");
        units.push(vec![
            Op::Fork,
            Op::Side(shared..at),
            Op::Compute(start..shared),
            Op::Join,
        ]);
    }
    Ok(units)
}

/// `launch_all` with the first launch's programmatic serialization off, so
/// that it waits for the joined shared expert to complete. Every launch is
/// resolved before the first is submitted.
fn launch_after_join(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    if launches.is_empty() || launches.iter().any(|launch| !launch.plan.pdl) {
        return Err(DispatchError::InvalidPlan);
    }
    let resolved = launches
        .iter()
        .enumerate()
        .map(|(i, launch)| {
            let plan = LaunchPlan {
                pdl: i != 0,
                ..launch.plan
            };
            dispatch::resolve(
                symbols,
                &launch.symbol,
                plan,
                &launch.arguments,
                &launch.bindings,
            )
            .inspect_err(|error| {
                let _ =
                    services.log(format!("launch {} failed: {error:?}", launch.symbol).as_bytes());
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (launch, resolved) in launches.iter().zip(&resolved) {
        resolved.submit(services, stream).inspect_err(|error| {
            let _ = services.log(format!("launch {} failed: {error:?}", launch.symbol).as_bytes());
        })?;
    }
    Ok(())
}

impl Lanes {
    /// Issues `unit`'s submissions of `program`. Once the fork reached the
    /// side stream, the join is issued last even after a failed submission,
    /// so the side stream never stays in an open capture; the first error
    /// is returned.
    pub fn issue(
        &self,
        services: &HostServicesV1,
        symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
        program: &[PreparedLaunch],
        unit: &[Op],
    ) -> Result<(), DispatchError> {
        let mut forked = false;
        let mut joined = false;
        let mut issued = Ok(());
        for op in unit {
            issued = match op {
                // Only an r2 sublayer's gate onward follows a join in its unit.
                Op::Compute(range) if joined => {
                    launch_after_join(services, self.compute, symbols, &program[range.clone()])
                }
                Op::Compute(range) => {
                    launch_all(services, self.compute, symbols, &program[range.clone()])
                }
                Op::Side(range) => {
                    launch_all(services, self.side, symbols, &program[range.clone()])
                }
                Op::Fork => {
                    let recorded = services.record_event(self.fork, self.compute);
                    forked = recorded.is_ok();
                    recorded
                        .and_then(|()| services.wait_event(self.side, self.fork))
                        .map_err(DispatchError::from)
                }
                Op::Join => {
                    forked = false;
                    joined = true;
                    self.join(services)
                }
            };
            if issued.is_err() {
                break;
            }
        }
        if forked {
            return issued.and(self.join(services));
        }
        issued
    }

    fn join(&self, services: &HostServicesV1) -> Result<(), DispatchError> {
        services
            .record_event(self.join, self.side)
            .and_then(|()| services.wait_event(self.compute, self.join))
            .map_err(DispatchError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Phase, Sublayer, stages};
    use bundle_runtime::dispatch::BufferBindings;
    use kernel_core::contract::{ArgumentList, LaunchPlan};
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn strict_prefill_audits_cover_pointer_arrays_maps_and_invalid_views() {
        use kernel_core::contract::{
            TensorMap, TensorMapElement, TensorMapInterleave, TensorMapL2Promotion,
            TensorMapOobFill, TensorMapSwizzle,
        };
        let span = |buffer, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(buffer, 1),
                offset,
                length,
            )
        };
        let launch = |view, kind| {
            let mut bindings = BufferBindings::default();
            let id = bindings.bind(view);
            // Unreferenced views are not accesses and must not enter the audit.
            bindings.bind(span(7, 128, 128));
            let pointer = match kind {
                0 => Argument::DeviceBuffer(id),
                1 => Argument::DevicePointerArray(vec![id, id]),
                _ => Argument::TensorMap(TensorMap {
                    base_buffer: id,
                    dims: vec![128, 64],
                    strides_bytes: vec![1, 128],
                    box_dims: vec![128, 1],
                    element_strides: vec![1, 1],
                    swizzle: TensorMapSwizzle::B128,
                    interleave: TensorMapInterleave::None,
                    l2_promotion: TensorMapL2Promotion::B256,
                    oob_fill: TensorMapOobFill::Zero,
                    element: TensorMapElement::Fp8E4M3,
                }),
            };
            PreparedLaunch {
                // Even a const input may not alias in the strict prefill audit.
                symbol: "chop_deepseek41_mxfp8_prefetch_n576_k5120_g1_pair_f32_v3".into(),
                rows: 1,
                plan: LaunchPlan {
                    grid: [1; 3],
                    block: [32, 1, 1],
                    cluster: [1; 3],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    pdl: true,
                },
                arguments: ArgumentList(vec![pointer]),
                bindings,
            }
        };
        let side = [launch(span(7, 128, 128), 0)];
        for kind in 0..3 {
            for (view, safe) in [
                (span(7, 256, 128), true),
                (span(7, 0, 128), true),
                (span(8, 128, 128), true),
                (span(7, 128, 0), true),
                (span(7, 255, 128), false),
                (span(7, 0, 129), false),
                (span(7, 128, 128), false),
                (span(8, u64::MAX, 1), false),
            ] {
                let main = [launch(view, kind)];
                assert_eq!(disjoint_branches(&side, &main).is_ok(), safe);
                assert_eq!(
                    disjoint_copy_spans(&[span(7, 128, 128)], &main).is_ok(),
                    safe
                );
            }
        }
        let mut missing = launch(span(8, 0, 128), 0);
        missing.arguments.0 = vec![Argument::DeviceBuffer(u64::MAX)];
        assert!(disjoint_branches(&side, std::slice::from_ref(&missing)).is_err());
        assert!(disjoint_copy_spans(&[span(7, 128, 128)], &[missing]).is_err());
        assert!(disjoint_copy_spans(&[span(8, u64::MAX, 1)], &side).is_err());
    }

    #[test]
    fn shared_overlap_checks_resolved_ranges_before_the_fork() {
        let span = |handle, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(handle, 1),
                offset,
                length,
            )
        };
        let stages = crate::program::moe(0);
        let mut program = stages
            .iter()
            .enumerate()
            .map(|(i, stage)| PreparedLaunch {
                symbol: format!("{stage:?}"),
                rows: 1,
                plan: LaunchPlan {
                    grid: [1; 3],
                    block: [32, 1, 1],
                    cluster: [1; 3],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    pdl: true,
                },
                arguments: ArgumentList(vec![Argument::U32(i as u32)]),
                bindings: BufferBindings::default(),
            })
            .collect::<Vec<_>>();
        let shared = stages
            .iter()
            .position(|&s| s == Stage::SharedGateUp(0))
            .unwrap();
        let bind = |launch: &mut PreparedLaunch, view| {
            let id = launch.bindings.bind(view);
            launch.arguments = ArgumentList(vec![Argument::DevicePointerArray(vec![id])]);
        };
        // The router and shared pair's flags and partial sums have different
        // planner IDs but intersect in the same physical allocation.
        bind(&mut program[0], span(7, 128, 61_440));
        bind(&mut program[shared], span(7, 256, 184_320));
        assert!(
            checked_units(&stages, &program, &[], false)
                .unwrap_err()
                .contains("unordered scratch alias")
        );
        bind(&mut program[shared], span(8, 256, 184_320));
        assert!(checked_units(&stages, &program, &[], false).is_ok());
        // Disjoint suballocations in one arena are safe, including adjacency.
        bind(&mut program[shared], span(7, 61_568, 4));
        assert!(checked_units(&stages, &program, &[], false).is_ok());
        bind(&mut program[shared], span(7, 61_567, 4));
        assert!(checked_units(&stages, &program, &[], false).is_err());
        let input = span(9, 512, 1024);
        let bind_at = |launch: &mut PreparedLaunch, index, view| {
            launch.symbol = "chop_deepseek41_mxfp8_prefetch_n576_k5120_g1_pair_f32_v3".into();
            let id = launch.bindings.bind(view);
            launch.arguments = ArgumentList(vec![Argument::U32(0); 12]);
            launch.arguments.0[index] = Argument::DeviceBuffer(id);
        };
        // Argument zero is const input; argument six is a mutable output.
        bind_at(&mut program[0], 0, input);
        bind_at(&mut program[shared], 0, span(9, 640, 128));
        assert!(checked_units(&stages, &program, &[input], false).is_ok());
        bind_at(&mut program[shared], 6, span(9, 640, 128));
        assert!(checked_units(&stages, &program, &[input], false).is_err());
        bind_at(&mut program[shared], 0, span(9, 640, 128));
        bind_at(&mut program[0], 6, input);
        assert!(checked_units(&stages, &program, &[input], false).is_err());
        bind_at(&mut program[0], 0, input);
        assert!(checked_units(&stages, &program, &[span(9, 640, 128)], false).is_err());
        bind(&mut program[shared], span(9, u64::MAX, 4));
        assert!(checked_units(&stages, &program, &[input], false).is_err());
    }

    #[test]
    fn an_r2_router_finish_joins_the_shared_expert_before_the_gate() {
        let span = |handle, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(handle, 1),
                offset,
                length,
            )
        };
        let stages = crate::program::moe(0);
        let at = |stage| stages.iter().position(|&s| s == stage).unwrap();
        let (route, gate, down, shared) = (
            at(Stage::Route(0)),
            at(Stage::ExpertsGateUp(0)),
            at(Stage::ExpertsDown(0)),
            at(Stage::SharedGateUp(0)),
        );
        let program = |finish: &str| {
            stages
                .iter()
                .enumerate()
                .map(|(i, stage)| PreparedLaunch {
                    symbol: if i == route {
                        finish.to_string()
                    } else {
                        format!("{stage:?}")
                    },
                    rows: 1,
                    plan: LaunchPlan {
                        grid: [1; 3],
                        block: [32, 1, 1],
                        cluster: [1; 3],
                        dynamic_shared_bytes: 0,
                        workspace_bytes: 0,
                        pdl: true,
                    },
                    arguments: ArgumentList(vec![Argument::U32(i as u32)]),
                    bindings: BufferBindings::default(),
                })
                .collect::<Vec<_>>()
        };
        let served = "chop_deepseek41_moe_router_finish_e384_k6_finish_v1";
        let r2 = "chop_deepseek41_moe_router_finish_e384_k6_finish_r2v7_v1";
        assert!(!joins_before_gate(served));
        let units_of = |p: &[PreparedLaunch]| checked_units(&stages, p, &[], true);
        assert_eq!(
            units_of(&program(served)).unwrap()[0],
            vec![
                Op::Fork,
                Op::Side(shared..stages.len() - 1),
                Op::Compute(0..shared),
                Op::Join
            ]
        );
        let edge = units_of(&program(r2)).unwrap();
        assert_eq!(
            edge[0],
            vec![
                Op::Fork,
                Op::Side(shared..stages.len() - 1),
                Op::Compute(0..gate),
                Op::Join,
                Op::Compute(gate..shared),
            ]
        );
        assert_eq!(edge.len(), units_of(&program(served)).unwrap().len());
        let bind = |launch: &mut PreparedLaunch, view| {
            let id = launch.bindings.bind(view);
            launch.arguments = ArgumentList(vec![Argument::DevicePointerArray(vec![id])]);
        };
        // After the join the routed down may reuse the shared expert's
        // scratch; the router, still concurrent, may not.
        for (finish, ordered) in [(r2, true), (served, false)] {
            let mut p = program(finish);
            bind(&mut p[down], span(7, 0, 256));
            bind(&mut p[shared], span(7, 128, 256));
            assert_eq!(units_of(&p).is_ok(), ordered, "{finish}");
            let mut p = program(finish);
            bind(&mut p[0], span(7, 0, 256));
            bind(&mut p[shared], span(7, 128, 256));
            assert!(units_of(&p).is_err(), "{finish}");
        }
    }

    #[test]
    fn shared_overlap_keeps_each_launch_and_the_input_and_sum_dependencies() {
        for phase in [
            Phase::Decode(1),
            Phase::Decode(32),
            Phase::Verify {
                rows: 192,
                items: 32,
            },
        ] {
            // Include repeated tags: lowering may emit several kernels per stage.
            let stages = stages(&FLASH, phase)
                .into_iter()
                .flat_map(|s| [s, s])
                .collect::<Vec<_>>();
            for kv in [false, true] {
                let mut seen = vec![0; stages.len()];
                let (mut moe, mut attention) = (0, 0);
                for unit in units(&stages, kv).unwrap() {
                    if unit.len() == 4 {
                        let [Op::Fork, Op::Side(side), Op::Compute(compute), Op::Join] = &unit[..]
                        else {
                            panic!("whole fork/join region")
                        };
                        match stages[compute.start] {
                            Stage::Project(layer, "moe.router") => {
                                moe += 1;
                                assert_eq!(
                                    stages[compute.start - 1],
                                    Stage::QuantizeInput(layer, Sublayer::Mlp)
                                );
                                assert_eq!(stages[side.end], Stage::RoutedSum(layer));
                                assert!(stages[side.clone()].iter().all(|&s| matches!(
                                    s,
                                    Stage::SharedGateUp(_)
                                        | Stage::SharedActivation(_)
                                        | Stage::SharedDown(_)
                                )));
                            }
                            Stage::Project(layer, "attention.q-a") => {
                                attention += 1;
                                // The fork follows the quantized input both
                                // branches read; the join precedes the
                                // window KV stage that reads the projection.
                                assert_eq!(
                                    stages[compute.start - 1],
                                    Stage::QuantizeInput(layer, Sublayer::Attention)
                                );
                                assert_eq!(compute.end, side.start);
                                assert_eq!(stages[compute.end - 1], Stage::QueryRope(layer));
                                assert_eq!(stages[side.end], Stage::WindowKv(layer));
                                assert!(
                                    stages[side.clone()]
                                        .iter()
                                        .all(|&s| s == Stage::Project(layer, "attention.kv"))
                                );
                            }
                            other => panic!("unexpected region start {other:?}"),
                        }
                    }
                    for op in unit {
                        if let Op::Compute(range) | Op::Side(range) = op {
                            for i in range {
                                seen[i] += 1;
                            }
                        }
                    }
                }
                assert!(moe > 0);
                assert_eq!(attention, if kv { FLASH.layers() as usize } else { 0 });
                assert!(seen.iter().all(|&count| count == 1));
            }
        }
    }

    #[test]
    fn the_window_kv_projection_runs_beside_the_query_chain_only() {
        let span = |handle, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(handle, 1),
                offset,
                length,
            )
        };
        let stages = crate::program::attention(&FLASH, 3, Phase::Decode(32));
        let at = |stage| stages.iter().position(|&s| s == stage).unwrap();
        let (kv, window) = (
            at(Stage::Project(3, "attention.kv")),
            at(Stage::WindowKv(3)),
        );
        let fresh = || {
            stages
                .iter()
                .enumerate()
                .map(|(i, stage)| PreparedLaunch {
                    symbol: format!("{stage:?}"),
                    rows: 32,
                    plan: LaunchPlan {
                        grid: [1; 3],
                        block: [32, 1, 1],
                        cluster: [1; 3],
                        dynamic_shared_bytes: 0,
                        workspace_bytes: 0,
                        pdl: true,
                    },
                    arguments: ArgumentList(vec![Argument::U32(i as u32)]),
                    bindings: BufferBindings::default(),
                })
                .collect::<Vec<_>>()
        };
        let mut program = fresh();
        let planned = checked_units(&stages, &program, &[], true).unwrap();
        assert_eq!(
            planned[0],
            vec![Op::Fork, Op::Side(kv..window), Op::Compute(0..kv), Op::Join]
        );
        assert_eq!(planned[1], vec![Op::Compute(window..window + 1)]);
        assert_eq!(
            checked_units(&stages, &program, &[], false).unwrap()[0],
            vec![Op::Compute(0..1)]
        );
        let bind = |launch: &mut PreparedLaunch, view| {
            let id = launch.bindings.bind(view);
            launch.arguments = ArgumentList(vec![Argument::DevicePointerArray(vec![id])]);
        };
        // The query chain and the projection share no scratch; the window
        // KV stage after the join may read the projection's output.
        bind(&mut program[kv], span(7, 0, 256));
        bind(&mut program[window], span(7, 0, 256));
        assert!(checked_units(&stages, &program, &[], true).is_ok());
        for chain in 0..kv {
            let mut p = fresh();
            bind(&mut p[kv], span(7, 0, 256));
            bind(&mut p[chain], span(7, 128, 256));
            assert!(checked_units(&stages, &p, &[], true).is_err());
            assert!(checked_units(&stages, &p, &[], false).is_ok());
        }
        // A chain without its projection, as at the dual C1 fold, stays serial.
        let without: Vec<_> = stages
            .iter()
            .copied()
            .filter(|&s| s != stages[kv])
            .collect();
        assert!(
            units(&without, true)
                .unwrap()
                .iter()
                .all(|unit| unit.len() == 1)
        );
    }

    #[test]
    fn shared_overlap_refuses_incomplete_shared_sequences() {
        let stages = crate::program::moe(0);
        for omitted in 1..stages.len() {
            let mut broken = stages.clone();
            broken.remove(omitted);
            assert!(
                units(&broken, true).is_err(),
                "missing {:?}",
                stages[omitted]
            );
        }
        assert!(Settings::read(None, None).serves(192));
        assert!(!Settings::read(Some("0"), None).serves(1));
        assert!(!Settings::read(Some("8"), None).serves(32));
        assert!(Settings::read(None, None).serves_kv(64));
        assert!(!Settings::read(None, None).serves_kv(65));
        assert!(!Settings::read(None, Some("0")).serves_kv(32));
        assert!(Settings::read(None, Some("0")).serves(32));
        assert!(!Settings::read(Some("0"), None).serves_kv(1));
        assert!(Settings::read(Some("16"), None).serves_kv(16));
        assert!(!Settings::read(Some("16"), None).serves_kv(32));
        assert_eq!(Settings::read(None, None), Settings::ALL);
        assert_eq!(Settings::read(Some("0"), Some("1")), Settings::OFF);
    }
}
