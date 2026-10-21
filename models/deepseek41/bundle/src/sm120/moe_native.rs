//! Default native block-scaled MoE prefill, a within-bound path that
//! changes bits (models/deepseek41/lab/native_moe_prefill.md). With
//! `CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL` unset or `1`, Flash TP4's target expert
//! bank runs its gate/up and down tiles through native MXFP4 x MXFP8 MMA
//! in every prefill step, whatever its row count, so a prompt row's
//! arithmetic still does not depend on the chunking. Routing, the gate/up
//! epilogue and the routed sum are the serving ones. In steps of at most
//! [`FUSED_CEIL_ROWS`] rows the intermediate ceil quantizer runs in the
//! gate/up tiles' epilogue (`native_grouped::Ceil`) with a fixup launch,
//! bit for bit the serving quantizer's codes, scales and status
//! (models/deepseek41/lab/native_ceil.md); larger steps keep the separate
//! quantizer, run gate/up through the wider r2w6s3 tile and down through
//! the 64-row r1w4s2 tile at two CTAs per SM, both with the lean inner
//! loop behind an activation admission scan (native_tune.md,
//! native_admission.md); every tile and loop gives the same bits and
//! refuses the same inputs. Decode,
//! verify and the drafter's bank are unchanged.
use super::moe_native_entries as entries;
use kernel_core::backend::{Counts, Launch, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::moe::native_grouped::{self as native, Shape, Tile, Variant};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    moe::Experts,
};

pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL";
pub const CUBIN: &str = entries::CUBIN;
pub const CEIL_CUBIN: &str = entries::CEIL_CUBIN;
pub const WIDE_CUBIN: &str = entries::WIDE_CUBIN;
pub const LEAN_CUBIN: &str = entries::LEAN_CUBIN;
/// The intermediate quantizer the gate/up op keeps after its tiles.
const QUANTIZER: &str = "_moe_ceil_";
/// The largest step whose gate/up runs the fused ceil. Its paired lab
/// timing (native_ceil.md, 250 W) was 0.85 of the separate quantizer's at
/// 512 rows and 0.95 at 2048, but 1.01 to 1.015 at 8192; the bits are the
/// same either way.
pub const FUSED_CEIL_ROWS: u32 = 2048;

fn selected(setting: Option<&str>) -> bool {
    matches!(setting, None | Some("1"))
}

/// Whether the instance selects native MoE prefill, on by default.
pub fn requested() -> bool {
    #[cfg(test)]
    return IN_TESTS.get();
    #[cfg(not(test))]
    match std::env::var(SWITCH) {
        Ok(value) => selected(Some(&value)),
        Err(std::env::VarError::NotPresent) => selected(None),
        Err(_) => false,
    }
}

#[cfg(test)]
thread_local! { static IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }

#[cfg(test)]
pub(crate) fn with_native<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(IN_TESTS.replace(enabled));
    run()
}

/// The topology and bank the entries compile for.
pub fn eligible(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub fn applies(g: &Geometry, ranks: u32, set: Experts, usage: OperationUse) -> bool {
    eligible(g, ranks)
        && set == entries::target_set()
        && usage.phase == ExecutionPhase::Prefill
        && !usage.graph
        && usage.capacity <= entries::CAPACITY
}

/// Replace `base`'s expert tiles with the native ones when requested and
/// `usage` is a prefill of the target bank; otherwise return `base`.
pub fn prepare(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
    base: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    if !requested() || !applies(g, ranks, set, usage) {
        return Ok(base);
    }
    wrap(kind, usage, base).map_err(Refused::from)
}

/// The wrapper itself, for the lab; it does not read the switch.
pub fn wrap(
    kind: GatedMlpProjection,
    usage: OperationUse,
    base: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refusal> {
    let gate_up = kind == GatedMlpProjection::GateUp;
    let shape = entries::shape(kind);
    let tile = entries::tile(kind);
    let symbols = &base.info().symbols;
    let split = symbols
        .iter()
        .position(|(s, _)| s.contains(QUANTIZER))
        .unwrap_or(symbols.len());
    if split == 0
        || gate_up != (split < symbols.len())
        || symbols[split..].iter().any(|(s, _)| !s.contains(QUANTIZER))
        || base.route_tile() != 64
    {
        return Err(Refusal(
            "native prefill needs grouped tiles and the ceil quantizer",
        ));
    }
    // Gate/up's fused steps keep the tile the fused ceil was qualified on;
    // unfused down steps run the two-CTA tile.
    let fused_tile = entries::GATE_UP_TILE;
    let unfused_tile = if gate_up {
        tile
    } else {
        entries::PAIRED_DOWN_TILE
    };
    let queue = [tile, fused_tile, unfused_tile]
        .iter()
        .map(|t| native::workspace_bytes(&shape, t))
        .max()
        .unwrap();
    let mut info = base.info().clone();
    info.implementation.push_str("+native_mxf8f6f4_prefill_v1");
    // The queue takes the workspace's first bytes. A base that carves
    // decoded operands from the workspace's end keeps its layout.
    let decoded = info.bytes_of("scratch.partial").unwrap_or(0);
    info.workspace = info.workspace.max(queue + decoded);
    for (name, bytes) in [
        (
            "routing.routes.counts".to_string(),
            u64::from(shape.experts) * 4,
        ),
        (
            "routing.routes.active_experts".to_string(),
            u64::from(shape.experts + 1) * 4,
        ),
    ] {
        if let Some(existing) = info.bytes.iter_mut().find(|(key, _)| key == &name) {
            existing.1 = existing.1.max(bytes);
        } else {
            info.bytes.push((name, bytes));
        }
    }
    // Gate/up: the wide pack and lean tiles, then the fused steps' pack,
    // tiles and fixup; it keeps its quantizer for the unfused steps. Down:
    // the served pack and tiles, then the unfused steps' two-CTA pack and
    // lean tiles.
    let lean = (
        native::tune_symbol(
            &shape,
            &unfused_tile,
            entries::UNFUSED_LOOP,
            Variant::Serving,
        ),
        native::shared_bytes(&shape, &unfused_tile),
    );
    let mut replaced = vec![(
        native::symbol(&shape, &tile, native::Stage::Pack, Variant::Serving),
        0,
    )];
    if gate_up {
        replaced.push(lean.clone());
    } else {
        replaced.push((
            native::symbol(&shape, &tile, native::Stage::Compute, Variant::Serving),
            native::shared_bytes(&shape, &tile),
        ));
        replaced.push((
            native::symbol(&shape, &unfused_tile, native::Stage::Pack, Variant::Serving),
            0,
        ));
        replaced.push(lean.clone());
    }
    if gate_up {
        let ceil = entries::ceil();
        replaced.push((
            native::symbol(&shape, &fused_tile, native::Stage::Pack, Variant::Serving),
            0,
        ));
        replaced.push((
            ceil.tiles_symbol(&shape, &fused_tile),
            native::shared_bytes(&shape, &fused_tile),
        ));
        replaced.push((ceil.fixup_symbol(&shape), 0));
    }
    // The unfused steps' admission scan.
    replaced.push((native::admission_scan_symbol(&shape), 0));
    info.symbols.splice(..split, replaced);
    Ok(Box::new(Prepared {
        base,
        shape,
        tile,
        fused_tile,
        unfused_tile,
        info,
        capacity: usage.capacity,
    }))
}

struct Prepared {
    base: Box<dyn PreparedExperts>,
    shape: Shape,
    tile: Tile,
    fused_tile: Tile,
    /// The tile of steps above [`FUSED_CEIL_ROWS`], run with the lean loop.
    unfused_tile: Tile,
    info: OpInfo,
    capacity: u32,
}

impl PreparedExperts for Prepared {
    fn route_tile(&self) -> u32 {
        self.base.route_tile()
    }
    fn padded_rows(&self) -> u32 {
        self.base.padded_rows()
    }
}

impl PreparedOp<GatedMlpBuffers<View>> for Prepared {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("native prefill rows exceed the prepared capacity"));
        }
        workspace.covers(self.info.workspace)?;
        // The base validates every operand and supplies the quantizer.
        let mut old = self.base.launches(b, workspace, counts)?;
        let split = old
            .iter()
            .position(|l| l.symbol.contains(QUANTIZER))
            .unwrap_or(old.len());
        if split == 0
            || self.shape.gate_up != (split < old.len())
            || old[split..].iter().any(|l| !l.symbol.contains(QUANTIZER))
        {
            return Err(Refusal("native prefill base launch sequence changed"));
        }
        let quantizer = old.split_off(split);
        let routing = b.routing.ok_or(Refusal("native prefill requires routes"))?;
        let route_counts = routing
            .routes
            .counts
            .ok_or(Refusal("native prefill requires route counts"))?;
        let active = routing
            .routes
            .active_experts
            .ok_or(Refusal("native prefill requires active experts"))?;
        route_counts.covers(self.info.bytes_of("routing.routes.counts").unwrap())?;
        active.covers(self.info.bytes_of("routing.routes.active_experts").unwrap())?;
        let required =
            |v: Option<View>| v.ok_or(Refusal("native prefill requires scaled operands"));
        let buffers = native::Buffers {
            input: b.activation.handle,
            weights: b.weights.codes.handle,
            weight_scales: required(b.weights.local_scales)?.handle,
            activation_scales: required(b.activation_scales)?.handle,
            offsets: routing.routes.offsets.handle,
            sorted_rows: routing.routes.sorted_rows.handle,
            padded_rows: routing.routes.padded_rows.handle,
            live_tokens: b.live_tokens.handle,
            route_weights: if self.shape.gate_up {
                routing.weights.handle
            } else {
                0
            },
            output: b.output.handle,
            counts: route_counts.handle,
            active: active.handle,
            scratch: workspace.handle,
            raw: 0,
        };
        let operands = [
            buffers.input,
            buffers.weights,
            buffers.weight_scales,
            buffers.activation_scales,
            buffers.offsets,
            buffers.sorted_rows,
            buffers.padded_rows,
            buffers.live_tokens,
            routing.weights.handle,
            buffers.output,
            buffers.counts,
            buffers.active,
        ];
        if workspace.handle == 0 || operands.contains(&workspace.handle) {
            return Err(Refusal("native prefill queue aliases an operand"));
        }
        let small = counts.rows <= FUSED_CEIL_ROWS;
        let fused = self.shape.gate_up && small;
        let mut launches = if fused {
            let outputs = fused_outputs(&quantizer, b)?;
            entries::ceil().launches(
                &self.shape,
                &self.fused_tile,
                counts.rows,
                &buffers,
                outputs,
            )?
        } else if small {
            native::launches(
                &self.shape,
                &self.tile,
                counts.rows,
                &buffers,
                Variant::Serving,
            )?
        } else {
            native::tune_launches(
                &self.shape,
                &self.unfused_tile,
                counts.rows,
                &buffers,
                entries::UNFUSED_LOOP,
                Variant::Serving,
            )?
        };
        // The pack and the tiles carry the queue's workspace; the unfused
        // steps' admission scan before them uses none.
        let scan = native::admission_scan_symbol(&self.shape);
        let first = usize::from(launches[0].symbol == scan);
        if first == 1 {
            launches[0].plan.workspace_bytes = 0;
        }
        for op in &mut launches[first..first + 2] {
            op.plan.workspace_bytes = self.info.workspace;
        }
        if !fused {
            launches.extend(quantizer);
        }
        Ok(launches)
    }
}

/// The serving quantizer's codes, scales and status, which the fused tiles
/// and fixup write instead: its one launch must read gate/up's output rows
/// and live count and update status rows from zero.
fn fused_outputs(
    quantizer: &[Launch],
    b: &GatedMlpBuffers<View>,
) -> Result<(u64, u64, u64), Refusal> {
    use kernel_core::contract::Argument::{DeviceBuffer, U64};
    let [q] = quantizer else {
        return Err(Refusal("the fused ceil replaces exactly one quantizer"));
    };
    match q.arguments.0.as_slice() {
        [
            DeviceBuffer(input),
            DeviceBuffer(codes),
            DeviceBuffer(scales),
            DeviceBuffer(live),
            DeviceBuffer(status),
            U64(0),
            ..,
        ] if *input == b.output.handle
            && *live == b.live_tokens.handle
            && Some(*codes) == b.scratch.output_codes.map(|v| v.handle)
            && Some(*scales) == b.scratch.output_scales.map(|v| v.handle)
            && Some(*status) == b.scratch.quantizer_status.map(|v| v.handle)
            && b.scratch.quantizer_status_row_offset == 0 =>
        {
            Ok((*codes, *scales, *status))
        }
        _ => Err(Refusal(
            "the fused ceil needs the quantizer of gate/up's rows at status row zero",
        )),
    }
}

impl crate::Kernels {
    pub(crate) fn native_moe_prefill(
        shape: &crate::shape::Shape,
        set: Experts,
        kind: GatedMlpProjection,
        usage: OperationUse,
        base: Box<dyn PreparedExperts>,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        prepare(shape.geometry, shape.ranks(), set, kind, usage, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use kernel_core::moe::RouteMetadata;
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
    use kernels_sm120::backend::Sm120;
    use kernels_sm120::moe::native_grouped::Tune;
    use model_deepseek41_contract::moe;

    #[test]
    fn switch_defaults_on_and_keeps_the_explicit_opt_out() {
        assert!(selected(None));
        assert!(selected(Some("1")));
        for value in [Some("0"), Some(""), Some("true")] {
            assert!(!selected(value));
        }
        assert!(requested());
        assert!(!with_native(false, requested));
        assert!(with_native(true, requested));
        assert!(requested());
    }

    #[test]
    fn applies_to_every_target_prefill_and_nothing_else() {
        let target = entries::target_set();
        for set in moe::expert_sets(&FLASH) {
            for phase in [
                ExecutionPhase::Prefill,
                ExecutionPhase::Decode,
                ExecutionPhase::Verify,
            ] {
                for capacity in [1, 65, 512, 2048, 4096, 8192, 8193] {
                    for ranks in [1, 2, 4] {
                        let usage = OperationUse {
                            phase,
                            capacity,
                            graph: phase != ExecutionPhase::Prefill,
                        };
                        assert_eq!(
                            applies(&FLASH, ranks, set, usage),
                            ranks == 4
                                && set == target
                                && phase == ExecutionPhase::Prefill
                                && capacity <= 8192
                        );
                    }
                }
            }
        }
        let mut other = FLASH;
        other.id = "unqualified-same-shape";
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        assert!(!applies(&other, 4, target, usage));
    }

    fn views(op: &dyn PreparedExperts) -> (GatedMlpBuffers<View>, View) {
        let mut handle = 0;
        let mut sized = |name: &str| {
            handle += 1;
            View {
                handle,
                bytes: op.info().bytes_of(name).unwrap_or(1 << 30),
            }
        };
        let b = GatedMlpBuffers {
            activation: sized("activation"),
            activation_scales: Some(sized("activation_scales")),
            activation_global: None,
            live_tokens: sized("live_tokens"),
            output: sized("output"),
            weights: QuantizedWeights {
                codes: sized("weights.codes"),
                local_scales: Some(sized("weights.local_scales")),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: sized("routing.weights"),
                routes: RouteMetadata {
                    offsets: sized("routing.routes.offsets"),
                    sorted_rows: sized("routing.routes.sorted_rows"),
                    padded_rows: sized("routing.routes.padded_rows"),
                    counts: Some(sized("routing.routes.counts")),
                    active_experts: Some(sized("routing.routes.active_experts")),
                },
            }),
            scratch: GatedMlpScratch {
                partial: op
                    .info()
                    .bytes_of("scratch.partial")
                    .map(|_| sized("scratch.partial")),
                output_codes: Some(sized("scratch.output_codes")),
                output_scales: Some(sized("scratch.output_scales")),
                quantizer_status: Some(sized("scratch.quantizer_status")),
                quantizer_status_row_offset: 0,
                unused: View::EMPTY,
            },
        };
        (
            b,
            View {
                handle: 1000,
                bytes: op.info().workspace,
            },
        )
    }

    #[test]
    fn wraps_every_prefill_kind_and_fuses_the_quantizer() {
        let backend = super::super::moe::register(Sm120::new(), &FLASH, 4).unwrap();
        let set = entries::target_set();
        // The exact buckets and the general prefill kind (any row count).
        for (capacity, rows, bucket) in [
            (8192, 8192, true),
            (8192, 65, false),
            (8192, 2048, false),
            (8192, 2049, false),
            (8192, 8192, false),
            (4096, 4096, true),
            (2048, 2048, true),
            (512, 512, true),
        ] {
            let usage = OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity,
                graph: false,
            };
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                let query = moe::projection(&FLASH, 4, set, kind, capacity).unwrap();
                // The serving baseline, including the measured vector gate/up.
                let serving = || {
                    let base = backend
                        .chained_mlp_projection(&query, usage, Need::rows(capacity))
                        .unwrap();
                    if bucket {
                        super::super::moe_pipeline::prepare(&FLASH, 4, set, kind, usage, base)
                            .unwrap()
                    } else {
                        base
                    }
                };
                let base = serving();
                let base_symbols = base.info().symbols.clone();
                let (b, workspace) = views(base.as_ref());
                let old = base.launches(&b, workspace, Counts::rows(rows)).unwrap();
                let unchanged =
                    with_native(false, || prepare(&FLASH, 4, set, kind, usage, serving())).unwrap();
                assert_eq!(unchanged.info().symbols, base_symbols);
                let op = with_native(true, || prepare(&FLASH, 4, set, kind, usage, base)).unwrap();
                let quantizer = base_symbols
                    .iter()
                    .filter(|(s, _)| s.contains(QUANTIZER))
                    .cloned()
                    .collect::<Vec<_>>();
                let gate_up = kind == GatedMlpProjection::GateUp;
                assert_eq!(quantizer.len(), usize::from(gate_up));
                // Down: the served pack and tiles, then the unfused steps'
                // two-CTA pack and scanned tiles. Gate/up: the wide pack and
                // scanned tiles, the fused steps' pack, tiles and fixup.
                // Then the unfused steps' admission scan, and gate/up's
                // quantizer of the unfused steps.
                let symbols = &op.info().symbols;
                assert_eq!(symbols.len(), 5 + 2 * usize::from(gate_up));
                let scan = (native::admission_scan_symbol(&entries::shape(kind)), 0);
                assert_eq!(symbols[if gate_up { 5 } else { 4 }], scan);
                let shape = entries::shape(kind);
                let tile = if gate_up {
                    entries::WIDE_GATE_UP_TILE
                } else {
                    entries::DOWN_TILE
                };
                assert_eq!(entries::tile(kind), tile);
                let pack = |t: &Tile| {
                    (
                        native::symbol(&shape, t, native::Stage::Pack, Variant::Serving),
                        0,
                    )
                };
                assert_eq!(entries::UNFUSED_LOOP, Tune::Scanned);
                let lean = |t: &Tile| {
                    (
                        native::tune_symbol(&shape, t, entries::UNFUSED_LOOP, Variant::Serving),
                        native::shared_bytes(&shape, t),
                    )
                };
                if gate_up {
                    assert_eq!(symbols[..2], [pack(&tile), lean(&tile)]);
                } else {
                    let paired = entries::PAIRED_DOWN_TILE;
                    assert_eq!(
                        symbols[..4],
                        [
                            pack(&tile),
                            (
                                native::symbol(
                                    &shape,
                                    &tile,
                                    native::Stage::Compute,
                                    Variant::Serving
                                ),
                                native::shared_bytes(&shape, &tile)
                            ),
                            pack(&paired),
                            lean(&paired),
                        ]
                    );
                    assert_eq!(
                        native::grid(&shape, &paired),
                        2 * native::grid(&shape, &tile)
                    );
                }
                if gate_up {
                    let fused = entries::GATE_UP_TILE;
                    let ceil = entries::ceil();
                    assert_ne!(fused, tile);
                    assert_eq!(
                        symbols[2..5],
                        [
                            (
                                native::symbol(
                                    &shape,
                                    &fused,
                                    native::Stage::Pack,
                                    Variant::Serving
                                ),
                                0
                            ),
                            (
                                ceil.tiles_symbol(&shape, &fused),
                                native::shared_bytes(&shape, &fused)
                            ),
                            (ceil.fixup_symbol(&shape), 0),
                        ]
                    );
                    assert_eq!(symbols[6..], quantizer[..]);
                } else {
                    assert!(symbols.iter().all(|(s, _)| !s.contains(QUANTIZER)));
                }
                assert!(
                    op.info()
                        .implementation
                        .ends_with("+native_mxf8f6f4_prefill_v1")
                );
                for (symbol, _) in &op.info().symbols {
                    assert!(
                        crate::sm120::fold_cubin_for_symbol(symbol).is_some(),
                        "{symbol}"
                    );
                }
                let (b, workspace) = views(op.as_ref());
                let all = op.launches(&b, workspace, Counts::rows(rows)).unwrap();
                // Unfused steps start with the admission scan of the input
                // and its scales, with no workspace.
                let unfused = rows > FUSED_CEIL_ROWS;
                assert_eq!(all.len(), 2 + usize::from(gate_up) + usize::from(unfused));
                if unfused {
                    use kernel_core::contract::Argument;
                    assert_eq!(all[0].symbol, scan.0);
                    assert_eq!(all[0].plan.workspace_bytes, 0);
                    // The tiles' input, activation scales and live count.
                    let tiles = &all[2].arguments.0;
                    assert_eq!(
                        all[0].arguments.0,
                        [tiles[0].clone(), tiles[3].clone(), tiles[6].clone()]
                    );
                    assert!(
                        tiles[..7]
                            .iter()
                            .all(|a| matches!(a, Argument::DeviceBuffer(_)))
                    );
                }
                let launches = &all[usize::from(unfused)..];
                assert!(
                    launches
                        .iter()
                        .all(|l| op.info().symbols.iter().any(|(s, _)| s == &l.symbol))
                );
                let fused = gate_up && rows <= FUSED_CEIL_ROWS;
                // The launches' pack and tiles: gate/up's fused ones or the
                // wide lean ones; down's served ones up to the fused steps'
                // rows, the two-CTA lean ones above.
                let first = if fused || (!gate_up && rows > FUSED_CEIL_ROWS) {
                    2
                } else {
                    0
                };
                assert_eq!(launches[0].symbol, op.info().symbols[first].0);
                assert_eq!(launches[1].symbol, op.info().symbols[first + 1].0);
                if fused {
                    // The fused tiles and the fixup write the quantizer's
                    // codes, scales and status.
                    use kernel_core::contract::Argument;
                    let q = &old.last().unwrap().arguments.0;
                    assert!(old.last().unwrap().symbol.contains(QUANTIZER));
                    assert_eq!(launches[2].symbol, op.info().symbols[4].0);
                    let tiles = &launches[1].arguments.0;
                    assert_eq!(
                        tiles[tiles.len() - 3..],
                        [q[1].clone(), q[2].clone(), q[4].clone()]
                    );
                    assert_eq!(
                        launches[2].arguments.0,
                        [
                            Argument::DeviceBuffer(b.live_tokens.handle),
                            q[4].clone(),
                            q[1].clone(),
                            q[2].clone()
                        ]
                    );
                    assert_eq!(launches[2].plan.grid, [rows, 1, 1]);
                    let mut offset = b;
                    offset.scratch.quantizer_status_row_offset = 1;
                    assert!(op.launches(&offset, workspace, Counts::rows(rows)).is_err());
                } else {
                    // The served tiles, then the base's quantizer unchanged.
                    if gate_up {
                        assert_eq!(launches[2], *old.last().unwrap());
                    }
                }
                assert_eq!(
                    launches[1].plan.dynamic_shared_bytes,
                    op.info().symbols[first + 1].1
                );
                assert!(
                    launches[..2]
                        .iter()
                        .all(|l| l.plan.workspace_bytes == op.info().workspace)
                );
                assert!(
                    op.launches(&b, workspace, Counts::rows(capacity + 1))
                        .is_err()
                );
                assert!(
                    op.launches(
                        &b,
                        View {
                            bytes: workspace.bytes - 1,
                            ..workspace
                        },
                        Counts::rows(rows)
                    )
                    .is_err()
                );
                let mut alias = b;
                alias.output.handle = workspace.handle;
                assert!(op.launches(&alias, workspace, Counts::rows(rows)).is_err());
            }
        }
        // Decode is never wrapped.
        let usage = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        let query = moe::projection(&FLASH, 4, set, GatedMlpProjection::Down, 8).unwrap();
        let base = backend
            .chained_mlp_projection(&query, usage, Need::rows(8))
            .unwrap();
        let symbols = base.info().symbols.clone();
        let op = with_native(true, || {
            prepare(&FLASH, 4, set, GatedMlpProjection::Down, usage, base)
        })
        .unwrap();
        assert_eq!(op.info().symbols, symbols);
    }
}
