//! Routed expert ops over the TP4 backbone gate/up bank in the layout the
//! served plan stores (`moe_row_tiles`). With the bank in row tiles a
//! gate/up op launches each entry's row-tiled build under its symbol and
//! refuses any other launch that reads the bank, so no step can read tiles
//! as rows. Native-off prefill refuses at prepare: its exact kernels read
//! the bank as rows.
use super::moe_row_tiles;
use kernel_core::backend::{Counts, DeferredBf16Round, Launch, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, Refusal};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::layout::ROW_TILES_64X128;
use kernel_core::precision::{ExecutionPhase, OperationUse, WeightLayout};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe::Experts;
use model_deepseek41_contract::weight_plan::RoutedLayout;
use std::collections::BTreeMap;

/// Why a tiled install refuses native-off prefill.
pub const NATIVE_OFF: &str = "DeepSeek's TP4 backbone gate/up bank is stored in row tiles, \
which only native MoE prefill reads: CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL=0 selects exact \
prefill kernels that read it as rows";

/// The layout the gate/up consumers of `set` read at `ranks`: the served
/// plan's for the TP4 backbone bank, rows for every other bank and rank
/// count.
pub fn layout(g: &Geometry, ranks: u32, set: Experts) -> RoutedLayout {
    if ranks == moe_row_tiles::RANKS
        && set.count == moe_row_tiles::EXPERTS
        && super::moe::streamed_qualified(g, ranks)
    {
        RoutedLayout::served(ranks)
    } else {
        RoutedLayout::Rows
    }
}

/// `op`, a routed expert projection of `set` at `ranks`, reading its bank
/// in the layout [`layout`] names.
pub fn prepare(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
    op: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    prepare_for(
        g,
        layout(g, ranks, set),
        kind,
        usage,
        super::moe_native::requested(),
        op,
    )
}

fn prepare_for(
    g: &Geometry,
    layout: RoutedLayout,
    kind: GatedMlpProjection,
    usage: OperationUse,
    native: bool,
    op: Box<dyn PreparedExperts>,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    if layout != RoutedLayout::Tiles || kind != GatedMlpProjection::GateUp {
        return Ok(op);
    }
    if usage.phase == ExecutionPhase::Prefill && !native {
        return Err(Refused::Rule(NATIVE_OFF.into()));
    }
    let symbols: BTreeMap<String, String> = moe_row_tiles::builds_for(g, layout)
        .map_err(Refused::Rule)?
        .into_iter()
        .map(|(rows, tiled)| (rows, tiled.symbol))
        .collect();
    let mut info = op.info().clone();
    for (symbol, _) in &mut info.symbols {
        if let Some(tiled) = symbols.get(symbol) {
            symbol.clone_from(tiled);
        }
    }
    let tiles = WeightLayout {
        id: ROW_TILES_64X128.into(),
        version: 1,
    };
    info.weight_layouts.retain(|(role, _)| role != "weights");
    info.weight_layouts.push(("weights".into(), tiles));
    Ok(Box::new(RowTiled { op, info, symbols }))
}

struct RowTiled {
    op: Box<dyn PreparedExperts>,
    info: OpInfo,
    /// Each row-major symbol with its row-tiled build's.
    symbols: BTreeMap<String, String>,
}

impl RowTiled {
    /// `launches` with each row-major symbol renamed to its row-tiled
    /// build's; a launch of another entry over the bank is refused.
    fn tiled(
        &self,
        b: &GatedMlpBuffers<View>,
        mut launches: Vec<Launch>,
    ) -> Result<Vec<Launch>, Refusal> {
        let bank: Vec<Argument> = [Some(b.weights.codes), b.weights.local_scales]
            .into_iter()
            .flatten()
            .map(|v| Argument::DeviceBuffer(v.handle))
            .collect();
        for launch in &mut launches {
            if let Some(tiled) = self.symbols.get(&launch.symbol) {
                launch.symbol.clone_from(tiled);
            } else if launch.arguments.0.iter().any(|a| bank.contains(a)) {
                return Err(Refusal(
                    "a row-major entry would read the row-tiled backbone gate/up bank",
                ));
            }
        }
        Ok(launches)
    }
}

impl PreparedExperts for RowTiled {
    fn route_tile(&self) -> u32 {
        self.op.route_tile()
    }
    fn padded_rows(&self) -> u32 {
        self.op.padded_rows()
    }
}

impl PreparedOp<GatedMlpBuffers<View>> for RowTiled {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn defer_bf16_round(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Option<DeferredBf16Round>, Refusal> {
        match self.op.defer_bf16_round(b, workspace, counts)? {
            Some(mut round) => {
                round.launches = self.tiled(b, round.launches)?;
                Ok(Some(round))
            }
            None => Ok(None),
        }
    }
    fn initialize(&self, b: &GatedMlpBuffers<View>) -> Result<Vec<Launch>, Refusal> {
        self.tiled(b, self.op.initialize(b)?)
    }
    fn launches(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.tiled(b, self.op.launches(b, workspace, counts)?)
    }
}

impl crate::Kernels {
    /// The layout the gate/up consumers of `set` read its bank in
    /// ([`layout`]).
    pub(crate) fn moe_gate_up_layout(shape: &crate::shape::Shape, set: Experts) -> RoutedLayout {
        layout(shape.geometry, shape.ranks(), set)
    }

    /// `op`, a routed expert projection, reading the bank of `set` in the
    /// layout the shape's served plan stores ([`layout`]).
    pub(crate) fn row_tiled_moe_projection(
        shape: &crate::shape::Shape,
        set: Experts,
        kind: GatedMlpProjection,
        usage: OperationUse,
        op: Box<dyn PreparedExperts>,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        prepare(shape.geometry, shape.ranks(), set, kind, usage, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ArgumentList, LaunchPlan};
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::buffers::QuantizedWeights;
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::moe;

    /// An op launching `symbols`, each with the arguments its index names:
    /// 1 the bank's codes, 2 its scales, 3 another buffer.
    struct Fake(Vec<(String, u64)>);
    impl PreparedExperts for Fake {
        fn route_tile(&self) -> u32 {
            64
        }
        fn padded_rows(&self) -> u32 {
            0
        }
    }
    impl PreparedOp<GatedMlpBuffers<View>> for Fake {
        fn info(&self) -> &OpInfo {
            unreachable!()
        }
        fn launches(
            &self,
            _: &GatedMlpBuffers<View>,
            _: View,
            _: Counts,
        ) -> Result<Vec<Launch>, Refusal> {
            Ok(self
                .0
                .iter()
                .map(|(symbol, handle)| Launch {
                    symbol: symbol.clone(),
                    rows: 1,
                    plan: LaunchPlan {
                        grid: [1; 3],
                        block: [32, 1, 1],
                        dynamic_shared_bytes: 0,
                        workspace_bytes: 0,
                        cluster: [1; 3],
                        pdl: false,
                    },
                    arguments: ArgumentList(vec![Argument::DeviceBuffer(*handle)]),
                    collective: false,
                    persistent: false,
                })
                .collect())
        }
    }

    fn buffers() -> GatedMlpBuffers<View> {
        let v = |handle| View { handle, bytes: 1 };
        GatedMlpBuffers {
            activation: v(3),
            activation_global: None,
            activation_scales: None,
            live_tokens: v(3),
            output: v(3),
            weights: QuantizedWeights {
                codes: v(1),
                local_scales: Some(v(2)),
                global_scales: None,
            },
            routing: None,
            scratch: GatedMlpScratch {
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: v(3),
            },
        }
    }

    fn tiled(symbols: Vec<(String, u64)>) -> RowTiled {
        let builds = moe_row_tiles::builds_for(&FLASH, RoutedLayout::Tiles).unwrap();
        RowTiled {
            op: Box::new(Fake(symbols)),
            info: OpInfo::new(
                "fake",
                kernel_core::properties::Invariance::any_rows(1),
                Vec::new(),
            ),
            symbols: builds.into_iter().map(|(r, t)| (r, t.symbol)).collect(),
        }
    }

    /// Over the tiled bank each served gate/up entry launches as its tiled
    /// build; an entry without one that reads the bank is refused, and one
    /// that does not read it keeps its symbol.
    #[test]
    fn tiled_ops_launch_the_tiled_builds_and_refuse_row_major_readers() {
        let builds = moe_row_tiles::builds_for(&FLASH, RoutedLayout::Tiles).unwrap();
        let (row, build) = builds.iter().next().unwrap();
        let op = tiled(vec![(row.clone(), 1), ("pack_v1".into(), 3)]);
        let out = op
            .launches(&buffers(), View::EMPTY, Counts::rows(1))
            .unwrap();
        assert_eq!(
            out.iter().map(|l| l.symbol.as_str()).collect::<Vec<_>>(),
            [build.symbol.as_str(), "pack_v1"]
        );
        for handle in [1, 2] {
            let op = tiled(vec![("row_major_reader_v1".into(), handle)]);
            assert!(
                op.launches(&buffers(), View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
    }

    /// Only the TP4 backbone gate/up bank takes the served layout; native-off
    /// prefill refuses a tiled bank with its reason, and rows leave every op
    /// as it is.
    #[test]
    fn only_the_tp4_backbone_bank_reads_the_served_layout() {
        let [target, drafter] = [(384, 6), (128, 3)].map(|(count, top_k)| Experts { count, top_k });
        assert!(moe::expert_sets(&FLASH).contains(&target));
        assert_eq!(layout(&FLASH, 4, target), RoutedLayout::served(4));
        for (ranks, set) in [(4, drafter), (2, target), (1, target)] {
            assert_eq!(layout(&FLASH, ranks, set), RoutedLayout::Rows);
        }
        let usage = |phase| OperationUse {
            phase,
            capacity: 8192,
            graph: false,
        };
        let fake = || -> Box<dyn PreparedExperts> { Box::new(Fake(Vec::new())) };
        let refused = prepare_for(
            &FLASH,
            RoutedLayout::Tiles,
            GatedMlpProjection::GateUp,
            usage(ExecutionPhase::Prefill),
            false,
            fake(),
        );
        assert_eq!(refused.err(), Some(Refused::Rule(NATIVE_OFF.into())));
        for (layout, kind, native) in [
            (RoutedLayout::Rows, GatedMlpProjection::GateUp, false),
            (RoutedLayout::Tiles, GatedMlpProjection::Down, false),
        ] {
            assert!(
                prepare_for(
                    &FLASH,
                    layout,
                    kind,
                    usage(ExecutionPhase::Prefill),
                    native,
                    fake()
                )
                .is_ok()
            );
        }
    }
}
