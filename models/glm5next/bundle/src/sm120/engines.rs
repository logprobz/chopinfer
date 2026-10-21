//! Every engine entry the bundle embeds, as the contract's SM120 module
//! selects them for a shape and checkpoint layout: the backend registers
//! them (`super::backend`) and module loading reads their dynamic shared
//! memory (`Engines::shared_bytes`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use kernels_sm120::comm::sinkhorn_split;
use kernels_sm120::linear::channel_decay;
use kernels_sm120::moe::chained::{self, LocalRoute, Projection, RoutedSum};
use kernels_sm120::{pooled_indexer, sinkhorn_hc};
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::quantization::{FP8_BLOCK128, Layout};
use model_glm5next_contract::sm120::engines as selected;

/// Every engine entry the bundle embeds, as the contract selects them.
#[derive(Debug)]
pub struct Engines {
    /// The shape they serve.
    pub shape: Shape,
    pub latent: Vec<kernels_sm120::latent::Entry>,
    /// The FP8 prefill attention entries' slice-widened forms, which every
    /// launch of those entries runs in their place.
    pub latent_slices: Vec<kernels_sm120::latent::slices::Entry>,
    /// The decode and verify attention entries' column merges, which every
    /// launch of those entries' split merges runs in their place.
    pub latent_merge: Vec<kernels_sm120::latent::merge_columns::Entry>,
    /// The KDA entries, then the prefill's streamed prepare and scan.
    pub kda: Vec<channel_decay::Entry>,
    pub indexer: Vec<pooled_indexer::Entry>,
    /// The selections' tree merges, which a selection over two or more
    /// splits runs in place of its merge.
    pub indexer_merge: Vec<pooled_indexer::merge_tree::Entry>,
    /// The global selection's radix merge, which a global selection over two
    /// or more splits runs in place of the tree merge.
    pub indexer_radix: Vec<pooled_indexer::merge_radix::Entry>,
    /// The selections' one-split forms, which a selection launch whose
    /// bucket takes one split runs in place of its split kernel and merge.
    pub indexer_rows: Vec<pooled_indexer::rows::Entry>,
    pub sinkhorn: Vec<sinkhorn_hc::Entry>,
    /// The mix-collapse split over CTAs, which the prefill mixer stages run
    /// where the served schedule picks it (`sinkhorn::mix_schedule`).
    pub sinkhorn_split: sinkhorn_hc::split::Entry,
    /// The mix-collapse with rows tiled over a CTA, one entry per tile the
    /// served schedule picks.
    pub sinkhorn_tiled: Vec<sinkhorn_hc::tiled::Entry>,
    /// The split mix-collapse fused with the input norm (its sum of squares
    /// in the 256-lane tree), which the decode and verify graphs run in place
    /// of each mixer and its input norm.
    pub sinkhorn_mix_norm: sinkhorn_hc::mix_norm_tree::Entry,
    /// The split Sinkhorn mHC comm epilogue, after the all-reduce.
    pub comm_epilogue: sinkhorn_split::Entry,
    /// The chained projections of every checkpoint layout: the dense MLP's
    /// and shared expert's in block FP8, then in the NVFP4 exports' formats
    /// (`selected::moe_format_projections`), then the routed experts split
    /// over the ranks in every routed format
    /// (`selected::moe_routed_projections`).
    pub moe: Vec<Projection>,
    /// The checkpoint layout the instance serves, which picks each chained
    /// projection's format (`super::formats`).
    pub layout: &'static Layout,
    /// The decode route (at most `DECODE_ROUTE_MAX_TOKENS` tokens, row
    /// tile 16) and the prefill route (row tile 64), over every expert.
    pub routes: [LocalRoute; 2],
    pub sum: RoutedSum,
    /// Each route and sum kernel's ABI, by symbol.
    pub(crate) route_params: BTreeMap<String, Vec<kernel_build::EntryParam>>,
    /// The embedding gather of the rank's vocabulary shard.
    pub embedding: kernels_sm120::row::RowInstantiation,
    /// The RMSNorm of each width the program normalizes, by requirement and
    /// symbol; the row engine's norms take no dynamic shared memory.
    pub norms: Vec<(kernel_core::rmsnorm::RmsNormRequirement, String)>,
    /// The context-parallel entries and group all-gather.
    context_parallel: Vec<String>,
    /// The projections' shared GEMM entries
    /// (`model_glm5next_contract::sm120::precision`, then
    /// `model_glm5next_contract::sm120::projection_formats`) and their dynamic shared
    /// memory at full capacity, their layout copies' entries and the
    /// stacked entries (`stacked`).
    pub projections: BTreeMap<String, u32>,
    /// The KDA layers' stacked input projections, one per decode and verify
    /// bucket (`model_glm5next_contract::sm120::stacked`).
    pub stacked: Vec<kernels_sm120::gemm::stacked::Stacked>,
}

type Lowered = Result<&'static Engines, String>;

/// Every set of engines [`Engines::shared`] lowered, with its key, in
/// first-request order.
static LOWERED: Mutex<Vec<(Shape, Layout, Lowered)>> = Mutex::new(Vec::new());

/// Each projection GEMM entry's dynamic shared memory, planned at its
/// compiled row capacity, and each layout copy's (none).
fn projection_shared_bytes(shape: Shape) -> Result<BTreeMap<String, u32>, String> {
    use kernel_core::contract::StaticQuery;
    // The copies take no dynamic shared memory.
    let copies = model_glm5next_contract::sm120::precision::copies(shape)?
        .into_iter()
        .map(|copy| Ok((copy.stage().entry().symbol.clone(), 0)));
    // The block-FP8 export's entries, then the NVFP4 layouts' added ones.
    model_glm5next_contract::sm120::precision::instantiations(shape)?
        .into_iter()
        .chain(model_glm5next_contract::sm120::projection_formats::instantiations_added(shape)?)
        .map(|spec| {
            let StaticQuery::Gemm(query) = spec.key.query else {
                return Err(format!("{} is not a GEMM", spec.symbol));
            };
            let plan = kernels_sm120::gemm::plan(
                &spec.key,
                kernels_sm120::gemm::row_counts(&spec.key, query.m_capacity),
            )
            .map_err(|r| format!("{}: {r}", spec.symbol))?;
            Ok((spec.symbol, plan.dynamic_shared_bytes))
        })
        .chain(copies)
        .collect()
}

/// Each stacked KDA input projection entry's dynamic shared memory at its
/// bucket's rows.
fn stacked_shared_bytes(
    stacked: &[kernels_sm120::gemm::stacked::Stacked],
) -> Result<Vec<(String, u32)>, String> {
    stacked
        .iter()
        .map(|entry| {
            let plan = entry
                .plan(entry.capacity)
                .map_err(|r| format!("{}: {r}", entry.symbol))?;
            Ok((entry.symbol.clone(), plan.dynamic_shared_bytes))
        })
        .collect()
}

impl Engines {
    /// The engines of the block-FP8 export.
    pub fn new(shape: Shape) -> Result<Self, String> {
        Self::for_layout(shape, &FP8_BLOCK128)
    }

    /// [`Self::for_layout`], lowered once per process for each (shape,
    /// checkpoint layout) and shared read-only by every instance after, for
    /// the process's life: the engines are a pure function of the two, and
    /// lowering them takes most of a create's host time.
    pub fn shared(shape: Shape, layout: &'static Layout) -> Lowered {
        // A panic while lowering leaves no entry, so the list stays usable.
        let mut lowered = LOWERED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let found = lowered
            .iter()
            .find(|(of, stored_as, _)| *of == shape && stored_as == layout);
        if let Some((.., engines)) = found {
            return engines.clone();
        }
        let engines = Self::for_layout(shape, layout).map(|engines| &*Box::leak(Box::new(engines)));
        lowered.push((shape, *layout, engines.clone()));
        engines
    }

    /// The engines of an export stored in `layout`.
    pub fn for_layout(shape: Shape, layout: &'static Layout) -> Result<Self, String> {
        let served = selected::moe_shapes(shape)?;
        let routes = served.routes();
        let sum = selected::routed_sum(shape)?;
        let mut route_params = BTreeMap::new();
        let prefix = chained::split::PREFIX;
        for route in routes {
            for spec in route.entries(prefix).map_err(|r| r.to_string())? {
                route_params.insert(spec.symbol, spec.params);
            }
        }
        // The route cubin also holds the BF16 sum, which nothing launches.
        let bf16 = sum.with_output(chained::RoutedSumOutput::Bf16);
        for spec in [sum.entry(prefix), bf16.entry(prefix), sum.add_entry(prefix)] {
            let spec = spec.map_err(|r| r.to_string())?;
            route_params.insert(spec.symbol, spec.params);
        }
        let moe = [
            selected::moe_projections(shape)?,
            selected::moe_format_projections(shape)?,
            selected::moe_routed_projections(shape)?,
        ]
        .concat();
        let stacked = model_glm5next_contract::sm120::stacked::entries(shape)?;
        Ok(Self {
            shape,
            latent: selected::latent_entries(shape),
            latent_slices: selected::latent_slices_entries(shape),
            latent_merge: selected::latent_merge_columns_entries(shape),
            kda: [
                selected::channel_decay_entries(shape),
                selected::channel_decay_streamed_entries(shape),
            ]
            .concat(),
            indexer: selected::pooled_indexer_entries(shape),
            indexer_merge: selected::pooled_indexer_merge_tree_entries(shape),
            indexer_radix: selected::pooled_indexer_merge_radix_entries(shape),
            indexer_rows: selected::pooled_indexer_rows_entries(shape),
            sinkhorn: selected::sinkhorn_entries(shape),
            sinkhorn_split: selected::sinkhorn_split_entry(shape),
            sinkhorn_tiled: selected::sinkhorn_tiled_entries(shape),
            sinkhorn_mix_norm: selected::sinkhorn_mix_norm_tree_entry(shape),
            comm_epilogue: selected::comm_epilogue_entry(shape)?,
            moe,
            layout,
            routes,
            sum,
            route_params,
            embedding: selected::embedding_entry(shape),
            norms: shape
                .rmsnorm_widths()
                .map(|width| {
                    let requirement = shape.rmsnorm_requirement(width);
                    (requirement, selected::rmsnorm_symbol(width))
                })
                .to_vec(),
            context_parallel: model_glm5next_contract::sm120::context_parallel::symbols(shape),
            projections: projection_shared_bytes(shape)?
                .into_iter()
                .chain(stacked_shared_bytes(&stacked)?)
                .collect(),
            stacked,
        })
    }

    /// Dynamic shared bytes of every embedded entry, for module loading.
    pub fn shared_bytes(&self) -> BTreeMap<String, u32> {
        let mut bytes = BTreeMap::new();
        for entry in &self.latent {
            bytes.insert(entry.symbol(), entry.dynamic_shared_bytes());
            if let Some(merge) = entry.merge_symbol() {
                bytes.insert(merge, 0);
            }
        }
        for entry in &self.latent_slices {
            bytes.insert(entry.symbol(), entry.dynamic_shared_bytes());
        }
        for entry in &self.latent_merge {
            bytes.insert(entry.symbol(), 0);
        }
        for entry in &self.kda {
            bytes.insert(entry.symbol.clone(), entry.dynamic_shared_bytes());
        }
        for entry in &self.indexer {
            bytes.insert(entry.symbol(), entry.dynamic_shared_bytes());
            if let Some(merge) = entry.merge_symbol() {
                bytes.insert(merge, 0);
            }
        }
        for entry in &self.indexer_merge {
            bytes.insert(entry.symbol(), 0);
        }
        for entry in &self.indexer_radix {
            bytes.insert(entry.symbol(), 0);
        }
        for entry in &self.indexer_rows {
            bytes.insert(entry.symbol(), entry.dynamic_shared_bytes());
        }
        for entry in &self.sinkhorn {
            bytes.insert(entry.symbol.clone(), 0);
        }
        bytes.insert(self.sinkhorn_split.symbol.clone(), 0);
        for entry in &self.sinkhorn_tiled {
            bytes.insert(entry.symbol.clone(), 0);
        }
        bytes.insert(self.sinkhorn_mix_norm.symbol.clone(), 0);
        // Chained projections at their compiled capacity: decoded tiles take
        // their stages, the other schedules static shared memory only.
        for projection in &self.moe {
            let shared = projection
                .plan(projection.tokens_capacity)
                .map_or(0, |plan| plan.dynamic_shared_bytes);
            bytes.insert(projection.symbol.clone(), shared);
        }
        let norms = self.norms.iter().map(|(_, symbol)| symbol);
        for symbol in self.route_params.keys().chain(norms) {
            bytes.insert(symbol.clone(), 0);
        }
        bytes.insert(self.embedding.symbol.clone(), 0);
        for row in selected::ModelRow::ALL {
            bytes.insert(row.symbol(), 0);
        }
        // The shared row copies (`selected::ROW_COPY_CUBIN`) and gather by
        // index (`selected::ROW_GATHER_CUBIN`).
        for entry in kernels_sm120::row::copy::entries() {
            bytes.insert(entry.symbol, 0);
        }
        bytes.insert(selected::ROW_GATHER_SYMBOL.into(), 0);
        bytes.insert(kernels_sm120::comm::owned_sum_f32::SYMBOL.into(), 0);
        for symbol in selected::COMM_SYMBOLS {
            bytes.insert(symbol.into(), 0);
        }
        // The shared argmax entries the drafter's argmax runs.
        for entry in selected::argmax_entries() {
            bytes.insert(entry.symbol, 0);
        }
        bytes.insert(selected::LL_SYMBOL.into(), 0);
        for symbol in selected::PHASED_SYMBOLS {
            bytes.insert(symbol.into(), 0);
        }
        // The F32 low-latency and phased entries, whose rounding forms the
        // MoE sublayers' all-reduce runs.
        let ll_f32 = kernels_sm120::comm::ll_f32::cubin_spec(selected::LL_F32_CUBIN.into());
        let phased_f32 =
            kernels_sm120::comm::phased_f32::cubin_spec(selected::PHASED_F32_CUBIN.into());
        for entry in ll_f32.entries.into_iter().chain(phased_f32.entries) {
            bytes.insert(entry.symbol, 0);
        }
        bytes.insert(self.comm_epilogue.symbol.clone(), 0);
        // The payload admission entries (`model_glm5next_contract::sm120::payload`).
        for check in model_glm5next_contract::payload::Check::ALL {
            bytes.insert(
                model_glm5next_contract::sm120::payload::symbol(check).into(),
                0,
            );
        }
        // Commit's state copies (`model_glm5next_contract::sm120::state_copy`).
        bytes.insert(model_glm5next_contract::sm120::state_copy::SYMBOL.into(), 0);
        for symbol in &self.context_parallel {
            bytes.insert(symbol.clone(), 0);
        }
        bytes.extend(self.projections.clone());
        bytes
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use model_glm5next_contract::quantization::representative_layouts;

    /// For every representative layout, the shared engines are the ones
    /// lowered fresh (every field, and the symbol table module loading
    /// reads), and later requests return that same lowering.
    #[test]
    fn shared_engines_equal_fresh_ones() {
        for layout in representative_layouts() {
            let fresh = Engines::for_layout(Shape::TP4, layout).unwrap();
            let shared = Engines::shared(Shape::TP4, layout).unwrap();
            assert_eq!(format!("{shared:?}"), format!("{fresh:?}"), "{}", layout.id);
            assert_eq!(shared.shared_bytes(), fresh.shared_bytes(), "{}", layout.id);
            let again = Engines::shared(Shape::TP4, layout).unwrap();
            assert!(std::ptr::eq(shared, again), "{}", layout.id);
        }
    }

    /// The comm state regions the all-reduces' ops keep are those create
    /// allocated before the ops sized them (`engines::ll_state_bytes`,
    /// `phased_state_bytes`, `two_shot_bytes`, the F32 sum of the scratch
    /// rows, and the MoE sublayers' `rounded_ll_state_bytes` and
    /// `rounded_phased_state_bytes`), so every rank allocates the same
    /// buffers; each drafter argmax pair gather keeps one tile's flags and
    /// epoch of the comm engine's all-gather.
    #[test]
    fn comm_state_regions_are_the_ones_create_allocated() {
        use model_glm5next_contract::sm120::engines;
        let shape = Shape::TP4;
        let regions = crate::journal_tests::comm_regions();
        let ll = engines::ll_state_bytes(shape).unwrap();
        let phased = engines::phased_state_bytes(shape).unwrap();
        let pair = engines::two_shot_bytes(shape).unwrap();
        let rounded_ll = engines::rounded_ll_state_bytes(shape).unwrap();
        let rounded = engines::rounded_phased_state_bytes(shape).unwrap();
        assert!(engines::comm_layout(shape).unwrap().one_shot.is_empty());
        assert_eq!(
            regions,
            crate::ops::StateRegions {
                low_latency: [ll.inbox, ll.epochs],
                phased: [phased.inbox, phased.flags, phased.epoch],
                pair: [pair.inbox, pair.flags, pair.epoch],
                sum: u64::from(crate::buffers::SCRATCH_ROWS) * u64::from(shape.profile.hidden) * 4,
                rounded_low_latency: [rounded_ll.inbox, rounded_ll.epochs],
                rounded_phased: [rounded.inbox, rounded.flags, rounded.epoch],
            }
        );
        // The phased F32 state holds F32 slots, twice the BF16 state's
        // two-shot inbox of the prefill chunk: about 144 MiB per rank.
        assert_eq!(
            rounded.inbox,
            (kernels_sm120::comm::phased_f32::ONE_SHOT_REGION + 4 * 8192 * 1024) * 4
        );
        // Per source rank a flag and an acknowledgment, and one epoch cell.
        let ranks = u64::from(shape.ranks());
        assert_eq!(crate::journal_tests::draft_regions(), [2 * ranks * 4, 4]);
    }
}
