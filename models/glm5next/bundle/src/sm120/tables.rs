//! SM120 tables the generic tests compare lowered launches against: the
//! parameter names of the entries a comm epilogue boundary launches, the
//! symbols a context-parallel group lowers itself and the key localization
//! entry of a group, and the latent attention's and the low-latency
//! all-reduce's fixed sizes.

use std::collections::{BTreeMap, BTreeSet};

use kernel_build::EntryParam;
use model_glm5next_contract::contract::{PREFILL_TOKENS, Shape};
use model_glm5next_contract::sm120::engines::{self as selected, ModelRow};

use super::engines::Engines;

fn name(param: &EntryParam) -> String {
    match param {
        EntryParam::DevicePointer { name, .. }
        | EntryParam::U64 { name }
        | EntryParam::U32 { name }
        | EntryParam::I32 { name }
        | EntryParam::TensorMap { name }
        | EntryParam::PointerArray { name, .. } => name.clone(),
    }
}

impl Engines {
    /// The latent attention's split count, which every attention launch
    /// passes and its merge reads.
    pub(crate) const LATENT_MAX_SPLITS: u32 = kernels_sm120::latent::MAX_SPLITS;

    /// Elements of one slice of the low-latency all-reduce's payload.
    pub(crate) const LL_SLICE_ELEMENTS: u64 = kernels_sm120::comm::ll_bf16::SLICE_ELEMENTS;

    /// The parameter names of every entry a comm epilogue boundary
    /// launches, by symbol, for the tests.
    pub(crate) fn parameters(&self, shape: Shape) -> BTreeMap<String, Vec<String>> {
        let names = |params: &[EntryParam]| params.iter().map(name).collect::<Vec<_>>();
        let mut out = BTreeMap::new();
        for spec in kernels_sm120::comm::cubin_spec().entries {
            out.insert(spec.symbol, names(&spec.params));
        }
        for spec in kernels_sm120::comm::phased_bf16::entries() {
            out.insert(spec.symbol, names(&spec.params));
        }
        out.insert(
            selected::LL_SYMBOL.to_owned(),
            names(&kernels_sm120::comm::ll_bf16::params()),
        );
        for entry in &self.sinkhorn {
            out.insert(entry.symbol.clone(), names(&entry.params));
        }
        out.insert(
            self.sinkhorn_split.symbol.clone(),
            names(&self.sinkhorn_split.params),
        );
        out.insert(
            self.sinkhorn_mix_norm.symbol.clone(),
            names(&self.sinkhorn_mix_norm.params),
        );
        out.insert(
            self.comm_epilogue.symbol.clone(),
            names(&self.comm_epilogue.params),
        );
        for row in [ModelRow::Round, ModelRow::Widen] {
            out.insert(row.symbol(), names(&row.entry(shape).params));
        }
        let width = shape.profile.hidden;
        let norm = selected::rmsnorm(shape, width, PREFILL_TOKENS).unwrap();
        let symbol = selected::rmsnorm_symbol(width);
        out.insert(symbol.clone(), names(&norm.entry(symbol).params));
        out
    }

    /// The symbols of the stages a group lowers itself: the latent KV write,
    /// the index key write (which takes the group rank), the selection and its
    /// merges, the latent attention and its merges, and the group's own
    /// entries and all-gather.
    pub(crate) fn group_stage_symbols() -> BTreeSet<String> {
        use kernel_core::latent_attention::LatentAttentionQuery;
        use kernels_sm120::pooled_indexer::Kind;
        let engines = Engines::new(Shape::TP4).unwrap();
        let mut symbols = model_glm5next_contract::sm120::context_parallel::symbols(Shape::TP4)
            .into_iter()
            .collect::<BTreeSet<_>>();
        for entry in &engines.latent {
            if matches!(
                entry.query,
                kernels_sm120::latent::Query::Latent(
                    LatentAttentionQuery::KvWrite { .. } | LatentAttentionQuery::Attention { .. }
                )
            ) {
                symbols.extend(std::iter::once(entry.symbol()).chain(entry.merge_symbol()));
            }
        }
        // The FP8 prefill entries' slice-widened forms, which the prefill
        // pieces' partial attentions run.
        symbols.extend(engines.latent_slices.iter().map(|entry| entry.symbol()));
        // The decode and verify entries' column merges, which their merges run.
        symbols.extend(engines.latent_merge.iter().map(|entry| entry.symbol()));
        for entry in &engines.indexer {
            if matches!(
                entry.kind,
                Kind::KeyWrite(_) | Kind::Select { .. } | Kind::TopKMerge { .. }
            ) {
                symbols.extend(std::iter::once(entry.symbol()).chain(entry.merge_symbol()));
            }
        }
        // The selections' tree merges, which merge two or more splits.
        symbols.extend(engines.indexer_merge.iter().map(|entry| entry.symbol()));
        // The global selection's radix merge, which merges its two or more
        // splits in place of the tree merge.
        symbols.extend(engines.indexer_radix.iter().map(|entry| entry.symbol()));
        // The selections' one-split forms, which the prefill pieces' selections
        // of 376 rows or more run.
        symbols.extend(engines.indexer_rows.iter().map(|entry| entry.symbol()));
        symbols
    }

    /// The context-parallel key localization entry of groups of `size`
    /// over the latent attention's key lists.
    pub(crate) fn localize_keys(shape: Shape, size: u32) -> kernels_sm120::context_parallel::Entry {
        model_glm5next_contract::sm120::context_parallel::entries(shape)
            .into_iter()
            .find(|entry| {
                entry.kind == kernels_sm120::context_parallel::Kind::LocalizeKeys { ranks: size }
                    && entry.geometry.list_width == shape.latent_keys()
            })
            .unwrap()
    }
}
