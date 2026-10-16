//! MiMo's attention on the SM120 backend: its catalog's variants, the
//! queries they state, and the switches that pick among their entries.

use kernel_core::contract::{AttentionLayout, AttentionPhase, AttentionQuery, DType, Refusal};
use kernels_sm120::attention::Variant;
use kernels_sm120::backend::AttentionEntries;

use crate::switches::Switches;
use crate::topology::Tp;

/// The attention entries of a `tp` instance with `switches`: exact verify
/// rows of paged layers share their keys per row group unless
/// `selection::EXACT_ATTENTION` is `rows`, and wide verify buckets run
/// per-row exact entries under `selection::VERIFY_ATTENTION`.
pub(super) fn entries(tp: Tp, switches: Switches) -> AttentionEntries {
    AttentionEntries {
        variants: variants(tp).to_vec(),
        per_row_exact: !switches.exact_attention_grouped,
        rows_above: switches
            .verify_attention_rows
            .then_some(crate::contract::EXACT_VERIFY_ROWS),
    }
}

fn variants(tp: Tp) -> &'static [Variant] {
    model_mimo26_contract::sm120::workspace::attention::mimo_variants(tp)
}

/// The query of `tp`'s target attention over a `layout` cache of `kv`
/// entries at `phase`, as MiMo's catalog states it.
pub fn query(
    tp: Tp,
    layout: AttentionLayout,
    phase: AttentionPhase,
    kv: DType,
) -> Result<AttentionQuery, Refusal> {
    variants(tp)
        .iter()
        .find(|variant| {
            variant.layout == layout
                && variant.query.phase == phase
                && variant.query.kv_dtype == kv
                && !variant.name.ends_with("dflash_block")
        })
        .map(|variant| variant.query)
        .ok_or(Refusal("MiMo attention entry missing"))
}

/// The query of DFlash's block attention.
pub fn block_query(tp: Tp) -> Result<AttentionQuery, Refusal> {
    variants(tp)
        .iter()
        .find(|variant| variant.name.ends_with("dflash_block"))
        .map(|variant| variant.query)
        .ok_or(Refusal("DFlash block attention entry missing"))
}

#[cfg(test)]
mod tests {
    use bundle_runtime::ModelLimits;
    use kernel_core::contract::AttentionLayout;
    use model_mimo26_contract::sm120::attention::catalog_for;
    use model_mimo26_contract::sm120::geometry::{GeometryModel, MODELS};

    /// Paged attention reads item `i`'s block-table row from entry
    /// `i * max_context_capacity / page_size`, while a step binds rows of
    /// the geometry's pages (`ModelLimits::max_pages`, which the RoPE and
    /// KV write index too). Every compiled geometry's paged entries, and
    /// those of a geometry of the synthetic recipe lane's 8192-token
    /// context, index those rows.
    #[test]
    fn paged_entries_index_the_block_table_rows_a_step_binds() {
        let mut short = MODELS[0].geometry;
        short.limits.context = 8192;
        let short = GeometryModel {
            geometry: short,
            model: MODELS[0].model,
        };
        for owner in MODELS.iter().copied().chain([short]) {
            let g = owner.geometry;
            let pages = ModelLimits::new(
                g.vocab(),
                u64::from(g.context_tokens()),
                crate::contract::PAGE_TOKENS,
            )
            .unwrap()
            .max_pages();
            let paged: Vec<_> = catalog_for(owner)
                .into_iter()
                .filter(|variant| variant.layout == AttentionLayout::Paged)
                .collect();
            assert!(!paged.is_empty(), "{}", g.id);
            for variant in paged {
                let query = variant.query;
                assert_eq!(
                    query.max_context_capacity.div_ceil(query.page_size),
                    pages,
                    "{} context {}: {}",
                    g.id,
                    g.context_tokens(),
                    variant.name
                );
            }
        }
    }
}
