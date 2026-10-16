//! One shape for the engine entries a lab test launches: a symbol, its
//! declared parameters, a launch plan for a prepared row bucket, and the
//! argument packing. [`crate::conformance`] runs any [`LabEntry`].
//!
//! The kernels-sm120 engines below already define these with the same
//! signatures; their implementations only forward, so no engine changes.

use kernel_build::EntryParam;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernels_sm120::{
    attention::gated, context_parallel, latent, owned_writes, paged_partials, pooled_indexer,
    selected_partials,
};

/// An entry a lab test launches.
pub trait LabEntry {
    /// The generated CUDA entry's symbol.
    fn symbol(&self) -> String;

    /// The entry's parameters, in launch order.
    fn params(&self) -> Vec<EntryParam>;

    /// The launch for `prepared_rows` rows (a graph bucket, at most the
    /// capacity). Rows past the device live count must not be written, so
    /// one plan serves every live count.
    fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal>;

    /// Packs `values`, one per parameter in order. By default the count and
    /// each value's kind must match [`LabEntry::params`].
    fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        pack_by_kind(&self.params(), values)
    }
}

/// `values` as an argument list when each matches its parameter's kind.
pub fn pack_by_kind(params: &[EntryParam], values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
    if values.len() != params.len() {
        return Err(Refusal(
            "lab entry argument count differs from its parameters",
        ));
    }
    let matches = params.iter().zip(&values).all(|(param, value)| {
        matches!(
            (param, value),
            (
                EntryParam::DevicePointer { .. },
                Argument::DeviceBuffer(_) | Argument::NullDevicePointer
            ) | (EntryParam::U32 { .. }, Argument::U32(_))
                | (EntryParam::U64 { .. }, Argument::U64(_))
                | (EntryParam::I32 { .. }, Argument::I32(_))
                | (EntryParam::TensorMap { .. }, Argument::TensorMap(_))
                | (
                    EntryParam::PointerArray { .. },
                    Argument::DevicePointerArray(_)
                )
        )
    });
    if !matches {
        return Err(Refusal(
            "lab entry argument kind differs from its parameter",
        ));
    }
    let arguments = ArgumentList(values);
    arguments.validate()?;
    Ok(arguments)
}

/// Implements [`LabEntry`] by forwarding to an engine's own methods.
macro_rules! forward {
    ($($engine:ty),* $(,)?) => {$(
        impl LabEntry for $engine {
            fn symbol(&self) -> String {
                <$engine>::symbol(self)
            }

            fn params(&self) -> Vec<EntryParam> {
                <$engine>::params(self)
            }

            fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
                <$engine>::plan(self, prepared_rows)
            }
        }
    )*};
}

forward!(
    latent::Entry,
    paged_partials::Entry,
    selected_partials::Entry,
    pooled_indexer::Entry,
    pooled_indexer::block_merge::Entry,
    pooled_indexer::head_sum::Entry,
    context_parallel::Entry,
    owned_writes::Entry,
);

/// The split merge kernel of an engine entry whose bucket's splits a second
/// kernel folds, as an entry of its own: a chain launches it after the entry
/// ([`crate::conformance::Case::then`]). It plans only where the entry
/// itself plans the bucket, so a merge launched alone keeps the entry's
/// contract and capacity checks, and the bucket has a merge launch.
pub struct SplitMerge<'a, E>(pub &'a E);

/// Implements [`LabEntry`] for [`SplitMerge`] by forwarding to an engine's
/// `merge_symbol`, `merge_params` and `merge_plan`.
macro_rules! split_merge {
    ($($engine:ty),* $(,)?) => {$(
        impl LabEntry for SplitMerge<'_, $engine> {
            fn symbol(&self) -> String {
                self.0
                    .merge_symbol()
                    .unwrap_or_else(|| format!("{} split merge", self.0.symbol()))
            }

            fn params(&self) -> Vec<EntryParam> {
                self.0.merge_params()
            }

            fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
                // The entry's own plan checks its contract and capacity,
                // which a merge plan alone may not.
                self.0.plan(prepared_rows)?;
                self.0
                    .merge_plan(prepared_rows)?
                    .ok_or(Refusal("the entry has no split merge at this bucket"))
            }
        }
    )*};
}

split_merge!(latent::Entry, pooled_indexer::Entry);

/// Its symbol and parameters are fields, and it packs its own arguments.
impl LabEntry for gated::Entry {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }

    fn params(&self) -> Vec<EntryParam> {
        self.params.clone()
    }

    fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        gated::Entry::plan(self, prepared_rows)
    }

    fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        gated::Entry::pack_arguments(self, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::latent_attention::LatentAttentionQuery;
    use kernels_sm120::latent::Query;

    fn pointer(name: &str) -> EntryParam {
        EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: "const uint16_t".into(),
        }
    }

    #[test]
    fn packing_refuses_a_wrong_count_or_kind() {
        let params = [
            pointer("input"),
            EntryParam::U32 {
                name: "rows".into(),
            },
        ];
        let packed = pack_by_kind(&params, vec![Argument::DeviceBuffer(1), Argument::U32(4)])
            .expect("matching kinds");
        assert_eq!(packed.0.len(), 2);
        assert!(pack_by_kind(&params, vec![Argument::DeviceBuffer(1)]).is_err());
        assert!(pack_by_kind(&params, vec![Argument::U32(4), Argument::DeviceBuffer(1)]).is_err());
        assert!(pack_by_kind(&params, vec![Argument::DeviceBuffer(1), Argument::U64(4)]).is_err());
        assert!(pack_by_kind(&params, vec![Argument::NullDevicePointer, Argument::U32(0)]).is_ok());
    }

    #[test]
    fn engines_plan_through_the_trait_as_they_do_directly() {
        let geometry = paged_partials::Geometry {
            q_heads: 4,
            kv_heads: 2,
            head_dim: 128,
            page: 64,
        };
        let entries = paged_partials::entries(kernels_sm120::Profile::StdDenseMoe, geometry);
        let entry = entries.first().expect("a paged partials entry");
        let lab: &dyn LabEntry = entry;
        assert_eq!(lab.symbol(), entry.symbol());
        assert_eq!(lab.params(), entry.params());
        let rows = entry.capacity;
        assert_eq!(lab.plan(rows), entry.plan(rows));
        assert!(lab.plan(0).is_err());
    }

    #[test]
    fn split_merges_plan_through_the_trait_where_their_entry_merges() {
        let geometry = latent::Geometry {
            heads: 16,
            latent: 512,
            page_size: 64,
            max_keys: 2051,
            softmax_scale: 0.0625,
            nope: 256,
            v_dim: 256,
        };
        let entries = latent::entries(kernels_sm120::Profile::StdDenseMoe, geometry);
        let named = |suffix: &str| {
            entries
                .iter()
                .find(|entry| entry.name.ends_with(suffix))
                .expect("a latent attention entry")
        };
        let decode = named("attn_bf16_decode");
        let merge = SplitMerge(decode);
        assert_eq!(Some(merge.symbol()), decode.merge_symbol());
        assert_eq!(merge.params(), decode.merge_params());
        assert!(decode.splits() > 1);
        assert_eq!(merge.plan(4).ok(), decode.merge_plan(4).unwrap());
        // Prefill takes one split, which no merge folds.
        let prefill = named("attn_bf16_prefill");
        assert!(SplitMerge(prefill).plan(4).is_err());
    }

    fn latent_decode() -> latent::Entry {
        let geometry = latent::Geometry {
            heads: 16,
            latent: 512,
            page_size: 64,
            max_keys: 2051,
            softmax_scale: 0.0625,
            nope: 256,
            v_dim: 256,
        };
        latent::entries(kernels_sm120::Profile::StdDenseMoe, geometry)
            .into_iter()
            .find(|entry| entry.name.ends_with("attn_bf16_decode"))
            .expect("a latent decode entry")
    }

    /// A split merge launched alone refuses a bucket its entry refuses.
    #[test]
    fn a_split_merge_refuses_rows_past_its_entry_capacity() {
        let mut decode = latent_decode();
        let Query::Latent(LatentAttentionQuery::Attention { capacity, .. }) = &mut decode.query
        else {
            panic!("an attention entry");
        };
        *capacity = 1;
        let refusal = decode.plan(4).expect_err("rows past the capacity");
        assert!(decode.merge_plan(4).is_ok_and(|plan| plan.is_some()));
        assert_eq!(SplitMerge(&decode).plan(4), Err(refusal));
        assert_eq!(
            SplitMerge(&decode).plan(1).ok(),
            decode.merge_plan(1).unwrap()
        );

        let geometry = pooled_indexer::Geometry {
            heads: 32,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: true,
            position_dtype: kernel_core::contract::DType::Bf16,
            key_norm_eps: 1e-6,
            verify_rows: 6,
        };
        let mut select = pooled_indexer::entries(kernels_sm120::Profile::StdDenseMoe, geometry)
            .into_iter()
            .find(|entry| entry.merge_symbol().is_some())
            .expect("a pooled selection entry");
        let rows = select.capacity;
        assert!(SplitMerge(&select).plan(rows).is_ok());
        select.capacity = rows - 1;
        let refusal = select.plan(rows).expect_err("rows past the capacity");
        assert_eq!(SplitMerge(&select).plan(rows), Err(refusal));
    }

    /// A split merge launched alone refuses an entry whose contract fails.
    #[test]
    fn a_split_merge_refuses_an_entry_its_contract_refuses() {
        let mut decode = latent_decode();
        let Query::Latent(LatentAttentionQuery::Attention { heads, .. }) = &mut decode.query else {
            panic!("an attention entry");
        };
        *heads = 24;
        let refusal = decode.contract().expect_err("heads not in 16s");
        assert!(decode.merge_plan(4).is_ok_and(|plan| plan.is_some()));
        assert_eq!(SplitMerge(&decode).plan(4), Err(refusal));
    }
}
