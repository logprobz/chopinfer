//! Ordered recipes and lock destinations for the model's standalone builds.
mod bundle;
mod exact;
mod lab;
mod moe;
mod native;
mod special;
#[cfg(test)]
mod tests;

use kernel_build::CubinSpec;
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    Bundle,
    Lab,
    Shared,
    Output(&'static str),
}

impl Root {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bundle => "bundle",
            Self::Lab => "lab",
            Self::Shared => "shared",
            Self::Output(_) => "output",
        }
    }

    pub fn path(self, out: &Path) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        match self {
            Self::Bundle => root.join("../bundle/kernels.lock.d"),
            Self::Lab => root.join("kernels.lock.d"),
            Self::Shared => root.join("../../../crates/kernel-lab/kernels.lock.d"),
            Self::Output(path) => out.join(path),
        }
    }
}

pub struct Step {
    pub root: Root,
    pub specs: Vec<(&'static str, CubinSpec)>,
    pub subdir: &'static str,
    pub initialize: bool,
}

impl Step {
    fn at(mut self, subdir: &'static str) -> Self {
        self.subdir = subdir;
        self
    }

    fn pinned(mut self) -> Self {
        self.initialize = false;
        self
    }
}

pub enum Export {
    None,
    Modules(&'static str),
    Attention,
    RouterReferences,
}

pub struct Recipe {
    pub root: Root,
    pub family: &'static str,
    pub steps: fn() -> Result<Vec<Step>>,
    pub export: Export,
}

impl Recipe {
    pub fn key(&self) -> String {
        format!("{}/{}", self.root.label(), self.family)
    }
}

fn recipe(
    root: Root,
    family: &'static str,
    steps: fn() -> Result<Vec<Step>>,
    export: Export,
) -> Recipe {
    Recipe {
        root,
        family,
        steps,
        export,
    }
}

fn ordered(root: Root, specs: Vec<(&'static str, CubinSpec)>) -> Step {
    Step {
        root,
        specs,
        subdir: "",
        initialize: true,
    }
}

fn family(root: Root, name: &'static str, specs: Vec<CubinSpec>) -> Step {
    ordered(root, specs.into_iter().map(|spec| (name, spec)).collect())
}

pub fn all() -> Vec<Recipe> {
    let mut recipes = simple();
    recipes.extend(special::recipes());
    recipes.sort_by_key(Recipe::key);
    assert!(
        recipes
            .windows(2)
            .all(|pair| pair[0].key() != pair[1].key())
    );
    recipes
}

fn cubins(specs: Vec<(&'static str, CubinSpec)>) -> Vec<CubinSpec> {
    specs.into_iter().map(|(_, spec)| spec).collect()
}

fn simple() -> Vec<Recipe> {
    macro_rules! one {
        ($root:ident, $name:expr, $specs:expr) => {
            one!($root, $name, $specs, "")
        };
        ($root:ident, $name:expr, $specs:expr, $subdir:expr) => {
            recipe(
                Root::$root,
                $name,
                || Ok(vec![family(Root::$root, $name, $specs).at($subdir)]),
                Export::None,
            )
        };
    }
    vec![
        one!(
            Lab,
            "ceil_split_reference",
            lab::pruned_references::ceil_split()?
        ),
        one!(
            Lab,
            "mxfp8_prefill_matrix_reference",
            lab::pruned_references::prefill_matrix()?
        ),
        one!(
            Bundle,
            bundle::ceil_prefill::FAMILY,
            cubins(bundle::ceil_prefill::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::ceil_split::FAMILY,
            cubins(bundle::ceil_split::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::f32_blocked::SERVING_FAMILY,
            bundle::f32_blocked::serving_cubins(GEOMETRIES)?
        ),
        one!(
            Bundle,
            bundle::f32_decode::FAMILY,
            cubins(bundle::f32_decode::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::f32_split::FAMILY,
            cubins(bundle::f32_split::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::f32_wide::FAMILY,
            bundle::f32_wide::cubins(GEOMETRIES)?
        ),
        one!(
            Bundle,
            bundle::mxfp8_fold_stream::FAMILY,
            cubins(bundle::mxfp8_fold_stream::cubins(GEOMETRIES)?),
            "stream"
        ),
        one!(
            Bundle,
            bundle::moe_grouped_entries::FAMILY,
            vec![bundle::moe_grouped_entries::cubin()]
        ),
        one!(
            Bundle,
            bundle::hc_cols::FAMILY,
            bundle::hc_cols::cubins()?.into_iter().collect()
        ),
        one!(
            Lab,
            lab::hc_completion::FAMILY,
            vec![lab::hc_completion::cubin()?]
        ),
        one!(
            Bundle,
            bundle::hc_finish_twin::FAMILY,
            vec![bundle::hc_finish_twin::cubin()?]
        ),
        one!(
            Bundle,
            bundle::hc::PREFILL_FAMILY,
            cubins(bundle::hc::prefill_cubins(GEOMETRIES))
        ),
        one!(
            Bundle,
            bundle::hc::SPLIT_FAMILY,
            cubins(bundle::hc::split_cubins(GEOMETRIES))
        ),
        one!(
            Bundle,
            bundle::moe::ACTIVE_FAMILY,
            cubins(bundle::moe::active_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::moe::PREFILL_FAMILY,
            cubins(bundle::moe::prefill_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::scale_expand::FAMILY,
            vec![bundle::scale_expand::cubin()]
        ),
        one!(
            Bundle,
            bundle::indexer::RUNTIME_CANDIDATES_FAMILY,
            cubins(bundle::indexer::runtime_candidates_cubins(GEOMETRIES)?)
        ),
        one!(
            Lab,
            "position_hierarchical",
            cubins(bundle::indexer_hierarchical::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            "position_hierarchical",
            cubins(bundle::indexer_hierarchical::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer::LIVE_FAMILY,
            cubins(bundle::indexer::live_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer_mma::FAMILY,
            cubins(bundle::indexer_mma::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer_mma_guard::FAMILY,
            cubins(bundle::indexer_mma_guard::cubins(GEOMETRIES)?)
        ),
        one!(
            Lab,
            "position_mma_guard_fast",
            cubins(bundle::indexer_mma_guard_fast::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            "position_mma_guard_fast",
            cubins(bundle::indexer_mma_guard_fast::cubins(GEOMETRIES)?)
        ),
        one!(Bundle, "position_mma_guard_probe", {
            let families = bundle::indexer_mma_guard::cubins(GEOMETRIES)?
                .into_iter()
                .map(|(_, mut spec)| {
                    spec.name.push_str("_counts");
                    spec.entries = spec
                        .entries
                        .into_iter()
                        .filter(|entry| entry.symbol.ends_with("_mma_fp4_guard_v1"))
                        .map(kernels_sm120::pooled_indexer::position_mma_guard::diagnostic)
                        .collect();
                    ("position_mma_guard_probe", spec)
                })
                .collect::<Vec<_>>();
            families
                .iter()
                .map(|(_, spec)| spec.clone())
                .collect::<Vec<_>>()
        }),
        one!(
            Bundle,
            bundle::indexer_mma_prefill_tile::FAMILY,
            cubins(bundle::indexer_mma_prefill_tile::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer_mma_tile::FAMILY,
            cubins(bundle::indexer_mma_tile::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer_query_parallel::PARALLEL_FAMILY,
            cubins(bundle::indexer_query_parallel::parallel_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer_select_cached::FAMILY,
            cubins(bundle::indexer_select_cached::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::indexer::VALID_FAMILY,
            cubins(bundle::indexer::valid_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::prefill_attention::FAMILY,
            vec![bundle::prefill_attention::cubin()?]
        ),
        one!(
            Bundle,
            bundle::prefill_cache::FAMILY,
            vec![bundle::prefill_cache::cubin()?]
        ),
        one!(
            Bundle,
            bundle::mxfp8_fold::PREFILL_FAMILY,
            cubins(bundle::mxfp8_fold::prefill_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::mxfp8_fold::compact::FAMILY,
            cubins(bundle::mxfp8_fold::compact::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::mxfp8_fold::PREFILL_SPLIT_FAMILY,
            cubins(bundle::mxfp8_fold::prefill_split_cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::query_window::FAMILY,
            cubins(bundle::query_window::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::f32_zero::FAMILY,
            bundle::f32_zero::cubins(model_deepseek41_contract::geometry::GEOMETRIES)?
        ),
        one!(Lab, lab::ceil_cached::FAMILY, lab::ceil_cached::specs()),
        one!(
            Bundle,
            bundle::ceil_parallel::FAMILY,
            cubins(bundle::ceil_parallel::cubins(GEOMETRIES)?)
        ),
        one!(
            Lab,
            lab::compact_dense::FAMILY,
            lab::compact_dense::cubins()
        ),
        one!(
            Bundle,
            bundle::f32_verify::FAMILY,
            cubins(bundle::f32_verify::cubins(&[
                model_deepseek41_contract::geometry::FLASH
            ])?)
        ),
        one!(Lab, lab::fold_core::FAMILY, lab::fold_core::cubins()),
        one!(Lab, lab::fold_deep::FAMILY, lab::fold_deep::cubins()),
        one!(Lab, lab::fold_fused::FAMILY, lab::fold_fused::cubins()),
        one!(
            Lab,
            lab::fold_local_rows::FAMILY,
            lab::fold_local_rows::cubins()
        ),
        one!(Lab, lab::fold_plain::FAMILY, lab::fold_plain::cubins()),
        one!(
            Bundle,
            bundle::fold_verify::FAMILY,
            cubins(bundle::fold_verify::cubins(&[
                model_deepseek41_contract::geometry::FLASH
            ])?)
        ),
        one!(
            Bundle,
            bundle::fold_verify_local::FAMILY,
            cubins(bundle::fold_verify_local::cubins(&[
                model_deepseek41_contract::geometry::FLASH,
            ])?)
        ),
        one!(
            Bundle,
            lab::finish_norm_registration::FAMILY,
            cubins(lab::finish_norm_registration::cubins(GEOMETRIES)?)
        ),
        one!(
            Lab,
            lab::hc_prefill_fused::EXPAND_MIX_FAMILY,
            lab::hc_prefill_fused::expand_mix_specs()
        ),
        one!(
            Lab,
            lab::hc_prefill_fused::EXPAND_MIX_NORM_FAMILY,
            lab::hc_prefill_fused::expand_mix_norm_specs()
        ),
        one!(
            Lab,
            lab::hc_prefill_fused::FAMILY,
            lab::hc_prefill_fused::specs()
        ),
        one!(
            Lab,
            lab::hc_prefill_warp::FAMILY,
            lab::hc_prefill_warp::specs()
        ),
        one!(
            Bundle,
            bundle::hc_round_chain::FAMILY,
            cubins(bundle::hc_round_chain::cubins(GEOMETRIES)?)
        ),
        one!(
            Bundle,
            bundle::moe::FAMILY,
            moe::routed::cubins(GEOMETRIES)?
        ),
        one!(
            Bundle,
            bundle::moe::FLOOR_FAMILY,
            cubins(bundle::moe::floor_cubins(GEOMETRIES)?)
        ),
        one!(Lab, native::family::FAMILY, native::family::cubins()),
        one!(Lab, native::tune::FAMILY, native::tune::cubins()),
        one!(
            Bundle,
            bundle::norm_quantize_cached::FAMILY,
            cubins(bundle::norm_quantize_cached::cubins(GEOMETRIES)?)
        ),
        one!(
            Lab,
            lab::norm_query_cached::FAMILY,
            lab::norm_query_cached::specs()
        ),
        one!(
            Lab,
            lab::packed_serving::FAMILY,
            lab::packed_serving::cubins()
        ),
        one!(
            Lab,
            lab::projection_zero::FAMILY,
            lab::projection_zero::cubins()
        ),
        one!(Lab, lab::router_norm::FAMILY, lab::router_norm::cubins()),
    ]
}
