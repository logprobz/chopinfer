//! The bundle's build script, which compiles and locks SM120 cubins, so it
//! lives in the sm120 module (`build` in Cargo.toml). Each engine's cubins
//! gate on their own family of `kernels.lock.d` (`kernel_build::LockDir`),
//! one cubin per geometry within it where its entries depend on the
//! geometry, so a geometry added later only appends records. Each engine
//! adds its selection here as it registers.

use std::env;
use std::path::PathBuf;

use kernel_build::{LockDir, ModelLock, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;

mod acceptance_cubin;
mod acceptance_entries;
#[allow(dead_code)]
mod ceil_parallel;
#[allow(dead_code)]
mod ceil_prefill;
mod confidence_parallel;
#[allow(dead_code, reason = "the bundle registers the auxiliary entries")]
mod dspark_aux;
mod dspark_cubin;
mod dspark_entries;
#[path = "dspark_router.rs"]
#[allow(dead_code)]
mod dspark_router;
#[path = "encoder.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod encoder;
mod engram_hash_cubin;
mod engram_hash_entries;
#[path = "hc.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod hc;
#[allow(dead_code)]
mod hc_expand_mix;
#[allow(dead_code, reason = "the bundle selects measured fused tails")]
mod hc_finish_norm;
#[allow(dead_code)]
mod hc_mixer_probe;
#[allow(
    dead_code,
    reason = "the bundle registers the measured prefill entries"
)]
mod hc_prefill;
#[path = "hc_round_chain.rs"]
#[allow(dead_code, reason = "the bundle registers the measured entries")]
mod hc_round_chain;
#[allow(dead_code)]
mod hc_round_expand_mix;
#[allow(dead_code, reason = "the bundle registers measured row tiles")]
mod hc_round_rows;
mod host_acceptance_cubin;
mod host_acceptance_entries;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent;
#[path = "latent_cache.rs"]
#[allow(dead_code)]
mod latent_cache;
#[allow(dead_code)]
mod latent_decode;
#[allow(dead_code)]
mod latent_scheduled;
#[allow(dead_code)]
mod norm_prefill;
#[allow(dead_code)]
mod norm_quantize;
#[path = "norm_quantize_cached.rs"]
#[allow(dead_code, reason = "the bundle registers the measured entries")]
mod norm_quantize_cached;
#[path = "norm_quantize_tiles.rs"]
mod norm_quantize_tiles;
mod own_cubin;
#[allow(dead_code, reason = "the bundle prepares the paired schedule")]
mod prefill_attention;
#[allow(dead_code, reason = "the bundle prepares the decoded schedule")]
mod prefill_cache;
#[allow(dead_code, reason = "the bundle prepares the qualified head schedules")]
mod prefill_heads;
#[allow(dead_code)]
mod query_prefill;
#[allow(dead_code)]
mod replicated_embedding;
mod ring_gather_cubin;
mod ring_gather_entries;
#[allow(dead_code)]
mod window_fused;
// The build compiles the entries; the bundle launches them.
#[allow(dead_code, reason = "the bundle registers the entries")]
mod compressor;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod compressor_bf16;
#[path = "dense.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod dense;
#[allow(dead_code)]
mod f32_blocked;
#[allow(dead_code)]
mod f32_decode;
#[path = "f32_dense.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_dense;
#[path = "f32_head.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_head;
#[allow(dead_code)]
mod f32_prefill;
#[allow(dead_code)]
mod f32_split;
mod f32_wide;
#[allow(dead_code)]
mod head_last;
#[path = "indexer.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod indexer;
mod indexer_hierarchical;
mod indexer_mma;
mod indexer_mma_guard;
mod indexer_mma_guard_fast;
mod indexer_mma_prefill_tile;
mod indexer_mma_tile;
#[path = "indexer_query.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod indexer_query;
#[allow(dead_code)]
mod indexer_query_parallel;
mod indexer_select_cached;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent_glue;
#[allow(dead_code)]
mod latent_window_bf16;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent_window_stage;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod moe;
#[allow(dead_code)]
mod moe_direct_kernels;
#[allow(dead_code)]
mod moe_grouped_entries;
#[allow(dead_code)]
mod moe_pipeline_entries;
#[allow(dead_code)]
mod moe_pipeline_vector;
#[path = "mxfp8_fold.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod mxfp8_fold;
#[allow(dead_code)]
mod mxfp8_fold_direct_entries;
#[allow(dead_code)]
mod mxfp8_fold_stream;
#[allow(dead_code)]
mod own_entries;
#[allow(dead_code)]
mod projection_reuse;
#[path = "reduce.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod reduce;
#[path = "routed_sum.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod routed_sum;
#[path = "rows.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod rows;
#[path = "shared.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod shared;

/// The cubins the bundle compiles for `geometries`, each with its lock
/// family, and the geometry of each cubin that serves one geometry alone;
/// the other cubins serve every geometry. One cubin per geometry within a
/// family, so a geometry added later only appends records
/// (`tests/sm120/projection_assembly.rs`).
pub struct Selection {
    pub specs: Vec<(&'static str, kernel_build::CubinSpec)>,
    pub owners: Vec<(String, &'static str)>,
}

pub fn selection(
    manifest: &std::path::Path,
    geometries: &'static [model_deepseek41_contract::geometry::Geometry],
) -> Selection {
    let mut specs = vec![(own_entries::FAMILY, own_cubin::rope_cubin_spec(manifest))];
    specs.push((dspark_entries::FAMILY, dspark_cubin::cubin_spec(manifest)));
    specs.push((
        confidence_parallel::FAMILY,
        confidence_parallel::cubin_spec(manifest),
    ));
    specs.push((
        ring_gather_entries::FAMILY,
        ring_gather_cubin::cubin_spec(manifest),
    ));
    specs.push((
        engram_hash_entries::FAMILY,
        engram_hash_cubin::cubin_spec(manifest),
    ));
    specs.push((
        acceptance_entries::FAMILY,
        acceptance_cubin::cubin_spec(manifest),
    ));
    specs.push((
        host_acceptance_entries::FAMILY,
        host_acceptance_cubin::cubin_spec(manifest),
    ));
    specs.extend(dspark_aux::cubins(geometries).expect("DSpark auxiliary cubins"));
    specs.extend(dspark_router::cubins(geometries).expect("DSpark router cubins"));
    let mut owners = Vec::new();
    if geometries
        .iter()
        .any(|g| g.id == model_deepseek41_contract::geometry::FLASH.id)
    {
        let spec = prefill_attention::cubin().expect("paired prefill cubin");
        owners.push((
            spec.name.clone(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
        specs.push((prefill_attention::FAMILY, spec));
        specs.extend(
            moe_pipeline_entries::cubins()
                .into_iter()
                .map(|s| (moe_pipeline_entries::FAMILY, s)),
        );
        owners.push((
            moe_pipeline_entries::CUBIN.into(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
        specs.extend(
            moe_pipeline_vector::cubins(moe_pipeline_entries::cubins())
                .into_iter()
                .map(|s| (moe_pipeline_vector::FAMILY, s)),
        );
        owners.push((
            moe_pipeline_vector::CUBIN.into(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
        let spec = moe_grouped_entries::cubin();
        owners.push((
            spec.name.clone(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
        specs.push((moe_grouped_entries::FAMILY, spec));
        let spec = prefill_cache::cubin().expect("decoded prefill cubin");
        owners.push((
            spec.name.clone(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
        specs.push((prefill_cache::FAMILY, spec));
        for (family, spec) in prefill_heads::cubins().expect("shared-head prefill cubins") {
            owners.push((
                spec.name.clone(),
                model_deepseek41_contract::geometry::FLASH.id,
            ));
            specs.push((family, spec));
        }
        owners.push((
            mxfp8_fold::compact::CUBIN.into(),
            model_deepseek41_contract::geometry::FLASH.id,
        ));
    }

    for g in geometries
        .iter()
        .filter(|g| latent_scheduled::qualified(g, 4))
    {
        let spec = latent_decode::cubin(g).expect("measured decode staging cubin");
        owners.push((spec.name.clone(), g.id));
        specs.push((latent_decode::FAMILY, spec));
        let spec = latent_cache::cubin(g).expect("qualified selected KV cache cubin");
        owners.push((spec.name.clone(), g.id));
        specs.push((latent_cache::FAMILY, spec));
    }
    for geometry in geometries.iter().filter(|g| own_entries::has_gates(g)) {
        let spec = own_cubin::gate_cubin_spec(manifest, geometry);
        owners.push((spec.name.clone(), geometry.id));
        specs.push((own_entries::FAMILY, spec));
    }
    for geometry in geometries {
        owners.extend(dspark_aux::geometry_cubins(geometry).map(|name| (name, geometry.id)));
        owners.extend(
            dspark_router::cubins(std::slice::from_ref(geometry))
                .expect("DSpark router cubins")
                .into_iter()
                .map(|(_, spec)| (spec.name, geometry.id)),
        );
        owners.push((norm_quantize::cubin_name(geometry), geometry.id));
        if geometry == &model_deepseek41_contract::geometry::FLASH {
            owners.push((norm_prefill::CUBIN.into(), geometry.id));
            owners.push((query_prefill::CUBIN.into(), geometry.id));
        }
        if hc_prefill::qualified(geometry) {
            owners.push((hc_prefill::CUBIN.into(), geometry.id));
        }
        owners.push((hc_expand_mix::cubin_name(geometry), geometry.id));
        owners.push((hc_round_expand_mix::cubin_name(geometry), geometry.id));
        if geometry.id == model_deepseek41_contract::geometry::FLASH.id {
            owners.push((hc_mixer_probe::cubin_name(geometry), geometry.id));
            owners.push((norm_quantize_tiles::cubin_name(geometry), geometry.id));
            owners.push((norm_quantize_cached::cubin_name(geometry), geometry.id));
            owners.push((hc_round_chain::cubin_name(geometry), geometry.id));
            owners.push((hc_finish_norm::cubin_name(geometry), geometry.id));
            owners.push((hc_round_rows::cubin_name(geometry), geometry.id));
        }
        owners.push((replicated_embedding::cubin_name(geometry), geometry.id));
        owners.push((ceil_parallel::cubin_name(geometry), geometry.id));
        owners.push((ceil_prefill::cubin_name(geometry), geometry.id));
        owners.push((rows::cubin_name(geometry), geometry.id));
        owners.push((rows::final_norm_cubin_name(geometry), geometry.id));
        owners.push((reduce::rounding_cubin(geometry), geometry.id));
        owners.extend(
            hc::cubins(std::slice::from_ref(geometry))
                .into_iter()
                .map(|(_, spec)| (spec.name, geometry.id)),
        );
        owners.push((
            format!("latent_sources_scheduled_{}_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        for cubin in ["sm120", "p32_sm120", "bf16_sm120"] {
            owners.push((
                format!("latent_sources_{}_{cubin}", geometry.kernel_tag),
                geometry.id,
            ));
        }
        owners.push((
            format!("position_indexer_{}_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        owners.push((indexer::blocks_cubin_name(geometry), geometry.id));
        owners.push((indexer::live_cubin_name(geometry), geometry.id));
        owners.push((indexer::valid_cubin_name(geometry), geometry.id));
        if geometry.native_indexer_supported() {
            owners.push((indexer_mma::cubin_name(geometry), geometry.id));
            owners.push((indexer_mma_guard::cubin_name(geometry), geometry.id));
            owners.push((indexer_mma_guard_fast::cubin_name(geometry), geometry.id));
            owners.push((indexer_hierarchical::cubin_name(geometry), geometry.id));
        }
        if geometry == &model_deepseek41_contract::geometry::FLASH {
            owners.push((indexer_select_cached::CUBIN.into(), geometry.id));
            owners.push((indexer_mma_tile::CUBIN.into(), geometry.id));
        }
        owners.push((
            indexer::runtime_candidates_cubin_name(geometry),
            geometry.id,
        ));
        owners.extend(indexer::bf16_cubin_names(geometry).map(|name| (name, geometry.id)));
        for part in ["compute", "routes", "quantize", "active", "active_routes"] {
            owners.push((
                format!("moe_{}_{part}_sm120", geometry.kernel_tag),
                geometry.id,
            ));
        }
        owners.push((
            format!("prefill_moe_{}_compute_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        if moe::streamed_qualified(geometry, 4) {
            for part in ["compute", "route"] {
                owners.push((
                    format!("floor_moe_{}_{part}_sm120", geometry.kernel_tag),
                    geometry.id,
                ));
            }
        }
        for part in ["compute", "routes"] {
            owners.push((
                format!("streamed_moe_{}_{part}_sm120", geometry.kernel_tag),
                geometry.id,
            ));
        }
        for cubin in routed_sum::cubin_names(geometry) {
            owners.push((cubin, geometry.id));
        }
        owners.push((dense::cubin_name(geometry), geometry.id));
        owners.push((f32_head::cubin_name(geometry), geometry.id));
        owners.push((f32_dense::cubin_name(geometry), geometry.id));
        owners.push((f32_decode::cubin_name(geometry), geometry.id));
        if geometry.id == model_deepseek41_contract::geometry::FLASH.id {
            owners.push((projection_reuse::FOLD_CUBIN.into(), geometry.id));
            owners.push((projection_reuse::HEAD_CUBIN.into(), geometry.id));
            owners.push((head_last::CUBIN.into(), geometry.id));
        }
        owners.push((f32_split::cubin_name(geometry), geometry.id));
        for features in f32_prefill::TILE_FEATURES {
            owners.push((f32_prefill::cubin_name(geometry, features), geometry.id));
        }
        if geometry.id == model_deepseek41_contract::geometry::FLASH.id {
            owners.push((f32_blocked::serving_cubin_name(geometry), geometry.id));
            owners.push((f32_wide::cubin_name(geometry), geometry.id));
        }
        owners.push((shared::cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::decode_cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::ticket_cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::prefill_cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::prefill_split_cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::prefetch_cubin_name(geometry), geometry.id));
        if geometry.id == model_deepseek41_contract::geometry::FLASH.id {
            owners.push((mxfp8_fold_stream::cubin_name(geometry), geometry.id));
            owners.push((mxfp8_fold_direct_entries::CUBIN.into(), geometry.id));
        }
        owners.push((
            format!("latent_q_norm_{}_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        owners.push((
            format!("position_query_{}_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        owners.push((
            indexer_query_parallel::parallel_cubin_name(geometry),
            geometry.id,
        ));
        owners.push((compressor_bf16::cubin_name(geometry), geometry.id));
        for part in ["pooling", "keys", "ratio_one"] {
            owners.push((
                format!("{}_compressor_{part}_v1", geometry.kernel_tag),
                geometry.id,
            ));
        }
    }
    specs.extend(norm_quantize::cubins(geometries).expect("fused norm cubins"));
    specs.extend(norm_prefill::cubins(geometries).expect("prefill fused norm cubins"));
    specs.extend(query_prefill::cubins(geometries).expect("prefill query norm cubins"));
    specs.extend(hc_expand_mix::cubins(geometries).expect("fused expansion mixer cubins"));
    specs.extend(hc_round_expand_mix::cubins(geometries).expect("fused rounded expansion cubins"));
    for geometry in geometries
        .iter()
        .filter(|g| g.id == model_deepseek41_contract::geometry::FLASH.id)
    {
        specs.extend(
            norm_quantize_cached::cubins(std::slice::from_ref(geometry))
                .expect("qualified cached norm cubin"),
        );
        specs.extend(
            hc_finish_norm::cubins(std::slice::from_ref(geometry))
                .expect("qualified HC finish norm cubin"),
        );
        specs.extend(
            hc_round_chain::cubins(std::slice::from_ref(geometry))
                .expect("qualified rounded chain cubin"),
        );
        specs.extend(
            hc_round_rows::cubins(std::slice::from_ref(geometry))
                .expect("qualified rounded row cubin"),
        );
        specs.extend(
            norm_quantize_tiles::cubins(std::slice::from_ref(geometry))
                .expect("qualified norm tiles"),
        );
        specs.extend(
            hc_mixer_probe::cubins(std::slice::from_ref(geometry))
                .expect("qualified warp mixer cubin"),
        );
    }
    specs.extend(replicated_embedding::cubins(geometries).expect("replicated embedding cubins"));
    specs.push((
        window_fused::FAMILY,
        window_fused::cubin().expect("fused window cubin"),
    ));
    specs.extend(ceil_parallel::cubins(geometries).expect("parallel ceil cubins"));
    specs.extend(ceil_prefill::cubins(geometries).expect("prefill ceil cubins"));
    specs.extend(rows::cubins(geometries).expect("deepseek41 row cubins"));
    specs.extend(reduce::cubins(geometries));
    specs.extend(hc::cubins(geometries));
    specs.extend(hc_prefill::cubins(geometries).expect("measured HC prefill cubin"));
    specs.extend(latent::cubins(geometries).expect("deepseek41 latent cubins"));
    specs.extend(indexer::runtime_candidates_cubins(geometries).expect("runtime candidate cubins"));
    specs.extend(indexer::cubins(geometries).expect("deepseek41 indexer cubins"));
    specs.extend(indexer::live_cubins(geometries).expect("live position cubins"));
    specs.extend(indexer::valid_cubins(geometries).expect("valid position cubins"));
    specs.extend(indexer_mma::cubins(geometries).expect("native MMA position cubins"));
    specs.extend(indexer_mma_guard::cubins(geometries).expect("guarded MMA position cubins"));
    specs.extend(
        indexer_mma_guard_fast::cubins(geometries).expect("coarse guarded MMA position cubins"),
    );
    specs.extend(indexer_hierarchical::cubins(geometries).expect("ordered position merge cubins"));
    specs.extend(indexer_select_cached::cubins(geometries).expect("cached position select cubin"));
    specs.extend(indexer_mma_tile::cubins(geometries).expect("tiled position score cubin"));
    specs
        .extend(indexer_mma_prefill_tile::cubins(geometries).expect("prefill position tile cubin"));
    specs.extend(indexer::bf16_cubins(geometries).expect("deepseek41 BF16 indexer cubins"));
    specs.extend(compressor::cubins(geometries).expect("deepseek41 compressor cubins"));
    specs.extend(compressor_bf16::cubins(geometries).expect("deepseek41 BF16 index-key cubins"));
    specs.extend(moe::cubins(geometries).expect("deepseek41 routed MoE cubins"));
    specs.extend(moe::active_cubins(geometries).expect("active routed MoE cubins"));
    specs.extend(moe::streamed_cubins(geometries).expect("streamed routed MoE cubins"));
    specs.extend(moe::prefill_cubins(geometries).expect("grouped prefill MoE cubins"));
    specs.extend(moe::floor_cubins(geometries).expect("fused decode MoE cubins"));
    specs.extend(routed_sum::cubins(geometries).expect("deepseek41 routed sum cubins"));
    specs.extend(dense::cubins(geometries).expect("deepseek41 dense quantizer cubins"));
    specs.extend(f32_head::cubins(geometries).expect("deepseek41 F32 head cubins"));
    specs.extend(f32_dense::cubins(geometries).expect("deepseek41 F32 dense cubins"));
    specs.extend(f32_decode::cubins(geometries).expect("cached F32 cubins"));
    specs.extend(projection_reuse::cubins(geometries).expect("qualified projection reuse cubins"));
    specs.extend(head_last::cubins(geometries).expect("qualified ordered head cubin"));
    specs.extend(f32_split::cubins(geometries).expect("fused split F32 cubins"));
    specs.extend(
        f32_prefill::cubins(geometries)
            .expect("measured prefill F32 cubins")
            .into_iter()
            .map(|spec| (f32_prefill::FAMILY, spec)),
    );
    specs.extend(shared::cubins(geometries).expect("deepseek41 shared expert cubins"));
    specs.extend(
        f32_blocked::serving_cubins(geometries)
            .expect("blocked prefill F32 cubins")
            .into_iter()
            .map(|spec| (f32_blocked::SERVING_FAMILY, spec)),
    );
    specs.extend(
        f32_wide::cubins(geometries)
            .expect("wide prefill F32 cubins")
            .into_iter()
            .map(|s| (f32_wide::FAMILY, s)),
    );
    specs.push(("dependencies", kernels_sm120::probes::cubin_spec()));
    specs.extend(indexer_query::cubins(geometries).expect("deepseek41 query preparation cubins"));
    specs.extend(
        indexer_query_parallel::parallel_cubins(geometries).expect("parallel query cubins"),
    );
    specs.extend(latent_glue::cubins(geometries).expect("deepseek41 latent glue cubins"));
    specs.extend(
        latent_scheduled::cubins(geometries)
            .expect("scheduled latent source cubins")
            .into_iter()
            .map(|spec| (latent_scheduled::FAMILY, spec)),
    );
    specs.push((
        latent_scheduled::QUERY_FAMILY,
        latent_scheduled::query_cubin(),
    ));
    specs.extend(latent_window_stage::cubins(geometries).expect("window stage cubins"));
    specs.extend(latent_window_bf16::cubins(geometries).expect("BF16 window cubins"));
    specs.extend(latent_window_stage::key_cubins(geometries).expect("staged window key cubins"));
    specs.extend(mxfp8_fold::cubins(geometries).expect("deepseek41 MXFP8 fold cubins"));
    specs.extend(mxfp8_fold::decode_cubins(geometries).expect("deepseek41 MXFP8 decode cubins"));
    specs.extend(mxfp8_fold::ticket_cubins(geometries).expect("deepseek41 MXFP8 ticket cubins"));
    specs.extend(mxfp8_fold::prefill_cubins(geometries).expect("MXFP8 prefill cubins"));
    specs.extend(mxfp8_fold::prefill_split_cubins(geometries).expect("MXFP8 split prefill cubins"));
    specs.extend(mxfp8_fold::compact::cubins(geometries).expect("compact MXFP8 prefill cubins"));
    specs
        .extend(mxfp8_fold::prefetch_cubins(geometries).expect("deepseek41 MXFP8 prefetch cubins"));
    specs.extend(mxfp8_fold_stream::cubins(geometries).expect("MXFP8 packed fold cubins"));
    if geometries
        .iter()
        .any(|g| g.id == model_deepseek41_contract::geometry::FLASH.id)
    {
        specs.push((
            mxfp8_fold_direct_entries::FAMILY,
            mxfp8_fold_direct_entries::cubin().expect("direct fold cubin"),
        ));
    }
    specs.extend(encoder::cubins().expect("deepseek41 encoder cubins"));
    // A composition may already carry the floor family for a drafter route.
    // Reuse that identical locked module; add only absent direct dependencies.
    for geometry in geometries {
        for (family, spec) in moe_direct_kernels::cubins(std::slice::from_ref(geometry))
            .expect("locked direct MoE cubins")
        {
            if let Some((_, owner)) = owners.iter().find(|(name, _)| name == &spec.name) {
                assert_eq!(*owner, geometry.id, "MoE module has a different owner");
            } else {
                owners.push((spec.name.clone(), geometry.id));
            }
            if let Some((prior_family, existing)) = specs
                .iter()
                .find(|(_, existing)| existing.name == spec.name)
            {
                assert_eq!(*prior_family, family);
                assert_eq!(existing.sources_root, spec.sources_root);
                assert_eq!(existing.include_dirs, spec.include_dirs);
                assert_eq!(existing.defines, spec.defines);
                assert_eq!(
                    kernel_build::generate(existing).unwrap().translation_unit,
                    kernel_build::generate(&spec).unwrap().translation_unit,
                    "conflicting locked MoE declarations"
                );
                assert_eq!(
                    existing
                        .entries
                        .iter()
                        .map(|e| e.allowed_local_bytes)
                        .collect::<Vec<_>>(),
                    spec.entries
                        .iter()
                        .map(|e| e.allowed_local_bytes)
                        .collect::<Vec<_>>()
                );
            } else {
                specs.push((family, spec));
            }
        }
    }
    Selection { specs, owners }
}

fn main() {
    println!("cargo:rerun-if-changed=src/sm120");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    let Selection { specs, owners } = selection(&manifest, GEOMETRIES);
    let mut families: Vec<(&str, Vec<&str>)> = Vec::new();
    for (family, spec) in specs.iter().map(|(family, spec)| (*family, spec)) {
        match families.iter_mut().find(|(name, _)| *name == family) {
            Some((_, cubins)) => cubins.push(&spec.name),
            None => families.push((family, vec![&spec.name])),
        }
    }
    // Model-owned: `tools.model_sdk relock` redirects it in an export.
    let model_lock =
        ModelLock::from_env(manifest.join("kernels.lock.d")).expect("deepseek41 kernel lock");
    let lock = families
        .into_iter()
        .fold(LockDir::new(model_lock.path()), |lock, (family, cubins)| {
            lock.family(family, cubins)
        });
    let specs: Vec<_> = specs.into_iter().map(|(_, spec)| spec).collect();
    let built = kernel_build::build_in_families(
        &Toolchain::detect().expect("CUDA 13.2 toolchain"),
        &specs,
        &out,
        &lock,
        model_lock.options(),
        None,
    )
    .expect("locked deepseek41 cubin build and audit");
    // Keep argument access tied to the exact locked entry declarations.
    // Tensor maps have no pointee qualifier, so they cannot grant a shared
    // input exemption. Unknown symbols and argument positions also refuse it.
    let mut access = std::collections::BTreeMap::new();
    for entry in specs.iter().flat_map(|spec| &spec.entries) {
        let inputs: Vec<_> = entry
            .params
            .iter()
            .map(|param| match param {
                kernel_build::EntryParam::DevicePointer { cxx_type, .. }
                | kernel_build::EntryParam::PointerArray { cxx_type, .. } => {
                    cxx_type.trim_start().starts_with("const ")
                }
                _ => false,
            })
            .collect();
        if let Some(previous) = access.insert(entry.symbol.as_str(), inputs.clone()) {
            assert_eq!(previous, inputs, "conflicting argument access declarations");
        }
    }
    let rows: String = access
        .iter()
        .map(|(symbol, inputs)| format!("({symbol:?}, &{inputs:?}),"))
        .collect();
    std::fs::write(
        out.join("argument_access.rs"),
        format!("const CONST_INPUT_ARGUMENTS: &[(&str, &[bool])] = &[{rows}];\n"),
    )
    .expect("write declared argument access");
    // Every cubin serves each rank count it holds entries for.
    kernel_build::write_embedded(&out, &built, |_| 0).expect("write embedded cubin list");
    // The exact F32 projection cubins' identity, which authenticates their
    // offer: each cubin's name, digest and source closure digest.
    let f32: String = built
        .iter()
        .filter(|cubin| {
            GEOMETRIES.iter().any(|g| {
                f32_prefill::TILE_FEATURES
                    .iter()
                    .any(|&features| cubin.name == f32_prefill::cubin_name(g, features))
                    || cubin.name == f32_split::cubin_name(g)
                    || cubin.name == f32_blocked::serving_cubin_name(g)
                    || cubin.name == f32_wide::cubin_name(g)
                    || cubin.name == f32_decode::cubin_name(g)
                    || cubin.name == projection_reuse::HEAD_CUBIN
                    || cubin.name == head_last::CUBIN
                    || cubin.name == f32_head::cubin_name(g)
                    || cubin.name == f32_dense::cubin_name(g)
                    || cubin.name == dspark_router::cubin_name(g)
            })
        })
        .map(|cubin| {
            format!(
                "{} {} {}\n",
                cubin.name, cubin.sha256, cubin.source_closure_sha256
            )
        })
        .collect();
    std::fs::write(
        out.join("f32_dependency.txt"),
        kernel_build::digest(format!("deepseek41.f32-projections.v1\n{f32}").as_bytes()),
    )
    .expect("write the F32 projection dependency");
    // The geometry of each cubin that serves one geometry; the others serve
    // every geometry.
    let owners: String = owners
        .iter()
        .map(|(cubin, geometry)| format!("({cubin:?}, {geometry:?}),"))
        .collect();
    std::fs::write(
        out.join("geometry_cubins.rs"),
        format!("pub(crate) const GEOMETRY_CUBINS: &[(&str, &str)] = &[{owners}];\n"),
    )
    .expect("write geometry cubin list");
}
