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
#[path = "hc.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod hc;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent;
mod own_cubin;
// The build compiles the entries; the bundle launches them.
#[allow(dead_code, reason = "the bundle registers the entries")]
mod compressor;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod compressor_bf16;
#[path = "dense.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod dense;
#[allow(dead_code)]
mod f32_decode;
#[path = "f32_dense.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_dense;
#[path = "f32_head.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_head;
#[path = "indexer.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod indexer;
#[path = "indexer_query.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod indexer_query;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent_glue;
#[allow(dead_code)]
mod latent_window_bf16;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod latent_window_stage;
#[allow(dead_code, reason = "the bundle registers the entries")]
mod moe;
#[path = "mxfp8_fold.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod mxfp8_fold;
#[allow(dead_code)]
mod own_entries;
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
    specs.extend(dspark_aux::cubins(geometries).expect("DSpark auxiliary cubins"));
    specs.extend(dspark_router::cubins(geometries).expect("DSpark router cubins"));
    let mut owners = Vec::new();
    for geometry in geometries.iter().filter(|g| own_entries::has_gates(g)) {
        let spec = own_cubin::gate_cubin_spec(manifest, geometry);
        owners.push((spec.name.clone(), geometry.id));
        specs.push((own_entries::FAMILY, spec));
    }
    for geometry in geometries {
        owners.extend(dspark_aux::geometry_cubins(geometry).map(|name| (name, geometry.id)));
        owners.push((dspark_router::cubin_name(geometry), geometry.id));
        owners.extend(dspark_router::sum_cubins(geometry).map(|name| (name, geometry.id)));
        owners.push((rows::cubin_name(geometry), geometry.id));
        owners.push((rows::final_norm_cubin_name(geometry), geometry.id));
        owners.push((reduce::rounding_cubin(geometry), geometry.id));
        owners.push((hc::cubin_name(geometry), geometry.id));
        owners.push((hc::head_f32_cubin_name(geometry), geometry.id));
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
        owners.push((shared::cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::decode_cubin_name(geometry), geometry.id));
        owners.push((mxfp8_fold::ticket_cubin_name(geometry), geometry.id));
        owners.push((
            format!("latent_q_norm_{}_sm120", geometry.kernel_tag),
            geometry.id,
        ));
        owners.push((
            format!("position_query_{}_sm120", geometry.kernel_tag),
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
    specs.extend(rows::cubins(geometries).expect("deepseek41 row cubins"));
    specs.extend(reduce::cubins(geometries));
    specs.extend(hc::cubins(geometries));
    specs.extend(latent::cubins(geometries).expect("deepseek41 latent cubins"));
    specs.extend(indexer::runtime_candidates_cubins(geometries).expect("runtime candidate cubins"));
    specs.extend(indexer::cubins(geometries).expect("deepseek41 indexer cubins"));
    specs.extend(indexer::live_cubins(geometries).expect("live position cubins"));
    specs.extend(indexer::bf16_cubins(geometries).expect("deepseek41 BF16 indexer cubins"));
    specs.extend(compressor::cubins(geometries).expect("deepseek41 compressor cubins"));
    specs.extend(compressor_bf16::cubins(geometries).expect("deepseek41 BF16 index-key cubins"));
    specs.extend(moe::cubins(geometries).expect("deepseek41 routed MoE cubins"));
    specs.extend(moe::active_cubins(geometries).expect("active routed MoE cubins"));
    specs.extend(moe::streamed_cubins(geometries).expect("streamed routed MoE cubins"));
    specs.extend(moe::prefill_cubins(geometries).expect("grouped prefill MoE cubins"));
    specs.extend(routed_sum::cubins(geometries).expect("deepseek41 routed sum cubins"));
    specs.extend(dense::cubins(geometries).expect("deepseek41 dense quantizer cubins"));
    specs.extend(f32_head::cubins(geometries).expect("deepseek41 F32 head cubins"));
    specs.extend(f32_dense::cubins(geometries).expect("deepseek41 F32 dense cubins"));
    specs.extend(f32_decode::cubins(geometries).expect("cached F32 cubins"));
    specs.extend(shared::cubins(geometries).expect("deepseek41 shared expert cubins"));
    specs.push(("dependencies", kernels_sm120::probes::cubin_spec()));
    specs.extend(indexer_query::cubins(geometries).expect("deepseek41 query preparation cubins"));
    specs.extend(latent_glue::cubins(geometries).expect("deepseek41 latent glue cubins"));
    specs.extend(latent_window_stage::cubins(geometries).expect("window stage cubins"));
    specs.extend(latent_window_bf16::cubins(geometries).expect("BF16 window cubins"));
    specs.extend(latent_window_stage::key_cubins(geometries).expect("staged window key cubins"));
    specs.extend(mxfp8_fold::cubins(geometries).expect("deepseek41 MXFP8 fold cubins"));
    specs.extend(mxfp8_fold::decode_cubins(geometries).expect("deepseek41 MXFP8 decode cubins"));
    specs.extend(mxfp8_fold::ticket_cubins(geometries).expect("deepseek41 MXFP8 ticket cubins"));
    specs.extend(encoder::cubins().expect("deepseek41 encoder cubins"));
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
    // Every cubin serves each rank count it holds entries for.
    kernel_build::write_embedded(&out, &built, |_| 0).expect("write embedded cubin list");
    // The exact F32 projection cubins' identity, which authenticates their
    // offer: each cubin's name, digest and source closure digest.
    let f32: String = built
        .iter()
        .filter(|cubin| {
            GEOMETRIES.iter().any(|g| {
                cubin.name == f32_decode::cubin_name(g)
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
