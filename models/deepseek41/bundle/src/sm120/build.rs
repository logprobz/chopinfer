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

#[path = "encoder.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod encoder;
#[path = "hc.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod hc;
mod own_cubin;
// The build compiles the entries; the bundle launches them.
#[path = "f32_dense.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_dense;
#[path = "f32_head.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod f32_head;
#[allow(dead_code)]
mod own_entries;
#[path = "reduce.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod reduce;
#[path = "rows.rs"]
#[allow(dead_code, reason = "the bundle registers the entries")]
mod rows;

fn main() {
    println!("cargo:rerun-if-changed=src/sm120");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    // (family, cubin) pairs; one family per engine or own-op set.
    let mut specs = vec![(own_entries::FAMILY, own_cubin::rope_cubin_spec(&manifest))];
    for geometry in GEOMETRIES.iter().filter(|g| own_entries::has_gates(g)) {
        specs.push((
            own_entries::FAMILY,
            own_cubin::gate_cubin_spec(&manifest, geometry),
        ));
    }
    specs.extend(rows::cubins(GEOMETRIES).expect("deepseek41 row cubins"));
    specs.extend(reduce::cubins(GEOMETRIES));
    specs.extend(hc::cubins(GEOMETRIES));
    specs.extend(f32_head::cubins(GEOMETRIES).expect("deepseek41 F32 head cubins"));
    specs.extend(f32_dense::cubins(GEOMETRIES).expect("deepseek41 F32 dense cubins"));
    specs.push(("dependencies", kernels_sm120::probes::cubin_spec()));
    specs.extend(encoder::cubins().expect("deepseek41 encoder cubins"));
    let mut families: Vec<(&str, Vec<&str>)> = Vec::new();
    for (family, spec) in &specs {
        match families.iter_mut().find(|(name, _)| name == family) {
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
                cubin.name == f32_head::cubin_name(g) || cubin.name == f32_dense::cubin_name(g)
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
}
