//! Compiles the bundle's cubins of the merged engines, the projections'
//! shared GEMM entries, the KDA layers' stacked input projections, the
//! vision encoder and the shared row copies, each gated by its own family
//! of `kernels.lock.d` (`kernel_build::LockDir`), and embeds them.
use std::env;
use std::path::PathBuf;

use kernel_build::{LockDir, ModelLock, Toolchain};
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::sm120::{encoder, engines, precision, projection_formats, stacked};

fn main() {
    println!("cargo:rerun-if-changed=src/sm120/build.rs");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    let mut specs = engines::cubin_specs(Shape::TP4).expect("glm5next engine cubins");
    specs.extend(encoder::cubin_specs().expect("glm5next encoder cubins"));
    specs.extend(
        precision::cubin_specs(Shape::TP4)
            .expect("glm5next projection cubins")
            .into_iter()
            .map(|spec| (precision::FAMILY, spec)),
    );
    specs.push((
        precision::DECODED_FAMILY,
        precision::decoded_cubin_spec(Shape::TP4).expect("glm5next decoded-tile projection cubin"),
    ));
    specs.push((
        precision::STAGED_FAMILY,
        precision::staged_cubin_spec(Shape::TP4).expect("glm5next staged-tile projection cubin"),
    ));
    specs.extend(
        projection_formats::cubin_specs(Shape::TP4)
            .expect("glm5next NVFP4 layouts' projection cubins")
            .into_iter()
            .map(|spec| (projection_formats::FAMILY, spec)),
    );
    specs.extend(
        stacked::cubin_specs(Shape::TP4)
            .expect("glm5next stacked KDA input projection cubins")
            .into_iter()
            .map(|spec| (stacked::FAMILY, spec)),
    );
    // The shared row copies, loaded after every other text cubin, then the
    // shared gather by index.
    specs.push((engines::ROW_COPY_FAMILY, engines::row_copy_cubin_spec()));
    specs.push((engines::ROW_GATHER_FAMILY, engines::row_gather_cubin_spec()));
    // One family per engine, holding every cubin it builds.
    let mut families: Vec<(&str, Vec<&str>)> = Vec::new();
    for (family, spec) in &specs {
        match families.iter_mut().find(|(name, _)| name == family) {
            Some((_, cubins)) => cubins.push(&spec.name),
            None => families.push((family, vec![&spec.name])),
        }
    }
    // Model-owned: `tools.model_sdk relock` redirects it in an export.
    let model_lock =
        ModelLock::from_env(manifest.join("kernels.lock.d")).expect("glm5next kernel lock");
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
    .expect("locked glm5next cubin build and audit");
    kernel_build::write_embedded(&out, &built, |_| 4).expect("write embedded cubin list");
}
