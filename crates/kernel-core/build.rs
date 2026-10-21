//! Marks an unoptimized build of this package with
//! `cfg(kernel_core_unoptimized)`. Only there does f64 arithmetic keep the
//! source order of its NaN operands, which `tests/rotary.rs` checks; the CI
//! profile (ci/rust-profile.env) optimizes this package like every other,
//! and the gate's `kernel-core-unoptimized` step runs that check at
//! opt-level 0.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(kernel_core_unoptimized)");
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("OPT_LEVEL").as_deref() == Ok("0") {
        println!("cargo::rustc-cfg=kernel_core_unoptimized");
    }
}
