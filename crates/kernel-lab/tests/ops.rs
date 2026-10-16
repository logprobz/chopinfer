#[path = "ops/moe.rs"]
mod moe;
#[test]
fn gpu_entry() {
    kernel_lab::gpu_test(0, |_| Ok(())).expect("GPU gate");
}
#[path = "ops/attention.rs"]
mod attention;
#[path = "ops/bundle_invariance.rs"]
mod bundle_invariance;
#[path = "ops/context_parallel_invariance.rs"]
mod context_parallel_invariance;
#[path = "ops/dflash.rs"]
mod dflash;
#[path = "ops/gemm.rs"]
mod gemm;
#[path = "ops/latent_invariance.rs"]
mod latent_invariance;
#[path = "ops/linear.rs"]
mod linear;
#[path = "ops/pooled_invariance.rs"]
mod pooled_invariance;
#[path = "ops/prepared_invariance.rs"]
mod prepared_invariance;
#[path = "ops/row.rs"]
mod row;
#[path = "ops/sinkhorn_invariance.rs"]
mod sinkhorn_invariance;
#[path = "ops/sparse_attention.rs"]
mod sparse_attention;
#[rustfmt::skip] #[path = "ops/sampling.rs"] mod sampling;
#[path = "common/invariance.rs"]
mod invariance;
#[path = "ops/invariance_tests.rs"]
mod invariance_tests;
#[path = "common/pool_offsets.rs"]
mod pool_offsets;
#[path = "ops/verify_drift_stats.rs"]
mod verify_drift_stats;

#[path = "common/owned_cubins.rs"]
mod owned_cubins;

#[path = "ops/generated_projections.rs"]
mod generated_projections;
