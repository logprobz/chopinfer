//! A whole context-parallel latent attention step on one GPU at group sizes
//! 2 and 4 against the single-rank result (`common/latent_checks.rs`,
//! `context_parallel_steps`): every group rank holds only its own pages,
//! localizes the key lists, runs the existing partial attention entry once
//! per query block over its own keys, and each owner merges its own heads
//! from the partials as the row exchange and the all-gather deliver them.
//! The collectives themselves are checked on several GPUs by
//! `tests/context_parallel_comm.rs`.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test context_parallel_attention`; without a GPU it skips.

#[path = "common/latent_checks.rs"]
mod latent_checks;
#[path = "../context_parallel_shapes.rs"]
mod shapes;

use kernel_lab::gpu_test;
use latent_checks::{ContextParallelKernels, Kernels, context_parallel_steps};

#[test]
fn context_parallel_steps_match_the_single_rank_result() {
    gpu_test(8 << 30, |session| {
        let cp = ContextParallelKernels {
            module: session.module(shapes::CUBIN)?,
            entries: shapes::lab_entries(),
        };
        context_parallel_steps(session, &Kernels::lab(session)?, &cp)
    })
    .expect("context-parallel attention steps");
}
