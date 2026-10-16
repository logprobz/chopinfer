//! Latent attention engine (`kernels_sm120::latent`) against the
//! kernel-core oracles (`latent_attention_oracle`) on one GPU at the lab's
//! served shape (`kernel_lab::latent_shapes`): 16 heads, latent 512, 64-token
//! pages, 2051-entry key lists, FP8 and BF16 pages.
//!
//! - The latent KV write matches the oracle bit for bit, including the
//!   bytes between pages it must not touch.
//! - Attention in both output modes matches the oracle within the
//!   decision-0081 gate for decode (batches 1, 8 and 32), verify (items of
//!   4 and 6 rows) and prefill (512 and 8192 rows), with selected pools, the
//!   incomplete tail, -1 holes and contexts across the 2051-token boundary.
//! - The 128-K test: the partial maximum (one 512-wide score) and numerator
//!   and the F32 projections at K = 256 and 512 hold against
//!   Neumaier-compensated f64 within 2^-22 K S on cancelling operands.
//! - Partials over 2 and 4 fake context-parallel ranks (whole pages per
//!   rank), merged by the contract's F32 arithmetic, pass the same gate, and
//!   one rank reproduces the normalized output bit for bit.
//! - All-hole rows (and fake ranks owning none of a row's keys) write +0 or
//!   the exact empty partial (-inf, +0, +0) over poisoned outputs, and rows
//!   of a larger prepared bucket past the live count stay poisoned.
//! - Poisoned eager, relaunch, graph and PDL-producer runs of the KV write,
//!   attention with its merge, and both projections write exactly their
//!   maps and agree bit for bit; the producer writes a real input of the
//!   consumer after releasing it.
//! - The KV write matches the oracle bit for bit at widths 128, 384, 640 and
//!   1024, and the split merge folds crafted partials in ascending split
//!   order bit for bit.
//! - Batch invariance: a full 2,051-key row, a 70-key row and the full row's
//!   keys of one context-parallel rank of 2 and of 4 give the same decode
//!   and verify bytes, in both output modes, alone and among other rows at
//!   buckets 8 to 256, as the one-split-per-CTA schedule that buckets of up
//!   to 8 rows ran before the splits were fixed; prefill (prompt rows and
//!   the drafter's widest passes) gives each probe its alone bytes in 30, 36
//!   and 2048-row launches.
//!
//! The checks live in `common/latent_checks.rs`, so a bundle's test can run
//! the served ones on the cubin it embeds.
//!
//! - The drafter's attention before and after PERF-B14b (one split per row,
//!   the decode entry's 64 fixed splits with their merge) against the f64
//!   oracle on every output of representative and adversarial cases: max
//!   and mean absolute and relative error and the BF16 outputs other than
//!   the correctly rounded oracle value; the fixed splits must be no less
//!   accurate (`latent_drafter_attention_accuracy`, Accuracy decision v2):
//!   every output of both arms within the oracle bound and finite, the
//!   fixed splits' max errors within one BF16 rounding step of one split's,
//!   and no more misrounded outputs. Its verdict refuses a gross error, a
//!   NaN, an infinity and a short output on CPU
//!   (`latent_drafter_accuracy_verdict_controls`).
//! - With `CHOPINFER_KERNEL_BENCH=1`, the drafter's attention at its row
//!   counts: the one-split prefill entry against the decode entry's fixed
//!   splits (`latent_drafter_attention_timing`), both gated on the oracle.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test latent_attention`; without a GPU every test skips.

#[path = "common/latent_checks.rs"]
mod latent_checks;

use kernel_lab::gpu_test;
use latent_checks::{
    Kernels, Report, accuracy_verdict_controls, attention_accumulation, attention_groups_timing,
    batch_invariance, drafter_attention_accuracy, drafter_attention_timing, kv_write_widths,
    oracle_cases, poisoned_replay, projection_accumulation, projections, split_merge_order,
};
use std::collections::BTreeMap;

#[test]
fn latent_attention_matches_the_oracle() {
    gpu_test(8 << 30, |session| {
        oracle_cases(session, &Kernels::lab(session)?)
    })
    .expect("latent attention oracle cases");
}

#[test]
fn latent_attention_accumulates_in_128_k_chains() {
    gpu_test(4 << 30, |session| {
        let kernels = &Kernels::lab(session)?;
        let mut report = Report::new(kernels, "accumulation");
        attention_accumulation(session, kernels, &mut report)?;
        projection_accumulation(session, kernels, &mut report)?;
        report.finish(session, BTreeMap::new())
    })
    .expect("latent attention accumulation");
}

#[test]
fn head_batched_projections_match_the_oracle() {
    gpu_test(4 << 30, |session| {
        projections(session, &Kernels::lab(session)?)
    })
    .expect("latent projections");
}

#[test]
fn latent_attention_replays_on_poisoned_buffers() {
    gpu_test(4 << 30, |session| {
        poisoned_replay(session, &Kernels::lab(session)?)
    })
    .expect("latent attention poisoned replay");
}

#[test]
fn latent_kv_write_matches_the_oracle_at_every_width() {
    gpu_test(1 << 30, |session| {
        kv_write_widths(session, &Kernels::lab(session)?)
    })
    .expect("latent KV write widths");
}

#[test]
fn latent_split_merge_folds_in_split_order() {
    gpu_test(1 << 30, |session| {
        split_merge_order(session, &Kernels::lab(session)?)
    })
    .expect("latent split merge order");
}

#[test]
fn latent_rows_do_not_depend_on_their_bucket() {
    gpu_test(4 << 30, |session| {
        batch_invariance(session, &Kernels::lab(session)?)
    })
    .expect("latent attention batch invariance");
}

/// With `CHOPINFER_KERNEL_BENCH=1`: the drafter's attention in one split
/// per row against the decode entry's fixed splits
/// (`drafter_attention_timing`).
#[test]
fn latent_drafter_attention_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("latent drafter attention timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 << 30, |session| {
        drafter_attention_timing(session, &Kernels::lab(session)?)
    })
    .expect("latent drafter attention timing");
}

/// With `CHOPINFER_KERNEL_BENCH=1`: the verify entry's attention at each
/// CTA grouping of its fixed splits, and its split merge against the column
/// merge (`attention_groups_timing`).
#[test]
fn latent_attention_groups_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("latent attention groups timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 << 30, |session| {
        attention_groups_timing(session, &Kernels::lab(session)?)
    })
    .expect("latent attention groups timing");
}

/// The drafter's attention in one split against the fixed splits, on every
/// output against the f64 oracle (`drafter_attention_accuracy`).
#[test]
fn latent_drafter_attention_accuracy() {
    gpu_test(8 << 30, |session| {
        drafter_attention_accuracy(session, &Kernels::lab(session)?)
    })
    .expect("latent drafter attention accuracy");
}

/// The drafter accuracy verdict's controls on a synthetic oracle (CPU):
/// gross errors, NaN, infinity and short outputs fail, one misrounded
/// output passes (`accuracy_verdict_controls`).
#[test]
fn latent_drafter_accuracy_verdict_controls() {
    accuracy_verdict_controls().expect("drafter accuracy verdict controls");
}
