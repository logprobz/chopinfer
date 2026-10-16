//! Lab-only decoded-tile chained projections
//! (`chained::Schedule::DecodedTiles`): the dense MLP's and the shared
//! expert's FP8 prefill gate/up with each weight code decoded once per CTA,
//! and the deliberate defects of `chop::moe::decoded_tiles::Mutation`.
//! Shared by the `moe_chained_decoded` lock family, `tests/moe_chained.rs`
//! (bitwise sweep, producer and mutant checks) and
//! `tests/projection_latency.rs` (prefill timings).

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Weights};

/// The candidates' cubin prefix: `{PREFIX}_fp8block_dt`.
pub const PREFIX: &str = "chop_moe_chain_lab";

/// The mutants' cubin.
pub const MUTANTS_CUBIN: &str = "chop_moe_chain_lab_dt_mutants";

/// `chop::moe::decoded_tiles::Mutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "long_chain"];

/// The served dense FP8 prefill (64-row) column gate/up projections: the
/// dense MLP's, then the shared expert's.
pub fn references() -> [Projection; 2] {
    chained::TP4_288_TOP8.dense_intermediates.map(|intermediate| {
        chained::catalog()
            .into_iter()
            .find(|p| {
                p.row_tile == 64
                    && p.weights() == Weights::Fp8Block128
                    && p.query.gemm.grouped.is_none()
                    && p.gate_up()
                    && p.columns() == intermediate
            })
            .expect("the dense FP8 prefill gate/up projections are cataloged")
    })
}

/// [`references`] in decoded tiles.
pub fn projections() -> [Projection; 2] {
    references().map(|reference| {
        reference
            .decoded_tiles()
            .expect("decoded tiles take the dense prefill gate/up")
    })
}

/// The mutated projection: the shared expert's gate/up in decoded tiles.
pub fn mutant_projection() -> Projection {
    let [_, shared] = projections();
    shared
}

/// The symbol of the mutant projection's instance with `mutation`.
pub fn mutant_symbol(mutation: &str) -> String {
    format!("chop_lab_dt_{mutation}_gateup_v1")
}

/// The candidates' cubin.
pub fn cubins() -> Vec<CubinSpec> {
    chained::projection_cubin_specs(PREFIX, &projections())
}

/// The mutants' cubin: the mutant projection's entry once per mutation,
/// each body given its mutation as the kernel's last template argument.
pub fn mutants_cubin() -> CubinSpec {
    let mut cubin = chained::projection_cubin_specs(MUTANTS_CUBIN, &[mutant_projection()])
        .pop()
        .expect("one decoded-tile cubin");
    cubin.name = MUTANTS_CUBIN.into();
    let base = std::mem::take(&mut cubin.entries);
    let [entry] = base.as_slice() else {
        panic!("one mutant projection entry");
    };
    for mutation in MUTATIONS {
        let mut entry = entry.clone();
        assert_eq!(entry.body.matches(">(input").count(), 1, "{}", entry.symbol);
        entry.symbol = mutant_symbol(mutation);
        entry.body = entry.body.replace(
            ">(input",
            &format!(",chop::moe::decoded_tiles::Mutation::{mutation}>(input"),
        );
        cubin.entries.push(entry);
    }
    cubin
}
