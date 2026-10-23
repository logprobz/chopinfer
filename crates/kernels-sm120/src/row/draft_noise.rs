//! The target sampler's Gumbel noise on a rank's shard of F32 drafter
//! logits on SM120 (`kernel_core::draft_noise::DraftNoise`,
//! `cuda/row/draft_noise.cuh`), in place. A launch grids one CTA per row
//! and [`TILE`]-column tile; the vocabulary, the shard's first global id
//! and the draft position are launch arguments, so a build compiles the
//! entry once under the symbol of [`Entry`].
//!
//! Each column's value is a function of its own logit and its row's noise
//! record, so a launch keeps every row, chunk and batch claim
//! ([`INVARIANCE`]) up to the backend's capacity.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;

use super::{ptr, u32_arg};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "row/draft_noise.cuh";
/// Threads of a CTA.
pub const THREADS: u32 = 256;
/// Columns one CTA perturbs: eight per thread.
pub const TILE: u32 = 2048;
/// The shared entry.
pub const SYMBOL: &str = "chop_row_draft_noise_f32_v1";
/// The lab's cubin of the shared entry.
pub const LAB_CUBIN: &str = "row_draft_noise_sm120";
/// What every launch of the shared entry keeps.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);

/// A draft noise entry a build compiles under `symbol`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub symbol: String,
}

impl Entry {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
        }
    }

    /// The plan of a launch of `rows` rows of `vocab` columns: one CTA per
    /// row and tile.
    pub fn plan(&self, rows: u32, vocab: u32) -> Result<LaunchPlan, Refusal> {
        plan(rows, vocab)
    }

    /// The entry's code under its symbol.
    pub fn spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("logits", "float"),
                ptr("noise", "const chop::row::draft_noise::Noise"),
                ptr("live_rows", "const uint32_t"),
                ptr("live_width", "const uint32_t"),
                u32_arg("vocab"),
                u32_arg("vocab_offset"),
                u32_arg("position"),
            ],
            format!(
                "chop::row::draft_noise::perturb<{TILE}>(logits,noise,live_rows,live_width,vocab,vocab_offset,position);"
            ),
            THREADS,
            1,
        )
    }

    /// [`INVARIANCE`] under the shared symbol; nothing under a build's own,
    /// which declares it itself.
    pub fn guarantees(&self) -> Invariance {
        if self.symbol == SYMBOL {
            INVARIANCE
        } else {
            Invariance::NONE
        }
    }
}

/// A cubin `name` holding `entries`.
pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// The lab's cubin of the shared entry ([`LAB_CUBIN`]).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    vec![cubin_spec(LAB_CUBIN, &[Entry::new(SYMBOL)])]
}

/// The plan of a launch of `rows` rows of `vocab` columns.
pub fn plan(rows: u32, vocab: u32) -> Result<LaunchPlan, Refusal> {
    if rows == 0 || rows > crate::backend::MAX_CAPACITY || vocab == 0 {
        return Err(Refusal(
            "an SM120 draft noise launches 1 to MAX_CAPACITY rows of a vocabulary shard",
        ));
    }
    let plan = LaunchPlan {
        grid: [rows, vocab.div_ceil(TILE), 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// The arguments, in the ABI's order: the logits, the noise records, the
/// device live rows and width, the shard's columns, its first global id
/// and the draft position.
pub fn arguments(
    [logits, noise, live_rows, live_width]: [Argument; 4],
    vocab: u32,
    vocab_offset: u32,
    position: u32,
) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(vec![
        logits,
        noise,
        live_rows,
        live_width,
        Argument::U32(vocab),
        Argument::U32(vocab_offset),
        Argument::U32(position),
    ]);
    args.validate()?;
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lab_cubin_holds_the_shared_entry() {
        let [spec] = &lab_cubin_specs()[..] else {
            panic!("one lab cubin");
        };
        assert_eq!(spec.name, LAB_CUBIN);
        assert_eq!(spec.entries.len(), 1);
        assert_eq!(spec.entries[0].symbol, SYMBOL);
        assert_eq!(spec.entries[0].max_threads, THREADS);
        assert!(spec.sources_root.join(HEADER).is_file());
        assert!(spec.sources_root.join("sampling/sampling.cuh").is_file());
        assert_eq!(Entry::new(SYMBOL).guarantees(), INVARIANCE);
        let renamed = Entry::new("chop_row_draft_noise_other_v1");
        assert_eq!(renamed.guarantees(), Invariance::NONE);
        assert_eq!(renamed.spec().body, spec.entries[0].body);
    }

    #[test]
    fn launches_plan_one_cta_per_row_and_tile() {
        let plan = plan(5, 32_320).unwrap();
        assert_eq!(
            (plan.grid, plan.block, plan.pdl),
            ([5, 16, 1], [THREADS, 1, 1], true)
        );
        assert_eq!(super::plan(1, TILE).unwrap().grid, [1, 1, 1]);
        assert_eq!(super::plan(1, TILE + 1).unwrap().grid, [1, 2, 1]);
        assert!(super::plan(0, 64).is_err());
        assert!(super::plan(1, 0).is_err());
        assert!(super::plan(crate::backend::MAX_CAPACITY + 1, 64).is_err());
        let args = arguments([1, 2, 3, 4].map(Argument::DeviceBuffer), 64, 128, 3).unwrap();
        assert_eq!(
            args.0[4..],
            [Argument::U32(64), Argument::U32(128), Argument::U32(3)]
        );
    }
}
