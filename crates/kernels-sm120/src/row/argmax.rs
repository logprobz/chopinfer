//! The shared sharded argmax on SM120 (`cuda/row/argmax.cuh`): a rank's
//! partial over its vocabulary shard of F32 or BF16 logits
//! (`RowQuery::ShardedArgmaxPartial`) and the final over every rank's pairs
//! (`RowQuery::ShardedArgmaxFinal`), in the contract's order (NaN below
//! every value, -0 as +0, ties to the lower id). The final reads row-major
//! or rank-major pairs through two strides and writes each row's id to a
//! dense output and up to [`MAX_SCATTER`] scattered outputs. Neither names a shape: the vocabulary, its offset, the ranks
//! and the strides are launch arguments, so a build compiles each pattern
//! once, under the symbols of [`entries`] or its own.
//!
//! Each launch grids one CTA per row of its launch. A row's pair or id is a
//! function of that row's inputs alone and the reduction picks the one
//! largest key of a total order, so every partial of up to the backend's
//! capacity keeps every row, chunk and batch claim ([`INVARIANCE`]). A
//! final keeps the row and chunk claims under its scatter precondition
//! (distinct destinations, `rows::ArgmaxScatter`) but claims no batch
//! independence ([`FINAL_INVARIANCE`]): rows whose inputs are copies of one
//! row share its destinations, which breaks the precondition.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal, Validate};
use kernel_core::properties::{Batch, Invariance};

use super::{ptr, u32_arg};

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "row/argmax.cuh";
/// Threads of a partial's CTA.
pub const PARTIAL_THREADS: u32 = 256;
/// Threads of a final's CTA, of which one reads the row's pairs.
pub const FINAL_THREADS: u32 = 32;
/// The most scattered outputs a final writes.
pub const MAX_SCATTER: usize = 2;
/// The shared partial over F32 logits.
pub const PARTIAL_F32_SYMBOL: &str = "chop_row_argmax_partial_f32_v1";
/// The shared partial over BF16 logits.
pub const PARTIAL_BF16_SYMBOL: &str = "chop_row_argmax_partial_bf16_v1";
/// The shared final.
pub const FINAL_SYMBOL: &str = "chop_row_argmax_final_v1";
/// The lab's cubin of the shared entries.
pub const LAB_CUBIN: &str = "row_argmax_sm120";
/// What every launch of a shared partial keeps: each row's pair is a
/// function of its own inputs, at any row count up to the most rows an
/// SM120 op serves, in any batch or chunk split.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);
/// What every launch of the shared final keeps whose scattered rows have
/// distinct destinations: [`INVARIANCE`] without batch independence.
pub const FINAL_INVARIANCE: Invariance = Invariance {
    batch: Batch::None,
    ..INVARIANCE
};

/// The step an entry performs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Pattern {
    PartialF32,
    PartialBf16,
    Final,
}

impl Pattern {
    /// The partial over `logits`, if SM120 has one.
    pub fn partial(logits: DType) -> Option<Pattern> {
        match logits {
            DType::F32 => Some(Pattern::PartialF32),
            DType::Bf16 => Some(Pattern::PartialBf16),
            _ => None,
        }
    }
}

/// An argmax entry a build compiles: its pattern, under `symbol`, with
/// `threads` per CTA.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub pattern: Pattern,
    pub symbol: String,
    pub threads: u32,
}

impl Entry {
    pub fn new(pattern: Pattern, symbol: impl Into<String>) -> Self {
        let threads = match pattern {
            Pattern::PartialF32 | Pattern::PartialBf16 => PARTIAL_THREADS,
            Pattern::Final => FINAL_THREADS,
        };
        Self {
            pattern,
            symbol: symbol.into(),
            threads,
        }
    }

    /// A partial with `threads` per CTA, a multiple of 32 up to 1024. The
    /// key order is total, so every width picks the same pair; a wider CTA
    /// only keeps more loads of a row in flight.
    pub fn with_threads(mut self, threads: u32) -> Result<Self, Refusal> {
        if self.pattern == Pattern::Final {
            return Err(Refusal("an SM120 argmax final has a fixed CTA"));
        }
        if threads == 0 || threads % 32 != 0 || threads > 1024 {
            return Err(Refusal(
                "an SM120 argmax partial uses 32 to 1024 threads in whole warps",
            ));
        }
        self.threads = threads;
        Ok(self)
    }

    /// The plan of a launch of `rows` rows: one CTA of the entry's threads
    /// per row.
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        let mut plan = plan(self.pattern, rows)?;
        plan.block = [self.threads, 1, 1];
        plan.validate()?;
        Ok(plan)
    }

    /// The entry's code under its symbol.
    pub fn spec(&self) -> EntrySpec {
        let symbol = self.symbol.clone();
        match self.pattern {
            Pattern::PartialF32 | Pattern::PartialBf16 => {
                let logit = if self.pattern == Pattern::PartialF32 {
                    "float"
                } else {
                    "uint16_t"
                };
                EntrySpec::new(
                    symbol,
                    partial_params(&format!("const {logit}")),
                    "chop::row::argmax::partial(logits,scores,ids,live_rows,live_width,vocab,vocab_offset);"
                        .into(),
                    self.threads,
                    1,
                )
            }
            Pattern::Final => EntrySpec::new(
                symbol,
                final_params(),
                "chop::row::argmax::final_pairs(scores,ids,output,rows_a,out_a,rows_b,out_b,live_rows,ranks,row_stride,rank_stride,len_a,len_b);"
                    .into(),
                FINAL_THREADS,
                1,
            ),
        }
    }

    /// What a launch of the entry claims: [`INVARIANCE`] for a shared
    /// partial and [`FINAL_INVARIANCE`] for the shared final, nothing under
    /// a build's own symbol, which declares it itself.
    pub fn guarantees(&self) -> Invariance {
        match self.pattern {
            _ if !entries().contains(self) => Invariance::NONE,
            Pattern::Final => FINAL_INVARIANCE,
            Pattern::PartialF32 | Pattern::PartialBf16 => INVARIANCE,
        }
    }
}

fn partial_params(logits: &str) -> Vec<EntryParam> {
    vec![
        ptr("logits", logits),
        ptr("scores", "float"),
        ptr("ids", "int32_t"),
        ptr("live_rows", "const uint32_t"),
        ptr("live_width", "const uint32_t"),
        u32_arg("vocab"),
        u32_arg("vocab_offset"),
    ]
}

fn final_params() -> Vec<EntryParam> {
    vec![
        ptr("scores", "const float"),
        ptr("ids", "const int32_t"),
        ptr("output", "int32_t"),
        ptr("rows_a", "const int32_t"),
        ptr("out_a", "int32_t"),
        ptr("rows_b", "const int32_t"),
        ptr("out_b", "int32_t"),
        ptr("live_rows", "const uint32_t"),
        u32_arg("ranks"),
        u32_arg("row_stride"),
        u32_arg("rank_stride"),
        u32_arg("len_a"),
        u32_arg("len_b"),
    ]
}

/// The shared entries under their kernel-base symbols.
pub fn entries() -> Vec<Entry> {
    vec![
        Entry::new(Pattern::PartialF32, PARTIAL_F32_SYMBOL),
        Entry::new(Pattern::PartialBf16, PARTIAL_BF16_SYMBOL),
        Entry::new(Pattern::Final, FINAL_SYMBOL),
    ]
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

/// The lab's cubin of the shared entries ([`LAB_CUBIN`]).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    vec![cubin_spec(LAB_CUBIN, &entries())]
}

/// The plan of a launch of `rows` rows of an entry of `pattern`: one CTA
/// per row.
pub fn plan(pattern: Pattern, rows: u32) -> Result<LaunchPlan, Refusal> {
    if rows == 0 || rows > crate::backend::MAX_CAPACITY {
        return Err(Refusal(
            "an SM120 sharded argmax launches 1 to MAX_CAPACITY rows",
        ));
    }
    let threads = match pattern {
        Pattern::PartialF32 | Pattern::PartialBf16 => PARTIAL_THREADS,
        Pattern::Final => FINAL_THREADS,
    };
    let plan = LaunchPlan {
        grid: [rows, 1, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// A partial's arguments, in the ABI's order: the logits, the pairs'
/// scores and ids, the device live rows and width, the shard's columns and
/// its first global id.
pub fn partial_arguments(
    [logits, scores, ids, live_rows, live_width]: [Argument; 5],
    vocab: u32,
    vocab_offset: u32,
) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(vec![
        logits,
        scores,
        ids,
        live_rows,
        live_width,
        Argument::U32(vocab),
        Argument::U32(vocab_offset),
    ]);
    args.validate()?;
    Ok(args)
}

/// Where a final reads and writes, as pointers (`NullDevicePointer` for
/// an output it does not write) and the counts its ABI takes.
pub struct FinalArguments {
    pub scores: Argument,
    pub ids: Argument,
    pub output: Argument,
    /// Each scattered output's row indexes, output and elements, at most
    /// 2^31 (the entry reads an index unsigned, so a negative one must
    /// fall past the elements); the absent ones null with zero elements.
    pub scatter: [(Argument, Argument, u32); MAX_SCATTER],
    pub live_rows: Argument,
    pub ranks: u32,
    pub row_stride: u32,
    pub rank_stride: u32,
}

/// A final's arguments, in the ABI's order.
pub fn final_arguments(arguments: FinalArguments) -> Result<ArgumentList, Refusal> {
    let FinalArguments {
        scores,
        ids,
        output,
        scatter: [(rows_a, out_a, len_a), (rows_b, out_b, len_b)],
        live_rows,
        ranks,
        row_stride,
        rank_stride,
    } = arguments;
    let args = ArgumentList(vec![
        scores,
        ids,
        output,
        rows_a,
        out_a,
        rows_b,
        out_b,
        live_rows,
        Argument::U32(ranks),
        Argument::U32(row_stride),
        Argument::U32(rank_stride),
        Argument::U32(len_a),
        Argument::U32(len_b),
    ]);
    args.validate()?;
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lab_cubin_holds_every_pattern_under_its_symbol() {
        let [spec] = &lab_cubin_specs()[..] else {
            panic!("one lab cubin");
        };
        assert_eq!(spec.name, LAB_CUBIN);
        let symbols: Vec<_> = spec.entries.iter().map(|e| e.symbol.as_str()).collect();
        assert_eq!(
            symbols,
            [PARTIAL_F32_SYMBOL, PARTIAL_BF16_SYMBOL, FINAL_SYMBOL]
        );
        assert!(spec.sources_root.join(HEADER).is_file());
        assert_eq!(spec.entries[0].params[0], ptr("logits", "const float"));
        assert_eq!(spec.entries[1].params[0], ptr("logits", "const uint16_t"));
        assert_eq!(spec.entries[0].body, spec.entries[1].body);
        assert_eq!(spec.entries[2].max_threads, FINAL_THREADS);
        let renamed = Entry::new(Pattern::Final, "chop_row_argmax_other_v1");
        assert_eq!(renamed.spec().body, spec.entries[2].body);
        assert_eq!(renamed.guarantees(), Invariance::NONE);
        assert_eq!(
            Entry::new(Pattern::Final, FINAL_SYMBOL).guarantees(),
            FINAL_INVARIANCE
        );
        assert_eq!(
            Entry::new(Pattern::PartialF32, PARTIAL_F32_SYMBOL).guarantees(),
            INVARIANCE
        );
        assert_eq!(FINAL_INVARIANCE.batch, Batch::None);
        assert!(FINAL_INVARIANCE.one_row_at(crate::backend::MAX_CAPACITY));
    }

    #[test]
    fn launches_plan_one_cta_per_row() {
        for (pattern, threads) in [
            (Pattern::PartialF32, PARTIAL_THREADS),
            (Pattern::PartialBf16, PARTIAL_THREADS),
            (Pattern::Final, FINAL_THREADS),
        ] {
            let plan = plan(pattern, 5).unwrap();
            assert_eq!(
                (plan.grid, plan.block, plan.pdl),
                ([5, 1, 1], [threads, 1, 1], true)
            );
            assert!(super::plan(pattern, 0).is_err());
            assert!(super::plan(pattern, crate::backend::MAX_CAPACITY + 1).is_err());
        }
        assert_eq!(Pattern::partial(DType::F32), Some(Pattern::PartialF32));
        assert_eq!(Pattern::partial(DType::Bf16), Some(Pattern::PartialBf16));
        assert_eq!(Pattern::partial(DType::F16), None);
        assert_eq!(
            partial_arguments([1, 2, 3, 4, 5].map(Argument::DeviceBuffer), 64, 128)
                .unwrap()
                .0[5..],
            [Argument::U32(64), Argument::U32(128)]
        );
    }

    #[test]
    fn a_wider_partial_keeps_the_body_and_plans_its_threads() {
        let default = Entry::new(Pattern::PartialF32, "chop_row_argmax_narrow_v1");
        let wide = default.clone().with_threads(1024).unwrap();
        assert_eq!(default.threads, PARTIAL_THREADS);
        assert_eq!(default.spec().max_threads, PARTIAL_THREADS);
        assert_eq!(wide.spec().max_threads, 1024);
        assert_eq!(wide.spec().body, default.spec().body);
        assert_eq!(wide.spec().params, default.spec().params);
        assert_eq!(wide.plan(3).unwrap().block, [1024, 1, 1]);
        assert_eq!(wide.plan(3).unwrap().grid, [3, 1, 1]);
        assert_eq!(
            default.plan(3).unwrap(),
            plan(Pattern::PartialF32, 3).unwrap()
        );
        assert!(wide.plan(0).is_err());
        for threads in [0, 48, 1056] {
            assert!(default.clone().with_threads(threads).is_err());
        }
        assert!(
            Entry::new(Pattern::Final, FINAL_SYMBOL)
                .with_threads(64)
                .is_err()
        );
        assert!(!entries().contains(&wide));
        assert!(
            entries()
                .iter()
                .all(|e| e.plan(1).unwrap() == plan(e.pattern, 1).unwrap())
        );
    }
}
