//! The partial rounding rows on SM120 (`cuda/row/rounding.cuh`): an F32 row
//! rounded once to BF16 (round to nearest even) and a BF16 row widened to
//! F32 (exact), over rows of a compiled width. Each launch grids one CTA of
//! [`THREADS`] per row of the row bucket that holds its rows (rows at or
//! past the device live count return), with the arguments `input`,
//! `output` and `live_rows`.
//!
//! The BF16-partials all-reduce composes them with the comm engine's F32
//! pair, and its comm epilogue widens its rounded sums, through the pair a
//! model registers ([`PartialRounding`], `Sm120::with_partial_rounding`):
//! its own compiled entries of this ABI or the shared ones ([`shared`]),
//! with what its entries declare.
//!
//! The shared entries declare [`shared_invariance`]: each CTA rounds or
//! widens one row's elements alone, one conversion each, and rows at or
//! past the live count return, so a row has its bits at every bucket and
//! beside any rows.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;

use super::ptr;

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "row/rounding.cuh";
/// Threads of every entry's CTA.
pub const THREADS: u32 = 256;
/// The lab's cubin of the shared entries at [`LAB_WIDTHS`].
pub const LAB_CUBIN: &str = "row_rounding_sm120";
/// The row widths the lab's cubin compiles.
pub const LAB_WIDTHS: [u32; 2] = [2048, 4096];

/// What a rounding entry computes per element.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Rounding {
    /// F32 input, BF16 output, rounded once to nearest even.
    Round,
    /// BF16 input, F32 output, exact.
    Widen,
}

impl Rounding {
    /// The entry's parameters: its input, output and live count.
    pub fn params(self) -> Vec<EntryParam> {
        let (input, output) = match self {
            Self::Round => ("const float", "uint16_t"),
            Self::Widen => ("const uint16_t", "float"),
        };
        vec![
            ptr("input", input),
            ptr("output", output),
            ptr("live_rows", "const uint32_t"),
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Round => "round",
            Self::Widen => "widen",
        }
    }
}

/// The shared entry `rounding` over rows of `width` elements.
pub fn shared_symbol(rounding: Rounding, width: u32) -> String {
    format!("chop_row_rounding_{}_w{width}_v1", rounding.name())
}

/// The shared entry `rounding` over rows of `width` elements, under
/// `symbol`.
pub fn entry(rounding: Rounding, width: u32, symbol: impl Into<String>) -> EntrySpec {
    let function = match rounding {
        Rounding::Round => "round_rows",
        Rounding::Widen => "widen_rows",
    };
    EntrySpec::new(
        symbol.into(),
        rounding.params(),
        format!("chop::row::rounding::{function}<{width}>(input,output,live_rows);"),
        THREADS,
        1,
    )
}

/// A model's rounding pair over rows of `width` elements: entries its build
/// compiles that round and widen as the shared ones do, with their ABI,
/// and what the two entries declare together (the meet of their
/// declarations; none for entries that declare nothing).
#[derive(Clone, Debug)]
pub struct PartialRounding {
    pub width: u32,
    pub round: EntrySpec,
    pub widen: EntrySpec,
    pub invariance: Invariance,
}

impl PartialRounding {
    /// Refuses entries whose parameters or threads are not the ABI's, or a
    /// width the entries cannot take.
    pub fn check(&self) -> Result<(), Refusal> {
        if self.width == 0 || !self.width.is_multiple_of(4) {
            return Err(Refusal(
                "a partial rounding row takes widths of whole four-element vectors",
            ));
        }
        for (entry, rounding) in [
            (&self.round, Rounding::Round),
            (&self.widen, Rounding::Widen),
        ] {
            if entry.params != rounding.params() || entry.max_threads != THREADS {
                return Err(Refusal(
                    "a partial rounding entry takes input, output and live_rows in CTAs of 256 threads",
                ));
            }
        }
        Ok(())
    }

    /// The symbol of the entry `rounding`.
    pub fn symbol(&self, rounding: Rounding) -> &str {
        match rounding {
            Rounding::Round => &self.round.symbol,
            Rounding::Widen => &self.widen.symbol,
        }
    }
}

/// What the shared entries declare (the module docs): one-row bits up to
/// the largest row bucket, any chunk split and row independence.
pub fn shared_invariance() -> Invariance {
    let largest = super::tuning().buckets.iter().copied().max().unwrap_or(0);
    Invariance::any_rows(largest)
}

/// The shared pair at `width` ([`shared_symbol`]).
pub fn shared(width: u32) -> PartialRounding {
    PartialRounding {
        width,
        invariance: shared_invariance(),
        round: entry(
            Rounding::Round,
            width,
            shared_symbol(Rounding::Round, width),
        ),
        widen: entry(
            Rounding::Widen,
            width,
            shared_symbol(Rounding::Widen, width),
        ),
    }
}

/// The plan of a launch of `rows` rows: one CTA per row of the smallest row
/// bucket that holds them.
pub fn plan(rows: u32) -> Result<LaunchPlan, Refusal> {
    let bucket = super::smallest_bucket(rows)
        .filter(|_| rows > 0)
        .ok_or(Refusal("no row bucket holds the partial rounding's rows"))?;
    let plan = LaunchPlan {
        grid: [bucket, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// A launch's arguments, in the ABI's order.
pub fn arguments(input: u64, output: u64, live_rows: u64) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(input),
        Argument::DeviceBuffer(output),
        Argument::DeviceBuffer(live_rows),
    ]);
    args.validate()?;
    Ok(args)
}

/// A cubin `name` holding the pairs `pairs`.
pub fn cubin_spec(name: impl Into<String>, pairs: &[PartialRounding]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: pairs
            .iter()
            .flat_map(|pair| [pair.round.clone(), pair.widen.clone()])
            .collect(),
    }
}

/// The lab's cubin of the shared pairs at [`LAB_WIDTHS`].
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let pairs: Vec<_> = LAB_WIDTHS.into_iter().map(shared).collect();
    vec![cubin_spec(LAB_CUBIN, &pairs)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_pairs_take_the_abi_and_plan_the_row_bucket() {
        let pair = shared(4096);
        assert_eq!(pair.check(), Ok(()));
        assert_eq!(
            pair.symbol(Rounding::Round),
            "chop_row_rounding_round_w4096_v1"
        );
        assert!(pair.widen.body.contains("widen_rows<4096>"));
        for rows in [1, 2, 3, 5, 33, 1000, 8192] {
            let plan = plan(rows).unwrap();
            assert_eq!(plan.grid[0], super::super::smallest_bucket(rows).unwrap());
            assert_eq!((plan.block, plan.pdl), ([THREADS, 1, 1], true));
        }
        assert!(plan(0).is_err());
        assert!(plan(1 << 20).is_err());
        assert_eq!(
            arguments(1, 2, 3).unwrap().0,
            [1, 2, 3].map(Argument::DeviceBuffer)
        );
        let lab = lab_cubin_specs();
        assert_eq!(lab[0].entries.len(), 2 * LAB_WIDTHS.len());
        assert!(lab[0].sources_root.join(HEADER).is_file());
    }

    #[test]
    fn pairs_off_the_abi_are_refused() {
        let mut swapped = shared(4096);
        std::mem::swap(&mut swapped.round, &mut swapped.widen);
        assert!(swapped.check().is_err());
        let mut wide = shared(4096);
        wide.round.max_threads = 512;
        assert!(wide.check().is_err());
        assert!(shared(4098).check().is_err());
        assert!(shared(0).check().is_err());
    }
}
