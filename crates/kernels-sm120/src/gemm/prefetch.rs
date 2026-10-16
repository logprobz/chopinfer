//! L2 prefetch of weight ranges ahead of their GEMMs
//! (`cuda/gemm/weight_prefetch.cuh`). A launch in a PDL chain releases its
//! dependents, requests every range from L2 while the kernel it depends on
//! runs, then waits for that kernel, so the chain's order and every output
//! are unchanged; only where the following kernels find their weights
//! moves. It writes no memory.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};

/// The entries' header, relative to the CUDA source root.
pub const HEADER: &str = "gemm/weight_prefetch.cuh";
/// The most ranges one launch requests (`kPrefetchRanges`).
pub const MAX_RANGES: usize = 8;
/// The bulk form's alignment of range starts, lengths and pieces.
pub const BULK_ALIGN: u64 = 16;

/// How a thread requests a piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One `cp.async.bulk.prefetch.L2` per piece.
    Bulk,
    /// One `prefetch.global.L2` per 128-byte line of a piece.
    Lines,
}

/// One prefetch entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefetch {
    pub symbol: String,
    pub mode: Mode,
}

impl Prefetch {
    pub fn params() -> Vec<EntryParam> {
        let mut params: Vec<EntryParam> = (0..MAX_RANGES)
            .map(|r| EntryParam::DevicePointer {
                name: format!("base{r}"),
                cxx_type: "const unsigned char".into(),
            })
            .collect();
        params.extend((0..MAX_RANGES).map(|r| EntryParam::U64 {
            name: format!("bytes{r}"),
        }));
        params.push(EntryParam::U32 {
            name: "count".into(),
        });
        params.push(EntryParam::U32 {
            name: "chunk".into(),
        });
        params
    }

    /// The entry, run by up to `threads` threads per CTA.
    pub fn entry(&self, threads: u32) -> EntrySpec {
        let list = |field: &str| {
            (0..MAX_RANGES)
                .map(|r| format!("{field}{r}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let body = format!(
            "const chop::gemm::PrefetchRanges ranges{{{{{}}},{{{}}}}};\n\
             chop::gemm::prefetch_ranges_l2<{}>(ranges, count, chunk);",
            list("base"),
            list("bytes"),
            self.mode == Mode::Lines
        );
        EntrySpec::new(self.symbol.clone(), Self::params(), body, threads, 1)
    }

    /// `grid` CTAs of `threads` threads, PDL, no shared memory.
    pub fn plan(&self, grid: u32, threads: u32) -> Result<LaunchPlan, Refusal> {
        if grid == 0 || threads == 0 || !threads.is_multiple_of(32) {
            return Err(Refusal("a prefetch launch takes whole warps"));
        }
        let plan = LaunchPlan {
            grid: [grid, 1, 1],
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The arguments for `ranges` (each a bound buffer and its bytes, in
    /// the order the following kernels read them) in `chunk`-byte pieces.
    pub fn arguments(&self, ranges: &[(u64, u64)], chunk: u32) -> Result<ArgumentList, Refusal> {
        if ranges.is_empty() || ranges.len() > MAX_RANGES {
            return Err(Refusal("a prefetch launch requests 1 to 8 ranges"));
        }
        if chunk == 0
            || (self.mode == Mode::Lines && !chunk.is_multiple_of(128))
            || (self.mode == Mode::Bulk && !u64::from(chunk).is_multiple_of(BULK_ALIGN))
        {
            return Err(Refusal(
                "prefetch pieces are whole lines, or 16-byte multiples in bulk",
            ));
        }
        if self.mode == Mode::Bulk
            && ranges
                .iter()
                .any(|&(_, bytes)| !bytes.is_multiple_of(BULK_ALIGN))
        {
            return Err(Refusal("bulk prefetch ranges are 16-byte multiples"));
        }
        let mut args: Vec<Argument> = (0..MAX_RANGES)
            .map(|r| match ranges.get(r) {
                Some(&(buffer, _)) => Argument::DeviceBuffer(buffer),
                None => Argument::NullDevicePointer,
            })
            .collect();
        args.extend((0..MAX_RANGES).map(|r| Argument::U64(ranges.get(r).map_or(0, |&(_, b)| b))));
        args.push(Argument::U32(ranges.len() as u32));
        args.push(Argument::U32(chunk));
        let args = ArgumentList(args);
        args.validate()?;
        Ok(args)
    }
}

/// A cubin named `name` holding `entries`, each run by up to `threads`
/// threads per CTA.
pub fn cubin_spec(name: &str, entries: &[Prefetch], threads: u32) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().map(|e| e.entry(threads)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_cover_every_range_slot() {
        let bulk = Prefetch {
            symbol: "chop_prefetch_test_v1".into(),
            mode: Mode::Bulk,
        };
        let args = bulk.arguments(&[(1, 1 << 20), (2, 4096)], 65536).unwrap();
        assert_eq!(args.0.len(), Prefetch::params().len());
        assert_eq!(args.0[2], Argument::NullDevicePointer);
        assert!(bulk.arguments(&[(1, 24)], 65536).is_err());
        assert!(bulk.arguments(&[], 65536).is_err());
        let lines = Prefetch {
            mode: Mode::Lines,
            ..bulk
        };
        assert!(lines.arguments(&[(1, 4096)], 100).is_err());
        assert!(lines.entry(256).body.contains("prefetch_ranges_l2<true>"));
    }
}
