//! The gather by index (`kernel_core::row_copy::GatherByIndex`) on SM120
//! (`cuda/row/gather.cuh`): one CTA per row of the smallest row bucket
//! holding the launch's grid rows, moving each gathered row in 16-byte
//! vectors. It moves bytes only, so every launch of up to the backend's
//! capacity keeps every row, chunk and batch claim ([`INVARIANCE`],
//! declared for the shared symbol by [`declared_entries`]). One entry
//! serves every row width; a build compiles it in its own cubin.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::row_copy::GatherByIndex;

use super::{ptr, smallest_bucket, u32_arg};
use crate::invariance::{Declared, OpKind};

/// The gather's header, relative to the CUDA source root.
pub const HEADER: &str = "row/gather.cuh";
/// Threads of the gather entry.
pub const THREADS: u32 = 256;
/// Bytes of one gathered vector: a gathered row holds whole vectors.
pub const VECTOR_BYTES: u64 = 16;
/// The shared gather entry.
pub const SYMBOL: &str = "chop_row_gather_by_index_v1";
/// The cubin of the shared entry, in the lab and in a model's build.
pub const CUBIN: &str = "row_gather_sm120";
/// What every gather launch keeps: rows move without arithmetic, so each
/// row's bytes are its source row's at any row count up to the most rows
/// an SM120 op serves, in any batch or chunk split.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);

/// The entry's code under `symbol`.
pub fn entry_spec(symbol: &str) -> EntrySpec {
    EntrySpec::new(
        symbol.into(),
        vec![
            ptr("source", "const uint4"),
            ptr("row_indices", "const uint32_t"),
            ptr("count", "const uint32_t"),
            ptr("destination", "uint4"),
            u32_arg("vectors"),
        ],
        "chop::row::gather::by_index(source,row_indices,count,destination,vectors);".into(),
        THREADS,
        1,
    )
}

/// The declaration of the shared entry: a row entry every profile shares,
/// claiming [`INVARIANCE`].
pub fn declared_entries() -> Vec<Declared> {
    vec![Declared {
        symbol: SYMBOL.into(),
        profile: None,
        op: OpKind::Row,
        invariance: INVARIANCE,
    }]
}

/// The cubin ([`CUBIN`]) holding the shared entry.
pub fn cubin_spec() -> CubinSpec {
    CubinSpec {
        name: CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: vec![entry_spec(SYMBOL)],
    }
}

/// A gather by index of up to `capacity` rows through entry `symbol`.
#[derive(Clone, Debug)]
pub struct Gather {
    op: GatherByIndex,
    capacity: u32,
    vectors: u32,
    symbol: String,
}

impl Gather {
    /// Refuses views aligned below a vector, a row of a partial vector,
    /// widths past the entry's loop bound, a capacity no row bucket holds
    /// and footprints past u64: SM120's limits, not the contract's.
    pub fn new(op: GatherByIndex, capacity: u32, symbol: &str) -> Result<Self, Refusal> {
        op.validate()?;
        if u64::from(op.alignment_bytes) < VECTOR_BYTES {
            return Err(Refusal(
                "an SM120 gather by index needs views of 16-byte alignment",
            ));
        }
        let row = op.row_bytes()?;
        if !row.is_multiple_of(VECTOR_BYTES) {
            return Err(Refusal(
                "an SM120 gather by index moves rows of whole 16-byte vectors",
            ));
        }
        let vectors = u32::try_from(row / VECTOR_BYTES)
            .map_err(|_| Refusal("an SM120 gather by index's row exceeds its loop bound"))?;
        if capacity == 0 || smallest_bucket(capacity).is_none() {
            return Err(Refusal(
                "an SM120 gather by index's capacity exceeds its row buckets",
            ));
        }
        op.destination_bytes(capacity)?;
        Ok(Self {
            op,
            capacity,
            vectors,
            symbol: symbol.into(),
        })
    }

    pub fn op(&self) -> GatherByIndex {
        self.op
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// The launch's gathered rows and grid: at most `rows` rows gathered,
    /// gridded over `padded` rows when nonzero, else `rows`.
    pub fn rows(&self, rows: u32, padded: u32) -> Result<(u32, u32), Refusal> {
        let grid = if padded == 0 { rows } else { padded };
        if rows > grid || grid == 0 || grid > self.capacity {
            return Err(Refusal(
                "a gather by index grids from one row to its capacity and gathers at most those rows",
            ));
        }
        Ok((rows, grid))
    }

    /// One CTA per row of the smallest row bucket holding `grid` rows.
    pub fn plan(&self, grid: u32) -> Result<LaunchPlan, Refusal> {
        if grid == 0 || grid > self.capacity {
            return Err(Refusal("gather rows exceed the prepared capacity"));
        }
        let bucket = smallest_bucket(grid).ok_or(Refusal("no row bucket holds the gather"))?;
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

    pub fn arguments(
        &self,
        source: u64,
        row_indices: u64,
        count: u64,
        destination: u64,
    ) -> Result<ArgumentList, Refusal> {
        if source == destination {
            return Err(Refusal(
                "a gather by index requires nonoverlapping source and destination",
            ));
        }
        let args = ArgumentList(vec![
            Argument::DeviceBuffer(source),
            Argument::DeviceBuffer(row_indices),
            Argument::DeviceBuffer(count),
            Argument::DeviceBuffer(destination),
            Argument::U32(self.vectors),
        ]);
        args.validate()?;
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::DType;

    fn op(width: u32) -> GatherByIndex {
        GatherByIndex {
            dtype: DType::Bf16,
            width,
            alignment_bytes: 16,
        }
    }

    #[test]
    fn gathers_plan_one_cta_per_row_of_the_bucket_of_their_grid() {
        let gather = Gather::new(op(6144), 8192, SYMBOL).unwrap();
        assert_eq!(gather.rows(5, 0), Ok((5, 5)));
        assert_eq!(gather.rows(5, 2048), Ok((5, 2048)));
        assert!(gather.rows(6, 4).is_err());
        assert!(gather.rows(0, 0).is_err());
        assert!(gather.rows(1, 8193).is_err());
        let plan = gather.plan(2048).unwrap();
        assert_eq!(
            (plan.grid, plan.block, plan.pdl),
            ([2048, 1, 1], [256, 1, 1], true)
        );
        let bucket = smallest_bucket(5).unwrap();
        assert_eq!(gather.plan(5).unwrap().grid, [bucket, 1, 1]);
        assert_eq!(
            gather.arguments(1, 2, 3, 4).unwrap().0[4..],
            [Argument::U32(6144 * 2 / 16)]
        );
        assert!(gather.arguments(1, 2, 3, 1).is_err());
        assert!(gather.plan(8193).is_err());
    }

    #[test]
    fn widths_alignments_and_capacities_past_the_entry_are_refused() {
        // A row of a partial vector and views aligned below a vector are
        // SM120's to refuse; the contract takes them.
        let odd = op(6148);
        assert_eq!(odd.validate(), Ok(()));
        assert!(Gather::new(odd, 32, SYMBOL).is_err());
        let narrow = GatherByIndex {
            alignment_bytes: 8,
            ..op(6144)
        };
        assert_eq!(narrow.validate(), Ok(()));
        assert!(Gather::new(narrow, 32, SYMBOL).is_err());
        assert!(Gather::new(op(6144), 0, SYMBOL).is_err());
        assert!(Gather::new(op(6144), crate::backend::MAX_CAPACITY * 2, SYMBOL).is_err());
        let f32 = GatherByIndex {
            dtype: DType::F32,
            width: 4,
            alignment_bytes: 16,
        };
        assert!(Gather::new(f32, 32, SYMBOL).is_ok());
    }

    #[test]
    fn the_cubin_holds_the_shared_entry_alone() {
        let spec = cubin_spec();
        assert_eq!(spec.name, CUBIN);
        let symbols: Vec<_> = spec.entries.iter().map(|e| e.symbol.as_str()).collect();
        assert_eq!(symbols, [SYMBOL]);
        assert_eq!(spec.entries[0].params.len(), 5);
        let declared = declared_entries();
        assert_eq!(declared.len(), 1);
        assert_eq!(
            (declared[0].op, declared[0].profile, declared[0].invariance),
            (OpKind::Row, None, INVARIANCE)
        );
    }
}
