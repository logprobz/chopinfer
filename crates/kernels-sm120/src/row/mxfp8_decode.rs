//! MXFP8 row decoding on SM120, one CTA per row of the launch bucket.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::mxfp8_rows::Mxfp8Rows;
use kernel_core::properties::Invariance;

use super::{ptr, smallest_bucket, u32_arg};
use crate::invariance::{Declared, OpKind};

/// The decode's header, relative to the CUDA source root.
pub const HEADER: &str = "row/mxfp8_decode.cuh";
/// Threads of the decode entry.
pub const THREADS: u32 = 256;
/// The shared decode entry.
pub const SYMBOL: &str = "chop_row_mxfp8_decode_v1";
/// The cubin of the shared entry, in the lab and in a model's build.
pub const CUBIN: &str = "row_mxfp8_decode_sm120";
/// Independent element conversion keeps row, chunk and batch bits.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);

/// The entry's code under `symbol`.
pub fn entry_spec(symbol: &str) -> EntrySpec {
    EntrySpec::new(
        symbol.into(),
        vec![
            ptr("codes", "const uint8_t"),
            ptr("scales", "const uint8_t"),
            ptr("live_rows", "const uint32_t"),
            ptr("output", "uint16_t"),
            u32_arg("width"),
        ],
        "chop::row::mxfp8_decode::rows(codes,scales,live_rows,output,width);".into(),
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

/// An MXFP8 row decode of up to `capacity` rows through entry `symbol`.
#[derive(Clone, Debug)]
pub struct Decode {
    op: Mxfp8Rows,
    capacity: u32,
    symbol: String,
}

impl Decode {
    /// Refuses capacities outside SM120's row buckets and overflowing output.
    pub fn new(op: Mxfp8Rows, capacity: u32, symbol: &str) -> Result<Self, Refusal> {
        op.validate()?;
        if capacity == 0 || smallest_bucket(capacity).is_none() {
            return Err(Refusal(
                "an SM120 MXFP8 row decode's capacity exceeds its row buckets",
            ));
        }
        op.output_bytes(capacity)?;
        Ok(Self {
            op,
            capacity,
            symbol: symbol.into(),
        })
    }

    pub fn op(&self) -> Mxfp8Rows {
        self.op
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// The launch's decoded rows and grid: at most `rows` rows decoded,
    /// gridded over `padded` rows when nonzero, else `rows`.
    pub fn rows(&self, rows: u32, padded: u32) -> Result<(u32, u32), Refusal> {
        let grid = if padded == 0 { rows } else { padded };
        if rows > grid || grid == 0 || grid > self.capacity {
            return Err(Refusal(
                "an MXFP8 row decode grids from one row to its capacity and decodes at most those rows",
            ));
        }
        Ok((rows, grid))
    }

    /// One CTA per row of the smallest row bucket holding `grid` rows.
    pub fn plan(&self, grid: u32) -> Result<LaunchPlan, Refusal> {
        if grid == 0 || grid > self.capacity {
            return Err(Refusal("decode rows exceed the prepared capacity"));
        }
        let bucket = smallest_bucket(grid).ok_or(Refusal("no row bucket holds the decode"))?;
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
        codes: u64,
        scales: u64,
        live_rows: u64,
        output: u64,
    ) -> Result<ArgumentList, Refusal> {
        if [codes, scales, live_rows].contains(&output) {
            return Err(Refusal(
                "an MXFP8 row decode requires nonoverlapping input and output",
            ));
        }
        let args = ArgumentList(vec![
            Argument::DeviceBuffer(codes),
            Argument::DeviceBuffer(scales),
            Argument::DeviceBuffer(live_rows),
            Argument::DeviceBuffer(output),
            Argument::U32(self.op.width),
        ]);
        args.validate()?;
        Ok(args)
    }
}
