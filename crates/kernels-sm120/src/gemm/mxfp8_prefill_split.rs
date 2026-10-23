//! Split full prefill tiles from a short tail using the existing fold engine.
use super::{mxfp8_fold::Entry, mxfp8_fold_packed, mxfp8_prefill as tiled};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

/// A model's measured bounds for choosing the split tail. Bounds are on
/// launch rows and remainder rows, not device data or arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TailPolicy {
    pub max_rows: u32,
    pub max_tail_rows: u32,
}
impl TailPolicy {
    pub const ALL: Self = Self {
        max_rows: u32::MAX,
        max_tail_rows: 127,
    };
    pub fn validate(self) -> Result<(), Refusal> {
        if self.max_rows < 128 || !(1..128).contains(&self.max_tail_rows) {
            return Err(Refusal("invalid split-tail row bounds"));
        }
        Ok(())
    }
    pub fn splits(self, rows: u32) -> bool {
        let tail = rows % tiled::TILE;
        rows >= tiled::TILE && rows <= self.max_rows && tail != 0 && tail <= self.max_tail_rows
    }
}

pub fn symbol(e: &Entry) -> String {
    format!("{}_split_full_v1", tiled::symbol(e))
}
pub fn tail_symbol(e: &Entry) -> String {
    format!("{}_split_tail_v1", tiled::symbol(e))
}
pub fn validation_symbol(e: &Entry) -> String {
    format!("{}_admit_v1", symbol(e))
}

pub fn specs(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    if e.decode() {
        return Err(Refusal("split prefill requires the non-decode fold entry"));
    }
    let mut specs = tiled::specs(e)?;
    specs[0].symbol = validation_symbol(e);
    specs[1].symbol = symbol(e);
    specs[1].body = specs[1]
        .body
        .replace("mxfp8_prefill::run<", "mxfp8_prefill_split::full<");
    let mut tail = e.spec();
    tail.symbol = tail_symbol(e);
    tail.body = tail
        .body
        .replace("mxfp8_fold::run<", "mxfp8_prefill_split::tail<");
    specs.push(tail);
    Ok(specs)
}

/// The full-tile entry reading `mxfp8-fold-packed-8x128-v1` weight codes,
/// launched with `mxfp8_prefill::packed_arguments`, and [`packed_tail_spec`].
pub fn packed_specs(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    if e.decode() {
        return Err(Refusal("split prefill requires the non-decode fold entry"));
    }
    let mut full = tiled::packed_spec(e)?;
    full.symbol = mxfp8_fold_packed::symbol(&symbol(e));
    full.body = full
        .body
        .replace("mxfp8_prefill::run<", "mxfp8_prefill_split::full<");
    Ok(vec![full, packed_tail_spec(e)?])
}

/// `launches` of a prepared split prefill op
/// (`Sm120::with_mxfp8_prefill_split`) moved to packed weight codes. Full
/// tiles, split or not, take their packed entries and the packed weight map;
/// the tail takes its packed entry with unchanged arguments. Validation
/// launches read no codes and are kept.
pub fn packed_launches(e: &Entry, launches: Vec<Launch>) -> Result<Vec<Launch>, Refusal> {
    let mut launches = tiled::packed_launches(e, launches)?;
    let (full, tail) = (symbol(e), tail_symbol(e));
    for launch in &mut launches {
        if launch.symbol == full {
            tiled::move_to_packed(e, launch, mxfp8_fold_packed::symbol(&full))?;
        } else if launch.symbol == tail {
            launch.symbol = mxfp8_fold_packed::symbol(&tail);
        }
    }
    Ok(launches)
}

/// The tail entry reading `mxfp8-fold-packed-8x128-v1` weight codes
/// (`gemm::mxfp8_fold_packed`), with the parameters, plan and arithmetic
/// of [`tail_symbol`]'s entry.
pub fn packed_tail_spec(e: &Entry) -> Result<EntrySpec, Refusal> {
    if e.decode() {
        return Err(Refusal("split prefill requires the non-decode fold entry"));
    }
    let mut tail = e.packed_spec()?;
    tail.symbol = mxfp8_fold_packed::symbol(&tail_symbol(e));
    tail.body = tail
        .body
        .replace("mxfp8_fold::run<", "mxfp8_prefill_split::tail<");
    Ok(tail)
}

pub fn plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let mut plan = tiled::plan(e, rows)?;
    plan.grid[1] = (rows / tiled::TILE).max(1);
    plan.validate()?;
    Ok(plan)
}

pub fn tail_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    if e.decode() {
        return Err(Refusal("split prefill requires the non-decode fold entry"));
    }
    let mut plan = e.plan(rows)?;
    // Device replay can move the tail; reserve at most 127 live rows.
    plan.grid[1] = rows.clamp(1, 127).div_ceil(16);
    plan.validate()?;
    Ok(plan)
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = tiled::cubin_spec(name, entries)?;
    spec.headers = vec!["#include \"gemm/mxfp8_prefill_split.cuh\"".into()];
    spec.entries = entries
        .iter()
        .map(specs)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(spec)
}

/// Deliberately wrong tail accumulator, used only by the lab witness.
pub fn control_spec(e: &Entry) -> EntrySpec {
    let mut tail = e.single_chain_control_spec();
    tail.symbol = format!("{}_single_chain_control_v1", tail_symbol(e));
    tail.body = tail
        .body
        .replace("mxfp8_fold::run<", "mxfp8_prefill_split::tail<");
    tail
}
