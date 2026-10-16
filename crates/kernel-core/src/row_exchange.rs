//! SM120's row exchange layout: the flag and epoch sizes, launch geometry
//! and arguments of `chop_comm_exchange_rows_{bf16,f32}_v1`.
//!
//! The exchange's math is `collective::ShardedRows` and its owner blocks
//! `collective::owned_rows`; a backend prepares it through
//! `Backend::exchange_rows`. This layout stays here for the bundle callers
//! that build the launch themselves, until they move to the backend.

use crate::contract::{Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate};

pub use crate::collective::owned_rows as exchange_rows_owned;

/// Decode logits rows exchanged by `chop_comm_exchange_rows_{bf16,f32}_v1`:
/// rows split into contiguous blocks owned by ranks in order (see
/// [`exchange_rows_owned`]), and every rank's shard of a row lands at
/// `outputs[owner] + (r * ranks + q) * shard`, the gather's layout.
pub struct ExchangeRowsCall {
    pub input: u64,
    pub outputs: Vec<u64>,
    pub flags: Vec<u64>,
    pub epoch: u64,
    pub rows: u32,
    pub shard: u64,
    pub rank: u32,
    pub ranks: u32,
}

/// One cluster of eight CTAs of 256 threads. Each CTA needs its own epoch
/// cell; the flags hold one word per (source, CTA) and one per (receiver,
/// CTA). Both start zeroed and are not shared with other entries.
pub const EXCHANGE_ROWS_EPOCH_BYTES: u64 = EXCHANGE_ROWS_CTAS * 4;
pub const EXCHANGE_ROWS_FLAG_BYTES: u64 = 2 * MAX_COMM_RANKS as u64 * EXCHANGE_ROWS_CTAS * 4;
const EXCHANGE_ROWS_CTAS: u64 = 8;

pub fn exchange_rows_plan() -> LaunchPlan {
    LaunchPlan {
        grid: [8, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: EXCHANGE_ROWS_EPOCH_BYTES + EXCHANGE_ROWS_FLAG_BYTES,
        cluster: [8, 1, 1],
        pdl: true,
    }
}

pub fn exchange_rows_arguments(call: ExchangeRowsCall) -> Result<ArgumentList, Refusal> {
    let ranks = call.ranks as usize;
    if call.ranks == 0
        || ranks > MAX_COMM_RANKS
        || call.rank >= call.ranks
        || call.outputs.len() != ranks
        || call.flags.len() != ranks
        || call.rows == 0
        || call.shard == 0
    {
        return Err(Refusal("invalid row exchange geometry"));
    }
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(call.input),
        Argument::DevicePointerArray(call.outputs),
        Argument::DevicePointerArray(call.flags),
        Argument::DeviceBuffer(call.epoch),
        Argument::U32(call.rows),
        Argument::U64(call.shard),
        Argument::U32(call.rank),
        Argument::U32(call.ranks),
    ]);
    args.validate()?;
    Ok(args)
}
