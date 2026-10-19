//! Frozen direct-fold catalog, including unselected qualification controls.
use super::mxfp8_fold_stream;
use kernels_sm120::gemm::mxfp8_fold_direct::{Entry, Schedule};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "mxfp8_fold_direct";
pub const CUBIN: &str = "mxfp8_fold_direct_sm120";
pub const ROWS: [u32; 8] = [1, 2, 4, 6, 8, 16, 32, 64];

pub fn catalog() -> Result<Vec<Entry>, String> {
    let mut entries = vec![];
    for packed in mxfp8_fold_stream::catalog(&FLASH)? {
        let q = packed.ticket.query;
        let bytes = packed.packed_bytes() * if packed.ticket.paired { 2 } else { 1 };
        // Large matrices already fill the machine without narrow tiles.
        let schedules: &[Schedule] = if bytes > 32 << 20 {
            &[Schedule::Block]
        } else {
            &[Schedule::Warp, Schedule::Block, Schedule::LocalSplit]
        };
        for &schedule in schedules {
            entries.push(Entry {
                symbol: format!(
                    "chop_deepseek41_mxfp8_direct_n{}_k{}_g{}_{}_{}_v1",
                    q.n,
                    q.k,
                    q.groups,
                    if packed.ticket.paired {
                        "pair"
                    } else {
                        "single"
                    },
                    schedule.name()
                ),
                packed: packed.clone(),
                schedule,
                defer_admission: false,
                omit_first: false,
            });
        }
    }
    Ok(entries)
}
pub fn probe(deferred: bool) -> Result<Entry, String> {
    let mut entry = catalog()?
        .into_iter()
        .find(|e| {
            e.packed.ticket.query.n == 512
                && !e.packed.ticket.paired
                && e.schedule == Schedule::Warp
        })
        .ok_or("missing fold probe")?;
    entry.defer_admission = deferred;
    entry.omit_first = !deferred;
    entry.symbol = format!(
        "{}_{}_v1",
        entry.symbol.trim_end_matches("_v1"),
        if deferred { "deferred" } else { "omit" }
    );
    Ok(entry)
}
pub fn cubin() -> Result<kernel_build::CubinSpec, String> {
    let mut entries = catalog()?;
    entries.extend([probe(false)?, probe(true)?]);
    kernels_sm120::gemm::mxfp8_fold_direct::cubin_spec(CUBIN, &entries).map_err(|e| e.to_string())
}
