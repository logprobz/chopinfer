//! Unselected native fold core variants with their exact arithmetic controls.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{
    mxfp8_fold, mxfp8_fold_stream, mxfp8_fold_tiled as decode, mxfp8_prefill_matrix as prefill,
};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_core";
pub const DECODE: &str = "fold_core_decode_sm120";
pub const PREFILL: &str = "fold_core_prefill_sm120";
pub const TILES: &[(u32, u32)] = &[(32, 3), (64, 2), (64, 3), (128, 2)];
pub fn bases() -> Vec<mxfp8_fold::Entry> {
    bundle_deepseek41::sm120::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .filter(|e| e.query.k != 15360)
        .collect()
}
pub fn decode_entries() -> Vec<decode::Entry> {
    bases()
        .into_iter()
        .filter(|e| e.query.n != 576)
        .flat_map(|base| {
            [
                decode::Schedule::NarrowRows,
                decode::Schedule::WideColumns,
                decode::Schedule::ReuseRows,
            ]
            .map(move |schedule| {
                let mut q = base.query;
                q.m_capacity = 32;
                let symbol = format!(
                    "chop_fold_core_n{}_k{}_g{}_{}_v1",
                    q.n,
                    q.k,
                    q.groups,
                    schedule.name()
                );
                decode::Entry {
                    packed: mxfp8_fold_stream::Entry::new(q, &symbol, false).unwrap(),
                    symbol,
                    schedule,
                    defer_admission: false,
                    omit_first: false,
                }
            })
        })
        .collect()
}
pub fn prefill_entries() -> Vec<prefill::Entry> {
    bases()
        .into_iter()
        .flat_map(|base| {
            TILES
                .iter()
                .map(move |&(m, s)| prefill::Entry::new(base.clone(), s, m).unwrap())
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut entries = decode_entries();
    for e in entries.clone() {
        let mut omit = e.clone();
        omit.omit_first = true;
        omit.symbol.push_str("_omit_v1");
        let mut delayed = e;
        delayed.defer_admission = true;
        delayed.symbol.push_str("_delayed_v1");
        entries.extend([omit, delayed]);
    }
    vec![
        decode::cubin_spec(DECODE, &entries).unwrap(),
        prefill::cubin_spec(PREFILL, &prefill_entries()).unwrap(),
    ]
}
