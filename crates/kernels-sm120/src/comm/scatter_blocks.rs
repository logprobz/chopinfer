//! Additive F32 scatter entries with a bounded last-destination block count.
use kernel_build::CubinSpec;

pub const BLOCKS: [u32; 3] = [2, 4, 8];

pub fn symbol(blocks: u32) -> String {
    assert!(BLOCKS.contains(&blocks));
    format!("chop_comm_reduce_scatter_f32_blocks{blocks}_v1")
}

/// The same arguments and state as the ordinary phased F32 scatter. Only
/// the partition of its last destination changes; use the ordinary gather.
pub fn cubin_spec(name: &str) -> CubinSpec {
    let mut spec = super::cubin_spec();
    let old = spec
        .entries
        .iter()
        .find(|e| e.symbol == "chop_comm_reduce_scatter_f32_v1")
        .unwrap();
    spec.entries = BLOCKS.into_iter().map(|blocks| {
        let mut entry = old.clone();
        entry.symbol = symbol(blocks);
        entry.body = format!("chop::comm::scatter_blocks::run<{blocks}>(input, partial, inboxes, flags, epoch, elements, rank, ranks);");
        entry
    }).collect();
    spec.name = name.into();
    spec.headers = vec!["#include \"comm/scatter_blocks.cuh\"".into()];
    spec
}
