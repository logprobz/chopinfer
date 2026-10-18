//! Device defects checked by the unchanged SwiGLU status cases.
use kernel_build::CubinSpec;
use kernels_sm120::row::silu_mul_quantize::{self, Entry};

pub const FAMILY: &str = "silu_mul_quantize_mutants";
pub const CUBIN: &str = "silu_mul_quantize_mutants_sm120";
pub const FAULTS: [&str; 4] = ["drop_flag", "wrong_row", "partial_row", "missing_skip"];

pub fn entry() -> Entry {
    silu_mul_quantize::lab_entries().remove(1)
}

pub fn symbol(symbol: &str, fault: &str) -> String {
    format!("{}_{fault}_v2", symbol.strip_suffix("_v2").unwrap())
}

pub fn cubins() -> Vec<CubinSpec> {
    let entry = entry();
    let entries = FAULTS.into_iter().map(|fault| {
        let mut spec = entry.spec();
        spec.symbol = symbol(&entry.symbol, fault);
        let policy = match fault {
            "drop_flag" => "chop::row::silu_mul_quantize::mutants::DropNonfinite,true,true",
            "wrong_row" => "chop::row::silu_mul_quantize::mutants::WrongRow,true,true",
            "partial_row" => "chop::row::block_quantize::StatusOr,true,false",
            _ => "chop::row::block_quantize::StatusOr,false,true",
        };
        spec.body = spec.body.replacen(">(gate", &format!(",{policy}>(gate"), 1);
        spec
    }).collect();
    crate::declared(&[CUBIN], vec![CubinSpec {
        name: CUBIN.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["crates/kernels-sm120/cuda".into()],
        headers: vec!["#include \"crates/kernel-lab/cuda/probes/silu_mul_quantize_mutants.cuh\"".into()],
        defines: vec![],
        entries,
    }])
}
