//! Status defects over unchanged trained-quantizer cases.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantization, QuantizeOutput};
use kernels_sm120::row::block_quantize::{self, Entry};
pub const FAMILY: &str = "block_quantize_mutants";
pub const CUBIN: &str = "block_quantize_mutants_sm120";
pub const FAULTS: [&str; 5] = [
    "drop_nonfinite",
    "drop_scale",
    "drop_output",
    "wrong_row",
    "missing_or",
];
pub fn entries() -> Vec<Entry> {
    block_quantize::lab_entries()
        .into_iter()
        .filter(|e| {
            e.query.width == 128
                && e.query.input == DType::F32
                && ((e.query.format == BlockQuantization::E2M1E4M3Scale16
                    && e.query.output == QuantizeOutput::Codes)
                    || (e.query.format == BlockQuantization::E4M3_CEIL32
                        && e.query.output == QuantizeOutput::Bf16))
        })
        .collect()
}
pub fn symbol(symbol: &str, fault: &str) -> String {
    format!("{}_{fault}_v2", symbol.strip_suffix("_v2").unwrap())
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries()
        .into_iter()
        .flat_map(|e| {
            FAULTS.map(|fault| {
                let mut spec = e.spec();
                spec.symbol = symbol(&e.symbol, fault);
                let policy = match fault {
                    "drop_nonfinite" => "Drop<1>",
                    "drop_scale" => "DropPriorScale",
                    "drop_output" => "Drop<4>",
                    "wrong_row" => "WrongRow",
                    _ => "MissingOr",
                };
                spec.body = spec.body.replacen(
                    ">(input",
                    &format!(",chop::row::block_quantize::mutants::{policy}>(input"),
                    1,
                );
                spec
            })
        })
        .collect();
    vec![CubinSpec {
        name: CUBIN.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["crates/kernels-sm120/cuda".into()],
        headers: vec![
            "#include \"crates/kernel-lab/cuda/probes/block_quantize_mutants.cuh\"".into(),
        ],
        defines: vec![],
        entries,
    }]
}
