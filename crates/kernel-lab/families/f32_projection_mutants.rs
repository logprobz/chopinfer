//! Device controls over the same projection entry and unchanged kit cases.
use kernel_build::CubinSpec;
use kernel_core::contract::WeightFormat;
pub const FAMILY: &str = "f32_projection_mutants";
pub const CUBIN: &str = "f32_projection_mutants_v1";
pub const FAULTS: [&str; 3] = ["drop_lo", "merged_chains", "tf32"];
pub fn symbol(symbol: &str, fault: &str) -> String {
    format!(
        "{}_{fault}_v1",
        symbol.strip_suffix("_v1").expect("versioned entry")
    )
}
pub fn cubins() -> Vec<CubinSpec> {
    let entry = super::f32_projection::entry(17, 257, 9, WeightFormat::Bf16);
    let entries = FAULTS.into_iter().map(|fault| {
        let mut spec = entry.entry_spec();
        spec.symbol = symbol(entry.symbol(),fault);
        let (split,chains) = match fault {
            "drop_lo" => ("mutants::DropLo","SeparateChains"),
            "merged_chains" => ("Components","mutants::MergedChains"),
            _ => ("mutants::Tf32","SeparateChains"),
        };
        spec.body = format!("chop::exact_f32::run<17,257,false,chop::exact_f32::{split},chop::exact_f32::{chains}>(activation,weight,output,live_rows,rows);");
        spec
    }).collect();
    crate::declared(
        &[CUBIN],
        vec![CubinSpec {
            name: CUBIN.into(),
            sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
            include_dirs: vec!["crates/kernels-sm120/cuda".into()],
            headers: vec![
                "#include \"crates/kernel-lab/cuda/probes/f32_projection_mutants.cuh\"".into(),
            ],
            defines: vec![],
            entries,
        }],
    )
}
