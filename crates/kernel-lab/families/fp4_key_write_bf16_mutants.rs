//! Change only the BF16 store expression in the current writer's body.

use kernel_build::CubinSpec;
use kernels_sm120::gated_pooling::key_write::cubin_spec;

pub const FAMILY: &str = "fp4_key_write_bf16_mutants";
pub const CUBIN: &str = "fp4_key_write_bf16_mutants_v1";
pub const FAULTS: [&str; 2] = ["truncate_input", "drop_scale"];

pub fn symbol(served: &str, fault: &str) -> String {
    let (stem, version) = served.rsplit_once("_v").unwrap();
    format!("{stem}_{fault}_v{version}")
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = crate::fp4_key_write_bf16::all_entries().into_iter()
        .filter(|entry| !entry.packed() && entry.page_size()==64/entry.ratio()).collect::<Vec<_>>();
    assert_eq!(entries.len(),4);
    let mut spec = cubin_spec(CUBIN, &entries).unwrap();
    let originals = spec.entries.clone();
    spec.entries.clear();
    spec.sources_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    spec.include_dirs = vec!["kernels-sm120/cuda".into()];
    spec.headers = vec!["#include \"kernel-lab/cuda/probes/fp4_key_write_bf16_mutants.cuh\"".into()];
    let serving = include_str!("../../kernels-sm120/cuda/gated_pooling/key_write.cuh");
    let store = "        uint16_t* target = reinterpret_cast<uint16_t*>(code_row);";
    assert_eq!(serving.matches(store).count(),1);
    let expected = serving.replace("#pragma once", "")
        .replace("namespace chop::gated_pooling", "namespace chop::bf16_write_tests")
        .replace("class Status = chop::row::block_quantize::StatusOr>", "class Status, class Store>")
        .replace(store, &format!("        value.pair = Store::pair(values,at,value,flags);\n{store}"));
    let mirrored = include_str!("../cuda/probes/fp4_key_write_bf16_mutants.cuh")
        .split_once("// BEGIN mirrored writer, checked against the serving body by the family builder.\n").unwrap().1;
    assert_eq!(mirrored, expected, "mutant body must track the current serving kernel");
    for (fault, policy) in FAULTS.into_iter().zip(["TruncateInput","DropScale"]) {
        for original in &originals {
            let mut entry = original.clone();
            entry.symbol = symbol(&entry.symbol,fault);
            entry.body = entry.body.replace("chop::gated_pooling::key_write", "chop::bf16_write_tests::key_write")
                .replacen(">(key,", &format!(",chop::row::block_quantize::StatusOr,chop::bf16_write_tests::{policy}>(key,"),1);
            spec.entries.push(entry);
        }
    }
    crate::declared(&[CUBIN], vec![spec])
}
