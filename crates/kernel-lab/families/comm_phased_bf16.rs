//! The phased all-reduce of BF16 partials from F32 inputs
//! (`comm::phased_bf16::lab_cubin_specs`): the served cubin, one cubin per
//! lab mutant (`comm::phased_bf16::MUTANTS`) and the slow-rank control
//! (`comm::phased_bf16::SLOW_CONTROL`).

use kernel_build::CubinSpec;
use kernels_sm120::comm::phased_bf16;

pub const FAMILY: &str = "comm_phased_bf16";

pub fn cubins() -> Vec<CubinSpec> {
    let mutants: Vec<String> = phased_bf16::MUTANTS
        .iter()
        .map(|(tag, _)| phased_bf16::mutant_cubin(tag))
        .chain([phased_bf16::mutant_cubin(phased_bf16::SLOW_CONTROL)])
        .collect();
    let mut names = vec![phased_bf16::LAB_CUBIN];
    names.extend(mutants.iter().map(String::as_str));
    crate::declared(&names, phased_bf16::lab_cubin_specs())
}
