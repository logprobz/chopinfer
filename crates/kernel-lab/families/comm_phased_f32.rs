//! The phased F32 all-reduce of large row sets
//! (`comm::phased_f32::lab_cubin_specs`): the served cubin and one cubin per
//! lab mutant (`comm::phased_f32::MUTANTS`).

use kernel_build::CubinSpec;
use kernels_sm120::comm::phased_f32;

pub const FAMILY: &str = "comm_phased_f32";

pub fn cubins() -> Vec<CubinSpec> {
    let mutants: Vec<String> = phased_f32::MUTANTS
        .iter()
        .map(|(tag, _)| phased_f32::mutant_cubin(tag))
        .collect();
    let mut names = vec![phased_f32::LAB_CUBIN];
    names.extend(mutants.iter().map(String::as_str));
    crate::declared(&names, phased_f32::lab_cubin_specs())
}
