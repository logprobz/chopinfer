//! The phased all-reduce of BF16 partials whose writers take turns at each
//! destination (`comm::phased_turns_bf16::lab_cubin_specs`): the served
//! cubin, one cubin per lab mutant (`comm::phased_turns_bf16::MUTANTS`),
//! the slow-rank and late-scatter controls
//! (`comm::phased_turns_bf16::SLOW_CONTROL`, `LATE_SCATTER_CONTROL`) and
//! the timing variants (`comm::phased_turns_bf16::TIMING_VARIANTS`).

use kernel_build::CubinSpec;
use kernels_sm120::comm::phased_turns_bf16;

pub const FAMILY: &str = "comm_phased_turns_bf16";

pub fn cubins() -> Vec<CubinSpec> {
    let tagged: Vec<String> = phased_turns_bf16::MUTANTS
        .iter()
        .map(|(tag, _)| *tag)
        .chain([
            phased_turns_bf16::SLOW_CONTROL,
            phased_turns_bf16::LATE_SCATTER_CONTROL,
        ])
        .chain(phased_turns_bf16::TIMING_VARIANTS.iter().map(|(tag, _)| *tag))
        .map(phased_turns_bf16::mutant_cubin)
        .collect();
    let mut names = vec![phased_turns_bf16::LAB_CUBIN];
    names.extend(tagged.iter().map(String::as_str));
    crate::declared(&names, phased_turns_bf16::lab_cubin_specs())
}
