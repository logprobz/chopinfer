//! Lab instances of the split's F32 routed sum that adds a row
//! (`chained::RoutedSum::add_entry`, routed_sum_add.cuh) at the split's
//! route sum (`chained::split::sums`, all 288 experts the rank's): the
//! unmodified kernel and every deliberate defect of
//! `chop::moe::chained::SumAddMutation` (the addend never added, the addend
//! starting the sum, no dependency wait). Shared with
//! `tests/routed_sum_add.rs` and the split invariance cases.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, RoutedSum, RoutedSumOutput, split};

pub const FAMILY: &str = "moe_chained_split_sum_add";

/// The lab cubin.
pub const CUBIN: &str = "chop_moe_chain_lab_split_routed_sum_add";

/// The instances' symbol prefix: the unmodified kernel's symbol differs
/// from the split's served one (`{split::PREFIX}_sum_add`).
pub const PREFIX: &str = "chop_moe_chain_lab_split";

/// `chop::moe::chained::SumAddMutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "no_addend", "addend_first", "drop_wait"];

/// The split's route sum in its F32 form.
pub fn sum() -> RoutedSum {
    split::sums()[0].with_output(RoutedSumOutput::F32)
}

/// The symbol of the instance with `mutation`.
pub fn symbol(mutation: &str) -> String {
    sum()
        .add_entry_with(PREFIX, mutation)
        .expect("the F32 route sum adds a row")
        .symbol
}

/// Every instance, in [`MUTATIONS`] order.
pub fn cubin() -> CubinSpec {
    let mut spec = chained::routed_sum_add_cubin_spec(CUBIN.into(), PREFIX, &[sum()])
        .expect("the F32 route sum's cubin");
    spec.entries = MUTATIONS
        .iter()
        .map(|mutation| {
            sum()
                .add_entry_with(PREFIX, mutation)
                .expect("the F32 route sum adds a row")
        })
        .collect();
    spec
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[CUBIN], vec![cubin()])
}
