//! The bundle API over the deepseek41 instance, which every stage of the
//! native KV layout lowers for; the BF16 layout's create refuses until its
//! writers land.

use crate::instance::Instance;
use bundle_runtime::entry;

// The shared entry bodies over the deepseek41 instance
// (`bundle_runtime::entry`): retirement frees a request's slot, a verify
// step commits the rows the host accepted, and prefix snapshots capture and
// restore a slot's rings and tails; a restore fills the host's state slot.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_state_slot BUNDLE_DEEPSEEK41_API_V1,
    BUNDLE_DEEPSEEK41_EXTENDED_API_V1,
    entry::create::<Instance>,
    entry::execute::<Instance>,
    entry::commit::<Instance>,
    entry::cancel::<Instance>,
    entry::destroy::<Instance>,
    entry::retire_request::<Instance>,
    [entry::commit_accepted::<Instance>],
    entry::describe_prefix_state::<Instance>,
    entry::capture_prefix_state::<Instance>,
    entry::restore_prefix_state::<Instance>,
    entry::restore_prefix_state_in_slot::<Instance>,
);
