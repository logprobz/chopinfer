//! Negotiated prefill execution. Legacy execute always runs the full schedule.

use super::Instance;
use bundle_runtime::entry::Bundle;
use bundle_runtime::table::with_instance;
use runtime_bundle_abi::prefill::BundlePrefillPolicyV1;
use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1, ExecuteResultSinkV1};

pub(crate) fn describe(handle: BundleHandleV1) -> BundlePrefillPolicyV1 {
    let mut chunk = 0;
    let status = Instance::instances().with(|table| {
        with_instance(table, handle, |instance| {
            chunk = instance.config.prefill.chunk;
            BundleStatusV1::Success
        })
    });
    if status == BundleStatusV1::Success {
        BundlePrefillPolicyV1::guaranteed(chunk)
    } else {
        BundlePrefillPolicyV1::failure(status)
    }
}

pub(crate) fn execute(
    handle: BundleHandleV1,
    bytes: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> BundleStatusV1 {
    let envelope = match runtime_bundle_sdk::prefill::decode(bytes) {
        Ok(envelope) => envelope,
        Err(_) => return BundleStatusV1::InvalidArgument,
    };
    Instance::instances().with(|table| {
        with_instance(table, handle, |instance| {
            if instance.ledger.step().is_some() {
                return BundleStatusV1::InvalidState;
            }
            if instance.config.prefill.chunk == 0
                || envelope
                    .boundaries
                    .iter()
                    .any(|b| b.chunk_tokens != instance.config.prefill.chunk)
            {
                return BundleStatusV1::InvalidArgument;
            }
            let all_prefill = envelope
                .batch
                .items()
                .iter()
                .filter(|item| !item.is_decode() && !item.is_verify())
                .all(|item| {
                    envelope
                        .boundaries
                        .iter()
                        .any(|b| b.request_id == item.request_id())
                });
            instance.skip_prefill =
                instance.config.prefill.skip && all_prefill && !instance.taps.text();
            instance.prefill_boundaries = envelope.boundaries;
            instance.execute_step(envelope.original, sink)
        })
    })
}
