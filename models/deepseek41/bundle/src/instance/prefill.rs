//! Negotiated prefill execution. Legacy execute always runs the full schedule.

use super::Instance;
use bundle_runtime::entry::Bundle;
use bundle_runtime::table::with_instance;
use runtime_bundle_abi::prefill::{BundlePrefillAnchorsV2, BundlePrefillPolicyV1};
use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1, ExecuteResultSinkV1};

runtime_bundle_abi::export_prefill_cuts_api_v1!(PREFILL_CUTS_API, describe_cuts, execute_cuts,);

/// The instance's allocated and exported allowance, bounded by its served
/// sequence count rather than the package maximum or allocation bucket.
pub(super) fn anchor_allowance(config: &crate::config::Configuration) -> u32 {
    let shape = config.shape;
    if config.prefill.skip
        && config.prefill.scheduler_cuts()
        && shape.geometry.id == model_deepseek41_contract::geometry::FLASH.id
        && shape.ranks() == 4
    {
        config
            .execution
            .prefill_anchor_items
            .min(config.create.items.get())
    } else {
        0
    }
}

pub(crate) fn describe_anchors(handle: BundleHandleV1) -> BundlePrefillAnchorsV2 {
    let mut anchors = 0;
    let status = Instance::instances().with(|table| {
        with_instance(table, handle, |instance| {
            anchors = anchor_allowance(&instance.config);
            BundleStatusV1::Success
        })
    });
    if status == BundleStatusV1::Success {
        BundlePrefillAnchorsV2::kept(anchors)
    } else {
        BundlePrefillAnchorsV2::failure(status)
    }
}

pub(crate) fn describe(handle: BundleHandleV1) -> BundlePrefillPolicyV1 {
    policy(handle, false)
}

fn describe_cuts(handle: BundleHandleV1) -> BundlePrefillPolicyV1 {
    policy(handle, true)
}

fn policy(handle: BundleHandleV1, cuts: bool) -> BundlePrefillPolicyV1 {
    let mut chunk = 0;
    let status = Instance::instances().with(|table| {
        with_instance(table, handle, |instance| {
            if !cuts || instance.config.prefill.scheduler_cuts() {
                chunk = instance.config.prefill.chunk;
            }
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
    execute_envelope(handle, envelope, sink, false)
}

fn execute_cuts(
    handle: BundleHandleV1,
    bytes: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> BundleStatusV1 {
    let envelope = match runtime_bundle_sdk::prefill::decode_with_cuts(bytes) {
        Ok(envelope) => envelope,
        Err(_) => return BundleStatusV1::InvalidArgument,
    };
    execute_envelope(handle, envelope, sink, true)
}

fn execute_envelope(
    handle: BundleHandleV1,
    envelope: runtime_bundle_sdk::prefill::Envelope<'_>,
    sink: &mut ExecuteResultSinkV1,
    cuts: bool,
) -> BundleStatusV1 {
    Instance::instances().with(|table| {
        with_instance(table, handle, |instance| {
            if instance.ledger.step().is_some() {
                return BundleStatusV1::InvalidState;
            }
            if instance.config.prefill.chunk == 0
                || (cuts && !instance.config.prefill.scheduler_cuts())
                || envelope.boundaries.iter().any(|b| {
                    b.chunk_tokens != instance.config.prefill.chunk
                        || (cuts
                            && b.chunk_end != b.prompt_tokens
                            && !b.chunk_end.is_multiple_of(u64::from(
                                model_deepseek41_contract::serving::PREFILL_GRANULE,
                            )))
                })
            {
                return BundleStatusV1::InvalidArgument;
            }
            let all_prefill = contracted_text(&envelope);
            instance.skip_prefill =
                instance.config.prefill.skip && all_prefill && !instance.taps.text();
            instance.prefill_boundaries = envelope.boundaries;
            bundle_runtime::entry::with_route_counter(instance, |instance| {
                instance.execute_step(envelope.original, sink)
            })
        })
    })
}

fn contracted_text(envelope: &runtime_bundle_sdk::prefill::Envelope<'_>) -> bool {
    if !envelope.batch.media().is_empty() {
        return false;
    }
    envelope
        .batch
        .items()
        .iter()
        .filter(|item| !item.is_decode() && !item.is_verify())
        .all(|item| {
            envelope
                .boundaries
                .iter()
                .any(|b| b.request_id == item.request_id())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_sdk::prefill::{Boundary, decode, encode};
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1,
        BatchWireVersionV1,
    };

    #[test]
    fn media_records_keep_the_full_schedule_even_with_text_boundaries() {
        let text = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, false, 4, true, vec![1, 2, 3, 4])],
            vec![BatchAppendV1::new(0, 4, 4, vec![0], None)],
        );
        let boundary = Boundary {
            request_id: 7,
            prompt_tokens: 4,
            chunk_end: 4,
            chunk_tokens: 2048,
        };
        let bytes = encode(&text.encode().unwrap(), &[boundary]).unwrap();
        assert!(contracted_text(&decode(&bytes).unwrap()));
        let media = text.with_media(BatchMediaSectionV3 {
            encodes: vec![BatchMediaEncodeV3 {
                encoder: 7,
                geometry: [1; 4],
                output_rows: 1,
                pages: vec![0],
                payload_offset: 0,
                payload_bytes: 1,
            }],
            injects: Vec::new(),
            payload: vec![1],
        });
        let bytes = encode(&media.encode().unwrap(), &[boundary]).unwrap();
        assert!(!contracted_text(&decode(&bytes).unwrap()));
    }
}
