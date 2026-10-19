//! Storage and checked descriptors for the host's optional device result sink.
use super::execution::Input;
use super::state::{BLOCK_ROWS, Kind};
use crate::ops::{Bind, Lowered};
use bundle_runtime::resources::{Held, allocate_span};
use kernel_core::backend::{Counts, Need, Prepared, View};
use model_deepseek41_contract::own_ops::{
    OwnOps,
    acceptance::{Acceptance, AcceptanceBuffers},
};
use runtime_bundle_abi::device_results::DeviceCycleMetadataV1;
use runtime_bundle_abi::{
    BundleAbiError, BundleEventHandleV1, BundleMemoryPoolHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};

pub(crate) const ITEMS: u32 = 32;
pub(crate) const ROWS: u32 = ITEMS * (BLOCK_ROWS + 1);
pub(crate) const SAMPLE_OFFSET: u64 = ITEMS as u64 * 32;
pub(crate) const STOP_CAPACITY: u32 = 1 << 20;

pub(crate) struct Results {
    prepared: Prepared<AcceptanceBuffers<View>>,
    pub memory: Option<Memory>,
    pub samples: Vec<Span>,
    pub target_ready: Vec<BundleEventHandleV1>,
    pub sampled: Vec<BundleEventHandleV1>,
    pub ready: Option<BundleEventHandleV1>,
    pub metadata: Option<DeviceCycleMetadataV1>,
    pub last_epoch: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub items: Span,
    pub stops: Span,
    pub packet: Span,
}
impl Results {
    pub fn new(own: &dyn OwnOps, vocab: u32) -> Result<Self, String> {
        let prepared = own
            .dspark_host_acceptance(
                &Acceptance {
                    vocab,
                    sample_rows: ROWS,
                    draft_items: ITEMS,
                    stop_ids: STOP_CAPACITY,
                },
                Need::rows(ITEMS),
            )
            .map_err(|e| format!("DSpark host acceptance: {e:?}"))?;
        Ok(Self {
            prepared,
            memory: None,
            samples: Vec::new(),
            target_ready: Vec::new(),
            sampled: Vec::new(),
            ready: None,
            metadata: None,
            last_epoch: 0,
        })
    }
    pub fn allocate(
        &mut self,
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        held: &mut Vec<Held>,
    ) -> Result<(), BundleAbiError> {
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        if self.memory.is_none() {
            self.memory = Some(Memory {
                items: alloc(u64::from(ITEMS) * 40)?,
                stops: alloc(u64::from(STOP_CAPACITY) * 4)?,
                packet: alloc(SAMPLE_OFFSET + u64::from(ROWS) * 256)?,
            });
        }
        self.samples.push(alloc(u64::from(ROWS) * 256)?);
        Ok(())
    }
    pub fn launch(&self, proposals: Span, count: u32) -> Result<Vec<Lowered>, String> {
        let memory = self.memory.ok_or("DSpark result storage absent")?;
        let mut bind = Bind::default();
        let buffers = AcceptanceBuffers {
            items: bind.view(memory.items),
            proposals: bind.view(proposals),
            samples: bind.view(super::device::slice(
                memory.packet,
                SAMPLE_OFFSET,
                u64::from(ROWS) * 256,
            )?),
            stops: bind.view(memory.stops),
            results: bind.view(super::device::slice(
                memory.packet,
                0,
                u64::from(ITEMS) * 32,
            )?),
        };
        bind.launch(&self.prepared, &buffers, Counts::rows(count))
            .map_err(|e| e.to_string())
    }
}

/// The host owns its lease IDs. Check the same request and row order before
/// uploading descriptors, while the lifecycle separately validates model leases.
pub(crate) fn validate_metadata(
    metadata: &DeviceCycleMetadataV1,
    inputs: &[Input],
    vocab: u32,
    previous: u64,
) -> Result<(), String> {
    metadata
        .validate(runtime_bundle_abi::ResultSinkLimitsV1::new(
            u64::from(ROWS),
            u64::from(vocab),
            u64::from(ROWS) * u64::from(vocab),
        ))
        .map_err(|e| format!("DSpark result metadata: {e:?}"))?;
    if metadata.cycle_epoch <= previous
        || metadata.items.len() != inputs.len()
        || inputs.len() > ITEMS as usize
    {
        return Err("DSpark result epoch or item count differs".into());
    }
    let mut draft = 0;
    for (item, input) in metadata.items.iter().zip(inputs) {
        let Kind::Decode { drafts } = input.rows.kind else {
            return Err("DSpark results require decode rows".into());
        };
        let index = if drafts == 0 {
            u32::MAX
        } else {
            let index = draft;
            draft += 1;
            index
        };
        if drafts > BLOCK_ROWS
            || item.request_id != input.rows.request
            || item.acceptance.draft_tokens != drafts
            || item.acceptance.draft_item_index != index
        {
            return Err("DSpark result request, K or draft order differs".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::state::Rows;
    use super::*;
    use runtime_bundle_abi::device_results::{BundleAcceptanceInputV1, DeviceCycleItemV1};
    #[test]
    fn metadata_binds_epoch_requests_budgets_and_draft_offsets() {
        let inputs: Vec<_> = [0, 1, 5]
            .into_iter()
            .enumerate()
            .map(|(slot, drafts)| Input {
                rows: Rows {
                    request: slot as u64 + 1,
                    slot: slot as u32,
                    start: 4,
                    kind: Kind::Decode { drafts },
                },
                tokens: vec![1],
            })
            .collect();
        let metadata = DeviceCycleMetadataV1 {
            cycle_epoch: 5,
            termination_stop_ids: vec![3, 4],
            items: inputs
                .iter()
                .enumerate()
                .map(|(index, input)| DeviceCycleItemV1 {
                    request_id: input.rows.request,
                    slot_generation: 4,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: 5,
                        remaining_tokens: 9,
                        first_sample_row: [0, 1, 3][index],
                        draft_item_index: [u32::MAX, 0, 1][index],
                        draft_tokens: input.rows.kind.drafts(),
                        stop_offset: 0,
                        stop_count: 2,
                        reserved: 0,
                    },
                })
                .collect(),
        };
        validate_metadata(&metadata, &inputs, 100, 4).unwrap();
        assert!(
            validate_metadata(&metadata, &inputs, 100, 5)
                .unwrap_err()
                .contains("epoch")
        );
        for mutation in 0..5 {
            let mut bad = metadata.clone();
            match mutation {
                0 => bad.items[0].request_id = 99,
                1 => bad.items[2].acceptance.draft_tokens = 4,
                2 => {
                    bad.items[1].acceptance.draft_item_index = 1;
                    bad.items[2].acceptance.draft_item_index = 0;
                }
                3 => bad.items[0].slot_generation = 0,
                _ => bad.termination_stop_ids.reverse(),
            }
            assert!(
                validate_metadata(&bad, &inputs, 100, 4).is_err(),
                "mutation {mutation}"
            );
        }
    }
}
