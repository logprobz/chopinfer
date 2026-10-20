//! A private successor draft is useful only after its predicted commit matches.
use super::execution::{DraftInput, Input};
use super::state::{BLOCK_ROWS, CommitPlan, DraftBlock, Kind, RowProjection, State};
use bundle_runtime::graphs::GraphCache;
use model_deepseek41_contract::serving::DeepSeek;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use runtime_bundle_abi::device_results::{
    BundleAcceptanceResultV1, DeviceAcceptanceV1, DeviceCycleMetadataV1,
};
use std::collections::BTreeMap;

pub(crate) struct Cache {
    pub enabled: bool,
    pub ranks: Vec<Memory>,
    pub graphs: BTreeMap<u32, GraphCache<DeepSeek>>,
    pub plan: Option<Plan>,
    pub feedback: Option<Feedback>,
    pub reused: bool,
    pub done: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub joined: Option<runtime_bundle_abi::BundleEventHandleV1>,
}

pub(crate) struct Memory {
    pub undo_ring: Span,
    pub next_ring: Span,
    pub draft_ids: Span,
    pub confidence: Span,
    pub status: super::status::Status,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            enabled: enabled(),
            ranks: Vec::new(),
            graphs: BTreeMap::new(),
            plan: None,
            feedback: None,
            reused: false,
            done: Vec::new(),
            joined: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Feedback {
    pub metadata: DeviceCycleMetadataV1,
    pub inputs: Vec<Input>,
    pub packet: Vec<u8>,
}

pub(crate) struct Plan {
    pub committed: State,
    pub retain: CommitPlan,
    pub projections: Vec<RowProjection>,
    pub drafts: Vec<DraftInput>,
}

impl Feedback {
    /// The device result predicts work, never authorizes a host commit.
    pub fn plan(&self, state: &State, vocab: u32) -> Result<Option<Plan>, String> {
        if self.inputs.is_empty()
            || self.inputs.len() > 8
            || self.metadata.items.len() != self.inputs.len()
            || self.inputs.iter().any(|input| {
                !matches!(
                    input.rows.kind,
                    Kind::Decode {
                        drafts: 1..=BLOCK_ROWS
                    }
                )
            })
        {
            return Ok(None);
        }
        super::results::validate_metadata(&self.metadata, &self.inputs, vocab, 0)?;
        let mut kept = Vec::new();
        let mut drafts = Vec::new();
        for (index, (input, item)) in self.inputs.iter().zip(&self.metadata.items).enumerate() {
            // The host validates the same record and owns any failure. An
            // unreadable or stale record only declines the prediction.
            let Ok(DeviceAcceptanceV1::Prefix {
                emitted_rows,
                terminal: false,
                ..
            }) = BundleAcceptanceResultV1::from_le_bytes(
                self.packet
                    .get(index * 32..index * 32 + 32)
                    .ok_or("DSpark successor acceptance record is absent")?,
            )
            .and_then(|result| result.validate(item.acceptance))
            else {
                return Ok(None);
            };
            if u64::from(emitted_rows) + 1 >= item.acceptance.remaining_tokens {
                return Ok(None);
            }
            let sample = u64::from(item.acceptance.first_sample_row + emitted_rows - 1);
            let at = (super::results::SAMPLE_OFFSET + sample * 256) as usize;
            let words = self
                .packet
                .get(at..at + 8)
                .ok_or("DSpark successor sample is absent")?;
            let bonus = u32::from_le_bytes(words[..4].try_into().expect("sample word"));
            let status = u32::from_le_bytes(words[4..].try_into().expect("sample word"));
            if status != 0 || bonus >= vocab {
                return Ok(None);
            }
            let end = input
                .rows
                .start
                .checked_add(u64::from(emitted_rows))
                .ok_or("DSpark successor position overflow")?;
            if end >= state.config().context_tokens
                || end
                    .checked_add(u64::from(BLOCK_ROWS))
                    .is_none_or(|end| end > state.config().position_capacity)
            {
                return Ok(None);
            }
            kept.push((input.rows.request, emitted_rows));
            drafts.push(DraftInput {
                block: DraftBlock {
                    request: input.rows.request,
                    slot: input.rows.slot,
                    main_rows: end,
                    compute_rows: BLOCK_ROWS,
                    emit_rows: BLOCK_ROWS,
                    verify_rows: BLOCK_ROWS + 1,
                },
                bonus,
            });
        }
        let mut committed = state.clone();
        let retain = committed.begin_commit(&kept)?;
        if retain.items().len() != self.inputs.len()
            || retain
                .items()
                .iter()
                .zip(&self.inputs)
                .any(|(item, input)| item.rows != input.rows)
        {
            return Err("DSpark successor rows differ from the pending transaction".into());
        }
        committed.complete_commit(retain.clone())?;
        let mut projected = committed.clone();
        let mut projections = Vec::new();
        for draft in &drafts {
            projected.skip_unread(draft.block.request)?;
            if projected.has_bulk(draft.block.request) {
                return Ok(None);
            }
            if let Some(pass) = projected.row_projection(draft.block.request)? {
                projections.push(pass);
            }
        }
        if projections.iter().map(|pass| pass.count).sum::<u32>() > state.batch_rows() {
            return Ok(None);
        }
        Ok(Some(Plan {
            committed,
            retain,
            projections,
            drafts,
        }))
    }
}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_ENABLED.with(std::cell::Cell::get) {
        return value;
    }
    std::env::var_os("CHOPINFER_DSPARK_PREFETCH").is_some_and(|v| v == "1")
}

#[cfg(test)]
thread_local! {
    static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(value: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_ENABLED.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(TEST_ENABLED.with(|cell| cell.replace(Some(value))));
    f()
}

#[cfg(test)]
mod tests;

impl Plan {
    pub fn matches(&self, state: &State, inputs: &[Input]) -> bool {
        state == &self.committed
            && inputs.len() == self.drafts.len()
            && inputs.iter().zip(&self.drafts).all(|(input, draft)| {
                matches!(
                    input.rows.kind,
                    Kind::Decode {
                        drafts: 1..=BLOCK_ROWS
                    }
                ) && input.rows.request == draft.block.request
                    && input.rows.slot == draft.block.slot
                    && input.rows.start == draft.block.main_rows
                    && input.tokens == [draft.bonus]
            })
    }
}
