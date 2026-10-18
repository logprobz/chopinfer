//! Lowering of the MTP drafter's own rows through the instance's own row
//! ops (`crate::ops::own`): its hidden inputs and its residual add; of its
//! head rows on the shared gather by index (`crate::ops::logits`); and of
//! its argmax on the shared sharded argmax (`crate::ops::draft`).

use kernel_core::row_copy::GatherByIndexBuffers;
use model_glm5next_contract::contract::DRAFT_ITEMS;
use model_glm5next_contract::own_ops::draft::DraftGatherBuffers;
use model_glm5next_contract::own_ops::rows::ResidualAddBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::Lowering;
use crate::buffers::Buffer;
use crate::ops::draft::{DraftViews, GatherPeer};
use crate::ops::{Lowered, one};
use crate::program::{Phase, draft_rows};

impl Lowering<'_> {
    /// The drafter rows' hidden inputs and tokens by the run's source codes:
    /// retained rows, or the normalized rows the previous program left in
    /// `Normed`: the target's final norm (`model.norm`, as retained rows
    /// are), or a previous draft step's `shared_head.norm`.
    pub(super) fn draft_gather(&self, phase: Phase) -> Result<Lowered, String> {
        let spans = DraftGatherBuffers {
            retained: self.buffers.state.retained()?,
            last: self.scratch(Buffer::Normed),
            codes: self.scratch(Buffer::DraftCodes),
            drafts: self.scratch(Buffer::Drafts),
            live_rows: self.inputs.live_tokens,
            output: self.scratch(Buffer::DraftSource),
            tokens: self.inputs.token_ids,
        };
        self.own_launch("the draft gather", || {
            self.ops.own.draft_gather(phase.rows(), spans)
        })
    }

    /// The draft rows of the head norm's output, which alone the head
    /// scores (`draft_rows`).
    pub(super) fn draft_head_rows(&self, phase: Phase) -> Result<Lowered, String> {
        let spans = GatherByIndexBuffers {
            source: self.scratch(Buffer::Normed),
            row_indices: self.scratch(Buffer::DraftRows),
            count: self.scratch(Buffer::DraftCount),
            destination: self.scratch(Buffer::HeadRows),
        };
        self.own_launch("the draft head rows", || {
            self.ops.logits.head_rows(draft_rows(phase), 0, spans)
        })
    }

    /// Each draft's token on the shared sharded argmax
    /// (`model_glm5next_contract::draft_argmax`): every rank's pair over
    /// its F32 logits shard, the gathers of every rank's scores and ids onto
    /// every rank, then the best of them written to the drafting item's
    /// latest draft and its verify row's input token. The run's draft rows
    /// (`draft_rows`) bound the grids; the draft rows table holds their run
    /// rows, then their drafting items, then their verify rows.
    pub(super) fn draft_argmax(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let drafter = self
            .buffers
            .drafter
            .as_ref()
            .ok_or("the drafter's argmax runs on an instance with the drafter")?;
        let words = u64::from(DRAFT_ITEMS) * 4;
        let part = |buffer: Buffer, at: u64| {
            let span = self.scratch(buffer);
            DeviceBufferSpanV1::new(span.buffer, span.offset + at * words, words)
        };
        let gathers = [0, 1].map(|which| {
            self.peers
                .iter()
                .map(|peer| {
                    let gather = peer.draft_gathers.ok_or("no drafter argmax gather state")?[which];
                    Ok(GatherPeer {
                        output: gather.output,
                        flags: gather.flags,
                        epoch: gather.epoch,
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        });
        let [scores, ids] = gathers;
        let views = DraftViews {
            logits: self.scratch(Buffer::Logits),
            live_width: drafter.vocab_width,
            scores: part(Buffer::DraftPairs, 0),
            ids: part(Buffer::DraftPairs, 1),
            gathers: [scores?, ids?],
            items: part(Buffer::DraftRows, 1),
            drafts: self.scratch(Buffer::Drafts),
            verify_rows: part(Buffer::DraftRows, 2),
            verify_tokens: self.verify_tokens,
            live_drafts: self.scratch(Buffer::DraftCount),
        };
        self.ops
            .draft
            .launches(self.rank, draft_rows(phase), &views)
            .map_err(|refusal| format!("drafter argmax: {refusal}"))
    }

    /// The drafter's single stream plus the sublayer's reduced output.
    pub(super) fn residual_add(&self, phase: Phase) -> Result<Lowered, String> {
        let spans = ResidualAddBuffers {
            x: self.scratch(Buffer::DraftHidden),
            y: self.scratch(Buffer::Reduced),
            live_rows: self.inputs.live_tokens,
        };
        self.own_launch("the drafter's residual add", || {
            self.ops.own.residual_add(phase.rows(), spans)
        })
    }

    /// The one launch of an own row op, `what` in its refusal.
    pub(super) fn own_launch(
        &self,
        what: &str,
        launch: impl FnOnce() -> Result<Vec<Lowered>, kernel_core::contract::Refusal>,
    ) -> Result<Lowered, String> {
        one(launch().map_err(|refusal| format!("{what}: {refusal}"))?)
    }
}
