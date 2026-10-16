//! The MTP drafter's argmax over the head's vocabulary shards
//! (`model_glm5next_contract::draft_argmax`): each rank's shared F32
//! partial, the all-gathers of the pairs' scores and ids, and the shared
//! final scattered into the drafts and the verify input tokens.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews};
use kernel_core::contract::Refusal;
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::draft_argmax::{self, DraftArgmax, DraftArgmaxBuffers};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered};

/// Each rank's draft argmax, in rank order, and the comm state regions of
/// each of its pair gathers.
pub(crate) struct DraftOps {
    ranks: Vec<Prepared<DraftArgmaxBuffers<View>>>,
    regions: [u64; 2],
}

/// One rank's views of a pair gather that every rank's launch reaches: the
/// rank's gathered pairs and its flags and epoch.
#[derive(Clone, Copy)]
pub(crate) struct GatherPeer {
    pub output: DeviceBufferSpanV1,
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

/// A step's spans of the draft argmax on one rank: its logits shard rows
/// and their live width, its pairs' scores and ids, every rank's views of
/// the scores' and the ids' gathers in rank order, the drafting items and
/// drafts, the verify rows and the optional verify tokens, and the live
/// draft count.
pub(crate) struct DraftViews {
    pub logits: DeviceBufferSpanV1,
    pub live_width: DeviceBufferSpanV1,
    pub scores: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub gathers: [Vec<GatherPeer>; 2],
    pub items: DeviceBufferSpanV1,
    pub drafts: DeviceBufferSpanV1,
    pub verify_rows: DeviceBufferSpanV1,
    pub verify_tokens: Option<DeviceBufferSpanV1>,
    pub live_drafts: DeviceBufferSpanV1,
}

impl DraftOps {
    /// Every rank's draft argmax of `shape` on `backend`, each asked
    /// through `asks`, refused unless every rank's two pair gathers keep
    /// the same flag and epoch regions. The drafts only choose what the
    /// target's verify accepts or rejects, so the argmax asks for its rows
    /// alone.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        asks: &Asks,
    ) -> Result<Self, Refused> {
        let ranks = (0..shape.ranks())
            .map(|rank| {
                asks.ask(
                    format!("draft.argmax rank {rank}"),
                    Need::rows(draft_argmax::CAPACITY),
                    |need| {
                        DraftArgmax::prepare_for(backend, shape, rank, need)
                            .map(|op| Box::new(op) as Prepared<DraftArgmaxBuffers<View>>)
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let refused = || {
            Refused::Rule(
                "GLM 5.3 Flash's draft argmax keeps other comm state regions than it binds".into(),
            )
        };
        let regions = match ranks.first().map(|op| op.info().state.as_slice()) {
            Some(&[flags, epoch, ..]) => [flags, epoch],
            _ => return Err(refused()),
        };
        if ranks
            .iter()
            .any(|op| op.info().state[..] != [regions, regions].concat())
        {
            return Err(refused());
        }
        Ok(Self { ranks, regions })
    }

    /// The comm state regions of each pair gather, the scores' and the
    /// ids' alike: a flag and an epoch region of the bytes its op keeps.
    pub(crate) fn regions(&self) -> [u64; 2] {
        self.regions
    }

    /// The launches of rank `rank`'s draft argmax over `rows` draft rows.
    pub(crate) fn launches(
        &self,
        rank: u32,
        rows: u32,
        views: &DraftViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .ranks
            .get(rank as usize)
            .ok_or(Refusal("GLM 5.3 Flash has no draft argmax for the rank"))?;
        let mut bind = Bind::default();
        let mut gather = |input, peers: &[GatherPeer]| GatherBuffers {
            input: bind.view(input),
            ranks: peers
                .iter()
                .map(|peer| RankViews {
                    output: bind.view(peer.output),
                    state: vec![bind.view(peer.flags), bind.view(peer.epoch)],
                })
                .collect(),
        };
        let scores = gather(views.scores, &views.gathers[0]);
        let ids = gather(views.ids, &views.gathers[1]);
        let buffers = DraftArgmaxBuffers {
            logits: bind.view(views.logits),
            live_width: bind.view(views.live_width),
            scores,
            ids,
            items: bind.view(views.items),
            drafts: bind.view(views.drafts),
            verify_rows: bind.view(views.verify_rows),
            verify_tokens: views.verify_tokens.map(|span| bind.view(span)),
            live_drafts: bind.view(views.live_drafts),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }
}
