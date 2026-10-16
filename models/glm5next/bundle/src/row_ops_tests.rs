//! The row family's prepared ops (RMSNorms, the embedding gather) on every
//! rank's buffers of one KV format: they refuse what an instance did not
//! prepare them for.

use bundle_runtime::fake_host::FakeDevice;
use bundle_runtime::kv::KvFormat;
use bundle_runtime::upload::DeviceInputs;
use kernel_core::latent_attention::LatentKvFormat;
use model_glm5next_contract::contract::{Shape, has_mtp_layer, latent_format};

use crate::buffers::{CommState, RankBuffers};
use crate::journal_tests::{head_rows, inputs, rank_buffers};
use crate::lower::Lowering;
use crate::ops::Ops;
use crate::program::{DraftPass, Phase, Stage};

/// One phase of each kind, for the stages of every program.
pub(crate) const KINDS: [Phase; 3] = [
    Phase::Prefill { rows: 2048 },
    Phase::Decode(8),
    Phase::Verify { rows: 32, items: 8 },
];

/// Every rank's buffers over one KV format, with the ops an instance of
/// them prepares and each rank's inputs of each phase kind.
pub(crate) struct Rig {
    pub shape: Shape,
    pub format: KvFormat,
    pub ops: Ops,
    pub ranks: Vec<RankBuffers>,
    pub peers: Vec<CommState>,
    /// Each rank's inputs of `KINDS`, in rank order.
    pub inputs: Vec<[DeviceInputs; 3]>,
}

impl Rig {
    /// The rig of the KV format `kv` that holds the drafter's layer when
    /// `drafter`, over its own fake device.
    pub(crate) fn new(kv: LatentKvFormat, drafter: bool) -> Rig {
        let shape = Shape::TP4;
        let format = shape
            .kv_formats()
            .into_iter()
            .find(|format| latent_format(format) == kv && has_mtp_layer(format) == drafter)
            .unwrap();
        let device = FakeDevice::default();
        let ranks = (0..shape.ranks())
            .map(|rank| rank_buffers(&device, rank, &format))
            .collect::<Vec<_>>();
        let peers = ranks.iter().map(|buffers| buffers.comm.clone()).collect();
        Rig {
            shape,
            format,
            ops: Ops::for_tests(shape, 1),
            ranks,
            peers,
            inputs: (0..shape.ranks())
                .map(|rank| KINDS.map(|kind| inputs(rank, kind)))
                .collect(),
        }
    }

    /// Rank `rank`'s inputs of `phase`'s kind.
    pub(crate) fn inputs_of(&self, rank: u32, phase: Phase) -> &DeviceInputs {
        let kind = match phase {
            Phase::Prefill { .. } => 0,
            Phase::Decode(_) => 1,
            Phase::Verify { .. } => 2,
        };
        &self.inputs[rank as usize][kind]
    }

    /// Rank `rank`'s lowering of `phase`, in drafter pass `draft` if any.
    pub(crate) fn lowering(
        &self,
        rank: u32,
        phase: Phase,
        draft: Option<DraftPass>,
    ) -> Lowering<'_> {
        Lowering {
            shape: self.shape,
            format: &self.format,
            buffers: &self.ranks[rank as usize],
            inputs: self.inputs_of(rank, phase),
            ops: &self.ops,
            rank,
            index_topk: self.shape.profile.index_topk_max,
            pieces: &[],
            row_offset: 0,
            peers: &self.peers,
            head_rows: head_rows(phase),
            context_parallel: None,
            draft,
            verify_tokens: draft.map(|_| self.inputs_of(rank, KINDS[2]).token_ids),
            projections: None,
        }
    }
}

/// The prepared norms refuse a width the program does not normalize, and
/// each rank's embedding refuses another rank.
#[test]
fn row_ops_refuse_what_they_were_not_prepared_for() {
    let rig = Rig::new(LatentKvFormat::Fp8E4M3Block128, false);
    let phase = Phase::Decode(8);
    let lowering = rig.lowering(0, phase, None);
    assert!(lowering.norm(phase, Stage::Embedding).is_err());
    let beyond = Lowering {
        rank: rig.shape.ranks(),
        ..rig.lowering(0, phase, None)
    };
    assert!(beyond.embedding(phase).is_err());
}
