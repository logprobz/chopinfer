//! Lowering of the rows around the logits: the embedding fills every
//! residual stream and the router's scores are compacted, both through the
//! shared row copies (`crate::ops`), and an eager prefill gathers the rows
//! that need logits before the head. The head's F32
//! logits shards move to full-vocabulary rows through the instance's logits
//! moves (`crate::ops`): decode and verify rows to the rank that owns each
//! block, prefill rows to rank 0.

use model_glm5next_contract::own_ops::rows::GatherRowsBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::Lowering;
use crate::buffers::{Buffer, CommState};
use crate::ops::{CopyViews, Lowered, PeerViews};
use crate::program::Phase;

impl Lowering<'_> {
    /// The embedding into every stream of the first sublayer's input, over
    /// the phase's row bucket.
    pub(super) fn expand_streams(&self, phase: Phase) -> Result<Lowered, String> {
        let views = self.copy_views(Buffer::Reduced, Buffer::Streams0);
        let launches = self
            .ops
            .rows
            .expand_streams(self.row_bucket(phase)?, &views);
        self.one_copy(launches, "stream expansion")
    }

    /// The router's scores of every expert, from the padded rows its GEMM
    /// writes, over the phase's row bucket.
    pub(crate) fn compact_scores(&self, phase: Phase) -> Result<Lowered, String> {
        let views = self.copy_views(Buffer::RouterScores, Buffer::RouterLogits);
        let launches = self.ops.rows.router_scores(self.row_bucket(phase)?, &views);
        self.one_copy(launches, "score compaction")
    }

    fn copy_views(&self, source: Buffer, destination: Buffer) -> CopyViews {
        CopyViews {
            source: self.scratch(source),
            destination: self.scratch(destination),
            live_rows: self.inputs.live_tokens,
        }
    }

    fn one_copy(
        &self,
        launches: Result<Vec<Lowered>, kernel_core::contract::Refusal>,
        what: &str,
    ) -> Result<Lowered, String> {
        let launches =
            launches.map_err(|refusal| format!("rank {}'s {what}: {refusal}", self.rank))?;
        crate::ops::one(launches).map_err(|error| format!("rank {}'s {what}: {error}", self.rank))
    }

    /// The final norm's rows that need logits, in the prefill's row order:
    /// the head's rows gathered over the prefill's row bucket.
    pub(super) fn gather_head_rows(&self, phase: Phase) -> Result<Lowered, String> {
        let spans = GatherRowsBuffers {
            input: self.scratch(Buffer::Normed),
            row_indices: self.inputs.row_indices,
            count: self.inputs.logit_rows,
            output: self.scratch(Buffer::HeadRows),
        };
        self.own_launch("the head rows' gather", || {
            self.ops
                .own
                .gather_rows(self.head_rows, phase.rows(), spans)
        })
    }

    /// The head's logits shards into full-vocabulary rows: a decode or
    /// verify phase's rows exchanged to their block owners, a prefill's
    /// logits rows gathered to rank 0. A prefill chunk without logits rows
    /// moves nothing.
    pub(crate) fn gather_logits(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let local = self.scratch(Buffer::Logits);
        let moves = &self.ops.logits;
        let launches = if !matches!(phase, Phase::Prefill { .. }) {
            let peers = self.logits_peers(|peer| {
                (peer.decode_logits, peer.exchange.flags, peer.exchange.epoch)
            });
            moves.exchange(self.rank, self.head_rows, local, &peers)
        } else if self.head_rows == 0 {
            return Ok(Vec::new());
        } else {
            let peers = self
                .logits_peers(|peer| (peer.prefill_logits, peer.gather_flags, peer.gather_epoch));
            moves.gather(self.rank, self.head_rows, local, &peers)
        };
        launches.map_err(|refusal| format!("rank {}'s logits move: {refusal}", self.rank))
    }

    /// Every rank's views of a logits move: its full rows, flags and epoch.
    fn logits_peers(&self, views: impl Fn(&CommState) -> (Span, Span, Span)) -> Vec<PeerViews> {
        self.peers
            .iter()
            .map(|peer| {
                let (output, flags, epoch) = views(peer);
                PeerViews {
                    output,
                    flags,
                    epoch,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::dispatch;
    use kernel_core::latent_attention::LatentKvFormat;
    use runtime_bundle_abi::KernelArgumentV1;

    use super::*;
    use crate::buffers::{SCORED_ROWS, head_rows};
    use crate::row_ops_tests::Rig;

    /// The logits moves on every rank at their row limits: a decode or
    /// verify phase's exchange of `head_rows()` rows and a prefill's gather
    /// of `SCORED_ROWS` rows to rank 0 are one collective launch each,
    /// binding the rank's logits shards, every rank's full rows and flags
    /// and the rank's epoch. With no rows a prefill moves nothing and an
    /// exchange is refused; one row past the most is refused.
    #[test]
    fn logits_moves_launch_at_their_row_limits_on_every_rank() {
        let rig = Rig::new(LatentKvFormat::Fp8E4M3Block128, false);
        for (phase, most) in [
            (Phase::Decode(32), head_rows()),
            (Phase::Verify { rows: 32, items: 8 }, head_rows()),
            (Phase::Prefill { rows: 2048 }, SCORED_ROWS),
        ] {
            let prefill = matches!(phase, Phase::Prefill { .. });
            for rank in 0..rig.shape.ranks() {
                let lowering = |head_rows| Lowering {
                    head_rows,
                    ..rig.lowering(rank, phase, None)
                };
                let what = |rows: u32| format!("rank {rank} {phase:?} {rows} rows");
                let none = lowering(0).gather_logits(phase);
                if prefill {
                    assert!(none.unwrap().is_empty(), "{}", what(0));
                } else {
                    assert!(none.is_err(), "{}", what(0));
                }
                assert!(
                    lowering(most + 1).gather_logits(phase).is_err(),
                    "{}",
                    what(most + 1)
                );
                let at_most = lowering(most);
                let launches = at_most.gather_logits(phase).unwrap();
                let [moved] = launches.as_slice() else {
                    panic!("{}: {} launches", what(most), launches.len());
                };
                assert!(moved.collective, "{}", what(most));
                assert_eq!(moved.launch.rows, most, "{}", what(most));
                let views = |view: fn(&CommState) -> Span| {
                    KernelArgumentV1::DevicePointerArray(
                        rig.peers.iter().map(|peer| Some(view(peer))).collect(),
                    )
                };
                let own = &rig.peers[rank as usize];
                let (outputs, flags, epoch) = if prefill {
                    (
                        views(|peer| peer.prefill_logits),
                        views(|peer| peer.gather_flags),
                        own.gather_epoch,
                    )
                } else {
                    (
                        views(|peer| peer.decode_logits),
                        views(|peer| peer.exchange.flags),
                        own.exchange.epoch,
                    )
                };
                let arguments =
                    dispatch::arguments(&moved.launch.arguments, &moved.launch.bindings).unwrap();
                assert_eq!(
                    arguments[0],
                    KernelArgumentV1::DeviceBuffer(at_most.scratch(Buffer::Logits)),
                    "{}",
                    what(most)
                );
                for expected in [
                    outputs,
                    flags,
                    KernelArgumentV1::DeviceBuffer(epoch),
                    KernelArgumentV1::ScalarU32(most),
                ] {
                    assert!(
                        arguments.contains(&expected),
                        "{}: {expected:?} in {arguments:?}",
                        what(most)
                    );
                }
            }
        }
    }
}
