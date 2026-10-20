//! Lowering of the rows around the logits: a prefill gathers its F32
//! final-norm rows that need logits into the head's input, the head
//! projects the rank's vocabulary shard (`crate::ops::head`), and its F32
//! logits shards move to full-vocabulary rows through the logits ops
//! (`crate::ops::logits`): decode and verify rows to the rank that owns
//! each block, prefill rows to rank 0.

use kernel_core::row_copy::GatherByIndexBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::Lowering;
use crate::buffers::{LogitsComm, Scratch, rows};
use crate::ops::Lowered;
use crate::ops::head::HeadViews;
use crate::ops::logits::PeerViews;
use crate::program::Phase;

impl Lowering<'_> {
    /// The final norm's rows that need logits, in the prefill's row order,
    /// gathered over the prefill's rows into the head's input.
    pub(super) fn gather_head_rows(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.logits else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let spans = GatherByIndexBuffers {
            source: rows(scratch.normed_wide, phase.rows()),
            row_indices: self.inputs.row_indices,
            count: self.inputs.logit_rows,
            destination: scratch.head_input,
        };
        ops.head_rows(self.head_rows, phase.rows(), spans)
            .map(Some)
            .map_err(|refusal| format!("rank {}'s head rows: {refusal}", self.rank))
    }

    /// The head over the rank's vocabulary shard: a decode or verify
    /// phase's final-norm rows, or a prefill's gathered head rows (none
    /// without logits rows).
    pub(super) fn lm_head(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.head else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let (input, rows, live_rows) = if phase.prefill() {
            if self.head_rows == 0 {
                return Ok(Some(Vec::new()));
            }
            (scratch.head_input, self.head_rows, self.inputs.logit_rows)
        } else {
            (
                rows(scratch.normed_wide, phase.rows()),
                phase.rows(),
                self.inputs.live_tokens,
            )
        };
        let views = HeadViews {
            input,
            weight: self.buffers.view("head.weight")?,
            logits: scratch.head,
            live_rows,
            state: crate::ops::exact_f32::state(self.buffers, "head.weight"),
        };
        ops.head(rows, &views)
            .map(Some)
            .map_err(|refusal| format!("rank {}'s LM head: {refusal}", self.rank))
    }

    /// The head's logits shards into full-vocabulary rows: a decode or
    /// verify phase's rows exchanged to their block owners, a prefill's
    /// logits rows gathered to rank 0. A prefill chunk without logits rows
    /// moves nothing.
    pub(super) fn gather_logits(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.logits else {
            return Ok(None);
        };
        let local = self.buffers.scratch.head;
        let launches = if !phase.prefill() {
            let peers = self.logits_peers(|scratch, comm| {
                (
                    scratch.decode_logits,
                    comm.exchange_flags,
                    comm.exchange_epoch,
                )
            })?;
            ops.exchange(self.rank, self.head_rows, local, &peers)
        } else if self.head_rows == 0 {
            return Ok(Some(Vec::new()));
        } else {
            let peers = self.logits_peers(|scratch, comm| {
                (scratch.prefill_logits, comm.gather_flags, comm.gather_epoch)
            })?;
            ops.gather(self.rank, self.head_rows, local, &peers)
        };
        launches
            .map(Some)
            .map_err(|refusal| format!("rank {}'s logits move: {refusal}", self.rank))
    }

    /// Every rank's views of a logits move: its full rows, flags and epoch.
    fn logits_peers(
        &self,
        views: impl Fn(&Scratch<Span>, &LogitsComm) -> (Span, Span, Span),
    ) -> Result<Vec<PeerViews>, String> {
        self.peers
            .iter()
            .map(|peer| {
                let comm = peer
                    .logits
                    .as_ref()
                    .ok_or("a rank lacks the logits moves' comm state")?;
                let (output, flags, epoch) = views(&peer.scratch, comm);
                Ok(PeerViews {
                    output,
                    flags,
                    epoch,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::dispatch;
    use bundle_runtime::geometry::{Items, Ranks};
    use bundle_runtime::state::Extent;
    use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::serving::{DECODE_BATCH, SCORED_ROWS};
    use model_deepseek41_contract::state::KvStorage;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    use super::*;
    use crate::buffers::{RankBuffers, head_rows};
    use crate::lower::KvExtent;
    use crate::ops::Ops;
    use crate::shape::Shape;

    /// The rows of the prefill the tests lower.
    const PREFILL_ROWS: u32 = 2048;

    struct Rig {
        shape: Shape,
        ops: Ops,
        peers: Vec<RankBuffers>,
        inputs: DeviceInputs,
    }

    impl Rig {
        fn new() -> Rig {
            let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops = Ops::prepare(&kernels, &shape, KvStorage::Native);
            assert!(ops.logits.is_some(), "{:?}", ops.missing);
            // Each rank's logits comm state, of the sizes the ops state.
            let state = ops.logits.as_ref().unwrap().state();
            let peers = (0..shape.ranks())
                .map(|rank| {
                    let span = |region: usize| {
                        Span::new(
                            BundleDeviceBufferHandleV1::from_parts(1000 + region as u32, rank + 1),
                            0,
                            state[region],
                        )
                    };
                    RankBuffers {
                        logits: Some(LogitsComm {
                            exchange_flags: span(0),
                            exchange_epoch: span(1),
                            gather_flags: span(2),
                            gather_epoch: span(3),
                        }),
                        ..crate::lower::tests::buffers(&shape, rank)
                    }
                })
                .collect();
            let arena = Span::new(BundleDeviceBufferHandleV1::from_parts(1, 999), 0, 1 << 20);
            let inputs = DeviceInputs::within(
                PhaseArena {
                    arena,
                    block_tables: arena,
                },
                &InputLayout::within(PREFILL_ROWS, SCORED_ROWS, Items::new(DECODE_BATCH).unwrap()),
            );
            Rig {
                shape,
                ops,
                peers,
                inputs,
            }
        }

        fn lowering(&self, rank: u32, head_rows: u32) -> Lowering<'_> {
            Lowering {
                prefill_row_origin: None,
                shape: &self.shape,
                ops: &self.ops,
                buffers: &self.peers[rank as usize],
                peers: &self.peers,
                inputs: &self.inputs,
                rank,
                kv: KvExtent {
                    storage: KvStorage::Native,
                    pages: 1,
                    extent: Extent {
                        served: 1,
                        bucket: 1,
                        verify_rows: 0,
                    },
                    sequences: 1,
                    max_pages: 1,
                },
                head_rows,
                prefill_hc_fusion: true,
            }
        }
    }

    /// The logits moves on every rank at their row limits: a decode or
    /// verify phase's exchange of `head_rows()` rows and a prefill's gather
    /// of `SCORED_ROWS` rows to rank 0 are one collective launch each,
    /// binding the rank's logits shards, every rank's full rows and flags
    /// and the rank's epoch. With no rows a prefill moves nothing and an
    /// exchange is refused; one row past the most is refused.
    #[test]
    fn logits_moves_launch_at_their_row_limits_on_every_rank() {
        let rig = Rig::new();
        for (phase, most) in [
            (Phase::Decode(DECODE_BATCH), head_rows()),
            (
                Phase::Verify {
                    rows: head_rows(),
                    items: DECODE_BATCH,
                },
                head_rows(),
            ),
            (Phase::Prefill { rows: PREFILL_ROWS }, SCORED_ROWS),
        ] {
            for rank in 0..rig.shape.ranks() {
                let what = |rows: u32| format!("rank {rank} {phase:?} {rows} rows");
                let none = rig.lowering(rank, 0).gather_logits(phase);
                if phase.prefill() {
                    assert!(none.unwrap().unwrap().is_empty(), "{}", what(0));
                } else {
                    assert!(none.is_err(), "{}", what(0));
                }
                assert!(
                    rig.lowering(rank, most + 1).gather_logits(phase).is_err(),
                    "{}",
                    what(most + 1)
                );
                let launches = rig
                    .lowering(rank, most)
                    .gather_logits(phase)
                    .unwrap()
                    .unwrap();
                let [moved] = launches.as_slice() else {
                    panic!("{}: {} launches", what(most), launches.len());
                };
                assert!(moved.collective, "{}", what(most));
                assert_eq!(moved.launch.rows, most, "{}", what(most));
                let views = |view: fn(&RankBuffers) -> Span| {
                    KernelArgumentV1::DevicePointerArray(
                        rig.peers.iter().map(|peer| Some(view(peer))).collect(),
                    )
                };
                let own = &rig.peers[rank as usize];
                let comm = own.logits.unwrap();
                let (outputs, flags, epoch) = if phase.prefill() {
                    (
                        views(|peer| peer.scratch.prefill_logits),
                        views(|peer| peer.logits.unwrap().gather_flags),
                        comm.gather_epoch,
                    )
                } else {
                    (
                        views(|peer| peer.scratch.decode_logits),
                        views(|peer| peer.logits.unwrap().exchange_flags),
                        comm.exchange_epoch,
                    )
                };
                let arguments =
                    dispatch::arguments(&moved.launch.arguments, &moved.launch.bindings).unwrap();
                assert_eq!(
                    arguments[0],
                    KernelArgumentV1::DeviceBuffer(own.scratch.head),
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

    /// A prefill's head rows are one gather over its rows of the final
    /// norm's F32 output, by the uploaded indices and count, into the
    /// head's input; more rows than the prefill runs are refused.
    #[test]
    fn head_rows_gather_the_scored_rows_of_a_prefill() {
        let rig = Rig::new();
        let phase = Phase::Prefill { rows: PREFILL_ROWS };
        let lowering = rig.lowering(0, SCORED_ROWS);
        let launches = lowering.gather_head_rows(phase).unwrap().unwrap();
        let [gather] = launches.as_slice() else {
            panic!("{} launches", launches.len());
        };
        assert_eq!(gather.launch.rows, PREFILL_ROWS);
        let arguments =
            dispatch::arguments(&gather.launch.arguments, &gather.launch.bindings).unwrap();
        let scratch = &rig.peers[0].scratch;
        for expected in [
            rows(scratch.normed_wide, phase.rows()),
            rig.inputs.row_indices,
            rig.inputs.logit_rows,
            scratch.head_input,
        ] {
            assert!(
                arguments.contains(&KernelArgumentV1::DeviceBuffer(expected)),
                "{expected:?} in {arguments:?}"
            );
        }
        assert!(
            rig.lowering(0, PREFILL_ROWS + 1)
                .gather_head_rows(phase)
                .is_err()
        );
    }
}
