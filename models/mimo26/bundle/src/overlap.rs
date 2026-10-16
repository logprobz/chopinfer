//! A prefill chunk pipelined over two token halves and two streams.
//!
//! Each layer has four row-parallel blocks per half: attention, the attention
//! all-reduce, the MLP and the MLP all-reduce. Compute runs on the rank's
//! stream and the all-reduces on a second stream, ordered with events:
//!
//! ```text
//! compute: A0  A1  M0  M1 | A0' A1' ...
//! comm:          R0  R1  S0  S1 | ...
//! ```
//!
//! `R0` (half 0's attention all-reduce) runs during `A1`, `R1` during `M0`,
//! `S0` during `M1`, and `S1` during the next layer's `A0`. The second half's
//! attention reads the first half's keys from the caches, which the compute
//! stream wrote earlier. The ring backup and the vocabulary head run once
//! for the whole chunk.
//!
//! The schedule is generic; the target owns the lanes' settings and the
//! grids a lane moves (`Target::lanes`, `Target::adjust`). A launch carries
//! the facts of the op that made it, which the target's grid choice reads:
//! its persistence as the op's `Launch` states it, and the phased pair, the
//! two collective launches (scatter, then gather) of a phased all-reduce
//! (`LaunchFacts::of_op`).

use crate::program::Stage;

pub use bundle_runtime::lanes::Config;

/// Splits a chunk of `tokens` near its middle on a 64-token (KV page)
/// boundary, the boundaries 1024-token chunks would have for 2048-token ones.
pub const fn split_point(tokens: u32) -> u32 {
    (tokens / 2).div_ceil(64) * 64
}

pub use bundle_runtime::lanes::LaunchFacts as Facts;
pub use bundle_runtime::lanes::{Lane, Mark, Op};

/// The two halves of a chunk.
pub const PARTS: usize = 2;

/// Per rank and half, the events that mark attention, the attention
/// all-reduce, the MLP and the MLP all-reduce done.
pub const EVENTS: usize = 4 * PARTS;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Group {
    Whole,
    HalfPrologue,
    Attention(u32),
    AttentionReduce(u32),
    Mlp(u32),
    MlpReduce(u32),
    Epilogue,
}

fn group(tp: crate::topology::Tp, stage: Stage) -> Group {
    match stage {
        Stage::RingBackup => Group::Whole,
        Stage::Embedding | Stage::MediaInject | Stage::InputNorm => Group::HalfPrologue,
        Stage::GatherHeadRows | Stage::LmHead | Stage::GatherLogits => Group::Epilogue,
        Stage::FinalReduceNorm => Group::MlpReduce(tp.geometry.layers() - 1),
        Stage::AttentionReduceNorm(layer) => Group::AttentionReduce(layer),
        Stage::MlpReduceNorm(layer) => Group::MlpReduce(layer),
        Stage::DenseGateUp(layer)
        | Stage::DenseDown(layer)
        | Stage::RouterProjection(layer)
        | Stage::RouterTop8(layer)
        | Stage::PermuteCount(layer)
        | Stage::PermuteScan(layer)
        | Stage::PermuteScatter(layer)
        | Stage::GroupedGateUp(layer)
        | Stage::GroupedDown(layer)
        | Stage::CombineExperts(layer) => Group::Mlp(layer),
        Stage::Qkv(layer)
        | Stage::RopeAndPagedWrite(layer)
        | Stage::RopeForSlidingWindow(layer)
        | Stage::AttentionPartial(layer)
        | Stage::AttentionMerge(layer)
        | Stage::SlidingWindowWrite(layer)
        | Stage::AttentionOutput(layer) => Group::Attention(layer),
    }
}

/// Orders a prefill program's `stages` for two halves and two lanes: per
/// layer, attention (compute), its all-reduce (comm), the MLP (compute)
/// and its all-reduce (comm), marked in that order.
pub fn schedule_for(tp: crate::topology::Tp, stages: &[Stage]) -> Vec<Op<Stage>> {
    let of = |wanted: Group| {
        stages
            .iter()
            .copied()
            .filter(|&stage| group(tp, stage) == wanted)
            .collect::<Vec<_>>()
    };
    let block = |lane, mark, group| bundle_runtime::lanes::Block {
        lane,
        mark,
        stages: of(group),
    };
    let program = bundle_runtime::lanes::Program {
        whole: of(Group::Whole),
        prologue: of(Group::HalfPrologue),
        blocks: (0..tp.geometry.layers())
            .flat_map(|layer| {
                [
                    block(Lane::Compute, 0, Group::Attention(layer)),
                    block(Lane::Comm, 1, Group::AttentionReduce(layer)),
                    block(Lane::Compute, 2, Group::Mlp(layer)),
                    block(Lane::Comm, 3, Group::MlpReduce(layer)),
                ]
            })
            .collect(),
        epilogue: of(Group::Epilogue),
    };
    debug_assert_eq!(program.events(PARTS), EVENTS);
    program.schedule(PARTS)
}

#[cfg(test)]
pub fn schedule(stages: &[Stage]) -> Vec<Op<Stage>> {
    schedule_for(crate::topology::Tp::TP4, stages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill_stages() -> Vec<Stage> {
        crate::program::stages()
            .into_iter()
            .filter(|stage| {
                !matches!(stage, Stage::AttentionMerge(layer)
                    if !crate::program::GLOBAL_LAYERS.contains(layer))
            })
            .collect()
    }

    #[test]
    fn every_stage_runs_once_per_half_or_once_whole() {
        let stages = prefill_stages();
        let ops = schedule(&stages);
        for &stage in &stages {
            let halves = ops
                .iter()
                .filter_map(|op| match op {
                    Op::Launch {
                        stage: launched,
                        part,
                        ..
                    } if *launched == stage => Some(*part),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let expected = match group(crate::topology::Tp::TP4, stage) {
                Group::Whole | Group::Epilogue => vec![None],
                _ => vec![Some(0), Some(1)],
            };
            assert_eq!(halves, expected, "{stage:?}");
        }
    }

    /// Replays the schedule with events as happens-before edges and checks
    /// that each half's blocks keep the unpipelined program's order.
    #[test]
    fn events_order_each_half_like_the_unpipelined_program() {
        let stages = prefill_stages();
        let ops = schedule(&stages);
        // Position of each launch in its lane, and the lane positions each
        // event's record and wait sit at.
        let mut lane_position = [0_usize; 2];
        let mut recorded: [Option<usize>; EVENTS] = [None; EVENTS];
        let mut waits_seen = Vec::new();
        let mut launches = Vec::new();
        for op in &ops {
            match *op {
                Op::Launch { stage, part, lane } => {
                    launches.push((stage, part, lane, lane_position[lane as usize]));
                    lane_position[lane as usize] += 1;
                }
                Op::Record(_, mark) => recorded[mark.index(PARTS)] = Some(launches.len()),
                Op::Wait(lane, mark) => {
                    assert!(
                        recorded[mark.index(PARTS)].is_some(),
                        "wait on {mark:?} before record"
                    );
                    waits_seen.push((lane, mark));
                }
            }
        }
        // All-reduces run on the comm lane, everything else on compute.
        for (stage, _, lane, _) in &launches {
            let comm = matches!(
                group(crate::topology::Tp::TP4, *stage),
                Group::AttentionReduce(_) | Group::MlpReduce(_)
            );
            assert_eq!(*lane == Lane::Comm, comm, "{stage:?}");
        }
        // Each lane change within one half is guarded by a wait.
        assert_eq!(
            waits_seen.len(),
            4 * 2 * crate::program::LAYERS as usize - 2 + 2
        );
        // The first comm op waits for half 0's attention, and the second
        // half's attention of layer 0 comes before half 0's MLP.
        let position = |wanted: Stage, half| {
            launches
                .iter()
                .position(|(stage, h, _, _)| *stage == wanted && *h == Some(half))
                .unwrap()
        };
        assert!(position(Stage::AttentionOutput(0), 1) < position(Stage::DenseGateUp(0), 0));
        assert!(position(Stage::AttentionReduceNorm(0), 0) < position(Stage::DenseGateUp(0), 0));
        assert_eq!(
            ops.iter().find(|op| matches!(op, Op::Wait(Lane::Comm, _))),
            Some(&Op::Wait(Lane::Comm, Mark { class: 0, part: 0 }))
        );
        assert_eq!(
            ops.last(),
            Some(&Op::Launch {
                stage: Stage::GatherLogits,
                part: None,
                lane: Lane::Compute
            })
        );
    }

    #[test]
    fn halves_split_on_page_boundaries() {
        assert_eq!(split_point(2048), 1024);
        assert_eq!(split_point(2045), 1024);
        assert_eq!(split_point(600), 320);
    }
}
