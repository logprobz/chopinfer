//! Overlap a rounded reduction with the other half of attention.o-b.
//! The full-chunk quantizer and all KV/compressor updates retain their order.
use super::{Run, Stage};
use crate::instance::Instance;
use crate::lower::Lowering;
use crate::program::Sublayer;
use bundle_runtime::acquire::RankHandles;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::lanes::{self, Block, Lane, Op, SideLane};
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch, check_collectives};
use bundle_runtime::resources::{Held, allocate_span, upload_bytes};
use bundle_runtime::upload::DeviceInputs;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1, HostServicesV1};
use std::ops::Range;

/// Keep the sequential default until the paired GPU measurements qualify it.
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    return IN_TESTS.get();
    #[cfg(not(test))]
    {
        std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP").as_deref() == Ok("1")
    }
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(enabled: bool, create: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(IN_TESTS.replace(enabled));
    create()
}

pub(crate) struct Rank {
    lane: SideLane,
    counts: DeviceBufferSpanV1,
}

pub(crate) fn acquire(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<Rank>, BundleAbiError> {
    ranks
        .iter()
        .map(|&(context, pool, stream)| {
            Ok(Rank {
                lane: SideLane::create(services, context, 4, held)?,
                counts: allocate_span(services, pool, stream, 8, held)?,
            })
        })
        .collect()
}

fn split(rows: u32) -> Option<[u32; 2]> {
    (rows >= 1024).then(|| {
        let first = (rows / 2) / 16 * 16;
        [first, rows - first]
    })
}

#[derive(Clone, Copy)]
enum Work {
    Project,
    Reduce,
}

fn schedule() -> Vec<Op<Work>> {
    lanes::Program {
        whole: vec![],
        prologue: vec![],
        epilogue: vec![],
        blocks: vec![
            Block {
                lane: Lane::Compute,
                mark: 0,
                stages: vec![Work::Project],
            },
            Block {
                lane: Lane::Comm,
                mark: 1,
                stages: vec![Work::Reduce],
            },
        ],
    }
    .schedule(2)
}

struct Part {
    project: Vec<PreparedLaunch>,
    reduce: Vec<PreparedLaunch>,
    flags: Vec<bool>,
}

struct Section {
    before: Vec<Range<usize>>,
    parts: Vec<[Part; 2]>,
}

pub(super) struct Plan {
    rows: [u32; 2],
    sections: Vec<Section>,
    tail: Vec<Range<usize>>,
}

impl Plan {
    pub(super) fn prepare(instance: &Instance, run: &Run) -> Result<Option<Self>, String> {
        let Some(rows) = split(run.phase.rows()).filter(|_| run.phase.prefill()) else {
            return Ok(None);
        };
        if instance.overlap.len() != instance.ranks.len() || instance.overlap.is_empty() {
            return Ok(None);
        }
        #[cfg(feature = "diagnostic-taps")]
        if instance.taps.text() {
            return Ok(None);
        }
        check_collectives(&run.programs, &run.collectives)?;
        let mut ends = vec![0; instance.ranks.len()];
        let mut sections = Vec::new();
        for layer in 0..instance.config.shape.geometry.layers() {
            let mut before = Vec::new();
            let mut parts: Vec<[Part; 2]> = Vec::new();
            for (rank, buffers) in instance.buffers.iter().enumerate() {
                let stages = &run.stages[rank];
                let at = stages
                    .iter()
                    .enumerate()
                    .filter_map(|(at, stage)| {
                        (*stage == Stage::Project(layer, "attention.o-b")).then_some(at)
                    })
                    .collect::<Vec<_>>();
                // Retain the full quantizer, then split the prepared producer
                // (including TMA admission) and the composed rounded reduce.
                if at.len() < 2 || at.windows(2).any(|p| p[1] != p[0] + 1) || at[1] < ends[rank] {
                    return Ok(None);
                }
                let reduce_start = at[at.len() - 1] + 1;
                let reduce_end = reduce_start
                    + stages[reduce_start..]
                        .iter()
                        .take_while(|&&s| s == Stage::Reduce(layer, Sublayer::Attention))
                        .count();
                if reduce_end == reduce_start {
                    return Ok(None);
                }
                let inputs = DeviceInputs::within(instance.arenas[run.arena][rank], &run.layout);
                let lowering = Lowering {
                    shape: &instance.config.shape,
                    ops: &instance.ops,
                    buffers,
                    peers: &instance.buffers,
                    inputs: &inputs,
                    rank: rank as u32,
                    kv: instance.kv_extent(),
                    head_rows: run.head_rows,
                };
                let mut rank_parts = Vec::new();
                for (part, first) in [0, rows[0]].into_iter().enumerate() {
                    let counts = instance.overlap[rank].counts;
                    let live =
                        DeviceBufferSpanV1::new(counts.buffer, counts.offset + part as u64 * 4, 4);
                    let (project, reduce) =
                        match lowering.output_b_part(layer, first, rows[part], live) {
                            Ok(value) => value,
                            Err(_) => return Ok(None),
                        };
                    if project
                        .iter()
                        .map(|p| &p.launch.symbol)
                        .ne(run.programs[rank][at[1]..reduce_start]
                            .iter()
                            .map(|p| &p.symbol))
                        || reduce
                            .iter()
                            .map(|p| &p.launch.symbol)
                            .ne(run.programs[rank][reduce_start..reduce_end]
                                .iter()
                                .map(|p| &p.symbol))
                    {
                        return Ok(None);
                    }
                    rank_parts.push(Part {
                        project: project.into_iter().map(|p| p.launch).collect(),
                        flags: reduce.iter().map(|p| p.collective).collect(),
                        reduce: reduce.into_iter().map(|p| p.launch).collect(),
                    });
                }
                before.push(ends[rank]..at[1]);
                ends[rank] = reduce_end;
                crate::shared_overlap::disjoint_branches(
                    &rank_parts[0].reduce,
                    &rank_parts[1].project,
                )?;
                parts.push(rank_parts.try_into().ok().expect("two parts"));
            }
            for part in 0..2 {
                let calls = parts
                    .iter()
                    .map(|p| p[part].reduce.as_slice())
                    .collect::<Vec<_>>();
                let flags = parts
                    .iter()
                    .map(|p| p[part].flags.as_slice())
                    .collect::<Vec<_>>();
                check_collectives(&calls, &flags)?;
            }
            sections.push(Section { before, parts });
        }
        let tail = ends
            .into_iter()
            .zip(&run.programs)
            .map(|(first, p)| first..p.len())
            .collect();
        Ok(Some(Self {
            rows,
            sections,
            tail,
        }))
    }

    pub(super) fn execute(&self, instance: &Instance, run: &Run) -> Result<(), DispatchError> {
        let main = instance.ranks.iter().map(|r| r.2).collect::<Vec<_>>();
        let comm = instance
            .overlap
            .iter()
            .map(|r| r.lane.stream)
            .collect::<Vec<_>>();
        let normal = |ranges: &[Range<usize>]| {
            let programs = ranges
                .iter()
                .enumerate()
                .map(|(r, range)| &run.programs[r][range.clone()])
                .collect::<Vec<_>>();
            let flags = ranges
                .iter()
                .enumerate()
                .map(|(r, range)| &run.collectives[r][range.clone()])
                .collect::<Vec<_>>();
            instance.launch_eager(&main, &programs, &flags)
        };
        let _unwind = PoisonOnUnwind(&instance.collectives);
        let result = (|| {
            let bytes = self
                .rows
                .iter()
                .flat_map(|rows| rows.to_le_bytes())
                .collect::<Vec<_>>();
            for (rank, state) in instance.overlap.iter().enumerate() {
                upload_bytes(&instance.services, main[rank], state.counts, &bytes)?;
            }
            for section in &self.sections {
                normal(&section.before)?;
                for op in schedule() {
                    match op {
                        Op::Launch { stage, part, lane } => {
                            let part = part.expect("split blocks have parts");
                            let programs = section
                                .parts
                                .iter()
                                .map(|p| match stage {
                                    Work::Project => p[part].project.as_slice(),
                                    Work::Reduce => p[part].reduce.as_slice(),
                                })
                                .collect::<Vec<_>>();
                            let flags = section
                                .parts
                                .iter()
                                .map(|p| match stage {
                                    Work::Project => vec![false; p[part].project.len()],
                                    Work::Reduce => p[part].flags.clone(),
                                })
                                .collect::<Vec<_>>();
                            instance.launch_eager(
                                if lane == Lane::Compute { &main } else { &comm },
                                &programs,
                                &flags,
                            )?;
                        }
                        Op::Record(lane, mark) | Op::Wait(lane, mark) => {
                            for (rank, state) in instance.overlap.iter().enumerate() {
                                let stream = if lane == Lane::Compute {
                                    main[rank]
                                } else {
                                    comm[rank]
                                };
                                let event = state.lane.events[mark.index(2)];
                                match op {
                                    Op::Record(..) => {
                                        instance.services.record_event(event, stream)?
                                    }
                                    Op::Wait(..) => instance.services.wait_event(stream, event)?,
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                }
            }
            normal(&self.tail)
        })();
        // A failed event or partial rank submission may leave a peer
        // waiting forever. Poison instead of draining through missing events.
        if result.is_err() {
            instance.collectives.poison();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_keeps_tiles_aligned_and_covers_every_row() {
        assert_eq!(split(1023), None);
        for rows in 1024..=8192 {
            let [a, b] = split(rows).unwrap();
            assert_eq!(a % 16, 0);
            assert!(a > 0 && b > 0);
            assert_eq!(a + b, rows);
        }
    }
}
