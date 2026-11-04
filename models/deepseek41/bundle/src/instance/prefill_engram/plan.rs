//! Split eager programs at the same layer-consumer boundaries as DSpark.
use super::Gathers;
use crate::instance::Instance;
use crate::instance::execute::Executor;
use crate::program::Stage;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::{PreparedLaunch, check_collectives};
use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::instance) struct Plan {
    ranges: Vec<Vec<Range<usize>>>,
    pub(in crate::instance) dma: Vec<Vec<DmaPiece>>,
}

/// One original compute interval, split only at an Engram consumer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::instance) struct DmaPiece {
    pub spans: Vec<Range<usize>>,
    /// Join before this piece, on every target stream.
    pub join: Option<usize>,
}

impl Plan {
    pub(in crate::instance) fn new(
        gathers: &Gathers,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
        stages: &[Vec<Stage>],
    ) -> Result<Self, String> {
        if !gathers.idle()
            || programs.is_empty()
            || programs.len() != flags.len()
            || programs.len() != stages.len()
        {
            return Err("prefill Engram program extent or lifecycle".into());
        }
        let mut ranges = Vec::new();
        for (rank, program) in programs.iter().enumerate() {
            if program.len() != flags[rank].len() || program.len() != stages[rank].len() {
                return Err("prefill Engram stage extent".into());
            }
            let mut start = 0;
            let mut sections = Vec::new();
            for (index, layer) in gathers.layers.iter().enumerate() {
                let mut matches = stages[rank].iter().enumerate().filter_map(|(at, stage)| {
                    (*stage == Stage::EngramRows(layer.layer)).then_some(at)
                });
                let end = matches.next().ok_or("prefill Engram consumer absent")?;
                if end < start || matches.next().is_some() {
                    return Err("prefill Engram consumer repeated or reordered".into());
                }
                let pending: Vec<_> = gathers.layers[index..]
                    .iter()
                    .flat_map(|l| std::iter::once(l.indices).chain(l.destinations.iter().copied()))
                    .collect();
                crate::shared_overlap::disjoint_copy_spans(&pending, &program[..end])?;
                sections.push(start..end);
                start = end;
            }
            if !gathers.verify.is_empty() {
                crate::shared_overlap::disjoint_copy_spans(&gathers.verify, &program[start..])?;
            }
            sections.push(start..program.len());
            ranges.push(sections);
        }
        let plan = Self {
            ranges,
            dma: Vec::new(),
        };
        plan.check_sections(programs, flags)?;
        Ok(plan)
    }

    /// Each section's collectives match across the ranks.
    fn check_sections(
        &self,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
    ) -> Result<(), String> {
        for piece in 0..self.ranges.first().map_or(0, Vec::len) {
            let ops: Vec<_> = programs
                .iter()
                .enumerate()
                .map(|(rank, p)| &p[self.ranges[rank][piece].clone()])
                .collect();
            let waits: Vec<_> = flags
                .iter()
                .enumerate()
                .map(|(rank, p)| &p[self.ranges[rank][piece].clone()])
                .collect();
            check_collectives(&ops, &waits)
                .map_err(|e| format!("prefill Engram collective boundary: {e:?}"))?;
        }
        Ok(())
    }

    /// This plan for `programs`, which issue the stages and flags of the
    /// programs it was made for and differ from them only in each rank's
    /// launches `changed`: `new`'s checks, with the pending-span audit of
    /// the changed launches alone. `new` audits each launch on its own
    /// (`audit`), so the unchanged launches pass as they passed before.
    pub(in crate::instance) fn reuse(
        &self,
        gathers: &Gathers,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
        changed: impl IntoIterator<Item = (usize, Range<usize>)>,
    ) -> Result<Self, String> {
        let extent =
            self.ranges.len() == programs.len()
                && flags.len() == programs.len()
                && programs.iter().zip(flags).zip(&self.ranges).all(
                    |((program, flags), sections)| {
                        flags.len() == program.len()
                            && sections.len() == gathers.layers.len() + 1
                            && sections.last().map(|last| last.end) == Some(program.len())
                    },
                );
        if !gathers.idle() || !extent {
            return Err("prefill Engram program extent or lifecycle".into());
        }
        for (rank, range) in changed {
            let launches = programs
                .get(rank)
                .and_then(|program| program.get(range.clone()))
                .ok_or("prefill Engram changed launches")?;
            self.audit(gathers, rank, range.start, launches)?;
        }
        self.check_sections(programs, flags)?;
        Ok(self.clone())
    }

    /// The audit `new` makes of rank `rank`'s launches from `at`: a launch
    /// before a layer's consumer touches none of the spans that layer and
    /// the later layers still have pending, and a later launch none of a
    /// mixed step's verify-row gathers'.
    fn audit(
        &self,
        gathers: &Gathers,
        rank: usize,
        at: usize,
        launches: &[PreparedLaunch],
    ) -> Result<(), String> {
        let sections = self.ranges.get(rank).ok_or("prefill Engram rank extent")?;
        for (offset, launch) in launches.iter().enumerate() {
            let index = at + offset;
            let first = sections[..gathers.layers.len()]
                .iter()
                .position(|section| index < section.end)
                .unwrap_or(gathers.layers.len());
            if first == gathers.layers.len() && gathers.verify.is_empty() {
                continue;
            }
            crate::shared_overlap::disjoint_copy_spans(
                &gathers.pending_spans(first),
                std::slice::from_ref(launch),
            )?;
        }
        Ok(())
    }

    /// Build ordered joins inside the actual DMA compute intervals. A moved
    /// producer or consumer may not swallow a boundary, even on one rank.
    pub(in crate::instance) fn split_dma(
        &self,
        spans: &[Range<usize>],
        joined: &mut usize,
    ) -> Result<Vec<DmaPiece>, String> {
        if spans.len() != self.ranges.len() || spans.is_empty() {
            return Err("DMA Engram rank extent".into());
        }
        let mut cursor = spans.iter().map(|s| s.start).collect::<Vec<_>>();
        let mut pieces = Vec::new();
        let mut join = None;
        while *joined + 1 < self.ranges[0].len() {
            let boundaries = self
                .ranges
                .iter()
                .map(|r| r[*joined].end)
                .collect::<Vec<_>>();
            if boundaries.iter().zip(spans).all(|(&b, s)| b >= s.end) {
                break;
            }
            if boundaries
                .iter()
                .zip(spans)
                .any(|(&b, s)| b < s.start || b >= s.end)
            {
                return Err("DMA moved or reordered an Engram join boundary".into());
            }
            pieces.push(DmaPiece {
                spans: cursor
                    .iter()
                    .zip(&boundaries)
                    .map(|(&a, &b)| a..b)
                    .collect(),
                join,
            });
            cursor = boundaries;
            join = Some(*joined);
            *joined += 1;
        }
        pieces.push(DmaPiece {
            spans: cursor.iter().zip(spans).map(|(&a, s)| a..s.end).collect(),
            join,
        });
        Ok(pieces)
    }

    pub(in crate::instance) fn check_dma_collectives(
        &self,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
    ) -> Result<(), String> {
        for piece in self.dma.iter().flatten() {
            let launches = piece
                .spans
                .iter()
                .enumerate()
                .map(|(rank, span)| &programs[rank][span.clone()])
                .collect::<Vec<_>>();
            let waits = piece
                .spans
                .iter()
                .enumerate()
                .map(|(rank, span)| &flags[rank][span.clone()])
                .collect::<Vec<_>>();
            check_collectives(&launches, &waits)
                .map_err(|error| format!("DMA Engram collective boundary: {error:?}"))?;
        }
        Ok(())
    }

    /// Moves the first DMA piece's join to another layer; whether there was
    /// one to move.
    #[cfg(test)]
    pub(in crate::instance) fn move_first_join(&mut self) -> bool {
        let join = self
            .dma
            .iter_mut()
            .flatten()
            .find_map(|piece| piece.join.as_mut());
        join.map(|join| *join += 42).is_some()
    }

    pub(in crate::instance) fn layer_count(&self) -> usize {
        self.ranges[0].len() - 1
    }

    pub(in crate::instance) fn execute(
        &self,
        instance: &Instance,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
    ) -> Result<(), DispatchError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        if instance.prefill_engram.is_none()
            || programs.len() != streams.len()
            || flags.len() != streams.len()
        {
            return Err(DispatchError::InvalidPlan);
        }
        let executor = Executor::new(instance, &streams);
        executor.run(|| {
            instance.launch_layered_pieces(&streams, &self.ranges, programs, flags, |layer| {
                instance.join_prefill_engram(&streams, layer)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::layer_gather::LayerGather;
    use bundle_runtime::dispatch::BufferBindings;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{
        BundleDeviceBufferHandleV1, BundleMappedGatherHandleV1, DeviceBufferSpanV1 as Span,
    };
    use std::cell::Cell;

    fn span(id: u32) -> Span {
        Span::new(BundleDeviceBufferHandleV1::from_parts(1, id), 0, 64)
    }
    fn op(input: Span) -> PreparedLaunch {
        let mut bindings = BufferBindings::default();
        let id = bindings.bind(input);
        PreparedLaunch {
            symbol: "audit".into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1; 3],
                block: [32, 1, 1],
                cluster: [1; 3],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                pdl: true,
            },
            arguments: ArgumentList(vec![Argument::DeviceBuffer(id)]),
            bindings,
        }
    }
    fn gathers() -> Gathers {
        Gathers {
            layers: [(1, 1, 3), (14, 2, 4)]
                .map(|(id, index, output)| LayerGather {
                    layer: id,
                    gather: BundleMappedGatherHandleV1::from_parts(1, index),
                    indices: span(index),
                    destinations: vec![span(output)],
                    graphs: (),
                })
                .into(),
            verify: Vec::new(),
            columns: 24,
            pending: Cell::new(false),
            failed: Cell::new(false),
            joined: Cell::new(0),
            sequence: Cell::new(0),
            rows: Cell::new(0),
            start: Cell::new(0),
        }
    }
    fn program() -> Vec<Vec<PreparedLaunch>> {
        vec![vec![op(span(9)), op(span(3)), op(span(9)), op(span(4))]]
    }
    #[test]
    fn join_audit_rejects_early_consumers_and_reordered_layers() {
        let g = gathers();
        let stages = vec![vec![
            Stage::Embedding,
            Stage::EngramRows(1),
            Stage::ExpandStreams,
            Stage::EngramRows(14),
        ]];
        let flags = vec![vec![false; 4]];
        let clean = program();
        let plan = Plan::new(&g, &clean, &flags, &stages).unwrap();
        assert_eq!(plan.ranges, [vec![0..1, 1..3, 3..4]]);
        for early in [span(1), span(2), span(3), span(4)] {
            let mut bad = program();
            bad[0][0] = op(early);
            assert!(
                Plan::new(&g, &bad, &flags, &stages)
                    .unwrap_err()
                    .contains("alias")
            );
        }
        let mut later = program();
        later[0][2] = op(span(4));
        assert!(
            Plan::new(&g, &later, &flags, &stages)
                .unwrap_err()
                .contains("alias")
        );
        let mut bad = stages.clone();
        bad[0].swap(1, 3);
        assert!(Plan::new(&g, &clean, &flags, &bad).is_err());
        let mut bad = stages.clone();
        bad[0][2] = Stage::EngramRows(1);
        assert!(Plan::new(&g, &clean, &flags, &bad).is_err());
        assert!(Plan::new(&g, &clean, &[vec![false; 3]], &stages).is_err());
        g.pending.set(true);
        assert!(Plan::new(&g, &clean, &flags, &stages).is_err());
    }

    /// A plan reused for a program that changes one launch refuses it
    /// exactly when a fresh plan does, and equals the fresh plan otherwise.
    #[test]
    fn reused_plans_refuse_exactly_the_programs_fresh_plans_refuse() {
        let g = gathers();
        let stages = vec![vec![
            Stage::Embedding,
            Stage::EngramRows(1),
            Stage::ExpandStreams,
            Stage::EngramRows(14),
            Stage::FinalNorm,
        ]];
        let flags = vec![vec![false; 5]];
        let inputs = [9, 3, 9, 4, 1];
        let program = |at: usize, input: u32| {
            vec![
                (0..5)
                    .map(|index| op(span(if index == at { input } else { inputs[index] })))
                    .collect::<Vec<_>>(),
            ]
        };
        let plan = Plan::new(&g, &program(0, inputs[0]), &flags, &stages).unwrap();
        let mut outcomes = [0; 2];
        for at in 0..5 {
            for input in [1, 2, 3, 4, 9] {
                let changed = program(at, input);
                let fresh = Plan::new(&g, &changed, &flags, &stages);
                let reused = plan.reuse(&g, &changed, &flags, [(0, at..at + 1)]);
                match (fresh, reused) {
                    (Ok(fresh), Ok(reused)) => {
                        assert_eq!(fresh, reused);
                        outcomes[0] += 1;
                    }
                    (Err(fresh), Err(reused)) => {
                        assert_eq!(fresh, reused);
                        outcomes[1] += 1;
                    }
                    (fresh, reused) => panic!("launch {at} reads {input}: {fresh:?} {reused:?}"),
                }
            }
        }
        // The first launch may touch no pending span, the two before layer
        // 14's consumer only layer 1's, the rest any.
        assert_eq!(outcomes, [17, 8]);
        let clean = program(0, inputs[0]);
        assert!(plan.reuse(&g, &clean, &flags, [(1, 0..1)]).is_err());
        assert!(plan.reuse(&g, &clean, &flags, [(0, 4..6)]).is_err());
        assert!(plan.reuse(&g, &clean[..0], &flags, []).is_err());
        assert!(plan.reuse(&g, &clean, &[vec![false; 4]], []).is_err());
        g.pending.set(true);
        assert!(plan.reuse(&g, &clean, &flags, []).is_err());
    }

    #[test]
    fn reused_plans_check_every_section_collective_again() {
        let g = gathers();
        let stages = vec![
            vec![
                Stage::Embedding,
                Stage::EngramRows(1),
                Stage::EngramRows(14)
            ];
            2
        ];
        let programs = || {
            (0..2)
                .map(|_| vec![op(span(9)), op(span(5)), op(span(6))])
                .collect::<Vec<_>>()
        };
        let flags = vec![vec![false, true, false]; 2];
        let plan = Plan::new(&g, &programs(), &flags, &stages).unwrap();
        assert_eq!(plan.reuse(&g, &programs(), &flags, []).unwrap(), plan);
        let mut other = programs();
        other[1][1].symbol = "other".into();
        let fresh = Plan::new(&g, &other, &flags, &stages).unwrap_err();
        assert_eq!(plan.reuse(&g, &other, &flags, []).unwrap_err(), fresh);
    }

    #[test]
    fn dma_splits_only_compute_spans_and_refuses_moved_join_boundaries() {
        let plan = Plan {
            ranges: vec![vec![0..2, 2..7, 7..10]; 4],
            dma: Vec::new(),
        };
        let mut joined = 0;
        let pieces = plan.split_dma(&vec![0..4; 4], &mut joined).unwrap();
        assert_eq!(joined, 1);
        assert_eq!(
            pieces.iter().map(|p| p.join).collect::<Vec<_>>(),
            [None, Some(0)]
        );
        assert_eq!(pieces[0].spans, vec![0..2; 4]);
        assert_eq!(pieces[1].spans, vec![2..4; 4]);
        let pieces = plan.split_dma(&vec![6..10; 4], &mut joined).unwrap();
        assert_eq!(joined, 2);
        assert_eq!(pieces[1].join, Some(1));
        assert!(plan.split_dma(&vec![3..4; 4], &mut 0).is_err());
        assert!(plan.split_dma(&[0..4, 0..4, 0..2, 0..4], &mut 0).is_err());
        let mut joined = 0;
        assert_eq!(
            plan.split_dma(&vec![0..2; 4], &mut joined).unwrap().len(),
            1
        );
        assert_eq!(joined, 0);
        assert_eq!(
            plan.split_dma(&vec![2..4; 4], &mut joined).unwrap()[1].join,
            Some(0)
        );
    }

    #[test]
    fn dma_piece_collective_imbalance_is_refused_before_enqueue() {
        let programs = vec![vec![op(span(1))], vec![op(span(2))]];
        let flags = vec![vec![true], vec![true]];
        let mut plan = Plan {
            ranges: Vec::new(),
            dma: vec![vec![DmaPiece {
                spans: vec![0..1, 0..1],
                join: None,
            }]],
        };
        plan.check_dma_collectives(&programs, &flags).unwrap();
        plan.dma[0][0].spans[1] = 0..0;
        assert!(plan.check_dma_collectives(&programs, &flags).is_err());
    }
}
