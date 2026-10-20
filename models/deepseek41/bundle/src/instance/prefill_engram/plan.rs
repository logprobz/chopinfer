//! Split eager programs at the same layer-consumer boundaries as DSpark.
use super::Gathers;
use crate::instance::Instance;
use crate::program::Stage;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch, check_collectives};
use std::ops::Range;

#[derive(Debug)]
pub(in crate::instance) struct Plan {
    ranges: Vec<Vec<Range<usize>>>,
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
                    (*stage == Stage::EngramRows(layer.id)).then_some(at)
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
            sections.push(start..program.len());
            ranges.push(sections);
        }
        for piece in 0..=gathers.layers.len() {
            let ops: Vec<_> = programs
                .iter()
                .enumerate()
                .map(|(rank, p)| &p[ranges[rank][piece].clone()])
                .collect();
            let waits: Vec<_> = flags
                .iter()
                .enumerate()
                .map(|(rank, p)| &p[ranges[rank][piece].clone()])
                .collect();
            check_collectives(&ops, &waits)
                .map_err(|e| format!("prefill Engram collective boundary: {e:?}"))?;
        }
        Ok(Self { ranges })
    }

    pub(in crate::instance) fn execute(
        &self,
        instance: &Instance,
        programs: &[Vec<PreparedLaunch>],
        flags: &[Vec<bool>],
    ) -> Result<(), DispatchError> {
        let gathers = instance
            .prefill_engram
            .as_ref()
            .ok_or(DispatchError::InvalidPlan)?;
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        if programs.len() != streams.len() || flags.len() != streams.len() {
            return Err(DispatchError::InvalidPlan);
        }
        let _unwind = PoisonOnUnwind(&instance.collectives);
        for piece in 0..=gathers.layers.len() {
            if piece > 0 {
                gathers.join(&instance.services, &streams, piece - 1)?;
            }
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
            instance.launch_eager(&streams, &ops, &waits)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Layer;
    use super::*;
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
                .map(|(id, index, output)| Layer {
                    id,
                    gather: BundleMappedGatherHandleV1::from_parts(1, index),
                    indices: span(index),
                    destinations: vec![span(output)],
                })
                .into(),
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
}
