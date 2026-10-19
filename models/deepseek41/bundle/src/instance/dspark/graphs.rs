//! Capture catch-up and full drafts over fixed staging and scratch spans.

use super::Instance;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::pipeline::{self, PoisonOnUnwind, Unit};

#[derive(Clone, Copy)]
enum Program {
    Draft,
    CatchUp,
}

impl Instance {
    pub(super) fn replay_drafter(&mut self, items: u32) -> Result<(), String> {
        self.replay_dspark_graph(Program::Draft, items)
    }

    pub(super) fn replay_catch_up(&mut self, rows: u32) -> Result<(), String> {
        self.replay_dspark_graph(Program::CatchUp, rows)
    }

    fn replay_dspark_graph(&mut self, program: Program, count: u32) -> Result<(), String> {
        // Each exact row or item count owns a separate cache. Its sole key is an
        // opaque program unit, so no arithmetic row is padded for capture.
        let key = |rank| GraphKey {
            rank,
            bucket: 1,
            items: 0,
        };
        let cached = match program {
            Program::Draft => self.draft_graphs.get(&count),
            Program::CatchUp => self
                .dspark
                .as_ref()
                .ok_or("DSpark is absent")?
                .catch_up_graphs
                .get(&count),
        };
        let ready = cached.is_some_and(|cache| {
            (0..self.ranks.len()).all(|rank| cache.contains(key(rank as u32)))
        });
        let (programs, waits): (Vec<Vec<_>>, Vec<Vec<_>>) = if ready {
            (Vec::new(), Vec::new())
        } else {
            match program {
                Program::Draft => self.drafter_programs(count)?,
                Program::CatchUp => self.catch_up_programs(count)?,
            }
            .into_iter()
            .map(|rank| {
                rank.into_iter()
                    .map(|op| (op.launch, op.collective))
                    .unzip()
            })
            .unzip()
        };
        if !ready {
            pipeline::check_collectives(&programs, &waits)?;
        }
        let streams: Vec<_> = self.ranks.iter().map(|rank| rank.2).collect();
        let caches = match program {
            Program::Draft => &mut self.draft_graphs,
            Program::CatchUp => {
                &mut self
                    .dspark
                    .as_mut()
                    .ok_or("DSpark is absent")?
                    .catch_up_graphs
            }
        };
        let cache = caches
            .entry(count)
            .or_insert_with(|| GraphCache::sharing(&self.graphs));
        for (rank, &stream) in streams.iter().enumerate() {
            if cache.contains(key(rank as u32)) {
                continue;
            }
            // A single capture unit contains the whole program, including
            // byte gathers. Metadata uploads and status reads stay outside.
            cache
                .capture::<DispatchError>(&self.services, stream, key(rank as u32), 1, |_| {
                    pipeline::launch_all(
                        &self.services,
                        stream,
                        &self.symbols[rank],
                        &programs[rank],
                    )
                })
                .map_err(|e| format!("DSpark graph capture: {e:?}"))?;
        }
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let replayed = Unit::default()
            .record(
                &self.collectives,
                cache
                    .replay_piece_major(&self.services, &streams, (1, 0), |_| {})
                    .map_err(DispatchError::from),
            )
            .map_err(|e| format!("DSpark graph replay: {e:?}"))?;
        if !replayed {
            return Err("DSpark graph cache is incomplete".into());
        }
        Ok(())
    }
}
