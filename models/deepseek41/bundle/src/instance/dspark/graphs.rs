//! Capture every stage and all five Markov positions as one rank graph.

use super::Instance;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::pipeline::{self, PoisonOnUnwind, Unit};

impl Instance {
    pub(super) fn replay_drafter(&mut self, items: u32) -> Result<(), String> {
        // Each exact item count owns a separate cache. Its sole key is an
        // opaque program unit, so no arithmetic row is padded for capture.
        let key = |rank| GraphKey {
            rank,
            bucket: 1,
            items: 0,
        };
        let ready = self.draft_graphs.get(&items).is_some_and(|cache| {
            (0..self.ranks.len()).all(|rank| cache.contains(key(rank as u32)))
        });
        let (programs, waits): (Vec<Vec<_>>, Vec<Vec<_>>) = if ready {
            (Vec::new(), Vec::new())
        } else {
            self.drafter_programs(items)?
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
        let cache = self
            .draft_graphs
            .entry(items)
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
