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
        let missing = (0..self.ranks.len())
            .filter(|&rank| cached.is_none_or(|cache| !cache.contains(key(rank as u32))))
            .count();
        let mut reservation = (!ready)
            .then(|| self.graphs.reserve_retaining(missing))
            .flatten();
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
        cache.label(
            match program {
                Program::Draft => "draft",
                Program::CatchUp => "catch-up",
            },
            Some(count),
            0,
        );
        if !ready && reservation.is_none() {
            cache.trace(&self.services, "eager", Some(key(0)), missing, 0);
            return self
                .launch_eager(&streams, &programs, &waits)
                .map_err(|e| format!("DSpark eager program: {e:?}"));
        }
        for (rank, &stream) in streams.iter().enumerate() {
            if cache.contains(key(rank as u32)) {
                continue;
            }
            // A single capture unit contains the whole program, including
            // byte gathers. Metadata uploads and status reads stay outside.
            cache
                .capture_reserved::<DispatchError>(
                    &self.services,
                    stream,
                    key(rank as u32),
                    1,
                    reservation.as_mut().expect("missing graphs were admitted"),
                    |_| {
                        pipeline::launch_all(
                            &self.services,
                            stream,
                            &self.symbols[rank],
                            &programs[rank],
                        )
                    },
                )
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

/// One drafter launch as the journal export prints it: symbol, rows,
/// grid, block, dynamic shared bytes and whether it is a collective.
#[cfg(test)]
pub(crate) type DrafterLaunch = (String, u32, [u32; 3], [u32; 3], u32, bool);

/// Each rank's complete drafter program for `items` drafting items, as the
/// draft graph would capture it.
#[cfg(test)]
pub(crate) fn drafter_launches_for_test(
    handle: runtime_bundle_abi::BundleHandleV1,
    items: u32,
) -> Result<Vec<Vec<DrafterLaunch>>, String> {
    crate::instance::INSTANCES.with(|table| {
        let mut table = table.borrow_mut();
        let instance = table.get_mut(handle).ok_or("no live instance")?;
        Ok(instance
            .drafter_programs(items)?
            .into_iter()
            .map(|rank| {
                rank.into_iter()
                    .map(|op| {
                        let plan = op.launch.plan;
                        (
                            op.launch.symbol,
                            op.launch.rows,
                            plan.grid,
                            plan.block,
                            plan.dynamic_shared_bytes,
                            op.collective,
                        )
                    })
                    .collect()
            })
            .collect())
    })
}
