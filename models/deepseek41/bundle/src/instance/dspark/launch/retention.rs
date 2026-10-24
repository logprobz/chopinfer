use super::*;
use bundle_runtime::dispatch::{self, Resolved};

pub(super) struct Bound {
    encoded: Vec<u8>,
    ranks: Vec<(Span, Resolved)>,
}

impl Instance {
    pub(super) fn prepare_indexed_retention(
        &self,
        plan: &CommitPlan,
        copies: &[(usize, crate::dspark::features::Copy)],
    ) -> Result<Option<Bound>, String> {
        let device = self.dspark.as_ref().ok_or("DSpark is absent")?;
        let Some(prepared) = &device.retention else {
            return Ok(None);
        };
        if plan
            .items()
            .iter()
            .any(|item| matches!(item.rows.kind, Kind::Prefill { .. }) || item.project_from_staging)
        {
            return Ok(None);
        }
        let count = copies.iter().filter(|(rank, _)| *rank == 0).count();
        if count < crate::dspark::retention::MIN_COPIES {
            return Ok(None);
        }
        let mut common = None;
        let mut ranks = Vec::new();
        for (rank, buffers) in self.buffers.iter().enumerate() {
            let retained = buffers
                .state
                .named("dspark.features")
                .ok_or("retention state is absent")?;
            let stored = &device.ranks[rank];
            let metadata = stored
                .retention_segments
                .ok_or("retention metadata is absent")?;
            let copies: Vec<_> = copies
                .iter()
                .filter(|(owner, _)| *owner == rank)
                .map(|(_, copy)| *copy)
                .collect();
            if copies.len() != count {
                return Err("retention counts differ across ranks".into());
            }
            let (encoded, lowered) = prepared.bind(retained, stored.packed, metadata, &copies)?;
            match &common {
                Some(expected) if expected != &encoded => {
                    return Err("retention mappings differ across ranks".into());
                }
                None => common = Some(encoded),
                _ => {}
            }
            let launch = lowered.launch;
            let resolved = dispatch::resolve(
                &self.symbols[rank],
                &launch.symbol,
                launch.plan,
                &launch.arguments,
                &launch.bindings,
            )
            .map_err(|error| format!("retention launch binding: {error:?}"))?;
            ranks.push((metadata, resolved));
        }
        Ok(Some(Bound {
            encoded: common.ok_or("retention has no ranks")?,
            ranks,
        }))
    }

    pub(super) fn submit_indexed_retention(&self, bound: Bound) -> Result<(), String> {
        for (rank, (metadata, launch)) in bound.ranks.into_iter().enumerate() {
            let stream = self.ranks[rank].2;
            upload(&self.services, stream, metadata, &bound.encoded)?;
            launch
                .submit(&self.services, stream)
                .map_err(|error| format!("indexed retention: {error:?}"))?;
        }
        Ok(())
    }
}
