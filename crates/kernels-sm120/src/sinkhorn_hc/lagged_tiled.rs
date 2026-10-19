//! Exact lagged mix-collapse with several rows sharing each fn load.
//! The prefill entry keeps the one-row ABI, chain order and F32 outputs.

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{InstantiationSpec, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;

use super::{CHAIN, Geometry, THREADS, lagged, tiled};
use crate::Profile;

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: lagged::Entry,
    pub rows: u32,
    /// Small buckets keep the registered decode implementation.
    pub from_rows: u32,
}

impl Entry {
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let mut spec = self.reference.contract()?;
        let Query::MixCollapse { mix, capacity } = self.reference.query else {
            return Err(Refusal("the lagged tiled entry requires a mix-collapse"));
        };
        if !matches!(self.reference.op, lagged::Op::MixCollapse { .. }) {
            return Err(Refusal("the lagged tiled entry requires a mix-collapse"));
        }
        tiled::fits(mix, self.rows)?;
        if !mix.width.is_multiple_of(8) {
            return Err(Refusal(
                "the lagged tiled entry needs 16-byte collapse packs",
            ));
        }
        if self.from_rows == 0 || self.from_rows > capacity {
            return Err(Refusal("lagged tiled minimum rows exceed its capacity"));
        }
        spec.key.knobs.tile_m = self.rows as u16;
        spec.key.knobs.tile_k = (tiled::chunk_chains(self.rows) * CHAIN) as u16;
        spec.key.knobs.stages = 2;
        spec.validate()?;
        Ok(spec)
    }

    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.contract()?;
        self.reference.capacity()
    }

    pub fn invariance(&self) -> Invariance {
        if self.contract().is_ok() {
            self.reference.invariance()
        } else {
            Invariance::NONE
        }
    }

    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let mut plan = self.reference.plan(rows, live)?;
        plan.grid = [rows.div_ceil(self.rows), 1, 1];
        plan.validate()?;
        Ok(plan)
    }

    pub fn body(&self) -> Result<String, Refusal> {
        self.contract()?;
        let probe = matches!(self.reference.op, lagged::Op::MixCollapse { probe: true });
        Ok(self
            .reference
            .body()
            .replace("mix_collapse_lagged<", "mix_collapse_lagged_tiled<")
            .replace(&format!(",{probe}>("), &format!(",{},{probe}>(", self.rows)))
    }
}

pub fn entry(
    profile: Profile,
    geometry: Geometry,
    rows: u32,
    from_rows: u32,
    probe: bool,
) -> Entry {
    let mut reference = lagged::entries(profile, geometry, probe)
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .expect("lagged mix-collapse");
    reference.symbol = reference.symbol.replace(
        "lagged_mix_collapse",
        &format!("lagged_mix_collapse_tiled_r{rows}"),
    );
    Entry {
        reference,
        rows,
        from_rows,
    }
}

pub fn cubin_spec(name: String, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/lagged_tiled.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|e| {
                Ok(EntrySpec::new(
                    e.reference.symbol.clone(),
                    e.reference.params.clone(),
                    e.body()?,
                    THREADS,
                    1,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_launches_preserve_capacity_and_reject_invalid_geometry() {
        for tile in tiled::ROWS {
            let e = entry(lagged::LAB_PROFILE, lagged::LAB, tile, 256, false);
            for rows in [1, 7, 33, 1024, 2048, 8191, 8192] {
                let p = e.plan(rows, 0).unwrap();
                assert_eq!(p.grid, [rows.div_ceil(tile), 1, 1]);
                assert_eq!(p.block, [256, 1, 1]);
                assert_eq!(p.cluster, [1, 1, 1]);
                assert_eq!(p.dynamic_shared_bytes, 0);
                assert_eq!(p.workspace_bytes, 0);
                assert!(p.pdl);
            }
            assert_eq!(e.invariance(), Invariance::any_rows(8192));
            assert!(e.plan(0, 0).is_err());
            assert!(e.plan(8193, 0).is_err());
            assert!(e.plan(8192, 8193).is_err());
            assert_eq!(e.reference.params.len(), 10);
        }
        for tile in [0, 1, 2, 3, 32] {
            assert!(
                entry(lagged::LAB_PROFILE, lagged::LAB, tile, 256, false)
                    .contract()
                    .is_err()
            );
        }
        for minimum in [0, 8193] {
            assert!(
                entry(lagged::LAB_PROFILE, lagged::LAB, 8, minimum, false)
                    .contract()
                    .is_err()
            );
        }
        let probe = entry(lagged::LAB_PROFILE, lagged::LAB, 8, 256, true);
        assert_eq!(probe.invariance(), Invariance::NONE);
        assert!(probe.body().unwrap().contains(",8,true>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,projected,rstd,cells,live_rows)"));
    }
    #[test]
    fn backend_keeps_small_buckets_and_requires_unambiguous_tiles() {
        use crate::backend::Sm120;
        use kernel_core::backend::{Backend, Counts, Need, View};
        use kernel_core::sinkhorn_hyper_connection::{
            LaggedMixCollapseBuffers, MixScratch, MixerWeights,
        };
        let view = |handle| View {
            handle,
            bytes: u64::MAX,
        };
        let buffers = LaggedMixCollapseBuffers {
            x: view(1),
            mixer: MixerWeights {
                fn_: view(2),
                base: view(3),
                scale: view(4),
            },
            pre_in: view(5),
            collapsed: view(6),
            pre: view(7),
            post: view(8),
            comb: view(9),
            scratch: MixScratch {
                partials: None,
                tickets: None,
            },
            live_rows: view(10),
        };
        let e = entry(lagged::LAB_PROFILE, lagged::LAB, 8, 256, false);
        let backend = || Sm120::new().with_sinkhorn_lagged(lagged::catalog());
        let op = backend()
            .with_sinkhorn_lagged_tiled([e.clone()])
            .sinkhorn_lagged_mix_collapse(&e.reference.query, Need::rows(8192))
            .unwrap();
        for rows in [1, 32, 128, 255, 256, 1024, 2048, 8192] {
            let launches = op
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol.contains("_tiled_"), rows >= 256);
            assert_eq!(
                launches[0].plan.grid[0],
                if rows >= 256 { rows.div_ceil(8) } else { rows }
            );
            assert_eq!(launches[0].plan.workspace_bytes, 0);
        }
        let other = entry(lagged::LAB_PROFILE, lagged::LAB, 4, 256, false);
        for entries in [[e.clone(), other.clone()], [other, e.clone()]] {
            assert!(
                backend()
                    .with_sinkhorn_lagged_tiled(entries)
                    .sinkhorn_lagged_mix_collapse(&e.reference.query, Need::rows(8192))
                    .is_err()
            );
        }
    }
}
