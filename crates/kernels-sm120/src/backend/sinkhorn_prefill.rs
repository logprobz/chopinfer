//! Measured prefill routes over the ordinary lagged mixer's validation.
use std::collections::BTreeMap;

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{LaunchPlan, Refusal};
use kernel_core::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, SinkhornHyperConnectionQuery as Query,
};

use super::{sinkhorn::Selection, sinkhorn::uncapped, sinkhorn_lagged};
use crate::sinkhorn_hc::{lagged, prefill_warp::Entry};

pub(super) fn prepare(
    selection: &Selection,
    query: &Query,
    need: Need,
) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
    let inner = sinkhorn_lagged::mix_collapse(selection, query, need)?;
    let mut info = inner.info().clone();
    let mut choices = BTreeMap::new();
    for (entry, buckets) in &selection.lagged_prefill {
        if uncapped(entry.reference.query) != uncapped(*query) {
            continue;
        }
        entry.contract()?;
        if entry.reference.op != (lagged::Op::MixCollapse { probe: false })
            || buckets.is_empty()
            || buckets[0] == 0
            || buckets.windows(2).any(|pair| pair[0] >= pair[1])
            || *buckets.last().unwrap() > entry.reference.capacity()?
        {
            return Err(Refused::Rule(
                "invalid measured prefill mixer buckets".into(),
            ));
        }
        Need::rows(need.capacity)
            .with_exact(info.guarantees)
            .check(&OpInfo::new("prefill mixer", entry.invariance(), vec![]))?;
        for &rows in buckets.iter().filter(|&&rows| rows <= need.capacity) {
            let plan = entry.plan(rows, rows)?;
            if choices.insert(rows, (entry.clone(), plan)).is_some() {
                return Err(Refused::Rule(
                    "ambiguous measured prefill mixer bucket".into(),
                ));
            }
            let symbol = (entry.reference.symbol.clone(), entry.shared_bytes());
            if !info.symbols.contains(&symbol) {
                info.symbols.push(symbol);
            }
        }
    }
    need.check(&info)?;
    Ok(Box::new(Op {
        inner,
        info,
        choices,
    }))
}

struct Op {
    inner: Prepared<LaggedMixCollapseBuffers<View>>,
    info: OpInfo,
    choices: BTreeMap<u32, (Entry, LaunchPlan)>,
}

impl PreparedOp<LaggedMixCollapseBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LaggedMixCollapseBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(buffers, workspace, counts)?;
        let Some((entry, plan)) = self.choices.get(&counts.rows) else {
            return Ok(launches);
        };
        let [launch] = launches.as_mut_slice() else {
            return Err(Refusal("prefill mixer replaces one validated launch"));
        };
        // The selected entry preserves the ordinary ten-pointer ABI.
        launch.arguments = entry.reference.pack_arguments(launch.arguments.0.clone())?;
        launch.symbol = entry.reference.symbol.clone();
        launch.plan = *plan;
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::sinkhorn_hc::lagged::{LAB, LAB_PROFILE};
    use crate::sinkhorn_hc::prefill_warp;
    use kernel_core::backend::Backend;
    use kernel_core::sinkhorn_hyper_connection::{MixScratch, MixerWeights};

    fn backend(buckets: Vec<u32>) -> Sm120 {
        Sm120::new()
            .with_sinkhorn_lagged(lagged::entries(LAB_PROFILE, LAB, false))
            .with_sinkhorn_lagged_prefill(
                prefill_warp::pipeline(LAB_PROFILE, LAB, 8, 4, false),
                buckets,
            )
    }

    #[test]
    fn measured_prefill_keeps_the_abi_refusals_and_other_phase_routes() {
        let query = Query::MixCollapse {
            mix: LAB.mix,
            capacity: 8192,
        };
        let need = Need::rows(8192).with_exact(kernel_core::properties::Invariance::any_rows(8192));
        let backend = backend(vec![64, 8192]);
        let old = backend.sinkhorn_lagged_mix_collapse(&query, need).unwrap();
        let new = backend.sinkhorn_lagged_prefill(&query, need).unwrap();
        let buffers = LaggedMixCollapseBuffers {
            x: view(1, u64::MAX),
            mixer: MixerWeights {
                fn_: view(2, u64::MAX),
                base: view(3, u64::MAX),
                scale: view(4, u64::MAX),
            },
            pre_in: view(5, u64::MAX),
            collapsed: view(6, u64::MAX),
            pre: view(7, u64::MAX),
            post: view(8, u64::MAX),
            comb: view(9, u64::MAX),
            scratch: MixScratch {
                partials: None,
                tickets: None,
            },
            live_rows: view(10, 4),
        };
        assert_eq!(new.info().guarantees, old.info().guarantees);
        assert_eq!(new.info().bytes, old.info().bytes);
        assert_eq!(new.info().workspace, 0);
        for rows in [1, 63, 64, 65, 192, 8191, 8192] {
            let a = old
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let b = new
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(a[0].arguments, b[0].arguments);
            assert_eq!(a[0].rows, b[0].rows);
            assert_eq!(a[0].symbol != b[0].symbol, [64, 8192].contains(&rows));
            if ![64, 8192].contains(&rows) {
                assert_eq!(a, b);
            }
        }
        let mut short = buffers;
        short.collapsed.bytes = 8192 * 5120 * 2 - 1;
        assert!(
            old.launches(&short, View::EMPTY, Counts::rows(8192))
                .is_err()
        );
        assert!(
            new.launches(&short, View::EMPTY, Counts::rows(8192))
                .is_err()
        );
    }

    #[test]
    fn prefill_registration_refuses_ambiguous_or_unqualified_buckets() {
        let query = Query::MixCollapse {
            mix: LAB.mix,
            capacity: 8192,
        };
        for rows in [vec![], vec![0], vec![64, 64], vec![192, 64], vec![8193]] {
            assert!(
                backend(rows)
                    .sinkhorn_lagged_prefill(&query, Need::rows(8192))
                    .is_err()
            );
        }
        let backend = backend(vec![64]).with_sinkhorn_lagged_prefill(
            prefill_warp::pipeline(LAB_PROFILE, LAB, 8, 8, false),
            vec![64],
        );
        assert!(
            backend
                .sinkhorn_lagged_prefill(&query, Need::rows(8192))
                .is_err()
        );
    }
}
