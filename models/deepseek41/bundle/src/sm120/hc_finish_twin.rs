//! The served HC finish/norm tail on two CTAs per row: one runs the fold and
//! Sinkhorn, the other the collapse and cached norm.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::{
    completion::Mutation,
    finish_twin::{self, Entry, WARPS},
};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "sinkhorn_hc_finish_twin";
pub const CUBIN: &str = "sinkhorn_hc_deepseek41_finish_twin_sm120";
pub fn served() -> Entry {
    Entry {
        reference: super::hc_finish_norm::entry(&FLASH),
        prefix: format!("chop_{}_hc_finish_twin", FLASH.kernel_tag),
        mutation: Mutation::None,
    }
}
/// Lab controls: each must differ from the served tail on real rows.
pub fn controls() -> Vec<Entry> {
    [
        Mutation::ClearStatus,
        Mutation::WrongFloor,
        Mutation::ReverseFold,
    ]
    .into_iter()
    .map(|mutation| Entry {
        prefix: format!(
            "chop_hc_finish_twin_{}",
            format!("{mutation:?}").to_lowercase()
        ),
        mutation,
        ..served()
    })
    .collect()
}
/// Every candidate warp count, then each control at 32 warps.
pub fn entries() -> Vec<(Entry, u32)> {
    WARPS
        .iter()
        .map(|&w| (served(), w))
        .chain(controls().into_iter().map(|e| (e, 32)))
        .collect()
}
pub fn cubin() -> Result<CubinSpec, String> {
    finish_twin::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        backend::Launch,
        contract::{Argument, ArgumentList},
    };
    #[test]
    fn symbols_plans_and_arguments_keep_the_served_tail() {
        let spec = cubin().unwrap();
        assert_eq!(spec.entries.len(), 6);
        let tail = super::super::hc_finish_norm::entry(&FLASH);
        let reference = tail.cubin_spec("tail").unwrap().entries.remove(1);
        let mut symbols = std::collections::BTreeSet::new();
        for e in &spec.entries {
            assert!(symbols.insert(e.symbol.clone()), "{}", e.symbol);
            assert_eq!(e.params, reference.params);
        }
        let e = served();
        for rows in [1, 6, 8, 32] {
            for w in WARPS {
                let p = e.plan(rows, rows, w).unwrap();
                let served = tail.plan(rows, rows, 32).unwrap();
                assert_eq!((p.grid, p.block), ([2, rows, 1], [w * 32, 1, 1]));
                assert_eq!(p.workspace_bytes, served.workspace_bytes);
                assert_eq!(p.dynamic_shared_bytes, served.dynamic_shared_bytes);
                assert!(p.pdl);
                assert!(e.plan(rows, rows + 1, w).is_err());
            }
            let launch = Launch {
                symbol: tail.symbols()[1].clone(),
                plan: tail.plan(rows, rows, 32).unwrap(),
                rows,
                arguments: ArgumentList((0..18).map(Argument::DeviceBuffer).collect()),
                collective: false,
                persistent: false,
            };
            let twin = e.compose(&launch, 16).unwrap();
            assert_eq!(twin.arguments, launch.arguments);
            assert_eq!(twin.symbol, "chop_deepseek41_hc_finish_twin_w16_v1");
            assert_eq!(twin.plan, e.plan(rows, rows, 16).unwrap());
            // Only the served tail at its own plan is replaced.
            let mut other = launch.clone();
            other.symbol = tail.symbols()[0].clone();
            assert!(e.compose(&other, 16).is_err());
            let mut other = launch.clone();
            other.plan.block = [512, 1, 1];
            assert!(e.compose(&other, 16).is_err());
            let mut other = launch.clone();
            other.arguments.0.pop();
            assert!(e.compose(&other, 16).is_err());
        }
        assert!(e.plan(1, 1, 4).is_err());
        assert_eq!(
            CUBIN,
            format!("sinkhorn_hc_{}_finish_twin_sm120", FLASH.kernel_tag)
        );
    }

    #[test]
    fn backend_replaces_only_the_tail_at_twin_buckets() {
        use crate::sm120 as b;
        use kernel_core::{
            backend::{Backend, Counts, Need, View},
            properties::Invariance,
            rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
            sinkhorn_expand_mix::{
                SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers,
                SinkhornRoundNormQuery,
            },
        };
        use kernels_sm120::backend::Sm120;
        // The served registration without its two-CTA buckets.
        let base = || {
            let backend = b::norm_quantize::register(Sm120::new(), &FLASH).unwrap();
            let backend = b::norm_quantize_cached::register(backend, &FLASH);
            let backend = b::hc_round_expand_mix::register(backend, &FLASH);
            let backend = b::hc_round_chain::register(backend, &FLASH);
            b::hc_finish_norm::register_pair(backend, &FLASH).with_sinkhorn_finish_norm_completion(
                b::hc_completion::served(),
                b::hc_finish_norm::COMPLETION_BUCKETS.to_vec(),
            )
        };
        let entry = b::hc_finish_norm::entry(&FLASH);
        let query = SinkhornRoundNormQuery {
            mixer: entry.mixer.query,
            norm: entry.norm.reference.query,
        };
        let need = Need::rows(32).with_exact(Invariance::any_rows(32));
        let v = |handle: u64| View {
            handle,
            bytes: 1 << 30,
        };
        let buffers = SinkhornRoundNormBuffers {
            hc: Hc {
                x: v(1),
                sum: v(2),
                rounded: v(3),
                post_in: v(4),
                comb_in: v(5),
                expanded: v(6),
                fn_: v(7),
                base: v(8),
                scale: v(9),
                pre_in: v(10),
                collapsed: v(11),
                pre: v(12),
                post: v(4),
                comb: v(5),
                live_rows: v(13),
            },
            norm: Norm {
                input: v(11),
                weight: v(14),
                normed: v(15),
                widened: Some(v(16)),
                codes: v(17),
                scales: v(18),
                live_rows: v(13),
                status: v(19),
                status_row_offset: 0,
            },
            tickets: Some(v(20)),
        };
        let served = base().sinkhorn_round_expand_mix_norm(&query, need).unwrap();
        let twin = base()
            .with_sinkhorn_finish_norm_twin(super::served(), vec![(6, 16)])
            .sinkhorn_round_expand_mix_norm(&query, need)
            .unwrap();
        assert!(
            twin.info()
                .symbols
                .contains(&("chop_deepseek41_hc_finish_twin_w16_v1".into(), 0))
        );
        for rows in [1, 4, 6, 8, 32] {
            let workspace = View {
                handle: 21,
                bytes: entry.mixer.workspace(32).unwrap(),
            };
            let old = served
                .launches(&buffers, workspace, Counts::rows(rows))
                .unwrap();
            let new = twin
                .launches(&buffers, workspace, Counts::rows(rows))
                .unwrap();
            if rows != 6 {
                assert_eq!(new, old, "{rows}");
                continue;
            }
            assert_eq!(old.len(), 2);
            assert_eq!(new.len(), 2);
            assert_eq!(new[0], old[0]);
            assert_eq!(new[1].symbol, "chop_deepseek41_hc_finish_twin_w16_v1");
            assert_eq!(new[1].arguments, old[1].arguments);
            assert_eq!(new[1].plan, super::served().plan(6, 6, 16).unwrap());
        }
        // A twin bucket needs a separate tail: none at an unmeasured row
        // count or at a one-launch completion bucket. Controls never serve.
        for (bucket, entry) in [
            ((2, 16), super::served()),
            ((8, 16), super::served()),
            ((6, 16), controls().remove(0)),
        ] {
            assert!(
                base()
                    .with_sinkhorn_finish_norm_twin(entry, vec![bucket])
                    .sinkhorn_round_expand_mix_norm(&query, need)
                    .is_err()
            );
        }
    }
}
