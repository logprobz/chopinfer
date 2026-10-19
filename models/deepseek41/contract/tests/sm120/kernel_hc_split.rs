#[path = "../../../bundle/src/sm120/hc.rs"]
#[allow(dead_code)]
mod registration;
#[path = "../../../bundle/src/sm120/rows.rs"]
#[allow(dead_code)]
mod rows;

use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::sinkhorn_hyper_connection::{LaggedMixCollapseBuffers, MixScratch, MixerWeights};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{geometry::GEOMETRIES, hc, serving::PREFILL_TOKENS};

#[test]
fn decode_buckets_use_split_and_prefill_uses_row_tiles() {
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
            partials: Some(view(10)),
            tickets: Some(view(11)),
        },
        live_rows: view(12),
    };
    let mut count = 0;
    for g in GEOMETRIES {
        let entry = registration::prefill_entry(g);
        let shared = kernel_lab_families::sinkhorn_hc_lagged_tiled::entry(8, false);
        assert_eq!(entry.body().unwrap(), shared.body().unwrap());
        assert_eq!(entry.reference.params, shared.reference.params);
        let backend = registration::register(Sm120::new(), g);
        let op = backend
            .sinkhorn_lagged_mix_collapse(
                &hc::mix_collapse(g, PREFILL_TOKENS),
                Need::rows(PREFILL_TOKENS),
            )
            .unwrap();
        for rows in [
            1,
            2,
            4,
            8,
            16,
            31,
            32,
            33,
            128,
            255,
            256,
            1024,
            2048,
            PREFILL_TOKENS,
        ] {
            let launch = op
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap()
                .remove(0);
            assert_eq!(
                launch.symbol.contains("lagged_mix_collapse_split"),
                rows <= 32,
                "{} rows={rows}: {}",
                g.id,
                launch.symbol
            );
            assert_eq!(
                launch.symbol.contains("_tiled_"),
                rows >= 256,
                "{} rows={rows}: {}",
                g.id,
                launch.symbol
            );
            if rows >= 256 {
                assert_eq!(launch.plan.grid, [rows.div_ceil(8), 1, 1]);
                assert_eq!(launch.plan.block, [256, 1, 1]);
                assert_eq!(launch.plan.dynamic_shared_bytes, 0);
                assert_eq!(launch.plan.workspace_bytes, 0);
            }
            count += 1;
        }
        assert_eq!(op.info().bytes_of("scratch.tickets"), Some(32 * 4));
    }
    assert!(count >= 10);
}
