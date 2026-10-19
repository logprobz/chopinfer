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
fn decode_buckets_use_split_and_prefill_keeps_the_original_mixer() {
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
        let backend = registration::register(Sm120::new(), g);
        let op = backend
            .sinkhorn_lagged_mix_collapse(
                &hc::mix_collapse(g, PREFILL_TOKENS),
                Need::rows(PREFILL_TOKENS),
            )
            .unwrap();
        for rows in [1, 2, 4, 8, 16, 31, 32, 33, 128, PREFILL_TOKENS] {
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
            count += 1;
        }
        assert_eq!(op.info().bytes_of("scratch.tickets"), Some(32 * 4));
    }
    assert!(count >= 10);
}
