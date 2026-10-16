//! The model's selected RMSNorm rows preserve their typed launch plans.

use kernel_core::contract::{RmsNormKind, RuntimeCounts};
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};

#[test]
fn prepared_capacity_and_legacy_launches() {
    let catalog = kernels_sm120::row::catalog(model_mimo26_contract::sm120::catalogs());
    for name in [
        "mimo_hf4096",
        "mimo_plain1280",
        "mimo_plain1024",
        "mimo_hf4096_residual",
        "plain4096",
        "plain8192",
        "gemma4096",
        "qk128",
    ] {
        let symbol = format!("chop_row_rms_{name}_v1");
        let item = catalog.iter().find(|item| item.symbol == symbol).unwrap();
        let kernel_core::contract::RowQuery::RmsNorm {
            width,
            eps,
            kind,
            fused_residual,
            output_quantization: None,
        } = item.query
        else {
            continue;
        };
        let numerics = if kind == RmsNormKind::HfBf16Intermediate {
            RmsNormNumerics::HF_ORDERED
        } else {
            RmsNormNumerics::TREE_256
        };
        for capacity in [1, 32, 64, 8192] {
            let requirement = RmsNormRequirement::bf16(width, eps, kind, fused_residual, numerics);
            let selected = RmsNorm::new(requirement, capacity, RmsNormTarget::SM120F).unwrap();
            for live in [0, 1, capacity] {
                let counts = RuntimeCounts {
                    m: live,
                    batch: 1,
                    context: 0,
                    routed_rows: 0,
                    split_k: 1,
                };
                assert_eq!(
                    selected.plan(live).unwrap(),
                    item.plan(capacity, counts).unwrap()
                );
            }
            assert!(selected.plan(capacity + 1).is_err());
        }
    }
    let mut requirement = RmsNormRequirement::bf16(
        1280,
        1e-6,
        RmsNormKind::Plain,
        false,
        RmsNormNumerics::TREE_256,
    );
    requirement.width = 1536;
    let selected = RmsNorm::new(requirement, 4, RmsNormTarget::SM120F).unwrap();
    assert_eq!(selected.entry("local_shape_v1").max_threads, 256);
    assert_eq!(selected.plan(3).unwrap().grid, [4, 1, 1]);
}
