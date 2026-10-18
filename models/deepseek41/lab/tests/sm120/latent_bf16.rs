//! The comparison layout preserves every native model attention result.
use super::latent::Modules as ModelModules;
use bundle_deepseek41::sm120::latent;
use kernel_core::backend::{Backend, Need};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernel_harness::kit::{self, Modules, OpResult, Outcome, cases};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::GEOMETRIES;

fn required<'a>(entries: &[kernels_sm120::latent::sources::Entry]) -> Vec<kit::Required<'a>> {
    entries
        .iter()
        .flat_map(|entry| {
            if matches!(
                entry.query,
                LatentAttentionQuery::Attention {
                    second_fp4_keys: true,
                    ..
                }
            ) {
                cases::latent_sources::paged_family(std::slice::from_ref(entry), true)
            } else {
                cases::latent_sources::numerical_family(std::slice::from_ref(entry))
            }
        })
        .collect()
}
#[test]
fn bf16_model_cases_prepare() {
    let mut count = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let entries = latent::bf16_entries(g, ranks).unwrap();
            let backend = latent::register(kernels_sm120::backend::Sm120::new(), g, ranks).unwrap();
            let report = kit::conform(&backend, &ModelModules, None, &required(&entries)).unwrap();
            assert!(report.failures().is_empty(), "{:?}", report.failures());
            assert_eq!(
                report.prepared().len(),
                7 + 3 * usize::from(g.shape.compress_ratios.contains(&2))
            );
            for op in &report.ops {
                let OpResult::Prepared { checks, .. } = &op.result else {
                    panic!("BF16 model case refused")
                };
                assert!(checks.iter().any(|c| c.name == "declaration" && matches!(c.outcome, Outcome::Passed(_))));
            }
            count += report.prepared().len();
        }
    }
    assert_eq!(
        count,
        GEOMETRIES
            .iter()
            .map(|g| 21 + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
    );
}
#[test]
fn bf16_model_instances_preserve_native_bits() {
    kernel_harness::gpu_test(1 << 30, |session| {
        let functions = ModelModules.load(session)?;
        let mut count = 0;
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                let backend = latent::register(kernels_sm120::backend::Sm120::new(), g, ranks)?;
                for (native, bf16) in latent::entries(g, ranks)?
                    .into_iter()
                    .zip(latent::bf16_entries(g, ranks)?)
                {
                    assert!(matches!(
                        bf16.query,
                        LatentAttentionQuery::Attention {
                            kv: LatentKvFormat::Bf16,
                            ..
                        }
                    ));
                    let mut before = None;
                    for entry in [native, bf16] {
                        let LatentAttentionQuery::Attention {
                            second_fp4_keys, ..
                        } = entry.query
                        else {
                            unreachable!()
                        };
                        let prepared = backend
                            .latent_attention_sources(&entry.query, Need::rows(4))
                            .unwrap();
                        let case = if second_fp4_keys {
                            cases::latent_sources::paged_case(entry.query, 4, prepared)?
                        } else {
                            cases::latent_sources::numerical_case(entry.query, 4, prepared)?
                        };
                        let images = case.images();
                        let actual = case.launch(session, functions.as_ref(), &images, 4)?;
                        case.judge_outputs(&images, 4, &actual)?;
                        if let Some(previous) = &before {
                            assert_eq!(&actual["output"], previous, "{}", entry.name);
                        }
                        before = Some(actual["output"].clone());
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(
            count,
            GEOMETRIES
                .iter()
                .map(|g| 21 + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
                .sum::<usize>()
        );
        session.receipt(
            "model-latent-bf16-native-pairs",
            std::collections::BTreeMap::from([("bitwise_pairs".into(), serde_json::json!(count))]),
        )?;
        Ok(())
    })
    .expect("BF16 comparison storage preserves native model output bits");
}
