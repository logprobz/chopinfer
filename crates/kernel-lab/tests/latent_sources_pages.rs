//! Native compressed source pages, boundaries, holes and address mutants.
use kernel_core::backend::{Backend, Need};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery};
use kernel_lab::families::latent_sources_pages as family;
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, cases};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn backend() -> Sm120 {
    family::catalog()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_latent_sources)
}
fn check(report: &kit::Report, gpu: bool) {
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.ops.len(), family::REQUIRED);
    assert_eq!(report.prepared().len(), family::REQUIRED);
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required page entry refused")
        };
        for check in checks.iter().filter(|c| gpu || c.name == "declaration") {
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {check:?}",
                op.name
            );
        }
    }
}
#[test]
fn native_page_geometries_prepare_all_required_cases() {
    for nonzero in [false, true] {
        check(
            &kit::conform(
                &backend(),
                &LabModules,
                None,
                &cases::latent_sources::paged_family(&family::catalog(), nonzero),
            )
            .unwrap(),
            false,
        );
    }
}
#[test]
fn native_page_geometries_meet_f64_bounds_with_replay_and_pdl() {
    kernel_lab::gpu_test(1 << 30, |session| {
        for nonzero in [false, true] {
            let report = kit::conform(
                &backend(),
                &LabModules,
                Some(session),
                &cases::latent_sources::paged_family(&family::catalog(), nonzero),
            )?;
            check(&report, true);
            session.receipt(
                "native-latent-pages",
                BTreeMap::from([
                    ("nonzero".into(), serde_json::json!(nonzero)),
                    ("report".into(), report.json()),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("page-specific native source conformance");
}

#[test]
fn native_page_address_mutants_are_rejected_without_device_faults() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = family::catalog()
            .into_iter()
            .find(|e| {
                matches!(
                    e.query,
                    LatentAttentionQuery::Attention {
                        page_size: 32,
                        phase: AttentionPhase::Decode,
                        output: LatentAttentionOutput::Normalized,
                        ..
                    }
                )
            })
            .unwrap();
        let prepared = backend()
            .latent_attention_sources(&entry.query, Need::rows(4))
            .unwrap();
        let case = cases::latent_sources::paged_case(entry.query, 4, prepared)?;
        let mut images = case.images();
        let (width, page) = (512, 32);
        let stride = page * (width / 2 + width / 16) + 64;
        let mut pool = vec![0x38; images.bytes("compressed.cache.pool")?.len()];
        // One extra physical page backs the wrong scale-offset reads.
        // Every mutant address contains finite bytes, including guard bytes.
        for physical in 0..pool.len() / stride {
            pool[physical * stride..physical * stride + page * width / 2]
                .fill([0x22, 0x44, 0x66, 0x77][physical % 4]);
        }
        let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        images = images
            .with("compressed.cache.pool", pool)
            .with("q", vec![0; 4 * 4 * width * 2])
            .with("row_batch", i32s(&[0; 4]))
            .with("window.key_lists", i32s(&vec![-1; 4 * 128]))
            .with("sinks", vec![0; 4 * 4])
            .with(
                "cos_sin",
                [1.0_f32, 0.0]
                    .repeat(4 * 32)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            );
        let functions = LabModules.load(session)?;
        let correct = case.launch(session, functions.as_ref(), &images, 1)?;
        case.judge_outputs(&images, 1, &correct)?;
        let values = correct["output"]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        assert!(
            values[..4 * width]
                .iter()
                .all(|&v| v == f32_to_bf16(11.0 / 4.0))
        );
        let mut rejected = Vec::new();
        for mutant in family::mutants(&entry) {
            let source = entry.symbol();
            let wrong = |symbol: &str| {
                functions(if symbol == source {
                    &mutant.symbol
                } else {
                    symbol
                })
            };
            let actual = case.launch(session, &wrong, &images, 1)?;
            assert!(
                actual["output"][..4 * width * 2]
                    .chunks_exact(2)
                    .all(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])).is_finite())
            );
            let failure = case
                .judge_outputs(&images, 1, &actual)
                .expect_err("address mutant must fail the original f64 bound");
            rejected.push((mutant.symbol, failure));
        }
        assert_eq!(rejected.len(), 3);
        session.receipt(
            "native-latent-page-mutants",
            BTreeMap::from([("rejected".into(), serde_json::json!(rejected))]),
        )?;
        Ok(())
    })
    .expect("safe address mutants");
}

#[path = "latent_sources/address.rs"]
mod address;
