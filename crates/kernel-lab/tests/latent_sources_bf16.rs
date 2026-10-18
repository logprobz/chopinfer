//! BF16 comparison storage reads the same operands through the same sums.
use kernel_core::backend::{Backend, Need};
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery};
use kernel_lab::families::latent_sources_bf16 as family;
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, cases};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn backend() -> Sm120 {
    family::native_catalog()
        .into_iter()
        .chain(family::catalog())
        .fold(Sm120::new(), Sm120::with_latent_sources)
}
fn required<'a>(entries: &[kernels_sm120::latent::sources::Entry]) -> Vec<kit::Required<'a>> {
    entries
        .iter()
        .flat_map(|e| {
            if matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    second_fp4_keys: true,
                    ..
                }
            ) {
                cases::latent_sources::paged_family(std::slice::from_ref(e), true)
            } else {
                cases::latent_sources::numerical_family(std::slice::from_ref(e))
            }
        })
        .collect()
}
fn check(report: &kit::Report, gpu: bool) {
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.prepared().len(), family::REQUIRED);
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required BF16 source refused")
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
fn bf16_native_sources_prepare_every_required_case() {
    check(
        &kit::conform(&backend(), &LabModules, None, &required(&family::catalog())).unwrap(),
        false,
    );
}
#[test]
fn bf16_native_sources_meet_f64_bounds_replay_and_pdl() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = kit::conform(
            &backend(),
            &LabModules,
            Some(session),
            &required(&family::catalog()),
        )?;
        check(&report, true);
        session.receipt(
            "latent-bf16-native-conformance",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        Ok(())
    })
    .expect("BF16 native operand source conformance");
}
#[test]
fn bf16_native_sources_preserve_served_and_partial_bits() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        let mut paired = 0;
        for native in family::native_catalog() {
            let bf16 = family::paired(&native);
            let mut before = None;
            for entry in [native, bf16] {
                let LatentAttentionQuery::Attention {
                    output,
                    second_fp4_keys,
                    ..
                } = entry.query
                else {
                    unreachable!()
                };
                let prepared = backend()
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
                // The oracle checks empty partials as (-inf, 0, 0).
                // Compare those sentinels byte for byte with the finite rows.
                let names = if output == LatentAttentionOutput::Partial {
                    &["partial.max", "partial.den", "partial.num"][..]
                } else {
                    &["output"][..]
                };
                let bits = names
                    .iter()
                    .map(|name| actual[*name].clone())
                    .collect::<Vec<_>>();
                if let Some(old) = &before {
                    assert_eq!(&bits, old, "{}", entry.name);
                }
                before = Some(bits);
            }
            paired += 1;
        }
        assert_eq!(paired, family::REQUIRED);
        session.receipt(
            "latent-bf16-native-pairs",
            BTreeMap::from([("bitwise_pairs".into(), serde_json::json!(paired))]),
        )?;
        Ok(())
    })
    .expect("BF16 comparison storage preserves native bits");
}

#[path = "latent_sources/address.rs"]
mod address;
