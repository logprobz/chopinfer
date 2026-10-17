//! Initial native-format attention smoke, with exact unit probabilities.
//! This isolates storage and split ownership from the later nonzero-Q
//! numerical and accumulation-chain gates.

use kernel_lab::families::latent_sources as family;
use kernel_lab::kit::{self, LabModules, OpResult, Outcome, cases};
use kernels_sm120::backend::Sm120;

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
            panic!("required native op refused")
        };
        if gpu {
            for check in checks {
                assert!(
                    matches!(check.outcome, Outcome::Passed(_)),
                    "{}: {check:?}",
                    op.name
                );
            }
        }
        for name in if gpu {
            &["declaration", "conformance", kit::PDL][..]
        } else {
            &["declaration"][..]
        } {
            let check = checks
                .iter()
                .find(|c| c.name == *name)
                .expect("required check exists");
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {check:?}",
                op.name
            );
        }
    }
}

#[test]
fn native_sources_prepare_every_required_case() {
    for cases in [
        cases::latent_sources::family(&family::catalog()),
        cases::latent_sources::numerical_family(&family::catalog()),
    ] {
        let report = kit::conform(&backend(), &LabModules, None, &cases).unwrap();
        check(&report, false);
    }
}

#[test]
fn native_sources_nonzero_queries_meet_f64_bounds() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = kit::conform(
            &backend(),
            &LabModules,
            Some(session),
            &cases::latent_sources::numerical_family(&family::catalog()),
        )?;
        session.receipt(
            "native-latent-numerical",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        check(&report, true);
        Ok(())
    })
    .expect("native source numerical gates");
}

#[test]
fn native_sources_pass_storage_split_and_replay_smoke() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = kit::conform(
            &backend(),
            &LabModules,
            Some(session),
            &cases::latent_sources::family(&family::catalog()),
        )?;
        session.receipt(
            "native-latent-source-smoke",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        check(&report, true);
        Ok(())
    })
    .expect("native source smoke");
}
