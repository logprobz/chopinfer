//! Required F32 pair-pooling kit cases over the lab's own lock family.

use kernel_lab::families::gated_pooling_f32;
use kernel_lab::kit::{self, LabModules, cases};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn run(session: Option<&Session>) -> LabResult<()> {
    let entries = gated_pooling_f32::entries();
    let mut ops = cases::gated_pooling::family(&entries);
    ops.extend(cases::gated_pooling::steps::family(&entries));
    assert_eq!(ops.len(), 6);
    let backend = Sm120::new().with_gated_pooling(entries.clone());
    let report = kit::conform(&backend, &LabModules, session, &ops)?;
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.prepared().len(), 6);
    if let Some(session) = session {
        assert!(
            report.not_exercised().is_empty(),
            "{:?}",
            report.not_exercised()
        );
        session.receipt(
            "gated-pooling-f32",
            BTreeMap::from([
                ("report".into(), report.json()),
                (
                    "prefill".into(),
                    cases::gated_pooling::steps::prefill_splits(
                        &backend,
                        &LabModules,
                        session,
                        &entries,
                    )?,
                ),
            ]),
        )?;
    }
    Ok(())
}

#[test]
fn pooling_family_prepares_and_declares_every_required_case() {
    run(None).unwrap();
}

#[test]
fn pooling_family_passes_on_a_gpu() {
    gpu_test(1 << 30, |session| run(Some(session))).unwrap();
}

#[path = "common/pooling_write_mutants.rs"]
mod mutants;

#[test]
fn wrong_gate_and_dropped_split_tail_are_rejected_on_a_gpu() {
    use kernel_lab::families::pooling_write_mutants::Fault;
    gpu_test(1 << 30, |session| {
        let entries = gated_pooling_f32::entries();
        // Positive control: the same cases and split checks must pass first.
        run(Some(session))?;
        let wrong_gate = mutants::Mutant {
            backend: Sm120::new().with_gated_pooling(entries.clone()),
            fault: Fault::WrongGate,
        };
        let report = kit::conform(
            &wrong_gate,
            &LabModules,
            Some(session),
            &cases::gated_pooling::family(&entries),
        )?;
        mutants::rejected(&report, 2, "pooled");
        let dropped = mutants::Mutant {
            backend: Sm120::new().with_gated_pooling(entries.clone()),
            fault: Fault::DropSplitTail,
        };
        let error =
            cases::gated_pooling::steps::prefill_splits(&dropped, &LabModules, session, &entries)
                .expect_err("device mutant dropped the split tail but passed");
        assert!(error.contains("split 4, row 5: pooled differs"), "{error}");
        session.receipt(
            "pooling-device-mutants",
            BTreeMap::from([
                ("wrong_gate".into(), report.json()),
                ("dropped_split_tail".into(), serde_json::json!(error)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}
