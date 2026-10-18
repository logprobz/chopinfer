//! Required FP4 writer kit cases, exact against the shared fake-quant rule.

use kernel_lab::families::fp4_key_write;
use kernel_lab::kit::{self, LabModules, cases};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn run(session: Option<&Session>) -> LabResult<()> {
    let ops = cases::fp4_key_write::family();
    assert_eq!(ops.len(), 6);
    let backend = Sm120::new().with_fp4_key_writes(fp4_key_write::entries());
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
            "fp4-key-write",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
    }
    Ok(())
}

#[test]
fn fp4_write_family_prepares_and_declares_every_required_case() {
    run(None).unwrap();
}

#[test]
fn fp4_write_family_passes_on_a_gpu() {
    gpu_test(1 << 30, |session| run(Some(session))).unwrap();
}

#[path = "common/pooling_write_mutants.rs"]
mod mutants;

#[test]
fn swapped_nibbles_and_wrong_pages_are_rejected_on_a_gpu() {
    use cases::fp4_key_write::{Query, queries};
    use kernel_core::compressed_kv::CompressedKvStorage;
    use kernel_lab::families::pooling_write_mutants::Fault;
    gpu_test(1 << 30, |session| {
        run(Some(session))?;
        for fault in [Fault::SwapNibbles, Fault::WrongPage] {
            let backend = mutants::Mutant {
                backend: Sm120::new().with_fp4_key_writes(fp4_key_write::entries()),
                fault,
            };
            let queries = queries().into_iter().filter(|q| {
                !matches!(q,
                Query::Compressed(q) if q.storage == CompressedKvStorage::Bf16)
            });
            let ops = cases::fp4_key_write::family_for(queries);
            let report = kit::conform(&backend, &LabModules, Some(session), &ops)?;
            mutants::rejected(&report, 4, "codes");
            session.receipt(
                "fp4-write-device-mutants",
                BTreeMap::from([
                    ("fault".into(), serde_json::json!(fault.tag())),
                    ("report".into(), report.json()),
                ]),
            )?;
        }
        Ok(())
    })
    .unwrap();
}
