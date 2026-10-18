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

fn status_cases() -> Vec<kit::Required<'static>> {
    cases::fp4_key_write::STATUS_CASES
        .into_iter()
        .flat_map(cases::fp4_key_write::status_family)
        .collect()
}
#[test]
fn every_key_write_status_case_prepares() {
    let backend = Sm120::new().with_fp4_key_writes(fp4_key_write::entries());
    let report = kit::conform(&backend, &LabModules, None, &status_cases()).unwrap();
    assert_eq!(report.prepared().len(), 40);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
}
#[test]
fn key_write_status_and_saturation_pass_on_a_gpu() {
    gpu_test(1 << 30, |session| {
        let backend = Sm120::new().with_fp4_key_writes(fp4_key_write::entries());
        let report = kit::conform(&backend, &LabModules, Some(session), &status_cases())?;
        assert_eq!(report.prepared().len(), 40);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        assert!(
            report.not_exercised().is_empty(),
            "{:?}",
            report.not_exercised()
        );
        session.receipt(
            "fp4-write-status",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        Ok(())
    })
    .unwrap();
}
#[test]
fn faulty_key_write_status_and_saturation_are_rejected_on_a_gpu() {
    use cases::fp4_key_write::{StatusCase, status_family};
    use kernel_lab::families::pooling_write_mutants::Fault;
    use kit::{OpResult, Outcome};
    gpu_test(1<<30,|session| {
        for (fault,variant,count) in [(Fault::DropInputStatus,StatusCase::InputNonfinite,6),
            (Fault::DropDecodeStatus,StatusCase::DecodeOverflow,2),
            (Fault::MissingStatusOr,StatusCase::PriorFlags,6),
            (Fault::FlagSaturation,StatusCase::Saturation,4),
            (Fault::HoleStatus,StatusCase::Holes,6)] {
            let cases=status_family(variant);
            let real=Sm120::new().with_fp4_key_writes(fp4_key_write::entries());
            let positive=kit::conform(&real,&LabModules,Some(session),&cases)?;
            assert_eq!(positive.prepared().len(),count);
            assert!(positive.failures().is_empty(),"{:?}",positive.failures());
            let report=kit::conform(&mutants::Mutant { backend:real,fault },&LabModules,Some(session),&cases)?;
            assert_eq!(report.prepared().len(),count);
            for op in &report.ops {
                let OpResult::Prepared {checks,..}=&op.result else {panic!("mutant refused")};
                if fault == Fault::HoleStatus {
                    // live=1 contains only valid row 0 and must pass. The
                    // next prefix (live=3) first exposes the hole at row 1.
                    // Pin its U32 status byte, so changing valid rows or
                    // cache payloads cannot satisfy this negative control.
                    assert!(checks.iter().any(|c|c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if e.ends_with(" live 3: status byte 4 is 0x01, the reference 0x00"))),"{fault:?}: {checks:?}");
                } else {
                    assert!(checks.iter().any(|c|c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if ["status","codes","scales"].iter().any(|s|e.contains(s)) && !e.contains("CUDA") && !e.contains("unbound"))),"{fault:?}: {checks:?}");
                }
            }
            session.receipt("fp4-write-status-mutant",BTreeMap::from([("fault".into(),serde_json::json!(fault.tag())),("report".into(),report.json())]))?;
        }
        Ok(())
    }).unwrap();
}
#[test]
fn fused_key_writer_has_no_traps_or_tf32() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = kernel_lab::receipt_root().join(format!("fp4-status-sass-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("write.cubin");
    std::fs::write(&path, kernel_lab::cubin(fp4_key_write::CUBIN).unwrap()).unwrap();
    let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success());
    let sass = String::from_utf8(output.stdout).unwrap();
    for forbidden in ["TRAP", "TF32", "LDL", "STL"] {
        assert!(!sass.contains(forbidden), "contains {forbidden}");
    }
    assert!(sass.contains("ACQBULK") && sass.contains("PREEXIT"));
    std::fs::remove_dir_all(root).unwrap();
}
