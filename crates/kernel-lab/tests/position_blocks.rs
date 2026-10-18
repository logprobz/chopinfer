//! Compact Position v3 conformance, dense equivalence and validation controls.
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::Refusal,
    position_blocks::{PositionBlockIndexerQuery, PositionBlockSelectBuffers},
};
use kernel_harness::kit::{
    self, OpResult, Outcome,
    cases::position_blocks::{self as cases, Witness},
};
use kernel_lab::families::position_blocks as family;
use kernels_sm120::{backend::Sm120, pooled_indexer::position_blocks::control_symbol};

#[path = "position_blocks/chain.rs"]
mod chain;

fn backend() -> Sm120 {
    Sm120::new()
        .with_position_block_indexer(family::entries())
        .with_indexer_candidate_block_ids(family::candidates())
        .with_position_indexer(family::dense())
        .with_indexer_candidate_blocks(family::dense_candidates())
}
fn ops() -> Vec<kit::Required<'static>> {
    let mut ops = cases::family(&family::entries());
    for witness in [
        Witness::Unsorted,
        Witness::Duplicate,
        Witness::Stale,
        Witness::DroppedPin,
        Witness::NanWeight,
        Witness::NanDot,
        Witness::Overflow,
    ] {
        ops.push(cases::witness(&family::entries()[0], witness));
    }
    ops
}
fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 9);
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("compact op refused")
            };
            for name in [
                "conformance",
                kit::PDL,
                "property rows",
                "property chunk",
                "property batch",
            ] {
                assert!(
                    checks
                        .iter()
                        .any(|c| c.name == name && matches!(c.outcome, Outcome::Passed(_))),
                    "{} {name}: {checks:?}",
                    op.name
                );
            }
        }
    }
}
#[test]
fn compact_cases_declare_and_prepare() {
    check(
        &kit::conform(&backend(), &kernel_lab::kit::LabModules, None, &ops()).unwrap(),
        false,
    );
}
#[test]
fn compact_cases_pass_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = kit::conform(
            &backend(),
            &kernel_lab::kit::LabModules,
            Some(session),
            &ops(),
        )?;
        session.receipt(
            "compact-position-conformance",
            std::collections::BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        Ok(())
    })
    .expect("compact position conformance");
}
#[test]
fn compact_cubin_has_no_tf32_spills_or_stack_and_keeps_pdl_waits() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("chopinfer-compact-audit-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("compact.cubin");
    std::fs::write(&path, kernel_lab::cubin(family::CUBIN).unwrap()).unwrap();
    let specs = family::cubins();
    let resources = kernel_build::audit_cubin(&toolchain, &specs[0], &path).unwrap();
    assert_eq!(resources.len(), 38);
    for resource in resources {
        assert_eq!(
            (resource.local_bytes, resource.stack_bytes),
            (0, 0),
            "{}",
            resource.symbol
        );
    }
    let result = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    let sass = String::from_utf8(result.stdout).unwrap();
    for spec in &specs[0].entries {
        let label = format!("Function : {}\n", spec.symbol);
        let body = &sass[sass.find(&label).unwrap() + label.len()..];
        let body = &body[..body.find("Function : ").unwrap_or(body.len())];
        for required in ["PREEXIT", "ACQBULK"] {
            assert!(body.contains(required), "{} lacks {required}", spec.symbol);
        }
        for forbidden in ["TF32", "LDL", "STL", "TRAP"] {
            assert!(!body.contains(forbidden), "{} has {forbidden}", spec.symbol);
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

struct Mutant {
    inner: Sm120,
    control: u32,
}
struct MutantOp {
    inner: Prepared<PositionBlockSelectBuffers<View>>,
    info: OpInfo,
    control: u32,
}
impl PreparedOp<PositionBlockSelectBuffers<View>> for MutantOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &PositionBlockSelectBuffers<View>,
        w: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, w, counts)?;
        for launch in &mut launches {
            launch.symbol = control_symbol(launch.symbol.clone(), self.control)?;
        }
        Ok(launches)
    }
}
impl Backend for Mutant {
    fn name(&self) -> &str {
        "compact-control"
    }
    fn position_block_indexer(
        &self,
        q: &PositionBlockIndexerQuery,
        top: u32,
        width: u32,
        need: Need,
    ) -> Result<Prepared<PositionBlockSelectBuffers<View>>, Refused> {
        let inner = self.inner.position_block_indexer(q, top, width, need)?;
        let mut info = inner.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = control_symbol(symbol.clone(), self.control)?;
        }
        Ok(Box::new(MutantOp {
            inner,
            info,
            control: self.control,
        }))
    }
}
#[test]
fn compact_validation_mutants_fail_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        for (witness, control) in [(Witness::Unsorted,1), (Witness::Duplicate,1), (Witness::Stale,2), (Witness::DroppedPin,3)] {
            let mutant = Mutant { inner: backend(), control };
            let ops = vec![cases::witness(&family::entries()[0], witness)];
            let report = kit::conform(&mutant, &kernel_lab::kit::LabModules, Some(session), &ops)?;
            assert_eq!(report.prepared().len(), 1);
            let intended = report.ops.iter().any(|op| {
                let OpResult::Prepared { checks, .. } = &op.result else { return false; };
                checks.iter().any(|c| c.name == "conformance" && matches!(&c.outcome, Outcome::Failed(reason) if reason.contains("status") && reason.contains("byte")))
            });
            session.receipt(&format!("compact-control-{witness:?}"), std::collections::BTreeMap::from([("report".into(), report.json())]))?;
            assert!(intended, "{witness:?} missed its validation assertion: {report:?}");
        }
        Ok(())
    }).expect("compact validation controls");
}
