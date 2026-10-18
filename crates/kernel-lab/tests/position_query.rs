//! F32 tail rotation and shared MXFP4 fake quantization as one prepared op.

use kernel_harness::kit::{self, OpResult, Outcome, cases::position_query};
use kernel_lab::families::position_query as family;
use kernels_sm120::backend::Sm120;

fn report(session: Option<&kernel_lab::Session>) -> kernel_lab::LabResult<kit::Report> {
    let mut backend = Sm120::new().with_position_queries(family::entries());
    for quantizer in family::quantizers() {
        backend = backend.with_block_quantize(quantizer);
    }
    kit::conform(&backend, &kernel_lab::kit::LabModules, session, &cases())
}

fn cases() -> Vec<kit::Required<'static>> {
    let queries = family::queries();
    let mut cases = position_query::family(&queries);
    for q in queries {
        for kind in [
            position_query::Witness::Bounds,
            position_query::Witness::Extreme,
            position_query::Witness::Prior,
        ] {
            cases.push(position_query::witness(q, kind));
        }
    }
    cases
}

fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 8, "{:#?}", report.ops);
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("query preparation refused")
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
                    "{}: {name}",
                    op.name
                );
            }
        }
    }
}

#[test]
fn position_query_cases_declare_and_prepare() {
    check(&report(None).unwrap(), false);
}

#[test]
fn position_query_cubin_has_f32_rotation_and_no_tf32_spills_or_stack() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("chopinfer-query-audit-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("query.cubin");
    std::fs::write(&path, kernel_lab::cubin(family::CUBIN).unwrap()).unwrap();
    let specs = family::cubins();
    let resources = kernel_build::audit_cubin(&toolchain, &specs[0], &path).unwrap();
    assert_eq!(resources.len(), 6);
    for entry in resources {
        assert_eq!(
            (entry.stack_bytes, entry.local_bytes),
            (0, 0),
            "{}",
            entry.symbol
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
        if spec.symbol.contains("_rotate") {
            assert!(
                !body.contains("BF16"),
                "rotation prematurely rounds to BF16"
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn position_query_cases_pass_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let report = report(Some(session))?;
        session.receipt(
            "position-query",
            std::collections::BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        Ok(())
    })
    .expect("query preparation GPU conformance");
}

struct IgnoreBounds {
    inner: Sm120,
}
struct IgnoreBoundsOp {
    inner: kernel_core::backend::Prepared<
        kernel_core::pooled_indexer::PositionQueryBuffers<kernel_core::backend::View>,
    >,
    info: kernel_core::backend::OpInfo,
}
fn ignored(symbol: &str) -> String {
    symbol.replace("_rotate_v3", "_rotate_ignore_bounds_v3")
}
impl
    kernel_core::backend::PreparedOp<
        kernel_core::pooled_indexer::PositionQueryBuffers<kernel_core::backend::View>,
    > for IgnoreBoundsOp
{
    fn info(&self) -> &kernel_core::backend::OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &kernel_core::pooled_indexer::PositionQueryBuffers<kernel_core::backend::View>,
        workspace: kernel_core::backend::View,
        counts: kernel_core::backend::Counts,
    ) -> Result<Vec<kernel_core::backend::Launch>, kernel_core::contract::Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        launches[0].symbol = ignored(&launches[0].symbol);
        Ok(launches)
    }
}
impl kernel_core::backend::Backend for IgnoreBounds {
    fn name(&self) -> &str {
        "query-bounds-control"
    }
    fn position_indexer_query(
        &self,
        q: &kernel_core::pooled_indexer::PositionIndexerQuery,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::pooled_indexer::PositionQueryBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        let inner = self.inner.position_indexer_query(q, need)?;
        let mut info = inner.info().clone();
        info.symbols[0].0 = ignored(&info.symbols[0].0);
        Ok(Box::new(IgnoreBoundsOp { inner, info }))
    }
}
#[test]
fn position_query_bounds_mutant_fails_on_a_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let mut inner=Sm120::new().with_position_queries(family::entries());
        for q in family::quantizers() { inner=inner.with_block_quantize(q); }
        let mutant=IgnoreBounds {inner};
        for query in family::queries() {
            for kind in [position_query::Witness::Bounds,position_query::Witness::Extreme] {
                let cases=vec![position_query::witness(query,kind)];
                let report=kit::conform(&mutant,&kernel_lab::kit::LabModules,Some(session),&cases)?;
                assert_eq!(report.prepared().len(),1);
                assert!(report.ops.iter().any(|op| {
                    let OpResult::Prepared { checks,.. }=&op.result else { return false; };
                    checks.iter().any(|c| c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(reason) if reason.contains("quantizer_status") && reason.contains("byte")))
                }),"bounds mutant missed status assertion: {report:?}");
                session.receipt(&format!("position-query-control-h{}-{kind:?}",query.heads), std::collections::BTreeMap::from([("report".into(),report.json())]))?;
            }
        }
        Ok(())
    }).expect("position query bounds control");
}
