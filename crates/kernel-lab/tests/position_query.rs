//! F32 tail rotation and shared MXFP4 fake quantization as one prepared op.

use kernel_harness::kit::{self, OpResult, Outcome, cases::position_query};
use kernel_lab::families::position_query as family;
use kernels_sm120::backend::Sm120;

fn report(session: Option<&kernel_lab::Session>) -> kernel_lab::LabResult<kit::Report> {
    let mut backend = Sm120::new().with_position_queries(family::entries());
    for quantizer in family::quantizers() {
        backend = backend.with_block_quantize(quantizer);
    }
    kit::conform(
        &backend,
        &kernel_lab::kit::LabModules,
        session,
        &position_query::family(&family::queries()),
    )
}

fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 2, "{:#?}", report.ops);
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
    assert_eq!(resources.len(), 4);
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
        for forbidden in ["TF32", "LDL", "STL"] {
            assert!(!body.contains(forbidden), "{} has {forbidden}", spec.symbol);
        }
        if spec.symbol.ends_with("_rotate_v2") {
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
