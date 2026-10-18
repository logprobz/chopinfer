//! The model's registered F32 query preparation through the shared backend.

use crate::indexer_query_entries as entries;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::{LabResult, Session, kit};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[test]
fn position_query_model_cubins_match_locks() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("position-model");
    std::fs::create_dir_all(&out).unwrap();
    let specs = entries::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    let lock = LockDir::new(bundle.join("kernels.lock.d"))
        .family(entries::FAMILY, specs.iter().map(|s| s.name.as_str()));
    let built = kernel_build::build_in_families(
        &Toolchain::detect().unwrap(),
        &specs,
        &out,
        &lock,
        BuildOptions {
            lock_update: false,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )
    .unwrap();
    assert_eq!(built.len(), GEOMETRIES.len());
    for cubin in built {
        let bytes = std::fs::read(&cubin.path).unwrap();
        assert!(!bytes.is_empty());
        std::fs::write(out.join(format!("{}.cubin", cubin.name)), bytes).unwrap();
    }
}

struct Modules<'a>(&'a Geometry);

impl kit::Modules for Modules<'_> {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Box<kit::Lookup<'s>>> {
        let module = super::indexer::frozen_module(session, &entries::cubin(self.0)?.name)?;
        Ok(Box::new(move |symbol| {
            module.function(symbol).map_err(|e| e.to_string())
        }))
    }
    fn probes(&self, session: &Session) -> LabResult<Option<kernel_harness::conformance::Probes>> {
        super::indexer::frozen_probes(session).map(Some)
    }
}

fn report(g: &Geometry, session: Option<&Session>) -> LabResult<kit::Report> {
    kit::conform(
        &entries::register(Sm120::new(), g),
        &Modules(g),
        session,
        &kit::cases::position_query::family(&[entries::query(g)]),
    )
}

fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 1, "{:#?}", report.ops);
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    if gpu {
        for op in &report.ops {
            let kit::OpResult::Prepared { checks, .. } = &op.result else {
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
                        .any(|c| c.name == name && matches!(c.outcome, kit::Outcome::Passed(_))),
                    "{}: {name}",
                    op.name
                );
            }
        }
    }
}

#[test]
fn position_query_model_cases_declare_and_prepare() {
    for g in GEOMETRIES {
        check(&report(g, None).unwrap(), false);
    }
}

#[test]
fn position_query_model_cases_pass_on_a_gpu() {
    kernel_harness::gpu_test(1 << 30, |session| {
        for g in GEOMETRIES {
            let report = report(g, Some(session))?;
            session.receipt(
                &format!("position-query-{}", g.id),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            check(&report, true);
        }
        Ok(())
    })
    .expect("registered query preparation GPU conformance");
}
