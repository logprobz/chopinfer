//! The registered model instances, with the same own-F32 selection oracle
//! as the shared family: the dense selection of the Full sources. The
//! compact selection and the block-ID producer are the shared engine's
//! cases. GPU jobs load frozen cubins and never compile.

use crate::indexer_entries as entries;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::{LabResult, Session, kit};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[test]
fn position_model_cubins_match_locks() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle");
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("position-model");
    std::fs::create_dir_all(&out).unwrap();
    let native = entries::cubins(GEOMETRIES).unwrap();
    let comparison = entries::bf16_cubins(GEOMETRIES).unwrap();
    let lock = LockDir::new(bundle.join("kernels.lock.d"))
        .family(entries::FAMILY, native.iter().map(|(_, s)| s.name.as_str()))
        .family(
            entries::BF16_FAMILY,
            comparison.iter().map(|(_, s)| s.name.as_str()),
        );
    let specs = native
        .into_iter()
        .chain(comparison)
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
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
    .expect("registered indexer cubins match their locks");
    // Each geometry's native and comparison dense/compact cubins.
    assert_eq!(built.len(), 4 * GEOMETRIES.len());
    for cubin in built {
        let bytes = std::fs::read(&cubin.path).unwrap();
        assert!(!bytes.is_empty());
        std::fs::write(out.join(format!("{}.cubin", cubin.name)), bytes).unwrap();
    }
}

struct FrozenModules<'a>(&'a Geometry);

pub(super) fn frozen_module(session: &Session, name: &str) -> LabResult<runtime_cuda::Module> {
    let root = std::env::var_os("CHOPINFER_INDEXER_CUBINS")
        .ok_or("GPU model checks need the frozen indexer cubin directory")?;
    let bytes = std::fs::read(PathBuf::from(root).join(format!("{name}.cubin")))
        .map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

pub(super) fn frozen_probes(session: &Session) -> LabResult<kernel_harness::conformance::Probes> {
    use kernel_harness::poison::{CopyProducer, LiveProducer};
    let live = frozen_module(session, "probe_pdl_prefetch_sm120")?;
    let copy = frozen_module(session, "latent_attention_latent_h16_l512_sm120")?;
    let live = live
        .function("chop_probe_pdl_prefetch_producer_v1")
        .map_err(|e| e.to_string())?;
    Ok(kernel_harness::conformance::Probes {
        live: LiveProducer::with_function(session, live)?,
        copy: Some(CopyProducer::new(&copy)?),
    })
}

impl kit::Modules for FrozenModules<'_> {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Box<kit::Lookup<'s>>> {
        let name = entries::cubin(self.0)?.name;
        let module = frozen_module(session, &name)?;
        Ok(Box::new(move |symbol| {
            module.function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(&self, session: &Session) -> LabResult<Option<kernel_harness::conformance::Probes>> {
        frozen_probes(session).map(Some)
    }
}

fn report(g: &Geometry, session: Option<&Session>) -> LabResult<kit::Report> {
    let entries = entries::entries(g);
    let cases = kit::cases::position_indexer::family(&entries, &[]);
    let backend = crate::indexer_entries::register(Sm120::new(), g);
    kit::conform(&backend, &FrozenModules(g), session, &cases)
}

fn check(g: &Geometry, report: &kit::Report, gpu: bool) {
    // Each dense entry's exact and multi-head case.
    let cases = 2 * entries::entries(g).len();
    assert_eq!(report.prepared().len(), cases, "{:#?}", report.ops);
    assert!(report.failures().is_empty(), "{:#?}", report.failures());
    if gpu {
        for op in &report.ops {
            let kit::OpResult::Prepared { checks, .. } = &op.result else {
                panic!("required indexer case refused")
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
fn position_model_cases_declare_and_prepare() {
    for g in GEOMETRIES {
        check(g, &report(g, None).unwrap(), false);
    }
}

#[test]
fn position_model_cases_pass_on_a_gpu() {
    kernel_harness::gpu_test(4 << 30, |session| {
        for g in GEOMETRIES {
            let report = report(g, Some(session))?;
            session.receipt(
                &format!("position-indexer-{}", g.id),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            check(g, &report, true);
        }
        Ok(())
    })
    .expect("registered indexer GPU conformance");
}
