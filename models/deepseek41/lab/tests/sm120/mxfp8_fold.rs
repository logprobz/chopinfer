use bundle_deepseek41::sm120::{fold_cubin_for_symbol, mxfp8_fold as registration};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{
    self, Modules, OpResult, Outcome, Report, Required, cases::mxfp8_fold as cases,
};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry;
use std::cell::RefCell;
use std::collections::BTreeMap;

pub(super) struct ModelModules;
impl Modules for ModelModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let loaded = RefCell::new(BTreeMap::new());
        Ok(Box::new(move |symbol| {
            let (name, bytes) = fold_cubin_for_symbol(symbol)
                .ok_or_else(|| format!("model owns no cubin for {symbol}"))?;
            let mut modules = loaded.borrow_mut();
            if !modules.contains_key(name) {
                modules.insert(name, session.load_cubin(bytes)?);
            }
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }
    /// The bundle's locked dependency probes.
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let (bytes, _) = bundle_deepseek41::sm120::dependency_probes();
        let module = session.load_cubin(bytes)?;
        let function = |name| module.function(name).map_err(|e| e.to_string());
        Ok(Some(Probes {
            live: LiveProducer::with_function(session, function(kernels_sm120::probes::LIVE)?)?,
            copy: Some(CopyProducer::with_function(function(
                kernels_sm120::probes::COPY,
            )?)),
        }))
    }
}

fn required<'a>() -> (Sm120, Vec<Required<'a>>, Vec<u32>) {
    let mut backend = Sm120::new().with_packed_fold_weights();
    let mut required = Vec::new();
    let mut capacities = Vec::new();
    for g in geometry::GEOMETRIES {
        for entry in registration::catalog(g).unwrap() {
            assert!(
                fold_cubin_for_symbol(&entry.packed_spec().unwrap().symbol).is_some(),
                "{} is not built",
                entry.symbol
            );
            let q = entry.query;
            // Exercise both sides of the 16-row prefill tile, and every
            // decode/verify row, while retaining production N and K.
            let capacity = q.m_capacity.min(17);
            backend = backend.with_mxfp8_fold(entry);
            required.push(
                Required::new(
                    format!("{} {q:?}", g.id),
                    cases::need(capacity),
                    move |backend, need| backend.mxfp8_fold_gemm(&q, need),
                    move |op| cases::packed_case(q, capacity, op),
                )
                .required(),
            );
            capacities.push(capacity);
        }
    }
    assert!(required.len() > 10);
    (backend, required, capacities)
}

fn check(report: &Report, capacities: &[u32], gpu: bool) {
    assert_eq!(report.ops.len(), capacities.len());
    assert_eq!(report.prepared().len(), capacities.len());
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    for (op, &capacity) in report.ops.iter().zip(capacities) {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required model fold refused")
        };
        for name in ["declaration", "need", "conformance", kit::PDL] {
            if !gpu && !["declaration", "need"].contains(&name) {
                continue;
            }
            let check = checks.iter().find(|c| c.name == name).unwrap();
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{} {name}: {:?}",
                op.name,
                check.outcome
            );
        }
        if gpu {
            for property in cases::need(capacity).exact.claims() {
                let name = kit::properties::name(property);
                let check = checks.iter().find(|c| c.name == name).unwrap();
                assert!(
                    matches!(check.outcome, Outcome::Passed(_)),
                    "{} {name}: {:?}",
                    op.name,
                    check.outcome
                );
            }
        }
    }
}

#[test]
fn model_fold_entries_prepare() {
    let (backend, required, capacities) = required();
    let report = kit::conform(&backend, &ModelModules, None, &required).unwrap();
    check(&report, &capacities, false);
    println!("prepared {} model fold entries", report.prepared().len());
}

#[test]
fn model_fold_gpu() {
    kernel_harness::gpu_test(2 << 30, |session| {
        assert!(geometry::GEOMETRIES.iter().any(|g| g.id == "flash"));
        assert!(
            geometry::GEOMETRIES
                .iter()
                .any(|g| g.id == "synthetic_small"),
            "build the model lab with its synthetic geometry"
        );
        let (backend, required, capacities) = required();
        let report = kit::conform(&backend, &ModelModules, Some(session), &required)?;
        check(&report, &capacities, true);
        session.receipt(
            "model-mxfp8-fold",
            BTreeMap::from([
                ("report".into(), report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}
