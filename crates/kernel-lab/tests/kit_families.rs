//! The conformance kit's cases of the MoE (and its routed experts), gated
//! and sparse attention, Sinkhorn, multimodal encoder and context-parallel
//! families, through SM120, one required op set per family.
//!
//! The CPU tests prepare every op: SM120 prepares each one from the lab's
//! catalogs and cubins and its case declares. The GPU tests run every check
//! (`CHOPINFER_KERNEL_GPU`) and skip without a GPU.

use kernel_core::backend::Backend;
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, Report, Required};
use kernel_lab::{LabResult, Session, gpu_test};
use serde_json::json;
use std::collections::BTreeMap;

/// SM120's MoE ops over the lab's catalogs and chained cubins.
fn moe() -> kernels_sm120::backend::Sm120 {
    kit::cases::moe::sm120(kernel_lab::catalogs(), kernel_lab::lab_resources())
}

/// One family's SM120 selection and required ops.
struct Family {
    name: &'static str,
    sm120: fn() -> LabResult<kernels_sm120::backend::Sm120>,
    required: fn() -> LabResult<Vec<Required<'static>>>,
}

const FAMILIES: &[Family] = &[
    Family {
        name: "context parallel",
        sm120: || {
            Ok(kit::cases::context_parallel::sm120(
                kernel_lab::families::context_parallel::shapes::lab_entries(),
            ))
        },
        required: || {
            Ok(kit::cases::context_parallel::required(
                &kernel_lab::families::context_parallel::shapes::lab_entries(),
            ))
        },
    },
    Family {
        name: "gated sparse",
        sm120: || Ok(kit::cases::gated_sparse::sm120(kernel_lab::catalogs())),
        required: || Ok(kit::cases::gated_sparse::required(kernel_lab::catalogs())),
    },
    Family {
        name: "moe",
        sm120: || Ok(moe()),
        required: || Ok(kit::cases::moe::required(kernel_lab::catalogs())),
    },
    Family {
        name: "moe routed split",
        sm120: || Ok(moe()),
        required: || Ok(kit::cases::moe_routed::required_split()),
    },
    Family {
        name: "multimodal",
        sm120: || {
            Ok(kit::cases::multimodal::sm120(
                kernel_lab::families::encoder_vision::shapes::entries(),
            ))
        },
        required: || {
            Ok(kit::cases::multimodal::required(
                &kernel_lab::families::encoder_vision::shapes::entries(),
            ))
        },
    },
    Family {
        name: "sinkhorn",
        sm120: || Ok(kit::cases::sinkhorn::sm120(false)),
        required: || Ok(kit::cases::sinkhorn::required()),
    },
    Family {
        name: "sinkhorn split",
        sm120: || Ok(kit::cases::sinkhorn::sm120(true)),
        required: || Ok(kit::cases::sinkhorn::required()),
    },
    Family {
        name: "sinkhorn tiled",
        sm120: || Ok(kit::cases::sinkhorn::sm120_tiled()),
        required: || Ok(kit::cases::sinkhorn::required_tiled()),
    },
    Family {
        name: "sinkhorn mix norm",
        sm120: || Ok(kit::cases::sinkhorn::sm120_mix_norm()),
        required: || Ok(kit::cases::sinkhorn::required_mix_norm()),
    },
];

fn family(name: &str) -> &'static Family {
    FAMILIES
        .iter()
        .find(|family| family.name == name)
        .expect("a family")
}

fn run(
    family: &Family,
    backend: &dyn Backend,
    modules: &dyn Modules,
    session: Option<&Session>,
) -> LabResult<Report> {
    kit::conform(backend, modules, session, &(family.required)()?)
}

/// SM120 prepares every op the family marks required, each case it
/// prepares declares and every launched symbol is in a lab cubin; on a GPU
/// every prepared op also passes its conformance run.
fn check_sm120(report: &Report, gpu: bool) {
    let failures = report.failures();
    assert!(failures.is_empty(), "{}: {failures:#?}", report.backend);
    assert!(!report.prepared().is_empty());
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            continue;
        };
        let outcome = |name: &str| {
            &checks
                .iter()
                .find(|check| check.name == name)
                .expect("every op has the check")
                .outcome
        };
        let Outcome::Passed(declared) = outcome("declaration") else {
            panic!("{}: {:?}", op.name, outcome("declaration"));
        };
        for symbol in declared["launches"].as_array().expect("launches") {
            let symbol = symbol.as_str().expect("a symbol");
            assert!(
                LabModules::holds(symbol),
                "{}: no lab cubin holds {symbol}",
                op.name
            );
        }
        if gpu {
            assert!(
                matches!(outcome("conformance"), Outcome::Passed(_)),
                "{}: {:?}",
                op.name,
                outcome("conformance")
            );
        }
    }
}

fn cpu(name: &str) {
    let family = family(name);
    let report =
        run(family, &(family.sm120)().expect("SM120"), &LabModules, None).expect("SM120 report");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    assert!(!report.ops.is_empty());
    check_sm120(&report, false);
}

/// A family's receipt kind: receipt kinds take ASCII letters, digits, `-`
/// and `_` only.
fn receipt_kind(name: &str) -> String {
    format!("conformance-kit-{}", name.replace(' ', "-"))
}

#[test]
fn every_family_names_a_valid_receipt_kind() {
    for family in FAMILIES {
        let kind = receipt_kind(family.name);
        assert!(
            kind.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "{kind}"
        );
    }
}

fn gpu(name: &str) {
    gpu_with(name, 4 << 30);
}

/// [`gpu`] on a device with at least `bytes` free.
fn gpu_with(name: &str, bytes: u64) {
    let family = family(name);
    gpu_test(bytes, |session| {
        let report = run(family, &(family.sm120)()?, &LabModules, Some(session))?;
        session.receipt(
            &receipt_kind(name),
            BTreeMap::from([
                ("report".into(), report.json()),
                ("not_exercised".into(), json!(report.not_exercised())),
            ]),
        )?;
        check_sm120(&report, true);
        Ok(())
    })
    .unwrap_or_else(|error| panic!("SM120 passes the {name} family: {error}"));
}

#[test]
fn moe_cases_declare_on_sm120() {
    cpu("moe");
}

#[test]
fn sm120_passes_the_moe_cases_on_a_gpu() {
    gpu("moe");
}

#[test]
fn moe_routed_split_cases_declare_on_sm120() {
    cpu("moe routed split");
}

/// The split down case's output alone is 2,048 tokens of eight F32 routes of
/// 4,096 columns, and every expert's weights are on the device.
#[test]
fn sm120_passes_the_moe_routed_split_cases_on_a_gpu() {
    gpu_with("moe routed split", 16 << 30);
}

#[test]
fn sinkhorn_cases_declare_on_sm120() {
    cpu("sinkhorn");
    cpu("sinkhorn split");
}

#[test]
fn sinkhorn_tiled_and_mix_norm_cases_declare_on_sm120() {
    cpu("sinkhorn tiled");
    cpu("sinkhorn mix norm");
}

#[test]
fn sm120_passes_the_sinkhorn_cases_on_a_gpu() {
    gpu("sinkhorn");
    gpu("sinkhorn split");
}

#[test]
fn sm120_passes_the_sinkhorn_tiled_and_mix_norm_cases_on_a_gpu() {
    gpu("sinkhorn tiled");
    gpu("sinkhorn mix norm");
}

#[test]
fn gated_sparse_cases_declare_on_sm120() {
    cpu("gated sparse");
}

#[test]
fn sm120_passes_the_gated_sparse_cases_on_a_gpu() {
    gpu("gated sparse");
}

#[test]
fn multimodal_cases_declare_on_sm120() {
    cpu("multimodal");
}

#[test]
fn sm120_passes_the_multimodal_cases_on_a_gpu() {
    gpu("multimodal");
}

#[test]
fn context_parallel_cases_declare_on_sm120() {
    cpu("context parallel");
}

#[test]
fn sm120_passes_the_context_parallel_cases_on_a_gpu() {
    gpu("context parallel");
}
