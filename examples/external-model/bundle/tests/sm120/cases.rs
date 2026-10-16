//! SM120 checks of the example's selection: its builder refusals and its
//! locked artifacts.

use super::ops::{CAPACITY, WIDTH};
use super::selection;
use kernel_build::{BuildReport, KernelSelection};
use kernel_core::contract::{DType, RmsNormKind};
use kernel_core::rmsnorm::{RmsNormLayout, RmsNormNumerics, RmsNormRounding};
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};

/// Exercise near-matches through the shared builder and selected contract.
pub fn check_refusals(selected: &KernelSelection) {
    let req = selection::requirement();
    let rounding = kernel_core::rmsnorm::RmsNormRequirement {
        numerics: Some(RmsNormNumerics {
            output_rounding: RmsNormRounding::TowardZero,
            ..RmsNormNumerics::TREE_256
        }),
        ..req
    };
    for (bad, field) in [
        (
            kernel_core::rmsnorm::RmsNormRequirement {
                input: DType::F16,
                ..req
            },
            "formats",
        ),
        (
            kernel_core::rmsnorm::RmsNormRequirement {
                layout: RmsNormLayout::StridedRows {
                    stride_elements: WIDTH as u64 + 8,
                    alignment_bytes: 16,
                },
                ..req
            },
            "contiguous",
        ),
        (rounding, "rounding"),
    ] {
        let error = selected
            .context()
            .require(
                "RMSNorm",
                RmsNorm::new(bad, CAPACITY, RmsNormTarget::SM120F),
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("example.kernels.v1")
                && error.contains("RMSNorm")
                && error.contains(field),
            "{error}"
        );
    }
    let hf = kernel_core::rmsnorm::RmsNormRequirement {
        kind: RmsNormKind::HfBf16Intermediate,
        numerics: Some(RmsNormNumerics::HF_ORDERED),
        ..req
    };
    RmsNorm::new(hf, CAPACITY, RmsNormTarget::SM120F).unwrap();
    assert!(selection::rmsnorm().unwrap().require(&hf).is_err());
}

/// Bind the example's current declarations and generated code to verified bytes.
pub fn check_artifacts(report: &BuildReport, selection: &KernelSelection) {
    report
        .verify_current()
        .expect("current locked artifacts and source membership");
    assert!(report.source_ownership_complete && report.entry_declarations_complete);
    assert_eq!(
        report.context.as_ref().unwrap().consumer,
        selection.context().consumer
    );
    assert_eq!(
        report.context.as_ref().unwrap().selection,
        selection.context().selection
    );
    assert_eq!(report.cubins.len(), 2);
    for spec in selection.specs() {
        let artifact = report.cubins.iter().find(|c| c.name == spec.name).unwrap();
        assert_eq!(
            &artifact.declarations,
            &selection.declarations()[&spec.name]
        );
        let generated = kernel_build::generate(spec).unwrap();
        for (name, contents) in [
            (format!("{}.cu", spec.name), generated.translation_unit),
            (format!("{}_params.h", spec.name), generated.header),
        ] {
            let file = artifact
                .membership
                .iter()
                .map(|m| &report.files[&m.file])
                .find(|f| {
                    f.identity.kind == kernel_build::report::SourceKind::Generated
                        && f.identity.path == name
                })
                .unwrap();
            assert_eq!(std::fs::read_to_string(&file.location).unwrap(), contents);
        }
        assert_eq!(artifact.identity["entries"].as_array().unwrap().len(), 1);
    }
}
