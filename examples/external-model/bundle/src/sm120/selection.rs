//! The example's SM120 kernel selection: the entries its build compiles, and
//! the backend that prepares the example's RMSNorm from them. This bundle
//! owns its choices. Neither its name nor its execution program is
//! registered in the shared catalogs.

use super::ops::{self, CAPACITY};
use kernel_build::{
    BuildContext, BuildError, CubinSpec, EntryDeclaration, EntryParam, EntrySpec, KernelSelection,
    SelectedCubin,
};
use kernel_core::contract::Refusal;
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};
use std::path::Path;

pub const RMS_SYMBOL: &str = "example_rmsnorm_v1";
pub const PROBE_SYMBOL: &str = "example_probe_v1";
pub const RMS_CASE: &str = "example.plain-rmsnorm.v1";
pub const PROBE_CASE: &str = "example.sigmoid.v1";
/// Threads of the probe entry's one block.
pub const PROBE_THREADS: u32 = 32;

/// The RMSNorm entry's requirement: the example's op with the 256-thread
/// tree reduction SM120 compiles for it.
pub fn requirement() -> RmsNormRequirement {
    RmsNormRequirement {
        numerics: Some(RmsNormNumerics::TREE_256),
        ..ops::rmsnorm()
    }
}

pub fn rmsnorm() -> Result<RmsNorm, Refusal> {
    RmsNorm::new(requirement(), CAPACITY, RmsNormTarget::SM120F)
}

/// The SM120 backend over the RMSNorm entry this selection compiles.
pub fn backend() -> Sm120 {
    Sm120::new().with_rmsnorm(requirement(), RMS_SYMBOL)
}

pub fn selection(model: &Path, sdk_cuda: &Path) -> Result<KernelSelection, BuildError> {
    let context = BuildContext::new(
        "example-bundle@0.1.0",
        "external-model",
        "example.kernels.v1",
    )
    .source_root("external-model", model)
    .source_root("chopinfer", sdk_cuda.join("../../.."));
    let include = context.require("local probe include", sdk_cuda.strip_prefix(model))?;
    // Keep this legacy recipe byte-for-byte identical, including its cubin name.
    let local = CubinSpec {
        name: "external_probe".into(),
        sources_root: model.into(),
        include_dirs: vec!["bundle/cuda".into(), include.into()],
        headers: vec!["#include \"probe.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            PROBE_SYMBOL.into(),
            vec![
                EntryParam::DevicePointer {
                    name: "result".into(),
                    cxx_type: "float".into(),
                },
                EntryParam::U32 {
                    name: "value".into(),
                },
            ],
            "example::probe(result, float(value));".into(),
            PROBE_THREADS,
            1,
        )],
    };
    let declaration = EntryDeclaration {
        symbol: PROBE_SYMBOL.into(),
        operation: "SigmoidProbe".into(),
        contract: "example.sigmoid-u32-f32.v1".into(),
        implementation: "example.native-probe.v1".into(),
        requirements: [
            ("input", "U32 converted to F32, nearest even"),
            ("output", "one aligned F32 scalar"),
            (
                "arithmetic",
                "F32 1 / (1 + expf(-F32(value))) via common/sigmoid.cuh",
            ),
            ("state", "stateless; thread zero writes result[0]"),
            ("target", "sm120f; 32 threads"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect(),
        correctness_case: Some(PROBE_CASE.into()),
    };
    let rms = context.require("RMSNorm", rmsnorm())?;
    let mut shared = rms.cubin_spec("external_rmsnorm_v1", RMS_SYMBOL);
    // Also allow the in-tree lab to reconstruct this exported recipe.
    shared.sources_root = sdk_cuda.into();
    let mut rms_declaration = rms.declaration(RMS_SYMBOL);
    rms_declaration.correctness_case = Some(RMS_CASE.into());
    KernelSelection::new(
        context,
        vec![
            SelectedCubin::existing(local, vec![declaration])?,
            SelectedCubin::new(shared, vec![rms_declaration])?,
        ],
    )
}
