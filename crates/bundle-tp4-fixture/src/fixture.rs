//! The canonical `chop-model-bundle-v1` package this fixture emits.
//!
//! Everything here is a declaration: the TP4 target and topology, the four
//! components and the bytes behind them, the planned program component whose
//! four synthetic rank images the weight plan loads from a checkpoint, the KV page layout the instance allocates a
//! rank-local pool from, the persistent state its committed history lives in,
//! the one operation it requires, and the one implementation it carries. The
//! emitter stages those bytes, measures the files it wrote, builds the one
//! canonical manifest from those measurements, and publishes the directory only
//! after the canonical verifier accepted it.
//!
//! The one operation is explicitly private. It is declared as a bundle custom
//! operation, so no host-optimized or generic candidate may satisfy it and no
//! host can be asked to stand in for it. That is the honest shape for what it
//! does: it produces a synthetic row of TP4's declared width, and nothing
//! model-neutral could be substituted for that. The 35 TP4 operations a real
//! bundle would declare are not here, are not stubbed, and are not claimed.

use std::path::Path;

use model_compiler_sdk::{
    BundleDraftV1, CompilerError, ComponentContentV1, ComponentDraftV1, EmittedBundleV1,
    ImplementationArtifactDraftV1, ImplementationDraftV1, NativeModuleDraftV1, StagedContentV1,
    emit_bundle_directory_v1,
};
use runtime_bundle::{
    BundleBuildIdentityV1, BundleError, BundleGraphBucketV1, BundleLaunchContractV1,
    BundleOperationKeyV1, BundleOperationProvisioningV1, BundleOperationRequirementV1,
    BundleRelativePathV1, BundleResourceRequirementsV1, BundleSemanticContractV1,
    BundleSourceIdentityV1, BundleSpecializationV1, BundleSpeculationV1, BundleTargetV1,
    BundleTensorAccessV1, BundleTensorAliasV1, BundleTensorDependencyV1, BundleTensorRequirementV1,
    DTYPE_F32_V1, DTYPE_U32_V1, LAYOUT_DENSE_ROW_MAJOR_V1, SCALE_NONE_V1,
};
use runtime_weights::sha256_bytes;

use crate::contract::{
    TP4_FIXTURE_ARCHITECTURE_V1, TP4_FIXTURE_BUFFER_ALIGNMENT_V1, TP4_FIXTURE_BUNDLE_ID_V1,
    TP4_FIXTURE_FRONTEND_COMPONENT_V1, TP4_FIXTURE_FRONTEND_FORMAT_V1,
    TP4_FIXTURE_FRONTEND_PATH_V1, TP4_FIXTURE_FRONTEND_ROLE_V1, TP4_FIXTURE_GRAPH_BUCKET_V1,
    TP4_FIXTURE_MARKER_COMPONENT_V1, TP4_FIXTURE_MARKER_FORMAT_V1, TP4_FIXTURE_MARKER_PATH_V1,
    TP4_FIXTURE_MARKER_ROLE_V1, TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1,
    TP4_FIXTURE_NATIVE_MODULE_PATH_V1, TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
    TP4_FIXTURE_PROGRAM_COMPONENT_V1, TP4_FIXTURE_PROGRAM_FORMAT_V1, TP4_FIXTURE_PROGRAM_ROLE_V1,
    TP4_FIXTURE_ROW_INPUT_TOKENS_V1, TP4_FIXTURE_ROW_OPERATION_V1,
    TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1, TP4_FIXTURE_ROW_VARIANT_V1,
    TP4_FIXTURE_TOKENIZER_CONFIGURATION_COMPONENT_V1, TP4_FIXTURE_TOKENIZER_CONFIGURATION_PATH_V1,
    TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1, TP4_FIXTURE_TOKENIZER_DEFINITION_COMPONENT_V1,
    TP4_FIXTURE_TOKENIZER_DEFINITION_PATH_V1, TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1,
    TP4_FIXTURE_TOKENIZER_FORMAT_V1, id, tp4_fixture_kv_layout_v1, tp4_fixture_state_layout_v1,
    tp4_fixture_topology_v1, tp4_fixture_vocabulary_v1,
};
use crate::frontend::tp4_fixture_frontend_declaration_v1;
use crate::marker::tp4_fixture_marker_v1;
use crate::operation::TP4_FIXTURE_ROW_OPERATION_ID_V1;
use crate::tokenizer::{
    tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
};

/// The implementation this package carries for its own private operation.
const TP4_FIXTURE_IMPLEMENTATION_ID_V1: &str = "fixture.tp4.synthetic-row.bundle.v1";

/// The operation ABI of this fixture's own synthetic row.
const TP4_FIXTURE_ROW_ABI_V1: &str = "fixture.tp4.synthetic-row.abi";

/// The version of that ABI.
const TP4_FIXTURE_ROW_ABI_VERSION_V1: u32 = 1;

/// The numerical contract the synthetic row is computed under.
///
/// It says in its own name that this is not model arithmetic. A route that
/// computed TP4 logits would declare a different contract, and would therefore
/// be a different semantic key that no host could confuse with this one.
const TP4_FIXTURE_ROW_NUMERICS_V1: &str = "fixture.tp4.synthetic-row-fixture-only.v1";

/// The fixed-capacity operand the private operation reads its tokens from.
const TP4_FIXTURE_TOKENS_ROLE_V1: &str = "input-tokens";

/// The operand that says how many of those token slots are live.
const TP4_FIXTURE_TOKEN_COUNT_ROLE_V1: &str = "input-token-count";

/// The operand the private operation writes.
const TP4_FIXTURE_LOGITS_ROLE_V1: &str = "output-logits";

/// The Rust source the private operation is compiled from.
const TP4_FIXTURE_OPERATION_SOURCE_V1: &str = include_str!("operation.rs");

/// The invocation `build-tp4-fixture` builds the native module with.
///
/// It is the argument list that program actually runs, feature selection
/// included, because a module built under a different feature selection is a
/// different artifact and this identity is what a reviewer compares against.
/// The profile is not part of it: the builder inherits its own, and the artifact
/// the manifest measures is the one it then staged.
const TP4_FIXTURE_BUILD_RECIPE_V1: &str =
    "cargo build --locked --offline -p bundle-tp4-fixture --lib (crate-type cdylib)";

/// The feature flags the builder adds when it was built with them itself.
const TP4_FIXTURE_FAKE_HOST_SERVICES_FLAGS_V1: &str =
    " --no-default-features --features fake-host-services";

/// The whole recipe, as the bytes the build identity is measured over.
fn tp4_fixture_build_recipe_v1() -> Vec<u8> {
    let mut recipe = TP4_FIXTURE_BUILD_RECIPE_V1.to_owned();
    if cfg!(feature = "fake-host-services") {
        recipe.push_str(TP4_FIXTURE_FAKE_HOST_SERVICES_FLAGS_V1);
    }
    recipe.into_bytes()
}

/// Emits this fixture's canonical bundle directory at `output`.
///
/// `native_module` is the shared object `build-tp4-fixture` compiled for this
/// crate. It is staged, measured, and digested by the emitter like any other
/// file, and the manifest records the one V1 entry symbol it must export.
pub fn emit_tp4_fixture_v1(
    native_module: &Path,
    output: &Path,
) -> Result<EmittedBundleV1, CompilerError> {
    emit_bundle_directory_v1(&tp4_fixture_bundle_draft_v1(native_module)?, output)
}

/// The complete declaration this fixture emits, before anything is staged.
pub fn tp4_fixture_bundle_draft_v1(native_module: &Path) -> Result<BundleDraftV1, BundleError> {
    let components = tp4_fixture_components_v1()?;
    let component_ids = components
        .iter()
        .map(|component| component.id().clone())
        .collect::<Vec<_>>();
    Ok(BundleDraftV1::new(
        id(TP4_FIXTURE_BUNDLE_ID_V1)?,
        BundleTargetV1::new(
            vec![id(TP4_FIXTURE_ARCHITECTURE_V1)?],
            vec![tp4_fixture_topology_v1()?],
        )?,
        BundleResourceRequirementsV1::new(
            TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1,
            vec![
                id("chop.prompt-rows.v2")?,
                id("chop.request-retirement.v1")?,
                id("chop.speculative.v1")?,
                id("chop.media-inputs.v1")?,
                id("chop.prefix-cache.v1")?,
                id(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1)?,
                id(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1)?,
            ],
        )?,
        tp4_fixture_kv_layout_v1()?,
        tp4_fixture_state_layout_v1()?,
    )
    .with_speculation(BundleSpeculationV1::new(
        3,
        true,
        4,
        vec![2, 4],
        Some(id(TP4_FIXTURE_PROGRAM_COMPONENT_V1)?),
    )?)
    .with_components(components)
    .with_graph_buckets(vec![BundleGraphBucketV1::new(
        id(TP4_FIXTURE_GRAPH_BUCKET_V1)?,
        component_ids,
        vec![tp4_fixture_row_key_v1()?],
    )?])
    .with_operation_requirements(vec![tp4_fixture_row_requirement_v1()?])
    .with_implementations(vec![tp4_fixture_row_implementation_v1()?])
    .with_native_module(NativeModuleDraftV1::new(
        BundleRelativePathV1::new(TP4_FIXTURE_NATIVE_MODULE_PATH_V1)?,
        StagedContentV1::file(native_module),
    )))
}

/// The five components this package carries. The program component is
/// planned: a direct load asks the fixture's weight plan for its four rank
/// images, which it reads from the synthetic checkpoint.
fn tp4_fixture_components_v1() -> Result<Vec<ComponentDraftV1>, BundleError> {
    let program = ComponentDraftV1::planned(
        id(TP4_FIXTURE_PROGRAM_COMPONENT_V1)?,
        id(TP4_FIXTURE_PROGRAM_ROLE_V1)?,
        id(TP4_FIXTURE_PROGRAM_FORMAT_V1)?,
    )?;
    Ok(vec![
        program,
        whole_component_v1(
            TP4_FIXTURE_FRONTEND_COMPONENT_V1,
            TP4_FIXTURE_FRONTEND_ROLE_V1,
            TP4_FIXTURE_FRONTEND_FORMAT_V1,
            TP4_FIXTURE_FRONTEND_PATH_V1,
            tp4_fixture_frontend_declaration_v1(),
        )?,
        whole_component_v1(
            TP4_FIXTURE_TOKENIZER_DEFINITION_COMPONENT_V1,
            TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1,
            TP4_FIXTURE_TOKENIZER_FORMAT_V1,
            TP4_FIXTURE_TOKENIZER_DEFINITION_PATH_V1,
            tp4_fixture_tokenizer_definition_v1(),
        )?,
        whole_component_v1(
            TP4_FIXTURE_TOKENIZER_CONFIGURATION_COMPONENT_V1,
            TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1,
            TP4_FIXTURE_TOKENIZER_FORMAT_V1,
            TP4_FIXTURE_TOKENIZER_CONFIGURATION_PATH_V1,
            tp4_fixture_tokenizer_configuration_v1(),
        )?,
        whole_component_v1(
            TP4_FIXTURE_MARKER_COMPONENT_V1,
            TP4_FIXTURE_MARKER_ROLE_V1,
            TP4_FIXTURE_MARKER_FORMAT_V1,
            TP4_FIXTURE_MARKER_PATH_V1,
            tp4_fixture_marker_v1(),
        )?,
    ])
}

fn whole_component_v1(
    component: &str,
    role: &str,
    format: &str,
    path: &str,
    bytes: Vec<u8>,
) -> Result<ComponentDraftV1, BundleError> {
    Ok(ComponentDraftV1::new(
        id(component)?,
        id(role)?,
        id(format)?,
        BundleRelativePathV1::new(path)?,
        ComponentContentV1::Whole(StagedContentV1::bytes(bytes)),
    ))
}

/// The key of the one operation only this package may implement.
pub fn tp4_fixture_row_key_v1() -> Result<BundleOperationKeyV1, BundleError> {
    Ok(BundleOperationKeyV1::new(
        id(TP4_FIXTURE_ROW_OPERATION_V1)?,
        id(TP4_FIXTURE_ROW_VARIANT_V1)?,
    ))
}

/// The complete semantic contract of the synthetic row.
///
/// The shape is exact and it is the shape every call is lent. One PrefillChunk's
/// capacity of token identifiers, how many of them are live, and one row of
/// TP4's declared width in binary32. A DecodeStep carries one token and a tail
/// carries between one and 32, and all of them are lent this: the operand is
/// zero past the count, so the declaration and the lend cannot drift. The
/// numerical contract says the values are a fixture's, so a host offering any
/// other width, dtype, or numerics would be offering a different operation and
/// could not satisfy this one.
pub fn tp4_fixture_row_contract_v1() -> Result<BundleSemanticContractV1, BundleError> {
    BundleSemanticContractV1::new(
        id(TP4_FIXTURE_ROW_ABI_V1)?,
        TP4_FIXTURE_ROW_ABI_VERSION_V1,
        vec![
            operand_v1(
                TP4_FIXTURE_TOKENS_ROLE_V1,
                u64::from(TP4_FIXTURE_ROW_INPUT_TOKENS_V1),
                DTYPE_U32_V1,
                BundleTensorAccessV1::ReadOnly,
                Vec::new(),
            )?,
            operand_v1(
                TP4_FIXTURE_TOKEN_COUNT_ROLE_V1,
                TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1,
                DTYPE_U32_V1,
                BundleTensorAccessV1::ReadOnly,
                Vec::new(),
            )?,
            operand_v1(
                TP4_FIXTURE_LOGITS_ROLE_V1,
                u64::from(tp4_fixture_vocabulary_v1()),
                DTYPE_F32_V1,
                BundleTensorAccessV1::WriteOnly,
                vec![
                    BundleTensorDependencyV1::new(0, id(TP4_FIXTURE_TOKENS_ROLE_V1)?),
                    BundleTensorDependencyV1::new(1, id(TP4_FIXTURE_TOKEN_COUNT_ROLE_V1)?),
                ],
            )?,
        ],
        Vec::new(),
        id(TP4_FIXTURE_ROW_NUMERICS_V1)?,
        Vec::new(),
    )
}

/// The one requirement this package declares, and implements itself.
fn tp4_fixture_row_requirement_v1() -> Result<BundleOperationRequirementV1, BundleError> {
    BundleOperationRequirementV1::new(
        tp4_fixture_row_key_v1()?,
        BundleOperationProvisioningV1::BundleCustomOperation,
        tp4_fixture_row_contract_v1()?,
    )
}

/// The one implementation this package declares, for its own private operation.
///
/// The artifact is this package's native module, because that is where the code
/// that computes the row is: the program component carries geometry words, a
/// history seed, and filler, not an implementation. The symbol stays the
/// identifier the host runs the operation by, and the source identity stays the
/// file the callback is compiled from. The emitter fills the artifact's length
/// and digest in from the module it staged, so the record can never point
/// outside the closed file set.
fn tp4_fixture_row_implementation_v1() -> Result<ImplementationDraftV1, BundleError> {
    Ok(ImplementationDraftV1::new(
        id(TP4_FIXTURE_IMPLEMENTATION_ID_V1)?,
        tp4_fixture_row_key_v1()?,
        ImplementationArtifactDraftV1::NativeModule,
        id(
            std::str::from_utf8(TP4_FIXTURE_ROW_OPERATION_ID_V1).map_err(|error| {
                BundleError::validation(format!("the operation identifier is not UTF-8: {error}"))
            })?,
        )?,
        id(TP4_FIXTURE_ARCHITECTURE_V1)?,
        // What a launch would use, rather than what one did: the operation is
        // native and the host runs it directly, so nothing is launched.
        BundleLaunchContractV1::new([1, 1, 1], [TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1, 1, 1], 0)?,
        None,
        vec![
            BundleSpecializationV1::new(
                id("vocabulary")?,
                id(&tp4_fixture_vocabulary_v1().to_string())?,
            )?,
            BundleSpecializationV1::new(
                id("prefill-tokens")?,
                id(&TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1.to_string())?,
            )?,
        ],
        BundleSourceIdentityV1::new(
            id("bundle-tp4-fixture")?,
            id(env!("CARGO_PKG_VERSION"))?,
            id("src.operation.rs")?,
            TP4_FIXTURE_OPERATION_SOURCE_V1.len() as u64,
            sha256_bytes(TP4_FIXTURE_OPERATION_SOURCE_V1.as_bytes()),
        )?,
        BundleBuildIdentityV1::new(
            id("rustc")?,
            id(env!("CARGO_PKG_RUST_VERSION"))?,
            id("cargo-cdylib")?,
            sha256_bytes(&tp4_fixture_build_recipe_v1()),
        )?,
    ))
}

/// One unscaled, disjoint, densely laid-out operand of the declared operation.
fn operand_v1(
    role: &str,
    elements: u64,
    dtype: &str,
    access: BundleTensorAccessV1,
    dependencies: Vec<BundleTensorDependencyV1>,
) -> Result<BundleTensorRequirementV1, BundleError> {
    BundleTensorRequirementV1::new(
        id(role)?,
        vec![elements],
        id(dtype)?,
        id(LAYOUT_DENSE_ROW_MAJOR_V1)?,
        id(SCALE_NONE_V1)?,
        Vec::new(),
        TP4_FIXTURE_BUFFER_ALIGNMENT_V1,
        access,
        BundleTensorAliasV1::Disjoint,
        dependencies,
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use runtime_bundle::BundleOperationProvisioningV1;

    use crate::contract::{
        TP4_FIXTURE_ROW_INPUT_TOKENS_V1, TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1,
        tp4_fixture_vocabulary_v1,
    };

    use super::{
        TP4_FIXTURE_BUILD_RECIPE_V1, TP4_FIXTURE_ROW_NUMERICS_V1, tp4_fixture_build_recipe_v1,
        tp4_fixture_bundle_draft_v1, tp4_fixture_row_contract_v1, tp4_fixture_row_requirement_v1,
    };

    /// A draft that cannot be described is one no package could be built from,
    /// so every record is constructed here before a byte is staged.
    #[test]
    fn the_declaration_this_fixture_emits_is_well_formed() {
        let draft = tp4_fixture_bundle_draft_v1(Path::new("native/module.bin"))
            .expect("this fixture's declaration is valid");

        assert_eq!(draft.components().len(), 5);
        let requirement = tp4_fixture_row_requirement_v1().expect("the one requirement is valid");
        assert_eq!(
            requirement.provisioning(),
            &BundleOperationProvisioningV1::BundleCustomOperation,
            "the one operation has to be private, so no host can be asked to stand in for it"
        );
    }

    /// The declared operation is the model's width under a fixture's numerics,
    /// over the fixed-capacity operands every call is lent.
    #[test]
    fn the_declared_operation_is_the_fixture_width_under_fixture_numerics() {
        let contract = tp4_fixture_row_contract_v1().expect("the row contract is valid");

        let [tokens, count, logits] = contract.tensors() else {
            panic!("this operation reads a fixed-capacity token operand and its count");
        };
        assert_eq!(tokens.dtype().as_str(), "chop.dtype.u32.v1");
        assert_eq!(
            tokens.dimensions(),
            [u64::from(TP4_FIXTURE_ROW_INPUT_TOKENS_V1)],
            "the token operand is one PrefillChunk's capacity, whatever a batch carries"
        );
        assert_eq!(count.dtype().as_str(), "chop.dtype.u32.v1");
        assert_eq!(
            count.dimensions(),
            [TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1]
        );
        assert_eq!(
            logits.dimensions(),
            [u64::from(tp4_fixture_vocabulary_v1())]
        );
        assert_eq!(logits.dtype().as_str(), "chop.dtype.f32.v1");
        assert_eq!(
            contract.numerical_contract().as_str(),
            TP4_FIXTURE_ROW_NUMERICS_V1
        );
        assert!(
            TP4_FIXTURE_ROW_NUMERICS_V1.contains("fixture-only"),
            "the numerical identity must say this computes no model arithmetic"
        );
        contract
            .semantic_key()
            .expect("the row contract has a semantic key");
    }

    /// The recorded recipe is the invocation the builder makes, feature
    /// selection included, rather than a shorter line that omits it.
    #[test]
    fn the_recorded_build_recipe_is_the_invocation_the_builder_makes() {
        let recipe =
            String::from_utf8(tp4_fixture_build_recipe_v1()).expect("the recipe is one UTF-8 line");

        assert!(recipe.starts_with(TP4_FIXTURE_BUILD_RECIPE_V1));
        for flag in ["--locked", "--offline", "-p bundle-tp4-fixture", "--lib"] {
            assert!(recipe.contains(flag), "the recipe must record {flag}");
        }
        assert_eq!(
            recipe.contains("--features fake-host-services"),
            cfg!(feature = "fake-host-services"),
            "the recipe must record the feature selection the builder compiled with"
        );
    }
}
