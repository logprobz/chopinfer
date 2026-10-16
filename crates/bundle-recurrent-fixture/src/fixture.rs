//! The canonical `chop-model-bundle-v1` package this fixture emits.
//!
//! Everything here is a declaration: the target and topology, the components and
//! the bytes behind them, the KV layout this fixture declares and never uses,
//! the persistent state it does use, the two operations it requires, and the one
//! implementation it carries. The emitter stages those bytes, measures the files
//! it wrote, builds the one canonical manifest from those measurements, and
//! publishes the directory only after the canonical verifier accepted it.
//!
//! The two operation requirements are the point of this fixture, and one
//! execution goes through both. The recurrent step is a bundle custom
//! operation, which only a bundle may implement and which this
//! package therefore carries an implementation for; it writes its row into an
//! intermediate operand. The exact F32 copy of that row into the logits a
//! caller receives is the model-neutral primitive the shared contract crate
//! declares, and this package carries no implementation of it at all:
//! a host that offers an exact match satisfies it generically, and a host that
//! offers nothing, or something whose contract differs, leaves it unsupported
//! and this fixture emits no row.

use std::path::Path;

use model_compiler_sdk::{
    BundleDraftV1, CompilerError, ComponentContentV1, ComponentDraftV1, EmittedBundleV1,
    ImplementationArtifactDraftV1, ImplementationDraftV1, NativeModuleDraftV1, StagedContentV1,
    emit_bundle_directory_v1,
};
use runtime_bundle::{
    BundleBuildIdentityV1, BundleError, BundleGraphBucketV1, BundleLaunchContractV1,
    BundleOperationKeyV1, BundleOperationProvisioningV1, BundleOperationRequirementV1,
    BundleRelativePathV1, BundleResourceRequirementsV1, BundleScalarArgumentV1,
    BundleSemanticContractV1, BundleSourceIdentityV1, BundleSpecializationV1, BundleTargetV1,
    BundleTensorAccessV1, BundleTensorAliasV1, BundleTensorDependencyV1, BundleTensorRequirementV1,
    DTYPE_F32_V1, DTYPE_U32_V1, GENERIC_COPY_F32_PRIMITIVE_V1, LAYOUT_DENSE_ROW_MAJOR_V1,
    SCALE_NONE_V1, generic_copy_f32_contract_v1, generic_copy_f32_key_v1,
};
use runtime_weights::sha256_bytes;

use crate::contract::{
    RECURRENT_ARCHITECTURE_V1, RECURRENT_BUFFER_ALIGNMENT_V1, RECURRENT_BUNDLE_ID_V1,
    RECURRENT_FRONTEND_COMPONENT_V1, RECURRENT_FRONTEND_DECLARATION_V1,
    RECURRENT_FRONTEND_FORMAT_V1, RECURRENT_FRONTEND_PATH_V1, RECURRENT_FRONTEND_ROLE_V1,
    RECURRENT_GRAPH_BUCKET_V1, RECURRENT_MAXIMUM_TOKENS_V1,
    RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1, RECURRENT_NATIVE_MODULE_PATH_V1,
    RECURRENT_PROGRAM_COMPONENT_V1, RECURRENT_PROGRAM_FORMAT_V1, RECURRENT_PROGRAM_ROLE_V1,
    RECURRENT_STEP_OPERATION_V1, RECURRENT_STEP_VARIANT_V1, RECURRENT_VOCABULARY_V1, id,
    recurrent_kv_layout_v1, recurrent_state_layout_v1, recurrent_topology_v1,
};
use crate::operation::RECURRENT_STEP_OPERATION_ID_V1;
use crate::recurrent::{RECURRENT_LANES_V1, RECURRENT_STATE_BYTES_V1};

/// The implementation this package carries for its own custom operation.
const RECURRENT_IMPLEMENTATION_ID_V1: &str = "fixture.recurrent.step.bundle.v1";

/// The operation ABI of this fixture's own recurrent step.
const RECURRENT_STEP_ABI_V1: &str = "fixture.recurrent.step.abi";

/// The exact-integer contract the recurrent fold computes under.
const RECURRENT_FOLD_NUMERICS_V1: &str = "fixture.recurrent.exact-integer-fold.v1";

/// The Rust source the custom operation is compiled from.
///
/// The implementation record names the exact bytes behind the operation, so the
/// declaration is provenance rather than a label: this is the file, and the
/// digest below is taken over it at compile time.
const RECURRENT_OPERATION_SOURCE_V1: &str = include_str!("operation.rs");

/// The recipe the native module is built with, as one auditable line.
const RECURRENT_BUILD_RECIPE_V1: &[u8] =
    b"cargo build --package bundle-recurrent-fixture --lib (crate-type cdylib)";

/// Emits this fixture's canonical bundle directory at `output`.
///
/// `native_module` is the shared object `build-recurrent-fixture` compiled for
/// this crate.
/// It is staged, measured, and digested by the emitter like any other file, and
/// the manifest records the one V1 entry symbol it must export.
pub fn emit_recurrent_fixture_v1(
    native_module: &Path,
    output: &Path,
) -> Result<EmittedBundleV1, CompilerError> {
    emit_bundle_directory_v1(&recurrent_bundle_draft_v1(native_module)?, output)
}

/// The complete declaration this fixture emits, before anything is staged.
pub fn recurrent_bundle_draft_v1(native_module: &Path) -> Result<BundleDraftV1, BundleError> {
    let draft = BundleDraftV1::new(
        id(RECURRENT_BUNDLE_ID_V1)?,
        BundleTargetV1::new(
            vec![id(RECURRENT_ARCHITECTURE_V1)?],
            vec![recurrent_topology_v1()?],
        )?,
        BundleResourceRequirementsV1::new(
            RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1,
            vec![
                id("chop.prompt-rows.v2")?,
                id("chop.prefix-cache.state-slot.v1")?,
                id("chop.prefix-cache.v1")?,
                id("chop.serving-kv.v1")?,
                id("chop.recurrent-state.v1")?,
                id("chop.request-retirement.v1")?,
                id("chop.speculative.v1")?,
                id(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1)?,
                id(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1)?,
            ],
        )?,
        recurrent_kv_layout_v1()?,
        recurrent_state_layout_v1()?,
    )
    .with_recurrent_state_layouts(vec![runtime_bundle::BundleStateLayoutV1::new(
        id("fixture.recurrent.request.v1")?,
        RECURRENT_STATE_BYTES_V1,
        crate::contract::RECURRENT_STATE_ALIGNMENT_V1,
        runtime_bundle::BundleStateLifetimeV1::Request,
    )?])
    .with_speculation(runtime_bundle::BundleSpeculationV1::new(
        3,
        false,
        4,
        vec![2, 3, 4],
        None,
    )?)
    .with_components(recurrent_components_v1()?)
    .with_graph_buckets(vec![BundleGraphBucketV1::new(
        id(RECURRENT_GRAPH_BUCKET_V1)?,
        vec![
            id(RECURRENT_PROGRAM_COMPONENT_V1)?,
            id(RECURRENT_FRONTEND_COMPONENT_V1)?,
        ],
        vec![recurrent_step_key_v1()?, recurrent_generic_copy_key_v1()?],
    )?])
    .with_operation_requirements(vec![
        recurrent_step_requirement_v1()?,
        recurrent_generic_copy_requirement_v1()?,
    ])
    .with_implementations(vec![recurrent_step_implementation_v1()?])
    .with_native_module(NativeModuleDraftV1::new(
        BundleRelativePathV1::new(RECURRENT_NATIVE_MODULE_PATH_V1)?,
        StagedContentV1::file(native_module),
    ));
    Ok(draft)
}

/// The two components this package carries. The program component is
/// planned: a direct load asks the fixture's weight plan for its rank image,
/// which it reads from the synthetic checkpoint.
fn recurrent_components_v1() -> Result<Vec<ComponentDraftV1>, BundleError> {
    let program = ComponentDraftV1::planned(
        id(RECURRENT_PROGRAM_COMPONENT_V1)?,
        id(RECURRENT_PROGRAM_ROLE_V1)?,
        id(RECURRENT_PROGRAM_FORMAT_V1)?,
    )?;
    let frontend = ComponentDraftV1::new(
        id(RECURRENT_FRONTEND_COMPONENT_V1)?,
        id(RECURRENT_FRONTEND_ROLE_V1)?,
        id(RECURRENT_FRONTEND_FORMAT_V1)?,
        BundleRelativePathV1::new(RECURRENT_FRONTEND_PATH_V1)?,
        ComponentContentV1::Whole(StagedContentV1::bytes(
            RECURRENT_FRONTEND_DECLARATION_V1.as_bytes().to_vec(),
        )),
    );
    Ok(vec![program, frontend])
}

/// The key of the operation only this package may implement.
pub fn recurrent_step_key_v1() -> Result<BundleOperationKeyV1, BundleError> {
    Ok(BundleOperationKeyV1::new(
        id(RECURRENT_STEP_OPERATION_V1)?,
        id(RECURRENT_STEP_VARIANT_V1)?,
    ))
}

/// The complete semantic contract of the recurrent step.
///
/// The step reads the batch's tokens and writes one row of the declared width.
/// It writes that row into an intermediate operand rather than into the logits
/// the caller receives, because the copy into those logits is the other
/// operation, and that one is the host's.
pub fn recurrent_step_contract_v1() -> Result<BundleSemanticContractV1, BundleError> {
    BundleSemanticContractV1::new(
        id(RECURRENT_STEP_ABI_V1)?,
        1,
        vec![
            tensor_v1(
                "input-tokens",
                vec![u64::from(RECURRENT_MAXIMUM_TOKENS_V1)],
                DTYPE_U32_V1,
                BundleTensorAccessV1::ReadOnly,
                Vec::new(),
            )?,
            tensor_v1(
                "step-output",
                vec![u64::from(RECURRENT_VOCABULARY_V1)],
                DTYPE_F32_V1,
                BundleTensorAccessV1::WriteOnly,
                vec![BundleTensorDependencyV1::new(0, id("input-tokens")?)],
            )?,
        ],
        vec![BundleScalarArgumentV1::new(
            id("token-count")?,
            id(DTYPE_U32_V1)?,
            32,
        )?],
        id(RECURRENT_FOLD_NUMERICS_V1)?,
        Vec::new(),
    )
}

/// The key of the generic copy this package leaves to the host tier.
///
/// It is the model-neutral primitive the shared contract crate declares,
/// required at the width this fixture's program image declares.
pub fn recurrent_generic_copy_key_v1() -> Result<BundleOperationKeyV1, BundleError> {
    generic_copy_f32_key_v1()
}

/// The complete semantic contract of that generic copy.
pub fn recurrent_generic_copy_contract_v1() -> Result<BundleSemanticContractV1, BundleError> {
    generic_copy_f32_contract_v1(RECURRENT_VOCABULARY_V1)
}

fn recurrent_step_requirement_v1() -> Result<BundleOperationRequirementV1, BundleError> {
    BundleOperationRequirementV1::new(
        recurrent_step_key_v1()?,
        BundleOperationProvisioningV1::BundleCustomOperation,
        recurrent_step_contract_v1()?,
    )
}

/// The requirement this package carries no implementation for.
///
/// A host resolves it through its own catalog or leaves it unsupported. This
/// fixture's execution goes through it either way: the row a caller receives is
/// the destination this copy produced, so a host with no candidate for it can
/// load the bundle and still never see a row.
fn recurrent_generic_copy_requirement_v1() -> Result<BundleOperationRequirementV1, BundleError> {
    BundleOperationRequirementV1::new(
        recurrent_generic_copy_key_v1()?,
        BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1)?),
        recurrent_generic_copy_contract_v1()?,
    )
}

/// The one implementation this package declares, for its own custom operation.
///
/// The artifact is this package's native module, because that is where the code
/// that runs the step is: the program component carries geometry words and a
/// state seed, not an implementation, and a record naming it would describe
/// bytes the host uploads rather than the callback it calls. The symbol stays
/// the identifier the host runs the operation by, and the source identity stays
/// the file the callback is compiled from. The emitter fills the artifact's
/// length and digest in from the module it staged, so the record can never
/// point outside the closed file set.
fn recurrent_step_implementation_v1() -> Result<ImplementationDraftV1, BundleError> {
    Ok(ImplementationDraftV1::new(
        id(RECURRENT_IMPLEMENTATION_ID_V1)?,
        recurrent_step_key_v1()?,
        ImplementationArtifactDraftV1::NativeModule,
        id(
            std::str::from_utf8(RECURRENT_STEP_OPERATION_ID_V1).map_err(|error| {
                BundleError::validation(format!("the operation identifier is not UTF-8: {error}"))
            })?,
        )?,
        id(RECURRENT_ARCHITECTURE_V1)?,
        // One block per row of the one sequence a step runs. Nothing launches
        // this: the operation is native and the host runs it directly, so the
        // geometry is what a launch would use rather than what one did.
        BundleLaunchContractV1::new([1, 1, 1], [RECURRENT_VOCABULARY_V1, 1, 1], 0)?,
        None,
        vec![
            BundleSpecializationV1::new(id("lanes")?, id(&RECURRENT_LANES_V1.to_string())?)?,
            BundleSpecializationV1::new(
                id("vocabulary")?,
                id(&RECURRENT_VOCABULARY_V1.to_string())?,
            )?,
        ],
        BundleSourceIdentityV1::new(
            id("bundle-recurrent-fixture")?,
            id(env!("CARGO_PKG_VERSION"))?,
            id("src.operation.rs")?,
            RECURRENT_OPERATION_SOURCE_V1.len() as u64,
            sha256_bytes(RECURRENT_OPERATION_SOURCE_V1.as_bytes()),
        )?,
        BundleBuildIdentityV1::new(
            id("rustc")?,
            id(env!("CARGO_PKG_RUST_VERSION"))?,
            id("cargo-cdylib")?,
            sha256_bytes(RECURRENT_BUILD_RECIPE_V1),
        )?,
    ))
}

/// One unscaled, disjoint, densely laid-out operand of a declared operation.
fn tensor_v1(
    role: &str,
    dimensions: Vec<u64>,
    dtype: &str,
    access: BundleTensorAccessV1,
    dependencies: Vec<BundleTensorDependencyV1>,
) -> Result<BundleTensorRequirementV1, BundleError> {
    BundleTensorRequirementV1::new(
        id(role)?,
        dimensions,
        id(dtype)?,
        id(LAYOUT_DENSE_ROW_MAJOR_V1)?,
        id(SCALE_NONE_V1)?,
        Vec::new(),
        RECURRENT_BUFFER_ALIGNMENT_V1,
        access,
        BundleTensorAliasV1::Disjoint,
        dependencies,
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        recurrent_bundle_draft_v1, recurrent_generic_copy_contract_v1, recurrent_step_contract_v1,
    };

    /// A draft that cannot be described is one no package could be built from,
    /// so every record is constructed here before a byte is staged.
    #[test]
    fn the_declaration_this_fixture_emits_is_well_formed() {
        let draft = recurrent_bundle_draft_v1(Path::new("native/module.bin"))
            .expect("this fixture's declaration is valid");

        assert_eq!(draft.components().len(), 2);
    }

    /// Two operations that shared one semantic key could not be told apart, and
    /// the resolver would have to guess which provisioning owns it.
    #[test]
    fn the_two_declared_operations_are_different_semantic_keys() {
        let step = recurrent_step_contract_v1()
            .expect("the step contract is valid")
            .semantic_key()
            .expect("the step contract has a semantic key");
        let copy = recurrent_generic_copy_contract_v1()
            .expect("the generic copy contract is valid")
            .semantic_key()
            .expect("the generic copy contract has a semantic key");

        assert_ne!(step, copy);
    }
}
