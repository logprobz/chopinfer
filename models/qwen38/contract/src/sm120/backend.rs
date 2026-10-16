//! Qwen's SM120 kernels as the contract's backend ([`crate::backend::Kernels`]).

use super::{Sm120, admission, context_parallel, generated, owned, registered};
use crate::backend::{
    EngineConfig, FusionChoice, FusionMember, Kernels, LegacyStage, OwnedKernel, StageChoice,
};
use crate::context_parallel::GatherBytes;
use crate::contract::Shape;
use crate::precision::{DenseRole, Projection};
use kernel_core::contract::{DType, Refusal, Target, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::{ExecutionPhase, OperationSpec, OperationUse};
use kernel_core::properties::Invariance;
use kernels_sm120::invariance::{Declared, OpKind};

impl Kernels for Sm120 {
    const TARGET: Target = Target::Sm120f;

    const ARCHITECTURE: &'static str = "sm_120f";

    const LINEAR_CHUNK: u32 = kernels_sm120::linear::CHUNK;

    const ROUTE_TICKET_BYTES: u64 = kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES;

    fn expert_route_rows(routed: u32, experts: u32) -> Result<u32, Refusal> {
        kernels_sm120::moe::padded_capacity(routed, experts, 16)
    }

    fn registered_dense(
        shape: Shape,
        role: DenseRole,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal> {
        registered::dense_capability(shape, role, usage)
    }

    fn registered_expert(
        shape: Shape,
        projection: Projection,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal> {
        registered::expert_capability(shape, projection, usage)
    }

    fn registered_lookup(
        shape: Shape,
        ngram: bool,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal> {
        registered::lookup_capability(shape, ngram, usage)
    }

    fn generated_capabilities(
        geometry: Option<&'static crate::geometry::Geometry>,
    ) -> Result<Vec<OperationCapability>, String> {
        match geometry {
            Some(geometry) => generated::generated_for(geometry),
            None => generated::all_capabilities(),
        }
    }

    fn published_ports(shape: Shape) -> Result<Vec<(&'static str, WeightFormat)>, String> {
        generated::published_ports(shape)
    }

    fn generated_capability(
        shape: Shape,
        port: &str,
        weight: WeightFormat,
        usage: OperationUse,
    ) -> Result<OperationCapability, String> {
        generated::publish(shape, port, weight, usage).map(|p| p.capability.clone())
    }

    fn capability_domain(
        capability: &OperationCapability,
        local_scale: Option<DType>,
    ) -> Result<OperationSpec, String> {
        generated::capability_domain(capability, local_scale)
    }

    fn legacy_stage(
        shape: Shape,
        stage: LegacyStage,
        usage: OperationUse,
    ) -> Result<StageChoice, String> {
        registered::legacy_stage(shape, stage, usage)
    }

    fn legacy_fusion(
        shape: Shape,
        member: FusionMember<'_>,
        usage: OperationUse,
    ) -> Result<FusionChoice, String> {
        registered::legacy_fusion(shape, member, usage)
    }

    type Declared = Declared;

    fn bundle_declaration(symbol: String, invariance: Invariance) -> Declared {
        Declared {
            symbol,
            profile: None,
            op: OpKind::Bundle,
            invariance,
        }
    }

    fn owned_entry(kernel: OwnedKernel, tag: &str) -> String {
        owned::per_rank(owned::entry(kernel), tag)
    }

    fn ngram_hash_symbol(shape: Shape, phase: ExecutionPhase) -> String {
        generated::ngram_hash_symbol(shape, phase)
    }

    fn mixed_cubin(name: &str) -> bool {
        generated::mixed_cubin(name)
    }

    fn ngram_hash_symbols() -> Result<Vec<(String, String)>, String> {
        Ok(generated::ngram_hash_forms()?
            .into_iter()
            .map(|form| (form.prefill, form.verify))
            .collect())
    }

    type EncoderKernel = kernels_sm120::encoder::Entry;

    fn encoder_kernels(
        declarations: &[encoder_spec::Declaration],
    ) -> Result<Vec<kernels_sm120::encoder::Entry>, encoder_spec::Error> {
        kernels_sm120::encoder::model_catalog(declarations, &kernels_sm120::encoder::Shared)
    }

    const ENGINE_PHASES: [&'static str; 5] = admission::ENGINE_PHASES;

    fn admit_engine(
        config: &EngineConfig,
        geometry: Option<&'static crate::geometry::Geometry>,
        limits: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        admission::admit(config, geometry, limits)
    }

    fn context_parallel_entries(shape: Shape) -> bool {
        context_parallel::has_entries(shape)
    }

    fn group_gather(size: u32, words: u64) -> Result<GatherBytes, Refusal> {
        kernels_sm120::context_parallel::group::gather_buffers(size, words).map(|bytes| {
            GatherBytes {
                output_bytes: bytes.output_bytes,
                flag_bytes: bytes.flag_bytes,
                epoch_bytes: bytes.epoch_bytes,
            }
        })
    }

    fn selection_workspace(shape: Shape, rows: u32) -> u64 {
        context_parallel::select_entry(shape).workspace_bytes(rows)
    }
}
