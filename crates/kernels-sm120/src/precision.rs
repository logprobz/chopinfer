//! Bind supplied dense, grouped and lookup implementations to portable operands.
//! Construction emits build inputs. Inventory resolution never invokes a compiler.

use crate::{gemm, moe, row::precision as row};
use kernel_build::{CubinSpec, EntryDeclaration, EntrySpec, SelectedCubin};
use kernel_core::contract::{
    ActivationFormat, DType, GemmQuery, InstantiationSpec, NumericalContract, OutputQuantization,
    Refusal, RuntimeCounts, StaticQuery, WeightFormat,
};
use kernel_core::precision::capability::{
    ActivationCapability, CapabilityDocument, CapabilityInventory, GlobalScaleCapability,
    OperationCapability,
};
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernel_core::precision::{
    ArithmeticSpec, DecodeOrder, DynamicScale, MatrixGeometry, OperationKind, OperationSpec,
    OperationUse, Rounding, WeightLayout,
};
use std::collections::{BTreeMap, BTreeSet};

pub mod generated;

#[derive(Clone, Debug)]
pub enum Compute {
    Dense(InstantiationSpec),
    Grouped(moe::GroupedOperation),
    Lookup(row::Lookup),
}

/// Retain this at instance creation. Execution supplies buffers/live counts to
/// these planners, without searching formats or catalogs again.
#[derive(Clone, Debug)]
pub struct OperationHandle {
    pub compute: Compute,
    pub gather: Option<row::Gather>,
    pub quantizer: Option<row::Quantizer>,
}

impl OperationHandle {
    pub fn row_stages(&self) -> Vec<row::RowStage> {
        let mut stages = Vec::new();
        if let Some(gather) = &self.gather {
            stages.push(gather.stage().clone());
        }
        if let Some(quantizer) = &self.quantizer {
            stages.extend_from_slice(quantizer.stages());
        }
        if let Compute::Lookup(lookup) = &self.compute {
            stages.push(lookup.stage().clone());
        }
        stages
    }
    pub fn cubin_specs(&self, owner: &str) -> Result<Vec<CubinSpec>, Refusal> {
        let mut specs = Vec::new();
        let stages = self.row_stages();
        if !stages.is_empty() {
            specs.push(row::cubin_spec(&format!("{owner}_rows_v1"), &stages)?);
        }
        specs.extend(self.compute_cubin_spec(owner)?);
        Ok(specs)
    }
    /// The cubin of the compute entry; row stages have their own.
    fn compute_cubin_spec(&self, owner: &str) -> Result<Option<CubinSpec>, Refusal> {
        Ok(match &self.compute {
            Compute::Dense(spec) => {
                let mut cubin = gemm::cubin_spec(std::slice::from_ref(spec))?;
                cubin.name = format!("{owner}_dense_v1");
                Some(cubin)
            }
            // Packed ring entries have their own cubin, so the engine's
            // cubin keeps its single header.
            Compute::Grouped(operation) => Some(moe::cubin_spec(
                &if operation.packed_ring() {
                    format!("{owner}_grouped_ring_v1")
                } else {
                    format!("{owner}_grouped_v1")
                },
                std::slice::from_ref(operation),
            )?),
            Compute::Lookup(_) => None,
        })
    }
    pub fn entries(&self) -> Result<Vec<EntrySpec>, Refusal> {
        Ok(self
            .cubin_specs("precision")?
            .into_iter()
            .flat_map(|s| s.entries)
            .collect())
    }
    pub fn declarations(&self) -> Result<Vec<EntryDeclaration>, Refusal> {
        let mut declarations: Vec<_> = self.row_stages().iter().map(|s| s.declaration()).collect();
        let (query, knobs, launch, extra, implementation) = match &self.compute {
            Compute::Lookup(_) => return Ok(declarations),
            Compute::Dense(spec) => {
                let StaticQuery::Gemm(query) = spec.key.query else {
                    unreachable!()
                };
                let split_k =
                    gemm::required_split_k(&spec.key).unwrap_or(gemm::choose_split_k(&spec.key)?);
                let launch = gemm::plan(
                    &spec.key,
                    RuntimeCounts {
                        m: query.m_capacity,
                        batch: 1,
                        context: 0,
                        routed_rows: 0,
                        split_k,
                    },
                )?;
                (
                    query,
                    spec.key.knobs,
                    launch,
                    format!(
                        "fragment={}; kernel={:?}",
                        gemm::fragment_weight_enabled(&spec.key),
                        gemm::kernel(&spec.key)?
                    ),
                    "sm120.dense.v1",
                )
            }
            Compute::Grouped(operation) => {
                let query = operation.query;
                let launch = operation.plan(RuntimeCounts {
                    m: operation.tokens_capacity,
                    batch: 1,
                    context: 0,
                    routed_rows: query.m_capacity,
                    split_k: operation.knobs.split_k_capacity,
                })?;
                (
                    query,
                    operation.knobs,
                    launch,
                    format!(
                        "phase={:?}; layout={:?}; packed={:?}; scale_tile={:?}; weight_stages={}",
                        operation.phase,
                        operation.layout,
                        operation.packed_contract,
                        operation.weight_scale_layout(),
                        operation.weight_stages
                    ),
                    "sm120.grouped.v1",
                )
            }
        };
        let rows: BTreeSet<_> = declarations.iter().map(|d| d.symbol.clone()).collect();
        for entry in self
            .entries()?
            .into_iter()
            .filter(|e| !rows.contains(&e.symbol))
        {
            declarations.push(EntryDeclaration {
                symbol: entry.symbol,
                operation: if query.grouped.is_some() {
                    "grouped projection"
                } else {
                    "dense projection"
                }
                .into(),
                contract: query.numerical_contract_id()?.to_string(),
                implementation: implementation.into(),
                requirements: [
                    ("target", "sm120f".into()),
                    ("query", format!("{query:?}")),
                    ("knobs", format!("{knobs:?}")),
                    ("launch", format!("{launch:?}")),
                    ("packing", extra.clone()),
                    ("arguments", format!("{:?}", entry.params)),
                    ("entry", entry.body),
                    (
                        "quantizer_codes_bytes",
                        self.quantizer
                            .as_ref()
                            .map_or(0, |q| q.codes_bytes())
                            .to_string(),
                    ),
                    (
                        "quantizer_scales_bytes",
                        self.quantizer
                            .as_ref()
                            .map_or(0, |q| q.scales_bytes())
                            .to_string(),
                    ),
                    (
                        "quantizer_scratch_bytes",
                        self.quantizer
                            .as_ref()
                            .map_or(0, |q| q.scratch_bytes())
                            .to_string(),
                    ),
                ]
                .into_iter()
                .map(|(k, v)| (k.into(), v))
                .collect(),
                correctness_case: None,
            });
        }
        Ok(declarations)
    }
}

/// Where a selection places launch-shape row stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchRows {
    /// In the owner's rows cubin with its other row stages.
    Owner,
    /// Left out: the caller places them in one cubin its owners share
    /// ([`launch_row_cubin`]), since they serve every shape.
    Shared,
}

/// Group a consumer's supplied operations without registering a model profile.
/// Several graph/phase uses may share exactly the same compiled entry; a
/// conflicting symbol or resource declaration is an error, never first-wins.
pub fn selected_cubins(owner: &str, offers: &[KernelOffer]) -> Result<Vec<SelectedCubin>, String> {
    selected_cubins_at(owner, offers, &crate::cuda_source_root())
}

/// Reconstruct a supplied selection from an exported SDK's source root.
pub fn selected_cubins_at(
    owner: &str,
    offers: &[KernelOffer],
    cuda: &std::path::Path,
) -> Result<Vec<SelectedCubin>, String> {
    selected_cubins_with(owner, offers, cuda, LaunchRows::Owner)
}

/// One cubin of launch-shape row stages that several owners share. Uses of
/// one entry repeat its symbol; each must have the same code and declaration.
pub fn launch_row_cubin(name: &str, stages: &[row::RowStage]) -> Result<SelectedCubin, String> {
    let mut unique = BTreeMap::<String, row::RowStage>::new();
    for stage in stages {
        if stage.shapes() != row::ShapeBinding::Launch {
            return Err(format!(
                "{name} holds launch-shape row stages only, not {}",
                stage.entry().symbol
            ));
        }
        if let Some(prior) = unique.get(&stage.entry().symbol) {
            if format!("{:?}", prior.entry()) != format!("{:?}", stage.entry())
                || prior.declaration() != stage.declaration()
            {
                return Err(format!(
                    "conflicting launch-shape row stage {}",
                    stage.entry().symbol
                ));
            }
        } else {
            unique.insert(stage.entry().symbol.clone(), stage.clone());
        }
    }
    let stages = unique.into_values().collect::<Vec<_>>();
    SelectedCubin::new(
        row::cubin_spec(name, &stages).map_err(|e| e.to_string())?,
        stages.iter().map(|s| s.declaration()).collect(),
    )
    .map_err(|e| e.to_string())
}

/// [`selected_cubins_at`], placing launch-shape row stages per `rows`.
pub fn selected_cubins_with(
    owner: &str,
    offers: &[KernelOffer],
    cuda: &std::path::Path,
    rows: LaunchRows,
) -> Result<Vec<SelectedCubin>, String> {
    if offers.is_empty() || offers.len() > 8192 {
        return Err("invalid precision offer count".into());
    }
    let mut groups = BTreeMap::<String, CubinSpec>::new();
    let mut declarations = BTreeMap::<String, EntryDeclaration>::new();
    // The owner's row stages form one cubin whose headers follow from all of
    // them, since one offer's stages may bind shapes another way than another's.
    let mut row_stages = BTreeMap::<String, row::RowStage>::new();
    for offer in offers {
        for declaration in offer.handle.declarations().map_err(|e| e.to_string())? {
            if let Some(prior) =
                declarations.insert(declaration.symbol.clone(), declaration.clone())
                && prior != declaration
            {
                return Err(format!(
                    "conflicting precision declaration for {}",
                    declaration.symbol
                ));
            }
        }
        for stage in offer.handle.row_stages() {
            if rows == LaunchRows::Shared && stage.shapes() == row::ShapeBinding::Launch {
                continue;
            }
            if let Some(prior) = row_stages.get(&stage.entry().symbol) {
                if format!("{:?}", prior.entry()) != format!("{:?}", stage.entry()) {
                    return Err(format!(
                        "conflicting precision entry {}",
                        stage.entry().symbol
                    ));
                }
            } else {
                row_stages.insert(stage.entry().symbol.clone(), stage);
            }
        }
        let Some(mut spec) = offer
            .handle
            .compute_cubin_spec(owner)
            .map_err(|e| e.to_string())?
        else {
            continue;
        };
        spec.sources_root = cuda.into();
        if let Some(prior) = groups.get_mut(&spec.name) {
            if prior.sources_root != spec.sources_root
                || prior.include_dirs != spec.include_dirs
                || prior.headers != spec.headers
                || prior.defines != spec.defines
            {
                return Err(format!(
                    "conflicting precision cubin inputs for {}",
                    spec.name
                ));
            }
            for entry in spec.entries {
                if let Some(old) = prior.entries.iter().find(|e| e.symbol == entry.symbol) {
                    if format!("{old:?}") != format!("{entry:?}") {
                        return Err(format!("conflicting precision entry {}", entry.symbol));
                    }
                } else {
                    prior.entries.push(entry);
                }
            }
        } else {
            groups.insert(spec.name.clone(), spec);
        }
    }
    if !row_stages.is_empty() {
        let stages = row_stages.into_values().collect::<Vec<_>>();
        let mut spec =
            row::cubin_spec(&format!("{owner}_rows_v1"), &stages).map_err(|e| e.to_string())?;
        spec.sources_root = cuda.into();
        groups.insert(spec.name.clone(), spec);
    }
    groups
        .into_values()
        .map(|mut spec| {
            spec.entries.sort_by(|a, b| a.symbol.cmp(&b.symbol));
            let records = spec
                .entries
                .iter()
                .map(|e| declarations[&e.symbol].clone())
                .collect();
            SelectedCubin::new(spec, records).map_err(|e| e.to_string())
        })
        .collect()
}

/// How offers bind their quantizer and gather stages' shapes: compiled,
/// since their launch-shape forms measured slower (kernel-lab
/// `launch_shape_row_stage_timing`).
const ROW_SHAPES: row::ShapeBinding = row::ShapeBinding::Compiled;

#[derive(Clone, Debug)]
pub struct KernelOffer {
    pub(crate) usage: OperationUse,
    pub(crate) handle: OperationHandle,
    pub(crate) nv_global: row::Nvfp4Global,
    pub(crate) lookup: Option<row::LookupRequest>,
    pub(crate) invariance: crate::invariance::Invariance,
}

fn quantizer(
    format: ActivationFormat,
    width: u32,
    capacity: u32,
    global: row::Nvfp4Global,
    symbol: &str,
    scale_rule: Option<DynamicScale>,
) -> Result<Option<row::Quantizer>, Refusal> {
    let output = match format {
        ActivationFormat::Bf16 => {
            if global != row::Nvfp4Global::Absent || scale_rule.is_some() {
                return Err(Refusal("BF16 operands cannot bind an activation global"));
            }
            return Ok(None);
        }
        ActivationFormat::Fp8Group128(mode) => OutputQuantization::Fp8Group128(mode),
        ActivationFormat::Mxfp8 => OutputQuantization::Mxfp8,
        ActivationFormat::Nvfp4 => OutputQuantization::Nvfp4,
    };
    let quantizer = row::Quantizer::new(
        width,
        capacity,
        output,
        global,
        symbol,
        ROW_SHAPES,
        row::RowResources::SM120F,
    )?;
    Ok(Some(match scale_rule {
        Some(rule) => quantizer.with_dynamic_scale(rule, symbol)?,
        None => quantizer,
    }))
}

/// What a generated dense offer keeps of its rows' bits, from the kernel
/// it launches. Every claimed form sums each output element over the
/// whole K in one fixed order (no split, one warp per element, its weights
/// decoded exactly to BF16 before the MMA) and stores only its own row:
/// - BF16 fragment prefill: a row's bits in any chunk and beside any rows;
/// - an unsplit tiled entry over MXFP8 weights: the same at prefill
///   capacities, and at decode capacities (32 rows or fewer) each row's
///   bits launched alone through the operation's capacity-1 offer, which
///   the generator builds from the same kernel and tile (`run_a16`);
/// - an unsplit BF16 fragment decode entry: likewise one-row exact;
/// - the one-split tiled arithmetic in the forms and weights kernel-lab's
///   dense rows kit cases cover (`dense_rows`, over a bundle's offers):
///   `Tiled { splits: 1 }` over BF16 or NVFP4 weights and `TiledChains`
///   over BF16 weights at decode capacities, `StripChains` over BF16 or FP8
///   128x128 block weights at decode capacities, `Tiled { splits: 1 }`
///   over NVFP4 and `TiledChains` over BF16 or FP8 128x128 block weights
///   at prefill capacities (at any tile knobs). Each output element is a
///   function of its row alone at every compiled capacity: the tokens are
///   the MMA's columns, which never mix, every form sums each element's K
///   chains in K order (`TiledChains` and `StripChains` keep each chain
///   and add them in K order at any split or strip count), and no form
///   splits by row count. So each row has the bits of that row launched
///   alone through the operation's capacity-1 offer: one-row bits at
///   decode capacities, and every row claim at prefill ones.
///
/// Other forms claim nothing until kernel-lab proves them.
fn dense_invariance(
    kernel: kernel_core::contract::DenseKernel,
    query: &GemmQuery,
) -> crate::invariance::Invariance {
    use crate::invariance::Invariance;
    use kernel_core::contract::DenseKernel;
    let decode = query.m_capacity <= 32;
    if query.activation != ActivationFormat::Bf16 || query.grouped.is_some() {
        return Invariance::NONE;
    }
    match (kernel, query.weight) {
        (DenseKernel::FragmentPrefill { .. }, _) => Invariance::PREFILL,
        (DenseKernel::Tiled { splits: 1 }, WeightFormat::Mxfp8) if decode => {
            Invariance::one_row(query.m_capacity)
        }
        (DenseKernel::Tiled { splits: 1 }, WeightFormat::Mxfp8) => Invariance::PREFILL,
        (DenseKernel::FragmentDecode { splits: 1 }, WeightFormat::Bf16) if decode => {
            Invariance::one_row(query.m_capacity)
        }
        (DenseKernel::Tiled { splits: 1 }, WeightFormat::Bf16 | WeightFormat::Nvfp4)
        | (DenseKernel::TiledChains { .. }, WeightFormat::Bf16)
        | (
            DenseKernel::StripChains { .. },
            WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_),
        ) if decode => Invariance::one_row(query.m_capacity),
        (DenseKernel::Tiled { splits: 1 }, WeightFormat::Nvfp4)
        | (
            DenseKernel::TiledChains { .. },
            WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_),
        ) if !decode => Invariance::any_rows(query.m_capacity),
        _ => Invariance::NONE,
    }
}

/// What a generated grouped offer keeps of its token rows' bits. Its
/// kernels never split K: every routed row's accumulator adds its K16 MMAs
/// in ascending K from zero over exact BF16 weights, the packed engine and
/// the decode ring alike (`cuda/moe/grouped_packed.cuh`), and each routed
/// row is stored once. So with BF16 activations and one K split:
/// - prefill: a token's rows in any chunk and beside any tokens;
/// - decode over NVFP4 weights: each token's rows launched alone through
///   the operation's capacity-1 offer.
///
/// The combine over a token's experts belongs to the model's own op.
fn grouped_invariance(compute: &moe::GroupedOperation) -> crate::invariance::Invariance {
    use crate::invariance::Invariance;
    if compute.query.activation != ActivationFormat::Bf16
        || compute.knobs.split_k_capacity != 1
        || compute.reduction_symbol().is_some()
    {
        return Invariance::NONE;
    }
    match compute.phase {
        moe::Phase::Prefill => Invariance::PREFILL,
        moe::Phase::Decode if compute.query.weight == WeightFormat::Nvfp4 => {
            Invariance::one_row(compute.tokens_capacity)
        }
        moe::Phase::Decode => Invariance::NONE,
    }
}

impl KernelOffer {
    pub fn dense(
        request: gemm::DenseRequest,
        usage: OperationUse,
        global: row::Nvfp4Global,
        quantizer_symbol: &str,
    ) -> Result<Self, Refusal> {
        if usage.capacity > request.query.m_capacity {
            return Err(Refusal(
                "dense offer does not cover its advertised capacity",
            ));
        }
        let query = request.query;
        let invariance = dense_invariance(request.kernel, &query);
        let compute = gemm::instantiate(request)?;
        let quantizer = quantizer(
            query.activation,
            query.k,
            query.m_capacity,
            global,
            quantizer_symbol,
            None,
        )?;
        Ok(Self {
            usage,
            handle: OperationHandle {
                compute: Compute::Dense(compute),
                gather: None,
                quantizer,
            },
            nv_global: global,
            lookup: None,
            invariance,
        })
    }
    /// An offer of a compiled catalog entry (`gemm::catalog`): a dense GEMM
    /// with BF16 activations, whose capability is derived as a constructed
    /// offer's is. The entry is already in its catalog's cubin, so nothing
    /// new is instantiated.
    pub fn catalog_dense(entry: InstantiationSpec, usage: OperationUse) -> Result<Self, Refusal> {
        let StaticQuery::Gemm(query) = entry.key.query else {
            return Err(Refusal("catalog offer requires a GEMM entry"));
        };
        if query.grouped.is_some() || query.activation != ActivationFormat::Bf16 {
            return Err(Refusal(
                "catalog offer requires a dense entry with BF16 activations",
            ));
        }
        if usage.capacity > query.m_capacity {
            return Err(Refusal(
                "dense offer does not cover its advertised capacity",
            ));
        }
        Ok(Self {
            usage,
            handle: OperationHandle {
                compute: Compute::Dense(entry),
                gather: None,
                quantizer: None,
            },
            nv_global: row::Nvfp4Global::Absent,
            lookup: None,
            invariance: crate::invariance::Invariance::NONE,
        })
    }
    pub fn grouped(
        request: moe::GroupedRequest,
        usage: OperationUse,
        global: row::Nvfp4Global,
        gather_symbol: &str,
        quantizer_symbol: &str,
    ) -> Result<Self, Refusal> {
        if usage.capacity > request.tokens_capacity {
            return Err(Refusal(
                "grouped offer does not cover its advertised capacity",
            ));
        }
        let shape = request.shape;
        let compute = moe::instantiate(request)?;
        let invariance = grouped_invariance(&compute);
        let gather = if compute.query.activation == ActivationFormat::Bf16 {
            None
        } else {
            Some(row::Gather::new(
                compute.query.k,
                compute.tokens_capacity,
                shape.experts,
                shape.top_k,
                16,
                if compute.projection == moe::Projection::GateUp {
                    row::GatherInput::Tokens
                } else {
                    row::GatherInput::Routed
                },
                false,
                gather_symbol,
                ROW_SHAPES,
                row::RowResources::SM120F,
            )?)
        };
        let capacity = gather
            .as_ref()
            .map_or(compute.tokens_capacity, |g| g.stage().capacity());
        let quantizer = quantizer(
            compute.query.activation,
            compute.query.k,
            capacity,
            global,
            quantizer_symbol,
            None,
        )?;
        Ok(Self {
            usage,
            handle: OperationHandle {
                compute: Compute::Grouped(compute),
                gather,
                quantizer,
            },
            nv_global: global,
            lookup: None,
            invariance,
        })
    }
    pub fn lookup(
        request: row::LookupRequest,
        usage: OperationUse,
        symbol: &str,
    ) -> Result<Self, Refusal> {
        if usage.capacity > request.capacity {
            return Err(Refusal(
                "lookup offer does not cover its advertised capacity",
            ));
        }
        let compute = row::Lookup::new(request, symbol, row::RowResources::SM120F)?;
        Ok(Self {
            usage,
            handle: OperationHandle {
                compute: Compute::Lookup(compute),
                gather: None,
                quantizer: None,
            },
            nv_global: row::Nvfp4Global::Absent,
            lookup: Some(request),
            invariance: crate::invariance::Invariance::COPY,
        })
    }
    pub fn lookup_request(&self) -> Option<row::LookupRequest> {
        self.lookup
    }
    pub fn usage(&self) -> OperationUse {
        self.usage
    }
    /// Bind the declared input scale rule to this offer's quantizer. The
    /// compute instruction path remains the one the builder actually emits.
    pub fn with_dynamic_scale(mut self, rule: DynamicScale, symbol: &str) -> Result<Self, Refusal> {
        let (format, width, capacity) = match &self.handle.compute {
            Compute::Dense(spec) => {
                let StaticQuery::Gemm(q) = spec.key.query else {
                    unreachable!()
                };
                (q.activation, q.k, q.m_capacity)
            }
            Compute::Grouped(g) => (
                g.query.activation,
                g.query.k,
                self.handle
                    .gather
                    .as_ref()
                    .map_or(g.tokens_capacity, |g| g.stage().capacity()),
            ),
            Compute::Lookup(_) => return Err(Refusal("lookup has no activation quantizer")),
        };
        self.handle.quantizer =
            quantizer(format, width, capacity, self.nv_global, symbol, Some(rule))?;
        Ok(self)
    }

    /// Check a requested MX instruction claim against this supplied entry.
    /// A claim cannot change a decoded-BF16 entry into a native MMA entry.
    pub fn require_execution(
        &self,
        execution: kernel_core::precision::execution::BlockScaledExecution,
    ) -> Result<(), Refusal> {
        let capability = self.capability()?;
        execution
            .require(capability.weight, &capability.arithmetic)
            .map_err(|_| Refusal("MX execution claim disagrees with the supplied instruction path"))
    }
    pub fn handle(&self) -> &OperationHandle {
        &self.handle
    }

    /// Retain the catalog's claim even when the same kernel can serve other uses.
    pub fn catalog_variant(
        entry: &gemm::GemmVariant,
        usage: OperationUse,
    ) -> Result<Self, Refusal> {
        let mut offer = Self::catalog_dense(entry.instantiation.clone(), usage)?;
        offer.invariance = entry.invariance;
        Ok(offer)
    }
    /// Retain the owning declaration's exactness claims after schedule selection.
    pub fn with_invariance(mut self, invariance: crate::invariance::Invariance) -> Self {
        self.invariance = invariance;
        self
    }
    pub fn invariance(&self) -> crate::invariance::Invariance {
        self.invariance
    }

    /// Derive portable admission from the same validated implementation that
    /// supplies generated code, launch bounds and packing. No model names enter
    /// this contract, and compiler SDK consumers need not link this crate.
    pub fn capability(&self) -> Result<OperationCapability, Refusal> {
        let (
            kind,
            n,
            k,
            operands,
            weight,
            activation,
            epilogues,
            layout,
            scales,
            globals,
            arithmetic,
        ) = match &self.handle.compute {
            Compute::Lookup(lookup) => {
                let r = self.lookup.expect("lookup offer");
                (
                    OperationKind::Lookup,
                    r.vocab,
                    u64::from(r.width),
                    1,
                    r.weight,
                    None,
                    Default::default(),
                    WeightLayout::row_major(),
                    match r.weight {
                        WeightFormat::Bf16 => vec![],
                        WeightFormat::Nvfp4 => vec![DType::Fp8E4M3, DType::Ue4M3],
                        _ => vec![r.global_scale.expect("lookup scale dtype")],
                    },
                    if r.weight == WeightFormat::Nvfp4 {
                        GlobalScaleCapability::SharedScalar
                    } else {
                        GlobalScaleCapability::Absent
                    },
                    ArithmeticSpec {
                        weight_operand: DType::Bf16,
                        accumulator: DType::F32,
                        output: DType::Bf16,
                        decode_order: if r.weight == WeightFormat::Nvfp4 {
                            DecodeOrder::ExactWeightToBf16
                        } else {
                            DecodeOrder::CodeThenScale
                        },
                        rounding: Rounding::NearestEven,
                        numerical_contract: lookup.stage().numerical_contract().into(),
                    },
                )
            }
            compute => {
                let (query, layout, globals, order, algorithm) = match compute {
                    Compute::Dense(spec) => {
                        let StaticQuery::Gemm(query) = spec.key.query else {
                            unreachable!()
                        };
                        let fragment = gemm::fragment_weight_enabled(&spec.key);
                        (
                            query,
                            if fragment {
                                WeightLayout {
                                    id: if query.weight == WeightFormat::Bf16 {
                                        "bf16-fragment"
                                    } else {
                                        "fp8-fragment"
                                    }
                                    .into(),
                                    version: 1,
                                }
                            } else {
                                WeightLayout::row_major()
                            },
                            if query.weight == WeightFormat::Nvfp4 {
                                GlobalScaleCapability::SharedScalar
                            } else {
                                GlobalScaleCapability::Absent
                            },
                            dense_decode_order(&query, fragment),
                            if fragment {
                                "sm120.dense.fragment"
                            } else {
                                "sm120.dense.packed"
                            },
                        )
                    }
                    Compute::Grouped(operation) => {
                        let globals = operation
                            .packed_contract
                            .expect("typed offer")
                            .scales
                            .weight_globals;
                        (
                            operation.query,
                            WeightLayout {
                                id: format!("moe-packed-{globals:?}"),
                                version: 1,
                            },
                            match globals {
                                moe::GlobalScaleLayout::Absent => GlobalScaleCapability::Absent,
                                moe::GlobalScaleLayout::Scalar => {
                                    GlobalScaleCapability::SharedScalar
                                }
                                moe::GlobalScaleLayout::ExpertProjection => {
                                    GlobalScaleCapability::ExpertProjection
                                }
                            },
                            grouped_decode_order(
                                operation.query.activation,
                                operation.query.weight,
                            ),
                            "sm120.grouped.packed",
                        )
                    }
                    Compute::Lookup(_) => unreachable!(),
                };
                let operands = if query.epilogues.swiglu.is_some() {
                    2
                } else {
                    1
                };
                let activation = match (query.activation, self.nv_global) {
                    (ActivationFormat::Bf16, _) => ActivationCapability::Bf16,
                    (ActivationFormat::Fp8Group128(mode), _) => {
                        ActivationCapability::Fp8Group128(mode)
                    }
                    (ActivationFormat::Mxfp8, _) => match self
                        .handle
                        .quantizer
                        .as_ref()
                        .and_then(row::Quantizer::scale_rule)
                    {
                        Some(DynamicScale::CeilPowerOfTwo { amax_floor_bits }) => {
                            ActivationCapability::Mxfp8Ceil { amax_floor_bits }
                        }
                        Some(DynamicScale::MxPowerOfTwo) => ActivationCapability::Mxfp8,
                        _ => {
                            return Err(Refusal(
                                "MXFP8 capability requires its declared scale rule",
                            ));
                        }
                    },
                    (ActivationFormat::Nvfp4, row::Nvfp4Global::Supplied) => {
                        ActivationCapability::Nvfp4StaticScalar
                    }
                    (ActivationFormat::Nvfp4, row::Nvfp4Global::TensorAmax) => {
                        ActivationCapability::Nvfp4TensorAmax
                    }
                    _ => return Err(Refusal("NVFP4 capability requires an explicit quantizer")),
                };
                (
                    query
                        .grouped
                        .map_or(OperationKind::Dense, |g| OperationKind::Grouped {
                            experts: g.experts,
                            top_k: g.top_k,
                        }),
                    u64::from(query.n / operands),
                    u64::from(query.k),
                    operands as u8,
                    query.weight,
                    Some(activation),
                    query.epilogues,
                    layout,
                    match query.weight {
                        WeightFormat::Bf16 => vec![],
                        WeightFormat::Nvfp4 => vec![DType::Fp8E4M3, DType::Ue4M3],
                        WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => vec![DType::Ue8M0],
                        WeightFormat::Fp8PerTensor => vec![DType::F32],
                        _ => vec![DType::Bf16, DType::F32],
                    },
                    globals,
                    ArithmeticSpec {
                        weight_operand: mma_weight(&query),
                        accumulator: DType::F32,
                        output: query.output,
                        decode_order: order,
                        rounding: Rounding::NearestEven,
                        numerical_contract: format!(
                            "{algorithm}.{}",
                            query.numerical_contract_id()?
                        ),
                    },
                )
            }
        };
        let capability = OperationCapability {
            kind,
            target: kernel_core::contract::Target::Sm120f,
            usage: self.usage,
            geometry: MatrixGeometry { rows: n, k },
            operands,
            weight,
            activation,
            epilogues,
            layout,
            local_scale_dtypes: scales,
            globals,
            shared_local_scale: weight == WeightFormat::Fp8PerTensor && operands > 1,
            implementation: arithmetic.numerical_contract.clone(),
            arithmetic,
            stages: self
                .handle
                .entries()?
                .into_iter()
                .map(|e| e.symbol)
                .collect(),
        };
        capability
            .validate()
            .map_err(|_| Refusal("invalid derived operation capability"))?;
        Ok(capability)
    }
}

pub fn dense_decode_order(query: &GemmQuery, fragment: bool) -> DecodeOrder {
    match query.activation {
        ActivationFormat::Bf16 => {
            if query.weight == WeightFormat::Nvfp4 {
                return DecodeOrder::ExactLocalThenGlobal;
            }
            if matches!(query.weight, WeightFormat::Mxfp8 | WeightFormat::Mxfp4) {
                return DecodeOrder::ExactScaledOperand;
            }
            if matches!(
                query.weight,
                WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
            ) && (fragment
                || query.epilogues.swiglu
                    != Some(kernel_core::contract::SwigluColumns::Interleaved))
            {
                DecodeOrder::BlockScaleAfterMma
            } else {
                DecodeOrder::ScaleThenCode
            }
        }
        ActivationFormat::Fp8Group128(_) => DecodeOrder::ScaleProductAfterBlockMma,
        ActivationFormat::Mxfp8 => DecodeOrder::NativeBlockScaledMma,
        ActivationFormat::Nvfp4 => DecodeOrder::NativeBlockScaledMmaThenGlobals,
    }
}

fn grouped_decode_order(activation: ActivationFormat, weight: WeightFormat) -> DecodeOrder {
    match activation {
        ActivationFormat::Bf16 if weight == WeightFormat::Nvfp4 => {
            DecodeOrder::ExactLocalThenGlobal
        }
        ActivationFormat::Bf16 if matches!(weight, WeightFormat::Mxfp8 | WeightFormat::Mxfp4) => {
            DecodeOrder::ExactScaledOperand
        }
        ActivationFormat::Bf16 => DecodeOrder::CodeThenScale,
        ActivationFormat::Fp8Group128(_) => DecodeOrder::SuccessiveScalesAfterBlockMma,
        ActivationFormat::Mxfp8 => DecodeOrder::NativeBlockScaledMma,
        ActivationFormat::Nvfp4 => DecodeOrder::NativeBlockScaledMmaThenGlobals,
    }
}

/// The capability of a compiled MXFP4 fragment grouped catalog entry
/// (`moe::catalog`) with BF16 activations gathered by token
/// (`InputPath::A16Gather`), for `usage` within its token capacity: its
/// experts, routing and per-operand matrix, its UE8M0 block-32 scales and
/// its arithmetic, naming the entry itself as its stage. The fragment
/// converter folds each scale into the BF16 operand before the MMA
/// (also used by `mimo26_fragments.cuh`), with an exact scaled operand. Other
/// grouped entries are refused: the FP8 fragment entries scale the MMA's
/// partial sums instead, which this derivation does not state.
pub fn catalog_grouped_capability(
    variant: &moe::Variant,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let query = variant.query;
    let grouped = query.grouped.ok_or(Refusal(
        "grouped catalog capability requires a grouped entry",
    ))?;
    if query.activation != ActivationFormat::Bf16 || variant.path != moe::InputPath::A16Gather {
        return Err(Refusal(
            "grouped catalog capability requires BF16 activations gathered by token",
        ));
    }
    if query.weight != WeightFormat::Mxfp4 || variant.layout != moe::WeightLayout::Mxfp4Fragment {
        return Err(Refusal(
            "grouped catalog capability states MXFP4 fragment entries only",
        ));
    }
    if usage.capacity > variant.tokens_capacity {
        return Err(Refusal(
            "grouped offer does not cover its advertised capacity",
        ));
    }
    let operands: u8 = if query.epilogues.swiglu.is_some() {
        2
    } else {
        1
    };
    let capability = OperationCapability {
        kind: OperationKind::Grouped {
            experts: grouped.experts,
            top_k: grouped.top_k,
        },
        target: kernel_core::contract::Target::Sm120f,
        usage,
        geometry: MatrixGeometry {
            rows: u64::from(query.n / u32::from(operands)),
            k: u64::from(query.k),
        },
        operands,
        weight: query.weight,
        activation: Some(ActivationCapability::Bf16),
        epilogues: query.epilogues,
        layout: WeightLayout {
            id: "moe-mxfp4-fragment".into(),
            version: 1,
        },
        local_scale_dtypes: vec![DType::Ue8M0],
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: ArithmeticSpec {
            weight_operand: mma_weight(&query),
            accumulator: DType::F32,
            output: query.output,
            decode_order: DecodeOrder::ExactScaledOperand,
            rounding: Rounding::NearestEven,
            numerical_contract: format!("sm120.grouped.catalog.{}", query.numerical_contract_id()?),
        },
        implementation: variant.symbol.clone(),
        stages: vec![variant.symbol.clone()],
    };
    capability
        .validate()
        .map_err(|_| Refusal("invalid grouped catalog capability"))?;
    Ok(capability)
}

/// The capability of a compiled embedding-gather row entry (`row::catalog`):
/// a BF16 table of `vocab` rows of `width` copied by token ID, for `usage`,
/// naming the entry itself as its stage.
pub fn catalog_lookup_capability(
    entry: &crate::row::RowInstantiation,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let kernel_core::contract::RowQuery::EmbeddingGather { vocab, width } = entry.query else {
        return Err(Refusal(
            "lookup catalog capability requires an embedding gather entry",
        ));
    };
    let capability = OperationCapability {
        kind: OperationKind::Lookup,
        target: kernel_core::contract::Target::Sm120f,
        usage,
        geometry: MatrixGeometry {
            rows: u64::from(vocab),
            k: u64::from(width),
        },
        operands: 1,
        weight: WeightFormat::Bf16,
        activation: None,
        epilogues: Default::default(),
        layout: WeightLayout::row_major(),
        local_scale_dtypes: vec![],
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: DecodeOrder::CodeThenScale,
            rounding: Rounding::NearestEven,
            numerical_contract: format!(
                "sm120.row.catalog.{}",
                entry.query.numerical_contract_id()?
            ),
        },
        implementation: entry.symbol.clone(),
        stages: vec![entry.symbol.clone()],
    };
    capability
        .validate()
        .map_err(|_| Refusal("invalid lookup catalog capability"))?;
    Ok(capability)
}

fn mma_weight(query: &GemmQuery) -> DType {
    if query.activation == ActivationFormat::Bf16 {
        DType::Bf16
    } else if matches!(query.weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) {
        DType::Fp4E2M1
    } else {
        DType::Fp8E4M3
    }
}

/// `available` comes from the authenticated native capability inventory, whose
/// entries the publisher builds and audits. Missing entries remain refusals so
/// auto can fall back for the complete logical operation before materialization.
pub struct CompiledInventory {
    offers: Vec<KernelOffer>,
    portable: CapabilityInventory,
}

impl CompiledInventory {
    pub fn new(
        offers: Vec<KernelOffer>,
        available: BTreeSet<String>,
        native_identity: String,
    ) -> Result<Self, Refusal> {
        let document = capabilities(&offers)?;
        let portable = CapabilityInventory::new(document, available, native_identity)
            .map_err(|_| Refusal("invalid supplied implementation inventory or native digest"))?;
        Ok(Self { offers, portable })
    }
    pub fn offers(&self) -> &[KernelOffer] {
        &self.offers
    }
}

pub fn capabilities(offers: &[KernelOffer]) -> Result<CapabilityDocument, Refusal> {
    let document = CapabilityDocument {
        version: 1,
        operations: offers
            .iter()
            .map(KernelOffer::capability)
            .collect::<Result<_, _>>()?,
    };
    document
        .validate()
        .map_err(|_| Refusal("invalid supplied capability document"))?;
    Ok(document)
}

/// Build-time publication data. Embed `bytes` in the portable capability ELF
/// section and the three digests in the existing native consumption note.
pub struct PublishedCapabilities {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub selection: String,
    pub dependencies: String,
    pub binding_dependency: String,
}

pub fn publish_capabilities(
    offers: &[KernelOffer],
    selection: &kernel_build::KernelSelection,
    built: &[kernel_build::BuiltCubin],
) -> Result<PublishedCapabilities, String> {
    let declared = selection
        .specs()
        .iter()
        .flat_map(|s| &s.entries)
        .map(|e| (e.symbol.as_str(), e))
        .collect::<BTreeMap<_, _>>();
    for offer in offers {
        for entry in offer.handle.entries().map_err(|e| e.to_string())? {
            if declared
                .get(entry.symbol.as_str())
                .is_none_or(|actual| format!("{actual:?}") != format!("{entry:?}"))
            {
                return Err(format!(
                    "precision entry {} differs from the audited selection",
                    entry.symbol
                ));
            }
        }
    }
    let document = capabilities(offers).map_err(|e| e.to_string())?;
    publish_capability_document(&document, selection, built)
}

/// Publish portable contracts for caller-owned adapters over existing typed
/// entries, together with shared offers. Each referenced stage must have an
/// explicit declaration in the audited selection. The caller owns the
/// arithmetic and buffer composition of its adapters, just as for other
/// model-local entry declarations.
pub fn publish_capability_document(
    document: &CapabilityDocument,
    selection: &kernel_build::KernelSelection,
    built: &[kernel_build::BuiltCubin],
) -> Result<PublishedCapabilities, String> {
    document.validate()?;
    let declared = selection
        .declarations()
        .values()
        .flatten()
        .map(|d| d.symbol.as_str())
        .collect::<BTreeSet<_>>();
    for stage in document.entries() {
        if !declared.contains(stage.as_str()) {
            return Err(format!(
                "precision stage {stage} has no declaration in the audited selection"
            ));
        }
    }
    let bytes = serde_json::to_vec(&document).map_err(|e| e.to_string())?;
    if bytes.len() > kernel_core::precision::capability::MAX_CAPABILITY_BYTES {
        return Err("supplied capability document exceeds its byte bound".into());
    }
    let sha256 = kernel_build::digest(&bytes);
    let contract = selection
        .audited_contract_identity(built)
        .map_err(|e| e.to_string())?;
    let dependencies = selection
        .audited_dependency_identity(built)
        .map_err(|e| e.to_string())?;
    let binding_dependency = kernel_build::digest(
        &serde_json::to_vec(&(
            "native-operation-bindings-v1",
            &contract,
            &dependencies,
            &sha256,
        ))
        .map_err(|e| e.to_string())?,
    );
    Ok(PublishedCapabilities {
        bytes,
        sha256,
        selection: contract,
        dependencies,
        binding_dependency,
    })
}

impl ImplementationProvider for CompiledInventory {
    type Handle = OperationHandle;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String> {
        let binding = self.portable.bind(operation, usage)?;
        Ok(Binding {
            identity: binding.identity,
            handle: self.offers[binding.handle].handle.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariance::Invariance;
    use kernel_core::contract::{DenseKernel, Target};
    use kernel_core::precision::ExecutionPhase;

    fn dense(m: u32, weight: WeightFormat, kernel: DenseKernel) -> Invariance {
        let usage = OperationUse {
            phase: if m <= 32 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            capacity: m,
            graph: m <= 32,
        };
        KernelOffer::dense(
            gemm::DenseRequest {
                query: GemmQuery {
                    weight,
                    ..gemm::bf16_query(m, 512, 1024, DType::Bf16)
                },
                kernel,
                target: Target::Sm120f,
                symbol: "chop_gemm_test_claims_v1".into(),
                knobs: None,
            },
            usage,
            row::Nvfp4Global::Absent,
            "unused",
        )
        .unwrap()
        .invariance()
    }

    /// The one-split tiled forms claim one-row bits at decode capacities
    /// and every row claim at prefill ones, over the weights their kit
    /// cases cover; other weights, and a tiled entry that splits K, claim
    /// nothing.
    #[test]
    fn one_split_tiled_offers_claim_their_rows() {
        let block = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
        let tiled = DenseKernel::Tiled { splits: 1 };
        let chains = DenseKernel::TiledChains { splits: 4 };
        let strips = DenseKernel::StripChains {
            strips: 2,
            splits: 2,
        };
        for m in [1, 4, 32] {
            for (kernel, weight) in [
                (tiled, WeightFormat::Bf16),
                (tiled, WeightFormat::Nvfp4),
                (chains, WeightFormat::Bf16),
                (strips, WeightFormat::Bf16),
                (strips, block),
            ] {
                let claim = dense(m, weight, kernel);
                assert_eq!(claim, Invariance::one_row(m), "{kernel:?} {weight:?}");
            }
            assert_eq!(dense(m, block, tiled), Invariance::NONE);
            assert_eq!(dense(m, block, chains), Invariance::NONE);
        }
        let one = DenseKernel::TiledChains { splits: 1 };
        for (kernel, weight) in [
            (tiled, WeightFormat::Nvfp4),
            (one, WeightFormat::Bf16),
            (one, block),
        ] {
            let claim = dense(8192, weight, kernel);
            assert_eq!(claim, Invariance::any_rows(8192), "{kernel:?} {weight:?}");
        }
        for weight in [WeightFormat::Bf16, block] {
            assert_eq!(dense(8192, weight, tiled), Invariance::NONE, "{weight:?}");
        }
        assert_eq!(
            dense(4, WeightFormat::Bf16, DenseKernel::Tiled { splits: 2 }),
            Invariance::NONE
        );
    }

    /// Unsplit tiled MXFP8 offers keep one-row bits at decode capacities
    /// and chunk bits at prefill ones; a split K, or a format no evidence
    /// covers, claims nothing.
    #[test]
    fn dense_offers_claim_from_their_kernel() {
        let tiled = DenseKernel::Tiled { splits: 1 };
        for m in [1, 2, 4, 8, 32] {
            assert_eq!(dense(m, WeightFormat::Mxfp8, tiled), Invariance::one_row(m));
        }
        assert_eq!(dense(8192, WeightFormat::Mxfp8, tiled), Invariance::PREFILL);
        assert_eq!(
            dense(4, WeightFormat::Mxfp8, DenseKernel::Tiled { splits: 2 }),
            Invariance::NONE
        );
        assert_eq!(
            dense(4, WeightFormat::Fp8PerChannel, tiled),
            Invariance::NONE
        );
        assert_eq!(
            dense(
                4,
                WeightFormat::Bf16,
                DenseKernel::FragmentDecode { splits: 1 }
            ),
            Invariance::one_row(4)
        );
        assert_eq!(
            dense(
                8192,
                WeightFormat::Bf16,
                DenseKernel::FragmentPrefill { segments: 4 }
            ),
            Invariance::PREFILL
        );
    }
}
