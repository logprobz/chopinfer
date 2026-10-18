//! Projections through a model's SM120 offers.

use std::collections::BTreeSet;
use std::sync::Arc;

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    ActivationFormat, ArgumentList, DType, LaunchPlan, Refusal, RuntimeCounts, StaticQuery,
    WeightFormat,
};
use kernel_core::precision::buffers::{ActivationGlobal, ProjectionBuffers};
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernel_core::precision::{OperationSpec, OperationUse};
use kernel_core::properties::Invariance;

use super::check_plan;
use super::select::{Candidate, choose};
use crate::gemm::{self, GemmBuffers};
use crate::moe::GroupedOperation;
use crate::precision::{Compute, KernelOffer, OperationHandle, capabilities};
use crate::row::precision::{Gather, LayoutCopy, LookupBuffers, QuantizerBuffers};

/// A projection use's SM120 implementation: the kernel offer bound to it,
/// an optional copy into the use's output layout, how its dense compute
/// launches, and whether its output holds only the step's rows.
#[derive(Clone, Debug)]
pub struct ProjectionOffer {
    pub offer: KernelOffer,
    pub copy: Option<LayoutCopy>,
    pub dense: DenseForm,
    /// The output holds the rows of each launch rather than the entry's
    /// capacity, as a vocabulary head's logits buffer does: the fragment
    /// kernels store only rows below the live count.
    pub live_output: bool,
}

/// How a dense offer's compute launches, which the model that built the
/// offer chose. Each form computes the same bits.
#[derive(Clone, Debug, Default)]
pub enum DenseForm {
    /// The entry over [`gemm::tensor_maps`].
    #[default]
    Tensor,
    /// The entry over [`gemm::fragment_maps`], which promote 256-byte L2
    /// lines, as a model that planned its fragment entries so binds them.
    Fragment,
    /// A medium-M entry ([`gemm::mid`]) in place of the offer's fragment
    /// prefill entry, over the fragment maps with the medium entry's
    /// activation box. It runs one persistent CTA per SM, at most its
    /// `max_rows` rows.
    Medium(Box<gemm::mid::Entry>),
}

/// The model's SM120 projection offers: every offer that binds an
/// operation at a use, each with the identity the precision document
/// authenticates. SM120 refuses an op that two offers would bind
/// differently, so the registration order of the offers never decides.
/// SM120 implements it twice: [`OfferList`] for a list of offers, which it
/// searches whole, and [`NamedOffers`] for a provider that names the one
/// offer of each operation and use.
pub trait ProjectionOffers: Send + Sync {
    /// Every offer that binds `op` at `usage`.
    fn bind_all(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Vec<Binding<ProjectionOffer>>, String>;
}

/// The offers of a provider that binds an operation and use by the
/// authenticated identity the model composes for it (a named composition),
/// never by searching a list: each binding is the only offer of its
/// operation and use. A provider that searches offers, such as one over a
/// `CapabilityInventory`, takes the first that binds, so its offers
/// register as an [`OfferList`] instead. A provider is never offers by
/// itself; it names this adapter:
///
/// ```compile_fail
/// use kernel_core::precision::execution::{Binding, ImplementationProvider};
/// use kernel_core::precision::{OperationSpec, OperationUse};
/// use kernels_sm120::backend::{ProjectionOffer, ProjectionOffers};
///
/// struct Search;
/// impl ImplementationProvider for Search {
///     type Handle = ProjectionOffer;
///     fn bind(&self, _: &OperationSpec, _: OperationUse) -> Result<Binding<ProjectionOffer>, String> {
///         Err("unbound".into())
///     }
/// }
/// let offers: Box<dyn ProjectionOffers> = Box::new(Search);
/// ```
pub struct NamedOffers<P>(pub P);

impl<P> ProjectionOffers for NamedOffers<P>
where
    P: ImplementationProvider<Handle = ProjectionOffer> + Send + Sync,
{
    fn bind_all(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Vec<Binding<ProjectionOffer>>, String> {
        self.0.bind(op, usage).map(|binding| vec![binding])
    }
}

/// A model's list of SM120 offers, bound through the capability inventory
/// of their document (`precision::capabilities`): every offer whose
/// capability binds an operation and use binds it.
pub struct OfferList {
    offers: Vec<ProjectionOffer>,
    inventory: CapabilityInventory,
}

impl OfferList {
    /// `offers`, whose compiled entries are `available`, under the native
    /// `dependency` digest their publisher authenticates.
    pub fn new(
        offers: Vec<ProjectionOffer>,
        available: BTreeSet<String>,
        dependency: String,
    ) -> Result<Self, Refusal> {
        // A capability names its offer's stages, the reference entry's for
        // a medium-M form, which launches another entry; such forms bind by
        // name ([`NamedOffers`]), under the identity of the stage they launch.
        if offers
            .iter()
            .any(|o| matches!(o.dense, DenseForm::Medium(_)))
        {
            return Err(Refusal(
                "a medium-M form binds by name, not through an offer list",
            ));
        }
        let kernels = offers.iter().map(|o| o.offer.clone()).collect::<Vec<_>>();
        let inventory = CapabilityInventory::new(capabilities(&kernels)?, available, dependency)
            .map_err(|_| Refusal("invalid supplied implementation inventory or native digest"))?;
        Ok(Self { offers, inventory })
    }
}

impl ProjectionOffers for OfferList {
    fn bind_all(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Vec<Binding<ProjectionOffer>>, String> {
        // The inventory binds the first capability its predicate accepts,
        // so each pass excludes the capabilities already bound.
        let operations = &self.inventory.document().operations;
        let mut bound = Vec::<Binding<usize>>::new();
        loop {
            let next = self.inventory.bind_matching(op, usage, |capability| {
                !bound
                    .iter()
                    .any(|b| std::ptr::eq(capability, &operations[b.handle]))
            });
            match next {
                Ok(binding) => bound.push(binding),
                Err(reason) if bound.is_empty() => return Err(reason),
                Err(_) => break,
            }
        }
        Ok(bound
            .into_iter()
            .map(|binding| Binding {
                identity: binding.identity,
                handle: self.offers[binding.handle].clone(),
            })
            .collect())
    }
}

/// The binding at an index of the bound offers, declared by its offer and
/// identity alone, so the index (the order the offers were listed in)
/// never decides which of equal launches stands for them.
struct Bound<'a>(usize, &'a Binding<ProjectionOffer>);

impl std::fmt::Debug for Bound<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.1.handle, self.1.identity)
    }
}

/// What a dense form launches: the form, and for a medium-M entry the
/// kernel and K segments it runs. The rest of the entry is settled by the
/// op or claims nothing about its launches.
fn dense_launch(dense: &DenseForm) -> String {
    match dense {
        DenseForm::Tensor => "Tensor".into(),
        DenseForm::Fragment => "Fragment".into(),
        DenseForm::Medium(medium) => {
            let gemm::mid::Entry {
                profile: _,    // where the entry is compiled
                reference: _,  // the op's own entry (`ProjectionOp::with_form`)
                query: _,      // the reference's query, which the op holds
                kernel,        // the kernel the launch runs
                segments,      // the reduction segments it passes
                rows: _,       // the rows a model selects it at; an op's served rows are bounded
                invariance: _, // claims
            } = medium.as_ref();
            format!("Medium {} {segments}", kernel.symbol())
        }
    }
}

/// What a bound offer launches: its operation, copy, dense form and output
/// bound; the binding returns its identity too.
fn bound_launch(binding: &Binding<ProjectionOffer>) -> String {
    let Binding {
        identity, // returned with the prepared op
        handle:
            ProjectionOffer {
                offer,
                copy,
                dense,
                live_output,
            },
    } = binding;
    let KernelOffer {
        usage: _,      // the use its capability binds at; the binding selects by it
        handle,        // the operation: its gather, quantizer and compute stages
        nv_global: _,  // an NVFP4 offer's capability, whose identity the key holds
        lookup: _,     // a lookup offer's capability, whose identity the key holds
        invariance: _, // claims
    } = offer;
    format!(
        "{handle:?} {copy:?} {} {live_output} {identity:?}",
        dense_launch(dense)
    )
}

/// The projection `op` at `usage` through the model's offer for it, serving
/// launches of up to `need.capacity` rows, at most the use's capacity.
/// Offers that would launch the same are one; two that would launch
/// differently are a tie, and the op is refused.
pub(super) fn prepare(
    offers: Option<&dyn ProjectionOffers>,
    op: &OperationSpec,
    usage: OperationUse,
    need: Need,
) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
    let offers = offers.ok_or(Refused::Unsupported)?;
    if need.capacity == 0 || need.capacity > usage.capacity {
        return Err(Refused::Rule(
            "a projection serves from one row to its use's capacity".into(),
        ));
    }
    let mut bound = offers.bind_all(op, usage).map_err(Refused::Rule)?;
    let chosen = choose(
        "projection",
        bound.iter().enumerate().map(|(index, binding)| {
            let name = binding.identity.stages.join("+");
            let invariance = binding.handle.offer.invariance();
            Candidate::new(
                Bound(index, binding),
                name,
                bound_launch(binding),
                invariance,
            )
        }),
        need,
    )?;
    let (index, guarantees) = (chosen.entry.0, chosen.guarantees);
    let binding = bound.swap_remove(index);
    let global = ActivationGlobal::of(op).map_err(Refused::Rule)?;
    let ProjectionOffer {
        offer,
        copy,
        dense,
        live_output,
    } = binding.handle;
    let mut prepared = ProjectionOp::with_rows(
        Arc::new(offer.handle().clone()),
        copy.map(Box::new),
        usage,
        need.capacity,
        global,
        dense,
        live_output,
    )?;
    // A model's dense form launches one entry, which the identity it
    // authenticates must name: a medium-M form's is not its reference's.
    if !prepared.default_form()
        && binding.identity.stages.last() != prepared.info.symbols.last().map(|(symbol, _)| symbol)
    {
        return Err(Refused::Rule(format!(
            "the identity's stages {:?} do not name the stage the op launches",
            binding.identity.stages
        )));
    }
    // The op claims no more than every offer of its launch declares.
    prepared.info.guarantees = guarantees;
    // One packed weight record holds every operand, in one layout.
    let mut layouts = binding.identity.weight_layouts.clone();
    layouts.dedup();
    prepared.info.weight_layouts = layouts
        .into_iter()
        .map(|layout| ("weights".to_owned(), layout))
        .collect();
    need.check(&prepared.info)?;
    Ok(Binding {
        identity: binding.identity,
        handle: Box::new(prepared),
    })
}

/// A projection through a supplied SM120 offer: an optional row gather,
/// activation quantizer stages, the dense, grouped or lookup compute (with a
/// grouped reduction), and an optional copy into the output layout.
pub struct ProjectionOp {
    operation: Arc<OperationHandle>,
    copy: Option<Box<LayoutCopy>>,
    /// The most rows a launch serves, at most the use's capacity.
    rows: u32,
    global: ActivationGlobal,
    dense: DenseForm,
    live_output: bool,
    info: OpInfo,
}

impl ProjectionOp {
    /// The op over `operation` at `usage`, whose activation global comes
    /// from `global`. Its info lists the scratch each stage needs at the
    /// use's capacity, and every stage with its dynamic shared memory.
    pub fn new(
        operation: Arc<OperationHandle>,
        copy: Option<Box<LayoutCopy>>,
        usage: OperationUse,
        global: ActivationGlobal,
    ) -> Result<Self, Refusal> {
        Self::with_form(operation, copy, usage, global, DenseForm::Tensor, false)
    }

    /// [`ProjectionOp::new`] whose dense compute launches in `dense` form,
    /// with an output of each launch's rows when `live_output`.
    pub fn with_form(
        operation: Arc<OperationHandle>,
        copy: Option<Box<LayoutCopy>>,
        usage: OperationUse,
        global: ActivationGlobal,
        dense: DenseForm,
        live_output: bool,
    ) -> Result<Self, Refusal> {
        let rows = usage.capacity;
        Self::with_rows(operation, copy, usage, rows, global, dense, live_output)
    }

    /// [`ProjectionOp::with_form`] serving launches of up to `rows` rows, at
    /// most the use's capacity. Stages, scratch and plans stay those of the
    /// use's capacity, so every launch is the one the use's op makes.
    pub fn with_rows(
        operation: Arc<OperationHandle>,
        copy: Option<Box<LayoutCopy>>,
        usage: OperationUse,
        rows: u32,
        global: ActivationGlobal,
        dense: DenseForm,
        live_output: bool,
    ) -> Result<Self, Refusal> {
        if rows == 0 || rows > usage.capacity {
            return Err(Refusal(
                "a projection serves from one row to its use's capacity",
            ));
        }
        if copy.is_some() && !matches!(operation.compute, Compute::Dense(_)) {
            return Err(Refusal("projection layout copies require dense compute"));
        }
        let default = matches!(dense, DenseForm::Tensor) && !live_output;
        if !default {
            let Compute::Dense(spec) = &operation.compute else {
                return Err(Refusal(
                    "a projection's dense form and live output need dense compute",
                ));
            };
            let StaticQuery::Gemm(query) = spec.key.query else {
                return Err(Refusal("projection dense query missing"));
            };
            if copy.is_some() || operation.gather.is_some() || operation.quantizer.is_some() {
                return Err(Refusal(
                    "a projection's dense form and live output take no row stages",
                ));
            }
            if !matches!(dense, DenseForm::Tensor) && query.activation != ActivationFormat::Bf16 {
                return Err(Refusal(
                    "the fragment and medium dense forms take BF16 activations",
                ));
            }
            if let DenseForm::Medium(medium) = &dense {
                if medium.reference != spec.symbol
                    || medium.query != query
                    || gemm::prefill_segments(&spec.key) != Ok(medium.segments)
                {
                    return Err(Refusal(
                        "a medium-M entry stands in only for its own fragment prefill \
                         reference, with its K segments",
                    ));
                }
                if rows > medium.max_rows() {
                    return Err(Refusal("a medium-M form serves at most its rows"));
                }
            }
        }
        let mut bytes = Bytes::default();
        let mut symbols = Vec::new();
        // Row stages (gather, quantizer, lookup and copy) take no dynamic
        // shared memory.
        if let Some(gather) = &operation.gather {
            symbols.push((gather.stage().entry().symbol.clone(), 0));
            // The typed row contract retains the gather width.
            let StaticQuery::Row {
                width_capacity: width,
                ..
            } = gather.stage().contract().key.query
            else {
                return Err(Refusal("projection gather query missing"));
            };
            let elements = u64::from(gather.stage().capacity()) * u64::from(width);
            if quantized_gather(gather) {
                bytes.codes = elements;
                bytes.scales = elements / 32;
            } else {
                bytes.gathered = elements * 2;
            }
        }
        if let Some(q) = &operation.quantizer {
            symbols.extend(
                q.stages()
                    .iter()
                    .map(|stage| (stage.entry().symbol.clone(), 0)),
            );
            bytes.codes = q.codes_bytes();
            bytes.scales = q.scales_bytes();
            bytes.quantizer_scratch = q.scratch_bytes();
        }
        let medium_symbol;
        let (symbol, plan) = match &operation.compute {
            Compute::Dense(spec) => {
                let StaticQuery::Gemm(query) = spec.key.query else {
                    return Err(Refusal("projection dense query missing"));
                };
                let columns = query.n / (1 + u32::from(query.epilogues.swiglu.is_some()));
                // A live output's size follows from each launch's rows.
                if !live_output {
                    bytes.output = u64::from(query.m_capacity)
                        * u64::from(columns)
                        * if query.output == DType::F32 { 4 } else { 2 };
                }
                bytes.semaphores = GemmBuffers::semaphore_capacity_bytes(&spec.key)?;
                if copy.is_some() {
                    if query.output != DType::Bf16 {
                        return Err(Refusal("projection layout copy requires BF16 output"));
                    }
                    bytes.temporary = u64::from(query.m_capacity) * u64::from(columns) * 2;
                }
                match &dense {
                    DenseForm::Medium(medium) => {
                        medium_symbol = medium.kernel.symbol();
                        (&medium_symbol, medium.plan(medium.max_rows()))
                    }
                    DenseForm::Tensor | DenseForm::Fragment => (
                        &spec.symbol,
                        gemm::plan(&spec.key, gemm::row_counts(&spec.key, query.m_capacity)),
                    ),
                }
            }
            Compute::Grouped(g) => {
                bytes.output = u64::from(g.query.m_capacity)
                    * u64::from(g.query.n / (1 + u32::from(g.query.epilogues.swiglu.is_some())))
                    * 2;
                (&g.symbol, g.plan(grouped_counts(g)))
            }
            Compute::Lookup(l) => {
                bytes.output = l.output_bytes();
                (&l.stage().entry().symbol, l.stage().plan(usage.capacity))
            }
        };
        let plan = plan?;
        check_plan(&plan)?;
        bytes.partial = plan.workspace_bytes;
        symbols.push((symbol.clone(), plan.dynamic_shared_bytes));
        if let Compute::Grouped(g) = &operation.compute
            && let Some(reduction) = g.reduction_symbol()
        {
            let plan = g
                .reduction_plan(grouped_counts(g))?
                .ok_or(Refusal("missing grouped reduction plan"))?;
            check_plan(&plan)?;
            symbols.push((reduction, plan.dynamic_shared_bytes));
        }
        if let Some(copy) = &copy {
            bytes.temporary = bytes.temporary.max(copy.source_bytes());
            bytes.output = copy.destination_bytes();
            symbols.push((copy.stage().entry().symbol.clone(), 0));
        }
        // The stages, then a digest of the complete typed selection, whose
        // keys and knobs hold every numerics choice (splits, segments,
        // schedules) whatever its entries are named.
        let stages: Vec<&str> = symbols.iter().map(|(symbol, _)| symbol.as_str()).collect();
        let selection = if default {
            kernel_build::digest(format!("{:?}", (&operation, &copy)).as_bytes())
        } else {
            kernel_build::digest(
                format!(
                    "{:?}",
                    (&operation, &copy, dense_launch(&dense), live_output)
                )
                .as_bytes(),
            )
        };
        let implementation = format!(
            "sm120.projection.v1[{}]#{}",
            stages.join(","),
            &selection[..16]
        );
        let mut info = OpInfo::new(implementation, Invariance::NONE, symbols);
        info.bytes = bytes.roles();
        Ok(Self {
            operation,
            copy,
            rows,
            global,
            dense,
            live_output,
            info,
        })
    }

    /// Whether the op launches as [`ProjectionOp::new`] does: the tensor
    /// form with an output of the entry's capacity.
    fn default_form(&self) -> bool {
        matches!(self.dense, DenseForm::Tensor) && !self.live_output
    }
}

/// The scratch and output bytes of a projection at its use's capacity.
#[derive(Default)]
struct Bytes {
    partial: u64,
    semaphores: u64,
    output: u64,
    temporary: u64,
    gathered: u64,
    codes: u64,
    scales: u64,
    /// Per-row amax and, after them, the tensor global of a dynamic NVFP4
    /// quantizer.
    quantizer_scratch: u64,
}

impl Bytes {
    fn roles(&self) -> Vec<(String, u64)> {
        let (row_amax, global) = match self.quantizer_scratch {
            0 => (0, 0),
            bytes => (bytes - 4, 4),
        };
        [
            ("scratch.partial", self.partial),
            ("scratch.semaphores", self.semaphores),
            ("output", self.output),
            ("scratch.temporary", self.temporary),
            ("scratch.gathered", self.gathered),
            ("scratch.codes", self.codes),
            ("scratch.scales", self.scales),
            ("scratch.row_amax", row_amax),
            ("scratch.activation_global", global),
        ]
        .into_iter()
        .filter(|&(_, bytes)| bytes > 0)
        .map(|(role, bytes)| (role.to_owned(), bytes))
        .collect()
    }
}

/// A grouped operation's counts at its full capacity.
fn grouped_counts(g: &GroupedOperation) -> RuntimeCounts {
    RuntimeCounts {
        m: g.tokens_capacity,
        batch: 1,
        context: 1,
        routed_rows: g.query.m_capacity,
        split_k: g.knobs.split_k_capacity,
    }
}

fn quantized_gather(gather: &Gather) -> bool {
    gather.stage().numerical_contract() == "gather.mxfp8.group32.ue8m0.v1"
}

/// A grouped expert entry's scale operands in the slots its kernels read
/// (`run` in `cuda/moe/moe.cuh`): byte block scales of the MX and NV formats
/// at 1 (activation) and 3 (weight), F32 scales at 2 and 4 (NVFP4 global
/// scales, FP8 block scales). Slots a format does not read hold `unused`.
pub(super) struct GroupedScales {
    pub activation_bytes: View,
    pub activation_f32: View,
    pub weight_bytes: View,
    pub weight_f32: View,
}

impl GroupedScales {
    /// `activation_scales` and `weight_scales` are the format's block
    /// scales; the globals are NVFP4's per-tensor (or per-expert) scales.
    pub(super) fn of(
        query: &kernel_core::contract::GemmQuery,
        activation_scales: Option<View>,
        activation_global: Option<View>,
        weight_scales: Option<View>,
        weight_global: Option<View>,
        unused: View,
    ) -> Result<GroupedScales, Refusal> {
        let (activation_bytes, activation_f32) = match query.activation {
            ActivationFormat::F32 => {
                return Err(Refusal(
                    "legacy grouped projections do not support F32 activations",
                ));
            }
            ActivationFormat::Bf16 => (unused, unused),
            ActivationFormat::Mxfp8 => (required(activation_scales)?, unused),
            ActivationFormat::Nvfp4 => (required(activation_scales)?, required(activation_global)?),
            ActivationFormat::Fp8Group128(_) => (unused, required(activation_scales)?),
        };
        let (weight_bytes, weight_f32) = match query.weight {
            WeightFormat::F32 => {
                return Err(Refusal(
                    "legacy grouped projections do not support F32 weights",
                ));
            }
            WeightFormat::Fp8Block32x32 => {
                return Err(Refusal(
                    "32x32 UE8M0 weights require a lossless per-row repack before grouped execution",
                ));
            }
            WeightFormat::Bf16 => (unused, unused),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => (required(weight_scales)?, unused),
            WeightFormat::Nvfp4 => (required(weight_scales)?, required(weight_global)?),
            WeightFormat::Fp8PerTensor
            | WeightFormat::Fp8PerChannel
            | WeightFormat::Fp8Block128x128(_)
            | WeightFormat::Fp8Rows64x128(_) => (unused, required(weight_scales)?),
        };
        Ok(GroupedScales {
            activation_bytes,
            activation_f32,
            weight_bytes,
            weight_f32,
        })
    }
}

fn required(view: Option<View>) -> Result<View, Refusal> {
    view.ok_or(Refusal("missing projection operand or scratch region"))
}

fn covers(view: View, bytes: u64) -> Result<(), Refusal> {
    if view.bytes < bytes {
        Err(Refusal(
            "projection region does not cover selected capacity",
        ))
    } else {
        Ok(())
    }
}

fn launch(symbol: &str, rows: u32, plan: LaunchPlan, arguments: ArgumentList) -> Launch {
    Launch {
        symbol: symbol.into(),
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    }
}

/// Whether a compute launch is persistent: fragment prefill GEMMs and
/// prefill grouped experts fill every SM with one block each.
fn persistent(compute: &Compute, plan: &LaunchPlan) -> Result<bool, Refusal> {
    let kind = match compute {
        Compute::Dense(spec) => matches!(
            gemm::kernel(&spec.key)?,
            kernel_core::contract::DenseKernel::FragmentPrefill { .. }
        ),
        Compute::Grouped(g) => g.phase == crate::moe::Phase::Prefill,
        Compute::Lookup(_) => false,
    };
    Ok(kind && plan.grid == [gemm::SM_COUNT, 1, 1])
}

impl PreparedOp<ProjectionBuffers<View>> for ProjectionOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &ProjectionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.rows {
            return Err(Refusal("projection phase does not match bound use"));
        }
        let operation = &self.operation;
        let scratch = &buffers.scratch;
        let bytes = |role: &str| self.info.bytes_of(role).unwrap_or(0);
        let mut launches = Vec::new();
        let mut activation = buffers.activation;
        let activation_global = match self.global {
            ActivationGlobal::Absent => None,
            ActivationGlobal::Supplied => buffers.activation_global,
            ActivationGlobal::TensorAmax => scratch.activation_global,
        };
        let row_amax = if self.global == ActivationGlobal::TensorAmax {
            scratch.row_amax
        } else {
            None
        };
        let grouped = matches!(operation.compute, Compute::Grouped(_));
        let routed_live = if grouped {
            required(buffers.routing.map(|r| r.routes.padded_rows))?
        } else {
            buffers.live_rows
        };
        let quant_rows = if grouped { counts.padded_rows } else { rows };
        if let Some(gather) = &operation.gather {
            let route = buffers
                .routing
                .ok_or(Refusal("projection gather needs routing"))?;
            let gathered = if quantized_gather(gather) {
                let codes = required(scratch.codes)?;
                covers(codes, bytes("scratch.codes"))?;
                covers(required(scratch.scales)?, bytes("scratch.scales"))?;
                codes
            } else {
                let gathered = required(scratch.gathered)?;
                covers(gathered, bytes("scratch.gathered"))?;
                gathered
            };
            let args = gather.arguments(
                activation.handle,
                route.routes.sorted_rows.handle,
                gathered.handle,
                if quantized_gather(gather) {
                    Some(required(scratch.scales)?.handle)
                } else {
                    None
                },
                route.routes.padded_rows.handle,
            )?;
            launches.push(launch(
                &gather.stage().entry().symbol,
                rows,
                gather.stage().plan(counts.padded_rows)?,
                args,
            ));
            activation = gathered;
        }
        if let Some(q) = &operation.quantizer {
            let codes = required(scratch.codes)?;
            let scales = required(scratch.scales)?;
            covers(codes, q.codes_bytes())?;
            covers(scales, q.scales_bytes())?;
            if q.scratch_bytes() > 0 {
                covers(required(row_amax)?, q.scratch_bytes() - 4)?;
                covers(required(activation_global)?, 4)?;
            }
            let args = q.arguments(QuantizerBuffers {
                input: activation.handle,
                codes: codes.handle,
                scales: scales.handle,
                live_rows: routed_live.handle,
                global_scale: activation_global.map(|v| v.handle),
                row_amax: row_amax.map(|v| v.handle),
            })?;
            for (stage, args) in q.stages().iter().zip(args) {
                launches.push(launch(
                    &stage.entry().symbol,
                    rows,
                    stage.plan(quant_rows)?,
                    args,
                ));
            }
            activation = codes;
        }
        let destination = if self.copy.is_some() {
            let view = required(scratch.temporary)?;
            covers(view, bytes("scratch.temporary"))?;
            view
        } else {
            buffers.output
        };
        let weights = &buffers.weights;
        match &operation.compute {
            Compute::Dense(spec) => {
                let StaticQuery::Gemm(query) = spec.key.query else {
                    return Err(Refusal("projection dense query missing"));
                };
                let gemm_counts = gemm::row_counts(&spec.key, rows);
                let width_bytes = u64::from(query.m_capacity)
                    * u64::from(query.k)
                    * if query.activation == ActivationFormat::Bf16 {
                        2
                    } else {
                        1
                    }
                    / if query.activation == ActivationFormat::Nvfp4 {
                        2
                    } else {
                        1
                    };
                covers(activation, width_bytes)?;
                let element_bytes = if query.output == DType::F32 { 4 } else { 2 };
                let output_rows = if self.live_output {
                    rows
                } else {
                    query.m_capacity
                };
                covers(
                    destination,
                    u64::from(output_rows)
                        * u64::from(query.n / (1 + u32::from(query.epilogues.swiglu.is_some())))
                        * element_bytes,
                )?;
                let maps = match self.dense {
                    DenseForm::Tensor => {
                        gemm::tensor_maps(&spec.key, weights.codes.handle, activation.handle)?
                    }
                    DenseForm::Fragment | DenseForm::Medium(_) => gemm::fragment_maps(
                        &spec.key,
                        &query,
                        activation.handle,
                        weights.codes.handle,
                        query.m_capacity <= 32,
                    ),
                };
                let gemm_buffers = GemmBuffers {
                    output: destination.handle,
                    partial: scratch.partial.handle,
                    semaphores: scratch.semaphores.handle,
                    weight_scales: if query.weight == WeightFormat::Bf16 {
                        None
                    } else {
                        Some(required(weights.local_scales)?.handle)
                    },
                    activation_scales: if query.activation == ActivationFormat::Bf16 {
                        None
                    } else {
                        Some(required(scratch.scales)?.handle)
                    },
                    residual: buffers.residual.map(|v| v.handle),
                    weight_global_scale: weights.global_scales.map(|v| v.handle),
                    activation_global_scale: if query.activation == ActivationFormat::Nvfp4 {
                        activation_global.map(|v| v.handle)
                    } else {
                        None
                    },
                };
                if let DenseForm::Medium(medium) = &self.dense {
                    // The reference's activation map with the medium tile's box.
                    let [activation_map, weight_map] = maps;
                    let mut medium_map = activation_map;
                    medium_map.box_dims[1] = medium.activation_box_rows();
                    let args = medium.arguments_from(
                        gemm_counts.m,
                        medium_map,
                        weight_map.base_buffer,
                        gemm_buffers,
                    )?;
                    let plan = medium.plan(gemm_counts.m)?;
                    let mut compute = launch(&medium.kernel.symbol(), rows, plan, args);
                    // One CTA per SM for as long as the GEMM runs.
                    compute.persistent = true;
                    launches.push(compute);
                } else {
                    let args = gemm::arguments(&spec.key, gemm_counts, maps, gemm_buffers)?;
                    let plan = gemm::plan(&spec.key, gemm_counts)?;
                    let mut compute = launch(&spec.symbol, rows, plan, args);
                    compute.persistent = persistent(&operation.compute, &plan)?;
                    launches.push(compute);
                }
            }
            Compute::Grouped(g) => {
                let activation_rows = if g.query.activation == ActivationFormat::Bf16 {
                    if g.query.epilogues.swiglu.is_some() {
                        g.tokens_capacity
                    } else {
                        g.query.m_capacity
                    }
                } else if let Some(gather) = &operation.gather {
                    gather.stage().capacity()
                } else if let Some(quantizer) = &operation.quantizer {
                    quantizer.stages()[0].capacity()
                } else {
                    g.query.m_capacity
                };
                let activation_bytes = u64::from(activation_rows)
                    * u64::from(g.query.k)
                    * if g.query.activation == ActivationFormat::Bf16 {
                        2
                    } else {
                        1
                    }
                    / if g.query.activation == ActivationFormat::Nvfp4 {
                        2
                    } else {
                        1
                    };
                covers(activation, activation_bytes)?;
                let route = buffers
                    .routing
                    .ok_or(Refusal("projection grouped compute needs routing"))?;
                let scales = GroupedScales::of(
                    &g.query,
                    scratch.scales,
                    activation_global,
                    weights.local_scales,
                    weights.global_scales,
                    scratch.unused,
                )?;
                let (output_codes, output_scales) =
                    if g.query.epilogues.output_quantization.is_some() {
                        (
                            required(scratch.output_codes)?,
                            required(scratch.output_scales)?,
                        )
                    } else {
                        (scratch.unused, scratch.unused)
                    };
                let views = [
                    activation,
                    scales.activation_bytes,
                    scales.activation_f32,
                    scales.weight_bytes,
                    scales.weight_f32,
                    destination,
                    route.weights,
                    route.routes.offsets,
                    route.routes.sorted_rows,
                    route.routes.padded_rows,
                    buffers.live_rows,
                    required(route.routes.active_experts)?,
                    output_codes,
                    output_scales,
                    if bytes("scratch.partial") > 0 {
                        scratch.partial
                    } else {
                        scratch.unused
                    },
                ];
                let handles = views.map(|v| v.handle);
                let args = g.arguments(handles, g.weight_tensor_map(weights.codes.handle)?)?;
                let top_k = g
                    .query
                    .grouped
                    .ok_or(Refusal("projection grouped query missing"))?
                    .top_k;
                let grouped_counts = RuntimeCounts {
                    m: rows,
                    batch: 1,
                    context: 1,
                    routed_rows: rows
                        .checked_mul(top_k)
                        .ok_or(Refusal("projection routed rows overflow"))?,
                    split_k: g.knobs.split_k_capacity,
                };
                let plan = g.plan(grouped_counts)?;
                let reduction = g.reduction_plan(grouped_counts)?;
                let mut compute = launch(&g.symbol, rows, plan, args);
                compute.persistent = persistent(&operation.compute, &plan)?;
                launches.push(compute);
                if let Some(plan) = reduction {
                    let symbol = g
                        .reduction_symbol()
                        .ok_or(Refusal("missing grouped reduction entry"))?;
                    let args = g
                        .reduction_arguments(handles)?
                        .ok_or(Refusal("missing grouped reduction arguments"))?;
                    launches.push(launch(&symbol, rows, plan, args));
                }
            }
            Compute::Lookup(l) => {
                covers(activation, l.input_bytes())?;
                let args = l.arguments(LookupBuffers {
                    table: weights.codes.handle,
                    ids: activation.handle,
                    local_scales: weights.local_scales.map(|v| v.handle),
                    global_scale: weights.global_scales.map(|v| v.handle),
                    output: destination.handle,
                    live_rows: buffers.live_rows.handle,
                    row_offset: buffers.row_offset,
                    shard_rows: buffers.shard_rows,
                })?;
                launches.push(launch(
                    &l.stage().entry().symbol,
                    rows,
                    l.stage().plan(rows)?,
                    args,
                ));
            }
        }
        if let Some(copy) = &self.copy {
            let args = copy.arguments(
                destination.handle,
                buffers.output.handle,
                buffers.live_rows.handle,
            )?;
            launches.push(launch(
                &copy.stage().entry().symbol,
                rows,
                copy.stage().plan(rows)?,
                args,
            ));
        }
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::view;

    #[test]
    fn grouped_scales_take_the_slots_the_expert_kernels_read() {
        // `run`'s operands 1 to 4, in the order the grouped arguments pass them.
        let source = include_str!("../../cuda/moe/moe.cuh");
        let signature = source.split("__device__ void run(").nth(1).unwrap();
        let order = [
            "const uint8_t* activation,",
            "const uint8_t* activation_scales,",
            "const float* activation_f32_scales,",
            "const uint8_t* weight_scales,",
            "const float* weight_f32_scales,",
        ];
        let at = order.map(|operand| signature.find(operand).unwrap());
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]));

        let query = |weight, activation| kernel_core::contract::GemmQuery {
            weight,
            activation,
            ..gemm::bf16_query(64, 256, 256, DType::Bf16)
        };
        let (activation_scales, activation_global) = (view(1, 64), view(2, 4));
        let (weight_scales, weight_global, unused) = (view(3, 64), view(4, 4), view(5, 4));
        let slots = |weight, activation| {
            let s = GroupedScales::of(
                &query(weight, activation),
                Some(activation_scales),
                Some(activation_global),
                Some(weight_scales),
                Some(weight_global),
                unused,
            )
            .unwrap();
            [
                s.activation_bytes,
                s.activation_f32,
                s.weight_bytes,
                s.weight_f32,
            ]
            .map(|v| v.handle)
        };
        let fp8 = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
        let a8 = ActivationFormat::Fp8Group128(kernel_core::contract::ScaleMode::ExactF32);
        // FP8 block scales are F32: the activation's at 2, the weight's at 4.
        assert_eq!(slots(fp8, ActivationFormat::Bf16), [5, 5, 5, 3]);
        assert_eq!(slots(fp8, a8), [5, 1, 5, 3]);
        assert_eq!(
            slots(WeightFormat::Nvfp4, ActivationFormat::Nvfp4),
            [1, 2, 3, 4]
        );
        assert_eq!(
            slots(WeightFormat::Mxfp8, ActivationFormat::Mxfp8),
            [1, 5, 3, 5]
        );
        assert_eq!(
            slots(WeightFormat::Mxfp4, ActivationFormat::Bf16),
            [5, 5, 3, 5]
        );
        assert_eq!(
            slots(WeightFormat::Bf16, ActivationFormat::Bf16),
            [5, 5, 5, 5]
        );
        let missing = GroupedScales::of(
            &query(fp8, a8),
            None,
            None,
            Some(weight_scales),
            None,
            unused,
        );
        assert!(missing.is_err());
    }

    fn dense_op(kernel: kernel_core::contract::DenseKernel, symbol: &str) -> ProjectionOp {
        let spec = gemm::instantiate(gemm::DenseRequest {
            query: gemm::bf16_query(8192, 4096, 4096, DType::Bf16),
            kernel,
            target: kernel_core::contract::Target::Sm120f,
            symbol: symbol.into(),
            knobs: None,
        })
        .unwrap();
        let operation = OperationHandle {
            compute: Compute::Dense(spec),
            gather: None,
            quantizer: None,
        };
        let usage = OperationUse {
            phase: kernel_core::precision::ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        ProjectionOp::new(Arc::new(operation), None, usage, ActivationGlobal::Absent).unwrap()
    }

    #[test]
    fn projection_identity_names_its_entries_and_persistent_launches_are_marked() {
        use kernel_core::contract::DenseKernel;
        use kernel_core::precision::buffers::{ProjectionScratch, QuantizedWeights};
        let fragment = dense_op(
            DenseKernel::FragmentPrefill { segments: 4 },
            "chop_gemm_test_fragment_v1",
        );
        let tiled = dense_op(DenseKernel::Tiled { splits: 2 }, "chop_gemm_test_tiled_v1");
        assert!(
            fragment
                .info()
                .implementation
                .starts_with("sm120.projection.v1[chop_gemm_test_fragment_v1]#")
        );
        assert_ne!(fragment.info().implementation, tiled.info().implementation);
        // Entries of one name with different schedules are different
        // implementations.
        let split = |splits| dense_op(DenseKernel::Tiled { splits }, "chop_gemm_test_same_v1");
        assert_ne!(
            split(2).info().implementation,
            split(4).info().implementation
        );
        assert_eq!(
            split(2).info().implementation,
            split(2).info().implementation
        );
        let any = view(1, u64::MAX);
        let buffers = ProjectionBuffers {
            activation: any,
            live_rows: any,
            output: any,
            residual: None,
            weights: QuantizedWeights {
                codes: any,
                local_scales: None,
                global_scales: None,
            },
            activation_global: None,
            routing: None,
            scratch: ProjectionScratch {
                partial: any,
                semaphores: any,
                unused: any,
                temporary: None,
                gathered: None,
                codes: None,
                scales: None,
                activation_global: None,
                row_amax: None,
                output_codes: None,
                output_scales: None,
            },
            row_offset: 0,
            shard_rows: 0,
        };
        let launches = fragment
            .launches(&buffers, View::EMPTY, Counts::rows(8192))
            .unwrap();
        assert_eq!(launches[0].plan.grid, [gemm::SM_COUNT, 1, 1]);
        assert!(launches[0].persistent);
        let launches = tiled
            .launches(&buffers, View::EMPTY, Counts::rows(8192))
            .unwrap();
        assert!(!launches[0].persistent);
    }

    #[test]
    fn listed_offers_prepare_alike_in_every_order() {
        use crate::backend::Sm120;
        use crate::backend::select::orders;
        use kernel_core::backend::Backend;
        use kernel_core::contract::{DenseKernel, Target};
        use kernel_core::precision::ExecutionPhase;

        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let offer = |splits, symbol: &str| ProjectionOffer {
            offer: KernelOffer::dense(
                gemm::DenseRequest {
                    query: gemm::bf16_query(8192, 4096, 4096, DType::Bf16),
                    kernel: DenseKernel::Tiled { splits },
                    target: Target::Sm120f,
                    symbol: symbol.into(),
                    knobs: None,
                },
                usage,
                crate::row::precision::Nvfp4Global::Absent,
                "unused",
            )
            .unwrap(),
            copy: None,
            dense: DenseForm::Tensor,
            live_output: false,
        };
        let two = offer(2, "chop_gemm_test_split2_v1");
        let four = offer(4, "chop_gemm_test_split4_v1");
        let op =
            crate::precision::generated::comparison::operation(&two.offer.capability().unwrap())
                .unwrap();
        let prepare = |order: &[ProjectionOffer]| {
            let kernels = order.iter().map(|o| o.offer.clone()).collect::<Vec<_>>();
            let available = capabilities(&kernels).unwrap().entries();
            let list = OfferList::new(order.to_vec(), available, "a".repeat(64)).unwrap();
            Sm120::new()
                .with_projections(Box::new(list))
                .projection(&op, usage, Need::rows(8192))
                .map(|bound| bound.handle.info().clone())
        };
        let in_every_order = |registered: &[ProjectionOffer]| {
            let results = orders(registered)
                .iter()
                .map(|o| prepare(o))
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        // An offer listed twice is one; two splits of one operation tie.
        let alone = prepare(std::slice::from_ref(&two)).unwrap();
        assert_eq!(alone.symbols[0].0, "chop_gemm_test_split2_v1");
        assert_eq!(in_every_order(&[two.clone(), two.clone()]), Ok(alone));
        assert!(matches!(
            in_every_order(&[two, four]),
            Err(Refused::Rule(rule))
                if rule.contains("chop_gemm_test_split2_v1 and chop_gemm_test_split4_v1")
        ));
    }

    /// The projection key holds what a bound offer launches and returns,
    /// every field changed alone: its operation, copy, dense form, output
    /// bound and identity; never its claims. Its use and capability shapes decide what it binds.
    #[test]
    fn launch_keys_hold_what_bound_offers_launch() {
        use crate::backend::Sm120;
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        use crate::row::precision::{CopyRows, Nvfp4Global, RowResources, ShapeBinding};
        use kernel_core::backend::Backend;
        use kernel_core::contract::{DenseKernel, Target};
        use kernel_core::precision::ExecutionPhase;
        use kernel_core::precision::buffers::{ProjectionScratch, QuantizedWeights};

        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let offer = move |splits, symbol: &str| ProjectionOffer {
            offer: KernelOffer::dense(
                gemm::DenseRequest {
                    query: gemm::bf16_query(8192, 4096, 4096, DType::Bf16),
                    kernel: DenseKernel::Tiled { splits },
                    target: Target::Sm120f,
                    symbol: symbol.into(),
                    knobs: None,
                },
                usage,
                Nvfp4Global::Absent,
                "unused",
            )
            .unwrap(),
            copy: None,
            dense: DenseForm::Tensor,
            live_output: false,
        };
        let base = offer(2, "chop_gemm_test_split2_v1");
        let op =
            crate::precision::generated::comparison::operation(&base.offer.capability().unwrap())
                .unwrap();
        let list = |o: &ProjectionOffer| {
            let available = capabilities(std::slice::from_ref(&o.offer)).ok()?.entries();
            OfferList::new(vec![o.clone()], available, "a".repeat(64)).ok()
        };
        let mut names = field_names!(ProjectionOffer {
            offer,
            copy,
            dense,
            live_output
        });
        names.retain(|name| *name != "offer");
        names.extend(field_names!(KernelOffer {
            usage,
            handle,
            nv_global,
            lookup,
            invariance
        }));
        let rows = CopyRows {
            row_stride: 4096,
            column_stride: 1,
            column_offset: 0,
        };
        let copy = LayoutCopy::new(
            4096,
            8192,
            rows,
            rows,
            "chop_row_copy_test_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F,
        )
        .unwrap();
        let split4 = offer(4, "chop_gemm_test_split2_v1").offer.handle;
        let fields: Vec<Field<ProjectionOffer>> = vec![
            (
                "usage",
                Class::Selects,
                Box::new(move |o| o.offer.usage.phase = ExecutionPhase::Decode),
            ),
            (
                "handle",
                Class::Launch,
                Box::new(move |o| o.offer.handle = split4.clone()),
            ),
            (
                "nv_global",
                Class::Inert,
                Box::new(|o| o.offer.nv_global = Nvfp4Global::Supplied),
            ),
            (
                "lookup",
                Class::Inert,
                Box::new(|o| {
                    o.offer.lookup = Some(crate::row::precision::LookupRequest {
                        vocab: 1024,
                        width: 4096,
                        heads: 1,
                        capacity: 8192,
                        weight: kernel_core::contract::WeightFormat::Bf16,
                        global_scale: None,
                        ids: crate::row::precision::LookupIds::I32,
                        shapes: ShapeBinding::Compiled,
                    })
                }),
            ),
            (
                "invariance",
                Class::Inert,
                Box::new(|o| o.offer.invariance = Invariance::PREFILL),
            ),
            (
                "copy",
                Class::Launch,
                Box::new(move |o| o.copy = Some(copy.clone())),
            ),
            (
                "dense",
                Class::Launch,
                Box::new(|o| o.dense = DenseForm::Fragment),
            ),
            (
                "live_output",
                Class::Bound,
                Box::new(|o| o.live_output = true),
            ),
        ];
        let any = view(1, u64::MAX);
        let buffers = ProjectionBuffers {
            activation: any,
            live_rows: any,
            output: any,
            residual: None,
            weights: QuantizedWeights {
                codes: any,
                local_scales: None,
                global_scales: None,
            },
            activation_global: None,
            routing: None,
            scratch: ProjectionScratch {
                partial: any,
                semaphores: any,
                unused: any,
                temporary: Some(view(9, u64::MAX)),
                gathered: None,
                codes: None,
                scales: None,
                activation_global: None,
                row_amax: None,
                output_codes: None,
                output_scales: None,
            },
            row_offset: 0,
            shard_rows: 0,
        };
        keys_follow_launches(
            &base,
            names,
            fields,
            |o| {
                list(o).map_or_else(String::new, |list| {
                    bound_launch(&list.bind_all(&op, usage).unwrap()[0])
                })
            },
            |o| {
                let list = list(o).ok_or(Refused::Unsupported)?;
                let bound = Sm120::new().with_projections(Box::new(list)).projection(
                    &op,
                    usage,
                    Need::rows(8192),
                )?;
                let launches = [1, 8192].map(|rows| {
                    bound
                        .handle
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", bound.identity), launches))
            },
        );
    }

    /// R1 finding 1: a provider registers only through an adapter; the
    /// named one passes its one binding through, and an op it binds
    /// prepares as the same offer listed alone.
    #[test]
    fn named_offers_bind_the_providers_one_offer() {
        use crate::backend::Sm120;
        use kernel_core::backend::Backend;
        use kernel_core::contract::{DenseKernel, Target};
        use kernel_core::precision::ExecutionPhase;
        use kernel_core::precision::execution::ImplementationIdentity;

        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let listed = ProjectionOffer {
            offer: KernelOffer::dense(
                gemm::DenseRequest {
                    query: gemm::bf16_query(8192, 4096, 4096, DType::Bf16),
                    kernel: DenseKernel::Tiled { splits: 2 },
                    target: Target::Sm120f,
                    symbol: "chop_gemm_test_named_v1".into(),
                    knobs: None,
                },
                usage,
                crate::row::precision::Nvfp4Global::Absent,
                "unused",
            )
            .unwrap(),
            copy: None,
            dense: DenseForm::Tensor,
            live_output: false,
        };
        let op =
            crate::precision::generated::comparison::operation(&listed.offer.capability().unwrap())
                .unwrap();
        let kernels = [listed.offer.clone()];
        let available = capabilities(&kernels).unwrap().entries();
        let list = OfferList::new(vec![listed.clone()], available, "a".repeat(64)).unwrap();
        let bound = list.bind_all(&op, usage).unwrap();
        assert_eq!(bound.len(), 1);
        let identity = bound[0].identity.clone();

        struct Named(ImplementationIdentity, ProjectionOffer);
        impl ImplementationProvider for Named {
            type Handle = ProjectionOffer;
            fn bind(
                &self,
                _: &OperationSpec,
                _: OperationUse,
            ) -> Result<Binding<ProjectionOffer>, String> {
                Ok(Binding {
                    identity: self.0.clone(),
                    handle: self.1.clone(),
                })
            }
        }
        let prepare = |offers: Box<dyn ProjectionOffers>| {
            let prepared = Sm120::new()
                .with_projections(offers)
                .projection(&op, usage, Need::rows(8192))
                .unwrap();
            (prepared.identity, prepared.handle.info().clone())
        };
        let named = prepare(Box::new(NamedOffers(Named(identity, listed))));
        assert_eq!(named, prepare(Box::new(list)));
    }

    /// A fragment prefill entry of `capacity` rows (decode below 33), its
    /// op in `dense` form with `live_output`, and the buffers it launches
    /// over: role handles 1 to 9.
    fn fragment(
        capacity: u32,
        dense: impl FnOnce(&kernel_core::contract::InstantiationSpec) -> DenseForm,
        live_output: bool,
    ) -> (
        kernel_core::contract::InstantiationSpec,
        Result<ProjectionOp, Refusal>,
    ) {
        let (spec, usage) = fragment_entry(capacity);
        let form = dense(&spec);
        let op = ProjectionOp::with_form(
            Arc::new(OperationHandle {
                compute: Compute::Dense(spec.clone()),
                gather: None,
                quantizer: None,
            }),
            None,
            usage,
            ActivationGlobal::Absent,
            form,
            live_output,
        );
        (spec, op)
    }

    /// A fragment prefill entry of `capacity` rows (decode below 33) and
    /// the use of that capacity it serves.
    fn fragment_entry(capacity: u32) -> (kernel_core::contract::InstantiationSpec, OperationUse) {
        use kernel_core::contract::DenseKernel;
        let kernel = if capacity > 32 {
            DenseKernel::FragmentPrefill { segments: 4 }
        } else {
            DenseKernel::FragmentDecode { splits: 1 }
        };
        let spec = gemm::instantiate(gemm::DenseRequest {
            query: gemm::bf16_query(capacity, 4096, 4096, DType::Bf16),
            kernel,
            target: kernel_core::contract::Target::Sm120f,
            symbol: "chop_gemm_test_fragment_v1".into(),
            knobs: None,
        })
        .unwrap();
        let usage = OperationUse {
            phase: if capacity > 32 {
                kernel_core::precision::ExecutionPhase::Prefill
            } else {
                kernel_core::precision::ExecutionPhase::Decode
            },
            capacity,
            graph: capacity <= 32,
        };
        (spec, usage)
    }

    /// A verify use of `capacity` rows of the 8192-row fragment prefill
    /// entry, as a model runs its graph buckets above the decode rows.
    fn verify(capacity: u32) -> OperationUse {
        OperationUse {
            phase: kernel_core::precision::ExecutionPhase::Verify,
            capacity,
            graph: true,
        }
    }

    /// The medium-M entry that stands in for fragment prefill entry `spec`.
    fn medium_entry(spec: &kernel_core::contract::InstantiationSpec) -> gemm::mid::Entry {
        let StaticQuery::Gemm(query) = spec.key.query else {
            unreachable!()
        };
        gemm::mid::Entry {
            profile: crate::Profile::StdDenseMoe,
            reference: spec.symbol.clone(),
            query,
            kernel: gemm::mid::Kernel {
                fp8: false,
                rows64: false,
                swiglu: false,
            },
            segments: 4,
            rows: 33..=64,
            invariance: Invariance::NONE,
        }
    }

    /// The op of the 8192-row fragment prefill entry in the medium-M form
    /// `change` makes of its entry, at `usage`.
    fn medium(
        usage: OperationUse,
        change: impl FnOnce(&mut gemm::mid::Entry),
    ) -> (
        kernel_core::contract::InstantiationSpec,
        Result<ProjectionOp, Refusal>,
    ) {
        let (spec, _) = fragment_entry(8192);
        let mut entry = medium_entry(&spec);
        change(&mut entry);
        let op = ProjectionOp::with_form(
            Arc::new(OperationHandle {
                compute: Compute::Dense(spec.clone()),
                gather: None,
                quantizer: None,
            }),
            None,
            usage,
            ActivationGlobal::Absent,
            DenseForm::Medium(Box::new(entry)),
            false,
        );
        (spec, op)
    }

    /// Binds every operation to the one offer the identity names, a named
    /// composition.
    struct Named(
        kernel_core::precision::execution::ImplementationIdentity,
        ProjectionOffer,
    );
    impl ImplementationProvider for Named {
        type Handle = ProjectionOffer;
        fn bind(
            &self,
            _: &OperationSpec,
            _: OperationUse,
        ) -> Result<Binding<ProjectionOffer>, String> {
            Ok(Binding {
                identity: self.0.clone(),
                handle: self.1.clone(),
            })
        }
    }

    /// The 8192-row fragment prefill entry's offer at verify use `usage` in
    /// `dense` form, the operation it binds, and the identity its offer
    /// list authenticates for it with `stages`.
    fn medium_offer(
        usage: OperationUse,
        dense: impl FnOnce(&kernel_core::contract::InstantiationSpec) -> DenseForm,
        stages: impl FnOnce(&kernel_core::contract::InstantiationSpec) -> Vec<String>,
    ) -> (
        ProjectionOffer,
        OperationSpec,
        kernel_core::precision::execution::ImplementationIdentity,
    ) {
        let (spec, _) = fragment_entry(8192);
        let reference = ProjectionOffer {
            offer: KernelOffer::catalog_dense(spec.clone(), usage).unwrap(),
            copy: None,
            dense: DenseForm::Fragment,
            live_output: false,
        };
        let op = crate::precision::generated::comparison::operation(
            &reference.offer.capability().unwrap(),
        )
        .unwrap();
        let kernels = [reference.offer.clone()];
        let available = capabilities(&kernels).unwrap().entries();
        let list = OfferList::new(vec![reference.clone()], available, "a".repeat(64)).unwrap();
        let mut identity = list.bind_all(&op, usage).unwrap().remove(0).identity;
        identity.stages = stages(&spec);
        let offer = ProjectionOffer {
            dense: dense(&spec),
            ..reference
        };
        (offer, op, identity)
    }

    fn fragment_buffers(output: u64) -> ProjectionBuffers<View> {
        use kernel_core::precision::buffers::{ProjectionScratch, QuantizedWeights};
        let any = |handle| view(handle, u64::MAX);
        ProjectionBuffers {
            activation: any(1),
            live_rows: any(2),
            output: view(3, output),
            residual: None,
            weights: QuantizedWeights {
                codes: any(4),
                local_scales: None,
                global_scales: None,
            },
            activation_global: None,
            routing: None,
            scratch: ProjectionScratch {
                partial: any(5),
                semaphores: any(6),
                unused: any(7),
                temporary: None,
                gathered: None,
                codes: None,
                scales: None,
                activation_global: None,
                row_amax: None,
                output_codes: None,
                output_scales: None,
            },
            row_offset: 0,
            shard_rows: 0,
        }
    }

    fn gemm_buffers() -> GemmBuffers {
        GemmBuffers {
            output: 3,
            partial: 5,
            semaphores: 6,
            weight_scales: None,
            activation_scales: None,
            residual: None,
            weight_global_scale: None,
            activation_global_scale: None,
        }
    }

    /// A fragment entry in fragment form launches over the fragment maps,
    /// as a model that plans its fragment entries so launches them; the
    /// tensor form keeps the tensor maps. A decode entry reads no map, so
    /// both forms launch it alike.
    #[test]
    fn fragment_forms_launch_over_the_fragment_maps() {
        for capacity in [8, 8192] {
            for rows in [1, capacity / 2, capacity] {
                let (spec, op) = fragment(capacity, |_| DenseForm::Fragment, false);
                let op = op.unwrap();
                let StaticQuery::Gemm(query) = spec.key.query else {
                    unreachable!()
                };
                let launches = op
                    .launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let counts = gemm::row_counts(&spec.key, rows);
                let maps = gemm::fragment_maps(&spec.key, &query, 1, 4, capacity <= 32);
                let plan = gemm::plan(&spec.key, counts).unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(launches[0].symbol, spec.symbol);
                assert_eq!(launches[0].plan, plan);
                assert_eq!(
                    launches[0].arguments,
                    gemm::arguments(&spec.key, counts, maps, gemm_buffers()).unwrap()
                );
                let (_, tensor) = fragment(capacity, |_| DenseForm::Tensor, false);
                let tensor = tensor
                    .unwrap()
                    .launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                    .unwrap();
                // The decode entry takes its weights by pointer, not by map.
                assert_eq!(tensor[0].arguments == launches[0].arguments, capacity <= 32);
                assert_eq!(tensor[0].plan, launches[0].plan);
            }
        }
    }

    /// A medium-M form launches its medium entry in place of the fragment
    /// prefill entry: the reference's fragment activation map with the
    /// medium box, the medium plan, one persistent CTA per SM. It stands in
    /// only for its own reference.
    #[test]
    fn medium_forms_launch_the_medium_entry() {
        let (spec, op) = medium(verify(64), |_| {});
        let op = op.unwrap();
        let entry = medium_entry(&spec);
        let symbol = entry.kernel.symbol();
        assert_eq!(
            op.info().symbols,
            [(symbol.clone(), entry.kernel.dynamic_shared_bytes())]
        );
        assert_eq!(
            op.info().bytes_of("scratch.partial"),
            Some(entry.plan(entry.max_rows()).unwrap().workspace_bytes)
        );
        let StaticQuery::Gemm(query) = spec.key.query else {
            unreachable!()
        };
        for rows in [1, 33, 48, 64] {
            let launches = op
                .launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                .unwrap();
            let counts = gemm::row_counts(&spec.key, rows);
            let [activation, weight] = gemm::fragment_maps(&spec.key, &query, 1, 4, false);
            let mut map = activation;
            map.box_dims[1] = entry.activation_box_rows();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol, symbol);
            assert_eq!(launches[0].plan, entry.plan(counts.m).unwrap());
            assert_eq!(
                launches[0].arguments,
                entry
                    .arguments_from(counts.m, map, weight.base_buffer, gemm_buffers())
                    .unwrap()
            );
            assert!(launches[0].persistent);
        }
        // Another entry's medium stand-in is refused when prepared.
        let (_, other) = medium(verify(64), |entry| {
            entry.reference = "chop_gemm_test_other_v1".into()
        });
        assert!(other.is_err());
    }

    /// A medium-M entry stands in only for a fragment prefill reference
    /// with the reference's K segments, and serves no use past its rows:
    /// both are refused when the op is prepared, not when it launches.
    #[test]
    fn medium_forms_refuse_other_segments_and_wider_uses() {
        let (_, segments) = medium(verify(64), |entry| entry.segments = 2);
        assert!(segments.is_err());
        // A tiled entry of the same query is not a fragment prefill entry.
        let tiled = gemm::instantiate(gemm::DenseRequest {
            query: gemm::bf16_query(8192, 4096, 4096, DType::Bf16),
            kernel: kernel_core::contract::DenseKernel::Tiled { splits: 2 },
            target: kernel_core::contract::Target::Sm120f,
            symbol: "chop_gemm_test_fragment_v1".into(),
            knobs: None,
        })
        .unwrap();
        let tiled = ProjectionOp::with_form(
            Arc::new(OperationHandle {
                compute: Compute::Dense(tiled.clone()),
                gather: None,
                quantizer: None,
            }),
            None,
            verify(64),
            ActivationGlobal::Absent,
            DenseForm::Medium(Box::new(medium_entry(&tiled))),
            false,
        );
        assert!(tiled.is_err());
        let max_rows = medium_entry(&fragment_entry(8192).0).max_rows();
        assert!(medium(verify(max_rows), |_| {}).1.is_ok());
        assert!(medium(verify(2 * max_rows), |_| {}).1.is_err());
    }

    /// A medium-M form binds by name under the identity of the stage it
    /// launches: an offer list, whose capabilities name the reference's
    /// stage, refuses it, and so does a prepare whose identity names the
    /// reference.
    #[test]
    fn medium_forms_bind_under_the_identity_of_their_stage() {
        use crate::backend::Sm120;
        use kernel_core::backend::Backend;
        let usage = verify(64);
        let launched = |spec: &kernel_core::contract::InstantiationSpec| {
            vec![medium_entry(spec).kernel.symbol()]
        };
        let form = |spec: &kernel_core::contract::InstantiationSpec| {
            DenseForm::Medium(Box::new(medium_entry(spec)))
        };
        let (offer, op, identity) = medium_offer(usage, form, launched);
        let kernels = [offer.offer.clone()];
        let available = capabilities(&kernels).unwrap().entries();
        assert!(OfferList::new(vec![offer.clone()], available, "a".repeat(64)).is_err());
        let prepare = |identity| {
            Sm120::new()
                .with_projections(Box::new(NamedOffers(Named(identity, offer.clone()))))
                .projection(&op, usage, Need::rows(64))
        };
        let prepared = prepare(identity.clone()).unwrap();
        assert_eq!(prepared.identity.stages, identity.stages);
        let (_, _, reference) = medium_offer(usage, form, |spec| vec![spec.symbol.clone()]);
        assert!(matches!(prepare(reference), Err(Refused::Rule(_))));
    }

    /// A use wider than a medium-M entry's rows is served when the need
    /// asks for at most those rows: its launches are the ones the use of
    /// the entry's rows makes, and wider launches are refused. A need of no
    /// rows, past the use's capacity, or past the medium rows is refused
    /// when the op is prepared.
    #[test]
    fn projections_serve_needs_within_their_use() {
        use crate::backend::Sm120;
        use kernel_core::backend::Backend;
        let form = |spec: &kernel_core::contract::InstantiationSpec| {
            DenseForm::Medium(Box::new(medium_entry(spec)))
        };
        let launched = |spec: &kernel_core::contract::InstantiationSpec| {
            vec![medium_entry(spec).kernel.symbol()]
        };
        let prepare = |usage: OperationUse, rows: u32| {
            let (offer, op, identity) = medium_offer(usage, form, launched);
            Sm120::new()
                .with_projections(Box::new(NamedOffers(Named(identity, offer))))
                .projection(&op, usage, Need::rows(rows))
                .map(|binding| binding.handle)
        };
        let max_rows = medium_entry(&fragment_entry(8192).0).max_rows();
        let narrow = prepare(verify(max_rows), max_rows).unwrap();
        for wide in [2 * max_rows, 4 * max_rows] {
            let op = prepare(verify(wide), max_rows).unwrap();
            assert_eq!(op.info(), narrow.info());
            for rows in [1, max_rows / 2 + 1, max_rows] {
                let launch = |op: &Prepared<ProjectionBuffers<View>>| {
                    op.launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                        .unwrap()
                };
                assert_eq!(launch(&op), launch(&narrow));
            }
            assert!(
                op.launches(
                    &fragment_buffers(u64::MAX),
                    View::EMPTY,
                    Counts::rows(max_rows + 1)
                )
                .is_err()
            );
            assert!(matches!(prepare(verify(wide), wide), Err(Refused::Rule(_))));
            assert!(matches!(prepare(verify(wide), 0), Err(Refused::Rule(_))));
            assert!(matches!(
                prepare(verify(wide), wide + 1),
                Err(Refused::Rule(_))
            ));
        }
        // A tensor-form op serves a smaller need with the use's launches.
        let (spec, usage) = fragment_entry(8192);
        let full = ProjectionOp::new(
            Arc::new(OperationHandle {
                compute: Compute::Dense(spec.clone()),
                gather: None,
                quantizer: None,
            }),
            None,
            usage,
            ActivationGlobal::Absent,
        )
        .unwrap();
        let served = ProjectionOp::with_rows(
            Arc::new(OperationHandle {
                compute: Compute::Dense(spec),
                gather: None,
                quantizer: None,
            }),
            None,
            usage,
            64,
            ActivationGlobal::Absent,
            DenseForm::Tensor,
            false,
        )
        .unwrap();
        assert_eq!(served.info(), full.info());
        let launch = |op: &ProjectionOp, rows| {
            op.launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
        };
        assert_eq!(launch(&served, 64).unwrap(), launch(&full, 64).unwrap());
        assert!(launch(&served, 65).is_err());
        assert!(launch(&full, 65).is_ok());
    }

    /// The projection key holds what a medium-M form launches, its nested
    /// fields changed alone: the entry's kernel and segments, which the
    /// reference settles, and never its profile, rows or claims, so offers
    /// that differ only there are one.
    #[test]
    fn medium_launch_keys_hold_what_the_medium_entry_launches() {
        use crate::backend::Sm120;
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        use kernel_core::backend::Backend;
        let usage = verify(64);
        let form = |spec: &kernel_core::contract::InstantiationSpec| {
            DenseForm::Medium(Box::new(medium_entry(spec)))
        };
        let (base, op, identity) =
            medium_offer(usage, form, |spec| vec![medium_entry(spec).kernel.symbol()]);
        fn entry(o: &mut ProjectionOffer) -> &mut gemm::mid::Entry {
            let DenseForm::Medium(entry) = &mut o.dense else {
                unreachable!()
            };
            entry
        }
        let names = field_names!(gemm::mid::Entry {
            profile,
            reference,
            query,
            kernel,
            segments,
            rows,
            invariance
        });
        let fields: Vec<Field<ProjectionOffer>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(|o| {
                    entry(o).profile = crate::Profile::synthetic(crate::ProfileNames {
                        name: "other",
                        tag: "other",
                    })
                }),
            ),
            (
                "reference",
                Class::Selects,
                Box::new(|o| entry(o).reference = "chop_gemm_test_other_v1".into()),
            ),
            (
                "query",
                Class::Selects,
                Box::new(|o| entry(o).query.k = 8192),
            ),
            (
                "kernel",
                Class::Selects,
                Box::new(|o| entry(o).kernel.fp8 = true),
            ),
            (
                "segments",
                Class::Selects,
                Box::new(|o| entry(o).segments = 2),
            ),
            ("rows", Class::Inert, Box::new(|o| entry(o).rows = 1..=64)),
            (
                "invariance",
                Class::Inert,
                Box::new(|o| entry(o).invariance = Invariance::PREFILL),
            ),
        ];
        let buffers = fragment_buffers(u64::MAX);
        let named = |o: &ProjectionOffer| Binding {
            identity: identity.clone(),
            handle: o.clone(),
        };
        keys_follow_launches(
            &base,
            names,
            fields,
            |o| bound_launch(&named(o)),
            |o| {
                let bound = Sm120::new()
                    .with_projections(Box::new(NamedOffers(Named(identity.clone(), o.clone()))))
                    .projection(&op, usage, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    bound
                        .handle
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", bound.identity), launches))
            },
        );

        // Two offers that differ only in the entry's claims launch alike,
        // so they are one, not a tie.
        struct Twins(Vec<Binding<ProjectionOffer>>);
        impl ProjectionOffers for Twins {
            fn bind_all(
                &self,
                _: &OperationSpec,
                _: OperationUse,
            ) -> Result<Vec<Binding<ProjectionOffer>>, String> {
                Ok(self
                    .0
                    .iter()
                    .map(|binding| Binding {
                        identity: binding.identity.clone(),
                        handle: binding.handle.clone(),
                    })
                    .collect())
            }
        }
        let mut twin = base.clone();
        entry(&mut twin).invariance = Invariance::PREFILL;
        let twins = Twins(vec![named(&base), named(&twin)]);
        assert!(
            Sm120::new()
                .with_projections(Box::new(twins))
                .projection(&op, usage, Need::rows(64))
                .is_ok()
        );
    }

    /// A live output holds each launch's rows; any other output holds the
    /// entry's capacity, and the op lists those bytes.
    #[test]
    fn live_outputs_hold_the_launched_rows() {
        let row = 4096 * 2;
        let (_, capacity) = fragment(8192, |_| DenseForm::Fragment, false);
        let capacity = capacity.unwrap();
        assert_eq!(capacity.info().bytes_of("output"), Some(8192 * row));
        assert!(
            capacity
                .launches(&fragment_buffers(3 * row), View::EMPTY, Counts::rows(3))
                .is_err()
        );
        let (_, live) = fragment(8192, |_| DenseForm::Fragment, true);
        let live = live.unwrap();
        assert_eq!(live.info().bytes_of("output"), None);
        let at = |rows| live.launches(&fragment_buffers(3 * row), View::EMPTY, Counts::rows(rows));
        assert_eq!(
            at(3).unwrap(),
            capacity
                .launches(&fragment_buffers(u64::MAX), View::EMPTY, Counts::rows(3))
                .unwrap()
        );
        assert!(at(4).is_err());
        // The tensor form with its default output keeps the identity it had.
        let (_, tensor) = fragment(8192, |_| DenseForm::Tensor, false);
        assert_ne!(
            tensor.unwrap().info().implementation,
            capacity.info().implementation
        );
    }
}
