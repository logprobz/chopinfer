//! Retained precision bindings lowered through the existing typed planners.
//! Models supply routing, state inputs and registered entries. This module
//! owns the conversion, compute and output-copy sequence, without selecting
//! arithmetic again at execution.
use kernel_core::allocation::Traversal as _;
use std::{collections::BTreeMap, sync::Arc};

use kernel_core::backend::{Counts, OpInfo, PreparedOp, View};
use kernel_core::contract::{Refusal, Validate, WeightFormat};
use kernel_core::moe::RouteMetadata;
pub use kernel_core::precision::buffers::{ActivationGlobal, ProjectionScratch as ScratchBuffers};
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionRouting, QuantizedWeights};
use kernel_core::precision::{
    OperationUse,
    execution::{ImplementationIdentity, ImplementationProvider, PrecisionDocument},
};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use crate::{
    dispatch::{self, BufferBindings},
    pipeline::PreparedLaunch,
};

/// Persistent operand views. Their owner authenticates the image and layouts.
#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub codes: Span,
    pub local_scales: Option<Span>,
    pub global_scales: Option<Span>,
    /// Supplied activation calibration, authenticated with the persistent image.
    pub activation_global: Option<Span>,
}

impl Weights {
    /// The views of operation `id` under checkpoint suffixes: `weight`
    /// codes, then `weight_scale_inv` (FP8 block) or `weight_scale` local
    /// scales and the NVFP4 `weight_scale_2` global.
    pub fn named(
        id: &str,
        format: WeightFormat,
        views: &BTreeMap<String, Span>,
    ) -> Result<Self, String> {
        let view = |suffix| {
            views
                .get(&format!("{id}.{suffix}"))
                .copied()
                .ok_or_else(|| format!("missing projection weight {id}.{suffix}"))
        };
        Ok(Self {
            codes: view("weight")?,
            local_scales: match format {
                WeightFormat::Bf16 => None,
                WeightFormat::Fp8Block128x128(_) => Some(view("weight_scale_inv")?),
                _ => Some(view("weight_scale")?),
            },
            global_scales: if format == WeightFormat::Nvfp4 {
                Some(view("weight_scale_2")?)
            } else {
                None
            },
            activation_global: None,
        })
    }
}

/// The projection op a backend prepared, over buffers by role.
pub type PreparedProjection = Arc<dyn PreparedOp<ProjectionBuffers<View>>>;

/// Typed recipe supplied by the admitted implementation provider.
#[derive(Clone)]
pub enum Recipe {
    /// An op a backend prepared (`kernel_core::backend::Backend::projection`).
    Prepared {
        /// Shared by the uses a provider supplies one op for, which then
        /// share one authenticated use ([`Authenticated::new`]).
        op: PreparedProjection,
        /// Registered stages before the op.
        prefix: Vec<String>,
    },
    Registered,
}

/// A registry callback handles a complete legacy recipe or only the prefix
/// of a prepared recipe. It must return launches of its authenticated stages.
pub trait Launcher: Send + Sync {
    /// Capacity requirements of the registered recipe or prefix.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// Per-record executions of each authenticated callback stage. Empty
    /// means one execution each; entries in the identity are never collapsed.
    fn repetitions(&self, _phase: Phase, _input: Input<'_>) -> Result<Vec<u32>, Refusal> {
        Ok(vec![])
    }

    /// Regions written by this callback. Prefixes of prepared recipes may
    /// write only side spans; complete registered recipes may also write output and scratch.
    fn writes(
        &self,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<Span>, Refusal>;

    fn launch(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal>;

    /// [`Launcher::launch`], with whether each launch is persistent
    /// (`kernel_core::backend::Launch::persistent`) where the launcher
    /// knows it, such as one that launches a backend's prepared op: `None`
    /// by default.
    fn launch_with_persistence(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<(PreparedLaunch, Option<bool>)>, Refusal> {
        Ok(self
            .launch(phase, input, output, scratch)?
            .into_iter()
            .map(|launch| (launch, None))
            .collect())
    }
}

/// Keys include dependency, arithmetic, layouts, activations and stage order.
/// Several uses of one implementation may have distinct identities.
/// The launchers of one implementation name, by complete identity.
type Launchers = Vec<(ImplementationIdentity, Arc<dyn Launcher>)>;

#[derive(Default)]
pub struct Registry {
    entries: BTreeMap<String, Launchers>,
}
impl Registry {
    pub fn register(
        &mut self,
        identity: ImplementationIdentity,
        launcher: Arc<dyn Launcher>,
    ) -> Result<(), Refusal> {
        if self.find(&identity).is_some() {
            return Err(Refusal("duplicate projection launcher identity"));
        }
        self.entries
            .entry(identity.implementation.clone())
            .or_default()
            .push((identity, launcher));
        Ok(())
    }
    /// One launcher per distinct authenticated identity of `document`, in
    /// document order. `launcher` returns `None` for uses a prepared recipe
    /// serves without a registered stage.
    pub fn for_document(
        document: &PrecisionDocument,
        mut launcher: impl FnMut(
            &kernel_core::precision::execution::PrecisionSelection,
            &kernel_core::precision::execution::BoundUse,
        ) -> Result<Option<Arc<dyn Launcher>>, String>,
    ) -> Result<Self, String> {
        let mut registry = Self::default();
        let mut seen: BTreeMap<&str, Vec<&ImplementationIdentity>> = BTreeMap::new();
        for selection in &document.operations {
            for binding in &selection.bindings {
                let identity = &binding.implementation;
                let seen = seen.entry(identity.implementation.as_str()).or_default();
                if seen.contains(&identity) {
                    continue;
                }
                seen.push(identity);
                if let Some(launcher) = launcher(selection, binding)? {
                    registry
                        .register(identity.clone(), launcher)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(registry)
    }
    fn find(&self, identity: &ImplementationIdentity) -> Option<&Arc<dyn Launcher>> {
        self.entries
            .get(&identity.implementation)?
            .iter()
            .find(|(key, _)| key == identity)
            .map(|(_, launcher)| launcher)
    }
    fn get(&self, identity: &ImplementationIdentity) -> Result<Arc<dyn Launcher>, String> {
        self.find(identity).map(Arc::clone).ok_or_else(|| {
            format!(
                "unregistered projection implementation {}",
                identity.implementation
            )
        })
    }
}

/// Counts for this launch. The use includes capacity and graph mode, so a
/// decode handle cannot silently serve verification or another graph bucket.
#[derive(Clone, Copy, Debug)]
pub struct Phase {
    pub usage: OperationUse,
    pub rows: u32,
    pub padded_rows: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Routing {
    pub weights: Span,
    pub offsets: Span,
    pub sorted_rows: Span,
    pub padded_rows: Span,
    pub active_experts: Span,
}

#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    pub activation: Span,
    pub live_rows: Span,
    pub routing: Option<Routing>,
    pub residual: Option<Span>,
    pub row_offset: u64,
    pub shard_rows: u64,
    /// Per-record inputs interpreted only by the registered implementation.
    pub side: &'a [Span],
    pub scalars: &'a [u64],
}

/// Regions belong to the caller's arena (`ScratchBuffers`). Dynamic global
/// scales are writable scratch; supplied globals can also be retained in
/// Weights::activation_global.
pub type Scratch = ScratchBuffers<Span>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Needs {
    pub workspace_bytes: u64,
    /// External counter storage, initialized to zero before the first launch.
    pub semaphore_bytes: u64,
    pub output_bytes: u64,
    pub temporary_bytes: u64,
    pub gathered_bytes: u64,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    /// Includes the global scalar for a dynamic tensor-amax recipe.
    pub quantizer_scratch_bytes: u64,
}

impl Needs {
    /// The scratch and output bytes a prepared op lists by role.
    fn of(info: &OpInfo) -> Self {
        let bytes = |role: &str| info.bytes_of(role).unwrap_or(0);
        Self {
            workspace_bytes: bytes("scratch.partial"),
            semaphore_bytes: bytes("scratch.semaphores"),
            output_bytes: bytes("output"),
            temporary_bytes: bytes("scratch.temporary"),
            gathered_bytes: bytes("scratch.gathered"),
            codes_bytes: bytes("scratch.codes"),
            scales_bytes: bytes("scratch.scales"),
            quantizer_scratch_bytes: bytes("scratch.row_amax") + bytes("scratch.activation_global"),
        }
    }

    /// The larger of each requirement.
    pub fn max(self, other: Self) -> Self {
        Self {
            workspace_bytes: self.workspace_bytes.max(other.workspace_bytes),
            semaphore_bytes: self.semaphore_bytes.max(other.semaphore_bytes),
            output_bytes: self.output_bytes.max(other.output_bytes),
            temporary_bytes: self.temporary_bytes.max(other.temporary_bytes),
            gathered_bytes: self.gathered_bytes.max(other.gathered_bytes),
            codes_bytes: self.codes_bytes.max(other.codes_bytes),
            scales_bytes: self.scales_bytes.max(other.scales_bytes),
            quantizer_scratch_bytes: self
                .quantizer_scratch_bytes
                .max(other.quantizer_scratch_bytes),
        }
    }
}

#[derive(Clone)]
/// One rank's use: its persistent views and the instance's authenticated use.
pub struct Handle {
    activation_global: ActivationGlobal,
    usage: OperationUse,
    weights: Weights,
    needs: Needs,
    bound: Arc<Bound>,
}
impl Handle {
    pub fn activation_global(&self) -> ActivationGlobal {
        self.activation_global
    }
    pub fn identity(&self) -> &ImplementationIdentity {
        &self.bound.identity
    }
    pub fn needs(&self) -> Needs {
        self.needs
    }
    pub fn recipe(&self) -> &Recipe {
        &self.bound.recipe
    }
    pub fn weights(&self) -> Weights {
        self.weights
    }
}

/// One authenticated use without persistent views, shared by every rank.
struct Bound {
    activation_global: ActivationGlobal,
    identity: ImplementationIdentity,
    usage: OperationUse,
    recipe: Recipe,
    /// The prepared recipe's op after any registered prefix.
    op: Option<PreparedProjection>,
    launcher: Option<Arc<dyn Launcher>>,
    needs: Needs,
    /// A prepared recipe's stages with their dynamic shared memory.
    shared_memory: Option<Vec<(String, u32)>>,
}

/// A precision component authenticated once per instance: every use is
/// rebound to the supplied inventory and its typed adapter, stages, launcher
/// and capacity needs are retained. Ranks then attach only their views
/// ([`Authenticated::bind`]), so a multi-rank instance binds the inventory
/// once rather than once per rank.
pub struct Authenticated {
    document: Arc<PrecisionDocument>,
    /// Each selection's position in the document, which every rank shares.
    positions: Arc<BTreeMap<String, usize>>,
    /// The uses of each selection, in document order.
    bound: Vec<Vec<Arc<Bound>>>,
}

impl Authenticated {
    /// Decode the component bytes, then authenticate them ([`Self::new`]).
    pub fn decode<P: ImplementationProvider<Handle = Recipe>>(
        bytes: &[u8],
        provider: &P,
        registry: &Registry,
    ) -> Result<Self, String> {
        Self::decoded(
            kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?,
            provider,
            registry,
        )
    }

    /// Authenticate `document` as [`kernel_core::precision::component::decode`]
    /// returned it ([`Self::new`]). The decoder validated it, so a caller
    /// that inspects the decoded document first does not validate it again.
    pub fn decoded<P: ImplementationProvider<Handle = Recipe>>(
        document: PrecisionDocument,
        provider: &P,
        registry: &Registry,
    ) -> Result<Self, String> {
        Self::authenticate(document, provider, registry)
    }

    /// Rebind each authenticated use to the supplied inventory, then retain
    /// the exact typed adapter. No model is inferred from an ID, symbol or
    /// format. The provider also owns layout validation.
    pub fn new<P: ImplementationProvider<Handle = Recipe>>(
        document: PrecisionDocument,
        provider: &P,
        registry: &Registry,
    ) -> Result<Self, String> {
        document.validate().map_err(|e| e.to_string())?;
        Self::authenticate(document, provider, registry)
    }

    fn authenticate<P: ImplementationProvider<Handle = Recipe>>(
        document: PrecisionDocument,
        provider: &P,
        registry: &Registry,
    ) -> Result<Self, String> {
        // Uses of one identity, use and activation recipe that the provider
        // supplies the same recipe for share one bound use.
        let mut authenticated: BTreeMap<String, Vec<Arc<Bound>>> = BTreeMap::new();
        let mut positions = BTreeMap::new();
        let mut bound = Vec::with_capacity(document.operations.len());
        for selection in &document.operations {
            let activation_global = ActivationGlobal::of(&selection.effective)?;
            let mut uses = Vec::new();
            for binding in &selection.bindings {
                let supplied = provider.bind(&selection.effective, binding.usage)?;
                if supplied.identity != binding.implementation {
                    return Err(format!(
                        "operation {}: projection identity differs from authenticated binding",
                        selection.source.id
                    ));
                }
                let earlier = authenticated
                    .get(&supplied.identity.implementation)
                    .and_then(|bound| {
                        bound.iter().find(|b| {
                            b.usage == binding.usage
                                && b.activation_global == activation_global
                                && b.identity == supplied.identity
                                && same_recipe(&b.recipe, &supplied.handle)
                        })
                    });
                if let Some(earlier) = earlier {
                    uses.push(Arc::clone(earlier));
                    continue;
                }
                let (op, stages) = recipe_op(&supplied.handle);
                let mut needs = op
                    .as_ref()
                    .map_or_else(Needs::default, |op| Needs::of(op.info()));
                let registered = stages.as_ref().is_none_or(|s| s.prefixed);
                if stages
                    .as_ref()
                    .is_some_and(|s| s.stages != supplied.identity.stages)
                {
                    return Err(format!(
                        "operation {}: projection adapter stage order differs from authenticated binding",
                        selection.source.id
                    ));
                }
                let launcher = if registered {
                    Some(registry.get(&supplied.identity)?)
                } else {
                    None
                };
                if let Some(launcher) = &launcher {
                    needs = needs.max(launcher.needs());
                }
                let bound = Arc::new(Bound {
                    activation_global,
                    identity: supplied.identity,
                    usage: binding.usage,
                    recipe: supplied.handle,
                    op,
                    launcher,
                    needs,
                    shared_memory: stages.map(|s| s.shared_memory),
                });
                authenticated
                    .entry(bound.identity.implementation.clone())
                    .or_default()
                    .push(Arc::clone(&bound));
                uses.push(bound);
            }
            if positions
                .insert(selection.source.id.clone(), bound.len())
                .is_some()
            {
                return Err(format!(
                    "operation {}: duplicate authenticated projection",
                    selection.source.id
                ));
            }
            bound.push(uses);
        }
        Ok(Self {
            document: Arc::new(document),
            positions: Arc::new(positions),
            bound,
        })
    }

    pub fn document(&self) -> &PrecisionDocument {
        &self.document
    }

    /// Where every store bound from this authentication holds projection
    /// `id`'s use at `usage` ([`Store::at`]), found once so a step indexes
    /// instead of searching.
    pub fn locate(&self, id: &str, usage: OperationUse) -> Option<UseIndex> {
        let position = *self.positions.get(id)?;
        let index = self.bound[position].iter().position(|b| b.usage == usage)?;
        Some(UseIndex { position, index })
    }

    /// Each shared stage with the dynamic shared memory its symbol is
    /// configured for at load, and the largest needs of any prepared recipe.
    /// Registered recipes and prefixes size their own stages.
    pub fn shared_requirements(&self) -> (BTreeMap<String, u32>, Needs) {
        let mut symbols = BTreeMap::new();
        let mut needs = Needs::default();
        for bound in self.bound.iter().flatten() {
            if let Some(stages) = &bound.shared_memory {
                for (stage, bytes) in stages {
                    if !symbols.contains_key(stage) {
                        symbols.insert(stage.clone(), *bytes);
                    }
                }
                needs = needs.max(bound.needs);
            }
        }
        (symbols, needs)
    }

    /// Attach one rank's persistent operand views. `weights` returns `None`
    /// for a selection this rank does not serve; the store then holds no
    /// handle for it, and its document lists only the retained selections.
    pub fn bind(
        &self,
        mut weights: impl FnMut(
            &kernel_core::precision::execution::PrecisionSelection,
        ) -> Result<Option<Weights>, String>,
    ) -> Result<Store, String> {
        let mut retained = Vec::new();
        let mut handles = Vec::with_capacity(self.bound.len());
        for (selection, uses) in self.document.operations.iter().zip(&self.bound) {
            let Some(views) = weights(selection)? else {
                handles.push(Vec::new());
                continue;
            };
            let mut bound = Vec::with_capacity(uses.len());
            for u in uses {
                validate_global_view(u.activation_global, views).map_err(|e| e.to_string())?;
                bound.push(Handle {
                    activation_global: u.activation_global,
                    usage: u.usage,
                    weights: views,
                    needs: u.needs,
                    bound: Arc::clone(u),
                });
            }
            handles.push(bound);
            retained.push(selection);
        }
        // A rank serving every selection shares the authenticated document.
        let document = if retained.len() == self.document.operations.len() {
            Arc::clone(&self.document)
        } else {
            Arc::new(PrecisionDocument {
                version: self.document.version,
                operations: retained.into_iter().cloned().collect(),
            })
        };
        Ok(Store {
            document,
            positions: Arc::clone(&self.positions),
            handles,
        })
    }
}

pub struct Store {
    document: Arc<PrecisionDocument>,
    positions: Arc<BTreeMap<String, usize>>,
    /// Each authenticated selection's handles, in its document order;
    /// empty for a selection this rank does not serve.
    handles: Vec<Vec<Handle>>,
}
impl Store {
    pub fn document(&self) -> &PrecisionDocument {
        &self.document
    }

    /// Authenticate and bind one store ([`Authenticated`]); every selection
    /// needs persistent views.
    pub fn decode<P: ImplementationProvider<Handle = Recipe>>(
        bytes: &[u8],
        provider: &P,
        registry: &Registry,
        mut weights: impl FnMut(
            &kernel_core::precision::execution::PrecisionSelection,
        ) -> Result<Weights, String>,
    ) -> Result<Self, String> {
        Authenticated::decode(bytes, provider, registry)?.bind(|s| weights(s).map(Some))
    }
    pub fn handle(&self, id: &str, usage: OperationUse) -> Result<&Handle, Refusal> {
        self.positions
            .get(id)
            .and_then(|&position| self.handles[position].iter().find(|h| h.usage == usage))
            .ok_or(Refusal("missing bound projection use"))
    }

    /// The handle at `index`, which [`Authenticated::locate`] of the
    /// authentication this store was bound from returned; refused when this
    /// store does not serve its selection.
    pub fn at(&self, index: UseIndex) -> Result<&Handle, Refusal> {
        self.handles
            .get(index.position)
            .and_then(|uses| uses.get(index.index))
            .ok_or(Refusal("missing bound projection use"))
    }
}

/// An authenticated use's place in the stores bound from its
/// authentication: its selection's position and its index among the
/// selection's uses ([`Authenticated::locate`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UseIndex {
    position: usize,
    index: usize,
}

/// Whether two supplied recipes are one: both registered, or one prepared
/// op with the same prefix.
fn same_recipe(a: &Recipe, b: &Recipe) -> bool {
    match (a, b) {
        (Recipe::Registered, Recipe::Registered) => true,
        (
            Recipe::Prepared {
                op: a,
                prefix: a_prefix,
            },
            Recipe::Prepared {
                op: b,
                prefix: b_prefix,
            },
        ) => Arc::ptr_eq(a, b) && a_prefix == b_prefix,
        _ => false,
    }
}

/// A prepared recipe's stage order the binding must authenticate, whether a
/// registered prefix runs first, and each op stage's dynamic shared memory.
struct RecipeStages {
    stages: Vec<String>,
    prefixed: bool,
    shared_memory: Vec<(String, u32)>,
}

/// The op of a prepared recipe and its stages after any registered
/// prefix. A complete registered recipe declares its needs
/// itself.
fn recipe_op(recipe: &Recipe) -> (Option<PreparedProjection>, Option<RecipeStages>) {
    let (op, prefix) = match recipe {
        Recipe::Registered => return (None, None),
        Recipe::Prepared { op, prefix } => (Arc::clone(op), prefix),
    };
    let symbols = &op.info().symbols;
    let stages = prefix
        .iter()
        .cloned()
        .chain(symbols.iter().map(|(symbol, _)| symbol.clone()))
        .collect();
    let stages = RecipeStages {
        stages,
        prefixed: !prefix.is_empty(),
        shared_memory: symbols.clone(),
    };
    (Some(op), Some(stages))
}

fn validate_global_view(recipe: ActivationGlobal, weights: Weights) -> Result<(), Refusal> {
    match (recipe, weights.activation_global) {
        (ActivationGlobal::Supplied, Some(view)) => covers(view, 4),
        (ActivationGlobal::Supplied, None) => Err(Refusal(
            "supplied activation global needs a persistent view",
        )),
        (_, Some(_)) => Err(Refusal(
            "persistent activation global disagrees with bound recipe",
        )),
        (_, None) => Ok(()),
    }
}

fn covers(view: Span, bytes: u64) -> Result<(), Refusal> {
    if view.length < bytes {
        Err(Refusal(
            "projection region does not cover selected capacity",
        ))
    } else {
        Ok(())
    }
}
fn overlap(a: Span, b: Span) -> Result<bool, Refusal> {
    let end_a = a
        .offset
        .checked_add(a.length)
        .ok_or(Refusal("projection region range overflows"))?;
    let end_b = b
        .offset
        .checked_add(b.length)
        .ok_or(Refusal("projection region range overflows"))?;
    Ok(
        a.buffer == b.buffer
            && a.length > 0
            && b.length > 0
            && a.offset < end_b
            && b.offset < end_a,
    )
}

/// Writable scratch regions in declaration order. `unused` only fills kernel
/// arguments that the selected recipe never reads, so callers may point it at
/// any live region; it is neither checked for overlap nor writable.
fn scratch_regions(s: &Scratch) -> Vec<Span> {
    let mut regions = vec![];
    s.visit_named(&mut |name, span| {
        if name != "unused" {
            regions.push(*span);
        }
    });
    regions
}

fn check_regions(
    handle: &Handle,
    input: Input<'_>,
    output: Span,
    s: &Scratch,
) -> Result<(), Refusal> {
    let regions = scratch_regions(s);
    for (index, &region) in regions.iter().enumerate() {
        for &other in &regions[index + 1..] {
            if overlap(region, other)? {
                return Err(Refusal("projection scratch regions overlap"));
            }
        }
    }
    let n = handle.needs;
    let active = [
        (n.workspace_bytes > 0).then_some(s.partial),
        (n.semaphore_bytes > 0).then_some(s.semaphores),
        if n.temporary_bytes > 0 {
            s.temporary
        } else {
            None
        },
        if n.gathered_bytes > 0 {
            s.gathered
        } else {
            None
        },
        if n.codes_bytes > 0 { s.codes } else { None },
        if n.scales_bytes > 0 { s.scales } else { None },
        if handle.activation_global == ActivationGlobal::TensorAmax {
            s.activation_global
        } else {
            None
        },
        if n.quantizer_scratch_bytes > 0 {
            s.row_amax
        } else {
            None
        },
        s.output_codes,
        s.output_scales,
    ];
    let mut reads = vec![
        input.activation,
        input.live_rows,
        handle.weights.codes,
        output,
    ];
    reads.extend(
        [
            input.residual,
            handle.weights.local_scales,
            handle.weights.global_scales,
            handle.weights.activation_global,
        ]
        .into_iter()
        .flatten(),
    );
    reads.extend_from_slice(input.side);
    if let Some(r) = input.routing {
        reads.extend([
            r.weights,
            r.offsets,
            r.sorted_rows,
            r.padded_rows,
            r.active_experts,
        ]);
    }
    for region in active.into_iter().flatten() {
        for &other in &reads {
            if overlap(region, other)? {
                return Err(Refusal(
                    "projection scratch overlaps input, output or persistent views",
                ));
            }
        }
    }
    if let Some(launcher) = &handle.bound.launcher {
        let prepared = !matches!(handle.bound.recipe, Recipe::Registered);
        let mut allowed = input.side.to_vec();
        if !prepared {
            allowed.push(output);
            allowed.extend(regions);
        }
        let written = launcher.writes(input, output, s)?;
        for (index, &region) in written.iter().enumerate() {
            for immutable in [
                Some(handle.weights.codes),
                handle.weights.local_scales,
                handle.weights.global_scales,
                handle.weights.activation_global,
            ]
            .into_iter()
            .flatten()
            {
                if overlap(region, immutable)? {
                    return Err(Refusal("registered projection writes persistent weights"));
                }
            }
            if !allowed.iter().any(|view| {
                view.buffer == region.buffer
                    && view.offset <= region.offset
                    && region
                        .offset
                        .checked_add(region.length)
                        .zip(view.offset.checked_add(view.length))
                        .is_some_and(|(end, bound)| end <= bound)
            }) {
                return Err(Refusal("registered projection writes an undeclared region"));
            }
            for &other in &written[index + 1..] {
                if overlap(region, other)? {
                    return Err(Refusal("registered projection written regions overlap"));
                }
            }
        }
    }
    Ok(())
}

/// Produce the entire ordered sequence for pipeline::launch_all or recorded
/// graph submission. Arguments retain spans and tensor maps, never CUDA addresses.
pub fn launch(
    handle: &Handle,
    phase: Phase,
    input: Input<'_>,
    output: Span,
    scratch: &Scratch,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    Ok(
        launch_with_persistence(handle, phase, input, output, scratch)?
            .into_iter()
            .map(|(launch, _)| launch)
            .collect(),
    )
}

/// [`launch`], with whether each launch of the recipe's op is persistent
/// (`kernel_core::backend::Launch::persistent`: its grid strides over the
/// GPU's SMs). A registered stage's launch has what its launcher states
/// ([`Launcher::launch_with_persistence`]), `None` by default.
pub fn launch_with_persistence(
    handle: &Handle,
    phase: Phase,
    input: Input<'_>,
    output: Span,
    scratch: &Scratch,
) -> Result<Vec<(PreparedLaunch, Option<bool>)>, Refusal> {
    if phase.usage != handle.usage || phase.rows == 0 || phase.rows > phase.usage.capacity {
        return Err(Refusal("projection phase does not match bound use"));
    }
    validate_global_view(handle.activation_global, handle.weights)?;
    covers(scratch.partial, handle.needs.workspace_bytes)?;
    covers(scratch.semaphores, handle.needs.semaphore_bytes)?;
    covers(output, handle.needs.output_bytes)?;
    check_regions(handle, input, output, scratch)?;
    let mut launches = if let Some(launcher) = &handle.bound.launcher {
        launcher.launch_with_persistence(phase, input, output, scratch)?
    } else {
        Vec::new()
    };
    let expected_prefix = match &handle.bound.recipe {
        Recipe::Registered => &handle.bound.identity.stages,
        Recipe::Prepared { prefix, .. } => prefix,
    };
    let repetitions = if let Some(launcher) = &handle.bound.launcher {
        launcher.repetitions(phase, input)?
    } else {
        vec![]
    };
    if (!repetitions.is_empty() && repetitions.len() != expected_prefix.len())
        || repetitions.contains(&0)
    {
        return Err(Refusal("invalid registered projection stage repetitions"));
    }
    let expected = expected_prefix
        .iter()
        .enumerate()
        .flat_map(|(index, symbol)| {
            std::iter::repeat_n(
                symbol,
                repetitions.get(index).copied().unwrap_or(1) as usize,
            )
        });
    if !launches.iter().map(|(item, _)| &item.symbol).eq(expected) {
        return Err(Refusal(
            "registered projection stages differ from bound recipe",
        ));
    }
    for (item, _) in &launches {
        item.plan.validate()?;
        dispatch::arguments(&item.arguments, &item.bindings)
            .map_err(|_| Refusal("invalid registered projection arguments"))?;
    }
    let Some(op) = &handle.bound.op else {
        return Ok(launches);
    };
    // One binding per role; the op's arguments name the roles it reads.
    let mut b = BufferBindings::default();
    let mut view = |span: Span| View {
        handle: b.bind(span),
        bytes: span.length,
    };
    let weights = handle.weights;
    let buffers = ProjectionBuffers {
        activation: view(input.activation),
        live_rows: view(input.live_rows),
        output: view(output),
        residual: input.residual.map(&mut view),
        weights: QuantizedWeights {
            codes: view(weights.codes),
            local_scales: weights.local_scales.map(&mut view),
            global_scales: weights.global_scales.map(&mut view),
        },
        activation_global: weights.activation_global.map(&mut view),
        routing: input.routing.map(|r| ProjectionRouting {
            weights: view(r.weights),
            routes: RouteMetadata {
                counts: None,
                offsets: view(r.offsets),
                sorted_rows: view(r.sorted_rows),
                padded_rows: view(r.padded_rows),
                active_experts: Some(view(r.active_experts)),
            },
        }),
        scratch: scratch.map(&mut view),
        row_offset: input.row_offset,
        shard_rows: input.shard_rows,
    };
    // A registered prefix shares the op's scratch, so each region the op
    // sizes holds the larger of the op's bytes and the handle's needs.
    let needs = handle.needs;
    let combined = |role: &str| match role {
        "output" => needs.output_bytes,
        "scratch.partial" => needs.workspace_bytes,
        "scratch.semaphores" => needs.semaphore_bytes,
        "scratch.temporary" => needs.temporary_bytes,
        "scratch.gathered" => needs.gathered_bytes,
        "scratch.codes" => needs.codes_bytes,
        "scratch.scales" => needs.scales_bytes,
        // Quantizer scratch is per-row amax, then the 4-byte tensor global.
        "scratch.row_amax" => needs.quantizer_scratch_bytes.saturating_sub(4),
        _ => 0,
    };
    let mut short = false;
    buffers.visit_named(&mut |role, view| {
        if let Some(bytes) = op.info().bytes_of(role) {
            short |= view.bytes < bytes.max(combined(role));
        }
    });
    if short {
        return Err(Refusal(
            "projection region does not cover selected capacity",
        ));
    }
    if op.info().workspace > 0 {
        return Err(Refusal("the projection host binds no backend workspace"));
    }
    let counts = Counts {
        padded_rows: phase.padded_rows,
        ..Counts::rows(phase.rows)
    };
    for item in op.launches(&buffers, View::EMPTY, counts)? {
        let mut bindings = b.clone();
        if kernel_core::route::Metadata::enabled() {
            bindings.set_route(kernel_core::route::selected(op.as_ref(), counts, &item));
        }
        launches.push((
            PreparedLaunch {
                symbol: item.symbol,
                rows: item.rows,
                plan: item.plan,
                arguments: item.arguments,
                bindings,
            },
            Some(item.persistent),
        ));
    }
    Ok(launches)
}

#[cfg(test)]
mod tests;
