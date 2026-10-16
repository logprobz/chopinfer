//! Authenticate the precision component once per instance; each rank then
//! attaches its persistent views to the retained launch handles.
use crate::{contract::*, launch, program::Phase};
use bundle_runtime::projection as shared;
use kernel_core::backend::Need;
use kernel_core::contract::{Refusal, WeightFormat};
use kernel_core::precision::execution::{
    Binding, BoundUse, ImplementationIdentity, ImplementationProvider, PrecisionDocument,
};
use kernel_core::precision::{ExecutionPhase, OperationKind, OperationSpec, OperationUse};
use kernel_core::properties::{Anchors, Invariance};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use std::{collections::BTreeMap, sync::Arc};

mod registered;
pub(crate) use registered::legacy_lookup;

pub struct Prepared {
    authenticated: shared::Authenticated,
    stores: Vec<shared::Store>,
    pub dense_symbols: BTreeMap<String, String>,
    pub shared_memory: BTreeMap<String, u32>,
    pub workspace_bytes: u64,
    pub temporary_bytes: u64,
    /// The norm weights the rank views store in F32, whose consumers bind
    /// their F32 forms (`model_qwen38_contract::norms`).
    pub norms: model_qwen38_contract::norms::NormWeights,
    /// What the prepared generated projections guarantee, by each use and
    /// symbol they launch at it, for checks of the launches against what
    /// serving relies on. A symbol serves several uses (a layout copy every
    /// capacity), so its claims are those of the op of the use.
    pub guarantees: Vec<(OperationUse, String, Invariance)>,
}

/// Each distinct authenticated implementation's recipe, and each
/// operation's uses of them. `prepared::bind` has already bound every use
/// of the component to the supplied inventory.
#[derive(Default)]
struct Provider {
    recipes: Vec<(shared::Recipe, ImplementationIdentity)>,
    uses: BTreeMap<String, Vec<(OperationUse, usize)>>,
}

impl ImplementationProvider for Provider {
    type Handle = shared::Recipe;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String> {
        let (recipe, identity) = self
            .uses
            .get(&operation.id)
            .and_then(|uses| uses.iter().find(|(u, _)| *u == usage))
            .and_then(|&(_, index)| self.recipes.get(index))
            .ok_or("missing declared projection recipe")?;
        Ok(Binding {
            identity: identity.clone(),
            handle: recipe.clone(),
        })
    }
}

/// What serving relies on of a generated projection at `usage`: its rows
/// over chunks at the manifest's granule in prefill, one-row bits in the
/// exact verify steps, nothing else (`crate::ops::exact`). Projections
/// write no per-slot state.
fn projection_need(usage: OperationUse) -> Need {
    let key = crate::ops::Key::of_use(usage);
    Need::rows(usage.capacity).with_exact(crate::ops::exact(
        key,
        usage.capacity,
        Anchors::NotApplicable,
    ))
}

pub fn usage(phase: Phase) -> OperationUse {
    match phase {
        Phase::Prefill => OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: PREFILL_TOKENS,
            graph: false,
        },
        Phase::Decode(capacity) => OperationUse {
            phase: ExecutionPhase::Decode,
            capacity,
            graph: true,
        },
        Phase::Verify { rows, .. } => OperationUse {
            phase: ExecutionPhase::Verify,
            capacity: rows,
            graph: true,
        },
    }
}

impl Prepared {
    /// Authenticates the component `bytes` of an instance of `shape` and
    /// prepares its uses on `target`.
    pub(crate) fn decode(
        bytes: &[u8],
        shape: Shape,
        target: &dyn crate::target::Target,
    ) -> Result<Self, String> {
        let document =
            kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?;
        let inventory = crate::precision::inventory()?;
        model_qwen38_contract::prepared::bind(&document, shape, &inventory)?;
        // Generated uses prepare through the backend, registered lookups and
        // experts their launchers; each prepares its ops once, here. The
        // n-gram forms do not read the up mix layout.
        let (backend, own) = (target.backend(true), target.own());
        // Layers share each implementation's recipe. A row stage serves
        // every capacity under one symbol, so its planned capacity is part
        // of the recipe. A generated recipe's op serves every use sharing
        // it (a decode and a verify use of one bucket), so it is prepared
        // at the strongest of their needs.
        let key = |binding: &BoundUse| {
            (
                binding.implementation.implementation.clone(),
                binding.implementation.stages.clone(),
                binding.usage.capacity,
            )
        };
        let mut needs = BTreeMap::new();
        for selection in &document.operations {
            for binding in &selection.bindings {
                if registered::registered(selection, binding) {
                    continue;
                }
                let mut need = projection_need(binding.usage);
                if let Some(&held) = needs.get(&key(binding)) {
                    need = crate::ops::strongest(held, need)
                        .map_err(|r| format!("operation {}: {r:?}", selection.source.id))?;
                }
                needs.insert(key(binding), need);
            }
        }
        let mut provider = Provider::default();
        let mut indices = BTreeMap::new();
        let (mut dense_symbols, mut prefixes) = (BTreeMap::new(), Vec::new());
        for selection in &document.operations {
            let id = &selection.source.id;
            let mut uses = Vec::new();
            for binding in &selection.bindings {
                let usage = binding.usage;
                if let Some(&index) = indices.get(&key(binding)) {
                    uses.push((usage, index));
                    continue;
                }
                let first = &binding.implementation.stages[0];
                let recipe = if registered::registered(selection, binding) {
                    if selection.effective.kind == OperationKind::Dense {
                        dense_symbols
                            .entry(first.clone())
                            .or_insert_with(|| id.clone());
                    }
                    shared::Recipe::Registered
                } else {
                    let prepared = backend
                        .projection(&selection.effective, usage, needs[&key(binding)])
                        .map_err(|refused| format!("operation {id}: {refused:?}"))?;
                    if prepared.identity != binding.implementation {
                        return Err(format!(
                            "operation {id}: the prepared projection's identity differs from \
                             the authenticated binding"
                        ));
                    }
                    let op: shared::PreparedProjection = Arc::from(prepared.handle);
                    // The bound stages are a registered prefix (the n-gram
                    // hash), then the op's own.
                    let stages = &binding.implementation.stages;
                    let prefix = stages
                        .len()
                        .checked_sub(op.info().symbols.len())
                        .map(|at| stages[..at].to_vec())
                        .ok_or_else(|| {
                            format!("operation {id}: more launched than bound stages")
                        })?;
                    // Registered prefixes take no dynamic shared memory.
                    prefixes.extend(prefix.iter().cloned());
                    shared::Recipe::Prepared { op, prefix }
                };
                indices.insert(key(binding), provider.recipes.len());
                uses.push((usage, provider.recipes.len()));
                provider
                    .recipes
                    .push((recipe, binding.implementation.clone()));
            }
            provider.uses.insert(id.clone(), uses);
        }
        let mut guarantees = std::collections::HashSet::new();
        for uses in provider.uses.values() {
            for &(usage, index) in uses {
                if let shared::Recipe::Prepared { op, .. } = &provider.recipes[index].0 {
                    let info = op.info();
                    for (symbol, _) in &info.symbols {
                        guarantees.insert((usage, symbol.clone(), info.guarantees));
                    }
                }
            }
        }
        let registry = shared::Registry::for_document(&document, |selection, binding| {
            registered::launcher(
                shape,
                target,
                backend.as_ref(),
                own,
                &document,
                selection,
                binding,
            )
        })?;
        let authenticated = shared::Authenticated::decoded(document, &provider, &registry)?;
        let (mut shared_memory, needs) = authenticated.shared_requirements();
        shared_memory.extend(prefixes.into_iter().map(|symbol| (symbol, 0)));
        Ok(Self {
            authenticated,
            stores: vec![],
            dense_symbols,
            shared_memory,
            workspace_bytes: needs.workspace_bytes,
            temporary_bytes: needs.temporary_bytes,
            norms: Default::default(),
            guarantees: guarantees.into_iter().collect(),
        })
    }

    /// Record the norm weight dtypes of each rank's validated views, in rank
    /// order. Norm weights are replicated, so every rank stores rank 0's.
    pub fn bind_norms(&mut self, index: &crate::views::RankIndex) -> Result<(), String> {
        let norms = model_qwen38_contract::norms::NormWeights::from_index(index);
        if index.rank == 0 {
            self.norms = norms;
        } else if norms != self.norms {
            return Err(format!(
                "rank {} stores other norm weight dtypes than rank 0",
                index.rank
            ));
        }
        Ok(())
    }

    pub fn document(&self) -> &PrecisionDocument {
        self.authenticated.document()
    }

    /// Attach the next rank's persistent operand views.
    pub fn bind_rank(&mut self, views: &BTreeMap<String, Span>) -> Result<(), String> {
        let store = self.authenticated.bind(|selection| {
            let format = selection.effective.weights[0].format;
            shared::Weights::named(&selection.source.id, format, views).map(Some)
        })?;
        self.stores.push(store);
        Ok(())
    }

    pub fn operation(&self, rank: u32, id: &str, phase: Phase) -> Result<&shared::Handle, Refusal> {
        self.stores
            .get(rank as usize)
            .ok_or(Refusal("missing projection rank store"))?
            .handle(id, usage(phase))
    }
}
