//! Authenticated projection stores: dense uses prepared through the
//! target's projection backend (`Target::projection_backend`), and
//! model-owned launchers for the expert and lookup uses.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bundle_runtime::projection as shared;
use kernel_core::backend::Backend;
use kernel_core::contract::Refusal;
use kernel_core::precision::execution::{Binding, ImplementationProvider, PrecisionDocument};
use kernel_core::precision::{OperationSpec, OperationUse};
use model_mimo26_contract::prepared;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use crate::{
    pipeline::PreparedLaunch,
    selection::Phase,
    target::{Registered, Target},
    topology::Tp,
};

/// The text program's BF16 vocabulary head.
pub(crate) const BF16_HEAD: &str = "head.bf16";

/// What projection `id` bound at `usage` needs of its op: what its phase
/// needs (`ops::projection_need`), and a drafter's rows only, since its
/// steps propose tokens the target's verify rows check. An inexact
/// alternative (`.inexact`) serves the steps the exact verify setting
/// leaves inexact, so it needs what they need. The BF16 head's
/// decode buckets also project a prefill's few unscored logits rows
/// (`RankStage::prefill_head`), so they need each row's bits in any chunk
/// as well, which their catalog entries declare.
pub(crate) fn need(
    id: &str,
    usage: OperationUse,
    switches: crate::switches::Switches,
) -> Result<kernel_core::backend::Need, Refusal> {
    if drafter(id) {
        return Ok(kernel_core::backend::Need::rows(usage.capacity));
    }
    let switches = if id.ends_with(".inexact") {
        crate::switches::Switches {
            exact_verify: false,
            ..switches
        }
    } else {
        switches
    };
    let need = crate::ops::projection_need(usage, switches)?;
    Ok(
        if id == BF16_HEAD
            && usage.phase == kernel_core::precision::ExecutionPhase::Decode
            && usage.capacity <= crate::contract::DECODE_ENTRY_ROWS
        {
            need.with_exact(need.exact.with_chunk(kernel_core::properties::Chunk::Any))
        } else {
            need
        },
    )
}

/// Binds each use to the op the backend prepares for it, or to MiMo's
/// registered launcher where the backend prepares none.
struct Provider<'a> {
    target: &'a dyn Target,
    backend: Box<dyn Backend>,
    switches: crate::switches::Switches,
}
impl ImplementationProvider for Provider<'_> {
    type Handle = shared::Recipe;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String> {
        let bound = self.target.projection_rows(operation, usage)?;
        let Some(rows) = bound.handle else {
            return Ok(Binding {
                identity: bound.identity,
                handle: shared::Recipe::Registered,
            });
        };
        let need = kernel_core::backend::Need {
            capacity: rows,
            ..need(&operation.id, usage, self.switches).map_err(|e| e.to_string())?
        };
        let prepared = self
            .backend
            .projection(operation, usage, need)
            .map_err(|refused| format!("operation {}: {refused:?}", operation.id))?;
        Ok(Binding {
            identity: prepared.identity,
            handle: shared::Recipe::Prepared {
                op: Arc::from(prepared.handle),
                prefix: vec![],
            },
        })
    }
}

/// MiMo's launcher of an expert or lookup use, which the backend does not
/// prepare (`Target::registered_projection`): the expert projections or
/// embedding gathers prepared for its entry when it was authenticated.
enum Native {
    Experts {
        experts: crate::ops::Experts,
        workspace_bytes: u64,
    },
    Lookup {
        embeddings: crate::ops::Embeddings,
        switches: crate::switches::Switches,
    },
}
impl shared::Launcher for Native {
    fn needs(&self) -> shared::Needs {
        let workspace_bytes = match self {
            Self::Experts {
                workspace_bytes, ..
            } => *workspace_bytes,
            Self::Lookup { .. } => 0,
        };
        shared::Needs {
            workspace_bytes,
            ..Default::default()
        }
    }
    fn writes(
        &self,
        _: shared::Input<'_>,
        output: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        let mut writes = vec![output];
        let needs = self.needs();
        if needs.workspace_bytes > 0 {
            writes.push(scratch.partial);
        }
        if needs.semaphore_bytes > 0 {
            writes.push(scratch.semaphores);
        }
        Ok(writes)
    }
    fn launch(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Ok(crate::program_lower::launches(
            self.launch_ops(phase, input, output, scratch)?,
        ))
    }
    fn launch_with_persistence(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<(PreparedLaunch, Option<bool>)>, Refusal> {
        // Lookups and experts launch prepared ops, which state their
        // persistence.
        Ok(self
            .launch_ops(phase, input, output, scratch)?
            .into_iter()
            .map(|lowered| (lowered.launch, Some(lowered.facts.persistent)))
            .collect())
    }
}

impl Native {
    /// The prepared ops' launches of a lookup or an expert use.
    fn launch_ops(
        &self,
        phase: shared::Phase,
        input: shared::Input<'_>,
        output: Span,
        scratch: &shared::Scratch,
    ) -> Result<Vec<crate::program_lower::Lowered>, Refusal> {
        match self {
            Self::Lookup {
                embeddings,
                switches,
            } => Ok(vec![lookup(
                embeddings,
                *switches,
                crate::ops::phase_of(phase.usage)?,
                phase.rows,
                &input,
                output,
            )?]),
            Self::Experts { experts, .. } => self::experts(experts, phase, &input, output, scratch),
        }
    }
}

/// An expert projection of a `phase` step's token rows over its routes,
/// prepared for the use when it was authenticated.
fn experts(
    experts: &crate::ops::Experts,
    phase: shared::Phase,
    input: &shared::Input<'_>,
    output: Span,
    scratch: &shared::Scratch,
) -> Result<Vec<crate::program_lower::Lowered>, Refusal> {
    let route = input.routing.ok_or(Refusal("expert routing missing"))?;
    let side = |index: usize| {
        input
            .side
            .get(index)
            .copied()
            .ok_or(Refusal("expert weight, scale or unit scale missing"))
    };
    experts.launch(
        phase.usage,
        phase.rows,
        &crate::ops::ExpertViews {
            activation: input.activation,
            weight: side(0)?,
            weight_scales: side(2)?,
            output,
            route_weights: route.weights,
            offsets: route.offsets,
            sorted_rows: route.sorted_rows,
            padded_rows: route.padded_rows,
            active_experts: route.active_experts,
            live_tokens: input.live_rows,
            partials: scratch.partial,
            unused: side(1)?,
        },
    )
}

/// A lookup of `rows` token ids of a `phase` step: the embedding gather of
/// its bound table.
fn lookup(
    embeddings: &crate::ops::Embeddings,
    switches: crate::switches::Switches,
    phase: Phase,
    rows: u32,
    input: &shared::Input<'_>,
    output: Span,
) -> Result<crate::program_lower::Lowered, Refusal> {
    embeddings.tokens(
        phase,
        switches,
        rows,
        &crate::ops::EmbeddingViews {
            table: *input
                .side
                .first()
                .ok_or(Refusal("bound projection weight missing"))?,
            ids: input.activation,
            output,
            live_rows: input.live_rows,
        },
    )
}

pub struct Prepared {
    pub stores: Vec<shared::Store>,
    pub choices: Choices,
}

impl Prepared {
    /// Check that the component serves every port of the text program with
    /// complete use coverage and reads each port's declared checkpoint
    /// sources, then authenticate every binding once: the backend prepares
    /// each dense use with what `switches` need of its phase, and lookups
    /// grid their rows as `switches` say at `target`'s row buckets.
    pub fn authenticate(
        bytes: &[u8],
        tp: Tp,
        switches: crate::switches::Switches,
        target: &dyn Target,
    ) -> Result<shared::Authenticated, String> {
        let provider = Provider {
            target,
            backend: target.projection_backend(),
            switches,
        };
        let document =
            kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?;
        prepared::check_ports(&document, tp)?;
        let registry = shared::Registry::for_document(&document, |selection, binding| {
            // Launchers register by identity, which every use shares.
            let native = match target.registered_projection(&selection.effective, binding.usage)? {
                None => return Ok(None),
                Some(Registered::Experts {
                    backend,
                    op,
                    workspace_bytes,
                }) => {
                    // Every use the component binds to this implementation.
                    let mut uses = Vec::new();
                    for bound in document
                        .operations
                        .iter()
                        .flat_map(|selection| &selection.bindings)
                        .filter(|bound| bound.implementation == binding.implementation)
                    {
                        if !uses.iter().any(|&(usage, _)| usage == bound.usage) {
                            let need = need(&selection.effective.id, bound.usage, switches)
                                .map_err(|e| e.to_string())?;
                            uses.push((bound.usage, need));
                        }
                    }
                    Native::Experts {
                        experts: crate::ops::Experts::prepare(&*backend, &op, &uses).map_err(
                            |refused| format!("experts {}: {refused:?}", selection.effective.id),
                        )?,
                        workspace_bytes,
                    }
                }
                Some(Registered::Lookup { backend }) => Native::Lookup {
                    embeddings: crate::ops::Embeddings::prepare(
                        &*backend,
                        tp,
                        switches,
                        target.row_buckets(),
                    )
                    .map_err(|refused| format!("lookup {}: {refused:?}", selection.effective.id))?,
                    switches,
                },
            };
            Ok(Some(Arc::new(native)))
        })?;
        shared::Authenticated::decoded(document, &provider, &registry)
    }

    /// Attach every rank's resident views, and resolve the alternatives
    /// `switches` choose among `target`'s entries. Ranks without MTP or
    /// DFlash views serve only the text program.
    pub fn bind(
        authenticated: &shared::Authenticated,
        tp: Tp,
        switches: crate::switches::Switches,
        target: &dyn Target,
        ranks: &[BTreeMap<String, Span>],
    ) -> Result<Self, String> {
        let stores = ranks
            .iter()
            .map(|views| {
                let mtp = views.keys().any(|name| name.starts_with("model.mtp."));
                if mtp {
                    prepared::check_mtp_ports(authenticated.document(), tp)?;
                }
                // The DFlash drafter's context projection reads `fc.weight`.
                let dflash = views.contains_key("fc.weight");
                if dflash {
                    prepared::check_dflash_ports(authenticated.document(), tp)?;
                }
                rank_store(authenticated, tp, views, |id| {
                    if id.starts_with("dflash.") {
                        dflash
                    } else {
                        mtp || !drafter(id)
                    }
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            stores,
            choices: Choices::new(
                authenticated.document(),
                |id, usage| authenticated.locate(id, usage),
                switches,
                target,
            )?,
        })
    }

    /// Rank `rank`'s projections.
    pub fn rank(&self, rank: usize) -> Option<Rank<'_>> {
        Some(Rank {
            store: self.stores.get(rank)?,
            choices: &self.choices,
        })
    }
}

/// Whether projection `id` is an MTP or DFlash drafter's.
fn drafter(id: &str) -> bool {
    id.starts_with("mtp") || id.starts_with("dflash.")
}

/// One rank's store of the selections `serves` admits, over its views.
pub fn rank_store(
    authenticated: &shared::Authenticated,
    tp: Tp,
    views: &BTreeMap<String, Span>,
    serves: impl Fn(&str) -> bool,
) -> Result<shared::Store, String> {
    authenticated.bind(|selection| {
        if !serves(&selection.source.id) {
            return Ok(None);
        }
        let weights = &selection.effective.weights;
        let fused = weights.len() == 2;
        let view = |name: &str| {
            views
                .get(name)
                .copied()
                .ok_or_else(|| format!("missing bound resident view {name}"))
        };
        let get = |name: &str| view(&model_mimo26_contract::weight_plan::resident(name, fused));
        let assembled =
            model_mimo26_contract::weight_plan::assembled_resident(tp, &selection.source.id)
                .map_err(|e| e.to_string())?;
        Ok(Some(shared::Weights {
            codes: match assembled {
                Some(name) => view(&name)?,
                None => get(&weights[0].codes.tensor)?,
            },
            local_scales: weights[0]
                .local_scale
                .as_ref()
                .map(|s| get(&s.tensor.tensor))
                .transpose()?,
            global_scales: None,
            activation_global: None,
        }))
    })
}

/// The handle each declared use launches, located when the instance is
/// created from its switches: an expert use's inexact entry when exact
/// verify is off (`CHOPINFER_MIMO_VERIFY_EXACT=0`), and the alternative the
/// target launches over part of a use's live rows (`Target::row_alternative`)
/// at those rows. A step indexes the use's slot and the store; it searches
/// nothing.
pub struct Choices {
    /// Every selection's uses by slot ([`slot`]), by its id.
    ids: HashMap<String, [Option<Choice>; SLOTS]>,
}

/// The handle one use launches, and its row alternative's with the live
/// rows (first and last) it serves.
#[derive(Clone, Copy)]
struct Choice {
    handle: shared::UseIndex,
    alternative: Option<(u32, u32, shared::UseIndex)>,
}

/// Prefill, then the decode and the verify buckets of 1 to 256 rows.
const SLOTS: usize = 1 + 2 * 9;

/// A use's slot: prefill first, then each decode and each verify bucket by
/// its power of two.
fn slot(usage: OperationUse) -> Option<usize> {
    use kernel_core::precision::ExecutionPhase;
    let bucket = || {
        (usage.capacity.is_power_of_two() && usage.capacity <= 256)
            .then(|| usage.capacity.trailing_zeros() as usize)
    };
    match usage.phase {
        ExecutionPhase::Prefill => Some(0),
        ExecutionPhase::Decode => Some(1 + bucket()?),
        ExecutionPhase::Verify => Some(10 + bucket()?),
    }
}

impl Choices {
    /// The choices of `switches` among `document`'s selections, whose uses
    /// `locate` finds in the stores (`Authenticated::locate`). Every use
    /// `target` launches an alternative for over part of its rows needs
    /// that alternative at that use, and every expert use its inexact
    /// alternative when exact verify is off.
    pub fn new(
        document: &PrecisionDocument,
        locate: impl Fn(&str, OperationUse) -> Option<shared::UseIndex>,
        switches: crate::switches::Switches,
        target: &dyn Target,
    ) -> Result<Self, String> {
        let mut ids = HashMap::new();
        for selection in &document.operations {
            let id = &selection.source.id;
            let expert = id.starts_with("moe.experts.");
            let inexact = (expert && !id.ends_with(".inexact") && !switches.exact_verify)
                .then(|| format!("{id}.inexact"));
            let mut slots = [None; SLOTS];
            for binding in &selection.bindings {
                let usage = binding.usage;
                let at = slot(usage)
                    .ok_or_else(|| format!("operation {id}: use {usage:?} has no slot"))?;
                let located = |id: &str| {
                    locate(id, usage).ok_or_else(|| format!("missing {id} at {usage:?}"))
                };
                let handle = located(inexact.as_deref().unwrap_or(id))?;
                let alternative = match target.row_alternative(id, binding, switches)? {
                    Some((alternative, rows)) => {
                        let at_rows = locate(&alternative, usage)
                            .ok_or_else(|| format!("missing row alternative {alternative}"))?;
                        Some((*rows.start(), *rows.end(), at_rows))
                    }
                    None => None,
                };
                slots[at] = Some(Choice {
                    handle,
                    alternative,
                });
            }
            ids.insert(id.clone(), slots);
        }
        Ok(Self { ids })
    }

    /// The handle projection `id` launches at `usage` over `rows` live rows.
    fn resolve(
        &self,
        id: &str,
        usage: OperationUse,
        rows: u32,
    ) -> Result<shared::UseIndex, Refusal> {
        let choice = self
            .ids
            .get(id)
            .and_then(|slots| slots[slot(usage)?])
            .ok_or(Refusal("missing bound projection use"))?;
        Ok(match choice.alternative {
            Some((first, last, alternative)) if (first..=last).contains(&rows) => alternative,
            _ => choice.handle,
        })
    }
}

/// One rank's projections: its store and the instance's choices.
#[derive(Clone, Copy)]
pub struct Rank<'a> {
    pub store: &'a shared::Store,
    pub choices: &'a Choices,
}

pub struct Launch {
    pub phase: Phase,
    pub rows: u32,
    pub activation: Span,
    pub live_rows: Span,
    pub routing: Option<shared::Routing>,
    pub output: Span,
    pub partial: Span,
    pub semaphores: Span,
    /// A live 1.0 F32 constant that BF16 expert entries read as their
    /// activation scale.
    pub unit_scale: Span,
}

/// Launch projection `id`, or the alternative the instance chose for it,
/// each launch with the facts of the op that made it: dense ops and MiMo's
/// registered launchers of experts and lookups all state their
/// persistence.
pub fn launch(
    rank: Rank<'_>,
    id: &str,
    call: Launch,
) -> Result<Vec<crate::program_lower::Lowered>, Refusal> {
    let Launch {
        phase,
        rows,
        activation,
        live_rows,
        routing,
        output,
        partial,
        semaphores,
        unit_scale,
    } = call;
    let usage = prepared::usage(phase)?;
    let handle = rank.store.at(rank.choices.resolve(id, usage, rows)?)?;
    let weights = handle.weights();
    // Native adapters read the codes, the unit scale and, for quantized
    // weights, the local scales.
    let mut side = vec![weights.codes, unit_scale];
    side.extend(weights.local_scales);
    let launches = shared::launch_with_persistence(
        handle,
        shared::Phase {
            usage,
            rows,
            padded_rows: rows,
        },
        shared::Input {
            activation,
            live_rows,
            routing,
            residual: None,
            row_offset: 0,
            shard_rows: 0,
            side: &side,
            scalars: &[],
        },
        output,
        &shared::Scratch {
            partial,
            semaphores,
            // Registered entries read no unused argument; it may name any
            // live region and is never written.
            unused: unit_scale,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
        },
    )?;
    launches
        .into_iter()
        .map(|(launch, persistent)| {
            Ok(crate::program_lower::Lowered {
                launch,
                facts: crate::overlap::Facts {
                    persistent: persistent
                        .ok_or(Refusal("a MiMo projection launch states no persistence"))?,
                    phased_comm: false,
                    collective: false,
                },
            })
        })
        .collect()
}
