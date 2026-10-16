//! The projection API's stores (`bundle_runtime::projection`) over
//! the package's precision component (`model_glm5next_contract::prepared`):
//! authenticated once at create, every use bound to the op the backend
//! prepares over its shared tiled GEMM offer, and each rank's weights bound
//! by checkpoint name from its placed views. Each rank's scratch holds the
//! largest workspace and counters any use needs, allocated before the KV
//! pool.

use std::collections::BTreeMap;
use std::sync::Arc;

use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::projection::{
    Authenticated, Input, Launcher, Needs, Phase, Recipe, Registry, Scratch, Store, Weights,
};
use kernel_core::backend::Backend;
use kernel_core::contract::Refusal;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernel_core::precision::{OperationSpec, OperationUse};
use model_glm5next_contract::backend::{Kernels, Selected};
use model_glm5next_contract::direct::Stored;
use model_glm5next_contract::prepared;
use model_glm5next_contract::quantization::Layout;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use crate::ops::{Asks, exact};

pub use model_glm5next_contract::prepared::COMPONENT_ROLE;

/// Binds each use to the op the backend prepares for it over the offer the
/// compiler bound it to (with its copy into a strided output), asked
/// through `asks` for the exactness GLM relies on at the use's phase
/// (`exact::of_phase`: a prefill's chunk split, a verify bucket's one-row
/// bits, nothing of a decode bucket); a use bound to its part of a stacked
/// entry to that entry's registered stage (`StackedPart`), a GLM own op
/// asked where it is prepared (`ops::own`).
struct Provider<'a> {
    backend: &'a dyn Backend,
    asks: &'a Asks,
}

impl ImplementationProvider for Provider<'_> {
    type Handle = Recipe;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Recipe>, String> {
        let bound = prepared::bind(operation, usage)?;
        if Selected::own_launch(&bound.handle)?.is_some() {
            return Ok(Binding {
                identity: bound.identity,
                handle: Recipe::Registered,
            });
        }
        // Every use launches up to its capacity's rows.
        let need = Asks::asked_need(exact::need(
            usage.capacity,
            exact::of_phase(usage.phase, usage.capacity),
        ));
        let prepared = self
            .backend
            .projection(operation, usage, need)
            .map_err(|refused| format!("operation {}: {refused}", operation.id))?;
        let what = format!(
            "projection {} {:?} {}",
            operation.id, usage.phase, usage.capacity
        );
        self.asks.record(what, need, prepared.handle.info());
        let op = prepared.handle;
        #[cfg(test)]
        let op: kernel_core::backend::Prepared<_> = Box::new(crate::ops::Recorded(op));
        Ok(Binding {
            identity: prepared.identity,
            handle: Recipe::Prepared {
                op: Arc::from(op),
                prefix: Vec::new(),
            },
        })
    }
}

/// A KDA input projection's use bound to its part of a stacked entry
/// (`model_glm5next_contract::sm120::stacked`): the store holds its weights
/// and the entry's scratch needs, and the lowering launches the entry once
/// for all six parts (`Lowering::stacked_kda_inputs`); launching one part
/// alone is refused.
struct StackedPart {
    needs: Needs,
}

impl Launcher for StackedPart {
    fn needs(&self) -> Needs {
        self.needs
    }

    fn writes(&self, _: Input<'_>, _: Span, _: &Scratch) -> Result<Vec<Span>, Refusal> {
        Err(Refusal("a stacked part runs only in its stacked launch"))
    }

    fn launch(
        &self,
        _: Phase,
        _: Input<'_>,
        _: Span,
        _: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Err(Refusal("a stacked part runs only in its stacked launch"))
    }
}

/// The registered stages of `document`: each use bound to an entry that
/// runs in an own op's launch (`StackedPart`), with that entry's scratch
/// (`Kernels::own_launch`).
fn registry(
    document: &kernel_core::precision::execution::PrecisionDocument,
) -> Result<Registry, String> {
    Registry::for_document(document, |selection, binding| {
        let bound = prepared::bind(&selection.effective, binding.usage)?;
        let Some(own) = Selected::own_launch(&bound.handle)? else {
            return Ok(None);
        };
        let launcher: Arc<dyn Launcher> = Arc::new(StackedPart {
            needs: Needs {
                workspace_bytes: own.workspace_bytes,
                semaphore_bytes: own.semaphore_bytes,
                ..Needs::default()
            },
        });
        Ok(Some(launcher))
    })
}

/// Checks that the component, derived from an export stored in `layout`,
/// serves every lowered projection at every declared use from its declared
/// checkpoint tensors, then authenticates each binding against the op
/// `backend` prepares for it, asked through `asks`, and the stacked
/// entries' registered stages.
pub(crate) fn authenticate(
    bytes: &[u8],
    layout: &Layout,
    backend: &dyn Backend,
    asks: &Asks,
) -> Result<Authenticated, String> {
    let document = kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?;
    prepared::check(&document, layout)?;
    let registry = registry(&document)?;
    Authenticated::decoded(document, &Provider { backend, asks }, &registry)
}

/// One rank's store over its placed views, by checkpoint name: each
/// operation's codes and, for block FP8 and NVFP4, its local scales (and
/// NVFP4's tensor global), each stored as the
/// operation's offer reads it (`prepared::resident`, against the rank's
/// verified layout `stored`) and placed in a span that holds all of it.
pub fn bind(
    authenticated: &Authenticated,
    stored: &[Stored],
    views: &BTreeMap<String, Span>,
) -> Result<Store, String> {
    let stored: BTreeMap<&str, &Stored> = stored
        .iter()
        .map(|entry| (entry.view.name.as_str(), entry))
        .collect();
    authenticated.bind(|selection| {
        let resident = prepared::resident(selection, &stored)?;
        let view = |(name, bytes): &(String, u64)| {
            let span = views
                .get(name)
                .copied()
                .ok_or_else(|| format!("{}: no placed view {name}", selection.source.id))?;
            if span.length < *bytes {
                return Err(format!(
                    "{}: {name} is placed in {} bytes of the {bytes} it reads",
                    selection.source.id, span.length
                ));
            }
            Ok(span)
        };
        Ok(Some(Weights {
            codes: view(&resident.codes)?,
            local_scales: resident.scales.as_ref().map(view).transpose()?,
            global_scales: resident.global.as_ref().map(view).transpose()?,
            activation_global: None,
        }))
    })
}

/// One rank's projection scratch: the GEMM workspace, its counters (zeroed
/// once; each launch leaves them zero), the contiguous rows a copied use's
/// GEMM writes before its copy, and a placeholder for the operands a use
/// does not bind.
#[derive(Clone, Copy, Debug)]
pub struct RankScratch {
    pub workspace: Span,
    pub semaphores: Span,
    pub temporary: Span,
    pub unused: Span,
}

impl RankScratch {
    /// The scratch every use of `authenticated` fits, allocated through
    /// `allocate`.
    pub fn allocate<E>(
        authenticated: &Authenticated,
        mut allocate: impl FnMut(u64) -> Result<Span, E>,
    ) -> Result<Self, E> {
        let (_, needs) = authenticated.shared_requirements();
        Ok(Self {
            workspace: allocate(needs.workspace_bytes.max(256))?,
            semaphores: allocate(needs.semaphore_bytes.max(256))?,
            temporary: allocate(needs.temporary_bytes.max(256))?,
            unused: allocate(256)?,
        })
    }

    /// The counters, which the caller zeroes before the first launch.
    pub fn zeroed(&self) -> [Span; 1] {
        [self.semaphores]
    }

    /// The scratch a launch binds.
    pub fn scratch(&self) -> Scratch {
        Scratch {
            partial: self.workspace,
            semaphores: self.semaphores,
            unused: self.unused,
            temporary: Some(self.temporary),
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
        }
    }
}

/// Each authenticated operation's identity by its projection and execution
/// layer (`prepared::instance`).
pub type Instances = BTreeMap<(String, Option<u64>), String>;

/// The operation identities of `authenticated`'s document.
pub fn instances(authenticated: &Authenticated) -> Result<Instances, String> {
    authenticated
        .document()
        .operations
        .iter()
        .map(|selection| {
            let id = &selection.source.id;
            Ok((prepared::instance(id)?, id.clone()))
        })
        .collect()
}

/// The instance's authenticated component, its operations by projection
/// and layer, and each rank's store and scratch.
pub struct Projections {
    pub authenticated: Authenticated,
    pub instances: Instances,
    pub stores: Vec<Store>,
    pub scratch: Vec<RankScratch>,
    /// Each rank's scratch of the projection the decode and verify graphs
    /// run on the side stream (the shared expert's down projection,
    /// `crate::shared_overlap`), beside projections binding `scratch`.
    pub side_scratch: Vec<RankScratch>,
    /// What each use's op asked at create and was granted.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the invariance journal reads the asks")
    )]
    pub(crate) asked: Vec<crate::ops::Asked>,
}

/// One rank's projections as its lowering binds them.
#[derive(Clone, Copy)]
pub struct Rank<'a> {
    pub instances: &'a Instances,
    pub store: &'a Store,
    pub scratch: &'a RankScratch,
    pub side_scratch: &'a RankScratch,
}

impl Projections {
    pub fn rank(&self, rank: usize) -> Rank<'_> {
        Rank {
            instances: &self.instances,
            store: &self.stores[rank],
            scratch: &self.scratch[rank],
            side_scratch: &self.side_scratch[rank],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::quantization::FP8_BLOCK128;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    fn spans(layout: &[Stored]) -> BTreeMap<String, Span> {
        layout
            .iter()
            .map(|entry| {
                let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
                (
                    entry.view.name.clone(),
                    Span::new(handle, entry.offset, entry.bytes()),
                )
            })
            .collect()
    }

    /// The layout of the copy projection `id`'s BF16 use `usage` runs after
    /// its GEMM, which `handle`'s prepared op launches last: its width, then
    /// the source's row stride and the destination's row stride and first
    /// column.
    fn copied(
        handle: &bundle_runtime::projection::Handle,
        id: &str,
        usage: OperationUse,
    ) -> (u32, u32, u32, u32) {
        let copy = model_glm5next_contract::sm120::precision::copy(
            id,
            kernel_core::contract::WeightFormat::Bf16,
            usage,
        )
        .unwrap()
        .unwrap_or_else(|| panic!("{id} copies its rows"));
        let Recipe::Prepared { op, .. } = handle.recipe() else {
            panic!("{id} is prepared on the backend");
        };
        let (last, _) = op.info().symbols.last().unwrap();
        assert_eq!(*last, copy.stage().entry().symbol, "{id}");
        let (width, source, destination) = copy.layout();
        (
            width,
            source.row_stride,
            destination.row_stride,
            destination.column_offset,
        )
    }

    /// Each rank binds every operation to the views holding its codes and
    /// scales, the image storing the padded BF16 rows its offers read
    /// (linear.beta's [64, 4096] codes, not the rank's 16 rows). Stored
    /// unpadded, or placed in a span shorter than the stored view, they are
    /// refused. A KDA input projection's decode use binds its stacked
    /// entry's registered stage over the same weights; its prefill use keeps
    /// the GEMM and its copy.
    #[test]
    fn stores_bind_views_that_hold_what_their_offers_read() {
        let bytes = prepared::test_component();
        let engines = crate::sm120::engines::Engines::new(Shape::TP4).unwrap();
        let (backend, _) = crate::sm120::backend(&engines, Shape::TP4);
        let authenticated =
            authenticate(&bytes, &FP8_BLOCK128, backend.as_ref(), &Default::default()).unwrap();
        let decode = kernel_core::precision::OperationUse {
            phase: kernel_core::precision::ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        let prefill = model_glm5next_contract::precision::uses("linear.beta")
            .unwrap()
            .into_iter()
            .map(|(usage, _)| usage)
            .find(|usage| usage.phase == kernel_core::precision::ExecutionPhase::Prefill)
            .unwrap();
        let stacked = ["chop_gemm_stacked_glm5next_kda_inputs_m8_s4x1_pf_v1".to_owned()];
        let beta = "model.language_model.layers.0.self_attn.b_proj.weight";
        for rank in 0..Shape::TP4.ranks() {
            let layout = model_glm5next_contract::direct::layout(Shape::TP4, rank).unwrap();
            let views = spans(&layout);
            let store = bind(&authenticated, &layout, &views).unwrap();
            let handle = store.handle("linear.beta.0.bf16", decode).unwrap();
            let codes = handle.weights();
            assert_eq!(codes.codes, views[beta]);
            assert_eq!(codes.codes.length, 64 * 4096 * 2);
            assert!(matches!(handle.recipe(), Recipe::Registered));
            assert_eq!(handle.identity().stages, stacked);
            // In prefill its 64 padded rows compact to the rank's 16 heads.
            let handle = store.handle("linear.beta.0.bf16", prefill).unwrap();
            assert_eq!(handle.weights().codes, views[beta]);
            let (width, source, destination, _) = copied(handle, "linear.beta", prefill);
            assert_eq!((width, source, destination), (16, 64, 16));
            // The KDA value projection writes its contiguous rows to the
            // temporary, then copies them into the last third of the
            // convolution's input.
            assert_eq!(
                store
                    .handle("linear.v.0.bf16", decode)
                    .unwrap()
                    .identity()
                    .stages,
                stacked
            );
            let value = store.handle("linear.v.0.bf16", prefill).unwrap();
            let (width, _, stride, offset) = copied(value, "linear.v", prefill);
            assert_eq!((stride, offset), (3 * width, 2 * width));
            assert_eq!(
                value.needs().temporary_bytes,
                u64::from(prefill.capacity) * u64::from(width) * 2
            );
            let latent = store.handle("latent.q-b.3.bf16", decode).unwrap().weights();
            let stem = "model.language_model.layers.3.self_attn.q_b_proj";
            assert_eq!(latent.codes, views[&format!("{stem}.weight")]);
            assert_eq!(
                latent.local_scales,
                Some(views[&format!("{stem}.weight_scale_inv")])
            );
            // The rank's 16 rows of linear.beta alone.
            let mut unpadded = layout.clone();
            let entry = unpadded
                .iter_mut()
                .find(|entry| entry.view.name == beta)
                .unwrap();
            entry.view.shape = entry.pieces[0].shape.clone();
            let error = bind(&authenticated, &unpadded, &spans(&unpadded))
                .err()
                .unwrap();
            assert!(error.contains("[16, 4096] of the [64, 4096]"), "{error}");
            // A one-byte span of the query's codes, and of the latent
            // query's scales.
            for name in [
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.3.self_attn.q_b_proj.weight_scale_inv",
            ] {
                let mut short = views.clone();
                short.get_mut(name).unwrap().length = 1;
                let error = bind(&authenticated, &layout, &short).err().unwrap();
                assert!(error.contains("in 1 bytes"), "{error}");
            }
        }
    }

    /// A KDA input projection's stacked part (linear.beta's decode use)
    /// launches only in its stacked entry: launched alone, at one row or its
    /// capacity, with or without a residual, it is refused, while the same
    /// projection's prefill use launches its GEMM and copy.
    #[test]
    fn a_stacked_part_cannot_launch_alone() {
        use bundle_runtime::projection::{Phase, launch};
        use kernel_core::precision::ExecutionPhase;
        let bytes = prepared::test_component();
        let engines = crate::sm120::engines::Engines::new(Shape::TP4).unwrap();
        let (backend, _) = crate::sm120::backend(&engines, Shape::TP4);
        let authenticated =
            authenticate(&bytes, &FP8_BLOCK128, backend.as_ref(), &Default::default()).unwrap();
        let layout = model_glm5next_contract::direct::layout(Shape::TP4, 0).unwrap();
        let store = bind(&authenticated, &layout, &spans(&layout)).unwrap();
        let span = |buffer| {
            Span::new(
                BundleDeviceBufferHandleV1::from_parts(2, buffer),
                0,
                1 << 36,
            )
        };
        let mut next = 100;
        let scratch = RankScratch::allocate(&authenticated, |_| {
            next += 1;
            Ok::<_, ()>(span(next))
        })
        .unwrap();
        let run = |usage: OperationUse, rows: u32, residual: bool| {
            let handle = store.handle("linear.beta.0.bf16", usage).unwrap();
            let phase = Phase {
                usage,
                rows,
                padded_rows: rows,
            };
            let input = Input {
                activation: span(10),
                live_rows: span(11),
                routing: None,
                residual: residual.then(|| span(12)),
                row_offset: 0,
                shard_rows: 0,
                side: &[],
                scalars: &[],
            };
            launch(handle, phase, input, span(13), &scratch.scratch())
        };
        let decode = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        for rows in [1, decode.capacity] {
            for residual in [false, true] {
                let Err(refusal) = run(decode, rows, residual) else {
                    panic!("a stacked part launched alone at {rows} rows");
                };
                assert_eq!(
                    refusal.0, "a stacked part runs only in its stacked launch",
                    "{rows} {residual}"
                );
            }
        }
        let prefill = model_glm5next_contract::precision::uses("linear.beta")
            .unwrap()
            .into_iter()
            .map(|(usage, _)| usage)
            .find(|usage| usage.phase == ExecutionPhase::Prefill)
            .unwrap();
        for rows in [1, prefill.capacity] {
            assert!(!run(prefill, rows, false).unwrap().is_empty(), "{rows}");
        }
    }

    /// A backend whose projections keep no claims, as one without their
    /// declarations: each refuses exactness and otherwise prepares on the
    /// backend it wraps.
    struct Undeclared<'a>(&'a dyn Backend);

    impl Backend for Undeclared<'_> {
        fn name(&self) -> &str {
            "undeclared"
        }

        fn projection(
            &self,
            op: &OperationSpec,
            usage: OperationUse,
            need: kernel_core::backend::Need,
        ) -> Result<
            Binding<
                kernel_core::backend::Prepared<
                    kernel_core::precision::buffers::ProjectionBuffers<kernel_core::backend::View>,
                >,
            >,
            kernel_core::backend::Refused,
        > {
            use kernel_core::properties::Invariance;
            if need.exact != Invariance::NONE {
                return Err(kernel_core::backend::Refused::Inexact {
                    needed: need.exact,
                    guaranteed: Invariance::NONE,
                });
            }
            self.0.projection(op, usage, need)
        }
    }

    /// Create asks every prefill and verify use for its phase's exactness,
    /// so projections that do not keep it are refused.
    #[test]
    fn projections_that_do_not_keep_their_asks_refuse_create() {
        let bytes = prepared::test_component();
        let engines = crate::sm120::engines::Engines::new(Shape::TP4).unwrap();
        let (backend, _) = crate::sm120::backend(&engines, Shape::TP4);
        let asks = Asks::default();
        authenticate(&bytes, &FP8_BLOCK128, backend.as_ref(), &asks).unwrap();
        let asked = asks.asked();
        let at = |phase: &str| {
            asked
                .iter()
                .filter(|asked| asked.what.contains(phase))
                .map(|asked| asked.need.exact)
                .collect::<Vec<_>>()
        };
        assert!(at(" Prefill ").iter().all(|need| *need == exact::PREFILL));
        assert!(
            at(" Verify ")
                .iter()
                .all(|need| need.rows != kernel_core::properties::Rows::None)
        );
        assert!(!at(" Prefill ").is_empty() && !at(" Verify ").is_empty());
        let undeclared = Undeclared(backend.as_ref());
        let refused = authenticate(&bytes, &FP8_BLOCK128, &undeclared, &Asks::default());
        let error = refused.err().expect("refused");
        assert!(error.contains("does not cover the needed"), "{error}");
    }
}
