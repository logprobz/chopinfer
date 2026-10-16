//! GLM's own kernels on SM120: [`OwnOps`] over the bundle's cubins.
//!
//! Each prepared op launches the entry the bundle builds for its shape:
//! the row cubin's own rows (`engines::ModelRow`, one 256-thread CTA per
//! row of the smallest row bucket holding the step's rows, with PDL), the
//! drafter input halves' exact layout copies (`precision::eh_halves`), the
//! stacked KDA input projections (`sm120::stacked`), commit's state copy
//! (`sm120::state_copy`) and the payload admission checks
//! (`sm120::payload`). Ops are refused where an entry's compiled constants
//! (widths, shard, ranks, draft rows, stacked parts) do not fit the
//! contract.
//!
//! Views carry no offset, so a host binds each view at the alignment its
//! role record states. Guarantees are the declaration of the symbol an op
//! launches: GLM's own kernels' ([`declared_kernels`]), or the kernel
//! base's and GLM's catalogs', none for a symbol without one.
//!
//! The lowering launches GLM's own rows through these ops
//! (`crate::ops::own`), except the stream expansion and the router's score
//! compaction, which keep their builders until the shared row copies
//! replace them. The tests compare each op's launches with the lowering's
//! at every phase and bucket GLM runs.

use std::sync::OnceLock;

use kernel_core::backend::{Counts, Launch, Need, Prepared, Refused, View};
use kernel_core::contract::{Argument, Refusal, Validate, WeightFormat};
use kernel_core::precision::OperationUse;
use kernel_core::properties::Invariance;
use kernels_sm120::backend::own::{self, OwnKernels, buffer, bytes, launch, rows, rule};
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::own_ops::draft::{DraftGather, DraftGatherBuffers};
use model_glm5next_contract::own_ops::rows::{
    EhInput, EhInputBuffers, GatherRows, GatherRowsBuffers, ResidualAdd, ResidualAddBuffers,
    RoundRows, RoundRowsBuffers, WidenRows, WidenRowsBuffers,
};
use model_glm5next_contract::own_ops::stacked::{StackedProjection, StackedProjectionBuffers};
use model_glm5next_contract::own_ops::state::{
    Admission, AdmissionBuffers, StateCopy, StateCopyBuffers,
};
use model_glm5next_contract::payload::{self, Check};
use model_glm5next_contract::sm120::engines::ModelRow;
use model_glm5next_contract::sm120::payload as admit;
use model_glm5next_contract::sm120::stacked;
use model_glm5next_contract::sm120::state_copy;

/// GLM's own kernels as the bundle compiles them for one shape.
#[derive(Clone, Copy, Debug)]
pub struct Sm120Own {
    shape: Shape,
}

impl Sm120Own {
    pub fn new(shape: Shape) -> Sm120Own {
        Sm120Own { shape }
    }

    /// The shape the kernels are compiled for.
    pub fn shape(&self) -> Shape {
        self.shape
    }
}

/// GLM's own kernels: their implementation names, the bundle's cubins and
/// the declarations of what they keep.
const OWN: OwnKernels = OwnKernels {
    prefix: "sm120.glm",
    embedded,
    declared,
    check: own::listed,
};

/// Whether the bundle's cubins hold `symbol`.
fn embedded(symbol: &str) -> bool {
    crate::cubin_of(symbol).is_some()
}

/// The declarations of GLM's own kernels that the target's prefill and
/// verify launch, which kernel-lab's own-op cases check through the
/// conformance kit (`tests/glm5next_bundle_kernels.rs`):
/// - the rounding, widening and row gather rows run one CTA per row of the
///   smallest row bucket holding the step's rows; none reduces across rows, uses atomics in its
///   arithmetic or splits by row count, so each row's bits are its own at
///   every bucket up to the largest, 8192 rows: one-row bits, any chunk
///   split and row independence;
/// - each stacked KDA input projection sums every output element's 64-K
///   chunks in K order in one split, with the tokens as MMA columns that
///   never mix: one-row bits up to its bucket's rows. Decode and verify
///   share each bucket's entry, and prefill never runs it, so it claims no
///   chunk split.
///
/// The drafter's own rows only choose drafts and commit's state copy runs
/// outside the step; neither is declared.
pub fn declared_kernels() -> Vec<kernels_sm120::invariance::Declared> {
    use model_glm5next_contract::contract::PREFILL_TOKENS;
    let rows = [ModelRow::Round, ModelRow::Widen, ModelRow::GatherRows]
        .into_iter()
        .map(|row| (row.symbol(), Invariance::any_rows(PREFILL_TOKENS)));
    let stacked = stacked::entries(Shape::TP4)
        .expect("the stacked KDA input entries")
        .into_iter()
        .map(|entry| (entry.symbol.clone(), Invariance::one_row(entry.capacity)));
    rows.chain(stacked)
        .map(|(symbol, invariance)| kernels_sm120::invariance::Declared {
            symbol,
            profile: None,
            op: kernels_sm120::invariance::OpKind::Bundle,
            invariance,
        })
        .collect()
}

/// The invariance GLM's own kernels ([`declared_kernels`]) or the kernel
/// base's or GLM's catalog declare for `symbol`, none when it has no
/// declaration: a declaration covers its own symbol only.
pub(crate) fn declared(symbol: &str) -> Invariance {
    static DECLARED: OnceLock<Vec<kernels_sm120::invariance::Declared>> = OnceLock::new();
    let declared = DECLARED.get_or_init(|| {
        let catalogs = kernels_sm120::Catalogs::new(&[&model_glm5next_contract::sm120::CATALOG])
            .expect("GLM's catalogs are consistent");
        let mut declared = declared_kernels();
        declared.extend(kernels_sm120::invariance::declarations(&catalogs));
        declared
    });
    own::declared_in(declared, symbol)
}

/// The row bucket a row op of `rows` rows grids, as the lowering plans it
/// (`kernels_sm120::row::smallest_bucket`).
fn bucket(rows: u32) -> Result<u32, Refusal> {
    kernels_sm120::row::smallest_bucket(rows).ok_or(Refusal("no row bucket holds the rows"))
}

/// Prepares the op launching `symbols` (each with the most dynamic shared
/// memory its launches plan), with the guarantees their declarations all
/// claim and `workspace` bytes of backend workspace, after checking the
/// need.
fn prepare<B: 'static>(
    name: &str,
    symbols: Vec<(String, u32)>,
    need: Need,
    workspace: u64,
    planner: impl Fn(&B, View, Counts) -> Result<Vec<Launch>, Refusal> + Send + Sync + 'static,
) -> Result<Prepared<B>, Refused> {
    let mut info = OWN.info(name, symbols, need)?;
    info.workspace = workspace;
    OWN.prepare(info, need, planner)
}

impl Sm120Own {
    /// Prepares own row `row`: one launch over the smallest row bucket
    /// holding the step's rows, which it records. `arguments` gives the
    /// entry's arguments in parameter order over `Counts::rows` rows and
    /// the bucket, after checking the views cover them.
    fn row<B: 'static>(
        &self,
        name: &str,
        row: ModelRow,
        need: Need,
        arguments: impl Fn(&B, u32, u32) -> Result<Vec<Argument>, Refusal> + Send + Sync + 'static,
    ) -> Result<Prepared<B>, Refused> {
        self.row_over(name, row, need, false, arguments)
    }

    /// [`Self::row`]; with `padded`, a launch whose `Counts::padded_rows`
    /// is nonzero grids and records the smallest bucket holding those
    /// rows, at most the capacity, while `arguments` sees `Counts::rows`,
    /// zero to the padded rows.
    fn row_over<B: 'static>(
        &self,
        name: &str,
        row: ModelRow,
        need: Need,
        padded: bool,
        arguments: impl Fn(&B, u32, u32) -> Result<Vec<Argument>, Refusal> + Send + Sync + 'static,
    ) -> Result<Prepared<B>, Refused> {
        bucket(need.capacity)?;
        let symbol = row.symbol();
        let capacity = need.capacity;
        prepare(
            name,
            vec![(symbol.clone(), 0)],
            need,
            0,
            move |b: &B, _, counts| {
                let (rows, grid) = if padded && counts.padded_rows > 0 {
                    if counts.padded_rows > capacity || counts.rows > counts.padded_rows {
                        return Err(Refusal(
                            "a gather grids up to its capacity and gathers at most those rows",
                        ));
                    }
                    (counts.rows, counts.padded_rows)
                } else {
                    let rows = rows(counts, capacity)?;
                    (rows, rows)
                };
                let bucket = bucket(grid)?;
                let plan = row.plan(bucket).map_err(|_| Refusal("no own row plan"))?;
                Ok(vec![launch(
                    &symbol,
                    bucket,
                    plan,
                    arguments(b, rows, bucket)?,
                )])
            },
        )
    }

    /// The hidden rows' bytes.
    fn hidden_bytes(&self) -> u64 {
        u64::from(self.shape.profile.hidden) * 2
    }
}

impl OwnOps for Sm120Own {
    fn name(&self) -> &str {
        "sm120"
    }

    fn round_rows(
        &self,
        op: &RoundRows,
        need: Need,
    ) -> Result<Prepared<RoundRowsBuffers<View>>, Refused> {
        op.validate()?;
        if *op != RoundRows::of(self.shape) {
            return Err(rule("the round row is compiled for the hidden rows"));
        }
        let row = self.hidden_bytes();
        self.row(
            "round_rows",
            ModelRow::Round,
            need,
            move |b: &RoundRowsBuffers<View>, rows, _| {
                b.input.covers(bytes(rows, 2 * row))?;
                b.output.covers(bytes(rows, row))?;
                b.live_rows.covers(4)?;
                Ok(vec![buffer(b.input), buffer(b.output), buffer(b.live_rows)])
            },
        )
    }

    fn widen_rows(
        &self,
        op: &WidenRows,
        need: Need,
    ) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
        op.validate()?;
        if *op != WidenRows::of(self.shape) {
            return Err(rule("the widen row is compiled for the hidden rows"));
        }
        let row = self.hidden_bytes();
        self.row(
            "widen_rows",
            ModelRow::Widen,
            need,
            move |b: &WidenRowsBuffers<View>, rows, _| {
                b.input.covers(bytes(rows, row))?;
                b.output.covers(bytes(rows, 2 * row))?;
                b.live_rows.covers(4)?;
                Ok(vec![buffer(b.input), buffer(b.output), buffer(b.live_rows)])
            },
        )
    }

    fn residual_add(
        &self,
        op: &ResidualAdd,
        need: Need,
    ) -> Result<Prepared<ResidualAddBuffers<View>>, Refused> {
        op.validate()?;
        if *op != ResidualAdd::of(self.shape) {
            return Err(rule("the residual add is compiled for the hidden rows"));
        }
        let row = self.hidden_bytes();
        self.row(
            "residual_add",
            ModelRow::ResidualAdd,
            need,
            move |b: &ResidualAddBuffers<View>, rows, _| {
                b.x.covers(bytes(rows, row))?;
                b.y.covers(bytes(rows, row))?;
                b.live_rows.covers(4)?;
                Ok(vec![buffer(b.x), buffer(b.y), buffer(b.live_rows)])
            },
        )
    }

    fn gather_rows(
        &self,
        op: &GatherRows,
        need: Need,
    ) -> Result<Prepared<GatherRowsBuffers<View>>, Refused> {
        op.validate()?;
        if *op != GatherRows::of(self.shape) {
            return Err(rule("the row gather is compiled for the hidden rows"));
        }
        let row = self.hidden_bytes();
        // The indices and output hold the gathered rows alone: a prefill
        // supplies one index per scored row (`InputLayout::prefill_within`).
        self.row_over(
            "gather_rows",
            ModelRow::GatherRows,
            need,
            true,
            move |b: &GatherRowsBuffers<View>, rows, _| {
                b.input.covers(row)?;
                b.row_indices.covers(bytes(rows, 4))?;
                b.count.covers(4)?;
                b.output.covers(bytes(rows, row))?;
                Ok(vec![
                    buffer(b.input),
                    buffer(b.row_indices),
                    buffer(b.count),
                    buffer(b.output),
                ])
            },
        )
    }

    fn eh_input(
        &self,
        op: &EhInput,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<EhInputBuffers<View>>, Refused> {
        op.validate()?;
        if *op != EhInput::of(self.shape) {
            return Err(rule(
                "the drafter's input halves are compiled for the hidden rows",
            ));
        }
        let halves = model_glm5next_contract::sm120::precision::eh_halves(usage)
            .map_err(|_| rule("no drafter input halves at this use"))?;
        if need.capacity > halves[0].stage().capacity() {
            return Err(rule(
                "the drafter's input halves run at most their compiled capacity",
            ));
        }
        let symbols = halves
            .iter()
            .map(|half| (half.stage().entry().symbol.clone(), 0))
            .collect::<Vec<_>>();
        let (row, capacity) = (self.hidden_bytes(), need.capacity);
        prepare(
            "eh_input",
            symbols,
            need,
            0,
            move |b: &EhInputBuffers<View>, _, counts| {
                let rows = rows(counts, capacity)?;
                b.embed.covers(bytes(rows, row))?;
                b.hidden.covers(bytes(rows, row))?;
                b.output.covers(bytes(rows, 2 * row))?;
                b.live_rows.covers(4)?;
                halves
                    .iter()
                    .zip([b.embed, b.hidden])
                    .map(|(half, input)| {
                        let stage = half.stage();
                        Ok(Launch {
                            symbol: stage.entry().symbol.clone(),
                            rows,
                            plan: stage.plan(rows)?,
                            arguments: half.arguments(
                                input.handle,
                                b.output.handle,
                                b.live_rows.handle,
                            )?,
                            collective: false,
                            persistent: false,
                        })
                    })
                    .collect()
            },
        )
    }

    fn draft_gather(
        &self,
        op: &DraftGather,
        need: Need,
    ) -> Result<Prepared<DraftGatherBuffers<View>>, Refused> {
        op.validate()?;
        if *op != DraftGather::of(self.shape) {
            return Err(rule("the draft gather is compiled for the hidden rows"));
        }
        let row = self.hidden_bytes();
        self.row(
            "draft_gather",
            ModelRow::DraftGather,
            need,
            move |b: &DraftGatherBuffers<View>, rows, _| {
                b.retained.covers(row)?;
                b.last.covers(row)?;
                b.codes.covers(bytes(rows, 8))?;
                b.drafts.covers(4)?;
                b.live_rows.covers(4)?;
                b.output.covers(bytes(rows, row))?;
                b.tokens.covers(bytes(rows, 4))?;
                Ok(vec![
                    buffer(b.retained),
                    buffer(b.last),
                    buffer(b.codes),
                    buffer(b.drafts),
                    buffer(b.live_rows),
                    buffer(b.output),
                    buffer(b.tokens),
                ])
            },
        )
    }

    fn stacked_projection(
        &self,
        op: &StackedProjection,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<StackedProjectionBuffers<View>>, Refused> {
        op.validate()?;
        let entry = stacked::part(stacked::PROJECTIONS[0], WeightFormat::Bf16, usage)
            .map_err(|_| rule("no stacked KDA input projections at this use"))?
            .ok_or(rule("no stacked KDA input projections at this use"))?
            .entry;
        let parts = entry
            .parts
            .iter()
            .map(|p| (p.rows, p.width, p.row_stride))
            .collect::<Vec<_>>();
        if op.k != entry.k
            || op
                .parts
                .iter()
                .map(|p| (p.rows, p.width, p.row_stride))
                .collect::<Vec<_>>()
                != parts
        {
            return Err(rule(
                "the stacked entry is compiled for the KDA layer's projections",
            ));
        }
        if need.capacity > entry.capacity {
            return Err(rule("a stacked entry runs at most its bucket's rows"));
        }
        let largest = entry.plan(entry.capacity)?;
        let symbol = entry.symbol.clone();
        let parts = op.parts.clone();
        let (k, capacity) = (u64::from(op.k), need.capacity);
        prepare(
            "stacked_projection",
            vec![(symbol.clone(), largest.dynamic_shared_bytes)],
            need,
            largest.workspace_bytes,
            move |b: &StackedProjectionBuffers<View>, workspace, counts| {
                let rows = rows(counts, capacity)?;
                if b.weights.len() != parts.len() || b.outputs.len() != parts.len() {
                    return Err(Refusal(
                        "stacked projections take a weight and output per part",
                    ));
                }
                // The tensor maps cover the bucket's rows of the activation
                // and each weight's rows; each output is written at its
                // row stride through the live rows.
                b.x.covers(bytes(entry.capacity, k * 2))?;
                for ((part, weight), output) in parts.iter().zip(&b.weights).zip(&b.outputs) {
                    weight.covers(u64::from(part.rows) * k * 2)?;
                    output.covers(part.span(entry.capacity) * 2)?;
                }
                b.tickets.covers(entry.semaphore_bytes())?;
                let plan = entry.plan(rows)?;
                let handles = |views: &[View]| views.iter().map(|v| v.handle).collect::<Vec<_>>();
                let arguments = entry.arguments(
                    rows,
                    &handles(&b.weights),
                    b.x.handle,
                    &handles(&b.outputs),
                    workspace.handle,
                    b.tickets.handle,
                )?;
                Ok(vec![Launch {
                    symbol: symbol.clone(),
                    rows,
                    plan,
                    arguments,
                    collective: false,
                    persistent: false,
                }])
            },
        )
    }

    fn state_copy(
        &self,
        op: &StateCopy,
        need: Need,
    ) -> Result<Prepared<StateCopyBuffers<View>>, Refused> {
        op.validate()?;
        let copy = state_copy::LayerCopy {
            bytes: op.bytes,
            layers: op.layers,
            source_stride: op.source_stride,
            destination_stride: op.destination_stride,
        };
        let plan = copy.plan()?;
        let symbol = state_copy::SYMBOL.to_owned();
        prepare(
            "state_copy",
            vec![(symbol.clone(), 0)],
            need,
            0,
            move |b: &StateCopyBuffers<View>, _, _| {
                b.source.covers(copy.source_span())?;
                b.destination.covers(copy.destination_span())?;
                Ok(vec![launch(
                    &symbol,
                    1,
                    plan,
                    vec![
                        buffer(b.source),
                        buffer(b.destination),
                        Argument::U64(copy.source_stride),
                        Argument::U64(copy.destination_stride),
                        Argument::U64(copy.bytes),
                    ],
                )])
            },
        )
    }

    fn admission(
        &self,
        op: &Admission,
        need: Need,
    ) -> Result<Prepared<AdmissionBuffers<View>>, Refused> {
        op.validate()?;
        if op.count >= u64::from(payload::NONE) {
            return Err(rule("an admission's report indexes fewer elements"));
        }
        let plan = admit::plan(op.count)?;
        let (rule_of, count) = (op.rule, op.count);
        let (codes, scales) = op.bytes();
        let symbol = admit::symbol(op.rule).to_owned();
        prepare(
            "admission",
            vec![(symbol.clone(), 0)],
            need,
            0,
            move |b: &AdmissionBuffers<View>, _, _| {
                b.codes.covers(codes)?;
                b.first.covers(4)?;
                let mut arguments = vec![buffer(b.codes)];
                match (rule_of, b.scales) {
                    (Check::Mxfp8, Some(view)) => {
                        view.covers(scales)?;
                        arguments.push(buffer(view));
                    }
                    (Check::Mxfp8, None) => {
                        return Err(Refusal("an MXFP8 admission reads its scales"));
                    }
                    (_, Some(_)) => return Err(Refusal("only an MXFP8 admission reads scales")),
                    (_, None) => {}
                }
                arguments.extend([Argument::U64(count), buffer(b.first)]);
                Ok(vec![launch(&symbol, 1, plan, arguments)])
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    use bundle_runtime::dispatch::{self, BufferBindings};
    use bundle_runtime::fake_host::FakeDevice;
    use bundle_runtime::geometry::Items;
    use bundle_runtime::kv::KvFormat;
    use bundle_runtime::pipeline::PreparedLaunch;
    use bundle_runtime::state::Extent;
    use bundle_runtime::upload::{BlockTableMirror, DeviceInputs, InputLayout, PhaseArena};
    use kernel_core::precision::ExecutionPhase;
    use model_glm5next_contract::contract::{
        DRAFT_ITEMS, Glm, MAX_VERIFY_ROWS, MTP_DRAFTS, PREFILL_TOKENS, has_mtp_layer,
    };
    use model_glm5next_contract::own_ops::stacked::StackedPart;
    use runtime_bundle_abi::{
        BundleDeviceBufferHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1, KernelArgumentV1,
        LaunchConfigV1,
    };

    use super::*;
    use crate::buffers::{
        Buffer, CommState, DrafterBuffers, RankBuffers, StatePools, allocate_scratch,
    };
    use crate::journal_tests::head_rows;
    use crate::lower::Lowering;
    use crate::program::{DraftPass, Phase, Stage, Sublayer, draft_rows};
    use crate::sm120::engines::Engines;
    use kernel_core::contract::ArgumentList;

    const SHAPE: Shape = Shape::TP4;

    /// The view a span binds as: its buffer and offset name the handle.
    fn view(span: DeviceBufferSpanV1) -> View {
        View {
            handle: u64::from(span.buffer.slot()) << 40 | span.offset,
            bytes: span.length,
        }
    }

    /// `length` bytes of test buffer `id` from `offset`.
    fn part(id: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, id),
            offset,
            length,
        )
    }

    /// A launch as the host dispatches it: the symbol, the rows it records,
    /// its configuration and its arguments with every buffer resolved to
    /// its whole span (handle with generation, offset and length) and every
    /// tensor map to its base span (`dispatch::launch_config`,
    /// `dispatch::arguments`).
    #[derive(Debug, PartialEq)]
    struct Dispatched {
        symbol: String,
        rows: u32,
        config: LaunchConfigV1,
        arguments: Vec<KernelArgumentV1>,
    }

    /// A lowering's launch as the host dispatches it.
    fn built(launch: PreparedLaunch) -> Dispatched {
        Dispatched {
            config: dispatch::launch_config(launch.plan).unwrap(),
            arguments: dispatch::arguments(&launch.arguments, &launch.bindings).unwrap(),
            symbol: launch.symbol,
            rows: launch.rows,
        }
    }

    /// Views bound to spans: each view's handle is a binding of its span,
    /// so a prepared op's launches resolve to what a host dispatches over
    /// the same spans.
    #[derive(Default)]
    struct Bound(RefCell<BufferBindings>);

    impl Bound {
        fn view(&self, span: DeviceBufferSpanV1) -> View {
            View {
                handle: self.0.borrow_mut().bind(span),
                bytes: span.length,
            }
        }

        fn dispatched(&self, launches: Vec<Launch>) -> Vec<Dispatched> {
            let bindings = self.0.borrow();
            launches
                .into_iter()
                .map(|launch| Dispatched {
                    config: dispatch::launch_config(launch.plan).unwrap(),
                    arguments: dispatch::arguments(&launch.arguments, &bindings).unwrap(),
                    symbol: launch.symbol,
                    rows: launch.rows,
                })
                .collect()
        }

        /// The one launch of `prepared` over `buffers`, dispatched.
        fn one<B>(&self, prepared: &Prepared<B>, buffers: &B, counts: Counts) -> Dispatched {
            let launches = prepared.launches(buffers, View::EMPTY, counts).unwrap();
            assert_eq!(launches.len(), 1);
            self.dispatched(launches).remove(0)
        }
    }

    /// The lowering's launches of `stage`, dispatched.
    fn lowered(lowering: &Lowering<'_>, phase: Phase, stage: Stage) -> Vec<Dispatched> {
        lowered_with(lowering, phase, stage, false)
    }

    fn lowered_with(
        lowering: &Lowering<'_>,
        phase: Phase,
        stage: Stage,
        epilogue: bool,
    ) -> Vec<Dispatched> {
        lowering
            .lower_with(phase, stage, epilogue)
            .unwrap()
            .unwrap()
            .into_iter()
            .map(built)
            .collect()
    }

    /// Projection `id`'s use for `rows` rows of `phase`, as the lowering
    /// picks it (`lower/project.rs`, `usage`): its prefill use, or its
    /// smallest decode or verify bucket holding them.
    fn usage(id: &str, phase: Phase, rows: u32) -> OperationUse {
        let wanted = match phase {
            Phase::Prefill { .. } => ExecutionPhase::Prefill,
            Phase::Decode(_) => ExecutionPhase::Decode,
            Phase::Verify { .. } => ExecutionPhase::Verify,
        };
        model_glm5next_contract::precision::uses(id)
            .unwrap()
            .into_iter()
            .map(|(usage, _)| usage)
            .filter(|usage| usage.phase == wanted && usage.capacity >= rows)
            .min_by_key(|usage| usage.capacity)
            .unwrap()
    }

    /// Every verify phase an instance serving `items` runs (`execute.rs`):
    /// a graph step's token bucket with an item bucket at least the bucket
    /// its rows need and at most its rows, or a narrow instance's eager
    /// step of its live rows and live item bucket. As a superset: every row
    /// count up to the verify tokens with every item bucket up to the
    /// instance's capacity and the rows' bucket.
    fn verify_phases(items: Items) -> Vec<Phase> {
        let tokens = bundle_runtime::graphs::verify_tokens::<Glm>(items);
        (1..=tokens)
            .flat_map(|rows| {
                items
                    .buckets()
                    .filter(move |&bucket| bucket <= rows.next_power_of_two())
                    .map(move |items| Phase::Verify { rows, items })
            })
            .collect()
    }

    /// The prefill row counts of a configuration: every one from 1 to the
    /// capacity, or every row bucket's edges (each bucket, one less and one
    /// more, from the one-row bucket up) and an odd count.
    fn prefill_rows(full: bool) -> Vec<u32> {
        if full {
            return (1..=PREFILL_TOKENS).collect();
        }
        let mut rows = vec![7001];
        for bucket in [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ] {
            rows.extend([bucket - 1, bucket, bucket + 1]);
        }
        rows.retain(|&rows| (1..=PREFILL_TOKENS).contains(&rows));
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// Every phase of a configuration: its prefills, every decode bucket of
    /// its items and every verify phase.
    fn phases(items: Items, full: bool) -> Vec<Phase> {
        let mut phases = prefill_rows(full)
            .into_iter()
            .map(|rows| Phase::Prefill { rows })
            .collect::<Vec<_>>();
        phases.extend(items.buckets().map(Phase::Decode));
        phases.extend(verify_phases(items));
        phases
    }

    /// One instance configuration the own rows lower under: the KV format,
    /// the items it serves, the verify rows its drafts take, whether
    /// create left the low-latency and phased all-reduce entries on (off,
    /// the engine composition's round and widen rows reduce).
    #[derive(Clone, Copy, Debug)]
    struct Config {
        format: KvFormat,
        items: u32,
        verify_rows: u32,
        entries: bool,
    }

    /// The configurations: every KV format at the package's items with and
    /// without the reduce entries, and narrower instances (one, three and
    /// eight items) with three and five drafts.
    fn configs() -> Vec<Config> {
        let mut configs = Vec::new();
        for format in SHAPE.kv_formats() {
            for entries in [true, false] {
                configs.push(Config {
                    format,
                    items: model_glm5next_contract::contract::DECODE_BATCH,
                    verify_rows: MTP_DRAFTS + 1,
                    entries,
                });
            }
        }
        let format = SHAPE.kv_formats()[1];
        for (items, verify_rows) in [
            (1, MTP_DRAFTS + 1),
            (3, MAX_VERIFY_ROWS),
            (8, MTP_DRAFTS + 1),
        ] {
            configs.push(Config {
                format,
                items,
                verify_rows,
                entries: true,
            });
        }
        configs
    }

    /// Four ranks' buffers of a configuration, as create allocates them
    /// (`journal_tests::rank_buffers` at the configuration's extent), and
    /// the phase input arenas of its items.
    struct Fixture<'e> {
        config: Config,
        items: Items,
        engines: &'e Engines,
        ops: crate::ops::Ops,
        ranks: Vec<RankBuffers>,
        peers: Vec<CommState>,
    }

    impl<'e> Fixture<'e> {
        fn new(engines: &'e Engines, config: Config) -> Fixture<'e> {
            let device = FakeDevice::default();
            let items = Items::new(config.items).unwrap();
            let ranks = (0..SHAPE.ranks())
                .map(|rank| rank_buffers(&device, rank, config, items))
                .collect::<Vec<_>>();
            let peers = ranks.iter().map(|buffers| buffers.comm.clone()).collect();
            Fixture {
                config,
                items,
                engines,
                ops: crate::ops::Ops::for_tests(SHAPE, 1),
                ranks,
                peers,
            }
        }

        /// Rank `rank`'s lowering of `phase` over `inputs`, for the
        /// drafter's `draft` pass or the target.
        fn lowering<'a>(
            &'a self,
            inputs: &'a DeviceInputs,
            rank: u32,
            phase: Phase,
            draft: Option<DraftPass>,
            projections: Option<crate::projections::Rank<'a>>,
        ) -> Lowering<'a> {
            Lowering {
                shape: SHAPE,
                format: &self.config.format,
                buffers: &self.ranks[rank as usize],
                inputs,
                ops: &self.ops,
                rank,
                index_topk: SHAPE.profile.index_topk_max,
                pieces: &[],
                row_offset: 0,
                peers: &self.peers,
                head_rows: head_rows(phase),
                context_parallel: None,
                draft,
                verify_tokens: draft.map(|_| self.inputs(rank, verify_of(self.items)).token_ids),
                projections,
            }
        }

        /// Rank `rank`'s inputs of `phase` in its phase arena.
        fn inputs(&self, rank: u32, phase: Phase) -> DeviceInputs {
            let layout = match phase {
                Phase::Prefill { .. } => InputLayout::prefill_largest_within::<Glm>(self.items),
                Phase::Decode(_) => InputLayout::decode_within(self.items),
                Phase::Verify { .. } => InputLayout::verify_within::<Glm>(self.items),
            };
            let arena = |id: u32, kind: u32| {
                BundleDeviceBufferHandleV1::from_parts(2, 5000 + 10 * id + kind)
            };
            let kind = match phase {
                Phase::Prefill { .. } => 0,
                Phase::Decode(_) => 1,
                Phase::Verify { .. } => 2,
            };
            let tables = BlockTableMirror::<Glm>::device_bytes(SHAPE.model_limits(), self.items);
            DeviceInputs::within(
                PhaseArena {
                    arena: DeviceBufferSpanV1::new(arena(rank, kind), 0, layout.end),
                    block_tables: DeviceBufferSpanV1::new(arena(rank, kind + 3), 0, tables),
                },
                &layout,
            )
        }
    }

    /// Any verify phase of `items`: the drafter's argmax writes that arena's
    /// tokens.
    fn verify_of(items: Items) -> Phase {
        Phase::Verify {
            rows: 1,
            items: items.buckets().next().unwrap(),
        }
    }

    /// One rank's buffers in create's order (`journal_tests::rank_buffers`)
    /// at the configuration's extent and reduce entries.
    fn rank_buffers(device: &FakeDevice, rank: u32, config: Config, items: Items) -> RankBuffers {
        let format = &config.format;
        let handle = |id| BundleDeviceBufferHandleV1::from_parts(3, id);
        let views = model_glm5next_contract::direct::layout(SHAPE, rank)
            .unwrap()
            .into_iter()
            .map(|entry| {
                let span =
                    DeviceBufferSpanV1::new(handle(1000 + rank), entry.offset, entry.bytes());
                (entry.view.name, span)
            })
            .collect();
        let stream = BundleStreamHandleV1::from_parts(1, 3000 + rank);
        let mut allocate = |bytes| {
            let id = device.id();
            Ok::<_, ()>(DeviceBufferSpanV1::new(
                device.allocate(id, stream, bytes, 256),
                0,
                bytes,
            ))
        };
        let mtp = has_mtp_layer(format);
        let scratch = allocate_scratch(SHAPE, mtp, &mut allocate)
            .unwrap()
            .unwrap();
        let reduced = scratch[&Buffer::Reduced];
        let regions = crate::journal_tests::comm_regions();
        let draft = mtp.then(crate::journal_tests::draft_regions);
        let mut comm = CommState::allocate(SHAPE, draft, reduced, regions, &mut allocate)
            .unwrap()
            .unwrap();
        comm.low_latency &= config.entries;
        comm.phased &= config.entries;
        let tail = DeviceBufferSpanV1::new(
            handle(4000 + rank),
            0,
            u64::from(items.capacity()) * SHAPE.indexer_tail_slot_bytes(mtp),
        );
        let kv_pool = DeviceBufferSpanV1::new(handle(2000 + rank), 0, 1024 * format.page_bytes);
        let extent = Extent {
            served: config.items,
            bucket: items.capacity(),
            verify_rows: config.verify_rows,
        };
        let drafter = mtp.then(|| DrafterBuffers::allocate(SHAPE, &mut allocate).unwrap());
        let retained =
            StatePools::retained_bytes(SHAPE, format, extent).map(|bytes| allocate(bytes).unwrap());
        let state = StatePools::new(SHAPE, format, extent, retained, tail, &mut allocate)
            .unwrap()
            .unwrap();
        RankBuffers {
            scratch,
            comm,
            state,
            views,
            kv_pool,
            drafter,
        }
    }

    /// The rows a prefill of `rows` rows scores in the gather's domain (the
    /// host refuses more than `SCORED_ROWS`): every count from none to
    /// the most when `rows` is an edge of the prefill rows or at most
    /// `SCORED_ROWS` (the pair's whole joint domain), else the scored
    /// edges: none (a chunk before its prompt's end), one, two, a full
    /// batch's items and the most, each with its neighbours.
    fn scored_rows(rows: u32) -> Vec<u32> {
        use model_glm5next_contract::contract::{DECODE_BATCH, SCORED_ROWS};
        let most = rows.min(SCORED_ROWS);
        if rows <= SCORED_ROWS || prefill_rows(false).contains(&rows) {
            return (0..=most).collect();
        }
        let mut scored = vec![
            0,
            1,
            2,
            DECODE_BATCH - 1,
            DECODE_BATCH,
            DECODE_BATCH + 1,
            SCORED_ROWS - 1,
            SCORED_ROWS,
        ];
        scored.retain(|&scored| scored <= most);
        scored.sort_unstable();
        scored.dedup();
        scored
    }

    /// A prefill's gather of its `scored` rows of `rows` launches what the
    /// lowering launches over the inputs a step packs for them
    /// (`InputLayout::prefill_within`: one index per scored row): the
    /// prefill's grid, the scored rows gathered.
    fn check_prefill_gather(fixture: &Fixture<'_>, rank: u32, rows: u32, scored: u32) {
        let phase = Phase::Prefill { rows };
        let granule = model_glm5next_contract::contract::TOKEN_GRANULE;
        let layout = InputLayout::within(rows.div_ceil(granule) * granule, scored, fixture.items);
        let arena = BundleDeviceBufferHandleV1::from_parts(2, 5900 + rank);
        let tables = BlockTableMirror::<Glm>::device_bytes(SHAPE.model_limits(), fixture.items);
        let inputs = DeviceInputs::within(
            PhaseArena {
                arena: DeviceBufferSpanV1::new(arena, 0, layout.end),
                block_tables: DeviceBufferSpanV1::new(arena, layout.end, tables),
            },
            &layout,
        );
        // One index slot per scored row, one when none is.
        assert_eq!(inputs.row_indices.length, u64::from(scored.max(1)) * 4);
        let mut lowering = fixture.lowering(&inputs, rank, phase, None, None);
        lowering.head_rows = scored;
        let own = Sm120Own::new(SHAPE);
        let gather = own
            .gather_rows(&GatherRows::of(SHAPE), Need::rows(phase.capacity()))
            .unwrap();
        let bound = Bound::default();
        let buffers_of = GatherRowsBuffers {
            input: bound.view(lowering.scratch(Buffer::Normed)),
            row_indices: bound.view(inputs.row_indices),
            count: bound.view(inputs.logit_rows),
            output: bound.view(lowering.scratch(Buffer::HeadRows)),
        };
        let counts = Counts {
            padded_rows: rows,
            ..Counts::rows(scored)
        };
        let launch = lowered(&lowering, phase, Stage::GatherHeadRows).remove(0);
        assert_eq!(
            bound.one(&gather, &buffers_of, counts),
            launch,
            "{:?} {rows} rows scoring {scored}",
            fixture.config
        );
    }

    /// A served prefill's gather takes one index per scored row: prefills
    /// of 2, 65 and 8192 tokens scoring one row supply 4 index bytes, which
    /// the op takes, at the prefill's grid; more rows gathered than the
    /// grid, or a grid past the capacity, are refused.
    #[test]
    fn prefill_gathers_take_one_index_per_scored_row() {
        let engines = Engines::new(SHAPE).unwrap();
        let fixture = Fixture::new(&engines, configs()[2]);
        for rows in [2, 65, 8192] {
            check_prefill_gather(&fixture, 0, rows, 1);
        }
        let gather = Sm120Own::new(SHAPE)
            .gather_rows(&GatherRows::of(SHAPE), Need::rows(PREFILL_TOKENS))
            .unwrap();
        let big = view(part(1, 0, 1 << 40));
        let buffers_of = GatherRowsBuffers {
            input: big,
            row_indices: view(part(2, 0, 4)),
            count: view(part(3, 0, 4)),
            output: big,
        };
        let counts = |rows, padded_rows| Counts {
            padded_rows,
            ..Counts::rows(rows)
        };
        let launch = gather
            .launches(&buffers_of, View::EMPTY, counts(1, 65))
            .unwrap();
        assert_eq!((launch[0].rows, launch[0].plan.grid[0]), (128, 128));
        assert!(
            gather
                .launches(&buffers_of, View::EMPTY, counts(0, 65))
                .is_ok()
        );
        assert!(
            gather
                .launches(&buffers_of, View::EMPTY, counts(2, 65))
                .is_err()
        );
        assert!(
            gather
                .launches(&buffers_of, View::EMPTY, counts(66, 65))
                .is_err()
        );
        assert!(
            gather
                .launches(&buffers_of, View::EMPTY, counts(1, 8193))
                .is_err()
        );
    }

    /// The target's own rows of one lowering launch what it lowers: the
    /// embedding's widening, a prefill's gather of its logits rows, and the
    /// all-reduce's own rows: with the reduce entries the comm epilogue's
    /// widening into the F32 sum, without them the engine composition's
    /// round, widen and round.
    fn check_target(fixture: &Fixture<'_>, lowering: &Lowering<'_>, phase: Phase) {
        let own = Sm120Own::new(SHAPE);
        let need = Need::rows(phase.capacity());
        let counts = Counts::rows(phase.rows());
        let bound = Bound::default();
        let scratch = |buffer| bound.view(lowering.scratch(buffer));
        let live = bound.view(lowering.inputs.live_tokens);
        let at = format!("{:?} {phase:?}", fixture.config);
        let widen = own.widen_rows(&WidenRows::of(SHAPE), need).unwrap();
        let widened = |input, output| WidenRowsBuffers {
            input,
            output,
            live_rows: live,
        };
        // The embedding's widening: the lowering's launch through its own
        // op.
        let embedding = lowered(lowering, phase, Stage::EmbeddingReduce).remove(0);
        let buffers_of = widened(scratch(Buffer::Collapsed), scratch(Buffer::Partial));
        assert_eq!(bound.one(&widen, &buffers_of, counts), embedding, "{at}");
        if let Phase::Prefill { rows } = phase {
            for scored in scored_rows(rows) {
                check_prefill_gather(fixture, lowering.rank, rows, scored);
            }
        }
        let widen_symbol = ModelRow::Widen.symbol();
        if fixture.config.entries {
            // The comm epilogue widens the rounded sums into the rank's F32
            // sums before its launch.
            let epilogue =
                lowered_with(lowering, phase, Stage::Reduce(0, Sublayer::Attention), true)
                    .into_iter()
                    .filter(|launch| launch.symbol == widen_symbol)
                    .next_back()
                    .unwrap();
            let summed = bound.view(lowering.buffers.comm.summed);
            let buffers_of = widened(scratch(Buffer::Reduced), summed);
            assert_eq!(bound.one(&widen, &buffers_of, counts), epilogue, "{at}");
            return;
        }
        let round = own.round_rows(&RoundRows::of(SHAPE), need).unwrap();
        let reduce = lowered(lowering, phase, Stage::Reduce(0, Sublayer::Mlp));
        let rounded = |input| RoundRowsBuffers {
            input,
            output: scratch(Buffer::Reduced),
            live_rows: live,
        };
        assert_eq!(
            bound.one(&round, &rounded(scratch(Buffer::Partial)), counts),
            reduce[0],
            "{at}"
        );
        let buffers_of = widened(scratch(Buffer::Reduced), scratch(Buffer::Partial));
        assert_eq!(bound.one(&widen, &buffers_of, counts), reduce[1], "{at}");
        let summed = bound.view(lowering.buffers.comm.summed);
        assert_eq!(
            bound.one(&round, &rounded(summed), counts),
            reduce[reduce.len() - 1],
            "{at}"
        );
    }

    /// The drafter's own rows of one lowering launch what it lowers: the
    /// row inputs' gather, the input halves, the residual add and the draft
    /// rows' gather; its argmax runs on the shared entries.
    fn check_drafter(fixture: &Fixture<'_>, lowering: &Lowering<'_>, phase: Phase) {
        let own = Sm120Own::new(SHAPE);
        let (rows, need) = (phase.rows(), Need::rows(phase.capacity()));
        let (counts, drafts) = (Counts::rows(rows), Counts::rows(draft_rows(phase)));
        let draft_need = Need::rows(DRAFT_ITEMS);
        let bound = Bound::default();
        let scratch = |buffer| bound.view(lowering.scratch(buffer));
        let live = bound.view(lowering.inputs.live_tokens);
        let at = format!("{:?} {phase:?}", fixture.config);
        let gather = own.draft_gather(&DraftGather::of(SHAPE), need).unwrap();
        let buffers_of = DraftGatherBuffers {
            retained: bound.view(lowering.buffers.state.retained().unwrap()),
            last: scratch(Buffer::Normed),
            codes: scratch(Buffer::DraftCodes),
            drafts: scratch(Buffer::Drafts),
            live_rows: live,
            output: scratch(Buffer::DraftSource),
            tokens: bound.view(lowering.inputs.token_ids),
        };
        let launch = lowered(lowering, phase, Stage::DraftGather).remove(0);
        assert_eq!(bound.one(&gather, &buffers_of, counts), launch, "{at}");
        let halves = own
            .eh_input(
                &EhInput::of(SHAPE),
                usage("mtp.eh", phase, rows),
                Need::rows(rows),
            )
            .unwrap();
        let buffers_of = EhInputBuffers {
            embed: scratch(Buffer::Normed),
            hidden: scratch(Buffer::Collapsed),
            output: scratch(Buffer::DraftEhInput),
            live_rows: live,
        };
        let launches = halves.launches(&buffers_of, View::EMPTY, counts).unwrap();
        assert_eq!(
            bound.dispatched(launches),
            lowered(lowering, phase, Stage::DraftEhInput),
            "{at}"
        );
        let add = own.residual_add(&ResidualAdd::of(SHAPE), need).unwrap();
        let buffers_of = ResidualAddBuffers {
            x: scratch(Buffer::DraftHidden),
            y: scratch(Buffer::Reduced),
            live_rows: live,
        };
        for sublayer in [Sublayer::Attention, Sublayer::Mlp] {
            let launch = lowered(lowering, phase, Stage::DraftResidual(sublayer)).remove(0);
            assert_eq!(bound.one(&add, &buffers_of, counts), launch, "{at}");
        }
        let head_rows = own.gather_rows(&GatherRows::of(SHAPE), draft_need).unwrap();
        let buffers_of = GatherRowsBuffers {
            input: scratch(Buffer::Normed),
            row_indices: scratch(Buffer::DraftRows),
            count: scratch(Buffer::DraftCount),
            output: scratch(Buffer::HeadRows),
        };
        let launch = lowered(lowering, phase, Stage::DraftHeadRows).remove(0);
        assert_eq!(bound.one(&head_rows, &buffers_of, drafts), launch, "{at}");
        // The argmax runs on the shared entries: the partial, the pairs'
        // two gathers and the final (`model_glm5next_contract::draft_argmax`).
        let argmax = lowered(lowering, phase, Stage::DraftArgmax);
        assert_eq!(
            argmax
                .iter()
                .map(|launch| launch.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                kernels_sm120::row::argmax::PARTIAL_F32_SYMBOL,
                model_glm5next_contract::sm120::context_parallel::ALLGATHER_SYMBOL,
                model_glm5next_contract::sm120::context_parallel::ALLGATHER_SYMBOL,
                kernels_sm120::row::argmax::FINAL_SYMBOL,
            ],
            "{at}"
        );
    }

    /// The target's and the drafter's own rows launch what the lowering
    /// launches over the whole served domain: under the package's FP8
    /// configuration with the drafter's layer, every prefill row count
    /// from 1 to 8192, every decode bucket and every verify phase, on every
    /// rank, the drafter's extend pass at prefill and verify and its step
    /// pass at decode; under every other configuration (each KV format,
    /// the reduce entries off, one, three and eight served items, five
    /// drafts) the row buckets' edges.
    #[test]
    fn own_rows_launch_what_the_lowering_plans() {
        // The edges include a 65-row prefill and a one-item instance's
        // eager verify step of six rows.
        let narrow = phases(Items::new(1).unwrap(), false);
        assert!(narrow.contains(&Phase::Prefill { rows: 65 }));
        assert!(narrow.contains(&Phase::Verify { rows: 6, items: 1 }));
        let engines = Engines::new(SHAPE).unwrap();
        for (index, config) in configs().into_iter().enumerate() {
            let full = index == 2;
            let fixture = Fixture::new(&engines, config);
            assert!(!full || has_mtp_layer(&config.format));
            for phase in phases(fixture.items, full) {
                for rank in 0..SHAPE.ranks() {
                    let inputs = fixture.inputs(rank, phase);
                    let lowering = fixture.lowering(&inputs, rank, phase, None, None);
                    check_target(&fixture, &lowering, phase);
                    if !has_mtp_layer(&config.format) {
                        continue;
                    }
                    let pass = match phase {
                        Phase::Decode(_) => DraftPass::Step,
                        _ => DraftPass::Extend,
                    };
                    let lowering = fixture.lowering(&inputs, rank, phase, Some(pass), None);
                    check_drafter(&fixture, &lowering, phase);
                }
            }
        }
    }

    /// One rank's projections over the pinned FP8 test component, as
    /// create binds them (`crate::projections`).
    struct Projections {
        instances: crate::projections::Instances,
        stores: Vec<bundle_runtime::projection::Store>,
        scratch: Vec<crate::projections::RankScratch>,
    }

    impl Projections {
        fn new(fixture: &Fixture<'_>) -> Projections {
            let bytes = model_glm5next_contract::prepared::test_component();
            let (backend, _) = crate::sm120::backend(fixture.engines, SHAPE);
            let authenticated = crate::projections::authenticate(
                &bytes,
                &model_glm5next_contract::quantization::FP8_BLOCK128,
                backend.as_ref(),
                &Default::default(),
            )
            .unwrap();
            let mut next = 7000;
            let mut allocate = |bytes| {
                next += 1;
                Ok::<_, ()>(part(next, 0, bytes))
            };
            let stores = (0..SHAPE.ranks())
                .map(|rank| {
                    let layout = model_glm5next_contract::direct::layout(SHAPE, rank).unwrap();
                    crate::projections::bind(
                        &authenticated,
                        &layout,
                        &fixture.ranks[rank as usize].views,
                    )
                    .unwrap()
                })
                .collect();
            let scratch = (0..SHAPE.ranks())
                .map(|_| crate::projections::RankScratch::allocate(&authenticated, &mut allocate))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            Projections {
                instances: crate::projections::instances(&authenticated).unwrap(),
                stores,
                scratch,
            }
        }

        fn rank(&self, rank: u32) -> crate::projections::Rank<'_> {
            let rank = rank as usize;
            crate::projections::Rank {
                instances: &self.instances,
                store: &self.stores[rank],
                scratch: &self.scratch[rank],
                side_scratch: &self.scratch[rank],
            }
        }
    }

    /// The stacked KDA input projections of every decode and verify phase
    /// of at most 32 rows launch what the lowering launches over an
    /// instance's projections, on every rank.
    #[test]
    fn projection_rows_launch_what_the_lowering_plans() {
        let engines = Engines::new(SHAPE).unwrap();
        let fixture = Fixture::new(&engines, configs()[2]);
        let projections = Projections::new(&fixture);
        let own = Sm120Own::new(SHAPE);
        // The KDA projections' consumers (`lower/project.rs`, `io`).
        let consumers = [
            Buffer::KdaQkv,
            Buffer::KdaQkv,
            Buffer::KdaQkv,
            Buffer::KdaForgetDown,
            Buffer::KdaBeta,
            Buffer::KdaGateDown,
        ];
        let mut stacked_phases = 0;
        for phase in phases(fixture.items, true) {
            let rows = phase.rows();
            let counts = Counts::rows(rows);
            let stacked = (!matches!(phase, Phase::Prefill { .. }) && rows <= stacked::MAX_ROWS)
                .then(|| {
                    let used = usage(stacked::PROJECTIONS[0], phase, rows);
                    let placed = stacked::placed(used).unwrap();
                    let entry = stacked::part(stacked::PROJECTIONS[0], WeightFormat::Bf16, used)
                        .unwrap()
                        .unwrap()
                        .entry;
                    let op = StackedProjection {
                        k: entry.k,
                        parts: placed
                            .iter()
                            .map(|placed| StackedPart {
                                rows: placed.part.rows,
                                width: placed.part.width,
                                row_stride: placed.part.row_stride,
                            })
                            .collect(),
                    };
                    let prepared = own.stacked_projection(&op, used, Need::rows(rows)).unwrap();
                    (used, placed, prepared)
                });
            stacked_phases += usize::from(stacked.is_some());
            for rank in 0..SHAPE.ranks() {
                let inputs = fixture.inputs(rank, phase);
                let lowering =
                    fixture.lowering(&inputs, rank, phase, None, Some(projections.rank(rank)));
                let bound = Bound::default();
                let Some((used, placed, prepared)) = &stacked else {
                    continue;
                };
                let store = &projections.stores[rank as usize];
                for layer in [0, 1, 2, 4] {
                    let launch = lowered(
                        &lowering,
                        phase,
                        Stage::Project(layer, stacked::PROJECTIONS[0]),
                    );
                    let weights = placed
                        .iter()
                        .map(|placed| {
                            let operation = &projections.instances
                                [&(placed.id.to_owned(), Some(u64::from(layer)))];
                            bound.view(store.handle(operation, *used).unwrap().weights().codes)
                        })
                        .collect();
                    let outputs = placed
                        .iter()
                        .zip(consumers)
                        .map(|(placed, consumer)| {
                            let span = lowering.scratch(consumer);
                            let offset = u64::from(placed.column_offset) * 2;
                            bound.view(DeviceBufferSpanV1::new(
                                span.buffer,
                                span.offset + offset,
                                span.length - offset,
                            ))
                        })
                        .collect();
                    let scratch = projections.scratch[rank as usize].scratch();
                    let buffers_of = StackedProjectionBuffers {
                        x: bound.view(lowering.scratch(Buffer::Normed)),
                        weights,
                        outputs,
                        tickets: bound.view(scratch.semaphores),
                    };
                    let workspace = bound.view(scratch.partial);
                    let launches = prepared.launches(&buffers_of, workspace, counts).unwrap();
                    assert_eq!(
                        bound.dispatched(launches),
                        launch,
                        "{phase:?} layer {layer}"
                    );
                }
            }
        }
        // Decode buckets 1 to 32 and every verify phase of at most 32 rows.
        assert_eq!(stacked_phases, 6 + verify_phases(fixture.items).len());
    }

    /// The journal lines from `from` on that are launches.
    fn launched(host: &crate::fake_host_tests::FakeHost, from: usize) -> Vec<String> {
        crate::fake_host_tests::issued(host, from)
            .into_iter()
            .filter(|line| line.starts_with("launch "))
            .collect()
    }

    /// A journal launch line's stream, symbol, buffer spans and U64
    /// scalars, in argument order.
    fn parse_launch(line: &str) -> (u32, String, Vec<DeviceBufferSpanV1>, Vec<u64>) {
        let words = line.split(' ').collect::<Vec<_>>();
        let number = |text: &str| -> u64 {
            text.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .unwrap()
        };
        let mut spans = Vec::new();
        for piece in line
            .split("DeviceBuffer(DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
            .skip(1)
        {
            let raw = number(piece);
            let offset = number(&piece[piece.find("offset: ").unwrap() + 8..]);
            let length = number(&piece[piece.find("length: ").unwrap() + 8..]);
            spans.push(DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_raw(raw),
                offset,
                length,
            ));
        }
        let scalars = line.split("ScalarU64(").skip(1).map(number).collect();
        (number(words[1]) as u32, words[2].to_owned(), spans, scalars)
    }

    /// A dispatched launch as the fake device journals it on `stream`.
    fn journaled(stream: u32, launch: &Dispatched) -> String {
        format!(
            "launch {stream} {} {:?} {:?}",
            launch.symbol, launch.config, launch.arguments
        )
    }

    /// Commit's state copies as serving dispatches them: a verify step's
    /// commit through a fake-host instance (as
    /// `fake_host_tests::decode_and_verify_steps_run_through_the_ledger`
    /// runs it) journals one copy per checkpointing buffer and rank, each
    /// within one state pool. The op prepared from the buffer's entry
    /// bytes and committed layers (from the request state's declaration),
    /// over views of the journaled entries' starts spanning the op's own
    /// source and destination spans, journals the same line: stream,
    /// symbol, configuration and every argument; and it is the launch
    /// commit's builder formed (`state_copy::LayerCopy`) over those spans.
    #[test]
    fn state_copies_launch_what_commit_dispatches() {
        use crate::config::tests::{options, record};
        use crate::fake_host_tests::{FakeHost, NoRows};
        let host = FakeHost::new((MTP_DRAFTS, 32));
        let mut instance = {
            crate::fake_host_tests::api()
                .create(
                    runtime_bundle_abi::HostServicesTableV1::new(Rc::clone(&host)),
                    &record(
                        model_glm5next_contract::contract::DECODE_BATCH,
                        &options(2048),
                    ),
                )
                .unwrap()
        };
        let handle = instance.handle();
        crate::instance::seed(handle, 7, 0, 100);
        crate::instance::seed(handle, 8, 5, 64);
        crate::instance::run_lowered_stages_alone(handle);
        instance
            .execute(
                &step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]),
                &mut NoRows::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        let verify = step(&[(7, 0, 101, &[21, 22, 23, 24]), (8, 5, 65, &[13])]);
        instance.execute(&verify, &mut NoRows::default()).unwrap();
        let start = host.device.journal.borrow().len();
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
        let copies = launched(&host, start);
        assert_eq!(copies.len(), 4 * 3, "{copies:#?}");
        // Each checkpointing buffer's entry bytes and the layers commit
        // copies (`step::target_layers`: the tail's but the drafter's).
        let format = SHAPE.kv_formats()[0];
        let state = model_glm5next_contract::state::request_state(SHAPE, &format, false);
        let layers = |name: &str, layers: u64| {
            if name == model_glm5next_contract::contract::INDEXER_TAIL_STATE_ID {
                SHAPE.paged_layers(false)
            } else {
                layers
            }
        };
        let own = Sm120Own::new(SHAPE);
        let mut matched = 0;
        for line in &copies {
            let (stream, symbol, spans, scalars) = parse_launch(line);
            assert_eq!(symbol, state_copy::SYMBOL);
            let buffer = state
                .buffers
                .iter()
                .filter(|buffer| buffer.bytes_per_layer_rank() == scalars[2])
                .collect::<Vec<_>>();
            assert_eq!(buffer.len(), 1, "{line}");
            let op = StateCopy {
                bytes: buffer[0].bytes_per_layer_rank(),
                layers: layers(buffer[0].name, buffer[0].layers) as u32,
                source_stride: scalars[0],
                destination_stride: scalars[1],
            };
            // Both entries lie in the one state pool.
            assert_eq!(spans[0].buffer, spans[1].buffer);
            let bound = Bound::default();
            let source =
                DeviceBufferSpanV1::new(spans[0].buffer, spans[0].offset, op.source_span());
            let destination =
                DeviceBufferSpanV1::new(spans[1].buffer, spans[1].offset, op.destination_span());
            let prepared = own.state_copy(&op, Need::rows(1)).unwrap();
            let buffers_of = StateCopyBuffers {
                source: bound.view(source),
                destination: bound.view(destination),
            };
            let launch = bound.one(&prepared, &buffers_of, Counts::rows(1));
            assert_eq!(journaled(stream, &launch), *line);
            // What commit's builder formed over the same entries.
            let copy = state_copy::LayerCopy {
                bytes: op.bytes,
                layers: op.layers,
                source_stride: op.source_stride,
                destination_stride: op.destination_stride,
            };
            let mut bindings = BufferBindings::default();
            let arguments = vec![
                Argument::DeviceBuffer(bindings.bind(source)),
                Argument::DeviceBuffer(bindings.bind(destination)),
                Argument::U64(copy.source_stride),
                Argument::U64(copy.destination_stride),
                Argument::U64(copy.bytes),
            ];
            let formed = built(PreparedLaunch {
                symbol: state_copy::SYMBOL.into(),
                rows: 1,
                plan: copy.plan().unwrap(),
                arguments: ArgumentList(arguments),
                bindings,
            });
            assert_eq!(launch, formed);
            matched += 1;
        }
        assert_eq!(matched, 12);
    }

    /// A step of `items`: (request, slot, committed tokens, inputs), as
    /// `fake_host_tests` encodes it.
    fn step(items: &[(u64, u64, u64, &[u32])]) -> Vec<u8> {
        use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
        let mut batch = Vec::new();
        let mut appends = Vec::new();
        let mut slots = Vec::new();
        for &(request, slot, previous, inputs) in items {
            let count = inputs.len() as u64;
            let end = previous + count;
            let pages = (0..end.div_ceil(64)).map(|page| slot * 4 + page).collect();
            batch.push(if count == 1 {
                BatchItemV1::new(request, true, 1, true, inputs.to_vec())
            } else {
                BatchItemV1::verify(request, count as u32, false, inputs.to_vec())
            });
            appends.push(BatchAppendV1::new(previous, count, end, pages, None));
            slots.push(slot);
        }
        BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .with_state_slots(slots)
            .encode()
            .unwrap()
    }

    fn kept(request_id: u64, kept_rows: u32) -> runtime_bundle_abi::BundleKeptRowsV1 {
        runtime_bundle_abi::BundleKeptRowsV1 {
            struct_size: std::mem::size_of::<runtime_bundle_abi::BundleKeptRowsV1>() as u32,
            reserved: 0,
            request_id,
            kept_rows,
            reserved_tail: 0,
        }
    }

    /// Payload admission as create dispatches it: a fake-host create of the
    /// FP8 package journals one admission per checked view of each rank's
    /// image, in the order `payload::admissions` lists that rank's stored
    /// views. The op prepared from each admission's rule and count, over
    /// the journaled image buffer at the stored view's own offset and bytes
    /// (and its scales') and the journaled report, journals the same line.
    #[test]
    fn admissions_launch_what_create_dispatches() {
        use crate::config::tests::{options, record};
        use crate::fake_host_tests::FakeHost;
        let host = FakeHost::new((MTP_DRAFTS, 32));
        let start = host.device.journal.borrow().len();
        let _instance = {
            crate::fake_host_tests::api()
                .create(
                    runtime_bundle_abi::HostServicesTableV1::new(Rc::clone(&host)),
                    &record(
                        model_glm5next_contract::contract::DECODE_BATCH,
                        &options(2048),
                    ),
                )
                .unwrap()
        };
        let admitted = launched(&host, start)
            .into_iter()
            .filter(|line| line.contains("chop_admit_"))
            .collect::<Vec<_>>();
        let own = Sm120Own::new(SHAPE);
        let mut lines = admitted.iter();
        let mut count = 0;
        for rank in 0..SHAPE.ranks() {
            let stored = model_glm5next_contract::direct::layout(SHAPE, rank).unwrap();
            let entry = |name: &str| stored.iter().find(|entry| entry.view.name == name).unwrap();
            for admission in payload::admissions(&stored).unwrap() {
                let line = lines.next().expect("an admission line per checked view");
                let (stream, symbol, spans, _) = parse_launch(line);
                assert_eq!(symbol, admit::symbol(admission.check));
                let image = spans[0].buffer;
                let placed = |name: &str| {
                    let entry = entry(name);
                    DeviceBufferSpanV1::new(image, entry.offset, entry.bytes())
                };
                let bound = Bound::default();
                let op = Admission {
                    rule: admission.check,
                    count: admission.count,
                };
                let buffers_of = AdmissionBuffers {
                    codes: bound.view(placed(&admission.view)),
                    scales: admission
                        .scales
                        .as_deref()
                        .map(|name| bound.view(placed(name))),
                    first: bound.view(spans[spans.len() - 1]),
                };
                let prepared = own.admission(&op, Need::rows(1)).unwrap();
                let launch = bound.one(&prepared, &buffers_of, Counts::rows(1));
                assert_eq!(journaled(stream, &launch), *line);
                count += 1;
            }
        }
        assert!(lines.next().is_none());
        assert!(count > 0);
    }

    /// Every admission of every rank's placed views, in every checkpoint
    /// layout (create's dispatch is observed for the FP8 package above):
    /// the checked view, MXFP8's scales, the count and the report, at
    /// `sm120::payload::plan`, as `payload.rs` (`launch`) forms them.
    #[test]
    fn admissions_of_every_layout_launch_what_create_forms() {
        let own = Sm120Own::new(SHAPE);
        let mut checks = BTreeMap::new();
        for layout in model_glm5next_contract::quantization::LAYOUTS {
            for rank in 0..SHAPE.ranks() {
                let stored =
                    model_glm5next_contract::direct::checkpoint_layout(SHAPE, layout, rank)
                        .unwrap();
                let views = stored
                    .iter()
                    .map(|entry| {
                        let span = part(1000 + rank, entry.offset, entry.bytes());
                        (entry.view.name.clone(), span)
                    })
                    .collect::<BTreeMap<_, _>>();
                for (index, admission) in payload::admissions(&stored).unwrap().iter().enumerate() {
                    let report = part(2000 + rank, 4 * index as u64, 4);
                    let mut bindings = BufferBindings::default();
                    let mut arguments = vec![Argument::DeviceBuffer(
                        bindings.bind(views[&admission.view]),
                    )];
                    if let Some(scales) = &admission.scales {
                        arguments.push(Argument::DeviceBuffer(bindings.bind(views[scales])));
                    }
                    arguments.push(Argument::U64(admission.count));
                    arguments.push(Argument::DeviceBuffer(bindings.bind(report)));
                    let expected = built(PreparedLaunch {
                        symbol: admit::symbol(admission.check).into(),
                        rows: 1,
                        plan: admit::plan(admission.count).unwrap(),
                        arguments: ArgumentList(arguments),
                        bindings,
                    });
                    let bound = Bound::default();
                    let op = Admission {
                        rule: admission.check,
                        count: admission.count,
                    };
                    let prepared = own.admission(&op, Need::rows(1)).unwrap();
                    let buffers_of = AdmissionBuffers {
                        codes: bound.view(views[&admission.view]),
                        scales: admission
                            .scales
                            .as_ref()
                            .map(|scales| bound.view(views[scales])),
                        first: bound.view(report),
                    };
                    assert_eq!(bound.one(&prepared, &buffers_of, Counts::rows(1)), expected);
                    *checks.entry(format!("{:?}", admission.check)).or_insert(0) += 1;
                }
            }
        }
        // Each rule runs over some layout's views.
        assert_eq!(checks.len(), Check::ALL.len(), "{checks:?}");
    }

    /// Each op claims what its launched symbols' declarations do (GLM's
    /// target rows and stacked projections, none of the drafter's rows, the
    /// state copy or the admission) and refuses a need beyond it.
    #[test]
    fn guarantees_are_the_launched_symbols_own_declarations() {
        let own = Sm120Own::new(SHAPE);
        let need = Need::rows(32);
        let exact = need.with_exact(Invariance::COPY);
        let decode = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        let infos = [
            own.round_rows(&RoundRows::of(SHAPE), need)
                .map(|p| p.info().clone()),
            own.widen_rows(&WidenRows::of(SHAPE), need)
                .map(|p| p.info().clone()),
            own.residual_add(&ResidualAdd::of(SHAPE), need)
                .map(|p| p.info().clone()),
            own.gather_rows(&GatherRows::of(SHAPE), need)
                .map(|p| p.info().clone()),
            own.eh_input(&EhInput::of(SHAPE), decode, Need::rows(8))
                .map(|p| p.info().clone()),
            own.draft_gather(&DraftGather::of(SHAPE), need)
                .map(|p| p.info().clone()),
            own.state_copy(
                &StateCopy {
                    bytes: 16,
                    layers: 2,
                    source_stride: 32,
                    destination_stride: 32,
                },
                need,
            )
            .map(|p| p.info().clone()),
            own.admission(
                &Admission {
                    rule: Check::Ue4m3Scales,
                    count: 64,
                },
                need,
            )
            .map(|p| p.info().clone()),
        ];
        // The rounding, widening and gather rows (the first two ops and the
        // fourth) claim one-row bits at every bucket; the others nothing.
        for (at, info) in infos.into_iter().enumerate() {
            let info = info.unwrap();
            let claimed = match at {
                0 | 1 | 3 => Invariance::any_rows(8192),
                _ => Invariance::NONE,
            };
            for (symbol, _) in &info.symbols {
                assert!(crate::cubin_of(symbol).is_some(), "{symbol}");
                assert_eq!(declared(symbol), claimed, "{symbol}");
            }
            assert_eq!(info.guarantees, claimed, "{}", info.implementation);
        }
        // Each stacked bucket's entry claims one-row bits up to its rows.
        for entry in stacked::entries(SHAPE).unwrap() {
            let one_row = Invariance::one_row(entry.capacity);
            assert_eq!(declared(&entry.symbol), one_row, "{}", entry.symbol);
        }
        assert!(matches!(
            own.round_rows(&RoundRows::of(SHAPE), exact),
            Err(Refused::Inexact { .. })
        ));
    }

    /// Ops other than the compiled ones, capacities and rows past their
    /// limits, and views short of what a launch touches are refused.
    #[test]
    fn mismatched_ops_rows_and_views_are_refused() {
        let own = Sm120Own::new(SHAPE);
        let need = Need::rows(32);
        let hidden = SHAPE.profile.hidden;
        assert!(
            own.round_rows(&RoundRows { width: hidden / 2 }, need)
                .is_err()
        );
        // A row op grids at most the largest row bucket.
        assert!(
            own.widen_rows(&WidenRows::of(SHAPE), Need::rows(8193))
                .is_err()
        );
        assert!(
            own.widen_rows(&WidenRows::of(SHAPE), Need::rows(0))
                .is_err()
        );
        // Rows past the capacity, short views.
        let widen = own.widen_rows(&WidenRows::of(SHAPE), need).unwrap();
        let big = view(part(1, 0, 1 << 40));
        let buffers_of = WidenRowsBuffers {
            input: big,
            output: big,
            live_rows: view(part(2, 0, 4)),
        };
        assert!(
            widen
                .launches(&buffers_of, View::EMPTY, Counts::rows(32))
                .is_ok()
        );
        assert!(
            widen
                .launches(&buffers_of, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        assert!(
            widen
                .launches(&buffers_of, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        let short = WidenRowsBuffers {
            output: view(part(1, 0, 5 * u64::from(hidden) * 4 - 1)),
            ..buffers_of
        };
        assert!(
            widen
                .launches(&short, View::EMPTY, Counts::rows(5))
                .is_err()
        );
        assert!(widen.launches(&short, View::EMPTY, Counts::rows(4)).is_ok());
        // The drafter's input halves run at their use's capacity.
        let decode = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        assert!(
            own.eh_input(&EhInput::of(SHAPE), decode, Need::rows(9))
                .is_err()
        );
        // Launches stay within the prepared capacity, below the use's.
        let halves = own
            .eh_input(&EhInput::of(SHAPE), decode, Need::rows(4))
            .unwrap();
        let placed = EhInputBuffers {
            embed: view(part(5, 0, 1 << 30)),
            hidden: view(part(6, 0, 1 << 30)),
            output: big,
            live_rows: view(part(2, 0, 4)),
        };
        assert!(
            halves
                .launches(&placed, View::EMPTY, Counts::rows(4))
                .is_ok()
        );
        assert!(
            halves
                .launches(&placed, View::EMPTY, Counts::rows(5))
                .is_err()
        );
        // Stacked projections exist at decode and verify uses of at most 32
        // rows, for the KDA layer's parts only.
        let entry = stacked::part(stacked::PROJECTIONS[0], WeightFormat::Bf16, decode)
            .unwrap()
            .unwrap()
            .entry;
        let op = StackedProjection {
            k: entry.k,
            parts: entry
                .parts
                .iter()
                .map(|p| StackedPart {
                    rows: p.rows,
                    width: p.width,
                    row_stride: p.row_stride,
                })
                .collect(),
        };
        assert!(own.stacked_projection(&op, decode, Need::rows(8)).is_ok());
        let stacked = own.stacked_projection(&op, decode, Need::rows(4)).unwrap();
        let projected = StackedProjectionBuffers {
            x: big,
            weights: vec![big; op.parts.len()],
            outputs: vec![big; op.parts.len()],
            tickets: big,
        };
        assert!(stacked.launches(&projected, big, Counts::rows(4)).is_ok());
        assert!(stacked.launches(&projected, big, Counts::rows(5)).is_err());
        assert!(own.stacked_projection(&op, decode, Need::rows(9)).is_err());
        let prefill = usage(stacked::PROJECTIONS[0], Phase::Prefill { rows: 64 }, 64);
        assert!(own.stacked_projection(&op, prefill, Need::rows(8)).is_err());
        let fewer = StackedProjection {
            parts: op.parts[..5].to_vec(),
            ..op.clone()
        };
        assert!(
            own.stacked_projection(&fewer, decode, Need::rows(8))
                .is_err()
        );
        // Admissions: an MXFP8 check reads its scales, the others none.
        let mxfp8 = own
            .admission(
                &Admission {
                    rule: Check::Mxfp8,
                    count: 64,
                },
                Need::rows(1),
            )
            .unwrap();
        let checked = AdmissionBuffers {
            codes: big,
            scales: None,
            first: view(part(2, 0, 4)),
        };
        assert!(
            mxfp8
                .launches(&checked, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let scaled = AdmissionBuffers {
            scales: Some(view(part(4, 0, 2))),
            ..checked
        };
        assert!(
            mxfp8
                .launches(&scaled, View::EMPTY, Counts::rows(1))
                .is_ok()
        );
        let f32_scales = own
            .admission(
                &Admission {
                    rule: Check::F32Scales,
                    count: 64,
                },
                Need::rows(1),
            )
            .unwrap();
        assert!(
            f32_scales
                .launches(&scaled, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        assert!(
            own.admission(
                &Admission {
                    rule: Check::F32Scales,
                    count: u64::from(payload::NONE),
                },
                Need::rows(1)
            )
            .is_err()
        );
    }
}
