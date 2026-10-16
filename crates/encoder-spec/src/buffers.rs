//! Buffer plan: the device regions one rank allocates for its encoders, for
//! runs of up to `rows` rows, derived from their declarations. Every region
//! a run touches holds `rows` rows divided by the merges before the stage
//! that uses it; the payload holds a whole item when an item takes several
//! runs, and rows that a stage joins across runs hold the largest item's.
//!
//! The precision policy fixes the lowering, and with it the aliasing:
//! - `f32-pairs.v1` keeps the residual stream in `stream` (the projection
//!   and down GEMMs add into it) and alternates every other stage's output
//!   between two scratch regions `a` and `b`; q, k and v become BF16 pair
//!   rows in place in `qkv`'s region, a SwiGLU's gate and up columns fill
//!   `b` and its product `a`, and a reorder gathers the stream into `a`. A
//!   declaration whose items are one run each uploads every side input
//!   before the run's first launch, so each stage's side inputs (a
//!   transformer's rotary rows and sequence bounds, a position stage's taps)
//!   have regions of their own (`Region::owner`) and each distinct live row
//!   count has its own slot. One whose items take several runs
//!   (`Declaration::staged`) stages them in command order: shared
//!   `sequences`, `cos`, `sin` and `reorder` regions and one live row `count`.
//!
//! Under this lowering a stage never writes the region it reads, except
//! in place (position adds, the residual adds, rotary): a declaration whose
//! stages would (for example a linear between two transformers, which would
//! read and write the residual stream) is refused rather than given aliased
//! regions. Each transformer reads its residual stream where the stage
//! before it wrote it.

use crate::components::rotary::RotaryComponent;
use crate::components::{
    Boundary, Component, Conv1d, Data, Element, Linear, Merge, StageOp, Transformer, mul,
};
use crate::policy::Precision;
use crate::{Chunking, Declaration, Error, Item, Sequences};

/// What a rank's encoders serve into, besides their declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Serving {
    /// Rows of the media arena the encoders write their output rows into.
    pub arena_rows: u64,
    /// Rows of one prefill step, each with an i64 arena source row.
    pub prefill_rows: u64,
    /// Upload staging of a recorded driver, or zero for none.
    pub staging_bytes: u64,
}

/// One use of a region.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub encoder: String,
    /// The stage, or `None` for the payload and the serving regions.
    pub stage: Option<usize>,
    pub what: String,
    pub bytes: u64,
}

/// One allocation: as large as its largest use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Region {
    pub name: &'static str,
    /// The stage whose launches alone read the region, for a stage's own
    /// side inputs; `None` for a region its uses share.
    pub owner: Option<usize>,
    pub bytes: u64,
    pub uses: Vec<Use>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BufferPlan {
    /// In order of first use.
    pub regions: Vec<Region>,
}

impl BufferPlan {
    /// Every region's bytes.
    pub fn bytes(&self) -> u64 {
        self.regions.iter().map(|region| region.bytes).sum()
    }

    /// The first region named `name`.
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|region| region.name == name)
    }

    /// The region `name` of `owner` (`Region::owner`).
    pub fn owned(&self, name: &str, owner: Option<usize>) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.name == name && region.owner == owner)
    }

    fn add(&mut self, name: &'static str, at: (&str, Option<usize>), what: &str, bytes: u64) {
        self.add_owned(name, None, at, what, bytes);
    }

    fn add_owned(
        &mut self,
        name: &'static str,
        owner: Option<usize>,
        at: (&str, Option<usize>),
        what: &str,
        bytes: u64,
    ) {
        let found = self
            .regions
            .iter()
            .position(|region| region.name == name && region.owner == owner);
        let index = match found {
            Some(index) => index,
            None => {
                self.regions.push(Region {
                    name,
                    owner,
                    bytes: 0,
                    uses: Vec::new(),
                });
                self.regions.len() - 1
            }
        };
        let region = &mut self.regions[index];
        region.bytes = region.bytes.max(bytes);
        region.uses.push(Use {
            encoder: at.0.to_owned(),
            stage: at.1,
            what: what.to_owned(),
            bytes,
        });
    }
}

/// The use the scratch regions of stage `id` carry (a registered stage's
/// declared scratch, a convolution's unfolded taps): its output region
/// alone carries `id`, so a lowering finds the output by its use, whatever
/// scratch the stage declares for the planned rows.
pub fn scratch_use(id: &str) -> String {
    format!("{id} scratch")
}

/// Why a transformer of several carries has no lowering: one `carry`
/// region holds the rows of one carry, which a second would overwrite while
/// the first still needs them.
const ONE_CARRY: &str = "one `carry` region holds one carry's rows; no lowering of several";

/// The rows a full run holds after `op`, from `rows` before it.
fn rows_after(op: &StageOp, rows: u64) -> Result<u64, String> {
    match op {
        StageOp::Merge(Merge::Consecutive(params)) => {
            let group = u64::from(params.rows);
            if rows % group != 0 {
                return Err(format!("{rows} rows are not whole groups of {group}"));
            }
            Ok(rows / group)
        }
        _ => Ok(rows),
    }
}

/// Where a pairs-lowered stage's rows live.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Place {
    Payload,
    Stream,
    A,
    B,
    /// A region of its own: the codes or joined rows of a registered stage,
    /// a group's gathered rows.
    Own(&'static str),
}

impl Place {
    fn name(self) -> &'static str {
        match self {
            Place::Payload => "payload",
            Place::Stream => "stream",
            Place::A => "a",
            Place::B => "b",
            Place::Own(name) => name,
        }
    }

    /// The scratch region a stage reading here writes.
    fn other(self) -> Place {
        if self == Place::A { Place::B } else { Place::A }
    }
}

/// One declaration's stages with the rows a full run holds at each.
struct Lowering<'a> {
    declaration: &'a Declaration,
    /// Per stage: its component, the rows it reads and writes, and the
    /// full-run rows before and after it.
    stages: Vec<(&'a dyn Component, Boundary, Boundary, u64, u64)>,
}

impl<'a> Lowering<'a> {
    fn new(declaration: &'a Declaration, rows: u64) -> Result<Self, Error> {
        if rows == 0 || rows > u64::from(declaration.input.capacity_rows) {
            return Err(Error(format!(
                "{}: runs of {rows} rows are not within capacity_rows {}",
                declaration.name, declaration.input.capacity_rows
            )));
        }
        let steps = declaration.steps()?;
        let mut stages = Vec::new();
        let mut full = rows;
        for (index, stage) in declaration.stages.iter().enumerate() {
            let after = rows_after(&stage.op, full)
                .map_err(|error| Error(format!("{}: {error}", steps[index].at)))?;
            stages.push((
                steps[index].component,
                steps[index].input.clone(),
                steps[index + 1].input.clone(),
                full,
                after,
            ));
            full = after;
        }
        Ok(Self {
            declaration,
            stages,
        })
    }

    fn name(&self) -> &str {
        &self.declaration.name
    }

    /// The payload of one item at most, when an item takes several runs.
    fn whole_payload(&self) -> Result<Option<u64>, Error> {
        let input = &self.declaration.input;
        let chunked = input.chunking == Some(Chunking::WholeSequences)
            || matches!(self.declaration.sequences, Sequences::Segments { .. });
        if !chunked {
            return Ok(None);
        }
        let mut item = 0;
        for recipe in &input.recipes {
            item = item.max(recipe.rows().map_err(Error)?.item);
        }
        let rows = mul(item, u64::from(input.columns)).map_err(Error)?;
        let bytes = mul(rows, input.payload.bytes()).map_err(Error)?;
        Ok(Some(bytes))
    }

    /// The rows of the largest item after `stage`. Rows a stage joins
    /// across runs are held as many as the joining stage writes.
    fn largest(&self, stage: usize) -> Result<u64, Error> {
        let mut item = 0;
        for recipe in &self.declaration.input.recipes {
            item = item.max(recipe.rows().map_err(Error)?.item);
        }
        let trace = self.declaration.trace(Item {
            rows: item,
            grid: None,
        })?;
        Ok(trace[stage].rows)
    }

    /// The sequence bounds a run holds at most: two for one sequence per
    /// item, else one more than its rows.
    fn bounds(&self, rows: u64) -> u64 {
        let own = self.declaration.stages.iter().any(
            |stage| matches!(&stage.op, StageOp::Transformer(t) if t.params().sequences.is_some()),
        );
        if matches!(self.declaration.sequences, Sequences::Item {}) && !own {
            2
        } else {
            rows + 1
        }
    }

    /// A transformer's rotary rows, in regions its stage owns or shares.
    fn rotary(
        &self,
        plan: &mut BufferPlan,
        (at, owner): (Option<usize>, Option<usize>),
        transformer: &Transformer,
        rows: u64,
    ) -> Result<(), Error> {
        if let Some(slot) = &transformer.params().rotary {
            let rotary: &dyn RotaryComponent = slot.component().map_err(Error)?;
            let half = rotary.rotary_dims().map_err(Error)? / 2;
            for name in ["cos", "sin"] {
                plan.add_owned(name, owner, (self.name(), at), "rotary", rows * half * 4);
            }
        }
        Ok(())
    }

    fn pairs(&self, plan: &mut BufferPlan, rows: u64) -> Result<(), Error> {
        let name = self.name();
        let f32 = Element::F32.bytes();
        let input = &self.declaration.input;
        let staged = self.declaration.staged();
        let last = self.stages.len() - 1;
        let mut place = match self.whole_payload()? {
            Some(bytes) => {
                plan.add("payload", (name, None), "payload", bytes);
                Place::Payload
            }
            None => {
                let bytes = rows * u64::from(input.columns) * input.payload.bytes();
                plan.add("b", (name, None), "payload", bytes);
                Place::B
            }
        };
        // A stage writes the residual stream when the next stage that does
        // not work in place is a transformer.
        let feeds_stream = |index: usize| {
            self.declaration.stages[index + 1..]
                .iter()
                .find(|stage| !matches!(stage.op, StageOp::Position(_)))
                .is_some_and(|stage| matches!(stage.op, StageOp::Transformer(_)))
        };
        let mut counts = Vec::new();
        let mut transformers = Vec::new();
        for (index, (component, input, output, before, after)) in self.stages.iter().enumerate() {
            let at = (name, Some(index));
            let id = component.id();
            let fail = |error: String| Error(format!("{name} stages[{index}] ({id}): {error}"));
            let joins_next = self
                .stages
                .get(index + 1)
                .is_some_and(|next| next.0.joins_segments());
            for rows in [before, after] {
                if !counts.contains(rows) {
                    counts.push(*rows);
                }
            }
            let out_bytes = after * output.data.row_bytes();
            // The stage's output into the residual stream or the scratch
            // region it does not read: never the region it reads.
            let write = |plan: &mut BufferPlan, place: &mut Place| -> Result<(), Error> {
                if joins_next && index != last {
                    // A joining stage reads every run's rows of the item.
                    let bytes = self.largest(index + 1)? * output.data.row_bytes();
                    *place = Place::Own("joined");
                    plan.add("joined", at, &id, bytes);
                    return Ok(());
                }
                let region = if feeds_stream(index) {
                    Place::Stream
                } else if matches!(output.data, Data::Codes { .. }) {
                    Place::Own("codes")
                } else {
                    place.other()
                };
                if region == *place {
                    return Err(fail(format!(
                        "would write region `{}`, which it reads; no f32-pairs.v1 lowering",
                        region.name()
                    )));
                }
                *place = region;
                plan.add(region.name(), at, &id, out_bytes);
                Ok(())
            };
            match &self.declaration.stages[index].op {
                StageOp::Linear(Linear::Pairs(_))
                | StageOp::Norm(_)
                | StageOp::Cast(_)
                | StageOp::Pad(_) => {
                    write(plan, &mut place)?;
                }
                StageOp::Merge(Merge::SpaceToDepth(_)) => write(plan, &mut place)?,
                StageOp::Merge(Merge::Consecutive(_)) => {}
                StageOp::Position(_) => {
                    for region in ["tap-indices", "tap-weights"] {
                        plan.add_owned(region, Some(index), at, &id, before * 4 * f32);
                    }
                }
                StageOp::Conv1d(Conv1d::Im2col(params)) => {
                    let width = u64::from(input.width().map_err(fail)?);
                    let taps = mul(u64::from(params.kernel), width).map_err(fail)?;
                    plan.add("im2col", at, &scratch_use(&id), after * taps * f32);
                    write(plan, &mut place)?;
                }
                StageOp::Transformer(transformer) => {
                    let p = transformer.params();
                    if place != Place::Stream {
                        return Err(fail(
                            "the residual stream starts in the stage before".into(),
                        ));
                    }
                    if !staged && !p.reorder.is_empty() {
                        return Err(fail(
                            "a reorder stages its maps and rotary rows per block; no \
                             f32-pairs.v1 lowering of one run per item"
                                .into(),
                        ));
                    }
                    let attention = &p.attention;
                    let width = u64::from(p.width);
                    plan.add("stream", at, "residual", before * width * f32);
                    plan.add("a", at, "norm", before * width * f32);
                    let qkv = attention.qkv_width().map_err(fail)?;
                    plan.add("b", at, "qkv, then its pair rows", before * qkv * f32);
                    let heads = attention.attention_width().map_err(fail)?;
                    plan.add("a", at, "attention", before * heads * f32);
                    let hidden = p.mlp.inner();
                    if p.mlp.gated() {
                        plan.add("b", at, "mlp", before * 2 * hidden * f32);
                        plan.add("a", at, "swiglu", before * hidden * f32);
                    } else {
                        plan.add("b", at, "mlp", before * hidden * f32);
                    }
                    if !p.reorder.is_empty() {
                        plan.add("a", at, "reorder", before * width * f32);
                        plan.add("reorder", at, "reorder", before * 4);
                    }
                    if p.carry.len() > 1 {
                        return Err(fail(ONE_CARRY.into()));
                    }
                    if !p.carry.is_empty() {
                        plan.add("carry", at, "carry", before * width * f32);
                    }
                    let owner = (!staged).then_some(index);
                    self.rotary(plan, (Some(index), owner), transformer, *before)?;
                    transformers.push(index);
                }
                StageOp::Projector(projector) => {
                    // Only `a` and `b`: the norm's (then GELU's) rows in the
                    // region the stage does not read, the up projection's
                    // rows in the other one (the input is consumed by then),
                    // the SwiGLU product back in the first, and the down
                    // projection writes past the product.
                    let p = projector.params();
                    let width = u64::from(input.width().map_err(fail)?);
                    let inner = p.mlp.inner();
                    let up = if p.mlp.gated() { 2 * inner } else { inner };
                    let scratch = place.other();
                    let scratch_use = scratch_use(&id);
                    plan.add(scratch.name(), at, &scratch_use, before * width * f32);
                    let up_rows = scratch.other().name();
                    plan.add(up_rows, at, &scratch_use, before * up * f32);
                    if p.mlp.gated() {
                        plan.add(scratch.name(), at, &scratch_use, before * inner * f32);
                    }
                    place = if p.mlp.gated() {
                        scratch
                    } else {
                        scratch.other()
                    };
                    write(plan, &mut place)?;
                }
                StageOp::Group(_) => {
                    plan.add("gathered", at, &id, out_bytes);
                    place = Place::Own("gathered");
                }
                StageOp::Registered(_) if staged => {
                    for scratch in component.scratch(*before, input).map_err(fail)? {
                        plan.add(scratch.name, at, &scratch_use(&id), scratch.bytes);
                    }
                    write(plan, &mut place)?;
                }
                _ => return Err(fail("no f32-pairs.v1 lowering".into())),
            }
        }
        if staged {
            plan.add(
                "sequences",
                (name, None),
                "cu_seqlens",
                self.bounds(rows) * 4,
            );
            plan.add("count", (name, None), "live rows", 4);
            return Ok(());
        }
        for stage in transformers {
            let bytes = self.bounds(rows) * 4;
            plan.add_owned(
                "sequences",
                Some(stage),
                (name, Some(stage)),
                "cu_seqlens",
                bytes,
            );
        }
        plan.add("counts", (name, None), "live rows", counts.len() as u64 * 4);
        Ok(())
    }
}

/// The buffer plan of one rank serving `declarations` (writing one arena)
/// in runs of up to `rows` rows. Declarations under different precision
/// policies share a plan only if every one is staged
/// (`Declaration::staged`): their side-input regions then mean the same.
pub fn plan(
    declarations: &[&Declaration],
    rows: u64,
    serving: Serving,
) -> Result<BufferPlan, Error> {
    let first = declarations
        .first()
        .ok_or_else(|| Error("no declaration".into()))?;
    let hidden = first.output.op.params().hidden;
    let staged = declarations.iter().all(|declaration| declaration.staged());
    let mut plan = BufferPlan::default();
    for declaration in declarations {
        if declaration.precision != first.precision && !staged {
            return Err(Error(format!(
                "{}: one plan serves one precision policy, or staged declarations only",
                declaration.name
            )));
        }
        if declaration.output.op.params().hidden != hidden {
            return Err(Error(format!(
                "{}: one plan serves one arena",
                declaration.name
            )));
        }
        let lowering = Lowering::new(declaration, rows)?;
        match declaration.precision {
            Precision::Bf16Legacy => {
                return Err(Error(
                    "precision bf16-legacy.v1 is retired; use f32-pairs.v1".into(),
                ));
            }
            Precision::F32Pairs => lowering.pairs(&mut plan, rows)?,
        }
    }
    let serving_use = ("", None);
    plan.add(
        "source-rows",
        serving_use,
        "prefill source rows",
        serving.prefill_rows * 8,
    );
    if serving.staging_bytes > 0 {
        plan.add(
            "staging",
            serving_use,
            "recorded uploads",
            serving.staging_bytes,
        );
    }
    let arena = mul(
        mul(serving.arena_rows, u64::from(hidden)).map_err(Error)?,
        Element::Bf16.bytes(),
    )
    .map_err(Error)?;
    plan.add("arena", serving_use, "media arena", arena);
    Ok(plan)
}
