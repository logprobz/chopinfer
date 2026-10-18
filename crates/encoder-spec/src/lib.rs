//! Encoder declarations: one JSON file per encoder,
//! `models/<m>/contract/encoders/<name>.json`, read by every consumer
//! through this crate. A declaration names its input recipes with their
//! geometry limits, an ordered list of stages built from typed, versioned
//! components, the output layout into the language model, the weight
//! mapping, the precision policy and the admissible parallel policies.
//! Model-specific variants come from the caller's
//! `Environment`. See README.md.

pub mod blocks;
pub mod buffers;
pub mod components;
pub mod cost;
pub mod geometry;
pub mod id;
pub mod placement;
pub mod policy;
pub mod registry;
pub mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest as _, Sha256};

use components::{
    Boundary, Component, Context, Data, Element, Item, Layout, RotarySlot, Stage, StageOp, mul,
};
use policy::{ParallelClass, ParallelPolicy, Precision};
use registry::Registry;
use weights::{Place, Role, TensorInfo, Weights};

/// Why a declaration was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// What the caller admits for declarations of model `owner`: that model's
/// providers.
pub struct Environment {
    pub owner: String,
    pub registry: Registry,
}

impl Environment {
    /// No providers: shared variants and `f32-pairs.v1`
    /// only.
    pub fn new(owner: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            registry: Registry::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Schema {
    #[serde(rename = "chopinfer.encoder.v1")]
    V1,
}

/// `image.patch-grid.v1` limits: frames of `patch x patch` pixels, two
/// frames per temporal patch, merge blocks of `merge x merge` patches.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchGrid {
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub maximum_pixels: u64,
}

/// `video.timestamped-grids.v1` limits: a patch grid per pair of frames.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoGrids {
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub maximum_pixels_per_frame: u64,
    pub maximum_frames: u32,
}

/// `audio.log-mel.v1` limits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogMel {
    pub mel_bins: u32,
    pub maximum_frames: u64,
}

/// `image.centered-pad.v1` limits (`runtime_media::recipe`'s centered
/// pad): the image is fit so that its `downsample x downsample` patch
/// cells, laid out with the delimiters, take at most `maximum_tokens`
/// tokens (`h * (w + 1) + 2`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CenteredPad {
    pub patch: u32,
    pub downsample: u32,
    pub maximum_tokens: u32,
}

/// `image.zero-padded-token-budget.v1` limits: one patch grid of whole
/// `merge x merge` blocks, each block one token, at most `maximum_tokens`
/// of them; every temporal patch repeats the image.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenBudgetGrid {
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub maximum_tokens: u64,
}

/// A host preprocessing recipe of `runtime-media`, by its tag, with the
/// parameters that bound the rows it emits.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag")]
pub enum Recipe {
    #[serde(rename = "image.patch-grid.v1")]
    ImagePatchGrid(PatchGrid),
    #[serde(rename = "video.timestamped-grids.v1")]
    VideoTimestampedGrids(VideoGrids),
    #[serde(rename = "audio.log-mel.v1")]
    AudioLogMel(LogMel),
    #[serde(rename = "image.centered-pad.v1")]
    ImageCenteredPad(CenteredPad),
    #[serde(rename = "image.zero-padded-token-budget.v1")]
    ImageZeroPaddedTokenBudget(TokenBudgetGrid),
}

/// The rows one recipe emits at most.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecipeRows {
    pub columns: u64,
    pub patchify: Option<Patchify>,
    /// Every item's rows are a multiple of this.
    pub multiple: u64,
    /// The rows of one item.
    pub item: u64,
    /// The rows of one temporal grid (the item, for non-video recipes).
    pub grid: u64,
}

impl Recipe {
    pub fn tag(&self) -> &'static str {
        match self {
            Recipe::ImagePatchGrid(_) => "image.patch-grid.v1",
            Recipe::VideoTimestampedGrids(_) => "video.timestamped-grids.v1",
            Recipe::AudioLogMel(_) => "audio.log-mel.v1",
            Recipe::ImageCenteredPad(_) => "image.centered-pad.v1",
            Recipe::ImageZeroPaddedTokenBudget(_) => "image.zero-padded-token-budget.v1",
        }
    }

    /// The recipe's row geometry, from its limits.
    pub fn rows(&self) -> Result<RecipeRows, String> {
        let positive = |values: &[u64]| {
            if values.contains(&0) {
                Err(format!("recipe {} has a zero limit", self.tag()))
            } else {
                Ok(())
            }
        };
        let patch_columns = |patch: u32, temporal: u32| {
            mul(
                mul(3, u64::from(temporal))?,
                mul(u64::from(patch), u64::from(patch))?,
            )
        };
        let grid_rows = |patch: u32, merge: u32, pixels: u64| {
            let merge_rows = mul(u64::from(merge), u64::from(merge))?;
            let rows = pixels / mul(u64::from(patch), u64::from(patch))?;
            if rows < merge_rows {
                return Err(format!(
                    "{pixels} pixels hold no {merge} x {merge} merge block"
                ));
            }
            Ok::<_, String>((rows / merge_rows * merge_rows, merge_rows))
        };
        match self {
            Recipe::ImagePatchGrid(r) => {
                positive(&[
                    r.patch.into(),
                    r.temporal_patch.into(),
                    r.merge.into(),
                    r.maximum_pixels,
                ])?;
                let (grid, multiple) = grid_rows(r.patch, r.merge, r.maximum_pixels)?;
                Ok(RecipeRows {
                    columns: patch_columns(r.patch, r.temporal_patch)?,
                    patchify: Some(Patchify::MergeBlocks),
                    multiple,
                    item: grid,
                    grid,
                })
            }
            Recipe::VideoTimestampedGrids(r) => {
                positive(&[
                    r.patch.into(),
                    r.temporal_patch.into(),
                    r.merge.into(),
                    r.maximum_pixels_per_frame,
                    r.maximum_frames.into(),
                ])?;
                let (grid, multiple) = grid_rows(r.patch, r.merge, r.maximum_pixels_per_frame)?;
                let grids = u64::from(r.maximum_frames).div_ceil(u64::from(r.temporal_patch));
                Ok(RecipeRows {
                    columns: patch_columns(r.patch, r.temporal_patch)?,
                    patchify: Some(Patchify::MergeBlocks),
                    multiple,
                    item: mul(grids, grid)?,
                    grid,
                })
            }
            Recipe::AudioLogMel(r) => {
                positive(&[r.mel_bins.into(), r.maximum_frames])?;
                Ok(RecipeRows {
                    columns: r.mel_bins.into(),
                    patchify: None,
                    multiple: 1,
                    item: r.maximum_frames,
                    grid: r.maximum_frames,
                })
            }
            Recipe::ImageCenteredPad(r) => {
                positive(&[r.patch.into(), r.downsample.into()])?;
                if r.maximum_tokens < 4 {
                    return Err("a token budget below 4 holds no cell".into());
                }
                // h * (w + 1) + 2 <= budget gives at most budget - 3 cells
                // (one row of cells); each cell holds downsample^2 patches.
                let cells = u64::from(r.maximum_tokens) - 3;
                let item = mul(
                    cells,
                    mul(u64::from(r.downsample), u64::from(r.downsample))?,
                )?;
                Ok(RecipeRows {
                    columns: patch_columns(r.patch, 1)?,
                    patchify: Some(Patchify::RowMajor),
                    multiple: 1,
                    item,
                    grid: item,
                })
            }
            Recipe::ImageZeroPaddedTokenBudget(r) => {
                positive(&[
                    r.patch.into(),
                    r.temporal_patch.into(),
                    r.merge.into(),
                    r.maximum_tokens,
                ])?;
                let multiple = mul(u64::from(r.merge), u64::from(r.merge))?;
                let item = mul(r.maximum_tokens, multiple)?;
                Ok(RecipeRows {
                    columns: patch_columns(r.patch, r.temporal_patch)?,
                    patchify: Some(Patchify::MergeBlocks),
                    multiple,
                    item,
                    grid: item,
                })
            }
        }
    }
}

/// The order of patch rows and of the features inside each.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Patchify {
    /// Rows `(t, bh, bw, mh, mw)`, features `(c, dt, ph, pw)`: each merge
    /// block's rows consecutive.
    #[serde(rename = "merge-blocks.v1")]
    MergeBlocks,
    /// Rows `(y, x)`, features `(c, py, px)`.
    #[serde(rename = "row-major.v1")]
    RowMajor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub recipes: Vec<Recipe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patchify: Option<Patchify>,
    /// The payload element the host writes.
    pub payload: Element,
    /// Payload columns per row.
    pub columns: u32,
    /// Rows of one encoder run: the kernels' row capacity. Without
    /// `chunking` it is derived: the largest run the recipes' limits and the
    /// sequences allow, rounded up to a multiple of `CAPACITY_GRANULE`.
    pub capacity_rows: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunking: Option<Chunking>,
}

/// Derived capacities round up to whole multiples of this many rows.
pub const CAPACITY_GRANULE: u64 = 1024;

/// How runs are cut when the capacity is chosen rather than derived.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Chunking {
    /// A run packs whole sequences (temporal grids, segments, or the groups
    /// a group op forms) up to `capacity_rows`, a multiple of
    /// `CAPACITY_GRANULE` that holds the largest single one.
    #[serde(rename = "whole-sequences.v1")]
    WholeSequences,
}

/// How rows form attention sequences. Every variant is a struct variant:
/// serde refuses unknown fields only there.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Sequences {
    /// One sequence per media item; a run holds one item.
    Item {},
    /// One sequence per temporal grid; a run holds whole grids.
    TemporalGrid {},
    /// Consecutive segments of at most this many input rows, one per run,
    /// until a group op joins them.
    Segments { input_rows: u32 },
    /// Fixed groups of this many rows (a stage's own sequences only).
    Groups { rows: u32 },
}

/// The checkpoint config values a declaration requires: `values` must equal
/// the members of the object at JSON pointer `pointer` of `file`, compared
/// as JSON values (so `27` and `27.0` differ, as they do in the config).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigPin {
    pub file: String,
    pub pointer: String,
    pub values: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// Prefixed to every tensor name of the weight mapping.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix: String,
    /// Full tensor-name prefix to the safetensors file holding those
    /// tensors, when the checkpoint index does not say. When present, every
    /// tensor matches one entry; the longest wins.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    pub config: Vec<ConfigPin>,
}

impl Checkpoint {
    /// The file declared to hold `tensor`.
    pub fn file(&self, tensor: &str) -> Option<&str> {
        self.files
            .iter()
            .filter(|(prefix, _)| tensor.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, file)| file.as_str())
    }
}

/// How the per-rank encoder image is laid out from the weight mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ImageLayout {
    /// Each GEMM's weights stacked in program order, then its biases
    /// (zeros for a GEMM without one), then the remaining tensors by name.
    #[serde(rename = "program-order-stacked.v1")]
    ProgramOrderStacked,
    /// One view per mapped tensor, in stage and role order, 4096-aligned.
    #[serde(rename = "view-order.v1")]
    ViewOrder,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    pub schema: Schema,
    /// The owning model's directory name under `models/`.
    pub model: String,
    /// The frontend's `encoders[].name`.
    pub name: String,
    pub precision: Precision,
    pub checkpoint: Checkpoint,
    pub input: Input,
    pub sequences: Sequences,
    pub image_layout: ImageLayout,
    pub stages: Vec<Stage>,
    pub output: Stage<Layout>,
    /// The admissible parallel policies, `replicated` first.
    pub parallel: Vec<ParallelPolicy>,
}

/// Parses a declaration, decodes its model-namespaced variants with the
/// environment's registry, and validates it.
pub fn parse(text: &str, environment: &Environment) -> Result<Declaration, Error> {
    let mut declaration: Declaration =
        serde_json::from_str(text).map_err(|error| Error(error.to_string()))?;
    declaration.resolve(environment)?;
    declaration.validate(environment)?;
    Ok(declaration)
}

fn word(text: &str) -> bool {
    !text.is_empty()
        && !text.starts_with('-')
        && text
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// One stage (or the output layout) with the rows it reads.
struct Step<'a> {
    at: String,
    component: &'a dyn Component,
    weights: &'a Weights,
    input: Boundary,
}

/// Checks that `weights` maps exactly `roles`, each source fit for its role.
fn check_roles(
    roles: &[Role],
    weights: &Weights,
    place: Place,
    blocks: Option<u32>,
) -> Result<(), String> {
    let names: BTreeSet<&str> = roles.iter().map(|role| role.name.as_str()).collect();
    if let Some(unknown) = weights.keys().find(|role| !names.contains(role.as_str())) {
        return Err(format!(
            "weights: unknown role `{unknown}`; expected {}",
            names.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    for role in roles {
        let fail = |error: String| format!("weights.{}: {error}", role.name);
        let source = weights
            .get(&role.name)
            .ok_or_else(|| format!("weights: role `{}` is not mapped", role.name))?;
        source.check(place, role).map_err(fail)?;
        match (&role.blocks, source.blocks()) {
            (None, None) => {}
            (Some(expected), Some(set)) => {
                if set.canonical_members(blocks.unwrap_or(0)).map_err(fail)? != *expected {
                    return Err(fail(format!(
                        "block set `{}` is not the blocks that have it",
                        set.as_str()
                    )));
                }
            }
            (Some(_), None) => {
                return Err(fail("some blocks have none; name the ones that do".into()));
            }
            (None, Some(_)) => {
                return Err(fail(
                    "only a role some blocks lack names a block set".into(),
                ));
            }
        }
    }
    Ok(())
}

impl Declaration {
    /// Decodes every model-namespaced variant with the environment's
    /// registry, refusing ids no provider of the owner registers.
    pub fn resolve(&mut self, environment: &Environment) -> Result<(), Error> {
        self.precision.validate().map_err(Error)?;
        let registry = &environment.registry;
        let owner = environment.owner.as_str();
        for (index, stage) in self.stages.iter_mut().enumerate() {
            let fail = |error: String| Error(format!("stages[{index}]: {error}"));
            match &mut stage.op {
                StageOp::Unresolved(unresolved) => {
                    stage.op =
                        StageOp::Registered(registry.stage(owner, unresolved).map_err(fail)?);
                }
                StageOp::Transformer(transformer) => {
                    if let Some(RotarySlot::Unresolved(unresolved)) = &transformer.params().rotary {
                        let rotary = registry.rotary(owner, unresolved).map_err(fail)?;
                        transformer.params_mut().rotary = Some(RotarySlot::Registered(rotary));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn context(&self) -> Context<'_> {
        Context {
            model: &self.model,
            precision: self.precision,
            patchify: self.input.patchify,
            sequences: &self.sequences,
            capacity_rows: u64::from(self.input.capacity_rows),
        }
    }

    /// The rows the payload enters as, sized by the recipes' limits.
    fn input_boundary(&self) -> Result<Boundary, String> {
        let input = &self.input;
        if input.recipes.is_empty() {
            return Err("input: no recipe".into());
        }
        let mut run_rows = 0;
        let mut multiple = None;
        for (index, recipe) in input.recipes.iter().enumerate() {
            let tag = recipe.tag();
            if input.recipes[..index]
                .iter()
                .any(|other| other.tag() == tag)
            {
                return Err(format!("input: recipe {tag} is repeated"));
            }
            let rows = recipe.rows().map_err(|error| format!("input: {error}"))?;
            if rows.columns != u64::from(input.columns) {
                return Err(format!(
                    "input: recipe {tag} emits {} columns, the input declares {}",
                    rows.columns, input.columns
                ));
            }
            if rows.patchify != input.patchify {
                return Err(format!(
                    "input: recipe {tag} emits {} rows, not the declared patchify",
                    rows.patchify
                        .map_or("non-patch".into(), |p| components::op(&p))
                ));
            }
            if *multiple.get_or_insert(rows.multiple) != rows.multiple {
                return Err(format!("input: recipe {tag} groups rows differently"));
            }
            let run = match self.sequences {
                Sequences::Item {} => rows.item,
                Sequences::TemporalGrid {} => rows.grid,
                Sequences::Segments { input_rows } => rows.item.min(u64::from(input_rows)),
                Sequences::Groups { .. } => {
                    return Err("sequences: groups are a stage's own sequences".into());
                }
            };
            run_rows = run_rows.max(run);
        }
        let capacity = u64::from(input.capacity_rows);
        match input.chunking {
            None => {
                let derived = run_rows.div_ceil(CAPACITY_GRANULE) * CAPACITY_GRANULE;
                if capacity != derived {
                    return Err(format!(
                        "input: capacity_rows {capacity} is not the derived {derived}: the \
                         {run_rows}-row run rounded up to {CAPACITY_GRANULE} rows (or declare \
                         `chunking`)"
                    ));
                }
            }
            Some(Chunking::WholeSequences) => {
                if matches!(self.sequences, Sequences::Item {}) {
                    return Err("input: one item is one run; derive the capacity".into());
                }
                if capacity % CAPACITY_GRANULE != 0 || capacity < run_rows {
                    return Err(format!(
                        "input: capacity_rows {capacity} is not a multiple of \
                         {CAPACITY_GRANULE} holding the {run_rows}-row sequence"
                    ));
                }
            }
        }
        Ok(Boundary {
            data: Data::Rows {
                width: input.columns,
                element: input.payload,
            },
            run_rows,
            multiple: multiple.unwrap_or(1),
            grid: input.patchify.is_some(),
            segmented: matches!(self.sequences, Sequences::Segments { .. }),
        })
    }

    /// Every stage, then the output layout, with the rows it reads.
    fn steps(&self) -> Result<Vec<Step<'_>>, Error> {
        let context = self.context();
        let capacity = u64::from(self.input.capacity_rows);
        let mut rows = self.input_boundary().map_err(Error)?;
        let mut steps = Vec::new();
        let components = self
            .stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let component = stage
                    .op
                    .component()
                    .map_err(|error| Error(format!("stages[{index}]: {error}")))?;
                let at = format!("stages[{index}] ({})", component.id());
                Ok((at, component, &stage.weights, false))
            })
            .chain(std::iter::once(Ok((
                format!("output ({})", components::Variant::id(&self.output.op)),
                &self.output.op as &dyn Component,
                &self.output.weights,
                true,
            ))));
        for step in components {
            let (at, component, weights, output) = step?;
            let fail = |error: String| Error(format!("{at}: {error}"));
            if rows.run_rows > capacity {
                return Err(fail(format!(
                    "a run reaches {} rows here, above capacity_rows {capacity}",
                    rows.run_rows
                )));
            }
            let rounded = matches!(
                rows.data,
                Data::Rows {
                    element: Element::Bf16,
                    ..
                }
            );
            if self.precision == Precision::F32Pairs && rounded && !output {
                return Err(fail(
                    "reads rows already rounded to BF16; under f32-pairs.v1 rows round once, \
                     into the language model"
                        .into(),
                ));
            }
            context
                .admit(&component.id(), component.policies())
                .map_err(fail)?;
            let output = component.output(&context, &rows).map_err(fail)?;
            steps.push(Step {
                at,
                component,
                weights,
                input: rows,
            });
            rows = output;
        }
        Ok(steps)
    }

    /// The rows each stage reads, in stage order, then the rows the output
    /// layout reads.
    pub fn inputs(&self) -> Result<Vec<Boundary>, Error> {
        Ok(self.steps()?.into_iter().map(|step| step.input).collect())
    }

    /// Refuses a declaration of another owner, unregistered variants, shape,
    /// row and capacity mismatches between stages, weight mappings that do
    /// not fit the components' roles, and precision or parallel policy
    /// violations.
    pub fn validate(&self, environment: &Environment) -> Result<(), Error> {
        self.precision.validate().map_err(Error)?;
        if self.model != environment.owner {
            return Err(Error(format!(
                "declares model {} but is loaded for model {}",
                self.model, environment.owner
            )));
        }
        self.check_header().map_err(Error)?;
        self.check_admission(environment).map_err(Error)?;
        let context = self.context();
        for step in self.steps()? {
            let fail = |error: String| Error(format!("{}: {error}", step.at));
            let place = Place {
                per_block: step.component.blocks().is_some(),
                precision: self.precision,
            };
            let roles = step.component.roles(&context, &step.input).map_err(fail)?;
            check_roles(&roles, step.weights, place, step.component.blocks()).map_err(fail)?;
        }
        self.check_parallel().map_err(Error)?;
        self.check_files().map_err(Error)?;
        Ok(())
    }

    /// Every registered variant was decoded by a provider this
    /// environment's registry holds, whatever environment resolved it.
    fn check_admission(&self, environment: &Environment) -> Result<(), String> {
        for (index, stage) in self.stages.iter().enumerate() {
            let registered = match &stage.op {
                StageOp::Registered(registered) => {
                    Some((registered.provider, registered.component.id()))
                }
                StageOp::Transformer(transformer) => match &transformer.params().rotary {
                    Some(RotarySlot::Registered(registered)) => {
                        Some((registered.provider, registered.component.id()))
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some((provider, id)) = registered
                && !environment.registry.admits(provider, &id)
            {
                return Err(format!(
                    "stages[{index}]: `{id}` was decoded by a provider this environment does \
                     not register"
                ));
            }
        }
        Ok(())
    }

    fn check_header(&self) -> Result<(), String> {
        if self.model.is_empty()
            || !self
                .model
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        {
            return Err(format!(
                "model `{}` is not a model directory name",
                self.model
            ));
        }
        if !word(&self.name) {
            return Err(format!(
                "name `{}` is not lowercase words joined by `-`",
                self.name
            ));
        }
        if self.precision == Precision::F32Pairs && self.input.payload != Element::F32 {
            return Err("f32-pairs.v1 takes an F32 payload".into());
        }
        if self.input.columns == 0 || self.input.capacity_rows == 0 {
            return Err("input: `columns` and `capacity_rows` are positive".into());
        }
        match self.sequences {
            Sequences::Segments { input_rows: 0 } => {
                return Err("a segment has at least one row".into());
            }
            Sequences::TemporalGrid {} if self.input.patchify != Some(Patchify::MergeBlocks) => {
                return Err("temporal-grid sequences need merge-block rows".into());
            }
            _ => {}
        }
        if self.stages.is_empty() {
            return Err("no stage".into());
        }
        if self.checkpoint.config.is_empty() {
            return Err("checkpoint: no config pin".into());
        }
        for (index, pin) in self.checkpoint.config.iter().enumerate() {
            if pin.values.is_empty() || !(pin.pointer.is_empty() || pin.pointer.starts_with('/')) {
                return Err(format!(
                    "checkpoint.config[{index}]: an empty pin or a malformed pointer"
                ));
            }
            if self.checkpoint.config[..index]
                .iter()
                .any(|other| (&other.file, &other.pointer) == (&pin.file, &pin.pointer))
            {
                return Err(format!(
                    "checkpoint.config[{index}]: {} {} is pinned twice",
                    pin.file, pin.pointer
                ));
            }
        }
        Ok(())
    }

    fn check_parallel(&self) -> Result<(), String> {
        if self.parallel.first() != Some(&ParallelPolicy::Replicated) {
            return Err("parallel: `replicated` comes first".into());
        }
        for (index, policy) in self.parallel.iter().enumerate() {
            if self.parallel[..index].contains(policy) {
                return Err(format!("parallel: {} is repeated", components::op(policy)));
            }
            if *policy == ParallelPolicy::SequenceSplit {
                let split = matches!(self.sequences, Sequences::Item {})
                    && self.stages.iter().all(|stage| {
                        stage.op.component().is_ok_and(|component| {
                            let class = component.parallel();
                            class != ParallelClass::RowHalo
                                && class != ParallelClass::Permuting
                                && !component.joins_segments()
                        }) && !matches!(&stage.op, StageOp::Transformer(t) if t.params().sequences.is_some())
                    });
                if !split {
                    return Err(
                        "parallel: sequence-split needs one sequence per item and no \
                         convolution, reorder, group op or stage-level sequences"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }

    fn check_files(&self) -> Result<(), String> {
        if self.checkpoint.files.is_empty() {
            return Ok(());
        }
        for tensor in self.tensors() {
            if self.checkpoint.file(&tensor).is_none() {
                return Err(format!("checkpoint.files: no file holds `{tensor}`"));
            }
        }
        Ok(())
    }

    fn components(&self) -> Vec<(Option<&dyn Component>, &Weights)> {
        self.stages
            .iter()
            .map(|stage| (stage.op.component().ok(), &stage.weights))
            .chain(std::iter::once((
                Some(&self.output.op as &dyn Component),
                &self.output.weights,
            )))
            .collect()
    }

    /// Every checkpoint tensor the weight mapping reads, by full name.
    pub fn tensors(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for (component, weights) in self.components() {
            let blocks: Vec<u32> = (0..component.and_then(|c| c.blocks()).unwrap_or(0)).collect();
            for source in weights.values() {
                source.names(&self.checkpoint.prefix, &blocks, &mut out);
            }
        }
        out
    }

    /// Checks the weight mapping against checkpoint metadata: every tensor
    /// exists in its declared file, every transform applies, and every role
    /// gets exactly the shape and element it reads, block by block.
    pub fn bind(&self, lookup: &dyn Fn(&str) -> Option<TensorInfo>) -> Result<(), Error> {
        for tensor in self.tensors() {
            let info = lookup(&tensor)
                .ok_or_else(|| Error(format!("the checkpoint has no `{tensor}`")))?;
            if let (Some(expected), Some(file)) = (self.checkpoint.file(&tensor), &info.file)
                && expected != file
            {
                return Err(Error(format!(
                    "`{tensor}` is in {file}, the declaration says {expected}"
                )));
            }
        }
        let context = self.context();
        for step in self.steps()? {
            let roles = step
                .component
                .roles(&context, &step.input)
                .map_err(|error| Error(format!("{}: {error}", step.at)))?;
            for role in roles {
                let source = &step.weights[&role.name];
                let blocks = match (&role.blocks, step.component.blocks()) {
                    (Some(blocks), _) => blocks.clone(),
                    (None, Some(blocks)) => (0..blocks).collect(),
                    (None, None) => vec![0],
                };
                for block in blocks {
                    source
                        .bind(&role, &self.checkpoint.prefix, block, lookup)
                        .map_err(|error| {
                            Error(format!(
                                "{}: weights.{} block {block}: {error}",
                                step.at, role.name
                            ))
                        })?;
                }
            }
        }
        Ok(())
    }

    /// One item's rows after each stage and after the output layout, for an
    /// item entering as `item` (input rows, and the patch grid if any).
    /// Segments run separately until a stage joins them.
    pub fn trace(&self, item: Item) -> Result<Vec<Item>, Error> {
        let mut parts = match self.sequences {
            Sequences::Segments { input_rows } => {
                let size = u64::from(input_rows);
                (0..item.rows.div_ceil(size))
                    .map(|index| Item {
                        rows: size.min(item.rows - index * size),
                        grid: None,
                    })
                    .collect()
            }
            _ => vec![item],
        };
        let mut trace = Vec::new();
        for (index, (component, _)) in self.components().into_iter().enumerate() {
            let at = if index == self.stages.len() {
                "output".to_owned()
            } else {
                format!("stages[{index}]")
            };
            let component = component.ok_or_else(|| Error(format!("{at} is not registered")))?;
            let fail = |error: String| Error(format!("{at} ({}): {error}", component.id()));
            if component.joins_segments() && parts.len() > 1 {
                let rows = parts.iter().map(|part| part.rows).sum();
                parts = vec![Item { rows, grid: None }];
            }
            parts = parts
                .into_iter()
                .map(|part| component.item(part))
                .collect::<Result<_, _>>()
                .map_err(fail)?;
            trace.push(Item {
                rows: parts.iter().map(|part| part.rows).sum(),
                grid: if parts.len() == 1 {
                    parts[0].grid
                } else {
                    None
                },
            });
        }
        Ok(trace)
    }

    /// Checks the pins of `file` against that file's parsed contents.
    pub fn check_config(&self, file: &str, config: &Value) -> Result<(), Error> {
        let pins: Vec<&ConfigPin> = self
            .checkpoint
            .config
            .iter()
            .filter(|pin| pin.file == file)
            .collect();
        if pins.is_empty() {
            return Err(Error(format!("{}: no pin for {file}", self.name)));
        }
        for pin in pins {
            let object = config
                .pointer(&pin.pointer)
                .ok_or_else(|| Error(format!("{}: {file} has no `{}`", self.name, pin.pointer)))?;
            for (key, expected) in &pin.values {
                let found = object.get(key).unwrap_or(&Value::Null);
                if found != expected {
                    return Err(Error(format!(
                        "{}: {file} {}/{key} is {found}, the declaration pins {expected}",
                        self.name, pin.pointer
                    )));
                }
            }
        }
        Ok(())
    }

    /// The declaration as canonical JSON: the typed declaration (registered
    /// variants included) serialized with object keys sorted at every level
    /// and no whitespace.
    pub fn canonical_json(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(Value::Null);
        let mut out = String::new();
        canonical(&value, &mut out);
        out
    }

    /// Whether an item can take several runs (`sequences` other than
    /// `item`). The runtime then stages each run's side inputs in command
    /// order into regions every run shares, re-uploading the live row count
    /// as it changes, and records one program over an item's runs.
    pub fn staged(&self) -> bool {
        !matches!(self.sequences, Sequences::Item {})
    }

    /// SHA-256 of `canonical_json`, lowercase hex: the declaration's identity
    /// for bundles and media cache keys.
    pub fn digest(&self) -> String {
        Sha256::digest(self.canonical_json().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                canonical(&object[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_tags_are_their_serde_tags() {
        let recipes = [
            r#"{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2, "merge": 2, "maximum_pixels": 1}"#,
            r#"{"tag": "video.timestamped-grids.v1", "patch": 16, "temporal_patch": 2, "merge": 2, "maximum_pixels_per_frame": 1, "maximum_frames": 1}"#,
            r#"{"tag": "audio.log-mel.v1", "mel_bins": 1, "maximum_frames": 1}"#,
            r#"{"tag": "image.centered-pad.v1", "patch": 1, "downsample": 1, "maximum_tokens": 4}"#,
            r#"{"tag": "image.zero-padded-token-budget.v1", "patch": 14, "temporal_patch": 2, "merge": 2, "maximum_tokens": 1}"#,
        ];
        for text in recipes {
            let recipe: Recipe = serde_json::from_str(text).unwrap();
            assert_eq!(serde_json::to_value(&recipe).unwrap()["tag"], recipe.tag());
        }
    }

    /// The token budget bounds the merge blocks of one grid, and so the rows.
    #[test]
    fn token_budget_grids_hold_their_budget_of_merge_blocks() {
        use crate::geometry::Media;
        let recipe = Recipe::ImageZeroPaddedTokenBudget(TokenBudgetGrid {
            patch: 14,
            temporal_patch: 2,
            merge: 2,
            maximum_tokens: 8000,
        });
        let rows = recipe.rows().unwrap();
        assert_eq!(
            (rows.columns, rows.multiple, rows.item, rows.grid),
            (1176, 4, 32_000, 32_000)
        );
        assert_eq!(rows.patchify, Some(Patchify::MergeBlocks));
        assert_eq!(
            recipe.admit(Media::Grid([1, 160, 200])).unwrap().rows,
            32_000
        );
        for grid in [[1, 160, 202], [2, 8, 8], [1, 7, 8], [1, 0, 8]] {
            assert!(recipe.admit(Media::Grid(grid)).is_err(), "{grid:?}");
        }
    }
}
