//! Recurrent layers through the linear-attention engine (`linear`: the
//! causal convolution, gated delta rule and gated norm) and the
//! per-channel-decay engine (`linear::channel_decay`: its short convolution,
//! delta rule and gated norm), from the entries a model instantiates.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, RecurrentPhase, RecurrentQuery, Refusal,
};
use kernel_core::per_channel_decay_delta_rule::{
    PerChannelDecayDeltaRuleQuery, PerChannelDecayOp, QkvConvs,
};
use kernel_core::properties::Invariance;
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers, RecurrentSlots};

use super::check_plan;
use super::select::{Candidate, choose};
use crate::linear::channel_decay::{self, Query as DecayQuery};
use crate::linear::{self, Bucket, CHUNK, Stage};

/// The model's recurrent entries of both engines.
#[derive(Clone, Debug, Default)]
pub(super) struct Entries {
    pub linear: Vec<linear::Entry>,
    pub channel_decay: Vec<channel_decay::Entry>,
}

pub(super) fn causal_conv1d(
    entries: &Entries,
    op: &RecurrentQuery,
    items: u32,
    need: Need,
) -> Result<Prepared<ConvBuffers<View>>, Refused> {
    let RecurrentQuery::CausalConv1d {
        channels,
        kernel,
        phase,
        ..
    } = *op
    else {
        return Err(Refused::Rule("the op is not a causal convolution".into()));
    };
    linear::limits(*op)?;
    let shape = ConvShape {
        channels: u64::from(channels),
        weights: vec![product(&[u64::from(channels), u64::from(kernel), 2])?],
        null_weights: false,
        state_bytes: product(&[op.state_elements(), 2])?,
        phase,
    };
    let (stages, guarantees) = linear_stages(entries, op, need)?;
    let info = info("linear.conv", Some(phase), &stages, guarantees);
    RecurrentOp::prepare(stages, shape, need, Some(items), false, info)
}

pub(super) fn gated_delta_rule(
    entries: &Entries,
    op: &RecurrentQuery,
    items: u32,
    need: Need,
) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
    let RecurrentQuery::GatedDeltaRule {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        phase,
        ..
    } = *op
    else {
        return Err(Refused::Rule("the op is not a gated delta rule".into()));
    };
    linear::limits(*op)?;
    let shape = DeltaShape {
        qkv_columns: product(&[2, u64::from(key_heads), u64::from(key_dim)])?
            .checked_add(product(&[u64::from(value_heads), u64::from(value_dim)])?)
            .ok_or(SIZES)?,
        decay_columns: u64::from(value_heads),
        beta_columns: u64::from(value_heads),
        a_log_bytes: u64::from(value_heads) * 4,
        dt_bias_bytes: u64::from(value_heads) * 4,
        output_columns: product(&[u64::from(value_heads), u64::from(value_dim)])?,
        state_bytes: product(&[op.state_elements(), 4])?,
        phase,
    };
    let (stages, guarantees) = linear_stages(entries, op, need)?;
    let info = info("linear.delta-rule", Some(phase), &stages, guarantees);
    RecurrentOp::prepare(stages, shape, need, Some(items), false, info)
}

pub(super) fn gated_rmsnorm(
    entries: &Entries,
    op: &RecurrentQuery,
    need: Need,
) -> Result<Prepared<GatedNormBuffers<View>>, Refused> {
    let RecurrentQuery::GatedRmsNorm {
        width, head_dim, ..
    } = *op
    else {
        return Err(Refused::Rule("the op is not a gated RMSNorm".into()));
    };
    linear::limits(*op)?;
    let shape = NormShape {
        width: u64::from(width),
        head_dim: u64::from(head_dim),
    };
    // The engines share the norm template; an entry of either serves it.
    // An engine none of whose entries claims the exactness the op needs
    // does not serve it; of those left, two are a tie.
    let mut engines = Vec::new();
    let mut inexact = None;
    if entries.linear.iter().any(|entry| entry.query == *op) {
        match linear_stages(entries, op, need) {
            Ok((stages, guarantees)) => {
                engines.push(engine("linear.gated-norm", stages, guarantees));
            }
            Err(refused @ Refused::Inexact { .. }) => inexact = Some(refused),
            Err(refused) => return Err(refused),
        }
    }
    let decay = DecayQuery::Norm(*op);
    if entries
        .channel_decay
        .iter()
        .any(|entry| entry.query == decay)
    {
        match decay_stages(entries, decay, need) {
            Ok((stages, guarantees)) => {
                engines.push(engine("channel-decay.gated-norm", stages, guarantees))
            }
            Err(refused @ Refused::Inexact { .. }) => inexact = inexact.or(Some(refused)),
            Err(refused) => return Err(refused),
        }
    }
    if let (true, Some(refused)) = (engines.is_empty(), inexact) {
        return Err(refused);
    }
    let (name, stages, guarantees) = choose("gated norm", engines, need)?.entry;
    let info = info(name, None, &stages, guarantees);
    RecurrentOp::prepare(stages, shape, need, None, false, info)
}

pub(super) fn short_conv(
    entries: &Entries,
    op: &PerChannelDecayOp,
    items: u32,
    need: Need,
) -> Result<Prepared<ConvBuffers<View>>, Refused> {
    let PerChannelDecayOp {
        query: ref op,
        plan_at_capacity,
    } = *op;
    let PerChannelDecayDeltaRuleQuery::ShortConv {
        heads,
        head_dim,
        kernel,
        convs,
        phase,
        ..
    } = *op
    else {
        return Err(Refused::Rule("the op is not a short convolution".into()));
    };
    channel_decay::limits(*op)?;
    let part = product(&[u64::from(heads), u64::from(head_dim), u64::from(kernel), 4])?;
    let shape = ConvShape {
        channels: 3 * u64::from(heads) * u64::from(head_dim),
        weights: match convs {
            QkvConvs::Fused => vec![3 * part],
            QkvConvs::Separate => vec![part; 3],
        },
        null_weights: convs == QkvConvs::Fused,
        state_bytes: op.state_bytes(),
        phase,
    };
    let (stages, guarantees) = decay_stages(entries, DecayQuery::Rule(*op), need)?;
    let info = info("channel-decay.conv", Some(phase), &stages, guarantees);
    RecurrentOp::prepare(stages, shape, need, Some(items), plan_at_capacity, info)
}

pub(super) fn channel_decay_delta_rule(
    entries: &Entries,
    op: &PerChannelDecayOp,
    items: u32,
    need: Need,
) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
    let PerChannelDecayOp {
        query: ref op,
        plan_at_capacity,
    } = *op;
    let PerChannelDecayDeltaRuleQuery::DeltaRule {
        heads,
        head_dim,
        phase,
        ..
    } = *op
    else {
        return Err(Refused::Rule(
            "the op is not a per-channel-decay delta rule".into(),
        ));
    };
    channel_decay::limits(*op)?;
    let width = u64::from(heads) * u64::from(head_dim);
    let shape = DeltaShape {
        qkv_columns: 3 * width,
        decay_columns: width,
        beta_columns: u64::from(heads),
        a_log_bytes: u64::from(heads) * 4,
        dt_bias_bytes: width * 4,
        output_columns: width,
        state_bytes: op.state_bytes(),
        phase,
    };
    let (stages, guarantees) = decay_stages(entries, DecayQuery::Rule(*op), need)?;
    let info = info("channel-decay.delta-rule", Some(phase), &stages, guarantees);
    RecurrentOp::prepare(stages, shape, need, Some(items), plan_at_capacity, info)
}

/// An entry of either engine.
#[derive(Clone, Debug)]
enum Kernel {
    Linear(linear::Entry),
    ChannelDecay(channel_decay::Entry),
}

impl Kernel {
    fn symbol(&self) -> &str {
        match self {
            Kernel::Linear(entry) => &entry.symbol,
            Kernel::ChannelDecay(entry) => &entry.symbol,
        }
    }

    fn params(&self) -> &[EntryParam] {
        match self {
            Kernel::Linear(entry) => &entry.params,
            Kernel::ChannelDecay(entry) => &entry.params,
        }
    }

    /// What the entry launches: its symbol, query, stage and argument
    /// shape. Its profile and claims do not reach the launch.
    fn launch(&self) -> String {
        match self {
            Kernel::Linear(entry) => {
                let linear::Entry {
                    profile: _,    // filters catalogs; no launch reads it
                    symbol,        // the launched symbol
                    query,         // selects the op; its code and plans
                    stage,         // the op's form
                    params,        // the ABI
                    invariance: _, // claims
                } = entry;
                format!("linear {symbol} {query:?} {stage:?} {params:?}")
            }
            Kernel::ChannelDecay(entry) => {
                let channel_decay::Entry {
                    profile: _, // filters catalogs; no launch reads it
                    symbol,     // the launched symbol
                    query,      // selects the op; its code and plans
                    stage,      // the op's form
                    params,     // the ABI
                    schedule,   // the prefill stages' code and plans
                } = entry;
                format!("channel-decay {symbol} {query:?} {stage:?} {schedule:?} {params:?}")
            }
        }
    }

    fn stage(&self) -> Stage {
        match self {
            Kernel::Linear(entry) => entry.stage,
            Kernel::ChannelDecay(entry) => entry.stage,
        }
    }

    fn capacity(&self) -> Bucket {
        match self {
            Kernel::Linear(entry) => entry.capacity(),
            Kernel::ChannelDecay(entry) => entry.capacity(),
        }
    }

    fn dynamic_shared_bytes(&self) -> u32 {
        match self {
            Kernel::Linear(entry) => entry.dynamic_shared_bytes(),
            Kernel::ChannelDecay(entry) => entry.dynamic_shared_bytes(),
        }
    }

    /// The entry's plan for `bucket`, as the models plan it: for the
    /// bucket's own rows and sequences.
    fn plan(&self, bucket: Bucket) -> Result<LaunchPlan, Refusal> {
        match self {
            Kernel::Linear(entry) => entry.plan(bucket, bucket.rows, bucket.sequences),
            Kernel::ChannelDecay(entry) => entry.plan(bucket, bucket.rows, bucket.sequences),
        }
    }

    fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        match self {
            Kernel::Linear(entry) => entry.pack_arguments(values),
            Kernel::ChannelDecay(entry) => entry.pack_arguments(values),
        }
    }
}

/// An engine's stages for an op, named `name`, claiming `guarantees`.
fn engine(
    name: &'static str,
    stages: Vec<Kernel>,
    guarantees: Invariance,
) -> Candidate<(&'static str, Vec<Kernel>, Invariance)> {
    let symbols = stages
        .iter()
        .map(Kernel::symbol)
        .collect::<Vec<_>>()
        .join("+");
    let launches = stages.iter().map(Kernel::launch).collect::<Vec<_>>();
    let launches = format!("{name} {launches:?}");
    Candidate::new((name, stages, guarantees), symbols, launches, guarantees)
}

/// The linear-attention entries of `op`, in launch order: the prefill
/// The linear-attention entries of `op`, in launch order (the prefill
/// delta rule's prepare then scan, or the op's one entry), and the claims
/// they make together.
fn linear_stages(
    entries: &Entries,
    op: &RecurrentQuery,
    need: Need,
) -> Result<(Vec<Kernel>, Invariance), Refused> {
    let stages = stages(
        entries
            .linear
            .iter()
            .filter(|entry| entry.query == *op)
            .map(|entry| (Kernel::Linear(entry.clone()), entry.invariance)),
        need,
    )?;
    let guarantees = stages
        .iter()
        .map(|&(_, claims)| claims)
        .reduce(Invariance::meet)
        .unwrap_or(Invariance::NONE);
    Ok((
        stages.into_iter().map(|(kernel, _)| kernel).collect(),
        guarantees,
    ))
}

/// The per-channel-decay entries of `op`, in launch order, and the claims
/// they make together (`channel_decay::Entry::invariance`).
fn decay_stages(
    entries: &Entries,
    op: DecayQuery,
    need: Need,
) -> Result<(Vec<Kernel>, Invariance), Refused> {
    let stages = stages(
        entries
            .channel_decay
            .iter()
            .filter(|entry| entry.query == op)
            .map(|entry| (Kernel::ChannelDecay(entry.clone()), entry.invariance())),
        need,
    )?;
    let guarantees = stages
        .iter()
        .map(|&(_, claims)| claims)
        .reduce(Invariance::meet)
        .unwrap_or(Invariance::NONE);
    Ok((
        stages.into_iter().map(|(kernel, _)| kernel).collect(),
        guarantees,
    ))
}

/// The stages of one op, in launch order, each with what its registrations
/// declare, whatever order the model registered them (`found`, each with
/// its declared claims) in. Each stage is chosen through
/// [`select::choose`](super::select::choose): of its registrations, those
/// that claim `need.exact` remain, those that launch the same are one (the
/// op claims what all of them declare), and two that launch differently
/// are a tie.
fn stages(
    found: impl Iterator<Item = (Kernel, Invariance)>,
    need: Need,
) -> Result<Vec<(Kernel, Invariance)>, Refused> {
    let mut by_stage: Vec<(u8, Vec<(Kernel, Invariance)>)> = Vec::new();
    for (kernel, claims) in found {
        let order = match kernel.stage() {
            Stage::Prepare => 0,
            Stage::Single => 1,
            Stage::Scan => 2,
        };
        match by_stage.iter_mut().find(|(at, _)| *at == order) {
            Some((_, kernels)) => kernels.push((kernel, claims)),
            None => by_stage.push((order, vec![(kernel, claims)])),
        }
    }
    by_stage.sort_by_key(|&(order, _)| order);
    let order: Vec<u8> = by_stage.iter().map(|&(order, _)| order).collect();
    match order[..] {
        [1] | [0, 2] => {}
        [] => {
            return Err(Refused::Rule(
                "no SM120 recurrent entry serves the query".into(),
            ));
        }
        _ => {
            return Err(Refused::Rule(
                "the SM120 recurrent entries of the query do not form one op".into(),
            ));
        }
    }
    // A stage runs at the op's rows; its entries hold no capacity of their
    // own here (the prepared op checks the capacity).
    let need = Need {
        capacity: 1,
        ..need
    };
    by_stage
        .into_iter()
        .map(|(_, kernels)| {
            let chosen = choose(
                "recurrent",
                kernels.into_iter().map(|(kernel, claims)| {
                    let (symbol, launches) = (kernel.symbol().to_owned(), kernel.launch());
                    Candidate::new(kernel, symbol, launches, claims)
                }),
                need,
            )?;
            Ok((chosen.entry, chosen.guarantees))
        })
        .collect()
}

/// The op's info: `sm120.{name}.{phase}.v1`, the prefill delta rule's
/// chunk length named, every stage's symbol and shared memory.
fn info(
    name: &str,
    phase: Option<RecurrentPhase>,
    stages: &[Kernel],
    guarantees: Invariance,
) -> OpInfo {
    let phase = match phase {
        None => String::new(),
        Some(RecurrentPhase::Prefill) => ".prefill".into(),
        Some(RecurrentPhase::Decode) => ".decode".into(),
        Some(RecurrentPhase::Verify { max_rows }) => format!(".verify{max_rows}"),
    };
    let chunk = if stages.len() > 1 {
        format!("[chunk={CHUNK}]")
    } else {
        String::new()
    };
    OpInfo::new(
        format!("sm120.{name}{phase}.v1{chunk}"),
        guarantees,
        stages
            .iter()
            .map(|kernel| (kernel.symbol().to_owned(), kernel.dynamic_shared_bytes()))
            .collect(),
    )
}

/// What an entry parameter binds, resolved from its name when the op is
/// prepared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Input,
    InputStride,
    Weight(usize),
    State,
    Output,
    RowOffsets,
    StartPositions,
    SlotIds,
    Initial,
    AnchorIds,
    CheckpointIds,
    LiveSequences,
    Qkv,
    QkvStride,
    Decay,
    DecayStride,
    Beta,
    BetaStride,
    /// One stride of the decay and beta rows (the gated delta rule's `a`
    /// and `b`).
    DecayBetaStride,
    ALog,
    DtBias,
    Workspace,
    Gate,
    GateStride,
    NormWeight,
    LiveRows,
}

/// Rows and sequences of one launch.
#[derive(Clone, Copy)]
struct Step {
    rows: u32,
    /// The launch's sequences (`Counts::items`), which its per-sequence
    /// arrays hold; the plan's may be the op's items.
    sequences: u32,
    /// The workspace bytes of the launch's bucket.
    workspace: u64,
}

/// The sizes of an op's roles and how a role record binds them.
trait Shape: Send + Sync {
    type Buffers;
    /// The role of entry parameter `name`, if the op binds it.
    fn role(&self, name: &str) -> Option<Role>;
    /// The phase of the op's state; `None` for the row-local norm.
    fn phase(&self) -> Option<RecurrentPhase>;
    /// The argument of `role`, its view checked for `step`.
    fn argument(
        &self,
        buffers: &Self::Buffers,
        role: Role,
        step: Step,
    ) -> Result<Argument, Refusal>;
}

/// A recurrent op: its entries in launch order, each with the roles of its
/// parameters.
struct RecurrentOp<S> {
    stages: Vec<(Kernel, Vec<Role>)>,
    shape: S,
    capacity: u32,
    /// The most sequences one launch carries; `None` for the row-local
    /// norm, whose rows are its sequences.
    items: Option<u32>,
    /// Whether every launch plans for `items`, whatever sequences it
    /// carries (`PerChannelDecayOp::plan_at_capacity`).
    at_capacity: bool,
    info: OpInfo,
}

impl<S: Shape + 'static> RecurrentOp<S> {
    fn prepare(
        stages: Vec<Kernel>,
        shape: S,
        need: Need,
        items: Option<u32>,
        at_capacity: bool,
        mut info: OpInfo,
    ) -> Result<Prepared<S::Buffers>, Refused>
    where
        RecurrentOp<S>: PreparedOp<S::Buffers>,
    {
        let limit = stages[0].capacity();
        if need.capacity == 0 || need.capacity > limit.rows {
            return Err(Refused::Rule(
                "a recurrent op's capacity is at most its entries' rows".into(),
            ));
        }
        if items.is_some_and(|items| items == 0 || items > limit.sequences) {
            return Err(Refused::Rule(
                "a recurrent op's items are one to its entries' sequences".into(),
            ));
        }
        // SM120's limits at the widest bucket the op can launch: the entry's
        // own checks and the shared memory of every stage. A prefill or
        // verify bucket may carry all its items at any row count (each
        // sequence pads its own chunks), and a workspace grows with rows
        // and sequences, so the widest bucket's covers every launch. Decode
        // takes one row per sequence, so its rows are also its items; the
        // norm's rows are its sequences.
        let widest = match (shape.phase(), items) {
            (Some(RecurrentPhase::Decode), Some(items)) => Bucket {
                rows: need.capacity.min(items),
                sequences: need.capacity.min(items),
            },
            (_, Some(items)) => Bucket {
                rows: need.capacity,
                sequences: items,
            },
            (_, None) => Bucket {
                rows: need.capacity,
                sequences: need.capacity,
            },
        };
        let mut resolved = Vec::with_capacity(stages.len());
        let mut workspace = 0;
        for kernel in stages {
            let plan = kernel.plan(widest)?;
            check_plan(&plan)?;
            workspace = workspace.max(plan.workspace_bytes);
            let roles = kernel
                .params()
                .iter()
                .map(|param| {
                    shape.role(param_name(param)).ok_or_else(|| {
                        Refused::Rule(format!(
                            "{} takes a parameter the op does not bind",
                            kernel.symbol()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            resolved.push((kernel, roles));
        }
        if workspace > 0 {
            info.bytes.push(("workspace".to_owned(), workspace));
        }
        let op = RecurrentOp {
            stages: resolved,
            shape,
            capacity: need.capacity,
            items,
            at_capacity,
            info,
        };
        need.check(&op.info)?;
        Ok(Box::new(op))
    }

    fn launches_of(&self, buffers: &S::Buffers, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal(
                "recurrent rows must be one to the prepared capacity",
            ));
        }
        let sequences = match self.items {
            // The row-local norm takes each row as a sequence.
            None => counts.rows,
            Some(items) if counts.items == 0 || counts.items > items => {
                return Err(Refusal("recurrent items exceed the prepared items"));
            }
            Some(_) => counts.items,
        };
        // An op planned at capacity grids its items whatever the launch
        // carries; its arrays still hold only the launch's sequences.
        let bucket = Bucket {
            rows: counts.rows,
            sequences: match self.items {
                Some(items) if self.at_capacity => items,
                _ => sequences,
            },
        };
        let mut launches = Vec::with_capacity(self.stages.len());
        let mut plans = Vec::with_capacity(self.stages.len());
        for (kernel, _) in &self.stages {
            plans.push(kernel.plan(bucket)?);
        }
        let step = Step {
            rows: bucket.rows,
            sequences,
            workspace: plans
                .iter()
                .map(|plan| plan.workspace_bytes)
                .max()
                .unwrap_or(0),
        };
        for ((kernel, roles), plan) in self.stages.iter().zip(plans) {
            let values = roles
                .iter()
                .map(|&role| self.shape.argument(buffers, role, step))
                .collect::<Result<Vec<_>, _>>()?;
            launches.push(Launch {
                symbol: kernel.symbol().to_owned(),
                rows: counts.rows,
                plan,
                arguments: kernel.pack_arguments(values)?,
                collective: false,
                persistent: false,
            });
        }
        Ok(launches)
    }
}

impl<S: Shape<Buffers = B> + 'static, B> PreparedOp<B> for RecurrentOp<S> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &B,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.launches_of(buffers, counts)
    }
}

fn param_name(param: &EntryParam) -> &str {
    match param {
        EntryParam::DevicePointer { name, .. }
        | EntryParam::U64 { name }
        | EntryParam::U32 { name }
        | EntryParam::I32 { name }
        | EntryParam::TensorMap { name }
        | EntryParam::PointerArray { name, .. } => name,
    }
}

/// The role of a per-sequence parameter.
fn slot_role(name: &str) -> Option<Role> {
    Some(match name {
        "cu_seqlens" | "cu_rows" => Role::RowOffsets,
        "start_positions" => Role::StartPositions,
        "slot_ids" => Role::SlotIds,
        "initial" => Role::Initial,
        "anchor_ids" => Role::AnchorIds,
        "checkpoint_ids" => Role::CheckpointIds,
        "live_sequences" => Role::LiveSequences,
        _ => return None,
    })
}

fn pointer(view: View, bytes: u64) -> Result<Argument, Refusal> {
    view.covers(bytes)?;
    Ok(Argument::DeviceBuffer(view.handle))
}

fn present(view: Option<View>, bytes: u64) -> Result<Argument, Refusal> {
    pointer(
        view.ok_or(Refusal("a recurrent launch lacks a role its phase reads"))?,
        bytes,
    )
}

/// Bytes of `rows` BF16 rows of `columns` elements, `stride` apart.
fn strided(rows: u32, stride: u32, columns: u64) -> Result<u64, Refusal> {
    if u64::from(stride) < columns {
        return Err(Refusal("a recurrent row stride is narrower than its row"));
    }
    product(&[u64::from(rows - 1), u64::from(stride)])?
        .checked_add(columns)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or(SIZES)
}

/// Bytes of `rows` contiguous BF16 rows of `columns` elements.
fn packed(rows: u32, columns: u64) -> Result<u64, Refusal> {
    product(&[u64::from(rows), columns, 2])
}

const SIZES: Refusal = Refusal("recurrent sizes overflow u64");

/// The product of `factors`, refused where it overflows a u64: a size
/// that overflows is refused, never saturated or wrapped.
fn product(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1_u64, |total, &factor| total.checked_mul(factor))
        .ok_or(SIZES)
}

/// The slot argument of `role` for a launch of `phase`.
fn slot_argument(
    slots: &RecurrentSlots<View>,
    phase: RecurrentPhase,
    role: Role,
    step: Step,
) -> Result<Argument, Refusal> {
    let each = u64::from(step.sequences) * 4;
    match role {
        Role::RowOffsets => present(slots.row_offsets, each + 4),
        Role::StartPositions => present(slots.start_positions, each),
        Role::SlotIds => pointer(slots.slot_ids, each),
        Role::Initial => present(slots.initial, each),
        Role::AnchorIds => present(slots.anchor_ids, each),
        Role::CheckpointIds => {
            let RecurrentPhase::Verify { max_rows } = phase else {
                return Err(Refusal("only verify writes checkpoints"));
            };
            present(slots.checkpoint_ids, each * u64::from(max_rows))
        }
        Role::LiveSequences => pointer(slots.live_sequences, 4),
        _ => Err(Refusal("not a recurrent slot role")),
    }
}

struct ConvShape {
    channels: u64,
    /// Bytes of each weight, in role order.
    weights: Vec<u64>,
    /// A fused weight leaves the separate k and v weights null.
    null_weights: bool,
    state_bytes: u64,
    phase: RecurrentPhase,
}

impl Shape for ConvShape {
    type Buffers = ConvBuffers<View>;

    fn phase(&self) -> Option<RecurrentPhase> {
        Some(self.phase)
    }

    fn role(&self, name: &str) -> Option<Role> {
        Some(match name {
            "input" => Role::Input,
            "input_stride" => Role::InputStride,
            "weight" | "weight_q" => Role::Weight(0),
            "weight_k" => Role::Weight(1),
            "weight_v" => Role::Weight(2),
            "state" => Role::State,
            "output" => Role::Output,
            _ => return slot_role(name),
        })
    }

    fn argument(
        &self,
        buffers: &ConvBuffers<View>,
        role: Role,
        step: Step,
    ) -> Result<Argument, Refusal> {
        match role {
            Role::Input => pointer(
                buffers.input,
                strided(step.rows, buffers.input_stride, self.channels)?,
            ),
            Role::InputStride => Ok(Argument::U32(buffers.input_stride)),
            Role::Weight(index) => {
                if buffers.weights.len() != self.weights.len() {
                    return Err(Refusal(
                        "a convolution takes one weight, or its q, k and v weights",
                    ));
                }
                match self.weights.get(index) {
                    Some(&bytes) => pointer(buffers.weights[index], bytes),
                    None if self.null_weights => Ok(Argument::NullDevicePointer),
                    None => Err(Refusal("a convolution entry reads a weight it lacks")),
                }
            }
            Role::State => pointer(buffers.state, self.state_bytes),
            Role::Output => pointer(buffers.output, packed(step.rows, self.channels)?),
            _ => slot_argument(&buffers.slots, self.phase, role, step),
        }
    }
}

struct DeltaShape {
    qkv_columns: u64,
    decay_columns: u64,
    beta_columns: u64,
    a_log_bytes: u64,
    dt_bias_bytes: u64,
    output_columns: u64,
    state_bytes: u64,
    phase: RecurrentPhase,
}

impl Shape for DeltaShape {
    type Buffers = DeltaRuleBuffers<View>;

    fn phase(&self) -> Option<RecurrentPhase> {
        Some(self.phase)
    }

    fn role(&self, name: &str) -> Option<Role> {
        Some(match name {
            "qkv" => Role::Qkv,
            "qkv_stride" => Role::QkvStride,
            "a" | "f" => Role::Decay,
            "f_stride" => Role::DecayStride,
            "b" => Role::Beta,
            "b_stride" => Role::BetaStride,
            "gate_stride" => Role::DecayBetaStride,
            "a_log" => Role::ALog,
            "dt_bias" => Role::DtBias,
            "workspace" => Role::Workspace,
            "state" => Role::State,
            "output" => Role::Output,
            _ => return slot_role(name),
        })
    }

    fn argument(
        &self,
        buffers: &DeltaRuleBuffers<View>,
        role: Role,
        step: Step,
    ) -> Result<Argument, Refusal> {
        let rows = step.rows;
        match role {
            Role::Qkv => pointer(
                buffers.qkv,
                strided(rows, buffers.qkv_stride, self.qkv_columns)?,
            ),
            Role::QkvStride => Ok(Argument::U32(buffers.qkv_stride)),
            Role::Decay => pointer(
                buffers.decay,
                strided(rows, buffers.decay_stride, self.decay_columns)?,
            ),
            Role::DecayStride => Ok(Argument::U32(buffers.decay_stride)),
            Role::Beta => pointer(
                buffers.beta,
                strided(rows, buffers.beta_stride, self.beta_columns)?,
            ),
            Role::BetaStride => Ok(Argument::U32(buffers.beta_stride)),
            Role::DecayBetaStride if buffers.decay_stride == buffers.beta_stride => {
                Ok(Argument::U32(buffers.decay_stride))
            }
            Role::DecayBetaStride => Err(Refusal(
                "the gated delta rule reads its decay and beta rows at one stride",
            )),
            Role::ALog => pointer(buffers.a_log, self.a_log_bytes),
            Role::DtBias => pointer(buffers.dt_bias, self.dt_bias_bytes),
            Role::Workspace => present(buffers.workspace, step.workspace),
            Role::State => pointer(buffers.state, self.state_bytes),
            Role::Output => pointer(buffers.output, packed(rows, self.output_columns)?),
            _ => slot_argument(&buffers.slots, self.phase, role, step),
        }
    }
}

struct NormShape {
    width: u64,
    head_dim: u64,
}

impl Shape for NormShape {
    type Buffers = GatedNormBuffers<View>;

    fn phase(&self) -> Option<RecurrentPhase> {
        None
    }

    fn role(&self, name: &str) -> Option<Role> {
        Some(match name {
            "input" => Role::Input,
            "gate" => Role::Gate,
            "gate_stride" => Role::GateStride,
            "weight" => Role::NormWeight,
            "output" => Role::Output,
            "live_rows" => Role::LiveRows,
            _ => return None,
        })
    }

    fn argument(
        &self,
        buffers: &GatedNormBuffers<View>,
        role: Role,
        step: Step,
    ) -> Result<Argument, Refusal> {
        let rows = packed(step.rows, self.width)?;
        match role {
            Role::Input => pointer(buffers.input, rows),
            Role::Gate => pointer(
                buffers.gate,
                strided(step.rows, buffers.gate_stride, self.width)?,
            ),
            Role::GateStride => Ok(Argument::U32(buffers.gate_stride)),
            Role::NormWeight => pointer(buffers.weight, self.head_dim * 2),
            Role::Output => pointer(buffers.output, rows),
            Role::LiveRows => pointer(buffers.live_rows, 4),
            _ => Err(Refusal("not a gated RMSNorm role")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::linear::{conv_params, delta_params};
    use crate::{Profile, ProfileNames};
    use kernel_core::backend::Backend;
    use kernel_core::contract::{ConvActivation, GateActivation, Validate};
    use kernel_core::properties::Anchors;
    use kernel_core::properties::Chunk;

    const BIG: u64 = 1 << 40;

    fn p(handle: u64) -> Argument {
        Argument::DeviceBuffer(handle)
    }

    fn u(value: u32) -> Argument {
        Argument::U32(value)
    }

    /// Every slot role, handles 7 to 13.
    fn slots() -> RecurrentSlots<View> {
        RecurrentSlots {
            row_offsets: Some(view(7, BIG)),
            start_positions: Some(view(8, BIG)),
            slot_ids: view(9, BIG),
            initial: Some(view(10, BIG)),
            anchor_ids: Some(view(11, BIG)),
            checkpoint_ids: Some(view(12, BIG)),
            live_sequences: view(13, BIG),
        }
    }

    /// A launch of `rows` rows over `items` sequences.
    fn at(rows: u32, items: u32) -> Counts {
        Counts::rows(rows).with_items(items, 0)
    }

    fn conv_buffers(stride: u32, weights: usize) -> ConvBuffers<View> {
        ConvBuffers {
            input: view(1, BIG),
            input_stride: stride,
            weights: (0..weights).map(|i| view(2 + i as u64, BIG)).collect(),
            state: view(5, BIG),
            output: view(6, BIG),
            slots: slots(),
        }
    }

    fn delta_buffers(strides: [u32; 3]) -> DeltaRuleBuffers<View> {
        DeltaRuleBuffers {
            qkv: view(14, BIG),
            qkv_stride: strides[0],
            decay: view(15, BIG),
            decay_stride: strides[1],
            beta: view(16, BIG),
            beta_stride: strides[2],
            a_log: view(17, BIG),
            dt_bias: view(18, BIG),
            state: view(5, BIG),
            output: view(6, BIG),
            workspace: Some(view(19, BIG)),
            slots: slots(),
        }
    }

    // A linear-attention layer of 4 key and 12 value heads of 128, a
    // 2560-channel convolution of 4 taps, with the claims its catalog
    // declares: prefill 8192 rows, decode 256 sequences, verify 8 rows each.
    const KEY_HEADS: u32 = 4;
    const VALUE_HEADS: u32 = 12;
    const CHANNELS: u32 = 2560;
    const AB_STRIDE: u32 = 32;
    const VERIFY: RecurrentPhase = RecurrentPhase::Verify { max_rows: 8 };

    fn conv(phase: RecurrentPhase, capacity: u32) -> RecurrentQuery {
        RecurrentQuery::CausalConv1d {
            channels: CHANNELS,
            kernel: 4,
            dilation: 1,
            activation: ConvActivation::Silu,
            phase,
            capacity,
        }
    }

    fn delta(phase: RecurrentPhase, capacity: u32) -> RecurrentQuery {
        RecurrentQuery::GatedDeltaRule {
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            key_dim: 128,
            value_dim: 128,
            phase,
            capacity,
        }
    }

    const NORM: RecurrentQuery = RecurrentQuery::GatedRmsNorm {
        width: VALUE_HEADS * 128,
        head_dim: 128,
        eps_bits: 0x358637bd,
        gate: GateActivation::Sigmoid,
        capacity: 8192,
    };

    fn chunks() -> Invariance {
        Invariance::PREFILL.with_chunk(Chunk::Granule(CHUNK))
    }

    fn linear_entries() -> Vec<linear::Entry> {
        let profile = Profile::synthetic(ProfileNames {
            name: "linear-4x12x128",
            tag: "k4v12",
        });
        let entry = |name: &str, query, stage, params, invariance| linear::Entry {
            profile,
            symbol: format!("chop_linear_k4v12_{name}_v1"),
            query,
            stage,
            params,
            invariance,
        };
        let prefill = RecurrentPhase::Prefill;
        let decode = RecurrentPhase::Decode;
        let spare = Invariance::PREFILL.with_anchors(Anchors::Spare);
        let norm_params = channel_decay::norm_params();
        vec![
            entry(
                "conv_prefill",
                conv(prefill, 8192),
                Stage::Single,
                conv_params(prefill),
                spare,
            ),
            entry(
                "conv_decode",
                conv(decode, 256),
                Stage::Single,
                conv_params(decode),
                Invariance::NONE,
            ),
            entry(
                "conv_verify",
                conv(VERIFY, 2048),
                Stage::Single,
                conv_params(VERIFY),
                Invariance::one_row(4),
            ),
            entry(
                "delta_prefill_prepare",
                delta(prefill, 8192),
                Stage::Prepare,
                delta_params(prefill, Stage::Prepare),
                chunks(),
            ),
            entry(
                "delta_prefill_scan",
                delta(prefill, 8192),
                Stage::Scan,
                delta_params(prefill, Stage::Scan),
                chunks().with_anchors(Anchors::Spare),
            ),
            entry(
                "delta_decode",
                delta(decode, 256),
                Stage::Single,
                delta_params(decode, Stage::Single),
                Invariance::NONE,
            ),
            entry(
                "delta_verify",
                delta(VERIFY, 2048),
                Stage::Single,
                delta_params(VERIFY, Stage::Single),
                Invariance::one_row(4),
            ),
            entry(
                "gated_norm",
                NORM,
                Stage::Single,
                norm_params,
                Invariance::any_rows(8192),
            ),
        ]
    }

    fn linear_entry(name: &str) -> linear::Entry {
        let symbol = format!("chop_linear_k4v12_{name}_v1");
        linear_entries()
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap()
    }

    fn legacy(entry: &linear::Entry, bucket: Bucket, values: Vec<Argument>) -> Launch {
        Launch {
            symbol: entry.symbol.clone(),
            rows: bucket.rows,
            plan: entry.plan(bucket, bucket.rows, bucket.sequences).unwrap(),
            arguments: entry.pack_arguments(values).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    #[test]
    fn linear_ops_launch_what_the_entries_plan() {
        let backend = Sm120::new().with_linear(linear_entries());
        let qkv = 2 * KEY_HEADS * 128 + VALUE_HEADS * 128;

        // A prefill of 300 rows over 3 sequences, positional arguments in
        // the entries' ABI order.
        let bucket = Bucket {
            rows: 300,
            sequences: 3,
        };
        let counts = at(300, 3);
        let prefill = RecurrentPhase::Prefill;
        let op = backend
            .causal_conv1d(&conv(prefill, 8192), 256, Need::rows(8192))
            .unwrap();
        assert_eq!(op.info().implementation, "sm120.linear.conv.prefill.v1");
        assert_eq!(
            op.info().guarantees,
            Invariance::PREFILL.with_anchors(Anchors::Spare)
        );
        assert_eq!(
            op.launches(&conv_buffers(CHANNELS, 1), View::EMPTY, counts)
                .unwrap(),
            [legacy(
                &linear_entry("conv_prefill"),
                bucket,
                vec![
                    p(1),
                    u(CHANNELS),
                    p(2),
                    p(5),
                    p(6),
                    p(7),
                    p(9),
                    p(10),
                    p(11),
                    p(13)
                ],
            )]
        );
        let op = backend
            .gated_delta_rule(&delta(prefill, 8192), 256, Need::rows(8192))
            .unwrap();
        assert_eq!(
            op.info().implementation,
            "sm120.linear.delta-rule.prefill.v1[chunk=64]"
        );
        assert_eq!(op.info().guarantees, chunks().with_anchors(Anchors::Spare));
        let prepare = linear_entry("delta_prefill_prepare");
        let widest = Bucket {
            rows: 8192,
            sequences: linear::PREFILL_SEQUENCES,
        };
        assert_eq!(
            op.info().bytes_of("workspace"),
            Some(prepare.workspace_bytes(widest))
        );
        assert_eq!(
            op.info().symbols,
            [
                (prepare.symbol.clone(), prepare.dynamic_shared_bytes()),
                (
                    "chop_linear_k4v12_delta_prefill_scan_v1".to_owned(),
                    linear_entry("delta_prefill_scan").dynamic_shared_bytes()
                ),
            ]
        );
        let buffers = delta_buffers([qkv, AB_STRIDE, AB_STRIDE]);
        assert_eq!(
            op.launches(&buffers, View::EMPTY, counts).unwrap(),
            [
                legacy(
                    &prepare,
                    bucket,
                    vec![
                        p(14),
                        u(qkv),
                        p(15),
                        p(16),
                        u(AB_STRIDE),
                        p(17),
                        p(18),
                        p(19),
                        p(7),
                        p(13)
                    ],
                ),
                legacy(
                    &linear_entry("delta_prefill_scan"),
                    bucket,
                    vec![
                        p(14),
                        u(qkv),
                        p(19),
                        p(5),
                        p(6),
                        p(7),
                        p(9),
                        p(10),
                        p(11),
                        p(13)
                    ],
                ),
            ]
        );

        // A verify step of 4 sequences of up to 8 rows (decode runs it too).
        let bucket = Bucket {
            rows: 20,
            sequences: 4,
        };
        let counts = at(20, 4);
        let op = backend
            .causal_conv1d(&conv(VERIFY, 2048), 256, Need::rows(2048))
            .unwrap();
        assert_eq!(op.info().guarantees, Invariance::one_row(4));
        assert_eq!(
            op.launches(&conv_buffers(CHANNELS, 1), View::EMPTY, counts)
                .unwrap(),
            [legacy(
                &linear_entry("conv_verify"),
                bucket,
                vec![
                    p(1),
                    u(CHANNELS),
                    p(2),
                    p(5),
                    p(6),
                    p(7),
                    p(9),
                    p(12),
                    p(13)
                ],
            )]
        );
        let op = backend
            .gated_delta_rule(&delta(VERIFY, 2048), 256, Need::rows(2048))
            .unwrap();
        assert_eq!(
            op.info().implementation,
            "sm120.linear.delta-rule.verify8.v1"
        );
        assert_eq!(op.info().bytes_of("workspace"), None);
        assert_eq!(
            op.launches(
                &delta_buffers([qkv, AB_STRIDE, AB_STRIDE]),
                View::EMPTY,
                counts
            )
            .unwrap(),
            [legacy(
                &linear_entry("delta_verify"),
                bucket,
                vec![
                    p(14),
                    u(qkv),
                    p(15),
                    p(16),
                    u(AB_STRIDE),
                    p(17),
                    p(18),
                    p(5),
                    p(6),
                    p(7),
                    p(9),
                    p(12),
                    p(13),
                ],
            )]
        );

        // Decode: one row per sequence.
        let bucket = Bucket {
            rows: 16,
            sequences: 16,
        };
        let op = backend
            .gated_delta_rule(&delta(RecurrentPhase::Decode, 256), 256, Need::rows(256))
            .unwrap();
        assert_eq!(op.info().guarantees, Invariance::NONE);
        assert_eq!(
            op.launches(
                &delta_buffers([qkv, AB_STRIDE, AB_STRIDE]),
                View::EMPTY,
                at(16, 16)
            )
            .unwrap(),
            [legacy(
                &linear_entry("delta_decode"),
                bucket,
                vec![
                    p(14),
                    u(qkv),
                    p(15),
                    p(16),
                    u(AB_STRIDE),
                    p(17),
                    p(18),
                    p(5),
                    p(6),
                    p(9),
                    p(13),
                ],
            )]
        );
        assert!(
            op.launches(
                &delta_buffers([qkv, AB_STRIDE, AB_STRIDE]),
                View::EMPTY,
                at(16, 8)
            )
            .is_err()
        );

        // The gated norm: every row its own sequence.
        let op = backend.gated_rmsnorm(&NORM, Need::rows(8192)).unwrap();
        assert_eq!(op.info().implementation, "sm120.linear.gated-norm.v1");
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
        let norm = GatedNormBuffers {
            input: view(1, BIG),
            gate: view(20, BIG),
            gate_stride: 1536,
            weight: view(21, BIG),
            output: view(6, BIG),
            live_rows: view(22, BIG),
        };
        assert_eq!(
            op.launches(&norm, View::EMPTY, Counts::rows(300)).unwrap(),
            [legacy(
                &linear_entry("gated_norm"),
                Bucket {
                    rows: 300,
                    sequences: 300
                },
                vec![p(1), p(20), u(1536), p(21), p(6), p(22)],
            )]
        );
    }

    fn kda_entry(name: &str) -> channel_decay::Entry {
        let symbol = format!("chop_channel_decay_h16x128_{name}_v1");
        channel_decay::catalog()
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap()
    }

    fn kda_legacy(name: &str, bucket: Bucket, values: Vec<Argument>) -> Launch {
        let entry = kda_entry(name);
        Launch {
            symbol: entry.symbol.clone(),
            rows: bucket.rows,
            plan: entry.plan(bucket, bucket.rows, bucket.sequences).unwrap(),
            arguments: entry.pack_arguments(values).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    fn kda_rule(phase: RecurrentPhase) -> PerChannelDecayDeltaRuleQuery {
        let capacity = match phase {
            RecurrentPhase::Prefill => 8192,
            RecurrentPhase::Decode => 256,
            RecurrentPhase::Verify { .. } => 2048,
        };
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads: 16,
            head_dim: 128,
            gate: channel_decay::SERVED.gate,
            phase,
            capacity,
        }
    }

    fn kda_conv(phase: RecurrentPhase) -> PerChannelDecayDeltaRuleQuery {
        let PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads,
            head_dim,
            phase,
            capacity,
            ..
        } = kda_rule(phase)
        else {
            unreachable!()
        };
        PerChannelDecayDeltaRuleQuery::ShortConv {
            heads,
            head_dim,
            kernel: 4,
            convs: QkvConvs::Separate,
            phase,
            capacity,
        }
    }

    /// An op planned at capacity grids its prepared items at every launch
    /// and checks a launch's per-sequence arrays for the launch's own
    /// sequences: one sequence's entries (a prefill part's, from its offset
    /// in shared arrays) pass, a row offsets view short of them still
    /// refuses, and the op planned per launch refuses those views at the
    /// prepared items.
    #[test]
    fn channel_decay_ops_planned_at_capacity_plan_their_items_over_a_launchs_arrays() {
        let backend = Sm120::new().with_channel_decay(channel_decay::catalog());
        let width = 16 * 128;
        let prefill = RecurrentPhase::Prefill;
        let at_capacity = |query| PerChannelDecayOp {
            query,
            plan_at_capacity: true,
        };
        let conv = backend
            .short_conv(&at_capacity(kda_conv(prefill)), 32, Need::rows(8192))
            .unwrap();
        let rule = backend
            .channel_decay_delta_rule(&at_capacity(kda_rule(prefill)), 32, Need::rows(8192))
            .unwrap();
        // One sequence's two row offsets, start, flag and anchor, its slot
        // and the live count.
        let part = RecurrentSlots {
            row_offsets: Some(view(7, 8)),
            start_positions: Some(view(8, 4)),
            slot_ids: view(9, 4),
            initial: Some(view(10, 4)),
            anchor_ids: Some(view(11, 4)),
            checkpoint_ids: None,
            live_sequences: view(13, 4),
        };
        let one = at(300, 1);
        let planned = Bucket {
            rows: 300,
            sequences: 32,
        };
        let conv_buffers = ConvBuffers {
            slots: part,
            ..conv_buffers(3 * width, 3)
        };
        assert_eq!(
            conv.launches(&conv_buffers, View::EMPTY, one).unwrap(),
            [kda_legacy(
                "conv_prefill",
                planned,
                vec![
                    p(1),
                    u(3 * width),
                    p(2),
                    p(3),
                    p(4),
                    p(5),
                    p(6),
                    p(7),
                    p(8),
                    p(9),
                    p(10),
                    p(11),
                    p(13),
                ],
            )]
        );
        let rule_buffers = DeltaRuleBuffers {
            slots: part,
            ..delta_buffers([3 * width, width, 16])
        };
        let operands = vec![
            p(14),
            u(3 * width),
            p(15),
            u(width),
            p(16),
            u(16),
            p(17),
            p(18),
        ];
        assert_eq!(
            rule.launches(&rule_buffers, View::EMPTY, one).unwrap(),
            [
                kda_legacy(
                    "delta_prefill_prepare",
                    planned,
                    [operands, vec![p(19), p(7), p(8), p(13)]].concat(),
                ),
                kda_legacy(
                    "delta_prefill_scan",
                    planned,
                    vec![p(19), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(13)],
                ),
            ]
        );
        // A row offsets view of one entry does not hold one sequence's two.
        let short = RecurrentSlots {
            row_offsets: Some(view(7, 4)),
            ..part
        };
        assert!(
            conv.launches(
                &ConvBuffers {
                    slots: short,
                    ..conv_buffers.clone()
                },
                View::EMPTY,
                one
            )
            .is_err()
        );
        assert!(
            rule.launches(
                &DeltaRuleBuffers {
                    slots: short,
                    ..rule_buffers
                },
                View::EMPTY,
                one
            )
            .is_err()
        );
        // Planned per launch, the prepared items need their arrays whole.
        let per_launch = backend
            .short_conv(&kda_conv(prefill).into(), 32, Need::rows(8192))
            .unwrap();
        assert!(
            per_launch
                .launches(&conv_buffers, View::EMPTY, at(300, 32))
                .is_err()
        );
    }

    #[test]
    fn channel_decay_ops_launch_what_the_entries_plan_and_claim_their_declarations() {
        let backend = Sm120::new().with_channel_decay(channel_decay::catalog());
        let width = 16 * 128;
        let prefill = RecurrentPhase::Prefill;
        let bucket = Bucket {
            rows: 300,
            sequences: 3,
        };
        let counts = at(300, 3);
        let op = backend
            .short_conv(&kda_conv(prefill).into(), 256, Need::rows(8192))
            .unwrap();
        assert_eq!(
            op.info().guarantees,
            Invariance::PREFILL.with_anchors(Anchors::Spare)
        );
        assert_eq!(
            op.launches(&conv_buffers(3 * width, 3), View::EMPTY, counts)
                .unwrap(),
            [kda_legacy(
                "conv_prefill",
                bucket,
                vec![
                    p(1),
                    u(3 * width),
                    p(2),
                    p(3),
                    p(4),
                    p(5),
                    p(6),
                    p(7),
                    p(8),
                    p(9),
                    p(10),
                    p(11),
                    p(13),
                ],
            )]
        );
        // Separate q, k and v weights take three views.
        assert!(
            op.launches(&conv_buffers(3 * width, 1), View::EMPTY, counts)
                .is_err()
        );
        let op = backend
            .channel_decay_delta_rule(&kda_rule(prefill).into(), 256, Need::rows(8192))
            .unwrap();
        assert_eq!(
            op.info().implementation,
            "sm120.channel-decay.delta-rule.prefill.v1[chunk=64]"
        );
        let buffers = delta_buffers([3 * width, width, 16]);
        let operands = vec![
            p(14),
            u(3 * width),
            p(15),
            u(width),
            p(16),
            u(16),
            p(17),
            p(18),
        ];
        let prepare = [operands.clone(), vec![p(19), p(7), p(8), p(13)]].concat();
        let scan = vec![p(19), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(13)];
        assert_eq!(
            op.launches(&buffers, View::EMPTY, counts).unwrap(),
            [
                kda_legacy("delta_prefill_prepare", bucket, prepare),
                kda_legacy("delta_prefill_scan", bucket, scan),
            ]
        );
        // Prefill chunks start at absolute positions.
        let mut unpositioned = buffers;
        unpositioned.slots.start_positions = None;
        assert!(op.launches(&unpositioned, View::EMPTY, counts).is_err());

        let verify = RecurrentPhase::Verify { max_rows: 8 };
        let op = backend
            .channel_decay_delta_rule(&kda_rule(verify).into(), 256, Need::rows(2048))
            .unwrap();
        let bucket = Bucket {
            rows: 20,
            sequences: 4,
        };
        let values = [operands, vec![p(5), p(6), p(7), p(9), p(12), p(13)]].concat();
        assert_eq!(
            op.launches(
                &delta_buffers([3 * width, width, 16]),
                View::EMPTY,
                at(20, 4)
            )
            .unwrap(),
            [kda_legacy("delta_verify", bucket, values)]
        );
        // Verify rows equal decode steps up to the capacity, with no spare
        // entry; rows past it and anchors are refused.
        assert_eq!(
            op.info().guarantees,
            Invariance::one_row(2048).with_anchors(Anchors::None)
        );
        assert!(
            backend
                .channel_decay_delta_rule(
                    &kda_rule(verify).into(),
                    256,
                    Need::rows(2048).with_exact(Invariance::one_row(2048))
                )
                .is_ok()
        );
        for need in [
            Invariance::one_row(4096),
            Invariance::NONE.with_anchors(Anchors::Spare),
        ] {
            assert!(matches!(
                backend.channel_decay_delta_rule(
                    &kda_rule(verify).into(),
                    256,
                    Need::rows(2048).with_exact(need)
                ),
                Err(Refused::Inexact { .. })
            ));
        }

        let norm = channel_decay::catalog()
            .into_iter()
            .find_map(|entry| match entry.query {
                DecayQuery::Norm(query) => Some(query),
                DecayQuery::Rule(_) => None,
            })
            .unwrap();
        let op = backend.gated_rmsnorm(&norm, Need::rows(8192)).unwrap();
        assert_eq!(
            op.info().implementation,
            "sm120.channel-decay.gated-norm.v1"
        );
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
    }

    #[test]
    fn recurrent_prepare_refuses_inexact_unserved_and_over_limit_ops() {
        let backend = Sm120::new().with_linear(linear_entries());
        let prefill = RecurrentPhase::Prefill;
        // A granule the delta rule does not keep, and one it does.
        let granule =
            |n| Need::rows(8192).with_exact(Invariance::PREFILL.with_chunk(Chunk::Granule(n)));
        assert!(matches!(
            backend.gated_delta_rule(&delta(prefill, 8192), 256, granule(32)),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend
                .gated_delta_rule(&delta(prefill, 8192), 256, granule(128))
                .is_ok()
        );
        let spare = Invariance::PREFILL
            .with_chunk(Chunk::Granule(64))
            .with_anchors(Anchors::Spare);
        assert!(
            backend
                .gated_delta_rule(
                    &delta(prefill, 8192),
                    256,
                    Need::rows(8192).with_exact(spare)
                )
                .is_ok()
        );
        // Verify rows equal to decode rows up to the declared four rows.
        let verify = |rows| Need::rows(2048).with_exact(Invariance::one_row(rows));
        assert!(
            backend
                .gated_delta_rule(&delta(VERIFY, 2048), 256, verify(4))
                .is_ok()
        );
        assert!(matches!(
            backend.gated_delta_rule(&delta(VERIFY, 2048), 256, verify(8)),
            Err(Refused::Inexact { .. })
        ));
        // A query of the wrong kind, one no entry serves, and a capacity past
        // the entry's.
        assert!(matches!(
            backend.causal_conv1d(&delta(prefill, 8192), 256, Need::rows(8192)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.causal_conv1d(&conv(prefill, 4096), 256, Need::rows(4096)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.causal_conv1d(&conv(prefill, 8192), 256, Need::rows(8193)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            Sm120::new().short_conv(&kda_conv(prefill).into(), 256, Need::rows(8192)),
            Err(Refused::Rule(_))
        ));
        // SM120 limits: more than 8 taps, a head dim no template has, and
        // a prefill past the shared memory.
        let over = |query: RecurrentQuery| {
            let mut entry = linear_entry("conv_prefill");
            entry.query = query;
            if matches!(query, RecurrentQuery::GatedDeltaRule { .. }) {
                entry.params = delta_params(prefill, Stage::Single);
            }
            Sm120::new().with_linear([entry])
        };
        let taps = RecurrentQuery::CausalConv1d {
            channels: CHANNELS,
            kernel: 9,
            dilation: 1,
            activation: ConvActivation::Silu,
            phase: prefill,
            capacity: 8192,
        };
        assert!(matches!(
            over(taps).causal_conv1d(&taps, 256, Need::rows(8192)),
            Err(Refused::Rule(_))
        ));
        assert!(linear::limits(taps).is_err());
        let odd = RecurrentQuery::GatedDeltaRule {
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            key_dim: 96,
            value_dim: 96,
            phase: RecurrentPhase::Decode,
            capacity: 256,
        };
        assert!(matches!(
            over(odd).gated_delta_rule(&odd, 256, Need::rows(256)),
            Err(Refused::Rule(rule)) if rule.contains("64, 128 or 256")
        ));
        let wide = RecurrentQuery::GatedDeltaRule {
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            key_dim: 256,
            value_dim: 256,
            phase: prefill,
            capacity: 8192,
        };
        let backend = Sm120::new().with_linear(linear_entries().into_iter().map(|mut entry| {
            if entry.query == delta(prefill, 8192) {
                entry.query = wide;
            }
            entry
        }));
        assert!(matches!(
            backend.gated_delta_rule(&wide, 256, Need::rows(8192)),
            Err(Refused::Rule(rule)) if rule.contains("shared memory")
        ));
        // The per-channel-decay entries run head dim 128 and 8 taps only.
        let narrow = channel_decay::entries(
            channel_decay::SERVED_PROFILE,
            channel_decay::Geometry {
                head_dim: 64,
                ..channel_decay::SERVED
            },
        );
        let query = narrow[0].query;
        let DecayQuery::Rule(query) = query else {
            unreachable!()
        };
        assert!(matches!(
            Sm120::new()
                .with_channel_decay(narrow)
                .short_conv(&query.into(), 256, Need::rows(8192)),
            Err(Refused::Rule(rule)) if rule.contains("head dim 128")
        ));
    }

    #[test]
    fn invalid_queries_refuse_before_any_sizing() {
        // A matching entry of an invalid phase: the catalog capacity would
        // divide by zero before the entry refused it.
        let invalid = RecurrentPhase::Verify { max_rows: 0 };
        let mut entry = linear_entry("conv_verify");
        entry.query = conv(invalid, 2048);
        entry.params = conv_params(invalid);
        assert!(matches!(
            Sm120::new().with_linear([entry]).causal_conv1d(
                &conv(invalid, 2048),
                256,
                Need::rows(2048)
            ),
            Err(Refused::Rule(_))
        ));
        // A state whose elements overflow a u64, and one whose columns
        // overflow a u32: refused or sized in u64, never a panic.
        let backend = Sm120::new().with_linear(linear_entries());
        let heads = |value_heads, dim| RecurrentQuery::GatedDeltaRule {
            key_heads: 1,
            value_heads,
            key_dim: dim,
            value_dim: dim,
            phase: RecurrentPhase::Decode,
            capacity: 256,
        };
        assert!(matches!(
            backend.gated_delta_rule(&heads(1 << 31, 1 << 20), 256, Need::rows(256)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.gated_delta_rule(&heads(1 << 24, 256), 256, Need::rows(256)),
            Err(Refused::Rule(_))
        ));
        let rule = PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads: 1 << 30,
            head_dim: 128,
            gate: channel_decay::SERVED.gate,
            phase: RecurrentPhase::Decode,
            capacity: 256,
        };
        assert!(matches!(
            Sm120::new()
                .with_channel_decay(channel_decay::catalog())
                .channel_decay_delta_rule(&rule.into(), 256, Need::rows(256)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn verify_rows_past_sixteen_are_an_sm120_limit() {
        // The contracts take any verify rows; the SM120 templates of both
        // engines take at most 16 per sequence.
        let most = RecurrentPhase::Verify {
            max_rows: linear::MAX_VERIFY_ROWS,
        };
        let past = RecurrentPhase::Verify {
            max_rows: linear::MAX_VERIFY_ROWS + 1,
        };
        let sixteen = |refusal: &str| refusal.contains("at most 16 rows");
        for (query, served) in [
            (conv(past, 2048), conv(most, 2048)),
            (delta(past, 2048), delta(most, 2048)),
        ] {
            assert!(query.validate().is_ok());
            assert!(matches!(linear::limits(query), Err(refusal) if sixteen(refusal.0)));
            assert!(linear::limits(served).is_ok());
        }
        let mut entry = linear_entry("conv_verify");
        entry.query = conv(past, 2048);
        entry.params = conv_params(past);
        assert!(entry.contract().is_err());
        assert!(matches!(
            Sm120::new().with_linear([entry]).causal_conv1d(
                &conv(past, 2048),
                256,
                Need::rows(2048)
            ),
            Err(Refused::Rule(rule)) if sixteen(&rule)
        ));
        let backend = Sm120::new().with_linear(linear_entries());
        assert!(matches!(
            backend.gated_delta_rule(&delta(past, 2048), 256, Need::rows(2048)),
            Err(Refused::Rule(rule)) if sixteen(&rule)
        ));

        for query in [kda_conv(past), kda_rule(past)] {
            assert!(query.validate().is_ok());
            assert!(channel_decay::limits(query).is_err());
        }
        assert!(channel_decay::limits(kda_rule(most)).is_ok());
        let backend = Sm120::new().with_channel_decay(channel_decay::catalog());
        assert!(matches!(
            backend.short_conv(&kda_conv(past).into(), 256, Need::rows(2048)),
            Err(Refused::Rule(rule)) if sixteen(&rule)
        ));
        assert!(matches!(
            backend.channel_decay_delta_rule(&kda_rule(past).into(), 256, Need::rows(2048)),
            Err(Refused::Rule(rule)) if sixteen(&rule)
        ));
        let entries = channel_decay::entries(
            channel_decay::SERVED_PROFILE,
            channel_decay::Geometry {
                verify_rows: linear::MAX_VERIFY_ROWS + 1,
                ..channel_decay::SERVED
            },
        );
        let verify = entries
            .iter()
            .filter(|entry| entry.symbol.contains("verify"))
            .collect::<Vec<_>>();
        assert!(!verify.is_empty());
        assert!(verify.iter().all(|entry| entry.contract().is_err()));
    }

    #[test]
    fn the_workspace_covers_every_bucket_of_the_capacity() {
        // One row over the most sequences a prefill carries: every sequence
        // pads its own chunks.
        let backend = Sm120::new().with_channel_decay(channel_decay::catalog());
        let op = backend
            .channel_decay_delta_rule(
                &kda_rule(RecurrentPhase::Prefill).into(),
                256,
                Need::rows(1),
            )
            .unwrap();
        let prepare = kda_entry("delta_prefill_prepare");
        let bytes = op.info().bytes_of("workspace").unwrap();
        assert_eq!(
            bytes,
            prepare.workspace_bytes(Bucket {
                rows: 1,
                sequences: channel_decay::PREFILL_SEQUENCES,
            })
        );
        let width = 16 * 128;
        let mut buffers = delta_buffers([3 * width, width, 16]);
        buffers.workspace = Some(view(19, bytes));
        let one = Bucket {
            rows: 1,
            sequences: 32,
        };
        assert_eq!(prepare.workspace_bytes(one), 153_886_720);
        let launches = op.launches(&buffers, View::EMPTY, at(1, 32)).unwrap();
        assert_eq!(launches[0].plan, prepare.plan(one, 1, 32).unwrap());
        for sequences in [1, 255, 256] {
            assert!(op.launches(&buffers, View::EMPTY, at(1, sequences)).is_ok());
        }
        // The linear engine likewise.
        let qkv = 2 * KEY_HEADS * 128 + VALUE_HEADS * 128;
        let op = Sm120::new()
            .with_linear(linear_entries())
            .gated_delta_rule(&delta(RecurrentPhase::Prefill, 8192), 256, Need::rows(1))
            .unwrap();
        let bytes = op.info().bytes_of("workspace").unwrap();
        let mut buffers = delta_buffers([qkv, AB_STRIDE, AB_STRIDE]);
        buffers.workspace = Some(view(19, bytes));
        assert!(
            op.launches(&buffers, View::EMPTY, at(1, linear::PREFILL_SEQUENCES))
                .is_ok()
        );
        buffers.workspace = Some(view(19, bytes - 1));
        assert!(
            op.launches(&buffers, View::EMPTY, at(1, linear::PREFILL_SEQUENCES))
                .is_err()
        );
    }

    #[test]
    fn sizes_that_overflow_refuse_and_never_wrap() {
        // Elements that fit a u64 whose bytes do not.
        let wide = RecurrentQuery::CausalConv1d {
            channels: 1 << 31,
            kernel: 3,
            dilation: 1 << 31,
            activation: ConvActivation::Silu,
            phase: RecurrentPhase::Prefill,
            capacity: 1,
        };
        let mut entry = linear_entry("conv_prefill");
        entry.query = wide;
        assert!(matches!(
            Sm120::new().with_linear([entry]).causal_conv1d(&wide, 1, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("overflow")
        ));
        // Head counts whose workspace overflows u32, and more value heads
        // than a grid row holds.
        let heads = |value_heads| RecurrentQuery::GatedDeltaRule {
            key_heads: value_heads,
            value_heads,
            key_dim: 128,
            value_dim: 128,
            phase: RecurrentPhase::Prefill,
            capacity: 8192,
        };
        let stages = |query| {
            linear_entries().into_iter().map(move |mut entry| {
                if entry.query == delta(RecurrentPhase::Prefill, 8192) {
                    entry.query = query;
                }
                entry
            })
        };
        let query = heads(linear::MAX_VALUE_HEADS);
        let op = Sm120::new()
            .with_linear(stages(query))
            .gated_delta_rule(&query, 1, Need::rows(8192))
            .unwrap();
        let per_chunk = u64::from(linear::MAX_VALUE_HEADS)
            * (linear::WORK_FLOATS * 4 + 2 * 64 * 136 * 2 + 64 * 128 * 2);
        assert_eq!(op.info().bytes_of("workspace"), Some(129 * per_chunk));
        let query = heads(linear::MAX_VALUE_HEADS + 1);
        assert!(matches!(
            Sm120::new()
                .with_linear(stages(query))
                .gated_delta_rule(&query, 1, Need::rows(8192)),
            Err(Refused::Rule(rule)) if rule.contains("65535")
        ));
        // Rows of a view that holds every byte: the sizes overflow, so they
        // are refused rather than covered by a saturated size.
        let max = u64::from(u32::MAX);
        assert_eq!(packed(u32::MAX, max), Err(SIZES));
        assert_eq!(strided(u32::MAX, u32::MAX, max), Err(SIZES));
        // A norm of that many rows is past CUDA's grid before any size.
        let norm = RecurrentQuery::GatedRmsNorm {
            width: u32::MAX,
            head_dim: 255,
            eps_bits: 1e-6_f32.to_bits(),
            gate: GateActivation::Sigmoid,
            capacity: u32::MAX,
        };
        let mut entry = linear_entry("gated_norm");
        entry.query = norm;
        assert!(matches!(
            Sm120::new()
                .with_linear([entry])
                .gated_rmsnorm(&norm, Need::rows(u32::MAX)),
            Err(Refused::Rule(rule)) if rule.contains("launch limits")
        ));
    }

    #[test]
    fn items_bound_the_workspace_and_every_launch() {
        let qkv = 2 * KEY_HEADS * 128 + VALUE_HEADS * 128;
        let backend = Sm120::new().with_linear(linear_entries());
        let prefill = delta(RecurrentPhase::Prefill, 8192);
        let op = backend
            .gated_delta_rule(&prefill, 32, Need::rows(8192))
            .unwrap();
        let prepare = linear_entry("delta_prefill_prepare");
        let served = Bucket {
            rows: 8192,
            sequences: 32,
        };
        let bytes = op.info().bytes_of("workspace").unwrap();
        assert_eq!(bytes, prepare.workspace_bytes(served));
        assert!(
            bytes
                < prepare.workspace_bytes(Bucket {
                    rows: 8192,
                    sequences: linear::PREFILL_SEQUENCES
                })
        );
        let mut buffers = delta_buffers([qkv, AB_STRIDE, AB_STRIDE]);
        buffers.workspace = Some(view(19, bytes));
        for rows in [1, 8192] {
            assert!(op.launches(&buffers, View::EMPTY, at(rows, 32)).is_ok());
        }
        // A launch's items come from its counts: one to the prepared items.
        for items in [0, 33] {
            assert_eq!(
                op.launches(&buffers, View::EMPTY, at(64, items)),
                Err(Refusal("recurrent items exceed the prepared items"))
            );
        }
        // Items are one to the entries' sequences.
        for items in [0, linear::PREFILL_SEQUENCES + 1] {
            assert!(matches!(
                backend.gated_delta_rule(&prefill, items, Need::rows(8192)),
                Err(Refused::Rule(_))
            ));
        }
        // The per-channel-decay workspace likewise.
        let op = Sm120::new()
            .with_channel_decay(channel_decay::catalog())
            .channel_decay_delta_rule(
                &kda_rule(RecurrentPhase::Prefill).into(),
                32,
                Need::rows(8192),
            )
            .unwrap();
        assert_eq!(
            op.info().bytes_of("workspace"),
            Some(kda_entry("delta_prefill_prepare").workspace_bytes(served))
        );
        // Decode: one row per sequence, so its rows are bounded by both.
        let op = backend
            .gated_delta_rule(&delta(RecurrentPhase::Decode, 256), 16, Need::rows(256))
            .unwrap();
        let buffers = delta_buffers([qkv, AB_STRIDE, AB_STRIDE]);
        assert!(op.launches(&buffers, View::EMPTY, at(16, 16)).is_ok());
        assert!(op.launches(&buffers, View::EMPTY, at(17, 17)).is_err());
    }

    #[test]
    fn recurrent_launches_check_every_view_and_count() {
        let backend = Sm120::new().with_linear(linear_entries());
        let qkv = 2 * KEY_HEADS * 128 + VALUE_HEADS * 128;
        let op = backend
            .gated_delta_rule(&delta(RecurrentPhase::Prefill, 8192), 256, Need::rows(512))
            .unwrap();
        let buffers = delta_buffers([qkv, AB_STRIDE, AB_STRIDE]);
        let counts = at(512, 2);
        assert!(op.launches(&buffers, View::EMPTY, counts).is_ok());
        assert!(op.launches(&buffers, View::EMPTY, at(513, 2)).is_err());
        assert!(op.launches(&buffers, View::EMPTY, at(0, 2)).is_err());
        let workspace = op.launches(&buffers, View::EMPTY, counts).unwrap()[0]
            .plan
            .workspace_bytes;
        let mut short = buffers;
        short.workspace = Some(view(19, workspace - 1));
        assert!(op.launches(&short, View::EMPTY, counts).is_err());
        short.workspace = Some(view(19, workspace));
        assert!(op.launches(&short, View::EMPTY, counts).is_ok());
        // Strided rows: the last row needs only its own columns.
        let mut rows = buffers;
        rows.beta = view(
            16,
            (511 * u64::from(AB_STRIDE) + u64::from(VALUE_HEADS)) * 2,
        );
        assert!(op.launches(&rows, View::EMPTY, counts).is_ok());
        rows.beta.bytes -= 1;
        assert!(op.launches(&rows, View::EMPTY, counts).is_err());
        // One stride for decay and beta, at least a row wide.
        let split = delta_buffers([qkv, AB_STRIDE, 2 * AB_STRIDE]);
        assert!(op.launches(&split, View::EMPTY, counts).is_err());
        let narrow = delta_buffers([qkv - 8, AB_STRIDE, AB_STRIDE]);
        assert!(op.launches(&narrow, View::EMPTY, counts).is_err());
        // One slot entry of state, and every per-sequence list.
        let state = u64::from(VALUE_HEADS) * 128 * 128 * 4;
        let mut slot = buffers;
        slot.state = view(5, state - 1);
        assert!(op.launches(&slot, View::EMPTY, counts).is_err());
        slot.state = view(5, state);
        assert!(op.launches(&slot, View::EMPTY, counts).is_ok());
        slot.slots.row_offsets = Some(view(7, 8));
        assert!(op.launches(&slot, View::EMPTY, counts).is_err());
        slot.slots.row_offsets = Some(view(7, 12));
        assert!(op.launches(&slot, View::EMPTY, counts).is_ok());
        slot.slots.anchor_ids = None;
        assert!(op.launches(&slot, View::EMPTY, counts).is_err());
        // Verify checkpoints: max_rows per sequence.
        let op = backend
            .causal_conv1d(&conv(VERIFY, 2048), 256, Need::rows(2048))
            .unwrap();
        let mut buffers = conv_buffers(CHANNELS, 1);
        buffers.slots.checkpoint_ids = Some(view(12, 4 * 8 * 4 - 1));
        assert!(op.launches(&buffers, View::EMPTY, at(20, 4)).is_err());
        buffers.slots.checkpoint_ids = Some(view(12, 4 * 8 * 4));
        assert!(op.launches(&buffers, View::EMPTY, at(20, 4)).is_ok());
        // Verify rows past max_rows per sequence.
        assert!(op.launches(&buffers, View::EMPTY, at(33, 4)).is_err());
    }

    #[test]
    fn entries_prepare_alike_in_every_order() {
        let prefill = RecurrentPhase::Prefill;
        let prepare_entry = linear_entry("delta_prefill_prepare");
        let scan = linear_entry("delta_prefill_scan");
        let rule = |order: &[linear::Entry]| {
            Sm120::new()
                .with_linear(order.to_vec())
                .gated_delta_rule(&delta(prefill, 8192), 256, Need::rows(64))
                .map(|op| op.info().clone())
        };
        let in_every_order = |registered: &[linear::Entry]| {
            let results = orders(registered)
                .iter()
                .map(|o| rule(o))
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        // An equal duplicate is one entry; a different one of a stage does
        // not form one op.
        let one = rule(&[prepare_entry.clone(), scan.clone()]).unwrap();
        assert_eq!(
            in_every_order(&[scan.clone(), prepare_entry.clone(), scan.clone()]),
            Ok(one)
        );
        let mut twin = scan.clone();
        twin.symbol = "chop_linear_k4v12_delta_prefill_scan_twin_v1".into();
        assert!(matches!(
            in_every_order(&[scan, prepare_entry, twin]),
            Err(Refused::Rule(rule)) if rule.contains("two SM120 recurrent entries")
        ));
        // A gated norm both engines serve is a tie.
        let norm = channel_decay::catalog()
            .into_iter()
            .find(|entry| matches!(entry.query, DecayQuery::Norm(_)))
            .unwrap();
        let DecayQuery::Norm(query) = norm.query else {
            unreachable!()
        };
        let mut linear_norm = linear_entry("gated_norm");
        linear_norm.query = query;
        for both in [
            Sm120::new()
                .with_linear([linear_norm.clone()])
                .with_channel_decay([norm.clone()]),
            Sm120::new()
                .with_channel_decay([norm.clone()])
                .with_linear([linear_norm.clone()]),
        ] {
            assert!(matches!(
                both.gated_rmsnorm(&query, Need::rows(64)),
                Err(Refused::Rule(rule)) if rule.contains("two SM120 gated norm entries")
            ));
        }
        let decay = Sm120::new().with_channel_decay([norm.clone(), norm]);
        assert!(decay.gated_rmsnorm(&query, Need::rows(64)).is_ok());
        let linear = Sm120::new().with_linear([linear_norm.clone(), linear_norm]);
        assert!(linear.gated_rmsnorm(&query, Need::rows(64)).is_ok());
    }

    /// R1 finding 3: of a gated norm's engines, one whose entries cannot
    /// keep the needed exactness ties with nothing.
    #[test]
    fn exactness_filters_gated_norm_engines_before_ties() {
        let norm = channel_decay::catalog()
            .into_iter()
            .find(|entry| matches!(entry.query, DecayQuery::Norm(_)))
            .unwrap();
        let DecayQuery::Norm(query) = norm.query else {
            unreachable!()
        };
        let mut linear_norm = linear_entry("gated_norm");
        linear_norm.query = query;
        // A claim past the channel-decay norm's declaration, which only the
        // linear entry keeps.
        assert_eq!(norm.invariance(), Invariance::any_rows(8192));
        linear_norm.invariance = Invariance::any_rows(16384);
        let claims = linear_norm.invariance;
        assert!(!norm.invariance().provides(claims));
        let exact = Need::rows(64).with_exact(claims);
        let alone = Sm120::new()
            .with_linear([linear_norm.clone()])
            .gated_rmsnorm(&query, exact)
            .unwrap()
            .info()
            .clone();
        for both in [
            Sm120::new()
                .with_linear([linear_norm.clone()])
                .with_channel_decay([norm.clone()]),
            Sm120::new()
                .with_channel_decay([norm.clone()])
                .with_linear([linear_norm.clone()]),
        ] {
            assert_eq!(both.gated_rmsnorm(&query, exact).unwrap().info(), &alone);
            assert!(matches!(
                both.gated_rmsnorm(&query, Need::rows(64)),
                Err(Refused::Rule(rule)) if rule.contains("two SM120 gated norm entries")
            ));
        }
    }

    /// R2 finding 1: a stage registered twice, once with its claims and
    /// once declaring none, launches the same. A need the claims keep takes
    /// the claiming registration in every order; with no need, the op
    /// claims what both declare, in every order.
    #[test]
    fn claims_of_equal_stage_registrations_do_not_depend_on_order() {
        let claimed = linear_entry("gated_norm");
        assert_ne!(claimed.invariance, Invariance::NONE);
        let unclaimed = linear::Entry {
            invariance: Invariance::NONE,
            ..claimed.clone()
        };
        let prepare = |need: Need| {
            let results = orders(&[claimed.clone(), unclaimed.clone()])
                .into_iter()
                .map(|order| {
                    Sm120::new()
                        .with_linear(order)
                        .gated_rmsnorm(&NORM, need)
                        .map(|op| op.info().clone())
                })
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        let exact = Need::rows(1).with_exact(Invariance::one_row(1));
        let alone = Sm120::new()
            .with_linear([claimed.clone()])
            .gated_rmsnorm(&NORM, exact)
            .unwrap()
            .info()
            .clone();
        assert_eq!(prepare(exact), Ok(alone));
        let plain = prepare(Need::rows(1)).unwrap();
        assert_eq!(plain.guarantees, Invariance::NONE);
        assert_eq!(plain.symbols[0].0, claimed.symbol);
    }

    /// The recurrent key holds what an entry launches, every field changed
    /// alone: its symbol and ABI; never its profile or claims. The query
    /// selects the op and the stage its form.
    #[test]
    fn launch_keys_hold_what_recurrent_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let other = Profile::synthetic(ProfileNames {
            name: "recurrent-other",
            tag: "recurrent_other",
        });
        let names = field_names!(linear::Entry {
            profile,
            symbol,
            query,
            stage,
            params,
            invariance
        });
        let conv = linear_entry("conv_decode").query;
        let fields: Vec<Field<linear::Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_linear_other_v1".into()),
            ),
            ("query", Class::Selects, Box::new(move |e| e.query = conv)),
            ("stage", Class::Selects, Box::new(|e| e.stage = Stage::Scan)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = Invariance::NONE),
            ),
        ];
        let norm = GatedNormBuffers {
            input: view(1, BIG),
            gate: view(20, BIG),
            gate_stride: 1536,
            weight: view(21, BIG),
            output: view(6, BIG),
            live_rows: view(22, BIG),
        };
        keys_follow_launches(
            &linear_entry("gated_norm"),
            names,
            fields,
            |e| Kernel::Linear(e.clone()).launch(),
            |e| {
                let op = Sm120::new()
                    .with_linear([e.clone()])
                    .gated_rmsnorm(&NORM, Need::rows(8))?;
                let launches =
                    [1, 8].map(|rows| op.launches(&norm, View::EMPTY, Counts::rows(rows)).unwrap());
                Ok((format!("{:?}", e.params), launches))
            },
        );
        // A channel-decay convolution.
        let names = field_names!(channel_decay::Entry {
            profile,
            symbol,
            query,
            stage,
            params,
            schedule
        });
        let base = kda_entry("conv_prefill");
        let rule = kda_entry("delta_prefill_scan").query;
        let fields: Vec<Field<channel_decay::Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_channel_decay_other_v1".into()),
            ),
            ("query", Class::Selects, Box::new(move |e| e.query = rule)),
            ("stage", Class::Selects, Box::new(|e| e.stage = Stage::Scan)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
            // Only the delta rule's prefill stages have a tiled schedule.
            (
                "schedule",
                Class::Selects,
                Box::new(|e| e.schedule = channel_decay::PrefillSchedule::Tiled),
            ),
        ];
        let width = 16 * 128;
        let buffers = conv_buffers(3 * width, 3);
        let op = kda_conv(RecurrentPhase::Prefill).into();
        keys_follow_launches(
            &base,
            names,
            fields,
            |e| Kernel::ChannelDecay(e.clone()).launch(),
            |e| {
                let prepared = Sm120::new().with_channel_decay([e.clone()]).short_conv(
                    &op,
                    256,
                    Need::rows(512),
                )?;
                let launches = [at(1, 1), at(300, 3)]
                    .map(|counts| prepared.launches(&buffers, View::EMPTY, counts).unwrap());
                Ok((format!("{:?}", e.params), launches))
            },
        );
    }
}
