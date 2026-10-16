//! Grouped-MoE template instances (`cuda/moe/grouped*.cuh`).
//!
//! Every served A16 grouped entry on the MXFP4 fragment and FP8 fragment
//! engines, at every rank count, decode bucket, verify bucket and prefill
//! capacity, is a template instance at its rank shape, under the entry
//! symbols, cubins and ABI (`parameters`) of the per-model engines it
//! replaced. Kernel-lab keeps those engines as frozen references
//! (`chop_moe_eng_` cubins from `kernel-lab/cuda/baselines`) and compares
//! every instance with its engine bit for bit.
//!
//! The template takes its schedule and knobs from the row of its model's
//! schedule table keyed by the entry's operation, weight format, rank shape
//! and bucket ([`key`]), which the model's catalog resolves into the entry
//! ([`scheduled`], `Variant::schedule`): the entry body, its CTA threads and
//! the served launch plan ([`plan`]). A row may change only knobs that leave the
//! arithmetic unchanged ([`ScheduleKnobs::arithmetic`]): ring depths, warps
//! per CTA and staging flags, not K splits, so the
//! engine reference stays bitwise equal. The offline sweep enumerates those
//! knobs with [`candidates`].

use super::*;
use kernel_build::tuning::{self, Key, Knob, Row, Table};
use std::sync::OnceLock;

/// Prefix that replaces `chop_moe_` in the kernel-lab engine reference
/// symbols and cubin names.
pub const ENGINE_PREFIX: &str = "chop_moe_eng_";
/// Prefix that replaces `chop_moe_` in offline sweep candidate symbols.
pub const TUNE_PREFIX: &str = "chop_moe_tune_";
/// The kernel base's own schedule table, which holds the rows no model
/// owns.
const SHARED_TABLE: &str = include_str!("../../tuning/schedules.json");
/// Dynamic shared memory a template instance may use: the SM120 limit less
/// the 1 KB the driver reserves per CTA and 128 bytes of slack.
pub const SHARED_LIMIT: u32 = 101_376 - (1024 + 128);
/// Bytes of one routed BF16 row's K64 step in a ring slot, with its padding
/// (`kFragmentRowBytes`).
const RING_ROW_BYTES: u32 = 128 + 16;

/// Whether the template serves a grouped entry (every A16 fragment expert
/// entry); the others use the packed engine.
pub fn covers(v: &Variant) -> bool {
    schedule(v).is_some()
}

/// The template schedule an instance runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Schedule {
    /// Per-warp cp.async rings (`grouped_ring.cuh`).
    Ring,
    /// Persistent CTAs over 128-row token tiles (`grouped_prefill.cuh`).
    Prefill,
}

impl Schedule {
    /// The header of the schedule's template.
    pub fn header(self) -> &'static str {
        match self {
            Schedule::Ring => "#include \"moe/grouped_ring.cuh\"",
            Schedule::Prefill => "#include \"moe/grouped_prefill.cuh\"",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Schedule::Ring => "ring",
            Schedule::Prefill => "prefill",
        }
    }
}

/// `RingSchedule` in `grouped_ring.cuh`.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub struct RingKnobs {
    pub split_k: u32,
    pub warps: u32,
    pub depth: u32,
    pub early_row: bool,
    pub reload_rows: bool,
}

/// `PrefillSchedule` in `grouped_prefill.cuh`.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub struct PrefillKnobs {
    pub weight_stages: u32,
    pub activation_stages: u32,
    pub trim_gather: bool,
    pub bulk_weights: bool,
}

/// A schedule and its compile-time knobs.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum ScheduleKnobs {
    Ring(RingKnobs),
    Prefill(PrefillKnobs),
}

impl ScheduleKnobs {
    pub fn schedule(&self) -> Schedule {
        match self {
            ScheduleKnobs::Ring(_) => Schedule::Ring,
            ScheduleKnobs::Prefill(_) => Schedule::Prefill,
        }
    }

    /// The knobs that decide the F32 reduction order: the schedule and its
    /// K splits (ring). Two knob sets with equal
    /// arithmetic compute every output bit the same way.
    pub fn arithmetic(&self) -> (Schedule, u32) {
        match self {
            ScheduleKnobs::Ring(r) => (Schedule::Ring, r.split_k),
            ScheduleKnobs::Prefill(_) => (Schedule::Prefill, 1),
        }
    }

    /// The knobs as a table row holds them.
    pub fn row_knobs(&self) -> tuning::Knobs {
        let count = |name: &str, value: u32| (name.to_string(), Knob::Count(value));
        let flag = |name: &str, value: bool| (name.to_string(), Knob::Flag(value));
        match *self {
            ScheduleKnobs::Ring(r) => [
                count("split_k", r.split_k),
                count("warps", r.warps),
                count("depth", r.depth),
                flag("early_row", r.early_row),
                flag("reload_rows", r.reload_rows),
            ]
            .into(),
            ScheduleKnobs::Prefill(p) => [
                count("weight_stages", p.weight_stages),
                count("activation_stages", p.activation_stages),
                flag("trim_gather", p.trim_gather),
                flag("bulk_weights", p.bulk_weights),
            ]
            .into(),
        }
    }

    /// Reads a row's schedule and knobs; every knob of the schedule must be
    /// present with its type, and no other.
    pub fn from_row(row: &Row) -> Result<Self, String> {
        let knobs = &row.knobs;
        let count = |name: &str| {
            knobs
                .get(name)
                .and_then(|k| k.count())
                .ok_or_else(|| format!("{}: knob {name} needs a count", row.key))
        };
        let flag = |name: &str| {
            knobs
                .get(name)
                .and_then(|k| k.flag())
                .ok_or_else(|| format!("{}: knob {name} needs a flag", row.key))
        };
        let parsed = match row.schedule.as_str() {
            "ring" => ScheduleKnobs::Ring(RingKnobs {
                split_k: count("split_k")?,
                warps: count("warps")?,
                depth: count("depth")?,
                early_row: flag("early_row")?,
                reload_rows: flag("reload_rows")?,
            }),
            "prefill" => ScheduleKnobs::Prefill(PrefillKnobs {
                weight_stages: count("weight_stages")?,
                activation_stages: count("activation_stages")?,
                trim_gather: flag("trim_gather")?,
                bulk_weights: flag("bulk_weights")?,
            }),
            other => return Err(format!("{}: unknown schedule {other}", row.key)),
        };
        if parsed.row_knobs().len() != knobs.len() {
            return Err(format!("{}: unknown knobs for {}", row.key, row.schedule));
        }
        Ok(parsed)
    }

    /// A short symbol suffix naming every knob.
    fn tag(&self) -> String {
        let bit = u8::from;
        match *self {
            ScheduleKnobs::Ring(r) => format!(
                "k{}w{}d{}e{}r{}",
                r.split_k,
                r.warps,
                r.depth,
                bit(r.early_row),
                bit(r.reload_rows)
            ),
            ScheduleKnobs::Prefill(p) => format!(
                "ws{}as{}g{}b{}",
                p.weight_stages,
                p.activation_stages,
                bit(p.trim_gather),
                bit(p.bulk_weights)
            ),
        }
    }
}

/// A served template entry and its kernel-lab engine reference, or an
/// offline sweep candidate at other knobs.
#[derive(Clone, Debug)]
pub struct Instance {
    /// The served variant: its symbol, cubin, plan and arguments.
    pub baseline: Variant,
    pub schedule: Schedule,
    /// The template's knobs: its table row, or a sweep candidate's.
    pub knobs: ScheduleKnobs,
    /// The template entry: the served one, or a candidate's.
    pub template_symbol: String,
    pub template_cubin: String,
    /// The kernel-lab reference of the engine the template replaced.
    pub engine_symbol: String,
    pub engine_cubin: String,
    /// The template entry's body: a call of its schedule.
    pub body: String,
}

impl Instance {
    /// The template entry's launch plan at this instance's knobs.
    pub fn plan(&self, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
        plan_with(&self.baseline, self.knobs, counts)
    }

    /// This instance with other knobs of the same arithmetic, as a lab
    /// entry named by them: `chop_moe_tune_<served stem>_<knob tag>_v1` in
    /// cubin `tune_<served stem>_<knob tag>`.
    pub fn with_knobs(&self, knobs: ScheduleKnobs) -> Result<Instance, Refusal> {
        if knobs.arithmetic() != self.knobs.arithmetic() {
            return Err(Refusal("a schedule candidate must keep the K splits"));
        }
        let stem = format!(
            "{}_{}",
            self.baseline
                .symbol
                .trim_end_matches("_v1")
                .replacen("chop_moe_", TUNE_PREFIX, 1),
            knobs.tag()
        );
        Ok(Instance {
            baseline: self.baseline.clone(),
            schedule: knobs.schedule(),
            knobs,
            template_symbol: format!("{stem}_v1"),
            template_cubin: format!("tune_{}", stem.trim_start_matches(TUNE_PREFIX)),
            engine_symbol: self.engine_symbol.clone(),
            engine_cubin: self.engine_cubin.clone(),
            body: body(&self.baseline, &knobs),
        })
    }

    /// A one-entry cubin of this instance.
    pub fn cubin_spec(&self) -> CubinSpec {
        let mut spec = new_spec(self);
        spec.entries.push(entry(self));
        spec
    }
}

/// The template launch plan of `v` at `knobs`. Ring grids, blocks and rings
/// follow the knobs; prefill plans are the engine planner's at
/// the knobs' ring depths. Counts, workspace and PDL are the engine's.
fn plan_with(
    v: &Variant,
    knobs: ScheduleKnobs,
    counts: RuntimeCounts,
) -> Result<LaunchPlan, Refusal> {
    let plan = match knobs {
        ScheduleKnobs::Ring(r) => {
            let mut plan = v.engine_plan(counts)?;
            let group = v.query.grouped.ok_or(Refusal("not a grouped GEMM"))?;
            let active = group.experts.min(v.query.m_capacity);
            let strip_columns = r
                .warps
                .checked_div(r.split_k)
                .and_then(|warps| warps.checked_mul(16))
                .filter(|&columns| columns > 0)
                .ok_or(Refusal("ring warps and K splits tile no output columns"))?;
            let grid = active
                .checked_mul(out_n(v) / strip_columns)
                .ok_or(Refusal("ring grid overflows"))?;
            plan.grid = [grid, 1, 1];
            plan.block = [try_threads(&knobs)?, 1, 1];
            plan.dynamic_shared_bytes = ring_shared_bytes(v, r)?;
            plan
        }
        ScheduleKnobs::Prefill(PrefillKnobs {
            weight_stages,
            activation_stages,
            ..
        }) => {
            let mut staged = v.clone();
            staged.weight_stages =
                u8::try_from(weight_stages).map_err(|_| Refusal("weight stages overflow"))?;
            staged.knobs.stages = u8::try_from(activation_stages)
                .map_err(|_| Refusal("activation stages overflow"))?;
            staged.engine_plan(counts)?
        }
    };
    plan.validate()?;
    Ok(plan)
}

/// The served launch plan of an entry the template serves: the template's
/// at its table row. A variant without a valid row is no served entry (the
/// build refuses those, see [`knobs`]); the engine planner judges it.
pub fn plan(v: &Variant, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
    match v.schedule {
        Some(knobs) => plan_with(v, knobs, counts),
        None => v.engine_plan(counts),
    }
}

/// The knobs of `table`'s row for `v`, `None` if the template does not
/// cover it, or why the row is unusable: missing, another schedule than the
/// variant's, or other K splits than the engine's.
pub fn row_knobs(v: &Variant, table: &Table) -> Result<Option<ScheduleKnobs>, String> {
    v.check_numerics()
        .map_err(|Refusal(reason)| format!("{}: {reason}", v.symbol))?;
    if schedule(v).is_none() {
        return Ok(None);
    }
    try_shape(v)?;
    let row = table
        .require(&key(v))
        .map_err(|e| format!("{}: {e}", v.symbol))?;
    validate_row(v, row).map(Some)
}

/// `variants` with the template schedule of each entry the template covers
/// resolved from its row of `table` (`Variant::schedule`). Fails where a
/// row is unusable (see [`validate_row`]): building such an entry would
/// change output bits against the per-model engine or have no schedule at
/// all.
pub fn scheduled(variants: Vec<Variant>, table: &Table) -> Result<Vec<Variant>, String> {
    variants
        .into_iter()
        .map(|v| {
            let schedule = row_knobs(&v, table)?;
            Ok(Variant { schedule, ..v })
        })
        .collect()
}

/// The schedule table of `variants` from the measured tables of the
/// geometries they belong to (`measured`, by geometry ID): every measured
/// row, where tables sharing a key must agree, then for each entry the
/// template covers the row of its own geometry (`geometry`), which must be
/// legal, else the nearest legal row of another geometry
/// ([`tuning::inherit`]), else the engine's served knobs. Each entry's
/// fallback status (`inherited <id> row <key>` or `engine default template
/// schedule`) goes to `record`.
pub fn resolve_table<'g>(
    variants: &[Variant],
    measured: &[(String, Table)],
    geometry: impl Fn(&Variant) -> Option<&'g str>,
    mut record: impl FnMut(&Variant, String),
) -> Result<Table, String> {
    let mut table = measured.first().ok_or("no measured schedules")?.1.clone();
    for (id, extra) in measured.iter().skip(1) {
        for row in extra.rows() {
            if table.get(&row.key).is_some_and(|existing| existing != row) {
                return Err(format!(
                    "geometry {id}: conflicting measured schedules for shared shape key {}",
                    row.key
                ));
            }
            table.upsert(row.clone()).map_err(|e| e.to_string())?;
        }
    }
    for variant in variants.iter().filter(|v| covers(v)) {
        let key = key(variant);
        let geometry_id = geometry(variant);
        if let Some((id, own)) = measured
            .iter()
            .find(|(id, _)| geometry_id == Some(id.as_str()))
            && let Some(row) = own.get(&key)
        {
            validate_row(variant, row).map_err(|e| {
                format!(
                    "geometry {id} measured schedule {} for {} refused: {e}",
                    row.key, variant.symbol
                )
            })?;
        }
        let inherited = tuning::inherit(
            geometry_id,
            &key,
            measured.iter().map(|(id, table)| (id.as_str(), table)),
            |row| validate_row(variant, row).is_ok(),
        );
        let (row, status) = if let Some(inherited) = inherited {
            inherited
        } else {
            let knobs = served_knobs(variant)
                .ok_or_else(|| format!("{}: no engine schedule", variant.symbol))?;
            (
                Row {
                    key,
                    schedule: knobs.schedule().name().into(),
                    knobs: knobs.row_knobs(),
                },
                "engine default template schedule".into(),
            )
        };
        validate_row(variant, &row).map_err(|e| {
            format!(
                "{}: no legal inherited or default schedule: {e}",
                variant.symbol
            )
        })?;
        if table.get(&row.key).is_some_and(|existing| existing != &row) {
            return Err(format!(
                "{}: fallback conflicts with measured shared shape key {}",
                variant.symbol, row.key
            ));
        }
        record(variant, status);
        table.upsert(row).map_err(|e| e.to_string())?;
    }
    Ok(table)
}

/// The knobs of `row` for the served variant `v`, or why the build must
/// refuse them: another schedule than the engine's, other K splits, knobs
/// the template cannot run correctly ([`legality`]), or a
/// launch plan that does not fit at capacity ([`fits`]). The sweep's merge
/// applies the same checks.
pub fn validate_row(v: &Variant, row: &Row) -> Result<ScheduleKnobs, String> {
    let served = served_knobs(v).ok_or_else(|| format!("{}: no template schedule", v.symbol))?;
    let knobs = ScheduleKnobs::from_row(row)?;
    if knobs.arithmetic() != served.arithmetic() {
        return Err(format!(
            "{}: the schedule row changes the engine's reduction order",
            v.symbol
        ));
    }
    legality(v, &knobs).map_err(|e| format!("{}: {e}", v.symbol))?;
    fits(v, knobs).map_err(|e| format!("{}: {e}", v.symbol))?;
    Ok(knobs)
}

/// Live counts at `v`'s compiled capacity.
fn capacity_counts(v: &Variant) -> RuntimeCounts {
    RuntimeCounts {
        m: v.tokens_capacity,
        batch: 0,
        context: 0,
        routed_rows: v.query.m_capacity,
        split_k: v.knobs.split_k_capacity,
    }
}

/// Whether `knobs` have a planner-accepted launch plan at `v`'s capacity
/// within the dynamic shared memory a template instance may use. The rings
/// live in dynamic shared memory, so neither the template's static checks
/// nor the cubin audit enforce this.
pub fn fits(v: &Variant, knobs: ScheduleKnobs) -> Result<(), String> {
    match plan_with(v, knobs, capacity_counts(v)) {
        Ok(plan) if plan.dynamic_shared_bytes <= SHARED_LIMIT => Ok(()),
        Ok(plan) => Err(format!(
            "does not fit: {} bytes of dynamic shared memory",
            plan.dynamic_shared_bytes
        )),
        Err(e) => Err(format!("does not fit: {}", e.0)),
    }
}

/// Route slots of the persistent prefill schedule (`prefill::kRouteSlots`).
const PREFILL_ROUTE_SLOTS: u32 = 2;

/// Refuses knobs the template cannot execute correctly at `v`'s shape
/// although they compile, mirroring the template's static checks.
///
/// Persistent prefill: the row producer writes an item's route slot before
/// waiting on any activation ring slot, and the math warps read their item's
/// slot through its epilogue. With `PREFILL_ROUTE_SLOTS` slots and an
/// activation ring `activation_stages` deep, the producer can fill item
/// n + 2 while item n is still stored, unless an item has more K64 stages
/// than the ring: `activation_stages < (slots - 1) * K / 64`. A K = 256
/// instance (four stages) with four activation stages dropped output rows.
pub fn legality(v: &Variant, knobs: &ScheduleKnobs) -> Result<(), String> {
    if let ScheduleKnobs::Ring(r) = knobs {
        let granule = if v.fp8_fragment_weights() { 128 } else { 64 };
        if r.split_k == 0
            || !(1..=32).contains(&r.warps)
            || !r.warps.is_multiple_of(r.split_k)
            || !v.query.k.is_multiple_of(r.split_k * granule)
            || (v.projection == Projection::Down && r.split_k != 1)
            || !out_n(v).is_multiple_of(r.warps / r.split_k * 16)
        {
            return Err("illegal: ring warps and K splits must tile the projection".into());
        }
    }
    if let ScheduleKnobs::Prefill(p) = knobs {
        let k_tiles = v.query.k / 64;
        if p.activation_stages >= (PREFILL_ROUTE_SLOTS - 1) * k_tiles {
            return Err(format!(
                "illegal: {} activation stages reach the {} K64 stages of an item, so the \
                 row producer can reuse a route slot still in use",
                p.activation_stages, k_tiles
            ));
        }
    }
    Ok(())
}

/// The knobs of a variant's table row, if the template covers it.
///
/// Panics where [`try_knobs`] refuses the entry, such as when the template
/// covers it but its catalog resolved no schedule ([`scheduled`]).
pub fn knobs(v: &Variant) -> Option<ScheduleKnobs> {
    try_knobs(v).unwrap_or_else(|Refusal(reason)| panic!("{}: {reason}", v.symbol))
}

/// [`knobs`], or why the entry's engine and schedule disagree: a template
/// schedule replaces the engine it records, keeping its schedule kind and
/// K splits (the reduction order), so an entry has both or neither.
pub fn try_knobs(v: &Variant) -> Result<Option<ScheduleKnobs>, Refusal> {
    v.check_numerics()?;
    match (v.engine, v.schedule) {
        (Some(_), None) => Err(Refusal("no resolved schedule row")),
        (None, Some(_)) => Err(Refusal("a template schedule replaces no engine")),
        (Some(engine), Some(knobs)) if engine.arithmetic() != knobs.arithmetic() => {
            Err(Refusal("the schedule changes the engine's reduction order"))
        }
        (_, knobs) => Ok(knobs),
    }
}

/// The template's code for a served entry, if it has an instance. FP8
/// fragment cubins keep `CHOP_QWEN38_LOCAL_EXPERTS`, which the template does
/// not read, so their recorded configuration matches the replaced engine's.
///
/// Panics where [`try_implementation`] refuses the entry.
pub(crate) fn implementation(v: &Variant) -> Option<Implementation> {
    try_implementation(v).unwrap_or_else(|Refusal(reason)| panic!("{}: {reason}", v.symbol))
}

/// [`implementation`], or why the template cannot build the entry's code:
/// its engine and schedule disagree ([`try_knobs`]), or a scheduled entry
/// has no rank shape agreeing with its query, or a format the template
/// does not run.
pub(crate) fn try_implementation(v: &Variant) -> Result<Option<Implementation>, Refusal> {
    let Some(knobs) = try_knobs(v)? else {
        return Ok(None);
    };
    Ok(Some(Implementation {
        body: try_body(v, &knobs)?,
        header: knobs.schedule().header(),
        defines: if v.fp8_fragment_weights() {
            expert_parallel::cubin_defines(rank_shape(v)?.experts)
        } else {
            Vec::new()
        },
        threads: try_threads(&knobs)?,
    }))
}

/// The CTA threads of an instance at `knobs`.
pub fn threads(knobs: &ScheduleKnobs) -> u32 {
    try_threads(knobs).unwrap_or_else(|Refusal(reason)| panic!("{reason}"))
}

/// [`threads`], or why the knobs name more threads than a `u32` holds.
fn try_threads(knobs: &ScheduleKnobs) -> Result<u32, Refusal> {
    match knobs {
        ScheduleKnobs::Ring(r) => r
            .warps
            .checked_mul(32)
            .ok_or(Refusal("ring warps overflow the CTA threads")),
        ScheduleKnobs::Prefill(_) => Ok(FRAGMENT_PREFILL_THREADS),
    }
}

fn weights(v: &Variant) -> Result<&'static str, Refusal> {
    match v.query.weight {
        WeightFormat::Mxfp4 => Ok("mxfp4"),
        WeightFormat::Fp8Block128x128(_) => Ok("fp8_block128"),
        _ => Err(Refusal(
            "the template runs MXFP4 and FP8 block weights only",
        )),
    }
}

/// The rank shape of a variant's profile and class (`Shape::of_class`): its local
/// experts, top-k, hidden and per-rank intermediate. It must agree with the
/// variant's GEMM query.
pub fn shape(v: &Variant) -> Shape {
    try_shape(v).unwrap_or_else(|e| panic!("{e}"))
}

/// [`shape`], or why the variant has none: no MoE rank profile, or a query
/// that disagrees with it.
fn try_shape(v: &Variant) -> Result<Shape, String> {
    rank_shape(v).map_err(|Refusal(reason)| format!("{}: {reason}", v.symbol))
}

/// [`try_shape`], refused without the symbol.
fn rank_shape(v: &Variant) -> Result<Shape, Refusal> {
    let shape = v
        .profile
        .rank_shape()
        .and_then(|rank| Shape::of_class(rank, v.class))
        .ok_or(Refusal("not an MoE rank profile"))?;
    let group = v.query.grouped.ok_or(Refusal("not a grouped GEMM"))?;
    let (n, k) = shape.projection(v.projection);
    if (shape.experts, shape.top_k, n, k) != (group.experts, group.top_k, v.query.n, v.query.k) {
        return Err(Refusal("rank shape and query differ"));
    }
    Ok(shape)
}

fn out_n(v: &Variant) -> u32 {
    match v.projection {
        Projection::GateUp => v.query.n / 2,
        Projection::Down => v.query.n,
    }
}

/// The table key of a served variant: `grouped_moe.<projection>.<phase>`
/// (`exact` for exact entries), its weight format, rank shape and token
/// capacity.
pub fn key(v: &Variant) -> Key {
    let s = shape(v);
    let projection = match v.projection {
        Projection::GateUp => "gate_up",
        Projection::Down => "down",
    };
    let phase = match v.phase {
        Phase::Decode if v.exact => "exact",
        Phase::Decode => "decode",
        Phase::Prefill => "prefill",
    };
    Key {
        op: format!("grouped_moe.{projection}.{phase}"),
        weights: weights(v)
            .unwrap_or_else(|Refusal(reason)| panic!("{}: {reason}", v.symbol))
            .into(),
        shape: [
            ("experts".to_string(), s.experts),
            ("hidden".to_string(), s.hidden),
            ("intermediate".to_string(), s.intermediate),
            ("top_k".to_string(), s.top_k),
        ]
        .into(),
        bucket: v.tokens_capacity,
    }
}

/// The schedule table in `text`, for this crate's SM120 target.
pub fn parse_table(text: &str) -> Result<Table, Refusal> {
    let table = Table::parse(text).map_err(|_| Refusal("schedule table is not canonical"))?;
    if (table.target.compute_capability, table.target.sm_count) != (120, SM_COUNT) {
        return Err(Refusal("schedule table targets another device"));
    }
    Ok(table)
}

/// The kernel base's own schedule table.
pub fn shared_table() -> Table {
    static PARSED: OnceLock<Table> = OnceLock::new();
    PARSED
        .get_or_init(|| parse_table(SHARED_TABLE).expect("tuning/schedules.json"))
        .clone()
}

/// Bytes of one ring slot (`ring_slot_bytes`) and of the one-token row the
/// CTA stages beside the rings.
fn ring_slot_and_row_bytes(v: &Variant) -> Option<(u32, u32)> {
    let parts = if v.projection == Projection::GateUp {
        2
    } else {
        1
    };
    let part = if v.fp8_fragment_weights() {
        1024
    } else {
        512 + 128
    };
    let tokens = v.tokens_capacity;
    if tokens > 1 {
        Some((
            tokens
                .checked_mul(RING_ROW_BYTES)?
                .checked_add(parts * part)?,
            0,
        ))
    } else {
        Some((parts * part, v.query.k.checked_mul(2)?))
    }
}

fn ring_shared_bytes(v: &Variant, r: RingKnobs) -> Result<u32, Refusal> {
    ring_slot_and_row_bytes(v)
        .and_then(|(slot, row)| {
            r.warps
                .checked_mul(r.depth)?
                .checked_mul(slot)?
                .checked_add(row)
        })
        .ok_or(Refusal("ring shared memory overflows"))
}

/// The template schedule of a served A16 fragment grouped entry: that of the
/// engine its definition declares (`Variant::engine`).
pub fn schedule(v: &Variant) -> Option<Schedule> {
    v.engine.as_ref().map(ScheduleKnobs::schedule)
}

/// The knobs that reproduce the per-model engine's schedule, as the
/// entry's definition declares them.
pub fn served_knobs(v: &Variant) -> Option<ScheduleKnobs> {
    v.engine
}

/// The template entry body of `v` at `knobs`.
pub fn body(v: &Variant, knobs: &ScheduleKnobs) -> String {
    try_body(v, knobs).unwrap_or_else(|Refusal(reason)| panic!("{}: {reason}", v.symbol))
}

/// [`body`], or why the entry has no instance: no rank shape agreeing with
/// its query, or a format the template does not run.
fn try_body(v: &Variant, knobs: &ScheduleKnobs) -> Result<String, Refusal> {
    let s = rank_shape(v)?;
    let shape = format!(
        "chop::moe::grouped::Shape{{.experts = {}, .top_k = {}, .hidden = {}, .intermediate = {}}}",
        s.experts, s.top_k, s.hidden, s.intermediate
    );
    let format = format!("chop::moe::grouped::Weights::{}", weights(v)?);
    let gate_up = v.projection == Projection::GateUp;
    let (function, schedule, extra) = match knobs {
        ScheduleKnobs::Ring(r) => (
            format!(
                "ring_decode<{format}, {shape}, {gate_up}, {}",
                v.tokens_capacity
            ),
            format!(
                "chop::moe::grouped::RingSchedule{{.split_k = {}, .warps = {}, .depth = {}, .early_row = {}, .reload_rows = {}}}",
                r.split_k, r.warps, r.depth, r.early_row, r.reload_rows,
            ),
            "active_experts",
        ),
        ScheduleKnobs::Prefill(p) => (
            format!("persistent_prefill<{format}, {shape}, {gate_up}"),
            format!(
                "chop::moe::grouped::PrefillSchedule{{.weight_stages = {}, .activation_stages = {}, .trim_gather = {}, .bulk_weights = {}}}",
                p.weight_stages, p.activation_stages, p.trim_gather, p.bulk_weights
            ),
            "",
        ),
    };
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!(", {extra}")
    };
    Ok(format!(
        "chop::moe::grouped::{function}, {schedule}>(activation, weight_bytes, weight_scales, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens{extra});"
    ))
}

/// The served variant's cubin name.
fn cubin_name(v: &Variant) -> String {
    let base = v.symbol.trim_end_matches("_v1");
    if v.phase == Phase::Decode {
        base.rsplit_once("_t")
            .expect("decode bucket suffix")
            .0
            .into()
    } else {
        base.into()
    }
}

/// The template instance of `v` at `knobs`.
pub fn instance(v: Variant, knobs: ScheduleKnobs) -> Instance {
    let cubin = cubin_name(&v);
    Instance {
        schedule: knobs.schedule(),
        knobs,
        template_symbol: v.symbol.clone(),
        engine_symbol: v.symbol.replacen("chop_moe_", ENGINE_PREFIX, 1),
        engine_cubin: cubin.replacen("chop_moe_", ENGINE_PREFIX, 1),
        template_cubin: cubin,
        body: body(&v, &knobs),
        baseline: v,
    }
}

/// Instances of every served A16 fragment grouped entry of `catalogs`,
/// with the knobs of their table rows (see [`knobs`]).
pub fn lab_instances(catalogs: &Catalogs) -> Vec<Instance> {
    catalog(catalogs)
        .into_iter()
        .filter_map(|v| Some(instance(v.clone(), knobs(&v)?)))
        .collect()
}

/// Instances of every served A16 fragment grouped entry of `catalogs` at
/// the per-model engines' own schedules (`served_knobs`), whatever the
/// table rows hold: what the frozen engine references are built from.
pub fn engine_schedule_instances(catalogs: &Catalogs) -> Vec<Instance> {
    catalog(catalogs)
        .into_iter()
        .filter_map(|v| {
            let knobs = served_knobs(&v)?;
            Some(instance(v, knobs))
        })
        .collect()
}

/// The table whose rows reproduce the schedule of every served engine of
/// `catalogs`: a new model's initial rows.
pub fn served_table(catalogs: &Catalogs) -> Table {
    let mut table = Table::new(tuning::Target {
        compute_capability: 120,
        sm_count: SM_COUNT,
    });
    for v in catalog(catalogs) {
        let Some(knobs) = served_knobs(&v) else {
            continue;
        };
        let row = Row {
            key: key(&v),
            schedule: knobs.schedule().name().into(),
            knobs: knobs.row_knobs(),
        };
        if let Some(previous) = table.upsert(row.clone()).expect("valid served row")
            && previous != row
        {
            panic!("{}: two served entries share a key", v.symbol);
        }
    }
    table
}

/// Every knob set the offline sweep considers for `instance`, other than
/// its own, sorted by knobs: the same arithmetic
/// ([`ScheduleKnobs::arithmetic`]), every width and ring depth, and the
/// staging flags that apply to the bucket. Each is a candidate, or the
/// reason it is refused: illegal ([`legality`]), or no planner-accepted
/// launch plan within the shared-memory limit at capacity.
pub fn candidate_space(instance: &Instance) -> Vec<(ScheduleKnobs, Result<Instance, String>)> {
    let v = &instance.baseline;
    let gate_up = v.projection == Projection::GateUp;
    let tokens = v.tokens_capacity;
    let mut space = Vec::new();
    match instance.knobs {
        ScheduleKnobs::Ring(current) => {
            let fp8 = v.fp8_fragment_weights();
            let parts = if gate_up { 2 } else { 1 };
            let (slot, _) = ring_slot_and_row_bytes(v).expect("a served instance's ring slot fits");
            let k_part_tiles = v.query.k / current.split_k / 64;
            let early = if gate_up && tokens == 1 {
                vec![false, true]
            } else {
                vec![current.early_row]
            };
            let reload = if tokens > 1 {
                vec![false, true]
            } else {
                vec![current.reload_rows]
            };
            for strips in 1..=32 / current.split_k {
                let warps = strips * current.split_k;
                if !out_n(v).is_multiple_of(strips * 16) {
                    continue;
                }
                for depth in 1..=k_part_tiles {
                    if depth * slot < 32 * parts * tokens.div_ceil(8) * 16 {
                        continue;
                    }
                    for &early_row in &early {
                        for &reload_rows in &reload {
                            space.push(ScheduleKnobs::Ring(RingKnobs {
                                split_k: current.split_k,
                                warps,
                                depth,
                                early_row,
                                reload_rows,
                            }));
                        }
                    }
                }
            }
            debug_assert!(!fp8 || v.query.k.is_multiple_of(current.split_k * 128));
        }
        ScheduleKnobs::Prefill(_) => {
            let bulk = if gate_up && v.fp8_fragment_weights() {
                vec![false, true]
            } else {
                vec![false]
            };
            for weight_stages in 2..=8 {
                for activation_stages in 2..=4 {
                    for trim_gather in [false, true] {
                        for &bulk_weights in &bulk {
                            space.push(ScheduleKnobs::Prefill(PrefillKnobs {
                                weight_stages,
                                activation_stages,
                                trim_gather,
                                bulk_weights,
                            }));
                        }
                    }
                }
            }
        }
    }
    space.sort();
    space
        .into_iter()
        .filter(|knobs| *knobs != instance.knobs)
        .map(|knobs| {
            let candidate = legality(v, &knobs)
                .and_then(|()| fits(v, knobs))
                .and_then(|()| instance.with_knobs(knobs).map_err(|e| e.0.to_string()));
            (knobs, candidate)
        })
        .collect()
}

/// The candidates of [`candidate_space`] that are legal and fit.
pub fn candidates(instance: &Instance) -> Vec<Instance> {
    candidate_space(instance)
        .into_iter()
        .filter_map(|(_, candidate)| candidate.ok())
        .collect()
}

/// The entry of an instance's cubin.
pub fn entry(instance: &Instance) -> EntrySpec {
    EntrySpec::new(
        instance.template_symbol.clone(),
        parameters(),
        instance.body.clone(),
        threads(&instance.knobs),
        1,
    )
}

fn new_spec(instance: &Instance) -> CubinSpec {
    CubinSpec {
        name: instance.template_cubin.clone(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("common"), PathBuf::from("moe")],
        headers: vec![instance.schedule.header().to_string()],
        defines: Vec::new(),
        entries: Vec::new(),
    }
}

/// A packed NVFP4 or MXFP8 decode entry at one shape and bucket, as the
/// packed engine (`moe.cuh`) runs it and as the template's packed ring
/// (`grouped_packed.cuh`) runs it: the same request, bank, entry ABI and
/// tensor map, with the ring's knobs from `packed_ring_knobs`.
#[derive(Clone, Debug)]
pub struct PackedInstance {
    pub profile: Profile,
    /// The instance's schedule-table key ([`packed_key`]).
    pub key: Key,
    pub engine: GroupedOperation,
    pub engine_cubin: String,
    pub template: GroupedOperation,
    pub template_cubin: String,
}

/// The schedule-table key of a packed NVFP4 or MXFP8 decode projection:
/// `grouped_moe.<projection>.decode`, the format, the rank shape and the
/// token capacity. Its row's schedule is `packed_ring`.
pub fn packed_key(
    shape: Shape,
    projection: Projection,
    tokens_capacity: u32,
    weight: WeightFormat,
) -> Key {
    let projection = match projection {
        Projection::GateUp => "gate_up",
        Projection::Down => "down",
    };
    Key {
        op: format!("grouped_moe.{projection}.decode"),
        weights: packed_label(weight).into(),
        shape: [
            ("experts".to_string(), shape.experts),
            ("hidden".to_string(), shape.hidden),
            ("intermediate".to_string(), shape.intermediate),
            ("top_k".to_string(), shape.top_k),
        ]
        .into(),
        bucket: tokens_capacity,
    }
}

/// A packed staged schedule as a table row holds it.
pub fn packed_staged_row_knobs(staged: PackedStaged) -> tuning::Knobs {
    [
        ("depth".to_string(), Knob::Count(u32::from(staged.depth))),
        (
            "producers".to_string(),
            Knob::Count(u32::from(staged.producers)),
        ),
        ("stages".to_string(), Knob::Count(u32::from(staged.stages))),
    ]
    .into()
}

/// The schedule name and row knobs of packed knobs.
pub fn packed_row(knobs: CompileKnobs) -> (&'static str, tuning::Knobs) {
    if knobs.tile_k == PACKED_STAGED_TILE_K {
        (
            "packed_staged",
            packed_staged_row_knobs(PackedStaged::of(knobs)),
        )
    } else {
        ("packed_ring", packed_ring_row_knobs(PackedRing::of(knobs)))
    }
}

/// A packed ring as a table row holds it; `prefetch` only when nonzero.
pub fn packed_ring_row_knobs(ring: PackedRing) -> tuning::Knobs {
    let mut knobs: tuning::Knobs = [
        ("depth".to_string(), Knob::Count(u32::from(ring.depth))),
        ("split_parts".to_string(), Knob::Flag(ring.split_parts)),
        ("strips".to_string(), Knob::Count(u32::from(ring.strips))),
    ]
    .into();
    if ring.prefetch != 0 {
        knobs.insert("prefetch".into(), Knob::Count(u32::from(ring.prefetch)));
    }
    knobs
}

/// The packed ring knobs of `row` for a packed decode projection, or why
/// the build must refuse them, as [`validate_row`] does for the fragment
/// schedules: another schedule than `packed_ring`, missing or stray knobs,
/// split parts on down, knobs the ring cannot run, or a launch plan above
/// the template's shared-memory limit at capacity. The ring never splits
/// K, so no row can change the packed engine's reduction order.
pub fn validate_packed_row(
    shape: Shape,
    projection: Projection,
    tokens_capacity: u32,
    weight: WeightFormat,
    row: &Row,
) -> Result<CompileKnobs, String> {
    let key = &row.key;
    let count = |name: &str| {
        row.knobs
            .get(name)
            .and_then(|k| k.count())
            .ok_or_else(|| format!("{key}: knob {name} needs a count"))
    };
    let small = |name: &str| {
        count(name).and_then(|v| u8::try_from(v).map_err(|_| format!("{key}: {name} overflow")))
    };
    let knobs = match row.schedule.as_str() {
        "packed_ring" => {
            let ring = PackedRing {
                strips: u16::try_from(count("strips")?)
                    .map_err(|_| format!("{key}: strips overflow"))?,
                split_parts: row
                    .knobs
                    .get("split_parts")
                    .and_then(|k| k.flag())
                    .ok_or_else(|| format!("{key}: knob split_parts needs a flag"))?,
                depth: small("depth")?,
                prefetch: match row.knobs.get("prefetch") {
                    None => 0,
                    Some(k) => k
                        .count()
                        .filter(|p| matches!(p, 64 | 128 | 256))
                        .ok_or_else(|| format!("{key}: prefetch is 64, 128 or 256"))?
                        as u16,
                },
            };
            if packed_ring_row_knobs(ring) != row.knobs {
                return Err(format!("{key}: unknown knobs for packed_ring"));
            }
            if ring.split_parts && projection != Projection::GateUp {
                return Err(format!("{key}: only gate/up has parts to split"));
            }
            ring.checked_knobs().ok_or_else(|| {
                format!(
                    "{key}: {} strips need {} warps; a CTA has 1 to {PACKED_MAX_WARPS}",
                    ring.strips,
                    ring.warps()
                )
            })?
        }
        "packed_staged" => {
            let staged = PackedStaged {
                producers: small("producers")?,
                depth: small("depth")?,
                stages: small("stages")?,
            };
            if packed_staged_row_knobs(staged) != row.knobs {
                return Err(format!("{key}: unknown knobs for packed_staged"));
            }
            staged.checked_knobs().ok_or_else(|| {
                format!(
                    "{key}: {} producers and a consumer exceed {PACKED_MAX_WARPS} warps",
                    staged.producers
                )
            })?
        }
        _ => {
            return Err(format!(
                "{key}: packed decode rows use the packed_ring or packed_staged schedule"
            ));
        }
    };
    let operation = instantiate(packed_request(
        shape,
        projection,
        tokens_capacity,
        weight,
        Some(knobs),
        "packed_row_check_v1".into(),
    ))
    .map_err(|e| format!("{key}: {}", e.0))?;
    let plan = operation
        .plan(RuntimeCounts {
            m: tokens_capacity,
            batch: 1,
            context: 0,
            routed_rows: operation.query.m_capacity,
            split_k: 1,
        })
        .map_err(|e| format!("{key}: {}", e.0))?;
    if plan.dynamic_shared_bytes > SHARED_LIMIT {
        return Err(format!(
            "{key}: does not fit: {} bytes of dynamic shared memory",
            plan.dynamic_shared_bytes
        ));
    }
    Ok(knobs)
}

/// The expert request of a packed decode entry. NVFP4 carries the
/// per-expert, per-projection F32 globals of the checkpoint exports; MXFP8
/// has none.
pub fn packed_request(
    shape: Shape,
    projection: Projection,
    tokens_capacity: u32,
    weight: WeightFormat,
    knobs: Option<CompileKnobs>,
    symbol: String,
) -> GroupedRequest {
    GroupedRequest {
        shape,
        projection,
        route_weight: RouteWeight::AfterDown,
        swiglu_clamp: None,
        phase: Phase::Decode,
        tokens_capacity,
        weight,
        activation: ActivationFormat::Bf16,
        scales: if weight == WeightFormat::Nvfp4 {
            ScaleBindings {
                weight_globals: GlobalScaleLayout::ExpertProjection,
                activation_globals: GlobalScaleLayout::Absent,
                shared_gate_up_activation: true,
            }
        } else {
            ScaleBindings::NONE
        },
        output_quantization: None,
        swiglu_columns: SwigluColumns::ConcatenatedHalves,
        target: Target::Sm120f,
        symbol,
        knobs,
    }
}

fn packed_label(weight: WeightFormat) -> &'static str {
    if weight == WeightFormat::Nvfp4 {
        "nvfp4"
    } else {
        "mxfp8"
    }
}

/// Packed NVFP4 and MXFP8 decode instances of `profile` at the expert
/// `shape`, every decode bucket and both projections, with the ring knobs
/// of `table`'s `packed_ring` rows, named `chop_moe_{tag}_packed_*`. A
/// model's served packed entries are these template instances under their
/// own symbols; a format it does not serve is compared with the packed
/// engine at the same shape.
/// The schedule keys of the template entries among `variants`: the rows a
/// model's schedule table holds for its grouped entries.
pub fn schedule_keys(variants: &[Variant]) -> Vec<Key> {
    variants.iter().filter(|v| covers(v)).map(key).collect()
}

/// The `packed_ring` row keys of `shape`'s packed NVFP4 and MXFP8 decode
/// buckets, as [`packed_lab_instances`] reads them.
pub fn packed_keys(shape: Shape) -> Vec<Key> {
    let mut keys = Vec::new();
    for weight in [WeightFormat::Nvfp4, WeightFormat::Mxfp8] {
        for projection in [Projection::GateUp, Projection::Down] {
            for tokens in DECODE_BUCKETS {
                keys.push(packed_key(shape, projection, tokens, weight));
            }
        }
    }
    keys
}

pub fn packed_lab_instances(table: &Table, profile: Profile, shape: Shape) -> Vec<PackedInstance> {
    let tag = profile.tag();
    let mut result = Vec::new();
    for weight in [WeightFormat::Nvfp4, WeightFormat::Mxfp8] {
        let label = packed_label(weight);
        for projection in [Projection::GateUp, Projection::Down] {
            let name = if projection == Projection::GateUp {
                "gateup"
            } else {
                "down"
            };
            for tokens in DECODE_BUCKETS {
                let symbol = format!("chop_moe_{tag}_packed_{label}_{name}_decode_t{tokens}_v1");
                let knobs = packed_ring_knobs(
                    table,
                    shape,
                    projection,
                    Phase::Decode,
                    tokens,
                    weight,
                    ActivationFormat::Bf16,
                )
                .expect("a packed_ring row in the schedule table");
                let template = instantiate(packed_request(
                    shape,
                    projection,
                    tokens,
                    weight,
                    Some(knobs),
                    symbol.clone(),
                ))
                .expect("packed ring instance");
                let engine = instantiate(packed_request(
                    shape,
                    projection,
                    tokens,
                    weight,
                    None,
                    symbol.replacen("chop_moe_", ENGINE_PREFIX, 1),
                ))
                .expect("packed engine instance");
                let cubin = format!("chop_moe_{tag}_packed_{label}");
                result.push(PackedInstance {
                    profile,
                    key: packed_key(shape, projection, tokens, weight),
                    engine,
                    engine_cubin: cubin.replacen("chop_moe_", ENGINE_PREFIX, 1),
                    template,
                    template_cubin: cubin,
                });
            }
        }
    }
    result
}

/// Every packed schedule the offline packed sweep considers for
/// `instance`, other than its row's: rings of one to eight strips, split
/// gate/up parts, ring depths from 1 to K/64 on a coarse grid and no, 128-
/// or 256-byte L2 prefetch, and at
/// one token staged schedules of 1 to 8 producers with producer and
/// converted-step depths. Each is a candidate under
/// `chop_moe_tune_<model>_packed_<format>_<projection>_decode_t<T>_<tag>_v1`
/// (ring `s<S>[p]_d<D>`, staged `w<P>_d<D>_c<C>`), or the reason
/// [`validate_packed_row`] refuses it. No packed schedule splits K, so every
/// candidate keeps the packed engine's arithmetic.
pub fn packed_candidates(
    instance: &PackedInstance,
) -> Vec<(CompileKnobs, Result<GroupedOperation, String>)> {
    let t = &instance.template;
    let group = t.query.grouped.expect("grouped GEMM");
    let gate_up = t.projection == Projection::GateUp;
    let shape = Shape {
        top_k: group.top_k,
        hidden: if gate_up { t.query.k } else { t.query.n },
        experts: group.experts,
        intermediate: if gate_up { t.query.n / 2 } else { t.query.k },
    };
    let k_tiles = t.query.k / 64;
    let grid = |cap: u32| {
        let mut depths: Vec<u32> = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40]
            .into_iter()
            .filter(|&d| d <= cap)
            .collect();
        if !depths.contains(&cap) {
            depths.push(cap);
        }
        depths
    };
    let mut schedules = Vec::new();
    let splits: &[bool] = if gate_up { &[false, true] } else { &[false] };
    for prefetch in [0u16, 128, 256] {
        for strips in [1u16, 2, 4, 8] {
            for &split_parts in splits {
                for depth in grid(k_tiles) {
                    let ring = PackedRing {
                        strips,
                        split_parts,
                        depth: depth as u8,
                        prefetch,
                    };
                    let fetch = if prefetch == 0 {
                        String::new()
                    } else {
                        format!("_f{prefetch}")
                    };
                    schedules.push((
                        ring.knobs(),
                        format!(
                            "s{strips}{}_d{depth}{fetch}",
                            if split_parts { "p" } else { "" }
                        ),
                    ));
                }
            }
        }
    }
    if t.tokens_capacity == 1 {
        for producers in [1u32, 2, 3, 4, 5, 6, 8] {
            for depth in grid(k_tiles.div_ceil(producers)) {
                for stages in [producers, 2 * producers, 4 * producers] {
                    if stages > k_tiles {
                        continue;
                    }
                    let staged = PackedStaged {
                        producers: producers as u8,
                        depth: depth as u8,
                        stages: stages as u8,
                    };
                    schedules.push((staged.knobs(), format!("w{producers}_d{depth}_c{stages}")));
                }
            }
        }
    }
    let mut result = Vec::new();
    for (knobs, tag) in schedules {
        if knobs == t.knobs {
            continue;
        }
        let (schedule, row_knobs) = packed_row(knobs);
        let row = Row {
            key: instance.key.clone(),
            schedule: schedule.into(),
            knobs: row_knobs,
        };
        let candidate =
            validate_packed_row(shape, t.projection, t.tokens_capacity, t.query.weight, &row)
                .and_then(|knobs| {
                    let symbol = format!(
                        "{}_{tag}_v1",
                        t.symbol
                            .trim_end_matches("_v1")
                            .replacen("chop_moe_", TUNE_PREFIX, 1),
                    );
                    instantiate(packed_request(
                        shape,
                        t.projection,
                        t.tokens_capacity,
                        t.query.weight,
                        Some(knobs),
                        symbol,
                    ))
                    .map_err(|e| e.0.to_string())
                });
        result.push((knobs, candidate));
    }
    result
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn inherited_ring_warps_must_tile_output_columns() {
        use kernel_core::rank_shape::{Block, BlockClass, ExpertSplit, ModelConfig, MoeConfig};
        const MODEL: crate::Model = crate::Model::new(
            crate::profile_names!("small-experts", "small_experts", 4, 2),
            ModelConfig {
                hidden: 1024,
                vocab: 2048,
                linear: None,
                ngram_heads: 0,
                blocks: &[BlockClass {
                    name: "moe",
                    layers: (0, 1),
                    block: Block::Moe(MoeConfig {
                        experts: 128,
                        top_k: 8,
                        intermediate: 512,
                        split: ExpertSplit::Experts,
                        shared_intermediate: 0,
                        shared_tile: 128,
                    }),
                }],
            },
        );
        let variant = expert_parallel::variant(
            MODEL.at(4),
            (
                "moe",
                Shape {
                    experts: 32,
                    hidden: 1024,
                    intermediate: 512,
                    top_k: 8,
                },
            ),
            Projection::Down,
            16,
            expert_parallel::Tuning::Decode { warps: 4, split: 1 },
            expert_parallel::Numerics {
                route_weight: RouteWeight::AfterDown,
                swiglu_clamp: None,
            },
            crate::invariance::Invariance::NONE,
        )
        .unwrap();
        let ScheduleKnobs::Ring(mut ring) = served_knobs(&variant).unwrap() else {
            panic!("ring schedule");
        };
        ring.warps = 5;
        let row = Row {
            key: key(&variant),
            schedule: "ring".into(),
            knobs: ScheduleKnobs::Ring(ring).row_knobs(),
        };
        assert!(validate_row(&variant, &row).is_err());
        ring.warps = 4;
        let row = Row {
            knobs: ScheduleKnobs::Ring(ring).row_knobs(),
            ..row
        };
        validate_row(&variant, &row).unwrap();
    }
}
