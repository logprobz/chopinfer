//! Gated MLP projections through the entries a model selects: the grouped
//! expert engines' (`moe::Variant`: packed, MXFP4 fragment, template and
//! FP8 fragment entries) and the chained engine's routed and dense
//! projections (`moe::chained::Projection`). Each prepared op launches its
//! entry, and a grouped entry's split-K reduction, as the builders plan
//! them.
//!
//! A chained op holds a schedule per row count, prepared from the model's
//! chained registrations of its contract ([`chained_table`]): a step of
//! `rows` rows runs the forms of the least capacity that holds them, in
//! their tile-pair form from [`TILE_PAIR_MIN_ROWS`] rows, else at a prefill
//! use in their decoded-tile form, else in their one other form. Its
//! grouped forms' row tiles follow the chained routes the model registered,
//! which pad each row count's routes the same way ([`check_route_tiles`]).
//! A table that runs a tile-pair or decoded-tile form is admitted against
//! the resources the model's lock reports for it ([`admit`]).

use std::collections::BTreeMap;

use kernel_core::backend::{Counts, Launch, Need, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::{
    ActivationFormat, ArgumentList, CompileKnobs, LaunchPlan, Refusal, RoutedCapacity, Validate,
    WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::occupancy::EntryResources;
use kernel_core::precision::buffers::ProjectionRouting;
use kernel_core::precision::{ExecutionPhase, OperationUse, WeightLayout};
use kernel_core::properties::{Invariance, Rows};

use super::projection::GroupedScales;
use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, SHARED_MEMORY_BYTES, check_plan};
use crate::moe::chained::{self, LocalRoute, Schedule, Weights};
use crate::moe::{self, Phase, Projection, RouteWeight, Variant};

/// The gated MLP entries of a model's selection.
#[derive(Default)]
pub(super) struct Selection {
    pub(super) grouped: Vec<Variant>,
    pub(super) chained: Vec<chained::Projection>,
    /// The resources the model's locks report, by symbol.
    pub(super) resources: BTreeMap<String, EntryResources>,
}

/// Each entry's resources in a `kernels.lock`, as ptxas reported them, by
/// symbol; a symbol built in more than one cubin keeps the most of each
/// resource.
pub fn locked_resources(lock: &str) -> Result<BTreeMap<String, EntryResources>, Refusal> {
    let records: serde_json::Value =
        serde_json::from_str(lock).map_err(|_| Refusal("a kernels.lock is JSON"))?;
    let mut resources = BTreeMap::<String, EntryResources>::new();
    let unreadable = Refusal("a kernels.lock lists each record's resources");
    for record in records.as_array().ok_or(unreadable)? {
        for entry in record["resources"].as_array().ok_or(unreadable)? {
            let field = |name: &str| {
                entry[name]
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or(unreadable)
            };
            let symbol = entry["symbol"].as_str().ok_or(unreadable)?.to_owned();
            let found = EntryResources {
                registers: field("registers")?,
                shared_static_bytes: field("shared_static_bytes")?,
            };
            resources
                .entry(symbol)
                .and_modify(|kept| {
                    kept.registers = kept.registers.max(found.registers);
                    kept.shared_static_bytes =
                        kept.shared_static_bytes.max(found.shared_static_bytes);
                })
                .or_insert(found);
        }
    }
    Ok(resources)
}

/// CTAs of `threads` threads one SM of the target keeps resident at once
/// with `resources` and `dynamic_shared` bytes of dynamic shared memory
/// (`crate::device::TARGET`).
fn resident_ctas(resources: EntryResources, threads: u32, dynamic_shared: u32) -> u32 {
    crate::device::TARGET.ctas_per_sm(resources, threads, dynamic_shared)
}

/// Admits a chained form the op runs at `plan` against its locked
/// `resources`. A tile-pair form runs eight warps per CTA with static
/// shared memory only, and its resources keep at least one CTA resident per
/// SM (its CTAs do not wait on each other, so one is all a launch needs).
/// A decoded-tile form's static shared bytes are those
/// `chained::decoded_tile_static_bytes` bounds (so the engine's refusal of
/// longer K holds for the compiled entry), they and its stages fit one
/// SM120 block, and at least one CTA stays resident per SM. Other forms
/// need no admission.
fn admit(
    projection: &chained::Projection,
    plan: &LaunchPlan,
    resources: &BTreeMap<String, EntryResources>,
) -> Result<(), Refused> {
    let symbol = &projection.symbol;
    let locked = || {
        resources.get(symbol).copied().ok_or_else(|| {
            Refused::Rule(format!(
                "{symbol}: no resources registered (Sm120::with_entry_resources)"
            ))
        })
    };
    let dynamic = plan.dynamic_shared_bytes;
    match projection.schedule {
        Schedule::TilePairs { .. } => {
            if plan.block != [chained::TILE_PAIR_THREADS, 1, 1] || dynamic != 0 {
                return Err(Refused::Rule(format!(
                    "{symbol}: tile pairs run {} threads with no dynamic shared memory",
                    chained::TILE_PAIR_THREADS
                )));
            }
            let locked = locked()?;
            if resident_ctas(locked, plan.block[0], dynamic) == 0 {
                return Err(Refused::Rule(format!(
                    "{symbol}: {} registers and {} static shared bytes over {} threads keep no \
                     CTA resident",
                    locked.registers, locked.shared_static_bytes, plan.block[0]
                )));
            }
        }
        Schedule::DecodedTiles => {
            let locked = locked()?;
            let bound = chained::decoded_tile_static_bytes(projection.query.gemm.k);
            if locked.shared_static_bytes != bound
                || bound + dynamic > SHARED_MEMORY_BYTES
                || resident_ctas(locked, plan.block[0], dynamic) == 0
            {
                return Err(Refused::Rule(format!(
                    "{symbol}: {} static shared bytes (bounded at {bound}) and {dynamic} dynamic \
                     over {} threads do not fit an SM120 block",
                    locked.shared_static_bytes, plan.block[0]
                )));
            }
        }
        Schedule::Columns | Schedule::BlockWarps | Schedule::WarpStreams => {}
    }
    Ok(())
}

/// The fewest rows a step runs a chained projection's tile-pair form at
/// (`chained::Schedule::TilePairs`); fewer run its other form. Kernel-lab
/// measured tile pairs 1.05 to 1.14 times as fast as columns at 2,048 to
/// 8,192 tokens, uniform and captured routing (PERF-B11); nothing smaller
/// was measured.
pub const TILE_PAIR_MIN_ROWS: u32 = 2048;

/// A selected entry of either engine.
#[derive(Clone, Copy, Debug)]
enum Entry<'a> {
    Grouped(&'a Variant),
    Chained(&'a chained::Projection),
}

/// The projection `op` at `usage` through the selected entry that serves
/// it at `need`: its contract equals `op`, and of the entries that hold
/// `need.capacity` tokens and guarantee `need.exact`, it has the least
/// token capacity. The step's phase ranks the grouped entries of that
/// capacity: verify steps run exact entries (`Variant::exact`, which sum in
/// the one-token entry's order), other steps ordinary ones, and a step
/// runs the other kind where its capacity has only that. Entries left that
/// would launch differently are a tie. A grouped entry whose cubin the
/// build cannot compile (`Variant::buildable`) never serves; it refuses
/// the op only when no other entry of its contract is registered.
/// An entry's padded routed capacity is the engine's: `op` names the
/// routes without padding, and each entry is matched by its logical
/// contract. Chained entries are matched by the contract's math alone, so
/// every capacity of it is a form the op's schedule table may run
/// ([`chained_table`]); the table's form at the capacity competes with the
/// grouped entries. `local` are the model's chained routes, which a grouped
/// table's row tiles must agree with.
pub(super) fn prepare(
    selection: &Selection,
    local: &[(LocalRoute, String)],
    op: &GatedMlpProjectionQuery,
    usage: OperationUse,
    need: Need,
) -> Result<Box<dyn PreparedExperts>, Refused> {
    op.validate()?;
    if *op != op.logical() {
        return Err(Refused::Rule(
            "a grouped projection requests its routed rows without padding".into(),
        ));
    }
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(
            "a prepared op serves 1 to MAX_CAPACITY rows".into(),
        ));
    }
    let verify = usage.phase == ExecutionPhase::Verify;
    let (buildable, refused): (Vec<_>, Vec<_>) = selection
        .grouped
        .iter()
        .filter(|variant| grouped_contract(variant) == *op)
        .partition(|variant| variant.buildable().is_ok());
    let forms = selection
        .chained
        .iter()
        .filter(|projection| same_math(&projection.query, op))
        .collect::<Vec<_>>();
    if buildable.is_empty()
        && forms.is_empty()
        && let Some(Refusal(reason)) = refused
            .iter()
            .filter_map(|variant| variant.buildable().err())
            .min_by_key(|refusal| refusal.0)
    {
        return Err(Refused::Rule(format!(
            "no SM120 gated MLP entry for the op can be built: {reason}"
        )));
    }
    let grouped = buildable.into_iter().map(|variant| {
        let launches = grouped_launch(variant);
        Candidate::new(
            Entry::Grouped(variant),
            &variant.symbol,
            launches,
            variant.invariance,
        )
        .capacity(variant.tokens_capacity)
        .rank(u64::from(variant.exact != verify))
    });
    // The chained form that runs the capacity's rows stands for the
    // chained table; a chained projection declares its own claims
    // (`chained::Projection::invariance`), and the table's op claims what
    // its forms claim together (`chained_guarantees`).
    let at_capacity = (!forms.is_empty())
        .then(|| chained_form(&forms, need.capacity, usage.phase, need))
        .transpose();
    let at_capacity = match at_capacity {
        Ok(form) => form,
        Err(refused) if buildable_is_empty(&selection.grouped, op) => return Err(refused),
        Err(_) => None,
    };
    let chained = at_capacity.map(|projection| {
        Candidate::new(
            Entry::Chained(projection),
            &projection.symbol,
            chained_launch(projection),
            projection.invariance(),
        )
        .capacity(projection.tokens_capacity)
    });
    // The op claims no more than every registration of its launch
    // declares.
    let chosen = choose("gated MLP", grouped.chain(chained), need)?;
    let prepared: Box<dyn PreparedExperts> = match chosen.entry {
        Entry::Grouped(variant) => {
            let mut prepared = GroupedOp::new(variant.clone(), need.capacity)?;
            prepared.info.guarantees = prepared.info.guarantees.meet(chosen.guarantees);
            need.check(&prepared.info)?;
            Box::new(prepared)
        }
        Entry::Chained(projection) => {
            let table = chained_table(&forms, usage.phase, need)?;
            debug_assert_eq!(table.form_at(need.capacity), projection);
            check_route_tiles(&table, local, need.capacity)?;
            let mut prepared = ChainedOp::new(table, need.capacity, &selection.resources)?;
            prepared.info.guarantees = prepared.info.guarantees.meet(chosen.guarantees);
            need.check(&prepared.info)?;
            Box::new(prepared)
        }
    };
    Ok(prepared)
}

/// What a grouped entry launches: its symbol (and its reduction's), query,
/// layout, schedule and capacity, as the builder plans and packs them. Its
/// profile, MoE class, exact flag and claims do not reach the launch. A
/// template-scheduled entry's code and plans follow its schedule
/// (`moe::template::plan` replaces the prefill weight and activation
/// stages too), so of its engine knobs only the row tile, which its routes
/// are padded to, reaches the op; the others only gate support. Of an
/// engine entry's knobs, the packed engine never reads the stages or
/// swizzle, which only the packed ring reads (the stages also the
/// fragment engines), nor the weight stages, which only the fragment
/// prefill engines read.
fn grouped_launch(variant: &Variant) -> String {
    let Variant {
        profile: _,      // with the class, the rank shape the query must agree with
        class: _,        // with the profile, the rank shape the query must agree with
        layout,          // the code and weight views
        projection,      // selects the op
        phase,           // the code and plans
        tokens_capacity, // the code and plans
        path,            // the code
        query,           // selects the op
        knobs,           // below
        weight_stages,   // an engine prefill entry's plans and code; zero elsewhere
        exact: _,        // ranks entries by the step's use; its launch is its knobs
        route_weight,    // selects the op
        swiglu_clamp,    // selects the op
        engine,          // whether the template serves it, and its schedule kind
        schedule,        // the template's code and plans
        symbol,          // the launched symbol
        invariance: _,   // claims
    } = variant;
    let CompileKnobs {
        tile_m,           // the route tile
        tile_n,           // an engine entry's plans
        tile_k,           // an engine entry's plans; selects the packed ring
        stages,           // a packed ring's or fragment engine's code and plans
        swizzle,          // a packed ring's prefetch
        warps,            // an engine entry's code and plans
        split_k_capacity, // an engine entry's code and plans
    } = *knobs;
    let fragment_prefill = variant.fragment_prefill()
        || variant.fp8_fragment_weights() && *phase == moe::Phase::Prefill;
    let engine_knobs = schedule.is_none().then_some((
        tile_n,
        tile_k,
        (variant.packed_ring() || variant.fragment_prefill() || variant.fp8_fragment_weights())
            .then_some(stages),
        variant.packed_ring().then_some(swizzle),
        warps,
        split_k_capacity,
        fragment_prefill.then_some(*weight_stages),
    ));
    // The replaced engine's knob values are a record; only whether one was
    // replaced, and by which schedule kind, decides the template's code.
    let engine = engine.map(|engine| engine.schedule());
    format!(
        "{symbol} {:?} {layout:?} {projection:?} {phase:?} {tokens_capacity} {path:?} \
         {query:?} {tile_m} {engine_knobs:?} {route_weight:?} {swiglu_clamp:?} {engine:?} \
         {schedule:?}",
        variant.reduction_symbol()
    )
}

/// What a chained entry launches: its symbol, query, row tile and
/// schedule, which its code and plans follow.
fn chained_launch(projection: &chained::Projection) -> String {
    let chained::Projection {
        symbol,             // the launched symbol
        query,              // selects the op
        row_tile,           // the code and plans
        tokens_capacity: _, // bounds the rows the selection settles
        schedule,           // the code
    } = projection;
    format!("{symbol} {query:?} {row_tile} {schedule:?}")
}

/// Whether chained entry `query` computes `op`'s math: equal but for the
/// routed or token rows each holds, which only bound the rows it serves.
fn same_math(query: &GatedMlpProjectionQuery, op: &GatedMlpProjectionQuery) -> bool {
    let math = |query: &GatedMlpProjectionQuery| {
        let mut query = query.logical();
        query.gemm.m_capacity = 0;
        if let Some(grouped) = &mut query.gemm.grouped {
            grouped.capacity = RoutedCapacity::TotalRows(0);
        }
        query
    };
    math(query) == math(op)
}

/// Whether no grouped entry the build can compile computes `op`.
fn buildable_is_empty(grouped: &[Variant], op: &GatedMlpProjectionQuery) -> bool {
    !grouped
        .iter()
        .any(|variant| grouped_contract(variant) == *op && variant.buildable().is_ok())
}

/// The chained form of `forms` (the registered chained projections of an
/// op's math) that runs a step of `rows` rows at `phase`: of the forms that
/// hold the rows, those of the least capacity; of those, the tile-pair form
/// from [`TILE_PAIR_MIN_ROWS`] rows, else at a prefill the decoded-tile
/// form, else the one form of another schedule (columns, block warps or warp
/// streams). A tile-pair form never runs fewer rows, nor a decoded-tile form
/// another phase. Forms left that would launch differently tie.
fn chained_form<'a>(
    forms: &[&'a chained::Projection],
    rows: u32,
    phase: ExecutionPhase,
    need: Need,
) -> Result<&'a chained::Projection, Refused> {
    let pairs = rows >= TILE_PAIR_MIN_ROWS;
    let prefill = phase == ExecutionPhase::Prefill;
    let candidates = forms.iter().filter_map(|&projection| {
        let rank = match projection.schedule {
            Schedule::TilePairs { .. } if pairs => 0,
            Schedule::DecodedTiles if prefill => 1,
            Schedule::TilePairs { .. } | Schedule::DecodedTiles => return None,
            Schedule::Columns | Schedule::BlockWarps | Schedule::WarpStreams => 2,
        };
        Some(
            Candidate::new(
                projection,
                &projection.symbol,
                chained_launch(projection),
                projection.invariance(),
            )
            .capacity(projection.tokens_capacity)
            .rank(rank),
        )
    });
    // A form runs from `rows` up to its capacity at most, so it is asked
    // for one-row bits only that far: a form of fewer rows than the op's
    // ask keeps them at every row count it runs. The table's own claim
    // (`chained_guarantees`) is checked against the whole ask.
    let mut exact = need.exact;
    if let Rows::OneRow { max } = exact.rows {
        exact.rows = Rows::OneRow { max: max.min(rows) };
    }
    let need = Need {
        capacity: rows,
        exact,
    };
    Ok(choose("chained gated MLP", candidates, need)?.entry)
}

/// A chained op's forms and the form each row count runs.
#[derive(Debug)]
struct ChainedTable {
    /// The forms the table runs, in the order of the rows they first run.
    forms: Vec<chained::Projection>,
    /// Each run of row counts and its form: the first row count of the run
    /// and the index of the form in `forms`, ascending, the first at one.
    runs: Vec<(u32, usize)>,
}

impl ChainedTable {
    /// The form a step of `rows` rows runs, `rows` at least one.
    fn form_at(&self, rows: u32) -> &chained::Projection {
        let at = self.runs.partition_point(|&(first, _)| first <= rows) - 1;
        &self.forms[self.runs[at].1]
    }

    /// The most rows each form runs, by index.
    fn most_rows(&self, capacity: u32) -> Vec<u32> {
        let mut most = vec![0; self.forms.len()];
        for (at, &(_, form)) in self.runs.iter().enumerate() {
            let last = self
                .runs
                .get(at + 1)
                .map_or(capacity, |&(first, _)| first - 1);
            most[form] = most[form].max(last);
        }
        most
    }
}

/// The form of `forms` each row count from one to `need.capacity` runs at
/// `phase` ([`chained_form`]). The choice changes only where a capacity of
/// `forms` or [`TILE_PAIR_MIN_ROWS`] is passed, so the table is chosen at the
/// first row count of each run between those bounds.
fn chained_table(
    forms: &[&chained::Projection],
    phase: ExecutionPhase,
    need: Need,
) -> Result<ChainedTable, Refused> {
    let mut starts = forms
        .iter()
        .map(|projection| projection.tokens_capacity.saturating_add(1))
        .chain([1, TILE_PAIR_MIN_ROWS])
        .filter(|&rows| rows <= need.capacity)
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    let mut table = ChainedTable {
        forms: Vec::new(),
        runs: Vec::new(),
    };
    for first in starts {
        let form = chained_form(forms, first, phase, need)?;
        let at = match table.forms.iter().position(|known| known == form) {
            Some(at) => at,
            None => {
                table.forms.push(form.clone());
                table.forms.len() - 1
            }
        };
        if table.runs.last().is_none_or(|&(_, last)| last != at) {
            table.runs.push((first, at));
        }
    }
    Ok(table)
}

/// Refuses a grouped table whose row tiles the model's chained routes
/// (`local`) do not pad each row count up to `capacity` to. A chained route
/// op pads a step of `rows` rows to the route of the least capacity that
/// holds them (the route op's per-row choice in `moe_route`): for every
/// route geometry of the table's experts and top-k, each such route's pad
/// must be the row tile of the form the table runs at those rows. With no
/// such route registered, the table keeps one row tile, which a route op
/// takes as the op's route tile at every row count.
fn check_route_tiles(
    table: &ChainedTable,
    local: &[(LocalRoute, String)],
    capacity: u32,
) -> Result<(), Refused> {
    let Some(grouped) = table.forms[0].query.gemm.grouped else {
        return Ok(());
    };
    let routes = local
        .iter()
        .map(|(route, _)| *route)
        .filter(|route| route.local_experts == grouped.experts && route.top_k == grouped.top_k)
        .collect::<Vec<_>>();
    if routes.is_empty() {
        if table
            .forms
            .iter()
            .any(|form| form.row_tile != table.forms[0].row_tile)
        {
            return Err(Refused::Rule(
                "the chained projection's row tile changes with the rows, and no chained route \
                 pads them per row count"
                    .into(),
            ));
        }
        return Ok(());
    }
    // The pads change only where a route's capacity is passed, the tiles
    // where a run starts.
    let mut bounds = routes
        .iter()
        .map(|route| route.capacity.saturating_add(1))
        .chain(table.runs.iter().map(|&(first, _)| first))
        .filter(|&rows| rows <= capacity)
        .collect::<Vec<_>>();
    bounds.sort_unstable();
    bounds.dedup();
    for rows in bounds {
        let tile = table.form_at(rows).row_tile;
        for experts in routes.iter().map(|route| route.experts) {
            let holding = || {
                routes
                    .iter()
                    .filter(move |route| route.experts == experts && route.capacity >= rows)
            };
            let Some(least) = holding().map(|route| route.capacity).min() else {
                continue;
            };
            if holding().any(|route| route.capacity == least && route.pad != tile) {
                return Err(Refused::Rule(format!(
                    "the chained routes pad {rows} rows to another tile than the chained \
                     projection's {tile}"
                )));
            }
        }
    }
    Ok(())
}

/// The contract a grouped entry computes: the op a model asks the backend
/// for to run that entry, whose routed capacity is the routes without
/// padding.
pub fn grouped_contract(variant: &Variant) -> GatedMlpProjectionQuery {
    entry_contract(variant).logical()
}

/// The contract a grouped entry's query states, with its padded capacity.
fn entry_contract(variant: &Variant) -> GatedMlpProjectionQuery {
    GatedMlpProjectionQuery {
        gemm: variant.query,
        projection: match variant.projection {
            Projection::GateUp => GatedMlpProjection::GateUp,
            Projection::Down => GatedMlpProjection::Down,
        },
        swiglu_clamp: variant.swiglu_clamp.map(|clamp| SwigluClamp {
            gate_max_bits: clamp.gate_max.to_bits(),
            up_limit_bits: clamp.up_limit.to_bits(),
        }),
        route_weight: Some(match variant.route_weight {
            RouteWeight::AfterDown => RouteWeightPoint::AfterDown,
            RouteWeight::BeforeDown => RouteWeightPoint::BeforeDown,
        }),
    }
}

/// The versioned implementation of an entry: its stages and a digest of
/// its whole selection (query, knobs, schedule and layout), which fixes its
/// numerics.
fn implementation(engine: &str, stages: &[(String, u32)], selection: String) -> String {
    let stages: Vec<&str> = stages.iter().map(|(symbol, _)| symbol.as_str()).collect();
    let digest = kernel_build::digest(selection.as_bytes());
    format!(
        "sm120.moe.{engine}.v1[{}]#{}",
        stages.join(","),
        &digest[..16]
    )
}

fn product(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1_u64, |bytes, &factor| bytes.checked_mul(factor))
        .ok_or(Refusal("gated MLP buffer bytes overflow"))
}

fn required(view: Option<View>, role: &'static str) -> Result<View, Refusal> {
    view.ok_or(Refusal(role))
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

/// Checks each role `info` sizes against the bound views.
fn check_views(info: &OpInfo, buffers: &GatedMlpBuffers<View>) -> Result<(), Refusal> {
    for (role, bytes) in &info.bytes {
        let view = match role.as_str() {
            "activation" => Some(buffers.activation),
            "activation_scales" => buffers.activation_scales,
            "activation_global" => buffers.activation_global,
            "live_tokens" => Some(buffers.live_tokens),
            "output" => Some(buffers.output),
            "weights.codes" => Some(buffers.weights.codes),
            "weights.local_scales" => buffers.weights.local_scales,
            "weights.global_scales" => buffers.weights.global_scales,
            "routing.weights" => buffers.routing.map(|r| r.weights),
            "routing.routes.offsets" => buffers.routing.map(|r| r.routes.offsets),
            "routing.routes.sorted_rows" => buffers.routing.map(|r| r.routes.sorted_rows),
            "routing.routes.padded_rows" => buffers.routing.map(|r| r.routes.padded_rows),
            "routing.routes.active_experts" => {
                buffers.routing.and_then(|r| r.routes.active_experts)
            }
            "scratch.partial" => buffers.scratch.partial,
            "scratch.output_codes" => buffers.scratch.output_codes,
            "scratch.output_scales" => buffers.scratch.output_scales,
            _ => return Err(Refusal("gated MLP op sizes an unknown role")),
        };
        required(view, "a gated MLP role the op reads is missing")?.covers(*bytes)?;
    }
    Ok(())
}

/// A grouped expert entry (`moe::Variant`) and its split-K reduction. Its
/// plans are sized by the prepared capacity, as the builder sizes them:
/// live counts reach the kernels through device buffers.
struct GroupedOp {
    variant: Variant,
    capacity: u32,
    plan: LaunchPlan,
    reduction: Option<(String, LaunchPlan)>,
    persistent: bool,
    info: OpInfo,
}

impl GroupedOp {
    fn new(variant: Variant, capacity: u32) -> Result<Self, Refused> {
        if capacity > variant.tokens_capacity {
            return Err(Refused::Rule(
                "the grouped expert entry holds fewer tokens than the capacity".into(),
            ));
        }
        let grouped = variant
            .query
            .grouped
            .ok_or(Refusal("MoE requires grouped geometry"))?;
        let routed = capacity
            .checked_mul(grouped.top_k)
            .ok_or(Refusal("routed rows overflow"))?;
        let counts = variant.counts(capacity, routed);
        let plan = variant.plan(counts)?;
        check_plan(&plan)?;
        let mut symbols = vec![(variant.symbol.clone(), plan.dynamic_shared_bytes)];
        let reduction = match variant.reduction_plan(counts)? {
            Some(reduction) => {
                check_plan(&reduction)?;
                let symbol = variant
                    .reduction_symbol()
                    .ok_or(Refusal("missing grouped reduction entry"))?;
                symbols.push((symbol.clone(), reduction.dynamic_shared_bytes));
                Some((symbol, reduction))
            }
            None => None,
        };
        // Prefill entries that fill every SM with one CTA each.
        let persistent = variant.phase == Phase::Prefill && plan.grid == [moe::SM_COUNT, 1, 1];
        let mut info = OpInfo::new(
            implementation("grouped", &symbols, format!("{variant:?}")),
            variant.invariance,
            symbols,
        );
        info.bytes = grouped_bytes(&variant, routed, plan.workspace_bytes)?;
        info.weight_layouts = vec![("weights".into(), grouped_layout(&variant))];
        Ok(Self {
            variant,
            capacity,
            plan,
            reduction,
            persistent,
            info,
        })
    }
}

/// The layout a grouped entry reads its weights in, as the precision
/// layer names it (`moe-packed-*`, `moe-mxfp4-fragment`, `fp8-fragment`).
fn grouped_layout(variant: &Variant) -> WeightLayout {
    let id = match variant.layout {
        moe::WeightLayout::Packed if variant.query.weight == WeightFormat::Nvfp4 => {
            "moe-packed-Scalar"
        }
        moe::WeightLayout::Packed => "moe-packed-Absent",
        moe::WeightLayout::Mxfp4Fragment => "moe-mxfp4-fragment",
        moe::WeightLayout::Fp8Fragment => "fp8-fragment",
    };
    WeightLayout {
        id: id.into(),
        version: 1,
    }
}

/// Whether `variant`'s code reads the active expert ids after their count:
/// the template's ring decode does (`grouped_ring.cuh`), and so may an entry
/// whose engine schedule is a ring but whose served schedule is unresolved.
/// The packed and MXFP4 fragment engines (`run` in `moe.cuh`, and its split-K
/// reduction) read the count alone; the template's persistent prefill
/// (`grouped_prefill.cuh`) reads neither, so the four-byte count view it is
/// given goes unread. The packed
/// ring schedules (`grouped_packed.cuh`) build only precision-path
/// operations, which `Backend::projection` serves.
fn reads_expert_ids(variant: &Variant) -> bool {
    matches!(
        variant.schedule.or(variant.engine),
        Some(moe::template::ScheduleKnobs::Ring(_))
    )
}

/// The bytes of each role a grouped entry reads or writes at `routed`
/// routed rows (the capacity's) and its K-split workspace.
fn grouped_bytes(
    variant: &Variant,
    routed: u32,
    workspace: u64,
) -> Result<Vec<(String, u64)>, Refusal> {
    let query = variant.query;
    let grouped = query
        .grouped
        .ok_or(Refusal("MoE requires grouped geometry"))?;
    let padded = match grouped.capacity {
        RoutedCapacity::TotalRows(rows) => u64::from(rows),
        RoutedCapacity::PerExpert(rows) => product(&[rows.into(), grouped.experts.into()])?,
    };
    let gate_up = variant.projection == Projection::GateUp;
    let columns = u64::from(if gate_up { query.n / 2 } else { query.n });
    let (k, routed) = (u64::from(query.k), u64::from(routed));
    let tokens = routed / u64::from(grouped.top_k);
    let four_bit = matches!(query.weight, WeightFormat::Mxfp4 | WeightFormat::Nvfp4);
    // BF16 activations are the token rows (gate/up) or the routed rows
    // (down); quantized ones the routed rows gathered at their padded
    // positions.
    let activation = match query.activation {
        ActivationFormat::Bf16 => product(&[if gate_up { tokens } else { routed }, k, 2])?,
        ActivationFormat::Nvfp4 => product(&[padded, k / 2])?,
        _ => product(&[padded, k])?,
    };
    let mut bytes = vec![
        ("activation", activation),
        ("live_tokens", 4),
        ("output", product(&[routed, columns, 2])?),
        (
            "weights.codes",
            product(&[
                grouped.experts.into(),
                query.n.into(),
                if four_bit { k / 2 } else { k },
            ])?,
        ),
        ("routing.weights", product(&[routed, 4])?),
        (
            "routing.routes.offsets",
            product(&[u64::from(grouped.experts) + 1, 4])?,
        ),
        ("routing.routes.sorted_rows", product(&[padded, 4])?),
        ("routing.routes.padded_rows", 4),
        // The active count, then, for code that walks them, the active
        // expert ids.
        (
            "routing.routes.active_experts",
            if reads_expert_ids(variant) {
                product(&[u64::from(grouped.experts) + 1, 4])?
            } else {
                4
            },
        ),
        ("scratch.partial", workspace),
    ];
    // Quantized activations' scales, row-major over the padded rows: bytes
    // per 32 (MXFP8) or 16 (NVFP4) values, F32 per 128 (FP8 groups).
    match query.activation {
        ActivationFormat::Bf16 => {}
        ActivationFormat::Mxfp8 => bytes.push(("activation_scales", product(&[padded, k / 32])?)),
        ActivationFormat::Nvfp4 => {
            bytes.push(("activation_scales", product(&[padded, k / 16])?));
            bytes.push(("activation_global", 4));
        }
        ActivationFormat::Fp8Group128(_) => {
            bytes.push(("activation_scales", product(&[padded, k / 128, 4])?));
        }
    }
    // Weight scales over the experts' N rows: MXFP4 fragment scales (a byte
    // per 32 values), the packed 16-row by 8-block tiles of MX and NV byte
    // scales (`packed_scale_offset`), F32 128x128 FP8 blocks per expert, and
    // NVFP4's one global.
    let rows = product(&[grouped.experts.into(), query.n.into()])?;
    match query.weight {
        WeightFormat::Mxfp4 if variant.layout == moe::WeightLayout::Mxfp4Fragment => {
            bytes.push(("weights.local_scales", product(&[rows, k / 32])?));
        }
        WeightFormat::Mxfp4 | WeightFormat::Mxfp8 | WeightFormat::Nvfp4 => {
            let block = if query.weight == WeightFormat::Nvfp4 {
                16
            } else {
                32
            };
            let tiles = product(&[rows.div_ceil(16), (k / block).div_ceil(8)])?;
            bytes.push(("weights.local_scales", product(&[tiles, 128])?));
            if query.weight == WeightFormat::Nvfp4 {
                bytes.push(("weights.global_scales", 4));
            }
        }
        WeightFormat::Fp8Block128x128(_) => bytes.push((
            "weights.local_scales",
            product(&[
                grouped.experts.into(),
                u64::from(query.n.div_ceil(128)),
                k / 128,
                4,
            ])?,
        )),
        _ => {}
    }
    if query.epilogues.output_quantization.is_some() {
        bytes.push(("scratch.output_codes", product(&[routed, columns])?));
        bytes.push(("scratch.output_scales", product(&[routed, columns / 32])?));
    }
    Ok(bytes
        .into_iter()
        .filter(|&(_, bytes)| bytes > 0)
        .map(|(role, bytes)| (role.to_owned(), bytes))
        .collect())
}

impl PreparedExperts for GroupedOp {
    /// The entry's row block, to which its query's padded capacity pads
    /// each expert (`moe::padded_capacity`).
    fn route_tile(&self) -> u32 {
        u32::from(self.variant.knobs.tile_m)
    }

    /// The entry query's padded capacity, which sizes its sorted routes.
    fn padded_rows(&self) -> u32 {
        match self.variant.query.grouped.map(|grouped| grouped.capacity) {
            Some(RoutedCapacity::TotalRows(rows)) => rows,
            _ => unreachable!("a grouped entry holds its padded routed rows"),
        }
    }
}

impl PreparedOp<GatedMlpBuffers<View>> for GroupedOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatedMlpBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal(
                "grouped expert rows are outside the prepared capacity",
            ));
        }
        check_views(&self.info, buffers)?;
        let query = self.variant.query;
        let route = buffers
            .routing
            .ok_or(Refusal("grouped experts need routing"))?;
        let ProjectionRouting {
            weights: route_weights,
            routes,
        } = route;
        let active_experts = required(
            routes.active_experts,
            "grouped experts read the active experts",
        )?;
        let weights = &buffers.weights;
        let scratch = &buffers.scratch;
        let unused = scratch.unused;
        let read = |view: Option<View>, needed: bool, role| {
            if needed {
                required(view, role)
            } else {
                Ok(unused)
            }
        };
        let scales = GroupedScales::of(
            &query,
            buffers.activation_scales,
            buffers.activation_global,
            weights.local_scales,
            weights.global_scales,
            unused,
        )?;
        let output_quantization = query.epilogues.output_quantization.is_some();
        let handles = [
            buffers.activation,
            scales.activation_bytes,
            scales.activation_f32,
            scales.weight_bytes,
            scales.weight_f32,
            buffers.output,
            route_weights,
            routes.offsets,
            routes.sorted_rows,
            routes.padded_rows,
            buffers.live_tokens,
            active_experts,
            read(
                scratch.output_codes,
                output_quantization,
                "a quantized output needs codes",
            )?,
            read(
                scratch.output_scales,
                output_quantization,
                "a quantized output needs scales",
            )?,
            read(
                scratch.partial,
                self.plan.workspace_bytes > 0,
                "split-K experts need a partial workspace",
            )?,
        ]
        .map(|view| view.handle);
        let map = self.variant.weight_tensor_map(weights.codes.handle)?;
        let mut compute = launch(
            &self.variant.symbol,
            rows,
            self.plan,
            self.variant.arguments(handles, map)?,
        );
        compute.persistent = self.persistent;
        let mut launches = vec![compute];
        if let Some((symbol, plan)) = &self.reduction {
            let arguments = self
                .variant
                .reduction_arguments(handles)?
                .ok_or(Refusal("missing grouped reduction arguments"))?;
            launches.push(launch(symbol, rows, *plan, arguments));
        }
        Ok(launches)
    }
}

/// A chained engine projection (`moe::chained::Projection`), routed or
/// dense, through its schedule table ([`chained_table`]): each launch runs
/// the form its row count takes, whose grid covers the padded rows of that
/// launch's token rows.
struct ChainedOp {
    forms: Vec<chained::Projection>,
    /// The index in `forms` of the form each row count from one to the
    /// capacity runs.
    rows: Vec<u8>,
    capacity: u32,
    info: OpInfo,
}

impl ChainedOp {
    fn new(
        table: ChainedTable,
        capacity: u32,
        resources: &BTreeMap<String, EntryResources>,
    ) -> Result<Self, Refused> {
        let most = table.most_rows(capacity);
        let mut symbols = Vec::new();
        let mut bytes: Vec<(String, u64)> = Vec::new();
        for (projection, &rows) in table.forms.iter().zip(&most) {
            if rows > projection.tokens_capacity {
                return Err(Refused::Rule(
                    "the chained projection holds fewer tokens than the capacity".into(),
                ));
            }
            let plan = projection.plan(rows)?;
            check_plan(&plan)?;
            admit(projection, &plan, resources)?;
            symbols.push((projection.symbol.clone(), plan.dynamic_shared_bytes));
            // Every role holds what the widest launch of any form needs.
            for (role, size) in chained_bytes(projection, rows)? {
                match bytes.iter_mut().find(|(known, _)| *known == role) {
                    Some((_, held)) => *held = (*held).max(size),
                    None => bytes.push((role, size)),
                }
            }
        }
        let mut rows = Vec::with_capacity(capacity as usize);
        for (at, &(first, form)) in table.runs.iter().enumerate() {
            let last = table
                .runs
                .get(at + 1)
                .map_or(capacity, |&(next, _)| next - 1);
            let form = u8::try_from(form)
                .map_err(|_| Refusal("a chained table holds at most 256 forms"))?;
            rows.extend((first..=last).map(|_| form));
        }
        // One form is today's op; a table names its runs too.
        let selection = match table.forms.as_slice() {
            [projection] => format!("{projection:?}"),
            _ => format!("{table:?}"),
        };
        let mut info = OpInfo::new(
            implementation("chained", &symbols, selection),
            chained_guarantees(&table.forms, &rows),
            symbols,
        );
        info.bytes = bytes;
        // Weights keep their checkpoint layouts, per expert and row-major.
        info.weight_layouts = vec![("weights".into(), WeightLayout::row_major())];
        Ok(Self {
            forms: table.forms,
            rows,
            capacity,
            info,
        })
    }

    /// The form a launch of `rows` rows runs.
    fn form(&self, rows: u32) -> Result<&chained::Projection, Refusal> {
        rows.checked_sub(1)
            .and_then(|at| self.rows.get(at as usize))
            .map(|&form| &self.forms[usize::from(form)])
            .ok_or(Refusal(
                "chained projection rows are outside the prepared capacity",
            ))
    }
}

/// What a chained op claims that launches `forms[rows[r - 1]]` at each row
/// count `r` from one: the routes' claims over the one-token column
/// schedule every form's declaration names (`Invariance::routed`, as
/// `chained::Projection::invariance` says every schedule computes the same
/// chains), and any chunk split when every form declares it and the rows
/// claim covers every row count, since each row then has its one-row bits
/// in every launch and no form hands state to a later one.
fn chained_guarantees(forms: &[chained::Projection], rows: &[u8]) -> Invariance {
    let routes = rows
        .iter()
        .zip(1..)
        .map(|(&form, rows)| kernel_core::properties::RowRoute {
            rows,
            invariance: forms[usize::from(form)].invariance(),
            reference: "the chained column schedule for one token",
        })
        .collect::<Vec<_>>();
    let routed = Invariance::routed(&routes);
    let stateless = routes
        .iter()
        .all(|route| route.invariance.chunk == kernel_core::properties::Chunk::Any);
    if stateless && routed.one_row_at(routes.len() as u32) {
        routed.with_chunk(kernel_core::properties::Chunk::Any)
    } else {
        routed
    }
}

/// The bytes of each role a chained projection reads or writes at
/// `capacity` tokens.
fn chained_bytes(
    projection: &chained::Projection,
    capacity: u32,
) -> Result<Vec<(String, u64)>, Refusal> {
    let gemm = projection.query.gemm;
    let gate_up = projection.gate_up();
    let (k, columns, tokens) = (
        u64::from(gemm.k),
        u64::from(projection.columns()),
        u64::from(capacity),
    );
    let code_bytes = |elements: u64| match projection.weights() {
        Weights::Nvfp4 => Ok(elements / 2),
        Weights::Bf16 => product(&[elements, 2]),
        Weights::Fp8Block128 | Weights::Mxfp8 => Ok(elements),
    };
    // Checkpoint scales per expert (one for a dense projection): F32
    // 128x128 FP8 blocks, NVFP4's UE4M3 per 16 values with F32 globals per
    // expert and half (gate and up, or down), MXFP8's UE8M0 per 32.
    let experts = u64::from(gemm.grouped.map_or(1, |g| g.experts));
    let n = u64::from(gemm.n);
    let mut bytes = match projection.weights() {
        Weights::Fp8Block128 => vec![(
            "weights.local_scales",
            product(&[experts, n.div_ceil(128), k / 128, 4])?,
        )],
        Weights::Nvfp4 => vec![
            ("weights.local_scales", product(&[experts, n, k / 16])?),
            (
                "weights.global_scales",
                product(&[experts, if gate_up { 2 } else { 1 }, 4])?,
            ),
        ],
        Weights::Mxfp8 => vec![("weights.local_scales", product(&[experts, n, k / 32])?)],
        Weights::Bf16 => Vec::new(),
    };
    bytes.push(("weights.codes", code_bytes(product(&[experts, n, k])?)?));
    let Some(grouped) = gemm.grouped else {
        bytes.extend([
            ("activation", product(&[tokens, k, 2])?),
            ("live_tokens", 4),
            ("output", product(&[tokens, columns, 2])?),
        ]);
        return Ok(bytes
            .into_iter()
            .map(|(role, bytes)| (role.to_owned(), bytes))
            .collect());
    };
    let routes = product(&[tokens, grouped.top_k.into()])?;
    bytes.extend([
        (
            "activation",
            product(&[if gate_up { tokens } else { routes }, k, 2])?,
        ),
        (
            "output",
            product(&[routes, columns, if gate_up { 2 } else { 4 }])?,
        ),
        (
            "routing.routes.offsets",
            product(&[u64::from(grouped.experts) + 1, 4])?,
        ),
        // The sorted rows the routes hold (`moe::padded_capacity`, as the
        // route op that writes them sizes them): every route and each
        // routed expert's padding. The kernels read the device's padded
        // total of them, a whole number of row tiles within that bound.
        (
            "routing.routes.sorted_rows",
            product(&[
                moe::padded_capacity(
                    capacity * grouped.top_k,
                    grouped.experts,
                    projection.row_tile,
                )?
                .into(),
                4,
            ])?,
        ),
        ("routing.routes.padded_rows", 4),
    ]);
    if !gate_up {
        bytes.push(("routing.weights", product(&[routes, 4])?));
    }
    Ok(bytes
        .into_iter()
        .map(|(role, bytes)| (role.to_owned(), bytes))
        .collect())
}

impl PreparedExperts for ChainedOp {
    /// The row tile of the form the capacity's rows run; fewer rows may run
    /// a form of another tile, which the model's chained routes pad them to
    /// ([`check_route_tiles`]).
    fn route_tile(&self) -> u32 {
        match self.form(self.capacity) {
            Ok(projection) if projection.query.gemm.grouped.is_some() => projection.row_tile,
            _ => 1,
        }
    }

    /// The most padded routes any launch reads: each form's sorted rows at
    /// the most rows it runs (`chained::Projection::padded_rows`).
    fn padded_rows(&self) -> u32 {
        if self.forms[0].query.gemm.grouped.is_none() {
            return 0;
        }
        let most = self.most_rows();
        self.forms
            .iter()
            .zip(most)
            .map(|(projection, rows)| projection.padded_rows(rows))
            .max()
            .unwrap_or(0)
    }
}

impl ChainedOp {
    /// The most rows each form runs, by index.
    fn most_rows(&self) -> Vec<u32> {
        let mut most = vec![0; self.forms.len()];
        for (at, &form) in self.rows.iter().enumerate() {
            most[usize::from(form)] = at as u32 + 1;
        }
        most
    }
}

impl PreparedOp<GatedMlpBuffers<View>> for ChainedOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatedMlpBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let projection = self.form(rows)?;
        check_views(&self.info, buffers)?;
        let weights = &buffers.weights;
        let routing = buffers.routing;
        let grouped = projection.query.gemm.grouped.is_some();
        // The engine's arguments in entry order, the ones its format and
        // mode read; a zero handle passes a null pointer.
        // The format's block scales take the entry's F32 block-scale slot
        // for FP8 and its byte-scale slot otherwise.
        let views = [
            Some(buffers.activation),
            Some(weights.codes),
            weights.local_scales,
            weights.local_scales,
            weights.global_scales,
            routing.map(|r| r.routes.offsets),
            routing.map(|r| r.routes.sorted_rows),
            routing.map(|r| r.routes.padded_rows),
            Some(buffers.live_tokens),
            routing.map(|r| r.weights),
            Some(buffers.output),
        ];
        let format = projection.weights();
        let reads = [
            true,
            true,
            format == Weights::Fp8Block128,
            matches!(format, Weights::Nvfp4 | Weights::Mxfp8),
            format == Weights::Nvfp4,
            grouped,
            grouped,
            grouped,
            !grouped,
            grouped && !projection.gate_up(),
            true,
        ];
        let mut handles = [0; 11];
        for ((handle, view), read) in handles.iter_mut().zip(views).zip(reads) {
            if read {
                *handle = required(view, "a chained projection operand is missing")?.handle;
            }
        }
        Ok(vec![launch(
            &projection.symbol,
            rows,
            projection.plan(rows)?,
            projection.arguments(handles)?,
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::moe::{InputPath, Shape, expert_parallel};
    use kernel_core::backend::Backend;
    use kernel_core::contract::{Argument, ScaleMode};
    use kernel_core::moe::RouteMetadata;
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::buffers::QuantizedWeights;
    use kernel_core::rank_shape::{Block, BlockClass, ExpertSplit, ModelConfig, MoeConfig};

    /// A model of one MoE class of `$experts` experts, top 8, at `$hidden`
    /// with expert intermediate `$intermediate`, split over the ranks by
    /// `$split`.
    macro_rules! model {
        ($name:literal, $tag:literal, $hidden:expr, $experts:expr, $intermediate:expr, $split:expr) => {
            Model::new(
                crate::profile_names!($name, $tag, 4, 2),
                ModelConfig {
                    hidden: $hidden,
                    vocab: 4096,
                    linear: None,
                    ngram_heads: 0,
                    blocks: &[BlockClass {
                        name: "moe",
                        layers: (0, 1),
                        block: Block::Moe(MoeConfig {
                            experts: $experts,
                            top_k: 8,
                            intermediate: $intermediate,
                            split: $split,
                            shared_intermediate: 0,
                            shared_tile: 128,
                        }),
                    }],
                },
            )
        };
    }

    /// Expert-sharded experts: 128 (32 per rank at TP4) at hidden 2048 and
    /// intermediate 512, as FP8 fragment entries take them.
    const SHARDED: Model = model!(
        "synthetic-sharded-experts",
        "synthetic_sharded",
        2048,
        128,
        512,
        ExpertSplit::Experts
    );

    /// Column-split experts: 256 at hidden 4096 with intermediate 2048 (512
    /// per rank at TP4), as MXFP4 fragment entries take them.
    const SPLIT: Model = model!(
        "synthetic-split-experts",
        "synthetic_split",
        4096,
        256,
        2048,
        ExpertSplit::Intermediate
    );

    /// A step that is not a verify step: its entries are the ordinary
    /// ones.
    const DECODE: OperationUse = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: false,
    };

    const VERIFY: OperationUse = OperationUse {
        phase: ExecutionPhase::Verify,
        ..DECODE
    };

    const ACTIVATION: u64 = 1;
    const ACTIVATION_SCALES: u64 = 2;
    const ACTIVATION_GLOBAL: u64 = 3;
    const LIVE: u64 = 4;
    const OUTPUT: u64 = 5;
    const CODES: u64 = 6;
    const COUNTS: u64 = 7;
    const LOCAL_SCALES: u64 = 8;
    const GLOBALS: u64 = 9;
    const ROUTE_WEIGHTS: u64 = 10;
    const OFFSETS: u64 = 11;
    const SORTED: u64 = 12;
    const PADDED: u64 = 13;
    const ACTIVE: u64 = 14;
    const PARTIAL: u64 = 15;
    const OUTPUT_CODES: u64 = 16;
    const OUTPUT_SCALES: u64 = 17;
    const UNUSED: u64 = 18;

    /// Every role bound to its own view of `bytes`.
    fn buffers(bytes: u64) -> GatedMlpBuffers<View> {
        let v = |handle| view(handle, bytes);
        GatedMlpBuffers {
            activation: v(ACTIVATION),
            activation_scales: Some(v(ACTIVATION_SCALES)),
            activation_global: Some(v(ACTIVATION_GLOBAL)),
            live_tokens: v(LIVE),
            output: v(OUTPUT),
            weights: QuantizedWeights {
                codes: v(CODES),
                local_scales: Some(v(LOCAL_SCALES)),
                global_scales: Some(v(GLOBALS)),
            },
            routing: Some(ProjectionRouting {
                weights: v(ROUTE_WEIGHTS),
                routes: RouteMetadata {
                    counts: Some(v(COUNTS)),
                    offsets: v(OFFSETS),
                    sorted_rows: v(SORTED),
                    padded_rows: v(PADDED),
                    active_experts: Some(v(ACTIVE)),
                },
            }),
            scratch: GatedMlpScratch {
                partial: Some(v(PARTIAL)),
                output_codes: Some(v(OUTPUT_CODES)),
                output_scales: Some(v(OUTPUT_SCALES)),
                unused: v(UNUSED),
            },
        }
    }

    /// The fifteen buffers of a grouped entry in the order `run` in
    /// `cuda/moe/moe.cuh` reads them: byte block scales of the activation
    /// and weights at 1 and 3, F32 scales (FP8 blocks and groups, NVFP4
    /// globals) at 2 and 4, `UNUSED` where the format reads none, and the
    /// partial workspace or `UNUSED` (`partial`), which the entry nulls when
    /// it splits no K.
    fn legacy_handles(v: &Variant, partial: u64) -> [u64; 15] {
        let q = v.query;
        let pick = |read: bool, handle| if read { handle } else { UNUSED };
        let (activation_bytes, activation_f32) = match q.activation {
            ActivationFormat::Bf16 => (UNUSED, UNUSED),
            ActivationFormat::Mxfp8 => (ACTIVATION_SCALES, UNUSED),
            ActivationFormat::Nvfp4 => (ACTIVATION_SCALES, ACTIVATION_GLOBAL),
            ActivationFormat::Fp8Group128(_) => (UNUSED, ACTIVATION_SCALES),
        };
        let (weight_bytes, weight_f32) = match q.weight {
            WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => (LOCAL_SCALES, UNUSED),
            WeightFormat::Nvfp4 => (LOCAL_SCALES, GLOBALS),
            WeightFormat::Fp8Block128x128(_) => (UNUSED, LOCAL_SCALES),
            _ => (UNUSED, UNUSED),
        };
        [
            ACTIVATION,
            activation_bytes,
            activation_f32,
            weight_bytes,
            weight_f32,
            OUTPUT,
            ROUTE_WEIGHTS,
            OFFSETS,
            SORTED,
            PADDED,
            LIVE,
            ACTIVE,
            pick(q.epilogues.output_quantization.is_some(), OUTPUT_CODES),
            pick(q.epilogues.output_quantization.is_some(), OUTPUT_SCALES),
            partial,
        ]
    }

    /// The launches a model plans for `v` today at `live` tokens: the entry
    /// and its split-K reduction.
    fn legacy_grouped(v: &Variant, live: u32, handles: [u64; 15]) -> Vec<Launch> {
        let top_k = v.query.grouped.unwrap().top_k;
        let counts = v.counts(live, live * top_k);
        let map = v.weight_tensor_map(CODES).unwrap();
        let plan = v.plan(counts).unwrap();
        let mut launches = vec![Launch {
            persistent: v.phase == Phase::Prefill && plan.grid == [moe::SM_COUNT, 1, 1],
            ..launch(&v.symbol, live, plan, v.arguments(handles, map).unwrap())
        }];
        if let Some(plan) = v.reduction_plan(counts).unwrap() {
            launches.push(launch(
                &v.reduction_symbol().unwrap(),
                live,
                plan,
                v.reduction_arguments(handles).unwrap().unwrap(),
            ));
        }
        launches
    }

    fn live_counts(capacity: u32) -> Vec<u32> {
        if capacity <= 32 {
            (1..=capacity).collect()
        } else {
            vec![1, 17, capacity / 2, capacity - 1, capacity]
        }
    }

    /// Prepares `v` alone at its token capacity and checks its launches
    /// against the builder's at every count.
    fn assert_grouped_equal(v: &Variant) {
        let backend = Sm120::new().with_experts(v.clone());
        let op = grouped_contract(v);
        let prepared = backend
            .gated_mlp_projection(&op, DECODE, Need::rows(v.tokens_capacity))
            .unwrap_or_else(|refused| panic!("{}: {refused}", v.symbol));
        let info = prepared.info();
        assert_eq!(
            info.symbols[0],
            (v.symbol.clone(), v.capacity_shared_bytes().unwrap())
        );
        assert_eq!(info.guarantees, v.invariance);
        let bound = buffers(u64::MAX);
        // A model that splits no K may pass any view as the partials.
        let partials = if info.bytes_of("scratch.partial").is_some() {
            vec![PARTIAL]
        } else {
            vec![PARTIAL, UNUSED]
        };
        for live in live_counts(v.tokens_capacity) {
            let launches = prepared
                .launches(&bound, View::EMPTY, Counts::rows(live))
                .unwrap();
            for &partial in &partials {
                assert_eq!(
                    launches,
                    legacy_grouped(v, live, legacy_handles(v, partial)),
                    "{} at {live}",
                    v.symbol
                );
            }
        }
    }

    fn synthetic_shape() -> Shape {
        Shape::of_class(SHARDED.at(4).rank_shape().unwrap(), "moe").unwrap()
    }

    /// FP8 fragment entries at the decode buckets and prefill, each on its
    /// engine's schedule, as a catalog resolves it.
    fn fp8_fragment_variants() -> Vec<Variant> {
        let mut variants = Vec::new();
        for projection in [Projection::GateUp, Projection::Down] {
            let split = if projection == Projection::GateUp {
                2
            } else {
                1
            };
            for (tokens, tuning) in [
                (1, expert_parallel::Tuning::Decode { warps: 4, split: 1 }),
                (8, expert_parallel::Tuning::Decode { warps: 8, split }),
                (32, expert_parallel::Tuning::Decode { warps: 8, split: 1 }),
                (
                    4096,
                    expert_parallel::Tuning::Prefill {
                        weight_stages: 2,
                        activation_stages: 2,
                    },
                ),
            ] {
                let v = expert_parallel::variant(
                    SHARDED.at(4),
                    ("moe", synthetic_shape()),
                    projection,
                    tokens,
                    tuning,
                    expert_parallel::Numerics {
                        route_weight: RouteWeight::AfterDown,
                        swiglu_clamp: None,
                    },
                    Invariance::one_row(tokens),
                )
                .unwrap();
                variants.push(Variant {
                    schedule: v.engine,
                    ..v
                });
            }
        }
        variants
    }

    /// MXFP4 fragment entries: decode buckets, their exact entries and
    /// prefill, each on its engine's schedule, as a catalog resolves it.
    fn mxfp4_fragment_variants() -> Vec<Variant> {
        let table = moe::default_tuning();
        let profile = SPLIT.at(4);
        let mut variants = Vec::new();
        for projection in [Projection::GateUp, Projection::Down] {
            for (phase, tokens) in moe::DECODE_BUCKETS
                .map(|tokens| (Phase::Decode, tokens))
                .into_iter()
                .chain([(Phase::Prefill, 8192)])
            {
                variants.push(
                    moe::variant(
                        profile,
                        "moe",
                        moe::WeightLayout::Mxfp4Fragment,
                        projection,
                        phase,
                        tokens,
                        WeightFormat::Mxfp4,
                        InputPath::A16Gather,
                        &table,
                    )
                    .unwrap(),
                );
                if phase == Phase::Decode
                    && let Some(exact) =
                        moe::exact_variant(profile, "moe", projection, tokens, &table).unwrap()
                {
                    variants.push(exact);
                }
            }
        }
        variants
            .into_iter()
            .map(|v| Variant {
                schedule: v.engine,
                ..v
            })
            .collect()
    }

    #[test]
    fn grouped_launches_equal_the_builders() {
        // Packed entries: three formats, A16 and A8 paths, both
        // projections (and interleaved gate/up), decode buckets and prefill.
        let packed = moe::own_catalog();
        assert!(packed.iter().any(|v| v.reduction_symbol().is_some()));
        for activation in [
            ActivationFormat::Nvfp4,
            ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
        ] {
            assert!(packed.iter().any(|v| v.query.activation == activation));
        }
        // MXFP4 fragment entries with exact decode buckets; FP8 fragment
        // entries on their engines and template schedules.
        let mxfp4 = mxfp4_fragment_variants();
        assert!(mxfp4.iter().any(|v| v.exact));
        let fp8 = fp8_fragment_variants();
        assert!(
            fp8.iter()
                .any(|v| moe::template::covers(v) && v.schedule.is_some())
        );
        for v in packed.iter().chain(&mxfp4).chain(&fp8) {
            assert_grouped_equal(v);
        }
    }

    #[test]
    fn grouped_ops_report_layouts_bytes_and_persistence() {
        let fragment = mxfp4_fragment_variants();
        let prefill = fragment
            .iter()
            .find(|v| v.phase == Phase::Prefill && v.projection == Projection::GateUp)
            .unwrap();
        let op = Sm120::new()
            .with_experts(prefill.clone())
            .gated_mlp_projection(&grouped_contract(prefill), DECODE, Need::rows(8192))
            .unwrap();
        assert_eq!(
            op.info().layout_of("weights").unwrap().id,
            "moe-mxfp4-fragment"
        );
        assert!(
            op.info()
                .implementation
                .starts_with(&format!("sm120.moe.grouped.v1[{}]#", prefill.symbol))
        );
        let launches = op
            .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(8192))
            .unwrap();
        assert_eq!(launches[0].plan.grid, [moe::SM_COUNT, 1, 1]);
        assert!(launches[0].persistent);
        // Gate/up reads the token rows and writes the routed rows'
        // intermediate.
        let (k, columns) = (u64::from(prefill.query.k), u64::from(prefill.query.n / 2));
        assert_eq!(op.info().bytes_of("activation"), Some(8192 * k * 2));
        assert_eq!(op.info().bytes_of("output"), Some(8192 * 8 * columns * 2));
        // A view short of a role's bytes, a missing operand and a count
        // past the capacity are refused.
        let mut short = buffers(u64::MAX);
        short.output.bytes = 8192 * 8 * columns * 2 - 1;
        assert!(op.launches(&short, View::EMPTY, Counts::rows(1)).is_err());
        let mut missing = buffers(u64::MAX);
        missing.weights.local_scales = None;
        assert!(op.launches(&missing, View::EMPTY, Counts::rows(1)).is_err());
        assert!(
            op.launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(8193))
                .is_err()
        );
        assert!(
            op.launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(0))
                .is_err()
        );
        // NVFP4 weights read their global, FP8 blocks their block scales.
        for v in moe::own_catalog().iter().take(48) {
            let op = Sm120::new()
                .with_experts(v.clone())
                .gated_mlp_projection(&grouped_contract(v), DECODE, Need::rows(v.tokens_capacity))
                .unwrap();
            let mut missing = buffers(u64::MAX);
            match v.query.weight {
                WeightFormat::Nvfp4 => missing.weights.global_scales = None,
                WeightFormat::Fp8Block128x128(_) => missing.weights.local_scales = None,
                _ => continue,
            }
            assert!(
                op.launches(&missing, View::EMPTY, Counts::rows(1)).is_err(),
                "{}",
                v.symbol
            );
        }
    }

    /// The active experts' view in `buffers`.
    fn active(buffers: &mut GatedMlpBuffers<View>) -> &mut View {
        role(buffers, "routing.routes.active_experts")
    }

    /// The view of role `path` in `buffers`.
    fn role<'a>(buffers: &'a mut GatedMlpBuffers<View>, path: &str) -> &'a mut View {
        match path {
            "activation" => &mut buffers.activation,
            "activation_scales" => buffers.activation_scales.as_mut().unwrap(),
            "activation_global" => buffers.activation_global.as_mut().unwrap(),
            "live_tokens" => &mut buffers.live_tokens,
            "output" => &mut buffers.output,
            "weights.codes" => &mut buffers.weights.codes,
            "weights.local_scales" => buffers.weights.local_scales.as_mut().unwrap(),
            "weights.global_scales" => buffers.weights.global_scales.as_mut().unwrap(),
            "routing.weights" => &mut buffers.routing.as_mut().unwrap().weights,
            "routing.routes.offsets" => &mut buffers.routing.as_mut().unwrap().routes.offsets,
            "routing.routes.sorted_rows" => {
                &mut buffers.routing.as_mut().unwrap().routes.sorted_rows
            }
            "routing.routes.padded_rows" => {
                &mut buffers.routing.as_mut().unwrap().routes.padded_rows
            }
            "routing.routes.active_experts" => buffers
                .routing
                .as_mut()
                .unwrap()
                .routes
                .active_experts
                .as_mut()
                .unwrap(),
            "scratch.partial" => buffers.scratch.partial.as_mut().unwrap(),
            "scratch.output_codes" => buffers.scratch.output_codes.as_mut().unwrap(),
            "scratch.output_scales" => buffers.scratch.output_scales.as_mut().unwrap(),
            _ => panic!("unknown role {path}"),
        }
    }

    /// Every role `op` sizes is checked: views of exactly its bytes launch,
    /// and one byte short of any of them is refused. Returns the roles.
    fn assert_sized_roles(op: &dyn PreparedOp<GatedMlpBuffers<View>>) -> Vec<String> {
        let mut exact = buffers(u64::MAX);
        for (path, bytes) in &op.info().bytes {
            role(&mut exact, path).bytes = *bytes;
        }
        assert!(op.launches(&exact, View::EMPTY, Counts::rows(1)).is_ok());
        for (path, bytes) in &op.info().bytes {
            let mut short = exact;
            role(&mut short, path).bytes = bytes - 1;
            assert!(
                op.launches(&short, View::EMPTY, Counts::rows(1)).is_err(),
                "{}: {path}",
                op.info().implementation
            );
        }
        op.info()
            .bytes
            .iter()
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// The scale roles a format reads.
    fn scale_roles(weight: WeightFormat, activation: ActivationFormat) -> Vec<&'static str> {
        let mut roles = match weight {
            WeightFormat::Fp8Block128x128(_) => vec!["weights.local_scales"],
            WeightFormat::Nvfp4 => vec!["weights.local_scales", "weights.global_scales"],
            WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => vec!["weights.local_scales"],
            _ => Vec::new(),
        };
        roles.extend(match activation {
            ActivationFormat::Bf16 => Vec::new(),
            ActivationFormat::Nvfp4 => vec!["activation_scales", "activation_global"],
            _ => vec!["activation_scales"],
        });
        roles
    }

    #[test]
    fn every_scale_view_is_sized_from_its_layout() {
        for v in moe::own_catalog()
            .iter()
            .chain(&mxfp4_fragment_variants())
            .chain(&fp8_fragment_variants())
        {
            let op = Sm120::new()
                .with_experts(v.clone())
                .gated_mlp_projection(&grouped_contract(v), DECODE, Need::rows(v.tokens_capacity))
                .unwrap();
            let roles = assert_sized_roles(op.as_ref());
            for scale in scale_roles(v.query.weight, v.query.activation) {
                assert!(roles.iter().any(|r| r == scale), "{}: {scale}", v.symbol);
            }
            let rows = u64::from(v.query.grouped.unwrap().experts * v.query.n);
            let k = u64::from(v.query.k);
            let expected = match v.layout {
                // A byte per 32 values, as the fragment launchers check.
                moe::WeightLayout::Mxfp4Fragment => Some(("weights.local_scales", rows * k / 32)),
                // F32 per 128x128 block of each expert.
                moe::WeightLayout::Fp8Fragment => {
                    Some(("weights.local_scales", rows / 128 * (k / 128) * 4))
                }
                moe::WeightLayout::Packed => None,
            };
            if let Some((path, bytes)) = expected {
                assert_eq!(op.info().bytes_of(path), Some(bytes), "{}", v.symbol);
            }
        }
        let projections: Vec<_> = chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
            .chain(chained::formats::catalog())
            .collect();
        for projection in &projections {
            let op = Sm120::new()
                .with_chained_projection(projection.clone())
                .gated_mlp_projection(
                    &projection.query,
                    DECODE,
                    Need::rows(projection.tokens_capacity),
                )
                .unwrap();
            let roles = assert_sized_roles(op.as_ref());
            let gemm = projection.query.gemm;
            for scale in scale_roles(gemm.weight, gemm.activation) {
                assert!(
                    roles.iter().any(|r| r == scale),
                    "{}: {scale}",
                    projection.symbol
                );
            }
            // NVFP4 globals: gate and up per expert for gate/up, one per
            // expert for down.
            if projection.weights() == Weights::Nvfp4 {
                let experts = u64::from(gemm.grouped.map_or(1, |g| g.experts));
                let halves = if projection.gate_up() { 2 } else { 1 };
                assert_eq!(
                    op.info().bytes_of("weights.global_scales"),
                    Some(experts * halves * 4)
                );
            }
        }
    }

    #[test]
    fn ops_expose_the_route_tile_their_padding_uses() {
        // Grouped engines: the entry's row block, which their padded
        // capacity pads every expert to.
        for v in moe::own_catalog()
            .iter()
            .chain(&mxfp4_fragment_variants())
            .chain(&fp8_fragment_variants())
        {
            let op = Sm120::new()
                .with_experts(v.clone())
                .gated_mlp_projection(&grouped_contract(v), DECODE, Need::rows(v.tokens_capacity))
                .unwrap();
            let tile = op.route_tile();
            assert_eq!(tile, u32::from(v.knobs.tile_m), "{}", v.symbol);
            let grouped = v.query.grouped.unwrap();
            let padded = moe::padded_capacity(v.query.m_capacity, grouped.experts, tile).unwrap();
            assert_eq!(
                grouped.capacity,
                RoutedCapacity::TotalRows(padded),
                "{}",
                v.symbol
            );
            assert_eq!(op.padded_rows(), padded, "{}", v.symbol);
            if v.phase == Phase::Decode {
                assert_eq!(tile, 16, "{}", v.symbol);
            }
        }
        // The chained engine: its row tile, the pad of the served route at
        // its capacity, over unpadded routes, reading the sorted rows that
        // route holds; one for dense projections.
        let routes = chained::split::routes();
        for projection in chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
        {
            let op = Sm120::new()
                .with_chained_projection(projection.clone())
                .gated_mlp_projection(
                    &projection.query,
                    DECODE,
                    Need::rows(projection.tokens_capacity),
                )
                .unwrap();
            let gemm = projection.query.gemm;
            match gemm.grouped {
                Some(grouped) => {
                    let route = routes
                        .iter()
                        .find(|route| route.capacity == projection.tokens_capacity)
                        .unwrap();
                    assert_eq!(op.route_tile(), route.pad);
                    assert_eq!(op.route_tile(), projection.row_tile);
                    assert_eq!(grouped.capacity, RoutedCapacity::TotalRows(gemm.m_capacity));
                    assert_eq!(
                        op.padded_rows(),
                        projection.padded_rows(projection.tokens_capacity)
                    );
                    // The sorted rows it reads are every route and each routed
                    // expert's padding, within those the route holds (every
                    // expert's padding: at the split's decode capacity 256
                    // routes reach at most 256 of the 288 experts).
                    let reads =
                        moe::padded_capacity(gemm.m_capacity, grouped.experts, projection.row_tile)
                            .unwrap();
                    assert_eq!(
                        op.info().bytes_of("routing.routes.sorted_rows"),
                        Some(u64::from(reads) * 4),
                        "{}",
                        projection.symbol
                    );
                    assert!(
                        reads <= route.sorted_rows_capacity(),
                        "{}",
                        projection.symbol
                    );
                }
                None => assert_eq!((op.route_tile(), op.padded_rows()), (1, 0)),
            }
        }
    }

    #[test]
    fn models_request_routed_rows_without_padding() {
        // Each grouped engine is asked for the unpadded routes, which its
        // entry pads to its own tile; a padded request is refused.
        for v in moe::own_catalog()
            .iter()
            .chain(&mxfp4_fragment_variants())
            .chain(&fp8_fragment_variants())
        {
            let backend = Sm120::new().with_experts(v.clone());
            let logical = grouped_contract(v);
            assert_eq!(
                logical.gemm.grouped.unwrap().capacity,
                RoutedCapacity::TotalRows(v.query.m_capacity)
            );
            let need = Need::rows(v.tokens_capacity);
            assert!(backend.gated_mlp_projection(&logical, DECODE, need).is_ok());
            assert!(matches!(
                backend.gated_mlp_projection(&entry_contract(v), DECODE, need),
                Err(Refused::Rule(_))
            ));
        }
        // No two entries of a catalog that held different padded
        // capacities share a logical request.
        let catalog: Vec<Variant> = moe::own_catalog()
            .into_iter()
            .chain(mxfp4_fragment_variants())
            .chain(fp8_fragment_variants())
            .collect();
        for (i, a) in catalog.iter().enumerate() {
            for b in &catalog[i + 1..] {
                if grouped_contract(a) == grouped_contract(b) {
                    assert_eq!(
                        entry_contract(a),
                        entry_contract(b),
                        "{} {}",
                        a.symbol,
                        b.symbol
                    );
                }
            }
        }
    }

    #[test]
    fn ring_entries_size_the_whole_active_expert_list() {
        let packed = moe::own_catalog();
        // A packed decode entry reads only the active count (`moe.cuh`).
        let count_only_decode = packed
            .iter()
            .find(|v| v.symbol == "chop_moe_std_gateup_decode_fp8block_a16_interleaved_t1_v1")
            .unwrap();
        let op = Sm120::new()
            .with_experts(count_only_decode.clone())
            .gated_mlp_projection(&grouped_contract(count_only_decode), DECODE, Need::rows(1))
            .unwrap();
        assert_eq!(op.info().bytes_of("routing.routes.active_experts"), Some(4));
        let fp8 = fp8_fragment_variants();
        assert!(
            fp8.iter()
                .any(|v| v.phase == Phase::Decode && v.schedule.is_some())
        );
        for v in mxfp4_fragment_variants().iter().chain(&fp8).chain(&packed) {
            let op = Sm120::new()
                .with_experts(v.clone())
                .gated_mlp_projection(&grouped_contract(v), DECODE, Need::rows(v.tokens_capacity))
                .unwrap();
            // The code the entry's cubin compiles: the template's ring decode
            // walks the ids; the packed and fragment engines and the
            // prefill template read the count. An entry with an engine
            // schedule but none resolved builds no cubin; its op asks for
            // the list when the engine's schedule is a ring.
            let walks_ids = match (v.schedule, v.engine) {
                (None, Some(engine)) => {
                    matches!(engine, moe::template::ScheduleKnobs::Ring(_))
                }
                _ => crate::moe::template::implementation(v)
                    .is_some_and(|code| code.header == moe::template::Schedule::Ring.header()),
            };
            assert_eq!(
                walks_ids,
                v.phase == Phase::Decode && v.schedule.or(v.engine).is_some()
            );
            let list = (u64::from(v.query.grouped.unwrap().experts) + 1) * 4;
            let mut count_only = buffers(u64::MAX);
            active(&mut count_only).bytes = 4;
            let mut full = buffers(u64::MAX);
            active(&mut full).bytes = list;
            assert!(
                op.launches(&full, View::EMPTY, Counts::rows(1)).is_ok(),
                "{}",
                v.symbol
            );
            assert_eq!(
                op.launches(&count_only, View::EMPTY, Counts::rows(1))
                    .is_ok(),
                !walks_ids,
                "{}",
                v.symbol
            );
        }
    }

    #[test]
    fn grouped_ops_refuse_unserved_inexact_and_over_capacity_needs() {
        let fragment = fp8_fragment_variants();
        let decode = fragment
            .iter()
            .find(|v| (v.phase, v.tokens_capacity) == (Phase::Decode, 8))
            .unwrap();
        let op = grouped_contract(decode);
        let backend = Sm120::new().with_experts(decode.clone());
        // No entry serves a clamped SwiGLU or route weights before the down
        // projection, and an empty selection serves nothing.
        let mut clamped = op;
        clamped.swiglu_clamp = Some(SwigluClamp::symmetric(10.0));
        assert!(matches!(
            backend.gated_mlp_projection(&clamped, DECODE, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            Sm120::new().gated_mlp_projection(&op, DECODE, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        // A decode bucket holds its tokens; prepared ops at most
        // MAX_CAPACITY rows.
        assert!(
            backend
                .gated_mlp_projection(&op, DECODE, Need::rows(8))
                .is_ok()
        );
        assert!(
            backend
                .gated_mlp_projection(&op, DECODE, Need::rows(4))
                .is_ok()
        );
        for capacity in [0, 9, MAX_CAPACITY + 1] {
            assert!(matches!(
                backend.gated_mlp_projection(&op, DECODE, Need::rows(capacity)),
                Err(Refused::Rule(_))
            ));
        }
        // The entry declares one-row exactness up to its bucket.
        assert!(
            backend
                .gated_mlp_projection(
                    &op,
                    DECODE,
                    Need::rows(8).with_exact(Invariance::one_row(8))
                )
                .is_ok()
        );
        assert!(matches!(
            backend.gated_mlp_projection(
                &op,
                DECODE,
                Need::rows(8).with_exact(Invariance::any_rows(8))
            ),
            Err(Refused::Inexact { .. })
        ));
        // An invalid contract is refused before any entry is considered.
        let mut dense = op;
        dense.gemm.grouped = None;
        assert!(matches!(
            backend.gated_mlp_projection(&dense, DECODE, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
    }

    /// The launch a model plans for a chained projection today at `rows`:
    /// the operands its format and mode read, null elsewhere.
    fn legacy_chained(projection: &chained::Projection, rows: u32) -> Launch {
        let format = projection.weights();
        let grouped = projection.query.gemm.grouped.is_some();
        let some = |read: bool, handle| if read { handle } else { 0 };
        let handles = [
            ACTIVATION,
            CODES,
            some(format == Weights::Fp8Block128, LOCAL_SCALES),
            some(
                matches!(format, Weights::Nvfp4 | Weights::Mxfp8),
                LOCAL_SCALES,
            ),
            some(format == Weights::Nvfp4, GLOBALS),
            some(grouped, OFFSETS),
            some(grouped, SORTED),
            some(grouped, PADDED),
            some(!grouped, LIVE),
            some(grouped && !projection.gate_up(), ROUTE_WEIGHTS),
            OUTPUT,
        ];
        launch(
            &projection.symbol,
            rows,
            projection.plan(rows).unwrap(),
            projection.arguments(handles).unwrap(),
        )
    }

    #[test]
    fn chained_launches_equal_the_builder() {
        // Tile pairs alone serve no op (they never run fewer than
        // TILE_PAIR_MIN_ROWS rows).
        let projections: Vec<_> = chained::catalog()
            .into_iter()
            .chain(chained::block_warp_catalog())
            .chain(chained::split::catalog())
            .chain(
                chained::split::forms()
                    .into_iter()
                    .filter(|p| !matches!(p.schedule, Schedule::TilePairs { .. })),
            )
            .chain(chained::formats::catalog())
            .collect();
        assert!(projections.iter().any(|p| p.query.gemm.grouped.is_none()));
        assert!(projections.iter().any(|p| p.query.gemm.grouped.is_some()));
        for projection in &projections {
            let backend = Sm120::new().with_chained_projection(projection.clone());
            let prepared = backend
                .gated_mlp_projection(
                    &projection.query,
                    DECODE,
                    Need::rows(projection.tokens_capacity),
                )
                .unwrap_or_else(|refused| panic!("{}: {refused}", projection.symbol));
            assert_eq!(
                prepared.info().weight_layouts,
                vec![("weights".to_owned(), WeightLayout::row_major())]
            );
            assert_eq!(
                prepared.info().guarantees,
                Invariance::any_rows(projection.tokens_capacity)
            );
            for rows in live_counts(projection.tokens_capacity) {
                let launches = prepared
                    .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(
                    launches,
                    vec![legacy_chained(projection, rows)],
                    "{} at {rows}",
                    projection.symbol
                );
            }
        }
    }

    #[test]
    fn chained_ops_check_their_views_and_capacity() {
        let projection = chained::split::catalog()
            .into_iter()
            .find(|p| {
                p.query.gemm.grouped.is_some()
                    && !p.gate_up()
                    && p.weights() == Weights::Nvfp4
                    && p.tokens_capacity == 32
            })
            .unwrap();
        let backend = Sm120::new().with_chained_projection(projection.clone());
        let op = backend
            .gated_mlp_projection(&projection.query, DECODE, Need::rows(32))
            .unwrap();
        // The grouped down writes F32 weighted routes.
        let routes = 32 * 8;
        let output = routes * u64::from(projection.query.gemm.n) * 4;
        assert_eq!(op.info().bytes_of("output"), Some(output));
        let mut short = buffers(u64::MAX);
        short.output.bytes = output - 1;
        assert!(op.launches(&short, View::EMPTY, Counts::rows(1)).is_err());
        // A dense projection needs no routing; a grouped one does.
        let mut unrouted = buffers(u64::MAX);
        unrouted.routing = None;
        assert!(
            op.launches(&unrouted, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let mut missing = buffers(u64::MAX);
        missing.weights.global_scales = None;
        assert!(op.launches(&missing, View::EMPTY, Counts::rows(1)).is_err());
        // Null operands the format does not read.
        let launch = &op
            .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(1))
            .unwrap()[0];
        assert_eq!(launch.arguments.0[2], Argument::NullDevicePointer);
        assert_eq!(launch.arguments.0[8], Argument::NullDevicePointer);
        assert!(
            op.launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(33))
                .is_err()
        );
        assert!(matches!(
            backend.gated_mlp_projection(&projection.query, DECODE, Need::rows(33)),
            Err(Refused::Rule(_))
        ));
        // Its declared one-row bits are granted; a tie rule it does not
        // declare is refused.
        assert!(
            backend
                .gated_mlp_projection(
                    &projection.query,
                    DECODE,
                    Need::rows(32).with_exact(Invariance::one_row(32))
                )
                .is_ok()
        );
        assert!(matches!(
            backend.gated_mlp_projection(
                &projection.query,
                DECODE,
                Need::rows(32).with_exact(
                    Invariance::one_row(32).with_ties(kernel_core::properties::Ties::LowerIndex),
                )
            ),
            Err(Refused::Inexact { .. })
        ));
        let dense = chained::catalog()
            .into_iter()
            .find(|p| {
                p.query.gemm.grouped.is_none()
                    && p.weights() == Weights::Fp8Block128
                    && p.query.gemm.weight == WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
            })
            .unwrap();
        let op = Sm120::new()
            .with_chained_projection(dense.clone())
            .gated_mlp_projection(&dense.query, DECODE, Need::rows(dense.tokens_capacity))
            .unwrap();
        assert!(op.launches(&unrouted, View::EMPTY, Counts::rows(1)).is_ok());
    }

    /// The chained engine's served schedules of every cataloged contract:
    /// the decode forms in block warps (routed FP8 in warp streams, MXFP8
    /// in columns), the prefill forms in columns, the routed FP8 prefill
    /// gate/up also in tile pairs (two row tiles before pairing) and the
    /// dense FP8 prefill gate/up also in decoded tiles; and the decode and
    /// prefill routes their row tiles pad to.
    fn served_schedules() -> Sm120 {
        let mut backend = Sm120::new();
        for projection in chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
            .chain(chained::formats::catalog())
        {
            let fp8 = projection.weights() == Weights::Fp8Block128;
            let decode = if fp8 {
                projection
                    .warp_streams()
                    .or_else(|_| projection.block_warps())
            } else {
                projection.block_warps()
            };
            backend = backend.with_chained_projection(decode.unwrap_or(projection.clone()));
            if fp8 && projection.row_tile == 64 && projection.gate_up() {
                let other = match projection.query.gemm.grouped {
                    Some(_) => projection.tile_pairs(2),
                    None => projection.decoded_tiles(),
                };
                backend = backend.with_chained_projection(other.unwrap());
            }
        }
        for route in chained::split::routes() {
            backend = backend.with_local_route(route, chained::split::PREFIX);
        }
        backend.with_entry_resources(served_resources())
    }

    /// The resources ptxas reports for the tile-pair forms of the split's
    /// catalog and the decoded-tile forms of the chained catalog (kernel-lab
    /// `moe_chained_split_tile_pairs` and `moe_chained` locks): the
    /// tile-pair gate/up 252 registers and 47,636 static shared bytes, the
    /// down 159 and 29,204, a decoded tile 168 and its bounded static bytes.
    fn served_resources() -> Vec<(String, EntryResources)> {
        let locked = |registers, shared_static_bytes| EntryResources {
            registers,
            shared_static_bytes,
        };
        chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
            .filter_map(|p| match (p.tile_pairs(2), p.decoded_tiles()) {
                (Ok(pairs), _) if pairs.gate_up() => Some((pairs.symbol, locked(252, 47_636))),
                (Ok(pairs), _) => Some((pairs.symbol, locked(159, 29_204))),
                (_, Ok(decoded)) => {
                    let bound = chained::decoded_tile_static_bytes(decoded.query.gemm.k);
                    Some((decoded.symbol, locked(168, bound)))
                }
                _ => None,
            })
            .collect()
    }

    /// A table that runs a tile-pair or decoded-tile form is admitted on the
    /// form's locked resources when it is prepared: without them, or when
    /// they keep no CTA resident (or, for decoded tiles, differ from the
    /// static bytes the engine bounds), the op is refused. A table that
    /// never runs the form needs none.
    #[test]
    fn chained_forms_are_admitted_on_their_locked_resources() {
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let columns = chained::split::catalog()
            .into_iter()
            .find(|p| p.tile_pairs(2).is_ok() && p.gate_up())
            .unwrap();
        let pairs = columns.tile_pairs(2).unwrap();
        let dense = chained::catalog()
            .into_iter()
            .find(|p| p.decoded_tiles().is_ok())
            .unwrap();
        let decoded = dense.decoded_tiles().unwrap();
        let bound = chained::decoded_tile_static_bytes(decoded.query.gemm.k);
        assert_eq!(bound, 1408);
        let prepare = |form: &chained::Projection,
                       other: &chained::Projection,
                       resources: Option<EntryResources>,
                       rows: u32| {
            let backend = Sm120::new()
                .with_chained_projection(form.clone())
                .with_chained_projection(other.clone())
                .with_entry_resources(resources.map(|r| (form.symbol.clone(), r)));
            backend
                .gated_mlp_projection(&form.query, usage, Need::rows(rows))
                .map(|op| op.info().symbols.len())
        };
        let locked = |registers, shared_static_bytes| {
            Some(EntryResources {
                registers,
                shared_static_bytes,
            })
        };
        // Tile pairs: 252 registers over 256 threads keep exactly one CTA.
        assert_eq!(resident_ctas(locked(252, 47_636).unwrap(), 256, 0), 1);
        assert_eq!(prepare(&pairs, &columns, locked(252, 47_636), 8192), Ok(2));
        assert!(matches!(
            prepare(&pairs, &columns, None, 8192),
            Err(Refused::Rule(rule)) if rule.contains("no resources")
        ));
        assert!(matches!(
            prepare(&pairs, &columns, locked(252, 101_400), 8192),
            Err(Refused::Rule(rule)) if rule.contains("resident")
        ));
        // Below TILE_PAIR_MIN_ROWS the table never runs them.
        assert_eq!(
            prepare(&pairs, &columns, None, TILE_PAIR_MIN_ROWS - 1),
            Ok(1)
        );
        // Decoded tiles: the bounded static bytes, with the stages, fit.
        assert_eq!(prepare(&decoded, &dense, locked(168, bound), 8192), Ok(1));
        // 255 registers over 384 threads take more than the register file.
        for wrong in [
            None,
            locked(168, bound + 128),
            locked(168, bound - 128),
            locked(255, bound),
        ] {
            assert!(prepare(&decoded, &dense, wrong, 8192).is_err(), "{wrong:?}");
        }
    }

    /// A lock's resources by symbol, the most of each where a symbol is
    /// built in two cubins.
    #[test]
    fn locked_resources_read_each_records_entries() {
        let lock = r#"[
            {"name": "a", "resources": [
                {"symbol": "s", "registers": 40, "shared_static_bytes": 100, "local_bytes": 0},
                {"symbol": "t", "registers": 8, "shared_static_bytes": 0, "local_bytes": 0}
            ]},
            {"name": "b", "resources": [
                {"symbol": "s", "registers": 32, "shared_static_bytes": 200, "local_bytes": 0}
            ]}
        ]"#;
        let resources = locked_resources(lock).unwrap();
        assert_eq!(
            resources["s"],
            EntryResources {
                registers: 40,
                shared_static_bytes: 200
            }
        );
        assert_eq!(resources["t"].registers, 8);
        assert_eq!(resources.len(), 2);
        assert!(locked_resources("{}").is_err());
        assert!(locked_resources(r#"[{"resources": [{"symbol": "s"}]}]"#).is_err());
    }

    /// The registered form of `query`'s math under `schedule` at
    /// `row_tile`.
    fn form(
        query: &GatedMlpProjectionQuery,
        schedule: Schedule,
        row_tile: u32,
    ) -> chained::Projection {
        chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
            .chain(chained::formats::catalog())
            .filter(|p| same_math(&p.query, query) && p.row_tile == row_tile)
            .find_map(|p| match schedule {
                Schedule::Columns => Some(p),
                Schedule::BlockWarps => p.block_warps().ok(),
                Schedule::WarpStreams => p.warp_streams().ok(),
                Schedule::TilePairs { min_pair_tiles } => p.tile_pairs(min_pair_tiles).ok(),
                Schedule::DecodedTiles => p.decoded_tiles().ok(),
            })
            .unwrap_or_else(|| panic!("no {schedule:?} form at row tile {row_tile}"))
    }

    /// Each row count of a chained op runs the form the stated table names:
    /// up to the decode route's 32 tokens the 16-row form (block warps; the
    /// split's FP8 gate/up warp streams; MXFP8 columns), above them the
    /// 64-row form,
    /// which for the routed FP8 gate/up is tile pairs from 2,048 rows and
    /// for the dense FP8 gate/up decoded tiles at every prefill row count.
    /// Each launch equals the form's own at those rows, and the op's route
    /// tile is the capacity's form's.
    #[test]
    fn chained_tables_run_the_stated_form_per_row_count() {
        let backend = served_schedules();
        let prefill = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let verify = OperationUse {
            phase: ExecutionPhase::Verify,
            ..prefill
        };
        let tp2 = Schedule::TilePairs { min_pair_tiles: 2 };
        let ops = chained::catalog()
            .into_iter()
            .chain(chained::split::catalog())
            .chain(chained::formats::catalog())
            .filter(|p| p.tokens_capacity == 8192)
            .collect::<Vec<_>>();
        let mut stated = 0;
        for op in &ops {
            let fp8 = op.weights() == Weights::Fp8Block128;
            let grouped = op.query.gemm.grouped.is_some();
            // Warp streams refuse the split's down (K 512, one round).
            let small = match op.weights() {
                Weights::Mxfp8 => Schedule::Columns,
                Weights::Fp8Block128 if grouped && op.gate_up() => Schedule::WarpStreams,
                _ => Schedule::BlockWarps,
            };
            for usage in [prefill, verify] {
                // The stated runs: first row count, schedule and row tile.
                let mut runs = vec![(1, small, 16), (33, Schedule::Columns, 64)];
                if fp8 && op.gate_up() && grouped {
                    runs.push((TILE_PAIR_MIN_ROWS, tp2, 64));
                }
                if fp8 && op.gate_up() && !grouped && usage.phase == ExecutionPhase::Prefill {
                    runs[1].1 = Schedule::DecodedTiles;
                }
                let prepared = backend
                    .gated_mlp_projection(&op.query, usage, Need::rows(8192))
                    .unwrap_or_else(|refused| panic!("{}: {refused}", op.symbol));
                assert_eq!(prepared.route_tile(), if grouped { 64 } else { 1 });
                // Asking the claims the table keeps runs the same table: a
                // form of fewer rows than the ask still runs its rows.
                let asked = backend
                    .gated_mlp_projection(
                        &op.query,
                        usage,
                        Need::rows(8192).with_exact(Invariance::any_rows(8192)),
                    )
                    .unwrap_or_else(|refused| panic!("{}: {refused}", op.symbol));
                assert_eq!(asked.info(), prepared.info(), "{}", op.symbol);
                for rows in [1, 16, 32, 33, 2048, 8192] {
                    let at = |op: &dyn PreparedExperts| {
                        op.launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    };
                    assert_eq!(
                        at(asked.as_ref()),
                        at(prepared.as_ref()),
                        "{} at {rows}",
                        op.symbol
                    );
                }
                let mut listed = prepared
                    .info()
                    .symbols
                    .iter()
                    .map(|(symbol, _)| symbol.clone())
                    .collect::<Vec<_>>();
                listed.sort();
                let mut forms = runs
                    .iter()
                    .map(|&(_, schedule, tile)| form(&op.query, schedule, tile).symbol)
                    .collect::<Vec<_>>();
                forms.sort();
                assert_eq!(listed, forms, "{}", op.symbol);
                // Every form declares the one-token column schedule's
                // chains, so the table keeps them at every row count.
                assert_eq!(
                    prepared.info().guarantees,
                    Invariance::any_rows(8192),
                    "{}",
                    op.symbol
                );
                for rows in 1..=8192 {
                    let &(_, schedule, tile) = runs
                        .iter()
                        .rev()
                        .find(|&&(first, ..)| first <= rows)
                        .unwrap();
                    let expected = form(&op.query, schedule, tile);
                    let launches = prepared
                        .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    if [1, 32, 33, 2047, 2048, 8192].contains(&rows) {
                        assert_eq!(
                            launches,
                            vec![legacy_chained(&expected, rows)],
                            "{} at {rows}",
                            op.symbol
                        );
                    } else {
                        assert_eq!(launches[0].symbol, expected.symbol, "{rows}");
                    }
                }
                stated += 1;
            }
            // A decode bucket runs the 16-row form at its tile.
            let decode = OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: 8,
                graph: true,
            };
            let prepared = backend
                .gated_mlp_projection(&op.query, decode, Need::rows(8))
                .unwrap();
            assert_eq!(prepared.route_tile(), if grouped { 16 } else { 1 });
            assert_eq!(
                prepared
                    .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(8))
                    .unwrap(),
                vec![legacy_chained(&form(&op.query, small, 16), 8)]
            );
        }
        assert_eq!(stated, 2 * ops.len());
        assert!(ops.len() >= 12);
    }

    /// The forms of a table that changes its row tile must be the tiles the
    /// chained routes pad each row count to; with no chained route, a table
    /// keeps one tile. A table of one form needs no route.
    #[test]
    fn chained_tables_refuse_tiles_their_routes_do_not_pad() {
        let [decode, prefill] = chained::split::routes();
        let routed = chained::split::catalog()
            .into_iter()
            .filter(|p| {
                p.query.gemm.grouped.is_some() && p.weights() == Weights::Mxfp8 && p.gate_up()
            })
            .collect::<Vec<_>>();
        let wide = routed.iter().find(|p| p.row_tile == 64).unwrap();
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let with = |routes: &[LocalRoute]| {
            let backend = routed
                .iter()
                .cloned()
                .fold(Sm120::new(), Sm120::with_chained_projection);
            routes
                .iter()
                .fold(backend, |backend, &route| {
                    backend.with_local_route(route, chained::split::PREFIX)
                })
                .gated_mlp_projection(&wide.query, usage, Need::rows(8192))
                .map(|op| op.route_tile())
        };
        assert_eq!(with(&[decode, prefill]), Ok(64));
        for routes in [&[][..], &[prefill]] {
            assert!(
                matches!(with(routes), Err(Refused::Rule(rule)) if rule.contains("tile")),
                "{routes:?}"
            );
        }
        // Rows no route holds constrain nothing here: the route op refuses
        // a capacity past its routes'.
        assert_eq!(with(&[decode]), Ok(64));
        // Routes of another expert count are another model's.
        let other = LocalRoute {
            experts: 576,
            pad: 64,
            ..decode
        };
        assert!(with(&[decode, prefill, other]).is_err());
        // One form alone keeps its tile without routes.
        let alone = Sm120::new()
            .with_chained_projection(wide.clone())
            .gated_mlp_projection(&wide.query, usage, Need::rows(8192))
            .unwrap();
        assert_eq!(alone.route_tile(), 64);
        assert_eq!(
            alone
                .launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(1))
                .unwrap(),
            vec![legacy_chained(wide, 1)]
        );
        // A tile-pair form never runs fewer than TILE_PAIR_MIN_ROWS rows,
        // nor a decoded-tile form outside a prefill: alone, a tile-pair form
        // serves no op, since an op runs every row count to its capacity.
        let columns = chained::split::catalog()
            .into_iter()
            .find(|p| p.tile_pairs(2).is_ok())
            .unwrap();
        let pairs = columns.tile_pairs(2).unwrap();
        let alone = Sm120::new().with_chained_projection(pairs.clone());
        for rows in [TILE_PAIR_MIN_ROWS - 1, TILE_PAIR_MIN_ROWS] {
            assert!(
                alone
                    .gated_mlp_projection(&pairs.query, usage, Need::rows(rows))
                    .is_err()
            );
        }
        let op = Sm120::new()
            .with_chained_projection(pairs.clone())
            .with_chained_projection(columns.clone())
            .with_entry_resources(served_resources())
            .gated_mlp_projection(&pairs.query, usage, Need::rows(TILE_PAIR_MIN_ROWS))
            .unwrap();
        for (rows, form) in [
            (TILE_PAIR_MIN_ROWS - 1, &columns),
            (TILE_PAIR_MIN_ROWS, &pairs),
        ] {
            assert_eq!(
                op.launches(&buffers(u64::MAX), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![legacy_chained(form, rows)]
            );
        }
        let decoded = chained::catalog()
            .into_iter()
            .find_map(|p| p.decoded_tiles().ok())
            .unwrap();
        let backend = Sm120::new()
            .with_chained_projection(decoded.clone())
            .with_entry_resources(served_resources());
        assert!(
            backend
                .gated_mlp_projection(&decoded.query, usage, Need::rows(64))
                .is_ok()
        );
        let verify = OperationUse {
            phase: ExecutionPhase::Verify,
            ..usage
        };
        assert!(
            backend
                .gated_mlp_projection(&decoded.query, verify, Need::rows(64))
                .is_err()
        );
    }

    /// A registration of a permutation case.
    #[derive(Clone)]
    enum Registered {
        Grouped(Box<Variant>),
        Chained(chained::Projection),
    }

    /// The symbols `op` prepares at `usage` and `need` over every order of
    /// `registered`, or the refusal, the same in every order.
    fn in_every_order(
        registered: &[Registered],
        op: &GatedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Vec<String>, Refused> {
        let results = orders(registered)
            .iter()
            .map(|order| {
                let backend = order
                    .iter()
                    .cloned()
                    .fold(Sm120::new(), |backend, registered| match registered {
                        Registered::Grouped(v) => backend.with_experts(*v),
                        Registered::Chained(p) => backend.with_chained_projection(p),
                    });
                backend
                    .gated_mlp_projection(op, usage, need)
                    .map(|prepared| {
                        let symbols = &prepared.info().symbols;
                        symbols.iter().map(|(symbol, _)| symbol.clone()).collect()
                    })
            })
            .collect::<Vec<_>>();
        assert!(results.iter().all(|result| *result == results[0]));
        results[0].clone()
    }

    fn tied(result: Result<Vec<String>, Refused>, mut symbols: [&str; 2]) -> bool {
        symbols.sort_unstable();
        matches!(result, Err(Refused::Rule(rule)) if rule.contains(&symbols.join(" and ")))
    }

    #[test]
    fn entries_prepare_alike_in_every_order() {
        let variants = mxfp4_fragment_variants();
        let pairs = variants
            .iter()
            .filter(|v| v.exact)
            .map(|exact| {
                let ordinary = variants
                    .iter()
                    .find(|v| !v.exact && grouped_contract(v) == grouped_contract(exact))
                    .unwrap();
                (ordinary.clone(), exact.clone())
            })
            .collect::<Vec<_>>();
        assert!(!pairs.is_empty());
        for (ordinary, exact) in pairs {
            let op = grouped_contract(&exact);
            let tokens = exact.tokens_capacity;
            let need = Need::rows(tokens);
            let registered = [
                Registered::Grouped(Box::new(ordinary.clone())),
                Registered::Grouped(Box::new(exact.clone())),
            ];
            let symbols = |v: &Variant| {
                let mut symbols = vec![v.symbol.clone()];
                symbols.extend(v.reduction_symbol());
                Ok(symbols)
            };
            // Verify steps run the exact entry, other steps the ordinary
            // one, unless they need the one-token order it alone keeps.
            assert_eq!(
                in_every_order(&registered, &op, DECODE, need),
                symbols(&ordinary)
            );
            assert_eq!(
                in_every_order(&registered, &op, VERIFY, need),
                symbols(&exact)
            );
            if !ordinary.invariance.provides(exact.invariance) {
                let exact_need = need.with_exact(exact.invariance);
                assert_eq!(
                    in_every_order(&registered, &op, DECODE, exact_need),
                    symbols(&exact)
                );
            }
            // A verify step of a bucket without an exact entry runs the
            // ordinary one.
            let alone = [Registered::Grouped(Box::new(ordinary.clone()))];
            assert_eq!(
                in_every_order(&alone, &op, VERIFY, need),
                symbols(&ordinary)
            );
            // A second exact entry under another symbol ties for verify
            // steps only; a duplicate is one entry.
            let mut twin = exact.clone();
            twin.symbol = format!("{}_twin", exact.symbol);
            let registered = [
                Registered::Grouped(Box::new(ordinary.clone())),
                Registered::Grouped(Box::new(exact.clone())),
                Registered::Grouped(Box::new(twin.clone())),
            ];
            assert_eq!(
                in_every_order(&registered, &op, DECODE, need),
                symbols(&ordinary)
            );
            assert!(tied(
                in_every_order(&registered, &op, VERIFY, need),
                [&exact.symbol, &twin.symbol]
            ));
            let registered = [
                Registered::Grouped(Box::new(exact.clone())),
                Registered::Grouped(Box::new(ordinary)),
                Registered::Grouped(Box::new(exact.clone())),
            ];
            assert_eq!(
                in_every_order(&registered, &op, VERIFY, need),
                symbols(&exact)
            );
        }
        // Chained projections of one contract under two symbols tie.
        let projection = chained::catalog().into_iter().next().unwrap();
        let mut twin = projection.clone();
        twin.symbol = format!("{}_twin", projection.symbol);
        let need = Need::rows(projection.tokens_capacity);
        for usage in [DECODE, VERIFY] {
            let registered = [
                Registered::Chained(projection.clone()),
                Registered::Chained(projection.clone()),
            ];
            assert_eq!(
                in_every_order(&registered, &projection.query, usage, need),
                Ok(vec![projection.symbol.clone()])
            );
            let registered = [
                Registered::Chained(projection.clone()),
                Registered::Chained(twin.clone()),
            ];
            assert!(tied(
                in_every_order(&registered, &projection.query, usage, need),
                [&projection.symbol, &twin.symbol]
            ));
        }
    }

    /// Every field of a grouped entry, its knobs field by field.
    fn grouped_names() -> Vec<&'static str> {
        use crate::backend::select::field_names;
        let knobs = field_names!(kernel_core::contract::CompileKnobs {
            tile_m,
            tile_n,
            tile_k,
            stages,
            swizzle,
            warps,
            split_k_capacity
        });
        field_names!(Variant {
            profile,
            class,
            layout,
            projection,
            phase,
            tokens_capacity,
            path,
            query,
            knobs,
            weight_stages,
            exact,
            route_weight,
            swiglu_clamp,
            engine,
            schedule,
            symbol,
            invariance
        })
        .into_iter()
        .flat_map(|name| {
            if name == "knobs" {
                knobs
                    .iter()
                    .map(|knob| &*Box::leak(format!("knobs.{knob}").into_boxed_str()))
                    .collect()
            } else {
                vec![name]
            }
        })
        .collect()
    }

    /// A change of the grouped entry field `name` alone.
    fn grouped_change(name: &str) -> Box<dyn Fn(&mut Variant)> {
        use crate::moe::template::ScheduleKnobs;
        // A schedule with one ring or prefill knob changed, or one an
        // engine entry has none of.
        fn toggled(knobs: Option<ScheduleKnobs>) -> Option<ScheduleKnobs> {
            let Some(knobs) = knobs else {
                return Some(ScheduleKnobs::Prefill(crate::moe::template::PrefillKnobs {
                    weight_stages: 2,
                    activation_stages: 2,
                    trim_gather: false,
                    bulk_weights: false,
                }));
            };
            Some(match knobs {
                ScheduleKnobs::Ring(mut ring) => {
                    ring.early_row = !ring.early_row;
                    ScheduleKnobs::Ring(ring)
                }
                ScheduleKnobs::Prefill(mut prefill) => {
                    prefill.trim_gather = !prefill.trim_gather;
                    ScheduleKnobs::Prefill(prefill)
                }
            })
        }
        let other = crate::Profile::synthetic(crate::ProfileNames {
            name: "experts-other",
            tag: "experts_other",
        });
        match name {
            "profile" => Box::new(move |v| v.profile = other),
            "class" => Box::new(|v| v.class = "other"),
            "layout" => Box::new(|v| {
                v.layout = if v.layout == moe::WeightLayout::Packed {
                    moe::WeightLayout::Mxfp4Fragment
                } else {
                    moe::WeightLayout::Packed
                }
            }),
            "projection" => Box::new(|v| {
                v.projection = if v.projection == Projection::GateUp {
                    Projection::Down
                } else {
                    Projection::GateUp
                }
            }),
            "phase" => Box::new(|v| {
                v.phase = if v.phase == Phase::Decode {
                    Phase::Prefill
                } else {
                    Phase::Decode
                }
            }),
            "tokens_capacity" => Box::new(|v| v.tokens_capacity *= 2),
            "path" => Box::new(|v| v.path = moe::InputPath::A8Pregather),
            "query" => Box::new(|v| v.query.n *= 2),
            "knobs.tile_m" => Box::new(|v| v.knobs.tile_m *= 2),
            "knobs.tile_n" => Box::new(|v| v.knobs.tile_n *= 2),
            "knobs.tile_k" => Box::new(|v| v.knobs.tile_k *= 2),
            "knobs.stages" => Box::new(|v| v.knobs.stages += 1),
            "knobs.swizzle" => Box::new(|v| v.knobs.swizzle += 1),
            "knobs.warps" => Box::new(|v| v.knobs.warps *= 2),
            "knobs.split_k_capacity" => Box::new(|v| v.knobs.split_k_capacity *= 2),
            "weight_stages" => Box::new(|v| v.weight_stages += 1),
            "exact" => Box::new(|v| v.exact = !v.exact),
            "route_weight" => Box::new(|v| v.route_weight = moe::RouteWeight::BeforeDown),
            "swiglu_clamp" => Box::new(|v| {
                v.swiglu_clamp = Some(moe::SwigluClamp {
                    gate_max: 7.0,
                    up_limit: 7.0,
                })
            }),
            "engine" => Box::new(|v| v.engine = toggled(v.engine)),
            "schedule" => Box::new(|v| v.schedule = toggled(v.schedule)),
            "symbol" => Box::new(|v| v.symbol = format!("{}_twin_v1", v.symbol)),
            "invariance" => Box::new(|v| {
                v.invariance = if v.invariance == Invariance::NONE {
                    Invariance::one_row(1)
                } else {
                    Invariance::NONE
                }
            }),
            other => panic!("no change of {other}"),
        }
    }

    /// Checks the grouped key of `base`, each field of `classes` changed
    /// alone (unlisted ones `Inert`): the changed entry, prepared alone for
    /// `base`'s op at its capacity, launches as its class says, with the
    /// route tile and buffer bytes of the prepared op.
    fn grouped_keys_follow_launches(
        base: &Variant,
        classes: &[(&'static str, crate::backend::select::Class)],
    ) {
        use crate::backend::select::{Class, keys_follow_launches};
        let fields = grouped_names()
            .into_iter()
            .map(|name| {
                let class = classes
                    .iter()
                    .find(|(listed, _)| *listed == name)
                    .map_or(Class::Inert, |&(_, class)| class);
                (name, class, grouped_change(name))
            })
            .collect();
        let bound = buffers(u64::MAX);
        let op = grouped_contract(base);
        keys_follow_launches(base, grouped_names(), fields, grouped_launch, |v| {
            let prepared = Sm120::new().with_experts(v.clone()).gated_mlp_projection(
                &op,
                DECODE,
                Need::rows(base.tokens_capacity),
            )?;
            let launches = live_counts(base.tokens_capacity)
                .into_iter()
                .map(|live| {
                    prepared
                        .launches(&bound, View::EMPTY, Counts::rows(live))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let bytes = prepared.info().bytes.clone();
            Ok((
                moe::device_code(v),
                (prepared.route_tile(), bytes, launches),
            ))
        });
    }

    /// The grouped and chained keys hold what their entries launch, every
    /// field changed alone: the code (template instance or packed body),
    /// plans, route tile and symbol; never the profile, class, exact flag
    /// (which only ranks) or claims. The contract selects the op. Grouped
    /// entries on a decode ring schedule and on a prefill schedule each
    /// replaced an engine; a field with no other value that fits the entry
    /// serves no op when changed.
    #[test]
    fn launch_keys_hold_what_expert_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let variants = fp8_fragment_variants();
        let scheduled = |phase| {
            variants
                .iter()
                .find(|v| {
                    v.phase == phase
                        && v.schedule.is_some()
                        && v.engine.is_some()
                        && v.tokens_capacity > 1
                })
                .unwrap()
        };
        let selects = [
            ("profile", Class::Selects),
            ("class", Class::Selects),
            ("layout", Class::Selects),
            ("projection", Class::Selects),
            ("phase", Class::Selects),
            ("path", Class::Selects),
            ("query", Class::Selects),
            ("route_weight", Class::Selects),
            ("swiglu_clamp", Class::Selects),
            ("schedule", Class::Launch),
            ("symbol", Class::Launch),
        ];
        let mut decode = selects.to_vec();
        decode.extend([
            ("tokens_capacity", Class::Selects),
            ("knobs.tile_m", Class::Launch),
            ("knobs.stages", Class::Selects),
            ("knobs.warps", Class::Selects),
        ]);
        grouped_keys_follow_launches(scheduled(Phase::Decode), &decode);
        let mut prefill = selects.to_vec();
        prefill.extend([
            ("tokens_capacity", Class::Bound),
            ("knobs.tile_m", Class::Launch),
            ("knobs.split_k_capacity", Class::Selects),
        ]);
        grouped_keys_follow_launches(scheduled(Phase::Prefill), &prefill);
        // A packed engine prefill entry, which no schedule replaced: its
        // profile and class are no rank shape, so no template schedule can
        // build it; it never reads the stages, swizzle or weight stages.
        let engine = moe::own_catalog()
            .into_iter()
            .find(|v| v.phase == Phase::Prefill && v.schedule.is_none())
            .unwrap();
        let classes = [
            ("layout", Class::Selects),
            ("projection", Class::Selects),
            ("phase", Class::Selects),
            ("tokens_capacity", Class::Bound),
            ("path", Class::Launch),
            ("query", Class::Selects),
            ("knobs.tile_m", Class::Selects),
            ("knobs.tile_n", Class::Selects),
            ("knobs.tile_k", Class::Selects),
            ("knobs.warps", Class::Selects),
            ("knobs.split_k_capacity", Class::Selects),
            ("route_weight", Class::Selects),
            ("swiglu_clamp", Class::Selects),
            ("engine", Class::Selects),
            ("schedule", Class::Selects),
            ("symbol", Class::Launch),
        ];
        grouped_keys_follow_launches(&engine, &classes);
        let bound = buffers(u64::MAX);
        let projection = chained::catalog().remove(0);
        let names = field_names!(chained::Projection {
            symbol,
            query,
            row_tile,
            tokens_capacity,
            schedule
        });
        let fields: Vec<Field<chained::Projection>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|p| p.symbol = format!("{}_twin_v1", p.symbol)),
            ),
            ("query", Class::Selects, Box::new(|p| p.query.gemm.n *= 2)),
            ("row_tile", Class::Launch, Box::new(|p| p.row_tile *= 2)),
            (
                "tokens_capacity",
                Class::Inert,
                Box::new(|p| p.tokens_capacity *= 2),
            ),
            (
                "schedule",
                Class::Launch,
                Box::new(|p| {
                    p.schedule = if p.schedule == chained::Schedule::Columns {
                        chained::Schedule::BlockWarps
                    } else {
                        chained::Schedule::Columns
                    }
                }),
            ),
        ];
        keys_follow_launches(&projection, names, fields, chained_launch, |p| {
            let prepared = Sm120::new()
                .with_chained_projection(p.clone())
                .gated_mlp_projection(
                    &projection.query,
                    DECODE,
                    Need::rows(projection.tokens_capacity),
                )?;
            let launches = live_counts(projection.tokens_capacity)
                .into_iter()
                .map(|live| {
                    prepared
                        .launches(&bound, View::EMPTY, Counts::rows(live))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            Ok((format!("{:?}", p.entry()), launches))
        });
    }

    /// An entry whose cubin the build cannot compile never serves: prepare
    /// refuses it with the reason, rather than the build panicking, and an
    /// entry beside it of the same contract serves in every order.
    #[test]
    fn entries_the_build_cannot_compile_are_refused() {
        use crate::moe::template::{PrefillKnobs, ScheduleKnobs};
        let prefill_knobs = ScheduleKnobs::Prefill(PrefillKnobs {
            weight_stages: 2,
            activation_stages: 2,
            trim_gather: false,
            bulk_weights: false,
        });
        let packed = moe::own_catalog()
            .into_iter()
            .find(|v| v.phase == Phase::Prefill && v.projection == Projection::Down)
            .unwrap();
        let fp8 = fp8_fragment_variants();
        let decode = fp8
            .iter()
            .find(|v| v.phase == Phase::Decode && v.tokens_capacity == 8)
            .unwrap();
        let changed = |v: &Variant, change: &dyn Fn(&mut Variant)| {
            let mut v = v.clone();
            change(&mut v);
            v
        };
        let cases = [
            // A packed profile given a schedule, with or without an engine.
            (
                Variant {
                    schedule: Some(prefill_knobs),
                    ..packed.clone()
                },
                "a template schedule replaces no engine",
            ),
            (
                Variant {
                    engine: Some(prefill_knobs),
                    schedule: Some(prefill_knobs),
                    ..packed.clone()
                },
                "not an MoE rank profile",
            ),
            // An engine the template covers with no resolved schedule.
            (
                changed(decode, &|v| v.schedule = None),
                "no resolved schedule row",
            ),
            // A schedule of another reduction order than its engine's.
            (
                changed(decode, &|v| v.schedule = Some(prefill_knobs)),
                "the schedule changes the engine's reduction order",
            ),
            // A decode symbol naming no bucket for its cubin.
            (
                changed(decode, &|v| v.symbol = "chop_moe_unbucketed_v1".into()),
                "a decode entry's symbol names no bucket",
            ),
            // Weights the packed engine does not run, where no template
            // schedule builds the entry.
            (
                changed(&packed, &|v| v.query.weight = WeightFormat::Bf16),
                "the packed engine runs MXFP4, FP8 block, NVFP4 and MXFP8 weights only",
            ),
            (
                changed(&packed, &|v| v.query.weight = WeightFormat::Fp8PerTensor),
                "the packed engine runs MXFP4, FP8 block, NVFP4 and MXFP8 weights only",
            ),
            // A ring schedule naming more threads than a u32 holds.
            (
                changed(decode, &|v| {
                    if let Some(ScheduleKnobs::Ring(ring)) = &mut v.schedule {
                        ring.warps = u32::MAX;
                    }
                }),
                "ring warps overflow the CTA threads",
            ),
        ];
        for (v, reason) in &cases {
            assert_eq!(v.buildable(), Err(Refusal(reason)), "{}", v.symbol);
            let op = grouped_contract(v);
            let Err(refused) = Sm120::new().with_experts(v.clone()).gated_mlp_projection(
                &op,
                DECODE,
                Need::rows(v.tokens_capacity.min(8)),
            ) else {
                panic!("{} serves", v.symbol);
            };
            assert_eq!(
                refused,
                Refused::Rule(format!(
                    "no SM120 gated MLP entry for the op can be built: {reason}"
                )),
                "{}",
                v.symbol
            );
        }
        // Beside a buildable entry of its contract, it changes nothing.
        for (broken, _) in [&cases[2], &cases[7]] {
            for order in crate::backend::select::orders(&[decode.clone(), broken.clone()]) {
                let prepared = order
                    .into_iter()
                    .fold(Sm120::new(), Sm120::with_experts)
                    .gated_mlp_projection(&grouped_contract(decode), DECODE, Need::rows(8))
                    .unwrap();
                assert_eq!(prepared.info().symbols[0].0, decode.symbol);
            }
        }
        // The auditor's case: a packed BF16 entry, which no engine builds,
        // beside a chained BF16 projection of its contract does not tie
        // with it, in either registration order.
        let mut chained = chained::split::catalog()
            .into_iter()
            .find(|p| {
                p.query.gemm.grouped.is_some()
                    && p.query.projection == kernel_core::gated_mlp::GatedMlpProjection::Down
            })
            .unwrap();
        chained.query.gemm.weight = WeightFormat::Bf16;
        let packed_bf16 = Variant {
            query: chained.query.gemm,
            tokens_capacity: chained.tokens_capacity,
            ..packed.clone()
        };
        assert_eq!(grouped_contract(&packed_bf16), chained.query);
        assert!(packed_bf16.buildable().is_err());
        for reverse in [false, true] {
            let backend = if reverse {
                Sm120::new()
                    .with_experts(packed_bf16.clone())
                    .with_chained_projection(chained.clone())
            } else {
                Sm120::new()
                    .with_chained_projection(chained.clone())
                    .with_experts(packed_bf16.clone())
            };
            let prepared = backend
                .gated_mlp_projection(&chained.query, DECODE, Need::rows(8))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, chained.symbol);
        }
    }

    /// The auditor's case: prefill template entries differing only in the
    /// weight or activation stages their schedule replaces, or in another
    /// engine knob that only gates support, are one entry in every order;
    /// so are packed engine prefill entries differing only in the stages,
    /// swizzle or weight stages the packed engine never reads.
    #[test]
    fn prefill_entries_differing_in_unread_knobs_are_one() {
        let scheduled = fp8_fragment_variants()
            .into_iter()
            .find(|v| v.phase == Phase::Prefill && v.schedule.is_some())
            .unwrap();
        let packed = moe::own_catalog()
            .into_iter()
            .find(|v| v.phase == Phase::Prefill && v.schedule.is_none())
            .unwrap();
        for base in [scheduled, packed] {
            let twins = [
                |v: &mut Variant| v.weight_stages += 1,
                |v: &mut Variant| v.knobs.stages += 1,
                |v: &mut Variant| v.knobs.swizzle += 1,
            ]
            .map(|change| {
                let mut twin = base.clone();
                change(&mut twin);
                twin
            });
            let mut registered = vec![base.clone()];
            registered.extend(twins);
            let op = grouped_contract(&base);
            for order in crate::backend::select::orders(&registered) {
                let prepared = order
                    .into_iter()
                    .fold(Sm120::new(), Sm120::with_experts)
                    .gated_mlp_projection(&op, DECODE, Need::rows(base.tokens_capacity))
                    .unwrap();
                assert_eq!(prepared.info().symbols[0].0, base.symbol);
            }
        }
    }

    /// Decode template entries differing only in the replaced engine's knob
    /// values or in prefill weight stages, which no decode launch reads,
    /// are one entry in every order.
    #[test]
    fn decode_entries_differing_in_unread_records_are_one() {
        use crate::moe::template::ScheduleKnobs;
        let base = fp8_fragment_variants()
            .into_iter()
            .find(|v| v.phase == moe::Phase::Decode && v.schedule.is_some() && v.engine.is_some())
            .unwrap();
        let mut engine = base.clone();
        engine.engine = engine.engine.map(|knobs| match knobs {
            ScheduleKnobs::Ring(mut ring) => {
                ring.early_row = !ring.early_row;
                ScheduleKnobs::Ring(ring)
            }
            other => other,
        });
        let mut stages = base.clone();
        stages.weight_stages += 1;
        let op = grouped_contract(&base);
        for order in crate::backend::select::orders(&[base.clone(), engine, stages]) {
            let prepared = order
                .into_iter()
                .fold(Sm120::new(), Sm120::with_experts)
                .gated_mlp_projection(&op, DECODE, Need::rows(base.tokens_capacity))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, base.symbol);
        }
    }
}
