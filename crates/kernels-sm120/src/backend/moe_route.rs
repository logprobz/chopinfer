//! The MoE ops around the experts through the route entries a model
//! instantiates: the entries of a route cubin
//! (`moe::expert_parallel::RouteEntry`: the router, decode routes, the
//! parallel decode route, the prefill permutation and the shared-expert
//! combine), chained local routes (`moe::chained::LocalRoute`: the one-CTA
//! decode route, or row selection, per-expert count and scatter) and the
//! chained routed sum (`moe::chained::RoutedSum`, and its F32 form that adds
//! a row), the one-CTA sigmoid-bias decode route of 256 experts and the
//! router GEMV with that route fused into its tail (`moe::sigmoid_route`),
//! and row engine entries
//! (`row::RowInstantiation`: routers, the tiled permutation of every
//! expert's routes and the combine).
//!
//! A route op serves its use through one path: a decode route that selects
//! and builds the route metadata in one launch, or the router followed by
//! the permutation. A chained local route is the exception: a step of fewer
//! rows takes the chained route of the least capacity that holds them
//! ([`chained_runs`]), as the chained expert ops run those rows at its pad.
//! Every choice of an entry or a path is made by the op's
//! own fields alone, whatever order the model registered its entries in
//! ([`choose`]): of those that match the op (for a route: pad to the expert
//! op's route tile), those that hold the use's rows and claim the needed
//! exactness remain, and the least capacity wins; entries that launch alike
//! count as one, and a tie that would launch differently is refused. Paths
//! are concrete combinations of entries, so a registration the use's path
//! does not take never decides or refuses a prepare. Kernels plan as the
//! model builders plan them, launches record the rows those builders record
//! (`Kernel::plan`), and arguments are bound by parameter name.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MoeQuery, Refusal, RouterKind, RowQuery, RuntimeCounts,
    Validate,
};
use kernel_core::moe::{
    CombineBuffers, ExpertCombineOp, RouteBuffers, RoutedSumBuffers, RouterBuffers,
    SharedCombineBuffers,
};
use kernel_core::precision::OperationUse;
use kernel_core::properties::{Batch, Invariance, Rows, Scope, Ties};

use super::rows::row_launch;
use super::select::{Candidate, Chosen, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::moe::chained::{LocalRoute, RouteKernel, RoutedSum, RoutedSumOutput};
use crate::moe::expert_parallel::{ROUTE_BLOCK, ROUTE_TILE, RouteContract, RouteEntry};
use crate::moe::padded_capacity;
use crate::moe::parallel::{PARALLEL_ROUTE_TICKET_BYTES, RouteScoring};
use crate::moe::sigmoid_route::{
    decode_route_arguments, decode_route_cubin_spec, decode_route_plan,
};
use crate::row::builders::expert_combine_entry;
use crate::row::{Grid, RowInstantiation, smallest_bucket};
use kernel_core::contract::DType;

use crate::gemm::{self, GemmVariant};
use crate::moe::sigmoid_route::{fused_router_arguments, fused_router_cubin_spec};
use kernel_core::contract::{DenseKernel, GemmQuery, InstantiationSpec, StaticQuery};
use kernel_core::moe::ProjectedRouteBuffers;
use kernel_core::precision::WeightLayout;

/// The route entries of a model's selection.
#[derive(Default)]
pub(super) struct Selection {
    /// Route cubin entries other than decode routes
    /// (`Sm120::with_route_entry`).
    pub(super) entries: Vec<RouteEntry>,
    /// Decode routes, each with the router kind it computes
    /// (`Sm120::with_decode_route`).
    pub(super) decode_routes: Vec<(RouteEntry, RouterKind)>,
    /// Chained local routes and the cubin prefix of their entries.
    pub(super) local: Vec<(LocalRoute, String)>,
    /// Row engine routers and combines.
    pub(super) rows: Vec<RowInstantiation>,
    /// Tiled permutations of every expert's routes: count, prefix, scan
    /// and scatter.
    pub(super) tiles: Vec<[RowInstantiation; 4]>,
    /// Sigmoid-bias decode routes: symbol, token capacity and declaration.
    pub(super) sigmoid: Vec<(String, u32, Invariance)>,
    /// Chained routed sums and the cubin prefix of their entries.
    pub(super) sums: Vec<(RoutedSum, String)>,
    /// Chained F32 routed sums that add a row, and the cubin prefix of
    /// their entries (`RoutedSum::add_entry`).
    pub(super) sum_adds: Vec<(RoutedSum, String)>,
    /// Router GEMVs with the sigmoid-bias route fused into their tail
    /// (`Sm120::with_fused_router`).
    pub(super) fused: Vec<FusedRouterEntry>,
}

/// The experts, top-k and padding the sigmoid-bias decode route compiles
/// (the one-CTA decode route of moe/route.cuh).
const SIGMOID_EXPERTS: u32 = 256;
const SIGMOID_TOP_K: u32 = 8;
const SIGMOID_PAD: u32 = 16;

/// The most rows a row engine entry serves: its largest capacity bucket.
const ROW_CAPACITY: u32 = crate::row::hyper_connection::MAX_ROWS;

/// How a kernel is planned.
#[derive(Clone, Debug)]
enum Source {
    /// A route cubin entry: a decode route at its own capacity, any other
    /// entry at the step's rows (`launch::route` in the model builders).
    Entry(RouteEntry),
    /// A chained route kernel at the step's rows.
    Local(LocalRoute, RouteKernel),
    /// A row engine entry at the smallest bucket that holds the rows, or at
    /// a fixed bucket (a tiled permutation at its tile tables' capacity).
    Row(RowInstantiation, Option<u32>),
    /// The sigmoid-bias decode route at its token capacity.
    Sigmoid(u32),
    /// A chained routed sum at the step's rows.
    Sum(RoutedSum),
    /// A chained F32 routed sum that adds a row, at the step's rows.
    SumAdd(RoutedSum),
    /// A router GEMV with the sigmoid-bias route in its tail, planned as
    /// its GEMV entry at the entry's token bucket.
    Fused(FusedRouterEntry),
}

#[derive(Clone, Debug)]
struct Kernel {
    symbol: String,
    params: Vec<EntryParam>,
    source: Source,
    invariance: Invariance,
}

impl Kernel {
    fn entry(entry: &RouteEntry) -> Kernel {
        Kernel {
            symbol: entry.symbol.clone(),
            params: entry.params.clone(),
            invariance: entry.invariance,
            source: Source::Entry(entry.clone()),
        }
    }

    /// A chained route kernel and its declaration
    /// (`chained::LocalRoute::invariance`).
    fn local(route: LocalRoute, kernel: RouteKernel, prefix: &str) -> Result<Kernel, Refusal> {
        let symbol = route.symbol(prefix, kernel);
        let params = route
            .entries(prefix)?
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .ok_or(Refusal("a chained route lacks one of its kernels"))?
            .params;
        Ok(Kernel {
            symbol,
            params,
            invariance: route.invariance(kernel),
            source: Source::Local(route, kernel),
        })
    }

    fn row(entry: &RowInstantiation, bucket: Option<u32>) -> Kernel {
        Kernel {
            symbol: entry.symbol.clone(),
            params: entry.params.clone(),
            invariance: entry.invariance,
            source: Source::Row(entry.clone(), bucket),
        }
    }

    fn sigmoid(symbol: &str, capacity: u32, invariance: Invariance) -> Kernel {
        let params = decode_route_cubin_spec(symbol, symbol)
            .entries
            .remove(0)
            .params;
        Kernel {
            symbol: symbol.into(),
            params,
            invariance,
            source: Source::Sigmoid(capacity),
        }
    }

    /// A chained routed sum and its declaration
    /// (`chained::RoutedSum::invariance`).
    fn sum(sum: RoutedSum, prefix: &str) -> Result<Kernel, Refusal> {
        let entry = sum.entry(prefix)?;
        Ok(Kernel {
            symbol: entry.symbol,
            params: entry.params,
            invariance: sum.invariance(),
            source: Source::Sum(sum),
        })
    }

    /// A chained F32 routed sum that adds a row and its declaration, the
    /// sum's (`chained::RoutedSum::invariance`).
    fn sum_add(sum: RoutedSum, prefix: &str) -> Result<Kernel, Refusal> {
        let entry = sum.add_entry(prefix)?;
        Ok(Kernel {
            symbol: entry.symbol,
            params: entry.params,
            invariance: sum.invariance(),
            source: Source::SumAdd(sum),
        })
    }

    /// The launch of a step of `rows` rows: the rows its builder plans and
    /// records it at (a decode route's capacity, a row entry's bucket, a
    /// tiled permutation's fixed capacity, otherwise the step's rows) and
    /// its plan.
    fn plan(&self, rows: u32) -> Result<(u32, LaunchPlan), Refusal> {
        let (planned, plan) = match &self.source {
            Source::Entry(entry) => {
                let capacity = if decode_route(entry) {
                    entry.tokens_capacity
                } else {
                    rows
                };
                (capacity, entry.plan(capacity, capacity)?)
            }
            Source::Local(route, kernel) => (rows, route.plan(*kernel, rows)?),
            Source::Row(entry, bucket) => {
                let bucket = match bucket {
                    Some(bucket) => *bucket,
                    None => smallest_bucket(rows).ok_or(Refusal("no row bucket holds the rows"))?,
                };
                (bucket, entry.plan(bucket, row_counts(rows))?)
            }
            Source::Sigmoid(capacity) => (*capacity, decode_route_plan(*capacity, rows)?),
            Source::Sum(sum) | Source::SumAdd(sum) => (rows, sum.plan(rows)?),
            Source::Fused(entry) => fused_plan(entry, rows)?,
        };
        check_plan(&plan)?;
        Ok((planned, plan))
    }

    /// What the kernel launches: its symbol and argument shape and the
    /// record it plans from, without its profile or claims.
    fn launch(&self) -> String {
        let Kernel {
            symbol,        // the launched symbol
            params,        // the ABI
            invariance: _, // claims
            source,        // below
        } = self;
        let source = match source {
            Source::Entry(entry) => {
                let RouteEntry {
                    profile: _,      // filters catalogs; no launch reads it
                    experts, // a rank's entries' rank check; a router's contract holds its width
                    local_experts, // a rank's entries' plans and rank check
                    top_k,   // a rank's entries' plans
                    symbol: _, // the kernel's
                    contract, // selects the op
                    tokens_capacity, // a decode route's plan
                    threads, // the plan
                    params: _, // the kernel's
                    body,    // the code
                    parallel, // a parallel route's plan and code
                    invariance: _, // claims
                } = entry;
                // A router plans at the step's rows, its contract holding
                // its width, whatever experts a rank holds; a rank's entry
                // plans and checks its rank by its experts, local experts
                // and top-k; a decode route plans at its token capacity.
                // Other capacities only bound the rows the selection
                // settles.
                let rank = match contract {
                    RouteContract::Router(_) => String::new(),
                    RouteContract::Moe(_) => format!(" {experts} {local_experts} {top_k}"),
                };
                let planned = if decode_route(entry) {
                    format!(" {tokens_capacity}")
                } else {
                    String::new()
                };
                format!("entry {contract:?} {threads} {body} {parallel:?}{rank}{planned}")
            }
            Source::Local(route, kernel) => {
                let LocalRoute {
                    experts,       // selects the op
                    local_experts, // selects the op
                    top_k,         // selects the op
                    capacity,      // the plans
                    pad,           // the route tile; selects the op
                } = route;
                format!(
                    "local {:?} {kernel:?}",
                    (experts, local_experts, top_k, capacity, pad)
                )
            }
            Source::Row(entry, bucket) => format!("row {} {bucket:?}", row_launch(entry)),
            Source::Sigmoid(capacity) => format!("sigmoid {capacity}"),
            Source::Sum(sum) => {
                let RoutedSum {
                    top_k,         // selects the op
                    width,         // selects the op
                    local_experts, // selects the op
                    capacity,      // in the symbol
                    output,        // selects the op
                } = sum;
                format!("sum {:?}", (top_k, width, local_experts, capacity, output))
            }
            Source::SumAdd(sum) => {
                let RoutedSum {
                    top_k,         // selects the op
                    width,         // selects the op
                    local_experts, // selects the op
                    capacity,      // in the symbol
                    output,        // F32, the only form that adds
                } = sum;
                format!(
                    "sum add {:?}",
                    (top_k, width, local_experts, capacity, output)
                )
            }
            Source::Fused(entry) => fused_launch(entry),
        };
        format!("{symbol} {params:?} {source}")
    }

    fn takes(&self, name: &str) -> bool {
        self.params
            .iter()
            .any(|param| param_name(param) == Some(name))
    }

    /// Whether the kernel selects experts.
    fn selects(&self) -> bool {
        self.takes("logits")
    }

    /// The kernel's claims: its declaration, about the routed operation
    /// (`Scope::RoutedOperation`) when it binds a route metadata role. Route
    /// metadata (counts, offsets, sorted rows, the padded total, the active
    /// experts, tile tables) depends on every route of the launch, so no
    /// claim of a kernel that writes it is about its own rows alone,
    /// whatever scope the model declared: a one-row claim keeps each token's
    /// logits, expert ids and weights equal to the token's alone, and the
    /// metadata is the launch's own.
    fn claims(&self) -> Invariance {
        let metadata = [
            "counts",
            "offsets",
            "sorted_rows",
            "num_padded_rows",
            "padded_rows",
            "active_experts",
            "tile_counts",
            "tile_offsets",
        ];
        if metadata.iter().any(|name| self.takes(name)) {
            self.invariance.of_routed_operation()
        } else {
            self.invariance
        }
    }

    /// The arguments of `bind`'s values in parameter order, through the
    /// entry's own argument builder.
    fn arguments(
        &self,
        bind: &dyn Fn(&str) -> Result<Argument, Refusal>,
    ) -> Result<ArgumentList, Refusal> {
        let values = self
            .params
            .iter()
            .map(|param| {
                let name = param_name(param).ok_or(Refusal("unsupported route parameter"))?;
                let value = bind(name)?;
                match (param, &value) {
                    (
                        EntryParam::DevicePointer { .. },
                        Argument::DeviceBuffer(_) | Argument::NullDevicePointer,
                    )
                    | (EntryParam::U32 { .. }, Argument::U32(_)) => Ok(value),
                    _ => Err(Refusal("a route parameter binds the wrong kind")),
                }
            })
            .collect::<Result<Vec<_>, Refusal>>()?;
        match &self.source {
            Source::Entry(entry) => {
                let mut handles = Vec::new();
                let mut rank = None;
                for value in values {
                    match value {
                        Argument::DeviceBuffer(handle) => handles.push(handle),
                        Argument::U32(value) => rank = Some(value),
                        _ => return Err(Refusal("unsupported route argument")),
                    }
                }
                entry.arguments(&handles, rank)
            }
            Source::Local(..) | Source::Sum(_) | Source::SumAdd(_) => {
                let arguments = ArgumentList(values);
                arguments.validate()?;
                Ok(arguments)
            }
            Source::Row(entry, _) => entry.pack_arguments(values),
            Source::Sigmoid(_) => {
                let mut handles = Vec::new();
                let mut scale = None;
                for value in values {
                    match value {
                        Argument::DeviceBuffer(handle) => handles.push(handle),
                        Argument::U32(bits) => scale = Some(f32::from_bits(bits)),
                        _ => return Err(Refusal("unsupported route argument")),
                    }
                }
                let handles: [u64; 10] = handles
                    .try_into()
                    .map_err(|_| Refusal("the decode route takes ten buffers"))?;
                decode_route_arguments(handles, scale.ok_or(Refusal("the decode route scales"))?)
            }
            Source::Fused(_) => {
                let mut handles = Vec::new();
                let mut scale = None;
                for value in values {
                    match value {
                        Argument::DeviceBuffer(handle) => handles.push(handle),
                        Argument::U32(bits) => scale = Some(f32::from_bits(bits)),
                        _ => return Err(Refusal("unsupported route argument")),
                    }
                }
                let handles: [u64; 14] = handles
                    .try_into()
                    .map_err(|_| Refusal("the fused router takes fourteen buffers"))?;
                fused_router_arguments(handles, scale.ok_or(Refusal("the fused router scales"))?)
            }
        }
    }
}

fn param_name(param: &EntryParam) -> Option<&str> {
    match param {
        EntryParam::DevicePointer { name, .. } | EntryParam::U32 { name } => Some(name),
        _ => None,
    }
}

/// Whether a route cubin entry is a decode route, which selects and builds
/// the route metadata at its own capacity (as `RouteEntry::plan` tells).
pub(super) fn decode_route(entry: &RouteEntry) -> bool {
    entry.parallel.is_some() || entry.symbol.contains("_decode_route_")
}

fn row_counts(rows: u32) -> RuntimeCounts {
    RuntimeCounts {
        m: rows,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    }
}

/// The versioned implementation of a prepared op: a digest of the
/// declarations of every kernel it may launch.
fn implementation(op: &str, declaration: String) -> String {
    let digest = kernel_build::digest(declaration.as_bytes());
    format!("sm120.moe.{op}.v1#{}", &digest[..16])
}

/// Refuses a capacity no prepared op serves, before any per-row table is
/// built.
fn served(need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(
            "a prepared op serves 1 to MAX_CAPACITY rows".into(),
        ));
    }
    Ok(())
}

/// A kernel's claims over the rows it serves, at most `served`: one-row
/// bits up to its served rows hold at every row count it launches at, and
/// one row has no other rows to depend on.
fn effective(invariance: Invariance, served: u32) -> Invariance {
    if invariance.one_row_at(served) {
        Invariance {
            rows: Rows::OneRow { max: u32::MAX },
            batch: Batch::Independent,
            ..invariance
        }
    } else {
        invariance
    }
}

/// One way to route up to `capacity` rows: its kernels in launch order.
#[derive(Clone, Debug)]
struct Path {
    kernels: Vec<Kernel>,
    capacity: u32,
    /// The routes each expert's sorted rows are padded to.
    pad: u32,
    /// The token rows the path's tile tables are sized for, when they do
    /// not follow the step's rows.
    tile_rows: Option<u32>,
}

impl Path {
    /// The path's claims when it serves up to `served` rows. The tie rule
    /// is that of the kernels that select.
    fn guarantees(&self, served: u32) -> Invariance {
        let served = served.min(self.capacity);
        let ties = if self
            .kernels
            .iter()
            .filter(|kernel| kernel.selects())
            .all(|kernel| kernel.invariance.ties == Ties::LowerIndex)
        {
            Ties::LowerIndex
        } else {
            Ties::NotApplicable
        };
        // The kernels run in sequence: with route metadata among them, a
        // kernel's claim about its own rows is one about the routed
        // operation's rows, so claims take the path's scope before they meet.
        let scope = if self
            .kernels
            .iter()
            .any(|kernel| kernel.claims().scope == Scope::RoutedOperation)
        {
            Scope::RoutedOperation
        } else {
            Scope::Entry
        };
        let claims = self
            .kernels
            .iter()
            .map(|kernel| Invariance {
                scope,
                ..effective(kernel.claims(), served)
            })
            .reduce(Invariance::meet)
            .unwrap_or(Invariance::NONE);
        claims.with_ties(ties)
    }

    fn takes(&self, name: &str) -> bool {
        self.kernels.iter().any(|kernel| kernel.takes(name))
    }

    /// Whether the path is a chained local route
    /// (`moe::chained::LocalRoute`).
    fn chained(&self) -> bool {
        self.kernels
            .iter()
            .all(|kernel| matches!(kernel.source, Source::Local(..)))
    }

    /// What the path launches: its kernels' launches and the padding and
    /// tile rows it plans them at. Its capacity only bounds the rows.
    fn launch(&self) -> String {
        let Path {
            kernels,     // their launches
            capacity: _, // bounds the rows the selection settles
            pad,         // the route tile the plans pad to
            tile_rows,   // the tile tables' plans
        } = self;
        let kernels = kernels.iter().map(Kernel::launch).collect::<Vec<_>>();
        format!("{kernels:?} {pad} {tile_rows:?}")
    }
}

/// A route op's geometry: the router's and the rank's.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    experts: u32,
    local: u32,
    top_k: u32,
    rank: u32,
    /// The scale of a sigmoid-bias router's weights.
    scale: Option<f32>,
}

impl Geometry {
    fn of(router: &RowQuery, routes: &MoeQuery, rank: u32) -> Result<Geometry, Refused> {
        router.validate()?;
        routes.validate()?;
        let RowQuery::Router {
            experts,
            top_k,
            kind,
        } = *router
        else {
            return Err(Refused::Rule("expected a router".into()));
        };
        let MoeQuery::LocalRoutes {
            experts: route_experts,
            local_experts,
            top_k: route_top_k,
            ..
        } = *routes
        else {
            return Err(Refused::Rule("expected local routes".into()));
        };
        if route_experts != experts || route_top_k != top_k {
            return Err(Refused::Rule(
                "the local routes are over another router's experts or top-k".into(),
            ));
        }
        if rank >= experts / local_experts {
            return Err(Refused::Rule("the rank is past the expert shards".into()));
        }
        let scale = match kind {
            RouterKind::SigmoidBiasSelectNormalized { scaling_factor } => Some(scaling_factor),
            _ => None,
        };
        Ok(Geometry {
            experts,
            local: local_experts,
            top_k,
            rank,
            scale,
        })
    }

    fn of_entry(&self, entry: &RouteEntry) -> bool {
        entry.experts == self.experts
            && entry.local_experts == self.local
            && entry.top_k == self.top_k
    }

    /// The I32 sorted rows `pad` gives `rows` token rows: every route and
    /// each routed expert's padding.
    fn sorted_bytes(&self, rows: u32, pad: u32) -> Result<u64, Refusal> {
        let routes = rows
            .checked_mul(self.top_k)
            .ok_or(Refusal("route rows overflow"))?;
        Ok(u64::from(padded_capacity(routes, self.local, pad)?) * 4)
    }

    /// The bytes of one route tile table at `rows` token rows: an I32 per
    /// local expert per tile of routes.
    fn tile_bytes(&self, rows: u32) -> u64 {
        (u64::from(rows) * u64::from(self.top_k)).div_ceil(u64::from(ROUTE_TILE))
            * u64::from(self.local)
            * 4
    }
}

/// Whether a parallel route's scoring computes `kind`.
fn scores(scoring: RouteScoring, kind: RouterKind, experts: u32, groups: u32, take: u32) -> bool {
    match (scoring, kind) {
        (RouteScoring::Softmax, RouterKind::SoftmaxTopK { renormalize: true }) => true,
        (RouteScoring::SigmoidBias, RouterKind::SigmoidBiasSelectNormalized { .. }) => true,
        (
            RouteScoring::GroupedSoftmax,
            RouterKind::GroupLimitedSelectedSoftmax {
                n_group,
                topk_group,
            },
        ) => n_group == groups && topk_group == take && experts.is_multiple_of(groups),
        _ => false,
    }
}

/// Every concrete path of a selection that routes `router`'s selection
/// into the metadata of `geometry`'s rank, for an op of up to `capacity`
/// rows: one per combination of the entries each path takes, so that only
/// the path the op's use takes is chosen among ([`choose`]) and an unused
/// registration never decides or refuses a prepare. A decode route serves
/// only ops of the router kind it registered with (and a parallel one only
/// those its geometry's scoring computes). The sigmoid-bias decode route and a tiled permutation route every
/// expert (one rank); the tiled permutation follows a row engine router of
/// the same query and plans at the bucket of `capacity`, which its tile
/// tables assume.
fn paths(
    selection: &Selection,
    router: &RowQuery,
    geometry: &Geometry,
    capacity: u32,
) -> Vec<Path> {
    let RowQuery::Router { kind, .. } = *router else {
        return Vec::new();
    };
    let entries = || selection.entries.iter();
    let local_routes = |entry: &&RouteEntry| {
        geometry.of_entry(entry)
            && matches!(
                entry.contract,
                RouteContract::Moe(MoeQuery::LocalRoutes { .. })
            )
    };
    let permute = |suffix: &str| {
        entries()
            .filter(local_routes)
            .filter(|entry| !decode_route(entry) && entry.symbol.ends_with(suffix))
            .collect::<Vec<_>>()
    };
    let routers = entries()
        .filter(|entry| {
            entry.experts == geometry.experts
                && entry.top_k == geometry.top_k
                && entry.contract == RouteContract::Router(*router)
        })
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    // A decode route computes the router kind it was registered with (its
    // route cubin's router's), which its entry does not record; a parallel
    // one also computes its geometry's scoring, which must agree.
    for (entry, computes) in &selection.decode_routes {
        let scored = *computes == kind
            && entry.parallel.is_none_or(|parallel| {
                scores(
                    parallel.scoring,
                    kind,
                    geometry.experts,
                    parallel.groups,
                    parallel.take_groups,
                )
            });
        if !local_routes(&entry) || !scored {
            continue;
        }
        paths.push(Path {
            kernels: vec![Kernel::entry(entry)],
            capacity: entry.tokens_capacity,
            pad: entry
                .parallel
                .map_or(ROUTE_BLOCK, |parallel| parallel.padding),
            tile_rows: None,
        });
    }
    if !routers.is_empty() {
        let [counts, prefixes, scans, scatters] = [
            "_permute_count_v1",
            "_permute_prefix_v1",
            "_permute_scan_v1",
            "_permute_scatter_v1",
        ]
        .map(permute);
        for router_entry in &routers {
            for count in &counts {
                for prefix in &prefixes {
                    for scan in &scans {
                        for scatter in &scatters {
                            let kernels = [*router_entry, *count, *prefix, *scan, *scatter];
                            paths.push(Path {
                                capacity: kernels
                                    .iter()
                                    .map(|entry| entry.tokens_capacity)
                                    .min()
                                    .unwrap_or(0),
                                kernels: kernels.into_iter().map(Kernel::entry).collect(),
                                pad: ROUTE_BLOCK,
                                tile_rows: None,
                            });
                        }
                    }
                }
            }
        }
    }
    let sigmoid = matches!(kind, RouterKind::SigmoidBiasSelectNormalized { .. });
    // Chained routes score sigmoid plus bias.
    if sigmoid {
        for (route, prefix) in &selection.local {
            if route.experts != geometry.experts
                || route.local_experts != geometry.local
                || route.top_k != geometry.top_k
            {
                continue;
            }
            let Ok(kernels) = route
                .kernels()
                .iter()
                .map(|&kernel| Kernel::local(*route, kernel, prefix))
                .collect::<Result<Vec<_>, Refusal>>()
            else {
                continue;
            };
            paths.push(Path {
                kernels,
                capacity: route.capacity,
                pad: route.pad,
                tile_rows: None,
            });
        }
    }
    let every_expert = geometry.local == geometry.experts;
    if sigmoid
        && every_expert
        && geometry.experts == SIGMOID_EXPERTS
        && geometry.top_k == SIGMOID_TOP_K
    {
        for (symbol, tokens, invariance) in &selection.sigmoid {
            paths.push(Path {
                kernels: vec![Kernel::sigmoid(symbol, *tokens, *invariance)],
                capacity: *tokens,
                pad: SIGMOID_PAD,
                tile_rows: None,
            });
        }
    }
    let row_routers = selection.rows.iter().filter(|entry| entry.query == *router);
    if let (true, Some(bucket)) = (every_expert, smallest_bucket(capacity)) {
        for row_router in row_routers {
            for tiles in &selection.tiles {
                let RowQuery::ExpertPermute {
                    experts,
                    block_size,
                } = tiles[0].query
                else {
                    continue;
                };
                let tiled = matches!(
                    tiles[0].grid,
                    Grid::RouteTiles { top_k, tile } if top_k == geometry.top_k && tile == ROUTE_TILE
                );
                if experts != geometry.experts
                    || !tiled
                    || tiles.iter().any(|entry| entry.query != tiles[0].query)
                {
                    continue;
                }
                let mut kernels = vec![Kernel::row(row_router, None)];
                kernels.extend(tiles.iter().map(|entry| Kernel::row(entry, Some(bucket))));
                paths.push(Path {
                    kernels,
                    capacity: bucket.min(ROW_CAPACITY),
                    pad: block_size,
                    tile_rows: Some(bucket),
                });
            }
        }
    }
    paths
}

/// Rank `rank`'s routes of `router` into `routes` at `usage`, padded to
/// `route_tile`, through the model's route entries: the path [`choose`]
/// takes of those that pad to the tile and scale as the router does. SM120
/// chooses by the use's capacity (the decode routes hold a decode or verify
/// bucket; the router and permutation hold prefill rows and wider buckets);
/// the use's phase and graph choose nothing among today's entries.
///
/// A chained route ([`chained_runs`]) routes a step of fewer rows through
/// the chained route of the least capacity that holds them, padded to its
/// own pad, as the chained expert ops that read the routes run those rows
/// at that row tile (`moe_experts::check_route_tiles`); at the capacity's
/// rows it pads to `route_tile`.
pub(super) fn route(
    selection: &Selection,
    router: &RowQuery,
    routes: &MoeQuery,
    rank: u32,
    route_tile: u32,
    usage: OperationUse,
    need: Need,
) -> Result<Prepared<RouteBuffers<View>>, Refused> {
    served(need)?;
    if need.capacity != usage.capacity {
        return Err(Refused::Rule(
            "a route op's capacity is its use's capacity".into(),
        ));
    }
    let geometry = Geometry::of(router, routes, rank)?;
    if let MoeQuery::LocalRoutes { capacity, .. } = *routes
        && need.capacity > capacity
    {
        return Err(Refused::Rule(
            "the capacity exceeds the local routes' capacity".into(),
        ));
    }
    let paths = paths(selection, router, &geometry, need.capacity);
    if !paths.is_empty() && !paths.iter().any(|path| path.pad == route_tile) {
        return Err(Refused::Rule(
            "no SM120 route entries pad to the route tile".into(),
        ));
    }
    let chained = paths
        .iter()
        .filter(|path| path.chained())
        .cloned()
        .collect::<Vec<_>>();
    let candidates = paths
        .into_iter()
        .filter(|path| path.pad == route_tile)
        .map(|path| path_candidate(path, need))
        .collect::<Vec<_>>();
    let Chosen {
        entry: path,
        guarantees: claimed,
    } = choose("route", candidates, need)?;
    let (paths, runs, mut guarantees) = if path.chained() {
        chained_runs(&chained, &path, route_tile, need)?
    } else {
        (vec![path], vec![(1, 0)], claimed)
    };
    let mut rows = Vec::with_capacity(need.capacity as usize);
    for (at, &(first, path)) in runs.iter().enumerate() {
        let last = runs
            .get(at + 1)
            .map_or(need.capacity, |&(next, _)| next - 1);
        let path =
            u8::try_from(path).map_err(|_| Refusal("a route table holds at most 256 paths"))?;
        rows.extend((first..=last).map(|_| path));
    }
    let plans = rows
        .iter()
        .zip(1..)
        .map(|(&path, rows)| {
            paths[usize::from(path)]
                .kernels
                .iter()
                .map(|kernel| kernel.plan(rows))
                .collect::<Result<Vec<_>, Refusal>>()
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    // The paths claim no more than every registration of their launches
    // declares.
    if let Rows::OneRow { max } = guarantees.rows {
        guarantees.rows = Rows::OneRow {
            max: max.min(need.capacity),
        };
    }
    // Each kernel at the dynamic shared memory the capacity plans, or of a
    // chained table the most any row count plans.
    let mut symbols: Vec<(String, u32)> = Vec::new();
    let planned = if paths.len() == 1 {
        &plans[plans.len() - 1..]
    } else {
        &plans[..]
    };
    for (&path, plans) in rows[rows.len() - planned.len()..].iter().zip(planned) {
        for (kernel, (_, plan)) in paths[usize::from(path)].kernels.iter().zip(plans) {
            match symbols
                .iter_mut()
                .find(|(symbol, _)| *symbol == kernel.symbol)
            {
                Some((_, shared)) => *shared = (*shared).max(plan.dynamic_shared_bytes),
                None => symbols.push((kernel.symbol.clone(), plan.dynamic_shared_bytes)),
            }
        }
    }
    let declaration = match paths.as_slice() {
        [path] => format!("{path:?}"),
        _ => format!("{:?}", (&paths, &runs)),
    };
    let mut info = OpInfo::new(implementation("route", declaration), guarantees, symbols);
    // Each role holds what the widest launch of any path needs.
    let most = |path: usize| {
        rows.iter()
            .rposition(|&at| usize::from(at) == path)
            .map_or(0, |at| at as u32 + 1)
    };
    let mut sorted = 0;
    let mut tiles = None;
    for (at, path) in paths.iter().enumerate() {
        sorted = sorted.max(geometry.sorted_bytes(most(at), path.pad)?);
        if path.takes("tile_counts") {
            let held = geometry.tile_bytes(path.tile_rows.unwrap_or(most(at)));
            tiles = Some(tiles.map_or(held, |known: u64| known.max(held)));
        }
    }
    info.bytes.push(("routes.sorted_rows".into(), sorted));
    if paths.iter().any(|path| path.takes("active_experts")) {
        info.bytes.push((
            "routes.active_experts".into(),
            (u64::from(geometry.local) + 1) * 4,
        ));
    }
    if let Some(tiles) = tiles {
        info.bytes.push(("tile_counts".into(), tiles));
        info.bytes.push(("tile_offsets".into(), tiles));
    }
    if paths.iter().any(|path| path.takes("ticket")) {
        info.bytes
            .push(("ticket".into(), PARALLEL_ROUTE_TICKET_BYTES));
    }
    need.check(&info)?;
    Ok(Box::new(Route {
        paths,
        rows,
        plans,
        geometry,
        info,
    }))
}

/// Runs of row counts, each its first row count and the index of the path
/// it takes, ascending from one.
type Runs = Vec<(u32, usize)>;

/// `path` as a [`choose`] candidate at `need`.
fn path_candidate(path: Path, need: Need) -> Candidate<Path> {
    let name = path
        .kernels
        .iter()
        .map(|kernel| kernel.symbol.as_str())
        .collect::<Vec<_>>()
        .join("+");
    let (capacity, guarantees) = (path.capacity, path.guarantees(need.capacity));
    let launches = path.launch();
    Candidate::new(path, name, launches, guarantees).capacity(capacity)
}

/// The chained paths (`chained`, whatever their pad) a chained route op of
/// `need.capacity` rows runs, and the first row count of each run of rows
/// with the index of its path: a step of `rows` rows takes the chained path
/// of the least capacity that holds them ([`choose`] at those rows). The
/// choice changes only where a path's capacity is passed, so it is made at
/// the first row count of each run between those bounds. The capacity's
/// rows must take `chosen`, the path of `route_tile`'s pad; with the claims
/// every run's registrations declare.
fn chained_runs(
    chained: &[Path],
    chosen: &Path,
    route_tile: u32,
    need: Need,
) -> Result<(Vec<Path>, Runs, Invariance), Refused> {
    let mut starts = chained
        .iter()
        .map(|path| path.capacity.saturating_add(1))
        .chain([1])
        .filter(|&rows| rows <= need.capacity)
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    let mut paths: Vec<Path> = Vec::new();
    let mut runs: Runs = Vec::new();
    let mut claims = Invariance::NONE;
    let mut first_run = true;
    for first in starts {
        let candidates = chained
            .iter()
            .cloned()
            .map(|path| path_candidate(path, need));
        let Chosen { entry, guarantees } = choose(
            "route",
            candidates,
            Need {
                capacity: first,
                ..need
            },
        )?;
        claims = if first_run {
            guarantees
        } else {
            claims.meet(guarantees)
        };
        first_run = false;
        let at = match paths
            .iter()
            .position(|known| known.launch() == entry.launch())
        {
            Some(at) => at,
            None => {
                paths.push(entry);
                paths.len() - 1
            }
        };
        if runs.last().is_none_or(|&(_, last)| last != at) {
            runs.push((first, at));
        }
    }
    let last = runs.last().map(|&(_, at)| &paths[at]);
    if last.is_none_or(|path| path.launch() != chosen.launch()) {
        return Err(Refused::Rule(format!(
            "SM120's chained routes pad {} rows to another tile than {route_tile}",
            need.capacity
        )));
    }
    Ok((paths, runs, claims))
}

struct Route {
    /// The paths the op runs.
    paths: Vec<Path>,
    /// The index in `paths` of the path each row count from one to the
    /// capacity runs.
    rows: Vec<u8>,
    /// The recorded rows and plan of each of its path's kernels at each row
    /// count from one to the capacity.
    plans: Vec<Vec<(u32, LaunchPlan)>>,
    geometry: Geometry,
    info: OpInfo,
}

impl PreparedOp<RouteBuffers<View>> for Route {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RouteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let at = counts
            .rows
            .checked_sub(1)
            .ok_or(Refusal("row count outside the prepared capacity"))? as usize;
        let (Some(plans), Some(&path)) = (self.plans.get(at), self.rows.get(at)) else {
            return Err(Refusal("row count outside the prepared capacity"));
        };
        let path = &self.paths[usize::from(path)];
        let rows = counts.rows;
        let g = &self.geometry;
        let local = u64::from(g.local);
        buffers.logits.covers(dense(rows, g.experts, 4))?;
        buffers.ids.covers(dense(rows, g.top_k, 4))?;
        buffers.weights.covers(dense(rows, g.top_k, 4))?;
        buffers.routes.offsets.covers((local + 1) * 4)?;
        buffers
            .routes
            .sorted_rows
            .covers(g.sorted_bytes(rows, path.pad)?)?;
        buffers.routes.padded_rows.covers(4)?;
        buffers.live_rows.covers(4)?;
        let required = |view: Option<View>, name: &str, bytes: u64| -> Result<(), Refusal> {
            if path.takes(name) {
                view.ok_or(Refusal("a route kernel reads a role the caller left out"))?
                    .covers(bytes)?;
            }
            Ok(())
        };
        required(buffers.bias, "bias", u64::from(g.experts) * 4)?;
        let tiles = g.tile_bytes(path.tile_rows.unwrap_or(rows));
        required(buffers.tile_counts, "tile_counts", tiles)?;
        required(buffers.tile_offsets, "tile_offsets", tiles)?;
        required(buffers.ticket, "ticket", PARALLEL_ROUTE_TICKET_BYTES)?;
        required(buffers.routes.counts, "counts", local * 4)?;
        required(
            buffers.routes.active_experts,
            "active_experts",
            (local + 1) * 4,
        )?;
        let bind = |name: &str| -> Result<Argument, Refusal> {
            let view = match name {
                "logits" => Some(buffers.logits),
                "bias" => buffers.bias,
                "ids" => Some(buffers.ids),
                "weights" => Some(buffers.weights),
                "counts" => buffers.routes.counts,
                "offsets" => Some(buffers.routes.offsets),
                "sorted_rows" => Some(buffers.routes.sorted_rows),
                "num_padded_rows" | "padded_rows" => Some(buffers.routes.padded_rows),
                "active_experts" => buffers.routes.active_experts,
                "live_tokens" | "live_rows" => Some(buffers.live_rows),
                "tile_counts" => buffers.tile_counts,
                "tile_offsets" => buffers.tile_offsets,
                "ticket" => buffers.ticket,
                "rank" => return Ok(Argument::U32(g.rank)),
                "scale_bits" => return Ok(Argument::U32(scale_bits(g.scale))),
                _ => return Err(Refusal("a route kernel takes a parameter no role binds")),
            };
            view.map(buffer)
                .ok_or(Refusal("a route kernel reads a role the caller left out"))
        };
        path.kernels
            .iter()
            .zip(plans)
            .map(|(kernel, &(rows, plan))| {
                Ok(Launch {
                    symbol: kernel.symbol.clone(),
                    rows,
                    plan,
                    arguments: kernel.arguments(&bind)?,
                    collective: false,
                    persistent: false,
                })
            })
            .collect()
    }
}

/// The `scale_bits` argument of a router kernel: the router's scaling
/// factor, or a neutral 1.0 for a router kind without one. The softmax and
/// grouped bodies ignore the argument (`row/router.cuh`), and a product by
/// 1.0 is exact, so every router kind binds the shared router ABI.
fn scale_bits(scale: Option<f32>) -> u32 {
    scale.unwrap_or(1.0).to_bits()
}

fn dense(rows: u32, elements: u32, bytes: u64) -> u64 {
    u64::from(rows) * u64::from(elements) * bytes
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// One kernel holding `capacity` rows as a [`choose`] candidate, launched
/// at every row count up to the prepared capacity.
fn one(kernel: Kernel, capacity: u32) -> Candidate<(Kernel, u32)> {
    let (name, guarantees) = (kernel.symbol.clone(), kernel.invariance);
    let launches = kernel.launch();
    Candidate::new((kernel, capacity), name, launches, guarantees).capacity(capacity)
}

struct Single {
    kernel: Kernel,
    /// The recorded rows and plan at each row count from one to the
    /// capacity.
    plans: Vec<(u32, LaunchPlan)>,
    info: OpInfo,
}

impl Single {
    /// `chosen`'s kernel at its capacity, claiming what the registrations
    /// of its launch declare.
    fn new(chosen: Chosen<(Kernel, u32)>, op: &str, need: Need) -> Result<Single, Refused> {
        let Chosen {
            entry: (kernel, capacity),
            guarantees,
        } = chosen;
        served(need)?;
        if need.capacity > capacity {
            return Err(Refused::Rule(
                "the capacity exceeds the entry's token capacity".into(),
            ));
        }
        let largest = kernel.plan(need.capacity)?;
        let mut plans = (1..need.capacity)
            .map(|rows| kernel.plan(rows))
            .collect::<Result<Vec<_>, Refusal>>()?;
        plans.push(largest);
        let info = OpInfo::new(
            implementation(op, format!("{kernel:?}")),
            guarantees,
            vec![(kernel.symbol.clone(), largest.1.dynamic_shared_bytes)],
        );
        need.check(&info)?;
        Ok(Single {
            kernel,
            plans,
            info,
        })
    }

    fn launch(
        &self,
        counts: Counts,
        bind: &dyn Fn(&str) -> Result<Argument, Refusal>,
    ) -> Result<Vec<Launch>, Refusal> {
        let &(rows, plan) = counts
            .rows
            .checked_sub(1)
            .and_then(|index| self.plans.get(index as usize))
            .ok_or(Refusal("row count outside the prepared capacity"))?;
        Ok(vec![Launch {
            symbol: self.kernel.symbol.clone(),
            rows,
            plan,
            arguments: self.kernel.arguments(bind)?,
            collective: false,
            persistent: false,
        }])
    }
}

/// A router alone through the model's router entry: a route cubin's or a
/// row engine's.
pub(super) fn router(
    selection: &Selection,
    op: &RowQuery,
    need: Need,
) -> Result<Prepared<RouterBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::Router {
        experts,
        top_k,
        kind,
    } = *op
    else {
        return Err(Refused::Rule("expected a router".into()));
    };
    let found = selection
        .entries
        .iter()
        .filter(|entry| entry.contract == RouteContract::Router(*op))
        .map(|entry| (Kernel::entry(entry), entry.tokens_capacity))
        .chain(
            selection
                .rows
                .iter()
                .filter(|entry| entry.query == *op)
                .map(|entry| (Kernel::row(entry, None), ROW_CAPACITY)),
        )
        .map(|(kernel, capacity)| one(kernel, capacity))
        .collect::<Vec<_>>();
    let chosen = choose("router", found, need)?;
    let scale = match kind {
        RouterKind::SigmoidBiasSelectNormalized { scaling_factor } => Some(scaling_factor),
        _ => None,
    };
    Ok(Box::new(Router {
        single: Single::new(chosen, "router", need)?,
        experts,
        top_k,
        scale,
    }))
}

struct Router {
    single: Single,
    experts: u32,
    top_k: u32,
    scale: Option<f32>,
}

impl PreparedOp<RouterBuffers<View>> for Router {
    fn info(&self) -> &OpInfo {
        &self.single.info
    }

    fn launches(
        &self,
        buffers: &RouterBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        buffers.logits.covers(dense(rows, self.experts, 4))?;
        buffers.ids.covers(dense(rows, self.top_k, 4))?;
        buffers.weights.covers(dense(rows, self.top_k, 4))?;
        buffers.live_rows.covers(4)?;
        if self.single.kernel.takes("bias") {
            buffers
                .bias
                .ok_or(Refusal("the router reads a bias the caller left out"))?
                .covers(u64::from(self.experts) * 4)?;
        }
        let bind = |name: &str| -> Result<Argument, Refusal> {
            let view = match name {
                "logits" => Some(buffers.logits),
                "bias" => buffers.bias,
                "ids" => Some(buffers.ids),
                "weights" => Some(buffers.weights),
                "live_tokens" | "live_rows" => Some(buffers.live_rows),
                "scale_bits" => return Ok(Argument::U32(scale_bits(self.scale))),
                _ => return Err(Refusal("a router takes a parameter no role binds")),
            };
            view.map(buffer)
                .ok_or(Refusal("the router reads a role the caller left out"))
        };
        self.single.launch(counts, &bind)
    }
}

/// Rank `rank`'s shared-expert gate combine through the model's combine
/// entry over the routes of `routes`.
pub(super) fn shared_expert_combine(
    selection: &Selection,
    op: &MoeQuery,
    routes: &MoeQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<SharedCombineBuffers<View>>, Refused> {
    op.validate()?;
    routes.validate()?;
    let MoeQuery::SharedExpertGateCombine { width, capacity } = *op else {
        return Err(Refused::Rule(
            "expected a shared-expert gate combine".into(),
        ));
    };
    let MoeQuery::LocalRoutes {
        experts,
        local_experts,
        top_k,
        ..
    } = *routes
    else {
        return Err(Refused::Rule("expected local routes".into()));
    };
    if rank >= experts / local_experts {
        return Err(Refused::Rule("the rank is past the expert shards".into()));
    }
    if need.capacity > capacity {
        return Err(Refused::Rule(
            "the capacity exceeds the combine's capacity".into(),
        ));
    }
    let found = selection
        .entries
        .iter()
        .filter(|entry| {
            entry.experts == experts
                && entry.local_experts == local_experts
                && entry.top_k == top_k
                && matches!(
                    entry.contract,
                    RouteContract::Moe(MoeQuery::SharedExpertGateCombine { width: w, .. }) if w == width
                )
        })
        .map(|entry| one(Kernel::entry(entry), entry.tokens_capacity))
        .collect::<Vec<_>>();
    let chosen = choose("shared-expert combine", found, need)?;
    Ok(Box::new(SharedCombine {
        single: Single::new(chosen, "shared-combine", need)?,
        width,
        top_k,
        rank,
    }))
}

struct SharedCombine {
    single: Single,
    width: u32,
    top_k: u32,
    rank: u32,
}

impl PreparedOp<SharedCombineBuffers<View>> for SharedCombine {
    fn info(&self) -> &OpInfo {
        &self.single.info
    }

    fn launches(
        &self,
        buffers: &SharedCombineBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let routes = rows
            .checked_mul(self.top_k)
            .ok_or(Refusal("route rows overflow"))?;
        buffers.routed.covers(dense(routes, self.width, 2))?;
        buffers.ids.covers(dense(routes, 1, 4))?;
        buffers.shared.covers(dense(rows, self.width, 2))?;
        buffers.x.covers(dense(rows, self.width, 2))?;
        buffers.gate_weight.covers(u64::from(self.width) * 2)?;
        buffers.output.covers(dense(rows, self.width, 2))?;
        buffers.live_rows.covers(4)?;
        let bind = |name: &str| -> Result<Argument, Refusal> {
            Ok(match name {
                "routed" => buffer(buffers.routed),
                "ids" => buffer(buffers.ids),
                "shared_out" | "shared" => buffer(buffers.shared),
                "x" => buffer(buffers.x),
                "gate_weight" => buffer(buffers.gate_weight),
                "output" => buffer(buffers.output),
                "live_tokens" | "live_rows" => buffer(buffers.live_rows),
                "rank" => Argument::U32(self.rank),
                _ => return Err(Refusal("a combine takes a parameter no role binds")),
            })
        };
        self.single.launch(counts, &bind)
    }
}

/// The combine of each token's routed rows through the model's row engine
/// combine entry.
pub(super) fn expert_combine(
    selection: &Selection,
    op: &ExpertCombineOp,
    need: Need,
) -> Result<Prepared<CombineBuffers<View>>, Refused> {
    op.validate()?;
    let found = selection
        .rows
        .iter()
        .filter(|entry| combines(entry, op))
        .map(|entry| one(Kernel::row(entry, None), ROW_CAPACITY))
        .collect::<Vec<_>>();
    let chosen = choose("combine", found, need)?;
    Ok(Box::new(ExpertCombine {
        single: Single::new(chosen, "combine", need)?,
        top_k: op.top_k,
        width: op.width,
        weighted: op.route_weight.is_none(),
    }))
}

/// Whether `entry` is the row engine combine of `op`: the catalog record
/// `op.query()` and the entry `row::builders::expert_combine_entry` builds
/// for the op's width and weighting (rows weighted before the combine, at
/// any point, are added as they are).
fn combines(entry: &RowInstantiation, op: &ExpertCombineOp) -> bool {
    let Ok(built) = expert_combine_entry(
        entry.profile,
        "",
        op.width,
        op.experts,
        op.route_weight.is_some(),
    ) else {
        return false;
    };
    entry.query == op.query()
        && entry.width == built.width
        && entry.grid == built.grid
        && entry.threads == built.threads
        && entry.params == built.params
        && entry.body == built.body
}

struct ExpertCombine {
    single: Single,
    top_k: u32,
    width: u32,
    /// Whether the combine applies the route weights.
    weighted: bool,
}

impl PreparedOp<CombineBuffers<View>> for ExpertCombine {
    fn info(&self) -> &OpInfo {
        &self.single.info
    }

    fn launches(
        &self,
        buffers: &CombineBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let routes = rows
            .checked_mul(self.top_k)
            .ok_or(Refusal("route rows overflow"))?;
        buffers.routed.covers(dense(routes, self.width, 2))?;
        buffers.output.covers(dense(rows, self.width, 2))?;
        buffers.live_rows.covers(4)?;
        match buffers.weights {
            Some(weights) => weights.covers(dense(routes, 1, 4))?,
            None if self.weighted => {
                return Err(Refusal("the combine applies weights the caller left out"));
            }
            None => {}
        }
        let bind = |name: &str| -> Result<Argument, Refusal> {
            Ok(match name {
                "routed" => buffer(buffers.routed),
                // A combine of weighted rows takes no weights.
                "weights" => buffers.weights.map_or(Argument::NullDevicePointer, buffer),
                "output" => buffer(buffers.output),
                "live_rows" | "live_tokens" => buffer(buffers.live_rows),
                _ => return Err(Refusal("a combine takes a parameter no role binds")),
            })
        };
        self.single.launch(counts, &bind)
    }
}

/// Rank `rank`'s routed sum through the model's chained routed sum, and
/// through its sum that adds a row when the model registered one for an
/// F32 sum of the op's geometry (`Sm120::with_routed_sum_add`): a step that
/// binds an addend launches that one.
pub(super) fn routed_sum(
    selection: &Selection,
    op: &MoeQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<RoutedSumBuffers<View>>, Refused> {
    op.validate()?;
    let MoeQuery::LocalRoutedSum {
        experts,
        local_experts,
        top_k,
        width,
        output,
        capacity,
    } = *op
    else {
        return Err(Refused::Rule("expected a local routed sum".into()));
    };
    if rank >= experts / local_experts {
        return Err(Refused::Rule("the rank is past the expert shards".into()));
    }
    if need.capacity > capacity {
        return Err(Refused::Rule(
            "the capacity exceeds the routed sum's capacity".into(),
        ));
    }
    let form = match output {
        DType::Bf16 => RoutedSumOutput::Bf16,
        _ => RoutedSumOutput::F32,
    };
    let of_op = |sum: &RoutedSum| {
        sum.output == form
            && sum.top_k == top_k
            && sum.width == width
            && sum.local_experts == local_experts
    };
    let sum = registered("routed sum", &selection.sums, of_op, Kernel::sum, need)?
        .ok_or_else(|| Refused::Rule("no SM120 routed sum entry serves the op".into()))?;
    let Source::Sum(entry) = sum.kernel.source else {
        return Err(Refused::Rule("expected a routed sum".into()));
    };
    let add = registered(
        "routed sum add",
        &selection.sum_adds,
        of_op,
        Kernel::sum_add,
        need,
    )?;
    let mut info = sum.info.clone();
    if let Some(add) = &add {
        info.implementation =
            implementation("routed-sum", format!("{:?}", (&sum.kernel, &add.kernel)));
        info.guarantees = info.guarantees.meet(add.info.guarantees);
        info.symbols.extend(add.info.symbols.iter().cloned());
        need.check(&info)?;
    }
    Ok(Box::new(LocalRoutedSum {
        single: sum,
        add,
        info,
        top_k,
        width,
        output_bytes: entry.output_row_bytes(),
        rank,
    }))
}

/// The one of `registered`'s sums (each with its cubin prefix) that
/// `matches` the op and serves it at `need`, as `kernel` builds it, or
/// `None` when none matches. A registration its builder refuses never
/// serves, so it refuses the op only when no other sum matches.
fn registered(
    family: &str,
    registered: &[(RoutedSum, String)],
    matches: impl Fn(&RoutedSum) -> bool,
    kernel: fn(RoutedSum, &str) -> Result<Kernel, Refusal>,
    need: Need,
) -> Result<Option<Single>, Refused> {
    let (found, refused): (Vec<_>, Vec<_>) = registered
        .iter()
        .filter(|(sum, _)| matches(sum))
        .map(|(sum, prefix)| kernel(*sum, prefix).map(|kernel| one(kernel, sum.capacity)))
        .partition(Result::is_ok);
    if found.is_empty() {
        return match refused
            .into_iter()
            .filter_map(Result::err)
            .min_by_key(|refusal| refusal.0)
        {
            Some(refusal) => Err(refusal.into()),
            None => Ok(None),
        };
    }
    let found = found.into_iter().map(Result::unwrap).collect::<Vec<_>>();
    let chosen = choose(family, found, need)?;
    Ok(Some(Single::new(chosen, "routed-sum", need)?))
}

struct LocalRoutedSum {
    single: Single,
    /// The sum that adds a row, launched at a step that binds an addend.
    add: Option<Single>,
    info: OpInfo,
    top_k: u32,
    width: u32,
    output_bytes: u64,
    rank: u32,
}

impl PreparedOp<RoutedSumBuffers<View>> for LocalRoutedSum {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RoutedSumBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        let routes = rows
            .checked_mul(self.top_k)
            .ok_or(Refusal("route rows overflow"))?;
        buffers.routed.covers(dense(routes, self.width, 4))?;
        buffers.ids.covers(dense(routes, 1, 4))?;
        buffers.output.covers(u64::from(rows) * self.output_bytes)?;
        buffers.live_rows.covers(4)?;
        let kernel = match buffers.addend {
            Some(addend) => {
                addend.covers(dense(rows, self.width, 4))?;
                self.add
                    .as_ref()
                    .ok_or(Refusal("no SM120 routed sum of the op adds a row"))?
            }
            None => &self.single,
        };
        let bind = |name: &str| -> Result<Argument, Refusal> {
            Ok(match name {
                "routed" => buffer(buffers.routed),
                "ids" => buffer(buffers.ids),
                "addend" => buffer(
                    buffers
                        .addend
                        .ok_or(Refusal("a routed sum add reads an addend"))?,
                ),
                "output" => buffer(buffers.output),
                "live_tokens" | "live_rows" => buffer(buffers.live_rows),
                "rank" => Argument::U32(self.rank),
                _ => return Err(Refusal("a routed sum takes a parameter no role binds")),
            })
        };
        kernel.launch(counts, &bind)
    }
}

/// A model's router GEMV of one token bucket with the sigmoid-bias route of
/// 256 experts, top 8, fused into its tail: the entry
/// `{prefix}_t{tokens}_v1` of `moe::sigmoid_route::fused_router_cubin_spec`,
/// and the claims the model declares for it. Its GEMV entry's query is the
/// projection it computes, the query's row capacity its token bucket, and
/// its key the plan it launches with.
#[derive(Clone, Debug)]
pub struct FusedRouterEntry {
    /// The fused entry's symbol, `{prefix}_t{tokens}_v1`.
    pub symbol: String,
    /// The router GEMV entry of the same token bucket
    /// (`DenseKernel::FragmentRouter`).
    pub gemm: GemmVariant,
    /// What the model declares every launch of the fused entry keeps.
    pub invariance: Invariance,
}

impl FusedRouterEntry {
    /// The projection the entry computes.
    fn query(&self) -> Option<GemmQuery> {
        match self.gemm.key.query {
            StaticQuery::Gemm(query) => Some(query),
            _ => None,
        }
    }

    /// The entry's token bucket: its GEMV's row capacity.
    fn tokens(&self) -> Result<u32, Refusal> {
        let query = self
            .query()
            .ok_or(Refusal("a fused router's GEMV entry is no GEMM"))?;
        if !matches!(
            self.gemm.key.dense,
            Some(DenseKernel::FragmentRouter { .. })
        ) {
            return Err(Refusal("a fused router plans as the fragment router GEMV"));
        }
        Ok(query.m_capacity)
    }

    /// Whether the entry computes `projection`, whose row capacity only
    /// bounds the rows.
    fn projects(&self, projection: &GemmQuery) -> bool {
        self.query().is_some_and(|query| {
            GemmQuery {
                m_capacity: query.m_capacity,
                ..*projection
            } == query
        })
    }
}

impl Kernel {
    /// A fused router, whose symbol names its GEMV's token bucket in the
    /// fused cubin. It writes route metadata, so its claims are about the
    /// routed operation ([`Kernel::claims`]).
    fn fused(entry: &FusedRouterEntry) -> Result<Kernel, Refusal> {
        let tokens = entry.tokens()?;
        let prefix = entry
            .symbol
            .strip_suffix(&format!("_t{tokens}_v1"))
            .ok_or(Refusal(
                "a fused router's symbol names another token bucket than its GEMV",
            ))?;
        let params = fused_router_cubin_spec(prefix)
            .entries
            .into_iter()
            .find(|spec| spec.symbol == entry.symbol)
            .ok_or(Refusal(
                "the fused router cubin compiles no such token bucket",
            ))?
            .params;
        Ok(Kernel {
            symbol: entry.symbol.clone(),
            params,
            invariance: entry.invariance,
            source: Source::Fused(entry.clone()),
        })
    }
}

/// A fused router launch as the model's builder plans and records it: its
/// GEMV entry at the token bucket's rows with the entry's K splits, whatever
/// rows are live (the kernel reads the live count).
fn fused_plan(entry: &FusedRouterEntry, rows: u32) -> Result<(u32, LaunchPlan), Refusal> {
    let tokens = entry.tokens()?;
    if rows > tokens {
        return Err(Refusal(
            "the fused router's token bucket does not hold the rows",
        ));
    }
    let key = &entry.gemm.key;
    let plan = gemm::plan(
        key,
        RuntimeCounts {
            m: tokens,
            batch: tokens,
            context: 0,
            routed_rows: 0,
            split_k: gemm::choose_split_k(key)?,
        },
    )?;
    Ok((tokens, plan))
}

/// What a fused router launches beyond its symbol and parameters: the GEMV
/// key it plans with, which holds its token bucket.
fn fused_launch(entry: &FusedRouterEntry) -> String {
    let FusedRouterEntry {
        symbol: _,     // the kernel's
        gemm,          // below
        invariance: _, // claims
    } = entry;
    let GemmVariant {
        profile: _,    // filters catalogs; no launch reads it
        instantiation, // below
        invariance: _, // the GEMV entry's claims, not the fused entry's
    } = gemm;
    let InstantiationSpec {
        family: _, // a GEMM's; the plan reads the key alone
        key,       // the plan and the token bucket
        symbol: _, // the GEMV entry's, which never launches here
    } = instantiation;
    format!("fused {key:?}")
}

/// The arrival words a fused router of `tokens` takes
/// (`gemm/fragment_router.cuh`): a bucket of at most eight tokens reduces
/// in the grid's last CTA on word 0; a wider one reduces each 16-expert
/// strip on its own word and then publishes the logits on word 16.
fn fused_semaphore_bytes(tokens: u32) -> u64 {
    if tokens <= 8 {
        4
    } else {
        (SIGMOID_EXPERTS / 16 + 1) as u64 * 4
    }
}

/// The layout of the router weight the GEMV entry reads, named as the
/// SM120 precision capabilities name a dense entry's.
fn router_layout(gemm: &GemmVariant) -> WeightLayout {
    if gemm::fragment_weight_enabled(&gemm.key) {
        WeightLayout {
            id: "bf16-fragment".into(),
            version: 1,
        }
    } else {
        WeightLayout::row_major()
    }
}

/// Rank `rank`'s routes of the logits `projection` gives, through the
/// model's fused router entries ([`FusedRouterEntry`]): the router GEMV of a
/// token bucket with the sigmoid-bias route of 256 experts, top 8, every
/// expert local, in its tail, which pads each expert's routes to 16. Of the
/// entries that compute the projection, [`choose`] takes the least bucket
/// that holds the use's capacity and claims `need.exact`; the op claims what
/// the entry declares, for the routed operation ([`Kernel::claims`]), so a
/// model states `need.exact` in that scope. A registration whose symbol,
/// GEMV or plan the fused cubin does not take serves nothing: it is dropped
/// before the choice, and refuses the op only when no other entry is left.
#[allow(clippy::too_many_arguments)]
pub(super) fn projected_route(
    selection: &Selection,
    projection: &GemmQuery,
    router: &RowQuery,
    routes: &MoeQuery,
    rank: u32,
    route_tile: u32,
    usage: OperationUse,
    need: Need,
) -> Result<Prepared<ProjectedRouteBuffers<View>>, Refused> {
    served(need)?;
    if need.capacity != usage.capacity {
        return Err(Refused::Rule(
            "a route op's capacity is its use's capacity".into(),
        ));
    }
    let geometry = Geometry::of(router, routes, rank)?;
    if let MoeQuery::LocalRoutes { capacity, .. } = *routes
        && need.capacity > capacity
    {
        return Err(Refused::Rule(
            "the capacity exceeds the local routes' capacity".into(),
        ));
    }
    projection.validate()?;
    if projection.n != geometry.experts
        || projection.output != DType::F32
        || projection.epilogues != Default::default()
        || projection.grouped.is_some()
    {
        return Err(Refused::Rule(
            "a router projection writes the F32 logits of every expert".into(),
        ));
    }
    if need.capacity > projection.m_capacity {
        return Err(Refused::Rule(
            "the capacity exceeds the router projection's rows".into(),
        ));
    }
    // The tail selects as the one-CTA sigmoid-bias decode route does.
    let tail = geometry.scale.is_some()
        && geometry.local == geometry.experts
        && geometry.experts == SIGMOID_EXPERTS
        && geometry.top_k == SIGMOID_TOP_K;
    let matched = selection
        .fused
        .iter()
        .filter(|entry| tail && entry.projects(projection))
        .collect::<Vec<_>>();
    if !matched.is_empty() && route_tile != SIGMOID_PAD {
        return Err(Refused::Rule(
            "no SM120 route entries pad to the route tile".into(),
        ));
    }
    let (found, refused): (Vec<_>, Vec<_>) = matched
        .into_iter()
        .map(|entry| {
            let kernel = Kernel::fused(entry)?;
            // A registration that does not plan at its own bucket is not
            // the op's, so it ties with nothing.
            kernel.plan(entry.tokens()?)?;
            let path = Path {
                capacity: entry.tokens()?,
                kernels: vec![kernel],
                pad: SIGMOID_PAD,
                tile_rows: None,
            };
            let (name, launches) = (entry.symbol.clone(), path.launch());
            let (capacity, guarantees) = (path.capacity, path.guarantees(need.capacity));
            Ok(Candidate::new(path, name, launches, guarantees).capacity(capacity))
        })
        .partition(Result::is_ok);
    if found.is_empty()
        && let Some(refusal) = refused
            .into_iter()
            .filter_map(Result::<_, Refusal>::err)
            .min_by_key(|refusal| refusal.0)
    {
        return Err(refusal.into());
    }
    let found = found.into_iter().map(Result::unwrap).collect::<Vec<_>>();
    let Chosen {
        entry: path,
        guarantees: claimed,
    } = choose("projected route", found, need)?;
    let kernel = path.kernels[0].clone();
    let Source::Fused(entry) = &kernel.source else {
        return Err(Refused::Rule("expected a fused router".into()));
    };
    let planned = kernel.plan(need.capacity)?;
    // The op claims no more than every registration of its launch declares.
    let mut guarantees = claimed;
    if let Rows::OneRow { max } = guarantees.rows {
        guarantees.rows = Rows::OneRow {
            max: max.min(need.capacity),
        };
    }
    let mut info = OpInfo::new(
        implementation("projected-route", format!("{path:?}")),
        guarantees,
        vec![(kernel.symbol.clone(), planned.1.dynamic_shared_bytes)],
    );
    let partials = planned.1.workspace_bytes;
    let semaphores = fused_semaphore_bytes(planned.0);
    info.bytes = vec![
        (
            "route.routes.sorted_rows".into(),
            geometry.sorted_bytes(need.capacity, path.pad)?,
        ),
        (
            "route.routes.active_experts".into(),
            (u64::from(geometry.local) + 1) * 4,
        ),
        ("partials".into(), partials),
        ("semaphores".into(), semaphores),
    ];
    info.weight_layouts = vec![("weight".into(), router_layout(&entry.gemm))];
    need.check(&info)?;
    Ok(Box::new(ProjectedRoute {
        planned,
        capacity: need.capacity,
        k: projection.k,
        partials,
        semaphores,
        geometry,
        info,
        kernel,
    }))
}

struct ProjectedRoute {
    kernel: Kernel,
    /// The rows the launch records and its plan, the same at every row
    /// count up to the capacity.
    planned: (u32, LaunchPlan),
    capacity: u32,
    /// The projection's K: the hidden rows' and the weight rows' elements.
    k: u32,
    partials: u64,
    semaphores: u64,
    geometry: Geometry,
    info: OpInfo,
}

impl PreparedOp<ProjectedRouteBuffers<View>> for ProjectedRoute {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &ProjectedRouteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("row count outside the prepared capacity"));
        }
        let g = &self.geometry;
        let route = &buffers.route;
        let local = u64::from(g.local);
        buffers.x.covers(dense(rows, self.k, 2))?;
        buffers.weight.covers(dense(g.experts, self.k, 2))?;
        buffers.partials.covers(self.partials)?;
        buffers.semaphores.covers(self.semaphores)?;
        route.logits.covers(dense(rows, g.experts, 4))?;
        route.ids.covers(dense(rows, g.top_k, 4))?;
        route.weights.covers(dense(rows, g.top_k, 4))?;
        route.routes.offsets.covers((local + 1) * 4)?;
        route
            .routes
            .sorted_rows
            .covers(g.sorted_bytes(rows, SIGMOID_PAD)?)?;
        route.routes.padded_rows.covers(4)?;
        route.live_rows.covers(4)?;
        let required = |view: Option<View>, bytes: u64| -> Result<View, Refusal> {
            let view = view.ok_or(Refusal("a route kernel reads a role the caller left out"))?;
            view.covers(bytes)?;
            Ok(view)
        };
        let bias = required(route.bias, u64::from(g.experts) * 4)?;
        let expert_counts = required(route.routes.counts, local * 4)?;
        let active = required(route.routes.active_experts, (local + 1) * 4)?;
        let bind = |name: &str| -> Result<Argument, Refusal> {
            let view = match name {
                "weight" => buffers.weight,
                "activation" => buffers.x,
                "logits" => route.logits,
                "partials" => buffers.partials,
                "semaphores" => buffers.semaphores,
                "live_tokens" => route.live_rows,
                "bias" => bias,
                "topk_ids" => route.ids,
                "topk_weights" => route.weights,
                "counts" => expert_counts,
                "offsets" => route.routes.offsets,
                "sorted_rows" => route.routes.sorted_rows,
                "num_padded_rows" => route.routes.padded_rows,
                "active_experts" => active,
                "scale_bits" => {
                    let scale = g.scale.ok_or(Refusal("the router scales no weights"))?;
                    return Ok(Argument::U32(scale.to_bits()));
                }
                _ => return Err(Refusal("a route kernel takes a parameter no role binds")),
            };
            Ok(buffer(view))
        };
        let (planned, plan) = self.planned;
        Ok(vec![Launch {
            symbol: self.kernel.symbol.clone(),
            rows: planned,
            plan,
            arguments: self.kernel.arguments(&bind)?,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;
    use crate::backend::{Sm120, view};
    use crate::moe::chained::{PREFIX, TP4_288_TOP8};
    use crate::moe::expert_parallel::rank_arg;
    use crate::moe::parallel::ParallelRoute;
    use crate::moe::ptr;
    use crate::moe::sigmoid_route::{decode_route_arguments, decode_route_plan};
    use crate::row::builders::router_entry;
    use kernel_core::backend::Backend;
    use kernel_core::moe::RouteMetadata;
    use kernel_core::properties::Chunk;

    const PROFILE: Profile = Profile::StdDenseMoe;

    /// The decode use of `need`'s capacity: phase and graph choose nothing
    /// among today's route entries.
    fn use_of(need: Need) -> OperationUse {
        OperationUse {
            phase: kernel_core::precision::ExecutionPhase::Decode,
            capacity: need.capacity,
            graph: true,
        }
    }
    /// A softmax top-10 router over 512 experts, 128 per rank.
    const EXPERTS: u32 = 512;
    const LOCAL: u32 = 128;
    const TOP_K: u32 = 10;
    const HIDDEN: u32 = 2560;
    const PREFILL: u32 = 8192;
    const DECODE: [u32; 6] = [1, 2, 4, 8, 16, 32];

    fn softmax() -> RowQuery {
        RowQuery::Router {
            experts: EXPERTS,
            top_k: TOP_K,
            kind: RouterKind::SoftmaxTopK { renormalize: true },
        }
    }

    fn local(capacity: u32) -> MoeQuery {
        MoeQuery::LocalRoutes {
            experts: EXPERTS,
            local_experts: LOCAL,
            top_k: TOP_K,
            capacity,
        }
    }

    fn entry(
        symbol: &str,
        contract: RouteContract,
        tokens_capacity: u32,
        threads: u32,
        params: Vec<EntryParam>,
        invariance: Invariance,
    ) -> RouteEntry {
        RouteEntry {
            profile: PROFILE,
            experts: EXPERTS,
            local_experts: LOCAL,
            top_k: TOP_K,
            symbol: format!("chop_moe_test_{symbol}"),
            contract,
            tokens_capacity,
            threads,
            params,
            body: format!("{symbol}();"),
            parallel: None,
            invariance,
        }
    }

    fn decode_params() -> Vec<EntryParam> {
        vec![
            ptr("logits", "const float"),
            ptr("ids", "int32_t"),
            ptr("weights", "float"),
            ptr("counts", "int32_t"),
            ptr("offsets", "int32_t"),
            ptr("sorted_rows", "int32_t"),
            ptr("num_padded_rows", "int32_t"),
            ptr("live_tokens", "const int32_t"),
            ptr("active_experts", "int32_t"),
            rank_arg(),
        ]
    }

    /// A route cubin as a model with expert parallelism declares it: the
    /// router, decode routes, prefill permutation and shared-expert combine.
    fn route_cubin() -> Vec<RouteEntry> {
        let mut entries = vec![entry(
            "router_v1",
            RouteContract::Router(softmax()),
            PREFILL,
            256,
            vec![
                ptr("logits", "const float"),
                ptr("ids", "int32_t"),
                ptr("weights", "float"),
                ptr("live_tokens", "const int32_t"),
            ],
            Invariance::PREFILL.with_ties(Ties::LowerIndex),
        )];
        for tokens in DECODE {
            entries.push(entry(
                &format!("decode_route_t{tokens}_v1"),
                RouteContract::Moe(local(tokens)),
                tokens,
                (tokens * 32).max(LOCAL),
                decode_params(),
                if tokens <= 4 {
                    Invariance::one_row(tokens)
                } else {
                    Invariance::NONE
                },
            ));
        }
        let permute = |name: &str, threads, params| {
            entry(
                &format!("permute_{name}_v1"),
                RouteContract::Moe(local(PREFILL)),
                PREFILL,
                threads,
                params,
                Invariance::PREFILL.of_routed_operation(),
            )
        };
        entries.extend([
            permute(
                "count",
                ROUTE_TILE,
                vec![
                    ptr("ids", "const int32_t"),
                    ptr("tile_counts", "int32_t"),
                    ptr("live_tokens", "const int32_t"),
                    rank_arg(),
                ],
            ),
            permute(
                "prefix",
                512,
                vec![
                    ptr("tile_counts", "const int32_t"),
                    ptr("counts", "int32_t"),
                    ptr("tile_offsets", "int32_t"),
                    ptr("live_tokens", "const int32_t"),
                ],
            ),
            permute(
                "scan",
                LOCAL,
                vec![
                    ptr("counts", "const int32_t"),
                    ptr("offsets", "int32_t"),
                    ptr("sorted_rows", "int32_t"),
                    ptr("num_padded_rows", "int32_t"),
                    ptr("active_experts", "int32_t"),
                ],
            ),
            permute(
                "scatter",
                ROUTE_TILE,
                vec![
                    ptr("ids", "const int32_t"),
                    ptr("tile_offsets", "const int32_t"),
                    ptr("offsets", "const int32_t"),
                    ptr("sorted_rows", "int32_t"),
                    ptr("live_tokens", "const int32_t"),
                    rank_arg(),
                ],
            ),
            entry(
                "combine_v1",
                RouteContract::Moe(MoeQuery::SharedExpertGateCombine {
                    width: HIDDEN,
                    capacity: PREFILL,
                }),
                PREFILL,
                HIDDEN / 8,
                vec![
                    ptr("routed", "const uint16_t"),
                    ptr("ids", "const int32_t"),
                    ptr("shared_out", "const uint16_t"),
                    ptr("x", "const uint16_t"),
                    ptr("gate_weight", "const uint16_t"),
                    ptr("output", "uint16_t"),
                    ptr("live_tokens", "const int32_t"),
                    rank_arg(),
                ],
                Invariance::any_rows(PREFILL),
            ),
        ]);
        entries
    }

    /// The parallel decode route of `tokens`, made from the plain one as a
    /// model's catalog makes it: a ticket before the rank.
    fn parallel(tokens: u32) -> RouteEntry {
        let mut entry = route_cubin()
            .into_iter()
            .find(|entry| {
                entry
                    .symbol
                    .ends_with(&format!("decode_route_t{tokens}_v1"))
            })
            .unwrap();
        let geometry = ParallelRoute {
            experts: EXPERTS,
            top_k: TOP_K,
            local_experts: LOCAL,
            capacity: tokens,
            padding: 16,
            groups: 1,
            take_groups: 1,
            scoring: RouteScoring::Softmax,
        };
        entry.symbol = format!("chop_moe_test_decode_route_parallel_t{tokens}_v1");
        entry.threads = geometry.plan(tokens).unwrap().block[0];
        entry.parallel = Some(geometry);
        let rank = entry.params.len() - 1;
        entry.params.insert(rank, ptr("ticket", "uint32_t"));
        entry.body = geometry.body("rank", "nullptr", "1.f").unwrap();
        entry
    }

    /// The test router's kind, which the cubin's decode routes compute.
    const SOFTMAX: RouterKind = RouterKind::SoftmaxTopK { renormalize: true };

    /// `entries` of a route cubin registered as a model registers them:
    /// decode routes with the router kind they compute, `SOFTMAX`.
    fn with_cubin(backend: Sm120, entries: impl IntoIterator<Item = RouteEntry>) -> Sm120 {
        entries.into_iter().fold(backend, |backend, entry| {
            if decode_route(&entry) {
                backend.with_decode_route(entry, SOFTMAX)
            } else {
                backend.with_route_entry(entry)
            }
        })
    }

    fn cubin_backend() -> Sm120 {
        with_cubin(Sm120::new(), route_cubin())
    }

    const BIG: u64 = 1 << 40;

    fn buffers() -> RouteBuffers<View> {
        RouteBuffers {
            logits: view(1, BIG),
            bias: Some(view(2, BIG)),
            ids: view(3, BIG),
            weights: view(4, BIG),
            routes: RouteMetadata {
                counts: Some(view(5, BIG)),
                offsets: view(6, BIG),
                sorted_rows: view(7, BIG),
                padded_rows: view(8, BIG),
                active_experts: Some(view(9, BIG)),
            },
            tile_counts: Some(view(10, BIG)),
            tile_offsets: Some(view(11, BIG)),
            ticket: Some(view(12, BIG)),
            live_rows: view(13, BIG),
        }
    }

    /// What `launch::route` in a model's builders gives: the entry planned
    /// and recorded at `capacity` (a decode route's bucket, otherwise the
    /// step's rows), its handles and the rank.
    fn built(entry: &RouteEntry, capacity: u32, handles: &[u64], rank: Option<u32>) -> Launch {
        Launch {
            symbol: entry.symbol.clone(),
            rows: capacity,
            plan: entry.plan(capacity, capacity).unwrap(),
            arguments: entry.arguments(handles, rank).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    fn find(entries: &[RouteEntry], suffix: &str) -> RouteEntry {
        entries
            .iter()
            .find(|entry| entry.symbol.ends_with(suffix))
            .unwrap()
            .clone()
    }

    #[test]
    fn decode_routes_equal_the_builders_at_each_bucket() {
        let entries = route_cubin();
        let rank = 2;
        let route = |backend: &Sm120, rows| {
            backend.moe_route(
                &softmax(),
                &local(32),
                rank,
                16,
                use_of(Need::rows(rows)),
                Need::rows(rows),
            )
        };
        // Each decode bucket's op takes the least decode route that holds
        // it, at every row count up to it.
        for bucket in DECODE {
            let op = route(&cubin_backend(), bucket).unwrap();
            let entry = find(&entries, &format!("decode_route_t{bucket}_v1"));
            assert_eq!(op.info().symbols, vec![(entry.symbol.clone(), 0)]);
            for rows in [1, bucket] {
                assert_eq!(
                    op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    vec![built(
                        &entry,
                        bucket,
                        &[1, 3, 4, 5, 6, 7, 8, 13, 9],
                        Some(rank)
                    )]
                );
            }
        }
        // A capacity between buckets takes the next bucket's route.
        let op = route(&cubin_backend(), 20).unwrap();
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(20))
                .unwrap()[0]
                .symbol,
            "chop_moe_test_decode_route_t32_v1"
        );
        // The plain and parallel routes of one capacity launch differently:
        // registering both is ambiguous, and the model registers the one
        // it serves.
        let both = cubin_backend().with_decode_route(parallel(32), SOFTMAX);
        assert!(matches!(
            route(&both, 32),
            Err(Refused::Rule(rule)) if rule.contains(
                "chop_moe_test_decode_route_parallel_t32_v1 and chop_moe_test_decode_route_t32_v1"
            )
        ));
        assert!(route(&both, 16).is_ok());
        let parallel_only = with_cubin(
            Sm120::new(),
            route_cubin()
                .into_iter()
                .filter(|entry| !entry.symbol.ends_with("decode_route_t32_v1"))
                .chain([parallel(32)]),
        );
        let op = route(&parallel_only, 32).unwrap();
        assert_eq!(op.info().bytes_of("ticket"), Some(4));
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(20))
                .unwrap(),
            vec![built(
                &parallel(32),
                32,
                &[1, 3, 4, 5, 6, 7, 8, 13, 9, 12],
                Some(rank)
            )]
        );
        let mut no_ticket = buffers();
        no_ticket.ticket = None;
        assert!(
            op.launches(&no_ticket, View::EMPTY, Counts::rows(20))
                .is_err()
        );
        // The use's capacity is the op's.
        assert!(matches!(
            cubin_backend().moe_route(
                &softmax(),
                &local(32),
                rank,
                16,
                use_of(Need::rows(16)),
                Need::rows(32),
            ),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn prefill_routes_equal_the_builders_and_keep_chunk_invariance() {
        let entries = route_cubin();
        let rank = 3;
        let need = Need::rows(PREFILL).with_exact(Invariance::PREFILL.of_routed_operation());
        let op = cubin_backend()
            .moe_route(&softmax(), &local(PREFILL), rank, 16, use_of(need), need)
            .unwrap();
        let tiles = u64::from(PREFILL * TOP_K).div_ceil(256) * u64::from(LOCAL) * 4;
        assert_eq!(op.info().bytes_of("tile_counts"), Some(tiles));
        assert_eq!(
            op.info().bytes_of("routes.sorted_rows"),
            Some(u64::from(padded_capacity(PREFILL * TOP_K, LOCAL, 16).unwrap()) * 4)
        );
        assert_eq!(op.info().guarantees.ties, Ties::LowerIndex);
        assert_eq!(op.info().guarantees.scope, Scope::RoutedOperation);
        // Prefill rows run the router and permutation at every row count,
        // as the builders plan them at the step's rows.
        for rows in [1, 7, 32, 33, 1000, PREFILL] {
            let at = |suffix: &str| find(&entries, suffix);
            assert_eq!(
                op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![
                    built(&at("router_v1"), rows, &[1, 3, 4, 13], None),
                    built(&at("permute_count_v1"), rows, &[3, 10, 13], Some(rank)),
                    built(&at("permute_prefix_v1"), rows, &[10, 5, 11, 13], None),
                    built(&at("permute_scan_v1"), rows, &[5, 6, 7, 8, 9], None),
                    built(
                        &at("permute_scatter_v1"),
                        rows,
                        &[3, 11, 6, 7, 13],
                        Some(rank)
                    ),
                ],
                "{rows} rows"
            );
        }
        // Graph buckets past the decode routes take the permutation at every
        // row count.
        let op = cubin_backend()
            .moe_route(
                &softmax(),
                &local(PREFILL),
                rank,
                16,
                use_of(Need::rows(64)),
                Need::rows(64),
            )
            .unwrap();
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(20))
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(64))
                .unwrap()
                .len(),
            5
        );
    }

    #[test]
    fn exact_verify_takes_one_row_decode_routes_or_is_refused() {
        let backend = cubin_backend();
        let exact = Need::rows(4).with_exact(Invariance::one_row(4).of_routed_operation());
        let op = backend
            .moe_route(&softmax(), &local(32), 0, 16, use_of(exact), exact)
            .unwrap();
        assert_eq!(
            op.info().guarantees,
            Invariance::one_row(4).of_routed_operation()
        );
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(3))
                .unwrap()[0]
                .symbol,
            "chop_moe_test_decode_route_t4_v1"
        );
        let exact = Need::rows(8).with_exact(Invariance::one_row(8).of_routed_operation());
        assert!(matches!(
            backend.moe_route(&softmax(), &local(32), 0, 16, use_of(exact), exact),
            Err(Refused::Inexact { .. })
        ));
    }

    /// A decode route writes route metadata, which depends on every route
    /// of the launch, so its claims are the routed operation's whatever
    /// scope the model declared: a one-row need in that scope is served and
    /// launches as the builder does, the same need about the entry's own
    /// outputs is refused. Route cubin decode routes and the sigmoid-bias
    /// decode route alike.
    #[test]
    fn decode_route_claims_are_about_the_routed_operation() {
        let cubin = cubin_backend();
        let t4 = find(&route_cubin(), "decode_route_t4_v1");
        assert_eq!(t4.invariance.scope, Scope::Entry);
        let routed = Need::rows(4).with_exact(t4.invariance.of_routed_operation());
        let op = cubin
            .moe_route(&softmax(), &local(32), 0, 16, use_of(routed), routed)
            .unwrap();
        assert_eq!(op.info().guarantees.scope, Scope::RoutedOperation);
        let handles = [1, 3, 4, 5, 6, 7, 8, 13, 9];
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(3))
                .unwrap(),
            vec![built(&t4, 4, &handles, Some(0))]
        );
        let entry = Need::rows(4).with_exact(t4.invariance);
        assert!(matches!(
            cubin.moe_route(&softmax(), &local(32), 0, 16, use_of(entry), entry),
            Err(Refused::Inexact { guaranteed, .. }) if guaranteed.scope == Scope::RoutedOperation
        ));

        let (router, routes) = sigmoid256();
        let declared = Invariance::one_row(16).with_ties(Ties::LowerIndex);
        let sigmoid = Sm120::new().with_sigmoid_decode_route(
            "chop_moe_test_decode_route_b16_v1",
            16,
            declared,
        );
        let routed = Need::rows(16).with_exact(declared.of_routed_operation());
        let op = sigmoid
            .moe_route(&router, &routes, 0, 16, use_of(routed), routed)
            .unwrap();
        assert_eq!(op.info().guarantees, declared.of_routed_operation());
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(9))
                .unwrap(),
            vec![Launch {
                symbol: "chop_moe_test_decode_route_b16_v1".into(),
                rows: 16,
                plan: decode_route_plan(16, 9).unwrap(),
                arguments: decode_route_arguments([1, 2, 3, 4, 5, 6, 7, 8, 13, 9], 1.0).unwrap(),
                collective: false,
                persistent: false,
            }]
        );
        for need in [declared, Invariance::one_row(4)] {
            let entry = Need::rows(16).with_exact(need);
            assert!(matches!(
                sigmoid.moe_route(&router, &routes, 0, 16, use_of(entry), entry),
                Err(Refused::Inexact { guaranteed, .. })
                    if guaranteed.scope == Scope::RoutedOperation
            ));
        }
    }

    #[test]
    fn chained_routes_equal_the_lowering_bindings() {
        let [decode, prefill] = TP4_288_TOP8.routes();
        let backend = Sm120::new()
            .with_local_route(decode, PREFIX)
            .with_local_route(prefill, PREFIX);
        let router = RowQuery::Router {
            experts: 288,
            top_k: 8,
            kind: RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.5,
            },
        };
        let routes = MoeQuery::LocalRoutes {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            capacity: 8192,
        };
        let rank = 1;
        // The decode route pads to the 16-row tile, the prefill route to
        // 64; each op takes the expert op's tile.
        let op = backend
            .moe_route(
                &router,
                &routes,
                rank,
                16,
                use_of(Need::rows(32)),
                Need::rows(32),
            )
            .unwrap();
        assert_eq!(
            op.info().guarantees,
            Invariance::any_rows(32)
                .with_ties(Ties::LowerIndex)
                .of_routed_operation()
        );
        assert_eq!(
            op.info().bytes_of("routes.sorted_rows"),
            Some(u64::from(decode.sorted_rows_capacity()) * 4)
        );
        assert!(
            backend
                .moe_route(
                    &router,
                    &routes,
                    rank,
                    16,
                    use_of(Need::rows(33)),
                    Need::rows(33)
                )
                .is_err()
        );
        let wide = backend
            .moe_route(
                &router,
                &routes,
                rank,
                64,
                use_of(Need::rows(8192)),
                Need::rows(8192),
            )
            .unwrap();
        assert_eq!(
            wide.info().bytes_of("routes.sorted_rows"),
            Some(u64::from(prefill.sorted_rows_capacity()) * 4)
        );
        assert!(matches!(
            backend.moe_route(&router, &routes, rank, 32, use_of(Need::rows(32)), Need::rows(32)),
            Err(Refused::Rule(rule)) if rule.contains("route tile")
        ));
        let (scale, rank) = (Argument::U32(2.5_f32.to_bits()), Argument::U32(rank));
        let b = |handle| Argument::DeviceBuffer(handle);
        let launch = |route: LocalRoute, kernel, rows, arguments: Vec<Argument>| Launch {
            symbol: route.symbol(PREFIX, kernel),
            rows,
            plan: route.plan(kernel, rows).unwrap(),
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        };
        let decoded = |rows| {
            vec![launch(
                decode,
                RouteKernel::Decode,
                rows,
                vec![
                    b(1),
                    b(2),
                    b(3),
                    b(4),
                    b(5),
                    b(6),
                    b(7),
                    b(8),
                    b(13),
                    b(9),
                    rank.clone(),
                    scale.clone(),
                ],
            )]
        };
        for rows in [1, 5, 32] {
            assert_eq!(
                op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                decoded(rows)
            );
            // A prefill of the decode route's rows takes it too, as the
            // 16-row expert forms it runs read them.
            assert_eq!(
                wide.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                decoded(rows)
            );
        }
        for rows in [33, 4096, 8192] {
            assert_eq!(
                wide.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![
                    launch(
                        prefill,
                        RouteKernel::Select,
                        rows,
                        vec![b(1), b(2), b(3), b(4), b(13), scale.clone()]
                    ),
                    launch(
                        prefill,
                        RouteKernel::Count,
                        rows,
                        vec![b(3), b(5), b(13), rank.clone()]
                    ),
                    launch(
                        prefill,
                        RouteKernel::Scatter,
                        rows,
                        vec![b(3), b(5), b(6), b(7), b(8), b(13), b(9), rank.clone()]
                    ),
                ]
            );
        }
        // Chained routes read the score bias.
        let mut unbiased = buffers();
        unbiased.bias = None;
        assert!(
            op.launches(&unbiased, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        // Chained routes score sigmoid plus bias only.
        let softmax = RowQuery::Router {
            experts: 288,
            top_k: 8,
            kind: RouterKind::SoftmaxTopK { renormalize: true },
        };
        assert!(matches!(
            backend.moe_route(
                &softmax,
                &routes,
                1,
                16,
                use_of(Need::rows(32)),
                Need::rows(32)
            ),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn routers_equal_their_builders() {
        let entries = route_cubin();
        let op = cubin_backend()
            .moe_router(&softmax(), Need::rows(PREFILL))
            .unwrap();
        let buffers = RouterBuffers {
            logits: view(1, BIG),
            bias: None,
            ids: view(3, BIG),
            weights: view(4, BIG),
            live_rows: view(13, BIG),
        };
        for rows in [1, 9, PREFILL] {
            assert_eq!(
                op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![built(
                    &find(&entries, "router_v1"),
                    rows,
                    &[1, 3, 4, 13],
                    None
                )]
            );
        }
        // A row engine sigmoid-bias router at the smallest bucket.
        let row = router_entry(PROFILE, "test_sigmoid256", 256, 1, false).unwrap();
        let query = row.query;
        let op = Sm120::new()
            .with_router(row.clone())
            .moe_router(&query, Need::rows(PREFILL))
            .unwrap();
        let biased = RouterBuffers {
            bias: Some(view(2, BIG)),
            ..buffers
        };
        for rows in [1, 3, 100, PREFILL] {
            let bucket = smallest_bucket(rows).unwrap();
            let b = |handle| Argument::DeviceBuffer(handle);
            assert_eq!(
                op.launches(&biased, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![Launch {
                    symbol: row.symbol.clone(),
                    rows: bucket,
                    plan: row.plan(bucket, row_counts(rows)).unwrap(),
                    arguments: row
                        .pack_arguments(vec![
                            b(1),
                            b(2),
                            b(3),
                            b(4),
                            b(13),
                            Argument::U32(1.0_f32.to_bits())
                        ])
                        .unwrap(),
                    collective: false,
                    persistent: false,
                }]
            );
        }
        assert!(op.launches(&buffers, View::EMPTY, Counts::rows(4)).is_err());
    }

    /// Row-engine routers without a scale (softmax and grouped) share the
    /// router ABI's `scale_bits`, which their bodies ignore: they prepare
    /// alone and through the tiled permutation, binding a neutral 1.0.
    #[test]
    fn unscaled_row_routers_bind_a_neutral_scale() {
        let one = Argument::U32(1.0_f32.to_bits());
        let buffers = RouterBuffers {
            logits: view(1, BIG),
            bias: Some(view(2, BIG)),
            ids: view(3, BIG),
            weights: view(4, BIG),
            live_rows: view(13, BIG),
        };
        for (kind, grouped) in [(0, false), (2, true)] {
            let row =
                router_entry(PROFILE, &format!("test_unscaled{kind}"), 256, kind, grouped).unwrap();
            let query = row.query;
            let op = Sm120::new()
                .with_router(row.clone())
                .moe_router(&query, Need::rows(16))
                .unwrap();
            let launch = &op.launches(&buffers, View::EMPTY, Counts::rows(3)).unwrap()[0];
            assert_eq!(launch.symbol, row.symbol);
            assert_eq!(launch.arguments.0.last(), Some(&one));
            // Through the tiled permutation of every expert's routes.
            let (_, _, permute) = tiled_backend(16);
            let routes = MoeQuery::LocalRoutes {
                experts: 256,
                local_experts: 256,
                top_k: 8,
                capacity: 16,
            };
            let usage = OperationUse {
                phase: kernel_core::precision::ExecutionPhase::Decode,
                capacity: 16,
                graph: false,
            };
            let op = Sm120::new()
                .with_router(row.clone())
                .with_tile_permute(permute)
                .moe_route(&query, &routes, 0, 16, usage, Need::rows(16))
                .unwrap();
            assert_eq!(op.info().symbols[0].0, row.symbol);
            let launches = op
                .launches(&self::buffers(), View::EMPTY, Counts::rows(3))
                .unwrap();
            assert_eq!(launches[0].arguments.0.last(), Some(&one));
        }
    }

    #[test]
    fn shared_expert_combine_equals_its_builder() {
        let entries = route_cubin();
        let combine = MoeQuery::SharedExpertGateCombine {
            width: HIDDEN,
            capacity: PREFILL,
        };
        let rank = 1;
        let op = cubin_backend()
            .moe_shared_expert_combine(&combine, &local(PREFILL), rank, Need::rows(PREFILL))
            .unwrap();
        assert_eq!(op.info().guarantees, Invariance::any_rows(PREFILL));
        let buffers = SharedCombineBuffers {
            routed: view(1, BIG),
            ids: view(2, BIG),
            shared: view(3, BIG),
            x: view(4, BIG),
            gate_weight: view(5, BIG),
            output: view(6, BIG),
            live_rows: view(7, BIG),
        };
        for rows in [1, 9, PREFILL] {
            assert_eq!(
                op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![built(
                    &find(&entries, "combine_v1"),
                    rows,
                    &[1, 2, 3, 4, 5, 6, 7],
                    Some(rank)
                )]
            );
        }
        let short = SharedCombineBuffers {
            routed: view(1, u64::from(4 * TOP_K * HIDDEN * 2) - 1),
            ..buffers
        };
        assert!(op.launches(&short, View::EMPTY, Counts::rows(4)).is_err());
        assert!(
            op.launches(&buffers, View::EMPTY, Counts::rows(PREFILL + 1))
                .is_err()
        );
        assert!(matches!(
            cubin_backend().moe_shared_expert_combine(
                &combine,
                &local(PREFILL),
                EXPERTS / LOCAL,
                Need::rows(4)
            ),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn prepare_refuses_unsupported_unserved_and_over_limit_routes() {
        let backend = cubin_backend();
        // Another variant of the router's or routes' query.
        let permute = RowQuery::ExpertPermute {
            experts: EXPERTS,
            block_size: 16,
        };
        assert!(matches!(
            backend.moe_route(
                &permute,
                &local(32),
                0,
                16,
                use_of(Need::rows(32)),
                Need::rows(32)
            ),
            Err(Refused::Rule(_))
        ));
        let combine = MoeQuery::SharedExpertGateCombine {
            width: HIDDEN,
            capacity: 32,
        };
        assert!(matches!(
            backend.moe_route(
                &softmax(),
                &combine,
                0,
                16,
                use_of(Need::rows(32)),
                Need::rows(32)
            ),
            Err(Refused::Rule(_))
        ));
        // No entries, a router the entries do not compute, a rank past the
        // shards, and routes over another top-k.
        assert!(
            Sm120::new()
                .moe_route(
                    &softmax(),
                    &local(32),
                    0,
                    16,
                    use_of(Need::rows(32)),
                    Need::rows(32)
                )
                .is_err()
        );
        let sigmoid = RowQuery::Router {
            experts: EXPERTS,
            top_k: TOP_K,
            kind: RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 1.0,
            },
        };
        assert!(
            backend
                .moe_route(
                    &sigmoid,
                    &local(32),
                    0,
                    16,
                    use_of(Need::rows(32)),
                    Need::rows(32)
                )
                .is_err()
        );
        assert!(
            backend
                .moe_route(
                    &softmax(),
                    &local(32),
                    EXPERTS / LOCAL,
                    16,
                    use_of(Need::rows(32)),
                    Need::rows(32)
                )
                .is_err()
        );
        let other = MoeQuery::LocalRoutes {
            experts: EXPERTS,
            local_experts: LOCAL,
            top_k: 8,
            capacity: 32,
        };
        assert!(
            backend
                .moe_route(
                    &softmax(),
                    &other,
                    0,
                    16,
                    use_of(Need::rows(32)),
                    Need::rows(32)
                )
                .is_err()
        );
        // Capacities past the routes' query, past every path (the entries'
        // 8192 tokens) and past MAX_CAPACITY, refused before any table.
        assert!(
            backend
                .moe_route(
                    &softmax(),
                    &local(32),
                    0,
                    16,
                    use_of(Need::rows(33)),
                    Need::rows(33)
                )
                .is_err()
        );
        for capacity in [PREFILL + 1, MAX_CAPACITY + 1, u32::MAX] {
            assert!(matches!(
                backend.moe_route(
                    &softmax(),
                    &local(u32::MAX),
                    0,
                    16,
                    use_of(Need::rows(capacity)),
                    Need::rows(capacity)
                ),
                Err(Refused::Rule(_))
            ));
        }
        // The one-CTA chained decode route holds at most 32 tokens.
        let [decode, _] = TP4_288_TOP8.routes();
        assert!(
            LocalRoute {
                capacity: 64,
                ..decode
            }
            .plan(RouteKernel::Decode, 64)
            .is_err()
        );
        // Short route metadata and tile tables are refused at launch.
        let op = backend
            .moe_route(
                &softmax(),
                &local(PREFILL),
                0,
                16,
                use_of(Need::rows(PREFILL)),
                Need::rows(PREFILL),
            )
            .unwrap();
        let mut short = buffers();
        short.routes.sorted_rows = view(
            7,
            u64::from(padded_capacity(64 * TOP_K, LOCAL, 16).unwrap()) * 4 - 1,
        );
        assert!(op.launches(&short, View::EMPTY, Counts::rows(64)).is_err());
        assert!(op.launches(&short, View::EMPTY, Counts::rows(63)).is_ok());
        let mut untiled = buffers();
        untiled.tile_offsets = None;
        assert!(
            op.launches(&untiled, View::EMPTY, Counts::rows(64))
                .is_err()
        );
        assert!(op.launches(&untiled, View::EMPTY, Counts::rows(1)).is_err());
        assert!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(0))
                .is_err()
        );
        assert!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(PREFILL + 1))
                .is_err()
        );
    }

    /// A route of every expert as a model without expert sharding declares
    /// it: a sigmoid-bias row router and the tiled permutation, padded to
    /// `block` routes, and the 16-token decode route.
    fn tiled_backend(block: u32) -> (Sm120, RowInstantiation, [RowInstantiation; 4]) {
        let mut router = router_entry(PROFILE, "test_sigmoid256", 256, 1, false).unwrap();
        router.invariance = crate::row::row_local().with_ties(Ties::LowerIndex);
        let query = RowQuery::ExpertPermute {
            experts: 256,
            block_size: block,
        };
        let tiles = Grid::RouteTiles {
            top_k: 8,
            tile: 256,
        };
        let tile = |name: &str, grid, params, body: &str| {
            let mut entry = RowInstantiation::new(
                PROFILE,
                &format!("permute_tile_{name}_test_b{block}"),
                query,
                256,
                (grid, 256),
                params,
                body.into(),
            );
            entry.invariance = Invariance::PREFILL.of_routed_operation();
            entry
        };
        let permute = [
            tile(
                "count",
                tiles,
                vec![
                    ptr("ids", "const int32_t"),
                    ptr("tile_counts", "int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "count();",
            ),
            tile(
                "prefix",
                Grid::Experts(256),
                vec![
                    ptr("tile_counts", "const int32_t"),
                    ptr("counts", "int32_t"),
                    ptr("tile_offsets", "int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "prefix();",
            ),
            tile(
                "scan",
                Grid::One,
                vec![
                    ptr("counts", "const int32_t"),
                    ptr("offsets", "int32_t"),
                    ptr("sorted_rows", "int32_t"),
                    ptr("padded_rows", "int32_t"),
                ],
                "scan();",
            ),
            tile(
                "scatter",
                tiles,
                vec![
                    ptr("ids", "const int32_t"),
                    ptr("tile_offsets", "const int32_t"),
                    ptr("offsets", "const int32_t"),
                    ptr("sorted_rows", "int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "scatter();",
            ),
        ];
        let backend = Sm120::new()
            .with_router(router.clone())
            .with_tile_permute(permute.clone())
            .with_sigmoid_decode_route("chop_moe_test_decode_route_b16_v1", 16, Invariance::NONE);
        (backend, router, permute)
    }

    fn sigmoid256() -> (RowQuery, MoeQuery) {
        (
            RowQuery::Router {
                experts: 256,
                top_k: 8,
                kind: RouterKind::SigmoidBiasSelectNormalized {
                    scaling_factor: 1.0,
                },
            },
            MoeQuery::LocalRoutes {
                experts: 256,
                local_experts: 256,
                top_k: 8,
                capacity: PREFILL,
            },
        )
    }

    /// A row engine launch as a model's builder plans it at the smallest
    /// bucket, or at `bucket`.
    fn row_launch(
        entry: &RowInstantiation,
        rows: u32,
        bucket: Option<u32>,
        arguments: Vec<Argument>,
    ) -> Launch {
        let bucket = bucket.unwrap_or_else(|| smallest_bucket(rows).unwrap());
        Launch {
            symbol: entry.symbol.clone(),
            rows: bucket,
            plan: entry.plan(bucket, row_counts(rows)).unwrap(),
            arguments: entry.pack_arguments(arguments).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    #[test]
    fn every_expert_routes_equal_the_builders() {
        let (backend, router_row, [count, prefix, scan, scatter]) = tiled_backend(64);
        let (router, routes) = sigmoid256();
        let b = |handle| Argument::DeviceBuffer(handle);
        // The one-CTA decode route up to its 16 tokens.
        let op = backend
            .moe_route(
                &router,
                &routes,
                0,
                16,
                use_of(Need::rows(16)),
                Need::rows(16),
            )
            .unwrap();
        assert_eq!(op.info().bytes_of("routes.active_experts"), Some(257 * 4));
        for rows in [1, 9, 16] {
            assert_eq!(
                op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![Launch {
                    symbol: "chop_moe_test_decode_route_b16_v1".into(),
                    rows: 16,
                    plan: decode_route_plan(16, rows).unwrap(),
                    arguments: decode_route_arguments([1, 2, 3, 4, 5, 6, 7, 8, 13, 9], 1.0)
                        .unwrap(),
                    collective: false,
                    persistent: false,
                }]
            );
        }
        // Prefill rows: the row router at the rows' bucket, then the tiled
        // permutation at the capacity its tile tables assume.
        let need = Need::rows(PREFILL).with_exact(Invariance::PREFILL.of_routed_operation());
        let op = backend
            .moe_route(&router, &routes, 0, 64, use_of(need), need)
            .unwrap();
        assert_eq!(op.info().bytes_of("routes.active_experts"), None);
        assert_eq!(op.info().bytes_of("tile_counts"), Some(256 * 256 * 4));
        assert_eq!(op.info().guarantees.chunk, Chunk::Any);
        assert_eq!(op.info().guarantees.ties, Ties::LowerIndex);
        let at = Some(PREFILL);
        for rows in [1, 17, 100, PREFILL] {
            assert_eq!(
                op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![
                    row_launch(
                        &router_row,
                        rows,
                        None,
                        vec![
                            b(1),
                            b(2),
                            b(3),
                            b(4),
                            b(13),
                            Argument::U32(1.0_f32.to_bits())
                        ]
                    ),
                    row_launch(&count, rows, at, vec![b(3), b(10), b(13)]),
                    row_launch(&prefix, rows, at, vec![b(10), b(5), b(11), b(13)]),
                    row_launch(&scan, rows, at, vec![b(5), b(6), b(7), b(8)]),
                    row_launch(&scatter, rows, at, vec![b(3), b(11), b(6), b(7), b(13)]),
                ],
                "{rows} rows"
            );
        }
        // Tile tables follow the capacity, not the step's rows.
        let mut short = buffers();
        short.tile_counts = Some(view(10, 256 * 256 * 4 - 1));
        assert!(op.launches(&short, View::EMPTY, Counts::rows(1)).is_err());
        // An op whose row counts would write the active experts at some
        // and not at others is refused.
        // The 64-row tile takes the permutation at every row count; with a
        // 16-row permutation as well, rows up to 16 would take the decode
        // route.
        let op = backend
            .moe_route(
                &router,
                &routes,
                0,
                64,
                use_of(Need::rows(64)),
                Need::rows(64),
            )
            .unwrap();
        assert_eq!(
            op.launches(&buffers(), View::EMPTY, Counts::rows(3))
                .unwrap()
                .len(),
            5
        );
        // With a 16-row permutation registered as well, a 16-row use has
        // two routes of equal capacity that launch differently: ambiguous.
        // A wider use holds only the permutation.
        let (both, ..) = tiled_backend(16);
        let at16 = Need::rows(16);
        assert!(matches!(
            both.moe_route(&router, &routes, 0, 16, use_of(at16), at16),
            Err(Refused::Rule(rule)) if rule.contains("chop_moe_test_decode_route_b16_v1")
        ));
        let at64 = Need::rows(64);
        let op = both
            .moe_route(&router, &routes, 0, 16, use_of(at64), at64)
            .unwrap();
        assert_eq!(op.info().bytes_of("routes.active_experts"), None);
        // Every expert is local to the one rank only.
        assert!(
            backend
                .moe_route(
                    &router,
                    &routes,
                    1,
                    16,
                    use_of(Need::rows(16)),
                    Need::rows(16)
                )
                .is_err()
        );
        let sharded = MoeQuery::LocalRoutes {
            experts: 256,
            local_experts: 128,
            top_k: 8,
            capacity: PREFILL,
        };
        assert!(
            backend
                .moe_route(
                    &router,
                    &sharded,
                    0,
                    16,
                    use_of(Need::rows(16)),
                    Need::rows(16)
                )
                .is_err()
        );
    }

    #[test]
    fn expert_combines_equal_their_builder() {
        use kernel_core::gated_mlp::RouteWeightPoint;
        let entry =
            crate::row::builders::expert_combine_entry(PROFILE, "test4096", 4096, 256, true)
                .unwrap();
        let weighted = ExpertCombineOp {
            experts: 256,
            top_k: 8,
            width: 4096,
            route_weight: Some(RouteWeightPoint::AfterDown),
        };
        assert_eq!(weighted.query(), entry.query);
        let backend = Sm120::new().with_expert_combine(entry.clone());
        let op = backend
            .moe_expert_combine(&weighted, Need::rows(PREFILL))
            .unwrap();
        let b = |handle| Argument::DeviceBuffer(handle);
        let buffers = CombineBuffers {
            routed: view(1, BIG),
            weights: Some(view(2, BIG)),
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        for rows in [1, 3, PREFILL] {
            assert_eq!(
                op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                vec![row_launch(&entry, rows, None, vec![b(1), b(2), b(3), b(4)])]
            );
        }
        // Weighted rows need no weights.
        let unweighted = CombineBuffers {
            weights: None,
            ..buffers
        };
        assert_eq!(
            op.launches(&unweighted, View::EMPTY, Counts::rows(2))
                .unwrap()[0]
                .arguments,
            entry
                .pack_arguments(vec![b(1), Argument::NullDevicePointer, b(3), b(4)])
                .unwrap()
        );
        let short = CombineBuffers {
            routed: view(1, 2 * 8 * 4096 * 2 - 1),
            ..buffers
        };
        assert!(op.launches(&short, View::EMPTY, Counts::rows(2)).is_err());
        // The weighted entry does not serve a combine that applies the
        // weights, nor another width.
        let weighting = ExpertCombineOp {
            route_weight: None,
            ..weighted
        };
        assert!(matches!(
            backend.moe_expert_combine(&weighting, Need::rows(16)),
            Err(Refused::Rule(_))
        ));
        let wider = ExpertCombineOp {
            width: 8192,
            ..weighted
        };
        assert!(backend.moe_expert_combine(&wider, Need::rows(16)).is_err());
        // A combine that applies the weights needs them.
        let op = Sm120::new()
            .with_expert_combine(
                crate::row::builders::expert_combine_entry(PROFILE, "test4096w", 4096, 256, false)
                    .unwrap(),
            )
            .moe_expert_combine(&weighting, Need::rows(16))
            .unwrap();
        assert!(
            op.launches(&unweighted, View::EMPTY, Counts::rows(2))
                .is_err()
        );
        assert!(op.launches(&buffers, View::EMPTY, Counts::rows(2)).is_ok());
    }

    #[test]
    fn routed_sums_equal_the_lowering_bindings() {
        let [sum] = TP4_288_TOP8.sums();
        // A route cubin holds the forms its model passes: both here.
        let backend = Sm120::new()
            .with_routed_sum(sum, PREFIX)
            .with_routed_sum(sum.with_output(RoutedSumOutput::F32), PREFIX);
        let query = |output| MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output,
            capacity: PREFILL,
        };
        let b = |handle| Argument::DeviceBuffer(handle);
        let buffers = RoutedSumBuffers {
            routed: view(1, BIG),
            ids: view(2, BIG),
            addend: None,
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        for (output, form) in [
            (DType::F32, RoutedSumOutput::F32),
            (DType::Bf16, RoutedSumOutput::Bf16),
        ] {
            let sum = sum.with_output(form);
            let op = backend
                .moe_routed_sum(&query(output), 1, Need::rows(PREFILL))
                .unwrap();
            assert_eq!(op.info().guarantees, Invariance::any_rows(PREFILL));
            for rows in [1, 33, PREFILL] {
                assert_eq!(
                    op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    vec![Launch {
                        symbol: sum.symbol(PREFIX),
                        rows,
                        plan: sum.plan(rows).unwrap(),
                        arguments: ArgumentList(vec![b(1), b(2), b(4), Argument::U32(1), b(3)]),
                        collective: false,
                        persistent: false,
                    }]
                );
            }
            let short = RoutedSumBuffers {
                output: view(3, 2 * sum.output_row_bytes() - 1),
                ..buffers
            };
            assert!(op.launches(&short, View::EMPTY, Counts::rows(2)).is_err());
        }
        // A rank past the shards, another width or an F16 output, and a
        // capacity past the query's.
        assert!(
            backend
                .moe_routed_sum(&query(DType::F32), 4, Need::rows(8))
                .is_err()
        );
        let narrow = MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 2048,
            output: DType::F32,
            capacity: PREFILL,
        };
        assert!(backend.moe_routed_sum(&narrow, 0, Need::rows(8)).is_err());
        assert!(
            backend
                .moe_routed_sum(&query(DType::F16), 0, Need::rows(8))
                .is_err()
        );
        assert!(
            backend
                .moe_routed_sum(&query(DType::F32), 0, Need::rows(PREFILL + 1))
                .is_err()
        );
    }

    /// A step that binds an addend launches the F32 sum that adds it, with
    /// the sum's plan and the addend bound before the output, as the
    /// lowering binds `routed_sum_add`; a step without one launches the sum.
    /// A BF16 sum, or an F32 sum with no add registered, refuses an addend.
    #[test]
    fn routed_sum_adds_equal_the_lowering_bindings() {
        let sum = TP4_288_TOP8.sums()[0].with_output(RoutedSumOutput::F32);
        let backend = Sm120::new()
            .with_routed_sum(sum, PREFIX)
            .with_routed_sum(sum.with_output(RoutedSumOutput::Bf16), PREFIX)
            .with_routed_sum_add(sum, PREFIX);
        let query = |output| MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output,
            capacity: PREFILL,
        };
        let b = |handle| Argument::DeviceBuffer(handle);
        let plain = RoutedSumBuffers {
            routed: view(1, BIG),
            ids: view(2, BIG),
            addend: None,
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        let added = RoutedSumBuffers {
            addend: Some(view(5, BIG)),
            ..plain
        };
        let op = backend
            .moe_routed_sum(&query(DType::F32), 1, Need::rows(PREFILL))
            .unwrap();
        let symbols = op
            .info()
            .symbols
            .iter()
            .map(|(symbol, shared)| (symbol.as_str(), *shared))
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                (sum.symbol(PREFIX).as_str(), 0),
                (sum.add_symbol(PREFIX).as_str(), 0)
            ]
        );
        // The sum and its add form declare the same claims.
        assert_eq!(op.info().guarantees, Invariance::any_rows(PREFILL));
        for rows in [1, 33, PREFILL] {
            let launch = |symbol: String, arguments| Launch {
                symbol,
                rows,
                plan: sum.plan(rows).unwrap(),
                arguments: ArgumentList(arguments),
                collective: false,
                persistent: false,
            };
            let counts = Counts::rows(rows);
            assert_eq!(
                op.launches(&added, View::EMPTY, counts).unwrap(),
                vec![launch(
                    sum.add_symbol(PREFIX),
                    vec![b(1), b(2), b(4), Argument::U32(1), b(5), b(3)]
                )]
            );
            assert_eq!(
                op.launches(&plain, View::EMPTY, counts).unwrap(),
                vec![launch(
                    sum.symbol(PREFIX),
                    vec![b(1), b(2), b(4), Argument::U32(1), b(3)]
                )]
            );
        }
        // The addend holds a row per token.
        let short = RoutedSumBuffers {
            addend: Some(view(5, 2 * 4096 * 4 - 1)),
            ..plain
        };
        assert!(op.launches(&short, View::EMPTY, Counts::rows(2)).is_err());
        // A BF16 sum adds no row.
        let bf16 = backend
            .moe_routed_sum(&query(DType::Bf16), 1, Need::rows(PREFILL))
            .unwrap();
        assert_eq!(bf16.info().symbols.len(), 1);
        assert!(bf16.launches(&added, View::EMPTY, Counts::rows(2)).is_err());
        // Without the add registered, the F32 sum lists its own entry alone,
        // names another implementation and refuses an addend.
        let alone = Sm120::new()
            .with_routed_sum(sum, PREFIX)
            .moe_routed_sum(&query(DType::F32), 1, Need::rows(PREFILL))
            .unwrap();
        assert_ne!(alone.info().implementation, op.info().implementation);
        assert_eq!(alone.info().symbols.len(), 1);
        assert!(
            alone
                .launches(&added, View::EMPTY, Counts::rows(2))
                .is_err()
        );
        assert!(alone.launches(&plain, View::EMPTY, Counts::rows(2)).is_ok());
        // An add past the op's capacity refuses the op.
        let narrow = Sm120::new()
            .with_routed_sum(sum, PREFIX)
            .with_routed_sum_add(
                RoutedSum {
                    capacity: 64,
                    ..sum
                },
                PREFIX,
            );
        assert!(
            narrow
                .moe_routed_sum(&query(DType::F32), 1, Need::rows(65))
                .is_err()
        );
    }

    /// What an add registration's fields change: as the sum's, its output
    /// form selects (only the F32 sum adds).
    #[test]
    fn routed_sum_add_keys_follow_launches() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let sum = TP4_288_TOP8.sums()[0].with_output(RoutedSumOutput::F32);
        let op = MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output: DType::F32,
            capacity: PREFILL,
        };
        let added = RoutedSumBuffers {
            routed: view(1, BIG),
            ids: view(2, BIG),
            addend: Some(view(5, BIG)),
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        let mut names = field_names!(RoutedSum {
            top_k,
            width,
            local_experts,
            capacity,
            output
        });
        names.push("prefix");
        let fields: Vec<Field<(RoutedSum, String)>> = vec![
            ("top_k", Class::Selects, Box::new(|s| s.0.top_k = 4)),
            ("width", Class::Selects, Box::new(|s| s.0.width = 2048)),
            (
                "local_experts",
                Class::Selects,
                Box::new(|s| s.0.local_experts = 64),
            ),
            ("capacity", Class::Launch, Box::new(|s| s.0.capacity = 4096)),
            (
                "output",
                Class::Selects,
                Box::new(|s| s.0.output = RoutedSumOutput::Bf16),
            ),
            (
                "prefix",
                Class::Launch,
                Box::new(|s| s.1 = "chop_moe_other".into()),
            ),
        ];
        keys_follow_launches(
            &(sum, PREFIX.to_owned()),
            names,
            fields,
            |(sum, prefix)| {
                Kernel::sum_add(*sum, prefix)
                    .map(|kernel| kernel.launch())
                    .unwrap_or_default()
            },
            |(add, prefix)| {
                let prepared = Sm120::new()
                    .with_routed_sum(sum, PREFIX)
                    .with_routed_sum_add(*add, prefix.clone())
                    .moe_routed_sum(&op, 1, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&added, View::EMPTY, Counts::rows(rows))
                        .map_err(|refusal| Refused::Rule(refusal.0.into()))
                });
                let [one, all] = launches;
                Ok((String::new(), [one?, all?]))
            },
        );
    }

    /// One registration of a model's selection.
    #[derive(Clone)]
    enum Registered {
        Cubin(Vec<RouteEntry>),
        Local(LocalRoute),
        Router(RowInstantiation),
        Tiles(Box<[RowInstantiation; 4]>),
        Sigmoid,
        Combine(RowInstantiation),
        Sum(RoutedSum),
    }

    fn register(backend: Sm120, registered: Registered) -> Sm120 {
        match registered {
            Registered::Cubin(entries) => with_cubin(backend, entries),
            Registered::Local(route) => backend.with_local_route(route, PREFIX),
            Registered::Router(entry) => backend.with_router(entry),
            Registered::Tiles(tiles) => backend.with_tile_permute(*tiles),
            Registered::Sigmoid => backend.with_sigmoid_decode_route(
                "chop_moe_test_decode_route_b16_v1",
                16,
                Invariance::NONE,
            ),
            Registered::Combine(entry) => backend.with_expert_combine(entry),
            Registered::Sum(sum) => backend.with_routed_sum(sum, PREFIX),
        }
    }

    /// What every op of the family prepares and launches over `backend`.
    fn outcomes(backend: &Sm120) -> Vec<String> {
        use kernel_core::gated_mlp::RouteWeightPoint;
        let mut out = Vec::new();
        let mut route = |router: &RowQuery, routes: &MoeQuery, rank, tile, need: Need| {
            out.push(
                match backend.moe_route(router, routes, rank, tile, use_of(need), need) {
                    Ok(op) => format!(
                        "{:?} {:?} {:?}",
                        op.info(),
                        op.launches(&buffers(), View::EMPTY, Counts::rows(1)),
                        op.launches(&buffers(), View::EMPTY, Counts::rows(need.capacity)),
                    ),
                    Err(refused) => format!("{refused:?}"),
                },
            );
        };
        let prefill = Need::rows(PREFILL).with_exact(Invariance::PREFILL.of_routed_operation());
        for need in [Need::rows(4), Need::rows(32), Need::rows(64), prefill] {
            route(&softmax(), &local(PREFILL), 1, 16, need);
        }
        let chained = RowQuery::Router {
            experts: 288,
            top_k: 8,
            kind: RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.5,
            },
        };
        let chained_routes = MoeQuery::LocalRoutes {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            capacity: PREFILL,
        };
        route(&chained, &chained_routes, 1, 16, Need::rows(32));
        route(&chained, &chained_routes, 1, 64, Need::rows(PREFILL));
        let (sigmoid, every) = sigmoid256();
        route(&sigmoid, &every, 0, 16, Need::rows(16));
        route(&sigmoid, &every, 0, 64, prefill);
        let router = |op: &RowQuery| match backend.moe_router(op, Need::rows(64)) {
            Ok(op) => format!("{:?}", op.info()),
            Err(refused) => format!("{refused:?}"),
        };
        out.push(router(&softmax()));
        out.push(router(&sigmoid));
        let combine = ExpertCombineOp {
            experts: 256,
            top_k: 8,
            width: 4096,
            route_weight: Some(RouteWeightPoint::AfterDown),
        };
        out.push(match backend.moe_expert_combine(&combine, Need::rows(64)) {
            Ok(op) => format!("{:?}", op.info()),
            Err(refused) => format!("{refused:?}"),
        });
        let sum = MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output: DType::F32,
            capacity: PREFILL,
        };
        out.push(match backend.moe_routed_sum(&sum, 1, Need::rows(64)) {
            Ok(op) => format!("{:?}", op.info()),
            Err(refused) => format!("{refused:?}"),
        });
        let shared = MoeQuery::SharedExpertGateCombine {
            width: HIDDEN,
            capacity: PREFILL,
        };
        out.push(
            match backend.moe_shared_expert_combine(&shared, &local(PREFILL), 1, Need::rows(64)) {
                Ok(op) => format!("{:?}", op.info()),
                Err(refused) => format!("{refused:?}"),
            },
        );
        out
    }

    #[test]
    fn registration_order_does_not_change_the_prepared_launches() {
        let (_, router, tiles) = tiled_backend(64);
        let [decode, prefill] = TP4_288_TOP8.routes();
        let [sum] = TP4_288_TOP8.sums();
        // A second router of the same query holds fewer tokens: it serves
        // uses it holds, whatever its place in the selection.
        let mut small = find(&route_cubin(), "router_v1");
        small.symbol = "chop_moe_test_small_router_v1".into();
        small.tokens_capacity = 64;
        let mut registered = vec![
            Registered::Cubin(route_cubin()),
            Registered::Cubin(vec![small]),
        ];
        registered.extend([
            Registered::Local(decode),
            Registered::Local(prefill),
            Registered::Router(router),
            Registered::Tiles(Box::new(tiles)),
            Registered::Sigmoid,
            Registered::Combine(
                crate::row::builders::expert_combine_entry(PROFILE, "test4096", 4096, 256, true)
                    .unwrap(),
            ),
            Registered::Sum(sum.with_output(RoutedSumOutput::F32)),
        ]);
        let build = |order: &[Registered]| order.iter().cloned().fold(Sm120::new(), register);
        let forward = outcomes(&build(&registered));
        // Every op of the family prepares over this selection, and the
        // 64-row uses take the smaller router.
        assert!(forward.iter().all(|outcome| !outcome.starts_with("Rule")));
        assert!(forward[2].contains("chop_moe_test_small_router_v1"));
        assert!(!forward[3].contains("chop_moe_test_small_router_v1"));
        assert!(forward[8].contains("chop_moe_test_small_router_v1"));
        let mut reversed = registered.clone();
        reversed.reverse();
        let mut rotated = registered.clone();
        rotated.rotate_left(registered.len() / 3);
        let interleaved = registered
            .iter()
            .step_by(2)
            .chain(registered.iter().skip(1).step_by(2))
            .cloned()
            .collect::<Vec<_>>();
        let doubled = registered
            .iter()
            .flat_map(|registration| [registration.clone(), registration.clone()])
            .collect::<Vec<_>>();
        for order in [reversed, rotated, interleaved, doubled] {
            assert_eq!(outcomes(&build(&order)), forward);
        }
    }

    /// Finding R1-1: each launch records the rows its builder records, not
    /// the step's live rows.
    #[test]
    fn launches_record_each_builders_rows() {
        let (backend, ..) = tiled_backend(64);
        let (router, routes) = sigmoid256();
        let prefill = Need::rows(PREFILL).with_exact(Invariance::PREFILL.of_routed_operation());
        let op = backend
            .moe_route(&router, &routes, 0, 64, use_of(prefill), prefill)
            .unwrap();
        // 100 live prefill rows: the router at its 128-row bucket, the
        // permutation at the 8192 rows its tile tables assume.
        let rows = op
            .launches(&buffers(), View::EMPTY, Counts::rows(100))
            .unwrap()
            .iter()
            .map(|launch| launch.rows)
            .collect::<Vec<_>>();
        assert_eq!(rows, [128, PREFILL, PREFILL, PREFILL, PREFILL]);
        let entry =
            crate::row::builders::expert_combine_entry(PROFILE, "test4096", 4096, 256, true)
                .unwrap();
        let combine = ExpertCombineOp {
            experts: 256,
            top_k: 8,
            width: 4096,
            route_weight: Some(kernel_core::gated_mlp::RouteWeightPoint::AfterDown),
        };
        let op = Sm120::new()
            .with_expert_combine(entry)
            .moe_expert_combine(&combine, Need::rows(PREFILL))
            .unwrap();
        let buffers = CombineBuffers {
            routed: view(1, BIG),
            weights: Some(view(2, BIG)),
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        assert_eq!(
            op.launches(&buffers, View::EMPTY, Counts::rows(100))
                .unwrap()[0]
                .rows,
            128
        );
        // A decode route records its bucket, a prefill route cubin entry
        // the step's rows.
        let op = cubin_backend()
            .moe_route(
                &softmax(),
                &local(32),
                0,
                16,
                use_of(Need::rows(4)),
                Need::rows(4),
            )
            .unwrap();
        assert_eq!(
            op.launches(&self::buffers(), View::EMPTY, Counts::rows(3))
                .unwrap()[0]
                .rows,
            4
        );
        let op = cubin_backend()
            .moe_route(&softmax(), &local(PREFILL), 0, 16, use_of(prefill), prefill)
            .unwrap();
        assert!(
            op.launches(&self::buffers(), View::EMPTY, Counts::rows(100))
                .unwrap()
                .iter()
                .all(|launch| launch.rows == 100)
        );
    }

    /// Finding R1-2: a routed sum serves only the output form registered.
    #[test]
    fn routed_sums_serve_only_their_registered_form() {
        let [sum] = TP4_288_TOP8.sums();
        let backend = Sm120::new().with_routed_sum(sum.with_output(RoutedSumOutput::F32), PREFIX);
        let query = |output| MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output,
            capacity: PREFILL,
        };
        let op = backend
            .moe_routed_sum(&query(DType::F32), 0, Need::rows(64))
            .unwrap();
        assert_eq!(
            op.info().symbols[0].0,
            sum.with_output(RoutedSumOutput::F32).symbol(PREFIX)
        );
        assert!(matches!(
            backend.moe_routed_sum(&query(DType::Bf16), 0, Need::rows(64)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120 routed sum entry")
        ));
    }

    /// Finding R1-3: a registration the op's use does not take never
    /// decides or refuses its prepare.
    #[test]
    fn unused_registrations_do_not_change_a_prepare() {
        let (router, routes) = sigmoid256();
        let decode = Need::rows(16);
        let alone = Sm120::new().with_sigmoid_decode_route(
            "chop_moe_test_decode_route_b16_v1",
            16,
            Invariance::NONE,
        );
        let expected = format!(
            "{:?}",
            alone
                .moe_route(&router, &routes, 0, 16, use_of(decode), decode)
                .unwrap()
                .info()
        );
        // Two different row routers of the same query, with and without a
        // tiled permutation of another padding.
        let mut other = router_entry(PROFILE, "test_sigmoid256", 256, 1, false).unwrap();
        other.symbol = "chop_row_router_other_sigmoid256_v1".into();
        let (_, first, tiles) = tiled_backend(64);
        let routers = Sm120::new()
            .with_sigmoid_decode_route("chop_moe_test_decode_route_b16_v1", 16, Invariance::NONE)
            .with_router(first)
            .with_router(other);
        let op = routers
            .moe_route(&router, &routes, 0, 16, use_of(decode), decode)
            .unwrap();
        assert_eq!(format!("{:?}", op.info()), expected);
        let tiled = routers.with_tile_permute(tiles);
        let op = tiled
            .moe_route(&router, &routes, 0, 16, use_of(decode), decode)
            .unwrap();
        assert_eq!(format!("{:?}", op.info()), expected);
        // The use that takes the permutation meets the two routers: a tie.
        let prefill = Need::rows(PREFILL);
        assert!(matches!(
            tiled.moe_route(&router, &routes, 0, 64, use_of(prefill), prefill),
            Err(Refused::Rule(rule)) if rule.contains(
                "chop_row_router_other_sigmoid256_v1"
            )
        ));
        // Likewise two route cubin routers of one capacity tie only for the
        // uses that take the router and permutation.
        let mut second = find(&route_cubin(), "router_v1");
        second.symbol = "chop_moe_test_second_router_v1".into();
        second.body = "second();".into();
        let both = cubin_backend().with_route_entry(second);
        assert!(
            both.moe_route(&softmax(), &local(32), 0, 16, use_of(decode), decode)
                .is_ok()
        );
        assert!(matches!(
            both.moe_route(&softmax(), &local(PREFILL), 0, 16, use_of(prefill), prefill),
            Err(Refused::Rule(rule)) if rule.contains("chop_moe_test_second_router_v1")
        ));
    }

    /// A router registered again under another profile, or with another
    /// claim, launches the same: one entry in every order, claiming what
    /// the registrations of its launch declare.
    #[test]
    fn registrations_of_one_launch_are_one_router() {
        let mut claimed = router_entry(PROFILE, "test_sigmoid256", 256, 1, false).unwrap();
        claimed.invariance = Invariance::PREFILL.with_ties(Ties::LowerIndex);
        let unclaimed = RowInstantiation {
            invariance: Invariance::NONE,
            ..claimed.clone()
        };
        let profiled = RowInstantiation {
            profile: Profile::synthetic(crate::ProfileNames {
                name: "moe-route-other",
                tag: "moe_route_other",
            }),
            ..claimed.clone()
        };
        let query = claimed.query;
        let prepare = |order: &[&RowInstantiation], need: Need| {
            order
                .iter()
                .fold(Sm120::new(), |backend, entry| {
                    backend.with_router((*entry).clone())
                })
                .moe_router(&query, need)
                .map(|op| op.info().clone())
        };
        let alone = prepare(&[&claimed], Need::rows(64)).unwrap();
        let both = prepare(&[&claimed, &profiled], Need::rows(64)).unwrap();
        assert_eq!(
            prepare(&[&profiled, &claimed], Need::rows(64)),
            Ok(both.clone())
        );
        assert_eq!(
            (both.symbols, both.guarantees),
            (alone.symbols.clone(), alone.guarantees)
        );
        let need = Need::rows(64).with_exact(Invariance::PREFILL);
        for order in [[&claimed, &unclaimed], [&unclaimed, &claimed]] {
            assert_eq!(prepare(&order, need), prepare(&[&claimed], need));
            let plain = prepare(&order, Need::rows(64)).unwrap();
            assert_eq!(plain.symbols, alone.symbols);
            assert_eq!(plain.guarantees, claimed.invariance.meet(Invariance::NONE));
        }
    }

    /// Finding R1-4: entries that cannot give the needed exactness are
    /// dropped before the least capacity and the tie.
    #[test]
    fn exactness_is_filtered_before_the_choice() {
        let mut exact = router_entry(PROFILE, "test_sigmoid256", 256, 1, false).unwrap();
        exact.invariance = Invariance::PREFILL.with_ties(Ties::LowerIndex);
        let mut loose = exact.clone();
        loose.symbol = "chop_row_router_loose_sigmoid256_v1".into();
        loose.invariance = Invariance::NONE;
        let query = exact.query;
        let need = Need::rows(64).with_exact(Invariance::PREFILL);
        for backend in [
            Sm120::new().with_router(exact.clone()),
            Sm120::new()
                .with_router(exact.clone())
                .with_router(loose.clone()),
            Sm120::new()
                .with_router(loose.clone())
                .with_router(exact.clone()),
        ] {
            let op = backend.moe_router(&query, need).unwrap();
            assert_eq!(op.info().symbols[0].0, exact.symbol);
        }
        // Without an exactness need both serve and tie.
        assert!(matches!(
            Sm120::new()
                .with_router(exact.clone())
                .with_router(loose.clone())
                .moe_router(&query, Need::rows(64)),
            Err(Refused::Rule(rule)) if rule.contains("chop_row_router_loose_sigmoid256_v1")
        ));
        // Only the loose router: the exactness is refused.
        assert!(matches!(
            Sm120::new().with_router(loose).moe_router(&query, need),
            Err(Refused::Inexact { .. })
        ));
        // Combines alike: a declared one-row combine is taken over an
        // undeclared one of the same build.
        let mut declared =
            crate::row::builders::expert_combine_entry(PROFILE, "test4096", 4096, 256, true)
                .unwrap();
        declared.invariance = Invariance::any_rows(PREFILL);
        let mut plain = declared.clone();
        plain.symbol = "chop_row_combine_plain4096_v1".into();
        plain.invariance = Invariance::NONE;
        let combine = ExpertCombineOp {
            experts: 256,
            top_k: 8,
            width: 4096,
            route_weight: Some(kernel_core::gated_mlp::RouteWeightPoint::AfterDown),
        };
        let op = Sm120::new()
            .with_expert_combine(plain)
            .with_expert_combine(declared.clone())
            .moe_expert_combine(&combine, Need::rows(64).with_exact(Invariance::one_row(64)))
            .unwrap();
        assert_eq!(op.info().symbols[0].0, declared.symbol);
    }

    /// The grouped-softmax router of the test geometry: 8 groups, the best
    /// 4 taken.
    fn grouped() -> RowQuery {
        RowQuery::Router {
            experts: EXPERTS,
            top_k: TOP_K,
            kind: RouterKind::GroupLimitedSelectedSoftmax {
                n_group: 8,
                topk_group: 4,
            },
        }
    }

    /// A router entry of the grouped scoring, its own cubin's.
    fn grouped_router() -> RouteEntry {
        let mut entry = find(&route_cubin(), "router_v1");
        entry.contract = RouteContract::Router(grouped());
        entry.symbol = "chop_moe_test_grouped_router_v1".into();
        entry.body = "grouped();".into();
        entry
    }

    /// The auditor's case: a decode route computes the router kind it is
    /// registered with, so a grouped router registered beside it never
    /// admits it for a grouped op, in either order; the grouped op takes
    /// that router and the permutation. A decode route registered for the
    /// grouped kind serves the grouped op without a router beside it.
    #[test]
    fn decode_routes_serve_only_the_router_kind_they_compute() {
        let decode = Need::rows(4);
        let route = |backend: &Sm120, router: &RowQuery| {
            backend
                .moe_route(router, &local(32), 0, 16, use_of(decode), decode)
                .map(|op| op.info().symbols.clone())
        };
        let registrations: [fn(Sm120) -> Sm120; 2] = [
            |backend| with_cubin(backend, route_cubin()),
            |backend| backend.with_route_entry(grouped_router()),
        ];
        for order in [[0, 1], [1, 0]] {
            let backend = order
                .iter()
                .fold(Sm120::new(), |backend, &i| registrations[i](backend));
            let symbols = route(&backend, &grouped()).unwrap();
            assert_eq!(symbols[0].0, "chop_moe_test_grouped_router_v1");
            assert!(
                symbols
                    .iter()
                    .all(|(symbol, _)| !symbol.contains("decode_route"))
            );
            // The softmax op still takes the decode route.
            let symbols = route(&backend, &softmax()).unwrap();
            assert_eq!(symbols[0].0, "chop_moe_test_decode_route_t4_v1");
        }
        let RowQuery::Router { kind, .. } = grouped() else {
            unreachable!()
        };
        let grouped_route =
            Sm120::new().with_decode_route(find(&route_cubin(), "decode_route_t4_v1"), kind);
        assert_eq!(
            route(&grouped_route, &grouped()).unwrap()[0].0,
            "chop_moe_test_decode_route_t4_v1"
        );
        assert!(route(&grouped_route, &softmax()).is_err());
    }

    /// A decode route cannot register without the router kind it computes.
    #[test]
    #[should_panic(expected = "register it with with_decode_route")]
    fn decode_routes_register_only_with_their_router_kind() {
        let _ = Sm120::new().with_route_entry(find(&route_cubin(), "decode_route_t4_v1"));
    }

    /// The auditor's case: routers differing only in the local experts,
    /// which a router never reads, are one router in every order.
    #[test]
    fn routers_differing_in_unread_local_experts_are_one() {
        let router = find(&route_cubin(), "router_v1");
        let twin = RouteEntry {
            local_experts: 256,
            ..router.clone()
        };
        for order in crate::backend::select::orders(&[router.clone(), twin]) {
            let backend = order
                .into_iter()
                .fold(Sm120::new(), Sm120::with_route_entry);
            let op = backend.moe_router(&softmax(), Need::rows(8)).unwrap();
            assert_eq!(op.info().symbols[0].0, router.symbol);
        }
    }

    /// The auditor's case: a routed sum its builder refuses (capacity zero)
    /// never serves, so it does not refuse an op another sum serves, in
    /// either order; alone it refuses with the builder's reason.
    #[test]
    fn refused_routed_sums_do_not_refuse_the_op() {
        let valid = RoutedSum {
            top_k: 8,
            width: 4096,
            local_experts: 72,
            capacity: PREFILL,
            output: RoutedSumOutput::F32,
        };
        let unused = RoutedSum {
            capacity: 0,
            ..valid
        };
        let op = MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output: kernel_core::contract::DType::F32,
            capacity: PREFILL,
        };
        let prepare = |sums: &[RoutedSum]| {
            sums.iter()
                .fold(Sm120::new(), |backend, &sum| {
                    backend.with_routed_sum(sum, PREFIX)
                })
                .moe_routed_sum(&op, 0, Need::rows(64))
                .map(|op| op.info().clone())
        };
        let alone = prepare(&[valid]).unwrap();
        for order in crate::backend::select::orders(&[valid, unused]) {
            assert_eq!(prepare(&order), Ok(alone.clone()));
        }
        assert_eq!(
            prepare(&[unused]),
            Err(Refused::Rule(
                "unsupported routed sum geometry or token capacity".into()
            ))
        );
    }

    /// Every field of a route cubin entry.
    fn route_entry_names() -> Vec<&'static str> {
        crate::backend::select::field_names!(RouteEntry {
            profile,
            experts,
            local_experts,
            top_k,
            symbol,
            contract,
            tokens_capacity,
            threads,
            params,
            body,
            parallel,
            invariance
        })
    }

    /// A change of the route cubin entry field `name` alone.
    fn route_entry_change(name: &str) -> Box<dyn Fn(&mut RouteEntry)> {
        let other = Profile::synthetic(crate::ProfileNames {
            name: "moe-route-other",
            tag: "other",
        });
        match name {
            "profile" => Box::new(move |e| e.profile = other),
            "experts" => Box::new(|e| e.experts = 256),
            "local_experts" => Box::new(|e| e.local_experts = 256),
            "top_k" => Box::new(|e| e.top_k = 8),
            "symbol" => Box::new(|e| e.symbol = e.symbol.replace("_v1", "_twin_v1")),
            "contract" => Box::new(|e| {
                e.contract = if matches!(e.contract, RouteContract::Router(_)) {
                    RouteContract::Router(grouped())
                } else {
                    RouteContract::Router(softmax())
                }
            }),
            "tokens_capacity" => Box::new(|e| e.tokens_capacity *= 2),
            "threads" => Box::new(|e| e.threads *= 2),
            "params" => Box::new(|e| e.params.swap(0, 1)),
            "body" => Box::new(|e| e.body = "other();".into()),
            "parallel" => Box::new(|e| e.parallel = parallel(4).parallel),
            "invariance" => Box::new(|e| e.invariance = Invariance::one_row(2)),
            other => panic!("no change of {other}"),
        }
    }

    /// The fields of `names`, each `Inert` unless `classes` lists it.
    fn classified<E: 'static>(
        names: &[&'static str],
        classes: &[(&'static str, crate::backend::select::Class)],
        change: impl Fn(&str) -> Box<dyn Fn(&mut E)>,
    ) -> Vec<crate::backend::select::Field<E>> {
        use crate::backend::select::Class;
        names
            .iter()
            .map(|&name| {
                let class = classes
                    .iter()
                    .find(|(listed, _)| *listed == name)
                    .map_or(Class::Inert, |&(_, class)| class);
                (name, class, change(name))
            })
            .collect()
    }

    /// The route cubin entries' key holds what each kind launches, every
    /// field changed alone: a router's code and plan, never its experts,
    /// local experts, top-k or a capacity that only bounds its rows; a
    /// decode route's code, parallel geometry and the token capacity it
    /// plans at. Entries register as a model registers them.
    #[test]
    fn route_entry_keys_hold_what_each_kind_launches() {
        use crate::backend::select::{Class, keys_follow_launches};
        let code = |e: &RouteEntry| format!("{:?} {} {}", e.params, e.body, e.threads);
        let router = find(&route_cubin(), "router_v1");
        let classes = [
            ("symbol", Class::Launch),
            ("contract", Class::Selects),
            ("threads", Class::Launch),
            ("params", Class::Launch),
            ("body", Class::Launch),
            ("parallel", Class::Selects),
        ];
        let router_buffers = RouterBuffers {
            logits: view(1, BIG),
            bias: None,
            ids: view(3, BIG),
            weights: view(4, BIG),
            live_rows: view(13, BIG),
        };
        keys_follow_launches(
            &router,
            route_entry_names(),
            classified(&route_entry_names(), &classes, route_entry_change),
            |e| Kernel::entry(e).launch(),
            |e| {
                let op =
                    with_cubin(Sm120::new(), [e.clone()]).moe_router(&softmax(), Need::rows(8))?;
                let launches = [1, 8].map(|rows| {
                    op.launches(&router_buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
        let decode = find(&route_cubin(), "decode_route_t4_v1");
        let classes = [
            ("experts", Class::Selects),
            ("local_experts", Class::Selects),
            ("top_k", Class::Selects),
            ("symbol", Class::Launch),
            ("contract", Class::Selects),
            ("tokens_capacity", Class::Launch),
            ("threads", Class::Launch),
            ("params", Class::Launch),
            ("body", Class::Launch),
            ("parallel", Class::Launch),
        ];
        let need = Need::rows(4);
        keys_follow_launches(
            &decode,
            route_entry_names(),
            classified(&route_entry_names(), &classes, route_entry_change),
            |e| Kernel::entry(e).launch(),
            |e| {
                let op = with_cubin(Sm120::new().with_route_entry(router.clone()), [e.clone()])
                    .moe_route(&softmax(), &local(32), 0, 16, use_of(need), need)?;
                let launches = [1, 4].map(|rows| {
                    op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
    }

    /// The chained route, sigmoid decode route and routed sum keys hold
    /// what their kernels launch, every field changed alone: the symbols
    /// their cubin prefix and capacity name and the capacity a route plans
    /// at; never the sigmoid route's claims. The geometry selects the op.
    #[test]
    fn chained_keys_hold_what_their_kernels_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let [decode, _] = TP4_288_TOP8.routes();
        let router = RowQuery::Router {
            experts: 288,
            top_k: 8,
            kind: RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.5,
            },
        };
        let routes = MoeQuery::LocalRoutes {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            capacity: 8192,
        };
        let need = Need::rows(8);
        let mut names = field_names!(LocalRoute {
            experts,
            local_experts,
            top_k,
            capacity,
            pad
        });
        names.push("prefix");
        let fields: Vec<Field<(LocalRoute, String)>> = vec![
            ("experts", Class::Selects, Box::new(|r| r.0.experts = 256)),
            (
                "local_experts",
                Class::Selects,
                Box::new(|r| r.0.local_experts = 64),
            ),
            ("top_k", Class::Selects, Box::new(|r| r.0.top_k = 4)),
            ("capacity", Class::Launch, Box::new(|r| r.0.capacity = 16)),
            ("pad", Class::Selects, Box::new(|r| r.0.pad = 64)),
            (
                "prefix",
                Class::Launch,
                Box::new(|r| r.1 = "chop_moe_other".into()),
            ),
        ];
        let route_key = |(route, prefix): &(LocalRoute, String)| {
            route
                .kernels()
                .iter()
                .map(|&kernel| Kernel::local(*route, kernel, prefix).map(|k| k.launch()))
                .collect::<Result<Vec<_>, Refusal>>()
                .map(|launches| launches.join(" "))
                .unwrap_or_default()
        };
        keys_follow_launches(
            &(decode, PREFIX.to_owned()),
            names,
            fields,
            route_key,
            |(route, prefix)| {
                let op = Sm120::new()
                    .with_local_route(*route, prefix.clone())
                    .moe_route(&router, &routes, 1, 16, use_of(need), need)?;
                Ok((
                    String::new(),
                    [1, 8].map(|rows| {
                        op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    }),
                ))
            },
        );
        // The sigmoid decode route registers as (symbol, tokens, claims).
        let (router, routes) = sigmoid256();
        let fields: Vec<Field<(String, u32, Invariance)>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|s| s.0 = "chop_moe_test_other_route_b16_v1".into()),
            ),
            ("tokens", Class::Launch, Box::new(|s| s.1 = 8)),
            (
                "invariance",
                Class::Inert,
                Box::new(|s| s.2 = Invariance::one_row(8)),
            ),
        ];
        let base = (
            "chop_moe_test_decode_route_b16_v1".to_owned(),
            16,
            Invariance::NONE,
        );
        keys_follow_launches(
            &base,
            vec!["symbol", "tokens", "invariance"],
            fields,
            |(symbol, tokens, invariance)| Kernel::sigmoid(symbol, *tokens, *invariance).launch(),
            |(symbol, tokens, invariance)| {
                let op = Sm120::new()
                    .with_sigmoid_decode_route(symbol.clone(), *tokens, *invariance)
                    .moe_route(&router, &routes, 0, 16, use_of(need), need)?;
                Ok((
                    String::new(),
                    [1, 8].map(|rows| {
                        op.launches(&buffers(), View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    }),
                ))
            },
        );
        let [sum] = TP4_288_TOP8.sums();
        let op = MoeQuery::LocalRoutedSum {
            experts: 288,
            local_experts: 72,
            top_k: 8,
            width: 4096,
            output: DType::Bf16,
            capacity: PREFILL,
        };
        let sums = RoutedSumBuffers {
            routed: view(1, BIG),
            ids: view(2, BIG),
            addend: None,
            output: view(3, BIG),
            live_rows: view(4, BIG),
        };
        let mut names = field_names!(RoutedSum {
            top_k,
            width,
            local_experts,
            capacity,
            output
        });
        names.push("prefix");
        let fields: Vec<Field<(RoutedSum, String)>> = vec![
            ("top_k", Class::Selects, Box::new(|s| s.0.top_k = 4)),
            ("width", Class::Selects, Box::new(|s| s.0.width = 2048)),
            (
                "local_experts",
                Class::Selects,
                Box::new(|s| s.0.local_experts = 64),
            ),
            ("capacity", Class::Launch, Box::new(|s| s.0.capacity = 4096)),
            (
                "output",
                Class::Selects,
                Box::new(|s| s.0.output = RoutedSumOutput::F32),
            ),
            (
                "prefix",
                Class::Launch,
                Box::new(|s| s.1 = "chop_moe_other".into()),
            ),
        ];
        keys_follow_launches(
            &(sum, PREFIX.to_owned()),
            names,
            fields,
            |(sum, prefix)| {
                Kernel::sum(*sum, prefix)
                    .map(|kernel| kernel.launch())
                    .unwrap_or_default()
            },
            |(sum, prefix)| {
                let prepared = Sm120::new()
                    .with_routed_sum(*sum, prefix.clone())
                    .moe_routed_sum(&op, 1, Need::rows(64))?;
                Ok((
                    String::new(),
                    [1, 64].map(|rows| {
                        prepared
                            .launches(&sums, View::EMPTY, Counts::rows(rows))
                            .unwrap()
                    }),
                ))
            },
        );
    }

    /// A router of 256 experts over a hidden of 4096 at two rank counts,
    /// whose fused router symbols carry each rank count's tag.
    const FUSED_MODEL: crate::Model = crate::Model::new(
        crate::profile_names!("synthetic-fused-router", "synthetic_fused", 4, 2),
        kernel_core::rank_shape::ModelConfig {
            hidden: 4096,
            vocab: 4096,
            blocks: &[],
            linear: None,
            ngram_heads: 0,
        },
    );
    const FUSED_BUCKETS: [u32; 4] = [1, 2, 4, 8];
    const ROUTER_HIDDEN: u32 = 4096;

    /// The router projection's query at `rows`.
    fn router_projection(rows: u32) -> GemmQuery {
        gemm::bf16_query(rows, 256, ROUTER_HIDDEN, DType::F32)
    }

    /// The router GEMV entry of bucket `m`, which the rank-invariant
    /// profile owns, with one-row claims as a decode bucket of exact rows.
    fn router_gemv(m: u32) -> GemmVariant {
        gemm::instance(
            FUSED_MODEL.at(4),
            &format!("synthetic_fused_router_m{m}"),
            router_projection(m),
            64,
            m.max(8) as u16,
            DenseKernel::FragmentRouter {
                splits: gemm::ROUTER_SPLITS,
            },
        )
        .declare(Invariance::one_row(m))
    }

    /// The fused router prefix of rank count `tp`.
    fn fused_prefix(tp: u32) -> String {
        format!("chop_moe_{}_router_route", FUSED_MODEL.at(tp).tag())
    }

    /// The fused entry of bucket `m` at `tp`, declared as exact verify
    /// rows need: a bucket's one-row bits with lower-index ties.
    fn fused_entry(tp: u32, m: u32) -> FusedRouterEntry {
        FusedRouterEntry {
            symbol: format!("{}_t{m}_v1", fused_prefix(tp)),
            gemm: router_gemv(m),
            invariance: Invariance::one_row(m).with_ties(Ties::LowerIndex),
        }
    }

    fn fused_backend(tp: u32) -> Sm120 {
        FUSED_BUCKETS.into_iter().fold(Sm120::new(), |backend, m| {
            backend.with_fused_router(fused_entry(tp, m))
        })
    }

    fn projected_buffers() -> ProjectedRouteBuffers<View> {
        ProjectedRouteBuffers {
            x: view(20, BIG),
            weight: view(21, BIG),
            route: buffers(),
            partials: view(22, BIG),
            semaphores: view(23, BIG),
        }
    }

    /// What a model's lowering of the fused router gives at bucket `m`: the
    /// fused symbol at the bucket's rows, planned as its GEMV entry at the
    /// bucket with the entry's K splits, and its fourteen buffers in the
    /// fused entry's order (`projected_buffers`' handles) with scale one.
    fn lowered(tp: u32, m: u32) -> Launch {
        let key = &router_gemv(m).key;
        let plan = gemm::plan(
            key,
            RuntimeCounts {
                m,
                batch: m,
                context: 0,
                routed_rows: 0,
                split_k: gemm::choose_split_k(key).unwrap(),
            },
        )
        .unwrap();
        Launch {
            symbol: format!("{}_t{m}_v1", fused_prefix(tp)),
            rows: m,
            plan,
            arguments: fused_router_arguments([21, 20, 1, 22, 23, 13, 2, 3, 4, 5, 6, 7, 8, 9], 1.0)
                .unwrap(),
            collective: false,
            persistent: false,
        }
    }

    /// The fused router at each decode bucket and both rank counts launches
    /// as the model's lowering does, at every live row count, and claims
    /// the one-row bits and ties its declaration states, for the routed
    /// operation.
    #[test]
    fn projected_routes_equal_the_lowering_at_each_bucket() {
        let (router, routes) = sigmoid256();
        let exact = |m| {
            Invariance::one_row(m)
                .with_ties(Ties::LowerIndex)
                .of_routed_operation()
        };
        for tp in [4, 2] {
            let backend = fused_backend(tp);
            for m in FUSED_BUCKETS {
                let need = Need::rows(m).with_exact(exact(m));
                let op = backend
                    .moe_projected_route(
                        &router_projection(m),
                        &router,
                        &routes,
                        0,
                        16,
                        use_of(need),
                        need,
                    )
                    .unwrap();
                let expected = lowered(tp, m);
                let info = op.info();
                assert_eq!(
                    info.symbols,
                    vec![(expected.symbol.clone(), expected.plan.dynamic_shared_bytes)]
                );
                assert!(info.guarantees.provides(exact(m)));
                assert_eq!(info.guarantees.rows, Rows::OneRow { max: m });
                assert_eq!(info.guarantees.scope, Scope::RoutedOperation);
                assert_eq!(
                    info.bytes_of("partials"),
                    Some(expected.plan.workspace_bytes)
                );
                assert!(expected.plan.workspace_bytes > 0);
                assert_eq!(info.bytes_of("semaphores"), Some(4));
                assert_eq!(info.bytes_of("route.routes.active_experts"), Some(257 * 4));
                assert_eq!(
                    info.bytes_of("route.routes.sorted_rows"),
                    Some(u64::from(padded_capacity(m * 8, 256, 16).unwrap()) * 4)
                );
                assert_eq!(
                    info.layout_of("weight").map(|layout| layout.id.as_str()),
                    Some("bf16-fragment")
                );
                assert_eq!(info.workspace, 0);
                for rows in 1..=m {
                    assert_eq!(
                        op.launches(&projected_buffers(), View::EMPTY, Counts::rows(rows))
                            .unwrap(),
                        vec![expected.clone()],
                        "{tp:?} bucket {m}, {rows} rows"
                    );
                }
            }
            // A use between buckets takes the least bucket that holds it.
            let need = Need::rows(3);
            let op = backend
                .moe_projected_route(
                    &router_projection(3),
                    &router,
                    &routes,
                    0,
                    16,
                    use_of(need),
                    need,
                )
                .unwrap();
            assert_eq!(
                op.launches(&projected_buffers(), View::EMPTY, Counts::rows(3))
                    .unwrap(),
                vec![lowered(tp, 4)]
            );
            assert_eq!(op.info().guarantees.rows, Rows::OneRow { max: 3 });
        }
    }

    /// The fused routes refuse what they do not compute, rows past their
    /// buckets or capacities, exactness their declarations do not claim,
    /// malformed registrations and short or missing views.
    #[test]
    fn projected_routes_refuse_what_they_do_not_serve() {
        let backend = fused_backend(4);
        let (router, routes) = sigmoid256();
        let prepare = |backend: &Sm120,
                       projection: GemmQuery,
                       router: &RowQuery,
                       routes: &MoeQuery,
                       rank,
                       tile,
                       need: Need| {
            backend
                .moe_projected_route(&projection, router, routes, rank, tile, use_of(need), need)
                .map(|_| ())
        };
        let rule = |result: Result<(), Refused>, text: &str| {
            assert!(
                matches!(&result, Err(Refused::Rule(rule)) if rule.contains(text)),
                "{text}: {result:?}"
            );
        };
        let need = Need::rows(4);
        let projection = router_projection(4);
        assert!(prepare(&backend, projection, &router, &routes, 0, 16, need).is_ok());
        // Another router kind or geometry: no fused entry computes it.
        let softmax256 = RowQuery::Router {
            experts: 256,
            top_k: 8,
            kind: RouterKind::SoftmaxTopK { renormalize: true },
        };
        rule(
            prepare(&backend, projection, &softmax256, &routes, 0, 16, need),
            "no SM120 projected route entry",
        );
        let sharded = MoeQuery::LocalRoutes {
            experts: 256,
            local_experts: 128,
            top_k: 8,
            capacity: PREFILL,
        };
        rule(
            prepare(&backend, projection, &router, &sharded, 0, 16, need),
            "no SM120 projected route entry",
        );
        rule(
            prepare(&backend, projection, &router, &routes, 1, 16, need),
            "past the expert shards",
        );
        // A projection of other logits, or of another hidden.
        for other in [
            GemmQuery {
                output: DType::Bf16,
                ..projection
            },
            GemmQuery {
                n: 128,
                ..projection
            },
        ] {
            rule(
                prepare(&backend, other, &router, &routes, 0, 16, need),
                "F32 logits of every expert",
            );
        }
        rule(
            prepare(
                &backend,
                GemmQuery {
                    k: 2048,
                    ..projection
                },
                &router,
                &routes,
                0,
                16,
                need,
            ),
            "no SM120 projected route entry",
        );
        // The tail pads each expert's routes to 16 rows.
        rule(
            prepare(&backend, projection, &router, &routes, 0, 64, need),
            "pad to the route tile",
        );
        // Capacities: past every bucket, past the projection's rows or the
        // routes', past MAX_CAPACITY, or another than the use's.
        rule(
            prepare(
                &backend,
                router_projection(16),
                &router,
                &routes,
                0,
                16,
                Need::rows(9),
            ),
            "exceed every SM120 projected route entry's capacity",
        );
        rule(
            prepare(
                &backend,
                router_projection(4),
                &router,
                &routes,
                0,
                16,
                Need::rows(8),
            ),
            "router projection's rows",
        );
        let few = MoeQuery::LocalRoutes {
            experts: 256,
            local_experts: 256,
            top_k: 8,
            capacity: 2,
        };
        rule(
            prepare(&backend, projection, &router, &few, 0, 16, need),
            "local routes' capacity",
        );
        rule(
            prepare(
                &backend,
                router_projection(MAX_CAPACITY + 1),
                &router,
                &routes,
                0,
                16,
                Need::rows(MAX_CAPACITY + 1),
            ),
            "1 to MAX_CAPACITY",
        );
        assert!(matches!(
            backend.moe_projected_route(
                &projection,
                &router,
                &routes,
                0,
                16,
                use_of(Need::rows(8)),
                need
            ),
            Err(Refused::Rule(rule)) if rule.contains("its use's capacity")
        ));
        // Exactness the declarations do not claim: chunk invariance, or
        // one-row bits from a bucket that declares none (both asked of the
        // routed operation, the op's scope).
        assert!(matches!(
            prepare(
                &backend,
                projection,
                &router,
                &routes,
                0,
                16,
                need.with_exact(Invariance::PREFILL.of_routed_operation())
            ),
            Err(Refused::Inexact { .. })
        ));
        let plain = fused_backend(4).with_fused_router(FusedRouterEntry {
            invariance: Invariance::NONE,
            ..fused_entry(4, 16)
        });
        let sixteen = Need::rows(16);
        assert!(
            prepare(
                &plain,
                router_projection(16),
                &router,
                &routes,
                0,
                16,
                sixteen
            )
            .is_ok()
        );
        assert!(matches!(
            prepare(
                &plain,
                router_projection(16),
                &router,
                &routes,
                0,
                16,
                sixteen.with_exact(Invariance::one_row(16).of_routed_operation())
            ),
            Err(Refused::Inexact { guaranteed, .. })
                if guaranteed == Invariance::NONE.of_routed_operation()
        ));
        // A registration the fused cubin does not compile serves nothing:
        // a symbol of another bucket than its GEMV's, or a GEMV that is not
        // the fragment router.
        let renamed = Sm120::new().with_fused_router(FusedRouterEntry {
            symbol: format!("{}_t8_v1", fused_prefix(4)),
            ..fused_entry(4, 4)
        });
        rule(
            prepare(&renamed, projection, &router, &routes, 0, 16, need),
            "another token bucket",
        );
        let mut decode = fused_entry(4, 4);
        decode.gemm.instantiation.key.dense = Some(DenseKernel::FragmentDecode { splits: 4 });
        let decode = Sm120::new().with_fused_router(decode);
        rule(
            prepare(&decode, projection, &router, &routes, 0, 16, need),
            "fragment router GEMV",
        );
        // Launches: row counts outside the capacity, and every view one byte
        // short (and every optional role left out) of the least bytes the
        // launch at four rows takes, which serve.
        let op = backend
            .moe_projected_route(&projection, &router, &routes, 0, 16, use_of(need), need)
            .unwrap();
        let full = projected_buffers();
        assert!(op.launches(&full, View::EMPTY, Counts::rows(0)).is_err());
        assert!(op.launches(&full, View::EMPTY, Counts::rows(5)).is_err());
        let partials = op.info().bytes_of("partials").unwrap();
        let sorted = u64::from(padded_capacity(4 * 8, 256, 16).unwrap()) * 4;
        let least = ProjectedRouteBuffers {
            x: view(20, 4 * 4096 * 2),
            weight: view(21, 256 * 4096 * 2),
            route: RouteBuffers {
                logits: view(1, 4 * 256 * 4),
                bias: Some(view(2, 256 * 4)),
                ids: view(3, 4 * 8 * 4),
                weights: view(4, 4 * 8 * 4),
                routes: RouteMetadata {
                    counts: Some(view(5, 256 * 4)),
                    offsets: view(6, 257 * 4),
                    sorted_rows: view(7, sorted),
                    padded_rows: view(8, 4),
                    active_experts: Some(view(9, 257 * 4)),
                },
                tile_counts: None,
                tile_offsets: None,
                ticket: None,
                live_rows: view(13, 4),
            },
            partials: view(22, partials),
            semaphores: view(23, 4),
        };
        assert_eq!(
            op.launches(&least, View::EMPTY, Counts::rows(4)).unwrap(),
            vec![lowered(4, 4)]
        );
        type Shorten = fn(&mut ProjectedRouteBuffers<View>);
        let short: [(&str, Shorten); 17] = [
            ("x", |b| b.x = view(20, b.x.bytes - 1)),
            ("weight", |b| b.weight = view(21, b.weight.bytes - 1)),
            ("partials", |b| b.partials = view(22, b.partials.bytes - 1)),
            ("semaphores", |b| b.semaphores = view(23, 3)),
            ("logits", |b| b.route.logits = view(1, 4 * 256 * 4 - 1)),
            ("bias", |b| b.route.bias = Some(view(2, 256 * 4 - 1))),
            ("bias left out", |b| b.route.bias = None),
            ("ids", |b| b.route.ids = view(3, 4 * 8 * 4 - 1)),
            ("weights", |b| b.route.weights = view(4, 4 * 8 * 4 - 1)),
            ("counts", |b| {
                b.route.routes.counts = Some(view(5, 256 * 4 - 1))
            }),
            ("counts left out", |b| b.route.routes.counts = None),
            ("offsets", |b| b.route.routes.offsets = view(6, 257 * 4 - 1)),
            ("sorted rows", |b| {
                let v = b.route.routes.sorted_rows;
                b.route.routes.sorted_rows = view(v.handle, v.bytes - 1)
            }),
            ("padded rows", |b| b.route.routes.padded_rows = view(8, 3)),
            ("active experts", |b| {
                b.route.routes.active_experts = Some(view(9, 257 * 4 - 1))
            }),
            ("active experts left out", |b| {
                b.route.routes.active_experts = None
            }),
            ("live rows", |b| b.route.live_rows = view(13, 3)),
        ];
        for (name, shorten) in short {
            let mut buffers = least;
            shorten(&mut buffers);
            assert!(
                op.launches(&buffers, View::EMPTY, Counts::rows(4)).is_err(),
                "{name}"
            );
        }
    }

    /// The op writes route metadata, which depends on every route of the
    /// launch, so its claims are the routed operation's: a need in that scope
    /// is served, the same need about the entry's own outputs is refused,
    /// whatever scope the model declared the entry in.
    #[test]
    fn projected_route_claims_are_about_the_routed_operation() {
        let (router, routes) = sigmoid256();
        let need = Need::rows(4);
        let exact = Invariance::one_row(4).with_ties(Ties::LowerIndex);
        for declared in [exact, exact.of_routed_operation()] {
            let backend = Sm120::new().with_fused_router(FusedRouterEntry {
                invariance: declared,
                ..fused_entry(4, 4)
            });
            let prepare = |need: Need| {
                backend.moe_projected_route(
                    &router_projection(4),
                    &router,
                    &routes,
                    0,
                    16,
                    use_of(need),
                    need,
                )
            };
            let op = prepare(need.with_exact(exact.of_routed_operation())).unwrap();
            assert_eq!(op.info().guarantees, exact.of_routed_operation());
            assert_eq!(
                op.launches(&projected_buffers(), View::EMPTY, Counts::rows(2))
                    .unwrap(),
                vec![lowered(4, 4)]
            );
            for entry_scoped in [
                exact,
                Invariance::one_row(4),
                Invariance::NONE.with_ties(Ties::LowerIndex),
            ] {
                assert!(
                    matches!(
                        prepare(need.with_exact(entry_scoped)),
                        Err(Refused::Inexact { guaranteed, .. })
                            if guaranteed.scope == Scope::RoutedOperation
                                && guaranteed.provides(exact.of_routed_operation())
                    ),
                    "{declared:?} {entry_scoped:?}"
                );
            }
            // A need of nothing is served in any scope.
            assert!(prepare(need).is_ok());
        }
    }

    /// A registration that does not plan (here a router GEMV of three K
    /// splits, which the fragment router does not compile) serves nothing:
    /// beside a valid entry of the same bucket and symbol, in either order,
    /// the valid one serves; alone, or beside another malformed one, the op
    /// is refused with the same reason in every order.
    #[test]
    fn malformed_projected_routes_do_not_block_valid_ones() {
        let (router, routes) = sigmoid256();
        let need = Need::rows(4);
        let valid = fused_entry(4, 4);
        let mut splits = valid.clone();
        splits.gemm.instantiation.key.dense = Some(DenseKernel::FragmentRouter { splits: 3 });
        let renamed = FusedRouterEntry {
            symbol: format!("{}_t8_v1", fused_prefix(4)),
            ..valid.clone()
        };
        let prepare = |entries: &[FusedRouterEntry]| {
            entries
                .iter()
                .cloned()
                .fold(Sm120::new(), Sm120::with_fused_router)
                .moe_projected_route(
                    &router_projection(4),
                    &router,
                    &routes,
                    0,
                    16,
                    use_of(need),
                    need,
                )
        };
        for order in crate::backend::select::orders(&[valid.clone(), splits.clone()]) {
            let op = prepare(&order).unwrap();
            assert_eq!(
                op.launches(&projected_buffers(), View::EMPTY, Counts::rows(4))
                    .unwrap(),
                vec![lowered(4, 4)]
            );
        }
        let refusal = |entries: &[FusedRouterEntry]| match prepare(entries) {
            Err(Refused::Rule(rule)) => rule,
            other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
        };
        assert!(refusal(std::slice::from_ref(&splits)).contains("in 4 splits"));
        let reasons = crate::backend::select::orders(&[splits, renamed])
            .iter()
            .map(|order| refusal(order))
            .collect::<Vec<_>>();
        assert!(
            reasons.iter().all(|reason| *reason == reasons[0]),
            "{reasons:?}"
        );
    }

    /// The entry a fused route takes depends on the op alone: every order
    /// of a rank count's registrations, and duplicates, prepare the same
    /// launches; one rank count's entries beside the other's are two
    /// launches of one bucket, which the op does not choose between.
    #[test]
    fn projected_routes_do_not_depend_on_the_registration_order() {
        let (router, routes) = sigmoid256();
        let entries = FUSED_BUCKETS.map(|m| fused_entry(4, m));
        for mut order in crate::backend::select::orders(&entries) {
            order.push(order[0].clone());
            let backend = order
                .into_iter()
                .fold(Sm120::new(), Sm120::with_fused_router);
            for m in FUSED_BUCKETS {
                let need = Need::rows(m);
                let op = backend
                    .moe_projected_route(
                        &router_projection(m),
                        &router,
                        &routes,
                        0,
                        16,
                        use_of(need),
                        need,
                    )
                    .unwrap();
                assert_eq!(
                    op.launches(&projected_buffers(), View::EMPTY, Counts::rows(1))
                        .unwrap(),
                    vec![lowered(4, m)]
                );
            }
        }
        let both = fused_backend(4).with_fused_router(fused_entry(2, 4));
        let need = Need::rows(4);
        assert!(matches!(
            both.moe_projected_route(
                &router_projection(4),
                &router,
                &routes,
                0,
                16,
                use_of(need),
                need
            ),
            Err(Refused::Rule(rule))
                if rule.contains(&fused_entry(2, 4).symbol)
                    && rule.contains(&fused_entry(4, 4).symbol)
        ));
    }

    /// The fused router's key holds what it launches, every field of the
    /// entry, its GEMV entry and that entry's instantiation changed alone:
    /// the fused symbol, never the claims, the GEMV's profile, family or
    /// symbol; the GEMV key selects the op.
    #[test]
    fn fused_router_keys_hold_what_they_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let nested = |prefix: &str, names: Vec<&'static str>| {
            names
                .into_iter()
                .map(|name| &*Box::leak(format!("{prefix}.{name}").into_boxed_str()))
                .collect::<Vec<&'static str>>()
        };
        let mut names = field_names!(FusedRouterEntry {
            symbol,
            gemm,
            invariance
        });
        names.retain(|name| *name != "gemm");
        let mut gemm_names = nested(
            "gemm",
            field_names!(GemmVariant {
                profile,
                instantiation,
                invariance
            }),
        );
        gemm_names.retain(|name| *name != "gemm.instantiation");
        names.extend(gemm_names);
        names.extend(nested(
            "gemm.instantiation",
            field_names!(InstantiationSpec {
                family,
                key,
                symbol
            }),
        ));
        let other = Profile::synthetic(crate::ProfileNames {
            name: "fused-router-other",
            tag: "other",
        });
        let fields: Vec<Field<FusedRouterEntry>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = e.symbol.replace("_router_route", "_twin_router_route")),
            ),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = Invariance::NONE),
            ),
            (
                "gemm.profile",
                Class::Inert,
                Box::new(move |e| e.gemm.profile = other),
            ),
            (
                "gemm.invariance",
                Class::Inert,
                Box::new(|e| e.gemm.invariance = Invariance::NONE),
            ),
            (
                "gemm.instantiation.family",
                Class::Inert,
                Box::new(|e| e.gemm.instantiation.family = kernel_core::contract::OpFamily::Row),
            ),
            // A router bucket's key admits no other knobs (`gemm::shape`),
            // so a changed key computes another projection; the launch
            // equality test pins the plan it gives.
            (
                "gemm.instantiation.key",
                Class::Selects,
                Box::new(|e| {
                    if let StaticQuery::Gemm(query) = &mut e.gemm.instantiation.key.query {
                        query.k /= 2;
                    }
                }),
            ),
            (
                "gemm.instantiation.symbol",
                Class::Inert,
                Box::new(|e| e.gemm.instantiation.symbol = "chop_gemm_twin_v1".into()),
            ),
        ];
        let (router, routes) = sigmoid256();
        let need = Need::rows(4);
        keys_follow_launches(
            &fused_entry(4, 4),
            names,
            fields,
            |e| Kernel::fused(e).unwrap().launch(),
            |e| {
                let op = Sm120::new()
                    .with_fused_router(e.clone())
                    .moe_projected_route(
                        &router_projection(4),
                        &router,
                        &routes,
                        0,
                        16,
                        use_of(need),
                        need,
                    )?;
                let prefix = e.symbol.strip_suffix("_t4_v1").unwrap();
                let spec = fused_router_cubin_spec(prefix)
                    .entries
                    .into_iter()
                    .find(|spec| spec.symbol == e.symbol)
                    .unwrap();
                let code = format!("{:?} {}", spec.params, spec.body);
                let launches = [1, 4].map(|rows| {
                    op.launches(&projected_buffers(), View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code, launches))
            },
        );
    }
}
