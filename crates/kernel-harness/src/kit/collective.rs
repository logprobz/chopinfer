//! The kit's multi-rank runner: collective ops a backend prepares on every
//! rank of a group, run on one GPU per rank and checked against the
//! contracts' oracles.
//!
//! A collective's role record on rank `r` names its own buffers and every
//! rank's views (`collective::RankViews`): each leaf is a [`Slot`], a
//! buffer of the rank's own ([`Slot::Own`] over a [`Spec`]), a persistent
//! state region of its own ([`Slot::State`], `OpInfo::state`), or another
//! rank's buffer by that rank's path ([`Slot::Peer`]). The runner
//! ([`conform`]) prepares the op on every rank through the backend,
//! allocates every rank's buffers on its session (peer access enabled, as
//! [`crate::gpu_test_multi`] opens them), zeroes the state regions once,
//! and runs each step by launching every rank's `k`th launch before any
//! rank's next, so a collective launch waits only for peers that were
//! launched. Its checks per op:
//!
//! - `declaration` (CPU): an oracle; each rank's launches at every row
//!   count against its `OpInfo` (symbols, dynamic shared memory, rows,
//!   every role it sizes bound, every state region bound at its bytes,
//!   every handle a buffer of some rank); every rank launches the same
//!   symbols with the same plans and collective flags, at least one
//!   collective; every rank's info claims the same guarantees and state.
//! - `need` (CPU): the guarantees provide the needed exactness.
//! - `conformance` (GPU): at each live count (the default counts, both
//!   sides of the comm protocols' row thresholds and the counts the case
//!   adds for its op's own thresholds), two steps in a row over the
//!   case's images and over them with every row input's rows reversed, on
//!   the same state (so the steps' epochs and flags advance as a serving
//!   step's): every rank's outputs keep their fill outside their write
//!   maps and match the oracle, and the outputs the case names hold the
//!   same bits on every rank.
//! - `live below rows` (GPU, for a case that asks with
//!   [`CollectiveCase::live_below`]): each rank's launches of a planned row
//!   count captured into a graph and replayed with smaller device live
//!   counts, zero included, one after another on the state the
//!   conformance run left, as a served bucket's graph replays: the rows
//!   below the live count match the oracle at that count and hold the same
//!   bits on every rank, rows past the planned count keep their fill, and
//!   rows between are the contract's to leave or write.
//! - one property check per claimed or needed property
//!   ([`super::properties`]) over the rows of every rank at once (a row
//!   moves on every rank together), at every live count of two rows or
//!   more as the launch's rows, since the entries' forms change with the
//!   rows.
//!
//! Not exercised: PDL dependency order (the kit's producers are one
//! rank's), the null launch (a collective's chain drops no stage alone),
//! and graph replay and a device live count below the rows except in the
//! `live below rows` check.

use super::{Check, Leaf, Lookup, Modules, OpReport, OpResult, Report, Spec, Tag, properties};
use crate::conformance::{Case, Expected, Fill, Images, Results, Rows};
use crate::{LabResult, Session, launch, poison};
use kernel_core::allocation::Traversal;
use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, Refused, View};
use kernel_core::contract::Argument;
use kernel_core::properties::Property;
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// One line of a GPU run's progress on stderr, with the seconds since the
/// first: a run the watchdog stops names the op and step it was in.
fn progress(step: std::fmt::Arguments<'_>) {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    eprintln!(
        "kit collectives [{:.1} s]: {step}",
        start.elapsed().as_secs_f64()
    );
}

/// One leaf of a rank's role record.
pub enum Slot<'a> {
    /// A buffer of the rank's own.
    Own(Spec<'a>),
    /// A persistent state region of the rank's own, of these bytes: zeroed
    /// once before the first step and left to the op.
    State(u64),
    /// Rank `rank`'s buffer at its own path.
    Peer(u32, String),
}

impl Slot<'_> {
    /// Rank `rank`'s buffer at `path`.
    pub fn peer(rank: u32, path: impl Into<String>) -> Self {
        Self::Peer(rank, path.into())
    }
}

/// Every rank's expected results at a live count, by rank, over every
/// rank's images.
type Oracle<'a> = Box<dyn Fn(&[Images], u32) -> LabResult<Vec<Vec<(String, Expected)>>> + 'a>;

/// A rank's launches at a row count, over views the first argument gives
/// by rank and path.
type RankLaunches<'a> =
    Box<dyn Fn(&dyn Fn(u32, &str) -> LabResult<View>, u32) -> LabResult<Vec<Launch>> + 'a>;

/// One rank's part of a collective case.
struct Rank<'a> {
    /// Its own buffers, which judge its results.
    case: Case<'a>,
    /// Its own buffers' leaves, state regions included, and their bytes.
    leaves: Vec<Leaf>,
    bytes: BTreeMap<String, u64>,
    /// Its state regions' paths and bytes, in record order.
    states: Vec<(String, u64)>,
    info: OpInfo,
    launches: RankLaunches<'a>,
}

/// A collective op's case on every rank of a group.
pub struct CollectiveCase<'a> {
    ranks: Vec<Rank<'a>>,
    capacity: u32,
    oracle: Option<Oracle<'a>>,
    /// Outputs whose bytes every rank must hold alike.
    same: Vec<String>,
    /// The property checks that do not apply, with their reasons.
    excluded: BTreeMap<Property, String>,
    /// Live counts the case adds to [`PROTOCOL_COUNTS`].
    counts: Vec<u32>,
    /// Planned row counts and the device live counts each replays with
    /// (`live below rows`).
    live_below: Vec<(u32, Vec<u32>)>,
}

/// The check of launches replayed with fewer live rows than planned.
const LIVE_BELOW: &str = "live below rows";

/// Row counts on both sides of the comm protocols' thresholds: low latency
/// up to 2 rows, one-shot up to 16 rows (four ranks) or 32, push entries up
/// to 32 rows and wide or phased entries above.
const PROTOCOL_COUNTS: [u32; 7] = [1, 2, 3, 16, 17, 32, 33];

/// The device live count's name in a rank's case that declares none.
const NO_LIVE: &str = "#no device live count";

impl<'a> CollectiveCase<'a> {
    /// The case of `ops` (rank `r`'s prepared op at `r`) at `capacity`
    /// rows, over each rank's record of slots, launched with
    /// [`Counts::rows`].
    pub fn new<S, R, B>(ops: Vec<Prepared<B>>, capacity: u32, records: Vec<S>) -> LabResult<Self>
    where
        S: Traversal<Slot<'a>, Mapped<usize> = R>,
        R: Traversal<usize, Mapped<View> = B> + Clone + 'a,
        B: 'a,
    {
        if ops.len() != records.len() || ops.len() < 2 {
            return Err(
                "a collective case takes one op and one record per rank, two or more".into(),
            );
        }
        let mut ranks = Vec::with_capacity(ops.len());
        for (rank, (op, record)) in ops.into_iter().zip(records).enumerate() {
            let rank = rank as u32;
            let info = op.info().clone();
            if info.workspace > 0 {
                return Err("the collective runner binds no generic workspace".into());
            }
            let mut slots = Vec::new();
            let places = record.map(|slot| {
                slots.push(slot);
                slots.len() - 1
            });
            let mut paths = vec![String::new(); slots.len()];
            places.visit_named(&mut |path, &at| paths[at] = path.to_owned());
            let symbols = info
                .symbols
                .iter()
                .map(|(symbol, _)| symbol.clone())
                .collect();
            let mut case = Case::prepared(
                symbols,
                capacity,
                Box::new(|_, _| Err("the collective runner launches a collective's ranks".into())),
            );
            let (mut leaves, mut bytes, mut states) = (Vec::new(), BTreeMap::new(), Vec::new());
            let mut targets = Vec::with_capacity(slots.len());
            let mut counted = false;
            for (path, slot) in paths.iter().zip(slots) {
                match slot {
                    Slot::Own(spec) => {
                        let leaf = spec.leaf(path)?;
                        let rows = match leaf.rows {
                            Rows::Live => capacity as usize,
                            Rows::Fixed(rows) => rows,
                        };
                        counted |= leaf.tag == Tag::Live;
                        bytes.insert(path.clone(), (rows * leaf.row_bytes) as u64);
                        leaves.push(leaf);
                        case = spec.bind(case, path, capacity)?;
                        targets.push((rank, path.clone()));
                    }
                    Slot::State(size) => {
                        leaves.push(Leaf {
                            path: path.clone(),
                            rows: Rows::Fixed(1),
                            row_bytes: size as usize,
                            tag: Tag::Scratch,
                            route_metadata: false,
                        });
                        bytes.insert(path.clone(), size);
                        states.push((path.clone(), size));
                        targets.push((rank, path.clone()));
                    }
                    Slot::Peer(peer, path) => targets.push((peer, path)),
                }
            }
            if !counted {
                case = case.live(NO_LIVE);
            }
            let launches: RankLaunches<'a> = Box::new(move |view, rows| {
                let views = places.clone().try_map(&mut |at: usize| {
                    let (rank, path) = &targets[at];
                    view(*rank, path)
                })?;
                op.launches(&views, View::EMPTY, Counts::rows(rows))
                    .map_err(|refusal| format!("the op refuses {rows} rows: {refusal}"))
            });
            ranks.push(Rank {
                case,
                leaves,
                bytes,
                states,
                info,
                launches,
            });
        }
        Ok(Self {
            ranks,
            capacity,
            oracle: None,
            same: Vec::new(),
            excluded: BTreeMap::new(),
            counts: Vec::new(),
            live_below: Vec::new(),
        })
    }

    /// Every rank's expected results at each live count, by rank: they
    /// must cover every output that writes there.
    pub fn oracle(
        mut self,
        oracle: impl Fn(&[Images], u32) -> LabResult<Vec<Vec<(String, Expected)>>> + 'a,
    ) -> Self {
        self.oracle = Some(Box::new(oracle));
        self
    }

    /// Outputs whose bytes every rank must hold alike.
    pub fn same_on_every_rank(mut self, paths: &[&str]) -> Self {
        self.same = paths.iter().map(|&path| path.to_owned()).collect();
        self
    }

    /// Steps through `counts` too (up to the capacity): both sides of the
    /// op's own row thresholds.
    pub fn counts(mut self, counts: &[u32]) -> Self {
        self.counts.extend_from_slice(counts);
        self
    }

    /// Runs the `live below rows` check over `schedule`: each planned row
    /// count (at most the capacity) with the device live counts its graph
    /// replays with, in order, each at most the planned count. Only for
    /// exact oracles ([`Expected::Bits`]) whose outputs are `[rows, ...]`.
    pub fn live_below(mut self, schedule: &[(u32, &[u32])]) -> Self {
        self.live_below.extend(
            schedule
                .iter()
                .map(|&(planned, lives)| (planned, lives.to_vec())),
        );
        self
    }

    /// Reports the checks of `properties` as not exercised for `reason`,
    /// where the case's rows cannot show them.
    pub fn without_properties(mut self, properties: &[Property], reason: &str) -> Self {
        for &property in properties {
            self.excluded.insert(property, reason.into());
        }
        self
    }

    /// What the backend chose for rank 0's op.
    pub fn info(&self) -> &OpInfo {
        &self.ranks[0].info
    }

    /// Rank `rank`'s view of `path` with its token as the handle: rank
    /// `rank`'s leaves numbered from `rank * 2^20 + 1`.
    fn token(&self, rank: u32, path: &str) -> LabResult<View> {
        let own = self
            .ranks
            .get(rank as usize)
            .ok_or_else(|| format!("no rank {rank}"))?;
        let at = own
            .leaves
            .iter()
            .position(|leaf| leaf.path == path)
            .ok_or_else(|| format!("rank {rank} has no buffer {path}"))?;
        Ok(View {
            handle: (u64::from(rank) << 20) + at as u64 + 1,
            bytes: own.bytes[path],
        })
    }

    /// Every rank's launches at `rows` over token views.
    fn plans(&self, rows: u32) -> LabResult<Vec<Vec<Launch>>> {
        self.ranks
            .iter()
            .enumerate()
            .map(|(rank, own)| {
                (own.launches)(&|rank, path| self.token(rank, path), rows)
                    .map_err(|e| format!("rank {rank}: {e}"))
            })
            .collect()
    }

    /// Checks the declaration on the CPU (see the module docs).
    pub fn validate(&self) -> LabResult<Value> {
        if self.oracle.is_none() {
            return Err("the case declares no oracle".into());
        }
        let first = &self.ranks[0].info;
        for (rank, own) in self.ranks.iter().enumerate() {
            let info = &own.info;
            if (info.guarantees, &info.symbols, &info.state)
                != (first.guarantees, &first.symbols, &first.state)
            {
                return Err(format!(
                    "rank {rank}'s op claims other guarantees, symbols or state than rank 0's"
                ));
            }
            if own.states.len() != info.state.len()
                || own
                    .states
                    .iter()
                    .zip(&info.state)
                    .any(|((_, held), need)| held < need)
            {
                return Err(format!(
                    "rank {rank} binds {} state regions; the op lists {:?} bytes",
                    own.states.len(),
                    info.state
                ));
            }
            for (path, bytes) in &info.bytes {
                let held = own.bytes.get(path).ok_or_else(|| {
                    format!("the op sizes {path}, which rank {rank} does not bind")
                })?;
                if held < bytes {
                    return Err(format!(
                        "rank {rank}'s {path} holds {held} bytes of {bytes}"
                    ));
                }
            }
        }
        let known = |handle: u64| {
            let rank = (handle >> 20) as usize;
            let at = (handle & ((1 << 20) - 1)) as usize;
            self.ranks
                .get(rank)
                .is_some_and(|own| at >= 1 && at <= own.leaves.len())
        };
        // Every row count's launches: the forms differ by rows.
        let mut launched = Vec::new();
        for rows in 1..=self.capacity {
            let plans = self.plans(rows)?;
            for (rank, launches) in plans.iter().enumerate() {
                let info = &self.ranks[rank].info;
                let at = |e: String| format!("rank {rank} at {rows} rows: {e}");
                if launches.len() != plans[0].len() {
                    return Err(at(format!(
                        "{} launches, rank 0 {}",
                        launches.len(),
                        plans[0].len()
                    )));
                }
                for (launch, other) in launches.iter().zip(&plans[0]) {
                    let symbol = &launch.symbol;
                    let Some(&(_, shared)) = info.symbols.iter().find(|(s, _)| s == symbol) else {
                        return Err(at(format!(
                            "{symbol} is not among the symbols the op lists"
                        )));
                    };
                    if launch.plan.dynamic_shared_bytes > shared {
                        return Err(at(format!(
                            "{symbol} plans more dynamic shared memory than listed"
                        )));
                    }
                    if launch.rows == 0 {
                        return Err(at(format!("{symbol} records no rows")));
                    }
                    if (symbol, launch.plan, launch.collective)
                        != (&other.symbol, other.plan, other.collective)
                    {
                        return Err(at(format!("{symbol} is unlike rank 0's {}", other.symbol)));
                    }
                    if let Some(handle) = handles(&launch.arguments)
                        .into_iter()
                        .find(|&handle| !known(handle))
                    {
                        return Err(at(format!(
                            "{symbol} takes handle {handle}, no rank's buffer"
                        )));
                    }
                }
            }
            if !plans[0].iter().any(|launch| launch.collective) {
                return Err(format!("no launch at {rows} rows waits for its peers"));
            }
            for launch in &plans[0] {
                if !launched.contains(&launch.symbol) {
                    launched.push(launch.symbol.clone());
                }
            }
        }
        // The oracle at every live count the runs take, so a count it
        // cannot answer fails here rather than on a GPU.
        let oracle = self.oracle.as_ref().ok_or("the case declares no oracle")?;
        let images = self.images();
        let mut counts = self.live_counts();
        for (planned, lives) in &self.live_below {
            if *planned == 0 || *planned > self.capacity || lives.iter().any(|live| live > planned)
            {
                return Err(format!(
                    "{planned} planned rows with live counts {lives:?} past the plan or the \
                     capacity {}",
                    self.capacity
                ));
            }
            counts.extend(lives);
        }
        counts.sort_unstable();
        counts.dedup();
        for &live in &counts {
            let expected =
                oracle(&images, live).map_err(|e| format!("the oracle at live {live}: {e}"))?;
            if expected.len() != self.ranks.len() {
                return Err(format!(
                    "the oracle at live {live} gives no results for every rank"
                ));
            }
        }
        Ok(json!({
            "ranks": self.ranks.len(),
            "launches": launched,
            "state_bytes": first.state,
            "oracle_counts": counts,
        }))
    }

    /// Every rank's images.
    fn images(&self) -> Vec<Images> {
        self.ranks.iter().map(|own| own.case.images()).collect()
    }

    /// Every check (see the module docs) on `sessions` (one per rank) with
    /// each rank's functions.
    fn checks(&self, need: Need, sessions: Option<(&[Session], &[Box<Lookup<'_>>])>) -> Vec<Check> {
        let declaration = self.validate();
        let declared = declaration.is_ok();
        let mut checks = vec![Check::of("declaration", declaration)];
        let guaranteed = self.info().guarantees;
        checks.push(Check::of(
            super::NEED,
            if guaranteed.provides(need.exact) {
                Ok(json!({"needed": format!("{:?}", need.exact)}))
            } else {
                Err(format!(
                    "the op guarantees {guaranteed:?}, which does not provide the needed {:?}",
                    need.exact
                ))
            },
        ));
        let claimed = guaranteed.claims();
        let requested = need
            .exact
            .claims()
            .into_iter()
            .filter(|property| !claimed.contains(property))
            .collect::<Vec<_>>();
        for property in claimed.iter().chain(&requested) {
            if let Some(reason) = self.excluded.get(property) {
                checks.push(Check::not_exercised(&properties::name(*property), reason));
            }
        }
        let included = |properties: &[Property]| {
            properties
                .iter()
                .copied()
                .filter(|property| !self.excluded.contains_key(property))
                .collect::<Vec<_>>()
        };
        let (claimed, requested) = (included(&claimed), included(&requested));
        let (Some((sessions, functions)), true) = (sessions, declared) else {
            let reason = if declared {
                "no GPU session"
            } else {
                "the declaration fails"
            };
            checks.push(Check::not_exercised("conformance", reason));
            if !self.live_below.is_empty() {
                checks.push(Check::not_exercised(LIVE_BELOW, reason));
            }
            for property in claimed.iter().chain(&requested) {
                checks.push(Check::not_exercised(&properties::name(*property), reason));
            }
            return checks;
        };
        let rig = match Rig::new(self, sessions) {
            Ok(rig) => rig,
            Err(error) => {
                checks.push(Check::of("conformance", Err(error)));
                return checks;
            }
        };
        progress(format_args!("conformance"));
        checks.push(Check::of("conformance", self.conformance(&rig, functions)));
        if !self.live_below.is_empty() {
            checks.push(Check::of(LIVE_BELOW, self.below(&rig, functions)));
        }
        // Properties over every rank's rows at once, by `r<rank>.<path>`.
        let prefixed = |rank: usize, path: &str| format!("r{rank}.{path}");
        let leaves = self
            .ranks
            .iter()
            .enumerate()
            .flat_map(|(rank, own)| {
                own.leaves.iter().map(move |leaf| Leaf {
                    path: prefixed(rank, &leaf.path),
                    ..leaf.clone()
                })
            })
            .collect::<Vec<_>>();
        let mut images = Images::default();
        for (rank, own) in self.images().into_iter().enumerate() {
            for leaf in &self.ranks[rank].leaves {
                if let Ok(bytes) = own.bytes(&leaf.path) {
                    images = images.with(&prefixed(rank, &leaf.path), bytes.to_vec());
                }
            }
        }
        let split = |images: &Images| -> LabResult<Vec<Images>> {
            self.ranks
                .iter()
                .enumerate()
                .map(|(rank, own)| {
                    let mut mine = own.case.images();
                    for leaf in &own.leaves {
                        if let Ok(bytes) = images.bytes(&prefixed(rank, &leaf.path)) {
                            mine = mine.with(&leaf.path, bytes.to_vec());
                        }
                    }
                    Ok(mine)
                })
                .collect()
        };
        // Each property launches many steps: one line per change of rows.
        let last = std::cell::Cell::new((0_u32, 0_usize));
        let execute = |images: &Images, rows: u32| -> LabResult<Results> {
            let (previous, launched) = last.get();
            if rows != previous {
                progress(format_args!(
                    "property step of {rows} rows ({launched} steps so far)"
                ));
            }
            last.set((rows, launched + 1));
            let results = rig.step(self, functions, &split(images)?, rows)?;
            Ok(results
                .into_iter()
                .enumerate()
                .flat_map(|(rank, results)| {
                    results
                        .into_iter()
                        .map(move |(path, bytes)| (prefixed(rank, &path), bytes))
                })
                .collect())
        };
        let written = |state: &str, _: &Images, _: u32| -> LabResult<Vec<std::ops::Range<usize>>> {
            Err(format!("{state} is no in-place state"))
        };
        // Each property at every live count: the protocols change with the
        // rows.
        let counts = self.live_counts();
        progress(format_args!("property checks at live counts {counts:?}"));
        for (claims, list) in [(guaranteed, &claimed), (need.exact, &requested)] {
            checks.extend(properties::check_at(
                claims, list, &leaves, &images, &counts, &execute, &written,
            ));
        }
        checks
    }

    /// The live counts the conformance run and the property checks step
    /// through, up to the capacity: each positive count of
    /// [`poison::live_counts`], both sides of the comm protocols' row
    /// thresholds ([`PROTOCOL_COUNTS`]) and the case's own counts.
    fn live_counts(&self) -> Vec<u32> {
        let mut counts = poison::live_counts(self.capacity)
            .into_iter()
            .chain(PROTOCOL_COUNTS)
            .chain(self.counts.iter().copied())
            .filter(|&live| live > 0 && live <= self.capacity)
            .collect::<Vec<_>>();
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    /// The conformance run on `rig` (see the module docs).
    fn conformance(&self, rig: &Rig<'_>, functions: &[Box<Lookup<'_>>]) -> LabResult<Value> {
        let oracle = self.oracle.as_ref().ok_or("the case declares no oracle")?;
        let images = self.images();
        let reversed = images
            .iter()
            .zip(&self.ranks)
            .map(|(images, own)| reverse_rows(images, &own.leaves, self.capacity))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipts = Vec::new();
        for live in self.live_counts() {
            for (step, images) in [("images", &images), ("reversed rows", &reversed)] {
                progress(format_args!("conformance live {live}, {step}"));
                let at = |e: String| format!("live {live}, {step}: {e}");
                let results = rig.step(self, functions, images, live).map_err(at)?;
                let expected = oracle(images, live).map_err(at)?;
                if expected.len() != self.ranks.len() {
                    return Err(at("the oracle gives no results for every rank".into()));
                }
                for (rank, ((own, expected), results)) in
                    self.ranks.iter().zip(expected).zip(&results).enumerate()
                {
                    let receipt = own
                        .case
                        .judge(expected, &images[rank], live, results)
                        .map_err(|e| at(format!("rank {rank}: {e}")))?;
                    receipts
                        .push(json!({"live": live, "step": step, "rank": rank, "oracle": receipt}));
                }
                for path in &self.same {
                    let first = results[0]
                        .get(path)
                        .ok_or_else(|| at(format!("no {path}")))?;
                    for (rank, results) in results.iter().enumerate().skip(1) {
                        let bytes = results.get(path).ok_or_else(|| at(format!("no {path}")))?;
                        if let Some(byte) = bytes.iter().zip(first).position(|(a, b)| a != b) {
                            return Err(at(format!(
                                "{path} byte {byte} differs between rank {rank} and rank 0"
                            )));
                        }
                    }
                }
            }
        }
        Ok(json!({
            "live_counts": self.live_counts(),
            "steps_per_count": 2,
            "same_on_every_rank": self.same,
            "checks": receipts,
        }))
    }
}

impl CollectiveCase<'_> {
    /// The `live below rows` check on `rig` (see the module docs).
    fn below(&self, rig: &Rig<'_>, functions: &[Box<Lookup<'_>>]) -> LabResult<Value> {
        let oracle = self.oracle.as_ref().ok_or("the case declares no oracle")?;
        let images = self.images();
        let mut receipts = Vec::new();
        for (planned, lives) in &self.live_below {
            let planned = *planned;
            progress(format_args!("live below rows: capture {planned} rows"));
            let replays = rig.replays(self, functions, &images, planned, lives)?;
            for (&live, results) in lives.iter().zip(&replays) {
                let at = |e: String| format!("{planned} planned rows, live {live}: {e}");
                let expected = oracle(&images, live).map_err(at)?;
                if expected.len() != self.ranks.len() {
                    return Err(at("the oracle gives no results for every rank".into()));
                }
                let mut prefixes: BTreeMap<String, Vec<u8>> = BTreeMap::new();
                for (rank, ((own, expected), results)) in
                    self.ranks.iter().zip(expected).zip(results).enumerate()
                {
                    let fills = own
                        .case
                        .fills(&images[rank])?
                        .into_iter()
                        .filter_map(|(name, _, fill)| match fill {
                            Fill::Pattern(pattern) => Some((name, pattern)),
                            Fill::Image(_) => None,
                        })
                        .collect::<BTreeMap<_, _>>();
                    for (path, expected) in expected {
                        let at = |e: String| at(format!("rank {rank} {path}: {e}"));
                        let Expected::Bits(bits) = expected else {
                            return Err(at("the check compares exact oracles only".into()));
                        };
                        let leaf = own
                            .leaves
                            .iter()
                            .find(|leaf| leaf.path == path)
                            .ok_or_else(|| at("no such output".into()))?;
                        let bytes = results.get(&path).ok_or_else(|| at("not read".into()))?;
                        let active = live as usize * leaf.row_bytes;
                        let end = planned as usize * leaf.row_bytes;
                        if let Some(byte) = (0..active).find(|&i| bytes[i] != bits[i]) {
                            return Err(at(format!(
                                "byte {byte} is {:#04x}, the oracle's {:#04x}",
                                bytes[byte], bits[byte]
                            )));
                        }
                        let fill = fills
                            .get(&path)
                            .ok_or_else(|| at("has no fill pattern".into()))?;
                        if let Some(byte) =
                            (end..bytes.len()).find(|&i| bytes[i] != fill[i % fill.len()])
                        {
                            return Err(at(format!(
                                "byte {byte}, past the planned rows, was written"
                            )));
                        }
                        if self.same.contains(&path) {
                            let first = prefixes
                                .entry(path.clone())
                                .or_insert_with(|| bytes[..active].to_vec());
                            if first[..] != bytes[..active] {
                                return Err(at("differs from rank 0's live rows".into()));
                            }
                        }
                    }
                }
                receipts.push(json!({"planned": planned, "live": live}));
            }
        }
        Ok(json!({"replays": receipts, "captured": true}))
    }
}

/// `images` with the rows of every row input reversed over the capacity.
fn reverse_rows(images: &Images, leaves: &[Leaf], capacity: u32) -> LabResult<Images> {
    let mut reversed = images.clone();
    for leaf in leaves {
        if leaf.tag != Tag::Input || leaf.rows != Rows::Live {
            continue;
        }
        let bytes = images.bytes(&leaf.path)?;
        let row = leaf.row_bytes;
        let rows = (0..capacity as usize)
            .rev()
            .flat_map(|at| bytes[at * row..][..row].to_vec())
            .collect();
        reversed = reversed.with(&leaf.path, rows);
    }
    Ok(reversed)
}

/// The device handles `arguments` take.
fn handles(arguments: &kernel_core::contract::ArgumentList) -> Vec<u64> {
    arguments
        .0
        .iter()
        .flat_map(|argument| match argument {
            Argument::DeviceBuffer(handle) => vec![*handle],
            Argument::TensorMap(map) => vec![map.base_buffer],
            Argument::DevicePointerArray(handles) => handles.clone(),
            _ => Vec::new(),
        })
        .collect()
}

/// Every rank's buffers on its session, by token.
struct Rig<'s> {
    sessions: &'s [Session],
    buffers: Vec<BTreeMap<u64, DeviceBuffer>>,
}

impl<'s> Rig<'s> {
    /// Allocates every rank's buffers, with its state regions zeroed.
    fn new(case: &CollectiveCase<'_>, sessions: &'s [Session]) -> LabResult<Self> {
        if sessions.len() != case.ranks.len() {
            return Err(format!(
                "{} sessions for a case of {} ranks",
                sessions.len(),
                case.ranks.len()
            ));
        }
        let mut buffers = Vec::with_capacity(sessions.len());
        for (rank, (session, own)) in sessions.iter().zip(&case.ranks).enumerate() {
            let mut mine = BTreeMap::new();
            for leaf in &own.leaves {
                let view = case.token(rank as u32, &leaf.path)?;
                let buffer = session.allocate(view.bytes.max(16) as usize)?;
                if leaf.tag == Tag::Scratch && own.states.iter().any(|(path, _)| *path == leaf.path)
                {
                    poison::write(session, &buffer, 0, &vec![0; buffer.byte_len()])?;
                }
                mine.insert(view.handle, buffer);
            }
            buffers.push(mine);
        }
        poison::synchronize(sessions)?;
        Ok(Self { sessions, buffers })
    }

    /// Every rank's buffers set for a step over its `images` (inputs from
    /// the images, outputs filled, scratch poisoned, the device live count
    /// `live`, state regions kept) and every rank synchronized; the names
    /// each rank reads back.
    fn set(
        &self,
        case: &CollectiveCase<'_>,
        images: &[Images],
        live: u32,
    ) -> LabResult<Vec<Vec<String>>> {
        let mut names = Vec::with_capacity(case.ranks.len());
        for (rank, own) in case.ranks.iter().enumerate() {
            let session = &self.sessions[rank];
            let mut read = Vec::new();
            for (name, _, fill) in own.case.fills(&images[rank])? {
                let buffer = &self.buffers[rank][&case.token(rank as u32, &name)?.handle];
                match fill {
                    Fill::Image(bytes) => poison::write(session, buffer, 0, &bytes)?,
                    Fill::Pattern(pattern) => {
                        poison::fill_all(session, buffer, &pattern)?;
                        read.push(name);
                    }
                }
            }
            if let Ok(view) = case.token(rank as u32, own.case.live_name()) {
                poison::write(
                    session,
                    &self.buffers[rank][&view.handle],
                    0,
                    &live.to_le_bytes(),
                )?;
            }
            names.push(read);
        }
        // No rank launches before every rank's buffers are set: a peer
        // writes into them.
        poison::synchronize(self.sessions)?;
        Ok(names)
    }

    /// Every rank's functions of `plans`, resolved before any launch:
    /// resolving one can wait for the device's running work.
    fn resolve(
        &self,
        functions: &[Box<Lookup<'_>>],
        plans: &[Vec<Launch>],
    ) -> LabResult<Vec<Vec<Function>>> {
        plans
            .iter()
            .enumerate()
            .map(|(rank, launches)| {
                launches
                    .iter()
                    .map(|launch| {
                        let function = (functions[rank])(&launch.symbol)?;
                        if launch.plan.dynamic_shared_bytes
                            > crate::conformance::DEFAULT_SHARED_BYTES
                        {
                            function
                                .set_max_dynamic_shared_memory(launch.plan.dynamic_shared_bytes)
                                .map_err(|e| e.to_string())?;
                        }
                        Ok(function)
                    })
                    .collect::<LabResult<Vec<Function>>>()
            })
            .collect()
    }

    /// Every rank's buffers by token, as launches bind them.
    fn bindings(&self) -> launch::Bindings<'_> {
        launch::Bindings {
            buffers: self
                .buffers
                .iter()
                .flat_map(|mine| mine.iter().map(|(&token, buffer)| (token, buffer)))
                .collect(),
        }
    }

    /// Every rank's buffers of `names` after its work finished.
    fn read(&self, case: &CollectiveCase<'_>, names: Vec<Vec<String>>) -> LabResult<Vec<Results>> {
        poison::synchronize(self.sessions)?;
        names
            .into_iter()
            .enumerate()
            .map(|(rank, read)| {
                read.into_iter()
                    .map(|name| {
                        let buffer = &self.buffers[rank][&case.token(rank as u32, &name)?.handle];
                        Ok((name, poison::read(&self.sessions[rank], buffer)?))
                    })
                    .collect()
            })
            .collect()
    }

    /// One step of `rows` rows over every rank's `images`: every rank's
    /// buffers set with the device live count `rows`, every rank's
    /// launches in order, then every rank's outputs and scratch.
    fn step(
        &self,
        case: &CollectiveCase<'_>,
        functions: &[Box<Lookup<'_>>],
        images: &[Images],
        rows: u32,
    ) -> LabResult<Vec<Results>> {
        let names = self.set(case, images, rows)?;
        let plans = case.plans(rows)?;
        let functions = self.resolve(functions, &plans)?;
        let bindings = self.bindings();
        for at in 0..plans[0].len() {
            for (rank, launches) in plans.iter().enumerate() {
                let launch = &launches[at];
                let session = &self.sessions[rank];
                launch::launch(
                    &session.context,
                    &functions[rank][at],
                    &session.stream,
                    launch.plan,
                    &launch.arguments,
                    &bindings,
                )
                .map_err(|e| format!("rank {rank} {}: {e}", launch.symbol))?;
            }
        }
        self.read(case, names)
    }

    /// Every rank's launches of `planned` rows captured once into a graph,
    /// then replayed for each count of `lives` as the device live count
    /// (every rank's buffers set before each replay, its state kept), as a
    /// served graph of a bucket replays with fewer live rows; every
    /// replay's outputs and scratch.
    fn replays(
        &self,
        case: &CollectiveCase<'_>,
        functions: &[Box<Lookup<'_>>],
        images: &[Images],
        planned: u32,
        lives: &[u32],
    ) -> LabResult<Vec<Vec<Results>>> {
        let plans = case.plans(planned)?;
        let functions = self.resolve(functions, &plans)?;
        let bindings = self.bindings();
        let graphs = plans
            .iter()
            .enumerate()
            .map(|(rank, launches)| {
                let session = &self.sessions[rank];
                session.capture(|stream| {
                    for (launch, function) in launches.iter().zip(&functions[rank]) {
                        launch::launch(
                            &session.context,
                            function,
                            stream,
                            launch.plan,
                            &launch.arguments,
                            &bindings,
                        )
                        .map_err(|e| format!("rank {rank} {}: {e}", launch.symbol))?;
                    }
                    Ok(())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        lives
            .iter()
            .map(|&live| {
                progress(format_args!("replay {planned} rows at live {live}"));
                let names = self.set(case, images, live)?;
                for (rank, graph) in graphs.iter().enumerate() {
                    graph
                        .launch(&self.sessions[rank].stream)
                        .map_err(|e| format!("rank {rank}'s graph of {planned} rows: {e}"))?;
                }
                self.read(case, names)
            })
            .collect()
    }
}

/// How a backend prepares a collective on one rank and the case over every
/// rank's op: `Err` for a refusal, `Ok(Err)` for a case the kit could not
/// declare.
type PrepareRanks<'a> =
    Box<dyn Fn(&dyn Backend, u32, Need) -> Result<LabResult<CollectiveCase<'a>>, Refused> + 'a>;

/// One collective of a model's required set: its name, need, and how every
/// rank prepares it.
pub struct RequiredCollective<'a> {
    pub name: String,
    pub need: Need,
    /// Whether a refusal fails the run.
    pub required: bool,
    prepare: PrepareRanks<'a>,
}

impl<'a> RequiredCollective<'a> {
    /// The collective `name` at `need`, which `prepare` prepares on a
    /// backend for one rank of a group of `ranks`, and `case` declares
    /// over every rank's op. A refusal on any rank is a refusal.
    pub fn new<B: 'a>(
        name: impl Into<String>,
        need: Need,
        prepare: impl Fn(&dyn Backend, u32, u32, Need) -> Result<Prepared<B>, Refused> + 'a,
        case: impl Fn(u32, Vec<Prepared<B>>) -> LabResult<CollectiveCase<'a>> + 'a,
    ) -> Self {
        Self {
            name: name.into(),
            need,
            required: false,
            prepare: Box::new(move |backend, ranks, need| {
                let ops = (0..ranks)
                    .map(|rank| prepare(backend, ranks, rank, need))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(case(ranks, ops))
            }),
        }
    }

    /// Makes a refusal fail the run.
    pub fn required(self) -> Self {
        Self {
            required: true,
            ..self
        }
    }
}

/// Runs `ops` against `backend` at `ranks` ranks: prepares each op on
/// every rank, declares its case and, with `sessions` (one per rank, peer
/// access enabled), runs every check with symbols `modules` loads on each
/// rank's session.
pub fn conform(
    backend: &dyn Backend,
    modules: &dyn Modules,
    ranks: u32,
    sessions: Option<&[Session]>,
    ops: &[RequiredCollective<'_>],
) -> LabResult<Report> {
    if sessions.is_some_and(|sessions| sessions.len() != ranks as usize) {
        return Err(format!("a run of {ranks} ranks takes one session per rank"));
    }
    let functions = sessions
        .map(|sessions| {
            sessions
                .iter()
                .map(|session| modules.load(session))
                .collect::<LabResult<Vec<_>>>()
        })
        .transpose()?;
    let mut reports = Vec::with_capacity(ops.len());
    for op in ops {
        if sessions.is_some() {
            progress(format_args!("{} at {ranks} ranks", op.name));
        }
        let result = match (op.prepare)(backend, ranks, op.need) {
            Err(refused) => OpResult::Refused(refused.to_string()),
            Ok(Err(error)) => OpResult::Undeclared(error),
            Ok(Ok(case)) => OpResult::Prepared {
                implementation: case.info().implementation.clone(),
                guarantees: case.info().guarantees,
                checks: case.checks(op.need, sessions.zip(functions.as_deref())),
            },
        };
        reports.push(OpReport {
            name: op.name.clone(),
            required: op.required,
            result,
        });
    }
    Ok(Report {
        backend: format!("{} at {ranks} ranks", backend.name()),
        ops: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kit::{bf16_bytes, bf16_values};
    use kernel_core::backend::PreparedOp;
    use kernel_core::collective::{AllReduceBuffers, RankViews};
    use kernel_core::contract::{ArgumentList, DType, LaunchPlan, Refusal};
    use kernel_core::properties::Invariance;

    /// What a fake rank's op gets wrong.
    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        /// Rank 1 launches another symbol.
        Symbol,
        /// No launch waits for peers.
        Local,
        /// A launch takes a handle no rank bound.
        Handle,
    }

    const SYMBOL: &str = "chop_fake_allreduce_v1";
    const STATE: [u64; 2] = [64, 16];

    struct Fake {
        rank: u32,
        fault: Fault,
        info: OpInfo,
    }

    impl PreparedOp<AllReduceBuffers<View>> for Fake {
        fn info(&self) -> &OpInfo {
            &self.info
        }

        fn launches(
            &self,
            buffers: &AllReduceBuffers<View>,
            _workspace: View,
            counts: Counts,
        ) -> Result<Vec<Launch>, Refusal> {
            let mut arguments = vec![
                Argument::DeviceBuffer(buffers.input.handle),
                Argument::DeviceBuffer(buffers.output.handle),
                Argument::DevicePointerArray(
                    buffers.ranks.iter().map(|r| r.output.handle).collect(),
                ),
                Argument::DevicePointerArray(
                    buffers.ranks.iter().map(|r| r.state[0].handle).collect(),
                ),
                Argument::DeviceBuffer(buffers.ranks[self.rank as usize].state[1].handle),
            ];
            if self.fault == Fault::Handle {
                arguments.push(Argument::DeviceBuffer(7));
            }
            let symbol = match self.fault {
                Fault::Symbol if self.rank == 1 => "chop_fake_other_v1",
                _ => SYMBOL,
            };
            Ok(vec![Launch {
                symbol: symbol.into(),
                rows: counts.rows,
                plan: LaunchPlan {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: false,
                },
                arguments: ArgumentList(arguments),
                collective: self.fault != Fault::Local,
                persistent: false,
            }])
        }
    }

    /// A two-rank fake all-reduce case with `fault`.
    fn case(fault: Fault) -> CollectiveCase<'static> {
        let ops = (0..2)
            .map(|rank| {
                let mut info = OpInfo::new(
                    "fake.collective",
                    Invariance::NONE,
                    vec![(SYMBOL.into(), 0), ("chop_fake_other_v1".into(), 0)],
                );
                info.state = STATE.to_vec();
                Box::new(Fake { rank, fault, info }) as Prepared<AllReduceBuffers<View>>
            })
            .collect();
        let records = (0..2_u32)
            .map(|me| AllReduceBuffers {
                input: Slot::Own(Spec::input(
                    DType::Bf16,
                    (Rows::Live, 8),
                    bf16_bytes(&bf16_values(u64::from(me), 32)),
                )),
                output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, 8))),
                live_rows: None,
                ranks: (0..2_u32)
                    .map(|rank| RankViews {
                        output: Slot::peer(rank, "output"),
                        state: (0..STATE.len())
                            .map(|at| {
                                if rank == me {
                                    Slot::State(STATE[at])
                                } else {
                                    Slot::peer(rank, format!("ranks[{rank}].state[{at}]"))
                                }
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        CollectiveCase::new(ops, 4, records)
            .unwrap()
            .oracle(|images, _| Ok(images.iter().map(|_| Vec::new()).collect()))
    }

    /// A faithful collective declares, naming its launches; one whose
    /// ranks launch differently, that waits for no peer, or that takes an
    /// unbound handle does not.
    #[test]
    fn a_collective_declares_only_alike_peer_waiting_launches() {
        let declared = case(Fault::None).validate().unwrap();
        assert_eq!(declared["launches"], json!([SYMBOL]));
        assert_eq!(declared["ranks"], json!(2));
        for (fault, message) in [
            (Fault::Symbol, "unlike rank 0's"),
            (Fault::Local, "waits for its peers"),
            (Fault::Handle, "no rank's buffer"),
        ] {
            let error = case(fault).validate().unwrap_err();
            assert!(error.contains(message), "{error}");
        }
        let unknown = CollectiveCase::new(
            Vec::<Prepared<AllReduceBuffers<View>>>::new(),
            4,
            Vec::<AllReduceBuffers<Slot<'static>>>::new(),
        );
        assert!(unknown.is_err());
    }

    /// Peer views resolve to the peer's own buffers: rank 0's pointer
    /// array names rank 1's output and state region.
    #[test]
    fn peer_slots_resolve_to_the_peers_buffers() {
        let case = case(Fault::None);
        let plans = case.plans(4).unwrap();
        let Argument::DevicePointerArray(outputs) = &plans[0][0].arguments.0[2] else {
            panic!("pointer array");
        };
        assert_eq!(outputs[1], case.token(1, "output").unwrap().handle);
        let Argument::DevicePointerArray(states) = &plans[0][0].arguments.0[3] else {
            panic!("pointer array");
        };
        assert_eq!(
            states[1],
            case.token(1, "ranks[1].state[0]").unwrap().handle
        );
        assert_eq!(
            states[0],
            case.token(0, "ranks[0].state[0]").unwrap().handle
        );
    }
}
