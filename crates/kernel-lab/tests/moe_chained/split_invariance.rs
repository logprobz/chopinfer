//! The chained MoE invariance cases of the routed experts split over the
//! ranks (`chained::split`, `kernel_lab::declared::chained_moe_split`): every
//! served form in FP8, NVFP4, MXFP8 and BF16, the routes over all 288
//! experts and the route sum in its BF16, F32 and add forms, at their
//! served capacities (32 decode tokens, 8,192 prefill tokens), on rank 0,
//! where every route is the rank's. The `Chained` adapter hands these
//! entries here.
//!
//! A unit is one token, whose inputs follow from its seed alone: its hidden
//! row, its router logits (top 8 inside [`POOL`]), or its routes and routed
//! rows. Each format's weight banks are drawn once from fixed seeds and
//! written for the pool's experts only (every other expert keeps the
//! poison), and every layer of the format, decode, verify (which runs the
//! decode forms) or prefill, reads the same banks. So a token's bits compare
//! across every launch, phase and schedule.
//!
//! - A routed form runs in a whole expert layer of its phase's route: the
//!   form, the other projection's served form of the same format, row tile
//!   and schedule where one exists (else the one of the most tokens), and
//!   both route sums. A unit is, for each of the token's eight routes in
//!   slot order, the gate/up's BF16 intermediate row and the down's weighted
//!   F32 route row, then the token's F32 and BF16 sum rows. Its one-row
//!   launch is the format's decode layer at one token: the decode route, the
//!   served decode gate/up and the decode down a one-token step runs. So a
//!   prefill form's rows compare with the decode path's.
//! - A route kernel runs in its route; a unit is the token's expert ids and
//!   weights; its one-row launch is the decode route at one token. Ties as
//!   the `Chained` adapter's, against the f64 router over all experts.
//! - The route sum (BF16, F32) and the F32 sum-add: a unit is the token's
//!   output row from its routes over all experts, routed rows (and addend).
//!
//! Row counts: every count of a decode form's capacity (1 to 32, or 4); at
//! prefill the counts around the decode-prefill boundary (33), the route's
//! 64-row padding, the tile pairs' first step (`TILE_PAIR_MIN_ROWS` - 1 and
//! itself), 4,096 and the capacity, 8,192. A prefill chunk case spans the
//! capacity; a batch case 2,048 tokens. Every compared launch writes
//! poisoned outputs, checked afterwards over every route's rows.
//!
//! The negative control (`a_reordered_split_sum_add_fails_the_rows_case_gpu`)
//! runs the sum-add's tokens alone through the lab mutant that starts each
//! sum from the addend: the rows case must see other bits, and the lab's
//! unmodified instance alone must equal them.

use super::*;
use crate::invariance::{Compared, Element, Units, written};
use kernel_lab::families::moe_chained_split_sum_add as sum_add_shapes;
use kernels_sm120::backend::TILE_PAIR_MIN_ROWS;
use kernels_sm120::invariance::Declared;
use std::collections::HashMap;

/// The rank every case runs: every route is its own.
const RANK: usize = 0;
/// The split's per-rank expert intermediate.
const SPLIT_INTER: usize = split::SHAPES.expert_intermediate as usize;
/// The experts the tokens route to: spread over all 288, the first and the
/// last among them. Prefill launches load each with many row tiles, so tile
/// pairs pair.
pub const POOL: [usize; 24] = [
    0, 13, 26, 36, 49, 62, 72, 85, 98, 108, 121, 134, 144, 157, 170, 180, 193, 206, 216, 229, 242,
    252, 265, 287,
];
/// Prefill row counts of a rows case (see the module notes).
const PREFILL_COUNTS: [u32; 7] = [
    33,
    64,
    65,
    TILE_PAIR_MIN_ROWS - 1,
    TILE_PAIR_MIN_ROWS,
    4096,
    8192,
];
/// Tokens of a prefill batch case: a tile-pair step.
const PREFILL_BATCH: usize = TILE_PAIR_MIN_ROWS as usize;

/// What a split entry's cases launch.
#[derive(Clone, Debug)]
pub enum Role {
    Routed(Box<Projection>),
    Route(LocalRoute),
    Sum(RoutedSumOutput),
    SumAdd,
}

/// The split role of `symbol`, if it is a split entry.
pub fn role(symbol: &str) -> Option<Role> {
    if let Some(p) = split::forms().into_iter().find(|p| p.symbol == symbol) {
        return Some(Role::Routed(Box::new(p)));
    }
    for route in split::routes() {
        if route
            .kernels()
            .iter()
            .any(|&kernel| route.symbol(split::PREFIX, kernel) == symbol)
        {
            return Some(Role::Route(route));
        }
    }
    let [sum] = split::sums();
    for output in [RoutedSumOutput::Bf16, RoutedSumOutput::F32] {
        if sum.with_output(output).symbol(split::PREFIX) == symbol {
            return Some(Role::Sum(output));
        }
    }
    (sum.with_output(RoutedSumOutput::F32)
        .add_symbol(split::PREFIX)
        == symbol)
        .then_some(Role::SumAdd)
}

/// The split's served F32 sum-add entry.
fn sum_add() -> Entry {
    let [sum] = split::sums();
    Entry {
        cubin: format!("{}_sum_add", split::PREFIX),
        symbol: sum
            .with_output(RoutedSumOutput::F32)
            .add_symbol(split::PREFIX),
    }
}

/// A token's hidden row (as the `Chained` adapter's).
fn token_x(seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed ^ 0x0a11);
    (0..HIDDEN).map(|_| bf16_value(&mut rng, -4..=0)).collect()
}

/// A token's router logits over all experts: its top 8 inside [`POOL`].
fn token_logits(seed: u64) -> Vec<f32> {
    pool_logits(1, &POOL, &mut Rng(seed ^ 0x5b17))
}

/// A token's routes (eight experts of all 288, every one the rank's),
/// routed F32 rows and addend row.
fn token_sum_inputs(seed: u64) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed ^ 0x5a5b);
    let value =
        |rng: &mut Rng| ((rng.unit() * 2.0 - 1.0) * 2_f64.powi(rng.below(16) as i32 - 8)) as f32;
    let ids = (0..TOP_K).map(|_| rng.below(EXPERTS) as i32).collect();
    let routed = (0..TOP_K * HIDDEN).map(|_| value(&mut rng)).collect();
    let addend = (0..HIDDEN).map(|_| value(&mut rng)).collect();
    (ids, routed, addend)
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The output rows of `bytes` per token of tokens `0..tokens`.
fn rows_of(bytes: &[u8], tokens: usize, row: usize) -> Units {
    (0..tokens)
        .map(|t| bytes[t * row..(t + 1) * row].to_vec())
        .collect()
}

/// One format's weight banks, written for [`POOL`]'s experts from fixed
/// seeds.
struct Banks {
    weights: Weights,
    gate_up: Bank,
    down: Bank,
}

impl Banks {
    fn new(session: &Session, weights: Weights) -> LabResult<Self> {
        let experts = split::SHAPES.local_experts as usize;
        let build = |rows, k, halves, seed| {
            Bank::build(
                session,
                weights,
                experts,
                rows,
                k,
                halves,
                &POOL,
                &mut Rng(seed),
                false,
            )
        };
        Ok(Self {
            weights,
            gate_up: build(2 * SPLIT_INTER, HIDDEN, 2, 0x5b4c)?,
            down: build(HIDDEN, SPLIT_INTER, 1, 0x5b4d)?,
        })
    }
}

/// The served form of `weights`' gate/up or down at `row_tile` that runs
/// with `form` in a layer: the one of `form`'s schedule kind, else the one
/// of the most tokens.
fn partner(form: &Projection) -> LabResult<Projection> {
    let candidates = split::forms()
        .into_iter()
        .filter(|p| {
            p.weights() == form.weights()
                && p.row_tile == form.row_tile
                && p.gate_up() != form.gate_up()
        })
        .collect::<Vec<_>>();
    let same_kind = candidates
        .iter()
        .find(|p| std::mem::discriminant(&p.schedule) == std::mem::discriminant(&form.schedule));
    same_kind
        .or_else(|| candidates.iter().max_by_key(|p| p.tokens_capacity))
        .cloned()
        .ok_or_else(|| format!("{}: no served partner", form.symbol))
}

/// The format's decode forms of a one-token step: the served decode
/// gate/up and the decode down of the fewest tokens.
fn decode_forms(weights: Weights) -> LabResult<[Projection; 2]> {
    let decode = split::forms()
        .into_iter()
        .filter(|p| p.weights() == weights && p.row_tile == 16)
        .collect::<Vec<_>>();
    let pick = |gate_up: bool| {
        decode
            .iter()
            .filter(|p| p.gate_up() == gate_up)
            .min_by_key(|p| p.tokens_capacity)
            .cloned()
            .ok_or_else(|| format!("{weights:?}: no decode form"))
    };
    Ok([pick(true)?, pick(false)?])
}

/// One expert layer: a route, a gate/up and down form, both route sums,
/// with buffers for `tokens` tokens.
struct Layer {
    route: LocalRoute,
    forms: [Projection; 2],
    fns: [Function; 2],
    route_fns: Vec<Function>,
    sums: [Function; 2],
    routes: RouteBuffers,
    x: DeviceBuffer,
    inter: DeviceBuffer,
    routed: DeviceBuffer,
    out: DeviceBuffer,
    out_f32: DeviceBuffer,
    tokens: usize,
}

impl Layer {
    fn new(
        session: &Session,
        prefill: bool,
        forms: [Projection; 2],
        tokens: usize,
    ) -> LabResult<Self> {
        let route = split::routes()[usize::from(prefill)];
        let capacity = route.capacity as usize;
        let load = |p: &Projection| Entry::projection(split::PREFIX, p).load(session);
        let [sum] = split::sums();
        let sum = |output| {
            Entry::route_in(split::PREFIX, sum.with_output(output).symbol(split::PREFIX))
                .load(session)
        };
        let routes_per = tokens * TOP_K;
        Ok(Self {
            route,
            fns: [load(&forms[0])?, load(&forms[1])?],
            route_fns: route_entries_in(split::PREFIX, route)
                .iter()
                .map(|entry| entry.load(session))
                .collect::<LabResult<_>>()?,
            sums: [sum(RoutedSumOutput::Bf16)?, sum(RoutedSumOutput::F32)?],
            routes: RouteBuffers::new(
                session,
                route,
                &vec![-8.0; capacity * EXPERTS],
                &vec![0.0; EXPERTS],
            )?,
            x: zeroed(session, tokens * HIDDEN * 2)?,
            inter: zeroed(session, routes_per * SPLIT_INTER * 2)?,
            routed: zeroed(session, routes_per * HIDDEN * 4)?,
            out: zeroed(session, tokens * HIDDEN * 2)?,
            out_f32: zeroed(session, tokens * HIDDEN * 4)?,
            forms,
            tokens,
        })
    }

    /// The tokens of `seeds` through the layer over `banks`.
    fn run(&self, session: &Session, banks: &Banks, seeds: &[u64]) -> LabResult<Units> {
        let tokens = seeds.len();
        if tokens == 0 || tokens > self.tokens {
            return Err(format!("{tokens} tokens in a layer of {}", self.tokens));
        }
        let x = seeds.iter().flat_map(|&s| token_x(s)).collect::<Vec<_>>();
        poison::write(session, &self.x, 0, &u16_bytes(&x))?;
        let logits = seeds
            .iter()
            .flat_map(|&s| token_logits(s))
            .collect::<Vec<_>>();
        let r = &self.routes;
        poison::write(session, &r.logits, 0, &bytes_of(&logits, f32::to_ne_bytes))?;
        r.set_live(session, tokens as u32)?;
        poison::poison(
            session,
            &[&self.inter, &self.routed, &self.out, &self.out_f32],
        )?;
        let stream = &session.stream;
        r.launch(stream, self.route, &self.route_fns, RANK as u32, false)?;
        let b = KernelArgument::Buffer;
        for (at, (bank, input, output)) in [
            (&banks.gate_up, &self.x, &self.inter),
            (&banks.down, &self.inter, &self.routed),
        ]
        .into_iter()
        .enumerate()
        {
            let [codes, block, local, globals] = bank.arguments();
            let weights = match at {
                0 => KernelArgument::NullBuffer,
                _ => b(&r.weights),
            };
            launch_with(
                stream,
                &self.fns[at],
                self.forms[at].plan(tokens as u32).map_err(err)?,
                false,
                &[
                    b(input),
                    codes,
                    block,
                    local,
                    globals,
                    b(&r.offsets),
                    b(&r.sorted),
                    b(&r.padded),
                    b(&r.live),
                    weights,
                    b(output),
                ],
            )?;
        }
        let [sum] = split::sums();
        for (f, output) in [(&self.sums[0], &self.out), (&self.sums[1], &self.out_f32)] {
            launch_with(
                stream,
                f,
                sum.plan(tokens as u32).map_err(err)?,
                false,
                &[
                    b(&self.routed),
                    b(&r.ids),
                    b(&r.live),
                    KernelArgument::U32(RANK as u32),
                    b(output),
                ],
            )?;
        }
        for entry in route_entries_in(split::PREFIX, self.route) {
            kernel_lab::launched(&entry.symbol);
        }
        for form in &self.forms {
            kernel_lab::launched(&form.symbol);
        }
        for output in [RoutedSumOutput::Bf16, RoutedSumOutput::F32] {
            kernel_lab::launched(&sum.with_output(output).symbol(split::PREFIX));
        }
        let routes = tokens * TOP_K;
        let ids = words(&poison::read(session, &r.ids)?, i32::from_ne_bytes);
        if let Some(at) = ids[..routes]
            .iter()
            .position(|&id| !POOL.contains(&(id as usize)))
        {
            return Err(format!("route {at} left the pool: expert {}", ids[at]));
        }
        // Every route is the rank's: each writes its intermediate and
        // weighted route rows.
        let inter = poison::read(session, &self.inter)?;
        let routed = poison::read(session, &self.routed)?;
        let out = poison::read(session, &self.out)?;
        let out_f32 = poison::read(session, &self.out_f32)?;
        written(
            "gate/up intermediate",
            &inter,
            &WriteMap::of(DType::Bf16).prefix(routes * SPLIT_INTER)?,
            Element::Bf16,
        )?;
        written(
            "weighted routes",
            &routed,
            &WriteMap::of(DType::F32).prefix(routes * HIDDEN)?,
            Element::F32,
        )?;
        written(
            "route sum",
            &out,
            &WriteMap::of(DType::Bf16).prefix(tokens * HIDDEN)?,
            Element::Bf16,
        )?;
        written(
            "F32 route sum",
            &out_f32,
            &WriteMap::of(DType::F32).prefix(tokens * HIDDEN)?,
            Element::F32,
        )?;
        let (i_row, r_row) = (SPLIT_INTER * 2, HIDDEN * 4);
        Ok((0..tokens)
            .map(|t| {
                let mut unit = Vec::with_capacity(TOP_K * (i_row + r_row) + HIDDEN * 6);
                for route in t * TOP_K..(t + 1) * TOP_K {
                    unit.extend(&inter[route * i_row..(route + 1) * i_row]);
                    unit.extend(&routed[route * r_row..(route + 1) * r_row]);
                }
                unit.extend(&out_f32[t * HIDDEN * 4..(t + 1) * HIDDEN * 4]);
                unit.extend(&out[t * HIDDEN * 2..(t + 1) * HIDDEN * 2]);
                unit
            })
            .collect())
    }
}

/// The split cases' banks, layers and routes. Only one format's banks and
/// layers are held at a time (a format's banks take up to 3.6 GB).
#[derive(Default)]
pub struct Split {
    banks: Option<Banks>,
    layers: HashMap<String, Layer>,
    routes: HashMap<u32, (RouteBuffers, Vec<Function>)>,
}

impl Split {
    /// `weights`' banks, dropping another format's banks and layers.
    fn banks(&mut self, session: &Session, weights: Weights) -> LabResult<()> {
        if self.banks.as_ref().is_none_or(|b| b.weights != weights) {
            self.layers.clear();
            self.banks = None;
            self.banks = Some(Banks::new(session, weights)?);
        }
        Ok(())
    }

    /// The layer of `form` with its partner at its phase's capacity (at
    /// most the form's tokens), or with `alone` the format's one-token
    /// decode layer.
    fn layer(&mut self, session: &Session, form: &Projection, alone: bool) -> LabResult<&Layer> {
        self.banks(session, form.weights())?;
        let key = match alone {
            true => format!("{:?} alone", form.weights()),
            false => form.symbol.clone(),
        };
        if !self.layers.contains_key(&key) {
            let layer = if alone {
                Layer::new(session, false, decode_forms(form.weights())?, 1)?
            } else {
                let partner = partner(form)?;
                let forms = match form.gate_up() {
                    true => [form.clone(), partner],
                    false => [partner, form.clone()],
                };
                let prefill = form.row_tile == 64;
                let route = split::routes()[usize::from(prefill)];
                let tokens = forms
                    .iter()
                    .map(|p| p.tokens_capacity)
                    .chain([route.capacity])
                    .min()
                    .expect("forms");
                Layer::new(session, prefill, forms, tokens as usize)?
            };
            self.layers.insert(key.clone(), layer);
        }
        Ok(&self.layers[&key])
    }

    fn run_layer(
        &mut self,
        session: &Session,
        form: &Projection,
        alone: bool,
        seeds: &[u64],
    ) -> LabResult<Units> {
        self.layer(session, form, alone)?;
        let key = match alone {
            true => format!("{:?} alone", form.weights()),
            false => form.symbol.clone(),
        };
        let banks = self.banks.as_ref().expect("banks of the layer's format");
        self.layers[&key].run(session, banks, seeds)
    }

    fn route(
        &mut self,
        session: &Session,
        route: LocalRoute,
    ) -> LabResult<&(RouteBuffers, Vec<Function>)> {
        Ok(match self.routes.entry(route.capacity) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let capacity = route.capacity as usize;
                let buffers = RouteBuffers::new(
                    session,
                    route,
                    &vec![-8.0; capacity * EXPERTS],
                    &vec![0.0; EXPERTS],
                )?;
                let functions = route_entries_in(split::PREFIX, route)
                    .iter()
                    .map(|entry| entry.load(session))
                    .collect::<LabResult<Vec<_>>>()?;
                entry.insert((buffers, functions))
            }
        })
    }

    /// Router `logits` and `bias` of their tokens through `route`: each
    /// token's ids, then its weights.
    fn run_route(
        &mut self,
        session: &Session,
        route: LocalRoute,
        logits: &[f32],
        bias: &[f32],
    ) -> LabResult<(Vec<i32>, Units)> {
        let tokens = logits.len() / EXPERTS;
        let (buffers, functions) = self.route(session, route)?;
        poison::write(
            session,
            &buffers.logits,
            0,
            &bytes_of(logits, f32::to_ne_bytes),
        )?;
        poison::write(session, &buffers.bias, 0, &bytes_of(bias, f32::to_ne_bytes))?;
        buffers.set_live(session, tokens as u32)?;
        poison::poison(session, &[&buffers.ids, &buffers.weights])?;
        buffers.launch(&session.stream, route, functions, RANK as u32, false)?;
        for entry in route_entries_in(split::PREFIX, route) {
            kernel_lab::launched(&entry.symbol);
        }
        let ids = poison::read(session, &buffers.ids)?;
        let weights = poison::read(session, &buffers.weights)?;
        let routes = tokens * TOP_K;
        written(
            "route ids",
            &ids,
            &WriteMap::of(DType::I32).prefix(routes)?,
            Element::Raw(4),
        )?;
        written(
            "route weights",
            &weights,
            &WriteMap::of(DType::F32).prefix(routes)?,
            Element::F32,
        )?;
        let units = (0..tokens)
            .map(|t| {
                let mut unit = ids[t * TOP_K * 4..(t + 1) * TOP_K * 4].to_vec();
                unit.extend(&weights[t * TOP_K * 4..(t + 1) * TOP_K * 4]);
                unit
            })
            .collect();
        Ok((words(&ids, i32::from_ne_bytes)[..routes].to_vec(), units))
    }

    /// The tokens of `seeds` through the route sum in `output`'s form, or
    /// through the F32 sum-add entry `add` (the served one, [`sum_add`], or
    /// a lab instance).
    fn run_sum(
        session: &Session,
        output: RoutedSumOutput,
        add: Option<&Entry>,
        seeds: &[u64],
    ) -> LabResult<Units> {
        let tokens = seeds.len();
        let (mut ids, mut routed, mut addend) = (Vec::new(), Vec::new(), Vec::new());
        for &seed in seeds {
            let (i, r, a) = token_sum_inputs(seed);
            ids.extend(i);
            routed.extend(r);
            addend.extend(a);
        }
        let sum = split::sums()[0].with_output(output);
        let row = sum.output_row_bytes() as usize;
        let routed = upload(session, &bytes_of(&routed, f32::to_ne_bytes))?;
        let ids = upload(session, &bytes_of(&ids, i32::to_ne_bytes))?;
        let live = upload(session, &(tokens as u32).to_ne_bytes())?;
        let out = session.allocate(tokens * row)?;
        poison::poison(session, &[&out])?;
        let b = KernelArgument::Buffer;
        let plan = sum.plan(tokens as u32).map_err(err)?;
        let rank = KernelArgument::U32(RANK as u32);
        if let Some(entry) = add {
            let addend = upload(session, &bytes_of(&addend, f32::to_ne_bytes))?;
            launch_with(
                &session.stream,
                &entry.load(session)?,
                plan,
                false,
                &[b(&routed), b(&ids), b(&live), rank, b(&addend), b(&out)],
            )?;
            kernel_lab::launched(&entry.symbol);
        } else {
            let entry = Entry::route_in(split::PREFIX, sum.symbol(split::PREFIX));
            launch_with(
                &session.stream,
                &entry.load(session)?,
                plan,
                false,
                &[b(&routed), b(&ids), b(&live), rank, b(&out)],
            )?;
            kernel_lab::launched(&entry.symbol);
        }
        let bytes = poison::read(session, &out)?;
        let (map, element) = match output {
            RoutedSumOutput::Bf16 => (WriteMap::of(DType::Bf16), Element::Bf16),
            RoutedSumOutput::F32 => (WriteMap::of(DType::F32), Element::F32),
        };
        written("route sum", &bytes, &map.prefix(tokens * HIDDEN)?, element)?;
        Ok(rows_of(&bytes, tokens, row))
    }

    /// The tokens of `seeds` through `role`'s launch.
    pub fn launch(&mut self, session: &Session, role: &Role, seeds: &[u64]) -> LabResult<Units> {
        match role {
            Role::Routed(p) => self.run_layer(session, p, false, seeds),
            Role::Route(route) => {
                let logits = seeds
                    .iter()
                    .flat_map(|&s| token_logits(s))
                    .collect::<Vec<_>>();
                Ok(self.run_route(session, *route, &logits, &[0.0; EXPERTS])?.1)
            }
            Role::Sum(output) => Self::run_sum(session, *output, None, seeds),
            Role::SumAdd => Self::run_sum(session, RoutedSumOutput::F32, Some(&sum_add()), seeds),
        }
    }

    /// Each token of `seeds` alone through `role`'s one-row launch: the
    /// decode layer or route at one token.
    pub fn alone(&mut self, session: &Session, role: &Role, seeds: &[u64]) -> LabResult<Units> {
        let mut units = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            let one = [seed];
            let unit = match role {
                Role::Routed(p) => self.run_layer(session, p, true, &one)?,
                Role::Route(_) => {
                    let decode = split::routes()[0];
                    self.run_route(session, decode, &token_logits(seed), &[0.0; EXPERTS])?
                        .1
                }
                _ => self.launch(session, role, &one)?,
            };
            units.extend(unit);
        }
        Ok(units)
    }

    /// Router keys with ties through `route`: each token's ids against the
    /// f64 router's selection over all experts, which keeps the lower
    /// expert of equal keys.
    pub fn ties(&mut self, session: &Session, route: LocalRoute) -> LabResult<Vec<Compared>> {
        let tokens = 16;
        let mut compared = Vec::new();
        for (pattern, (logits, bias)) in tie_patterns(tokens, &mut Rng(0x7136))
            .into_iter()
            .enumerate()
        {
            let (ids, _) = self.run_route(session, route, &logits, &bias)?;
            let query = MoeQuery::LocalRoutes {
                experts: EXPERTS as u32,
                local_experts: route.local_experts,
                top_k: TOP_K as u32,
                capacity: route.capacity,
            };
            let kind = RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.5,
            };
            let wide = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
            let expected = rank_routes(&wide(&logits), Some(&wide(&bias)), kind, query, RANK)
                .map_err(|e| e.0.to_string())?;
            for (token, selection) in expected.selections.iter().enumerate() {
                compared.push(Compared {
                    label: format!("split tie pattern {pattern} token {token}"),
                    got: bytes_of(&ids[token * TOP_K..(token + 1) * TOP_K], i32::to_ne_bytes),
                    expected: selection
                        .expert_ids
                        .iter()
                        .flat_map(|&id| (id as i32).to_ne_bytes())
                        .collect(),
                });
            }
        }
        Ok(compared)
    }
}

/// The capacity of `role`'s launches.
fn capacity(role: &Role) -> u32 {
    match role {
        Role::Routed(p) => p.tokens_capacity,
        Role::Route(route) => route.capacity,
        Role::Sum(_) | Role::SumAdd => split::sums()[0].capacity,
    }
}

/// Tokens a chunk case's prompt spans: the capacity.
pub fn prompt_rows(role: &Role) -> u32 {
    capacity(role)
}

/// Row counts of a rows case: every count to a decode capacity; at prefill
/// capacities [`PREFILL_COUNTS`] and, for the sums, which every phase
/// runs, the decode counts' bounds too.
pub fn row_counts(role: &Role, max: u32) -> Vec<u32> {
    let most = max.min(capacity(role));
    if most <= split::SHAPES.decode_tokens {
        return (2..=most).collect();
    }
    let mut counts = match role {
        Role::Sum(_) | Role::SumAdd => vec![2, 3, 4, 5, 16, 31, 32],
        _ => Vec::new(),
    };
    counts.extend(PREFILL_COUNTS.into_iter().filter(|&n| n <= most));
    counts
}

/// Units of a batch case: a decode capacity, or a tile-pair step.
pub fn batch_units(role: &Role) -> usize {
    (capacity(role) as usize).min(PREFILL_BATCH)
}

/// The split declarations' roles, and every role kind among them (no GPU).
pub fn check_roles(declared: &[Declared]) -> LabResult<()> {
    let mut kinds = std::collections::BTreeSet::new();
    for entry in declared {
        let role = role(&entry.symbol).ok_or_else(|| format!("{}: no split role", entry.symbol))?;
        kinds.insert(match &role {
            Role::Routed(p) => format!("{:?} {:?}", p.weights(), p.schedule),
            Role::Route(route) => format!("route {}", route.capacity),
            Role::Sum(output) => format!("sum {output:?}"),
            Role::SumAdd => "sum add".into(),
        });
        let entry = match &role {
            Role::Routed(p) => Entry::projection(split::PREFIX, p),
            Role::Route(_) | Role::Sum(_) => Entry::route_in(split::PREFIX, entry.symbol.clone()),
            Role::SumAdd => Entry {
                cubin: format!("{}_sum_add", split::PREFIX),
                symbol: entry.symbol.clone(),
            },
        };
        if !kernel_lab::cubin_entries(&entry.cubin)?.contains(&entry.symbol.as_str()) {
            return Err(format!("{entry:?} is not built"));
        }
    }
    for weights in [
        Weights::Fp8Block128,
        Weights::Nvfp4,
        Weights::Mxfp8,
        Weights::Bf16,
    ] {
        decode_forms(weights)?;
        for form in split::forms().iter().filter(|p| p.weights() == weights) {
            partner(form)?;
        }
    }
    for kind in [
        "Fp8Block128 WarpStreams",
        "Fp8Block128 BlockWarps",
        "Fp8Block128 TilePairs { min_pair_tiles: 2 }",
        "Fp8Block128 TilePairs { min_pair_tiles: 0 }",
        "Nvfp4 BlockWarps",
        "Nvfp4 Columns",
        "Mxfp8 Columns",
        "Bf16 BlockWarps",
        "Bf16 Columns",
        "route 32",
        "route 8192",
        "sum Bf16",
        "sum F32",
        "sum add",
    ] {
        if !kinds.contains(kind) {
            return Err(format!("{kind}: no split entry ({kinds:?})"));
        }
    }
    Ok(())
}

/// Every split declaration has a role whose entry is built, every format
/// has its decode forms and every form a partner, the row counts reach each
/// capacity, and the tokens' inputs follow their seeds alone (no GPU).
#[test]
fn split_declarations_have_roles() {
    let declared = kernel_lab::declared::chained_moe_split();
    assert_eq!(declared.len(), 19 + 4 + 3, "{declared:?}");
    check_roles(&declared).unwrap();
    let decode = Role::Route(split::routes()[0]);
    assert_eq!(row_counts(&decode, 32), (2..=32).collect::<Vec<_>>());
    let prefill = Role::Route(split::routes()[1]);
    let counts = row_counts(&prefill, 8192);
    assert_eq!(counts.last(), Some(&8192));
    assert!(counts.contains(&(TILE_PAIR_MIN_ROWS - 1)) && counts.contains(&TILE_PAIR_MIN_ROWS));
    assert_eq!(prompt_rows(&prefill), 8192);
    assert_eq!(batch_units(&prefill), 2048);
    let sums = row_counts(&Role::SumAdd, 8192);
    assert!(sums.contains(&2) && sums.contains(&32) && sums.contains(&8192));
    for output in [RoutedSumOutput::Bf16, RoutedSumOutput::F32] {
        assert_eq!(capacity(&Role::Sum(output)), 8192);
    }
    let down_bw = split::forms()
        .into_iter()
        .find(|p| p.schedule == Schedule::BlockWarps && p.weights() == Weights::Fp8Block128)
        .unwrap();
    assert_eq!(
        row_counts(&Role::Routed(Box::new(down_bw.clone())), 4),
        [2, 3, 4]
    );
    assert_eq!(
        decode_forms(Weights::Fp8Block128).unwrap()[1].symbol,
        down_bw.symbol
    );
    assert_eq!(token_x(5), token_x(5));
    assert_ne!(token_x(5), token_x(6));
    let logits = token_logits(9);
    let mut order = (0..EXPERTS).collect::<Vec<_>>();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    assert!(order[..TOP_K].iter().all(|e| POOL.contains(e)));
    assert_eq!(token_sum_inputs(3).0, token_sum_inputs(3).0);
    let lab = kernel_lab::cubin_entries(sum_add_shapes::CUBIN).unwrap();
    for mutation in ["none", "addend_first"] {
        let symbol = sum_add_shapes::symbol(mutation);
        assert!(lab.contains(&symbol.as_str()), "{symbol}");
    }
}

/// The split sum-add's tokens alone through the lab mutant that starts each
/// sum from the addend (`moe_chained_split_sum_add`) give other bits than
/// the served kernel's launch of them all, which the rows case would catch;
/// the lab's unmodified instance alone gives the same.
#[test]
fn a_reordered_split_sum_add_fails_the_rows_case_gpu() {
    gpu_test(2 << 30, |session| {
        let seeds = (0..16_u64).map(|s| s * 31 + 7).collect::<Vec<_>>();
        let together = Split::run_sum(session, RoutedSumOutput::F32, Some(&sum_add()), &seeds)?;
        for (mutation, equal) in [("none", true), ("addend_first", false)] {
            let lab = Entry {
                cubin: sum_add_shapes::CUBIN.into(),
                symbol: sum_add_shapes::symbol(mutation),
            };
            let mut differing = 0;
            for (i, &seed) in seeds.iter().enumerate() {
                let alone = Split::run_sum(session, RoutedSumOutput::F32, Some(&lab), &[seed])?;
                if alone[0] != together[i] {
                    differing += 1;
                }
            }
            eprintln!("{mutation}: {differing} of {} tokens differ", seeds.len());
            if (differing == 0) != equal {
                return Err(format!("{mutation}: {differing} tokens differ"));
            }
        }
        Ok(())
    })
    .expect("the split sum-add negative control");
}
