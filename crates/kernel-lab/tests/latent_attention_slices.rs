//! The slice-widened latent attention (`kernels_sm120::latent::slices`,
//! README "Slice-widened attention" in crates/kernels-sm120/cuda/latent)
//! against the FP8 prefill entry it replaces, at the lab's latent shape
//! (`latent_shapes.rs`: 16 heads, latent 512, 64-token pages, 2051-entry key
//! lists), on the GPU. Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test
//! -p kernel-lab --release --test latent_attention_slices --
//! --test-threads=1`; without a GPU only the CPU contract test runs.
//!
//! The FP8 pool is written on the host: random E4M3 codes (no NaN codes,
//! which the satfinite KV write never stores) and positive F32 block scales
//! on pages scattered over one pool, for sequences of 3,000, 9,000 and
//! 33,000 tokens. Key lists are the indexer's: every pool up to 512 of
//! them, else 512 random pools ascending, then the incomplete tail, then -1.
//!
//! - Both entries' outputs (BF16 rows, or the F32 partial's maxima,
//!   denominators and numerators) are equal byte for byte over poisoned
//!   buffers at 1, 2, 30, 36, 512, 2048 and 8192 rows, at contexts across
//!   2K, 8K and 32K, with random holes, all-hole rows, whole-unit holes, a
//!   hole at every unit's first entry, sharp softmaxes, budgets 2051, 1027
//!   and 7, and rows past the live count.
//! - Rows whose queries hold NaN or infinities agree byte for byte too.
//! - A running output that underflows to -0 before an empty unit: both
//!   forms turn it to +0 there (`fmaf(-0, 1, +0)`), and keep -0 when no
//!   empty unit follows, in both output modes and at 1 to 8192 rows, each
//!   row's bytes the same at every row count.
//! - Poisoned replays (eager, relaunch, graph, PDL behind the live-count
//!   producer) at the live counts `poison::live_counts` lists for 1, 30,
//!   2048 and 8192 rows, each live count's eager output equal to the
//!   replaced entry's at that live count.
//! - One captured graph per capacity (1, 30, 52, 375, 376 and 2048 rows at
//!   every live count; 8064 and 8192 rows at the replay counts and the
//!   boundaries) on poisoned outputs, the device live count changed between
//!   launches: rows below it hold the replaced entry's bytes, the rest keep
//!   the poison.
//! - Mutants, each caught only when an output differs from the served
//!   bits (launch and readback errors propagate): a unit live only when
//!   its first key is (`_skip_first`), copies from the previous unit's page
//!   (`_stale_page`), and the dependency wait after the query loads
//!   (`_late_wait`, behind a producer that writes the queries after
//!   releasing it; up to four attempts, and the test fails unless one
//!   observes a difference).
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `latent_attention_slices_timing` times
//! both entries at 1, 16, 2048 and 8192 rows ending at 2K, 8K and 32K
//! contexts.

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{DType, LaunchPlan};
use kernel_core::latent_attention::LatentKvFormat;
use kernel_lab::latent_shapes::{self, GEOMETRY};
use kernel_lab::poison::{self, CopyProducer, Fill, POISON, Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::latent::Entry;
use kernels_sm120::latent::slices::{self, cubin_name};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use support::{Rng, bytes_i32, bytes_u16, download, upload};

const HEADS: usize = GEOMETRY.heads as usize;
const LATENT: usize = GEOMETRY.latent as usize;
const PAGE: usize = GEOMETRY.page_size as usize;
const MAX_KEYS: usize = GEOMETRY.max_keys as usize;
const POOL: usize = 4;
const TOP_POOLS: usize = 512;
const UNIT: usize = 32;
/// Bytes between consecutive pages that no kernel reads.
const PAGE_GAP: usize = 256;
/// The lab's sequences: 3K, 9K and 33K tokens.
const LENGTHS: [usize; 3] = [3_000, 9_000, 33_000];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The lab's FP8 prefill attention entries, normalized then partial.
fn prefill_entries() -> Vec<Entry> {
    latent_shapes::catalog()
        .into_iter()
        .filter(|entry| slices::Entry::of(entry).is_ok())
        .collect()
}

fn partial(entry: &Entry) -> bool {
    entry.name.ends_with("_partial")
}

/// A pool of FP8 pages for [`LENGTHS`], their tables and device copies.
struct Cache {
    max_pages: usize,
    page_stride: usize,
    d_pool: DeviceBuffer,
    d_tables: DeviceBuffer,
}

/// E4M3 -448, the code of a block's largest magnitude.
const MINUS_448: u8 = 0xfe;

impl Cache {
    /// A review's underflow construction (2026-10-09), the
    /// writer's per-block scales `RN(amax / 448)` and -448 codes, on two
    /// pages of one sequence: tokens 0 to 31 hold -2^-60 in latent block 0
    /// and -1 in the other blocks, tokens 32 to 63 are zero.
    fn underflow(session: &Session) -> LabResult<Cache> {
        let page_bytes = LatentKvFormat::Fp8E4M3Block128
            .page_bytes(GEOMETRY.latent, GEOMETRY.page_size) as usize;
        let page_stride = page_bytes + PAGE_GAP;
        let blocks = LATENT / 128;
        let mut pool = vec![0_u8; 2 * page_stride];
        for token in 0..32 {
            pool[token * LATENT..(token + 1) * LATENT].fill(MINUS_448);
            for block in 0..blocks {
                let amax = if block == 0 { 2_f64.powi(-60) } else { 1.0 };
                let scale = (amax / 448.0) as f32;
                let at = PAGE * LATENT + (token * blocks + block) * 4;
                pool[at..at + 4].copy_from_slice(&scale.to_ne_bytes());
            }
        }
        Ok(Cache {
            d_pool: upload(session, &pool)?,
            d_tables: upload(session, &bytes_i32(&[0, 1]))?,
            max_pages: 2,
            page_stride,
        })
    }

    fn new(session: &Session, rng: &mut Rng) -> LabResult<Cache> {
        let max_pages = LENGTHS.iter().map(|l| l.div_ceil(PAGE)).max().unwrap_or(1);
        let pages = LENGTHS.iter().map(|l| l.div_ceil(PAGE)).sum::<usize>();
        // Page ids in a shuffled order, so neighbours in a table are not
        // neighbours in the pool.
        let mut ids = (0..pages as i32).collect::<Vec<_>>();
        for i in (1..ids.len()).rev() {
            ids.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut tables = vec![-1; LENGTHS.len() * max_pages];
        let mut next = 0;
        for (s, &len) in LENGTHS.iter().enumerate() {
            for page in 0..len.div_ceil(PAGE) {
                tables[s * max_pages + page] = ids[next];
                next += 1;
            }
        }
        let page_bytes = LatentKvFormat::Fp8E4M3Block128
            .page_bytes(GEOMETRY.latent, GEOMETRY.page_size) as usize;
        let page_stride = page_bytes + PAGE_GAP;
        let mut pool = vec![0_u8; pages * page_stride];
        for page in pool.chunks_mut(page_stride) {
            for code in &mut page[..PAGE * LATENT] {
                // Any E4M3 code but NaN (0x7f and 0xff).
                let mut byte = rng.next() as u8;
                if byte & 0x7f == 0x7f {
                    byte ^= 1;
                }
                *code = byte;
            }
            for scale in page[PAGE * LATENT..page_bytes].chunks_mut(4) {
                let exponent = -9 - (rng.next() % 4) as i32;
                let value = (1.0 + 0.5 * rng.unit().abs()) * 2_f32.powi(exponent);
                scale.copy_from_slice(&value.to_ne_bytes());
            }
            for byte in &mut page[page_bytes..] {
                *byte = rng.next() as u8;
            }
        }
        Ok(Cache {
            d_pool: upload(session, &pool)?,
            d_tables: upload(session, &bytes_i32(&tables))?,
            max_pages,
            page_stride,
        })
    }
}

/// How a row's key list is cut.
#[derive(Clone, Copy, Debug)]
enum Holes {
    None,
    /// About one entry in nine.
    Random,
    /// Every entry.
    All,
    /// Units 3 to 5 and the last listed unit.
    Units,
    /// The first entry of every unit.
    UnitFirst,
}

/// The indexer's key list of a row at `position`, cut by `holes`.
fn key_list(rng: &mut Rng, position: usize, holes: Holes) -> Vec<i32> {
    let visible = position + 1;
    let complete = visible / POOL;
    let pools: Vec<usize> = if complete <= TOP_POOLS {
        (0..complete).collect()
    } else {
        let mut chosen = (0..complete).collect::<Vec<_>>();
        for i in 0..TOP_POOLS {
            let j = i + (rng.next() % (complete - i) as u64) as usize;
            chosen.swap(i, j);
        }
        chosen.truncate(TOP_POOLS);
        chosen.sort_unstable();
        chosen
    };
    let mut list = pools
        .iter()
        .flat_map(|&pool| pool * POOL..(pool + 1) * POOL)
        .chain(complete * POOL..visible)
        .map(|p| p as i32)
        .collect::<Vec<_>>();
    let listed = list.len();
    match holes {
        Holes::None => {}
        Holes::Random => {
            for entry in &mut list {
                if rng.next() % 9 == 0 {
                    *entry = -1;
                }
            }
        }
        Holes::All => list.fill(-1),
        Holes::Units => {
            let last = listed.saturating_sub(1) / UNIT;
            for (j, entry) in list.iter_mut().enumerate() {
                if (3..=5).contains(&(j / UNIT)) || j / UNIT == last {
                    *entry = -1;
                }
            }
        }
        Holes::UnitFirst => {
            for entry in list.iter_mut().step_by(UNIT) {
                *entry = -1;
            }
        }
    }
    list.resize(MAX_KEYS, -1);
    list
}

/// One row of a launch.
#[derive(Clone, Copy, Debug)]
struct Row {
    sequence: usize,
    position: usize,
    holes: Holes,
    /// Queries times this, for sharp softmaxes.
    q_scale: f32,
}

/// One launch's rows and operands.
struct Case {
    label: String,
    rows: Vec<Row>,
    budget: u32,
    q: Vec<u16>,
    lists: Vec<i32>,
}

impl Case {
    fn new(label: String, rows: Vec<Row>, budget: u32, seed: u64) -> Case {
        let mut rng = Rng(seed | 1);
        let mut q = Vec::with_capacity(rows.len() * HEADS * LATENT);
        let mut lists = Vec::with_capacity(rows.len() * MAX_KEYS);
        for row in &rows {
            assert!(row.position < LENGTHS[row.sequence], "{label}");
            lists.extend(key_list(&mut rng, row.position, row.holes));
            q.extend((0..HEADS * LATENT).map(|_| f32_to_bf16(rng.unit() * 2.0 * row.q_scale)));
        }
        Case {
            label,
            rows,
            budget,
            q,
            lists,
        }
    }

    /// `rows` rows ending at `context` in the 33K sequence, cut by `holes`.
    fn ending_at(label: String, rows: usize, context: usize, holes: Holes, seed: u64) -> Case {
        let rows = (0..rows)
            .map(|r| Row {
                sequence: 2,
                position: context - rows + r,
                holes,
                q_scale: 1.0,
            })
            .collect();
        Case::new(label, rows, MAX_KEYS as u32, seed)
    }

    /// Rows of every sequence at positions across the 2,048-token boundary
    /// and the contexts, every hole pattern among them.
    fn mixed(label: String, rows: usize, budget: u32, seed: u64) -> Case {
        const POSITIONS: [usize; 12] = [
            0, 3, 31, 33, 2046, 2047, 2048, 2051, 2999, 8191, 8999, 32_767,
        ];
        const HOLES: [Holes; 7] = [
            Holes::None,
            Holes::Random,
            Holes::None,
            Holes::Units,
            Holes::UnitFirst,
            Holes::All,
            Holes::Random,
        ];
        let mut rng = Rng(seed ^ 0x5eed);
        let rows = (0..rows)
            .map(|r| {
                let sequence = r % LENGTHS.len();
                let position = if r < POSITIONS.len() {
                    POSITIONS[r]
                } else {
                    (rng.next() % LENGTHS[sequence] as u64) as usize
                };
                Row {
                    sequence: if position < LENGTHS[sequence] {
                        sequence
                    } else {
                        2
                    },
                    position,
                    holes: HOLES[r % HOLES.len()],
                    q_scale: if r % 5 == 4 { 16.0 } else { 1.0 },
                }
            })
            .collect();
        Case::new(label, rows, budget, seed)
    }
}

/// A case's device operands and outputs.
struct Fixture<'a> {
    cache: &'a Cache,
    case: &'a Case,
    q: DeviceBuffer,
    row_batch: DeviceBuffer,
    lists: DeviceBuffer,
    live: DeviceBuffer,
    workspace: DeviceBuffer,
    out: DeviceBuffer,
    out_max: DeviceBuffer,
    out_den: DeviceBuffer,
    out_num: DeviceBuffer,
}

impl<'a> Fixture<'a> {
    fn new(session: &Session, cache: &'a Cache, case: &'a Case) -> LabResult<Fixture<'a>> {
        let rows = case.rows.len();
        let batch = case
            .rows
            .iter()
            .map(|row| row.sequence as i32)
            .collect::<Vec<_>>();
        Ok(Fixture {
            cache,
            case,
            q: upload(session, &bytes_u16(&case.q))?,
            row_batch: upload(session, &bytes_i32(&batch))?,
            lists: upload(session, &bytes_i32(&case.lists))?,
            live: upload(session, &(rows as u32).to_ne_bytes())?,
            workspace: session.allocate(16)?,
            out: session.allocate(rows * HEADS * LATENT * 2)?,
            out_max: session.allocate(rows * HEADS * 4)?,
            out_den: session.allocate(rows * HEADS * 4)?,
            out_num: session.allocate(rows * HEADS * LATENT * 4)?,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        poison::write(session, &self.live, 0, &live.to_ne_bytes())
    }

    /// Kernel arguments for `params` by name.
    fn args(&self, params: &[EntryParam]) -> LabResult<Vec<KernelArgument<'_>>> {
        params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => {
                    Ok(KernelArgument::Buffer(match name.as_str() {
                        "q" => &self.q,
                        "pool" => &self.cache.d_pool,
                        "block_tables" => &self.cache.d_tables,
                        "row_batch" => &self.row_batch,
                        "key_lists" => &self.lists,
                        "live_rows" => &self.live,
                        "workspace" => &self.workspace,
                        "out" => &self.out,
                        "out_max" => &self.out_max,
                        "out_den" => &self.out_den,
                        "out_num" => &self.out_num,
                        other => return Err(format!("unexpected pointer {other}")),
                    }))
                }
                EntryParam::U32 { name } => Ok(KernelArgument::U32(match name.as_str() {
                    "max_pages" => self.cache.max_pages as u32,
                    "budget" => self.case.budget,
                    "splits" => 1,
                    "scale_bits" => GEOMETRY.softmax_scale.to_bits(),
                    other => return Err(format!("unexpected scalar {other}")),
                })),
                EntryParam::U64 { name } if name == "page_stride_bytes" => {
                    Ok(KernelArgument::U64(self.cache.page_stride as u64))
                }
                other => Err(format!("unexpected parameter {other:?}")),
            })
            .collect()
    }

    fn launch(
        &self,
        session: &Session,
        function: &Function,
        entry: &Entry,
        mut plan: LaunchPlan,
        pdl: bool,
    ) -> LabResult<()> {
        plan.pdl = pdl;
        function
            .launch(
                &session.stream,
                launch::config(plan)?,
                &self.args(&entry.params())?,
            )
            .map_err(err)
    }

    /// The output buffers `entry` writes.
    fn buffers(&self, entry: &Entry) -> Vec<&DeviceBuffer> {
        if partial(entry) {
            vec![&self.out_max, &self.out_den, &self.out_num]
        } else {
            vec![&self.out]
        }
    }

    /// Every output buffer of `entry`, read back whole.
    fn read(&self, session: &Session, entry: &Entry) -> LabResult<Vec<u8>> {
        let mut bytes = Vec::new();
        for buffer in self.buffers(entry) {
            bytes.extend(download(session, buffer, buffer.byte_len())?);
        }
        Ok(bytes)
    }

    /// The outputs `entry` writes, and their write maps at `live` rows.
    fn outputs(
        &self,
        entry: &Entry,
        live: usize,
    ) -> LabResult<Vec<(&str, &DeviceBuffer, WriteMap)>> {
        let pairs = live * HEADS;
        Ok(if partial(entry) {
            vec![
                (
                    "out_max",
                    &self.out_max,
                    WriteMap::of(DType::F32).prefix(pairs)?,
                ),
                (
                    "out_den",
                    &self.out_den,
                    WriteMap::of(DType::F32).prefix(pairs)?,
                ),
                (
                    "out_num",
                    &self.out_num,
                    WriteMap::of(DType::F32).prefix(pairs * LATENT)?,
                ),
            ]
        } else {
            vec![(
                "out",
                &self.out,
                WriteMap::of(DType::Bf16).prefix(pairs * LATENT)?,
            )]
        })
    }

    /// Poisons every output, runs `launch` and reads every output back.
    fn run(
        &self,
        session: &Session,
        entry: &Entry,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<u8>> {
        session.stream.synchronize().map_err(err)?;
        for buffer in self.buffers(entry) {
            poison::fill_all(session, buffer, &[POISON])?;
        }
        launch()?;
        self.read(session, entry)
    }
}

/// The lab cubins: the latent entries, the served slice entries and each
/// mutant's.
struct Modules {
    latent: Module,
    slices: Module,
    mutants: BTreeMap<&'static str, Module>,
}

impl Modules {
    fn new(session: &Session) -> LabResult<Modules> {
        let mut mutants = BTreeMap::new();
        for (suffix, _) in slices::MUTANTS {
            mutants.insert(
                suffix,
                session.module(&cubin_name(latent_shapes::PROFILE, suffix))?,
            );
        }
        Ok(Modules {
            latent: session.module(&latent_shapes::cubin_name())?,
            slices: session.module(&cubin_name(latent_shapes::PROFILE, ""))?,
            mutants,
        })
    }
}

/// An entry's replaced and slice-widened kernels and plans at a row count.
struct Pair {
    entry: Entry,
    slices: slices::Entry,
    reference: Function,
    served: Function,
    plan: LaunchPlan,
    slices_plan: LaunchPlan,
}

fn function(module: &Module, symbol: &str, shared: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(err)?;
    function
        .set_max_dynamic_shared_memory(shared)
        .map_err(err)?;
    Ok(function)
}

impl Pair {
    fn new(modules: &Modules, entry: &Entry, rows: usize) -> LabResult<Pair> {
        let slices = slices::Entry::of(entry).map_err(|e| e.0.to_string())?;
        let plan = entry.plan(rows as u32).map_err(|e| e.0.to_string())?;
        let slices_plan = slices.plan(rows as u32).map_err(|e| e.0.to_string())?;
        Ok(Pair {
            reference: function(&modules.latent, &entry.symbol(), plan.dynamic_shared_bytes)?,
            served: function(
                &modules.slices,
                &slices.symbol(),
                slices_plan.dynamic_shared_bytes,
            )?,
            entry: entry.clone(),
            slices,
            plan,
            slices_plan,
        })
    }

    fn mutant(&self, modules: &Modules, suffix: &str) -> LabResult<Function> {
        let symbol = slices::Entry {
            suffix: suffix.into(),
            ..self.slices.clone()
        }
        .symbol();
        function(
            &modules.mutants[suffix],
            &symbol,
            self.slices_plan.dynamic_shared_bytes,
        )
    }

    fn reference(&self, session: &Session, fixture: &Fixture) -> LabResult<()> {
        fixture.launch(session, &self.reference, &self.entry, self.plan, false)
    }

    fn served(&self, session: &Session, fixture: &Fixture, pdl: bool) -> LabResult<()> {
        fixture.launch(session, &self.served, &self.entry, self.slices_plan, pdl)
    }
}

fn same(label: &str, actual: &[u8], expected: &[u8]) -> LabResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} bytes against {}",
            actual.len(),
            expected.len()
        ));
    }
    if let Some(at) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        return Err(format!(
            "{label}: {differing} of {} bytes differ, the first at byte {at}",
            actual.len()
        ));
    }
    Ok(())
}

/// The bitwise cases: every row count at the served budget over mixed
/// rows, whole prefill chunks ending at 2K, 8K and 32K, and the runtime
/// budgets of top-k 1024 and 4.
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for rows in [1, 2, 30, 36, 512, 2048] {
        cases.push(Case::mixed(
            format!("mixed rows {rows}"),
            rows,
            MAX_KEYS as u32,
            0x51ce + rows as u64,
        ));
    }
    for (context, rows, holes) in [
        (2_048, 2_048, Holes::None),
        (8_192, 2_048, Holes::Random),
        (8_192, 8_192, Holes::None),
        (32_768, 8_192, Holes::None),
        (32_768, 2_048, Holes::UnitFirst),
    ] {
        cases.push(Case::ending_at(
            format!("rows {rows} ending at {context} holes {holes:?}"),
            rows,
            context,
            holes,
            0xc0de + (context + rows) as u64,
        ));
    }
    for budget in [1_027, 7] {
        cases.push(Case::mixed(
            format!("mixed rows 36 budget {budget}"),
            36,
            budget,
            0xb0d6 + u64::from(budget),
        ));
    }
    cases
}

#[test]
fn latent_attention_slices_cpu_contract() {
    let entries = prefill_entries();
    assert_eq!(entries.len(), 2);
    for entry in &entries {
        let slices = slices::Entry::of(entry).unwrap();
        for rows in [1, 30, 2048, 8192] {
            let plan = slices.plan(rows).unwrap();
            assert_eq!(plan.grid, entry.plan(rows).unwrap().grid);
            assert_eq!(plan.block, [128, 1, 1]);
        }
        assert_eq!(slices.params(), entry.params());
    }
    for case in cases() {
        assert_eq!(
            case.lists.len(),
            case.rows.len() * MAX_KEYS,
            "{}",
            case.label
        );
        assert_eq!(
            case.q.len(),
            case.rows.len() * HEADS * LATENT,
            "{}",
            case.label
        );
    }
    // The underflow lists and the live-count matrix.
    for after in AFTER {
        let case = underflow_case(3, after);
        assert_eq!(case.lists.len(), 3 * MAX_KEYS);
        assert!(
            case.lists[..64]
                .iter()
                .enumerate()
                .all(|(j, &t)| t == j as i32)
        );
        assert!(case.lists[64..96].iter().all(|&t| t < 0));
    }
    assert_eq!(matrix_counts(2048), (0..=2048).collect::<Vec<_>>());
    for boundary in [0, 1, 375, 376, 4096, 8063, 8064, 8191, 8192] {
        assert!(matrix_counts(8192).contains(&boundary), "{boundary}");
    }
    assert!(matrix_counts(8064).iter().all(|&live| live <= 8064));
    // The hole patterns cut what they claim.
    let mut rng = Rng(7);
    let first = key_list(&mut rng, 20_000, Holes::UnitFirst);
    assert!(first.iter().step_by(UNIT).take(64).all(|&t| t < 0));
    assert!(first[1..UNIT].iter().all(|&t| t >= 0));
    let units = key_list(&mut rng, 20_000, Holes::Units);
    assert!(units[3 * UNIT..6 * UNIT].iter().all(|&t| t < 0));
    assert!(units[..3 * UNIT].iter().all(|&t| t >= 0));
    assert!(units[2048..].iter().all(|&t| t < 0));
}

/// Both entries at full live count over the same operands.
fn bitwise_case(
    session: &Session,
    modules: &Modules,
    cache: &Cache,
    entry: &Entry,
    case: &Case,
) -> LabResult<Value> {
    let fixture = Fixture::new(session, cache, case)?;
    let rows = case.rows.len();
    let pair = Pair::new(modules, entry, rows)?;
    let reference = fixture.run(session, entry, || pair.reference(session, &fixture))?;
    let served = fixture.run(session, entry, || pair.served(session, &fixture, false))?;
    same(
        &format!(
            "{} {}: slices against the prefill entry",
            entry.name, case.label
        ),
        &served,
        &reference,
    )?;
    Ok(json!({
        "entry": entry.name,
        "case": case.label,
        "rows": rows,
        "budget": case.budget,
        "grid": pair.slices_plan.grid,
        "dynamic_shared_bytes": pair.slices_plan.dynamic_shared_bytes,
    }))
}

#[test]
fn latent_attention_slices_match_the_prefill_entry_bitwise() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x1a7e17))?;
        let mut evidence = Vec::new();
        for case in cases() {
            for entry in prefill_entries() {
                evidence.push(bitwise_case(session, &modules, &cache, &entry, &case)?);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_bitwise"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_attention_slices_bitwise", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention against the prefill entry");
}

/// Rows whose queries hold NaN, +inf or -inf in one head.
#[test]
fn latent_attention_slices_match_on_non_finite_queries() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x1a7e19))?;
        let mut case = Case::mixed("non-finite queries".into(), 30, MAX_KEYS as u32, 0x0f);
        for (r, value) in [
            (1, f32::NAN),
            (5, f32::INFINITY),
            (9, f32::NEG_INFINITY),
            (14, f32::NAN),
        ] {
            let head = r % HEADS;
            case.q[(r * HEADS + head) * LATENT + 3 * r] = f32_to_bf16(value);
        }
        let mut evidence = Vec::new();
        for entry in prefill_entries() {
            evidence.push(bitwise_case(session, &modules, &cache, &entry, &case)?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_non_finite"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_attention_slices_non_finite", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention on non-finite queries");
}

/// How an underflow row's key list ends after its two units (tokens 0 to
/// 63 of [`Cache::underflow`]).
#[derive(Clone, Copy, Debug)]
enum After {
    /// The budget ends: no empty unit follows, the output stays -0.
    Nothing,
    /// One hole unit, then the budget ends.
    OneHoleUnit,
    /// Holes to the full budget.
    Holes,
    /// A hole unit, then the zero keys again.
    HoleUnitThenKeys,
}

const AFTER: [After; 4] = [
    After::Nothing,
    After::OneHoleUnit,
    After::Holes,
    After::HoleUnitThenKeys,
];

impl After {
    fn budget(self) -> u32 {
        match self {
            After::Nothing => 64,
            After::OneHoleUnit => 96,
            After::Holes => MAX_KEYS as u32,
            After::HoleUnitThenKeys => 128,
        }
    }

    /// Whether latent block 0 of the output ends at -0.
    fn negative_zero(self) -> bool {
        matches!(self, After::Nothing)
    }

    fn list(self) -> Vec<i32> {
        let mut list = vec![-1; MAX_KEYS];
        for (j, entry) in list.iter_mut().enumerate().take(64) {
            *entry = j as i32;
        }
        if let After::HoleUnitThenKeys = self {
            for (j, entry) in list.iter_mut().enumerate().take(128).skip(96) {
                *entry = (j - 64) as i32;
            }
        }
        list
    }
}

/// `rows` underflow rows ending as `after`, at its budget.
fn underflow_case(rows: usize, after: After) -> Case {
    let q = (0..rows * HEADS * LATENT)
        .map(|d| f32_to_bf16(if d % LATENT < 128 { 0.0 } else { 3.0 }))
        .collect();
    Case {
        label: format!("underflow {after:?} rows {rows}"),
        rows: (0..rows)
            .map(|_| Row {
                sequence: 0,
                position: 127,
                holes: Holes::None,
                q_scale: 1.0,
            })
            .collect(),
        budget: after.budget(),
        q,
        lists: (0..rows).flat_map(|_| after.list()).collect(),
    }
}

/// Row `r`'s bytes in each output buffer of `entry`, of a launch of `rows`
/// rows read back whole.
fn row_of(entry: &Entry, bytes: &[u8], rows: usize, r: usize) -> Vec<u8> {
    let widths: &[usize] = if partial(entry) {
        &[HEADS * 4, HEADS * 4, HEADS * LATENT * 4]
    } else {
        &[HEADS * LATENT * 2]
    };
    let mut offset = 0;
    let mut row = Vec::new();
    for &width in widths {
        row.extend_from_slice(&bytes[offset + r * width..][..width]);
        offset += width * rows;
    }
    row
}

/// The underflow construction: both forms' bytes equal at 1 to 8192 rows,
/// every row's bytes the same at every row count, and latent block 0 of the
/// output -0 exactly where no empty unit follows.
#[test]
fn latent_attention_slices_match_on_underflowed_outputs() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::underflow(session)?;
        let mut evidence = Vec::new();
        for entry in prefill_entries() {
            for after in AFTER {
                let mut first: Option<Vec<u8>> = None;
                for rows in [1_usize, 4, 375, 376, 2048, 8192] {
                    let case = underflow_case(rows, after);
                    evidence.push(bitwise_case(session, &modules, &cache, &entry, &case)?);
                    let fixture = Fixture::new(session, &cache, &case)?;
                    let pair = Pair::new(&modules, &entry, rows)?;
                    let bytes =
                        fixture.run(session, &entry, || pair.served(session, &fixture, false))?;
                    let row0 = first.get_or_insert_with(|| row_of(&entry, &bytes, rows, 0));
                    for r in 0..rows {
                        same(
                            &format!(
                                "{} {}: row {r} against the first row",
                                entry.name, case.label
                            ),
                            &row_of(&entry, &bytes, rows, r),
                            row0,
                        )?;
                    }
                }
                // Latent block 0 of every head: -0 where no empty unit
                // follows, else +0; the other blocks are negative.
                let row = first.ok_or("no underflow launch")?;
                let values: Vec<u32> = if partial(&entry) {
                    row[2 * HEADS * 4..]
                        .chunks_exact(4)
                        .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                } else {
                    row.chunks_exact(2)
                        .map(|b| u32::from(u16::from_ne_bytes([b[0], b[1]])) << 16)
                        .collect()
                };
                let zero = if after.negative_zero() {
                    0x8000_0000
                } else {
                    0
                };
                for (at, &bits) in values.iter().enumerate() {
                    let ok = if at % LATENT < 128 {
                        bits == zero
                    } else {
                        bits & 0x8000_0000 != 0 && bits & 0x7fff_ffff != 0
                    };
                    if !ok {
                        return Err(format!(
                            "{} underflow {after:?}: output {at} holds {bits:#010x}",
                            entry.name
                        ));
                    }
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_underflow"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_attention_slices_underflow", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention on underflowed outputs");
}

/// The served entry on poisoned outputs at every live count, each live
/// count's eager output compared with the replaced entry's.
#[test]
fn latent_attention_slices_replay_on_poisoned_outputs() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x1a7e23))?;
        let mut evidence = Vec::new();
        for rows in [1_usize, 30, 2048, 8192] {
            let case = Case::mixed(
                format!("replay rows {rows}"),
                rows,
                MAX_KEYS as u32,
                0x7e9 + rows as u64,
            );
            for entry in prefill_entries() {
                let fixture = Fixture::new(session, &cache, &case)?;
                let pair = Pair::new(&modules, &entry, rows)?;
                let mut counts = Vec::new();
                for live in poison::live_counts(rows as u32) {
                    let live = live as usize;
                    let label = format!("{} {}, live {live}", entry.name, case.label);
                    fixture.set_live(session, live as u32)?;
                    let reference =
                        fixture.run(session, &entry, || pair.reference(session, &fixture))?;
                    let mut replay = Replay::new(session);
                    for (name, buffer, map) in fixture.outputs(&entry, live)? {
                        replay = replay.output(name, buffer, map);
                    }
                    let mut receipt = BTreeMap::new();
                    replay_all(
                        session,
                        replay,
                        &fixture.live,
                        || pair.served(session, &fixture, true),
                        &mut receipt,
                    )?;
                    let eager = fixture.read(session, &entry)?;
                    same(
                        &format!("{label}: replayed slices against the prefill entry"),
                        &eager,
                        &reference,
                    )?;
                    counts.push(json!({"live": live, "replay": receipt}));
                }
                evidence.push(json!({
                    "entry": entry.name,
                    "case": case.label,
                    "live_counts": counts,
                }));
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_replay"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_attention_slices_replay", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention poisoned replays");
}

/// Capacities of the one-graph matrix: every live count up to 2048 rows;
/// above, the replay counts, the boundaries, and each multiple of 1024 with
/// its neighbours. The large capacities need no more counts: the form reads
/// the live count once per CTA, in its early return after the PDL wait,
/// and its CTAs share nothing through global memory
/// (`latent_attention_slices_read_the_live_count_once`), so a CTA's output
/// depends on the count only through whether its row is live. Counts 0 and
/// the capacity show every CTA both ways in one graph.
const MATRIX: [usize; 8] = [1, 30, 52, 375, 376, 2048, 8064, 8192];

fn matrix_counts(rows: usize) -> Vec<u32> {
    if rows <= 2048 {
        return (0..=rows as u32).collect();
    }
    let mut counts = poison::live_counts(rows as u32);
    counts.extend([2, 52, 375, 376, 2047, 2048, 2049, 8063, 8064]);
    for k in (1024..=rows as u32).step_by(1024) {
        counts.extend([k - 1, k, k + 1]);
    }
    counts.retain(|&live| live as usize <= rows);
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// The live-count argument of the large capacities ([`MATRIX`]), checked on
/// the form's source: `live_rows` is the parameter and one read, the
/// CTA-uniform early return after the PDL wait, and nothing synchronizes or
/// communicates across CTAs.
#[test]
fn latent_attention_slices_read_the_live_count_once() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../kernels-sm120/cuda/latent/attention_slices.cuh");
    let source = std::fs::read_to_string(&path).expect("the slice-widened header");
    assert_eq!(source.matches("live_rows").count(), 2);
    let wait = source.find("ptx::griddep_wait();").expect("the PDL wait");
    let read = source
        .find("if (row >= int(*live_rows)) return;")
        .expect("the early return");
    assert!(wait < read, "the live count is read after the PDL wait");
    let row = source
        .find("const int row = blockIdx.x;")
        .expect("the CTA's row");
    assert!(row < read);
    for word in [
        "atomic",
        "cluster",
        "grid_group",
        "__threadfence",
        "blockIdx.x +",
    ] {
        assert!(!source.contains(word), "{word}");
    }
}

/// One captured graph of the served entry per capacity, launched on
/// poisoned outputs at each live count the device count is set to: rows
/// below it hold the replaced entry's bytes, the rest keep the poison.
#[test]
fn latent_attention_slices_every_live_count_from_one_graph() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x1a7e31))?;
        let mut evidence = Vec::new();
        for rows in MATRIX {
            let case = Case::mixed(
                format!("live counts rows {rows}"),
                rows,
                MAX_KEYS as u32,
                0x11fe + rows as u64,
            );
            let counts = matrix_counts(rows);
            for entry in prefill_entries() {
                let fixture = Fixture::new(session, &cache, &case)?;
                let pair = Pair::new(&modules, &entry, rows)?;
                let reference =
                    fixture.run(session, &entry, || pair.reference(session, &fixture))?;
                let graph = session.capture(|_| pair.served(session, &fixture, true))?;
                for &live in &counts {
                    fixture.set_live(session, live)?;
                    let bytes = fixture.run(session, &entry, || {
                        graph.launch(&session.stream).map_err(err)
                    })?;
                    let mut offset = 0;
                    for (name, buffer, map) in fixture.outputs(&entry, live as usize)? {
                        let label = format!("{} {} {name}, live {live}", entry.name, case.label);
                        let len = buffer.byte_len();
                        let got = &bytes[offset..offset + len];
                        poison::check(&label, got, &map, Fill::Pattern(&[POISON]))?;
                        let written = map.written_bytes();
                        same(
                            &label,
                            &got[..written],
                            &reference[offset..offset + written],
                        )?;
                        offset += len;
                    }
                }
                evidence.push(json!({
                    "entry": entry.name,
                    "rows": rows,
                    "live_counts": counts.len(),
                    "first": counts.first(),
                    "last": counts.last(),
                }));
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_live_matrix"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_attention_slices_live_matrix", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention from one graph at every live count");
}

#[test]
fn latent_attention_slices_mutants_are_caught() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x1a7e29))?;
        let producer = CopyProducer::new(&modules.latent)?;
        let mut outcomes = Vec::new();
        for entry in prefill_entries() {
            for (suffix, holes) in [
                ("_skip_first", Holes::UnitFirst),
                ("_stale_page", Holes::None),
            ] {
                let case = Case::ending_at(
                    format!("{} {suffix}", entry.name),
                    30,
                    20_000,
                    holes,
                    0x3e7 + suffix.len() as u64,
                );
                let fixture = Fixture::new(session, &cache, &case)?;
                let pair = Pair::new(&modules, &entry, 30)?;
                let served =
                    fixture.run(session, &entry, || pair.served(session, &fixture, false))?;
                let mutant = pair.mutant(&modules, suffix)?;
                let caught = fixture.run(session, &entry, || {
                    fixture.launch(session, &mutant, &entry, pair.slices_plan, false)
                })?;
                if caught == served {
                    return Err(format!(
                        "{}: the mutant reproduced the served bits",
                        case.label
                    ));
                }
                outcomes.push(json!({"entry": entry.name, "mutant": suffix, "outcome": "caught"}));
            }

            // The queries arrive from a producer that releases its
            // dependent first; the operand starts poisoned.
            let case = Case::ending_at(
                format!("{} _late_wait", entry.name),
                30,
                20_000,
                Holes::None,
                0x1a7e,
            );
            let fixture = Fixture::new(session, &cache, &case)?;
            let pair = Pair::new(&modules, &entry, 30)?;
            let source = upload(session, &bytes_u16(&case.q))?;
            let eager = fixture.run(session, &entry, || pair.served(session, &fixture, false))?;
            let mutant = pair.mutant(&modules, "_late_wait")?;
            let behind = |late: bool| -> LabResult<Vec<u8>> {
                poison::fill_all(session, &fixture.q, &[POISON])?;
                fixture.run(session, &entry, || {
                    producer.copy(&session.stream, &source, &fixture.q)?;
                    if late {
                        fixture.launch(session, &mutant, &entry, pair.slices_plan, true)
                    } else {
                        pair.served(session, &fixture, true)
                    }
                })
            };
            if behind(false)? != eager {
                return Err(format!(
                    "{}: the served entry behind the query producer changed its output",
                    case.label
                ));
            }
            // A caught attempt is an observed difference; equal bytes on
            // every attempt fail the test.
            let mut attempts = 0;
            let mut caught = false;
            while attempts < 4 && !caught {
                attempts += 1;
                caught = behind(true)? != eager;
            }
            if !caught {
                return Err(format!(
                    "{}: the mutant reproduced the eager bits in {attempts} attempts",
                    case.label
                ));
            }
            let outcome = "caught";
            outcomes.push(json!({
                "entry": entry.name,
                "mutant": "_late_wait",
                "outcome": outcome,
                "attempts": attempts,
            }));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_mutants"));
        receipt.insert("mutants".into(), json!(outcomes));
        session.receipt("latent_attention_slices_mutants", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention mutants");
}

/// Graph replays of one launch (eight below 2048 rows), the prefill entry
/// (arm 0) or the slice-widened entry (arm 1), at row counts ending at 2K,
/// 8K and 32K contexts.
#[test]
fn latent_attention_slices_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("latent_attention_slices timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x7173))?;
        let entry = &prefill_entries()[0];
        let mut rows_out = Vec::new();
        for (context_name, context) in [("2k", 2_048_usize), ("8k", 8_192), ("32k", 32_768)] {
            for rows in [1_usize, 16, 2048, 8192] {
                if rows > context {
                    continue;
                }
                let case = Case::ending_at(
                    format!("timing {context_name} rows {rows}"),
                    rows,
                    context,
                    Holes::None,
                    0x7173 + rows as u64,
                );
                let fixture = Fixture::new(session, &cache, &case)?;
                let pair = Pair::new(&modules, entry, rows)?;
                let launches = if rows < 2048 { 8 } else { 1 };
                let graphs = [false, true].map(|slices| {
                    session.capture(|_| {
                        for _ in 0..launches {
                            if slices {
                                pair.served(session, &fixture, true)?;
                            } else {
                                fixture.launch(session, &pair.reference, entry, pair.plan, true)?;
                            }
                        }
                        Ok(())
                    })
                });
                let [reference, slices] = graphs;
                let graphs = [reference?, slices?];
                let paired = kernel_lab::time_ab(
                    session,
                    3,
                    15,
                    [kernel_lab::eviction::EvictionMode::Write; 2],
                    |arm, stream| graphs[arm].launch(stream).map_err(err),
                )?;
                let per = |ms: f32| f64::from(ms) * 1e3 / launches as f64;
                let row = json!({
                    "context": context_name,
                    "rows": rows,
                    "prefill_us": per(paired.arms[0].median_ms),
                    "slices_us": per(paired.arms[1].median_ms),
                    "slices_over_prefill": paired.paired_ratio,
                });
                eprintln!("latent_attention_slices timing: {row}");
                rows_out.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_attention_slices_timing"));
        receipt.insert("buckets".into(), json!(rows_out));
        session.receipt("latent_attention_slices_timing", receipt)?;
        Ok(())
    })
    .expect("slice-widened attention timing");
}
