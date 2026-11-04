//! Long-context eager prefill scoring and selection, 32K through 256K.
//!
//! Serving scores these full-source batches with the coarse guarded scorer
//! at 192 CTAs per row, then selects with the live selector and its merge,
//! or with the cached selector at a 131072-position extent. Each candidate
//! keeps the served entries, operands and arguments. It changes only the
//! scorer's symbol and grid x (the tile4 scorer at 8, 16, 32 or 192 CTAs per
//! row) and the selector's symbol (early stop, or cached). Layer 20 keeps the
//! served three-stage candidate producer.
//!
//! Before any timing, every case compares raw scores before selector
//! cleanup, final scores, status, ordered IDs and layer-20 candidate records
//! bit for bit, over one reused score plane per batch as serving writes it.
//! A one-bit corruption of an ordered ID must fail the comparator. Timing
//! interleaves an identical-operand A/A control with every visit, uses gated
//! CUDA events and cold caches, and refuses a changed power limit.
use super::*;
use kernels_sm120::pooled_indexer::{position_select_cached as cached, position_select_early};
use std::time::{Duration, Instant};

/// Batch extents: the first above the short tile band, 64K, the profiled
/// 100K and 200K mixed steps, the cached 128K extent and the 256K limit.
const CONTEXTS: [u32; 6] = [32832, 65536, 100544, 131072, 200512, 262144];
/// Step rows: a 512-row mixed step, a full chunk, and upper-skip and
/// adaptive tails.
const STEPS: [u32; 6] = [512, 8192, 64, 192, 448, 2048];
const CTAS: [u32; 4] = [8, 16, 32, 192];
/// Shared fixture kinds that apply to full sources (ties, repair scale
/// edges, NaN and off-grid queries, non-finite weights, scale bytes, page
/// holes, an inactive row) and `PACKED`, which mixes short and long rows.
const KINDS: [u32; 8] = [1, 2, 3, 4, 5, 7, 12, PACKED];
const PACKED: u32 = 13;
/// The adversarial cases run at both band edges and the cached extent.
const EDGES: [u32; 3] = [32832, 131072, 262144];
const LAYERS: [u32; 2] = [2, 20];
const BUDGET: Duration = Duration::from_secs(270);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Select {
    Early,
    Cached,
}

/// `None` is the served route.
type Arm = Option<(u32, Select)>;

fn arms() -> Vec<Arm> {
    let mut out = vec![None];
    for ctas in CTAS {
        for select in [Select::Early, Select::Cached] {
            out.push(Some((ctas, select)));
        }
    }
    out
}

fn arm_name(arm: Arm) -> String {
    match arm {
        None => "served".into(),
        Some((ctas, Select::Early)) => format!("tile{ctas}-early"),
        Some((ctas, Select::Cached)) => format!("tile{ctas}-cached"),
    }
}

/// The served cached-selector admission: a 128K extent, top-k 512 and the
/// table's row range.
fn served_cached(ratio: u32, width: u32, rows: u32) -> bool {
    use model_deepseek41_contract::tuning::{Family, select};
    u64::from(width) * u64::from(ratio) == 131072
        && select(
            &FLASH,
            Family::Indexer,
            "indexer.cached.prefill",
            "f32",
            &[("width", width), ("top_k", 512)],
            rows,
            rows,
        )
        .is_some()
}

/// One planned batch at absolute fixture rows.
#[derive(Clone, Debug)]
struct Unit {
    rows: std::ops::Range<u32>,
    width: u32,
    live: u32,
}

/// The served batches of the last `steps` rows of an 8192-row fixture.
fn units(layer: u32, context: u32, steps: u32, positions: &[i32], live: u32) -> Vec<Unit> {
    let first = 8192 - steps;
    plan(
        Shape {
            layer,
            context,
            rows: steps,
            case: 0,
        },
        &positions[first as usize..],
    )
    .into_iter()
    .map(|b| Unit {
        rows: first + b.rows.start..first + b.rows.end,
        width: b.width,
        live: live.saturating_sub(b.rows.start).min(b.rows.end - b.rows.start),
    })
    .collect()
}

/// The served launches of one batch over a reused score plane, before the
/// cached selector remap.
fn served(
    m: &mut Memory<'_>,
    layer: u32,
    input: &Input,
    out: &Outputs,
    u: &Unit,
    scratch: View,
) -> LabResult<Vec<Launch>> {
    let n = u.rows.end - u.rows.start;
    let b = backend(true);
    let q = m.row(input.dense.q, 8192, u.rows.clone());
    let head_weights = m.row(input.dense.head_weights, 128, u.rows.clone());
    let positions = m.row(input.dense.positions, 4, u.rows.clone());
    let row_batch = m.row(input.dense.row_batch, 4, u.rows.clone());
    let live_rows = m.input(&u.live.to_le_bytes())?;
    let scores = m.slice(out.scores, 0, u64::from(n) * u64::from(u.width) * 4);
    let status = m.row(out.status, 4, u.rows.clone());
    let output = m.row(out.ids, 2048, u.rows.clone());
    let op = b
        .position_indexer(
            &indexer::selection(&FLASH, layer, n).unwrap(),
            512,
            u.width,
            Need::rows(n),
        )
        .map_err(|e| e.to_string())?;
    let mut calls = op
        .launches(
            &PositionSelectBuffers {
                q,
                head_weights,
                keys: input.dense.keys,
                positions,
                row_batch,
                live_rows,
                candidate_mask: None,
                position_capacity: u.width,
                scores,
                status,
                output,
            },
            scratch,
            Counts::rows(n),
        )
        .map_err(|e| e.to_string())?;
    if layer == 20 {
        // Serving prepares the fixed-count producer when the runtime count
        // equals the geometry's candidate pool.
        let query = indexer::candidates(&FLASH, n);
        assert_eq!(query.top_k_blocks, 2048);
        let producer = b
            .indexer_candidate_block_ids(&query, u.width, Need::rows(n))
            .map_err(|e| e.to_string())?;
        let visible = m.row(input.visible, 4, u.rows.clone());
        let row_tags = m.row(input.tags, 8, u.rows.clone());
        let output = record(m, &out.candidates, u.rows.clone(), u.width);
        calls.extend(
            producer
                .launches(
                    &CandidateBlockIdsBuffers {
                        scores,
                        visible_positions: visible,
                        row_tags,
                        source_status: status,
                        live_rows,
                        output,
                    },
                    scratch,
                    Counts::rows(n),
                )
                .map_err(|e| e.to_string())?,
        );
    }
    Ok(calls)
}

/// Retarget the served launches of one batch to `arm`.
fn adapt(mut calls: Vec<Launch>, arm: Arm, ratio: u32, width: u32) -> LabResult<Vec<Launch>> {
    let n = calls[1].rows;
    match arm {
        None if served_cached(ratio, width, n) => {
            cached::remap(&mut calls[2]).map_err(|e| e.to_string())?
        }
        None => (),
        Some((ctas, select)) => {
            tile::remap(&mut calls[1], false, ctas).map_err(|e| e.to_string())?;
            match select {
                Select::Early => {
                    for call in &mut calls[2..4] {
                        position_select_early::remap(call).map_err(|e| e.to_string())?;
                    }
                }
                Select::Cached => cached::remap(&mut calls[2]).map_err(|e| e.to_string())?,
            }
        }
    }
    Ok(calls)
}

fn load_all(m: &Memory<'_>, modules: &mut BTreeMap<String, Module>, calls: &[Launch]) -> LabResult<()> {
    for call in calls {
        let (name, data) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
            .ok_or_else(|| format!("missing served cubin for {}", call.symbol))?;
        if !modules.contains_key(name) {
            modules.insert(name.into(), m.session.load_cubin(data)?);
        }
    }
    Ok(())
}

/// A view's leading `bytes`.
fn head(m: &mut Memory<'_>, view: View, bytes: u64) -> View {
    m.slice(view, 0, bytes)
}

/// Compare the per-row planes of `rows`.
fn compare_rows(
    m: &mut Memory<'_>,
    probe: &Module,
    a: &Outputs,
    b: &Outputs,
    rows: std::ops::Range<u32>,
    counters: View,
) -> LabResult<usize> {
    let pairs = [
        (a.status, b.status, 4),
        (a.ids, b.ids, 2048),
        (a.candidates.ids, b.candidates.ids, 8192),
        (a.candidates.row_tags, b.candidates.row_tags, 8),
        (a.candidates.visible_positions, b.candidates.visible_positions, 4),
        (a.candidates.status, b.candidates.status, 4),
    ];
    for (x, y, row) in pairs {
        let x = m.row(x, row, rows.clone());
        let y = m.row(y, row, rows.clone());
        compare(m, probe, x, y, counters)?;
    }
    Ok(pairs.len())
}

/// Bitwise pipelines of every candidate against the served route, over
/// poisoned outputs. Returns the comparison count.
#[allow(clippy::too_many_arguments)]
fn exact(
    m: &mut Memory<'_>,
    modules: &BTreeMap<String, Module>,
    probe: &Module,
    plans: &[Vec<Vec<Launch>>],
    units: &[Unit],
    a: &Outputs,
    b: &Outputs,
    counters: View,
) -> LabResult<usize> {
    let rows = units.first().unwrap().rows.start..units.last().unwrap().rows.end;
    let mut comparisons = 0;
    for candidate in &plans[1..] {
        a.poison(m, probe)?;
        b.poison(m, probe)?;
        for (k, u) in units.iter().enumerate() {
            let plane = u64::from(u.rows.end - u.rows.start) * u64::from(u.width) * 4;
            let (x, y) = (head(m, a.scores, plane), head(m, b.scores, plane));
            m.fire(modules, &plans[0][k][..2])?;
            m.fire(modules, &candidate[k][..2])?;
            compare(m, probe, x, y, counters)?;
            m.fire(modules, &plans[0][k][2..])?;
            m.fire(modules, &candidate[k][2..])?;
            compare(m, probe, x, y, counters)?;
            comparisons += 2;
        }
        comparisons += compare_rows(m, probe, a, b, rows.clone(), counters)?;
    }
    // A one-bit ordered-ID corruption must fail the same comparator.
    let first = m.row(a.ids, 2048, rows.start..rows.start + 1);
    let target = m.row(b.ids, 2048, rows.start..rows.start + 1);
    let word = m.slice(target, 0, 4);
    let bits = u32::from_le_bytes(m.read(word)?[..4].try_into().unwrap());
    m.fill(probe, word, bits ^ 1)?;
    m.fill(probe, counters, 0)?;
    let words = u32::try_from(first.bytes / 4).unwrap();
    m.compare(probe, first, target, counters, 1, 1, words, words, 0, 0)?;
    if m.read(counters)?[..4] != 1_u32.to_le_bytes() {
        return Err("ordered-ID corruption control escaped".into());
    }
    m.fill(probe, word, bits)?;
    m.fill(probe, counters, 0)?;
    Ok(comparisons)
}

fn gpu_snapshot(session: &Session) -> LabResult<serde_json::Value> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=enforced.power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
            "--id",
            session.metadata["uuid"]
                .as_str()
                .ok_or("missing device identity")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("GPU telemetry query failed".into());
    }
    let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let values = text
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    if values.len() != 2 || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("GPU power and clock telemetry is unavailable".into());
    }
    Ok(serde_json::json!({"enforced_power_w": values[0], "sm_mhz": values[1]}))
}

/// The fixture of one group. `PACKED` puts short rows on every odd row.
fn host(layer: u32, context: u32, rows: u32, kind: u32) -> LabResult<Host> {
    let base = if kind == PACKED { 0 } else { kind };
    let mut h = Host::new(Case {
        layer,
        context,
        rows,
        kind: base,
        capture: None,
    })?;
    if kind == PACKED {
        let ratio = FLASH.compress_ratio(layer) as i32;
        for (i, p) in h.positions.iter_mut().enumerate() {
            if i % 2 == 1 {
                *p = (i as i32 * 997) % 32768;
            }
        }
        h.visible = h.positions.iter().map(|p| (p + 1) / ratio).collect();
    }
    Ok(h)
}

fn selected_contexts() -> Vec<u32> {
    match std::env::var("CHOPINFER_DSIDX_CONTEXTS") {
        Ok(list) => list
            .split(',')
            .map(|c| c.trim().parse().expect("context list"))
            .collect(),
        Err(_) => CONTEXTS.to_vec(),
    }
}

/// The groups: each main (layer, context) and each adversarial case.
fn groups() -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for context in CONTEXTS {
        for layer in LAYERS {
            out.push((layer, context, 0));
        }
    }
    for context in EDGES {
        for layer in LAYERS {
            for kind in KINDS {
                out.push((layer, context, kind));
            }
        }
    }
    out
}

/// Alternate main groups between the shards by context and layer, so each
/// shard holds both sources; adversarial cases alternate by index.
fn shard_of(id: usize, (layer, context, kind): (u32, u32, u32)) -> usize {
    if kind == 0 {
        let i = CONTEXTS.iter().position(|&c| c == context).unwrap();
        let j = LAYERS.iter().position(|&l| l == layer).unwrap();
        (i + j) % 2
    } else {
        id % 2
    }
}

#[test]
fn cpu_long_prefill_arms_change_only_the_scorer_grid_and_selector() {
    let mut symbols = BTreeMap::new();
    let mut cached_units = 0;
    let mut timed = 0;
    for (layer, context, kind) in groups() {
        let ratio = FLASH.compress_ratio(layer);
        let rows = if kind == 0 { 8192 } else { 65 };
        let positions: Vec<i32> = if kind == PACKED {
            (0..rows)
                .map(|i| {
                    if i % 2 == 1 {
                        (i as i32 * 997) % 32768
                    } else {
                        (context - rows + i) as i32
                    }
                })
                .collect()
        } else {
            (context - rows..context).map(|p| p as i32).collect()
        };
        let steps: &[u32] = if kind == 0 { &STEPS } else { &[65] };
        for &step in steps {
            let mut padded = vec![0; 8192 - rows as usize];
            padded.extend(&positions);
            for u in units(layer, context, step, &padded, step) {
                let n = u.rows.end - u.rows.start;
                assert!(u64::from(n) * u64::from(u.width) * 4 <= SCORE_BYTES);
                assert!(n <= if layer == 20 { 512 } else { 4096 });
                // The band starts above the short tile band and ends at 256K.
                let extent = u.width * ratio;
                assert!((32768..=262144).contains(&extent), "{extent}");
                let b = backend(true);
                let dense = PositionSelectBuffers {
                    scores: v(12, u64::from(n) * u64::from(u.width) * 4),
                    ..cpu_buffers(n, ratio, u.width)
                };
                let base = b
                    .position_indexer(
                        &indexer::selection(&FLASH, layer, n).unwrap(),
                        512,
                        u.width,
                        Need::rows(n),
                    )
                    .unwrap()
                    .launches(&dense, v(15, 1 << 32), Counts::rows(n))
                    .unwrap();
                assert_eq!(base.len(), 4);
                assert!(base[1].symbol.ends_with("_mma_fp4_guard_fast_v1"));
                assert_eq!(base[1].plan.grid, [192, n, 1]);
                cached_units += usize::from(served_cached(ratio, u.width, n));
                for arm in arms() {
                    let now = adapt(base.clone(), arm, ratio, u.width).unwrap();
                    for (i, (old, new)) in base.iter().zip(&now).enumerate() {
                        assert_eq!(old.arguments, new.arguments);
                        assert_eq!(old.rows, new.rows);
                        assert_eq!(old.plan.block, new.plan.block);
                        assert_eq!(old.plan.grid[1..], new.plan.grid[1..]);
                        assert_eq!(
                            (old.plan.dynamic_shared_bytes, old.plan.workspace_bytes),
                            (new.plan.dynamic_shared_bytes, new.plan.workspace_bytes)
                        );
                        let changed = match (arm, i) {
                            (Some(_), 1) => true,
                            (Some((_, Select::Early)), 2 | 3) => true,
                            (Some((_, Select::Cached)), 2) => true,
                            (None, 2) => served_cached(ratio, u.width, n),
                            _ => false,
                        };
                        assert_eq!(old.symbol != new.symbol, changed, "{arm:?} {i}");
                        let grid = match (arm, i) {
                            (Some((ctas, _)), 1) => ctas,
                            _ => old.plan.grid[0],
                        };
                        assert_eq!(new.plan.grid[0], grid);
                        assert!(
                            bundle_deepseek41::sm120::fold_cubin_for_symbol(&new.symbol).is_some(),
                            "{}",
                            new.symbol
                        );
                        *symbols.entry(new.symbol.clone()).or_insert(0) += 1;
                    }
                }
                timed += 1;
            }
        }
    }
    // Ratio-2 scores and selectors, layer-20 scores and selectors, served,
    // tile4, live, merge, early, early merge and cached entries.
    let distinct = |suffix: &str| symbols.keys().filter(|s| s.ends_with(suffix)).count();
    assert_eq!(distinct("_mma_fp4_guard_fast_v1"), 2);
    assert_eq!(distinct("_mma_fp4_tile4_mode0_trace0_v1"), 2);
    assert_eq!(distinct("_select_v3_live_v1"), 2);
    assert_eq!(distinct("_select_v3_live_merge_v1"), 2);
    assert_eq!(distinct("_select_v3_live_early_v1"), 2);
    assert_eq!(distinct("_select_v3_live_merge_early_v1"), 2);
    assert_eq!(distinct("_select_v3_cached4_v1"), 2);
    // Every 131072-extent batch of at least 32 rows keeps the cached route.
    assert!(cached_units > 0);
    assert_eq!(groups().len(), 12 + EDGES.len() * LAYERS.len() * KINDS.len());
    assert!(timed > 0);
    assert_eq!(arms().len(), 9);
}

#[test]
#[ignore = "pair-lab: an idle GPU per shard, CUDA events, A/A control, at most five minutes"]
fn gpu_long_prefill_scores_and_selection_pair() {
    let start = Instant::now();
    let shard: usize = std::env::var("CHOPINFER_L4_SHARD")
        .expect("CHOPINFER_L4_SHARD")
        .parse()
        .unwrap();
    assert!(shard < 2);
    let contexts = selected_contexts();
    let ran = kernel_harness::gpu_test_with(test_cubins(), 16 << 30, |session| {
        let initial = gpu_snapshot(session)?;
        let limit = initial["enforced_power_w"].as_f64().ok_or("missing power limit")?;
        let snapshot = || -> LabResult<serde_json::Value> {
            let state = gpu_snapshot(session)?;
            if state["enforced_power_w"].as_f64() != Some(limit) {
                return Err("power limit changed; pair-lab job is void".into());
            }
            Ok(state)
        };
        let probe = session.load_cubin(&bytes(&artifact().join("indexer_prefill_probe.cubin"))?)?;
        let mut modules = BTreeMap::new();
        let mut results = Vec::new();
        for (id, (layer, context, kind)) in groups().into_iter().enumerate() {
            if shard_of(id, (layer, context, kind)) != shard || !contexts.contains(&context) {
                continue;
            }
            if start.elapsed() >= BUDGET {
                return Err("pair-lab time budget exhausted; no complete result".into());
            }
            let ratio = FLASH.compress_ratio(layer);
            let rows = if kind == 0 { 8192 } else { 65 };
            let mut m = Memory {
                session,
                buffers: Vec::new(),
                views: Vec::new(),
            };
            let h = host(layer, context, rows, kind)?;
            let c = Case {
                layer,
                context,
                rows,
                kind: if kind == PACKED { 0 } else { kind },
                capture: None,
            };
            let input = Input::upload(&mut m, c, &h)?;
            // Outputs keep all fixture rows; one score plane is reused per
            // batch, as in serving.
            let a = Outputs::allocate(&mut m, rows, SCORE_BYTES, context)?;
            let b = Outputs::allocate(&mut m, rows, SCORE_BYTES, context)?;
            let live = if kind == 12 { rows - 1 } else { rows };
            let mut positions = vec![0; 8192 - rows as usize];
            positions.extend(&h.positions);
            let steps: &[u32] = if kind == 0 { &STEPS } else { &[65] };
            // Fixture rows are the last `rows` of 8192.
            let shift = |u: Unit| Unit {
                rows: u.rows.start - (8192 - rows)..u.rows.end - (8192 - rows),
                ..u
            };
            let step_units: Vec<(u32, Vec<Unit>)> = steps
                .iter()
                .map(|&s| {
                    let live = if kind == 12 { s - 1 } else { s };
                    (s, units(layer, context, s, &positions, live).into_iter().map(shift).collect())
                })
                .collect();
            let workspace = step_units
                .iter()
                .flat_map(|(_, us)| us.iter())
                .map(|u| {
                    let n = u.rows.end - u.rows.start;
                    let b = backend(true);
                    let score = b
                        .position_indexer(&indexer::selection(&FLASH, layer, n).unwrap(), 512, u.width, Need::rows(n))
                        .unwrap()
                        .info()
                        .workspace;
                    if layer == 20 {
                        score.max(
                            b.indexer_candidate_block_ids(&indexer::candidates(&FLASH, n), u.width, Need::rows(n))
                                .unwrap()
                                .info()
                                .workspace,
                        )
                    } else {
                        score
                    }
                })
                .max()
                .unwrap();
            let scratch = [m.alloc(workspace as usize)?, m.alloc(workspace as usize)?];
            let counters = m.input(&[0; 32])?;
            let mut comparisons = 0;
            let mut timed = BTreeMap::new();
            for (step, us) in &step_units {
                let plans = arms()
                    .into_iter()
                    .enumerate()
                    .map(|(i, arm)| {
                        let (out, scratch) = if i == 0 { (&a, scratch[0]) } else { (&b, scratch[1]) };
                        us.iter()
                            .map(|u| adapt(served(&mut m, layer, &input, out, u, scratch)?, arm, ratio, u.width))
                            .collect::<LabResult<Vec<_>>>()
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                for arm in &plans {
                    for p in arm {
                        load_all(&m, &mut modules, p)?;
                    }
                }
                comparisons += exact(&mut m, &modules, &probe, &plans, us, &a, &b, counters)?;
                if kind != 0 {
                    // The shared fixture's documented outcomes, on the served route.
                    a.poison(&m, &probe)?;
                    for p in &plans[0] {
                        m.fire(&modules, p)?;
                    }
                    let first_status = m.slice(a.status, 0, 4);
                    if matches!(kind, 3..=5) && m.read(first_status)? != 1_i32.to_le_bytes() {
                        return Err(format!("kind {kind}: row 0 did not fail"));
                    }
                    let second_status = m.slice(a.status, 4, 4);
                    if kind == 3 && m.read(second_status)? != 0_i32.to_le_bytes() {
                        return Err("finite SIMT fallback row failed".into());
                    }
                    if kind == 1 {
                        let ids = m.slice(a.ids, 0, 512 * 4);
                        if m.read(first_status)? != 0_i32.to_le_bytes() || m.read(ids)? != ints(0..512) {
                            return Err("tied scores did not keep position order".into());
                        }
                    }
                    let inactive = m.slice(a.status, u64::from(live) * 4, 4);
                    if kind == 12 && m.read(inactive)? != POISON.to_le_bytes() {
                        return Err("inactive row was written".into());
                    }
                }
                println!(
                    "LONG_PREFILL_CASE {}",
                    serde_json::json!({"layer": layer, "context": context, "kind": kind, "rows": step,
                        "batches": us.iter().map(|u| [u.rows.end - u.rows.start, u.width]).collect::<Vec<_>>(),
                        "served_cached": us.iter().map(|u| served_cached(ratio, u.width, u.rows.end - u.rows.start)).collect::<Vec<_>>(),
                        "arms": arms().len(), "comparisons_total": comparisons, "mismatches": 0, "corruption_rejected": true})
                );
                if kind != 0 {
                    continue;
                }
                for (k, u) in us.iter().enumerate() {
                    let n = u.rows.end - u.rows.start;
                    if timed.contains_key(&n) {
                        continue;
                    }
                    timed.insert(n, *step);
                    if start.elapsed() >= BUDGET {
                        return Err("pair-lab time budget exhausted; no complete result".into());
                    }
                    let one: Vec<&Vec<Launch>> = plans.iter().map(|arm| &arm[k]).collect();
                    a.poison(&m, &probe)?;
                    b.poison(&m, &probe)?;
                    let graphs = one
                        .iter()
                        .map(|p| session.capture(|_| m.fire(&modules, p)))
                        .collect::<LabResult<Vec<_>>>()?;
                    let plane = u64::from(n) * u64::from(u.width) * 4;
                    let label = format!("long-prefill-l{layer}-c{context}-r{n}");
                    let mut visits = Vec::new();
                    for visit in 0..3 {
                        let gpu_start = snapshot()?;
                        let control = kernel_harness::with_timing_label(&format!("{label}-aa"), || {
                            kernel_harness::time_ab(session, 1, 3, [EvictionMode::Write; 2], |_, stream| {
                                graphs[0].launch(stream).map_err(|e| e.to_string())
                            })
                        })?;
                        let spread = (1.0 - control.paired_ratio).abs() * 100.0;
                        let mut candidates = Vec::new();
                        let order: Vec<usize> = if visit % 2 == 0 {
                            (1..graphs.len()).collect()
                        } else {
                            (1..graphs.len()).rev().collect()
                        };
                        for i in order {
                            let timing = kernel_harness::with_timing_label(
                                &format!("{label}-{}", arm_name(arms()[i])),
                                || {
                                    kernel_harness::time_ab(session, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                                        graphs[if arm == 0 { 0 } else { i }].launch(stream).map_err(|e| e.to_string())
                                    })
                                },
                            )?;
                            // The timed graphs must still agree.
                            let (x, y) = (head(&mut m, a.scores, plane), head(&mut m, b.scores, plane));
                            compare(&m, &probe, x, y, counters)?;
                            compare_rows(&mut m, &probe, &a, &b, u.rows.clone(), counters)?;
                            let gain = (1.0 - timing.paired_ratio) * 100.0;
                            candidates.push(serde_json::json!({"arm": arm_name(arms()[i]),
                                "served_ms": timing.arms[0].median_ms, "candidate_ms": timing.arms[1].median_ms,
                                "gain_pct": gain, "gain_exceeds_control": gain > spread}));
                        }
                        visits.push(serde_json::json!({"visit": visit, "gpu_start": gpu_start, "gpu_end": snapshot()?,
                            "aa_ms": control.arms.map(|t| t.median_ms), "aa_spread_pct": spread, "candidates": candidates}));
                    }
                    let row = serde_json::json!({"layer": layer, "context": context, "rows": n, "width": u.width,
                        "step": step, "served_cached": served_cached(ratio, u.width, n), "visits": visits});
                    println!("LONG_PREFILL_TIMING {row}");
                    results.push(row);
                }
            }
        }
        session.receipt(
            "deepseek41-indexer-prefill-long",
            BTreeMap::from([
                ("shard".into(), serde_json::json!(shard)),
                ("contexts".into(), serde_json::json!(contexts)),
                ("results".into(), serde_json::json!(results)),
                ("gpu_start".into(), initial.clone()),
                ("gpu_end".into(), snapshot()?),
            ]),
        )?;
        println!("LONG_PREFILL_COMPLETE shard={shard} timed={} seconds={:.1}", results.len(), start.elapsed().as_secs_f64());
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "GPU test skipped");
    assert!(start.elapsed() < Duration::from_secs(295));
}
