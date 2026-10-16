//! Selected attention partials against QSA's served selected attention
//! (`attn_{bf16,fp8}_{decode,verify,prefill}` of the TP4 profile, with its
//! split merge). Both run on the same rows, queries, KV and selected block
//! lists; QSA's output and the per-rank partials at group sizes 2 and 4,
//! merged by `partial_merge{N}`, must each pass their oracle gates: the f64
//! block-sparse attention's gate, the merge with the `exp2f` allowance as
//! well. Passing both gates bounds the served and merged outputs' difference
//! only by the sum of the two bounds, not by one gate. Their paired
//! difference (bitwise equal rows, differing elements, the largest absolute
//! and BF16-ulp difference) is reported as a diagnostic and not gated.
//!
//! - Synthetic: random selections of decode, verify and prefill rows across
//!   the 2051-token dense limit.
//! - Captured: `CHOPINFER_QSA_OPERANDS` names a `qwen38-oracle qsa-capture
//!   --export` directory or a directory of them (`exports-v2`). Per layer,
//!   QSA's own `index_key_write`, `index_query`, `block_scores_decode` and
//!   `block_topk` run on the projection rows and select the blocks of up to
//!   `SAMPLED_ROWS` evenly spaced exported rows. The exports hold no
//!   attention operands, so the queries and KV are synthetic BF16 and FP8
//!   over the layer's context; the selections are QSA's.
//!
//! Run in release with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test qsa_selected_partials`.

#[path = "common/selected_partials.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../paged_partials_shapes.rs"]
mod merge_shapes;
#[path = "../selected_partials_shapes.rs"]
mod shapes;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use fixture::{
    BLOCK, DIM, Kv, Launch, PAGE, Partials, Q_HEADS, Rng, Step, TOP, bf16_values, download, err,
    full_reference, gate, gate_merged, merge, queries, rank_cache, run, scales, selections, upload,
    words,
};
use kernel_core::contract::DType;
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::attention::sparse;
use kernels_sm120::selected_partials::Entry;
use runtime_cuda::{KernelArgument, Module};
use serde_json::{Value, json};

/// Indexer query and key width.
const INDEX_DIM: usize = 128;
const INDEX_HEADS: usize = 4;
const PROJECTION: usize = (INDEX_HEADS + 1) * INDEX_DIM;
/// Block-key page stride: 4 KiB of block keys, then a 1 KiB gap.
const KEY_STRIDE: usize = 5120;
const HISTORY_ROWS: usize = 8191;
/// Exported rows of a layer whose selections are checked.
const SAMPLED_ROWS: usize = 32;

fn entry(kv: &str, phase: &str) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name == format!("{}_{kv}_{phase}", shapes::PROFILE.tag()))
        .unwrap_or_else(|| panic!("no lab entry {kv} {phase}"))
}

fn qsa_entry(name: &str) -> sparse::Entry {
    sparse::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.name == format!("qwen38_{name}"))
        .unwrap_or_else(|| panic!("no QSA entry {name}"))
}

fn function(module: &Module, symbol: &str, shared: u32) -> LabResult<runtime_cuda::Function> {
    let function = module.function(symbol).map_err(err)?;
    if shared > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

/// The lab geometry is QSA's TP4 attention.
fn check_geometry() -> LabResult<()> {
    let attention = qsa_entry("attn_bf16_decode").geometry;
    let g = shapes::GEOMETRY;
    let qsa = (
        attention.q_heads / attention.kv_heads,
        attention.kv_heads,
        attention.head_dim,
        attention.page_size,
        attention.block,
        attention.top_blocks,
    );
    if qsa
        != (
            g.q_heads,
            g.kv_heads,
            g.head_dim,
            g.page,
            g.block,
            g.top_blocks,
        )
        || (Q_HEADS, DIM, PAGE, BLOCK, TOP) != (6, 256, 64, 4, 512)
    {
        return Err(format!("the lab geometry {g:?} is not QSA's {qsa:?}"));
    }
    Ok(())
}

/// QSA's served selected attention of `step` over `kv` (one rank holding
/// every page) and `selected`: its BF16 output `[rows, heads, DIM]`.
#[allow(clippy::too_many_arguments)]
fn qsa_attention(
    session: &Session,
    module: &Module,
    phase: &str,
    step: &Step,
    q: &[u16],
    selected: &[i32],
    kv: &Kv,
    rng: &mut Rng,
) -> LabResult<Vec<f32>> {
    let dtype = if kv.fp8 { "fp8" } else { "bf16" };
    let entry = qsa_entry(&format!("attn_{dtype}_{phase}"));
    let rows = step.rows.len();
    let context = step.rows.iter().map(|&(_, p)| p + 1).max().unwrap_or(1) as u32;
    let cache = rank_cache(kv, 0, 1, false, rng);
    let plan = entry
        .plan(rows as u32, context)
        .map_err(|e| e.0.to_string())?;
    let splits = plan.grid[2];
    let q_buffer = upload(
        session,
        &q.iter().flat_map(|b| b.to_ne_bytes()).collect::<Vec<_>>(),
    )?;
    let k = upload(session, &cache.k)?;
    let v = upload(session, &cache.v)?;
    let tables = upload(session, &words(&cache.table_words()))?;
    let row_batch = upload(
        session,
        &words(&step.rows.iter().map(|&(i, _)| i as i32).collect::<Vec<_>>()),
    )?;
    let positions = upload(
        session,
        &words(&step.rows.iter().map(|&(_, p)| p as i32).collect::<Vec<_>>()),
    )?;
    let live = upload(session, &(rows as u32).to_ne_bytes())?;
    let selected = upload(session, &words(selected))?;
    let workspace = session.allocate((plan.workspace_bytes as usize).max(16))?;
    let out = session.allocate(rows * Q_HEADS * DIM * 2)?;
    let (scale, k_scale, v_scale) = scales(kv.fp8);
    function(module, &entry.symbol(), entry.dynamic_shared_bytes())?
        .launch(
            &session.stream,
            launch::config(plan)?,
            &[
                KernelArgument::Buffer(&q_buffer),
                KernelArgument::Buffer(&k),
                KernelArgument::Buffer(&v),
                KernelArgument::Buffer(&tables),
                KernelArgument::Buffer(&row_batch),
                KernelArgument::Buffer(&positions),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&selected),
                KernelArgument::Buffer(&workspace),
                KernelArgument::Buffer(&out),
                KernelArgument::U32(cache.max_pages as u32),
                KernelArgument::U64(cache.stride as u64),
                KernelArgument::U64(cache.stride as u64),
                KernelArgument::U32(splits),
                KernelArgument::U32(scale.to_bits()),
                KernelArgument::U32(k_scale.to_bits()),
                KernelArgument::U32(v_scale.to_bits()),
            ],
        )
        .map_err(err)?;
    if let Some(merge_plan) = entry.merge_plan(rows as u32).map_err(|e| e.0.to_string())? {
        let symbol = entry.merge_symbol().ok_or("an attention entry")?;
        function(module, &symbol, 0)?
            .launch(
                &session.stream,
                launch::config(merge_plan)?,
                &[
                    KernelArgument::Buffer(&workspace),
                    KernelArgument::Buffer(&live),
                    KernelArgument::Buffer(&out),
                    KernelArgument::U32(splits),
                ],
            )
            .map_err(err)?;
    }
    Ok(bf16_values(&download(
        session,
        &out,
        rows * Q_HEADS * DIM * 2,
    )?))
}

/// The paired difference of the merged and the served BF16 outputs at one
/// group size: a diagnostic, not gated.
#[derive(Default)]
struct Paired {
    /// Rows whose merged output equals QSA's bit for bit.
    equal_rows: usize,
    elements: usize,
    differing_elements: usize,
    max_abs_difference: f64,
    /// The largest distance in BF16 values (ulps) between paired elements.
    max_bf16_ulps: u32,
}

impl Paired {
    fn add(&mut self, merged: &[f32], served: &[f32], width: usize) {
        // BF16 bits in value order: negative values below positive ones.
        let ordered = |x: f32| {
            let bits = (x.to_bits() >> 16) as i32;
            if bits & 0x8000 != 0 {
                -(bits & 0x7fff)
            } else {
                bits
            }
        };
        for (m, s) in merged.chunks_exact(width).zip(served.chunks_exact(width)) {
            let mut equal = true;
            for (&a, &b) in m.iter().zip(s) {
                self.elements += 1;
                if a.to_bits() == b.to_bits() {
                    continue;
                }
                equal = false;
                self.differing_elements += 1;
                self.max_abs_difference = self
                    .max_abs_difference
                    .max((f64::from(a) - f64::from(b)).abs());
                self.max_bf16_ulps = self.max_bf16_ulps.max(ordered(a).abs_diff(ordered(b)));
            }
            self.equal_rows += usize::from(equal);
        }
    }
}

/// What one comparison measured.
#[derive(Default)]
struct Tally {
    rows: usize,
    sparse_rows: usize,
    /// Per group size.
    paired: BTreeMap<usize, Paired>,
}

/// QSA's selected attention and the merged per-rank partials of `step`,
/// each against its oracle gate, and their paired difference.
#[allow(clippy::too_many_arguments)]
fn compare(
    session: &Session,
    modules: &(Module, Module, Module),
    phase: &str,
    step: &Step,
    selected: &[i32],
    fp8: bool,
    rng: &mut Rng,
    label: &str,
    tally: &mut Tally,
) -> LabResult<()> {
    let (qsa, partials, merges) = modules;
    let merge_entries = merge_shapes::merge_entries();
    let kv = Kv::new(&step.lengths, fp8, false, rng);
    let q = queries(step.rows.len(), false, rng);
    let rows = step.rows.len();
    let full = full_reference(step, &q, selected, &kv, scales(fp8))?;
    let served = qsa_attention(session, qsa, phase, step, &q, selected, &kv, rng)?;
    gate(
        &format!("{label} QSA attention"),
        &served.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        &full,
        DType::Bf16,
    )?;
    let entry = entry(if fp8 { "fp8" } else { "bf16" }, phase);
    let function = partials.function(&entry.symbol()).map_err(err)?;
    let width = Q_HEADS * DIM;
    for size in [2_usize, 4] {
        let parts = (0..size)
            .map(|rank| {
                let cache = rank_cache(&kv, rank, size, false, rng);
                let operands = Launch {
                    step,
                    q: &q,
                    selected,
                    cache: &cache,
                    scales: scales(fp8),
                    rank,
                    size,
                    prepared: rows,
                };
                run(session, &function, &entry, &operands, false, label)
            })
            .collect::<LabResult<Vec<Partials>>>()?;
        let merged = merge(session, merges, &merge_entries, &parts, rows)?;
        gate_merged(
            &format!("{label} merged over {size}"),
            &merged,
            &parts,
            &full,
        )?;
        tally
            .paired
            .entry(size)
            .or_default()
            .add(&merged, &served, width);
    }
    tally.rows += rows;
    tally.sparse_rows += step
        .rows
        .iter()
        .filter(|&&(_, p)| (p + 1) / BLOCK > TOP)
        .count();
    Ok(())
}

fn modules(session: &Session) -> LabResult<(Module, Module, Module)> {
    Ok((
        session.module("sparse_attention_qwen38_sm120")?,
        session.module(shapes::CUBIN)?,
        session.module(merge_shapes::MERGE_CUBIN)?,
    ))
}

fn record(session: &Session, case: &str, tally: &Tally) -> LabResult<Value> {
    let result = json!({
        "rows": tally.rows,
        "sparse_rows": tally.sparse_rows,
        "both_pass_oracle_gates": true,
        "paired_difference_diagnostic": tally
            .paired
            .iter()
            .map(|(size, p)| {
                (
                    size.to_string(),
                    json!({
                        "bitwise_equal_rows": p.equal_rows,
                        "elements": p.elements,
                        "differing_elements": p.differing_elements,
                        "max_abs_difference": p.max_abs_difference,
                        "max_bf16_ulps": p.max_bf16_ulps,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>(),
    });
    eprintln!("{case}: {result}");
    session.receipt(
        "qsa_selected_partials",
        BTreeMap::from([
            ("case".into(), json!(case)),
            ("result".into(), result.clone()),
            ("correct".into(), json!(true)),
        ]),
    )?;
    Ok(result)
}

/// Merged per-rank partials and QSA's served selected attention each pass
/// their oracle gates on random selections of decode, verify and prefill
/// rows.
#[test]
fn merged_partials_and_qsa_pass_their_gates_on_synthetic_selections() {
    gpu_test(4 << 30, |session| {
        check_geometry()?;
        let modules = modules(session)?;
        let mut rng = Rng(0x95a5_e1e0);
        for fp8 in [false, true] {
            for (phase, step) in [
                ("decode", Step::decode(&[1, 64, 2051, 2052, 5000, 30_000])),
                ("verify", Step::verify(&[8, 2100, 9000], 4)),
                ("prefill", Step::prefill(1950, 260)),
            ] {
                let selected = selections(&step, &mut rng);
                let case = format!("{phase} {}", if fp8 { "fp8" } else { "bf16" });
                let mut tally = Tally::default();
                compare(
                    session, &modules, phase, &step, &selected, fp8, &mut rng, &case, &mut tally,
                )?;
                record(session, &case, &tally)?;
            }
        }
        Ok(())
    })
    .expect("merged partials and QSA pass their gates on synthetic selections");
}

/// Directories under `root` (itself included, two levels down) that hold
/// layer manifests.
fn export_dirs(root: &Path) -> Vec<PathBuf> {
    let holds_layers = |dir: &Path| {
        fs::read_dir(dir).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with("layer") && name.ends_with(".json")
            })
        })
    };
    let subdirs = |dir: &Path| -> Vec<PathBuf> {
        let mut dirs = fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        dirs.sort();
        dirs
    };
    if holds_layers(root) {
        return vec![root.to_path_buf()];
    }
    let mut found = Vec::new();
    for dir in subdirs(root) {
        if holds_layers(&dir) {
            found.push(dir);
        } else {
            found.extend(subdirs(&dir).into_iter().filter(|d| holds_layers(d)));
        }
    }
    found
}

/// One exported layer: projection rows `[tokens][640]`, indexer norm
/// weights (query then key) and the exported rows.
struct Layer {
    qk: Vec<u16>,
    norms: Vec<u16>,
    rows: Vec<usize>,
}

fn read_layer(dir: &Path, layer: u64) -> LabResult<Layer> {
    let bits = |name: &str| -> LabResult<Vec<u16>> {
        let path = dir.join(format!("layer{layer:02}.{name}"));
        let data = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if data.len() % 2 != 0 {
            return Err(format!("{}: odd byte count", path.display()));
        }
        Ok(data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    };
    let manifest: Value =
        serde_json::from_slice(&fs::read(dir.join(format!("layer{layer:02}.json"))).map_err(err)?)
            .map_err(err)?;
    if manifest["table"] != "hf" {
        return Err(format!(
            "{} layer {layer}: operands use table {}, the kernels read the HF table",
            dir.display(),
            manifest["table"]
        ));
    }
    let layer = Layer {
        qk: bits("qk.bf16")?,
        norms: bits("norm.bf16")?,
        rows: serde_json::from_value(manifest["rows"].clone()).map_err(err)?,
    };
    let tokens = layer.qk.len() / PROJECTION;
    if layer.rows.is_empty()
        || layer.qk.len() != tokens * PROJECTION
        || layer.norms.len() != 2 * INDEX_DIM
        || layer.rows.iter().any(|&row| row >= tokens)
    {
        return Err(format!(
            "{}: layer shapes disagree or no rows",
            dir.display()
        ));
    }
    Ok(layer)
}

fn launch_qsa(
    session: &Session,
    module: &Module,
    entry: &sparse::Entry,
    rows: u32,
    context: u32,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    let plan = entry.plan(rows, context).map_err(|e| e.0.to_string())?;
    function(module, &entry.symbol(), entry.dynamic_shared_bytes())?
        .launch(&session.stream, launch::config(plan)?, arguments)
        .map_err(err)?;
    kernel_lab::launched(&entry.symbol());
    Ok(())
}

/// QSA's served selection of `rows` (positions of one sequence) on a
/// layer's projection rows: its key write over every row, its queries of
/// `rows`, `block_scores_decode` and `block_topk`. Returns the `[rows,
/// TOP]` block lists, -1 past each row's count.
fn qsa_selection(
    session: &Session,
    module: &Module,
    layer: &Layer,
    rows: &[usize],
) -> LabResult<Vec<i32>> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    let eps = geometry.eps.to_bits();
    let ring = geometry.ring as usize;
    let tokens = layer.qk.len() / PROJECTION;
    let context = tokens as u32;
    let cos_sin = (0..tokens)
        .flat_map(|position| {
            kernel_core::gated_attention_oracle::mrope_cos_sin(
                [position; 3],
                geometry.rotary_dim as usize,
                f64::from(geometry.theta),
                geometry.mrope,
            )
            .into_iter()
            .flat_map(|(cos, sin)| [cos as f32, sin as f32])
        })
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    let cos_sin = upload(session, &cos_sin)?;
    let half = |values: &[u16]| {
        values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let q_norm = upload(session, &half(&layer.norms[..INDEX_DIM]))?;
    let k_norm = upload(session, &half(&layer.norms[INDEX_DIM..]))?;
    let pages = (tokens / BLOCK).div_ceil(PAGE / BLOCK).max(1);
    let block_keys = upload(session, &vec![0xff; pages * KEY_STRIDE])?;
    let table = upload(session, &words(&(0..pages as i32).collect::<Vec<_>>()))?;
    let slots = upload(session, &words(&[0]))?;
    let ring_keys = upload(session, &vec![0xff; ring * INDEX_DIM * 2])?;
    let ring_positions = upload(session, &vec![0xff; ring * 3 * 4])?;
    let no_anchor = upload(session, &words(&[-1]))?;
    let key_write = qsa_entry("index_key_write");
    let mut first = 0;
    while first < tokens {
        let n = HISTORY_ROWS.min(tokens - first);
        let projection = upload(
            session,
            &half(&layer.qk[first * PROJECTION..(first + n) * PROJECTION]),
        )?;
        let positions: Vec<i32> = (first as i32..(first + n) as i32).collect();
        let rope = upload(
            session,
            &words(
                &positions
                    .iter()
                    .flat_map(|&p| [p, p, p])
                    .collect::<Vec<_>>(),
            ),
        )?;
        let row_batch = upload(session, &words(&vec![0; n]))?;
        let positions = upload(session, &words(&positions))?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        launch_qsa(
            session,
            module,
            &key_write,
            n as u32,
            context,
            &[
                KernelArgument::Buffer(&projection),
                KernelArgument::Buffer(&k_norm),
                KernelArgument::Buffer(&rope),
                KernelArgument::Buffer(&cos_sin),
                KernelArgument::Buffer(&row_batch),
                KernelArgument::Buffer(&positions),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&slots),
                KernelArgument::Buffer(&ring_keys),
                KernelArgument::Buffer(&ring_positions),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&table),
                KernelArgument::Buffer(&block_keys),
                KernelArgument::U32(pages as u32),
                KernelArgument::U64(KEY_STRIDE as u64),
                KernelArgument::U32(eps),
            ],
        )?;
        first += n;
    }
    let n = rows.len();
    let projection = rows
        .iter()
        .flat_map(|&r| half(&layer.qk[r * PROJECTION..(r + 1) * PROJECTION]))
        .collect::<Vec<_>>();
    let projection = upload(session, &projection)?;
    let rope = upload(
        session,
        &words(&rows.iter().flat_map(|&r| [r as i32; 3]).collect::<Vec<_>>()),
    )?;
    let live = upload(session, &(n as u32).to_ne_bytes())?;
    let q = upload(session, &vec![0xff; n * INDEX_HEADS * INDEX_DIM * 2])?;
    launch_qsa(
        session,
        module,
        &qsa_entry("index_query"),
        n as u32,
        context,
        &[
            KernelArgument::Buffer(&projection),
            KernelArgument::Buffer(&q_norm),
            KernelArgument::Buffer(&rope),
            KernelArgument::Buffer(&cos_sin),
            KernelArgument::Buffer(&live),
            KernelArgument::Buffer(&q),
            KernelArgument::U32(eps),
        ],
    )?;
    // Each row its own sequence over the same pages, as decode rows.
    let tables = upload(
        session,
        &words(&(0..n).flat_map(|_| 0..pages as i32).collect::<Vec<_>>()),
    )?;
    let row_batch = upload(session, &words(&(0..n as i32).collect::<Vec<_>>()))?;
    let positions = upload(
        session,
        &words(&rows.iter().map(|&r| r as i32).collect::<Vec<_>>()),
    )?;
    let stride = geometry.score_stride(context) as usize;
    let scores = upload(session, &vec![0xff; n * stride * 4])?;
    let selected = upload(session, &vec![0xff; n * TOP * 4])?;
    launch_qsa(
        session,
        module,
        &qsa_entry("block_scores_decode"),
        n as u32,
        context,
        &[
            KernelArgument::Buffer(&q),
            KernelArgument::Buffer(&block_keys),
            KernelArgument::Buffer(&tables),
            KernelArgument::Buffer(&row_batch),
            KernelArgument::Buffer(&positions),
            KernelArgument::Buffer(&live),
            KernelArgument::Buffer(&scores),
            KernelArgument::U32(pages as u32),
            KernelArgument::U64(KEY_STRIDE as u64),
            KernelArgument::U32(stride as u32),
        ],
    )?;
    launch_qsa(
        session,
        module,
        &qsa_entry("block_topk"),
        n as u32,
        context,
        &[
            KernelArgument::Buffer(&scores),
            KernelArgument::Buffer(&positions),
            KernelArgument::Buffer(&live),
            KernelArgument::Buffer(&selected),
            KernelArgument::U32(stride as u32),
        ],
    )?;
    let all = download(session, &selected, n * TOP * 4)?
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<_>>();
    let mut lists = vec![-1; n * TOP];
    for (i, &row) in rows.iter().enumerate() {
        let count = TOP.min((row + 1) / BLOCK);
        lists[i * TOP..][..count].copy_from_slice(&all[i * TOP..][..count]);
    }
    Ok(lists)
}

/// Merged per-rank partials and QSA's served selected attention each pass
/// their oracle gates on QSA's own selections of captured rows
/// (`CHOPINFER_QSA_OPERANDS`), with synthetic attention operands. With
/// `CHOPINFER_QSA_OPERANDS_OUT` the summary is written to that new file.
#[test]
fn merged_partials_and_qsa_pass_their_gates_on_captured_selections() {
    let Some(root) = std::env::var_os("CHOPINFER_QSA_OPERANDS").map(PathBuf::from) else {
        kernel_lab::skip("the captured-selection comparison: CHOPINFER_QSA_OPERANDS is unset")
            .expect("merged partials on captured selections");
        return;
    };
    let out = std::env::var_os("CHOPINFER_QSA_OPERANDS_OUT").map(PathBuf::from);
    gpu_test(8 << 30, |session| {
        check_geometry()?;
        let modules = modules(session)?;
        let dirs = export_dirs(&root);
        if dirs.is_empty() {
            return Err(format!("{}: no layer manifests", root.display()));
        }
        let mut rng = Rng(0xca97_5e1e);
        let mut summary = Vec::new();
        for dir in dirs {
            let mut layers: Vec<u64> = fs::read_dir(&dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .filter_map(|entry| {
                    let name = entry.ok()?.file_name().into_string().ok()?;
                    name.strip_prefix("layer")?
                        .strip_suffix(".json")?
                        .parse()
                        .ok()
                })
                .collect();
            layers.sort_unstable();
            for number in layers {
                let layer = read_layer(&dir, number)?;
                let count = layer.rows.len().min(SAMPLED_ROWS);
                let rows = (0..count)
                    .map(|i| layer.rows[i * layer.rows.len() / count])
                    .collect::<Vec<_>>();
                let selected = qsa_selection(session, &modules.0, &layer, &rows)?;
                let tokens = layer.qk.len() / PROJECTION;
                let step = Step {
                    lengths: vec![tokens],
                    rows: rows.iter().map(|&p| (0, p)).collect(),
                };
                for fp8 in [false, true] {
                    let case = format!(
                        "{} layer {number} {}",
                        dir.display(),
                        if fp8 { "fp8" } else { "bf16" }
                    );
                    let mut tally = Tally::default();
                    compare(
                        session, &modules, "decode", &step, &selected, fp8, &mut rng, &case,
                        &mut tally,
                    )?;
                    let mut result = record(session, &case, &tally)?;
                    result["case"] = json!(case);
                    summary.push(result);
                }
            }
        }
        if let Some(out) = &out {
            use std::io::Write;
            let value = json!({"operands": root.display().to_string(), "layers": summary});
            fs::File::create_new(out)
                .and_then(|mut f| f.write_all(&serde_json::to_vec_pretty(&value).expect("json")))
                .map_err(|e| format!("{}: {e}", out.display()))?;
        }
        Ok(())
    })
    .expect("merged partials and QSA pass their gates on captured selections");
}
