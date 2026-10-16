//! Qwen3.8 Flash gated attention against the Q-4 oracles:
//! the dense paged prefill, decode and verify variants at GQA group 6 and
//! 256-wide heads, the Gemma Q/K norm with M-RoPE and the KV write, the
//! sigmoid output gate, and the decode entries that fold the norm, RoPE and
//! KV write in. `CHOPINFER_QWEN38_RANKS=2` runs the gated rows (norm, RoPE
//! and KV write, output gate) of TP2 (12 query heads); the dense
//! variants and their decode folds exist at TP4 only. Those lean runs
//! (`kernel_lab::qwen38_lean`) check a few hundred gated rows per dtype and
//! mark their receipts `"lean": true`.

use super::*;
use kernel_core::contract::GatedAttentionQuery;
use kernel_core::gated_attention_oracle::{
    self as gated_oracle, GatedAttentionInputs, PagedKvPair, mrope_axis,
};
use kernels_sm120::attention::{GatedRopeInputs, gated};

pub(super) const Q_NORM: u64 = 30;
pub(super) const K_NORM: u64 = 31;
pub(super) const ROPE_POSITIONS: u64 = 32;
pub(super) const COS_SIN: u64 = 33;
const GATED: u64 = 34;

/// Contexts of the task: the first row, a partial page, the last dense QSA
/// context, and two sparse-region contexts that the dense engine also serves.
const CONTEXTS: [usize; 5] = [1, 127, 2051, 4096, 65_536];

fn qwen(name: &str) -> LabResult<Variant> {
    attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|variant| {
            variant.name == name && variant.profile == model_qwen38_contract::sm120::config::PROFILE
        })
        .ok_or_else(|| format!("missing Qwen3.8 attention variant {name}"))
}

fn decode_case(variant: Variant, batch: usize, context: usize, mixed: bool) -> Case {
    Case {
        variant,
        batch,
        rows: 1,
        prefix: context - 1,
        mixed,
    }
}

/// Batches 1, 8 and 32 at every context (65,536 at batches 1 and 8), the
/// native 262,144-token context at batch 1, and mixed rows whose stream-K
/// CTAs span several rows.
fn decode_cases(variant: Variant) -> Vec<Case> {
    let mut cases = Vec::new();
    for context in CONTEXTS {
        for batch in [1, 8, 32] {
            if context == 65_536 && batch == 32 {
                continue;
            }
            cases.push(decode_case(variant, batch, context, false));
        }
    }
    cases.push(decode_case(
        variant,
        1,
        model_qwen38_contract::sm120::attention::MAX_CONTEXT as usize,
        false,
    ));
    for (batch, context) in [(8, 4096), (32, 4096), (8, 65_536)] {
        cases.push(decode_case(variant, batch, context, true));
    }
    cases
}

fn filtered(mut cases: Vec<Case>) -> Vec<Case> {
    if let Ok(filter) = std::env::var("CHOPINFER_ATTENTION_CASE") {
        cases.retain(|case| case.label().starts_with(&filter));
    }
    cases
}

#[test]
fn qwen38_attention_decode_contexts_and_batches() {
    gpu_test(3 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            cases.extend(decode_cases(qwen(name)?));
        }
        run(session, &filtered(cases), false)
    })
    .expect("Qwen3.8 decode attention correctness");
}

/// Prefill chunks inside the dense QSA region, a one-token chunk, two
/// sequences sharing a chunk, and a chunk over a 32K prefix whose rows use
/// one probability term.
fn prefill_cases(variant: Variant) -> Vec<Case> {
    [
        (1, 2051, 0),
        (1, 512, 129),
        (1, 127, 0),
        (1, 1, 2050),
        (2, 1024, 1027),
        (1, 256, 32_768),
    ]
    .map(|(batch, rows, prefix)| Case {
        variant,
        batch,
        rows,
        prefix,
        mixed: false,
    })
    .to_vec()
}

#[test]
fn qwen38_attention_prefill_chunks() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in ["qwen38_bf16_prefill", "qwen38_fp8_prefill"] {
            cases.extend(prefill_cases(qwen(name)?));
        }
        run(session, &filtered(cases), false)
    })
    .expect("Qwen3.8 prefill attention correctness");
}

#[test]
fn qwen38_attention_verify_matches_oracle_and_prefill() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        for (verify, prefill) in [
            ("qwen38_bf16_verify", "qwen38_bf16_prefill"),
            ("qwen38_fp8_verify", "qwen38_fp8_prefill"),
        ] {
            let variant = qwen(verify)?;
            let cubin = module(session, variant.profile)?;
            let mut cases = Vec::new();
            for (batch, rows) in [(1, 8), (1, 1), (3, 4), (16, 2), (32, 1), (32, 2), (5, 8)] {
                for prefix in [0, 1, 127, 2043, 4095] {
                    for mixed in [false, true] {
                        if mixed && batch == 1 {
                            continue;
                        }
                        cases.push(Case {
                            variant,
                            batch,
                            rows,
                            prefix,
                            mixed,
                        });
                    }
                }
            }
            for case in filtered(cases) {
                let fixture = prepare(session, case)?;
                let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
                fixture.launch(session, &partial, &merge)?;
                let output = fixture.output(session)?;
                validate(session, case, &fixture, &output, "verify")?;
                let prefill_case = Case {
                    variant: qwen(prefill)?,
                    ..case
                };
                let reference = prepare(session, prefill_case)?;
                let (partial, merge) = functions(&cubin, prefill_case.variant, &reference.planned)?;
                reference.launch(session, &partial, &merge)?;
                close(
                    case,
                    &reference.output(session)?,
                    &output,
                    4.0,
                    "prefill kernel",
                )?;
            }
        }
        Ok(())
    })
    .expect("Qwen3.8 verify attention correctness");
}

/// Qwen3.8 verify above 32 items: its 96-thread CTAs scan
/// the item table in two or three passes, partly occupied at 97 and 193
/// items. Mixed contexts and query lengths of 2 and 1, at most 256 tokens,
/// against the oracle; one three-pass step also replays bit for bit, as a
/// graph and after a PDL producer.
#[test]
fn qwen38_attention_wide_verify_matches_oracle() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        for dtype in ["bf16", "fp8"] {
            let variant = qwen(&format!("qwen38_{dtype}_verify"))?
                .with_batch_capacity(attention::MAX_BATCH_ROWS)
                .map_err(|e| e.to_string())?;
            let cases: Vec<Case> = [
                (95, 2, 1500, true),
                (96, 2, 2043, true),
                (97, 2, 127, true),
                (191, 1, 1017, true),
                (192, 1, 4095, true),
                (193, 1, 0, false),
            ]
            .into_iter()
            .map(|(batch, rows, prefix, mixed)| Case {
                variant,
                batch,
                rows,
                prefix,
                mixed,
            })
            .collect();
            run(session, &filtered(cases), false)?;
            let replay = Case {
                variant,
                batch: 193,
                rows: 1,
                prefix: 2043,
                mixed: true,
            };
            run(session, &filtered(vec![replay]), true)?;
        }
        Ok(())
    })
    .expect("Qwen3.8 wide verify attention correctness");
}

#[test]
fn qwen38_attention_graph_determinism_and_pdl() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for dtype in ["bf16", "fp8"] {
            let decode = qwen(&format!("qwen38_{dtype}_decode"))?;
            cases.push(decode_case(decode, 1, 4096, false));
            cases.push(decode_case(decode, 32, 2051, true));
            cases.push(decode_case(decode, 8, 65_536, false));
            let prefill = qwen(&format!("qwen38_{dtype}_prefill"))?;
            for (batch, rows, prefix) in [(1, 2051, 0), (2, 1024, 1027)] {
                cases.push(Case {
                    variant: prefill,
                    batch,
                    rows,
                    prefix,
                    mixed: false,
                });
            }
            let verify = qwen(&format!("qwen38_{dtype}_verify"))?;
            for (batch, rows, mixed) in [(1, 8, false), (5, 8, true), (16, 2, true)] {
                cases.push(Case {
                    variant: verify,
                    batch,
                    rows,
                    prefix: 1500,
                    mixed,
                });
            }
        }
        run(session, &filtered(cases), true)
    })
    .expect("Qwen3.8 attention graph, deterministic replay and PDL");
}

/// Values on a 1/250 grid in [-limit, limit], rounded to BF16, so the norm,
/// rotation and FP8 quantization all round.
pub(super) fn grid_bf16(count: usize, limit: f32, state: &mut u64) -> Vec<u16> {
    let steps = (limit * 250.0) as u32;
    (0..count)
        .map(|_| {
            let value = (random_word(state) % (2 * steps + 1)) as f32 - steps as f32;
            f32_to_bf16(value / 250.0)
        })
        .collect()
}

pub(super) fn as_f64(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(bf16_to_f32(v))).collect()
}

/// The F32 table of decoded BF16 HF cos/sin, as uploaded by the bundle.
pub(super) fn cos_sin_table(positions: usize) -> Vec<f32> {
    cos_sin_table_for(positions, model_qwen38_contract::sm120::attention::GATED)
}

fn cos_sin_table_for(positions: usize, geometry: gated::Config) -> Vec<f32> {
    (0..positions)
        .flat_map(|position| {
            gated_oracle::mrope_cos_sin(
                [position; 3],
                geometry.rotary as usize,
                f64::from(geometry.theta),
                geometry.sections,
            )
            .into_iter()
            .flat_map(|(cos, sin)| [cos as f32, sin as f32])
        })
        .collect()
}

/// Each row's decoded table, as the oracle reads it.
pub(super) fn oracle_tables(table: &[f32], rope_positions: &[[usize; 3]]) -> Vec<Vec<(f64, f64)>> {
    oracle_tables_for(
        table,
        rope_positions,
        model_qwen38_contract::sm120::attention::GATED,
    )
}

fn oracle_tables_for(
    table: &[f32],
    rope_positions: &[[usize; 3]],
    geometry: gated::Config,
) -> Vec<Vec<(f64, f64)>> {
    let pairs = (geometry.rotary / 2) as usize;
    rope_positions
        .iter()
        .map(|position| {
            (0..pairs)
                .map(|pair| {
                    let at = position[mrope_axis(pair, geometry.sections)] * pairs + pair;
                    (f64::from(table[2 * at]), f64::from(table[2 * at + 1]))
                })
                .collect()
        })
        .collect()
}

/// RoPE positions of a text row at `position`, with the H and W axes moved
/// off it (by row) so every section reads its own axis.
pub(super) fn rope_position(position: usize, row: usize) -> [usize; 3] {
    [position, position.saturating_sub(row % 3 + 1), position / 2]
}

fn put_u16(session: &Session, values: &[u16]) -> LabResult<DeviceBuffer> {
    put_bytes(
        session,
        &values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}

fn put_f32(session: &Session, values: &[f32]) -> LabResult<DeviceBuffer> {
    put_bytes(
        session,
        &values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}

/// Half an E4M3 step of `value` (scaled), with slack for the kernel's F32
/// rounding before the conversion.
pub(super) fn e4m3_close(code: u8, reference: f64, scale: f64) -> bool {
    let actual = SmallFloat::E4M3.decode_f64(code) * scale;
    let magnitude = (reference / scale).abs().max(2_f64.powi(-6));
    let step = 2_f64.powi(magnitude.log2().floor() as i32 - 3) * scale;
    (actual - reference).abs() <= 0.5 * step * (1.0 + 2_f64.powi(-8)) + reference.abs() * 1e-6
}

/// The Qwen3.8 topology of the gated row tests
/// (`kernel_lab::qwen38_profile`).
pub(super) struct Topology {
    geometry: gated::Config,
    max_context: u32,
    pub(super) profile: Profile,
    ranks: u32,
    /// Query heads per rank over the one KV head.
    pub(super) q_heads: usize,
    /// Elements of one packed projection row.
    pub(super) width: usize,
}

pub(super) fn topology() -> LabResult<Topology> {
    let profile = kernel_lab::qwen38_profile()?;
    let shape = kernel_lab::qwen38_shape()?;
    let (q_heads, width) = (shape.q_heads(), shape.attn_row());
    if width
        != (q_heads + model_qwen38_contract::sm120::attention::KV_HEADS)
            * 2
            * model_qwen38_contract::sm120::attention::HEAD_DIM
    {
        return Err(format!(
            "packed row of {width} elements does not hold {q_heads} Q and gate heads and one K and V head"
        ));
    }
    Ok(Topology {
        geometry: model_qwen38_contract::sm120::attention::gated_config(profile)
            .ok_or("missing gated geometry")?,
        max_context: model_qwen38_contract::sm120::attention::MAX_CONTEXT,
        profile,
        ranks: shape.ranks,
        q_heads: q_heads as usize,
        width: width as usize,
    })
}

impl Topology {
    /// The gated row entry `chop_attn_{tag}_{name}_v1`, checked against the
    /// topology's heads.
    pub(super) fn entry(&self, name: &str) -> LabResult<gated::Entry> {
        let entry = row_entry(&format!("chop_attn_{}_{name}_v1", self.profile.tag()))?;
        let heads = match entry.query {
            GatedAttentionQuery::QkNormRopeKvWrite { q_heads, .. } => q_heads,
            GatedAttentionQuery::OutputGate { heads, .. } => heads,
        };
        if entry.profile != self.profile || heads as usize != self.q_heads {
            return Err(format!(
                "{} has {heads} heads for {:?}",
                entry.symbol, entry.profile
            ));
        }
        Ok(entry)
    }

    /// Marks receipts of other topologies than TP4, whose receipts stay as
    /// they were.
    pub(super) fn mark(&self, receipt: &mut BTreeMap<String, Value>) {
        if self.ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(self.ranks));
        }
    }
}

/// The selected topology's gated rows exist with its heads (CPU only).
#[test]
fn qwen38_gated_rows_cover_the_topology() {
    let topology = topology().expect("Qwen3.8 topology");
    for name in ["qk_norm_rope_kv_bf16", "qk_norm_rope_kv_fp8", "output_gate"] {
        topology.entry(name).expect("gated row entry");
    }
}

pub(super) fn rows_query(dtype: DType, q_heads: usize) -> GatedAttentionQuery {
    rows_query_for(
        dtype,
        gated::Config {
            q_heads: q_heads as u32,
            ..model_qwen38_contract::sm120::attention::GATED
        },
    )
}

fn rows_query_for(dtype: DType, geometry: gated::Config) -> GatedAttentionQuery {
    GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads: geometry.q_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        rotary_dim: geometry.rotary,
        theta_bits: geometry.theta.to_bits(),
        mrope: geometry.sections,
        eps_bits: geometry.eps.to_bits(),
        page_size: geometry.page,
        kv_dtype: dtype,
        capacity: geometry.max_rows,
    }
}

fn gate_query(heads: usize) -> GatedAttentionQuery {
    GatedAttentionQuery::OutputGate {
        heads: heads as u32,
        head_dim: model_qwen38_contract::sm120::attention::HEAD_DIM,
        capacity: model_qwen38_contract::sm120::attention::MAX_ROWS,
    }
}

pub(super) fn row_entry(symbol: &str) -> LabResult<gated::Entry> {
    gated::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .ok_or_else(|| format!("missing gated row entry {symbol}"))
}

/// The norm, RoPE and KV write rows of one launch: sequences of `rows` new
/// tokens after `prefix` cached ones, packed in order, with permuted pages.
/// Checks Q against the oracle, the written K rows within its bound (half an
/// E4M3 step for FP8), V bit for bit and every other cache byte unchanged,
/// then deterministic and graph replay; then the output gate on the same
/// rows, in place and out of place. With `bench` (peak DRAM GB/s and its
/// source) both entries are also timed against the bytes they move. `lean`
/// marks the receipts of a lean run.
fn run_rows(
    session: &Session,
    dtype: DType,
    sequences: &[(usize, usize)],
    bench: Option<&(f64, String)>,
    lean: bool,
) -> LabResult<()> {
    run_rows_for(session, dtype, sequences, bench, lean, topology()?)
}

fn run_rows_for(
    session: &Session,
    dtype: DType,
    sequences: &[(usize, usize)],
    bench: Option<&(f64, String)>,
    lean: bool,
    topology: Topology,
) -> LabResult<()> {
    let geometry = topology.geometry;
    let fp8 = dtype == DType::Fp8E4M3;
    let name = if fp8 { "fp8" } else { "bf16" };
    let tag = topology.profile.tag();
    let entry = topology.entry(&format!("qk_norm_rope_kv_{name}"))?;
    let cubin = module(session, topology.profile)?;
    let function = cubin.function(&entry.symbol).map_err(|e| e.to_string())?;
    let (qh, kh, d) = (
        topology.q_heads,
        geometry.kv_heads as usize,
        geometry.head_dim as usize,
    );
    let page = geometry.page as usize;
    let width = topology.width;
    let max_pages = (topology.max_context as usize).div_ceil(page);
    let rows: usize = sequences.iter().map(|&(_, n)| n).sum();
    let mut state = 0x51_7cc1_b727_220a_u64 ^ ((rows as u64) << 20) ^ u64::from(fp8);
    let qkv = grid_bf16(rows * width, 4.0, &mut state);
    let q_norm = grid_bf16(d, 0.5, &mut state);
    let k_norm = grid_bf16(d, 0.5, &mut state);
    // Pages: sequence s owns ceil((prefix + rows) / page) pages, numbered
    // from the end of the pool.
    let page_counts: Vec<usize> = sequences
        .iter()
        .map(|&(prefix, n)| (prefix + n).div_ceil(page))
        .collect();
    let pool_pages: usize = page_counts.iter().sum();
    let mut tables = vec![0_i32; sequences.len() * max_pages];
    let mut page_lists = Vec::new();
    let mut next = pool_pages;
    for (s, &count) in page_counts.iter().enumerate() {
        let list: Vec<usize> = (0..count).map(|p| next - 1 - p).collect();
        next -= count;
        for (p, &physical) in list.iter().enumerate() {
            tables[s * max_pages + p] = physical as i32;
        }
        page_lists.push(list);
    }
    let mut row_batch = Vec::new();
    let mut positions = Vec::new();
    let mut rope = Vec::new();
    for (s, &(prefix, n)) in sequences.iter().enumerate() {
        for t in 0..n {
            row_batch.push(s as i32);
            positions.push((prefix + t) as i32);
            rope.push(rope_position(prefix + t, row_batch.len()));
        }
    }
    let max_position = positions.iter().copied().max().unwrap_or(0) as usize;
    let table = cos_sin_table_for(max_position + 1, geometry);
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let bytes = cache.bytes()?;
    let page_bytes = page * kh * d * bytes;
    let fill = vec![0xa5_u8; pool_pages * page_bytes];
    // Unequal FP8 scales, so a swapped pair cannot pass.
    let (k_scale, v_scale) = if fp8 { (0.5_f32, 0.25_f32) } else { (1.0, 1.0) };
    let mut buffers = BTreeMap::new();
    buffers.insert(QKV, session.upload_bf16(&qkv)?);
    buffers.insert(Q_NORM, put_u16(session, &q_norm)?);
    buffers.insert(K_NORM, put_u16(session, &k_norm)?);
    buffers.insert(
        ROPE_POSITIONS,
        put_i32(
            session,
            &rope
                .iter()
                .flat_map(|p| p.map(|x| x as i32))
                .collect::<Vec<_>>(),
        )?,
    );
    buffers.insert(COS_SIN, put_f32(session, &table)?);
    buffers.insert(TOKEN_BATCH, put_i32(session, &row_batch)?);
    buffers.insert(POSITIONS, put_i32(session, &positions)?);
    buffers.insert(LIVE, put_i32(session, &[rows as i32])?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    buffers.insert(K, put_bytes(session, &fill)?);
    buffers.insert(V, put_bytes(session, &fill)?);
    buffers.insert(Q, session.allocate(rows * qh * d * 2)?);
    // The rows the step writes, one per token and each of K and V.
    let row_bytes = kh * d * bytes;
    let written_rows =
        cache.byte_ranges(sequences.iter().enumerate().flat_map(|(s, &(prefix, n))| {
            let list = &page_lists[s];
            (prefix..prefix + n).map(move |position| {
                let at = (list[position / page] * page + position % page) * row_bytes;
                at..at + row_bytes
            })
        }))?;
    let args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(QKV),
            Argument::DeviceBuffer(Q_NORM),
            Argument::DeviceBuffer(K_NORM),
            Argument::DeviceBuffer(ROPE_POSITIONS),
            Argument::DeviceBuffer(COS_SIN),
            Argument::DeviceBuffer(TOKEN_BATCH),
            Argument::DeviceBuffer(POSITIONS),
            Argument::DeviceBuffer(LIVE),
            Argument::DeviceBuffer(TABLE),
            Argument::DeviceBuffer(K),
            Argument::DeviceBuffer(V),
            Argument::DeviceBuffer(Q),
            Argument::U32(max_pages as u32),
            Argument::U64(page_bytes as u64),
            Argument::U64(page_bytes as u64),
            Argument::U32(geometry.eps.to_bits()),
            Argument::U32(k_scale.to_bits()),
            Argument::U32(v_scale.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    // Prepared capacity above the live rows: the extra CTAs return.
    let plan = entry
        .plan((rows + 3).min(geometry.max_rows as usize) as u32)
        .map_err(|e| e.to_string())?;
    let bindings = Bindings {
        buffers: buffers.iter().map(|(&id, buffer)| (id, buffer)).collect(),
    };
    let launch_rows = || {
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    };
    let q_buffer = &buffers[&Q];
    let mut checks = Replay::new(session)
        .output(
            "Q",
            q_buffer,
            WriteMap::of(DType::Bf16).byte_prefix(q_buffer.byte_len())?,
        )
        .state(
            "K cache",
            &buffers[&K],
            fill.clone(),
            written_rows.clone(),
            Vec::new(),
        )
        .state(
            "V cache",
            &buffers[&V],
            fill.clone(),
            written_rows,
            Vec::new(),
        );
    checks.eager(launch_rows)?;
    let q_out = session.download_bf16(q_buffer, rows * qh * d)?;
    let k_out = bytes_of(session, &buffers[&K])?;
    let v_out = bytes_of(session, &buffers[&V])?;

    // The oracle, one sequence at a time over f64 caches.
    let query = rows_query_for(dtype, geometry);
    let mut k_cache = vec![0.0; pool_pages * page * kh * d];
    let mut v_cache = vec![0.0; pool_pages * page * kh * d];
    let tables_f64 = oracle_tables_for(&table, &rope, geometry);
    let mut q_expected = oracle::OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    let mut k_expected = Vec::new();
    let mut first = 0;
    for (s, &(prefix, n)) in sequences.iter().enumerate() {
        let rows_of = |values: &[u16], offset: usize, len: usize| {
            (first..first + n)
                .flat_map(|r| as_f64(&values[r * width + offset..r * width + offset + len]))
                .collect::<Vec<_>>()
        };
        let q_gate = rows_of(&qkv, 0, qh * 2 * d);
        let k = rows_of(&qkv, qh * 2 * d, kh * d);
        let v = rows_of(&qkv, qh * 2 * d + kh * d, kh * d);
        let token_positions: Vec<usize> = (prefix..prefix + n).collect();
        let result = gated_oracle::qk_norm_rope_kv_write(
            query,
            GatedAttentionInputs {
                q_gate: &q_gate,
                k: &k,
                v: &v,
                q_norm: &as_f64(&q_norm),
                k_norm: &as_f64(&k_norm),
                rope_positions: &rope[first..first + n],
                token_positions: &token_positions,
                tables: Some(&tables_f64[first..first + n]),
            },
            PagedKvPair {
                keys: &mut k_cache,
                values: &mut v_cache,
                block_table: &page_lists[s],
            },
        )
        .map_err(|e| e.to_string())?;
        q_expected.values.extend(result.q.values);
        q_expected
            .sum_abs_products
            .extend(result.q.sum_abs_products);
        q_expected
            .reduction_lengths
            .extend(result.q.reduction_lengths);
        k_expected.push(result.k);
        first += n;
    }
    let mut receipt = BTreeMap::new();
    let label = format!(
        "{tag}_qk_norm_rope_kv_{name}_rows{rows}_seq{}",
        sequences.len()
    );
    receipt.insert("case".into(), json!(label));
    receipt.insert("sequences".into(), json!(sequences));
    topology.mark(&mut receipt);
    if lean {
        receipt.insert("lean".into(), json!(true));
    }
    let q_f32: Vec<f32> = q_out.iter().map(|&x| bf16_to_f32(x)).collect();
    check_output(&q_f32, &q_expected, DType::Bf16, &mut receipt)
        .map_err(|e| format!("{label} Q: {e}; {:?}", receipt.get("worst_element")))?;
    // Written rows: K within its bound, V exact; every other byte unchanged.
    let mut expected_k = fill.clone();
    let mut expected_v = fill.clone();
    let mut row = 0;
    for (s, &(prefix, n)) in sequences.iter().enumerate() {
        for t in 0..n {
            let position = prefix + t;
            let slot = page_lists[s][position / page] * page + position % page;
            let at = slot * kh * d * bytes;
            let reference = &k_expected[s];
            let k_values = &reference.values[t * kh * d..(t + 1) * kh * d];
            let v_bits = &qkv[row * width + qh * 2 * d + kh * d..][..kh * d];
            if fp8 {
                for i in 0..kh * d {
                    if !e4m3_close(k_out[at + i], k_values[i], f64::from(k_scale)) {
                        return Err(format!(
                            "{label} K row {row} dim {i}: code {} for {}",
                            k_out[at + i],
                            k_values[i]
                        ));
                    }
                    expected_k[at + i] = k_out[at + i];
                    expected_v[at + i] = kv_fp8::encode(bf16_to_f32(v_bits[i]), v_scale);
                }
            } else {
                let actual: Vec<f32> = k_out[at..at + kh * d * 2]
                    .chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_ne_bytes([b[0], b[1]])))
                    .collect();
                let bound = oracle::OracleOutput {
                    values: k_values.to_vec(),
                    sum_abs_products: reference.sum_abs_products[t * kh * d..(t + 1) * kh * d]
                        .to_vec(),
                    reduction_lengths: reference.reduction_lengths[t * kh * d..(t + 1) * kh * d]
                        .to_vec(),
                };
                check_output(&actual, &bound, DType::Bf16, &mut receipt)
                    .map_err(|e| format!("{label} K row {row}: {e}"))?;
                expected_k[at..at + kh * d * 2].copy_from_slice(&k_out[at..at + kh * d * 2]);
                for (i, &bits) in v_bits.iter().enumerate() {
                    expected_v[at + 2 * i..at + 2 * i + 2].copy_from_slice(&bits.to_ne_bytes());
                }
            }
            row += 1;
        }
    }
    if k_out != expected_k {
        return Err(format!(
            "{label}: K cache bytes outside the written rows changed"
        ));
    }
    if v_out != expected_v {
        return Err(format!("{label}: V cache differs from the V rows"));
    }
    checks
        .relaunch(launch_rows)
        .map_err(|e| format!("{label}: {e}"))?;
    let graph = session.capture(|_| launch_rows())?;
    checks.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
    receipt.insert("k_cache_checked".into(), json!(true));
    receipt.insert("v_cache_bitwise".into(), json!(true));
    checks.record(&mut receipt);
    session.receipt("attention", receipt)?;
    // Bytes of one launch: the packed rows, rotated Q, the K and V rows, the
    // weights, and each row's RoPE positions and table pairs.
    let row_bytes = rows * (width * 2 + qh * d * 2 + 2 * kh * d * bytes + 3 * 4 + 32 * 8) + 4 * d;
    if let Some((peak, source)) = bench {
        let timing = time(session, 2, 3, |_| launch_rows())?;
        rows_receipt(
            session, &topology, &label, rows, row_bytes, timing, *peak, source,
        )?;
    }

    // The output gate over random attention rows and the rows' gate logits.
    let gate = topology.entry("output_gate")?;
    let gate_function = cubin.function(&gate.symbol).map_err(|e| e.to_string())?;
    let attention = grid_bf16(rows * qh * d, 2.0, &mut state);
    let mut gate_buffers = BTreeMap::new();
    gate_buffers.insert(OUT, session.upload_bf16(&attention)?);
    gate_buffers.insert(QKV, session.upload_bf16(&qkv)?);
    let gated_out = session.allocate(rows * qh * d * 2)?;
    poison::poison(session, &[&gated_out])?;
    gate_buffers.insert(GATED, gated_out);
    gate_buffers.insert(LIVE, put_i32(session, &[rows as i32])?);
    let gate_plan = gate
        .plan((rows + 3).min(geometry.max_rows as usize) as u32)
        .map_err(|e| e.to_string())?;
    let logits: Vec<f64> = (0..rows)
        .flat_map(|r| (0..qh).flat_map(move |h| (0..d).map(move |i| r * width + h * 2 * d + d + i)))
        .map(|index| f64::from(bf16_to_f32(qkv[index])))
        .collect();
    let expected = gated_oracle::attention_output_gate(
        GatedAttentionQuery::OutputGate {
            heads: geometry.q_heads,
            head_dim: geometry.head_dim,
            capacity: geometry.max_rows,
        },
        &as_f64(&attention),
        &logits,
    )
    .map_err(|e| e.to_string())?;
    let gate_bytes = rows * qh * d * 2 * 3;
    for (output, in_place) in [(GATED, false), (OUT, true)] {
        let gate_args = gate
            .pack_arguments(vec![
                Argument::DeviceBuffer(OUT),
                Argument::DeviceBuffer(QKV),
                Argument::DeviceBuffer(output),
                Argument::DeviceBuffer(LIVE),
            ])
            .map_err(|e| e.to_string())?;
        let gate_bindings = Bindings {
            buffers: gate_buffers.iter().map(|(&id, b)| (id, b)).collect(),
        };
        let launch_gate = || {
            launch::launch(
                &session.context,
                &gate_function,
                &session.stream,
                gate_plan,
                &gate_args,
                &gate_bindings,
            )
        };
        launch_gate()?;
        let actual: Vec<f32> = session
            .download_bf16(&gate_buffers[&output], rows * qh * d)?
            .iter()
            .map(|&x| bf16_to_f32(x))
            .collect();
        let mut receipt = BTreeMap::new();
        let gate_label = format!(
            "{tag}_output_gate_rows{rows}{}",
            if in_place { "_in_place" } else { "" }
        );
        receipt.insert("case".into(), json!(gate_label));
        topology.mark(&mut receipt);
        if lean {
            receipt.insert("lean".into(), json!(true));
        }
        check_output(&actual, &expected, DType::Bf16, &mut receipt)
            .map_err(|e| format!("{gate_label}: {e}; {:?}", receipt.get("worst_element")))?;
        session.receipt("attention", receipt)?;
        if let Some((peak, source)) = bench
            && !in_place
        {
            let timing = time(session, 2, 3, |_| launch_gate())?;
            rows_receipt(
                session,
                &topology,
                &gate_label,
                rows,
                gate_bytes,
                timing,
                *peak,
                source,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rows_receipt(
    session: &Session,
    topology: &Topology,
    label: &str,
    rows: usize,
    bytes: usize,
    timing: kernel_lab::Timing,
    peak_gbps: f64,
    peak_source: &str,
) -> LabResult<()> {
    let roofline_us = bytes as f64 / peak_gbps / 1000.0;
    let median_us = f64::from(timing.median_ms) * 1000.0;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(label));
    receipt.insert("rows".into(), json!(rows));
    receipt.insert("median_us".into(), json!(median_us));
    receipt.insert("min_us".into(), json!(f64::from(timing.min_ms) * 1000.0));
    receipt.insert("rounds".into(), json!(timing.rounds));
    receipt.insert("logical_bytes".into(), json!(bytes));
    receipt.insert("roofline_kind".into(), json!("DRAM read and write"));
    receipt.insert("roofline_us".into(), json!(roofline_us));
    receipt.insert(
        "percent_peak".into(),
        json!(roofline_us / median_us * 100.0),
    );
    receipt.insert("peak_gb_per_second".into(), json!(peak_gbps));
    receipt.insert("peak_source".into(), json!(peak_source));
    topology.mark(&mut receipt);
    session.receipt("bench", receipt)?;
    eprintln!("{label}: {median_us:.2} us, roofline {roofline_us:.2} us");
    Ok(())
}

fn bytes_of(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    super::bytes(session, buffer)
}

/// Lean runs (TP2) take a verify item, decode rows at page edges and
/// position 65,535, and a chunk of three sequences whose rows cross page
/// boundaries: 311 rows per dtype.
#[test]
fn qwen38_gated_rows_match_oracle() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let lean = kernel_lab::qwen38_lean()?;
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            // Decode rows, a verify item, prefill chunks inside and past the
            // dense region, and chunks of several sequences.
            let cases = if lean {
                vec![
                    vec![(2043, 8)],
                    vec![(63, 1), (0, 64), (2050, 1), (65_535, 1)],
                    vec![(1000, 100), (0, 129), (5, 7)],
                ]
            } else {
                vec![
                    vec![(0, 1)],
                    vec![(4095, 1)],
                    vec![(2043, 8)],
                    vec![(0, 127)],
                    vec![(0, 2051)],
                    vec![(129, 8191)],
                    vec![(63, 1), (0, 64), (2050, 1), (65_535, 1)],
                    vec![(1000, 100), (0, 2051), (5, 7)],
                ]
            };
            for sequences in cases {
                run_rows(session, dtype, &sequences, None, lean)?;
            }
        }
        Ok(())
    })
    .expect("Qwen3.8 gated attention rows match the oracle");
}

/// The output gate at sigmoid logits from -150 to 4, including the review's
/// -88, -89 and -90, on attention values up to 2^100. `1 / (1 + exp(-x))`
/// overflowed below about -88 and returned 0 where `a * sigmoid(z)` is a
/// representable BF16 value; the oracle's bound is relative to it. Logits
/// between -80 and -5 are left out: the gate's `__expf` there is less exact
/// than the bound's F32 allowance, which the tail change does not affect.
#[test]
fn qwen38_output_gate_keeps_sigmoid_tails() {
    const LOGITS: [f32; 15] = [
        -150.0, -110.0, -100.0, -95.0, -90.0, -89.0, -88.0, -85.0, -81.0, -5.0, -3.0, -0.5, 0.0,
        1.5, 4.0,
    ];
    gpu_test(64 << 20, |session| {
        let topology = topology()?;
        let gate = topology.entry("output_gate")?;
        let cubin = module(session, topology.profile)?;
        let function = cubin.function(&gate.symbol).map_err(|e| e.to_string())?;
        let (qh, d) = (
            topology.q_heads,
            model_qwen38_contract::sm120::attention::HEAD_DIM as usize,
        );
        let width = topology.width;
        let rows = 2;
        let amplitudes = [
            2_f32.powi(40),
            -(2_f32.powi(60)),
            3.0 * 2_f32.powi(20),
            2_f32.powi(100),
            1.0,
        ];
        let attention: Vec<u16> = (0..rows * qh * d)
            .map(|i| f32_to_bf16(amplitudes[i / LOGITS.len() % amplitudes.len()]))
            .collect();
        let mut qkv = vec![0_u16; rows * width];
        let mut logits = Vec::with_capacity(rows * qh * d);
        for r in 0..rows {
            for h in 0..qh {
                for i in 0..d {
                    let logit = f32_to_bf16(LOGITS[(r * qh * d + h * d + i) % LOGITS.len()]);
                    qkv[r * width + h * 2 * d + d + i] = logit;
                    logits.push(f64::from(bf16_to_f32(logit)));
                }
            }
        }
        let expected =
            gated_oracle::attention_output_gate(gate_query(qh), &as_f64(&attention), &logits)
                .map_err(|e| e.to_string())?;
        let mut buffers = BTreeMap::new();
        buffers.insert(OUT, session.upload_bf16(&attention)?);
        buffers.insert(QKV, session.upload_bf16(&qkv)?);
        buffers.insert(GATED, session.allocate(rows * qh * d * 2)?);
        buffers.insert(LIVE, put_i32(session, &[rows as i32])?);
        let args = gate
            .pack_arguments(vec![
                Argument::DeviceBuffer(OUT),
                Argument::DeviceBuffer(QKV),
                Argument::DeviceBuffer(GATED),
                Argument::DeviceBuffer(LIVE),
            ])
            .map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            gate.plan(rows as u32).map_err(|e| e.to_string())?,
            &args,
            &Bindings {
                buffers: buffers.iter().map(|(&id, b)| (id, b)).collect(),
            },
        )?;
        let actual: Vec<f32> = session
            .download_bf16(&buffers[&GATED], rows * qh * d)?
            .iter()
            .map(|&x| bf16_to_f32(x))
            .collect();
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            json!(format!(
                "{}_output_gate_sigmoid_tails",
                topology.profile.tag()
            )),
        );
        topology.mark(&mut receipt);
        let result = check_output(&actual, &expected, DType::Bf16, &mut receipt);
        session.receipt("attention", receipt)?;
        result.map_err(|e| format!("output gate sigmoid tails: {e}"))
    })
    .expect("Qwen3.8 output gate sigmoid tails");
}

/// A decode case with packed Qwen rows and both ways to consume them: the
/// norm, RoPE and KV write row followed by the unfused decode, or the entry
/// that folds them in.
struct GatedFoldCase {
    case: Case,
    fixture: Fixture,
    folded: attention::PlannedAttention,
    row: Function,
    row_plan: LaunchPlan,
    row_args: ArgumentList,
    partial: Function,
    merge: Function,
    folded_function: Function,
    qkv: Vec<u16>,
}

fn prepare_gated_fold(session: &Session, cubin: &Module, case: Case) -> LabResult<GatedFoldCase> {
    prepare_gated_fold_with_page_layers(session, cubin, case, 1)
}

/// `prepare_gated_fold` over pages of `page_layers` layers.
fn prepare_gated_fold_with_page_layers(
    session: &Session,
    cubin: &Module,
    case: Case,
    page_layers: usize,
) -> LabResult<GatedFoldCase> {
    let mut fixture = prepare_with_page_layers(session, case, page_layers)?;
    let v = case.variant;
    let fold = v.gated_fold.ok_or("missing gated fold geometry")?;
    let geometry = model_qwen38_contract::sm120::attention::gated_config(v.profile)
        .unwrap_or_else(|| kernel_lab::lab_shapes::gated_config(v.profile));
    assert_eq!(geometry.fold(), fold);
    let batch = case.batch;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (case.context() as u64) ^ ((batch as u64) << 32);
    let qkv = grid_bf16(batch * geometry.row_width() as usize, 4.0, &mut state);
    let d = geometry.head_dim as usize;
    let q_norm = grid_bf16(d, 0.5, &mut state);
    let k_norm = grid_bf16(d, 0.5, &mut state);
    let lasts: Vec<usize> = case.contexts().iter().map(|&c| c - 1).collect();
    let rope: Vec<[usize; 3]> = lasts
        .iter()
        .enumerate()
        .map(|(b, &last)| rope_position(last, b))
        .collect();
    let table = cos_sin_table_for(lasts.iter().max().copied().unwrap_or(0) + 1, geometry);
    fixture.buffers.insert(QKV, session.upload_bf16(&qkv)?);
    fixture.buffers.insert(Q_NORM, put_u16(session, &q_norm)?);
    fixture.buffers.insert(K_NORM, put_u16(session, &k_norm)?);
    fixture.buffers.insert(
        ROPE_POSITIONS,
        put_i32(
            session,
            &rope
                .iter()
                .flat_map(|p| p.map(|x| x as i32))
                .collect::<Vec<_>>(),
        )?,
    );
    fixture.buffers.insert(COS_SIN, put_f32(session, &table)?);
    fixture.buffers.insert(
        POSITIONS,
        put_i32(
            session,
            &lasts.iter().map(|&p| p as i32).collect::<Vec<_>>(),
        )?,
    );
    fixture.buffers.insert(
        TOKEN_BATCH,
        put_i32(session, &(0..batch as i32).collect::<Vec<_>>())?,
    );
    let gated_out = session.allocate(batch * (v.query.q_heads * v.query.v_dim) as usize * 2)?;
    poison::poison(session, &[&gated_out])?;
    fixture.buffers.insert(GATED, gated_out);
    let row_profile = kernel_lab::lab_shapes::row_profile(v.profile);
    let row_cubin = module(session, row_profile)?;
    let entry = row_entry(&format!(
        "chop_attn_{}_qk_norm_rope_kv_{}_v1",
        row_profile.tag(),
        if fp8 { "fp8" } else { "bf16" }
    ))?;
    assert_eq!(entry.query, rows_query_for(v.query.kv_dtype, geometry));
    let arguments = fixture.arguments.clone();
    let bucket = batch.next_power_of_two() as u32;
    let row_plan = entry.plan(bucket).map_err(|e| e.to_string())?;
    let row_args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(QKV),
            Argument::DeviceBuffer(Q_NORM),
            Argument::DeviceBuffer(K_NORM),
            Argument::DeviceBuffer(ROPE_POSITIONS),
            Argument::DeviceBuffer(COS_SIN),
            Argument::DeviceBuffer(TOKEN_BATCH),
            Argument::DeviceBuffer(POSITIONS),
            Argument::DeviceBuffer(LIVE),
            Argument::DeviceBuffer(TABLE),
            Argument::DeviceBuffer(K),
            Argument::DeviceBuffer(V),
            Argument::DeviceBuffer(Q),
            Argument::U32(v.query.max_context_capacity.div_ceil(v.query.page_size)),
            Argument::U64(arguments.k_page_stride_bytes),
            Argument::U64(arguments.v_page_stride_bytes),
            Argument::U32(geometry.eps.to_bits()),
            Argument::U32(arguments.k_scale.to_bits()),
            Argument::U32(arguments.v_scale.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    let counts = RuntimeCounts {
        m: batch as u32,
        batch: batch as u32,
        context: case.context() as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let folded = v
        .plan_folded_gated(
            bucket,
            counts,
            arguments,
            GatedRopeInputs {
                qkv: QKV,
                q_norm: Q_NORM,
                k_norm: K_NORM,
                rope_positions: ROPE_POSITIONS,
                cos_sin: COS_SIN,
            },
        )
        .map_err(|e| e.to_string())?;
    if !folded.writes_cache || folded.partial_symbol != v.rope_symbol() {
        return Err(format!(
            "{} folded plan does not write the cache",
            case.label()
        ));
    }
    let (partial, merge) = functions(cubin, v, &fixture.planned)?;
    let folded_function = cubin
        .function(&folded.partial_symbol)
        .map_err(|e| e.to_string())?;
    let row = row_cubin
        .function(&entry.symbol)
        .map_err(|e| e.to_string())?;
    Ok(GatedFoldCase {
        case,
        fixture,
        folded,
        row,
        row_plan,
        row_args,
        partial,
        merge,
        folded_function,
        qkv,
    })
}

impl GatedFoldCase {
    fn launch_unfused(
        &self,
        session: &Session,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        launch::launch(
            &session.context,
            &self.row,
            &session.stream,
            self.row_plan,
            &self.row_args,
            &self.fixture.cache_bindings(cache),
        )?;
        self.fixture
            .launch_with_cache(session, &self.partial, &self.merge, cache)
    }
    fn launch_folded(
        &self,
        session: &Session,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        self.fixture.launch_planned(
            session,
            &self.folded,
            &self.folded_function,
            &self.merge,
            cache,
        )
    }

    /// The unfused decode's output against the attention oracle over the
    /// row kernel's Q and the caches it wrote, then the output gate against
    /// its oracle on that output.
    fn check_oracles(&self, session: &Session, cubin: &Module) -> LabResult<()> {
        let v = self.case.variant;
        let fixture = &self.fixture;
        let batch = self.case.batch;
        let (qh, d) = (v.query.q_heads as usize, v.query.qk_dim as usize);
        let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
        let decode_cache = |bytes: Vec<u8>, scale: f32| -> Vec<f64> {
            if fp8 {
                bytes
                    .iter()
                    .map(|&code| kv_fp8::decode(code, scale))
                    .collect()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|b| f64::from(bf16_to_f32(u16::from_ne_bytes([b[0], b[1]]))))
                    .collect()
            }
        };
        let keys = decode_cache(
            bytes_of(session, &fixture.buffers[&K])?,
            fixture.arguments.k_scale,
        );
        let values = decode_cache(
            bytes_of(session, &fixture.buffers[&V])?,
            fixture.arguments.v_scale,
        );
        let q = as_f64(&session.download_bf16(&fixture.buffers[&Q], batch * qh * d)?);
        let contexts = self.case.contexts();
        let starts: Vec<usize> = contexts.iter().map(|&c| c - 1).collect();
        let reference = oracle::attention(
            &v.query,
            &PagedAttention {
                q: &q,
                keys: &keys,
                values: &values,
                page_tables: &fixture.page_lists,
                query_lengths: &vec![1; batch],
                query_starts: &starts,
                key_lengths: &contexts,
                sink_logits: None,
            },
        )
        .map_err(|e| e.to_string())?;
        let attention_out = fixture.output(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(self.case.label()));
        receipt.insert("phase".into(), json!("gated_rows_then_decode"));
        let actual: Vec<f32> = attention_out.iter().map(|&x| bf16_to_f32(x)).collect();
        check_output(&actual, &reference, DType::Bf16, &mut receipt).map_err(|e| {
            format!(
                "{} attention: {e}; {:?}",
                self.case.label(),
                receipt.get("worst_element")
            )
        })?;
        session.receipt("attention", receipt)?;
        let gate = row_entry(&format!(
            "chop_attn_{}_output_gate_v1",
            kernel_lab::lab_shapes::row_profile(self.case.variant.profile).tag()
        ))?;
        let function = cubin.function(&gate.symbol).map_err(|e| e.to_string())?;
        let args = gate
            .pack_arguments(vec![
                Argument::DeviceBuffer(OUT),
                Argument::DeviceBuffer(QKV),
                Argument::DeviceBuffer(GATED),
                Argument::DeviceBuffer(LIVE),
            ])
            .map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            gate.plan(batch.next_power_of_two() as u32)
                .map_err(|e| e.to_string())?,
            &args,
            &fixture.bindings(),
        )?;
        let width = (qh + self.case.variant.query.kv_heads as usize) * 2 * d;
        let logits: Vec<f64> = (0..batch)
            .flat_map(|r| {
                (0..qh).flat_map(move |h| (0..d).map(move |i| r * width + h * 2 * d + d + i))
            })
            .map(|index| f64::from(bf16_to_f32(self.qkv[index])))
            .collect();
        let expected =
            gated_oracle::attention_output_gate(gate.query, &as_f64(&attention_out), &logits)
                .map_err(|e| e.to_string())?;
        let gated_out: Vec<f32> = session
            .download_bf16(&fixture.buffers[&GATED], batch * qh * d)?
            .iter()
            .map(|&x| bf16_to_f32(x))
            .collect();
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(self.case.label()));
        receipt.insert("phase".into(), json!("output_gate"));
        check_output(&gated_out, &expected, DType::Bf16, &mut receipt)
            .map_err(|e| format!("{} gate: {e}", self.case.label()))?;
        session.receipt("attention", receipt)?;
        Ok(())
    }
}

/// The folded entry against the norm, RoPE and KV write row followed by the
/// unfused decode, from the same caches: the output and both caches bit for
/// bit, then deterministic replay, graph replay and a PDL launch. Cases with
/// at most 262,144 cached tokens also check the unfused chain against the
/// attention and output gate oracles.
fn run_gated_fold(session: &Session, cases: &[Case]) -> LabResult<()> {
    for &case in cases {
        let cubin = module(session, case.variant.profile)?;
        let row_cubin = module(
            session,
            kernel_lab::lab_shapes::row_profile(case.variant.profile),
        )?;
        let fold = prepare_gated_fold(session, &cubin, case)?;
        let evidence = fold_checks(
            session,
            case,
            &fold.fixture,
            || fold.launch_unfused(session, None),
            |cache| fold.launch_folded(session, cache),
            fold.folded.partial.pdl,
            false,
        )?;
        let oracle_checked = case.contexts().iter().sum::<usize>() <= 262_144;
        if oracle_checked {
            fold.check_oracles(session, &row_cubin)?;
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("contexts".into(), json!(case.contexts()));
        receipt.insert("folded_symbol".into(), json!(fold.folded.partial_symbol));
        receipt.insert("output_bitwise".into(), json!(true));
        receipt.insert("cache_bitwise".into(), json!(true));
        evidence.record(&mut receipt);
        receipt.insert("oracle_checked".into(), json!(oracle_checked));
        session.receipt("attention", receipt)?;
    }
    Ok(())
}

/// Qwen's dense paged attention past the 32-bit lines (`pool_offsets`):
/// decode, verify and prefill over pages of the twelve full-attention
/// layers, and the decode fold with its norm, RoPE and KV-write row, in BF16
/// and FP8, give the low-id run's output and cache entries bit for bit;
/// writes leave every other pool byte at the sentinel.
#[test]
fn qwen38_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(18 << 30), |session| {
        for dtype in ["bf16", "fp8"] {
            for case in [
                decode_case(qwen(&format!("qwen38_{dtype}_decode"))?, 4, 4096, true),
                Case {
                    variant: qwen(&format!("qwen38_{dtype}_verify"))?,
                    batch: 4,
                    rows: 8,
                    prefix: 2040,
                    mixed: true,
                },
                Case {
                    variant: qwen(&format!("qwen38_{dtype}_prefill"))?,
                    batch: 2,
                    rows: 256,
                    prefix: 1024,
                    mixed: false,
                },
            ] {
                run_high_ids(session, case)?;
            }
        }
        let cubin = module(session, model_qwen38_contract::sm120::config::PROFILE)?;
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            let case = decode_case(qwen(name)?, 4, 4096, true);
            let label = case.label();
            let page_layers = serving_page_layers(case.variant);
            let mut fold = prepare_gated_fold_with_page_layers(session, &cubin, case, page_layers)?;
            // Low ids: the row then the unfused decode, and the fold, each on
            // copies of the starting caches.
            let copy = |buffer: &DeviceBuffer| put_bytes(session, &bytes_of(session, buffer)?);
            let (unfused_k, unfused_v) = (
                copy(&fold.fixture.buffers[&K])?,
                copy(&fold.fixture.buffers[&V])?,
            );
            fold.launch_unfused(session, Some((&unfused_k, &unfused_v)))?;
            let unfused = fold.fixture.output(session)?;
            let (folded_k, folded_v) = (
                copy(&fold.fixture.buffers[&K])?,
                copy(&fold.fixture.buffers[&V])?,
            );
            fold.launch_folded(session, Some((&folded_k, &folded_v)))?;
            let folded = fold.fixture.output(session)?;
            let mut receipt = json!({"page_layers": page_layers, "output_bitwise": true,
                "entries_bitwise": true});
            for (phase, expected, low_k, low_v) in [
                ("rows then unfused decode", &unfused, &unfused_k, &unfused_v),
                ("folded", &folded, &folded_k, &folded_v),
            ] {
                // The fixture's own caches still hold the starting rows.
                let high = HighCache::new(session, &fold.fixture)?;
                high.bind(session, &mut fold.fixture)?;
                if phase == "folded" {
                    fold.launch_folded(session, high.cache())?;
                } else {
                    fold.launch_unfused(session, high.cache())?;
                }
                exact(case, expected, &fold.fixture.output(session)?, phase)?;
                high.check_entries(session, &format!("{label} {phase}"), low_k, low_v)?;
                let scanned = high.k.check_untouched(session, &label)?
                    + high.v.check_untouched(session, &label)?;
                receipt[phase] =
                    json!({"cache": high.describe(), "sentinel_bytes_checked": scanned});
            }
            pool_offsets::receipt(session, "qwen38_gated_fold", &label, receipt)?;
        }
        Ok(())
    })
    .expect("Qwen3.8 attention at high page ids");
}

#[test]
fn qwen38_decode_fold_matches_rows_and_oracle() {
    gpu_test(3 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            let variant = qwen(name)?;
            // Current tokens at the start, end and inside of a 64-token page,
            // at the dense QSA limit, and over split rows.
            for context in [1, 2, 63, 64, 65, 127, 2051, 4096, 65_536, 262_144] {
                cases.push(decode_case(variant, 1, context, false));
            }
            for batch in [2, 8, 32] {
                for context in [1, 64, 2051, 4096] {
                    cases.push(decode_case(variant, batch, context, false));
                }
                cases.push(decode_case(variant, batch, 4096, true));
            }
            cases.push(decode_case(variant, 8, 65_536, true));
        }
        run_gated_fold(session, &filtered(cases))
    })
    .expect("Qwen3.8 folded decode matches the rows plus decode and the oracles");
}

/// Single launches (partial and merge) of decode at batches 1, 8 and 32,
/// prefill chunks and verify steps, against DRAM bytes (decode, verify) or
/// BF16 MMA FLOPs (prefill), after the oracle check of each case.
#[test]
fn qwen38_attention_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen3.8 attention benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(3 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for dtype in ["bf16", "fp8"] {
            let decode = qwen(&format!("qwen38_{dtype}_decode"))?;
            for batch in [1, 8, 32] {
                for context in [127, 2051, 4096] {
                    cases.push(decode_case(decode, batch, context, false));
                }
            }
            cases.push(decode_case(decode, 1, 65_536, false));
            cases.push(decode_case(decode, 8, 65_536, false));
            let prefill = qwen(&format!("qwen38_{dtype}_prefill"))?;
            for (rows, prefix) in [(512, 0), (2051, 0), (8192, 0), (512, 1539)] {
                cases.push(Case {
                    variant: prefill,
                    batch: 1,
                    rows,
                    prefix,
                    mixed: false,
                });
            }
            let verify = qwen(&format!("qwen38_{dtype}_verify"))?;
            for (batch, rows) in [(1, 8), (16, 2), (32, 1), (32, 2)] {
                cases.push(Case {
                    variant: verify,
                    batch,
                    rows,
                    prefix: 2051 - rows,
                    mixed: false,
                });
            }
        }
        benchmark(session, &cases)
    })
    .expect("Qwen3.8 attention benchmarks");
}

/// Decode and verify in the decode roofline graph: a PDL producer before
/// each layer and distinct KV caches of at least 512 MiB in total.
#[test]
fn qwen38_attention_decode_roofline_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen3.8 decode roofline: set CHOPINFER_KERNEL_GRAPH_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for dtype in ["bf16", "fp8"] {
            let decode = qwen(&format!("qwen38_{dtype}_decode"))?;
            for batch in [1, 8, 32] {
                for context in [2051, 4096, 32_768] {
                    cases.push(decode_case(decode, batch, context, false));
                }
            }
            cases.push(decode_case(decode, 1, 65_536, false));
            cases.push(decode_case(decode, 8, 65_536, false));
            cases.push(decode_case(decode, 32, 32_768, true));
            let verify = qwen(&format!("qwen38_{dtype}_verify"))?;
            for (batch, rows) in [(1, 8), (16, 2), (32, 1), (32, 2)] {
                cases.push(Case {
                    variant: verify,
                    batch,
                    rows,
                    prefix: 2051 - rows,
                    mixed: false,
                });
            }
        }
        decode_roofline_benchmark(session, &filtered(cases))
    })
    .expect("Qwen3.8 decode roofline benchmarks");
}

/// In-graph decode layers, the norm, RoPE and KV write row followed by the
/// unfused decode against the folded entry, each layer on its own KV cache
/// (at least 512 MiB in total) as `decode_roofline_benchmark` does. The
/// output gate row follows attention in both. As in the MiMo fold benchmark,
/// the bucket's MiMo decode QKV and O GEMMs surround each layer's attention
/// so that no path keeps its code in the SM instruction cache, as in
/// serving; the Qwen projections are not in the lab yet.
#[test]
fn qwen38_decode_fold_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen3.8 fold benchmarks: set CHOPINFER_KERNEL_GRAPH_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let cubin = module(session, model_qwen38_contract::sm120::config::PROFILE)?;
        let gate = row_entry("chop_attn_qwen38_output_gate_v1")?;
        let gate_function = cubin.function(&gate.symbol).map_err(|e| e.to_string())?;
        let mut cases = Vec::new();
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            let variant = qwen(name)?;
            for (batch, context, mixed) in [
                (1, 2051, false),
                (1, 32_768, false),
                (8, 2051, false),
                (32, 2051, false),
                (32, 4096, true),
            ] {
                cases.push(decode_case(variant, batch, context, mixed));
            }
        }
        for case in filtered(cases) {
            let fold = prepare_gated_fold(session, &cubin, case)?;
            let fixture = &fold.fixture;
            let gate_plan = gate
                .plan(case.batch.next_power_of_two() as u32)
                .map_err(|e| e.to_string())?;
            let gate_args = gate
                .pack_arguments(vec![
                    Argument::DeviceBuffer(OUT),
                    Argument::DeviceBuffer(QKV),
                    Argument::DeviceBuffer(GATED),
                    Argument::DeviceBuffer(LIVE),
                ])
                .map_err(|e| e.to_string())?;
            let bucket = case.batch.next_power_of_two() as u32;
            let qkv = super::super::gemm::Case::prepare(
                session,
                &format!("chop_gemm_mimo_qkv_ga_m{bucket}_v1"),
                case.batch as u32,
                false,
            )?;
            let o = super::super::gemm::Case::prepare(
                session,
                &format!("chop_gemm_mimo_o_m{bucket}_v1"),
                case.batch as u32,
                false,
            )?;
            let (qkv_function, o_function) = (qkv.function(session)?, o.function(session)?);
            let bindings = fixture.bindings();
            let launch_gate = |stream: &runtime_cuda::Stream| {
                launch::launch(
                    &session.context,
                    &gate_function,
                    stream,
                    gate_plan,
                    &gate_args,
                    &bindings,
                )
            };
            let cache_bytes = fixture.buffers[&K].byte_len() + fixture.buffers[&V].byte_len();
            let distinct = (512_usize << 20).div_ceil(cache_bytes).clamp(1, 256);
            let layers = distinct.max(GRAPH_LAYERS);
            let mut caches = Vec::with_capacity(distinct - 1);
            for _ in 1..distinct {
                let k = session.allocate(fixture.buffers[&K].byte_len())?;
                let v = session.allocate(fixture.buffers[&V].byte_len())?;
                k.copy_range_from_async(0, &fixture.buffers[&K], 0, k.byte_len(), &session.stream)
                    .map_err(|e| e.to_string())?;
                v.copy_range_from_async(0, &fixture.buffers[&V], 0, v.byte_len(), &session.stream)
                    .map_err(|e| e.to_string())?;
                caches.push((k, v));
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let cache = |layer: usize| {
                (layer % distinct)
                    .checked_sub(1)
                    .map(|index| (&caches[index].0, &caches[index].1))
            };
            // `Some(folded)` runs one attention path and the gate between the
            // GEMMs; `None` runs the GEMMs and the gate alone.
            let capture = |attention: Option<bool>| {
                session.capture(|stream| {
                    for layer in 0..layers {
                        qkv.launch(session, &qkv_function, stream)?;
                        match attention {
                            Some(true) => fold.launch_folded(session, cache(layer))?,
                            Some(false) => fold.launch_unfused(session, cache(layer))?,
                            None => {}
                        }
                        launch_gate(stream)?;
                        o.launch(session, &o_function, stream)?;
                    }
                    Ok(())
                })
            };
            let unfused = capture(Some(false))?;
            let folded = capture(Some(true))?;
            // The unfused graph is the reference; the attention output,
            // workspace and Q feed the gate inside each layer.
            let gated_out = &fixture.buffers[&GATED];
            let labelled = |error: String| format!("{}: {error}", case.label());
            let mut checks = Replay::new(session)
                .output(
                    "gated output",
                    gated_out,
                    WriteMap::of(DType::Bf16).byte_prefix(gated_out.byte_len())?,
                )
                .scratch("attention output", &fixture.buffers[&OUT])
                .scratch("attention workspace", &fixture.buffers[&WORK])
                .scratch("row Q", &fixture.buffers[&Q]);
            checks
                .eager(|| unfused.launch(&session.stream).map_err(|e| e.to_string()))
                .map_err(labelled)?;
            checks.graph(&folded).map_err(labelled)?;
            let gates = capture(None)?;
            let per_layer = |graph: &runtime_cuda::GraphExec| -> LabResult<f64> {
                let timing = time(session, 2, 5, |stream| {
                    graph.launch(stream).map_err(|e| e.to_string())
                })?;
                Ok(f64::from(timing.median_ms) * 1000.0 / layers as f64)
            };
            let unfused_us = per_layer(&unfused)?;
            let folded_us = per_layer(&folded)?;
            let gate_us = per_layer(&gates)?;
            let (peak_gbps, _, peak_source) = find_peak(session.ordinal);
            let roofline_us = fixture.logical_bytes as f64 / peak_gbps / 1000.0;
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!(case.label()));
            receipt.insert("contexts".into(), json!(case.contexts()));
            receipt.insert("layers".into(), json!(layers));
            receipt.insert("distinct_kv_caches".into(), json!(distinct));
            receipt.insert("unfused_us_per_layer".into(), json!(unfused_us));
            receipt.insert("folded_us_per_layer".into(), json!(folded_us));
            receipt.insert("gemms_and_gate_us_per_layer".into(), json!(gate_us));
            receipt.insert(
                "neighbours".into(),
                json!("MiMo decode QKV and O GEMMs of the bucket"),
            );
            receipt.insert("saving_us_per_layer".into(), json!(unfused_us - folded_us));
            receipt.insert("logical_bytes".into(), json!(fixture.logical_bytes));
            receipt.insert("roofline_kind".into(), json!("DRAM read"));
            receipt.insert("roofline_us".into(), json!(roofline_us));
            receipt.insert("peak_gb_per_second".into(), json!(peak_gbps));
            receipt.insert("peak_source".into(), json!(peak_source));
            receipt.insert("folded_symbol".into(), json!(fold.folded.partial_symbol));
            receipt.insert("partial_grid".into(), json!(fold.folded.partial.grid));
            session.receipt("bench", receipt)?;
            eprintln!(
                "{}: {unfused_us:.2} us/layer QKV + rows + attention + gate + O, {folded_us:.2} \
                 us/layer folded, {gate_us:.2} us/layer GEMMs and gate, roofline {roofline_us:.2} us",
                case.label(),
            );
        }
        Ok(())
    })
    .expect("Qwen3.8 decode fold benchmarks");
}

/// The norm, RoPE and KV write row and the output gate at decode, verify
/// and prefill row counts, against the bytes they move.
#[test]
fn qwen38_gated_rows_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen3.8 gated row benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(1024 * 1024 * 1024, |session| {
        let (peak_gbps, _, peak_source) = find_peak(session.ordinal);
        let peak = (peak_gbps, peak_source);
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            for sequences in [
                vec![(4095, 1)],
                vec![(4095, 1); 8],
                vec![(4095, 1); 32],
                vec![(2043, 8)],
                vec![(0, 2051)],
                vec![(0, 8192)],
            ] {
                run_rows(session, dtype, &sequences, Some(&peak), false)?;
            }
        }
        Ok(())
    })
    .expect("Qwen3.8 gated row benchmarks");
}

#[rustfmt::skip] #[path = "qwen_rope_positions.rs"] mod qwen_rope_positions;

#[test]
fn lab_shapes_attention_and_gated_rows_match_oracles() {
    gpu_test(256 * 1024 * 1024, |session| {
        let variants = (kernel_lab::lab_shapes::CATALOG.attention)();
        let cases: Vec<_> = variants
            .iter()
            .filter(|v| v.gated_fold.is_none() || v.profile.tag().starts_with("lab_shapes_w7"))
            .flat_map(|&variant| {
                [(2, 7), (3, 65), (1, 257)]
                    .into_iter()
                    .chain((variant.query.phase == AttentionPhase::Decode).then_some((1, 65_536)))
                    .map(move |(batch, prefix)| Case {
                        rows: if variant.query.phase == AttentionPhase::Decode {
                            1
                        } else {
                            3
                        },
                        variant,
                        batch,
                        prefix,
                        mixed: false,
                    })
            })
            .collect();
        run(session, &cases, true)?;
        for profile in kernel_lab::lab_shapes::row_profiles() {
            if !(kernel_lab::lab_shapes::CATALOG.gated)()
                .iter()
                .any(|e| e.profile == profile)
            {
                continue;
            }
            let geometry = kernel_lab::lab_shapes::gated_config(profile);
            for dtype in [DType::Bf16, DType::Fp8E4M3] {
                run_rows_for(
                    session,
                    dtype,
                    &[(7, 3), (63, 3), (1, 1)],
                    None,
                    false,
                    Topology {
                        geometry,
                        profile,
                        ranks: 4,
                        max_context: 262144,
                        q_heads: geometry.q_heads as usize,
                        width: geometry.row_width() as usize,
                    },
                )?;
            }
        }
        Ok(())
    })
    .expect("lab shape attention and gated row fixtures");
}

#[test]
fn lab_shapes_decode_fold_matches_rows_and_oracle() {
    gpu_test(256 * 1024 * 1024, |session| {
        let cases: Vec<_> = (kernel_lab::lab_shapes::CATALOG.attention)()
            .into_iter()
            .filter(|v| v.gated_fold.is_some())
            .flat_map(|variant| {
                [(1, 1), (2, 17), (3, 65), (2, 257)]
                    .map(|(batch, context)| decode_case(variant, batch, context, false))
            })
            .collect();
        run_gated_fold(session, &cases)
    })
    .expect("lab shape gated folds");
}

#[test]
fn lab_shapes_gated_tables_and_queries_keep_their_geometry() {
    for profile in kernel_lab::lab_shapes::row_profiles() {
        let geometry = kernel_lab::lab_shapes::gated_config(profile);
        let query = rows_query_for(DType::Bf16, geometry);
        assert_eq!(query, gated::entries(profile, geometry)[0].query);
        let table = cos_sin_table_for(11, geometry);
        assert_eq!(table.len(), 11 * geometry.rotary as usize);
        let positions = [[7, 5, 3]];
        assert_eq!(
            oracle_tables_for(&table, &positions, geometry)[0],
            gated_oracle::mrope_cos_sin(
                positions[0],
                geometry.rotary as usize,
                geometry.theta as f64,
                geometry.sections
            )
        );
    }
    for variant in (kernel_lab::lab_shapes::CATALOG.attention)()
        .into_iter()
        .filter(|v| v.gated_fold.is_some())
    {
        let geometry = kernel_lab::lab_shapes::gated_config(variant.profile);
        let profile = kernel_lab::lab_shapes::row_profile(variant.profile);
        let entry = row_entry(&format!(
            "chop_attn_{}_qk_norm_rope_kv_{}_v1",
            profile.tag(),
            if variant.query.kv_dtype == DType::Bf16 {
                "bf16"
            } else {
                "fp8"
            },
        ))
        .unwrap();
        assert_eq!(
            entry.query,
            rows_query_for(variant.query.kv_dtype, geometry)
        );
        assert_eq!(geometry.fold(), variant.gated_fold.unwrap());
    }
}
