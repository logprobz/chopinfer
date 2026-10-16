//! Paged decode's row-local split: at a fixed bucket, a row's output bits do
//! not depend on the other rows of its step, their number, lengths, order,
//! values or pages (decode.cuh `piece_units`). Probe rows are launched alone
//! and beside companions; every launch writes a poisoned output and
//! workspace, and every live row must be written and finite. The companions
//! move the step's CTA ranges across every probe: under the step-wide plan
//! this replaced, which cut those ranges across rows, all but a few probe
//! steps changed their partial boundaries.

use super::*;

/// One decode row: its seed and its context (`seq_lens`) in tokens.
#[derive(Clone, Copy)]
struct Row {
    seed: u64,
    len: usize,
}

/// A row's Q (its current token) and every position's K and V, encoded as
/// the cache holds them: the same bytes whatever step holds the row.
struct RowData {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
}

fn row_data(v: Variant, row: Row) -> RowData {
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let mut state = 0x726f_772d_6c6f_6361 ^ row.seed ^ ((row.len as u64) << 20);
    let (q, _) = encode(&samples(qh * qd, &mut state), false, 1.0);
    let [k_scale, v_scale] = scales(v);
    let (k, _) = encode(&samples(row.len * kh * qd, &mut state), fp8, k_scale);
    let (values, _) = encode(&samples(row.len * kh * vd, &mut state), fp8, v_scale);
    RowData { q, k, v: values }
}

fn scales(v: Variant) -> [f32; 2] {
    if v.query.kv_dtype == DType::Fp8E4M3 {
        [0.5, 0.25]
    } else {
        [1.0, 1.0]
    }
}

/// One planned paged decode step of `rows` at `bucket`. `gap` free pages
/// precede the first row's pages, so a probe's page IDs differ between
/// steps.
struct Step {
    buffers: BTreeMap<u64, DeviceBuffer>,
    planned: attention::PlannedAttention,
    rows: usize,
    row_bytes: usize,
}

fn step(
    session: &Session,
    v: Variant,
    data: &BTreeMap<(u64, usize), RowData>,
    rows: &[Row],
    bucket: u32,
    gap: usize,
) -> LabResult<Step> {
    let element = if v.query.kv_dtype == DType::Fp8E4M3 {
        1
    } else {
        2
    };
    let (kh, qd, vd) = (
        v.query.kv_heads as usize,
        v.query.qk_dim as usize,
        v.query.v_dim as usize,
    );
    let (k_row, v_row) = (kh * qd * element, kh * vd * element);
    let page = v.query.page_size as usize;
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let mut tables = vec![0_i32; rows.len() * max_pages];
    let (mut k_cache, mut v_cache) = (
        vec![0_u8; gap * page * k_row],
        vec![0_u8; gap * page * v_row],
    );
    let mut q = Vec::new();
    let mut next = gap;
    for (b, row) in rows.iter().enumerate() {
        let d = &data[&(row.seed, row.len)];
        let pages = row.len.div_ceil(page);
        for p in 0..pages {
            tables[b * max_pages + p] = (next + p) as i32;
        }
        next += pages;
        let mut k = d.k.clone();
        let mut values = d.v.clone();
        k.resize(pages * page * k_row, 0);
        values.resize(pages * page * v_row, 0);
        k_cache.extend(k);
        v_cache.extend(values);
        q.extend_from_slice(&d.q);
    }
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, put_bytes(session, &q)?);
    buffers.insert(K, put_bytes(session, &k_cache)?);
    buffers.insert(V, put_bytes(session, &v_cache)?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    let lens = rows.iter().map(|row| row.len as i32).collect::<Vec<_>>();
    buffers.insert(SEQ, put_i32(session, &lens)?);
    buffers.insert(LIVE, put_i32(session, &[rows.len() as i32])?);
    let qh = v.query.q_heads as usize;
    let sink: Vec<f32> = (0..qh).map(|h| (h as f32 - 8.0) / 16.0).collect();
    buffers.insert(SINK, put_bytes(session, &encode(&sink, false, 1.0).0)?);
    let row_bytes = qh * vd * 2;
    buffers.insert(OUT, session.allocate(bucket as usize * row_bytes)?);
    let [k_scale, v_scale] = scales(v);
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: (page * k_row) as u64,
        v_page_stride_bytes: (page * v_row) as u64,
        k_map: None,
        v_map: None,
        block_tables: Some(TABLE),
        slot_ids: None,
        seq_lens: Some(SEQ),
        q_start: None,
        q_len: None,
        prefix_len: None,
        live_batch: LIVE,
        k_new: None,
        v_new: None,
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale,
        v_scale,
    };
    let counts = RuntimeCounts {
        m: rows.len() as u32,
        batch: rows.len() as u32,
        context: lens.iter().copied().max().unwrap_or(1) as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let planned = v
        .plan_for_bucket(bucket, counts, arguments)
        .map_err(|e| e.to_string())?;
    buffers.insert(
        WORK,
        session.allocate((planned.partial.workspace_bytes as usize).max(4))?,
    );
    Ok(Step {
        buffers,
        planned,
        rows: rows.len(),
        row_bytes,
    })
}

impl Step {
    fn launch(&self, session: &Session, partial: &Function, merge: &Function) -> LabResult<()> {
        let bindings = Bindings {
            buffers: self.buffers.iter().map(|(&id, b)| (id, b)).collect(),
        };
        launch::launch(
            &session.context,
            partial,
            &session.stream,
            self.planned.partial,
            &self.planned.partial_args,
            &bindings,
        )?;
        launch::launch(
            &session.context,
            merge,
            &session.stream,
            self.planned.merge,
            &self.planned.merge_args,
            &bindings,
        )
    }

    /// Each live row's output after one launch on a poisoned output and
    /// workspace: every live row written and finite, the bucket's padding
    /// rows still poisoned.
    fn outputs(
        &self,
        session: &Session,
        module: &Module,
        v: Variant,
        label: &str,
    ) -> LabResult<Vec<Vec<u8>>> {
        let (partial, merge) = functions(module, v, &self.planned)?;
        poison::poison(session, &[&self.buffers[&OUT], &self.buffers[&WORK]])?;
        self.launch(session, &partial, &merge)?;
        kernel_lab::launched(&self.planned.partial_symbol);
        kernel_lab::launched(&self.planned.merge_symbol);
        let bytes = poison::read(session, &self.buffers[&OUT])?;
        super::invariance::written_rows(label, &bytes, self.rows, self.row_bytes)?;
        Ok(bytes[..self.rows * self.row_bytes]
            .chunks(self.row_bytes)
            .map(<[u8]>::to_vec)
            .collect())
    }
}

/// Probe rows: a few units to a long context, so pieces of one unit up to
/// pieces capped by the split slots occur.
const PROBES: [usize; 6] = [37, 700, 1500, 4097, 9000, 20_000];

fn probes() -> Vec<Row> {
    PROBES
        .iter()
        .enumerate()
        .map(|(i, &len)| Row {
            seed: 100 + i as u64,
            len,
        })
        .collect()
}

/// Companion sets for a bucket: balanced mid contexts, the same reversed,
/// one very long row among short ones, and a full bucket. Each is (name,
/// rows in batch order, free pages before the first row's).
fn steps_for(bucket: usize) -> Vec<(&'static str, Vec<Row>, usize)> {
    let probes = probes();
    let room = bucket - probes.len();
    let companion = |i: usize, len: usize| Row {
        seed: 1000 + i as u64,
        len,
    };
    let balanced = (0..room.min(40))
        .map(|i| companion(i, 300 + 97 * i % 1700))
        .collect::<Vec<_>>();
    let mut skewed = vec![companion(500, 30_000)];
    skewed.extend((1..room.min(24)).map(|i| companion(500 + i, 64 + i)));
    let full = (0..room)
        .map(|i| companion(2000 + i, 128 + 211 * i % 2900))
        .collect::<Vec<_>>();
    let interleave = |companions: &[Row]| {
        let mut rows = Vec::new();
        for i in 0..probes.len().max(companions.len()) {
            rows.extend(companions.get(i));
            rows.extend(probes.get(i));
        }
        rows
    };
    let mut reversed = interleave(&balanced);
    reversed.reverse();
    vec![
        ("alone", probes.clone(), 0),
        ("balanced", interleave(&balanced), 3),
        ("balanced reversed", reversed, 0),
        ("skewed", interleave(&skewed), 7),
        ("full", interleave(&full), 1),
    ]
}

/// Each step's name and its probes' outputs, in probe order.
type ProbeOutputs = Vec<(&'static str, Vec<Vec<u8>>)>;

/// Each probe's output in every step of `bucket`.
fn probe_outputs(
    session: &Session,
    module: &Module,
    v: Variant,
    data: &BTreeMap<(u64, usize), RowData>,
    bucket: usize,
) -> LabResult<ProbeOutputs> {
    let mut all = Vec::new();
    for (name, rows, gap) in steps_for(bucket) {
        let s = step(session, v, data, &rows, bucket as u32, gap)?;
        let label = format!("{} b{bucket} {name}", v.name);
        let outputs = s.outputs(session, module, v, &label)?;
        let by_probe = probes()
            .iter()
            .map(|probe| {
                let at = rows
                    .iter()
                    .position(|row| row.seed == probe.seed)
                    .expect("every step holds the probes");
                outputs[at].clone()
            })
            .collect();
        all.push((name, by_probe));
    }
    Ok(all)
}

fn data_for(v: Variant, buckets: &[usize]) -> BTreeMap<(u64, usize), RowData> {
    let mut data = BTreeMap::new();
    for &bucket in buckets {
        for (_, rows, _) in steps_for(bucket) {
            for row in rows {
                data.entry((row.seed, row.len))
                    .or_insert_with(|| row_data(v, row));
            }
        }
    }
    data
}

/// Every paged decode variant, prepared for buckets up to 256 rows (the
/// catalog's are 32; `with_batch_capacity` plans the same kernels).
fn paged_decode_variants() -> LabResult<Vec<Variant>> {
    let wide = variants()
        .into_iter()
        .filter(|v| v.query.phase == AttentionPhase::Decode && v.layout == CacheLayout::Paged)
        .map(|v| {
            v.with_batch_capacity(attention::MAX_BATCH_ROWS)
                .map_err(|e| format!("{}: {e}", v.name))
        })
        .collect::<LabResult<Vec<_>>>()?;
    if wide.is_empty() {
        return Err("no paged decode variants".into());
    }
    Ok(wide)
}

/// Buckets of the composition test: every bucket that holds the probes and
/// a companion.
const BUCKETS: [usize; 6] = [8, 16, 32, 64, 128, 256];

/// Every probe row's bits are the same alone and beside balanced, reversed,
/// skewed and full companions, with other page IDs, at buckets 8 to 256.
#[test]
fn attention_decode_rows_ignore_their_companions() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let variants = paged_decode_variants()?;
        let mut runs = 0;
        for &v in &variants {
            let module = module(session, v.profile)?;
            let data = data_for(v, &BUCKETS);
            let mut receipt = BTreeMap::new();
            receipt.insert("variant".into(), json!(v.name));
            for bucket in BUCKETS {
                let local = probe_outputs(session, &module, v, &data, bucket)?;
                let (_, alone) = &local[0];
                for (name, outputs) in &local[1..] {
                    for (p, (got, expected)) in outputs.iter().zip(alone).enumerate() {
                        if got != expected {
                            return Err(format!(
                                "{} b{bucket}: probe of {} tokens differs {name} against alone",
                                v.name, PROBES[p]
                            ));
                        }
                    }
                }
                let compared = (local.len() - 1) * PROBES.len();
                if compared != 4 * PROBES.len() {
                    return Err(format!(
                        "{} b{bucket}: {compared} probe comparisons",
                        v.name
                    ));
                }
                receipt.insert(
                    format!("b{bucket}_row_local_compared_probes"),
                    json!(compared),
                );
                runs += 1;
            }
            session.receipt("attention_decode_composition", receipt)?;
        }
        if runs != variants.len() * BUCKETS.len() {
            return Err(format!(
                "{runs} variant buckets ran, expected {}",
                variants.len() * BUCKETS.len()
            ));
        }
        Ok(())
    })
    .expect("row-local paged decode ignores its companions");
}

/// Bench shapes of a bucket: balanced mid contexts, one long row among
/// short ones, and uniform long contexts. Each is (name, rows).
fn bench_shapes(bucket: usize) -> Vec<(String, Vec<Row>)> {
    let mut shapes = vec![
        (
            "balanced".to_owned(),
            (0..bucket)
                .map(|i| Row {
                    seed: i as u64,
                    len: 512 + (i * 2731) % 3584,
                })
                .collect(),
        ),
        (
            "skewed".to_owned(),
            (0..bucket)
                .map(|i| Row {
                    seed: i as u64,
                    len: if i == bucket / 2 {
                        24_576
                    } else {
                        128 + (i * 97) % 384
                    },
                })
                .collect(),
        ),
    ];
    // Uniform rows share two seeds' data: the timing does not depend on
    // the values, and every row still has its own pages.
    for len in [4096, 16_384, 32_768] {
        shapes.push((
            format!("uniform{len}"),
            (0..bucket)
                .map(|i| Row {
                    seed: 5000 + (i % 2) as u64,
                    len,
                })
                .collect(),
        ));
    }
    shapes
}

/// KV bytes one bench step may hold; larger steps are skipped and listed.
const BENCH_KV_BYTES: u64 = 6 << 30;

/// Kernel time of paged decode (partial and merge) as planned, for
/// balanced, skewed and uniform 4K, 16K and 32K-token steps of buckets 8
/// to 256, with read-sweep eviction before each launch. A/B comparisons
/// run this bench in two builds. With `CHOPINFER_KERNEL_BENCH=1`;
/// `CHOPINFER_ATTN_BENCH_VARIANTS` (comma-separated names) narrows the
/// variants.
#[test]
fn attention_decode_timing_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("decode timing: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs").unwrap();
        return;
    }
    use kernel_lab::eviction::EvictionMode;
    gpu_test(12 * 1024 * 1024 * 1024, |session| {
        let wanted = std::env::var("CHOPINFER_ATTN_BENCH_VARIANTS").ok();
        let variants = paged_decode_variants()?
            .into_iter()
            .filter(|v| {
                wanted
                    .as_deref()
                    .is_none_or(|names| names.split(',').any(|name| name == v.name))
            })
            .collect::<Vec<_>>();
        if variants.is_empty() {
            return Err("no paged decode variant matches CHOPINFER_ATTN_BENCH_VARIANTS".into());
        }
        let (mut timed, mut skipped) = (0, 0);
        for v in variants {
            let module = module(session, v.profile)?;
            let element = if v.query.kv_dtype == DType::Fp8E4M3 {
                1
            } else {
                2
            };
            let token_bytes =
                u64::from(v.query.kv_heads * (v.query.qk_dim + v.query.v_dim)) * element;
            for bucket in [8_usize, 16, 32, 64, 128, 256] {
                for (shape, rows) in bench_shapes(bucket) {
                    let tokens = rows.iter().map(|row| row.len as u64).sum::<u64>();
                    let mut receipt = BTreeMap::new();
                    receipt.insert("case".into(), json!("attention_decode_timing_bench"));
                    receipt.insert("variant".into(), json!(v.name));
                    receipt.insert("bucket".into(), json!(bucket));
                    receipt.insert("shape".into(), json!(shape));
                    receipt.insert("tokens".into(), json!(tokens));
                    if tokens * token_bytes > BENCH_KV_BYTES {
                        receipt.insert("skipped".into(), json!("KV above the bench budget"));
                        skipped += 1;
                    } else {
                        let data = rows
                            .iter()
                            .map(|&row| ((row.seed, row.len), row))
                            .collect::<BTreeMap<_, _>>()
                            .into_iter()
                            .map(|(key, row)| (key, row_data(v, row)))
                            .collect::<BTreeMap<_, _>>();
                        let s = step(session, v, &data, &rows, bucket as u32, 0)?;
                        drop(data);
                        let (partial, merge) = functions(&module, v, &s.planned)?;
                        let bindings = Bindings {
                            buffers: s.buffers.iter().map(|(&id, b)| (id, b)).collect(),
                        };
                        let timing = kernel_lab::time_prepared(
                            session,
                            5,
                            31,
                            EvictionMode::Read,
                            |_| Ok(()),
                            |stream| {
                                launch::launch(
                                    &session.context,
                                    &partial,
                                    stream,
                                    s.planned.partial,
                                    &s.planned.partial_args,
                                    &bindings,
                                )?;
                                launch::launch(
                                    &session.context,
                                    &merge,
                                    stream,
                                    s.planned.merge,
                                    &s.planned.merge_args,
                                    &bindings,
                                )
                            },
                        )?;
                        receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
                        receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
                        timed += 1;
                    }
                    eprintln!(
                        "{}",
                        serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                    );
                    session.receipt("bench", receipt)?;
                }
            }
        }
        if timed == 0 {
            return Err("no bench step ran".into());
        }
        eprintln!("decode timing: {timed} steps timed, {skipped} skipped for KV size");
        Ok(())
    })
    .expect("decode timing");
}
