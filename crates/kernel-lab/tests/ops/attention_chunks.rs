//! Prefill attention steps over chunks of one prompt, after the cached
//! prefix of the earlier chunks, for the generated invariance tests
//! (`attention_invariance.rs`), and their timing case.

use super::*;

/// A foreign item that shares some steps: its prefix and chunk.
const FOREIGN: (usize, usize) = (700, 900);

/// Per-position K, V and Q rows of one prompt, encoded as the cache holds
/// them. Q rows start at position `q_from`.
pub(super) struct Prompt {
    q: Vec<u8>,
    q_from: usize,
    k: Vec<u8>,
    v: Vec<u8>,
    k_new: Vec<u8>,
    v_new: Vec<u8>,
    k_scale: f32,
    v_scale: f32,
}

impl Prompt {
    pub(super) fn new(v: &Variant, tokens: usize, seed: u64) -> Self {
        let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
        let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
        let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
        let mut state = seed;
        let (q, _) = encode(&samples(tokens * qh * qd, &mut state), false, 1.0);
        let keys = samples(tokens * kh * qd, &mut state);
        let values = samples(tokens * kh * vd, &mut state);
        let [k_scale, v_scale] = if fp8 { [0.5, 0.25] } else { [1.0, 1.0] };
        let (k, _) = encode(&keys, fp8, k_scale);
        let (v_bytes, _) = encode(&values, fp8, v_scale);
        let (k_new, _) = encode(&keys, false, 1.0);
        let (v_new, _) = encode(&values, false, 1.0);
        Self {
            q,
            q_from: 0,
            k,
            v: v_bytes,
            k_new,
            v_new,
            k_scale,
            v_scale,
        }
    }

    /// A prompt of `tokens` positions whose Q rows exist from `q_from` on:
    /// decode rows over long prefixes need Q only at their own positions.
    pub(super) fn decoding(v: &Variant, tokens: usize, seed: u64, q_from: usize) -> Self {
        let mut prompt = Self::new(v, 0, seed);
        let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
        let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
        let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
        let mut state = seed ^ 0x6465_636f_6465;
        prompt.q = encode(
            &samples((tokens - q_from) * qh * qd, &mut state),
            false,
            1.0,
        )
        .0;
        prompt.q_from = q_from;
        let keys = samples(tokens * kh * qd, &mut state);
        let values = samples(tokens * kh * vd, &mut state);
        prompt.k = encode(&keys, fp8, prompt.k_scale).0;
        prompt.v = encode(&values, fp8, prompt.v_scale).0;
        prompt.k_new = encode(&keys, false, 1.0).0;
        prompt.v_new = encode(&values, false, 1.0).0;
        prompt
    }

    /// Q rows `rows`, which start at or after `q_from`.
    fn queries(&self, width: usize, rows: std::ops::Range<usize>) -> &[u8] {
        Self::row(
            &self.q,
            width,
            rows.start - self.q_from..rows.end - self.q_from,
        )
    }

    fn row(bytes: &[u8], width: usize, rows: std::ops::Range<usize>) -> &[u8] {
        &bytes[rows.start * width..rows.end * width]
    }
}

/// One step item: a prompt's rows [prefix, prefix + len).
pub(super) struct Item<'a> {
    pub(super) prompt: &'a Prompt,
    pub(super) prefix: usize,
    pub(super) len: usize,
}

/// One planned prefill step on the device.
pub(super) struct Step {
    buffers: BTreeMap<u64, DeviceBuffer>,
    planned: attention::PlannedAttention,
    partial: Function,
    merge: Function,
    /// Each item's first packed row and row count.
    rows: Vec<(usize, usize)>,
    row_elements: usize,
}

impl Step {
    pub(super) fn launch(&self, session: &Session, stream: &runtime_cuda::Stream) -> LabResult<()> {
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect(),
        };
        launch::launch(
            &session.context,
            &self.partial,
            stream,
            self.planned.partial,
            &self.planned.partial_args,
            &bindings,
        )?;
        kernel_lab::launched(&self.planned.partial_symbol);
        if !self.planned.direct_output {
            launch::launch(
                &session.context,
                &self.merge,
                stream,
                self.planned.merge,
                &self.planned.merge_args,
                &bindings,
            )?;
            kernel_lab::launched(&self.planned.merge_symbol);
        }
        Ok(())
    }

    /// Each item's output rows as BF16 bits.
    pub(super) fn outputs(&self, session: &Session) -> LabResult<Vec<Vec<u16>>> {
        let tokens: usize = self.rows.iter().map(|&(_, len)| len).sum();
        let all = session.download_bf16(&self.buffers[&OUT], tokens * self.row_elements)?;
        Ok(self
            .rows
            .iter()
            .map(|&(start, len)| {
                all[start * self.row_elements..(start + len) * self.row_elements].to_vec()
            })
            .collect())
    }
}

/// Plans one prefill step of `items`. Paged items own consecutive pages
/// holding their whole prompt; a ring item's slot holds the 128 keys before
/// its prefix at their positions modulo 128.
pub(super) fn prepare_step(
    session: &Session,
    module: &Module,
    v: Variant,
    items: &[Item<'_>],
) -> LabResult<Step> {
    let ring = v.layout == CacheLayout::Ring;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let element = if fp8 { 1 } else { 2 };
    let (k_row, v_row) = (kh * qd * element, kh * vd * element);
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let mut tables = vec![0_i32; items.len() * max_pages];
    let (mut k_cache, mut v_cache) = (Vec::new(), Vec::new());
    let (mut q, mut k_new, mut v_new) = (Vec::new(), Vec::new(), Vec::new());
    let mut slots = 0;
    for (b, item) in items.iter().enumerate() {
        let end = item.prefix + item.len;
        if ring {
            let mut k_slot = vec![0_u8; page * k_row];
            let mut v_slot = vec![0_u8; page * v_row];
            for pos in item.prefix.saturating_sub(page)..item.prefix {
                let at = pos % page;
                k_slot[at * k_row..(at + 1) * k_row].copy_from_slice(Prompt::row(
                    &item.prompt.k,
                    k_row,
                    pos..pos + 1,
                ));
                v_slot[at * v_row..(at + 1) * v_row].copy_from_slice(Prompt::row(
                    &item.prompt.v,
                    v_row,
                    pos..pos + 1,
                ));
            }
            k_cache.extend(k_slot);
            v_cache.extend(v_slot);
            slots += 1;
        } else {
            let pages = end.div_ceil(page);
            for p in 0..pages {
                tables[b * max_pages + p] = (slots + p) as i32;
            }
            let mut k_pages = Prompt::row(&item.prompt.k, k_row, 0..end).to_vec();
            let mut v_pages = Prompt::row(&item.prompt.v, v_row, 0..end).to_vec();
            k_pages.resize(pages * page * k_row, 0);
            v_pages.resize(pages * page * v_row, 0);
            k_cache.extend(k_pages);
            v_cache.extend(v_pages);
            slots += pages;
        }
        let rows = item.prefix..end;
        q.extend_from_slice(item.prompt.queries(qh * qd * 2, rows.clone()));
        k_new.extend_from_slice(Prompt::row(&item.prompt.k_new, kh * qd * 2, rows.clone()));
        v_new.extend_from_slice(Prompt::row(&item.prompt.v_new, kh * vd * 2, rows));
    }
    let starts: Vec<i32> = items
        .iter()
        .scan(0, |next, item| {
            let start = *next;
            *next += item.len;
            Some(start as i32)
        })
        .collect();
    let tokens: usize = items.iter().map(|item| item.len).sum();
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, put_bytes(session, &q)?);
    buffers.insert(K, put_bytes(session, &k_cache)?);
    buffers.insert(V, put_bytes(session, &v_cache)?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    buffers.insert(
        SLOT,
        put_i32(session, &(0..items.len() as i32).collect::<Vec<_>>())?,
    );
    let ends: Vec<i32> = items.iter().map(|i| (i.prefix + i.len) as i32).collect();
    buffers.insert(SEQ, put_i32(session, &ends)?);
    buffers.insert(START, put_i32(session, &starts)?);
    buffers.insert(
        LEN,
        put_i32(
            session,
            &items.iter().map(|i| i.len as i32).collect::<Vec<_>>(),
        )?,
    );
    buffers.insert(
        PREFIX,
        put_i32(
            session,
            &items.iter().map(|i| i.prefix as i32).collect::<Vec<_>>(),
        )?,
    );
    buffers.insert(LIVE, put_i32(session, &[items.len() as i32])?);
    buffers.insert(K_NEW, put_bytes(session, &k_new)?);
    buffers.insert(V_NEW, put_bytes(session, &v_new)?);
    let sink: Vec<f32> = (0..qh).map(|h| (h as f32 - 8.0) / 16.0).collect();
    buffers.insert(SINK, put_bytes(session, &encode(&sink, false, 1.0).0)?);
    let out = session.allocate(tokens * qh * vd * 2)?;
    poison::poison(session, &[&out])?;
    buffers.insert(OUT, out);
    let (k_map, v_map) = v
        .cache_maps_with_strides(
            slots as u32,
            K,
            V,
            (page * k_row) as u64,
            (page * v_row) as u64,
        )
        .map_err(|e| e.to_string())?;
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: (page * k_row) as u64,
        v_page_stride_bytes: (page * v_row) as u64,
        k_map: Some(k_map),
        v_map: Some(v_map),
        block_tables: (!ring).then_some(TABLE),
        slot_ids: ring.then_some(SLOT),
        seq_lens: Some(SEQ),
        q_start: Some(START),
        q_len: Some(LEN),
        prefix_len: Some(PREFIX),
        live_batch: LIVE,
        k_new: ring.then_some(K_NEW),
        v_new: ring.then_some(V_NEW),
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale: items[0].prompt.k_scale,
        v_scale: items[0].prompt.v_scale,
    };
    let counts = RuntimeCounts {
        m: tokens as u32,
        batch: items.len() as u32,
        context: *ends.iter().max().expect("items") as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let planned = v
        .plan_for_bucket(items.len().next_power_of_two() as u32, counts, arguments)
        .map_err(|e| e.to_string())?;
    let work = session.allocate((planned.partial.workspace_bytes as usize).max(4))?;
    poison::poison(session, &[&work])?;
    buffers.insert(WORK, work);
    let (partial, merge) = functions(module, v, &planned)?;
    Ok(Step {
        buffers,
        planned,
        partial,
        merge,
        rows: items
            .iter()
            .zip(&starts)
            .map(|(item, &start)| (start as usize, item.len))
            .collect(),
        row_elements: qh * vd,
    })
}

/// Every prompt row's output under `ends`, beside the foreign item when
/// `foreign`.
pub(super) fn prefill(
    session: &Session,
    module: &Module,
    v: Variant,
    prompt: &Prompt,
    other: &Prompt,
    ends: &[usize],
    foreign: bool,
) -> LabResult<Vec<u16>> {
    let mut rows = Vec::new();
    let mut prefix = 0;
    for &end in ends {
        let mut items = vec![Item {
            prompt,
            prefix,
            len: end - prefix,
        }];
        if foreign {
            items.insert(
                0,
                Item {
                    prompt: other,
                    prefix: FOREIGN.0,
                    len: FOREIGN.1,
                },
            );
        }
        let step = prepare_step(session, module, v, &items)?;
        step.launch(session, &session.stream)?;
        let outputs = step.outputs(session)?;
        rows.extend_from_slice(outputs.last().expect("the prompt's item"));
        prefix = end;
    }
    Ok(rows)
}

/// Kernel time (partial and merge) of one prompt chunk at the rows and
/// cached prefixes of prefill: 2,048- and 1,024-token chunks at short and
/// 32K prefixes, and small chunks after long ones.
#[test]
fn attention_prefill_chunk_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention chunk benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let module = module(session, model_mimo26_contract::sm120::config::PROFILE)?;
        let cases: [(&str, &[(usize, usize)]); 3] = [
            (
                "mimo_ga_bf16_prefill",
                &[
                    (0, 2048),
                    (1024, 1024),
                    (4096, 2048),
                    (32_768, 2048),
                    (33_792, 1024),
                    (100_000, 64),
                    (100_000, 1),
                ],
            ),
            ("mimo_ga_fp8_prefill", &[(0, 2048), (32_768, 2048)]),
            (
                "mimo_swa_prefill",
                &[(129, 2048), (4133, 2048), (1061, 1024)],
            ),
        ];
        for (name, points) in cases {
            let v = named(name)?;
            for &(prefix, len) in points {
                let prompt = Prompt::new(&v, prefix + len, 0x0062_656e_6368);
                let step = prepare_step(
                    session,
                    &module,
                    v,
                    &[Item {
                        prompt: &prompt,
                        prefix,
                        len,
                    }],
                )?;
                let timing = time(session, 2, 5, |stream| step.launch(session, stream))?;
                let us = f64::from(timing.median_ms) * 1000.0;
                eprintln!("{name} prefix={prefix} rows={len}: {us:.1} us");
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("attention_prefill_chunk_bench"));
                receipt.insert("variant".into(), json!(name));
                receipt.insert("prefix".into(), json!(prefix));
                receipt.insert("rows".into(), json!(len));
                receipt.insert("median_us".into(), json!(us));
                receipt.insert("min_us".into(), json!(f64::from(timing.min_ms) * 1000.0));
                receipt.insert("merge".into(), json!(!step.planned.direct_output));
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("prefill attention chunk benchmarks");
}
