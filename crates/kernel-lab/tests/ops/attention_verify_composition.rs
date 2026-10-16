//! Verify attention's row-local split: at a fixed bucket, an item's output
//! bits do not depend on the other items of its step, their number, key
//! lengths, token counts, order, values, pages or ring slots (verify.cuh,
//! `piece_units`). Probe items are launched alone and beside companions,
//! causal and bidirectional, paged and ring; every launch writes a poisoned
//! output and workspace, and every live token row must be written and
//! finite. The companions move the step's CTA ranges across every probe:
//! under the step-wide plan this replaced, which cut those ranges across
//! items, most probe steps changed their partial boundaries.

use super::*;

/// One verify item: its seed, the cached prefix before its tokens and its
/// query token count.
#[derive(Clone, Copy)]
struct Item {
    seed: u64,
    prefix: usize,
    len: usize,
}

impl Item {
    fn keys(self) -> usize {
        self.prefix + self.len
    }
}

/// An item's Q and its K and V at every position up to its last token, as
/// the cache holds them (`k`, `v`), and its own rows in BF16 (`k_new`,
/// `v_new`, which ring layouts read): the same bytes in every step.
struct ItemData {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    k_new: Vec<u8>,
    v_new: Vec<u8>,
}

fn scales(v: Variant) -> [f32; 2] {
    if v.query.kv_dtype == DType::Fp8E4M3 {
        [0.5, 0.25]
    } else {
        [1.0, 1.0]
    }
}

fn item_data(v: Variant, item: Item) -> ItemData {
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let mut state = 0x7665_7269_6679_2d69
        ^ item.seed
        ^ ((item.prefix as u64) << 20)
        ^ ((item.len as u64) << 44);
    let (q, _) = encode(&samples(item.len * qh * qd, &mut state), false, 1.0);
    let [k_scale, v_scale] = scales(v);
    let k_values = samples(item.keys() * kh * qd, &mut state);
    let v_values = samples(item.keys() * kh * vd, &mut state);
    let own = |values: &[f32], dim: usize| encode(&values[item.prefix * kh * dim..], false, 1.0).0;
    ItemData {
        k_new: own(&k_values, qd),
        v_new: own(&v_values, vd),
        q,
        k: encode(&k_values, fp8, k_scale).0,
        v: encode(&v_values, fp8, v_scale).0,
    }
}

/// One planned verify step of `items` at `bucket`. `gap` free pages (paged)
/// or ring slots precede the first item's, so a probe's pages or slot
/// differ between steps.
struct Step {
    buffers: BTreeMap<u64, DeviceBuffer>,
    planned: attention::PlannedAttention,
    tokens: usize,
    row_bytes: usize,
}

fn step(
    session: &Session,
    v: Variant,
    data: &BTreeMap<(u64, usize, usize), ItemData>,
    items: &[Item],
    bucket: u32,
    gap: usize,
) -> LabResult<Step> {
    let ring = v.layout == CacheLayout::Ring;
    let element = if v.query.kv_dtype == DType::Fp8E4M3 {
        1
    } else {
        2
    };
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let (k_row, v_row) = (kh * qd * element, kh * vd * element);
    let page = v.query.page_size as usize;
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let mut tables = vec![0_i32; if ring { 0 } else { items.len() * max_pages }];
    let mut slot_ids = Vec::new();
    let blocks = if ring {
        gap + items.len()
    } else {
        gap + items.iter().map(|i| i.keys().div_ceil(page)).sum::<usize>()
    };
    let (mut k_cache, mut v_cache) = (
        vec![0_u8; blocks * page * k_row],
        vec![0_u8; blocks * page * v_row],
    );
    let (mut q, mut k_new, mut v_new) = (Vec::new(), Vec::new(), Vec::new());
    let (mut q_start, mut q_len, mut prefix_len, mut seq_lens) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut next = gap;
    for (b, item) in items.iter().enumerate() {
        let d = &data[&(item.seed, item.prefix, item.len)];
        if ring {
            // The row engine writes an item's own rows into the ring after
            // attention: the ring holds the positions before its prefix.
            let slot = gap + items.len() - 1 - b;
            slot_ids.push(slot as i32);
            for pos in item.prefix.saturating_sub(page)..item.prefix {
                let at = slot * page + pos % page;
                k_cache[at * k_row..(at + 1) * k_row]
                    .copy_from_slice(&d.k[pos * k_row..(pos + 1) * k_row]);
                v_cache[at * v_row..(at + 1) * v_row]
                    .copy_from_slice(&d.v[pos * v_row..(pos + 1) * v_row]);
            }
        } else {
            let pages = item.keys().div_ceil(page);
            for p in 0..pages {
                tables[b * max_pages + p] = (next + p) as i32;
            }
            let at = next * page;
            k_cache[at * k_row..at * k_row + d.k.len()].copy_from_slice(&d.k);
            v_cache[at * v_row..at * v_row + d.v.len()].copy_from_slice(&d.v);
            next += pages;
        }
        q_start.push((q.len() / (qh * qd * 2)) as i32);
        q_len.push(item.len as i32);
        prefix_len.push(item.prefix as i32);
        seq_lens.push(item.keys() as i32);
        q.extend_from_slice(&d.q);
        k_new.extend_from_slice(&d.k_new);
        v_new.extend_from_slice(&d.v_new);
    }
    let tokens = items.iter().map(|i| i.len).sum::<usize>();
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, put_bytes(session, &q)?);
    buffers.insert(K, put_bytes(session, &k_cache)?);
    buffers.insert(V, put_bytes(session, &v_cache)?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    buffers.insert(SLOT, put_i32(session, &slot_ids)?);
    buffers.insert(SEQ, put_i32(session, &seq_lens)?);
    buffers.insert(START, put_i32(session, &q_start)?);
    buffers.insert(LEN, put_i32(session, &q_len)?);
    buffers.insert(PREFIX, put_i32(session, &prefix_len)?);
    buffers.insert(LIVE, put_i32(session, &[items.len() as i32])?);
    buffers.insert(K_NEW, put_bytes(session, &k_new)?);
    buffers.insert(V_NEW, put_bytes(session, &v_new)?);
    let sink: Vec<f32> = (0..qh).map(|h| (h as f32 - 8.0) / 16.0).collect();
    buffers.insert(SINK, put_bytes(session, &encode(&sink, false, 1.0).0)?);
    let row_bytes = qh * vd * 2;
    let capacity = (bucket * v.query_capacity).min(attention::MAX_VERIFY_TOKENS) as usize;
    buffers.insert(OUT, session.allocate(capacity * row_bytes)?);
    let [k_scale, v_scale] = scales(v);
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: (page * k_row) as u64,
        v_page_stride_bytes: (page * v_row) as u64,
        k_map: None,
        v_map: None,
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
        k_scale,
        v_scale,
    };
    let counts = RuntimeCounts {
        m: tokens as u32,
        batch: items.len() as u32,
        context: items.iter().map(|i| i.keys()).max().unwrap_or(1) as u32,
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
        tokens,
        row_bytes,
    })
}

impl Step {
    /// Each live token row's output after one launch on a poisoned output
    /// and workspace: every live token row written and finite, the bucket's
    /// padding rows still poisoned.
    fn outputs(
        &self,
        session: &Session,
        module: &Module,
        v: Variant,
        label: &str,
    ) -> LabResult<Vec<Vec<u8>>> {
        let (partial, merge) = functions(module, v, &self.planned)?;
        poison::poison(session, &[&self.buffers[&OUT], &self.buffers[&WORK]])?;
        let bindings = Bindings {
            buffers: self.buffers.iter().map(|(&id, b)| (id, b)).collect(),
        };
        launch::launch(
            &session.context,
            &partial,
            &session.stream,
            self.planned.partial,
            &self.planned.partial_args,
            &bindings,
        )?;
        kernel_lab::launched(&self.planned.partial_symbol);
        if !self.planned.direct_output {
            launch::launch(
                &session.context,
                &merge,
                &session.stream,
                self.planned.merge,
                &self.planned.merge_args,
                &bindings,
            )?;
            kernel_lab::launched(&self.planned.merge_symbol);
        }
        let bytes = poison::read(session, &self.buffers[&OUT])?;
        super::invariance::written_rows(label, &bytes, self.tokens, self.row_bytes)?;
        Ok(bytes[..self.tokens * self.row_bytes]
            .chunks(self.row_bytes)
            .map(<[u8]>::to_vec)
            .collect())
    }
}

/// Probe items: prefixes from one key unit to a long context, with full,
/// partial and single-token item lengths.
const PROBES: [(usize, usize); 6] = [
    (37, 8),
    (700, 8),
    (1500, 3),
    (4097, 8),
    (9000, 1),
    (20_000, 8),
];

fn probes() -> Vec<Item> {
    PROBES
        .iter()
        .enumerate()
        .map(|(i, &(prefix, len))| Item {
            seed: 100 + i as u64,
            prefix,
            len,
        })
        .collect()
}

/// Companion sets for a bucket: balanced mid contexts, the same reversed,
/// one very long item among short ones, and a full bucket, each within the
/// bucket's query token rows. Each is (name, items in step order, free
/// pages or slots before the first item's).
fn steps_for(v: Variant, bucket: usize) -> Vec<(&'static str, Vec<Item>, usize)> {
    let probes = probes();
    let room = bucket - probes.len();
    let budget = (bucket as u32 * v.query_capacity).min(attention::MAX_VERIFY_TOKENS) as usize
        - probes.iter().map(|i| i.len).sum::<usize>();
    let max_len = v.query_capacity as usize;
    // Companion i of `count`, its length cut so the set fits the budget.
    let companion = |i: usize, count: usize, prefix: usize, len: usize| Item {
        seed: 1000 + i as u64,
        prefix,
        len: len.min(max_len).min((budget / count.max(1)).max(1)),
    };
    let balanced_count = room.min(40);
    let balanced = (0..balanced_count)
        .map(|i| companion(i, balanced_count, 300 + 97 * i % 1700, 1 + i % 8))
        .collect::<Vec<_>>();
    let skewed_count = room.min(24);
    let mut skewed = vec![companion(500, skewed_count, 30_000, 8)];
    skewed.extend((1..skewed_count).map(|i| companion(500 + i, skewed_count, 64 + i, 1 + i % 2)));
    let full = (0..room)
        .map(|i| companion(2000 + i, room, 128 + 211 * i % 2900, 1 + i % 4))
        .collect::<Vec<_>>();
    let interleave = |companions: &[Item]| {
        let mut items = Vec::new();
        for i in 0..probes.len().max(companions.len()) {
            items.extend(companions.get(i));
            items.extend(probes.get(i));
        }
        items
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

/// Each step's name and its probes' token rows, in probe order.
type ProbeOutputs = Vec<(&'static str, Vec<Vec<Vec<u8>>>)>;

fn probe_outputs(
    session: &Session,
    module: &Module,
    v: Variant,
    data: &BTreeMap<(u64, usize, usize), ItemData>,
    bucket: usize,
) -> LabResult<ProbeOutputs> {
    let mut all = Vec::new();
    for (name, items, gap) in steps_for(v, bucket) {
        let tokens = items.iter().map(|i| i.len).sum::<usize>();
        let capacity = (bucket as u32 * v.query_capacity).min(attention::MAX_VERIFY_TOKENS);
        if items.len() > bucket || tokens > capacity as usize {
            return Err(format!(
                "{} b{bucket} {name}: step exceeds its bucket",
                v.name
            ));
        }
        let s = step(session, v, data, &items, bucket as u32, gap)?;
        let label = format!("{} b{bucket} {name}", v.name);
        let rows = s.outputs(session, module, v, &label)?;
        let mut starts = Vec::new();
        let mut next = 0;
        for item in &items {
            starts.push(next);
            next += item.len;
        }
        let by_probe = probes()
            .iter()
            .map(|probe| {
                let at = items
                    .iter()
                    .position(|item| item.seed == probe.seed)
                    .expect("every step holds the probes");
                rows[starts[at]..starts[at] + probe.len].to_vec()
            })
            .collect();
        all.push((name, by_probe));
    }
    Ok(all)
}

/// Every verify variant (causal and bidirectional, paged and ring),
/// prepared for buckets up to 256 items.
fn verify_variants() -> LabResult<Vec<Variant>> {
    let wide = attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| {
            v.query.phase == AttentionPhase::Verify
                && !kernel_lab::lab_shapes::PROFILES.contains(&v.profile)
        })
        .map(|v| {
            v.with_batch_capacity(attention::MAX_BATCH_ROWS)
                .map_err(|e| format!("{}: {e}", v.name))
        })
        .collect::<LabResult<Vec<_>>>()?;
    if !wide.iter().any(|v| v.layout == CacheLayout::Ring) || !wide.iter().any(|v| !v.query.causal)
    {
        return Err("verify variants lack a ring or a bidirectional variant".into());
    }
    Ok(wide)
}

/// Buckets of the composition test: from the first that holds the probes
/// and companions to the widest whose query rows still exceed the probes'.
const BUCKETS: [usize; 4] = [8, 16, 32, 64];

/// Every probe item's bits are the same alone and beside balanced,
/// reversed, skewed and full companions, with other pages or ring slots, at
/// buckets 8 to 64.
#[test]
fn attention_verify_items_ignore_their_companions() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let variants = verify_variants()?;
        let mut runs = 0;
        for &v in &variants {
            let module = module(session, v.profile)?;
            let mut data = BTreeMap::new();
            for bucket in BUCKETS {
                for (_, items, _) in steps_for(v, bucket) {
                    for item in items {
                        data.entry((item.seed, item.prefix, item.len))
                            .or_insert_with(|| item_data(v, item));
                    }
                }
            }
            let mut receipt = BTreeMap::new();
            receipt.insert("variant".into(), json!(v.name));
            for bucket in BUCKETS {
                let local = probe_outputs(session, &module, v, &data, bucket)?;
                let (_, alone) = &local[0];
                for (name, outputs) in &local[1..] {
                    for (p, (got, expected)) in outputs.iter().zip(alone).enumerate() {
                        if got != expected {
                            return Err(format!(
                                "{} b{bucket}: probe {p} ({} keys) differs {name} against alone",
                                v.name,
                                probes()[p].keys()
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
            session.receipt("attention_verify_composition", receipt)?;
        }
        if runs != variants.len() * BUCKETS.len() {
            return Err(format!(
                "{runs} variant buckets ran, expected {}",
                variants.len() * BUCKETS.len()
            ));
        }
        Ok(())
    })
    .expect("row-local verify attention ignores its companions");
}
