//! MiMo exact verify rows (`Variant::plan_rows`): every token row of a verify
//! step of up to eight rows, run through the decode variant's rows entries,
//! equals the same row run alone as a one-row decode step at its position
//! (bucket 1's unfused decode entry, whose bits the folded AR entry shares:
//! `rope_fold_positions`), bit for bit. Paged layers read the step's rows
//! from pages the row engine wrote before attention; the sliding window reads
//! keys before the item's prefix from the ring and the step's own rows from
//! `k_new`/`v_new`, while the one-row reference reads a ring that holds them.
//! Cases cover one and two items, full and partial buckets, prefixes around
//! the window and ring size, and contexts long enough for a row's units to
//! span several stream-K CTAs, for BF16 and FP8 global KV at every rank count.
//! Paged variants check both rows entries: one CTA set per token row
//! (`plan_rows`) and the shared-key entry (`plan_rows_shared`), whose CTAs
//! run up to four rows of an item that share the one-row plan over one read
//! of their key range. Long contexts (131,000) reach the shared-key CTAs'
//! longest ranges. Wide steps of up to eight items at the 64-token bucket
//! and up to 32 items at the 256-token bucket, full and partial, check the
//! per-row entry alone (the shared-key entry holds at most 16 rows), as the
//! serving diagnostic that runs wide verify attention as exact rows does.
//!
//! `attention_exact_rows_bench` (`CHOPINFER_KERNEL_BENCH=1`) times the rows
//! entries against the verify entries of C1 steps of two, four and eight rows.

use super::*;

const EXACT_ROWS: usize = 8;

/// One verify item: `prefix` cached positions and `len` new rows.
#[derive(Clone, Copy, Debug)]
pub(super) struct Item {
    pub(super) prefix: usize,
    pub(super) len: usize,
}

/// Buffers of one step: the verify rows' inputs and the caches they read.
pub(super) struct Step {
    items: Vec<Item>,
    pub(super) buffers: BTreeMap<u64, DeviceBuffer>,
    pub(super) arguments: Buffers,
    /// Host copies used to build the one-row references.
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    q: Vec<u8>,
    k_new: Vec<u8>,
    v_new: Vec<u8>,
    sink: Vec<u8>,
    tables: Vec<i32>,
    max_pages: usize,
}

fn words(state: &mut u64, count: usize, fp8: bool, scale: f32) -> Vec<u8> {
    encode(&samples(count, state), fp8, scale).0
}

pub(super) fn step(session: &Session, v: Variant, items: &[Item], seed: u64) -> LabResult<Step> {
    let ring = v.layout == CacheLayout::Ring;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let element = if fp8 { 1 } else { 2 };
    let [k_scale, v_scale] = if fp8 { [0.5_f32, 0.25] } else { [1.0, 1.0] };
    let mut state = 0x7c3d_91a5_2e4b_0f67 ^ seed;
    let tokens: usize = items.iter().map(|item| item.len).sum();
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    // Paged: each item's pages in reverse physical order; ring: one slot per
    // item, holding the positions before its prefix.
    let slots = if ring {
        items.len()
    } else {
        items
            .iter()
            .map(|item| (item.prefix + item.len).div_ceil(page))
            .sum()
    };
    let k_cache = words(&mut state, slots * page * kh * qd, fp8, k_scale);
    let v_cache = words(&mut state, slots * page * kh * vd, fp8, v_scale);
    let mut tables = vec![0_i32; if ring { 0 } else { items.len() * max_pages }];
    if !ring {
        let mut base = 0;
        for (index, item) in items.iter().enumerate() {
            let pages = (item.prefix + item.len).div_ceil(page);
            for p in 0..pages {
                tables[index * max_pages + p] = (base + pages - 1 - p) as i32;
            }
            base += pages;
        }
    }
    let q = words(&mut state, tokens * qh * qd, false, 1.0);
    let k_new = words(&mut state, tokens * kh * qd, false, 1.0);
    let v_new = words(&mut state, tokens * kh * vd, false, 1.0);
    let sink = words(&mut state, qh, false, 1.0);
    let starts = items
        .iter()
        .scan(0, |next, item| {
            let start = *next;
            *next += item.len;
            Some(start as i32)
        })
        .collect::<Vec<_>>();
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, put_bytes(session, &q)?);
    buffers.insert(K, put_bytes(session, &k_cache)?);
    buffers.insert(V, put_bytes(session, &v_cache)?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    buffers.insert(
        SLOT,
        put_i32(session, &(0..items.len() as i32).rev().collect::<Vec<_>>())?,
    );
    buffers.insert(
        SEQ,
        put_i32(
            session,
            &items
                .iter()
                .map(|item| (item.prefix + item.len) as i32)
                .collect::<Vec<_>>(),
        )?,
    );
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
    buffers.insert(SINK, put_bytes(session, &sink)?);
    let out = session.allocate(EXACT_ROWS.max(tokens.next_power_of_two()) * qh * vd * 2)?;
    poison::poison(session, &[&out])?;
    buffers.insert(OUT, out);
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: (page * kh * qd * element) as u64,
        v_page_stride_bytes: (page * kh * vd * element) as u64,
        k_map: None,
        v_map: None,
        block_tables: (!ring).then_some(TABLE),
        slot_ids: ring.then_some(SLOT),
        seq_lens: Some(SEQ),
        q_start: Some(START),
        q_len: Some(LEN),
        prefix_len: Some(PREFIX),
        live_batch: LIVE,
        k_new: Some(K_NEW),
        v_new: Some(V_NEW),
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale,
        v_scale,
    };
    Ok(Step {
        items: items.to_vec(),
        buffers,
        arguments,
        k_cache,
        v_cache,
        q,
        k_new,
        v_new,
        sink,
        tables,
        max_pages,
    })
}

pub(super) fn counts(items: &[Item], tokens: usize) -> RuntimeCounts {
    RuntimeCounts {
        m: tokens as u32,
        batch: 1,
        context: items
            .iter()
            .map(|item| item.prefix + item.len)
            .max()
            .unwrap_or(1) as u32,
        routed_rows: 0,
        split_k: 1,
    }
}

pub(super) fn run(
    session: &Session,
    cubin: &Module,
    planned: &attention::PlannedAttention,
    buffers: &mut BTreeMap<u64, DeviceBuffer>,
) -> LabResult<()> {
    prepare(session, cubin, planned, buffers)?;
    enqueue(session, cubin, planned, buffers)?;
    session.stream.synchronize().map_err(|e| e.to_string())
}

/// A poisoned workspace for `planned`, and its partial entry's shared
/// memory opt-in.
pub(super) fn prepare(
    session: &Session,
    cubin: &Module,
    planned: &attention::PlannedAttention,
    buffers: &mut BTreeMap<u64, DeviceBuffer>,
) -> LabResult<()> {
    let work = session.allocate(planned.partial.workspace_bytes.max(4) as usize)?;
    poison::poison(session, &[&work])?;
    buffers.insert(WORK, work);
    if planned.partial.dynamic_shared_bytes > 48 * 1024 {
        cubin
            .function(&planned.partial_symbol)
            .map_err(|e| e.to_string())?
            .set_max_dynamic_shared_memory(planned.partial.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Launches `planned`'s partial (and merge) entries on the session stream.
pub(super) fn enqueue(
    session: &Session,
    cubin: &Module,
    planned: &attention::PlannedAttention,
    buffers: &BTreeMap<u64, DeviceBuffer>,
) -> LabResult<()> {
    let bindings = Bindings {
        buffers: buffers.iter().map(|(&id, buffer)| (id, buffer)).collect(),
    };
    let partial = cubin
        .function(&planned.partial_symbol)
        .map_err(|e| e.to_string())?;
    launch::launch(
        &session.context,
        &partial,
        &session.stream,
        planned.partial,
        &planned.partial_args,
        &bindings,
    )?;
    kernel_lab::launched(&planned.partial_symbol);
    if !planned.direct_output {
        let merge = cubin
            .function(&planned.merge_symbol)
            .map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &merge,
            &session.stream,
            planned.merge,
            &planned.merge_args,
            &bindings,
        )?;
        kernel_lab::launched(&planned.merge_symbol);
    }
    Ok(())
}

/// Token `token` of `step` (item `item`, at `position`) as the lone row of a
/// decode step: the same caches, and for the ring the item's slot holding
/// the step's rows up to `position` at their ring slots.
pub(super) fn one_row(
    session: &Session,
    v: Variant,
    step: &Step,
    item: usize,
    token: usize,
    position: usize,
) -> LabResult<(BTreeMap<u64, DeviceBuffer>, Buffers)> {
    let ring = v.layout == CacheLayout::Ring;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let q_row = qh * qd * 2;
    let mut k_cache = step.k_cache.clone();
    let mut v_cache = step.v_cache.clone();
    let start = token - (position - step.items[item].prefix);
    let slot = step.items.len() - 1 - item;
    if ring {
        let (k_row, v_row) = (kh * qd * 2, kh * vd * 2);
        for at in step.items[item].prefix..=position {
            let new = start + at - step.items[item].prefix;
            let cache = slot * page + at % page;
            k_cache[cache * k_row..(cache + 1) * k_row]
                .copy_from_slice(&step.k_new[new * k_row..(new + 1) * k_row]);
            v_cache[cache * v_row..(cache + 1) * v_row]
                .copy_from_slice(&step.v_new[new * v_row..(new + 1) * v_row]);
        }
    }
    let mut buffers = BTreeMap::new();
    buffers.insert(
        Q,
        put_bytes(session, &step.q[token * q_row..(token + 1) * q_row])?,
    );
    buffers.insert(K, put_bytes(session, &k_cache)?);
    buffers.insert(V, put_bytes(session, &v_cache)?);
    buffers.insert(
        TABLE,
        put_i32(
            session,
            if ring {
                &[0]
            } else {
                &step.tables[item * step.max_pages..(item + 1) * step.max_pages]
            },
        )?,
    );
    buffers.insert(SLOT, put_i32(session, &[slot as i32])?);
    buffers.insert(SEQ, put_i32(session, &[position as i32 + 1])?);
    buffers.insert(LIVE, put_i32(session, &[1])?);
    buffers.insert(SINK, put_bytes(session, &step.sink)?);
    let out = session.allocate(qh * vd * 2)?;
    poison::poison(session, &[&out])?;
    buffers.insert(OUT, out);
    let arguments = Buffers {
        q_start: None,
        q_len: None,
        prefix_len: None,
        k_new: None,
        v_new: None,
        ..step.arguments.clone()
    };
    Ok((buffers, arguments))
}

/// The exact rows entries a variant has for a bucket of `rows`: the
/// shared-key entry is paged only and holds at most 16 rows.
fn entries(v: Variant, rows: usize) -> &'static [bool] {
    if v.layout == CacheLayout::Paged && rows <= 16 {
        &[false, true]
    } else {
        &[false]
    }
}

/// Plans exact rows with the per-row entry or, `shared`, the shared-key one.
pub(super) fn plan_exact(
    v: Variant,
    shared: bool,
    rows: u32,
    live: RuntimeCounts,
    buffers: Buffers,
) -> LabResult<attention::PlannedAttention> {
    if shared {
        v.plan_rows_shared(rows, live, buffers)
    } else {
        v.plan_rows(rows, live, buffers)
    }
    .map_err(|e| e.to_string())
}

/// The positions of each token row of `items`, token-major.
pub(super) fn positions(items: &[Item]) -> Vec<(usize, usize)> {
    items
        .iter()
        .enumerate()
        .flat_map(|(index, item)| (0..item.len).map(move |j| (index, item.prefix + j)))
        .collect()
}

fn exact_case(session: &Session, v: Variant, cubin: &Module, items: &[Item]) -> LabResult<()> {
    let tokens: usize = items.iter().map(|item| item.len).sum();
    let bucket = tokens.next_power_of_two().max(2);
    let mut step = step(
        session,
        v,
        items,
        tokens as u64 ^ (items[0].prefix as u64) << 12,
    )?;
    let width = v.query.q_heads as usize * v.query.v_dim as usize;
    let mut outputs = Vec::new();
    for &shared in entries(v, bucket) {
        let label = format!("{} shared {shared} {items:?}", v.name);
        let planned = plan_exact(
            v,
            shared,
            bucket as u32,
            counts(items, tokens),
            step.arguments.clone(),
        )
        .map_err(|e| format!("{label}: {e}"))?;
        let symbol = if shared {
            v.rows_shared_symbol()
        } else {
            v.rows_symbol()
        };
        if planned.partial_symbol != symbol {
            return Err(format!(
                "{label}: rows plan runs {}",
                planned.partial_symbol
            ));
        }
        poison::poison(session, &[&step.buffers[&OUT]])?;
        run(session, cubin, &planned, &mut step.buffers)?;
        let rows = session.download_bf16(&step.buffers[&OUT], bucket * width)?;
        // Rows past the step stay untouched.
        if let Some(at) = rows[tokens * width..]
            .iter()
            .position(|&bits| bits != POISON_BF16)
        {
            return Err(format!(
                "{label}: padding row element {} was written",
                tokens * width + at
            ));
        }
        outputs.push((label, rows));
    }
    for (token, (item, position)) in positions(items).into_iter().enumerate() {
        let (mut buffers, arguments) = one_row(session, v, &step, item, token, position)?;
        let one = v
            .plan_for_bucket(1, counts(items, 1), arguments)
            .map_err(|e| format!("{} {items:?}: {e}", v.name))?;
        run(session, cubin, &one, &mut buffers)?;
        let alone = session.download_bf16(&buffers[&OUT], width)?;
        for (label, rows) in &outputs {
            let row = &rows[token * width..(token + 1) * width];
            if let Some(at) = row.iter().zip(&alone).position(|(a, b)| a != b) {
                return Err(format!(
                    "{label}: token {token} (item {item}, position {position}) element {at} is {:#06x}, alone {:#06x}",
                    row[at], alone[at]
                ));
            }
        }
    }
    Ok(())
}

/// The poison fill of an untouched BF16 output element.
const POISON_BF16: u16 = u16::from_ne_bytes([POISON, POISON]);

fn exact_items() -> Vec<Vec<Item>> {
    let mut cases = Vec::new();
    for prefix in [
        0, 1, 7, 64, 120, 125, 126, 127, 128, 129, 200, 1017, 4095, 12_020, 33_000,
    ] {
        for len in [2, 3, 4, 5, 8] {
            cases.push(vec![Item { prefix, len }]);
        }
    }
    for len in [2, 8] {
        cases.push(vec![Item {
            prefix: 131_000,
            len,
        }]);
    }
    // Two items: a partial bucket of four and eight rows, and one-row items.
    for (a, b) in [(126, 3), (3000, 130), (40, 1), (12_020, 127)] {
        cases.push(vec![Item { prefix: a, len: 3 }, Item { prefix: b, len: 4 }]);
        cases.push(vec![Item { prefix: a, len: 1 }, Item { prefix: b, len: 1 }]);
    }
    cases
}

/// Wide verify steps at the 64- and 256-token buckets, full and partial:
/// many items of up to eight rows, prefixes around the window and ring size,
/// and one longer context per step (multi-CTA rows at the 64-token bucket;
/// each row's plan is the one-row plan at every bucket).
fn wide_items() -> Vec<Vec<Item>> {
    let prefixes = [0, 1, 7, 64, 120, 125, 126, 127, 128, 129, 200, 1017, 4095];
    let items = |lens: &[usize], long: usize| {
        lens.iter()
            .enumerate()
            .map(|(index, &len)| Item {
                prefix: if index == 3 {
                    long
                } else {
                    prefixes[index % prefixes.len()] + 3 * (index / prefixes.len())
                },
                len,
            })
            .collect::<Vec<_>>()
    };
    vec![
        items(&[8; 8], 33_000),
        items(&[8, 1, 4, 8, 2, 8, 6], 12_020),
        items(&[8; 32], 1017),
        items(&[[8; 16].as_slice(), &[3]].concat(), 4095),
    ]
}

#[test]
fn attention_exact_rows_equal_one_row_decode() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let mut checked = 0;
        for profile in model_mimo26_contract::sm120::PROFILES {
            let cubin = module(session, profile)?;
            let names = variants()
                .into_iter()
                .filter(|v| v.profile == profile && v.has_rows())
                .collect::<Vec<_>>();
            if names.len() != 3 {
                return Err(format!(
                    "{}: {} decode variants with rows entries",
                    profile.as_str(),
                    names.len()
                ));
            }
            for v in names {
                for items in exact_items() {
                    exact_case(session, v, &cubin, &items)?;
                    checked += 1;
                }
                for items in wide_items() {
                    exact_case(session, v, &cubin, &items)?;
                    checked += 1;
                }
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("attention_exact_rows"));
                receipt.insert("variant".into(), json!(v.name));
                receipt.insert("steps".into(), json!(exact_items().len()));
                receipt.insert(
                    "wide_steps".into(),
                    json!(
                        wide_items()
                            .iter()
                            .map(|items| items.iter().map(|item| item.len).sum::<usize>())
                            .collect::<Vec<_>>()
                    ),
                );
                receipt.insert(
                    "entries".into(),
                    json!(
                        entries(v, EXACT_ROWS)
                            .iter()
                            .map(|&shared| if shared {
                                v.rows_shared_symbol()
                            } else {
                                v.rows_symbol()
                            })
                            .collect::<Vec<_>>()
                    ),
                );
                receipt.insert("bitwise_equal_to_one_row".into(), json!(true));
                session.receipt("exact_rows", receipt)?;
            }
        }
        eprintln!("exact attention rows: {checked} steps bitwise equal to one-row decode");
        Ok(())
    })
    .expect("MiMo exact attention rows");
}

/// Median microseconds of `planned`'s partial and merge as one graph, after
/// one checked run, with L2 flushed before each launch.
fn time_planned(
    session: &Session,
    cubin: &Module,
    planned: &attention::PlannedAttention,
    buffers: &mut BTreeMap<u64, DeviceBuffer>,
) -> LabResult<f32> {
    run(session, cubin, planned, buffers)?;
    let graph = session.capture(|_| enqueue(session, cubin, planned, buffers))?;
    let timing = time(session, 3, 7, |stream| {
        graph.launch(stream).map_err(|e| e.to_string())
    })?;
    Ok(timing.median_ms * 1000.0)
}

#[test]
fn attention_exact_rows_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("exact attention rows bench: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs")
            .unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let cubin = module(session, model_mimo26_contract::sm120::config::PROFILE)?;
        for (decode, verify) in [
            ("mimo_ga_bf16_decode", "mimo_ga_bf16_verify"),
            ("mimo_ga_fp8_decode", "mimo_ga_fp8_verify"),
            ("mimo_swa_decode", "mimo_swa_verify"),
        ] {
            let (decode, verify) = (named(decode)?, named(verify)?);
            for (prefix, len) in [128, 1024, 8192, 32_768, 131_072]
                .into_iter()
                .flat_map(|prefix| [2, 4, EXACT_ROWS].map(|len| (prefix, len)))
            {
                let items = [Item { prefix, len }];
                let mut step = step(session, decode, &items, 5)?;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("attention_exact_rows_bench"));
                receipt.insert("variant".into(), json!(decode.name));
                receipt.insert("prefix".into(), json!(prefix));
                receipt.insert("rows".into(), json!(len));
                let bucket = len.next_power_of_two() as u32;
                let mut arms = vec![("rows", decode, false)];
                if entries(decode, len).contains(&true) {
                    arms.push(("rows_shared", decode, true));
                }
                arms.push(("verify", verify, false));
                for (name, v, shared) in arms {
                    let planned = if name == "verify" {
                        v.plan_for_bucket(
                            1,
                            RuntimeCounts {
                                batch: 1,
                                ..counts(&items, len)
                            },
                            step.arguments.clone(),
                        )
                        .map_err(|e| e.to_string())
                    } else {
                        plan_exact(
                            v,
                            shared,
                            bucket,
                            counts(&items, len),
                            step.arguments.clone(),
                        )
                    }?;
                    let us = time_planned(session, &cubin, &planned, &mut step.buffers)?;
                    receipt.insert(format!("{name}_us"), json!(us));
                }
                eprintln!(
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                );
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("MiMo exact attention rows bench");
}

/// About 5 ms of producer spin before the verify step's row RoPE.
const PRODUCER_CYCLES: u64 = 10_000_000;

/// Buffers of the pipeline check beyond the attention ids.
const ANGLES: u64 = 40;
const LIVE_TOKENS: u64 = 41;
const QKV_ALL: u64 = 42;

/// The whole attention of `len` tokens after `prefix` cached positions, as
/// serving runs it: AR folds RoPE and the KV write into one-row decode steps
/// (the table entry), one token after another; a verify step runs the verify
/// phase's row RoPE (pages written before attention, or `k_new`/`v_new` for
/// the ring) and then the exact rows (`shared`: the shared-key entry). Outputs
/// and written K/V must agree bit for bit, so a verify row serves AR's bits
/// from the same packed QKV rows. As in serving, where the QKV GEMM runs
/// before the row RoPE, a PDL producer that releases its dependents at once
/// and then spins for `PRODUCER_CYCLES` precedes the row RoPE: the row RoPE
/// and then attention start early and run their pre-wait code while the
/// pages still hold the previous contents, so attention that copies the
/// step's own rows before its wait reads them stale. The chain runs as one
/// captured graph, as serving replays it, and the case fails unless a probe
/// in attention's place shows the overlap (it starts before the producer
/// ends).
fn pipeline_case(
    session: &Session,
    v: Variant,
    shared: bool,
    prefix: usize,
    len: usize,
) -> LabResult<()> {
    let label = format!("{} shared {shared} prefix {prefix} len {len}", v.name);
    let ring = v.layout == CacheLayout::Ring;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qk, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let element = if fp8 { 1 } else { 2 };
    let row_width = qh * qk + kh * (qk + vd);
    let items = [Item { prefix, len }];
    let base = step(session, v, &items, 0x51de ^ prefix as u64)?;
    let theta = if ring { 1e4_f64 } else { 1e7 };
    let context = prefix + len;
    let angles = (0..context)
        .flat_map(|p| {
            (0..32).flat_map(move |pair| {
                let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, theta);
                [c as f32, s as f32]
            })
        })
        .collect::<Vec<_>>();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (context as u64);
    let qkv: Vec<u16> = (0..len * row_width)
        .map(|_| f32_to_bf16(((random_word(&mut state) % 2001) as f32 - 1000.0) / 250.0))
        .collect();
    let cubin = module(session, v.profile)?;
    let rows = session.load_cubin(ROW_CUBIN)?;
    let slot = 0_usize;
    let k_row = kh * qk * element;
    let v_row = kh * vd * element;
    let cache_offsets = |position: usize, stride: u64, width: usize| {
        if ring {
            (slot * page + position % page) * width
        } else {
            base.tables[position / page] as usize * stride as usize + position % page * width
        }
    };

    // AR: one folded decode step per token, on its own cache copy.
    let mut ar = BTreeMap::new();
    for (id, bytes) in [(K, &base.k_cache), (V, &base.v_cache)] {
        ar.insert(id, put_bytes(session, bytes)?);
    }
    ar.insert(TABLE, put_i32(session, &base.tables)?);
    ar.insert(SLOT, put_i32(session, &[slot as i32])?);
    ar.insert(LIVE, put_i32(session, &[1])?);
    ar.insert(SINK, put_bytes(session, &base.sink)?);
    ar.insert(ANGLES, session.upload_f32(&angles)?);
    let width = qh * vd;
    let mut ar_rows = Vec::new();
    for token in 0..len {
        ar.insert(
            QKV,
            session.upload_bf16(&qkv[token * row_width..(token + 1) * row_width])?,
        );
        ar.insert(SEQ, put_i32(session, &[(prefix + token + 1) as i32])?);
        let out = session.allocate(width * 2)?;
        poison::poison(session, &[&out])?;
        ar.insert(OUT, out);
        let arguments = Buffers {
            q: QKV,
            q_start: None,
            q_len: None,
            prefix_len: None,
            k_new: None,
            v_new: None,
            slot_ids: ring.then_some(SLOT),
            ..base.arguments.clone()
        };
        let folded = v
            .plan_folded_table(
                1,
                counts(&items, 1),
                arguments,
                RopeInputs {
                    qkv: QKV,
                    theta: theta as f32,
                    v_multiplier: V_MULTIPLIER,
                },
                ANGLES,
            )
            .map_err(|e| format!("{label}: {e}"))?;
        run(session, &cubin, &folded, &mut ar)?;
        ar_rows.extend(session.download_bf16(&ar[&OUT], width)?);
    }

    // Verify: the verify phase's row RoPE, then the exact rows.
    let mut verify = BTreeMap::new();
    for (id, bytes) in [(K, &base.k_cache), (V, &base.v_cache)] {
        verify.insert(id, put_bytes(session, bytes)?);
    }
    verify.insert(QKV_ALL, session.upload_bf16(&qkv)?);
    verify.insert(Q, session.allocate(len * qh * qk * 2)?);
    verify.insert(K_NEW, session.allocate(len * kh * qk * 2)?);
    verify.insert(V_NEW, session.allocate(len * kh * vd * 2)?);
    verify.insert(TABLE, put_i32(session, &base.tables)?);
    verify.insert(SLOT, put_i32(session, &[slot as i32])?);
    verify.insert(TOKEN_BATCH, put_i32(session, &vec![0; len])?);
    verify.insert(
        POSITIONS,
        put_i32(
            session,
            &(prefix..context).map(|p| p as i32).collect::<Vec<_>>(),
        )?,
    );
    verify.insert(
        LIVE_TOKENS,
        put_bytes(session, &(len as u32).to_ne_bytes())?,
    );
    verify.insert(ANGLES, session.upload_f32(&angles)?);
    for id in [SEQ, START, LEN, PREFIX, LIVE, SINK] {
        verify.insert(
            id,
            put_bytes(session, &bytes(session, &base.buffers[&id])?)?,
        );
    }
    let out = session.allocate(EXACT_ROWS * width * 2)?;
    poison::poison(session, &[&out])?;
    verify.insert(OUT, out);
    let (tag, suffix) = match v.name {
        "mimo_ga_bf16_decode" => ("ga_bf16", "p64"),
        "mimo_ga_fp8_decode" => ("ga_fp8", "p64"),
        "mimo_swa_decode" => ("swa_prefill", "p32"),
        other => return Err(format!("{other}: no verify row RoPE")),
    };
    let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == format!("chop_row_rope_mimo_{tag}_table_{suffix}_pos32_v1"))
        .ok_or("verify row RoPE entry missing")?;
    let bucket = len.next_power_of_two().max(2) as u32;
    let row_plan = entry
        .plan(
            bucket,
            RuntimeCounts {
                m: len as u32,
                batch: 1,
                context: context as u32,
                routed_rows: 0,
                split_k: 1,
            },
        )
        .map_err(|e| e.to_string())?;
    let row_args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(QKV_ALL),
            Argument::DeviceBuffer(Q),
            Argument::DeviceBuffer(K_NEW),
            Argument::DeviceBuffer(V_NEW),
            Argument::DeviceBuffer(K),
            Argument::DeviceBuffer(V),
            if ring {
                Argument::NullDevicePointer
            } else {
                Argument::DeviceBuffer(TABLE)
            },
            if ring {
                Argument::DeviceBuffer(SLOT)
            } else {
                Argument::NullDevicePointer
            },
            Argument::DeviceBuffer(TOKEN_BATCH),
            Argument::DeviceBuffer(POSITIONS),
            Argument::DeviceBuffer(ANGLES),
            Argument::DeviceBuffer(LIVE_TOKENS),
            Argument::U32(step_max_pages(v)),
            Argument::U32(if ring { v.query.page_size } else { 0 }),
            Argument::U64(base.arguments.k_page_stride_bytes),
            Argument::U64(base.arguments.v_page_stride_bytes),
            Argument::U32((theta as f32).to_bits()),
            Argument::U32(V_MULTIPLIER.to_bits()),
            Argument::U32(base.arguments.k_scale.to_bits()),
            Argument::U32(base.arguments.v_scale.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    let planned = plan_exact(
        v,
        shared,
        bucket,
        counts(&items, len),
        base.arguments.clone(),
    )
    .map_err(|e| format!("{label}: {e}"))?;
    if !row_plan.pdl || !planned.partial.pdl {
        return Err(format!(
            "{label}: the row RoPE and attention must launch with PDL"
        ));
    }
    prepare(session, &cubin, &planned, &mut verify)?;
    let producer = PdlProducer::new(session)?;
    let function = rows.function(&entry.symbol).map_err(|e| e.to_string())?;
    let row_rope = || {
        let bindings = Bindings {
            buffers: verify.iter().map(|(&id, buffer)| (id, buffer)).collect(),
        };
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            row_plan,
            &row_args,
            &bindings,
        )
    };
    // One graph, as serving replays: nothing between the producer, the row
    // RoPE and attention, so each starts while the one before it runs.
    session
        .capture(|_| {
            producer.launch_for(session, PRODUCER_CYCLES)?;
            row_rope()?;
            enqueue(session, &cubin, &planned, &verify)
        })?
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    // The check is void unless the kernel after the row RoPE starts before
    // the producer ends: the same chain with a stamping probe in place of
    // attention (the pages already hold the step's rows, so this reads
    // nothing stale).
    session
        .capture(|_| {
            producer.launch_for(session, PRODUCER_CYCLES)?;
            row_rope()?;
            producer.launch_second(session)
        })?
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let [_, producer_end, second_start, _] = producer.stamps(session)?;
    if second_start >= producer_end {
        return Err(format!(
            "{label}: the kernel after the row RoPE started {} ns after the producer ended; PDL did not overlap",
            second_start - producer_end
        ));
    }
    let verify_rows = session.download_bf16(&verify[&OUT], len * width)?;
    if let Some(at) = verify_rows.iter().zip(&ar_rows).position(|(a, b)| a != b) {
        return Err(format!(
            "{label}: output row {} element {} is {:#06x}, AR {:#06x}",
            at / width,
            at % width,
            verify_rows[at],
            ar_rows[at]
        ));
    }
    // The K and V the step leaves: GA pages the row engine wrote, and for the
    // ring the k_new/v_new rows the ring write copies after attention.
    let ar_k = bytes(session, &ar[&K])?;
    let ar_v = bytes(session, &ar[&V])?;
    let (verify_k, verify_v) = if ring {
        (
            bytes(session, &verify[&K_NEW])?,
            bytes(session, &verify[&V_NEW])?,
        )
    } else {
        (bytes(session, &verify[&K])?, bytes(session, &verify[&V])?)
    };
    for position in prefix..context {
        let token = position - prefix;
        for (name, ar_cache, verify_cache, stride, row) in [
            (
                "K",
                &ar_k,
                &verify_k,
                base.arguments.k_page_stride_bytes,
                k_row,
            ),
            (
                "V",
                &ar_v,
                &verify_v,
                base.arguments.v_page_stride_bytes,
                v_row,
            ),
        ] {
            let at = cache_offsets(position, stride, row);
            let expected = &ar_cache[at..at + row];
            let actual = if ring {
                &verify_cache[token * row..(token + 1) * row]
            } else {
                &verify_cache[at..at + row]
            };
            if expected != actual {
                return Err(format!(
                    "{label}: {name} of position {position} differs from AR"
                ));
            }
        }
    }
    Ok(())
}

fn step_max_pages(v: Variant) -> u32 {
    v.query.max_context_capacity.div_ceil(v.query.page_size)
}

#[test]
fn attention_exact_rows_pipeline_equals_ar() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let mut checked = 0;
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_swa_decode",
        ] {
            let v = named(name)?;
            for &shared in entries(v, EXACT_ROWS) {
                for prefix in [0, 5, 120, 127, 128, 200, 1017, 4095, 33_000] {
                    for len in [2, 4, 8] {
                        pipeline_case(session, v, shared, prefix, len)?;
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("exact rows pipeline: {checked} steps bitwise equal to AR");
        Ok(())
    })
    .expect("MiMo exact rows pipeline against AR");
}

/// Cost of AR GA decode with key ranges of a fixed unit count, for the
/// design of a row-count-invariant split: the one-row decode step at each
/// context with today's plan, and with the plan's CTA target set so each
/// CTA owns about `chunk` units (the kernel's `split_target`; plans of at
/// least `(units - 2 + 512) / 511` units per CTA are realizable). With
/// `CHOPINFER_KERNEL_BENCH=1`.
#[test]
fn attention_chunked_decode_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("chunked decode bench: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs")
            .unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let cubin = module(session, model_mimo26_contract::sm120::config::PROFILE)?;
        for name in ["mimo_ga_bf16_decode", "mimo_ga_fp8_decode"] {
            let v = named(name)?;
            for context in [1024_usize, 8192, 32_768, 131_072] {
                let items = [Item {
                    prefix: context - 1,
                    len: 1,
                }];
                let mut step = step(session, v, &items, 7)?;
                let today = v
                    .plan_for_bucket(1, counts(&items, 1), step.arguments.clone())
                    .map_err(|e| e.to_string())?;
                let target = |plan: &attention::PlannedAttention| plan.partial.grid[0];
                let (partial_index, merge_index) = (17, 9);
                if today.partial_args.0[partial_index] != Argument::U32(target(&today))
                    || today.merge_args.0[merge_index] != Argument::U32(target(&today))
                {
                    return Err("split_target is not at the expected argument".into());
                }
                let units = (context - 1) / 32 + 1;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("attention_chunked_decode_bench"));
                receipt.insert("variant".into(), json!(v.name));
                receipt.insert("context".into(), json!(context));
                receipt.insert("units".into(), json!(units));
                receipt.insert(
                    "today_us".into(),
                    json!(time_planned(session, &cubin, &today, &mut step.buffers)?),
                );
                receipt.insert("today_ctas".into(), json!(target(&today)));
                for chunk in [1_usize, 2, 4, 8, 16, 32] {
                    let ctas = units.div_ceil(chunk).min(target(&today) as usize) as u32;
                    let mut planned = today.clone();
                    planned.partial.grid[0] = ctas;
                    planned.partial_args.0[partial_index] = Argument::U32(ctas);
                    planned.merge_args.0[merge_index] = Argument::U32(ctas);
                    receipt.insert(
                        format!("chunk{chunk}_us"),
                        json!(time_planned(session, &cubin, &planned, &mut step.buffers)?),
                    );
                    receipt.insert(format!("chunk{chunk}_ctas"), json!(ctas));
                }
                eprintln!(
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                );
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("MiMo chunked decode bench");
}
