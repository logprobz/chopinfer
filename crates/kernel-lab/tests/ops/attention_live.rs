//! One captured graph per bucket replayed at several live counts, as serving
//! replays one graph per bucket with the live count read on device: the
//! paged attention variants, the decode entries that fold RoPE or the gated
//! norm and M-RoPE in, and the Qwen3.8 norm, RoPE and KV-write rows. Every
//! live count gets new rows, contexts, page order, queries and caches; items
//! past the live count get valid rows, tables and lengths too. Outputs and
//! the cache rows a step writes start poisoned, only the live items' rows
//! may be written, and workspaces are poisoned scratch that only the live
//! items' rows may change. The eager launch at each live count meets its
//! oracle and the graph replay matches it bit for bit. A poisoned check fails
//! at once; oracle misses are collected so every live count still replays,
//! and fail the test at the end.

use super::*;
use kernel_lab::poison::{live_counts, write};
use kernels_sm120::attention::GatedRopeInputs;

/// A paged live-count bucket.
#[derive(Clone, Copy)]
struct Shape {
    /// Items the bucket prepares: decode rows, verify items or prefill
    /// chunks.
    items: usize,
    /// Query rows the step buffers hold.
    tokens: usize,
    /// Most query rows of one item.
    rows: usize,
    /// The largest context of the replays, which the plan covers.
    context: usize,
}

fn shape(phase: AttentionPhase) -> Shape {
    match phase {
        AttentionPhase::Decode => Shape {
            items: 32,
            tokens: 32,
            rows: 1,
            context: 4096,
        },
        AttentionPhase::Verify => Shape {
            items: 32,
            tokens: 256,
            rows: 8,
            context: 2056,
        },
        AttentionPhase::Prefill => Shape {
            items: 8,
            tokens: 1024,
            rows: 128,
            context: 2048,
        },
    }
}

fn catalog_variant(name: &str) -> LabResult<Variant> {
    attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|variant| variant.name == name)
        .ok_or_else(|| format!("missing attention variant {name}"))
}

/// The paged MiMo global and Qwen3.8 variants, BF16 and FP8 KV: decode at a
/// 32-row bucket, verify at 32 items of up to 8 rows and prefill at 8 chunks
/// in a 1024-token capacity, each planned and captured once at the largest
/// context of its replays, against the f64 attention oracle.
#[test]
fn paged_attention_replays_one_graph_at_every_live_count() {
    gpu_test(3 << 30, |session| {
        let mut misses = Vec::new();
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_ga_bf16_verify",
            "mimo_ga_fp8_verify",
            "mimo_ga_bf16_prefill",
            "mimo_ga_fp8_prefill",
            "qwen38_bf16_decode",
            "qwen38_fp8_decode",
            "qwen38_bf16_verify",
            "qwen38_fp8_verify",
            "qwen38_bf16_prefill",
            "qwen38_fp8_prefill",
        ] {
            misses.extend(replay_paged(session, catalog_variant(name)?)?);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(format!("oracle misses:\n{}", misses.join("\n")))
        }
    })
    .expect("paged attention live-count graph replay");
}

/// A partial reading a stale live count (the bucket's) while the merge reads
/// the true count of 0 writes workspace rows no merge reads, and the output
/// stays untouched. Unbounded workspace scratch accepts it; the live-row
/// bound rejects it, and passes the same launch with the true count.
#[test]
fn stale_live_count_partial_fails_the_workspace_bound() {
    gpu_test(1 << 30, |session| {
        let case = Case {
            variant: named("mimo_ga_bf16_decode")?,
            batch: 4,
            rows: 1,
            prefix: 128,
            mixed: false,
        };
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare(session, case)?;
        if fixture.planned.direct_output {
            return Err("the negative control needs a partial and a merge".into());
        }
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        let stale = put_i32(session, &[case.batch as i32])?;
        write(session, &fixture.buffers[&LIVE], 0, &0_i32.to_ne_bytes())?;
        let out = &fixture.buffers[&OUT];
        let work = &fixture.buffers[&WORK];
        let bound = workspace_rows(&fixture.planned, &case.variant.query, 0, 0)?;
        let stale_launch =
            || fixture.launch_rebound(session, &partial, &merge, &[(LIVE, &stale)], &[]);
        Replay::new(session)
            .output("attention output", out, WriteMap::of(DType::Bf16).none()?)
            .scratch("attention workspace", work)
            .eager(stale_launch)
            .map_err(|e| format!("the unbounded check already fails: {e}"))?;
        let error = match Replay::new(session)
            .output("attention output", out, WriteMap::of(DType::Bf16).none()?)
            .scratch_within("attention workspace", work, bound.clone())
            .eager(stale_launch)
        {
            Ok(()) => return Err("the workspace bound passed a stale live count".into()),
            Err(error) => error,
        };
        eprintln!("stale live count failed the workspace bound as required: {error}");
        Replay::new(session)
            .output("attention output", out, WriteMap::of(DType::Bf16).none()?)
            .scratch_within("attention workspace", work, bound)
            .eager(|| fixture.launch(session, &partial, &merge))
            .map_err(|e| format!("the true live count fails the workspace bound: {e}"))?;
        let receipt = BTreeMap::from([
            ("case".into(), json!(case.label())),
            ("live".into(), json!(0)),
            ("stale_live".into(), json!(case.batch)),
            ("unbounded_check_passed".into(), json!(true)),
            ("bounded_check_failed".into(), json!(true)),
            ("bounded_error".into(), json!(error)),
            ("true_live_passed".into(), json!(true)),
        ]);
        session.receipt("attention_negative_control", receipt)?;
        Ok(())
    })
    .expect("stale live count negative control");
}

/// One live count's items: query rows, prefixes and physical pages.
struct Items {
    lens: Vec<usize>,
    prefixes: Vec<usize>,
    pages: Vec<Vec<usize>>,
}

impl Items {
    /// Every item of the bucket: `rows` or fewer query rows after a prefix
    /// that keeps the context within the shape's, over pages drawn without
    /// repeats from a shuffled pool.
    fn draw(shape: Shape, page: usize, pool_pages: usize, state: &mut u64) -> Self {
        let mut lens = Vec::with_capacity(shape.items);
        let mut prefixes = Vec::with_capacity(shape.items);
        for _ in 0..shape.items {
            let len = 1 + random_word(state) as usize % shape.rows;
            lens.push(len);
            prefixes.push(random_word(state) as usize % (shape.context - len + 1));
        }
        let mut physical: Vec<usize> = (0..pool_pages).collect();
        for i in (1..physical.len()).rev() {
            physical.swap(i, random_word(state) as usize % (i + 1));
        }
        let mut next = 0;
        let pages = lens
            .iter()
            .zip(&prefixes)
            .map(|(&len, &prefix)| {
                let count = (prefix + len).div_ceil(page);
                next += count;
                physical[next - count..next].to_vec()
            })
            .collect();
        Self {
            lens,
            prefixes,
            pages,
        }
    }

    fn starts(&self) -> Vec<usize> {
        self.lens
            .iter()
            .scan(0, |next, &len| {
                let start = *next;
                *next += len;
                Some(start)
            })
            .collect()
    }
}

/// Captures one bucket of `variant` once and replays it at every live count;
/// returns the oracle misses.
fn replay_paged(session: &Session, variant: Variant) -> LabResult<Vec<String>> {
    let phase = variant.query.phase;
    let decode = phase == AttentionPhase::Decode;
    let shape = shape(phase);
    let mut v = variant;
    if phase == AttentionPhase::Prefill {
        v.query_capacity = shape.tokens as u32;
    }
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let element = if fp8 { 1 } else { 2 };
    let pool_pages = shape.items * shape.context.div_ceil(page);
    let (k_page_bytes, v_page_bytes) = (page * kh * qd * element, page * kh * vd * element);
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let [k_scale, v_scale] = if fp8 { [0.5, 0.25] } else { [1.0, 1.0] };
    let mut buffers = BTreeMap::new();
    for (id, bytes) in [
        (Q, shape.tokens * qh * qd * 2),
        (K, pool_pages * k_page_bytes),
        (V, pool_pages * v_page_bytes),
        (TABLE, shape.items * max_pages * 4),
        (SEQ, shape.items * 4),
        (START, shape.items * 4),
        (LEN, shape.items * 4),
        (PREFIX, shape.items * 4),
        (LIVE, 4),
        (SINK, qh * 2),
        (OUT, shape.tokens * qh * vd * 2),
    ] {
        buffers.insert(id, session.allocate(bytes)?);
    }
    let (k_map, v_map) = if phase == AttentionPhase::Prefill {
        let (k, v) = v
            .cache_maps_with_strides(
                pool_pages as u32,
                K,
                V,
                k_page_bytes as u64,
                v_page_bytes as u64,
            )
            .map_err(|e| e.to_string())?;
        (Some(k), Some(v))
    } else {
        (None, None)
    };
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: k_page_bytes as u64,
        v_page_stride_bytes: v_page_bytes as u64,
        k_map,
        v_map,
        block_tables: Some(TABLE),
        slot_ids: None,
        seq_lens: Some(SEQ),
        q_start: (!decode).then_some(START),
        q_len: (!decode).then_some(LEN),
        prefix_len: (!decode).then_some(PREFIX),
        live_batch: LIVE,
        k_new: None,
        v_new: None,
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale,
        v_scale,
    };
    let bucket = shape.items as u32;
    let counts = RuntimeCounts {
        m: shape.tokens as u32,
        batch: shape.items as u32,
        context: shape.context as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let planned = v
        .plan_for_bucket(bucket, counts, arguments.clone())
        .map_err(|e| e.to_string())?;
    buffers.insert(
        WORK,
        session.allocate(planned.partial.workspace_bytes as usize)?,
    );
    let fixture = Fixture {
        buffers,
        arguments,
        planned,
        reference: oracle::OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        },
        sampled: None,
        output_len: shape.tokens * qh * vd,
        logical_bytes: 0,
        // Each live count draws its own pages.
        page_lists: Vec::new(),
        slot_ids: Vec::new(),
        variant: v,
        bucket,
        counts,
    };
    let cubin = module(session, v.profile)?;
    let (partial, merge) = functions(&cubin, v, &fixture.planned)?;
    let launch = || fixture.launch(session, &partial, &merge);
    let graph = session.capture(|_| launch())?;
    let sink_values: Vec<f32> = (0..qh).map(|h| (h as f32 - 8.0) / 16.0).collect();
    let (sink_bytes, sink_ref) = encode(&sink_values, false, 1.0);
    write(session, &fixture.buffers[&SINK], 0, &sink_bytes)?;
    let mut misses = Vec::new();
    for live in live_counts(shape.items as u32) {
        let live = live as usize;
        let mut state = 0x5eed_1ea5_0000_0001_u64 ^ ((live as u64) << 40);
        for byte in v.name.bytes() {
            state = state.rotate_left(7) ^ u64::from(byte);
            random_word(&mut state);
        }
        let items = Items::draw(shape, page, pool_pages, &mut state);
        let starts = items.starts();
        let (q_bytes, q_ref) = encode(&samples(shape.tokens * qh * qd, &mut state), false, 1.0);
        let (k_bytes, k_ref) = encode(
            &samples(pool_pages * page * kh * qd, &mut state),
            fp8,
            k_scale,
        );
        let (v_bytes, v_ref) = encode(
            &samples(pool_pages * page * kh * vd, &mut state),
            fp8,
            v_scale,
        );
        let mut tables = vec![0_i32; shape.items * max_pages];
        for (b, pages) in items.pages.iter().enumerate() {
            for (p, &physical) in pages.iter().enumerate() {
                tables[b * max_pages + p] = physical as i32;
            }
        }
        let key_lengths: Vec<usize> = items
            .prefixes
            .iter()
            .zip(&items.lens)
            .map(|(prefix, len)| prefix + len)
            .collect();
        let as_i32 = |values: &[usize]| {
            values
                .iter()
                .flat_map(|&value| (value as i32).to_ne_bytes())
                .collect::<Vec<u8>>()
        };
        for (id, bytes) in [
            (Q, q_bytes),
            (K, k_bytes),
            (V, v_bytes),
            (
                TABLE,
                tables
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect(),
            ),
            (SEQ, as_i32(&key_lengths)),
            (START, as_i32(&starts)),
            (LEN, as_i32(&items.lens)),
            (PREFIX, as_i32(&items.prefixes)),
            (LIVE, (live as u32).to_ne_bytes().to_vec()),
        ] {
            write(session, &fixture.buffers[&id], 0, &bytes)?;
        }
        let live_tokens: usize = items.lens[..live].iter().sum();
        let row = qh * vd;
        let label = format!("{} live {live} of {}", v.name, shape.items);
        let labelled = |error: String| format!("{label}: {error}");
        let mut checks = Replay::new(session)
            .output(
                "attention output",
                &fixture.buffers[&OUT],
                WriteMap::of(DType::Bf16).prefix(live_tokens * row)?,
            )
            .scratch_within(
                "attention workspace",
                &fixture.buffers[&WORK],
                workspace_rows(&fixture.planned, &v.query, shape.tokens, live_tokens)?,
            );
        checks.eager(launch).map_err(labelled)?;
        let mut receipt = BTreeMap::new();
        let oracle = if live > 0 {
            let reference = paged_oracle(
                &v.query,
                &q_ref[..live_tokens * qh * qd],
                (&k_ref, &v_ref),
                &items.pages[..live],
                &items.lens[..live],
                &items.prefixes[..live],
                v.query.sink.then_some(sink_ref.as_slice()),
            )?;
            let eager = checks
                .expected("attention output")
                .ok_or("missing eager output")?;
            let actual: Vec<f32> = eager[..live_tokens * row * 2]
                .chunks_exact(2)
                .map(|pair| bf16_to_f32(u16::from_ne_bytes([pair[0], pair[1]])))
                .collect();
            check_output(&actual, &reference, DType::Bf16, &mut receipt).map_err(|error| {
                format!(
                    "{label}: {error}; worst element {:?}",
                    receipt.get("worst_element")
                )
            })
        } else {
            Ok(())
        };
        checks.graph(&graph).map_err(labelled)?;
        receipt.insert("family".into(), json!("paged_attention"));
        receipt.insert("variant".into(), json!(v.name));
        receipt.insert("profile".into(), json!(v.profile.as_str()));
        receipt.insert("phase".into(), json!(format!("{phase:?}").to_lowercase()));
        receipt.insert("bucket_items".into(), json!(shape.items));
        receipt.insert("token_capacity".into(), json!(shape.tokens));
        receipt.insert("planned_context".into(), json!(shape.context));
        receipt.insert("live".into(), json!(live));
        receipt.insert("live_tokens".into(), json!(live_tokens));
        receipt.insert("contexts".into(), json!(key_lengths[..live]));
        receipt.insert(
            "split_capacity".into(),
            json!(fixture.planned.split_capacity),
        );
        receipt.insert("merge".into(), json!(!fixture.planned.direct_output));
        receipt.insert("correct".into(), json!(oracle.is_ok()));
        if let Err(miss) = &oracle {
            receipt.insert("oracle_miss".into(), json!(miss));
        }
        checks.record(&mut receipt);
        session.receipt("live_replay", receipt)?;
        eprintln!(
            "{label}: {live_tokens} rows, {} splits, {} and graph bitwise",
            fixture.planned.split_capacity,
            if oracle.is_ok() {
                "oracle"
            } else {
                "an oracle miss"
            }
        );
        misses.extend(oracle.err());
    }
    Ok(misses)
}

/// The workspace bytes a paged step over `live_tokens` query rows (decode
/// rows) may write: the paged prefill merge records, one per partial CTA and
/// KV head, which every CTA writes (prefill.cuh `prefill_record`), then the
/// partials of the live rows. Partials are packed by row, each row holding
/// `q_heads * split_capacity * (v_dim + 2)` F32 words (decode.cuh
/// `decode_partial`, verify.cuh `verify_partial`, prefill.cuh), so rows past
/// the live count stay poisoned and at live count 0 only the records change.
/// `tokens` is the prefill query capacity.
fn workspace_rows(
    planned: &attention::PlannedAttention,
    query: &kernel_core::contract::AttentionQuery,
    tokens: usize,
    live_tokens: usize,
) -> LabResult<Vec<std::ops::Range<usize>>> {
    let total = planned.partial.workspace_bytes as usize;
    let row =
        query.q_heads as usize * planned.split_capacity as usize * (query.v_dim as usize + 2) * 4;
    let records = if query.phase == AttentionPhase::Prefill {
        total
            .checked_sub(tokens * row)
            .ok_or("prefill workspace smaller than its partials")?
    } else {
        0
    };
    if (total - records) % row != 0 || records + live_tokens * row > total {
        return Err(format!(
            "workspace of {total} bytes does not hold {live_tokens} rows of {row} after {records}"
        ));
    }
    Ok(std::iter::once(0..records + live_tokens * row).collect())
}

/// The first `live` of a buffer's `items` equal rows.
fn live_prefix(
    buffer: &DeviceBuffer,
    items: usize,
    live: usize,
) -> LabResult<Vec<std::ops::Range<usize>>> {
    let bytes = buffer.byte_len();
    if items == 0 || bytes % items != 0 {
        return Err(format!("{bytes} bytes do not hold {items} equal rows"));
    }
    Ok(std::iter::once(0..bytes / items * live).collect())
}

/// The paged attention oracle, one item per thread.
fn paged_oracle(
    query: &kernel_core::contract::AttentionQuery,
    q: &[f64],
    (keys, values): (&[f64], &[f64]),
    tables: &[Vec<usize>],
    lens: &[usize],
    prefixes: &[usize],
    sink: Option<&[f64]>,
) -> LabResult<oracle::OracleOutput> {
    let row = (query.q_heads * query.qk_dim) as usize;
    let mut start = 0;
    let parts = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lens.len())
            .map(|b| {
                let q = &q[start * row..(start + lens[b]) * row];
                start += lens[b];
                scope.spawn(move || {
                    oracle::attention(
                        query,
                        &PagedAttention {
                            q,
                            keys,
                            values,
                            page_tables: &tables[b..b + 1],
                            query_lengths: &lens[b..b + 1],
                            query_starts: &prefixes[b..b + 1],
                            key_lengths: &[prefixes[b] + lens[b]],
                            sink_logits: sink,
                        },
                    )
                    .map_err(|e| e.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| "oracle thread panicked")?)
            .collect::<LabResult<Vec<_>>>()
    })?;
    let mut output = oracle::OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    for part in parts {
        output.values.extend(part.values);
        output.sum_abs_products.extend(part.sum_abs_products);
        output.reduction_lengths.extend(part.reduction_lengths);
    }
    Ok(output)
}

/// Decode rows of a folded bucket, and the largest context it plans for.
const FOLD_ITEMS: usize = 32;
const FOLD_CONTEXT: usize = 4096;

/// The decode entries that fold RoPE (MiMo) or the gated norm and M-RoPE
/// (Qwen3.8) with the KV write in, captured once at a 32-row bucket. At each
/// live count the reference, the row step followed by the unfused entries,
/// runs on the fixture's caches and the folded entry on copies, both from
/// the same starting bytes: the current row of every decode row starts
/// poisoned, the live rows' current rows must be written and the others must
/// keep their poison. The folded eager launch matches the reference bit for
/// bit, output and caches, and the graph replay matches it. The reference
/// output meets the attention oracle over the Q and caches its row step
/// wrote; the row steps have their own live-count tests against their
/// oracles (`row::live` and `qwen38_rope_kv_replays_one_graph_at_every_live_count`).
#[test]
fn folded_decode_replays_one_graph_at_every_live_count() {
    gpu_test(3 << 30, |session| {
        let rows = session
            .context
            .load_module(ROW_CUBIN)
            .map_err(|e| e.to_string())?;
        let mut misses = Vec::new();
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_swa_decode",
            "qwen38_bf16_decode",
            "qwen38_fp8_decode",
        ] {
            misses.extend(replay_folded(session, &rows, catalog_variant(name)?)?);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(format!("oracle misses:\n{}", misses.join("\n")))
        }
    })
    .expect("folded decode live-count graph replay");
}

/// A generator seeded by the case name and the live count.
fn seeded(name: &str, live: usize) -> u64 {
    let mut state = 0x5eed_1ea5_0000_0001_u64 ^ ((live as u64) << 40);
    for byte in name.bytes() {
        state = state.rotate_left(7) ^ u64::from(byte);
        random_word(&mut state);
    }
    state
}

fn bytes_i32(values: impl IntoIterator<Item = usize>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| (value as i32).to_ne_bytes())
        .collect()
}

fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// Cache bytes as the values the kernels read.
fn decode_cache(bytes: &[u8], fp8: bool, scale: f32) -> Vec<f64> {
    if fp8 {
        bytes
            .iter()
            .map(|&code| kv_fp8::decode(code, scale))
            .collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|pair| f64::from(bf16_to_f32(u16::from_ne_bytes([pair[0], pair[1]]))))
            .collect()
    }
}

/// The eager bytes of `name` in two poisoned checks agree.
fn same_bytes(label: &str, expected: &Replay, actual: &Replay, name: &str) -> LabResult<()> {
    let (Some(left), Some(right)) = (expected.expected(name), actual.expected(name)) else {
        return Err(format!("{label}: no eager {name}"));
    };
    match left.iter().zip(right).position(|(a, b)| a != b) {
        None if left.len() == right.len() => Ok(()),
        None => Err(format!("{label}: {name} sizes differ")),
        Some(offset) => Err(format!(
            "{label}: folded {name} differs from the reference at byte {offset}"
        )),
    }
}

/// Captures the folded decode of `v` once and replays it at every live
/// count; returns the oracle misses.
fn replay_folded(session: &Session, rows: &Module, v: Variant) -> LabResult<Vec<String>> {
    let gated = v.gated_fold.is_some();
    let ring = v.layout == CacheLayout::Ring;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let items = FOLD_ITEMS;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let element = cache.bytes()?;
    // A ring of `page` rows per slot, or every row's pages at the largest
    // context.
    let pool_pages = if ring {
        items
    } else {
        items * FOLD_CONTEXT.div_ceil(page)
    };
    let (k_row, v_row) = (kh * qd * element, kh * vd * element);
    let (k_page_bytes, v_page_bytes) = (page * k_row, page * v_row);
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let [k_scale, v_scale] = if fp8 { [0.5_f32, 0.25] } else { [1.0, 1.0] };
    let width = if gated {
        model_qwen38_contract::sm120::attention::ROW_WIDTH as usize
    } else {
        qh * qd + kh * (qd + vd)
    };
    let pairs = (model_qwen38_contract::sm120::attention::ROTARY / 2) as usize;
    let mut buffers = BTreeMap::new();
    for (id, bytes) in [
        (Q, items * qh * qd * 2),
        (K, pool_pages * k_page_bytes),
        (V, pool_pages * v_page_bytes),
        (TABLE, items * max_pages * 4),
        (SLOT, items * 4),
        (SEQ, items * 4),
        (LIVE, 4),
        (SINK, qh * 2),
        (OUT, items * qh * vd * 2),
        (QKV, items * width * 2),
        (POSITIONS, items * 4),
        (TOKEN_BATCH, items * 4),
        (K_NEW, items * kh * qd * 2),
        (V_NEW, items * kh * vd * 2),
        (qwen38::Q_NORM, qd * 2),
        (qwen38::K_NORM, qd * 2),
        (qwen38::ROPE_POSITIONS, items * 12),
        (qwen38::COS_SIN, FOLD_CONTEXT * pairs * 8),
    ] {
        buffers.insert(id, session.allocate(bytes)?);
    }
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: k_page_bytes as u64,
        v_page_stride_bytes: v_page_bytes as u64,
        k_map: None,
        v_map: None,
        block_tables: (!ring).then_some(TABLE),
        slot_ids: ring.then_some(SLOT),
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
        m: items as u32,
        batch: items as u32,
        context: FOLD_CONTEXT as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let unfused = v
        .plan_for_bucket(items as u32, counts, arguments.clone())
        .map_err(|e| e.to_string())?;
    let folded = if gated {
        v.plan_folded_gated(
            items as u32,
            counts,
            arguments.clone(),
            GatedRopeInputs {
                qkv: QKV,
                q_norm: qwen38::Q_NORM,
                k_norm: qwen38::K_NORM,
                rope_positions: qwen38::ROPE_POSITIONS,
                cos_sin: qwen38::COS_SIN,
            },
        )
    } else {
        v.plan_folded(
            items as u32,
            counts,
            arguments.clone(),
            RopeInputs {
                qkv: QKV,
                theta: rope_theta(v),
                v_multiplier: V_MULTIPLIER,
            },
        )
    }
    .map_err(|e| e.to_string())?;
    if !folded.writes_cache || folded.partial_symbol != v.rope_symbol() {
        return Err(format!(
            "{}: the folded plan does not write the cache",
            v.name
        ));
    }
    buffers.insert(
        WORK,
        session.allocate(
            unfused
                .partial
                .workspace_bytes
                .max(folded.partial.workspace_bytes) as usize,
        )?,
    );
    let fold_k = session.allocate(pool_pages * k_page_bytes)?;
    let fold_v = session.allocate(pool_pages * v_page_bytes)?;
    let cubin = module(session, v.profile)?;
    let fixture = Fixture {
        buffers,
        arguments,
        planned: unfused,
        reference: oracle::OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        },
        sampled: None,
        output_len: items * qh * vd,
        logical_bytes: 0,
        // Each live count draws its own pages and slots.
        page_lists: Vec::new(),
        slot_ids: Vec::new(),
        variant: v,
        bucket: items as u32,
        counts,
    };
    let (partial, merge) = functions(&cubin, v, &fixture.planned)?;
    let folded_function = cubin
        .function(&folded.partial_symbol)
        .map_err(|e| e.to_string())?;
    if folded.partial.dynamic_shared_bytes > 48 * 1024 {
        folded_function
            .set_max_dynamic_shared_memory(folded.partial.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    // The row step the unfused entries follow, as `prepare_fold` and
    // `prepare_gated_fold` bind it.
    let (row, row_symbol, row_plan, row_args) = if gated {
        let entry = qwen38::row_entry(&format!(
            "chop_attn_qwen38_qk_norm_rope_kv_{}_v1",
            if fp8 { "fp8" } else { "bf16" }
        ))?;
        let plan = entry.plan(items as u32).map_err(|e| e.to_string())?;
        let args = entry
            .pack_arguments(vec![
                Argument::DeviceBuffer(QKV),
                Argument::DeviceBuffer(qwen38::Q_NORM),
                Argument::DeviceBuffer(qwen38::K_NORM),
                Argument::DeviceBuffer(qwen38::ROPE_POSITIONS),
                Argument::DeviceBuffer(qwen38::COS_SIN),
                Argument::DeviceBuffer(TOKEN_BATCH),
                Argument::DeviceBuffer(POSITIONS),
                Argument::DeviceBuffer(LIVE),
                Argument::DeviceBuffer(TABLE),
                Argument::DeviceBuffer(K),
                Argument::DeviceBuffer(V),
                Argument::DeviceBuffer(Q),
                Argument::U32(max_pages as u32),
                Argument::U64(k_page_bytes as u64),
                Argument::U64(v_page_bytes as u64),
                Argument::U32(model_qwen38_contract::sm120::attention::EPS.to_bits()),
                Argument::U32(k_scale.to_bits()),
                Argument::U32(v_scale.to_bits()),
            ])
            .map_err(|e| e.to_string())?;
        let function = cubin.function(&entry.symbol).map_err(|e| e.to_string())?;
        (function, entry.symbol, plan, args)
    } else {
        let row_name = match v.name {
            "mimo_ga_bf16_decode" => "ga_bf16_computed_p64_pos32",
            "mimo_ga_fp8_decode" => "ga_fp8_computed_p64_pos32",
            "mimo_swa_decode" => "swa_decode_computed_p32_pos32",
            _ => return Err(format!("{} folds no RoPE", v.name)),
        };
        let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| entry.symbol == format!("chop_row_rope_mimo_{row_name}_v1"))
            .ok_or("row RoPE entry missing")?;
        let plan = entry
            .plan(items as u32, counts)
            .map_err(|e| e.to_string())?;
        let args = entry
            .pack_arguments(vec![
                Argument::DeviceBuffer(QKV),
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
                Argument::NullDevicePointer,
                Argument::DeviceBuffer(LIVE),
                Argument::U32(1_048_576_u32.div_ceil(v.query.page_size)),
                Argument::U32(if ring { v.query.page_size } else { 0 }),
                Argument::U64(k_page_bytes as u64),
                Argument::U64(v_page_bytes as u64),
                Argument::U32(rope_theta(v).to_bits()),
                Argument::U32(V_MULTIPLIER.to_bits()),
                Argument::U32(k_scale.to_bits()),
                Argument::U32(v_scale.to_bits()),
            ])
            .map_err(|e| e.to_string())?;
        let function = rows.function(&entry.symbol).map_err(|e| e.to_string())?;
        (function, entry.symbol, plan, args)
    };
    let unfused_launch = || -> LabResult<()> {
        launch::launch(
            &session.context,
            &row,
            &session.stream,
            row_plan,
            &row_args,
            &fixture.bindings(),
        )?;
        fixture.launch(session, &partial, &merge)
    };
    let folded_launch = || {
        fixture.launch_planned(
            session,
            &folded,
            &folded_function,
            &merge,
            Some((&fold_k, &fold_v)),
        )
    };
    let graph = session.capture(|_| folded_launch())?;
    // Inputs every live count shares: the sinks, the norm weights and the
    // cos/sin table, and one decode row per batch entry.
    let mut state = seeded(v.name, usize::MAX);
    let sink_values: Vec<f32> = (0..qh).map(|h| (h as f32 - 8.0) / 16.0).collect();
    let (sink_bytes, sink_ref) = encode(&sink_values, false, 1.0);
    for (id, bytes) in [
        (SINK, sink_bytes),
        (TOKEN_BATCH, bytes_i32(0..items)),
        (
            qwen38::Q_NORM,
            bytes_u16(&qwen38::grid_bf16(qd, 0.5, &mut state)),
        ),
        (
            qwen38::K_NORM,
            bytes_u16(&qwen38::grid_bf16(qd, 0.5, &mut state)),
        ),
        (
            qwen38::COS_SIN,
            qwen38::cos_sin_table(FOLD_CONTEXT)
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        ),
    ] {
        write(session, &fixture.buffers[&id], 0, &bytes)?;
    }
    let mut misses = Vec::new();
    for live in live_counts(items as u32) {
        let live = live as usize;
        let mut state = seeded(v.name, live);
        let contexts: Vec<usize> = (0..items)
            .map(|_| 1 + random_word(&mut state) as usize % FOLD_CONTEXT)
            .collect();
        let positions: Vec<usize> = contexts.iter().map(|&context| context - 1).collect();
        let mut physical: Vec<usize> = (0..pool_pages).collect();
        for i in (1..physical.len()).rev() {
            physical.swap(i, random_word(&mut state) as usize % (i + 1));
        }
        // Ring rows use the shuffled pool as slots; paged rows take their
        // pages from it in turn.
        let mut next = 0;
        let page_lists: Vec<Vec<usize>> = contexts
            .iter()
            .map(|&context| {
                let count = if ring { 0 } else { context.div_ceil(page) };
                next += count;
                physical[next - count..next].to_vec()
            })
            .collect();
        let slots = physical[..items].to_vec();
        // The cache row each row's current token writes.
        let current: Vec<usize> = (0..items)
            .map(|b| {
                let position = positions[b];
                if ring {
                    slots[b] * page + position % page
                } else {
                    page_lists[b][position / page] * page + position % page
                }
            })
            .collect();
        let (mut k_image, _) = encode(
            &samples(pool_pages * page * kh * qd, &mut state),
            fp8,
            k_scale,
        );
        let (mut v_image, _) = encode(
            &samples(pool_pages * page * kh * vd, &mut state),
            fp8,
            v_scale,
        );
        for &row in &current {
            k_image[row * k_row..(row + 1) * k_row].fill(POISON);
            v_image[row * v_row..(row + 1) * v_row].fill(POISON);
        }
        let qkv = qwen38::grid_bf16(items * width, 4.0, &mut state);
        let mut tables = vec![0_i32; items * max_pages];
        for (b, pages) in page_lists.iter().enumerate() {
            for (p, &physical) in pages.iter().enumerate() {
                tables[b * max_pages + p] = physical as i32;
            }
        }
        let rope: Vec<usize> = positions
            .iter()
            .enumerate()
            .flat_map(|(b, &position)| qwen38::rope_position(position, b))
            .collect();
        for (id, bytes) in [
            (QKV, bytes_u16(&qkv)),
            (POSITIONS, bytes_i32(positions.iter().copied())),
            (SEQ, bytes_i32(contexts.iter().copied())),
            (
                TABLE,
                tables
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect(),
            ),
            (SLOT, bytes_i32(slots.iter().copied())),
            (qwen38::ROPE_POSITIONS, bytes_i32(rope)),
            (LIVE, (live as u32).to_ne_bytes().to_vec()),
        ] {
            write(session, &fixture.buffers[&id], 0, &bytes)?;
        }
        let label = format!("{} folded live {live} of {items}", v.name);
        let labelled = |error: String| format!("{label}: {error}");
        let written = |row_bytes: usize| {
            cache.byte_ranges(
                current[..live]
                    .iter()
                    .map(|&row| row * row_bytes..(row + 1) * row_bytes),
            )
        };
        let output = || WriteMap::of(DType::Bf16).prefix(live * qh * vd);
        let within = |id: u64| live_prefix(&fixture.buffers[&id], items, live);
        let mut reference = Replay::new(session)
            .output("attention output", &fixture.buffers[&OUT], output()?)
            .scratch_within(
                "attention workspace",
                &fixture.buffers[&WORK],
                workspace_rows(&fixture.planned, &v.query, 0, live)?,
            )
            .scratch_within("row Q", &fixture.buffers[&Q], within(Q)?)
            .scratch_within("row K", &fixture.buffers[&K_NEW], within(K_NEW)?)
            .scratch_within("row V", &fixture.buffers[&V_NEW], within(V_NEW)?)
            .state(
                "K cache",
                &fixture.buffers[&K],
                k_image.clone(),
                written(k_row)?,
                Vec::new(),
            )
            .state(
                "V cache",
                &fixture.buffers[&V],
                v_image.clone(),
                written(v_row)?,
                Vec::new(),
            );
        reference
            .eager(unfused_launch)
            .map_err(|e| labelled(format!("reference: {e}")))?;
        let row_q = poison::read(session, &fixture.buffers[&Q])?;
        let mut checks = Replay::new(session)
            .output("attention output", &fixture.buffers[&OUT], output()?)
            .scratch_within(
                "attention workspace",
                &fixture.buffers[&WORK],
                workspace_rows(&folded, &v.query, 0, live)?,
            )
            .state("K cache", &fold_k, k_image, written(k_row)?, Vec::new())
            .state("V cache", &fold_v, v_image, written(v_row)?, Vec::new());
        checks.eager(folded_launch).map_err(labelled)?;
        for name in ["attention output", "K cache", "V cache"] {
            same_bytes(&label, &reference, &checks, name)?;
        }
        checks.graph(&graph).map_err(labelled)?;
        let mut receipt = BTreeMap::new();
        let oracle = if live > 0 {
            let keys = decode_cache(
                reference.expected("K cache").ok_or("missing eager K")?,
                fp8,
                k_scale,
            );
            let values = decode_cache(
                reference.expected("V cache").ok_or("missing eager V")?,
                fp8,
                v_scale,
            );
            let q = decode_cache(&row_q[..live * qh * qd * 2], false, 1.0);
            let expected = if ring {
                oracle::ring_attention(
                    &v.query,
                    &RingAttention {
                        q: &q,
                        keys: &keys,
                        values: &values,
                        slot_ids: &slots[..live],
                        query_lengths: &vec![1; live],
                        prefix_lengths: &positions[..live],
                        k_new: None,
                        v_new: None,
                        sink_logits: v.query.sink.then_some(sink_ref.as_slice()),
                    },
                )
                .map_err(|e| e.to_string())?
            } else {
                paged_oracle(
                    &v.query,
                    &q,
                    (&keys, &values),
                    &page_lists[..live],
                    &vec![1; live],
                    &positions[..live],
                    v.query.sink.then_some(sink_ref.as_slice()),
                )?
            };
            let actual: Vec<f32> = decode_cache(
                &reference
                    .expected("attention output")
                    .ok_or("missing eager output")?[..live * qh * vd * 2],
                false,
                1.0,
            )
            .into_iter()
            .map(|value| value as f32)
            .collect();
            check_output(&actual, &expected, DType::Bf16, &mut receipt).map_err(|error| {
                format!(
                    "{label}: {error}; worst element {:?}",
                    receipt.get("worst_element")
                )
            })
        } else {
            Ok(())
        };
        receipt.insert("family".into(), json!("folded_decode"));
        receipt.insert("variant".into(), json!(v.name));
        receipt.insert("folded_symbol".into(), json!(folded.partial_symbol));
        receipt.insert("unfused_row_symbol".into(), json!(row_symbol));
        receipt.insert("bucket_items".into(), json!(items));
        receipt.insert("planned_context".into(), json!(FOLD_CONTEXT));
        receipt.insert("live".into(), json!(live));
        receipt.insert("contexts".into(), json!(contexts[..live]));
        receipt.insert(
            "split_capacity".into(),
            json!(fixture.planned.split_capacity),
        );
        receipt.insert("output_bitwise_unfused".into(), json!(true));
        receipt.insert("cache_bitwise_unfused".into(), json!(true));
        receipt.insert("correct".into(), json!(oracle.is_ok()));
        if let Err(miss) = &oracle {
            receipt.insert("oracle_miss".into(), json!(miss));
        }
        checks.record(&mut receipt);
        session.receipt("live_replay", receipt)?;
        eprintln!(
            "{label}: matches the unfused path, {} and graph bitwise",
            if oracle.is_ok() {
                "oracle"
            } else {
                "an oracle miss"
            }
        );
        misses.extend(oracle.err());
    }
    Ok(misses)
}

/// Rows of the Qwen3.8 norm, RoPE and KV-write bucket, and the largest
/// context of its sequences.
const GATED_ROWS: usize = 32;
const GATED_CONTEXT: usize = 4096;

/// The Qwen3.8 norm, RoPE and KV-write rows of the selected topology
/// (`CHOPINFER_QWEN38_RANKS`), BF16 and FP8, captured once at 32 rows. Each
/// live count gets new sequences of one to three rows over shuffled pages,
/// new rows, positions and weights. Q starts poisoned, and each row's cache
/// rows start poisoned: the live rows' must be written and the others must
/// keep their poison. The eager launch meets the gated-row oracles as
/// `run_rows` checks them (Q and BF16 K within their bounds, FP8 K within
/// half an E4M3 step, V exactly) and the graph replay matches it bit for
/// bit.
#[test]
fn qwen38_rope_kv_replays_one_graph_at_every_live_count() {
    gpu_test(1 << 30, |session| {
        let mut misses = Vec::new();
        for dtype in [DType::Bf16, DType::Fp8E4M3] {
            misses.extend(replay_gated_rows(session, dtype)?);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(format!("oracle misses:\n{}", misses.join("\n")))
        }
    })
    .expect("Qwen3.8 norm, RoPE and KV write live-count graph replay");
}

fn replay_gated_rows(session: &Session, dtype: DType) -> LabResult<Vec<String>> {
    let fp8 = dtype == DType::Fp8E4M3;
    let name = if fp8 { "fp8" } else { "bf16" };
    let topology = qwen38::topology()?;
    let entry = topology.entry(&format!("qk_norm_rope_kv_{name}"))?;
    let cubin = module(session, topology.profile)?;
    let function = cubin.function(&entry.symbol).map_err(|e| e.to_string())?;
    let (qh, kh, d) = (
        topology.q_heads,
        model_qwen38_contract::sm120::attention::KV_HEADS as usize,
        model_qwen38_contract::sm120::attention::HEAD_DIM as usize,
    );
    let (page, width, rows) = (
        model_qwen38_contract::sm120::attention::PAGE as usize,
        topology.width,
        GATED_ROWS,
    );
    let max_pages = (model_qwen38_contract::sm120::attention::MAX_CONTEXT as usize).div_ceil(page);
    let pool_pages = rows * GATED_CONTEXT.div_ceil(page);
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let element = cache.bytes()?;
    let row_bytes = kh * d * element;
    let page_bytes = page * row_bytes;
    let (k_scale, v_scale) = if fp8 { (0.5_f32, 0.25_f32) } else { (1.0, 1.0) };
    let mut buffers = BTreeMap::new();
    for (id, bytes) in [
        (QKV, rows * width * 2),
        (qwen38::Q_NORM, d * 2),
        (qwen38::K_NORM, d * 2),
        (qwen38::ROPE_POSITIONS, rows * 12),
        (
            qwen38::COS_SIN,
            GATED_CONTEXT * (model_qwen38_contract::sm120::attention::ROTARY as usize / 2) * 8,
        ),
        (TOKEN_BATCH, rows * 4),
        (POSITIONS, rows * 4),
        (LIVE, 4),
        (TABLE, rows * max_pages * 4),
        (K, pool_pages * page_bytes),
        (V, pool_pages * page_bytes),
        (Q, rows * qh * d * 2),
    ] {
        buffers.insert(id, session.allocate(bytes)?);
    }
    let table = qwen38::cos_sin_table(GATED_CONTEXT);
    write(
        session,
        &buffers[&qwen38::COS_SIN],
        0,
        &table
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(QKV),
            Argument::DeviceBuffer(qwen38::Q_NORM),
            Argument::DeviceBuffer(qwen38::K_NORM),
            Argument::DeviceBuffer(qwen38::ROPE_POSITIONS),
            Argument::DeviceBuffer(qwen38::COS_SIN),
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
            Argument::U32(model_qwen38_contract::sm120::attention::EPS.to_bits()),
            Argument::U32(k_scale.to_bits()),
            Argument::U32(v_scale.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    let plan = entry.plan(rows as u32).map_err(|e| e.to_string())?;
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
    let graph = session.capture(|_| launch_rows())?;
    let query = qwen38::rows_query(dtype, qh);
    let mut misses = Vec::new();
    for live in live_counts(rows as u32) {
        let live = live as usize;
        let mut state = seeded(&entry.symbol, live);
        // Sequences of one to three rows after random prefixes, covering the
        // bucket; the live count may end inside one.
        let mut sequences = Vec::new();
        let mut covered = 0;
        while covered < rows {
            let n = (1 + random_word(&mut state) as usize % 3).min(rows - covered);
            let prefix = random_word(&mut state) as usize % (GATED_CONTEXT - n + 1);
            sequences.push((prefix, n));
            covered += n;
        }
        let mut physical: Vec<usize> = (0..pool_pages).collect();
        for i in (1..physical.len()).rev() {
            physical.swap(i, random_word(&mut state) as usize % (i + 1));
        }
        let mut next = 0;
        let page_lists: Vec<Vec<usize>> = sequences
            .iter()
            .map(|&(prefix, n)| {
                let count = (prefix + n).div_ceil(page);
                next += count;
                physical[next - count..next].to_vec()
            })
            .collect();
        let mut tables = vec![0_i32; rows * max_pages];
        for (s, pages) in page_lists.iter().enumerate() {
            for (p, &physical) in pages.iter().enumerate() {
                tables[s * max_pages + p] = physical as i32;
            }
        }
        let (mut row_batch, mut positions, mut rope, mut targets) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (s, &(prefix, n)) in sequences.iter().enumerate() {
            for position in prefix..prefix + n {
                row_batch.push(s);
                positions.push(position);
                rope.push(qwen38::rope_position(position, row_batch.len()));
                targets.push(page_lists[s][position / page] * page + position % page);
            }
        }
        let qkv = qwen38::grid_bf16(rows * width, 4.0, &mut state);
        let q_norm = qwen38::grid_bf16(d, 0.5, &mut state);
        let k_norm = qwen38::grid_bf16(d, 0.5, &mut state);
        // Seeded cache bytes, with every row's step rows poisoned.
        let mut image: Vec<u8> = (0..pool_pages * page_bytes)
            .map(|_| random_word(&mut state) as u8 & 0x3f)
            .collect();
        for &target in &targets {
            image[target * row_bytes..(target + 1) * row_bytes].fill(POISON);
        }
        for (id, bytes) in [
            (QKV, bytes_u16(&qkv)),
            (qwen38::Q_NORM, bytes_u16(&q_norm)),
            (qwen38::K_NORM, bytes_u16(&k_norm)),
            (
                qwen38::ROPE_POSITIONS,
                bytes_i32(rope.iter().flatten().copied()),
            ),
            (TOKEN_BATCH, bytes_i32(row_batch.iter().copied())),
            (POSITIONS, bytes_i32(positions.iter().copied())),
            (
                TABLE,
                tables
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect(),
            ),
            (LIVE, (live as u32).to_ne_bytes().to_vec()),
        ] {
            write(session, &buffers[&id], 0, &bytes)?;
        }
        let label = format!("{} live {live} of {rows}", entry.symbol);
        let labelled = |error: String| format!("{label}: {error}");
        let written = || {
            cache.byte_ranges(
                targets[..live]
                    .iter()
                    .map(|&row| row * row_bytes..(row + 1) * row_bytes),
            )
        };
        let mut checks = Replay::new(session)
            .output(
                "Q",
                &buffers[&Q],
                WriteMap::of(DType::Bf16).prefix(live * qh * d)?,
            )
            .state(
                "K cache",
                &buffers[&K],
                image.clone(),
                written()?,
                Vec::new(),
            )
            .state("V cache", &buffers[&V], image, written()?, Vec::new());
        checks.eager(launch_rows).map_err(labelled)?;
        let mut receipt = BTreeMap::new();
        let oracle = if live > 0 {
            check_gated_rows(
                &checks,
                &mut receipt,
                GatedRows {
                    query,
                    qkv: &qkv,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    table: &table,
                    sequences: &sequences,
                    page_lists: &page_lists,
                    rope: &rope,
                    targets: &targets,
                    live,
                    scales: (k_scale, v_scale),
                },
            )
            .map_err(|error| format!("{label}: {error}"))
        } else {
            Ok(())
        };
        checks.graph(&graph).map_err(labelled)?;
        receipt.insert("family".into(), json!("qwen38_rope_kv"));
        receipt.insert("entry".into(), json!(entry.symbol));
        receipt.insert("bucket_rows".into(), json!(rows));
        receipt.insert("live".into(), json!(live));
        receipt.insert("sequences".into(), json!(sequences));
        receipt.insert("correct".into(), json!(oracle.is_ok()));
        if let Err(miss) = &oracle {
            receipt.insert("oracle_miss".into(), json!(miss));
        }
        topology.mark(&mut receipt);
        checks.record(&mut receipt);
        session.receipt("live_replay", receipt)?;
        eprintln!(
            "{label}: {} and graph bitwise",
            if oracle.is_ok() {
                "oracles"
            } else {
                "an oracle miss"
            }
        );
        misses.extend(oracle.err());
    }
    Ok(misses)
}

/// One live count of the gated rows, for the oracle check.
struct GatedRows<'a> {
    query: kernel_core::contract::GatedAttentionQuery,
    qkv: &'a [u16],
    q_norm: &'a [u16],
    k_norm: &'a [u16],
    table: &'a [f32],
    sequences: &'a [(usize, usize)],
    page_lists: &'a [Vec<usize>],
    rope: &'a [[usize; 3]],
    targets: &'a [usize],
    live: usize,
    scales: (f32, f32),
}

/// The eager rows below the live count against the gated-row oracle, as
/// `run_rows` checks them: Q within its bound, each written K row within its
/// bound (half an E4M3 step for FP8) and each V row exactly.
fn check_gated_rows(
    checks: &Replay,
    receipt: &mut BTreeMap<String, Value>,
    rows: GatedRows,
) -> LabResult<()> {
    use kernel_core::gated_attention_oracle::{
        GatedAttentionInputs, PagedKvPair, qk_norm_rope_kv_write,
    };
    let kernel_core::contract::GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads, kv_dtype, ..
    } = rows.query
    else {
        return Err("expected the gated norm and RoPE query".into());
    };
    let fp8 = kv_dtype == DType::Fp8E4M3;
    let (qh, kh, d) = (
        q_heads as usize,
        model_qwen38_contract::sm120::attention::KV_HEADS as usize,
        model_qwen38_contract::sm120::attention::HEAD_DIM as usize,
    );
    let width = rows.qkv.len() / rows.targets.len();
    let element = if fp8 { 1 } else { 2 };
    let row_bytes = kh * d * element;
    let page = model_qwen38_contract::sm120::attention::PAGE as usize;
    let pool_rows = rows
        .page_lists
        .iter()
        .flatten()
        .max()
        .map_or(0, |&last| (last + 1) * page);
    let tables = qwen38::oracle_tables(rows.table, rows.rope);
    let (mut q_expected, mut k_expected) = (
        oracle::OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        },
        Vec::new(),
    );
    let mut k_cache = vec![0.0; pool_rows * kh * d];
    let mut v_cache = vec![0.0; pool_rows * kh * d];
    let mut first = 0;
    for (s, &(prefix, n)) in rows.sequences.iter().enumerate() {
        let n = n.min(rows.live.saturating_sub(first));
        if n == 0 {
            break;
        }
        let rows_of = |offset: usize, len: usize| {
            (first..first + n)
                .flat_map(|r| qwen38::as_f64(&rows.qkv[r * width + offset..][..len]))
                .collect::<Vec<_>>()
        };
        let result = qk_norm_rope_kv_write(
            rows.query,
            GatedAttentionInputs {
                q_gate: &rows_of(0, qh * 2 * d),
                k: &rows_of(qh * 2 * d, kh * d),
                v: &rows_of(qh * 2 * d + kh * d, kh * d),
                q_norm: &qwen38::as_f64(rows.q_norm),
                k_norm: &qwen38::as_f64(rows.k_norm),
                rope_positions: &rows.rope[first..first + n],
                token_positions: &(prefix..prefix + n).collect::<Vec<_>>(),
                tables: Some(&tables[first..first + n]),
            },
            PagedKvPair {
                keys: &mut k_cache,
                values: &mut v_cache,
                block_table: &rows.page_lists[s],
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
    let q_bytes = checks.expected("Q").ok_or("missing eager Q")?;
    let q: Vec<f32> = q_bytes[..rows.live * qh * d * 2]
        .chunks_exact(2)
        .map(|pair| bf16_to_f32(u16::from_ne_bytes([pair[0], pair[1]])))
        .collect();
    check_output(&q, &q_expected, DType::Bf16, receipt)
        .map_err(|e| format!("Q: {e}; {:?}", receipt.get("worst_element")))?;
    let k_cache = checks.expected("K cache").ok_or("missing eager K cache")?;
    let v_cache = checks.expected("V cache").ok_or("missing eager V cache")?;
    let (k_scale, v_scale) = rows.scales;
    let mut row = 0;
    for (s, reference) in k_expected.iter().enumerate() {
        let n = reference.values.len() / (kh * d);
        for t in 0..n {
            let at = rows.targets[row] * row_bytes;
            let k_values = &reference.values[t * kh * d..(t + 1) * kh * d];
            let v_bits = &rows.qkv[row * width + qh * 2 * d + kh * d..][..kh * d];
            if fp8 {
                for i in 0..kh * d {
                    if !qwen38::e4m3_close(k_cache[at + i], k_values[i], f64::from(k_scale)) {
                        return Err(format!(
                            "K row {row} (sequence {s}) dim {i}: code {} for {}",
                            k_cache[at + i],
                            k_values[i]
                        ));
                    }
                    let code = kv_fp8::encode(bf16_to_f32(v_bits[i]), v_scale);
                    if v_cache[at + i] != code {
                        return Err(format!(
                            "V row {row} dim {i}: code {} instead of {code}",
                            v_cache[at + i]
                        ));
                    }
                }
            } else {
                let actual: Vec<f32> = k_cache[at..at + kh * d * 2]
                    .chunks_exact(2)
                    .map(|pair| bf16_to_f32(u16::from_ne_bytes([pair[0], pair[1]])))
                    .collect();
                let bound = oracle::OracleOutput {
                    values: k_values.to_vec(),
                    sum_abs_products: reference.sum_abs_products[t * kh * d..(t + 1) * kh * d]
                        .to_vec(),
                    reduction_lengths: reference.reduction_lengths[t * kh * d..(t + 1) * kh * d]
                        .to_vec(),
                };
                check_output(&actual, &bound, DType::Bf16, receipt)
                    .map_err(|e| format!("K row {row}: {e}"))?;
                if v_cache[at..at + kh * d * 2] != bytes_u16(v_bits)[..] {
                    return Err(format!("V row {row} differs from the packed V row"));
                }
            }
            row += 1;
        }
    }
    Ok(())
}
