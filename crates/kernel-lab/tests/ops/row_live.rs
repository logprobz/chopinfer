//! One captured graph per bucket replayed at several live token counts for
//! the MiMo computed RoPE and KV-write comparison entries (`rope_kv.cuh`): the
//! paged global-attention entries with BF16 and FP8 caches and the sliding-
//! window decode ring at a 32-token bucket, and the BF16 global entry at a
//! 1024-token bucket, whose grid holds four heads per CTA. Every live count
//! gets new rows, positions, page tables and ring slots. Q and the new K and
//! V rows start poisoned; each live token's cache row is written without
//! being read and starts poisoned, while the rows the tokens past the live
//! count would write start poisoned and must stay so. The eager launch at
//! each live count meets the HF F32-angle oracle and the graph replay matches
//! it bit for bit. A poisoned check fails at once; oracle misses are collected so
//! every live count still replays, and fail the test at the end. Serving uses
//! table-backed entries with the same HF angle and rotation arithmetic.

use super::*;
use kernel_lab::poison::{POISON, Replay, WriteMap, live_counts, write};

/// MiMo TP4 rotary geometry.
const Q_HEADS: usize = 16;
const QK: usize = 192;
const V: usize = 128;
const ROTARY: usize = 64;
const V_MULTIPLIER: f32 = 0.707;
const RING: usize = 128;
/// Pages each sequence may use; positions stay below `PAGES * page size`.
const PAGES: usize = 4;

#[test]
fn rope_kv_replays_one_graph_at_every_live_count() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let entries = catalog();
        let mut misses = Vec::new();
        for (name, bucket, sequences) in [
            ("rope_mimo_ga_bf16_computed_p64_pos32", 32, 32),
            ("rope_mimo_ga_fp8_computed_p64_pos32", 32, 32),
            ("rope_mimo_swa_decode_computed_p32_pos32", 32, 32),
            ("rope_mimo_ga_bf16_computed_p64_pos32", 1024, 8),
        ] {
            misses.extend(replay_entry(
                session,
                &module,
                find(&entries, name),
                name,
                bucket,
                sequences,
            )?);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(format!("oracle misses:\n{}", misses.join("\n")))
        }
    })
    .expect("RoPE and KV write live-count graph replay");
}

fn replay_entry(
    session: &Session,
    module: &Module,
    entry: &RowInstantiation,
    name: &str,
    bucket: usize,
    sequences: usize,
) -> LabResult<Vec<String>> {
    let ring = name.contains("swa");
    let fp8 = name.contains("fp8");
    let kvh = if ring { 2 } else { 1 };
    let page = if ring { 32 } else { 64 };
    let theta = if ring { 1e4_f32 } else { 1e7_f32 };
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let element = cache.bytes()?;
    let per_sequence = bucket / sequences;
    let stride = Q_HEADS * QK + kvh * (QK + V);
    let (k_row, v_row) = (kvh * QK * element, kvh * V * element);
    // Cache rows: a ring of RING rows per sequence slot, or PAGES pages of
    // `page` rows per sequence.
    let cache_rows = if ring {
        sequences * RING
    } else {
        sequences * PAGES * page
    };
    let mut fixture = Fixture::new(session, bucket)?;
    for (buffer, bytes) in [
        ("qkv", bucket * stride * 2),
        ("q", bucket * Q_HEADS * QK * 2),
        ("k_new", bucket * kvh * QK * 2),
        ("v_new", bucket * kvh * V * 2),
        ("k_pool", cache_rows * k_row),
        ("v_pool", cache_rows * v_row),
        ("block_tables", sequences * PAGES * 4),
        ("slot_ids", sequences * 4),
        ("token_batch", bucket * 4),
        ("positions", bucket * 4),
    ] {
        fixture.poisoned(buffer, bytes)?;
    }
    fixture.scalar("max_pages", PAGES as u32);
    fixture.scalar("ring_size", RING as u32);
    fixture.scalar64("k_page_stride_bytes", (page * k_row) as u64);
    fixture.scalar64("v_page_stride_bytes", (page * v_row) as u64);
    fixture.scalar("theta_bits", theta.to_bits());
    fixture.scalar("v_multiplier_bits", V_MULTIPLIER.to_bits());
    fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
    fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
    let token_batch: Vec<i32> = (0..bucket).map(|t| (t / per_sequence) as i32).collect();
    write(
        session,
        &fixture.buffers["token_batch"],
        0,
        &bytes_i32(&token_batch),
    )?;
    let launch = || fixture.launch(module, entry, bucket as u32, bucket as u32);
    let graph = session.capture(|_| launch())?;
    let mut misses = Vec::new();
    for live in live_counts(bucket as u32) {
        let live = live as usize;
        let seed = 0x7e11_0000 ^ ((live as u64) << 20) ^ ((bucket as u64) << 8) ^ name.len() as u64;
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // New rows, positions, pages and ring slots.
        let input = seeded_bf16(bucket * stride, seed);
        let limit = if ring { 4 * RING } else { PAGES * page };
        let bases: Vec<usize> = (0..sequences)
            .map(|_| (next() % (limit - per_sequence) as u64) as usize)
            .collect();
        let positions: Vec<usize> = (0..bucket)
            .map(|t| bases[t / per_sequence] + t % per_sequence)
            .collect();
        let mut pages: Vec<usize> = (0..sequences * PAGES).collect();
        let mut slots: Vec<usize> = (0..sequences).collect();
        for list in [&mut pages, &mut slots] {
            for i in (1..list.len()).rev() {
                list.swap(i, (next() % (i as u64 + 1)) as usize);
            }
        }
        write(session, &fixture.buffers["qkv"], 0, &bytes_u16(&input))?;
        write(
            session,
            &fixture.buffers["positions"],
            0,
            &bytes_i32(&positions.iter().map(|&p| p as i32).collect::<Vec<_>>()),
        )?;
        write(
            session,
            &fixture.buffers["block_tables"],
            0,
            &bytes_i32(&pages.iter().map(|&p| p as i32).collect::<Vec<_>>()),
        )?;
        write(
            session,
            &fixture.buffers["slot_ids"],
            0,
            &bytes_i32(&slots.iter().map(|&s| s as i32).collect::<Vec<_>>()),
        )?;
        write(
            session,
            &fixture.buffers["live_tokens"],
            0,
            &(live as u32).to_ne_bytes(),
        )?;
        // The cache row each token writes: through the ring slot or the
        // sequence's page table.
        let target = |t: usize| {
            let (sequence, position) = (t / per_sequence, positions[t]);
            if ring {
                slots[sequence] * RING + position % RING
            } else {
                pages[sequence * PAGES + position / page] * page + position % page
            }
        };
        let rows = |width: usize, tokens: std::ops::Range<usize>| {
            tokens
                .map(|t| target(t) * width..(target(t) + 1) * width)
                .collect::<Vec<_>>()
        };
        // Seeded cache contents; the rows tokens past the live count would
        // write hold poison and must keep it.
        let mut k_image: Vec<u8> = (0..cache_rows * k_row)
            .map(|_| next() as u8 & 0x3f)
            .collect();
        let mut v_image: Vec<u8> = (0..cache_rows * v_row)
            .map(|_| next() as u8 & 0x3f)
            .collect();
        for range in rows(k_row, live..bucket) {
            k_image[range].fill(POISON);
        }
        for range in rows(v_row, live..bucket) {
            v_image[range].fill(POISON);
        }
        let mut checks = Replay::new(session)
            .output(
                "q",
                &fixture.buffers["q"],
                WriteMap::of(DType::Bf16).prefix(live * Q_HEADS * QK)?,
            )
            .output(
                "k_new",
                &fixture.buffers["k_new"],
                WriteMap::of(DType::Bf16).prefix(live * kvh * QK)?,
            )
            .output(
                "v_new",
                &fixture.buffers["v_new"],
                WriteMap::of(DType::Bf16).prefix(live * kvh * V)?,
            )
            .state(
                "k_pool",
                &fixture.buffers["k_pool"],
                k_image,
                cache.byte_ranges(rows(k_row, 0..live))?,
                Vec::new(),
            )
            .state(
                "v_pool",
                &fixture.buffers["v_pool"],
                v_image,
                cache.byte_ranges(rows(v_row, 0..live))?,
                Vec::new(),
            );
        let label = format!("{name} bucket {bucket} live {live}");
        checks.eager(launch).map_err(|e| format!("{label}: {e}"))?;
        let oracle = if live > 0 {
            check_rope(
                &fixture,
                &checks,
                &label,
                &input[..live * stride],
                &positions[..live],
                (kvh, theta, fp8),
                &(0..live).map(target).collect::<Vec<_>>(),
            )
        } else {
            Ok(())
        };
        checks.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
        let mut receipt = BTreeMap::new();
        receipt.insert("family".into(), json!("rope_kv"));
        receipt.insert("entry".into(), json!(entry.symbol));
        receipt.insert("bucket".into(), json!(bucket));
        receipt.insert("sequences".into(), json!(sequences));
        receipt.insert("live".into(), json!(live));
        receipt.insert("correct".into(), json!(oracle.is_ok()));
        if let Err(miss) = &oracle {
            receipt.insert("oracle_miss".into(), json!(miss));
        }
        checks.record(&mut receipt);
        session.receipt("live_replay", receipt)?;
        misses.extend(oracle.err());
    }
    Ok(misses)
}

fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
        .collect()
}

/// The eager launch against the oracle: rotated Q and K and scaled V within
/// their bounds, and each live token's cache rows holding the new K and V
/// (BF16) or their E4M3 codes (FP8), as `rope_case` checks.
fn check_rope(
    fixture: &Fixture<'_>,
    checks: &Replay<'_>,
    label: &str,
    input: &[u16],
    positions: &[usize],
    (kvh, theta, fp8): (usize, f32, bool),
    targets: &[usize],
) -> LabResult<()> {
    let tokens = positions.len();
    let stride = Q_HEADS * QK + kvh * (QK + V);
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for row in input.chunks(stride) {
        q.extend_from_slice(&row[..Q_HEADS * QK]);
        k.extend_from_slice(&row[Q_HEADS * QK..(Q_HEADS + kvh) * QK]);
        v.extend_from_slice(&row[(Q_HEADS + kvh) * QK..]);
    }
    let expand = |heads: usize| {
        positions
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, heads))
            .collect::<Vec<_>>()
    };
    let rotate = |values: &[u16], heads: usize| {
        hf_rope_f32(
            &as_f64(values),
            &expand(heads),
            QK,
            ROTARY,
            f64::from(theta),
            None,
        )
    };
    let q_ref = oracle::OracleOutput::new(rotate(&q, Q_HEADS)?, vec![2.0; q.len()], 2)
        .map_err(|e| e.to_string())?;
    let k_ref = oracle::OracleOutput::new(rotate(&k, kvh)?, vec![2.0; k.len()], 2)
        .map_err(|e| e.to_string())?;
    let v_values: Vec<f64> = as_f64(&v)
        .into_iter()
        .map(|x| x * f64::from(V_MULTIPLIER))
        .collect();
    let v_ref = oracle::OracleOutput::new(v_values.clone(), vec![1.0; v.len()], 1)
        .map_err(|e| e.to_string())?;
    let read = |name: &str, len: usize| -> LabResult<Vec<u16>> {
        Ok(u16s(
            &checks.expected(name).ok_or("missing eager bytes")?[..len * 2],
        ))
    };
    let new_k = read("k_new", tokens * kvh * QK)?;
    let new_v = read("v_new", tokens * kvh * V)?;
    fixture.check_numeric(
        &format!("{label} Q"),
        &as_f32(&read("q", tokens * Q_HEADS * QK)?),
        &q_ref,
        DType::Bf16,
    )?;
    fixture.check_numeric(&format!("{label} K"), &as_f32(&new_k), &k_ref, DType::Bf16)?;
    fixture.check_numeric(&format!("{label} V"), &as_f32(&new_v), &v_ref, DType::Bf16)?;
    let k_pool = checks.expected("k_pool").ok_or("missing eager bytes")?;
    let v_pool = checks.expected("v_pool").ok_or("missing eager bytes")?;
    for (token, &row) in targets.iter().enumerate() {
        let (k_expected, v_expected): (Vec<u8>, Vec<u8>) = if fp8 {
            let encode = |values: &[f64]| {
                values
                    .iter()
                    .map(|&value| kernel_core::kv_fp8::encode(value as f32, 1.0))
                    .collect::<Vec<_>>()
            };
            (
                encode(&k_ref.values[token * kvh * QK..(token + 1) * kvh * QK]),
                encode(&v_values[token * kvh * V..(token + 1) * kvh * V]),
            )
        } else {
            (
                bytes_u16(&new_k[token * kvh * QK..(token + 1) * kvh * QK]),
                bytes_u16(&new_v[token * kvh * V..(token + 1) * kvh * V]),
            )
        };
        let element = if fp8 { 1 } else { 2 };
        let (k_bytes, v_bytes) = (kvh * QK * element, kvh * V * element);
        fixture.check_exact(
            &format!("{label} K cache row of token {token}"),
            &k_pool[row * k_bytes..(row + 1) * k_bytes],
            &k_expected[..],
        )?;
        fixture.check_exact(
            &format!("{label} V cache row of token {token}"),
            &v_pool[row * v_bytes..(row + 1) * v_bytes],
            &v_expected[..],
        )?;
    }
    Ok(())
}
