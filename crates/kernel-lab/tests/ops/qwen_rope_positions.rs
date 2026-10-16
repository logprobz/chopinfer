//! Exact-angle checks use tables=None in the oracle. Cache positions remain
//! small while independent RoPE coordinates sweep the full text domain.
use super::*;
use crate::row::rope_positions::{SEEDS, Stats, positions};

const LIMIT: usize = 262144;

fn write_words(session: &Session, buffer: &DeviceBuffer, data: &[u16]) -> LabResult<()> {
    buffer
        .copy_from_bytes(
            &data
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            &session.stream,
        )
        .map_err(|e| e.to_string())
}

fn current_rows(session: &Session, fold: &GatedFoldCase) -> LabResult<Vec<(u64, usize, Vec<u8>)>> {
    let case = fold.case;
    let v = case.variant;
    let page = v.query.page_size as usize;
    let max_pages = v.query.max_context_capacity as usize / page;
    let table = bytes(session, &fold.fixture.buffers[&TABLE])?;
    let table: Vec<_> = table
        .chunks_exact(4)
        .map(|x| i32::from_ne_bytes(x.try_into().unwrap()) as usize)
        .collect();
    let width = v.query.kv_heads as usize
        * v.query.qk_dim as usize
        * if v.query.kv_dtype == DType::Bf16 {
            2
        } else {
            1
        };
    let mut result = Vec::new();
    for b in 0..case.batch {
        let pos = case.context() - 1;
        for (handle, stride) in [
            (K, fold.fixture.arguments.k_page_stride_bytes),
            (V, fold.fixture.arguments.v_page_stride_bytes),
        ] {
            let at = table[b * max_pages + pos / page] * stride as usize + pos % page * width;
            let mut data = vec![0; width];
            fold.fixture.buffers[&handle]
                .copy_to_bytes_at_async(at, &mut data, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            result.push((handle, at, data));
        }
    }
    Ok(result)
}

fn parity(session: &Session, fold: &GatedFoldCase) -> LabResult<Vec<(u64, usize, Vec<u8>)>> {
    fold.launch_unfused(session, None)?;
    let expected = fold.fixture.output(session)?;
    let cache = current_rows(session, fold)?;
    for (handle, offset, data) in &cache {
        fold.fixture.buffers[handle]
            .copy_from_bytes_at(*offset, &vec![0xff; data.len()], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let graph = session.capture(|_| fold.launch_folded(session, None))?;
    graph.launch(&session.stream).map_err(|e| e.to_string())?;
    exact(
        fold.case,
        &expected,
        &fold.fixture.output(session)?,
        "Qwen position fold",
    )?;
    for ((_, _, expected), (_, _, actual)) in cache.iter().zip(current_rows(session, fold)?) {
        if expected != &actual { return Err("Qwen position cache differs bitwise".into()); }
    }
    Ok(cache)
}

#[test]
fn qwen_text_and_folded_rope_positions() {
    gpu_test(2 << 30, |session| {
        let cubin = module(session, model_qwen38_contract::sm120::config::PROFILE)?;
        let exhaustive = std::env::var("CHOPINFER_ROPE_EXHAUSTIVE").as_deref() == Ok("1");
        let mut failures = 0;
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            let variant = qwen(name)?;
            let fp8 = variant.query.kv_dtype != DType::Bf16;
            for rows in [1, 32, 128] {
                let selected: Vec<_> = positions(exhaustive && rows > 1).into_iter().filter(|&p| p < LIMIT).collect();
                let case = decode_case(variant.with_batch_capacity(rows as u32).map_err(|e| e.to_string())?, rows, 17, false);
                let mut fold = prepare_gated_fold(session, &cubin, case)?;
                fold.fixture.buffers.insert(COS_SIN, put_f32(session, &cos_sin_table(LIMIT))?);
                let q_norm = as_f64(&session.download_bf16(&fold.fixture.buffers[&Q_NORM], 256)?);
                let k_norm = as_f64(&session.download_bf16(&fold.fixture.buffers[&K_NORM], 256)?);
                for seed in SEEDS {
                    let mut stats = Stats::default();
                    stats.oracle_description = "f64 Gemma norm and computed M-RoPE; OracleOutput K/S bounds; FP8 K uses the existing storage bound";
                    stats.maximum_position = LIMIT - 1;
                    for chunk in selected.chunks(rows) {
                        let ps: Vec<_> = (0..rows).map(|i| chunk[i.min(chunk.len() - 1)]).collect();
                        let rope: Vec<_> = ps.iter().map(|&p| [p; 3]).collect();
                        let mut state = seed ^ ((ps[0] as u64 + 1) << 24);
                        let input = grid_bf16(rows * model_qwen38_contract::sm120::attention::ROW_WIDTH as usize, 0.7, &mut state);
                        write_words(session, &fold.fixture.buffers[&QKV], &input)?;
                        fold.fixture.buffers[&ROPE_POSITIONS].copy_from_bytes(&rope.iter().flat_map(|p| p.iter().flat_map(|&x| (x as i32).to_ne_bytes())).collect::<Vec<_>>(), &session.stream).map_err(|e| e.to_string())?;
                        write_words(session, &fold.fixture.buffers[&Q], &vec![0x7fc1; rows * 6 * 256])?;
                        let cache = parity(session, &fold)?;
                        let actual_q = session.download_bf16(&fold.fixture.buffers[&Q], rows * 6 * 256)?;
                        for (row, &p) in ps.iter().enumerate() {
                            let x = &input[row * model_qwen38_contract::sm120::attention::ROW_WIDTH as usize..][..model_qwen38_contract::sm120::attention::ROW_WIDTH as usize];
                            let mut keys = vec![0.0; 64 * 256];
                            let mut values = keys.clone();
                            let reference = gated_oracle::qk_norm_rope_kv_write(rows_query(variant.query.kv_dtype, 6), GatedAttentionInputs {
                                q_gate: &as_f64(&x[..6 * 512]), k: &as_f64(&x[6 * 512..7 * 512 - 256]), v: &as_f64(&x[7 * 512 - 256..]),
                                q_norm: &q_norm, k_norm: &k_norm, rope_positions: &[[p; 3]], token_positions: &[16], tables: None,
                            }, PagedKvPair { keys: &mut keys, values: &mut values, block_table: &[0] }).map_err(|e| e.to_string())?;
                            for i in 0..6 * 256 {
                                if i % 256 >= 64 { continue; }
                                stats.check_bound(f64::from(bf16_to_f32(actual_q[row * 6 * 256 + i])), reference.q.values[i], oracle::numeric_bound(&reference.q, i, DType::Bf16), p, i % 32);
                            }
                            let k_bytes = &cache[row * 2].2;
                            for i in 0..64 {
                                if fp8 {
                                    if !e4m3_close(k_bytes[i], reference.k.values[i], f64::from(fold.fixture.arguments.k_scale)) {
                                        return Err(format!("Qwen FP8 K fails storage bound at position {p}, dimension {i}"));
                                    }
                                } else {
                                    let value = bf16_to_f32(u16::from_ne_bytes([k_bytes[2 * i], k_bytes[2 * i + 1]]));
                                    stats.check_bound(f64::from(value), reference.k.values[i], oracle::numeric_bound(&reference.k, i, DType::Bf16), p, i % 32);
                                }
                            }
                        }
                    }
                    stats.receipt(session, &format!("{name}:exact_angles:fold_cache_and_output_bitwise"), rows, seed, exhaustive)?;
                    failures += stats.failures;
                }
            }
        }
        if failures > 0 { return Err(format!("Qwen exact-angle sweep: {failures} failures")); }
        Ok(())
    }).expect("Qwen text and folded exact-angle RoPE");
}

#[test]
fn qwen_rope_costs() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    gpu_test(2 << 30, |session| {
        let cubin = module(session, model_qwen38_contract::sm120::config::PROFILE)?;
        for name in ["qwen38_bf16_decode", "qwen38_fp8_decode"] {
            for rows in [1, 2, 4, 8, 16, 32, 128] {
                let mut fold =
                    prepare_gated_fold(session, &cubin, decode_case(qwen(name)?.with_batch_capacity(rows as u32).map_err(|e| e.to_string())?, rows, 17, false))?;
                fold.fixture
                    .buffers
                    .insert(COS_SIN, put_f32(session, &cos_sin_table(LIMIT))?);
                fold.fixture.buffers[&ROPE_POSITIONS]
                    .copy_from_bytes(
                        &vec![(LIMIT - 1) as i32; rows * 3]
                            .iter()
                            .flat_map(|x| x.to_ne_bytes())
                            .collect::<Vec<_>>(),
                        &session.stream,
                    )
                    .map_err(|e| e.to_string())?;
                parity(session, &fold)?;
                let mut receipt = BTreeMap::from([
                    ("entry".into(), json!(name)),
                    ("rows".into(), json!(rows)),
                    ("rope_position".into(), json!(LIMIT - 1)),
                    ("cache_context".into(), json!(17)),
                ]);
                for mode in ["row", "unfused", "folded"] {
                    let graph = session.capture(|_| match mode {
                        "row" => launch::launch(
                            &session.context,
                            &fold.row,
                            &session.stream,
                            fold.row_plan,
                            &fold.row_args,
                            &fold.fixture.cache_bindings(None),
                        ),
                        "unfused" => fold.launch_unfused(session, None),
                        _ => fold.launch_folded(session, None),
                    })?;
                    let timing = time(session, 3, 15, |stream| {
                        graph.launch(stream).map_err(|e| e.to_string())
                    })?;
                    receipt.insert(format!("{mode}_us"), json!(timing.median_ms * 1000.0));
                }
                session.receipt("qwen_rope_cost", receipt)?;
            }
        }
        Ok(())
    })
    .expect("Qwen RoPE row and fold costs");
}
