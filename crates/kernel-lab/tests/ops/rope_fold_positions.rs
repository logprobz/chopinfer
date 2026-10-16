//! Shared row/fold angle checks at absolute text positions.
use super::*;
use crate::row::rope_positions::{SEEDS, Stats, arm, uses_tables};

fn selected() -> Vec<Variant> {
    folded_variants()
        .into_iter()
        .filter(|v| {
            [
                "mimo_ga_bf16_decode",
                "mimo_ga_fp8_decode",
                "mimo_swa_decode",
            ]
            .contains(&v.name)
        })
        .collect()
}

fn modules(session: &Session) -> LabResult<(Module, Module)> {
    Ok((
        session.load_cubin(ROW_CUBIN)?,
        module(session, model_mimo26_contract::sm120::config::PROFILE)?,
    ))
}

fn bound_functions(
    fold: &FoldFixture,
    row: &Module,
    attention: &Module,
) -> LabResult<FoldFunctions> {
    let partial = attention
        .function(&fold.fixture.planned.partial_symbol)
        .map_err(|e| e.to_string())?;
    let merge = attention
        .function(&format!("chop_attn_{}_merge_v1", fold.variant.name))
        .map_err(|e| e.to_string())?;
    let folded = attention
        .function(&fold.folded.partial_symbol)
        .map_err(|e| e.to_string())?;
    for (function, bytes) in [
        (&partial, fold.fixture.planned.partial.dynamic_shared_bytes),
        (&merge, fold.fixture.planned.merge.dynamic_shared_bytes),
        (&folded, fold.folded.partial.dynamic_shared_bytes),
    ] {
        if bytes > 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(bytes)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(FoldFunctions {
        row: row
            .function(&fold.row_symbol)
            .map_err(|e| e.to_string())?,
        partial,
        merge,
        folded,
    })
}

fn cache_rows(
    session: &Session,
    fold: &FoldFixture,
    case: Case,
) -> LabResult<Vec<(u64, usize, Vec<u8>)>> {
    let f = &fold.fixture;
    let v = case.variant;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let page = v.query.page_size as usize;
    let ring = v.layout == CacheLayout::Ring;
    let ids = bytes(session, &f.buffers[&if ring { SLOT } else { TABLE }])?;
    let ids: Vec<_> = ids
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()) as usize)
        .collect();
    let max_pages = v.query.max_context_capacity.div_ceil(v.query.page_size) as usize;
    let mut result = Vec::new();
    for (row, context) in case.contexts().into_iter().enumerate() {
        let position = context - 1;
        for (handle, dim, stride) in [
            (K, v.query.qk_dim, f.arguments.k_page_stride_bytes),
            (V, v.query.v_dim, f.arguments.v_page_stride_bytes),
        ] {
            let width = v.query.kv_heads as usize * dim as usize * if fp8 { 1 } else { 2 };
            let offset = if ring {
                (ids[row] * page + position % page) * width
            } else {
                ids[row * max_pages + position / page] * stride as usize + position % page * width
            };
            let mut data = vec![0; width];
            f.buffers[&handle]
                .copy_to_bytes_at_async(offset, &mut data, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            result.push((handle, offset, data));
        }
    }
    Ok(result)
}

fn compare_fold(
    session: &Session,
    fold: &FoldFixture,
    functions: &FoldFunctions,
    case: Case,
) -> LabResult<()> {
    fold.launch_unfused(session, functions, None)?;
    let expected = fold.fixture.output(session)?;
    let cache = cache_rows(session, fold, case)?;
    for (handle, offset, data) in &cache {
        fold.fixture.buffers[handle]
            .copy_from_bytes_at(*offset, &vec![0xff; data.len()], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let graph = session.capture(|_| fold.launch_folded(session, functions, None))?;
    graph.launch(&session.stream).map_err(|e| e.to_string())?;
    exact(
        case,
        &expected,
        &fold.fixture.output(session)?,
        "RoPE position fold",
    )?;
    for ((_, _, expected), (_, _, actual)) in cache.iter().zip(cache_rows(session, fold, case)?) {
        if expected != &actual {
            return Err("RoPE position cache differs bitwise".into());
        }
    }
    Ok(())
}

// Repeated read-only prefix pages keep the high-position arithmetic probe
// small. Every sequence owns a separate terminal page, so its current K/V
// write cannot alias an earlier logical page. The ordinary full-cache
// fixture remains the benchmark input. No full-attention oracle is claimed
// for this synthetic prefix; the checks are exact row RoPE and fold parity.
fn position_fixture(session: &Session, case: Case) -> LabResult<FoldFixture> {
    if case.context() <= 128 {
        return prepare_fold(session, case);
    }
    let mut fold = prepare_fold(
        session,
        Case {
            prefix: 127,
            ..case
        },
    )?;
    let f = &mut fold.fixture;
    f.buffers.insert(
        SEQ,
        put_i32(session, &vec![case.context() as i32; case.batch])?,
    );
    f.buffers.insert(
        POSITIONS,
        put_i32(session, &vec![case.prefix as i32; case.batch])?,
    );
    f.buffers.insert(
        PREFIX,
        put_i32(session, &vec![case.prefix as i32; case.batch])?,
    );
    let variant = case.variant;
    if variant.layout != CacheLayout::Ring {
        let page = variant.query.page_size as usize;
        let max_pages = variant.query.max_context_capacity as usize / page;
        let mut tables = vec![0; case.batch * max_pages];
        for b in 0..case.batch {
            let list = &f.page_lists[b];
            assert!(list.len() >= 2);
            tables[b * max_pages..(b + 1) * max_pages].fill(list[0] as i32);
            tables[b * max_pages + case.prefix / page] = *list.last().unwrap() as i32;
        }
        f.buffers.insert(TABLE, put_i32(session, &tables)?);
    }
    f.counts.context = case.context() as u32;
    f.planned = variant
        .plan_for_bucket(f.bucket, f.counts, f.arguments.clone())
        .map_err(|e| e.to_string())?;
    fold.folded = variant
        .plan_folded(
            f.bucket,
            f.counts,
            f.arguments.clone(),
            RopeInputs {
                qkv: QKV,
                theta: rope_theta(variant),
                v_multiplier: V_MULTIPLIER,
            },
        )
        .map_err(|e| e.to_string())?;
    f.buffers.insert(
        WORK,
        session.allocate(
            f.planned
                .partial
                .workspace_bytes
                .max(fold.folded.partial.workspace_bytes) as usize,
        )?,
    );
    Ok(fold)
}

// Compare the served table fold with an independently launched table row.
// One device allocation owns the same angle bits for both paths.
fn use_table(session: &Session, fold: &mut FoldFixture, context: usize) -> LabResult<()> {
    if !uses_tables() {
        return Ok(());
    }
    const ANGLES: u64 = 10001;
    let theta = f64::from(rope_theta(fold.variant));
    let values: Vec<f32> = (0..context)
        .flat_map(|p| {
            (0..32).flat_map(move |pair| {
                let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, theta);
                [c as f32, s as f32]
            })
        })
        .collect();
    fold.fixture
        .buffers
        .insert(ANGLES, session.upload_f32(&values)?);
    fold.row_symbol = fold.row_symbol.replace("_computed_", "_table_");
    fold.row_args.0[10] = Argument::DeviceBuffer(ANGLES);
    fold.folded = fold
        .variant
        .plan_folded_table(
            fold.fixture.bucket,
            fold.fixture.counts,
            fold.fixture.arguments.clone(),
            RopeInputs {
                qkv: QKV,
                theta: theta as f32,
                v_multiplier: V_MULTIPLIER,
            },
            ANGLES,
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
fn folded_rope_position_sweep() {
    gpu_test(4 << 30, |session| {
        let (row_module, attention_module) = modules(session)?;
        let mut failures = 0;
        for variant in selected() {
            for position in [0, 36, 46, 127, 511, 4095, 65535, 262143, 1048575] {
                for batch in [1, 32] {
                    if batch > 1 && position > 4095 {
                        continue;
                    }
                    let case = Case {
                        variant,
                        batch,
                        rows: 1,
                        prefix: position,
                        mixed: false,
                    };
                    let mut fold = position_fixture(session, case)?;
                    use_table(session, &mut fold, case.context())?;
                    let functions = bound_functions(&fold, &row_module, &attention_module)?;
                    let (qh, kh, d, vd) = (
                        variant.query.q_heads as usize,
                        variant.query.kv_heads as usize,
                        variant.query.qk_dim as usize,
                        variant.query.v_dim as usize,
                    );
                    let stride = qh * d + kh * (d + vd);
                    for seed in SEEDS {
                        let mut state = seed ^ position as u64;
                        let input: Vec<_> = (0..batch * stride)
                            .map(|_| {
                                f32_to_bf16((random_word(&mut state) % 1401) as f32 / 1000.0 - 0.7)
                            })
                            .collect();
                        fold.fixture.buffers[&QKV]
                            .copy_from_bytes(
                                &input
                                    .iter()
                                    .flat_map(|x| x.to_ne_bytes())
                                    .collect::<Vec<_>>(),
                                &session.stream,
                            )
                            .map_err(|e| e.to_string())?;
                        compare_fold(session, &fold, &functions, case)?;
                        // Q and K_NEW are the separately launched row's outputs.
                        // Fold equality above proves the same Q rotation reaches
                        // attention and the same K bits reach the cache.
                        let q = session.download_bf16(&fold.fixture.buffers[&Q], batch * qh * d)?;
                        let k =
                            session.download_bf16(&fold.fixture.buffers[&K_NEW], batch * kh * d)?;
                        let mut stats = Stats::default();
                        for b in 0..batch {
                            for h in 0..qh + kh {
                                let source: Vec<_> = input[b * stride + h * d..][..d]
                                    .iter()
                                    .map(|&x| f64::from(bf16_to_f32(x)))
                                    .collect();
                                let expected = crate::row::hf_rope_bf16(
                                    &source,
                                    &[position],
                                    d,
                                    f64::from(rope_theta(variant)),
                                )?;
                                let actual = if h < qh {
                                    &q[(b * qh + h) * d..][..d]
                                } else {
                                    &k[(b * kh + h - qh) * d..][..d]
                                };
                                for i in 0..64 {
                                    stats.check_selected(actual[i], expected[i], position, i % 32);
                                }
                            }
                        }
                        stats.receipt(
                            session,
                            &format!(
                                "fold:{}:position={position}:cache_and_output_bitwise",
                                variant.name
                            ),
                            batch,
                            seed,
                            false,
                        )?;
                        failures += stats.failures;
                    }
                }
            }
        }
        if failures > 0 {
            return Err(format!("folded RoPE: {failures} elements exceed the bound"));
        }
        Ok(())
    })
    .expect("MiMo folded RoPE position sweep");
}

#[test]
fn folded_rope_position_costs() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    gpu_test(4 << 30, |session| {
        let (row_module, attention_module) = modules(session)?;
        for variant in selected() {
            for context in [128, 4096] {
                for batch in [1, 2, 4, 8, 16, 32] {
                    let case = Case {
                        variant,
                        batch,
                        rows: 1,
                        prefix: context - 1,
                        mixed: false,
                    };
                    let mut fold = prepare_fold(session, case)?;
                    use_table(session, &mut fold, case.context())?;
                    let functions = bound_functions(&fold, &row_module, &attention_module)?;
                    compare_fold(session, &fold, &functions, case)?;
                    let mut receipt = BTreeMap::from([
                        ("arm".into(), json!(arm())),
                        ("case".into(), json!(case.label())),
                        ("context".into(), json!(context)),
                        ("rows".into(), json!(batch)),
                        (
                            "scope".into(),
                            json!("captured attention path, cold L2, no neighboring GEMMs"),
                        ),
                    ]);
                    for folded in [false, true] {
                        let graph = session.capture(|_| {
                            if folded {
                                fold.launch_folded(session, &functions, None)
                            } else {
                                fold.launch_unfused(session, &functions, None)
                            }
                        })?;
                        let timing = time(session, 3, 15, |stream| {
                            graph.launch(stream).map_err(|e| e.to_string())
                        })?;
                        receipt.insert(
                            if folded { "folded_us" } else { "unfused_us" }.into(),
                            json!(timing.median_ms * 1000.0),
                        );
                    }
                    session.receipt("rope_fold_cost", receipt)?;
                }
            }
        }
        Ok(())
    })
    .expect("MiMo folded RoPE costs");
}
