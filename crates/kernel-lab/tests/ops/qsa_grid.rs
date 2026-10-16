//! Compare live work partitioning with the same header's fixed traversal.
use super::*;

#[test]
fn qsa_live_grid_is_bitwise_exact() {
    gpu_test(8_u64 << 30, |session| {
        let current = module(session)?;
        let baseline = session.module("qsa_fixed_grid")?;
        let max_blocks = 65_536;
        let keys: Vec<u16> = (0..max_blocks * INDEX_DIM)
            .map(|i| f32_to_bf16(((i * 37 % 127) as f32 - 63.0) / 64.0))
            .collect();
        let (bank, _, max_pages) = upload_block_keys(session, &keys)?;
        // Every sequence owns physical pages. Full-context C16 and wider
        // batches exceed L2; time() also flushes L2 before every sample.
        let bank_bytes = max_pages as usize * BLOCK_PAGE_STRIDE as usize;
        let d_keys = session.allocate(256 * bank_bytes)?;
        for row in 0..256 {
            d_keys
                .copy_range_from_async(row * bank_bytes, &bank, 0, bank_bytes, &session.stream)
                .map_err(|e| e.to_string())?;
        }
        for phase in ["decode", "verify"] {
            let score_entry = entry(&format!("block_scores_{phase}"));
            let old = function(
                &baseline,
                &score_entry.symbol().replace("chop_qsa_", "chop_qsa_fixed_"),
                score_entry.dynamic_shared_bytes(),
            )?;
            let new = function(
                &current,
                &score_entry.symbol(),
                score_entry.dynamic_shared_bytes(),
            )?;
            for rows in [1_u32, 2, 4, 8, 16, 32, 33, 64, 128, 256] {
                let prepared = rows.next_power_of_two();
                let q: Vec<u16> = (0..prepared as usize * INDEX_HEADS * INDEX_DIM)
                    .map(|i| f32_to_bf16(((i * 29 % 97) as f32 - 48.0) / 32.0))
                    .collect();
                let d_q = upload(session, &bytes_u16(&q))?;
                let batches: Vec<i32> = (0..prepared)
                    .map(|r| {
                        if phase == "verify" {
                            (r / 8) as i32
                        } else {
                            r as i32
                        }
                    })
                    .collect();
                let d_batches = upload(session, &bytes_i32(&batches))?;
                let table: Vec<i32> = (0..prepared * max_pages).map(|p| p as i32).collect();
                let d_table = upload(session, &bytes_i32(&table))?;
                let d_live = upload(session, &rows.to_ne_bytes())?;
                let d_positions = upload(session, &vec![0; prepared as usize * 4])?;
                let bytes = prepared as usize * max_blocks * 4;
                let d_old = session.allocate(bytes)?;
                let d_new = session.allocate(bytes)?;
                let plan = score_entry
                    .plan(prepared, 262_144)
                    .map_err(|e| e.0.to_string())?;
                let mut fixed = plan;
                fixed.grid = [prepared, if phase == "decode" { 1024 } else { 64 }, 1];
                let run = |stream: &Stream, original: bool| {
                    let (f, output, p) = if original {
                        (&old, &d_old, &fixed)
                    } else {
                        (&new, &d_new, &plan)
                    };
                    f.launch(
                        stream,
                        config(*p)?,
                        &[
                            KernelArgument::Buffer(&d_q),
                            KernelArgument::Buffer(&d_keys),
                            KernelArgument::Buffer(&d_table),
                            KernelArgument::Buffer(&d_batches),
                            KernelArgument::Buffer(&d_positions),
                            KernelArgument::Buffer(&d_live),
                            KernelArgument::Buffer(output),
                            KernelArgument::U32(max_pages),
                            KernelArgument::U64(BLOCK_PAGE_STRIDE),
                            KernelArgument::U32(max_blocks as u32),
                        ],
                    )
                    .map_err(|e| e.to_string())
                };
                let graph = session.capture(|stream| run(stream, false))?;
                let old_graph = session.capture(|stream| run(stream, true))?;
                // Replay the same capture with changed live counts, contexts,
                // and sequence grouping, including served MTP verify shapes.
                let cases = [
                    ("short", 2048_i32, rows),
                    ("boundary", 2052, rows),
                    ("ragged", 8195, rows),
                    ("long", 262_144, rows),
                    ("empty", 0, 0),
                    ("skew_first", 262_144, rows),
                    ("skew_last", 262_144, rows),
                    ("partly_live", 262_144, rows.div_ceil(2)),
                    ("single_live", 262_144, 1),
                    ("irregular_runs", 262_144, rows),
                    ("verify_short", 2052, rows),
                    ("verify_long", 262_144, rows),
                ];
                for (case, context, live) in cases {
                    let grouping: Vec<i32> = (0..prepared)
                        .map(|r| {
                            if case.starts_with("verify_") {
                                0
                            } else if case == "irregular_runs" {
                                // Repeated runs of lengths 1, 7, 3, 21.
                                (r / 32 * 4
                                    + match r % 32 {
                                        0 => 0,
                                        1..=7 => 1,
                                        8..=10 => 2,
                                        _ => 3,
                                    }) as i32
                            } else {
                                r as i32
                            }
                        })
                        .collect();
                    d_batches
                        .copy_from_bytes(&bytes_i32(&grouping), &session.stream)
                        .map_err(|e| e.to_string())?;
                    d_live
                        .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
                        .map_err(|e| e.to_string())?;
                    let positions: Vec<i32> = (0..prepared)
                        .map(|r| {
                            let length = match case {
                                "skew_first" | "partly_live" | "single_live" if r != 0 => 2052,
                                "skew_last" if r != live - 1 => 2052,
                                "irregular_runs" => {
                                    [2052, 262_144, 8195, 4099][grouping[r as usize] as usize % 4]
                                }
                                _ => context,
                            };
                            (length - 1 - (r % 8) as i32).max(0)
                        })
                        .collect();
                    d_positions
                        .copy_from_bytes(&bytes_i32(&positions), &session.stream)
                        .map_err(|e| e.to_string())?;
                    kernel_lab::poison::poison(session, &[&d_old, &d_new])?;
                    run(&session.stream, true)?;
                    graph.launch(&session.stream).map_err(|e| e.to_string())?;
                    let expected = download(session, &d_old, bytes)?;
                    let actual = download(session, &d_new, bytes)?;
                    // Only live rows with more than 512 complete blocks get
                    // scores, one per complete block (SPARSE.md); every other
                    // byte keeps its poison.
                    let written = kernel_lab::poison::WriteMap::of(DType::F32).byte_ranges(
                        (0..live as usize).filter_map(|r| {
                            let complete = (positions[r] as usize + 1) / 4;
                            (complete > 512)
                                .then(|| r * max_blocks * 4..(r * max_blocks + complete) * 4)
                        }),
                    )?;
                    kernel_lab::poison::check(
                        &format!("{phase} rows={rows} live={live} case={case} scores"),
                        &actual,
                        &written,
                        kernel_lab::poison::Fill::Pattern(&[kernel_lab::poison::POISON]),
                    )?;
                    if let Some(at) = expected.iter().zip(&actual).position(|(a, b)| a != b) {
                        return Err(format!(
                            "{phase} rows={rows} live={live} case={case}: score byte {at} differs"
                        ));
                    }
                    let mut receipt = BTreeMap::from([
                        ("correct".into(), json!(true)),
                        ("phase".into(), json!(phase)),
                        ("rows".into(), json!(rows)),
                        ("live_rows".into(), json!(live)),
                        ("case".into(), json!(case)),
                        ("distinct_physical_key_banks".into(), json!(prepared)),
                        ("flush_l2_before_each_sample".into(), json!(true)),
                        ("context".into(), json!(context)),
                        ("grid".into(), json!(plan.grid)),
                        ("bitwise_exact_including_padding".into(), json!(true)),
                    ]);
                    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1")
                        && phase == "decode"
                        && ([1, 16, 32, 256].contains(&rows)
                            && [
                                "short",
                                "boundary",
                                "long",
                                "skew_first",
                                "skew_last",
                                "partly_live",
                                "single_live",
                            ]
                            .contains(&case)
                            || [2, 4, 8].contains(&rows) && case.starts_with("verify_"))
                    {
                        let before = time(session, 2, 9, |stream| {
                            old_graph.launch(stream).map_err(|e| e.to_string())
                        })?;
                        let after = time(session, 2, 9, |stream| {
                            graph.launch(stream).map_err(|e| e.to_string())
                        })?;
                        receipt.insert("fixed_us".into(), json!(before.median_ms * 1000.0));
                        receipt.insert("bounded_us".into(), json!(after.median_ms * 1000.0));
                    }
                    session.receipt("c16_small_qsa", receipt)?;
                }
            }
        }
        Ok(())
    })
    .expect("QSA bounded grid exactness");
}
