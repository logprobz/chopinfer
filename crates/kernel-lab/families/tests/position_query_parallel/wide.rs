//! Explicit wide-row query pairs, with upper-row failures and reused graphs.
use super::*;
const ROWS: [u32; 8] = [32, 33, 40, 48, 49, 56, 63, 64];

fn check(
    s: &kernel_harness::Session,
    module: &runtime_cuda::Module,
    rows: u32,
    timing: bool,
) -> LabResult<()> {
    let mut a = Arena {
        session: s,
        buffers: vec![],
    };
    let original = inputs(rows + 1);
    let table: Vec<f32> = (0..8 * 32)
        .flat_map(|i| [[0.80013, 0.59991], [1., 0.], [-0.59993, 0.80017]][i % 3])
        .collect();
    let b = PositionQueryBuffers {
        q: a.input(&f32s(&original))?,
        cos_sin: a.input(&f32s(&table))?,
        positions: a.input(&i32s((0..rows).map(|r| (r % 8) as i32)))?,
        live_rows: a.input(&rows.to_le_bytes())?,
        output: a.output((rows as usize + 1) * 4096 * 2)?,
        quantizer_status: a.output((rows as usize + 4) * 4)?,
        quantizer_status_row_offset: 2,
    };
    let mut calls = vec![];
    for parallel in [false, true] {
        let selected = if parallel {
            backend(false)
                .with_parallel_position_query_rows(64)
                .map_err(|e| e.to_string())?
        } else {
            backend(false)
        };
        let op = selected
            .position_indexer_query(&family::query(), Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let scratch = a.output(op.info().workspace as usize)?;
        calls.push(
            op.launches(&b, scratch, Counts::rows(rows).with_positions(8))
                .map_err(|e| e.to_string())?,
        );
    }
    assert_eq!(calls[0].len(), 2);
    assert_eq!(calls[1].len(), 1);
    assert!(calls[1][0].symbol.ends_with("_parallel_v1"));
    assert_eq!(calls[1][0].plan.block, [1024, 1, 1]);
    let graphs = calls
        .iter()
        .map(|c| s.capture(|_| fire(s, module, &a, c)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut comparisons = 0;
    for row in [0, rows as usize - 1] {
        for case in 0..18 {
            let mut values = original.clone();
            let mut positions: Vec<_> = (0..rows).map(|r| (r % 8) as i32).collect();
            let mut cos_sin = table.clone();
            let mut status = vec![0xa5a5_a5a5_u32; rows as usize + 4];
            status[2..2 + rows as usize].fill(0);
            let start = row * 4096;
            match case {
                1 => values[start] = f32::NAN,
                2 => values[start + 4095] = f32::INFINITY,
                3 => values[start + 4094] = -f32::INFINITY,
                4 => values[start] = f32::MAX,
                5 => values[start] = f32::from_bits(0x7f7f_0000),
                6 => {
                    positions[row] = -1;
                    values[start..start + 4096].fill(f32::NAN);
                }
                7 => {
                    positions[row] = 8;
                    values[start..start + 4096].fill(f32::NAN);
                }
                8 => status[2 + row] = 128,
                9 => {
                    status[2 + row] = 128;
                    values[start + 4095] = f32::NAN;
                }
                10 => values[start..start + 4096].fill(-0.),
                11 => values[start..start + 4096].fill(f32::MIN_POSITIVE),
                12 => values[start..start + 4096].fill(f32::from_bits(0x0001_0000)),
                13 => {
                    values[start] = 6.015625;
                    values[start + 1] = 0.5;
                }
                16 => cos_sin[(row % 8) * 64] = f32::NAN,
                17 => {
                    values[start + 64] = f32::MAX;
                    values[start + 65] = f32::MAX;
                }
                _ => (),
            }
            let live = match case {
                14 => 0,
                15 => rows - 1,
                _ => rows,
            };
            a.write(b.q, &f32s(&values))?;
            a.write(b.cos_sin, &f32s(&cos_sin))?;
            a.write(b.positions, &i32s(positions))?;
            a.write(b.live_rows, &live.to_le_bytes())?;
            let flags: Vec<u8> = status.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut expected = None;
            for replay in 0..3 {
                for arm in 0..2 {
                    a.poison(&[b.output])?;
                    a.write(b.quantizer_status, &flags)?;
                    if replay == 0 {
                        fire(s, module, &a, &calls[arm])?;
                    } else {
                        graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let actual = vec![a.read(b.output)?, a.read(b.quantizer_status)?];
                    if a.read(b.q)? != f32s(&values)
                        || actual[0][live as usize * 4096 * 2..]
                            .iter()
                            .any(|&x| x != 0xa5)
                        || actual[1][..8] != flags[..8]
                        || actual[1][(live as usize + 2) * 4..] != flags[(live as usize + 2) * 4..]
                    {
                        return Err("wide query changed input, guards or inactive rows".into());
                    }
                    if let Some(expected) = &expected {
                        assert_eq!(
                            &actual, expected,
                            "rows={rows} case={case} row={row} replay={replay} arm={arm}"
                        );
                        comparisons += 1;
                    } else {
                        expected = Some(actual);
                    }
                }
            }
            if case == 8 {
                let mut mutant = calls[1].clone();
                mutant[0].symbol = mutant[0]
                    .symbol
                    .replace("_parallel_v1", "_parallel_ignore_status_v1");
                a.poison(&[b.output])?;
                a.write(b.quantizer_status, &flags)?;
                fire(s, module, &a, &mutant)?;
                assert_ne!(
                    a.read(b.output)?,
                    expected.unwrap()[0],
                    "wide admission control escaped"
                );
            }
        }
    }
    assert_eq!(comparisons, 180);
    a.write(b.q, &f32s(&original))?;
    a.write(b.cos_sin, &f32s(&table))?;
    a.write(b.positions, &i32s((0..rows).map(|r| (r % 8) as i32)))?;
    a.write(b.live_rows, &rows.to_le_bytes())?;
    a.write(b.quantizer_status, &vec![0; (rows as usize + 4) * 4])?;
    let times = if timing {
        Some(time_pair(s, module, &a, [&calls[0], &calls[1]], 9)?)
    } else {
        None
    };
    println!(
        "INDEX_QUERY_WIDE_CASE {}",
        serde_json::json!({"rows":rows,"witnesses":36,"comparisons":comparisons,
        "negative_controls":2,"graph_replays":2,"bitwise":true,"guards":true,"time_ms":times})
    );
    Ok(())
}

#[test]
#[ignore = "requires the staged pair GPU job"]
fn gpu_wide_query_parallel() {
    let small = std::env::var("CHOPINFER_INDEX_QUERY_WIDE_MODE").as_deref() == Ok("memcheck");
    let rows = if small { vec![33, 64] } else { ROWS.to_vec() };
    let ran = gpu_test(1 << 30, |s| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_QUERY_CUBINS").ok_or("query cubins required")?,
        );
        let module = s.load_cubin(
            &std::fs::read(root.join(format!("{}.cubin", family::CUBIN)))
                .map_err(|e| e.to_string())?,
        )?;
        for &row in &rows {
            check(s, &module, row, !small)?;
        }
        println!(
            "INDEX_QUERY_WIDE_COMPLETE rows={} witnesses={} comparisons={} graph=true",
            rows.len(),
            rows.len() * 36,
            rows.len() * 180
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
