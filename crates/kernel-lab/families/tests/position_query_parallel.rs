//! Old/new output and sticky status comparisons, including graph replay.
use kernel_core::{
    backend::{Backend, Counts, Need},
    pooled_indexer::PositionQueryBuffers,
};
use kernel_harness::LabResult;
use kernel_lab_families::position_query_parallel as family;
use kernels_sm120::backend::Sm120;
use std::path::PathBuf;
#[path = "position_query_parallel/support.rs"]
mod support;
use support::*;

fn backend(parallel: bool) -> Sm120 {
    let b = Sm120::new()
        .with_position_queries([family::entry()])
        .with_block_quantize(family::quantizer());
    if parallel {
        b.with_parallel_position_queries()
    } else {
        b
    }
}

#[test]
fn cpu_query_parallel_buckets_and_abi() {
    let specs = family::cubins().remove(0);
    for rows in family::ROWS.into_iter().chain([33, 64]) {
        let old = backend(false)
            .position_indexer_query(&family::query(), Need::rows(rows))
            .unwrap();
        let new = backend(true)
            .position_indexer_query(&family::query(), Need::rows(rows))
            .unwrap();
        assert_eq!(old.info().guarantees, new.info().guarantees);
        assert_eq!(new.info().symbols.len(), if rows <= 32 { 1 } else { 3 });
        for (symbol, _) in &new.info().symbols {
            assert!(specs.entries.iter().any(|e| e.symbol == *symbol));
        }
    }
    assert_eq!(family::ROWS, [1, 2, 4, 6, 8, 16, 32]);
}

#[test]
fn cpu_query_prepared_capacity_preserves_the_launch_crossover_and_refusals() {
    use kernel_core::backend::View;
    let op = backend(true)
        .position_indexer_query(&family::query(), Need::rows(64))
        .unwrap();
    let v = |handle, bytes| View { handle, bytes };
    let b = PositionQueryBuffers {
        q: v(1, 64 * 4096 * 4),
        cos_sin: v(2, 8 * 64 * 4),
        positions: v(3, 64 * 4),
        live_rows: v(4, 4),
        output: v(5, 64 * 4096 * 2),
        quantizer_status: v(6, 64 * 4),
        quantizer_status_row_offset: 0,
    };
    let workspace = v(7, op.info().workspace);
    for rows in [1, 2, 4, 6, 8, 16, 32, 33, 64] {
        let calls = op
            .launches(
                &b,
                workspace,
                Counts {
                    positions: 8,
                    ..Counts::rows(rows)
                },
            )
            .unwrap();
        if rows <= 32 {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].plan.grid, [rows, 1, 1]);
            assert_eq!(calls[0].plan.block, [1024, 1, 1]);
        } else {
            assert_eq!(calls.len(), 2);
            assert!(calls[0].symbol.ends_with("rotate_v3"));
        }
    }
    for alias in [
        b.q,
        b.output,
        b.positions,
        b.cos_sin,
        b.live_rows,
        workspace,
    ] {
        assert!(
            op.launches(
                &PositionQueryBuffers {
                    quantizer_status: alias,
                    ..b
                },
                workspace,
                Counts {
                    positions: 8,
                    ..Counts::rows(1)
                }
            )
            .is_err()
        );
    }
    assert!(
        op.launches(
            &b,
            View {
                bytes: workspace.bytes - 1,
                ..workspace
            },
            Counts {
                positions: 8,
                ..Counts::rows(1)
            }
        )
        .is_err()
    );
}

#[test]
fn cpu_explicit_wide_query_keeps_the_default_crossover() {
    use kernel_core::backend::View;
    for limit in [0, 65, u32::MAX] {
        assert!(
            backend(false)
                .with_parallel_position_query_rows(limit)
                .is_err()
        );
    }
    let wide = backend(false)
        .with_parallel_position_query_rows(64)
        .unwrap();
    let op = wide
        .position_indexer_query(&family::query(), Need::rows(64))
        .unwrap();
    assert_eq!(op.info().symbols.len(), 1);
    let v = |handle, bytes| View { handle, bytes };
    let buffers = PositionQueryBuffers {
        q: v(1, 64 * 4096 * 4),
        cos_sin: v(2, 8 * 64 * 4),
        positions: v(3, 64 * 4),
        live_rows: v(4, 4),
        output: v(5, 64 * 4096 * 2),
        quantizer_status: v(6, 66 * 4),
        quantizer_status_row_offset: 1,
    };
    let scratch = v(7, op.info().workspace);
    let default = backend(true)
        .position_indexer_query(&family::query(), Need::rows(64))
        .unwrap();
    for rows in 1..=64 {
        let counts = Counts::rows(rows).with_positions(8);
        let calls = op.launches(&buffers, scratch, counts).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].plan.grid, [rows, 1, 1]);
        assert_eq!(calls[0].plan.block, [1024, 1, 1]);
        assert!(calls[0].symbol.ends_with("_parallel_v1"));
        let old = default.launches(&buffers, scratch, counts).unwrap();
        if rows <= 32 {
            assert_eq!(calls, old);
        } else {
            assert_eq!(old.len(), 2);
            assert!(old[0].symbol.ends_with("rotate_v3"));
        }
    }
}

fn inputs(rows: u32) -> Vec<f32> {
    let mut seed = 0x6ab2_c7e9_1243_8ef1_u64;
    (0..rows as usize * 4096)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let bits = ((seed >> 32) as u32 & 0x807f_ffff) | ((110 + i as u32 % 35) << 23);
            f32::from_bits(bits)
        })
        .collect()
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn cpu_query_fixture_has_rounding_and_wide_exponents() {
    use kernel_core::{
        codec::{bf16_to_f32, f32_to_bf16},
        precision::quantizer::BlockQuantization,
    };
    let values = inputs(1);
    assert!(
        values
            .iter()
            .filter(|&&v| bf16_to_f32(f32_to_bf16(v)) != v)
            .count()
            > 4000
    );
    let out = BlockQuantization::E2M1Ceil32
        .fake_quant(&values, 1, 4096)
        .unwrap();
    assert!(
        out.iter()
            .zip(values)
            .filter(|(a, b)| bf16_to_f32(**a) != *b)
            .count()
            > 4000
    );
}

#[test]
#[ignore = "requires the staged pair GPU job"]
fn gpu_query_parallel() {
    let ran = gpu_test(1 << 30, |s| {
        let root = PathBuf::from(std::env::var_os("CHOPINFER_QUERY_CUBINS").ok_or("query cubins required")?);
        let module = s.load_cubin(&std::fs::read(root.join(format!("{}.cubin",family::CUBIN))).map_err(|e|e.to_string())?)?;
        let shard: usize = std::env::var("CHOPINFER_L4_SHARD").map_err(|e|e.to_string())?.parse().map_err(|_| "invalid shard")?;
        if shard > 1 { return Err("invalid shard".into()); }
        let mut comparisons = 0;
        for (index, rows) in family::ROWS.into_iter().enumerate().filter(|(i,_)| i % 2 == shard) {
            let mut a = Arena {session:s, buffers:vec![]};
            let original = inputs(rows);
            let table: Vec<f32> = (0..8*32).flat_map(|i| [[0.80013,0.59991],[1.,0.],[-0.59993,0.80017]][i%3]).collect();
            let b = PositionQueryBuffers {
                q:a.input(&f32s(&original))?, cos_sin:a.input(&f32s(&table))?,
                positions:a.input(&i32s((0..rows).map(|r| (r%8) as i32)))?, live_rows:a.input(&rows.to_le_bytes())?,
                output:a.output(rows as usize*4096*2)?, quantizer_status:a.output((rows as usize+4)*4)?, quantizer_status_row_offset:2,
            };
            let mut calls = vec![];
            for parallel in [false,true] {
                let op = backend(parallel).position_indexer_query(&family::query(),Need::rows(rows)).map_err(|e|e.to_string())?;
                let scratch = a.output(op.info().workspace as usize)?;
                calls.push(op.launches(&b,scratch,Counts { positions:8,..Counts::rows(rows)}).map_err(|e|e.to_string())?);
            }
            let graphs = calls.iter().map(|c| s.capture(|_|fire(s,&module,&a,c))).collect::<LabResult<Vec<_>>>()?;
            // Exercise bad values in both the first and last head. A failed
            // row must clear all heads, including otherwise valid groups.
            for case in 0..16 {
                let mut values = original.clone();
                let mut pos: Vec<_> = (0..rows).map(|r|(r%8) as i32).collect();
                let mut status = vec![0xa5a5_a5a5_u32;rows as usize+4];
                status[2..2+rows as usize].fill(0);
                match case {
                    1 => values[0] = f32::NAN,
                    2 => values[4095] = f32::INFINITY,
                    3 => values[4094] = -f32::INFINITY,
                    4 => values[0] = f32::MAX,
                    5 => values[0] = f32::from_bits(0x7f7f_0000),
                    6 => { pos[0] = -1; values[..4096].fill(f32::NAN); }
                    7 => { pos[0] = 8; values[..4096].fill(f32::NAN); }
                    8 => status[2] = 128,
                    9 => { status[2] = 128; values[4095] = f32::NAN; }
                    10 => values[..4096].fill(-0.),
                    11 => values[..4096].fill(f32::MIN_POSITIVE),
                    12 => values[..4096].fill(f32::from_bits(0x0001_0000)),
                    13 => { values[0]=6.015625; values[1]=0.5; }
                    _ => (),
                }
                let live = match case {14 => 0,15 => rows-1,_ => rows};
                a.write(b.q,&f32s(&values))?; a.write(b.positions,&i32s(pos))?; a.write(b.live_rows,&live.to_le_bytes())?;
                let flags: Vec<u8> = status.iter().flat_map(|v|v.to_le_bytes()).collect();
                let mut expected = vec![];
                for (arm, graph) in graphs.iter().enumerate() {
                    a.poison(&[b.output])?; a.write(b.quantizer_status,&flags)?;
                    graph.launch(&s.stream).map_err(|e|e.to_string())?;
                    s.stream.synchronize().map_err(|e|e.to_string())?;
                    let actual = vec![a.read(b.output)?,a.read(b.quantizer_status)?];
                    if arm == 0 { expected = actual; } else { assert_eq!(expected,actual,"rows={rows} case={case}"); }
                }
                if case == 8 {
                    let mut mutant = calls[1].clone(); mutant[0].symbol = mutant[0].symbol.replace("_parallel_v1", "_parallel_ignore_status_v1");
                    a.poison(&[b.output])?; a.write(b.quantizer_status,&flags)?;
                    fire(s,&module,&a,&mutant)?;
                    assert_ne!(a.read(b.output)?,expected[0],"admission negative control escaped");
                }
                comparisons += 1;
            }
            a.write(b.q,&f32s(&original))?; a.write(b.live_rows,&rows.to_le_bytes())?;
            a.write(b.quantizer_status,&vec![0;(rows as usize+4)*4])?;
            let [old,new]=time_pair(s,&module,&a,[&calls[0],&calls[1]],5)?;
            println!("QUERY_CASE {}",serde_json::json!({"rows":rows,"old_ms":old,"new_ms":new,"comparisons":16,"negative_control_rejected":true,"ordinal":index,"exclusive":true}));
        }
        println!("QUERY_COMPLETE shard={shard} comparisons={comparisons}");
        Ok(())
    }).unwrap();
    assert!(ran.is_some(), "GPU test skipped");
}
