//! Wide query rotation with unchanged arithmetic, status and inactive rows.
use super::*;
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::latent_glue::LatentQueryRopeBuffers;

const WIDTHS: [u32; 8] = [32, 33, 40, 48, 49, 56, 63, 64];
const ROW: usize = 16 * 512;

fn launches(rows: u32) -> LabResult<[Launch; 2]> {
    let query = model_deepseek41_contract::latent_glue::query_rope(geometry(), 4)
        .map_err(|e| e.to_string())?;
    let view = |handle, bytes| View { handle, bytes };
    let buffers = LatentQueryRopeBuffers {
        query: view(1, u64::from(rows + 1) * ROW as u64 * 4),
        native_q: view(2, u64::from(rows + 1) * ROW as u64 * 2),
        cos_sin: view(3, u64::from(rows) * 64 * 4),
        live_rows: view(4, 4),
        status: view(5, u64::from(rows + 2) * 4),
        status_row_offset: 1,
    };
    let plain = glue::entries()
        .into_iter()
        .find(|e| e.kernel == kernels_sm120::latent::glue::Kernel::QueryRope)
        .unwrap();
    let mut result = Vec::new();
    for entry in [plain, candidate::query().binding()] {
        let symbol = entry.symbol.clone();
        let op = kernels_sm120::backend::Sm120::new()
            .with_latent_glue(entry)
            .latent_query_rope(&query, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let mut calls = op
            .launches(&buffers, View::default(), Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].symbol, symbol);
        result.push(calls.remove(0));
    }
    result
        .try_into()
        .map_err(|_| "query pair is incomplete".into())
}

fn input(rows: usize, mode: usize) -> Vec<Vec<u8>> {
    let mut rng = fixture::Rng(0x7f3a_9156_846c_1023 ^ rows as u64);
    let mut query = fixture::f32s((0..(rows + 1) * ROW).map(|_| rng.value()));
    let mut table = fixture::f32s((0..rows * 64).map(|i| if i % 2 == 0 { 0.8 } else { 0.6 }));
    let mut status = vec![0; (rows + 2) * 4];
    status[..4].fill(0xa5);
    status[(rows + 1) * 4..].fill(0x5a);
    let live = match mode {
        1 => 0,
        2 => rows - 1,
        _ => rows,
    };
    let last = (rows - 1) * ROW;
    let put = |data: &mut [u8], at: usize, value: f32| {
        data[at * 4..at * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    };
    match mode {
        2 | 4 => put(&mut query, last, f32::NAN),
        3 => put(&mut query, 0, f32::NAN),
        5 => {
            put(&mut query, last + 448, f32::MAX);
            put(&mut query, last + 449, -f32::MAX);
        }
        6 => status[rows * 4..rows * 4 + 4].copy_from_slice(&2_u32.to_ne_bytes()),
        7 => put(&mut table, (rows - 1) * 64, f32::NAN),
        8 => {
            let values = [
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
            ];
            for (i, word) in query.chunks_exact_mut(4).enumerate() {
                word.copy_from_slice(&values[i % values.len()].to_ne_bytes());
            }
        }
        9 => put(&mut query, last, f32::INFINITY),
        _ => (),
    }
    vec![
        query,
        vec![0xa5; (rows + 1) * ROW * 2],
        table,
        fixture::u32s([live as u32]),
        status,
    ]
}

fn check(session: &Session, modules: &Modules, rows: u32, timing: bool) -> LabResult<()> {
    let calls = launches(rows)?;
    let initial = input(rows as usize, 0);
    let buffers = initial
        .iter()
        .map(|v| session.allocate(v.len()))
        .collect::<LabResult<Vec<_>>>()?;
    let bindings = kernel_harness::launch::Bindings {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    let reset = |data: &[Vec<u8>], stream: &runtime_cuda::Stream| -> LabResult<()> {
        for (buffer, bytes) in buffers.iter().zip(data) {
            buffer
                .copy_from_bytes(bytes, stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    reset(&initial, &session.stream)?;
    let graphs = calls
        .iter()
        .map(|call| session.capture(|stream| modules.launch(session, stream, call, &bindings)))
        .collect::<LabResult<Vec<_>>>()?;
    for mode in 0..10 {
        let data = input(rows as usize, mode);
        let live = u32::from_ne_bytes(data[3][..4].try_into().unwrap()) as usize;
        let mut reference = None;
        for replay in 0..3 {
            for arm in 0..2 {
                reset(&data, &session.stream)?;
                if replay == 0 {
                    modules.launch(session, &session.stream, &calls[arm], &bindings)?;
                } else {
                    graphs[arm]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                }
                let mut output = data.clone();
                for i in [0, 1, 4] {
                    buffers[i]
                        .copy_to_bytes(&mut output[i], &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                for (i, stride) in [(0, ROW * 4), (1, ROW * 2)] {
                    if output[i][live * stride..] != data[i][live * stride..] {
                        return Err(format!(
                            "wide query wrote inactive rows: rows={rows} mode={mode} arm={arm}"
                        ));
                    }
                }
                if output[4][..4] != data[4][..4]
                    || output[4][(live + 1) * 4..] != data[4][(live + 1) * 4..]
                {
                    return Err("wide query changed a status guard or inactive row".into());
                }
                if matches!(mode, 3..=7 | 9) {
                    let row = if mode == 3 { 0 } else { rows as usize - 1 };
                    if output[0][row * ROW * 4..(row + 1) * ROW * 4]
                        != data[0][row * ROW * 4..(row + 1) * ROW * 4]
                        || output[1][row * ROW * 2..(row + 1) * ROW * 2]
                            .iter()
                            .any(|&b| b != 0)
                        || output[4][(row + 1) * 4..(row + 2) * 4] == [0; 4]
                    {
                        return Err("wide query failure protocol did not fire".into());
                    }
                }
                if let Some(expected) = &reference {
                    if &output != expected {
                        return Err(format!(
                            "wide query bits differ: rows={rows} mode={mode} arm={arm} replay={replay}"
                        ));
                    }
                } else {
                    reference = Some(output);
                }
            }
        }
    }
    let times = if timing {
        let pair = kernel_harness::time_ab_prepared(
            session,
            2,
            9,
            [kernel_harness::eviction::EvictionMode::Write; 2],
            |_, stream| reset(&initial, stream),
            |arm, stream| modules.launch(session, stream, &calls[arm], &bindings),
        )?;
        Some(pair.arms.map(|arm| arm.median_ms * 1000.0))
    } else {
        None
    };
    println!(
        "QUERY_WIDE_CASE {}",
        json!({"rows":rows,"witnesses":10,"comparisons":50,
        "graph_replays":2,"bitwise":true,"guards":true,"time_us":times})
    );
    Ok(())
}

#[test]
fn wide_query_pair_forces_the_head_parallel_entry() {
    for rows in WIDTHS {
        let [old, new] = launches(rows).unwrap();
        assert_ne!(old.symbol, new.symbol);
        assert_eq!(new.symbol, candidate::query().symbol);
        assert_eq!(old.arguments, new.arguments);
        assert_eq!(new.plan.grid, [rows, 1, 1]);
        assert_eq!(new.plan.block, [1024, 1, 1]);
    }
}

#[test]
#[ignore = "requires allocated idle GPUs through the staged pair job"]
fn gpu_wide_query_rope() {
    let small = std::env::var("CHOPINFER_QUERY_WIDE_MODE").as_deref() == Ok("memcheck");
    let widths = if small { vec![33, 64] } else { WIDTHS.to_vec() };
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 1 << 30, |session| {
        let modules = Modules::load(session)?;
        for &rows in &widths {
            check(session, &modules, rows, !small)?;
        }
        println!(
            "QUERY_WIDE_COMPLETE rows={} witnesses={} comparisons={} graph=true",
            widths.len(),
            widths.len() * 10,
            widths.len() * 50
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
