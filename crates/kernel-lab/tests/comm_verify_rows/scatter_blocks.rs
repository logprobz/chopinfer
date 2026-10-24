//! Match the ordinary scatter/gather while splitting only the last sender phase.
use super::*;
use kernel_lab::families::comm_scatter_blocks::CUBIN;
use kernels_sm120::comm::scatter_blocks::{BLOCKS, symbol};
const VARIANTS: [u32; 5] = [0, 0, 2, 4, 8];

fn launch_blocks(
    f: &Fixture,
    modules: &CollectiveModules,
    sessions: &[Session],
    blocks: u32,
    only: Option<usize>,
) -> LabResult<()> {
    if blocks == 0 {
        return f.launch(sessions, [f.plan.grid[0]; 2], only);
    }
    for gather in [false, true] {
        launch::launch_collective(
            sessions,
            if gather { &f.modules } else { modules },
            if gather {
                "chop_comm_allgather_scattered_f32_v1".into()
            } else {
                symbol(blocks)
            }
            .as_str(),
            f.plan,
            only,
            |rank| f.stage(rank, gather),
        )?;
    }
    Ok(())
}

fn graphs(
    f: &Fixture,
    modules: &CollectiveModules,
    sessions: &[Session],
    blocks: u32,
    calls: usize,
) -> LabResult<Vec<GraphExec>> {
    sessions
        .iter()
        .enumerate()
        .map(|(rank, s)| {
            s.capture(|_| {
                for _ in 0..calls {
                    launch_blocks(f, modules, sessions, blocks, Some(rank))?;
                }
                Ok(())
            })
        })
        .collect()
}

fn run(sessions: &[Session]) -> LabResult<()> {
    assert_eq!(sessions.len(), 4);
    assert!(
        sessions
            .iter()
            .all(|s| s.metadata["measurement_label"] == "exclusive-device")
    );
    let modules = CollectiveModules::load(sessions, CUBIN)?;
    for rows in ROWS {
        let f = Fixture::new(sessions, rows)?;
        let mut checks = 0;
        for (arm, blocks) in VARIANTS.into_iter().enumerate() {
            let expected = f.reset(sessions, arm as u64)?;
            launch_blocks(&f, &modules, sessions, blocks, None)?;
            f.check(sessions, &expected)?;
            checks += 1;
            let graphs = graphs(&f, &modules, sessions, blocks, 1)?;
            for late in 0..4 {
                let expected = f.reset(sessions, (arm * 4 + late + 10) as u64)?;
                for rank in 0..4 {
                    if rank != late {
                        graphs[rank]
                            .launch(&sessions[rank].stream)
                            .map_err(|e| e.to_string())?;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                graphs[late]
                    .launch(&sessions[late].stream)
                    .map_err(|e| e.to_string())?;
                f.check(sessions, &expected)?;
                checks += 1;
            }
        }
        assert_eq!(checks, 25);
        let expected = f.reset(sessions, 51)?;
        let graphs = VARIANTS
            .into_iter()
            .map(|blocks| graphs(&f, &modules, sessions, blocks, CALLS))
            .collect::<LabResult<Vec<_>>>()?;
        let mut times = vec![Vec::new(); VARIANTS.len()];
        for round in 0..5 {
            let order: Vec<_> = if round % 2 == 0 {
                (0..VARIANTS.len()).collect()
            } else {
                (0..VARIANTS.len()).rev().collect()
            };
            for arm in order {
                let timing = kernel_harness::time_multi_graph(sessions, &graphs[arm], CALLS, 1, 3)?;
                times[arm].push(timing.critical_path.median_ms * 1000. / CALLS as f32);
                f.check(sessions, &expected)?;
            }
        }
        let record = serde_json::json!({"rows":rows,"width":WIDTH,"ranks":4,"blocks":VARIANTS,"grid":f.plan.grid[0],"checks":checks,"timed_checks":25,"calls_per_graph":CALLS,"critical_us":times,"bitwise":true,"guards":true,"delayed_ranks":4});
        println!("COMM_BLOCKS_CASE {record}");
        for s in sessions {
            s.receipt(
                "comm-blocks",
                BTreeMap::from([("result".into(), record.clone())]),
            )?;
        }
    }
    println!("COMM_BLOCKS_COMPLETE rows=7 variants=5 checks=175 timed_checks=175 bitwise=true");
    Ok(())
}

#[test]
fn blocked_scatter_keeps_the_existing_abi_and_flag_extent() {
    let old = comm::cubin_spec();
    let reference = old
        .entries
        .iter()
        .find(|e| e.symbol == "chop_comm_reduce_scatter_f32_v1")
        .unwrap();
    let spec = comm::scatter_blocks::cubin_spec(CUBIN);
    assert_eq!(spec.entries.len(), BLOCKS.len());
    for (entry, blocks) in spec.entries.iter().zip(BLOCKS) {
        assert_eq!(entry.params, reference.params);
        assert_eq!(entry.max_threads, reference.max_threads);
        assert_eq!(entry.symbol, symbol(blocks));
        for rows in ROWS {
            let tiles = (u64::from(rows) * WIDTH as u64).div_ceil(4096);
            for rank in 0..4 {
                let count = (rank + 1) * tiles / 4 - rank * tiles / 4;
                let parts = count.min(u64::from(blocks)).max(1);
                let spans: Vec<_> = (0..parts)
                    .map(|b| (b * count / parts, (b + 1) * count / parts))
                    .collect();
                assert_eq!(spans[0].0, 0);
                assert_eq!(spans.last().unwrap().1, count);
                assert!(spans.windows(2).all(|s| s[0].1 == s[1].0));
                assert!(parts <= 8);
            }
        }
    }
}

#[test]
#[ignore = "requires four allocated idle GPUs through the staged lab job"]
fn gpu_last_phase_blocks() {
    let ran = gpu_test_multi(512 << 20, run).unwrap();
    assert!(ran.is_some(), "four-rank collective test skipped");
}
