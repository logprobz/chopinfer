//! Bounded two-shot worker grids with unchanged rank-ordered F32 sums.
use kernel_core::contract::{Argument, ArgumentList, CommQuery, DType, LaunchPlan};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::{LabResult, Session, gpu_test_multi};
use kernels_sm120::comm;
use runtime_cuda::{DeviceBuffer, GraphExec};
use std::collections::BTreeMap;

const WIDTH: usize = 5120;
const ROWS: [u32; 7] = [32, 33, 40, 48, 56, 63, 64];
const GUARD: usize = 256;
const POISON: u8 = 0xa5;
const CALLS: usize = 16;

fn plan(rows: u32) -> LabResult<LaunchPlan> {
    comm::plan_variant(
        comm::Variant::TwoShot,
        &CommQuery::AllReduce {
            ranks: 4,
            elements: u64::from(rows) * WIDTH as u64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        },
        rows,
    )
    .map_err(|e| e.to_string())
}
fn grids(planned: u32) -> [[u32; 2]; 6] {
    [
        [planned, planned],
        [planned, planned],
        [planned, 16],
        [32, 16],
        [24, 16],
        [16, 16],
    ]
}
fn value(rank: usize, i: usize, seed: u64) -> f32 {
    if i % 7 == 0 {
        return [16_777_216.0, 0.125, -16_777_216.0, 0.25][(rank + seed as usize % 4) % 4];
    }
    let mut x = (i as u64 ^ seed ^ (rank as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 29;
    let f = ((x % 2001) as f32 - 1000.) / 1000.;
    f * 2f32.powi(((x >> 20) % 32) as i32 - 16)
}

struct Fixture {
    elements: usize,
    plan: LaunchPlan,
    input: Vec<DeviceBuffer>,
    output: Vec<DeviceBuffer>,
    inbox: Vec<DeviceBuffer>,
    flags: Vec<DeviceBuffer>,
    epoch: Vec<DeviceBuffer>,
    sizes: [usize; 3],
    modules: CollectiveModules,
}
impl Fixture {
    fn new(sessions: &[Session], rows: u32) -> LabResult<Self> {
        let elements = rows as usize * WIDTH;
        let shape =
            comm::phased_buffers(4, elements as u64, DType::F32).map_err(|e| e.to_string())?;
        let allocate = |bytes: usize| {
            sessions
                .iter()
                .map(|s| s.allocate(bytes + GUARD))
                .collect::<LabResult<Vec<_>>>()
        };
        let sizes = [
            shape.inbox_bytes as usize,
            shape.flag_bytes as usize,
            shape.epoch_bytes as usize,
        ];
        let result = Self {
            elements,
            plan: plan(rows)?,
            input: allocate(elements * 4)?,
            output: allocate(elements * 4)?,
            inbox: allocate(sizes[0])?,
            flags: allocate(sizes[1])?,
            epoch: allocate(sizes[2])?,
            sizes,
            modules: CollectiveModules::load(sessions, "comm_sm120")?,
        };
        for (rank, s) in sessions.iter().enumerate() {
            result.inbox[rank]
                .copy_from_bytes(&vec![POISON; sizes[0] + GUARD], &s.stream)
                .map_err(|e| e.to_string())?;
            for (buffer, size) in [
                (&result.flags[rank], sizes[1]),
                (&result.epoch[rank], sizes[2]),
            ] {
                let mut bytes = vec![POISON; size + GUARD];
                bytes[..size].fill(0);
                buffer
                    .copy_from_bytes(&bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(result)
    }
    fn stage(&self, rank: usize, gather: bool) -> LabResult<RankCall<'_>> {
        let mut buffers = BTreeMap::from([
            (100, &self.input[rank]),
            (101, &self.output[rank]),
            (102, &self.epoch[rank]),
        ]);
        for peer in 0..4 {
            buffers.insert(peer as u64 + 1, &self.inbox[peer]);
            buffers.insert(peer as u64 + 11, &self.flags[peer]);
            buffers.insert(peer as u64 + 21, &self.output[peer]);
        }
        let args = if gather {
            ArgumentList(vec![
                Argument::DeviceBuffer(101),
                Argument::DevicePointerArray((21..25).collect()),
                Argument::DevicePointerArray((11..15).collect()),
                Argument::DeviceBuffer(102),
                Argument::U64(self.elements as u64),
                Argument::U32(rank as u32),
                Argument::U32(4),
            ])
        } else {
            comm::arguments(comm::OneShotCall {
                input: 100,
                output: 101,
                peer_inboxes: (1..5).collect(),
                peer_flags: (11..15).collect(),
                epoch: 102,
                elements: self.elements as u64,
                rank: rank as u32,
                ranks: 4,
            })
            .map_err(|e| e.to_string())?
        };
        Ok(RankCall {
            args,
            bindings: Bindings { buffers },
        })
    }
    fn launch(&self, sessions: &[Session], grids: [u32; 2], only: Option<usize>) -> LabResult<()> {
        for (gather, grid) in [(false, grids[0]), (true, grids[1])] {
            launch::launch_collective(
                sessions,
                &self.modules,
                if gather {
                    "chop_comm_allgather_scattered_f32_v1"
                } else {
                    "chop_comm_reduce_scatter_f32_v1"
                },
                LaunchPlan {
                    grid: [grid, 1, 1],
                    ..self.plan
                },
                only,
                |rank| self.stage(rank, gather),
            )?;
        }
        Ok(())
    }
    fn graphs(
        &self,
        sessions: &[Session],
        grids: [u32; 2],
        calls: usize,
    ) -> LabResult<Vec<GraphExec>> {
        sessions
            .iter()
            .enumerate()
            .map(|(rank, s)| {
                s.capture(|_| {
                    for _ in 0..calls {
                        self.launch(sessions, grids, Some(rank))?;
                    }
                    Ok(())
                })
            })
            .collect()
    }
    fn reset(&self, sessions: &[Session], seed: u64) -> LabResult<Vec<u32>> {
        for s in sessions {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        let values: Vec<Vec<f32>> = (0..4)
            .map(|rank| (0..self.elements).map(|i| value(rank, i, seed)).collect())
            .collect();
        let expected = (0..self.elements)
            .map(|i| {
                let mut sum = 0.0_f32;
                for rank in &values {
                    sum += rank[i];
                }
                sum.to_bits()
            })
            .collect();
        for (rank, s) in sessions.iter().enumerate() {
            let mut input: Vec<u8> = values[rank].iter().flat_map(|x| x.to_ne_bytes()).collect();
            input.extend([POISON; GUARD]);
            self.input[rank]
                .copy_from_bytes(&input, &s.stream)
                .map_err(|e| e.to_string())?;
            self.output[rank]
                .copy_from_bytes(&vec![POISON; self.elements * 4 + GUARD], &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(expected)
    }
    fn check(&self, sessions: &[Session], expected: &[u32]) -> LabResult<()> {
        for s in sessions {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        for (rank, s) in sessions.iter().enumerate() {
            let bytes = kernel_lab::poison::read(s, &self.output[rank])?;
            let actual: Vec<u32> = bytes[..self.elements * 4]
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            kernel_core::oracle::check_exact(&actual, expected)
                .map_err(|i| format!("rank {rank} differs at {i}"))?;
            for (buffer, size) in [
                (&self.input[rank], self.elements * 4),
                (&self.output[rank], self.elements * 4),
                (&self.inbox[rank], self.sizes[0]),
                (&self.flags[rank], self.sizes[1]),
                (&self.epoch[rank], self.sizes[2]),
            ] {
                let bytes = kernel_lab::poison::read(s, buffer)?;
                if bytes[size..].iter().any(|&b| b != POISON) {
                    return Err(format!("rank {rank} wrote a collective guard"));
                }
            }
        }
        Ok(())
    }
}

fn run(sessions: &[Session]) -> LabResult<()> {
    assert_eq!(sessions.len(), 4);
    assert!(
        sessions
            .iter()
            .all(|s| s.metadata["measurement_label"] == "exclusive-device")
    );
    for rows in ROWS {
        let f = Fixture::new(sessions, rows)?;
        let variants = grids(f.plan.grid[0]);
        let mut checked = 0;
        for (arm, &grid) in variants.iter().enumerate() {
            let expected = f.reset(sessions, arm as u64)?;
            f.launch(sessions, grid, None)?;
            f.check(sessions, &expected)?;
            checked += 1;
            let graphs = f.graphs(sessions, grid, 1)?;
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
                checked += 1;
            }
        }
        assert_eq!(checked, 30);
        let expected = f.reset(sessions, 51)?;
        let graphs = variants
            .iter()
            .map(|&grid| f.graphs(sessions, grid, CALLS))
            .collect::<LabResult<Vec<_>>>()?;
        let mut times = vec![Vec::new(); variants.len()];
        for round in 0..5 {
            let order: Vec<_> = if round % 2 == 0 {
                (0..variants.len()).collect()
            } else {
                (0..variants.len()).rev().collect()
            };
            for arm in order {
                let timing = kernel_harness::time_multi_graph(sessions, &graphs[arm], CALLS, 1, 3)?;
                times[arm].push(timing.critical_path.median_ms * 1000.0 / CALLS as f32);
                f.check(sessions, &expected)?;
            }
        }
        let record = serde_json::json!({"rows":rows,"width":WIDTH,"ranks":4,"grids":variants,"checks":checked,
            "timed_checks":30,"calls_per_graph":CALLS,"critical_us":times,"bitwise":true,"guards":true,"delayed_ranks":4});
        println!("COMM_WIDE_CASE {record}");
        for s in sessions {
            s.receipt(
                "comm-wide",
                BTreeMap::from([("result".into(), record.clone())]),
            )?;
        }
    }
    println!("COMM_WIDE_COMPLETE rows=7 variants=6 checks=210 timed_checks=210 bitwise=true");
    Ok(())
}

#[test]
fn wide_grids_preserve_clusters_and_observe_rank_order() {
    assert_eq!(plan(32).unwrap().grid, [48, 1, 1]);
    assert_eq!(plan(64).unwrap().grid, [64, 1, 1]);
    for rows in ROWS {
        let p = plan(rows).unwrap();
        for grid in grids(p.grid[0]) {
            assert!(
                grid.into_iter()
                    .all(|n| n > p.cluster[0] && n % p.cluster[0] == 0)
            );
        }
    }
    let ordered = (0..4).fold(0_f32, |sum, rank| sum + value(rank, 0, 0));
    let reversed = (0..4)
        .rev()
        .fold(0_f32, |sum, rank| sum + value(rank, 0, 0));
    assert_ne!(ordered.to_bits(), reversed.to_bits());
}

#[test]
#[ignore = "requires four allocated idle GPUs through the staged lab job"]
fn gpu_wide_worker_grids() {
    let ran = gpu_test_multi(512 << 20, run).unwrap();
    assert!(ran.is_some(), "four-rank collective test skipped");
}
