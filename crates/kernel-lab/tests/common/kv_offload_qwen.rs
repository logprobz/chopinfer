//! Fixtures mirrored from qwen38_comm_decode.rs. Compare the decode entry
//! with the general entry at the selected live row counts, before and after
//! the interference measurements.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{DType, SigmoidMeanHyperConnectionQuery};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session, launch};
use model_qwen38_contract::sm120::comm::decode::{LL_MAX_ROWS, symbol_v2, symbol_v3};
use model_qwen38_contract::sm120::comm::{self as qwen38, decode as qwen38_decode};
use runtime_cuda::{DeviceBuffer, Function, GraphExec};

use std::collections::BTreeMap;

const WIDTH: usize = 2560;
const WIDE: usize = WIDTH * 4;
const ROWS: usize = 32;
const SENTINEL: u16 = 0x55aa;

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}

fn bytes(s: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(data.len())?;
    b.copy_from_bytes(data, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(b)
}

struct Lab<'a> {
    sessions: &'a [Session],
    fixtures: Vec<BTreeMap<u64, DeviceBuffer>>,
    general: Vec<Function>,
    /// Each rank's decode entries: v2, and v3 at the LL prepared rows.
    decode: Vec<[Function; 2]>,
    query: SigmoidMeanHyperConnectionQuery,
}

fn destinations(call: usize) -> [u64; 2] {
    if call == 0 {
        [114, 115]
    } else {
        [1000 + call as u64 * 2, 1001 + call as u64 * 2]
    }
}

fn checked(
    sessions: &[Session],
    replays: &mut [Replay<'_>],
    phase: Phase,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for replay in replays.iter() {
        replay.prepare(phase)?;
    }
    launch()?;
    poison::synchronize(sessions)?;
    let mut misses = Vec::new();
    for (rank, replay) in replays.iter_mut().enumerate() {
        if let Err(error) = replay.finish(phase, true) {
            misses.push(format!("rank {rank}: {error}"));
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}

impl Lab<'_> {
    fn bindings(&self, rank: usize, inbox: u64, flags: u64) -> launch::Bindings<'_> {
        let mut buffers: BTreeMap<_, _> =
            self.fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
        for (peer, fixture) in self.fixtures.iter().enumerate() {
            buffers.insert(peer as u64 + 1, &fixture[&inbox]);
            buffers.insert(peer as u64 + 11, &fixture[&flags]);
        }
        launch::Bindings { buffers }
    }

    fn general(&self, rank: usize) -> LabResult<()> {
        let args = qwen38::arguments(qwen38::Call {
            input: 100,
            residual: 101,
            logits: 102,
            weight: 103,
            residual_out: 104,
            normed: 105,
            inboxes: (1..=4).collect(),
            flags: (11..=14).collect(),
            epoch: 108,
            live_rows: 109,
            capacity: ROWS as u32,
            rank: rank as u32,
        })
        .map_err(|e| e.to_string())?;
        let mut plan = qwen38::plan(self.query, ROWS as u32).map_err(|e| e.to_string())?;
        plan.pdl = false;
        let s = &self.sessions[rank];
        launch::launch(
            &s.context,
            &self.general[rank],
            &s.stream,
            plan,
            &args,
            &self.bindings(rank, 106, 107),
        )
    }

    fn decode(&self, rank: usize, rows: u32, pdl: bool, ids: [u64; 2]) -> LabResult<()> {
        let args = qwen38_decode::arguments(qwen38_decode::Call {
            input: 100,
            residual: 101,
            logits: 102,
            weight: 103,
            residual_out: ids[0],
            normed: ids[1],
            inboxes: (1..=4).collect(),
            flags: (11..=14).collect(),
            epochs: 118,
            live_rows: 109,
            rows,
            rank: rank as u32,
        })
        .map_err(|e| e.to_string())?;
        let mut plan = qwen38_decode::plan(rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let s = &self.sessions[rank];
        launch::launch(
            &s.context,
            &self.decode[rank][usize::from(rows <= LL_MAX_ROWS)],
            &s.stream,
            plan,
            &args,
            &self.bindings(rank, 116, 117),
        )
    }

    fn sync(&self) -> LabResult<()> {
        for s in self.sessions {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn set_live(&self, live: u32) -> LabResult<()> {
        for (r, s) in self.sessions.iter().enumerate() {
            self.fixtures[r][&109]
                .copy_from_bytes(&live.to_ne_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
        }
        self.sync()
    }

    fn outputs(&self, rank: usize, ids: [u64; 2]) -> LabResult<[Vec<u16>; 2]> {
        let s = &self.sessions[rank];
        Ok([
            s.download_bf16(&self.fixtures[rank][&ids[0]], ROWS * WIDE)?,
            s.download_bf16(&self.fixtures[rank][&ids[1]], ROWS * WIDE)?,
        ])
    }

    /// Decode outputs equal the general entry's on the live rows and leave
    /// the rest untouched, on every rank.
    fn check(&self, live: usize, what: &str, ids: [u64; 2]) -> LabResult<()> {
        for rank in 0..4 {
            let expected = self.outputs(rank, [104, 105])?;
            let actual = self.outputs(rank, ids)?;
            for (e, a) in expected.iter().zip(&actual) {
                if a[..live * WIDE] != e[..live * WIDE] {
                    return Err(format!(
                        "{what}: rank {rank} differs from the general entry"
                    ));
                }
                if a[live * WIDE..].iter().any(|&v| v != 0xffff) {
                    return Err(format!("{what}: rank {rank} wrote a dead row"));
                }
            }
        }
        Ok(())
    }

    fn checks(&self, rows: usize, calls: usize, reference: bool) -> LabResult<Vec<Replay<'_>>> {
        self.sessions
            .iter()
            .enumerate()
            .map(|(rank, s)| {
                let mut replay = Replay::new(s);
                let ids = if reference {
                    vec![104, 105]
                } else {
                    (0..calls).flat_map(destinations).collect()
                };
                for id in ids {
                    replay = replay.output(
                        format!("bank destination {id}"),
                        &self.fixtures[rank][&id],
                        WriteMap::of(DType::Bf16).prefix(rows * WIDE)?,
                    );
                }
                Ok(replay)
            })
            .collect()
    }

    fn graphs(&self, rows: u32, calls: usize) -> LabResult<Vec<GraphExec>> {
        self.sessions
            .iter()
            .enumerate()
            .map(|(r, s)| {
                s.capture(|_| {
                    for call in 0..calls {
                        self.decode(r, rows, true, destinations(call))?;
                    }
                    Ok(())
                })
            })
            .collect()
    }
}

pub fn measure(
    sessions: &[Session],
    mut run: impl FnMut(&str, usize, &[GraphExec]) -> LabResult<()>,
) -> LabResult<()> {
    let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: 4,
        streams: 4,
        width: WIDTH as u32,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: ROWS as u32,
    };
    let general_sizes = qwen38::buffers(query).map_err(|e| e.to_string())?;
    let decode_sizes = qwen38_decode::buffers(4).map_err(|e| e.to_string())?;
    let x = codes(ROWS * WIDE, 1);
    let logits = codes(ROWS * 4, 2);
    let weight = codes(WIDE, 3);
    let mut fixtures = Vec::new();
    let mut general = Vec::new();
    let mut decode = Vec::new();
    for (rank, s) in sessions.iter().enumerate() {
        let mut b = BTreeMap::new();
        let partial = codes(ROWS * WIDTH, rank + 4);
        for (id, values) in [(100, &partial), (101, &x), (102, &logits), (103, &weight)] {
            b.insert(id, s.upload_bf16(values)?);
        }
        for id in [104, 105, 114, 115] {
            b.insert(id, s.upload_bf16(&vec![SENTINEL; ROWS * WIDE])?);
        }
        for call in 1..super::CALLS {
            for id in destinations(call) {
                b.insert(id, s.allocate(ROWS * WIDE * 2)?);
            }
        }
        b.insert(106, s.allocate(general_sizes.inbox_bytes as usize)?);
        b.insert(107, bytes(s, &vec![0; general_sizes.flags_bytes as usize])?);
        b.insert(108, bytes(s, &vec![0; general_sizes.epoch_bytes as usize])?);
        b.insert(109, bytes(s, &[0; 4])?);
        // Only the LL region must start zeroed; the rest holds garbage.
        let mut inbox = vec![0xff; decode_sizes.inbox_bytes as usize];
        inbox[..decode_sizes.zeroed_inbox_bytes as usize].fill(0);
        b.insert(116, bytes(s, &inbox)?);
        b.insert(117, bytes(s, &vec![0; decode_sizes.flags_bytes as usize])?);
        b.insert(118, bytes(s, &vec![0; decode_sizes.epoch_bytes as usize])?);
        fixtures.push(b);
        let load = |image: &[u8], symbol: &str| -> LabResult<Function> {
            s.context
                .load_module(image)
                .map_err(|e| e.to_string())?
                .function(symbol)
                .map_err(|e| e.to_string())
        };
        general.push(load(
            include_bytes!(concat!(env!("OUT_DIR"), "/hc-prefill/comm_qwen38.cubin")),
            qwen38::SYMBOL,
        )?);
        let image = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/hc-prefill/comm_qwen38_decode.cubin"
        ));
        decode.push([
            load(image, symbol_v2(4).map_err(|e| e.to_string())?)?,
            load(image, symbol_v3(4).map_err(|e| e.to_string())?)?,
        ]);
    }
    let lab = Lab {
        sessions,
        fixtures,
        general,
        decode,
        query,
    };
    lab.sync()?;

    let mut misses = Vec::new();
    for rows in [1, 32] {
        let checked = (|| -> kernel_lab::LabResult<()> {
            lab.set_live(rows)?;
            let mut general_checks = lab.checks(rows as usize, 1, true)?;
            checked(sessions, &mut general_checks, Phase::Eager, || {
                for rank in 0..4 {
                    lab.general(rank)?;
                }
                Ok(())
            })?;
            let mut replays = lab.checks(rows as usize, super::CALLS, false)?;
            checked(sessions, &mut replays, Phase::Eager, || {
                for call in 0..super::CALLS {
                    for rank in 0..4 {
                        lab.decode(rank, rows, false, destinations(call))?;
                    }
                }
                Ok(())
            })?;
            for call in 0..super::CALLS {
                lab.check(rows as usize, "KVO eager", destinations(call))?;
            }
            let graphs = lab.graphs(rows, super::CALLS)?;
            let launch_graphs = || {
                for (s, graph) in sessions.iter().zip(&graphs) {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            checked(sessions, &mut replays, Phase::Graph, launch_graphs)?;
            run("qwen_decode", rows as usize, &graphs)?;
            lab.check(rows as usize, "KVO interference", destinations(0))?;
            checked(sessions, &mut replays, Phase::Graph, launch_graphs)?;
            checked(sessions, &mut replays, Phase::Pdl, launch_graphs)?;
            for call in 0..super::CALLS {
                lab.check(rows as usize, "KVO replay", destinations(call))?;
            }
            let evidence = replays
                .iter()
                .map(|replay| {
                    let mut fields = BTreeMap::new();
                    replay.record(&mut fields);
                    fields
                })
                .collect::<Vec<_>>();
            sessions[0].receipt(
                "offload-compute-replay",
                BTreeMap::from([
                    ("rows".into(), serde_json::json!(rows)),
                    ("calls_per_graph".into(), serde_json::json!(super::CALLS)),
                    ("poisoned_ranks".into(), serde_json::json!(evidence)),
                ]),
            )?;

            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("rows={rows}: {error}");
            misses.push(error);
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}
