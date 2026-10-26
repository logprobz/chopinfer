//! Compare a staged finalizer cubin before and after catalog pruning.
//! The stage holds base.cubin, head.cubin and symbol.txt, sealed by the job.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::DType;
use kernel_lab::poison::{self, LiveProducer, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{Geometry, lagged, lagged_split};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A};
use serde_json::json;
use std::collections::BTreeMap;

const CAPACITY: u32 = 32;
const STREAMS: usize = 4;
const WIDTH: usize = 5120;
const FLAT: usize = STREAMS * WIDTH;
const MIXES: usize = 24;
const CHAINS: usize = FLAT / 128;
const NAMES: [&str; 5] = ["collapsed", "pre", "post", "comb", "partials"];
const LENGTHS: [usize; 5] = [WIDTH, STREAMS, STREAMS, STREAMS * STREAMS, MIXES * CHAINS];

fn entry() -> lagged_split::Entry {
    lagged_split::entry(
        lagged::LAB_PROFILE,
        Geometry {
            capacity: CAPACITY,
            ..lagged::LAB
        },
        false,
    )
}

#[test]
fn finalizer_twin_covers_every_small_row_count() {
    let entry = entry();
    assert_eq!(entry.params().len(), 12);
    for rows in 1..=CAPACITY {
        for live in 0..=rows {
            assert_eq!(entry.plan(rows, live).unwrap().grid, [20, rows, 1]);
            for (index, length) in LENGTHS.into_iter().enumerate() {
                let dtype = if index == 0 { DType::Bf16 } else { DType::F32 };
                let map = WriteMap::of(dtype).prefix(live as usize * length).unwrap();
                assert_eq!(
                    map.written_bytes(),
                    live as usize * length * if index == 0 { 2 } else { 4 }
                );
            }
        }
    }
}

struct Fixture {
    x: DeviceBuffer,
    weights: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    pre: DeviceBuffer,
    live: DeviceBuffer,
    outputs: Vec<DeviceBuffer>,
    tickets: DeviceBuffer,
}

impl Fixture {
    fn new(s: &Session, rows: usize, pattern: usize) -> LabResult<Self> {
        let values = |count: usize, scale: f32| {
            (0..count)
                .map(|i| (((i * 73 + pattern * 37) % 257) as f32 - 128.0) * scale)
                .collect::<Vec<_>>()
        };
        let x = values(rows * FLAT, if pattern == 0 { 0.015625 } else { 2.0 });
        Ok(Self {
            x: s.upload_bf16(&x.into_iter().map(f32_to_bf16).collect::<Vec<_>>())?,
            weights: s.upload_f32(&values(MIXES * FLAT, 1.0 / 4096.0))?,
            base: s.upload_f32(&values(MIXES, 0.00390625))?,
            scale: s.upload_f32(&[0.01, 0.02, 0.03])?,
            pre: s.upload_f32(&values(rows * STREAMS, 0.0078125))?,
            live: s.allocate(4)?,
            outputs: LENGTHS
                .into_iter()
                .enumerate()
                .map(|(index, length)| s.allocate(rows * length * if index == 0 { 2 } else { 4 }))
                .collect::<LabResult<_>>()?,
            tickets: s.allocate(rows * 4)?,
        })
    }

    fn launch(&self, s: &Session, function: &Function, rows: u32, pdl: bool) -> LabResult<()> {
        let mut plan = entry().plan(rows, 0).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        function
            .launch(
                &s.stream,
                kernel_lab::launch::config(plan)?,
                &[
                    A::Buffer(&self.x),
                    A::Buffer(&self.weights),
                    A::Buffer(&self.base),
                    A::Buffer(&self.scale),
                    A::Buffer(&self.pre),
                    A::Buffer(&self.outputs[0]),
                    A::Buffer(&self.outputs[1]),
                    A::Buffer(&self.outputs[2]),
                    A::Buffer(&self.outputs[3]),
                    A::Buffer(&self.outputs[4]),
                    A::Buffer(&self.tickets),
                    A::Buffer(&self.live),
                    A::NullBuffer,
                ],
            )
            .map_err(|e| e.to_string())
    }

    fn replay<'a>(&'a self, s: &'a Session, rows: usize, live: usize) -> LabResult<Replay<'a>> {
        let mut replay = Replay::new(s);
        for (index, (name, length)) in NAMES.into_iter().zip(LENGTHS).enumerate() {
            let dtype = if index == 0 { DType::Bf16 } else { DType::F32 };
            replay = replay.output(
                name,
                &self.outputs[index],
                WriteMap::of(dtype).prefix(live * length)?,
            );
        }
        Ok(replay.state(
            "tickets",
            &self.tickets,
            vec![0; rows * 4],
            WriteMap::of(DType::I32).none()?,
            vec![],
        ))
    }
}

#[test]
#[ignore = "requires staged cubins named by CHOPINFER_HC_FINALIZER_TWINS and an idle GPU"]
fn finalizer_catalog_twin_on_poisoned_buffers() {
    let stage = std::path::PathBuf::from(
        std::env::var_os("CHOPINFER_HC_FINALIZER_TWINS").expect("CHOPINFER_HC_FINALIZER_TWINS"),
    );
    let symbol = std::fs::read_to_string(stage.join("symbol.txt")).unwrap();
    let cubins = ["base.cubin", "head.cubin"].map(|name| std::fs::read(stage.join(name)).unwrap());
    kernel_lab::gpu_test(1 << 30, |s| {
        let modules = cubins
            .iter()
            .map(|bytes| s.load_cubin(bytes))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .map(|module| module.function(symbol.trim()).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let producer = LiveProducer::new(s)?;
        let mut cases = 0;
        for rows in 1..=CAPACITY {
            for pattern in 0..2 {
                let f = Fixture::new(s, rows as usize, pattern)?;
                let image = s.allocate(f.x.byte_len())?;
                poison::write(s, &image, 0, &poison::read(s, &f.x)?)?;
                let rewrites = Rewrites::new(s, &[(&image, &f.x)])?;
                for live in 0..=rows {
                    poison::write(s, &f.live, 0, &live.to_ne_bytes())?;
                    let mut replay = f.replay(s, rows as usize, live as usize)?;
                    replay.eager(|| f.launch(s, &functions[0], rows, false))?;
                    for function in &functions {
                        replay.relaunch(|| f.launch(s, function, rows, false))?;
                        let graph = s.capture(|_| f.launch(s, function, rows, false))?;
                        replay.graph(&graph)?;
                        replay.pdl(
                            || producer.launch_rewriting(s, &f.live, &rewrites),
                            || f.launch(s, function, rows, true),
                            true,
                        )?;
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 1120);
        eprintln!("FINALIZER_TWIN_CASES {cases}");
        let receipt = BTreeMap::from([
            ("cases".into(), json!(cases)),
            ("rows".into(), json!((1..=CAPACITY).collect::<Vec<_>>())),
            ("poisoned_relaunch_graph_pdl_twins".into(), json!(true)),
        ]);
        s.receipt("sinkhorn_hc_finalizer_twin", receipt)?;
        Ok(())
    })
    .expect("finalizer catalog twin");
}
