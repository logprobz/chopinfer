//! Compare sealed pre-template and mode-template modules on poisoned replays.
//! The staged manifest names each symbol and its original storage geometry.

use kernel_core::{codec::f32_to_bf16, contract::DType};
use kernel_lab::poison::{self, LiveProducer, Replay, Rewrites, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::latent::sources_prefill::heads;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A};
use serde::Deserialize;
use std::path::Path;

const WIDTH: usize = 512;
const MAX_ROWS: u32 = 8192;
const KEYS: usize = 65;

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len())?;
    poison::write(s, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn upload_i32(s: &Session, values: &[i32]) -> LabResult<DeviceBuffer> {
    upload(
        s,
        &values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}

#[derive(Deserialize)]
struct Entry {
    cubin: String,
    symbol: String,
    heads: usize,
    page: usize,
    decoded: bool,
    finish: bool,
}

fn row_range(value: &str) -> (u32, u32) {
    let (first, last) = value.split_once(':').expect("FIRST:LAST row range");
    let (first, last) = (first.parse().unwrap(), last.parse().unwrap());
    assert!(first > 0 && first <= last && last <= MAX_ROWS);
    (first, last)
}

#[test]
fn head_twins_cover_every_prefill_row_and_mode() {
    let entries = kernel_lab::families::latent_head_modes::entries();
    assert_eq!(entries.len(), 12);
    for entry in &entries {
        for rows in 1..=MAX_ROWS {
            let plan = heads::plan(entry, rows).unwrap();
            assert_eq!(plan.grid[0], rows);
            assert_eq!(plan.block, [256, 1, 1]);
        }
    }
    let specs = kernel_lab::families::latent_head_modes::cubins();
    assert_eq!(specs.len(), 6);
    assert_eq!(specs.iter().map(|s| s.entries.len()).sum::<usize>(), 60);
    assert_eq!(row_range("1:8192"), (1, MAX_ROWS));
}

struct Fixture {
    audit: bool,
    q: DeviceBuffer,
    batch: DeviceBuffer,
    live: DeviceBuffer,
    window: DeviceBuffer,
    scales: DeviceBuffer,
    slots: DeviceBuffer,
    keys: DeviceBuffer,
    pool: DeviceBuffer,
    compressed: DeviceBuffer,
    tables: DeviceBuffer,
    partial: DeviceBuffer,
    sinks: DeviceBuffer,
    rotation: DeviceBuffer,
    out: DeviceBuffer,
}

impl Fixture {
    fn new(s: &Session, e: &Entry, rows: usize, holes: bool, audit: bool) -> LabResult<Self> {
        let bf16 = |n: usize| {
            (0..n)
                .map(|i| f32_to_bf16(((i * 73 % 127) as f32 - 63.0) / 128.0))
                .collect::<Vec<_>>()
        };
        let page_stride = Self::page_stride(e);
        let pool = if e.decoded {
            s.upload_bf16(&bf16(2 * e.page * WIDTH))?
        } else {
            let mut bytes = vec![0x32; 2 * page_stride];
            for page in 0..2 {
                bytes[page * page_stride + e.page * WIDTH / 2..(page + 1) * page_stride].fill(0x38);
            }
            upload(s, &bytes)?
        };
        let keys = (0..rows * KEYS)
            .map(|i| {
                if holes || i % 7 == 0 {
                    -1
                } else {
                    (i % 4) as i32
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            audit,
            q: s.upload_bf16(&bf16(rows * e.heads * WIDTH))?,
            batch: upload_i32(s, &(0..rows).map(|i| (i % 2) as i32).collect::<Vec<_>>())?,
            live: s.allocate(4)?,
            window: if e.decoded {
                s.upload_bf16(&bf16(8 * WIDTH))?
            } else {
                upload(
                    s,
                    &(0..8 * WIDTH)
                        .map(|i| [0x30, 0xb8, 0x38][i % 3])
                        .collect::<Vec<_>>(),
                )?
            },
            scales: upload(s, &[124; 8 * WIDTH / 32])?,
            slots: upload_i32(s, &[1, 0])?,
            keys: upload_i32(s, &keys)?,
            pool,
            compressed: upload_i32(s, &keys)?,
            tables: upload_i32(s, &[1, 0])?,
            partial: s.allocate(rows * e.heads * (WIDTH + 2) * 4)?,
            sinks: s.upload_f32(&vec![0.125; e.heads])?,
            rotation: s.upload_f32(&[0.6, 0.8].repeat(rows * 32))?,
            out: s.allocate(rows * e.heads * WIDTH * 2)?,
        })
    }

    fn page_stride(e: &Entry) -> usize {
        e.page
            * if e.decoded {
                WIDTH * 2
            } else {
                WIDTH / 2 + WIDTH / 16
            }
    }

    fn launch(&self, s: &Session, f: &Function, e: &Entry, rows: u32, pdl: bool) -> LabResult<()> {
        let mut args = vec![
            A::Buffer(&self.q),
            A::Buffer(&self.batch),
            A::Buffer(&self.live),
            A::Buffer(&self.window),
            A::Buffer(&self.scales),
            A::Buffer(&self.slots),
            A::Buffer(&self.keys),
            if self.audit {
                A::Buffer(&self.partial)
            } else {
                A::NullBuffer
            },
            A::U64((4 * WIDTH * if e.decoded { 2 } else { 1 }) as u64),
            A::U64((4 * WIDTH / 32) as u64),
            A::U32(4),
            A::U32(2),
            A::U32(2),
            A::U32(KEYS as u32),
            A::U32(KEYS as u32),
            A::U32((1.0 / 512.0_f32.sqrt()).to_bits()),
            A::Buffer(&self.pool),
            A::Buffer(&self.compressed),
            A::Buffer(&self.tables),
            A::U64(Self::page_stride(e) as u64),
            A::U32(1),
            A::U32(2),
            A::U32(2),
            A::U32(KEYS as u32),
            A::U32(KEYS as u32),
        ];
        if e.finish {
            args.extend([
                A::Buffer(&self.sinks),
                A::Buffer(&self.rotation),
                A::Buffer(&self.out),
            ]);
        }
        let plan = kernel_core::contract::LaunchPlan {
            grid: [rows, e.heads as u32 / 16, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: heads::SHARED_BYTES,
            pdl,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
        };
        f.launch(&s.stream, kernel_lab::launch::config(plan)?, &args)
            .map_err(|e| e.to_string())
    }

    fn replay<'a>(&'a self, s: &'a Session, e: &Entry, live: usize) -> LabResult<Replay<'a>> {
        let replay = Replay::new(s).output(
            "partials",
            &self.partial,
            WriteMap::of(DType::F32).prefix(if self.audit {
                live * e.heads * (WIDTH + 2)
            } else {
                0
            })?,
        );
        Ok(if e.finish {
            replay.output(
                "output",
                &self.out,
                WriteMap::of(DType::Bf16).prefix(live * e.heads * WIDTH)?,
            )
        } else {
            replay
        })
    }
}

#[test]
#[ignore = "requires sealed CHOPINFER_LATENT_TWINS, ENTRY, ROWS and an idle GPU"]
fn latent_heads_match_before_and_after_on_poisoned_replays() {
    let stage = std::env::var("CHOPINFER_LATENT_TWINS").expect("CHOPINFER_LATENT_TWINS");
    let stage = Path::new(&stage);
    let entries: Vec<Entry> =
        serde_json::from_slice(&std::fs::read(stage.join("entries.json")).unwrap()).unwrap();
    assert_eq!(entries.len(), 30);
    let index: usize = std::env::var("CHOPINFER_LATENT_TWIN_ENTRY")
        .unwrap()
        .parse()
        .unwrap();
    let e = &entries[index];
    assert!([16, 32, 64].contains(&e.heads) && [32, 64].contains(&e.page));
    let (first, last) = row_range(&std::env::var("CHOPINFER_LATENT_TWIN_ROWS").unwrap());
    let cubins = ["base", "head"]
        .map(|arm| std::fs::read(stage.join(arm).join(format!("{}.cubin", e.cubin))).unwrap());
    kernel_lab::gpu_test(4 << 30, |s| {
        let modules = cubins
            .iter()
            .map(|c| s.load_cubin(c))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .map(|m| m.function(&e.symbol).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        for function in &functions {
            function
                .set_max_dynamic_shared_memory(heads::SHARED_BYTES)
                .map_err(|e| e.to_string())?;
        }
        let producer = LiveProducer::new(s)?;
        let mut cases = 0;
        for rows in first..=last {
            for holes in [false, true] {
                for &audit in if e.finish {
                    &[false, true][..]
                } else {
                    &[true][..]
                } {
                    let f = Fixture::new(s, e, rows as usize, holes, audit)?;
                    let image = s.allocate(f.q.byte_len())?;
                    poison::write(s, &image, 0, &poison::read(s, &f.q)?)?;
                    let rewrites = Rewrites::new(s, &[(&image, &f.q)])?;
                    for live in [0, rows / 2, rows] {
                        poison::write(s, &f.live, 0, &live.to_ne_bytes())?;
                        let mut replay = f.replay(s, e, live as usize)?;
                        replay.eager(|| f.launch(s, &functions[0], e, rows, false))?;
                        for function in &functions {
                            replay.relaunch(|| f.launch(s, function, e, rows, false))?;
                            let graph = s.capture(|_| f.launch(s, function, e, rows, false))?;
                            replay.graph(&graph)?;
                            replay.pdl(
                                || producer.launch_rewriting(s, &f.live, &rewrites),
                                || f.launch(s, function, e, rows, true),
                                true,
                            )?;
                        }
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, (last - first + 1) * if e.finish { 12 } else { 6 });
        eprintln!("LATENT_TWIN_CASES entry={index} first={first} last={last} cases={cases}");
        Ok(())
    })
    .expect("latent head mode twins");
}
