//! The harness of the shared row copy tests (`row_copy.rs`): every row
//! copy the models launch at every shape they serve, the shared op
//! (`kernels_sm120::row::copy`, lock family `row_copy`) as the SM120
//! backend prepares it, and its CPU oracle (`kernel_core::row_copy`).
//!
//! [`check_shared`] runs each copy at a 32-row bucket and live counts 32,
//! 0, 1, 16 and 31, into destinations filled with two poison patterns: the
//! shared op must write its oracle's bytes over the whole destination.

#![allow(dead_code)]

use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, View};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
use kernel_core::row_copy::{self, ColumnSlice, RowCopyBuffers, StreamBroadcast};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, poison};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::copy::{self as sm120, LAB_CUBIN};
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::copies;
use model_mimo26_contract::Tp;
use runtime_cuda::{DeviceBuffer, Module};
use std::collections::BTreeMap;

/// The bucket every case plans for.
pub const CAPACITY: u32 = 32;
/// The view alignment every copy states: the lab's device allocations
/// start at 256-byte boundaries, and every view is a whole allocation.
pub const ALIGNMENT: u32 = 16;
/// The destination fills: the lab's poison and a second pattern, so a
/// kernel that writes a fill's own value where it should not still shows.
pub const FILLS: [u8; 2] = [poison::POISON, 0x5a];

pub fn text(refusal: Refusal) -> String {
    refusal.to_string()
}

/// The SM120 backend over the shared copy entries, as a model registers
/// them.
pub fn backend() -> Sm120 {
    sm120::entries()
        .into_iter()
        .fold(Sm120::new(), |backend, entry| backend.with_row_copy(entry))
}

/// A row copy a model launches today: the shared op and the model kernel it
/// would replace (the shared entry, where the model launches it already).
pub struct Copy {
    pub name: String,
    pub op: Op,
    pub model: Model,
}

/// A row copy through the shared op alone.
pub struct Shared {
    pub name: String,
    pub op: Op,
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Broadcast(StreamBroadcast),
    Slice(ColumnSlice),
}

impl Op {
    pub fn prepare(self) -> LabResult<Prepared<RowCopyBuffers<View>>> {
        let need = Need::rows(CAPACITY);
        match self {
            Op::Broadcast(op) => backend().broadcast_streams(&op, need),
            Op::Slice(op) => backend().column_slice(&op, need),
        }
        .map_err(|refused| refused.to_string())
    }

    /// The source and destination bytes of the bucket's rows.
    pub fn bytes(self) -> LabResult<(usize, usize)> {
        let (source, destination) = match self {
            Op::Broadcast(op) => (op.source_row_bytes(), op.destination_row_bytes()),
            Op::Slice(op) => (op.source_row_bytes(), op.destination_row_bytes()),
        };
        let rows = |row: u64| (u64::from(CAPACITY) * row) as usize;
        Ok((
            rows(source.map_err(text)?),
            rows(destination.map_err(text)?),
        ))
    }

    pub fn oracle(self, source: &[u8], destination: &mut [u8], live: u32) -> LabResult<()> {
        match self {
            Op::Broadcast(op) => row_copy::broadcast_streams(&op, source, destination, live),
            Op::Slice(op) => row_copy::column_slice(&op, source, destination, live),
        }
        .map_err(text)
    }
}

/// A model kernel as its bundle launches it: (source, destination, live
/// rows) and its scalars, over `plan`, from cubin `cubin`.
pub struct Model {
    pub cubin: String,
    pub symbol: String,
    pub plan: LaunchPlan,
    pub scalars: Vec<u32>,
}

fn plan(grid: [u32; 3], block: u32) -> LaunchPlan {
    LaunchPlan {
        grid,
        block: [block, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    }
}

/// The Qwen3.8 stream expansion: the embedding into every residual stream.
/// The bundle launches the shared broadcast itself (its cubin compiles the
/// shared entry), so its kernel is the shared entry.
pub fn qwen38() -> Vec<Copy> {
    let (hidden, streams) = (
        model_qwen38_contract::contract::HIDDEN,
        model_qwen38_contract::contract::STREAMS,
    );
    vec![Copy {
        name: format!("qwen38 expand streams {hidden}x{streams}"),
        op: Op::Broadcast(StreamBroadcast {
            dtype: DType::Bf16,
            width: hidden,
            streams,
            alignment_bytes: ALIGNMENT,
        }),
        model: Model {
            cubin: LAB_CUBIN.into(),
            symbol: sm120::BROADCAST_SYMBOL.into(),
            plan: plan([CAPACITY, 1, 1], sm120::THREADS),
            scalars: vec![
                (u64::from(hidden) * 2 / sm120::VECTOR_BYTES) as u32,
                streams,
            ],
        },
    }]
}

/// GLM's stream expansion and the router's scores of every expert without
/// its GEMM's padded columns (`copies`), which its bundle launches through
/// the shared copies.
pub fn glm5next() -> Vec<Shared> {
    let shape = Shape::TP4;
    let broadcast = copies::expand_streams(shape);
    let slice = copies::router_scores(shape);
    vec![
        Shared {
            name: format!("expand streams {}x{}", broadcast.width, broadcast.streams),
            op: Op::Broadcast(broadcast),
        },
        Shared {
            name: format!("compact scores {} of {}", slice.width, slice.stride),
            op: Op::Slice(slice),
        },
    ]
}

/// MiMo's MTP EH shards at every rank count (`Tp::mtp_eh_slice`), which
/// its bundle launches through the shared slice.
pub fn mimo26() -> Vec<Shared> {
    let mut slices = Vec::new();
    for tp in Tp::SUPPORTED {
        for rank in 0..tp.ranks() as usize {
            let slice = tp.mtp_eh_slice(rank);
            if !slices.contains(&slice) {
                slices.push(slice);
            }
        }
    }
    slices
        .into_iter()
        .map(|slice| Shared {
            name: format!(
                "EH shard {} at {} of {}",
                slice.width, slice.offset, slice.stride
            ),
            op: Op::Slice(slice),
        })
        .collect()
}

/// Every copy that replaces a model kernel.
pub fn all() -> LabResult<Vec<Copy>> {
    Ok(qwen38())
}

/// Every shared op the models launch or would launch.
pub fn shared() -> LabResult<Vec<Shared>> {
    let mut shared: Vec<_> = all()?
        .into_iter()
        .map(|copy| Shared {
            name: copy.name,
            op: copy.op,
        })
        .collect();
    shared.extend(glm5next());
    shared.extend(mimo26());
    Ok(shared)
}

/// Deterministic bytes.
pub fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; len];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// The shared op's launches over `source`, `destination` and `live`.
fn run_shared(
    session: &Session,
    module: &Module,
    op: &Prepared<RowCopyBuffers<View>>,
    [source, destination, live]: [&DeviceBuffer; 3],
) -> LabResult<Vec<Launch>> {
    let view = |handle, buffer: &DeviceBuffer| View {
        handle,
        bytes: buffer.byte_len() as u64,
    };
    let buffers = RowCopyBuffers {
        source: view(1, source),
        destination: view(2, destination),
        live_rows: view(3, live),
    };
    let launches = op
        .launches(&buffers, View::EMPTY, Counts::rows(CAPACITY))
        .map_err(text)?;
    let bindings = Bindings {
        buffers: BTreeMap::from([(1, source), (2, destination), (3, live)]),
    };
    for one in &launches {
        let function = module.function(&one.symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            one.plan,
            &one.arguments,
            &bindings,
        )?;
    }
    Ok(launches)
}

/// The first byte where `got` differs from `expected`, if any.
fn differs(got: &[u8], expected: &[u8], what: &str) -> LabResult<()> {
    if got.len() != expected.len() {
        return Err(format!(
            "{what}: {} bytes, expected {}",
            got.len(),
            expected.len()
        ));
    }
    match got.iter().zip(expected).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(at) => Err(format!(
            "{what}: byte {at} is {:#04x}, expected {:#04x}",
            got[at], expected[at]
        )),
    }
}

/// Runs each of `shared` at every live count and fill against its oracle.
pub fn check_shared(session: &Session, shared: &[Shared]) -> LabResult<usize> {
    let module = session.module(LAB_CUBIN)?;
    let mut runs = 0;
    for (index, copy) in shared.iter().enumerate() {
        let (name, op) = (copy.name.as_str(), copy.op);
        let prepared = op.prepare()?;
        let (source_bytes, destination_bytes) = op.bytes()?;
        let source = noise(source_bytes, index as u64 + 1);
        let source_buffer = upload(session, &source)?;
        for live in poison::live_counts(CAPACITY) {
            let live_buffer = upload(session, &live.to_le_bytes())?;
            for fill in FILLS {
                let what = format!("{name} at live {live}, fill {fill:#04x}");
                let mut expected = vec![fill; destination_bytes];
                op.oracle(&source, &mut expected, live)?;
                let destination = upload(session, &vec![fill; destination_bytes])?;
                let buffers = [&source_buffer, &destination, &live_buffer];
                run_shared(session, &module, &prepared, buffers)?;
                let got = download(session, &destination, destination_bytes)?;
                differs(&got, &expected, &format!("shared {what}"))?;
                runs += 1;
            }
        }
    }
    Ok(runs)
}
