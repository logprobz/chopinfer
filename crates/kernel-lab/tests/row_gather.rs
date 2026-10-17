//! The shared gather by index (`kernels_sm120::row::gather`, lock family
//! `row_gather`) against its CPU oracle (`kernel_core::row_copy`) at the
//! row widths the bundles gather: the hidden rows a prefill scores before
//! the head, and the drafter's head rows.
//!
//! Each case gathers from a source of twice the bucket's rows by indices in
//! reverse order with repeats, at a 32-row bucket and every live count the
//! lab sweeps, gridded at the bucket and at the live count, into
//! destinations filled with two poison patterns: the shared op must write
//! its oracle's bytes over the whole destination.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test row_gather`; without a GPU the GPU test skips.

use std::collections::BTreeMap;

use kernel_core::backend::{Backend, Counts, Need, Prepared, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::row_copy::{self, GatherByIndex, GatherByIndexBuffers};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, poison};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::gather::{CUBIN, SYMBOL};
use runtime_cuda::DeviceBuffer;

/// The bucket every case plans for.
const CAPACITY: u32 = 32;
/// The source rows: indices reach past the bucket's rows.
const SOURCE_ROWS: u32 = 2 * CAPACITY;
/// The destination fills: the lab's poison and a second pattern.
const FILLS: [u8; 2] = [poison::POISON, 0x5a];
/// The row widths, in elements of each type: one vector, and the hidden
/// rows the bundles gather (2048 to 7168 BF16 columns).
const WIDTHS: [(DType, u32); 7] = [
    (DType::Bf16, 8),
    (DType::Bf16, 2048),
    (DType::Bf16, 4096),
    (DType::Bf16, 5120),
    (DType::Bf16, 6144),
    (DType::Bf16, 7168),
    (DType::F32, 4096),
];

fn text(refusal: Refusal) -> String {
    refusal.to_string()
}

fn op(dtype: DType, width: u32) -> GatherByIndex {
    GatherByIndex {
        dtype,
        width,
        alignment_bytes: 16,
    }
}

fn prepare(op: &GatherByIndex) -> LabResult<Prepared<GatherByIndexBuffers<View>>> {
    Sm120::new()
        .with_row_gather(SYMBOL)
        .gather_by_index(op, Need::rows(CAPACITY))
        .map_err(|refused| refused.to_string())
}

/// Indices in reverse order with every fourth row repeated: row `r`
/// gathers source row `SOURCE_ROWS - 1 - r`, or its predecessor's.
fn indices() -> Vec<u32> {
    (0..CAPACITY)
        .map(|r| SOURCE_ROWS - 1 - (r - u32::from(r % 4 == 3)))
        .collect()
}

/// Deterministic bytes.
fn noise(len: usize, seed: u64) -> Vec<u8> {
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

/// The prepared gather plans for every width; a launch's arguments fill
/// the entry's parameters and its grid is the bucket of its grid rows.
#[test]
fn the_shared_gather_serves_every_width_from_the_family_cubin() {
    let [spec] = &kernel_lab::families::row_gather::cubins()[..] else {
        panic!("the row gather family has one cubin");
    };
    assert_eq!(spec.name, CUBIN);
    assert_eq!(kernel_lab::cubin_entries(CUBIN).unwrap(), [SYMBOL]);
    for (dtype, width) in WIDTHS {
        let op = op(dtype, width);
        let prepared = prepare(&op).unwrap();
        let row = op.row_bytes().unwrap();
        let view = |handle, bytes: u64| View { handle, bytes };
        let buffers = GatherByIndexBuffers {
            source: view(1, row * u64::from(SOURCE_ROWS)),
            row_indices: view(2, u64::from(CAPACITY) * 4),
            count: view(3, 4),
            destination: view(4, row * u64::from(CAPACITY)),
        };
        let padded = Counts {
            padded_rows: CAPACITY,
            ..Counts::rows(5)
        };
        let launches = prepared.launches(&buffers, View::EMPTY, padded).unwrap();
        let [launch] = &launches[..] else {
            panic!("{dtype:?} {width}: one launch");
        };
        assert_eq!(launch.symbol, SYMBOL);
        assert_eq!(launch.arguments.0.len(), spec.entries[0].params.len());
        assert_eq!(launch.plan.grid, [CAPACITY, 1, 1], "{dtype:?} {width}");
        assert!(launch.plan.pdl);
    }
    // A row of a partial vector is refused when prepared.
    assert!(prepare(&op(DType::Bf16, 4100)).is_err());
}

/// The shared gather matches its oracle bit for bit at every live count,
/// gridded at the bucket and at the live count, leaving every byte past
/// the gathered rows as filled.
#[test]
fn the_shared_gather_matches_its_oracle_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.module(CUBIN)?;
        let function = module.function(SYMBOL).map_err(|e| e.to_string())?;
        let indices = indices();
        let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let index_buffer = upload(session, &index_bytes)?;
        let mut runs = 0;
        for (seed, (dtype, width)) in WIDTHS.into_iter().enumerate() {
            let op = op(dtype, width);
            let prepared = prepare(&op)?;
            let row = op.row_bytes().map_err(text)? as usize;
            let source = noise(row * SOURCE_ROWS as usize, seed as u64 + 1);
            let source_buffer = upload(session, &source)?;
            let destination_bytes = row * CAPACITY as usize;
            for live in poison::live_counts(CAPACITY) {
                let count_buffer = upload(session, &live.to_le_bytes())?;
                // Gridded at the bucket, and at the live count (a launch
                // of at least one row).
                for padded in [CAPACITY, live.max(1)] {
                    for fill in FILLS {
                        let what = format!(
                            "{dtype:?} {width} at live {live}, grid {padded}, fill {fill:#04x}"
                        );
                        let mut expected = vec![fill; destination_bytes];
                        row_copy::gather_by_index(&op, &source, &indices, live, &mut expected)
                            .map_err(text)?;
                        let destination = upload(session, &vec![fill; destination_bytes])?;
                        let view = |handle, buffer: &DeviceBuffer| View {
                            handle,
                            bytes: buffer.byte_len() as u64,
                        };
                        let buffers = GatherByIndexBuffers {
                            source: view(1, &source_buffer),
                            row_indices: view(2, &index_buffer),
                            count: view(3, &count_buffer),
                            destination: view(4, &destination),
                        };
                        let counts = Counts {
                            padded_rows: padded,
                            ..Counts::rows(live.max(1).min(padded))
                        };
                        let launches = prepared
                            .launches(&buffers, View::EMPTY, counts)
                            .map_err(text)?;
                        let bindings = Bindings {
                            buffers: BTreeMap::from([
                                (1, &source_buffer),
                                (2, &index_buffer),
                                (3, &count_buffer),
                                (4, &destination),
                            ]),
                        };
                        for one in &launches {
                            launch::launch(
                                &session.context,
                                &function,
                                &session.stream,
                                one.plan,
                                &one.arguments,
                                &bindings,
                            )?;
                        }
                        let got = download(session, &destination, destination_bytes)?;
                        if let Some(at) = got.iter().zip(&expected).position(|(a, b)| a != b) {
                            return Err(format!(
                                "{what}: byte {at} is {:#04x}, expected {:#04x}",
                                got[at], expected[at]
                            ));
                        }
                        runs += 1;
                    }
                }
            }
        }
        eprintln!("{runs} shared gather runs match their oracle");
        Ok(())
    })
    .expect("the shared gather against its oracle");
}
