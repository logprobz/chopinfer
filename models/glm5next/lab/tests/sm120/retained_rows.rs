//! GLM's ordered hidden-row copies, including strided runs and DMA fallback.

use bundle_glm5next::drafter::{RETAINED_ROWS, copies};
use bundle_runtime::drafter::Retain;
use kernel_core::contract::Argument;
use kernel_harness::{LabResult, Session, launch, poison};
use kernels_sm120::row::copy::SLICE_SYMBOL;
use model_glm5next_contract::contract::Shape;
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
use runtime_cuda::{DeviceBuffer, KernelArgument, Stream};

const ROW: usize = 4096 * 2;

fn cases() -> Vec<Vec<Retain>> {
    let mut cases = Vec::new();
    for rows in [1, 2, 3, 4, 8, 16, 32] {
        for position in 0..RETAINED_ROWS {
            cases.push(
                (0..rows)
                    .map(|row| Retain {
                        phase: 1,
                        staged: row,
                        retained: row * RETAINED_ROWS + position,
                    })
                    .collect(),
            );
        }
        cases.push(
            (0..rows)
                .map(|row| Retain {
                    phase: 0,
                    staged: row * 16 + 15,
                    retained: row,
                })
                .collect(),
        );
    }
    cases.push(
        [5, 0, 1, 2, 3, 4, 5, 0]
            .into_iter()
            .enumerate()
            .map(|(i, retained)| Retain {
                phase: 1,
                staged: i as u32,
                retained,
            })
            .collect(),
    );
    cases.push(
        (0..32)
            .map(|row| Retain {
                phase: (row % 2) as usize,
                staged: 31 - row,
                retained: row % 7,
            })
            .collect(),
    );
    cases.push(
        (0..32)
            .map(|row| Retain {
                phase: (row / 16) as usize,
                staged: row % 16,
                retained: (row / 4) * RETAINED_ROWS + row % 4,
            })
            .collect(),
    );
    assert_eq!(cases.len(), 52);
    cases
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

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

#[test]
fn batched_rows_match_scalar_copies_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| {
        let cubin = bundle_glm5next::cubin_of(SLICE_SYMBOL).ok_or("missing row-copy entry")?;
        let bytes = bundle_glm5next::embedded_cubin(cubin).ok_or("missing row-copy cubin")?;
        let module = session.load_cubin(bytes)?;
        let function = module.function(SLICE_SYMBOL).map_err(|e| e.to_string())?;
        let mut control_allocations = Vec::new();
        let staging = bundle_glm5next::buffers::DrafterBuffers::allocate(Shape::TP4, |bytes| {
            control_allocations.push(session.allocate(bytes as usize)?);
            Ok::<_, String>(DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, control_allocations.len() as u32),
                0,
                bytes,
            ))
        })?;
        let counts = &control_allocations[staging.copy_rows.buffer.slot() as usize - 1];
        poison::write(
            session,
            counts,
            staging.vocab_width.offset as usize,
            &Shape::TP4.profile.vocab_rows.to_le_bytes(),
        )?;
        poison::write(
            session,
            counts,
            staging.copy_rows.offset as usize,
            &copies::counts(),
        )?;
        let mut checks = 0;
        let mut controls = 0;
        for (index, rows) in cases().iter().enumerate() {
            let source_rows = rows.iter().map(|r| r.staged + 1).max().unwrap() as usize;
            let destination_rows = rows.iter().map(|r| r.retained + 1).max().unwrap() as usize;
            let bytes = destination_rows * ROW + 64;
            let input = [
                noise(source_rows * ROW, index as u64 + 1),
                noise(source_rows * ROW, index as u64 + 91),
            ];
            let sources = [upload(session, &input[0])?, upload(session, &input[1])?];
            let output = session.allocate(bytes)?;
            let scalar = session.allocate(bytes)?;
            let plan = copies::runs(rows);
            let batched = |stream: &Stream| -> LabResult<()> {
                for &run in &plan {
                    let source = &sources[run.first.phase];
                    let source_offset = run.first.staged as usize * ROW;
                    let destination_offset = run.first.retained as usize * ROW;
                    if run.contiguous() {
                        output
                            .copy_range_from_async(
                                destination_offset,
                                source,
                                source_offset,
                                run.rows as usize * ROW,
                                stream,
                            )
                            .map_err(|e| e.to_string())?;
                    } else {
                        let span = |id, buffer: &DeviceBuffer| {
                            DeviceBufferSpanV1::new(
                                BundleDeviceBufferHandleV1::from_parts(1, id),
                                0,
                                buffer.byte_len() as u64,
                            )
                        };
                        let one = bundle_glm5next::sm120::hidden_copy::prepare(
                            run,
                            4096,
                            span(1 + run.first.phase as u32, source),
                            span(4, &output),
                            staging.copy_rows,
                        )?;
                        let mut arguments = Vec::new();
                        for arg in &one.arguments.0 {
                            arguments.push(match arg {
                                Argument::DeviceBuffer(id) => {
                                    let view =
                                        one.bindings.get(*id).map_err(|e| format!("{e:?}"))?;
                                    let buffer = match view.buffer.slot() {
                                        1 => &sources[0],
                                        2 => &sources[1],
                                        3 => counts,
                                        4 => &output,
                                        _ => return Err("unknown hidden-copy buffer".into()),
                                    };
                                    KernelArgument::Range {
                                        buffer,
                                        offset: view.offset as usize,
                                        byte_len: view.length as usize,
                                    }
                                }
                                Argument::U32(value) => KernelArgument::U32(*value),
                                _ => return Err("unexpected row-copy argument".into()),
                            });
                        }
                        launch::launch_with(
                            &function,
                            stream,
                            launch::config(one.plan)?,
                            &arguments,
                        )?;
                    }
                }
                Ok(())
            };
            let graph = session.capture(batched)?;
            let result = (|| -> LabResult<()> {
                for fill in [0xa5, 0x5a] {
                    let mut expected = vec![fill; bytes];
                    for row in rows {
                        let from = row.staged as usize * ROW;
                        let to = row.retained as usize * ROW;
                        expected[to..to + ROW].copy_from_slice(&input[row.phase][from..from + ROW]);
                    }
                    poison::write(session, &scalar, 0, &vec![fill; bytes])?;
                    for row in rows {
                        scalar
                            .copy_range_from_async(
                                row.retained as usize * ROW,
                                &sources[row.phase],
                                row.staged as usize * ROW,
                                ROW,
                                &session.stream,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    let mut got = vec![0; bytes];
                    scalar
                        .copy_to_bytes(&mut got, &session.stream)
                        .map_err(|e| e.to_string())?;
                    if got != expected {
                        return Err(format!("case {index}: scalar oracle differs"));
                    }
                    for captured in [false, true, true] {
                        poison::write(session, &output, 0, &vec![fill; bytes])?;
                        if captured {
                            graph.launch(&session.stream).map_err(|e| e.to_string())?;
                        } else {
                            batched(&session.stream)?;
                        }
                        output
                            .copy_to_bytes(&mut got, &session.stream)
                            .map_err(|e| e.to_string())?;
                        if got != expected {
                            return Err(format!("case {index}: batched copy differs"));
                        }
                        checks += 1;
                    }
                    got[rows[0].retained as usize * ROW] ^= 1;
                    if got == expected {
                        return Err("byte comparison missed a corrupt row".into());
                    }
                    controls += 1;
                }
                Ok(())
            })();
            session.context.synchronize().map_err(|e| e.to_string())?;
            result?;
        }
        assert_eq!((checks, controls), (312, 104));
        eprintln!("GLM_ROW_COPY_PASS cases=52 checks={checks} controls={controls}");
        Ok(())
    })
    .expect("ordered GLM hidden rows remain byte-identical");
}
