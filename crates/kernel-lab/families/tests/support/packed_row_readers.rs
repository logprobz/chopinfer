//! Shared fixtures for the packed ticket, prefetch, split and reuse tests.
use kernel_core::backend::{Launch, View};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::packed_fold::{self, Image};
use kernel_harness::{LabResult, Session};
use kernels_sm120::gemm::mxfp8_fold;
use runtime_cuda::Module;
use std::path::PathBuf;

pub fn buffers(q: Mxfp8FoldGemmQuery, rows: u32) -> [Mxfp8FoldGemmBuffers<View>; 2] {
    let sizes = mxfp8_fold::Entry::new(q, "fixture").unwrap().bytes(rows);
    let v = |handle, bytes| View { handle, bytes };
    let first = Mxfp8FoldGemmBuffers {
        activations: v(1, sizes.activations),
        activation_scales: v(2, sizes.activation_scales),
        weights: v(3, sizes.weights),
        weight_scales: v(4, sizes.weight_scales),
        output: v(5, sizes.output),
        live_rows: v(6, 4),
        status: v(7, sizes.status + 12),
        status_row_offset: 3,
    };
    [
        first,
        Mxfp8FoldGemmBuffers {
            weights: v(8, sizes.weights),
            weight_scales: v(9, sizes.weight_scales),
            output: v(10, sizes.output),
            ..first
        },
    ]
}

pub fn images(
    q: Mxfp8FoldGemmQuery,
    rows: u32,
    paired: bool,
    workspace: u64,
    seed: u64,
) -> LabResult<Vec<Image>> {
    let [b, second] = buffers(q, rows);
    let layout = Mxfp8FoldLayout {
        groups: q.groups as usize,
        n: q.n as usize,
        k: q.k as usize,
    };
    let mut activations = packed_fold::codes(b.activations.bytes as usize, seed);
    if rows > 1 {
        activations[q.activation_row_stride as usize + 4] = 0x7f;
    }
    let mut status = vec![0; b.status.bytes as usize];
    status[..12].fill(0x5a);
    if rows > 2 {
        status[20] = 4;
    }
    let mut result = vec![
        Image::data(1, activations),
        Image::data(
            2,
            packed_fold::scales(b.activation_scales.bytes as usize, seed + 1),
        ),
        Image::codes(3, layout, seed + 2)?,
        Image::data(
            4,
            packed_fold::scales(b.weight_scales.bytes as usize, seed + 3),
        ),
        Image::data(5, vec![0xa5; b.output.bytes as usize]),
        Image::data(6, rows.to_le_bytes().to_vec()),
        Image::data(7, status),
        Image::data(11, vec![0xa5; workspace as usize]),
    ];
    if paired {
        result.extend([
            Image::codes(8, layout, seed + 4)?,
            Image::data(
                9,
                packed_fold::scales(second.weight_scales.bytes as usize, seed + 5),
            ),
            Image::data(10, vec![0xa5; second.output.bytes as usize]),
        ]);
    }
    Ok(result)
}

/// Timing uses admitted inputs in every row.
pub fn admit_all(images: &mut [Image], seed: u64) {
    let activations = images.iter_mut().find(|image| image.handle == 1).unwrap();
    activations.bytes = packed_fold::codes(activations.bytes.len(), seed);
    images
        .iter_mut()
        .find(|image| image.handle == 7)
        .unwrap()
        .bytes[12..]
        .fill(0);
}

pub fn load(session: &Session, variable: &str, cubin: &str) -> LabResult<Module> {
    let directory = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{variable} is required"))?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    let bytes =
        std::fs::read(directory.join(format!("{cubin}.cubin"))).map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

// Two arms of at most 2 GiB, with room for eviction, calibration and CUDA state.
pub const TIMING_MEMORY_BYTES: u64 = 5 << 30;
const ARM_BYTES: usize = 2 << 30;
const MAX_REPETITIONS: usize = 128;
pub const TIMING_ROWS: [u32; 9] = [1, 3, 4, 6, 16, 28, 32, 48, 64];

/// Baseline decode routing for the unpaired G=1 timing shapes below.
/// Split takes precedence over prefetch; ticket is the rest.
///
/// N     K     Split                 Prefetch
/// 5120  2048  none                  1-10, 12, 16, 24
/// 8192  1280  1-2, 5-12, 16         3
/// 5120   576  1-28                  29-32, 49-56
///
/// Rows outside 1-64 and shapes outside this table are not classified.
/// projection_reuse.rs overrides some TP4 buckets; reuse_served below
/// checks that separate registration, including its tile and fold count.
pub fn decode_served(family: &str, q: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> bool {
    if q.groups != 1 || paired || !(1..=64).contains(&rows) {
        return false;
    }
    let (split, prefetch) = match (q.n, q.k) {
        (5120, 2048) => (false, matches!(rows, 1..=10 | 12 | 16 | 24)),
        (8192, 1280) => (matches!(rows, 1..=2 | 5..=12 | 16), rows == 3),
        (5120, 576) => (rows <= 28, matches!(rows, 29..=32 | 49..=56)),
        _ => return false,
    };
    match family {
        "decode" => split,
        "prefetch" => !split && prefetch,
        "ticket" => !split && !prefetch,
        _ => false,
    }
}

/// Flash TP4 projection_reuse.rs::choice, restricted to the timed shapes.
/// N8192/K1280 uses tile 64, four folds, rows 9-12. N5120/K2048 uses
/// tile 32: eight folds at 17-23 and 25-28, two folds at 29-32.
/// N5120/K576 has no reuse registration.
pub fn reuse_served(q: Mxfp8FoldGemmQuery, rows: u32, paired: bool, tile: u32, folds: u32) -> bool {
    if q.groups != 1 || paired {
        return false;
    }
    match (q.n, q.k, tile, folds) {
        (8192, 1280, 64, 4) => (9..=12).contains(&rows),
        (5120, 2048, 32, 8) => matches!(rows, 17..=23 | 25..=28),
        (5120, 2048, 32, 2) => (29..=32).contains(&rows),
        _ => false,
    }
}

pub struct SequenceTiming {
    pub paired_ratio: f32,
    pub row_major_us: f32,
    pub packed_us: f32,
    pub repetitions: usize,
}

impl SequenceTiming {
    /// A served comparison cannot certify the bound with an unresolved timer.
    pub fn failure(&self) -> Option<String> {
        let row_sample_us = self.row_major_us * self.repetitions as f32;
        let reason = if !row_sample_us.is_finite() || row_sample_us < 1000.0 {
            "row-major sample below 1 ms at the repetition or memory cap"
        } else if !self.paired_ratio.is_finite() || self.paired_ratio > 1.03 {
            "packed/row-major exceeds 1.03"
        } else {
            return None;
        };
        Some(format!(
            "{reason}: ratio={:.6} row_major_us={:.6} packed_us={:.6} reps={}",
            self.paired_ratio, self.row_major_us, self.packed_us, self.repetitions
        ))
    }
}

fn repetitions(calibration_ms: f32, bytes_per_set: usize) -> LabResult<usize> {
    if !calibration_ms.is_finite() || calibration_ms <= 0.0 || bytes_per_set == 0 {
        return Err("invalid packed timing calibration".into());
    }
    let maximum = (ARM_BYTES / bytes_per_set).min(MAX_REPETITIONS);
    if maximum == 0 {
        return Err("one packed timing buffer set exceeds 2 GiB".into());
    }
    // Target 2 ms to leave headroom above 1 ms when repeated launches run
    // faster than the isolated calibration. Both resource caps still apply.
    Ok(((2.0 / calibration_ms).ceil() as usize).clamp(1, maximum))
}

pub fn time(
    session: &Session,
    module: &Module,
    calls: [&[Launch]; 2],
    images: &[Image],
) -> LabResult<SequenceTiming> {
    use kernel_harness::eviction::{Eviction, EvictionMode};
    use kernel_harness::launch::{Bindings, launch};
    use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Stream};
    let allocate = |packed: bool| -> LabResult<Vec<DeviceBuffer>> {
        images
            .iter()
            .map(|image| {
                let bytes = if packed {
                    image.packed.as_ref().unwrap_or(&image.bytes)
                } else {
                    &image.bytes
                };
                let buffer = session.allocate(bytes.len().max(1))?;
                kernel_harness::poison::write(session, &buffer, 0, bytes)?;
                Ok(buffer)
            })
            .collect()
    };
    let functions = calls
        .iter()
        .map(|arm| {
            arm.iter()
                .map(|call| module.function(&call.symbol).map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()
        })
        .collect::<LabResult<Vec<_>>>()?;
    let run = |arm: usize, buffers: &[DeviceBuffer], stream: &Stream| -> LabResult<()> {
        let bindings = Bindings {
            buffers: images
                .iter()
                .zip(buffers)
                .map(|(image, buffer)| (image.handle, buffer))
                .collect(),
        };
        for (call, function) in calls[arm].iter().zip(&functions[arm]) {
            launch(
                &session.context,
                function,
                stream,
                call.plan,
                &call.arguments,
                &bindings,
            )?;
        }
        Ok(())
    };
    let bytes_per_set = [false, true]
        .into_iter()
        .map(|packed| {
            images
                .iter()
                .map(|image| {
                    if packed {
                        image.packed.as_ref().unwrap_or(&image.bytes)
                    } else {
                        &image.bytes
                    }
                    .len()
                    .max(1)
                })
                .sum::<usize>()
        })
        .max()
        .unwrap();
    repetitions(1.0, bytes_per_set)?;

    // One cold calibration sequence, excluded from the paired samples.
    // Use the timing harness's gate protocol so host enqueue gaps cannot
    // inflate the calibration and select too few repetitions.
    let calibration_ms = {
        let buffers = allocate(false)?;
        let eviction = Eviction::new(session, EvictionMode::Read)?;
        let probe = session.module("probe_sm120")?;
        let gate_fn = probe
            .function("chop_probe_gate_v1")
            .map_err(|e| e.to_string())?;
        let gate = session.allocate(4)?;
        let release = session.context.stream().map_err(|e| e.to_string())?;
        gate.copy_from_bytes(&0_u32.to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        eviction.run(&session.stream)?;
        let start = session.context.timed_event().map_err(|e| e.to_string())?;
        let end = session.context.timed_event().map_err(|e| e.to_string())?;
        gate_fn
            .launch(
                &session.stream,
                kernel_harness::launch::simple_config(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[KernelArgument::Buffer(&gate)],
            )
            .map_err(|e| e.to_string())?;
        let scheduled = (|| {
            start.record(&session.stream).map_err(|e| e.to_string())?;
            run(0, &buffers, &session.stream)?;
            end.record(&session.stream).map_err(|e| e.to_string())
        })();
        gate.copy_from_bytes(&1_u32.to_ne_bytes(), &release)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        scheduled?;
        start.elapsed_ms(&end).map_err(|e| e.to_string())?
    };
    let mut repetitions = repetitions(calibration_ms, bytes_per_set)?;
    let maximum = (ARM_BYTES / bytes_per_set).min(MAX_REPETITIONS);
    let mut arms = [Vec::new(), Vec::new()];
    let timing = loop {
        for (arm, sets) in arms.iter_mut().enumerate() {
            while sets.len() < repetitions {
                sets.push(allocate(arm == 1)?);
            }
        }
        let timing =
            kernel_harness::time_ab(session, 5, 30, [EvictionMode::Read; 2], |arm, stream| {
                // Every sequence has distinct copies of every image, including
                // weights, workspace and flags. One eviction precedes the entire
                // sample; repetitions cannot reuse the previous sequence's data.
                for buffers in &arms[arm] {
                    run(arm, buffers, stream)?;
                }
                Ok(())
            })?;
        if timing.arms[0].median_ms >= 1.0 || repetitions == maximum {
            break timing;
        }
        // Keep the final result above 1 ms even if calibration overestimated
        // the sequence duration. A capped short result is still printed;
        // failure() refuses to certify a served case with it.
        repetitions = ((repetitions as f32 * 2.0 / timing.arms[0].median_ms).ceil() as usize)
            .clamp(repetitions + 1, maximum);
    };
    Ok(SequenceTiming {
        paired_ratio: timing.paired_ratio,
        row_major_us: timing.arms[0].median_ms * 1000.0 / repetitions as f32,
        packed_us: timing.arms[1].median_ms * 1000.0 / repetitions as f32,
        repetitions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(n: u32, k: u32) -> Mxfp8FoldGemmQuery {
        Mxfp8FoldGemmQuery {
            m_capacity: 64,
            n,
            k,
            groups: 1,
            output: kernel_core::contract::DType::F32,
            activation_group_stride: k,
            activation_row_stride: k,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        }
    }

    #[test]
    fn flash_decode_routing_boundaries() {
        for (n, k, split, prefetch) in [
            (5120, 2048, vec![], vec![1..=10, 12..=12, 16..=16, 24..=24]),
            (8192, 1280, vec![1..=2, 5..=12, 16..=16], vec![3..=3]),
            (5120, 576, vec![1..=28], vec![29..=32, 49..=56]),
        ] {
            let q = query(n, k);
            for rows in 0..=65 {
                let expected = if !(1..=64).contains(&rows) {
                    None
                } else if split.iter().any(|r| r.contains(&rows)) {
                    Some("decode")
                } else if prefetch.iter().any(|r| r.contains(&rows)) {
                    Some("prefetch")
                } else {
                    Some("ticket")
                };
                for family in ["decode", "prefetch", "ticket"] {
                    assert_eq!(
                        decode_served(family, q, rows, false),
                        expected == Some(family),
                        "{family} N{n} K{k} rows={rows}"
                    );
                    assert!(!decode_served(family, q, rows, true));
                    assert!(!decode_served(
                        family,
                        Mxfp8FoldGemmQuery { groups: 2, ..q },
                        rows,
                        false
                    ));
                }
            }
        }
        assert!(!decode_served("ticket", query(40, 576), 1, false));
    }

    #[test]
    fn reuse_routing_requires_selected_tile_and_fold_count() {
        for rows in 0..=65 {
            assert_eq!(
                reuse_served(query(8192, 1280), rows, false, 64, 4),
                (9..=12).contains(&rows)
            );
            assert!(!reuse_served(query(8192, 1280), rows, false, 32, 4));
            assert!(!reuse_served(query(5120, 2048), rows, false, 64, 4));
            assert_eq!(
                reuse_served(query(5120, 2048), rows, false, 32, 8),
                matches!(rows, 17..=23 | 25..=28)
            );
            assert_eq!(
                reuse_served(query(5120, 2048), rows, false, 32, 2),
                (29..=32).contains(&rows)
            );
            assert!(!reuse_served(query(5120, 576), rows, false, 64, 4));
        }
    }

    #[test]
    fn served_timing_requires_resolved_samples() {
        let mut timing = SequenceTiming {
            paired_ratio: 1.0,
            row_major_us: 20.0,
            packed_us: 20.0,
            repetitions: 50,
        };
        assert!(timing.failure().is_none());
        timing.repetitions = 49;
        assert!(timing.failure().unwrap().contains("below 1 ms"));
        timing.repetitions = 50;
        timing.paired_ratio = 1.04;
        assert!(timing.failure().unwrap().contains("exceeds 1.03"));
    }

    #[test]
    fn timing_repetitions_respect_duration_and_memory_caps() {
        assert_eq!(repetitions(0.02, 1 << 20).unwrap(), 100);
        assert_eq!(repetitions(0.001, 1 << 20).unwrap(), 128);
        assert_eq!(repetitions(0.02, 64 << 20).unwrap(), 32);
        assert_eq!(repetitions(3.0, 1 << 20).unwrap(), 1);
        assert!(repetitions(0.02, ARM_BYTES + 1).is_err());
        assert!(repetitions(0.0, 1).is_err());
        assert!(repetitions(f32::NAN, 1).is_err());
    }
}
