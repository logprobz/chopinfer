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

pub fn time(
    session: &Session,
    module: &Module,
    calls: [&[Launch]; 2],
    images: &[Image],
) -> LabResult<f32> {
    use kernel_harness::launch::{Bindings, launch};
    let mut arms = Vec::new();
    for packed in [false, true] {
        let buffers = images
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
            .collect::<LabResult<Vec<_>>>()?;
        arms.push(buffers);
    }
    let functions = calls
        .iter()
        .map(|arm| {
            arm.iter()
                .map(|call| module.function(&call.symbol).map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = kernel_harness::time_ab(
        session,
        5,
        30,
        [kernel_harness::eviction::EvictionMode::Read; 2],
        |arm, stream| {
            let bindings = Bindings {
                buffers: images
                    .iter()
                    .zip(&arms[arm])
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
        },
    )?;
    Ok(timing.paired_ratio)
}
