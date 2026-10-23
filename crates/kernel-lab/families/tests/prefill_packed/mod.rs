//! Shared parts of the packed TMA prefill lab tests: views and images of
//! one launch, the moved-strip control, cubin loading and paired timing.

use kernel_core::backend::{Launch, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::eviction::DEFAULT_EVICTION;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::packed_fold::{self, Image};
use kernel_harness::{LabResult, Session};
use kernels_sm120::gemm::mxfp8_fold::Entry;
use runtime_cuda::{DeviceBuffer, Function, Module};
use std::path::PathBuf;

pub const HANDLES: Mxfp8FoldGemmBuffers<u64> = Mxfp8FoldGemmBuffers {
    activations: 1,
    activation_scales: 2,
    weights: 3,
    weight_scales: 4,
    output: 5,
    live_rows: 6,
    status: 7,
    status_row_offset: 0,
};

/// The views a prepared op launches over at `rows` rows.
pub fn views(entry: &Entry, rows: u32) -> Mxfp8FoldGemmBuffers<View> {
    let b = entry.bytes(rows);
    let v = |handle, bytes| View { handle, bytes };
    Mxfp8FoldGemmBuffers {
        activations: v(HANDLES.activations, b.activations),
        activation_scales: v(HANDLES.activation_scales, b.activation_scales),
        weights: v(HANDLES.weights, b.weights),
        weight_scales: v(HANDLES.weight_scales, b.weight_scales),
        output: v(HANDLES.output, b.output),
        live_rows: v(HANDLES.live_rows, 4),
        status: v(HANDLES.status, b.status),
        status_row_offset: 0,
    }
}

pub fn layout(entry: &Entry) -> Mxfp8FoldLayout {
    Mxfp8FoldLayout {
        groups: entry.query.groups as usize,
        n: entry.query.n as usize,
        k: entry.query.k as usize,
    }
}

/// Every view of one launch; the weight codes are `images[2]`. Row 1 holds
/// a NaN code when there are two rows, so the admission and status paths
/// run too.
pub fn images(entry: &Entry, rows: u32, seed: u64) -> LabResult<Vec<Image>> {
    let sizes = entry.bytes(rows);
    let mut activations = packed_fold::codes(sizes.activations as usize, seed);
    if rows > 1 {
        activations[entry.query.activation_row_stride as usize + 4] = 0x7f;
    }
    Ok(vec![
        Image::data(HANDLES.activations, activations),
        Image::data(
            HANDLES.activation_scales,
            packed_fold::scales(sizes.activation_scales as usize, seed + 1),
        ),
        Image::codes(HANDLES.weights, layout(entry), seed + 2)?,
        Image::data(
            HANDLES.weight_scales,
            packed_fold::scales(sizes.weight_scales as usize, seed + 3),
        ),
        Image::data(HANDLES.output, vec![0xa5; sizes.output as usize]),
        Image::data(HANDLES.live_rows, rows.to_le_bytes().to_vec()),
        Image::data(HANDLES.status, vec![0; sizes.status as usize]),
    ])
}

/// Swaps packed strips `first` and `first + 1`.
pub fn move_strip(images: &mut [Image], entry: &Entry, first: usize) {
    let strip = 8 * entry.query.k as usize;
    let codes = images[2].packed.as_mut().unwrap();
    let (left, right) = codes.split_at_mut((first + 1) * strip);
    left[first * strip..].swap_with_slice(&mut right[..strip]);
}

/// The family cubin `cubin` from the verified build in directory `variable`.
pub fn load(session: &Session, variable: &str, cubin: &str) -> LabResult<Module> {
    let directory = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or(format!("{variable} is required"))?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    let bytes =
        std::fs::read(directory.join(format!("{cubin}.cubin"))).map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

/// Runs `row_major` on row-major codes and `packed` on packed codes over
/// `images` and requires every byte they write to match.
pub fn compare(
    session: &Session,
    module: &Module,
    row_major: &[Launch],
    packed: &[Launch],
    images: &[Image],
) -> LabResult<()> {
    let function = |symbol: &str| module.function(symbol).map_err(|e| e.to_string());
    packed_fold::compare(session, &function, row_major, packed, images)
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    kernel_harness::poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn functions(module: &Module, launches: &[Launch]) -> LabResult<Vec<Function>> {
    launches
        .iter()
        .map(|call| {
            let function = module.function(&call.symbol).map_err(|e| e.to_string())?;
            if call.plan.dynamic_shared_bytes > 48 << 10 {
                function
                    .set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            Ok(function)
        })
        .collect()
}

/// The median paired time ratio of `packed` launches on packed codes over
/// `row_major` launches on row-major codes, at `rows` rows of `entry`. Both
/// arms share every other view.
pub fn packed_over_row_major(
    session: &Session,
    module: &Module,
    entry: &Entry,
    rows: u32,
    row_major: &[Launch],
    packed: &[Launch],
) -> LabResult<f32> {
    let images = images(entry, rows, 11)?;
    let codes = [
        upload(session, &images[2].bytes)?,
        upload(session, images[2].packed.as_ref().unwrap())?,
    ];
    let data = images
        .iter()
        .filter(|image| image.packed.is_none())
        .map(|image| Ok((image.handle, upload(session, &image.bytes)?)))
        .collect::<LabResult<Vec<_>>>()?;
    let bindings = codes.each_ref().map(|codes| Bindings {
        buffers: data
            .iter()
            .map(|(handle, buffer)| (*handle, buffer))
            .chain([(HANDLES.weights, codes)])
            .collect(),
    });
    let launches = [row_major, packed];
    let functions = [functions(module, row_major)?, functions(module, packed)?];
    let paired = kernel_harness::time_ab(session, 3, 15, [DEFAULT_EVICTION; 2], |arm, stream| {
        for (call, function) in launches[arm].iter().zip(&functions[arm]) {
            launch(
                &session.context,
                function,
                stream,
                call.plan,
                &call.arguments,
                &bindings[arm],
            )?;
        }
        Ok(())
    })?;
    eprintln!(
        "{} rows={rows}: row-major {:.4} ms, packed {:.4} ms, paired packed/row-major {:.4}",
        entry.symbol, paired.arms[0].median_ms, paired.arms[1].median_ms, paired.paired_ratio
    );
    Ok(paired.paired_ratio)
}
