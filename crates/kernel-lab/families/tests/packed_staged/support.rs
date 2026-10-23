//! What the staged packed fold families' tests share: loading a family's
//! lab cubin and timing row-major launches against packed launches.

use kernel_build::EntrySpec;
use kernel_core::backend::Launch;
use kernel_harness::eviction::DEFAULT_EVICTION;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::packed_fold::Image;
use kernel_harness::{LabResult, Session};
use kernels_sm120::gemm::mxfp8_fold_packed;
use runtime_cuda::{DeviceBuffer, Module};
use std::path::PathBuf;

/// The cubin `cubin` of the build directory `variable` names, after checking
/// that the directory's build report is current.
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

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    kernel_harness::poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

/// The median paired ratio of `launches[1]` on packed codes to `launches[0]`
/// on row-major codes. Each arm replays one captured graph over its own
/// copy of `images`, after an L2 eviction.
pub fn time(
    session: &Session,
    module: &Module,
    launches: [&[Launch]; 2],
    images: &[Image],
) -> LabResult<f32> {
    let mut graphs = Vec::new();
    let mut kept = Vec::new();
    for (arm, calls) in launches.into_iter().enumerate() {
        let buffers = images
            .iter()
            .map(|image| match &image.packed {
                Some(packed) if arm == 1 => upload(session, packed),
                _ => upload(session, &image.bytes),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let functions = calls
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
            .collect::<LabResult<Vec<_>>>()?;
        let bindings = Bindings {
            buffers: images
                .iter()
                .zip(&buffers)
                .map(|(image, buffer)| (image.handle, buffer))
                .collect(),
        };
        graphs.push(session.capture(|stream| {
            for (call, function) in calls.iter().zip(&functions) {
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
        })?);
        kept.push(buffers);
    }
    let timing = kernel_harness::time_ab(session, 2, 15, [DEFAULT_EVICTION; 2], |arm, stream| {
        graphs[arm].launch(stream).map_err(|e| e.to_string())
    })?;
    // The graphs read these until the timing ends.
    drop(kept);
    Ok(timing.paired_ratio)
}

/// Checks that `packed` is `row_major` with `body` as its body and the
/// packed rule's symbol, and nothing else changed.
pub fn assert_packed_twin(row_major: &EntrySpec, packed: &EntrySpec, body: &str) {
    assert_eq!(packed.symbol, mxfp8_fold_packed::symbol(&row_major.symbol));
    assert_eq!(packed.body, body);
    assert_ne!(packed.body, row_major.body);
    assert_eq!(packed.params, row_major.params);
    assert_eq!(
        (
            packed.max_threads,
            packed.min_blocks,
            packed.allowed_local_bytes
        ),
        (
            row_major.max_threads,
            row_major.min_blocks,
            row_major.allowed_local_bytes
        )
    );
}
