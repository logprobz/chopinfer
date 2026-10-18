//! Actual allocations across the 32-bit byte-offset and 16-bit page boundaries.

use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::codec::{SmallFloat, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::latent_attention::{
    BlockTables, LatentCompressedBuffers, LatentKvCache, LatentKvFormat, LatentSourcesBuffers,
    LatentWindowBuffers, LatentWindowStorage,
};
use kernel_lab::families::latent_sources_controls as family;
use kernel_lab::kit::{LabModules, Modules};
use kernel_lab::launch::{Bindings, launch};
use kernel_lab::poison::{Replay, WriteMap};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

#[test]
fn native_sources_preserve_wide_strides_and_page_offsets() {
    kernel_lab::gpu_test(5_u64 << 30, |session| {
        let functions = LabModules.load(session)?;
        let (width, heads) = (512_usize, 4_usize);
        let code_bytes = 128 * width;
        let scale_bytes = 128 * width / 32;
        let page_bytes = LatentKvFormat::Fp4E2M1E4M3Block16.page_bytes(512, 16) as usize;
        let mut receipts = Vec::new();
        for entry in family::catalog() {
            for wide in ["codes", "scales", "compressed"] {
                let wide_window = wide != "compressed";
                let sequences = if wide_window { 2 } else { 1 };
                let code_stride = if wide == "codes" {
                    (1_usize << 32) + 64
                } else {
                    code_bytes
                };
                let scale_stride = if wide == "scales" {
                    (1_usize << 32) + 3
                } else {
                    scale_bytes + 3
                };
                let page_stride = if wide_window { page_bytes } else { 65536 };
                let page_id = if wide_window { 0_usize } else { 65537 };
                let page_offset = page_id * page_stride;
                let prepared = Sm120::new()
                    .with_latent_sources(entry.clone())
                    .latent_attention_sources(&entry.query, Need::rows(1))
                    .map_err(|e| e.to_string())?;
                let upload = |data: &[u8]| -> kernel_lab::LabResult<runtime_cuda::DeviceBuffer> {
                    let buffer = session.allocate(data.len())?;
                    buffer
                        .copy_from_bytes(data, &session.stream)
                        .map_err(|e| e.to_string())?;
                    Ok(buffer)
                };
                let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
                let buffers = BTreeMap::from([
                    ("q", upload(&vec![0; heads * width * 2])?),
                    ("row_batch", upload(&u32::from(wide_window).to_le_bytes())?),
                    ("live", upload(&1_u32.to_le_bytes())?),
                    (
                        "window_codes",
                        session.allocate((sequences - 1) as usize * code_stride + code_bytes)?,
                    ),
                    (
                        "window_scales",
                        session.allocate((sequences - 1) as usize * scale_stride + scale_bytes)?,
                    ),
                    ("window_keys", upload(&0_u32.to_le_bytes())?),
                    ("compressed", session.allocate(page_offset + page_bytes)?),
                    (
                        "tables",
                        upload(
                            &(0..sequences)
                                .flat_map(|_| (page_id as u32).to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?,
                    ),
                    ("compressed_keys", upload(&0_u32.to_le_bytes())?),
                    ("sinks", upload(&f32s(&vec![-1000.0; heads]))?),
                    ("cos_sin", upload(&f32s(&[1.0, 0.0].repeat(32)))?),
                    ("out", session.allocate(heads * width * 2)?),
                    (
                        "workspace",
                        session.allocate(entry.workspace_bytes(1) as usize)?,
                    ),
                ]);
                // The truncated address holds a different finite value, so narrowing
                // fails numerically instead of relying on an invalid-address fault.
                let write = |name: &str, offset: usize, data: &[u8]| {
                    buffers[name]
                        .copy_from_bytes_at(offset, data, &session.stream)
                        .map_err(|e| e.to_string())
                };
                if wide == "codes" {
                    assert!(code_stride > u32::MAX as usize);
                    for (offset, value) in [(code_stride, 2.0), (code_stride as u32 as usize, 1.0)]
                    {
                        write(
                            "window_codes",
                            offset,
                            &vec![SmallFloat::E4M3.encode_f32(value); width],
                        )?;
                    }
                    write("window_scales", scale_stride, &vec![127; width / 32])?;
                } else if wide == "scales" {
                    assert!(scale_stride > u32::MAX as usize);
                    write(
                        "window_codes",
                        code_stride,
                        &vec![SmallFloat::E4M3.encode_f32(1.0); width],
                    )?;
                    for (offset, code) in [(scale_stride, 128), (scale_stride as u32 as usize, 127)]
                    {
                        write("window_scales", offset, &vec![code; width / 32])?;
                    }
                } else {
                    assert!(page_id > u16::MAX as usize && page_offset > u32::MAX as usize);
                    for (offset, code) in [(page_offset, 0x44), (page_offset as u32 as usize, 0x22)]
                    {
                        write("compressed", offset, &vec![code; width / 2])?;
                        write(
                            "compressed",
                            offset + 16 * width / 2,
                            &vec![0x38; width / 16],
                        )?;
                    }
                }
                let handles = buffers
                    .keys()
                    .enumerate()
                    .map(|(i, &name)| (name, i as u64 + 1))
                    .collect::<BTreeMap<_, _>>();
                let view = |name: &str| View {
                    handle: handles[name],
                    bytes: buffers[name].byte_len() as u64,
                };
                let roles = LatentSourcesBuffers {
                    q: view("q"),
                    row_batch: view("row_batch"),
                    live_rows: view("live"),
                    window: LatentWindowBuffers {
                        codes: view("window_codes"),
                        scales: view("window_scales"),
                        sequences,
                        code_stride: code_stride as u64,
                        scale_stride: scale_stride as u64,
                        storage: LatentWindowStorage::Ring128,
                        key_lists: view("window_keys"),
                        key_stride: 1,
                        budget: u32::from(wide_window),
                    },
                    compressed: Some(LatentCompressedBuffers {
                        cache: LatentKvCache {
                            pool: view("compressed"),
                            pages: page_id as u32 + 1,
                            page_stride: page_stride as u64,
                        },
                        block_tables: BlockTables {
                            tables: view("tables"),
                            sequences,
                            max_pages: 1,
                        },
                        key_lists: view("compressed_keys"),
                        key_stride: 1,
                        budget: u32::from(!wide_window),
                    }),
                    sinks: Some(view("sinks")),
                    cos_sin: Some(view("cos_sin")),
                    output: Some(view("out")),
                    partial: None,
                    workspace: view("workspace"),
                };
                let calls = prepared
                    .launches(
                        &roles,
                        View {
                            handle: 0,
                            bytes: 0,
                        },
                        Counts::rows(1),
                    )
                    .map_err(|e| e.to_string())?;
                let bindings = Bindings {
                    buffers: buffers
                        .iter()
                        .map(|(&name, b)| (handles[name], b))
                        .collect(),
                };
                let stages = calls
                    .into_iter()
                    .map(|call| {
                        let function = functions(&call.symbol)?;
                        if call.plan.dynamic_shared_bytes > 48 * 1024 {
                            function
                                .set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
                                .map_err(|e| e.to_string())?;
                        }
                        Ok((function, call))
                    })
                    .collect::<kernel_lab::LabResult<Vec<_>>>()?;
                let run = |stream: &runtime_cuda::Stream| {
                    for (function, call) in &stages {
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
                let mut replay = Replay::new(session)
                    .output(
                        "output",
                        &buffers["out"],
                        WriteMap::of(DType::Bf16).prefix(heads * width)?,
                    )
                    .scratch("workspace", &buffers["workspace"]);
                replay.eager(|| run(&session.stream))?;
                assert_eq!(
                    replay.expected("output").unwrap(),
                    kernel_lab::kit::bf16_bytes(&vec![f32_to_bf16(2.0); heads * width])
                );
                replay.relaunch(|| run(&session.stream))?;
                replay.graph(&session.capture(run)?)?;
                let mut receipt = BTreeMap::new();
                replay.record(&mut receipt);
                receipt.insert("window_code_stride".into(), serde_json::json!(code_stride));
                receipt.insert(
                    "window_scale_stride".into(),
                    serde_json::json!(scale_stride),
                );
                receipt.insert("page_id".into(), serde_json::json!(page_id));
                receipt.insert("page_offset".into(), serde_json::json!(page_offset));
                receipts.push(receipt);
            }
        }
        assert_eq!(receipts.len(), 6);
        session.receipt(
            "native-latent-wide-addresses",
            BTreeMap::from([("cases".into(), serde_json::json!(receipts))]),
        )?;
        Ok(())
    })
    .expect("native wide addresses");
}
