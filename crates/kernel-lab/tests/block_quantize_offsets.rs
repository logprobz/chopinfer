//! Shared status pools with caller-selected row offsets and guard words.
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantizeBuffers, QuantizeOutput};
use kernel_lab::{
    LabResult, Session,
    launch::{self, Bindings},
    poison::{LiveProducer, Replay, WriteMap},
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::block_quantize::{self, Entry};
use std::collections::BTreeMap;

fn run(session: &Session, entry: Entry) -> LabResult<()> {
    let query = entry.query;
    let width = query.width as usize;
    let mut a = vec![1_f32; 16 * width];
    let mut b = a.clone();
    a[0] = f32::NAN;
    a[3 * width + width - 1] = f32::INFINITY;
    b[width] = f32::NEG_INFINITY;
    b[0] = 2704.;
    let upload = |v: &[f32]| {
        if query.input == DType::F32 {
            session.upload_f32(v)
        } else {
            session.upload_bf16(
                &v.iter()
                    .map(|&x| kernel_core::codec::f32_to_bf16(x))
                    .collect::<Vec<_>>(),
            )
        }
    };
    if query.input == DType::Bf16 {
        for v in a.iter_mut().chain(&mut b) {
            *v = kernel_core::codec::bf16_to_f32(kernel_core::codec::f32_to_bf16(*v));
        }
    }
    let inputs = [upload(&a)?, upload(&b)?];
    let outputs = [
        session.allocate(entry.output_bytes(16) as usize)?,
        session.allocate(entry.output_bytes(16) as usize)?,
        session.allocate(entry.output_bytes(16) as usize)?,
    ];
    let scales = [
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
    ];
    let status = session.allocate(39 * 4)?;
    let live = session.allocate(4)?;
    let module = session.module(block_quantize::LAB_CUBIN)?;
    let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
    let backend = Sm120::new().with_block_quantize(entry.clone());
    let op = backend
        .block_quantize(&query, Need::rows(16))
        .map_err(|e| format!("{e:?}"))?;
    let mut bindings = BTreeMap::from([(0, &inputs[0]), (1, &inputs[1]), (8, &status), (9, &live)]);
    for i in 0..3 {
        bindings.insert(2 + i as u64, &outputs[i]);
        bindings.insert(5 + i as u64, &scales[i]);
    }
    let view = |handle: u64| View {
        handle,
        bytes: bindings[&handle].byte_len() as u64,
    };
    let mut launches = Vec::new();
    for (i, input, offset) in [(0, 0, 3), (1, 1, 3), (2, 0, 19)] {
        launches.extend(
            op.launches(
                &BlockQuantizeBuffers {
                    input: view(input),
                    output: view(2 + i),
                    scales: (query.output == QuantizeOutput::Codes).then(|| view(5 + i)),
                    status: view(8),
                    status_row_offset: offset,
                    live_rows: view(9),
                },
                View::EMPTY,
                Counts::rows(16),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    let bindings = Bindings { buffers: bindings };
    let launch_all = || {
        for call in &launches {
            launch::launch(
                &session.context,
                &function,
                &session.stream,
                call.plan,
                &call.arguments,
                &bindings,
            )?;
        }
        Ok(())
    };
    let producer = LiveProducer::new(session)?;
    for live_count in [0u32, 1, 7, 16] {
        live.copy_from_bytes(&live_count.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        let mut initial = vec![0x5a5a5a5au32; 39];
        initial[3..35].fill(0);
        initial[3 + 6] = 0x80;
        let mut expected = initial.clone();
        let first = kernel_lab::kit::cases::block_quantize::expected_rows(
            query,
            &a,
            &expected[3..19],
            live_count,
        )?;
        expected[3..19].copy_from_slice(&first.2);
        let second = kernel_lab::kit::cases::block_quantize::expected_rows(
            query,
            &b,
            &expected[3..19],
            live_count,
        )?;
        expected[3..19].copy_from_slice(&second.2);
        let third = kernel_lab::kit::cases::block_quantize::expected_rows(
            query,
            &a,
            &expected[19..35],
            live_count,
        )?;
        expected[19..35].copy_from_slice(&third.2);
        let expected_bytes = expected
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let mut replay = Replay::new(session).state(
            "status",
            &status,
            initial.iter().flat_map(|v| v.to_le_bytes()).collect(),
            WriteMap::of(DType::I32).none()?,
            [3usize, 19]
                .into_iter()
                .flat_map(|off| {
                    (0..live_count as usize).map(move |r| (off + r) * 4..(off + r + 1) * 4)
                })
                .collect(),
        );
        for i in 0..3 {
            replay = replay.output(
                format!("output-{i}"),
                &outputs[i],
                WriteMap::of(DType::U8)
                    .rows(entry.output_bytes(1) as usize, 0..live_count as usize)?,
            );
            replay = replay.output(
                format!("scales-{i}"),
                &scales[i],
                if entry.scale_bytes(1) == 0 {
                    WriteMap::of(DType::U8).none()?
                } else {
                    WriteMap::of(DType::U8)
                        .rows(entry.scale_bytes(1) as usize, 0..live_count as usize)?
                },
            );
        }
        replay.eager(launch_all)?;
        assert_eq!(replay.expected("status").unwrap(), expected_bytes);
        for (i, (output, scale, _)) in [first, second, third].iter().enumerate() {
            let bytes = live_count as usize * entry.output_bytes(1) as usize;
            assert_eq!(
                &replay.expected(&format!("output-{i}")).unwrap()[..bytes],
                &output[..bytes]
            );
            let bytes = live_count as usize * entry.scale_bytes(1) as usize;
            assert_eq!(
                &replay.expected(&format!("scales-{i}")).unwrap()[..bytes],
                &scale[..bytes]
            );
        }
        replay.relaunch(launch_all)?;
        let graph = session.capture(|_| launch_all())?;
        replay.graph(&graph)?;
        replay.pdl(|| producer.launch(session, &live), launch_all, true)?;
        let mut receipt = BTreeMap::from([
            ("entry".into(), serde_json::json!(entry.symbol)),
            ("live".into(), serde_json::json!(live_count)),
            ("offsets".into(), serde_json::json!([3, 3, 19])),
        ]);
        replay.record(&mut receipt);
        session.receipt("quantizer-status-offsets", receipt)?;
    }
    Ok(())
}

#[test]
fn status_offsets_and_shared_launches_preserve_guards_on_a_gpu() {
    kernel_lab::gpu_test(256 << 20, |session| {
        let entries = block_quantize::lab_entries()
            .into_iter()
            .filter(|e| e.query.width == 128)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 12);
        for entry in entries {
            run(session, entry)?;
        }
        Ok(())
    })
    .unwrap();
}
