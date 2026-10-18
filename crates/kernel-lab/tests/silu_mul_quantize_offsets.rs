//! Shared status pools with caller-selected row offsets and guard words.
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::DType;
use kernel_core::gated_mlp::quantized::SiluMulQuantizeBuffers;
use kernel_lab::{
    LabResult, Session,
    launch::{self, Bindings},
    poison::{LiveProducer, Replay, WriteMap},
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::silu_mul_quantize::{self, Entry};
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
    let inputs = [session.upload_f32(&a)?, session.upload_f32(&b)?];
    let up_values = vec![3.0; 16 * width];
    let up = session.upload_f32(&up_values)?;
    let outputs = [
        session.allocate(entry.code_bytes(16) as usize)?,
        session.allocate(entry.code_bytes(16) as usize)?,
        session.allocate(entry.code_bytes(16) as usize)?,
    ];
    let scales = [
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
        session.allocate(entry.scale_bytes(16).max(1) as usize)?,
    ];
    let status = session.allocate(39 * 4)?;
    let live = session.allocate(4)?;
    let module = session.module(silu_mul_quantize::LAB_CUBIN)?;
    let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
    let backend = Sm120::new().with_silu_mul_quantize(entry.clone());
    let op = backend
        .silu_mul_quantize(&query, Need::rows(16))
        .map_err(|e| format!("{e:?}"))?;
    let mut bindings = BTreeMap::from([
        (0, &inputs[0]),
        (1, &inputs[1]),
        (8, &status),
        (9, &live),
        (10, &up),
    ]);
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
                &SiluMulQuantizeBuffers {
                    gate: view(input),
                    up: view(10),
                    codes: view(2 + i),
                    scales: view(5 + i),
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
        let expected_rows = |gate: &[f32], prior: &[u32]| {
            kernel_lab::kit::cases::silu_mul_quantize::outcomes_with_status(
                query,
                &gate[..live_count as usize * width],
                &up_values[..live_count as usize * width],
                &prior[..live_count as usize],
            )
        };
        let first = expected_rows(&a, &expected[3..19])?;
        expected[3..3 + live_count as usize].copy_from_slice(&first.status);
        let second = expected_rows(&b, &expected[3..19])?;
        expected[3..3 + live_count as usize].copy_from_slice(&second.status);
        let third = expected_rows(&a, &expected[19..35])?;
        expected[19..19 + live_count as usize].copy_from_slice(&third.status);
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
                    .rows(entry.code_bytes(1) as usize, 0..live_count as usize)?,
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
        for (i, outcome) in [first, second, third].iter().enumerate() {
            outcome.check(
                &replay.expected(&format!("output-{i}")).unwrap()[..live_count as usize * width],
                &replay.expected(&format!("scales-{i}")).unwrap()
                    [..live_count as usize * width / 32],
            )?;
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
        session.receipt("silu-status-offsets", receipt)?;
    }
    Ok(())
}

#[test]
fn status_offsets_and_shared_launches_preserve_guards_on_a_gpu() {
    kernel_lab::gpu_test(256 << 20, |session| {
        let entries = silu_mul_quantize::lab_entries()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 3);
        for entry in entries {
            run(session, entry)?;
        }
        Ok(())
    })
    .unwrap();
}
