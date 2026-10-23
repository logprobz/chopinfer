//! Load-time packed bytes must equal the locked device prepare permutation.
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, mxfp8_fold_stream};
use kernel_core::contract::{Argument, ArgumentList};
use kernel_core::weight_transform::{
    SourceTensor, WeightRecord, WeightTransform, mxfp8_fold::Mxfp8FoldLayout,
};
use kernel_harness::{
    LabResult,
    launch::{Bindings, launch},
};
use kernels_sm120::gemm::mxfp8_fold_stream::Entry;
use model_deepseek41_contract::geometry::FLASH;

fn cases() -> Vec<Entry> {
    let entries = mxfp8_fold_stream::catalog(&FLASH).unwrap();
    // Ordinary, grouped, 64-K tail, and paired banks, all already locked.
    [
        (1280, 5120, 1, false),
        (1024, 4096, 2, false),
        (5120, 576, 1, false),
        (576, 5120, 1, true),
    ]
    .into_iter()
    .map(|shape| {
        entries
            .iter()
            .find(|entry| {
                let q = entry.ticket.query;
                (q.n, q.k, q.groups, entry.ticket.paired) == shape
            })
            .expect("required packed prepare shape")
            .clone()
    })
    .collect()
}

#[test]
fn packed_layout_prepare_cases_are_embedded_and_sized() {
    let cases = cases();
    assert_eq!(cases.len(), 4);
    for entry in cases {
        let q = entry.ticket.query;
        let layout = Mxfp8FoldLayout {
            groups: q.groups as usize,
            n: q.n as usize,
            k: q.k as usize,
        };
        assert_eq!(layout.packed_bytes().unwrap() as u64, entry.packed_bytes());
        assert!(fold_cubin_for_symbol(&entry.symbols()[0]).is_some());
        assert!(!entry.initialize_plan().pdl);
    }
}

#[test]
fn packed_layout_transform_equals_prepare_on_a_gpu() {
    kernel_harness::gpu_test(256 << 20, |session| -> LabResult<()> {
        for entry in cases() {
            let q = entry.ticket.query;
            let paired = entry.ticket.paired;
            let count = entry.packed_bytes() as usize;
            let state_bytes = entry.state_bytes(q.m_capacity).map_err(|e| e.to_string())? as usize;
            let symbol = entry.symbols()[0].clone();
            let (_, cubin) =
                fold_cubin_for_symbol(&symbol).ok_or("missing locked prepare cubin")?;
            let module = session.load_cubin(cubin)?;
            let function = module.function(&symbol).map_err(|e| e.to_string())?;
            let shape = vec![u64::from(q.groups), u64::from(q.n), u64::from(q.k)];
            let record = WeightRecord {
                transform: WeightTransform::Mxfp8FoldPacked,
                sources: vec![SourceTensor {
                    dtype: "F8_E4M3".into(),
                    shape: shape.clone(),
                    byte_length: count as u64,
                }],
                axis: None,
                ranges: vec![],
                stored_shape: shape,
            }
            .prepare(count as u64)
            .map_err(|e| e.to_string())?;
            let sources: Vec<Vec<u8>> = (0..if paired { 2 } else { 1 })
                .map(|bank| {
                    (0..count)
                        .map(|i| {
                            let code =
                                ((i.wrapping_mul(2_654_435_761) >> 17) ^ (bank * 0x53)) as u8;
                            if code & 0x7f == 0x7f { code ^ 1 } else { code }
                        })
                        .collect()
                })
                .collect();
            let mut buffers = Vec::new();
            for index in 0..5 {
                let bytes = if index == 4 { state_bytes } else { count };
                let buffer = session.allocate(bytes)?;
                let initial = if index < sources.len() {
                    sources[index].clone()
                } else {
                    vec![0xa5; bytes]
                };
                buffer
                    .copy_from_bytes(&initial, &session.stream)
                    .map_err(|e| e.to_string())?;
                buffers.push(buffer);
            }
            let arguments = ArgumentList(vec![
                Argument::DeviceBuffer(0),
                if paired {
                    Argument::DeviceBuffer(1)
                } else {
                    Argument::NullDevicePointer
                },
                Argument::DeviceBuffer(2),
                if paired {
                    Argument::DeviceBuffer(3)
                } else {
                    Argument::NullDevicePointer
                },
                Argument::DeviceBuffer(4),
                Argument::U32((state_bytes / 4) as u32),
            ]);
            let bindings = Bindings {
                buffers: buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i as u64, b))
                    .collect(),
            };
            launch(
                &session.context,
                &function,
                &session.stream,
                entry.initialize_plan(),
                &arguments,
                &bindings,
            )?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            for (bank, source) in sources.iter().enumerate() {
                let expected = record
                    .produce_whole(&mut |index, offset, out| {
                        assert_eq!(index, 0);
                        let offset = offset as usize;
                        out.copy_from_slice(&source[offset..offset + out.len()]);
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
                let mut actual = vec![0; count];
                buffers[2 + bank]
                    .copy_to_bytes(&mut actual, &session.stream)
                    .map_err(|e| e.to_string())?;
                if let Some(at) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                    return Err(format!(
                        "{symbol} bank {bank} packed byte {at}: {} != {}",
                        actual[at], expected[at]
                    ));
                }
            }
            let mut state = vec![0xa5; state_bytes];
            buffers[4]
                .copy_to_bytes(&mut state, &session.stream)
                .map_err(|e| e.to_string())?;
            assert!(
                state.iter().all(|&b| b == 0),
                "{symbol}: prepare must clear state"
            );
        }
        Ok(())
    })
    .unwrap();
}
