//! Two layer uploads share a producer and join independent consumer streams.
use super::*;
use runtime_bundle_abi::MappedGatherPlaneV1;
use runtime_host::HostDeviceBackendV1;

const CAPACITY: u64 = 196_608;
const COUNTS: [u64; 5] = [8191, 8192, 8193, 43_600, CAPACITY];

fn exercise<D: HostDeviceBackendV1 + 'static>(device: D, ordinals: [i32; 2], label: &str) {
    let root = temp(label);
    write_checkpoint(&root.0);
    let image = planned(&root.0, (0..2).map(|rank| plan(2, rank)).collect()).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.to_vec()).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(device, config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let mut ranks = Vec::new();
    for (rank, ordinal) in ordinals.into_iter().enumerate() {
        let context = host.select_context(rank as u32, ordinal).unwrap();
        let pool = host.select_memory_pool(context).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        ranks.push((context, pool, stream));
    }
    let image = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let length = CAPACITY * 9 + 64;
    let mut buffers = Vec::new();
    let mut layers = Vec::new();
    for _ in 0..2 {
        let ids = host
            .allocate_device_buffer(ranks[0].1, ranks[0].2, CAPACITY * 4, 16)
            .unwrap();
        buffers.push(ids);
        let indices = DeviceBufferSpanV1::new(ids, 0, CAPACITY * 4);
        let mut planes = Vec::new();
        let mut destinations = Vec::new();
        for (rank, &(_, pool, stream)) in ranks.iter().enumerate() {
            let mut spans = Vec::new();
            for _ in 0..3 {
                let buffer = host
                    .allocate_device_buffer(pool, stream, length, 16)
                    .unwrap();
                buffers.push(buffer);
                spans.push(DeviceBufferSpanV1::new(buffer, 0, length));
            }
            for (view, width, offset) in [("codes", 8, 16), ("scales", 1, CAPACITY * 8 + 32)] {
                planes.push(MappedGatherPlaneV1 {
                    image,
                    rank: rank as u32,
                    view_id: view.as_bytes().to_vec(),
                    row_bytes: width,
                    index_offset: 0,
                    stream,
                    destination: DeviceBufferSpanV1::new(spans[0].buffer, offset, CAPACITY * width),
                });
            }
            destinations.push(spans);
        }
        layers.push((indices, planes, destinations));
    }
    let gathers: Vec<_> = layers
        .iter()
        .map(|(ids, planes, _)| host.create_mapped_gather(ranks[0].2, *ids, planes).unwrap())
        .collect();
    let sentinel = vec![0xa5; length as usize];
    let mut executed = 0;
    for (generation, count) in COUNTS.into_iter().enumerate() {
        let picked: Vec<Vec<u32>> = (0..2)
            .map(|layer| {
                (0..count)
                    .map(|row| ((row * 37 + layer * 73 + generation as u64 * 19) % 512) as u32)
                    .collect()
            })
            .collect();
        for (layer, (indices, _, destinations)) in layers.iter().enumerate() {
            host.copy_host_to_device(
                ranks[0].2,
                DeviceBufferSpanV1::new(indices.buffer, 0, count * 4),
                &picked[layer]
                    .iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            for (rank, spans) in destinations.iter().enumerate() {
                for &span in spans {
                    host.copy_host_to_device(ranks[rank].2, span, &sentinel)
                        .unwrap();
                }
            }
        }
        // Both uploads may run while the first layer computes. A copy is a
        // real device consumer and must observe its own layer's completed DMA.
        for &gather in &gathers {
            host.enqueue_mapped_gather(gather, count).unwrap();
        }
        for (layer, (_, _, destinations)) in layers.iter().enumerate() {
            for (rank, spans) in destinations.iter().enumerate() {
                host.wait_mapped_gather(gathers[layer], ranks[rank].2)
                    .unwrap();
                host.copy_device_to_device(ranks[rank].2, spans[1], spans[0])
                    .unwrap();
            }
        }
        for (layer, (_, planes, destinations)) in layers.iter().enumerate() {
            for (rank, spans) in destinations.iter().enumerate() {
                let stream = ranks[rank].2;
                for plane in planes.iter().filter(|plane| plane.rank == rank as u32) {
                    host.gather_mapped_rows(
                        stream,
                        DeviceBufferSpanV1::new(
                            spans[2].buffer,
                            plane.destination.offset,
                            count * plane.row_bytes,
                        ),
                        MappedRowGatherV1 {
                            image,
                            rank: rank as u32,
                            view_id: &plane.view_id,
                            row_bytes: plane.row_bytes,
                            rows: &picked[layer],
                        },
                    )
                    .unwrap();
                }
                let mut actual = vec![0; length as usize];
                let mut reference = vec![0; length as usize];
                host.copy_device_to_host(stream, &mut actual, spans[1])
                    .unwrap();
                host.copy_device_to_host(stream, &mut reference, spans[2])
                    .unwrap();
                if let Some(at) = actual.iter().zip(&reference).position(|(a, b)| a != b) {
                    panic!("layer={layer} rank={rank} count={count} differs at byte={at}");
                }
                let mut expected = sentinel.clone();
                for (row, &id) in picked[layer].iter().enumerate() {
                    for column in 0..8 {
                        expected[16 + row * 8 + column] =
                            ((u64::from(id) * 8 + column as u64) * 31 + 7) as u8;
                    }
                    expected[CAPACITY as usize * 8 + 32 + row] =
                        ((rank * 512 + id as usize) * 5 + 1) as u8;
                }
                assert!(
                    actual == expected,
                    "oracle/guard differs layer={layer} rank={rank} count={count}"
                );
                executed += 1;
            }
            assert!(
                host.complete_mapped_gather(gathers[layer])
                    .unwrap()
                    .is_some()
            );
        }
        println!("LAYER_GATHER_BITWISE rows={count} layers=2 ranks=2 PASS");
    }
    assert_eq!(executed, 20);
    for gather in gathers.into_iter().rev() {
        host.release_mapped_gather(gather).unwrap();
    }
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_compiled_image(image).unwrap();
    for (context, pool, stream) in ranks.into_iter().rev() {
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
    assert_eq!(backend.live_resource_count(), 0);
    println!("LAYER_GATHER_COMPLETE cases={executed}");
}

#[test]
fn large_layer_gathers_preserve_bytes_and_consumer_order() {
    exercise(FakeHostDeviceV1::new(), [0, 1], "layer-fake");
}

#[cfg(all(feature = "linux-cuda-host", not(feature = "cuda-fake-driver-tests")))]
#[test]
fn large_layer_gathers_gpu_preserve_bytes_and_consumer_order() {
    let Ok(ordinals) = std::env::var("CHOPINFER_KERNEL_GPUS") else {
        assert_ne!(
            std::env::var("CHOPINFER_KERNEL_REQUIRE").as_deref(),
            Ok("1"),
            "required GPU pair absent"
        );
        eprintln!("skipped: CHOPINFER_KERNEL_GPUS is not set");
        return;
    };
    let ordinals: Vec<i32> = ordinals
        .split(',')
        .map(|word| word.trim().parse().expect("GPU ordinal"))
        .collect();
    assert_eq!(ordinals.len(), 2);
    assert_ne!(ordinals[0], ordinals[1]);
    exercise(
        runtime_host::cuda::CudaHostDeviceV1::new().unwrap(),
        [ordinals[0], ordinals[1]],
        "layer-gpu",
    );
}
