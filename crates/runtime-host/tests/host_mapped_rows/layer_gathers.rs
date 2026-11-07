//! Two layer uploads share a producer and join independent consumer streams.
use super::*;
use runtime_bundle_abi::MappedGatherPlaneV1;
use runtime_host::HostDeviceBackendV1;

const CAPACITY: u64 = 196_608;
const COUNTS: [u64; 5] = [8191, 8192, 8193, 43_600, CAPACITY];
/// The same sizes shrinking and growing, so a reused gather would show a
/// previous gather's bytes in the live range.
const REUSED: [u64; 6] = [CAPACITY, 8193, 43_600, 8191, 8192, 1];

/// This process's descriptors that name `path`.
fn links(path: &Path) -> usize {
    let path = std::fs::canonicalize(path).unwrap();
    std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .ok()
                .and_then(|entry| std::fs::read_link(entry.path()).ok())
                .is_some_and(|target| target == path)
        })
        .count()
}

/// Each rank's streams, contexts and pools.
type Ranks = Vec<(
    runtime_bundle_abi::BundleContextHandleV1,
    runtime_bundle_abi::BundleMemoryPoolHandleV1,
    runtime_bundle_abi::BundleStreamHandleV1,
)>;

fn open_ranks(host: &runtime_bundle_abi::HostServicesV1, ordinals: [i32; 2]) -> Ranks {
    ordinals
        .into_iter()
        .enumerate()
        .map(|(rank, ordinal)| {
            let context = host.select_context(rank as u32, ordinal).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            (context, pool, stream)
        })
        .collect()
}

/// The gather's completion. The CUDA fake driver answers NOT_READY once per
/// event, even after a stream fence, so this asks again.
fn complete(
    host: &runtime_bundle_abi::HostServicesV1,
    gather: runtime_bundle_abi::BundleMappedGatherHandleV1,
) -> Result<Option<runtime_bundle_abi::MappedGatherStatsV1>, BundleAbiError> {
    for _ in 0..16 {
        match host.complete_mapped_gather(gather) {
            Ok(None) => continue,
            result => return result,
        }
    }
    panic!("prepared gather never completed");
}

fn close_ranks(host: &runtime_bundle_abi::HostServicesV1, ranks: Ranks) {
    for (context, pool, stream) in ranks.into_iter().rev() {
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
}

fn exercise<D: HostDeviceBackendV1 + 'static>(
    device: D,
    ordinals: [i32; 2],
    label: &str,
    counts: &[u64],
) {
    let root = temp(label);
    write_checkpoint(&root.0);
    let shard = root.0.join("model.safetensors");
    let image = planned(&root.0, (0..2).map(|rank| plan(2, rank)).collect()).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.to_vec()).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(device, config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let ranks = open_ranks(host, ordinals);
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
    for (generation, &count) in counts.iter().enumerate() {
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
            assert!(complete(host, gathers[layer]).unwrap().is_some());
        }
        println!("LAYER_GATHER_BITWISE rows={count} layers=2 ranks=2 PASS");
    }
    assert_eq!(executed, counts.len() * 4);
    let held = links(&shard);
    for gather in gathers.into_iter().rev() {
        host.release_mapped_gather(gather).unwrap();
    }
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_compiled_image(image).unwrap();
    close_ranks(host, ranks);
    assert_eq!(backend.live_resource_count(), 0);
    drop(table);
    drop(backend);
    // The mappings, and every description of the shard they held, close
    // with the image.
    assert!(held > 0);
    assert_eq!(links(&shard), 0);
    println!("LAYER_GATHER_COMPLETE cases={executed} shard_descriptors={held}");
}

/// A large prepared gather released while pending, a new one on the same
/// mappings, a settled gather (every consumer joined, every stream
/// synchronized, the gather completed) enqueued again, and a gather whose
/// reads fail once the checkpoint is truncated: it fails at completion,
/// refuses the next enqueue and still releases, and nothing leaks.
fn settle_and_fail<D: HostDeviceBackendV1 + 'static>(device: D, ordinals: [i32; 2], label: &str) {
    const ROWS: u64 = 9000;
    let root = temp(label);
    write_checkpoint(&root.0);
    let image = planned(&root.0, (0..2).map(|rank| plan(2, rank)).collect()).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.to_vec()).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(device, config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let ranks = open_ranks(host, ordinals);
    let image = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let ids = host
        .allocate_device_buffer(ranks[0].1, ranks[0].2, ROWS * 4, 16)
        .unwrap();
    let indices = DeviceBufferSpanV1::new(ids, 0, ROWS * 4);
    let picked = (0..ROWS)
        .map(|row| (row * 41 % 512) as u32)
        .collect::<Vec<_>>();
    host.copy_host_to_device(
        ranks[0].2,
        indices,
        &picked
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut outputs = Vec::new();
    let mut planes = Vec::new();
    for (rank, &(_, pool, stream)) in ranks.iter().enumerate() {
        let buffer = host
            .allocate_device_buffer(pool, stream, ROWS * 9, 16)
            .unwrap();
        outputs.push(buffer);
        for (view, width, offset) in [("codes", 8, 0), ("scales", 1, ROWS * 8)] {
            planes.push(MappedGatherPlaneV1 {
                image,
                rank: rank as u32,
                view_id: view.as_bytes().to_vec(),
                row_bytes: width,
                index_offset: 0,
                stream,
                destination: DeviceBufferSpanV1::new(buffer, offset, ROWS * width),
            });
        }
    }
    let read = |rank: usize| {
        let mut bytes = vec![0; (ROWS * 9) as usize];
        host.copy_device_to_host(
            ranks[rank].2,
            &mut bytes,
            DeviceBufferSpanV1::new(outputs[rank], 0, ROWS * 9),
        )
        .unwrap();
        bytes
    };
    let expected = |rank: usize| {
        let mut bytes = picked
            .iter()
            .flat_map(|&row| codes()[row as usize * 8..][..8].to_vec())
            .collect::<Vec<_>>();
        bytes.extend(
            picked
                .iter()
                .map(|&row| scales()[rank * 512 + row as usize]),
        );
        bytes
    };
    let clear = || {
        for (rank, &buffer) in outputs.iter().enumerate() {
            host.copy_host_to_device(
                ranks[rank].2,
                DeviceBufferSpanV1::new(buffer, 0, ROWS * 9),
                &vec![0xa5; (ROWS * 9) as usize],
            )
            .unwrap();
        }
    };
    let settle = |gather| {
        for &(_, _, stream) in &ranks {
            host.wait_mapped_gather(gather, stream).unwrap();
            host.synchronize_stream(stream).unwrap();
        }
        complete(host, gather)
    };
    // Released while pending: nothing it read reaches a later gather.
    let pending = host
        .create_mapped_gather(ranks[0].2, indices, &planes)
        .unwrap();
    host.enqueue_mapped_gather(pending, ROWS).unwrap();
    host.release_mapped_gather(pending).unwrap();
    let gather = host
        .create_mapped_gather(ranks[0].2, indices, &planes)
        .unwrap();
    for round in 0..2 {
        clear();
        host.enqueue_mapped_gather(gather, ROWS).unwrap();
        assert!(settle(gather).unwrap().is_some(), "round {round}");
        for rank in 0..2 {
            assert!(read(rank) == expected(rank), "round {round} rank {rank}");
        }
    }
    // Cut the shard inside the codes table: the scales rows, and the codes
    // rows from 400, are gone, so the large reads fail after writing rows.
    let shard = std::fs::OpenOptions::new()
        .write(true)
        .open(root.0.join("model.safetensors"))
        .unwrap();
    let header = u64::from_le_bytes(
        std::fs::read(root.0.join("model.safetensors")).unwrap()[..8]
            .try_into()
            .unwrap(),
    );
    shard.set_len(8 + header + 400 * 8).unwrap();
    clear();
    host.enqueue_mapped_gather(gather, ROWS).unwrap();
    assert_eq!(
        settle(gather),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(
        host.enqueue_mapped_gather(gather, ROWS),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    host.release_mapped_gather(gather).unwrap();
    for buffer in outputs.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_device_buffer(ids).unwrap();
    host.release_compiled_image(image).unwrap();
    close_ranks(host, ranks);
    assert_eq!(backend.live_resource_count(), 0);
}

#[test]
fn large_layer_gathers_preserve_bytes_and_consumer_order() {
    exercise(FakeHostDeviceV1::new(), [0, 1], "layer-fake", &COUNTS);
}

#[test]
fn large_gathers_settle_release_and_fail_without_leaks() {
    settle_and_fail(FakeHostDeviceV1::new(), [0, 1], "settle-fake");
}

/// One candidate: its name, its selectors' values, and the log line that
/// shows it engaged.
struct Candidate {
    name: &'static str,
    selectors: &'static [(&'static str, &'static str)],
    engaged: &'static str,
}

const CANDIDATES: [Candidate; 2] = [
    Candidate {
        name: "reader-files",
        selectors: &[("CHOPINFER_MAPPED_READER_FILES", "1")],
        engaged: "\"phase\":\"reader-files\",\"descriptions\"",
    },
    Candidate {
        name: "direct",
        selectors: &[
            ("CHOPINFER_MAPPED_READER_FILES", "1"),
            ("CHOPINFER_MAPPED_LARGE_GATHER_DIRECT", "1"),
        ],
        engaged: "\"phase\":\"direct-gather\"",
    },
];

/// Runs test `name` of this binary in a child process under each
/// candidate, which the child names in `child`, and checks the log line
/// that shows it engaged.
fn under_candidates(name: &str, child: &str) {
    for Candidate {
        name: candidate,
        selectors,
        engaged,
    } in CANDIDATES
    {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", name, "--nocapture", "--test-threads", "1"])
            .env(child, candidate)
            .env_remove("CHOPINFER_MAPPED_READER_FILES")
            .env_remove("CHOPINFER_MAPPED_LARGE_GATHER_DIRECT");
        for (key, value) in selectors {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        let (out, errors) = (
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(output.status.success(), "{candidate}: {out}\n{errors}");
        assert!(out.contains("1 passed; 0 failed"), "{candidate}: {out}");
        assert!(errors.contains(engaged), "{candidate}: {errors}");
    }
}

#[test]
fn large_layer_gathers_on_candidate_routes_preserve_bytes_and_release() {
    const CHILD: &str = "HOST_MAPPED_LAYER_ROUTE";
    if std::env::var_os(CHILD).is_some() {
        exercise(FakeHostDeviceV1::new(), [0, 1], "route-fake", &REUSED);
        settle_and_fail(FakeHostDeviceV1::new(), [0, 1], "route-settle-fake");
        runtime_engine::logging::flush();
        return;
    }
    under_candidates(
        "layer_gathers::large_layer_gathers_on_candidate_routes_preserve_bytes_and_release",
        CHILD,
    );
}

#[cfg(feature = "cuda-fake-driver-tests")]
#[test]
fn large_layer_gathers_through_the_cuda_fake_driver_on_every_route() {
    const CHILD: &str = "HOST_MAPPED_LAYER_DRIVER_ROUTE";
    let run = || {
        runtime_cuda::fake_driver::reset();
        let device = || runtime_host::cuda::CudaHostDeviceV1::new().unwrap();
        exercise(device(), [0, 1], "layer-driver", &REUSED);
        settle_and_fail(device(), [0, 1], "settle-driver");
        assert_eq!(runtime_cuda::fake_driver::ordering_violations(), 0);
    };
    if std::env::var_os(CHILD).is_some() {
        run();
        runtime_engine::logging::flush();
        return;
    }
    run();
    under_candidates(
        "layer_gathers::large_layer_gathers_through_the_cuda_fake_driver_on_every_route",
        CHILD,
    );
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
        &COUNTS,
    );
}
