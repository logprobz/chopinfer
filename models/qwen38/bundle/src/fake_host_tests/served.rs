//! Instances serving more than 32 sequences (CONC-QWEN): slot tables,
//! pools and the ring follow the served count, prefill phases carry more
//! than 32 items, and decode buckets above 32 run the prefill entries.

use super::*;
use crate::sm120::device::residency;

/// A step of `(request, slot, previous tokens, inputs)` items, each on one
/// KV page numbered by its slot.
fn slots_step(items: &[(u64, u64, u64, &[u32])]) -> BatchRequestV1 {
    let (batch, appends): (Vec<_>, Vec<_>) = items
        .iter()
        .map(|&(request, slot, previous, inputs)| {
            let count = inputs.len() as u64;
            (
                BatchItemV1::new(
                    request,
                    previous > 0 && count == 1,
                    count as u32,
                    true,
                    inputs.to_vec(),
                ),
                BatchAppendV1::new(previous, count, previous + count, vec![slot], None),
            )
        })
        .unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|item| item.1).collect())
}

fn serving(count: u32) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::default());
    host.served.set(Some(count));
    host
}

/// The last allocation of `bytes` bytes, as its journal line.
fn allocated(host: &FakeHost, bytes: u64) -> usize {
    host.journal
        .borrow()
        .iter()
        .filter(|line| **line == format!("allocate {bytes} 256"))
        .count()
}

/// Launches of `symbol` from launch `from` on, with their grids.
fn grids(host: &FakeHost, from: usize, symbol: &str) -> Vec<[u32; 3]> {
    host.launches.borrow()[from..]
        .iter()
        .filter(|(_, name, _)| name == symbol)
        .map(|&(_, _, grid)| grid)
        .collect()
}

/// Symbols of the decode kernels, which hold at most 32 rows.
fn decode_entry(name: &str) -> bool {
    let bucket = |prefix: &str| {
        name.strip_prefix(prefix).is_some_and(|rest| {
            rest.split('_')
                .next()
                .is_some_and(|digits| digits.chars().all(|c| c.is_ascii_digit()))
        })
    };
    name.contains("hc_up_mix")
        || name.contains("hc_up_decode")
        || name.contains("decode_route")
        || name.contains("allreduce_hc_decode")
        || name.contains("_decode_t")
        || [
            "chop_gemm_qwen38_lin_in_c",
            "chop_gemm_qwen38_attn_in_c",
            "chop_gemm_qwen38_hc_in_c",
            "chop_gemm_qwen38_hc_down_c",
            "chop_gemm_qwen38_lin_qkv_c",
            "chop_gemm_qwen38_out_c",
            "chop_gemm_qwen38_router_c",
            "chop_gemm_qwen38_ple_key_c",
            "chop_gemm_qwen38_shared_gate_up_c",
            "chop_gemm_qwen38_shared_down_c",
            "chop_gemm_qwen38_head_f32_m",
            "chop_gemm_qwen38_shared_gate_up_m",
            "chop_gemm_qwen38_shared_down_m",
        ]
        .into_iter()
        .any(bucket)
}

#[test]
fn served_counts_size_slot_state_rings_and_buffers() {
    let shape = Shape::TP4;
    for (count, capacity) in [(8, 8), (32, 32), (33, 64), (100, 128), (256, 256)] {
        let host = serving(count);
        let instance = create(&host);
        let handle = instance.handle();
        // The wide all-reduce's cubins load from 33 served sequences on, and
        // the instance logs whether graphs above 32 rows take it.
        assert_resolved(&host);
        let said = host
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("wide all-reduce on (CHOPINFER_QWEN38_WIDE_COMM=1)"));
        assert_eq!(said, capacity > 32, "{count}");
        let [conv, delta, ple] = crate::instance::pools(handle);
        let slots = u64::from(count);
        // The pools' planes add the anchor items' entries.
        let entries = slots + u64::from(ANCHOR_ITEMS);
        assert_eq!(conv.length, 36 * entries * shape.conv_entry_bytes);
        assert_eq!(delta.length, 36 * entries * shape.delta_entry_bytes);
        assert_eq!(ple.length, slots * PLE_ENTRY_BYTES);
        // The ring holds the largest bucket's slots: the state the host
        // allocated for the served count.
        let ring = crate::instance::state_views(handle, 0)[3];
        assert_eq!(ring.length, capacity * 12 * (4096 + 192));
        assert_eq!(2 * ring.length, host.layout().2);
        // Rows of the decode and verify graphs: never fewer than the 32 a
        // verify step may carry.
        let rows = if capacity <= 32 { 32 } else { 256 };
        let [local, _, _, gathered] = crate::instance::logits_spans(handle)[0];
        assert_eq!(local.length, 32 * u64::from(shape.vocab_per_rank) * 4);
        assert_eq!(gathered.length, rows * u64::from(VOCAB) * 4);
        // One block-table row per served sequence, per phase and rank.
        assert!(
            allocated(&host, slots * 4 * u64::from(MAX_PAGES)) >= 2 * 4,
            "{count}"
        );
        drop(instance);
    }
}

/// A prefill of more than 32 items, then decode buckets of 64, 128 and 256
/// items on the prefill entries, with graphs captured per bucket.
#[test]
fn wide_decode_buckets_run_the_prefill_entries() {
    for served in [100, 256] {
        let host = serving(served);
        let mut instance = create(&host);
        let prompts = (0..served)
            .map(|slot| [slot % 50 + 7, 9])
            .collect::<Vec<_>>();
        let prefill = (0..u64::from(served))
            .map(|slot| (1000 + slot, slot, 0, &prompts[slot as usize][..]))
            .collect::<Vec<_>>();
        let before = host.launches.borrow().len();
        let mut sink = ViewSink::default();
        instance
            .execute(&slots_step(&prefill).encode().unwrap(), &mut sink)
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        let names = launched(&host, before);
        // Every item emits one row: more than the decode heads' buckets.
        assert_eq!(count(&names, "chop_gemm_qwen38_head_f32_prefill_v1"), 4);
        // Every item leaves a partial chunk of its own.
        let prepare = grids(&host, before, "chop_linear_qwen38_delta_prefill_prepare_v1");
        assert_eq!(prepare.len(), 4 * 36);
        let chunks = (2 * served).div_ceil(64) + served;
        assert!(prepare.iter().all(|grid| *grid == [4, chunks, 1]));
        let rows = |sink: &ViewSink| sink.0.iter().map(|view| view.raw().row_count).sum::<u32>();
        assert_eq!(rows(&sink), served);

        // Decode every item, then fewer than half of them.
        let mut lengths = vec![2; served as usize];
        for live in [served, 33] {
            let bucket = live.next_power_of_two();
            let decode = (0..u64::from(live))
                .map(|slot| (1000 + slot, slot, lengths[slot as usize], &[11][..]))
                .collect::<Vec<_>>();
            let before = host.launches.borrow().len();
            let captures = host.captures.get();
            let mut sink = ViewSink::default();
            instance
                .execute(&slots_step(&decode).encode().unwrap(), &mut sink)
                .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            // One capture per piece and rank of the new bucket's graph.
            assert!(host.captures.get() > captures, "{served} {live}");
            let names = launched(&host, before);
            assert!(
                !names.iter().any(|name| decode_entry(name)),
                "{:?}",
                names.iter().find(|name| decode_entry(name))
            );
            for symbol in [
                "chop_gemm_qwen38_hc_down_prefill_v1",
                "chop_gemm_qwen38_hc_inject_v1",
                "chop_gemm_qwen38_hc_up_prefill_v1",
                "chop_row_qwen38_hc_bottleneck_v1",
                "chop_row_qwen38_hc_mix_v1",
                "chop_gemm_qwen38_lin_qkv_prefill_v1",
                "chop_gemm_qwen38_attn_qkv_prefill_v1",
                "chop_gemm_qwen38_router_prefill_v1",
                "chop_moe_qwen38_router_v1",
                "chop_moe_qwen38_permute_scatter_v1",
                "chop_gemm_qwen38_shared_gate_up_prefill_v1",
                "chop_gemm_qwen38_head_f32_prefill_v1",
            ] {
                assert!(count(&names, symbol) > 0, "{served} {live} {symbol}");
            }
            // The wide all-reduce up to its four-rank cap of 128 rows, the
            // phased pair above: 96 reduces per rank.
            let wide = model_qwen38_contract::sm120::comm::decode::wide_symbol(4).unwrap();
            let phased = prefill_comm_symbols()[0];
            let (expected_wide, expected_phased) = if bucket <= 128 { (96, 0) } else { (0, 96) };
            assert_eq!(
                (count(&names, wide), count(&names, phased)),
                (4 * expected_wide, 4 * expected_phased),
                "{served} {live}"
            );
            // The item kernels run the whole bucket.
            let attention = grids(&host, before, "chop_qsa_qwen38_attn_bf16_decode_v1");
            let splits = match bucket {
                64 => 8,
                128 => 4,
                _ => 2,
            };
            assert_eq!(attention.len(), 4 * 12);
            assert!(attention.iter().all(|grid| *grid == [bucket, 1, splits]));
            let scores = grids(&host, before, "chop_qsa_qwen38_block_scores_decode_v1");
            assert_eq!(scores.len(), 4 * 12);
            assert!(scores.iter().all(|grid| *grid == [376, 1, 1]));
            let delta = grids(&host, before, "chop_linear_qwen38_delta_verify_v1");
            assert!(delta.iter().all(|grid| *grid == [8, 12, bucket]));
            let conv = grids(&host, before, "chop_linear_qwen38_conv_verify_v1");
            assert!(conv.iter().all(|grid| *grid == [10, bucket, 1]));
            let ngram = grids(&host, before, "chop_row_qwen38_ngram_decode_v1");
            assert!(ngram.iter().all(|grid| grid[0] == bucket));
            assert_eq!(rows(&sink), live);
            instance.commit().unwrap();
            lengths[..live as usize]
                .iter_mut()
                .for_each(|length| *length += 1);
        }
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

/// Creates a 100-sequence instance on `host`, returning whether it created.
fn creates(host: &Rc<FakeHost>) -> bool {
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &configuration_served(host.shape(), host.layout(), &[], host.served()),
        )
        .is_ok()
}

/// A device short of clusters for the wide entry alone keeps the wide
/// buckets on the general entry's phased pair; one short for the phased pair
/// (both are 17 clusters at their largest), an MPS client, or a host that
/// reports no count fails create instead of risking a hang.
#[test]
fn wide_buckets_fall_back_unless_the_device_keeps_the_wide_grids_resident() {
    let wide = model_qwen38_contract::sm120::comm::decode::wide_symbol(4).unwrap();
    let host = serving(100);
    host.residency_by_symbol
        .borrow_mut()
        .insert(wide.into(), residency(8, 0));
    let mut instance = create(&host);
    let wide_env = std::env::var(crate::launch::WIDE_COMM).as_deref() != Ok("0");
    assert_eq!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("wide all-reduce off, graphs take the phased stages")),
        wide_env
    );
    let prompts = (0..40_u32)
        .map(|slot| [slot % 50 + 7, 9])
        .collect::<Vec<_>>();
    let prefill = (0..40_u64)
        .map(|slot| (1000 + slot, slot, 0, &prompts[slot as usize][..]))
        .collect::<Vec<_>>();
    instance
        .execute(
            &slots_step(&prefill).encode().unwrap(),
            &mut ViewSink::default(),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let decode = (0..40_u64)
        .map(|slot| (1000 + slot, slot, 2, &[11][..]))
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    instance
        .execute(
            &slots_step(&decode).encode().unwrap(),
            &mut ViewSink::default(),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let names = launched(&host, before);
    assert_eq!(count(&names, wide), 0);
    assert_eq!(count(&names, prefill_comm_symbols()[0]), 4 * 96);
    instance.cancel().unwrap();
    instance.destroy().unwrap();

    let failed = |host: &Rc<FakeHost>, needle: &str| {
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("create failed") && line.contains(needle))
    };
    // 16 clusters hold neither the wide entry's nor the phased pair's 17.
    let host = serving(100);
    host.residency.set(Some(residency(16, 0)));
    assert!(!creates(&host));
    assert!(failed(&host, "chop_comm_qwen38_"));
    // An MPS client: multi-rank serving under MPS is unsupported.
    let host = serving(100);
    host.residency
        .set(Some(residency(46, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT)));
    assert!(!creates(&host));
    assert!(failed(&host, "MPS"));
    // Every decode bucket's entry is checked: one short only for the decode
    // entry fails create.
    let decode = model_qwen38_contract::sm120::comm::decode::symbol_v2(4).unwrap();
    let host = serving(100);
    host.residency_by_symbol
        .borrow_mut()
        .insert(decode.into(), residency(0, 0));
    assert!(!creates(&host));
    assert!(failed(&host, decode));
    // No reported count authorizes nothing.
    let host = serving(100);
    host.residency
        .set(Some(BundleRawClusterResidencyV1::unreported()));
    assert!(!creates(&host));
    assert!(failed(&host, "CHOPINFER_COMM_RESIDENCY=unchecked"));
}

/// Buckets of 32 and fewer keep the decode entries at any served count.
#[test]
fn decode_buckets_keep_the_decode_entries_when_serving_more() {
    let host = serving(200);
    let mut instance = crate::instance::with_hc_precompute(true, || create(&host));
    let prefill = (0..32_u64)
        .map(|slot| (10 + slot, 150 + slot, 0, &PROMPT[..]))
        .collect::<Vec<_>>();
    run(&host, &mut instance, "prefill", slots_step(&prefill));
    instance.commit().unwrap();
    let decode = prefill
        .iter()
        .map(|&(request, slot, _, _)| (request, slot, 3, &[4][..]))
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "decode", slots_step(&decode));
    let names = launched(&host, before);
    for symbol in [
        super::sm120::HC_PRECOMPUTED_MIX,
        "chop_gemm_qwen38_lin_in_c32_v1",
        "chop_gemm_qwen38_head_f32_m32_v1",
        "chop_moe_qwen38_decode_route_parallel_t32_v1",
        super::decode_comm(4, 32),
    ] {
        assert!(count(&names, symbol) > 0, "{symbol}");
    }
    assert!(!names.iter().any(|name| name.ends_with("_prefill_v1")));
    let mut tickets = BTreeMap::new();
    for ((stream, symbol, grid), arguments) in host.launches.borrow()[before..]
        .iter()
        .zip(&host.launch_arguments.borrow()[before..])
    {
        if symbol != "chop_moe_qwen38_decode_route_parallel_t32_v1" {
            continue;
        }
        assert_eq!(*grid, [32, 1, 1]);
        assert_eq!(arguments.len(), 11);
        let KernelArgumentV1::DeviceBuffer(ticket) = arguments[9] else {
            panic!("parallel route requires a ticket buffer");
        };
        assert_eq!(ticket.length, super::sm120::PARALLEL_ROUTE_TICKET_BYTES);
        assert!(
            host.journal
                .borrow()
                .contains(&format!("zero {stream} {ticket:?}"))
        );
        if let Some(prior) = tickets.insert(*stream, ticket) {
            assert_eq!(prior, ticket);
        }
    }
    assert_eq!(tickets.len(), 4);
    // Only the parallel route binds a ticket, so nothing else can leave a
    // stale count in it; the zero at create is its only other write.
    for ((_, symbol, _), arguments) in host
        .launches
        .borrow()
        .iter()
        .zip(host.launch_arguments.borrow().iter())
    {
        if symbol.contains("decode_route_parallel") {
            continue;
        }
        for argument in arguments {
            if let KernelArgumentV1::DeviceBuffer(span) = argument {
                assert!(
                    tickets.values().all(|ticket| span.buffer != ticket.buffer
                        || span.offset + span.length <= ticket.offset
                        || ticket.offset + ticket.length <= span.offset),
                    "{symbol} binds a route ticket"
                );
            }
        }
    }
    for ticket in tickets.values() {
        let writes = host
            .journal
            .borrow()
            .iter()
            .filter(|line| !line.starts_with("launch ") && line.contains(&format!("{ticket:?}")))
            .count();
        assert_eq!(
            writes, 1,
            "the ticket is zeroed once and never copied or uploaded"
        );
    }
    instance.commit().unwrap();
}

#[test]
fn a_slot_past_32_captures_and_restores_its_snapshot() {
    let host = serving(100);
    let mut instance = create(&host);
    let handle = instance.handle();
    host.backing.set(true);
    run(
        &host,
        &mut instance,
        "prefill",
        slots_step(&[(10, 90, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let cold = fill_slot(&host, handle, 90);
    let first = snapshot_pages(&host, 100);
    instance.capture_prefix_state(10, 3, &first).unwrap();
    let arguments = host.launch_arguments.borrow().last().unwrap().clone();
    // The copy names the pools' 100 slots and 2 anchor entries, and the
    // ring's 128 slots.
    assert_eq!(
        arguments[12..14],
        [
            KernelArgumentV1::ScalarU32(102),
            KernelArgumentV1::ScalarU32(128)
        ]
    );
    instance
        .restore_prefix_state_in_slot(11, &PROMPT, &first, 99)
        .unwrap();
    assert_eq!(crate::instance::owner(handle, 99), Some(11));
    for (rank, cold) in cold.iter().enumerate() {
        assert!(slot_bytes(&host, handle, rank, 99) == *cold, "rank {rank}");
    }
    // The last slot is 99: 100 is no slot of this instance.
    assert!(
        instance
            .restore_prefix_state_in_slot(12, &PROMPT, &first, 100)
            .is_err()
    );
    let before = host.launches.borrow().len();
    run(
        &host,
        &mut instance,
        "hit",
        slots_step(&[(11, 99, 3, &[9, 8])]),
    );
    assert!(!launched(&host, before).is_empty());
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn served_counts_and_slots_past_them_are_refused() {
    // Past the package maximum.
    let host = serving(257);
    assert!(
        api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_served(Shape::TP4, host.layout(), &[], 257),
            )
            .is_err()
    );
    // Past what a TP2 rank fits (its package maximum, 64), and at it.
    for (served, fits) in [(128, false), (64, true)] {
        let host = Rc::new(FakeHost::at(Shape::TP2));
        host.ngram_host.set(ngram_in_host_memory(Shape::TP2));
        host.served.set(Some(served));
        let created = api().create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_served(Shape::TP2, host.layout(), &[], served),
        );
        match created {
            Ok(mut instance) => {
                assert!(fits, "TP2 C{served}");
                instance.destroy().unwrap();
            }
            Err(error) => {
                assert!(!fits, "TP2 C{served}: {error:?}");
                assert_eq!(
                    error,
                    BundleAbiError::Status(BundleStatusV1::InvalidArgument)
                );
            }
        }
    }
    // MTP ring allocation follows the served count too.
    let host = serving(64);
    host.drafter.set(true);
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_served(Shape::TP4, host.layout(), &[], 64),
        )
        .unwrap();
    instance.destroy().unwrap();
    // Ring state for another count than the record's.
    let host = serving(64);
    let error = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_served(Shape::TP4, host.layout(), &[], 32),
        )
        .map(|_| ())
        .unwrap_err();
    assert_eq!(error, BundleAbiError::Status(BundleStatusV1::Unsupported));
    // A step on a slot past the served count, or with more items.
    let host = serving(40);
    let mut instance = create(&host);
    let mut sink = ViewSink::default();
    assert!(
        instance
            .execute(
                &slots_step(&[(1, 40, 0, &PROMPT)]).encode().unwrap(),
                &mut sink
            )
            .is_err()
    );
    let prompts = (0..41_u64)
        .map(|slot| (slot + 1, slot, 0, &PROMPT[..]))
        .collect::<Vec<_>>();
    assert!(
        instance
            .execute(&slots_step(&prompts).encode().unwrap(), &mut sink)
            .is_err()
    );
    let prompts = (0..40_u64)
        .map(|slot| (slot + 1, slot, 0, &PROMPT[..]))
        .collect::<Vec<_>>();
    run(&host, &mut instance, "prefill", slots_step(&prompts));
    instance.commit().unwrap();
}

/// TP2's cap on the wide all-reduce: every bucket above 32, the phased
/// pair above. Its package maximum is 64 sequences.
#[test]
fn wide_all_reduce_caps_follow_the_rank_count() {
    let (shape, wide_buckets, lives) = (Shape::TP2, &[64][..], &[33_u64][..]);
    let served = model_qwen38_contract::package::maximum_sequences(
        shape,
        false,
        model_qwen38_contract::package::Encoders::NONE,
    );
    let host = Rc::new(FakeHost::at(shape));
    host.ngram_host.set(ngram_in_host_memory(shape));
    host.served.set(Some(served));
    let mut instance = create(&host);
    assert_resolved(&host);
    let ranks = shape.ranks();
    let prompts = (0..served).map(|slot| [slot % 50 + 7]).collect::<Vec<_>>();
    let prefill = (0..u64::from(served))
        .map(|slot| (1000 + slot, slot, 0, &prompts[slot as usize][..]))
        .collect::<Vec<_>>();
    run(&host, &mut instance, "prefill", slots_step(&prefill));
    instance.commit().unwrap();
    let wide = model_qwen38_contract::sm120::comm::decode::wide_symbol(ranks).unwrap();
    let phased = prefill_comm_symbols()[0];
    let mut lengths = vec![1; served as usize];
    for &live in lives {
        let bucket = (live as u32).next_power_of_two();
        let decode = (0..live)
            .map(|slot| (1000 + slot, slot, lengths[slot as usize], &[11][..]))
            .collect::<Vec<_>>();
        let before = host.launches.borrow().len();
        run(&host, &mut instance, "decode", slots_step(&decode));
        let names = launched(&host, before);
        let reduces = 96 * ranks as usize;
        let expected = if wide_buckets.contains(&bucket) {
            (reduces, 0)
        } else {
            (0, reduces)
        };
        assert_eq!(
            (count(&names, wide), count(&names, phased)),
            expected,
            "{shape:?} {live}"
        );
        instance.commit().unwrap();
        lengths[..live as usize]
            .iter_mut()
            .for_each(|length| *length += 1);
    }
    instance.destroy().unwrap();
}

#[test]
fn c16_keeps_the_original_route_without_a_measured_parallel_gain() {
    let host = serving(32);
    let mut instance = create(&host);
    let prefill = (0..16_u64)
        .map(|slot| (10 + slot, slot, 0, &PROMPT[..]))
        .collect::<Vec<_>>();
    run(&host, &mut instance, "prefill", slots_step(&prefill));
    instance.commit().unwrap();
    let decode = prefill
        .iter()
        .map(|&(request, slot, _, _)| (request, slot, 3, &[4][..]))
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "decode", slots_step(&decode));
    let names = launched(&host, before);
    assert!(count(&names, "chop_moe_qwen38_decode_route_t16_v1") > 0);
    assert!(
        !names
            .iter()
            .any(|name| name.contains("decode_route_parallel"))
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
}
