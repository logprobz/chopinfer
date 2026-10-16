//! Context-parallel stripes on the fake device: rank bytes, not metadata.
use runtime_bundle_abi::{DeviceBufferSpanV1, HostServicesBackendV1};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::offload::{KvOffloadExecutorV1, KvOffloadOptionsV1};
use runtime_host::{
    DeclaredKvContractV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
    ServingKvPolicyV1,
};
use runtime_kv::{
    ContextParallelPagesV1, DeclaredPageLayoutV1, HostTier, KvPool, KvTransferStatus,
    PrefixChainV1, RankPlacementV1, StateLifetimeV1,
};

/// Every rank's bytes survive a partial-stripe prefix copy-on-write, an
/// offload and a reload into new pages, for every tail length past one
/// stripe at group sizes 1, 2 and 4 on four fake ranks. Each rank's pages
/// hold distinct bytes, so a dropped or crossed rank copy fails.
#[test]
fn rank_bytes_survive_stripe_copies_offload_and_reload() {
    let mut cases = 0;
    for size in [1, 2, 4] {
        let pages = ContextParallelPagesV1::new(size, 4, 4).unwrap();
        for tail in 1..=pages.stripe_tokens() {
            let tokens = pages.stripe_tokens() + tail;
            let layout = DeclaredPageLayoutV1::new(
                16,
                1,
                4,
                64,
                4,
                RankPlacementV1::Sharded,
                StateLifetimeV1::Request,
                128,
            )
            .unwrap();
            let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
                .with_kv_contract(DeclaredKvContractV1::new(b"probe".to_vec(), layout))
                .with_context_parallel_size(size)
                .with_serving_kv(ServingKvPolicyV1 {
                    maximum_pages: Some(32),
                    ..Default::default()
                });
            let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
            let mut streams = Vec::new();
            for rank in 0..4 {
                let context = backend.select_context(rank, rank as i32).unwrap();
                backend.select_memory_pool(context).unwrap();
                streams.push(backend.create_stream(context, 0).unwrap());
            }
            assert_eq!(backend.acquire_serving_kv(b"probe").unwrap(), 32);
            let views: Vec<_> = (0..4)
                .map(|r| backend.resolve_serving_kv_pool(r).unwrap())
                .collect();
            let write_page = |rank: usize, page: usize, bytes: &[u8]| {
                backend
                    .copy_host_to_device(
                        streams[rank],
                        DeviceBufferSpanV1::new(views[rank].buffer, page as u64 * 64, 64),
                        bytes,
                    )
                    .unwrap();
            };
            let read_page = |rank: usize, page: usize| {
                let mut bytes = vec![0; 64];
                backend
                    .copy_device_to_host(
                        streams[rank],
                        &mut bytes,
                        DeviceBufferSpanV1::new(views[rank].buffer, page as u64 * 64, 64),
                    )
                    .unwrap();
                bytes
            };
            let mut pool = KvPool::new(pages.geometry(32).unwrap(), 8).unwrap();
            let mut tier = HostTier::new(&pool, 32).unwrap();
            let source = pool.create_sequence(128).unwrap();
            pool.append(source, tokens).unwrap();
            let chain = PrefixChainV1::new(
                pages.stripe_tokens(),
                (0..tokens as u32 + 4).collect::<Vec<_>>().into(),
                vec![],
            )
            .unwrap();
            let key = chain.key_at(tokens).unwrap();
            let state_pages = pool
                .insert_anchor(
                    key,
                    chain.chain(tokens / pages.stripe_tokens()).unwrap(),
                    source,
                    1,
                )
                .unwrap()
                .unwrap();
            assert_eq!(pool.lookup_anchor(&chain, tokens), Some((key, tokens)));
            let table = pool.snapshot(source).unwrap().blocks().to_vec();
            let all: Vec<_> = table.iter().chain(&state_pages).copied().collect();
            for rank in 0..4 {
                for (index, page) in all.iter().enumerate() {
                    write_page(rank, page.index(), &[(rank * 32 + index + 1) as u8; 64]);
                }
            }
            let hit = pool.resume_prefix(&chain, tokens, 128).unwrap().unwrap();
            pool.finish_restore(hit.handle()).unwrap();
            let append = pool.prepare_append_batch(&[(hit.handle(), 1)]).unwrap();
            backend.prepare_serving_batch(&append).unwrap();
            if let Some(copy) = append.appends()[0].copy() {
                for rank in 0..4 {
                    assert_eq!(
                        read_page(rank, copy.source().index()),
                        read_page(rank, copy.destination().index())
                    );
                }
            } else {
                assert_eq!(tail, pages.stripe_tokens());
            }
            pool.commit_prepared(append.id()).unwrap();
            pool.release(hit.handle()).unwrap();
            backend
                .initialize_kv_offload(
                    KvOffloadOptionsV1 {
                        budget_bytes: 32 * 4 * 64,
                        ..Default::default()
                    },
                    32 * 1024 * 1024 * 1024,
                )
                .unwrap()
                .unwrap();
            let (anchor, transfer) = tier.begin_offload(&mut pool, key, Some(1)).unwrap();
            let id = transfer.id();
            backend.submit(transfer).unwrap();
            let completion = backend.wait(id).unwrap().completion().unwrap();
            assert_eq!(completion.status(), KvTransferStatus::Complete);
            tier.complete_transfer(&mut pool, completion).unwrap();
            pool.release(source).unwrap();
            pool.remove_anchor(key).unwrap();
            for rank in 0..4 {
                backend
                    .copy_host_to_device(
                        streams[rank],
                        views[rank],
                        &vec![0; views[rank].length as usize],
                    )
                    .unwrap();
            }
            let destination = pool.create_initialized_sequence(tokens, 128, 1).unwrap();
            let transfer = tier.begin_reload(&mut pool, anchor, &destination).unwrap();
            let id = transfer.id();
            backend.submit(transfer).unwrap();
            let completion = backend.wait(id).unwrap().completion().unwrap();
            assert_eq!(completion.status(), KvTransferStatus::Complete);
            tier.complete_transfer(&mut pool, completion).unwrap();
            let restored = pool.snapshot(destination.handle()).unwrap();
            for rank in 0..4 {
                for (index, page) in restored
                    .blocks()
                    .iter()
                    .chain(destination.state())
                    .enumerate()
                {
                    assert_eq!(
                        read_page(rank, page.index()),
                        vec![(rank * 32 + index + 1) as u8; 64],
                        "size={size} tail={tail} rank={rank}"
                    );
                }
            }
            pool.finish_initialized_restore(destination.handle())
                .unwrap();
            tier.audit(&pool).unwrap();
            backend.drain().unwrap();
            cases += 1;
        }
    }
    assert_eq!(cases, 4 + 8 + 16);
}
