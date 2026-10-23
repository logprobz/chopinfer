//! A served glm5next instance at context-parallel sizes 2 and 4, with the
//! projection stores bound, against the size-1 instance: prefix-cache
//! reuse, offload to the host tier and preemption on striped pages, and
//! graph capture and replay at every decode and verify bucket.
//!
//! The host's side runs as the serving host runs it: a `runtime_kv::KvPool`
//! over the striped geometry (one pool block is one local page on every
//! rank), anchors holding the bundle's snapshot pages, and a `HostTier`
//! whose offload and reload plans this test carries out page by page on
//! every rank, as the host's copy executor does. The fake kernels write
//! nothing, so the test writes what a step's kernels would: physical page
//! `j` holds `page(j, rank)` on each of its owners (the ranks whose group
//! rank is `j % size`) at every size, and the request state holds the same
//! bytes per rank at every size.

use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::fake_host::check_journal;
use model_glm5next_contract::contract::{DECODE_BATCH, Shape};
use runtime_bundle_abi::{
    BundleDeviceBufferHandleV1, BundleInstanceV1, BundleKeptRowsV1, DeviceBufferSpanV1,
    HostServicesTableV1,
};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
use runtime_kv::{
    BlockId, ContextParallelPagesV1, HostTier, KvPool, KvTransferPlan, KvTransferStatus,
    PrefixChainV1,
};

use crate::context_parallel_served_tests::{assert_composes, outside_the_group, record};
use crate::context_parallel_tests::{fill_slot, slot_bytes};
use crate::fake_host_tests::{FakeHost, Host, KV_SLOT, NoRows, api, issued};

const RANKS: u32 = 4;
const PAGE_TOKENS: u64 = 64;
/// Local pages per rank of the fake host's serving KV, which the emulated
/// pool hands out.
const POOL_PAGES: usize = 256;

/// An FP8 instance at context-parallel size `size` with the stores bound.
fn create(host: &Rc<Host>, size: u32) -> BundleInstanceV1 {
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &record(size, Shape::TP4.kv_formats()[0]),
        )
        .unwrap_or_else(|error| panic!("size {size}: {error:?} {:?}", host.logs.borrow()))
}

/// One item of a step: `inputs` of `request` in state slot `slot` after
/// `previous` committed tokens over the striped page table `pages`, a
/// prompt chunk (logits at its end) when `prompt`, otherwise a decode row
/// or a verify item.
struct Item<'a> {
    request: u64,
    slot: u64,
    previous: u64,
    inputs: &'a [u32],
    prompt: bool,
    pages: Vec<u64>,
}

fn step(items: &[Item]) -> Vec<u8> {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for item in items {
        let count = item.inputs.len() as u32;
        let inputs = item.inputs.to_vec();
        batch.push(if item.prompt {
            BatchItemV1::new(item.request, false, count, true, inputs)
        } else if count == 1 {
            BatchItemV1::new(item.request, true, 1, true, inputs)
        } else {
            BatchItemV1::verify(item.request, count, false, inputs)
        });
        let end = item.previous + u64::from(count);
        appends.push(BatchAppendV1::new(
            item.previous,
            u64::from(count),
            end,
            item.pages.clone(),
            None,
        ));
        slots.push(item.slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

fn kept(request_id: u64, kept_rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id,
        kept_rows,
        reserved_tail: 0,
    }
}

fn execute(instance: &mut BundleInstanceV1, host: &FakeHost, size: u32, items: &[Item]) {
    instance
        .execute(&step(items), &mut NoRows::default())
        .unwrap_or_else(|error| panic!("size {size}: {error:?} {:?}", host.logs.borrow()));
}

fn ids(blocks: &[BlockId]) -> Vec<u64> {
    blocks.iter().map(|block| block.index() as u64).collect()
}

/// The bytes physical page `page` holds on rank `rank`, at every size.
fn page(page: u64, rank: u32, length: u64) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ page << 20 ^ u64::from(rank) << 52;
    (0..length.div_ceil(8))
        .flat_map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state.to_le_bytes()
        })
        .take(length as usize)
        .collect()
}

/// Local page `local` of rank `rank`'s serving KV.
fn local(rank: u32, local: u64, length: u64) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(
        BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
        local * length,
        length,
    )
}

/// The physical pages of a sequence's first `tokens` tokens over the
/// striped `table`, each with its owners' local pages: page `j` is local
/// page `table[j / size]` on every rank whose group rank is `j % size`.
fn owned(table: &[u64], tokens: u64, size: u32) -> Vec<(u64, u32, u64)> {
    let size = u64::from(size);
    (0..tokens.div_ceil(PAGE_TOKENS))
        .flat_map(|j| {
            (0..RANKS)
                .filter(move |&rank| u64::from(rank) % size == j % size)
                .map(move |rank| (j, rank, table[(j / size) as usize]))
        })
        .collect()
}

/// Writes the pages a prefill of `tokens` tokens over `table` fills.
fn write_kv(host: &FakeHost, table: &[u64], tokens: u64, size: u32, length: u64) {
    for (j, rank, at) in owned(table, tokens, size) {
        host.write(local(rank, at, length), &page(j, rank, length));
    }
}

/// Each owner's bytes of the first `tokens` tokens' pages over `table`, by
/// (physical page, rank).
fn held(
    host: &FakeHost,
    table: &[u64],
    tokens: u64,
    size: u32,
    length: u64,
) -> BTreeMap<(u64, u32), Vec<u8>> {
    owned(table, tokens, size)
        .into_iter()
        .map(|(j, rank, at)| ((j, rank), host.read(local(rank, at, length))))
        .collect()
}

/// Carries out a host-tier plan on every rank, page by page (`store` holds
/// the host pages per rank), and completes it.
fn transfer(
    host: &FakeHost,
    (pool, tier): (&mut KvPool, &mut HostTier),
    plan: KvTransferPlan,
    store: &mut BTreeMap<(u32, usize), Vec<u8>>,
    length: u64,
) {
    let out = plan.direction() == runtime_kv::KvTransferDirection::Out;
    for pair in plan.pages() {
        for rank in 0..RANKS {
            let device = local(rank, pair.device.index() as u64, length);
            if out {
                store.insert((rank, pair.host.index()), host.read(device));
            } else {
                host.write(device, &store[&(rank, pair.host.index())]);
            }
        }
    }
    let completion = plan
        .into_submission()
        .completion(KvTransferStatus::Complete);
    tier.complete_transfer(pool, completion).unwrap();
}

/// Tokens of the shared prompt: four full pages and part of a fifth, so
/// the last stripe is partial at every size (two full stripes and part of
/// a third at size 2, one and part of a second at size 4).
const PROMPT: u64 = 300;

/// What one size's prefix-cache run left: each check's KV bytes per owner
/// and request state per rank, and the launches the run issued.
struct Cached {
    kv: Vec<BTreeMap<(u64, u32), Vec<u8>>>,
    state: Vec<Vec<Vec<u8>>>,
    launches: Vec<String>,
}

/// Request 7 prefills the prompt in slot 0 and the host anchors it (the
/// bundle capturing slot 0 into the anchor's snapshot pages). Request 9
/// resumes from the anchor in slot 1, the host copying the shared partial
/// stripe on write, and decodes its next token. Then request 7 is
/// preempted: the host offloads the anchor to its tier, frees request 7's
/// sequence and slot and the anchor, and other work overwrites every
/// rank's pool and slot 0; the host reloads the anchor into new pages, the
/// bundle restores request 7 into slot 2, and it decodes on.
fn cached(size: u32) -> Cached {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host, size);
    let handle = instance.handle();
    host.backing.set(true);
    let start = host.device.journal.borrow().len();
    let stripes =
        ContextParallelPagesV1::new(size as usize, PAGE_TOKENS as usize, RANKS as usize).unwrap();
    let mut pool = KvPool::new(stripes.geometry(POOL_PAGES).unwrap(), 8).unwrap();
    let mut tier = HostTier::new(&pool, POOL_PAGES).unwrap();
    let mut store = BTreeMap::new();
    let state_pages = instance.describe_prefix_state().unwrap() as usize;
    let length = crate::instance::snapshot_layout(handle, 0).1.page_bytes;
    // The prompt and the token after it.
    let tokens = (0..=PROMPT as u32).map(|t| 1000 + t).collect::<Vec<_>>();
    let chain = PrefixChainV1::new(stripes.stripe_tokens(), tokens.clone().into(), vec![]).unwrap();
    let (prompt, next) = tokens.split_at(PROMPT as usize);
    let mut kv = Vec::new();
    let mut state = Vec::new();

    let source = pool.create_sequence(1024).unwrap();
    let append = pool
        .prepare_append_batch(&[(source, PROMPT as usize)])
        .unwrap();
    let table = ids(append.appends()[0].block_table());
    assert_eq!(
        table.len() as u64,
        PROMPT.div_ceil(PAGE_TOKENS * u64::from(size))
    );
    let item = |request, slot, previous, inputs, prompt, pages| Item {
        request,
        slot,
        previous,
        inputs,
        prompt,
        pages,
    };
    execute(
        &mut instance,
        &host,
        size,
        &[item(7, 0, 0, prompt, true, table.clone())],
    );
    instance.commit().unwrap();
    pool.commit_prepared(append.id()).unwrap();
    write_kv(&host, &table, PROMPT, size, length);
    let source_state = fill_slot(&host, handle, 0, 7);
    let key = chain.key_at(PROMPT as usize).unwrap();
    let full = PROMPT as usize / stripes.stripe_tokens();
    let snapshot = pool
        .insert_anchor(key, chain.chain(full).unwrap(), source, state_pages)
        .unwrap()
        .unwrap();
    instance
        .capture_prefix_state(7, PROMPT, &ids(&snapshot))
        .unwrap();

    // Request 9 reuses the prefix.
    let hit = pool
        .resume_prefix(&chain, PROMPT as usize, 1024)
        .unwrap()
        .unwrap();
    assert_eq!(hit.tokens(), PROMPT as usize);
    instance
        .restore_prefix_state_in_slot(9, prompt, &ids(hit.state()), 1)
        .unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    let append = pool.prepare_append_batch(&[(hit.handle(), 1)]).unwrap();
    let copy = append.appends()[0]
        .copy()
        .expect("the shared partial stripe copies on write");
    for rank in 0..RANKS {
        let bytes = host.read(local(rank, copy.source().index() as u64, length));
        host.write(
            local(rank, copy.destination().index() as u64, length),
            &bytes,
        );
    }
    let reused = ids(append.appends()[0].block_table());
    assert_eq!(
        reused[..full],
        table[..full],
        "size {size}: full stripes shared"
    );
    assert_ne!(reused[full], table[full]);
    execute(
        &mut instance,
        &host,
        size,
        &[item(9, 1, PROMPT, next, false, reused.clone())],
    );
    instance.commit().unwrap();
    pool.commit_prepared(append.id()).unwrap();
    kv.push(held(&host, &reused, PROMPT, size, length));
    state.push(slot_bytes(&host, handle, 1));
    instance.retire_request(9).unwrap();
    pool.release(hit.handle()).unwrap();

    // Request 7 is preempted through the host tier.
    let (anchor, plan) = tier.begin_offload(&mut pool, key, Some(7)).unwrap();
    assert_eq!(plan.pages().len(), table.len() + state_pages);
    transfer(&host, (&mut pool, &mut tier), plan, &mut store, length);
    instance.retire_request(7).unwrap();
    pool.release(source).unwrap();
    assert!(pool.remove_anchor(key).unwrap());
    let stale = vec![0xa5; length as usize];
    for rank in 0..RANKS {
        for at in 0..POOL_PAGES as u64 {
            host.write(local(rank, at, length), &stale);
        }
    }
    let other = fill_slot(&host, handle, 0, 99);
    let destination = pool
        .create_initialized_sequence(PROMPT as usize, 1024, state_pages)
        .unwrap();
    let plan = tier.begin_reload(&mut pool, anchor, &destination).unwrap();
    transfer(&host, (&mut pool, &mut tier), plan, &mut store, length);
    instance
        .restore_prefix_state_in_slot(7, prompt, &ids(destination.state()), 2)
        .unwrap();
    pool.finish_initialized_restore(destination.handle())
        .unwrap();
    let append = pool
        .prepare_append_batch(&[(destination.handle(), 1)])
        .unwrap();
    assert_eq!(append.appends()[0].copy(), None);
    let restored = ids(append.appends()[0].block_table());
    execute(
        &mut instance,
        &host,
        size,
        &[item(7, 2, PROMPT, next, false, restored.clone())],
    );
    instance.commit().unwrap();
    pool.commit_prepared(append.id()).unwrap();
    kv.push(held(&host, &restored, PROMPT, size, length));
    state.push(slot_bytes(&host, handle, 2));
    assert!(slot_bytes(&host, handle, 0) == other, "size {size}");

    let expected = owned(&table, PROMPT, size)
        .into_iter()
        .map(|(j, rank, _)| ((j, rank), page(j, rank, length)))
        .collect::<BTreeMap<_, _>>();
    for (check, (kv, state)) in kv.iter().zip(&state).enumerate() {
        assert!(*kv == expected, "size {size} check {check}: KV bytes");
        assert!(*state == source_state, "size {size} check {check}: state");
    }
    let launches = issued(&host, start)
        .into_iter()
        .filter(|line| line.starts_with("launch "))
        .collect();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get(), "size {size}");
    Cached {
        kv,
        state,
        launches,
    }
}

/// At sizes 2 and 4 a striped prefix is reused, offloaded and reloaded
/// after preemption with every owner's KV bytes and every rank's request
/// state intact, both as the size-1 instance holds them; the steps on the
/// reused and reloaded tables run the size-1 launches outside the group's
/// own stages; and each size's launches are pinned. Re-pinned when commit's
/// state copy symbol joined create (the handles), when the phased
/// all-reduce's state and cubin joined create (the handles), and when the
/// shared gate/up overlap became the default (the shared gate/ups on the
/// side streams; `fake_host_tests::shared_overlap` proves the launches the
/// one-stream program's).
/// Re-pinned when the strip-chain and prefill warp-tile entries took the
/// kernels that fence their ring reads before release (their symbols).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse
/// and its two symbols joined create (the handles).
/// Re-pinned when the decoded-tile projections took the balanced decoded
/// tiles (their symbols, grids and activation boxes).
/// Re-pinned when the projections' staged-tile cubin joined create (the
/// handles) and the BF16 prefill projections it holds took its entries.
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies (identity change, accuracy unaffected): their symbols and
/// shape arguments, and in create the row cubin's two symbols fewer and
/// `row_copy_sm120`'s two per rank (the handles).
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn striped_prefix_reuse_offload_and_preemption_match_size_one() {
    let single = cached(1);
    let pinned = [
        (2, (11388, 0xd3b2cb1c17e4a430)),
        (4, (11652, 0x1b6268c1341d4de6)),
    ];
    let mut failures = Vec::new();
    for (size, expected) in pinned {
        let group = cached(size);
        for (check, (kv, state)) in group.kv.iter().zip(&group.state).enumerate() {
            for (at, bytes) in kv {
                assert!(single.kv[check][at] == *bytes, "size {size} page {at:?}");
            }
            assert!(*state == single.state[check], "size {size} check {check}");
        }
        assert_composes(
            size,
            &outside_the_group(&group.launches),
            &outside_the_group(&single.launches),
        );
        let file = format!("glm5next-served-cp{size}-cached.txt");
        if let Err(error) = check_journal(&group.launches, expected, &file) {
            failures.push(format!("size {size}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A graph-captured step: decode rows of `items` items, or verify items
/// of `rows` rows each.
#[derive(Clone, Copy, Debug)]
enum Bucket {
    Decode { items: u32 },
    Verify { items: u32, rows: u32 },
}

/// Every decode bucket, and every verify bucket of the host's speculation
/// (up to four rows an item): 2 to 32 rows.
const BUCKETS: [Bucket; 11] = [
    Bucket::Decode { items: 1 },
    Bucket::Decode { items: 2 },
    Bucket::Decode { items: 4 },
    Bucket::Decode { items: 8 },
    Bucket::Decode { items: 16 },
    Bucket::Decode { items: 32 },
    Bucket::Verify { items: 1, rows: 2 },
    Bucket::Verify { items: 1, rows: 4 },
    Bucket::Verify { items: 2, rows: 4 },
    Bucket::Verify { items: 4, rows: 4 },
    Bucket::Verify { items: 8, rows: 4 },
];

/// Each bucket's launches as it captures its graphs, at size `size`: every
/// slot holds a request of 300 tokens; each bucket's first step captures
/// on every rank and its second step, of the same shape one row further,
/// replays every graph with no capture and no eager launch.
fn captures(size: u32) -> Vec<Vec<String>> {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host, size);
    let handle = instance.handle();
    let mut previous = vec![300_u64; DECODE_BATCH as usize];
    for slot in 0..DECODE_BATCH {
        crate::instance::seed(handle, 100 + u64::from(slot), slot, 300);
    }
    let mut captured = Vec::new();
    for bucket in BUCKETS {
        let (items, rows) = match bucket {
            Bucket::Decode { items } => (items, 1),
            Bucket::Verify { items, rows } => (items, rows),
        };
        let mut counts = None;
        for _ in 0..2 {
            let inputs = (0..rows).map(|row| 21 + row).collect::<Vec<_>>();
            let batch = (0..items)
                .map(|slot| {
                    let at = previous[slot as usize];
                    let end = at + u64::from(rows);
                    Item {
                        request: 100 + u64::from(slot),
                        slot: u64::from(slot),
                        previous: at,
                        inputs: &inputs,
                        prompt: false,
                        pages: (0..end.div_ceil(PAGE_TOKENS * u64::from(size)))
                            .map(|stripe| u64::from(slot) * 8 + stripe)
                            .collect(),
                    }
                })
                .collect::<Vec<_>>();
            let start = host.device.journal.borrow().len();
            execute(&mut instance, &host, size, &batch);
            if rows == 1 {
                instance.commit().unwrap();
            } else {
                let kept = (0..items)
                    .map(|slot| kept(100 + u64::from(slot), rows))
                    .collect::<Vec<_>>();
                instance.commit_accepted(&kept).unwrap();
            }
            for slot in 0..items {
                previous[slot as usize] += u64::from(rows);
            }
            let journal = host.device.journal.borrow();
            // Commit's state copies launch outside the graphs.
            let count = |prefix: &str| {
                journal[start..]
                    .iter()
                    .filter(|line| {
                        line.starts_with(prefix)
                            && !line.contains(model_glm5next_contract::sm120::state_copy::SYMBOL)
                    })
                    .count()
            };
            let (graphs, launched, replayed) =
                (count("captured "), count("launch "), count("replay "));
            match counts {
                None => {
                    assert!(graphs >= 4, "size {size} {bucket:?}: every rank captures");
                    assert_eq!(replayed, graphs, "size {size} {bucket:?}");
                    counts = Some(graphs);
                    drop(journal);
                    captured.push(
                        issued(&host, start)
                            .into_iter()
                            .filter(|line| {
                                line.starts_with("launch ")
                                    && !line.contains(
                                        model_glm5next_contract::sm120::state_copy::SYMBOL,
                                    )
                            })
                            .collect(),
                    );
                }
                Some(graphs_then) => {
                    assert_eq!(
                        (graphs, launched, replayed),
                        (0, 0, graphs_then),
                        "size {size} {bucket:?}: the second step replays"
                    );
                }
            }
        }
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get(), "size {size}");
    captured
}

/// At sizes 2 and 4 every decode and verify bucket captures its graphs on
/// every rank at its first step and replays them at the next; each
/// bucket's captured launches outside the group's own stages are the
/// size-1 instance's, and each size's and bucket's journal is pinned.
/// Re-pinned when commit's state copy symbol joined create (the handles);
/// its launches, outside the graphs, are not counted. Re-pinned when the
/// phased all-reduce's state and cubin joined create (the handles), and when
/// the shared gate/up overlap became the default (as above).
/// Re-pinned when the strip-chain and prefill warp-tile entries took the
/// kernels that fence their ring reads before release (their symbols).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse
/// and its two symbols joined create (the handles).
/// Re-pinned when the projections' staged-tile cubin joined create (the
/// handles).
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies (identity change, accuracy unaffected): their symbols and
/// shape arguments, and in create the row cubin's two symbols fewer and
/// `row_copy_sm120`'s two per rank (the handles).
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn served_groups_capture_and_replay_every_bucket() {
    let single = captures(1);
    let pinned: [(u32, [(usize, u64); 11]); 2] = [
        (
            2,
            [
                (4708, 0xa0509b596528bd70),
                (4708, 0xab7307aff4b24df7),
                (4708, 0x88dacfa878c7bbc5),
                (4708, 0xb6290f7f9dda0d49),
                (4708, 0xdf41bca14746c474),
                (4708, 0x0725e6d4a1a20517),
                (4708, 0x058e51a8428f2f35),
                (4708, 0xfee7485470ff48ef),
                (4708, 0x8170f5280aa06765),
                (4708, 0x85fec9749447d256),
                (4708, 0x586fc88220a80d53),
            ],
        ),
        (
            4,
            [
                (4884, 0xe8d803671b36cd1a),
                (4884, 0x2b76817fbb507c3d),
                (4884, 0xb367e922372a3891),
                (4884, 0xc8fa6d403f8bab95),
                (4884, 0x5a7f4489ea500332),
                (4884, 0x1bdaf29262a71463),
                (4884, 0x9ee9758182e079c7),
                (4884, 0xa5c1080167f3abff),
                (4884, 0x51ad7ce1fc2bbd73),
                (4884, 0x20512e56c303a91a),
                (4884, 0xab085d18524d4197),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (size, expected) in pinned {
        let group = captures(size);
        for (index, bucket) in BUCKETS.iter().enumerate() {
            assert_composes(
                size,
                &outside_the_group(&group[index]),
                &outside_the_group(&single[index]),
            );
            let name = match *bucket {
                Bucket::Decode { items } => format!("decode{items}"),
                Bucket::Verify { items, rows } => format!("verify{}", items * rows),
            };
            let file = format!("glm5next-served-cp{size}-{name}.txt");
            if let Err(error) = check_journal(&group[index], expected[index], &file) {
                failures.push(format!("size {size} {name}: {error}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
