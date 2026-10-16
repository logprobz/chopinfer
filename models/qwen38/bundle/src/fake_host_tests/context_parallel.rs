//! Context-parallel size 2 at TP4 on the fake host
//! (`crate::context_parallel`, `lower/context_parallel.rs`): create's
//! refusal switch, pinned launch journals per phase (BF16 and FP8, with and
//! without the MTP drafter), the composition with the size-1 instance's
//! launches, the owner-routed placement and group bindings, a striped
//! prefix snapshot, the per-rank bytes and KV capacity, and the group
//! gathers' protocol with one rank of each group delayed. The size-1
//! journals stay pinned unchanged in `fake_host_tests.rs`.

use kernel_core::context_parallel_oracle::{owned_block_row, owned_kv_positions};
use model_qwen38_contract::sm120::context_parallel::{self as cp, rank_bytes};

use super::speculation::{SpeculativeSink, batch, drafter_host_at, kept, truth, truths};
use super::*;
use crate::context_parallel::serve_in_tests;

/// The create record of `host`'s package, with the context-parallel tag of
/// group size `size` after the served count above 1, as the host sends it.
fn record(host: &FakeHost, size: u32) -> Vec<u8> {
    let mut record = configuration_at(host.shape(), host.layout(), &[]);
    if size > 1 {
        record.extend_from_slice(runtime_bundle_sdk::context_parallel::TAG);
        record.extend_from_slice(&size.to_le_bytes());
    }
    record
}

pub(super) fn create_at(host: &Rc<FakeHost>, size: u32) -> BundleInstanceV1 {
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &record(host, size),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    host.model(instance.handle());
    instance
}

/// Local pages of an item of `end` tokens in slot `slot`: one per stripe of
/// `size` pages.
pub(super) fn pages(slot: u64, end: u64, size: u64) -> Vec<u64> {
    (0..end.div_ceil(PAGE_TOKENS * size))
        .map(|page| slot * 64 + page)
        .collect()
}

/// An item of `inputs` after `previous` tokens in slot `slot`, striped.
pub(super) fn item(
    request: u64,
    slot: u64,
    previous: u64,
    inputs: &[u32],
    size: u64,
) -> (BatchItemV1, BatchAppendV1, u64) {
    let count = inputs.len() as u64;
    let end = previous + count;
    (
        BatchItemV1::new(
            request,
            previous > 0 && count == 1,
            count as u32,
            true,
            inputs.to_vec(),
        ),
        BatchAppendV1::new(previous, count, end, pages(slot, end, size), None),
        slot,
    )
}

/// A verify item the bundle drafts, striped.
fn drafted(
    request: u64,
    slot: u64,
    previous: u64,
    drafts: u32,
    size: u64,
) -> (BatchItemV1, BatchAppendV1, u64) {
    let rows = u64::from(drafts) + 1;
    let end = previous + rows;
    (
        BatchItemV1::verify(request, rows as u32, true, vec![truth(previous)]),
        BatchAppendV1::new(previous, rows, end, pages(slot, end, size), None),
        slot,
    )
}

fn execute(host: &FakeHost, instance: &mut BundleInstanceV1, name: &str, request: BatchRequestV1) {
    host.note(format!("step {name}"));
    let mut sink = SpeculativeSink::default();
    instance
        .execute(&request.encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{name}: {error:?} {:?}", host.logs.borrow()));
}

/// A host of the BF16 or FP8 package, with or without the MTP drafter.
fn host(drafter: bool, fp8: bool) -> Rc<FakeHost> {
    let host = if drafter {
        drafter_host_at(3, Shape::TP4)
    } else {
        Rc::new(FakeHost::default())
    };
    host.fp8_kv.set(fp8);
    host
}

/// The journal of a prefill of two items (without the drafter, one past a
/// selection piece), a decode of both and, with the drafter, a verify of three drafts, at group
/// size `size` with one prefill lane (as size 2 runs), split before each
/// step: create, then one segment per phase, the last up to destroy.
fn phases(drafter: bool, fp8: bool, size: u32) -> Vec<(String, Vec<String>)> {
    let _one_lane = legacy_overlap_off();
    let host = host(drafter, fp8);
    crate::pipeline::take_recorded();
    let n = u64::from(size);
    // Without the drafter the long item spans two selection pieces.
    let long = if drafter { 40 } else { 2050 };
    let mut instance = create_at(&host, size);
    execute(
        &host,
        &mut instance,
        "prefill",
        batch(vec![
            item(20, 0, 0, &truths(0..5), n),
            item(21, 1, 0, &truths(0..long), n),
        ]),
    );
    instance.commit().unwrap();
    execute(
        &host,
        &mut instance,
        "decode",
        batch(vec![
            item(20, 0, 5, &[truth(5)], n),
            item(21, 1, long, &[truth(long)], n),
        ]),
    );
    instance.commit().unwrap();
    if drafter {
        execute(
            &host,
            &mut instance,
            "verify",
            batch(vec![drafted(20, 0, 6, 3, n)]),
        );
        instance.commit_accepted(&kept(&[(20, 2)])).unwrap();
    }
    instance.retire_request(20).unwrap();
    instance.retire_request(21).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    super::export_rows(
        &format!(
            "dcp{size}-{}-{}",
            if drafter { "mtp" } else { "plain" },
            if fp8 { "fp8" } else { "bf16" }
        ),
        &[&host],
    );
    let mut segments = vec![("create".to_owned(), Vec::new())];
    for line in host.journal.take() {
        if let Some(name) = line.strip_prefix("step ") {
            segments.push((name.to_owned(), Vec::new()));
        }
        segments.last_mut().unwrap().1.push(line);
    }
    segments
}

/// The packages the journals cover: (MTP drafter, FP8 K/V, name).
const PACKAGES: [(bool, bool, &str); 4] = [
    (false, false, "bf16"),
    (false, true, "fp8"),
    (true, false, "mtp-bf16"),
    (true, true, "mtp-fp8"),
];

/// Journal lines and FNV-1a of each package's phases at size 2, recorded
/// with the lowering (2026-10-07; the create segments again after the
/// score rows' drop). The drafter packages' create segments are equal:
/// their FP8 and BF16 layouts allocate the same buffers.
/// Each create segment starts with one release line per rank: create
/// checks each text rank image's identity before opening it.
/// The shared stream broadcast changes the expansion's resolve and launch
/// lines, as for `EXPECTED`.
const DCP2_JOURNALS: [(&str, &str, (usize, u64)); 14] = [
    ("bf16", "create", (2336, 0x80E25E9929DBDEC1)),
    ("bf16", "prefill", (7522, 0x48B4FBF0F2836D18)),
    ("bf16", "decode", (6139, 0x0CE5B02B02E1889A)),
    ("fp8", "create", (2336, 0x69131047F95BF0C5)),
    ("fp8", "prefill", (7522, 0x72CE202BEB6E8FDE)),
    ("fp8", "decode", (6139, 0xA443B344982D7392)),
    ("mtp-bf16", "create", (2380, 0xA6EAF2F09108B506)),
    ("mtp-bf16", "prefill", (7234, 0x19D66D0D930AF5BC)),
    ("mtp-bf16", "decode", (4237, 0x2D13CE624654669C)),
    ("mtp-bf16", "verify", (6768, 0x5EA16EB6F21EAC36)),
    ("mtp-fp8", "create", (2380, 0xA6EAF2F09108B506)),
    ("mtp-fp8", "prefill", (7234, 0x44F77C4B70AE5F46)),
    ("mtp-fp8", "decode", (4237, 0xC753B55654872090)),
    ("mtp-fp8", "verify", (6768, 0x31AFE8C98B7B62B2)),
];

#[test]
fn dcp2_journals_are_pinned_per_package_and_phase() {
    serve_in_tests(true);
    let mut differing = Vec::new();
    let mut checked = 0;
    for (drafter, fp8, package) in PACKAGES {
        for (phase, lines) in phases(drafter, fp8, 2) {
            let expected = DCP2_JOURNALS
                .iter()
                .find(|(p, s, _)| *p == package && *s == phase)
                .unwrap_or_else(|| panic!("no pin for {package} {phase}"))
                .2;
            checked += 1;
            if let Err(error) = check_journal(
                &lines,
                expected,
                &format!("qwen38-dcp2-journal-{package}-{phase}.txt"),
            ) {
                differing.push(format!("{package} {phase}: {error}"));
            }
        }
    }
    assert!(differing.is_empty(), "{differing:#?}");
    assert_eq!(checked, DCP2_JOURNALS.len());
}

/// The size-1 launches the size-2 lowering replaces: the gated norm, RoPE
/// and K/V write, and QSA's key write, block selection and attention (the
/// index query stays).
fn replaced_at_one(symbol: &str) -> bool {
    symbol.contains("_qk_norm_rope_kv_")
        || (symbol.starts_with("chop_qsa_") && !symbol.ends_with("_index_query_v1"))
}

/// The launches the size-2 lowering adds: the family's entries and the
/// group all-gather.
fn added_at_two(symbol: &str) -> bool {
    symbol == cp::ALLGATHER_SYMBOL || cp::symbols(Shape::TP4).iter().any(|s| s == symbol)
}

/// The launch lines of `lines` without those `drop` names, with buffer
/// handles renamed by first appearance.
fn normalized(lines: &[String], drop: impl Fn(&str) -> bool) -> Vec<String> {
    let mut names = BTreeMap::new();
    lines
        .iter()
        .filter(|line| line.starts_with("launch ") && !drop(launch_symbol(line)))
        .map(|line| {
            let mut parts = line.split("raw: ");
            let mut renamed = parts.next().unwrap().to_owned();
            for part in parts {
                let digits = part.find(|c: char| !c.is_ascii_digit()).unwrap();
                let next = names.len();
                let name = *names.entry(part[..digits].to_owned()).or_insert(next);
                renamed.push_str(&format!("raw: #{name}{}", &part[digits..]));
            }
            renamed
        })
        .collect()
}

/// Every launch outside the replaced stages is the size-1 instance's: the
/// same symbols, geometry and arguments in the same order, buffers named
/// by first appearance. The replaced stages leave no size-1 launch behind,
/// and the size-1 instance launches none of the family.
#[test]
fn every_other_launch_is_the_size_one_instances() {
    serve_in_tests(true);
    for (drafter, fp8, package) in PACKAGES {
        let one = phases(drafter, fp8, 1);
        let two = phases(drafter, fp8, 2);
        assert_eq!(
            one.iter().map(|(phase, _)| phase).collect::<Vec<_>>(),
            two.iter().map(|(phase, _)| phase).collect::<Vec<_>>()
        );
        for ((phase, one), (_, two)) in one.iter().zip(&two) {
            assert!(
                !one.iter()
                    .any(|line| line.starts_with("launch ") && added_at_two(launch_symbol(line))),
                "{package} {phase}"
            );
            assert!(
                !two.iter()
                    .any(|line| line.starts_with("launch ") && replaced_at_one(launch_symbol(line))),
                "{package} {phase}"
            );
            let (one, two) = (
                normalized(one, replaced_at_one),
                normalized(two, added_at_two),
            );
            assert_eq!(one.len(), two.len(), "{package} {phase}");
            if let Some(index) = one.iter().zip(&two).position(|(a, b)| a != b) {
                panic!(
                    "{package} {phase} launch {index}:\n{}\n{}",
                    one[index], two[index]
                );
            }
        }
    }
}

/// Create serves size 2 at TP4 without the test switch, since `SERVED` is
/// on; it still refuses, before it acquires anything, a size the TP4
/// package does not declare, a malformed size and any size above 1 of a
/// TP2 package, whose ranks hold distinct KV heads.
#[test]
fn create_serves_only_the_declared_context_parallel_size() {
    serve_in_tests(false);
    assert!(crate::context_parallel::serves(2));
    let host = Rc::new(FakeHost::default());
    create_at(&host, 2);
    assert!(
        !host
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("awaits its GPU check"))
    );
    for (shape, size, status) in [
        (Shape::TP4, 4, BundleStatusV1::Unsupported),
        // A size that does not divide the ranks is malformed.
        (Shape::TP4, 3, BundleStatusV1::InvalidArgument),
        (
            Shape::of(Ranks::TP2).unwrap(),
            2,
            BundleStatusV1::Unsupported,
        ),
    ] {
        let host = Rc::new(FakeHost::at(shape));
        host.ngram_host.set(ngram_in_host_memory(shape));
        assert_eq!(
            api()
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &record(&host, size)
                )
                .err(),
            Some(BundleAbiError::Status(status)),
            "{} size {size}",
            shape.topology().id
        );
        assert_eq!(host.acquired.get(), 0);
    }
}

/// The scalar arguments of a launch, in order.
fn scalars(arguments: &[KernelArgumentV1]) -> Vec<u64> {
    arguments
        .iter()
        .filter_map(|argument| match argument {
            KernelArgumentV1::ScalarU32(value) => Some(u64::from(*value)),
            KernelArgumentV1::ScalarU64(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn span(argument: &KernelArgumentV1) -> DeviceBufferSpanV1 {
    match argument {
        KernelArgumentV1::DeviceBuffer(span) => *span,
        other => panic!("not a buffer: {other:?}"),
    }
}

fn members(argument: &KernelArgumentV1) -> Vec<u32> {
    match argument {
        KernelArgumentV1::DevicePointerArray(spans) => spans
            .iter()
            .map(|span| span.unwrap().buffer.slot())
            .collect(),
        other => panic!("not a pointer array: {other:?}"),
    }
}

/// The owner-routed writes place every K/V row and block key on exactly
/// one rank of its group, at the local page the host's striped table lists
/// for its stripe: the entries' placement references over a prefill's
/// positions, with the page tables the bundle packed. On every rank the
/// size-2 prefill binds those entries with the rank's group rank, the K/V
/// write reads the localization it just wrote, and each group launch
/// addresses only the rank's group.
#[test]
fn owner_writes_place_each_row_on_one_rank_of_its_group() {
    let shape = Shape::TP4;
    let entries = cp::owned_entries(shape);
    let kv = entries[0].query();
    let key = entries[2].query();
    let tokens = 2055_u64;
    let positions = (0..tokens as i32).collect::<Vec<_>>();
    let table = pages(1, tokens, 2);
    let mut owners = vec![Vec::new(); tokens as usize];
    for rank in 0..2 {
        let local = owned_kv_positions(kv, rank, &positions).unwrap();
        for (p, &l) in local.iter().enumerate() {
            if l < 0 {
                continue;
            }
            let (l, p) = (l as u64, p as u64);
            // Physical page p / 64 is stripe p / 128 on rank (p / 64) % 2.
            assert_eq!(u64::from(rank), p / PAGE_TOKENS % 2);
            assert_eq!(l / PAGE_TOKENS, p / (2 * PAGE_TOKENS));
            assert_eq!(l % PAGE_TOKENS, p % PAGE_TOKENS);
            owners[p as usize].push((rank, table[(l / PAGE_TOKENS) as usize], l % PAGE_TOKENS));
        }
    }
    assert!(owners.iter().all(|owners| owners.len() == 1));
    // Each (rank, page, slot) holds one row.
    let mut places = owners.iter().flatten().collect::<Vec<_>>();
    places.sort();
    places.dedup();
    assert_eq!(places.len() as u64, tokens);
    let blocks_per_page = PAGE_TOKENS / u64::from(crate::contract::QSA_BLOCK);
    for block in 0..tokens / u64::from(crate::contract::QSA_BLOCK) {
        let rows = (0..2)
            .filter_map(|rank| {
                owned_block_row(key, rank, block)
                    .unwrap()
                    .map(|row| (rank, row))
            })
            .collect::<Vec<_>>();
        let physical = block / blocks_per_page;
        assert_eq!(
            rows,
            [(
                (physical % 2) as u32,
                (physical / 2, block % blocks_per_page)
            )]
        );
    }

    // The bindings of a size-2 prefill on every rank.
    serve_in_tests(true);
    let _one_lane = legacy_overlap_off();
    let host = Rc::new(FakeHost::default());
    let mut instance = create_at(&host, 2);
    let from = host.launches.borrow().len();
    execute(
        &host,
        &mut instance,
        "prefill",
        batch(vec![item(21, 1, 0, &long_prompt(), 2)]),
    );
    let launches = host.launches.borrow()[from..].to_vec();
    let arguments = host.launch_arguments.borrow()[from..].to_vec();
    let symbol = |name: &str| {
        cp::symbols(shape)
            .into_iter()
            .find(|symbol| symbol.contains(name))
            .unwrap()
    };
    let (localize, write, key_write, select, partials) = (
        symbol("localize_positions2"),
        symbol("kv_write_bf16"),
        symbol("index_key_write"),
        symbol("select_per_rank2_v1"),
        symbol("bf16_prefill"),
    );
    // Rank streams in rank order; each rank's group is its pair.
    let mut streams = launches
        .iter()
        .map(|(stream, _, _)| *stream)
        .collect::<Vec<_>>();
    streams.sort();
    streams.dedup();
    assert_eq!(streams.len(), 4);
    let mut groups = BTreeMap::new();
    for (rank, &stream) in streams.iter().enumerate() {
        let rank = rank as u64;
        let own = launches
            .iter()
            .zip(&arguments)
            .filter(|((s, _, _), _)| *s == stream)
            .map(|((_, name, _), arguments)| (name.as_str(), arguments.as_slice()))
            .collect::<Vec<_>>();
        let mut localized = None;
        let mut writes = 0;
        for (name, arguments) in &own {
            if *name == localize {
                assert_eq!(scalars(arguments), [rank % 2]);
                localized = Some(span(&arguments[2]));
            } else if *name == write {
                // `local_positions` is the localization of this layer's rows.
                assert_eq!(Some(span(&arguments[6])), localized.take());
                assert_eq!(span(&arguments[9]).buffer.slot(), KV_SLOT + rank as u32);
                writes += 1;
            } else if *name == key_write || *name == select || *name == partials {
                assert_eq!(
                    scalars(arguments)[scalars(arguments).len() - 2..],
                    [rank % 2, 2]
                );
            } else if *name == cp::ALLGATHER_SYMBOL {
                // input, outputs, flags, epoch, words, rank, size, target.
                let [_, rank_scalar, size, target] = scalars(arguments)[..] else {
                    panic!("an all-gather's scalars")
                };
                assert_eq!((rank_scalar, size), (rank % 2, 2));
                assert!(target <= 2);
                groups
                    .entry(rank / 2)
                    .or_insert_with(BTreeSet::new)
                    .insert(members(&arguments[1]));
            }
        }
        assert_eq!(writes, 12, "rank {rank}");
    }
    // Both groups' gathers address only their own members' outputs.
    let slots = |group: u64| {
        groups[&group]
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
    };
    assert!(slots(0).is_disjoint(&slots(1)));
    assert!(groups.values().flatten().all(|outputs| outputs.len() == 2));
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

/// A striped prefix snapshot: each rank captures its own copy of the
/// per-slot state (the ring, linear and PLE entries are replicated, the K,
/// V and block-key pages are already split), restores it into another slot
/// byte for byte from its own origin, and the restored request continues
/// as the cold one does.
#[test]
fn striped_prefix_snapshots_move_each_ranks_bytes_from_its_origin() {
    serve_in_tests(true);
    let _one_lane = legacy_overlap_off();
    for fp8 in [false, true] {
        let host = Rc::new(FakeHost::default());
        host.fp8_kv.set(fp8);
        let mut instance = create_at(&host, 2);
        let handle = instance.handle();
        host.backing.set(true);
        execute(
            &host,
            &mut instance,
            "prefill",
            batch(vec![item(10, 3, 0, &PROMPT, 2)]),
        );
        instance.commit().unwrap();
        let cold = fill_slot(&host, handle, 3);
        // Every rank's state differs from its peers', so each byte has one
        // origin.
        for rank in 1..cold.len() {
            assert!(cold[..rank].iter().all(|other| *other != cold[rank]));
        }
        let first = snapshot_pages(&host, 100);
        let launches = host.launches.borrow().len();
        instance.capture_prefix_state(10, 3, &first).unwrap();
        assert_eq!(
            launched(&host, launches),
            vec![
                crate::launch::per_rank(
                    model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED,
                    host.shape()
                );
                4
            ]
        );
        for (rank, cold) in cold.iter().enumerate() {
            assert!(snapshot(&host, rank as u32, &first) == *cold, "rank {rank}");
        }
        instance
            .restore_prefix_state_in_slot(11, &PROMPT, &first, 7)
            .unwrap();
        for (rank, cold) in cold.iter().enumerate() {
            assert!(slot_bytes(&host, handle, rank, 7) == *cold, "rank {rank}");
        }
        let before = host.launches.borrow().len();
        execute(
            &host,
            &mut instance,
            "cold",
            batch(vec![item(10, 3, 3, &[9, 8], 2)]),
        );
        let cold_names = launched(&host, before);
        instance.commit().unwrap();
        let before = host.launches.borrow().len();
        execute(
            &host,
            &mut instance,
            "hit",
            batch(vec![item(11, 7, 3, &[9, 8], 2)]),
        );
        assert_eq!(launched(&host, before), cold_names);
        instance.commit().unwrap();
        instance.retire_request(10).unwrap();
        instance.retire_request(11).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

/// A late breakage of the restored request's continuation.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Late {
    None,
    /// Its continuation step is cancelled instead of committed.
    Uncommitted,
    /// Its committed ring goes back to the bytes it held before the
    /// continuation.
    StaleRing,
}

/// The QSA ring pieces of state slot `slot` on every rank, as
/// `offload_ring_state` reads them.
fn ring_pieces(host: &FakeHost, instance: &BundleInstanceV1, slot: u32) -> Vec<DeviceBufferSpanV1> {
    (0..4)
        .flat_map(|rank| {
            let ring = crate::instance::state_views(instance.handle(), rank)[3];
            super::host_table(host, slot)
                .descriptors
                .into_iter()
                .filter(|piece| piece.buffer == 3)
                .map(move |piece| {
                    DeviceBufferSpanV1::new(ring.buffer, ring.offset + piece.offset, piece.length)
                })
        })
        .collect()
}

/// How a striped continuation test moves the serving pages.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Move {
    /// Each rank's local page to the same local page of the new group.
    Stripes,
    /// Each rank's local pages to the next local page: a wrong stripe
    /// address.
    Rotated,
    /// Every rank's but rank 1's (an owner of the prompt's second page).
    WithoutRank1,
}

/// Block keys on `rank`'s local `pages` of the serving pool: per page, each
/// key's byte offset in the page and its (position, token) history.
fn rank_block_keys(host: &FakeHost, rank: u32, pages: &[u64]) -> Vec<Vec<u32>> {
    let pool = host.resolve_serving_kv_pool(rank).unwrap();
    let page_bytes = host.layout().1;
    let sim = host.sim.borrow();
    pages
        .iter()
        .map(|&page| {
            let start = pool.offset + page * page_bytes;
            let slot = pool.buffer.slot();
            sim.histories
                .range((slot, start)..(slot, start + page_bytes))
                .flat_map(|(&(_, offset), history)| {
                    std::iter::once((offset - start) as u32).chain(history.iter().copied())
                })
                .collect()
        })
        .collect()
}

/// Every rank's block keys in its local pages of `group` for `tokens`
/// tokens, after checking each key's origin: every position of a key on
/// group rank `r`'s local page `j` lies in physical page `2 j + r`.
fn striped_block_keys(host: &FakeHost, group: u64, tokens: u64) -> Vec<Vec<Vec<u32>>> {
    let local = pages(group, tokens, 2);
    (0..4)
        .map(|rank| {
            let keys = rank_block_keys(host, rank, &local);
            for (j, page) in keys.iter().enumerate() {
                let block = 2 * u64::from(crate::contract::QSA_BLOCK);
                // Each key is its offset, then (position, token) per row.
                for key in page.chunks(1 + block as usize) {
                    for position in key[1..].iter().step_by(2) {
                        let physical = u64::from(*position) / PAGE_TOKENS;
                        assert_eq!(
                            (physical / 2, physical % 2),
                            (j as u64, u64::from(rank % 2)),
                            "rank {rank} local page {j} holds position {position}"
                        );
                    }
                }
            }
            keys
        })
        .collect()
}

/// A request's prefix captured at 150 tokens (physical pages 0 and 2 on
/// group rank 0, page 1 on rank 1) moves its serving stripes to another
/// page group, restores into another state slot, and continues to 210
/// tokens (page 3, rank 1's second local page) as the uninterrupted request
/// does: per rank, the committed state, the replicated QSA ring, each
/// owner's block keys at their stripe addresses, the blocks the
/// continuation completes (from raw keys of the restored ring), and after
/// both commits the block keys, every rank's committed state (each entry's
/// history all 210 tokens) and the ring are the uninterrupted run's. A
/// rotated stripe address or a rank's missing copy makes them differ, as
/// does a restored continuation left uncommitted (stale state) or a ring
/// reverted after its commit (stale ring tail). K and V rows are not
/// modeled by the fake device; the block keys stand for the pages.
#[test]
fn striped_prefix_continuation_matches_the_uninterrupted_run() {
    serve_in_tests(true);
    let _one_lane = legacy_overlap_off();
    let (boundary, end) = (150, 210);
    for (moved, late) in [
        (Move::Stripes, Late::None),
        (Move::Rotated, Late::None),
        (Move::WithoutRank1, Late::None),
        (Move::Stripes, Late::Uncommitted),
        (Move::Stripes, Late::StaleRing),
    ] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_at(&host, 2);
        host.backing.set(true);
        host.sim.borrow_mut().track_qsa = true;
        // Serving page groups 1 and 7; state slots 1 and 5.
        let at = |request, slot, group, previous, tokens: std::ops::Range<u64>| {
            let mut item = item(request, group, previous, &truths(tokens), 2);
            item.2 = slot;
            item
        };
        execute(
            &host,
            &mut instance,
            "prefill",
            batch(vec![at(10, 1, 1, 0, 0..boundary)]),
        );
        instance.commit().unwrap();
        let snapshot = snapshot_pages(&host, 700);
        instance
            .capture_prefix_state(10, boundary, &snapshot)
            .unwrap();
        let source = striped_block_keys(&host, 1, boundary);
        assert!(source.iter().all(|pages| !pages[0].is_empty()));
        // The host moves each rank's serving stripes beside the snapshot.
        let local = pages(1, boundary, 2).len() as u64;
        for rank in 0..4 {
            if moved == Move::WithoutRank1 && rank == 1 {
                continue;
            }
            let pool = host.resolve_serving_kv_pool(rank).unwrap();
            let page = |group: u64, page: u64| {
                DeviceBufferSpanV1::new(
                    pool.buffer,
                    pool.offset + (group * 64 + page) * host.layout().1,
                    host.layout().1,
                )
            };
            for j in 0..local {
                let to = if moved == Move::Rotated {
                    (j + 1) % local
                } else {
                    j
                };
                host.sim
                    .borrow_mut()
                    .copy(page(7, to), page(1, j), &host.extents.borrow());
            }
        }
        instance
            .restore_prefix_state_in_slot(11, &truths(0..boundary), &snapshot, 5)
            .unwrap();
        for rank in 0..4 {
            assert_eq!(
                super::speculation::committed_state_on_rank(&host, &instance, 1, rank),
                super::speculation::committed_state_on_rank(&host, &instance, 5, rank),
                "rank {rank}"
            );
        }
        assert_eq!(
            super::speculation::offload_ring_state(&host, &instance, 1),
            super::speculation::offload_ring_state(&host, &instance, 5)
        );
        let restored = rank_block_keys_of(&host, 7, boundary);
        let mut same = restored == source;
        // Both continue to `end`, completing blocks over the restored ring.
        let mut completed = Vec::new();
        // The restored ring before its continuation, kept in slot 9's.
        let (restored_ring, kept_ring) = (
            ring_pieces(&host, &instance, 5),
            ring_pieces(&host, &instance, 9),
        );
        for (from, to) in restored_ring.iter().zip(&kept_ring) {
            host.sim
                .borrow_mut()
                .copy(*to, *from, &host.extents.borrow());
        }
        for (request, slot, group) in [(10, 1, 1), (11, 5, 7)] {
            host.sim.borrow_mut().qsa_blocks.clear();
            execute(
                &host,
                &mut instance,
                "continue",
                batch(vec![at(request, slot, group, boundary, boundary..end)]),
            );
            if request == 11 && late == Late::Uncommitted {
                instance.cancel().unwrap();
            } else {
                instance.commit().unwrap();
            }
            completed.push(host.sim.borrow().qsa_blocks.clone());
        }
        if late == Late::StaleRing {
            for (from, to) in kept_ring.iter().zip(&restored_ring) {
                host.sim
                    .borrow_mut()
                    .copy(*to, *from, &host.extents.borrow());
            }
        }
        // After both commits: every rank's committed state holds the whole
        // history, and the restored slot's state and ring are the
        // uninterrupted slot's.
        let state = |slot| {
            (0..4)
                .map(|rank| {
                    super::speculation::committed_state_on_rank(&host, &instance, slot, rank)
                })
                .collect::<Vec<_>>()
        };
        let uninterrupted_state = state(1);
        assert!(
            uninterrupted_state
                .iter()
                .flatten()
                .all(|history| *history == truths(0..end))
        );
        let state_same = state(5) == uninterrupted_state;
        let ring_same = super::speculation::offload_ring_state(&host, &instance, 1)
            == super::speculation::offload_ring_state(&host, &instance, 5);
        match late {
            Late::None => assert!(state_same && ring_same, "{moved:?}"),
            Late::Uncommitted => assert!(!state_same),
            Late::StaleRing => assert!(state_same && !ring_same),
        }
        same &= state_same && ring_same;
        assert!(!completed[0].is_empty());
        same &= completed[0] == completed[1];
        let uninterrupted = striped_block_keys(&host, 1, end);
        same &= rank_block_keys_of(&host, 7, end) == uninterrupted;
        if (moved, late) == (Move::Stripes, Late::None) {
            // The moved stripes keep their owners' addresses.
            assert_eq!(striped_block_keys(&host, 7, end), uninterrupted);
        }
        assert_eq!(
            same,
            (moved, late) == (Move::Stripes, Late::None),
            "{moved:?} {late:?}"
        );
        instance.retire_request(10).unwrap();
        instance.retire_request(11).unwrap();
        instance.destroy().unwrap();
    }
}

/// Every rank's block keys in its local pages of `group` for `tokens`
/// tokens, without the origin check.
fn rank_block_keys_of(host: &FakeHost, group: u64, tokens: u64) -> Vec<Vec<Vec<u32>>> {
    let local = pages(group, tokens, 2);
    (0..4)
        .map(|rank| rank_block_keys(host, rank, &local))
        .collect()
}

/// KV pages per rank the contract's floor leaves at context-parallel size
/// 1 for `slots` sequences and pages of `page_bytes`: the device's bytes
/// less the largest rank's weights, the vision encoder, the working set
/// (`package::working_bytes`) and the host's 1 GiB reserve, of which the
/// host's pool takes 0.9; and the pages at size 2, whose instance allocates
/// `group` bytes more per rank (negative: fewer).
fn planned_pages(slots: u32, page_bytes: u64, group: i64) -> (u64, u64) {
    use model_qwen38_contract::package;
    let shape = Shape::TP4;
    let room = package::DEVICE_BYTES
        - package::device_weight_bytes(shape, false)
        - package::vision_bytes_for(shape)
        - package::working_bytes(shape, slots)
        - (1 << 30);
    let pages = |room: u64| room / 10 * 9 / page_bytes;
    (pages(room), pages(room.checked_add_signed(-group).unwrap()))
}

/// Size 2 allocates the contract's group bytes per rank more than size 1
/// and drops QSA's score rows (`buffers::score_bytes`), which the group's
/// selection does not read, so at equal free memory each rank holds more
/// pages than at size 1, each of `2 * 64` group tokens. The planned pages
/// per rank at size 1 and 2: BF16 autoregressive at 64 sequences (tp4-ar)
/// and 32 (tp4-ar-dcp2), BF16 MTP at 32 (tp4-mtp, tp4-mtp-dcp2), and FP8
/// at the same counts; a GPU run's startup log measures them.
#[test]
fn size_two_drops_the_score_rows_and_holds_over_twice_the_tokens() {
    use model_qwen38_contract::buffers::score_bytes;
    serve_in_tests(true);
    let _one_lane = legacy_overlap_off();
    let allocated = |size| {
        let host = Rc::new(FakeHost::default());
        let mut instance = create_at(&host, size);
        let bytes = host.allocation_bytes.borrow().iter().sum::<u64>() as i64;
        instance.destroy().unwrap();
        bytes
    };
    let shape = Shape::TP4;
    let (scores, unread) = (score_bytes(shape, 1), score_bytes(shape, 2));
    assert_eq!((scores, unread), (512 << 20, 256));
    let group = rank_bytes(shape, 2).unwrap().total() as i64 - (scores - unread) as i64;
    assert_eq!(group, -423_139_644);
    assert_eq!(allocated(2) - allocated(1), 4 * group);
    let formats = shape.kv_formats();
    let planned = [
        (64, &formats[0]),
        (32, &formats[0]),
        (32, &formats[1]),
        (64, &formats[2]),
        (32, &formats[3]),
    ]
    .map(|(slots, format)| planned_pages(slots, format.page_bytes, group));
    assert_eq!(
        planned,
        [
            (34_191, 34_647),
            (36_495, 36_951),
            (33_688, 34_109),
            (64_584, 65_445),
            (59_561, 60_305)
        ]
    );
    for (one, two) in planned {
        let ratio = (2 * two) as f64 / one as f64;
        assert!((2.02..2.04).contains(&ratio), "{ratio}");
    }
}

/// A word of a fake buffer: its handle slot and byte offset.
type Word = (u32, u64);

/// One step of a simulated group launch.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Spin until the word holds the value.
    Wait(Word, u32),
    Release(Word, u32),
    /// Store gather `generation`'s tile from topology rank `writer` over
    /// the words from the first up to the byte offset.
    Store(Word, u64, u32, u32),
    /// Add one to a barrier's count, then wait for `members` arrivals.
    Arrive(Word),
    Barrier(Word, u32),
}

/// What a group all-gather moves, named by the launch that reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Moves {
    /// The selection's (score, block) pairs, which the top-k merge reads.
    Pairs,
    /// The queries the selected partials read.
    Queries,
    /// One owner's partial blocks, which its merge reads.
    Blocks,
    /// The barrier's word, which nothing reads.
    Barrier,
}

/// One rank's launch as the simulation runs it.
#[derive(Clone, Debug)]
enum Sim {
    /// The rank's `generation`th group all-gather: its outputs and flags in
    /// group-rank order, its epoch, words per rank, group rank, target (the
    /// group size for every member) and topology rank.
    Gather {
        moves: Moves,
        outputs: Vec<DeviceBufferSpanV1>,
        flags: Vec<DeviceBufferSpanV1>,
        epoch: DeviceBufferSpanV1,
        elements: u64,
        rank: u64,
        target: u64,
        writer: u32,
        generation: u32,
    },
    /// A launch reading the output of the rank's last gather that delivered
    /// there: every source block's tiles should hold that gather's
    /// generation, written by the group member of that slot (topology rank
    /// `group + source`).
    Consume {
        output: DeviceBufferSpanV1,
        elements: u64,
        group: u32,
        generation: u32,
        symbol: String,
    },
    /// The end of a step: a barrier of every rank.
    Reduce { key: Word, members: u32 },
}

/// Words per all-gather tile and CTA (`allgather` in `cuda/comm/comm.cuh`).
const TILE_WORDS: u64 = 4096;

fn tile_word(span: DeviceBufferSpanV1, word: u64) -> Word {
    (span.buffer.slot(), span.offset + word * 4)
}

/// The words of source `source`'s tile `tile` in a gather's output.
fn tile_words(source: u64, elements: u64, tile: u64) -> (u64, u64) {
    let first = source * elements + tile * TILE_WORDS;
    (first, (first + TILE_WORDS).min((source + 1) * elements))
}

/// A gather's tile as stored: its (generation, writer).
type Tile = (u32, u32);

/// What the gathers stored: per buffer, disjoint byte ranges from their
/// first byte, each with its end and the tile last stored there.
#[derive(Default)]
struct Stored(BTreeMap<u32, BTreeMap<u64, (u64, Tile)>>);

impl Stored {
    /// The ranges of `slot` overlapping bytes `first..end`.
    fn overlapping(&self, (slot, first): Word, end: u64) -> Vec<(u64, u64, Tile)> {
        self.0.get(&slot).map_or_else(Vec::new, |ranges| {
            ranges
                .range(..end)
                .rev()
                .take_while(|(_, (until, _))| *until > first)
                .map(|(&from, &(until, value))| (from, until, value))
                .collect()
        })
    }

    /// Stores `value` over bytes `first..end`, cutting what it overlaps.
    fn store(&mut self, (slot, first): Word, end: u64, value: Tile) {
        let cut = self.overlapping((slot, first), end);
        let ranges = self.0.entry(slot).or_default();
        for (from, until, old) in cut {
            ranges.remove(&from);
            if from < first {
                ranges.insert(from, (first, old));
            }
            if until > end {
                ranges.insert(end, (until, old));
            }
        }
        ranges.insert(first, (end, value));
    }

    /// What one store holds over every byte of `first..end`, if one does.
    fn read(&self, word: Word, end: u64) -> Option<Tile> {
        match self.overlapping(word, end)[..] {
            [(from, until, value)] if from <= word.1 && until >= end => Some(value),
            _ => None,
        }
    }
}

impl Sim {
    /// Each CTA's steps as `allgather` takes them from the words' current
    /// values: acknowledge the previous call to every source, wait for each
    /// delivered destination's acknowledgment (every member, or the target
    /// only), store the tile and raise its flag there, wait for every
    /// source's tile when it receives, then advance the tile's epoch.
    fn ctas(&self, words: &BTreeMap<Word, u32>) -> Vec<Vec<Op>> {
        match self {
            Sim::Gather {
                outputs,
                flags,
                epoch,
                elements,
                rank,
                target,
                writer,
                generation,
                ..
            } => {
                let (rank, ranks, target) = (*rank, outputs.len() as u64, *target);
                let delivers = |dest: u64| target == ranks || dest == target;
                let tiles = elements.div_ceil(TILE_WORDS);
                (0..tiles)
                    .map(|tile| {
                        let cell = tile_word(*epoch, tile);
                        let next = words.get(&cell).copied().unwrap_or(0) + 1;
                        let flag = rank * tiles + tile;
                        let own = flags[rank as usize];
                        let mut ops = Vec::new();
                        for source in 0..ranks {
                            let ack = ranks * tiles + source * tiles + tile;
                            ops.push(Op::Release(tile_word(own, ack), next - 1));
                        }
                        for (dest, &flags) in flags.iter().enumerate() {
                            if delivers(dest as u64) {
                                ops.push(Op::Wait(
                                    tile_word(flags, ranks * tiles + flag),
                                    next - 1,
                                ));
                            }
                        }
                        for (dest, (output, &flags)) in outputs.iter().zip(flags).enumerate() {
                            if delivers(dest as u64) {
                                let (first, end) = tile_words(rank, *elements, tile);
                                ops.push(Op::Store(
                                    tile_word(*output, first),
                                    tile_word(*output, end).1,
                                    *generation,
                                    *writer,
                                ));
                                ops.push(Op::Release(tile_word(flags, flag), next));
                            }
                        }
                        if delivers(rank) {
                            for source in 0..ranks {
                                ops.push(Op::Wait(tile_word(own, source * tiles + tile), next));
                            }
                        }
                        ops.push(Op::Release(cell, next));
                        ops
                    })
                    .collect()
            }
            Sim::Reduce { key, members } => {
                vec![vec![Op::Arrive(*key), Op::Barrier(*key, *members)]]
            }
            Sim::Consume { .. } => Vec::new(),
        }
    }

    /// What a consumer reads: the (generation, writer) of every source
    /// block's tiles, or the first tile that one store of its gather does
    /// not cover alone.
    fn consume(&self, stored: &Stored) -> Result<Read, String> {
        let Sim::Consume {
            output,
            elements,
            group,
            generation,
            symbol,
        } = self
        else {
            return Ok(Vec::new());
        };
        let mut read = Vec::new();
        for source in 0..u64::from(cp::RANKS) {
            for tile in 0..elements.div_ceil(TILE_WORDS) {
                let (first, end) = tile_words(source, *elements, tile);
                let found = stored.read(tile_word(*output, first), tile_word(*output, end).1);
                if found != Some((*generation, group + source as u32)) {
                    return Err(format!(
                        "{symbol} reads gather {generation} of rank {} but source {source} \
                         tile {tile} holds {found:?} (generation, writer)",
                        group + source as u32
                    ));
                }
                read.push(found);
            }
        }
        Ok(read)
    }
}

fn pointers(argument: &KernelArgumentV1) -> Vec<DeviceBufferSpanV1> {
    match argument {
        KernelArgumentV1::DevicePointerArray(spans) => spans.iter().flatten().copied().collect(),
        other => panic!("not a pointer array: {other:?}"),
    }
}

/// One step of a group simulation's requests.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A prefill of this many tokens in the next slot.
    Prefill(u64),
    /// A decode of one token in each of the first slots, as many as given.
    Decode(usize),
}

/// One launch as a step ran it: (stream, symbol, arguments).
type Ran = (u32, String, Vec<KernelArgumentV1>);

/// Every launch each of `steps` ran at size 2, in issue order: eager
/// launches as issued, and each graph launch as the launches captured into
/// it (which did not run at capture).
fn ran_steps(steps: &[Step]) -> Vec<Vec<Ran>> {
    let _one_lane = legacy_overlap_off();
    let host = Rc::new(FakeHost::default());
    let mut instance = create_at(&host, 2);
    let mut lengths = Vec::new();
    let (mut launched, mut replayed) = (
        host.launches.borrow().len(),
        host.graph_replays.borrow().len(),
    );
    let mut ran = Vec::new();
    for step in steps {
        let items = match *step {
            Step::Prefill(tokens) => {
                let slot = lengths.len() as u64;
                lengths.push(tokens);
                let tokens = (0..tokens)
                    .map(|token| token as u32 % 1000 + 5)
                    .collect::<Vec<_>>();
                vec![item(30 + slot, slot, 0, &tokens, 2)]
            }
            Step::Decode(items) => (0..items)
                .map(|slot| {
                    lengths[slot] += 1;
                    item(30 + slot as u64, slot as u64, lengths[slot] - 1, &[5], 2)
                })
                .collect(),
        };
        execute(&host, &mut instance, "step", batch(items));
        instance.commit().unwrap();
        let launches = host.launches.borrow();
        let arguments = host.launch_arguments.borrow();
        let graphs = host.graph_launches.borrow();
        let captured = graphs.values().flatten().collect::<BTreeSet<_>>();
        let replays = &host.graph_replays.borrow()[replayed..];
        // Decodes replay their graphs.
        assert_eq!(
            matches!(step, Step::Decode(_)),
            !replays.is_empty(),
            "{step:?}"
        );
        let run = |index: usize| {
            let (stream, symbol, _) = &launches[index];
            (*stream, symbol.clone(), arguments[index].clone())
        };
        let mut step = Vec::new();
        for index in launched..=launches.len() {
            for (_, graph) in replays.iter().filter(|(at, _)| *at == index) {
                step.extend(graphs[graph].iter().map(|&index| run(index)));
            }
            if index < launches.len() && !captured.contains(&index) {
                step.push(run(index));
            }
        }
        ran.push(step);
        launched = launches.len();
        replayed += replays.len();
    }
    instance.destroy().unwrap();
    ran
}

/// Every rank's group launches over size-2 `steps`, each prefill in its own
/// slot: each group all-gather, each launch that reads what a gather
/// delivered to the rank, and a barrier of every rank after each step. The
/// gathers of `dropped` kinds are left out, with their reads. The sublayer
/// all-reduces between layers are left out too, so a rank may run ahead of
/// its group as far as the group's own waits let it.
fn group_launches(steps: &[Step], dropped: &[Moves]) -> Vec<Vec<Sim>> {
    let ran = ran_steps(steps);
    let launches = ran.iter().flatten().collect::<Vec<_>>();
    // Each step's end, in launches.
    let ends = ran
        .iter()
        .scan(0, |end, step| {
            *end += step.len();
            Some(*end)
        })
        .collect::<Vec<_>>();
    // Rank streams in rank order.
    let mut streams = launches
        .iter()
        .map(|(stream, _, _)| *stream)
        .collect::<Vec<_>>();
    streams.sort();
    streams.dedup();
    assert_eq!(streams.len(), 4);
    // What each group gather's outputs hold, by the launch reading them.
    let readers = [
        ("topk_merge_blocks2", Moves::Pairs),
        ("selected_partials", Moves::Queries),
        ("partial_merge2", Moves::Blocks),
    ];
    let mut moves = BTreeMap::new();
    let mut sims = vec![Vec::new(); streams.len()];
    for (rank, &stream) in streams.iter().enumerate() {
        let out = &mut sims[rank];
        let group = rank as u32 / cp::RANKS * cp::RANKS;
        // Every rank issues the same gathers, so its count names a call.
        let mut generation = 0;
        // Per output buffer, the last gather that delivered there.
        let mut delivered = BTreeMap::new();
        let mut step = 0;
        for (index, (s, symbol, arguments)) in launches.iter().enumerate() {
            if *s == stream {
                if symbol == cp::ALLGATHER_SYMBOL {
                    // input, outputs, flags, epoch, words, rank, size, target.
                    let [elements, group_rank, size, target] = scalars(arguments)[..] else {
                        panic!("an all-gather's scalars")
                    };
                    assert_eq!(size, 2, "only group gathers run at size 2");
                    let outputs = pointers(&arguments[1]);
                    let output = outputs[group_rank as usize];
                    generation += 1;
                    if target == size || target == group_rank {
                        delivered.insert(output.buffer.slot(), (output, elements, generation));
                    }
                    out.push(Sim::Gather {
                        moves: if elements == cp::BARRIER_WORDS {
                            Moves::Barrier
                        } else {
                            Moves::Pairs
                        },
                        outputs,
                        flags: pointers(&arguments[2]),
                        epoch: span(&arguments[3]),
                        elements,
                        rank: group_rank,
                        target,
                        writer: rank as u32,
                        generation,
                    });
                } else if let Some(&(output, elements, generation)) =
                    arguments.iter().find_map(|argument| match argument {
                        KernelArgumentV1::DeviceBuffer(span) => delivered.get(&span.buffer.slot()),
                        _ => None,
                    })
                {
                    let (_, kind) = readers
                        .iter()
                        .find(|(reader, _)| symbol.contains(reader))
                        .unwrap_or_else(|| panic!("{symbol} reads a gathered buffer"));
                    moves.insert(output.buffer.slot(), *kind);
                    out.push(Sim::Consume {
                        output,
                        elements,
                        group,
                        generation,
                        symbol: symbol.clone(),
                    });
                }
            }
            if index + 1 == ends[step] {
                out.push(Sim::Reduce {
                    key: (u32::MAX - 100, step as u64),
                    members: streams.len() as u32,
                });
                step += 1;
            }
        }
    }
    // Name each gather by its output's reader, then leave out the dropped
    // gathers and their reads.
    for out in &mut sims {
        for sim in out.iter_mut() {
            if let Sim::Gather {
                moves: kind,
                outputs,
                rank,
                ..
            } = sim
                && *kind != Moves::Barrier
            {
                *kind = moves[&outputs[*rank as usize].buffer.slot()];
            }
        }
        out.retain(|sim| match sim {
            Sim::Gather { moves, .. } => !dropped.contains(moves),
            Sim::Consume { output, .. } => !dropped.contains(&moves[&output.buffer.slot()]),
            Sim::Reduce { .. } => true,
        });
    }
    sims
}

/// When a delayed rank takes a launch: only once no other rank can advance.
#[derive(Clone, Copy, Debug)]
enum Delay {
    /// Its entries into the query gathers and the owner gathers.
    Entries,
    /// Its reads of gathered buffers.
    Reads,
    /// Every step it takes.
    All,
}

impl Delay {
    fn holds(self, sim: &Sim) -> bool {
        match (self, sim) {
            (Delay::All, _) => true,
            (Delay::Reads, Sim::Consume { .. }) => true,
            (Delay::Entries, Sim::Gather { moves, .. }) => {
                matches!(moves, Moves::Queries | Moves::Blocks)
            }
            _ => false,
        }
    }
}

/// Reads of one consumer: per source block and tile, (generation, writer).
type Read = Vec<Option<Tile>>;

/// The simulated memory and every rank's position in its launches.
#[derive(Default)]
struct Run {
    words: BTreeMap<Word, u32>,
    stored: Stored,
    at: Vec<usize>,
    running: Vec<Option<Vec<std::collections::VecDeque<Op>>>>,
    reads: Vec<Vec<Read>>,
}

impl Run {
    /// Takes `rank`'s next launch or advances its running one, as far as
    /// its CTAs' waits allow; whether anything moved.
    fn advance(&mut self, ranks: &[Vec<Sim>], rank: usize) -> Result<bool, String> {
        let Some(sim) = ranks[rank].get(self.at[rank]) else {
            return Ok(false);
        };
        if matches!(sim, Sim::Consume { .. }) {
            let read = sim
                .consume(&self.stored)
                .map_err(|error| format!("rank {rank}: {error}"))?;
            self.reads[rank].push(read);
            self.at[rank] += 1;
            return Ok(true);
        }
        let mut progress = false;
        let words = &mut self.words;
        let ctas = self.running[rank].get_or_insert_with(|| {
            progress = true;
            sim.ctas(words).into_iter().map(Into::into).collect()
        });
        for cta in ctas.iter_mut() {
            while let Some(&op) = cta.front() {
                match op {
                    Op::Wait(word, value) | Op::Barrier(word, value) => {
                        if words.get(&word).copied().unwrap_or(0) != value {
                            break;
                        }
                    }
                    Op::Release(word, value) => {
                        words.insert(word, value);
                    }
                    Op::Store(word, end, generation, writer) => {
                        self.stored.store(word, end, (generation, writer));
                    }
                    Op::Arrive(word) => *words.entry(word).or_default() += 1,
                }
                cta.pop_front();
                progress = true;
            }
        }
        if ctas.iter().all(|cta| cta.is_empty()) {
            self.running[rank] = None;
            self.at[rank] += 1;
            progress = true;
        }
        Ok(progress)
    }
}

/// Runs every rank's launches in stream order with the words retained,
/// each launch's CTAs advancing as far as their waits allow. The `delayed`
/// ranks take the launches `delay` holds (once started, a launch's CTAs
/// run as any other's) only when no other step can advance, one at a
/// time: the latest legal schedule for them. Returns every rank's reads in
/// order, or the first stale read, and panics on a deadlock.
fn simulate(ranks: &[Vec<Sim>], delayed: &[usize], delay: Delay) -> Result<Vec<Vec<Read>>, String> {
    let mut run = Run {
        at: vec![0; ranks.len()],
        running: vec![None; ranks.len()],
        reads: vec![Vec::new(); ranks.len()],
        ..Run::default()
    };
    loop {
        let mut progress = false;
        for rank in 0..ranks.len() {
            let held = delayed.contains(&rank)
                && run.running[rank].is_none()
                && ranks[rank]
                    .get(run.at[rank])
                    .is_some_and(|sim| delay.holds(sim));
            if !held {
                progress |= run.advance(ranks, rank)?;
            }
        }
        if progress {
            continue;
        }
        // Nothing else advances: the first held launch goes now.
        let mut moved = false;
        for &rank in delayed {
            if run.advance(ranks, rank)? {
                moved = true;
                break;
            }
        }
        if !moved {
            assert!(
                run.at.iter().zip(ranks).all(|(&at, sims)| at == sims.len()),
                "deadlock at launches {:?}",
                run.at
            );
            return Ok(run.reads);
        }
    }
}

/// `ranks` with every bucket's pair gather delivering into one output per
/// member (the first bucket's), as a lowering sharing the pair output
/// across buckets would: the top-k merges read that output too.
fn shared_pair_outputs(ranks: &[Vec<Sim>]) -> Vec<Vec<Sim>> {
    let mut shared = BTreeMap::new();
    for (rank, sims) in ranks.iter().enumerate() {
        let group = rank as u32 / cp::RANKS * cp::RANKS;
        for sim in sims {
            if let Sim::Gather {
                moves: Moves::Pairs,
                outputs,
                ..
            } = sim
            {
                for (member, output) in outputs.iter().enumerate() {
                    shared.entry(group + member as u32).or_insert(output.buffer);
                }
            }
        }
    }
    let to = |member: u32, span: DeviceBufferSpanV1| {
        DeviceBufferSpanV1::new(shared[&member], span.offset, span.length)
    };
    ranks
        .iter()
        .enumerate()
        .map(|(rank, sims)| {
            let group = rank as u32 / cp::RANKS * cp::RANKS;
            sims.iter()
                .cloned()
                .map(|mut sim| {
                    match &mut sim {
                        Sim::Gather {
                            moves: Moves::Pairs,
                            outputs,
                            ..
                        } => {
                            for (member, output) in outputs.iter_mut().enumerate() {
                                *output = to(group + member as u32, *output);
                            }
                        }
                        Sim::Consume { output, symbol, .. }
                            if symbol.contains("topk_merge_blocks2") =>
                        {
                            *output = to(rank as u32, *output);
                        }
                        _ => {}
                    }
                    sim
                })
                .collect()
        })
        .collect()
}

/// One rank, or one rank of each group, arrives late at the query gathers
/// and the owner gathers (or reads every gathered buffer as late as the
/// protocols allow, or takes every step last) over size-2 prefills of
/// 2,560 tokens (selection pieces of 2,048 and 512 rows, cold buckets),
/// 4,096 tokens (one bucket twice) and 2,560 tokens again (warm buckets),
/// then decodes replaying the graphs of 3 rows three times, of 1 row, and
/// of 3 rows again. Every consumer on every rank reads what the undelayed
/// run reads, tile for tile, each block from its own call and group
/// member: the pairs the top-k merge reads, the queries the selected
/// partials read and the blocks each owner's merge reads. Pair outputs
/// shared across buckets let a cold bucket's gather overwrite pairs a
/// delayed member still merges. Without the barriers the next piece's
/// query gather still orders the received blocks; without both, a delayed
/// owner's blocks are overwritten.
#[test]
fn delayed_ranks_read_what_the_undelayed_run_reads() {
    serve_in_tests(true);
    let steps = [
        Step::Prefill(2560),
        Step::Prefill(4096),
        Step::Prefill(2560),
        Step::Decode(3),
        Step::Decode(3),
        Step::Decode(3),
        Step::Decode(1),
        Step::Decode(3),
    ];
    let ranks = group_launches(&steps, &[]);
    // Per layer and piece (two per prefill, one per decode): the pair,
    // query, two owner and barrier gathers.
    let gathers = |kind: Moves| {
        ranks[1]
            .iter()
            .filter(|sim| matches!(sim, Sim::Gather { moves, .. } if *moves == kind))
            .count()
    };
    let pieces = 12 * (2 * 3 + 5);
    assert_eq!(
        [Moves::Pairs, Moves::Queries, Moves::Blocks, Moves::Barrier].map(gathers),
        [pieces, pieces, 2 * pieces, pieces]
    );
    let undelayed = simulate(&ranks, &[], Delay::All).unwrap();
    assert!(undelayed.iter().all(|reads| reads.len() == 4 * pieces));
    let delays: [&[usize]; 8] = [&[0], &[1], &[2], &[3], &[0, 2], &[0, 3], &[1, 2], &[1, 3]];
    for delay in [Delay::Entries, Delay::Reads, Delay::All] {
        for delayed in delays {
            let reads = simulate(&ranks, delayed, delay)
                .unwrap_or_else(|error| panic!("{delay:?} {delayed:?}: {error}"));
            assert!(reads == undelayed, "{delay:?} {delayed:?}");
        }
    }
    let errors = |ranks: &[Vec<Sim>]| {
        delays
            .iter()
            .filter_map(|delayed| simulate(ranks, delayed, Delay::Reads).err())
            .collect::<Vec<_>>()
    };
    let shared = errors(&shared_pair_outputs(&ranks));
    assert!(!shared.is_empty());
    assert!(
        shared
            .iter()
            .all(|error| error.contains("topk_merge_blocks2")),
        "{shared:#?}"
    );
    assert!(errors(&group_launches(&steps, &[Moves::Barrier])).is_empty());
    let unordered = errors(&group_launches(&steps, &[Moves::Barrier, Moves::Queries]));
    assert!(!unordered.is_empty());
    assert!(
        unordered
            .iter()
            .all(|error| error.contains("partial_merge2")),
        "{unordered:#?}"
    );
}
