//! Graph capture, release and rank replay against the host's own resource
//! bookkeeping.
//!
//! The host releases resources newest first across a whole bundle instance,
//! and a bundle's fake host would accept any order, so these tests run the
//! shared graph cache over `HostServicesV1Backend<FakeHostDeviceV1>`, at
//! every supported rank count.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use bundle_runtime::BundleGeometry;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::geometry::{Items, MAX_ITEMS, Ranks, SUPPORTED_RANKS};
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::pipeline::{Collectives, PreparedLaunch, Program, run_ranks};
use kernel_core::contract::{ArgumentList, LaunchPlan};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesTableV1,
};
use runtime_host::fake::{FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

/// Step limits at MiMo's numbers; the rank count is the cache's.
struct Steps;

impl BundleGeometry for Steps {
    const PAGE_TOKENS: u64 = 64;
    const PREFILL_TOKENS: u32 = 8192;
    const DECODE_BATCH: u32 = 32;
    const MAX_VERIFY_ROWS: u32 = 8;
    const TOKEN_GRANULE: u32 = 256;
}

type Backend = HostServicesV1Backend<FakeHostDeviceV1>;

/// Every supported rank count.
fn rank_counts() -> impl Iterator<Item = Ranks> {
    SUPPORTED_RANKS
        .into_iter()
        .map(|count| Ranks::new(count).unwrap())
}

/// A host on `ranks` devices with one stream per rank, and its live resource
/// count once those exist.
fn host(
    ranks: Ranks,
) -> (
    HostServicesTableV1,
    Rc<Backend>,
    Vec<BundleStreamHandleV1>,
    usize,
) {
    let ordinals = (0..ranks.get() as i32).collect();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals).unwrap());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let streams = (0..ranks.get())
        .map(|rank| {
            let context = services.select_context(rank, rank as i32).unwrap();
            services.create_stream(context, 0).unwrap()
        })
        .collect();
    let baseline = backend.live_resource_count();
    (table, backend, streams, baseline)
}

fn key(rank: u32, bucket: u32, items: u32) -> GraphKey {
    GraphKey {
        rank,
        bucket,
        items,
    }
}

/// Captures `launches` for `key`, recording nothing on the stream.
fn capture<G: BundleGeometry>(
    cache: &mut GraphCache<G>,
    table: &HostServicesTableV1,
    streams: &[BundleStreamHandleV1],
    key: GraphKey,
    launches: usize,
) {
    cache
        .capture::<BundleAbiError>(
            table.caller(),
            streams[key.rank as usize],
            key,
            launches,
            |_| Ok(()),
        )
        .unwrap();
}

#[test]
fn teardown_releases_graphs_newest_first_across_bucket_histories() {
    // Buckets captured rank by rank in increasing, then decreasing, order,
    // one piece or three per program, interleave with the key order.
    for (ranks, buckets) in rank_counts().flat_map(|ranks| [(ranks, [1, 2, 4]), (ranks, [4, 2, 1])])
    {
        let (table, backend, streams, baseline) = host(ranks);
        let mut cache = GraphCache::<Steps>::new(ranks);
        for bucket in buckets {
            for rank in 0..ranks.get() {
                let launches = if bucket == 2 { 192 } else { 1 };
                capture(&mut cache, &table, &streams, key(rank, bucket, 0), launches);
            }
        }
        let last = ranks.get() - 1;
        capture(&mut cache, &table, &streams, key(last, 4, 2), 1);
        assert_eq!(
            backend.live_resource_count() - baseline,
            5 * ranks.count() + 1,
            "{ranks:?}"
        );
        cache.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
        assert!(cache.is_empty());
    }
}

#[test]
fn caches_sharing_a_ledger_release_their_interleaved_graphs_together() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut target = GraphCache::<Steps>::new(ranks);
        let mut drafters = [
            GraphCache::<Steps>::sharing(&target),
            GraphCache::<Steps>::sharing(&target),
        ];
        assert!(drafters.iter().all(|cache| cache.ranks() == ranks));
        // Decode, then a drafter group, verify, the other group, and a larger
        // decode bucket, all rank by rank.
        for rank in 0..ranks.get() {
            capture(&mut target, &table, &streams, key(rank, 1, 0), 1);
        }
        for rank in 0..ranks.get() {
            capture(&mut drafters[1], &table, &streams, key(rank, 2, 0), 192);
        }
        for rank in 0..ranks.get() {
            capture(&mut target, &table, &streams, key(rank, 8, 2), 192);
        }
        for rank in 0..ranks.get() {
            capture(&mut drafters[0], &table, &streams, key(rank, 1, 0), 1);
        }
        for rank in 0..ranks.get() {
            capture(&mut target, &table, &streams, key(rank, 4, 0), 1);
        }
        assert_eq!(
            backend.live_resource_count() - baseline,
            9 * ranks.count(),
            "{ranks:?}"
        );
        // Releasing any cache releases the whole ledger, and every cache
        // sharing it forgets its lookups.
        drafters[0].release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
        assert!(target.is_empty() && drafters[1].is_empty());
        assert!(!target.contains(key(0, 1, 0)));
        assert!(target.cached(&[key(0, 1, 0)]).is_none());
        // Every cache captures again after the release.
        capture(&mut target, &table, &streams, key(0, 1, 0), 1);
        capture(&mut drafters[1], &table, &streams, key(0, 2, 0), 1);
        assert_eq!((target.len(), drafters[1].len()), (1, 1));
        target.release_all(table.caller()).unwrap();
        drafters[1].release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

/// Two pieces are captured, then lowering the third fails after its capture
/// began. The rollback releases the ended third, then the second and the
/// first. A refused release keeps that graph and the older ones for teardown.
#[test]
fn a_failed_capture_releases_its_pieces_newest_first() {
    for refused in [None, Some(0), Some(1), Some(2)] {
        let (table, backend, streams, baseline) = host(Ranks::TP4);
        let mut cache = GraphCache::<Steps>::new(Ranks::TP4);
        capture(&mut cache, &table, &streams, key(0, 1, 0), 1);
        let decode = backend.live_resource_count();
        if let Some(allowed) = refused {
            backend.with_device_mut(|device| {
                device.refuse_after(FakeRefusalV1::ReleaseGraph, allowed);
            });
        }
        let mut piece = 0;
        let result =
            cache.capture::<BundleAbiError>(table.caller(), streams[0], key(0, 2, 0), 192, |_| {
                piece += 1;
                if piece == 3 {
                    Err(BundleAbiError::InvalidRange("injected lowering refusal"))
                } else {
                    Ok(())
                }
            });
        assert_eq!(
            result,
            Err(BundleAbiError::InvalidRange("injected lowering refusal"))
        );
        assert!(!cache.contains(key(0, 2, 0)) && cache.contains(key(0, 1, 0)));
        let kept = refused.map_or(0, |allowed| 3 - allowed);
        assert_eq!(backend.live_resource_count() - decode, kept, "{refused:?}");
        cache.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

/// A cache holding a one-piece bucket-1 decode graph for every rank.
fn decode_graphs(
    ranks: Ranks,
    table: &HostServicesTableV1,
    streams: &[BundleStreamHandleV1],
) -> GraphCache<Steps> {
    let mut cache = GraphCache::<Steps>::new(ranks);
    for rank in 0..ranks.get() {
        capture(&mut cache, table, streams, key(rank, 1, 0), 1);
    }
    cache
}

/// Per rank, `names` resolved in a module loaded on the rank's context.
fn symbols(
    table: &HostServicesTableV1,
    ranks: Ranks,
    names: &[String],
) -> Vec<BTreeMap<String, BundleCudaSymbolHandleV1>> {
    let services = table.caller();
    (0..ranks.get())
        .map(|rank| {
            let context = services.select_context(rank, rank as i32).unwrap();
            let module = services.load_cuda_module(context, b"fake-module").unwrap();
            names
                .iter()
                .map(|name| {
                    let symbol = services.resolve_cuda_symbol(module, name.as_bytes());
                    (name.clone(), symbol.unwrap())
                })
                .collect()
        })
        .collect()
}

/// A program of `launches`, each a symbol and whether it is a collective.
fn program(launches: &[(String, bool)]) -> Program {
    let launch = |symbol: &String| PreparedLaunch {
        symbol: symbol.clone(),
        rows: 1,
        plan: LaunchPlan {
            grid: [1, 1, 1],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: ArgumentList(Vec::new()),
        bindings: Default::default(),
    };
    Program {
        launches: launches.iter().map(|(symbol, _)| launch(symbol)).collect(),
        collective: launches.iter().map(|&(_, collective)| collective).collect(),
    }
}

/// Every launch the host was asked for, in order.
fn launched(backend: &Backend) -> Vec<String> {
    backend.with_device(|device| {
        device
            .refused_launches()
            .iter()
            .map(|name| String::from_utf8_lossy(name).into_owned())
            .collect()
    })
}

/// Eager programs are queued collective by collective: every rank's
/// launches up to and including its first collective, rank after rank,
/// then every rank's next segment, so no rank's stream fills with work
/// behind a collective its peers have not queued yet. Each rank's own
/// launches keep their order.
#[test]
fn eager_programs_queue_collective_by_collective_across_ranks() {
    for ranks in rank_counts() {
        let (table, backend, streams, _) = host(ranks);
        backend.with_device_mut(|device| device.accept_launches(true));
        // Rank r's program: a local launch, the first collective, two local
        // launches, the second collective, then a local tail.
        let rank_program = |rank: u32| {
            let local = |index: u32| (format!("local_{rank}_{index}"), false);
            vec![
                local(0),
                ("reduce_a".to_string(), true),
                local(1),
                local(2),
                ("reduce_b".to_string(), true),
                local(3),
            ]
        };
        let names = (0..ranks.get())
            .flat_map(rank_program)
            .map(|(name, _)| name)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let symbols = symbols(&table, ranks, &names);
        let mut cache = GraphCache::<Steps>::new(ranks);
        let collectives = Collectives::default();
        let mut submitted = Vec::new();
        run_ranks(
            table.caller(),
            &streams,
            &symbols,
            None,
            &mut cache,
            &collectives,
            |rank| Ok(program(&rank_program(rank as u32))),
            |rank| submitted.push(rank),
        )
        .unwrap();
        let segments = [0..2, 2..5, 5..6];
        let expected = segments
            .iter()
            .flat_map(|segment| {
                (0..ranks.get()).flat_map(move |rank| {
                    rank_program(rank)[segment.clone()]
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(launched(&backend), expected, "{ranks:?}");
        assert_eq!(submitted, (0..ranks.count()).collect::<Vec<_>>());
        assert!(!collectives.is_poisoned());
        assert!(cache.is_empty());
    }
}

/// Ranks whose eager programs issue different collectives are refused
/// before anything is queued: each would wait at a collective its peers
/// never reach.
#[test]
fn eager_programs_with_differing_collectives_launch_nothing() {
    for ranks in rank_counts().filter(|ranks| ranks.count() > 1) {
        let (table, backend, streams, _) = host(ranks);
        backend.with_device_mut(|device| device.accept_launches(true));
        let names = ["local", "reduce_a", "reduce_b", "reduce_c"].map(String::from);
        let symbols = symbols(&table, ranks, &names);
        let mut cache = GraphCache::<Steps>::new(ranks);
        let collectives = Collectives::default();
        let last = ranks.count() - 1;
        let peers = [(names[1].clone(), true), (names[2].clone(), true)];
        // The last rank's second launch is local, or another collective.
        for second in [(names[0].clone(), false), (names[3].clone(), true)] {
            let result = run_ranks(
                table.caller(),
                &streams,
                &symbols,
                None,
                &mut cache,
                &collectives,
                |rank| {
                    let mut launches = peers.to_vec();
                    if rank == last {
                        launches[1] = second.clone();
                    }
                    Ok(program(&launches))
                },
                |_| panic!("nothing is queued"),
            );
            assert!(
                matches!(result, Err(DispatchError::InvalidPlan)),
                "{ranks:?}"
            );
            assert!(launched(&backend).is_empty());
            assert!(!collectives.is_poisoned());
        }
    }
}

/// Every rank's graphs replay piece-major on its own stream, whatever the
/// rank count.
#[test]
fn every_rank_count_replays_one_graph_per_rank() {
    for ranks in rank_counts() {
        let (table, _backend, streams, _) = host(ranks);
        let mut cache = decode_graphs(ranks, &table, &streams);
        let symbols = vec![BTreeMap::new(); ranks.count()];
        let collectives = Collectives::default();
        let mut submitted = Vec::new();
        run_ranks(
            table.caller(),
            &streams,
            &symbols,
            Some((1, 0)),
            &mut cache,
            &collectives,
            |_| panic!("cached ranks need no lowering"),
            |rank| submitted.push(rank),
        )
        .unwrap();
        assert_eq!(submitted, (0..ranks.count()).collect::<Vec<_>>());
        // Missing graphs are captured for every rank, then replayed.
        let mut lowered = Vec::new();
        run_ranks(
            table.caller(),
            &streams,
            &symbols,
            Some((2, 0)),
            &mut cache,
            &collectives,
            |rank| {
                lowered.push(rank);
                Ok(Program::default())
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(lowered, (0..ranks.count()).collect::<Vec<_>>());
        assert_eq!(cache.len(), 2 * ranks.count());
        assert!(!collectives.is_poisoned());
        cache.release_all(table.caller()).unwrap();
    }
}

#[test]
fn a_rank_list_other_than_the_instance_launches_nothing() {
    for ranks in rank_counts() {
        let n = ranks.count();
        let (table, _backend, streams, _) = host(ranks);
        let mut cache = decode_graphs(ranks, &table, &streams);
        let mut all = streams.clone();
        all.push(streams[0]);
        let collectives = Collectives::default();
        for (count, symbols) in [
            (0, 0),
            (n - 1, n - 1),
            (n + 1, n + 1),
            (n, n - 1),
            (n - 1, n),
        ] {
            for graph in [Some((1, 0)), Some((2, 0)), None] {
                let calls = Cell::new(0);
                let result = run_ranks(
                    table.caller(),
                    &all[..count],
                    &vec![BTreeMap::new(); symbols],
                    graph,
                    &mut cache,
                    &collectives,
                    |_| {
                        calls.set(calls.get() + 1);
                        Ok(Program::default())
                    },
                    |_| calls.set(calls.get() + 1),
                );
                assert!(
                    matches!(result, Err(DispatchError::InvalidPlan)),
                    "{n} ranks, {count} streams, {symbols} symbols, graph {graph:?}"
                );
                assert_eq!(
                    calls.get(),
                    0,
                    "{n} ranks, {count} streams, {symbols} symbols"
                );
            }
            if count != n {
                let mut submitted = 0;
                let replayed =
                    cache.replay_piece_major(table.caller(), &all[..count], (1, 0), |_| {
                        submitted += 1;
                    });
                assert!(replayed.is_err(), "{n} ranks, {count} streams");
                assert_eq!(submitted, 0, "{n} ranks, {count} streams");
            }
        }
        // A graph key past the instance's ranks is refused before capture.
        let before = cache.len();
        let refused = cache.capture::<BundleAbiError>(
            table.caller(),
            streams[0],
            key(ranks.get(), 1, 0),
            1,
            |_| panic!("an out-of-range rank records nothing"),
        );
        assert_eq!(refused, Err(BundleAbiError::InvalidRange("graph bucket")));
        assert_eq!(cache.len(), before);
        assert!(!collectives.is_poisoned());
    }
}

/// A panic once a rank's graph is queued unwinds past the unit's result,
/// so it must still leave the instance poisoned, and a poisoned instance
/// submits nothing more.
#[test]
fn a_panic_after_a_rank_was_submitted_poisons() {
    for ranks in rank_counts() {
        let last = ranks.count() - 1;
        let (table, _backend, streams, _) = host(ranks);
        let mut cache = decode_graphs(ranks, &table, &streams);
        let symbols = vec![BTreeMap::new(); ranks.count()];
        let collectives = Collectives::default();
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            run_ranks(
                table.caller(),
                &streams,
                &symbols,
                Some((1, 0)),
                &mut cache,
                &collectives,
                |_| panic!("cached ranks need no lowering"),
                |rank| assert_ne!(rank, last, "injected panic after the last rank was queued"),
            )
        }));
        assert!(panicked.is_err());
        assert!(collectives.is_poisoned());
        let mut submitted = 0;
        let result = run_ranks(
            table.caller(),
            &streams,
            &symbols,
            Some((1, 0)),
            &mut cache,
            &collectives,
            |_| Ok(Program::default()),
            |_| submitted += 1,
        );
        assert!(result.is_err());
        assert_eq!(submitted, 0);
    }
}

#[test]
fn catch_up_rows_do_not_relax_target_verify_bounds() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut verify = GraphCache::<Steps>::new(ranks);
        let mut catch_up = GraphCache::<Steps>::catch_up_sharing(&verify);
        for bucket in [16, 32] {
            assert_eq!(
                verify.capture::<BundleAbiError>(
                    table.caller(),
                    streams[0],
                    key(0, bucket, 1),
                    1,
                    |_| panic!("oversized verify item must not capture"),
                ),
                Err(BundleAbiError::InvalidRange("graph bucket"))
            );
            catch_up
                .capture::<BundleAbiError>(table.caller(), streams[0], key(0, bucket, 1), 1, |_| {
                    Ok(())
                })
                .unwrap();
        }
        for invalid in [
            key(0, 64, 1),
            key(0, 32, 64),
            key(0, 3, 1),
            key(ranks.get(), 32, 1),
        ] {
            assert_eq!(
                catch_up.capture::<BundleAbiError>(
                    table.caller(),
                    streams[0],
                    invalid,
                    1,
                    |_| panic!("invalid catch-up key must not capture"),
                ),
                Err(BundleAbiError::InvalidRange("graph bucket"))
            );
        }
        verify
            .capture::<BundleAbiError>(table.caller(), streams[0], key(0, 32, 4), 1, |_| Ok(()))
            .unwrap();
        // All three captures share the strict host's global release order.
        catch_up.release_all(table.caller()).unwrap();
        assert!(verify.is_empty());
        assert!(catch_up.is_empty());
        verify.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

/// `Steps` whose package runs the most items any bundle may.
struct Wide;

impl BundleGeometry for Wide {
    const PAGE_TOKENS: u64 = Steps::PAGE_TOKENS;
    const PREFILL_TOKENS: u32 = Steps::PREFILL_TOKENS;
    const DECODE_BATCH: u32 = MAX_ITEMS;
    const MAX_VERIFY_ROWS: u32 = Steps::MAX_VERIFY_ROWS;
    const TOKEN_GRANULE: u32 = Steps::TOKEN_GRANULE;
}

#[test]
fn every_item_count_captures_its_buckets_and_refuses_larger_ones() {
    let ranks = Ranks::TP2;
    let (table, backend, streams, baseline) = host(ranks);
    // A cache without a count serves the package maximum.
    assert_eq!(GraphCache::<Steps>::new(ranks).items().get(), 32);
    assert_eq!(GraphCache::<Wide>::new(ranks).items().get(), MAX_ITEMS);
    for count in 1..=MAX_ITEMS {
        let items = Items::within::<Wide>(count).unwrap();
        let mut cache = GraphCache::<Wide>::with_items(ranks, items);
        assert_eq!(GraphCache::<Wide>::sharing(&cache).items(), items);
        assert_eq!(
            GraphCache::<Wide>::catch_up_sharing(&cache).items().get(),
            MAX_ITEMS
        );
        let buckets = items.buckets().collect::<Vec<_>>();
        for &bucket in &buckets {
            for rank in 0..ranks.get() {
                capture(&mut cache, &table, &streams, key(rank, bucket, 0), 1);
            }
        }
        assert_eq!(cache.len(), buckets.len() * ranks.count(), "{count} items");
        // The next bucket up, and verify item buckets above the capacity,
        // are refused before any capture.
        let above = 2 * items.capacity();
        for refused in [key(0, above, 0), key(0, above * 8, above)] {
            assert!(matches!(
                cache.capture::<BundleAbiError>(table.caller(), streams[0], refused, 1, |_| Ok(())),
                Err(BundleAbiError::InvalidRange("graph bucket"))
            ));
        }
        // A verify token bucket above the verify rows is refused even when
        // its items fit and carry at most `MAX_VERIFY_ROWS` rows each.
        let capacity = items.capacity();
        if capacity * Wide::MAX_VERIFY_ROWS >= 512 {
            assert!(matches!(
                cache.capture::<BundleAbiError>(
                    table.caller(),
                    streams[0],
                    key(0, 512, capacity),
                    1,
                    |_| Ok(())
                ),
                Err(BundleAbiError::InvalidRange("graph bucket"))
            ));
        }
        // A verify step over the largest item bucket fits.
        capture(&mut cache, &table, &streams, key(0, capacity, capacity), 1);
        cache.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

#[test]
fn the_item_bound_is_the_manifest_package_maximum() {
    assert_eq!(
        MAX_ITEMS,
        runtime_bundle::BundleBatchLimitsV1::MAXIMUM_SEQUENCES
    );
}
