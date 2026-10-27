//! Exact-program graphs (`graphs::programs`) against the host's own
//! resource bookkeeping, which releases graphs only newest first across
//! the whole instance: repeat-aware admission, whole-group reservation,
//! newest-first release that leaves warmed decode graphs captured, and
//! failures that stop captures without touching another cache's graphs.

use std::rc::Rc;

use bundle_runtime::BundleGeometry;
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::geometry::{Ranks, SUPPORTED_RANKS};
use bundle_runtime::graphs::programs::{Admission, Eager, Limits, Programs};
use bundle_runtime::graphs::{GraphCache, GraphKey};
use bundle_runtime::pipeline::{Collectives, PreparedLaunch};
use kernel_core::contract::{ArgumentList, LaunchPlan};
use runtime_bundle_abi::{BundleAbiError, BundleStreamHandleV1, HostServicesTableV1};
use runtime_host::fake::{FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

struct Steps;

impl BundleGeometry for Steps {
    const PAGE_TOKENS: u64 = 64;
    const PREFILL_TOKENS: u32 = 8192;
    const DECODE_BATCH: u32 = 32;
    const MAX_VERIFY_ROWS: u32 = 8;
    const TOKEN_GRANULE: u32 = 256;
}

type Backend = HostServicesV1Backend<FakeHostDeviceV1>;
type Store = Programs<u32, Steps>;

fn rank_counts() -> impl Iterator<Item = Ranks> {
    SUPPORTED_RANKS
        .into_iter()
        .map(|count| Ranks::new(count).unwrap())
}

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

fn launch() -> PreparedLaunch {
    PreparedLaunch {
        symbol: "segment".into(),
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
    }
}

/// A one-graph decode bucket of `bucket` on every rank of `owner`.
fn decode(
    owner: &mut GraphCache<Steps>,
    table: &HostServicesTableV1,
    streams: &[BundleStreamHandleV1],
    bucket: u32,
) {
    for rank in 0..owner.ranks().get() {
        let key = GraphKey {
            rank,
            bucket,
            items: 0,
        };
        owner
            .capture::<BundleAbiError>(table.caller(), streams[rank as usize], key, 1, |_| Ok(()))
            .unwrap();
    }
}

const LIMITS: Limits = Limits {
    programs: 2,
    seen: 4,
    headroom: 0,
};

/// Three segments per rank, the middle one empty: two graphs per rank.
fn segments(ranks: Ranks, launches: &[PreparedLaunch]) -> Vec<Vec<&[PreparedLaunch]>> {
    let empty: &[PreparedLaunch] = &[];
    vec![
        vec![&launches[..1]; ranks.count()],
        vec![empty; ranks.count()],
        vec![&launches[1..]; ranks.count()],
    ]
}

/// Admits `key`, capturing it when admitted.
fn run(
    store: &mut Store,
    owner: &GraphCache<Steps>,
    table: &HostServicesTableV1,
    streams: &[BundleStreamHandleV1],
    key: u32,
) -> Result<Option<Eager>, DispatchError> {
    let launches = [launch(), launch(), launch()];
    match store.admit(table.caller(), owner, &key, 2, || Ok(()))? {
        Admission::Replay => Ok(None),
        Admission::Eager(why) => Ok(Some(why)),
        Admission::Capture(mut reservation) => {
            store.capture(
                table.caller(),
                owner,
                key,
                &mut reservation,
                streams,
                &segments(owner.ranks(), &launches),
                ("prefill", Some(key)),
                |_, _| Ok(()),
            )?;
            Ok(None)
        }
    }
}

#[test]
fn a_key_captures_at_its_second_run_and_replays_between_its_gaps() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut owner = GraphCache::<Steps>::new(ranks);
        decode(&mut owner, &table, &streams, 1);
        let warmed = backend.live_resource_count();
        let mut store = Store::new(LIMITS);
        assert_eq!(
            run(&mut store, &owner, &table, &streams, 7).unwrap(),
            Some(Eager::FirstRun)
        );
        assert!(!store.contains(&7));
        assert_eq!(backend.live_resource_count(), warmed);
        // Each nonempty segment captures rank after rank.
        let launches = [launch(), launch(), launch()];
        let Admission::Capture(mut reservation) = store
            .admit(table.caller(), &owner, &7, 2, || {
                panic!("nothing to release")
            })
            .unwrap()
        else {
            panic!("a repeated key captures");
        };
        let mut issued = Vec::new();
        store
            .capture(
                table.caller(),
                &owner,
                7,
                &mut reservation,
                &streams,
                &segments(ranks, &launches),
                ("prefill", Some(7)),
                |rank, launches| {
                    issued.push((rank, launches.len()));
                    Ok(())
                },
            )
            .unwrap();
        drop(reservation);
        let expected: Vec<_> = [1, 2]
            .into_iter()
            .flat_map(|len| (0..ranks.count()).map(move |rank| (rank, len)))
            .collect();
        assert_eq!(issued, expected);
        assert!(store.contains(&7));
        assert_eq!(backend.live_resource_count() - warmed, 2 * ranks.count());
        assert_eq!(owner.stats().reserved, 0);
        // A replay runs the host work between every pair of segments.
        let collectives = Collectives::default();
        let mut gaps = Vec::new();
        assert!(matches!(
            store.admit(table.caller(), &owner, &7, 2, || panic!()),
            Ok(Admission::Replay)
        ));
        let replayed = store
            .replay(table.caller(), &7, &streams, &collectives, |gap| {
                gaps.push(gap);
                Ok(())
            })
            .unwrap();
        assert!(replayed && gaps == [0, 1]);
        // An uncaptured key replays nothing.
        assert!(
            !store
                .replay(table.caller(), &8, &streams, &collectives, |_| panic!())
                .unwrap()
        );
        let stats = store.stats();
        assert_eq!(
            (
                stats.programs,
                stats.captures,
                stats.first_runs,
                stats.replays
            ),
            (1, 1, 1, 1)
        );
        owner.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
        assert!(!store.contains(&7));
        assert_eq!(store.stats().programs, 0);
    }
}

#[test]
fn a_full_store_releases_its_newest_program_and_keeps_older_graphs() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut owner = GraphCache::<Steps>::new(ranks);
        decode(&mut owner, &table, &streams, 1);
        let warmed = backend.live_resource_count();
        let mut store = Store::new(LIMITS);
        for key in [1, 1, 2, 2] {
            run(&mut store, &owner, &table, &streams, key).unwrap();
        }
        assert!(store.contains(&1) && store.contains(&2));
        assert_eq!(
            run(&mut store, &owner, &table, &streams, 3).unwrap(),
            Some(Eager::FirstRun)
        );
        // The third program first releases the newest, after the drain.
        let mut drained = 0;
        let admitted = store
            .admit(table.caller(), &owner, &3, 2, || {
                drained += 1;
                Ok(())
            })
            .unwrap();
        assert!(matches!(admitted, Admission::Capture(_)));
        drop(admitted);
        assert_eq!(drained, 1);
        assert!(store.contains(&1) && !store.contains(&2));
        assert_eq!(backend.live_resource_count() - warmed, 2 * ranks.count());
        assert!(owner.contains(GraphKey {
            rank: 0,
            bucket: 1,
            items: 0
        }));
        assert_eq!(store.stats().released, 1);
        owner.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

#[test]
fn a_program_under_a_newer_decode_graph_stays_and_the_new_key_runs_eagerly() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut owner = GraphCache::<Steps>::new(ranks);
        let mut store = Store::new(Limits {
            programs: 1,
            ..LIMITS
        });
        for _ in 0..2 {
            run(&mut store, &owner, &table, &streams, 1).unwrap();
        }
        // A decode bucket captured lazily lies above the program.
        decode(&mut owner, &table, &streams, 2);
        let live = backend.live_resource_count();
        run(&mut store, &owner, &table, &streams, 2).unwrap();
        let full = store
            .admit(table.caller(), &owner, &2, 2, || panic!("no release"))
            .unwrap();
        assert!(matches!(full, Admission::Eager(Eager::Full)));
        assert_eq!(backend.live_resource_count(), live);
        assert!(store.contains(&1));
        // The refused key is remembered, so it captures once there is room.
        assert_eq!(store.stats().full, 1);
        owner.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
        assert!(matches!(
            store
                .admit(table.caller(), &owner, &2, 2, || Ok(()))
                .unwrap(),
            Admission::Capture(_)
        ));
    }
}

#[test]
fn a_bounded_ledger_keeps_its_headroom_for_lazy_decode_graphs() {
    for ranks in rank_counts() {
        let (table, _backend, streams, _) = host(ranks);
        let mut owner = GraphCache::<Steps>::new(ranks);
        let count = ranks.count();
        // Room for one decode bucket, one two-segment program and a
        // headroom of one handle per rank.
        owner.set_capacity(4 * count).unwrap();
        decode(&mut owner, &table, &streams, 1);
        let mut store = Store::new(Limits {
            programs: 4,
            seen: 4,
            headroom: 1,
        });
        for _ in 0..2 {
            run(&mut store, &owner, &table, &streams, 1).unwrap();
        }
        assert!(store.contains(&1));
        run(&mut store, &owner, &table, &streams, 2).unwrap();
        assert_eq!(
            run(&mut store, &owner, &table, &streams, 2).unwrap(),
            Some(Eager::Ledger)
        );
        // The headroom still admits the decode graph the ledger kept it for.
        decode(&mut owner, &table, &streams, 2);
        assert_eq!(owner.stats().live, 4 * count);
        assert!(!owner.stats().pressure);
        owner.release_all(table.caller()).unwrap();
    }
}

#[test]
fn a_failed_capture_releases_what_it_took_and_stops_captures() {
    for ranks in rank_counts() {
        let (table, backend, streams, baseline) = host(ranks);
        let mut owner = GraphCache::<Steps>::new(ranks);
        decode(&mut owner, &table, &streams, 1);
        let mut store = Store::new(LIMITS);
        for _ in 0..2 {
            run(&mut store, &owner, &table, &streams, 1).unwrap();
        }
        let before = backend.live_resource_count();
        run(&mut store, &owner, &table, &streams, 2).unwrap();
        let Admission::Capture(mut reservation) = store
            .admit(table.caller(), &owner, &2, 2, || Ok(()))
            .unwrap()
        else {
            panic!("admitted");
        };
        let launches = [launch(), launch(), launch()];
        let last = ranks.count() - 1;
        let failed = store.capture(
            table.caller(),
            &owner,
            2,
            &mut reservation,
            &streams,
            &segments(ranks, &launches),
            ("prefill", None),
            |rank, launches| {
                if rank == last && launches.len() == 2 {
                    Err(DispatchError::InvalidPlan)
                } else {
                    Ok(())
                }
            },
        );
        drop(reservation);
        assert!(matches!(failed, Err(DispatchError::InvalidPlan)));
        assert_eq!(backend.live_resource_count(), before);
        assert_eq!(owner.stats().reserved, 0);
        assert!(store.contains(&1) && !store.contains(&2));
        assert_eq!(store.stats().failures, 1);
        // Captured programs keep replaying; new keys stay eager.
        let collectives = Collectives::default();
        assert!(
            store
                .replay(table.caller(), &1, &streams, &collectives, |_| Ok(()))
                .unwrap()
        );
        for _ in 0..2 {
            assert_eq!(
                run(&mut store, &owner, &table, &streams, 3).unwrap(),
                Some(Eager::Stopped)
            );
        }
        owner.release_all(table.caller()).unwrap();
        assert_eq!(backend.live_resource_count(), baseline);
    }
}

#[test]
fn a_failed_release_leaves_its_graphs_to_the_ledger_and_stops_captures() {
    let ranks = Ranks::TP4;
    let (table, backend, streams, baseline) = host(ranks);
    let mut owner = GraphCache::<Steps>::new(ranks);
    let mut store = Store::new(Limits {
        programs: 1,
        ..LIMITS
    });
    for _ in 0..2 {
        run(&mut store, &owner, &table, &streams, 1).unwrap();
    }
    run(&mut store, &owner, &table, &streams, 2).unwrap();
    backend.with_device_mut(|device| device.refuse_after(FakeRefusalV1::ReleaseGraph, 1));
    assert!(run(&mut store, &owner, &table, &streams, 2).is_err());
    assert!(!store.contains(&1) && !store.contains(&2));
    assert_eq!(store.stats().failures, 1);
    assert_eq!(
        run(&mut store, &owner, &table, &streams, 2).unwrap(),
        Some(Eager::Stopped)
    );
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseGraph));
    owner.release_all(table.caller()).unwrap();
    assert_eq!(backend.live_resource_count(), baseline);
}

#[test]
fn segments_empty_on_some_ranks_only_are_refused_before_any_capture() {
    let ranks = Ranks::TP2;
    let (table, backend, streams, baseline) = host(ranks);
    let owner = GraphCache::<Steps>::new(ranks);
    let mut store = Store::new(LIMITS);
    let launches = [launch()];
    let empty: &[PreparedLaunch] = &[];
    let uneven = vec![vec![&launches[..], empty]];
    run(&mut store, &owner, &table, &streams, 1).unwrap();
    let Admission::Capture(mut reservation) = store
        .admit(table.caller(), &owner, &1, 1, || Ok(()))
        .unwrap()
    else {
        panic!("admitted");
    };
    let refused = store.capture(
        table.caller(),
        &owner,
        1,
        &mut reservation,
        &streams,
        &uneven,
        ("prefill", None),
        |_, _| panic!("nothing captures"),
    );
    assert!(matches!(refused, Err(DispatchError::InvalidPlan)));
    drop(reservation);
    assert_eq!(backend.live_resource_count(), baseline);
    assert_eq!(owner.stats().reserved, 0);
}

#[test]
fn a_store_without_programs_never_captures() {
    let (table, backend, streams, baseline) = host(Ranks::TP1);
    let owner = GraphCache::<Steps>::new(Ranks::TP1);
    let mut store = Store::new(Limits {
        programs: 0,
        ..LIMITS
    });
    for _ in 0..3 {
        assert_eq!(
            run(&mut store, &owner, &table, &streams, 1).unwrap(),
            Some(Eager::Stopped)
        );
    }
    assert_eq!(backend.live_resource_count(), baseline);
}
