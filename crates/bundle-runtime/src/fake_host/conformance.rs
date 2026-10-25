//! Additive CPU conformance coverage over a model's existing fake host.
//! Models supply requests, raw validation callbacks and release accounting.

mod accounting;
pub mod route_guard;
mod startup;
pub use route_guard::Envelope;

pub use accounting::ObservedHost;
pub use startup::{
    Coverage, Item, PackageSpeculation, Phase, Serving, Startup, serving_startup,
    serving_startup_of, startup_profiles,
};

pub use crate::conformance_suite as suite;

use std::rc::Rc;

use runtime_bundle_abi::{
    BundleAbiError, BundleApiV1, BundleHandleV1, BundleInstanceV1, BundleKeptRowsV1,
    BundleStatusV1, HostServicesBackendV1, HostServicesTableV1,
};

use super::{FakeDevice, JournalObserver, RecordingSink, SinkLimits};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    AllocationRelease,
    MalformedCommit,
    PoisonedClient,
    Cancellation,
    SlotReuse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancellationPolicy {
    PreserveCommitted,
    InvalidateTouched,
}

/// Every scenario is required. An adapter uses one served slot so a fresh
/// request after retirement must reuse the same slot. It names its model's
/// serving configurations, whose startup its suite (`suite!`) runs.
pub trait Adapter: Sized {
    type Startup: Startup;
    type Observer: JournalObserver;
    type Limits: SinkLimits;
    type Host: HostServicesBackendV1 + 'static;

    fn new() -> Self;
    fn host(&self) -> Rc<Self::Host>;
    fn device(&self) -> &FakeDevice<Self::Observer>;
    fn api(&self) -> BundleApiV1;
    fn configuration(&self) -> Vec<u8>;
    fn request(&self, request: u64, start: u64, tokens: &[u32], verify: bool) -> Vec<u8>;
    fn raw_commit(&self, handle: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1;
    /// Check all successful acquisitions, including non-buffer resources.
    fn assert_released(&self);
    fn release_count(&self) -> usize;
    fn callback_count(&self) -> usize;
    fn cancellation_policy(&self) -> CancellationPolicy;
    fn arm_cancel_failure(&self);
    fn assert_cancel_failure_consumed(&self);
    fn assert_poisoned_prefix_calls(&self, instance: &mut BundleInstanceV1);

    fn create(&self) -> Result<BundleInstanceV1, BundleAbiError> {
        self.api()
            .create(HostServicesTableV1::new(self.host()), &self.configuration())
    }
}

fn execute<A: Adapter>(
    adapter: &A,
    instance: &mut BundleInstanceV1,
    request: u64,
    start: u64,
    tokens: &[u32],
    verify: bool,
) -> Result<(), BundleAbiError> {
    instance.execute(
        &adapter.request(request, start, tokens, verify),
        &mut RecordingSink::<A::Limits>::default(),
    )
}

fn kept(request: u64, rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id: request,
        kept_rows: rows,
        reserved_tail: 0,
    }
}

fn prepared<A: Adapter>() -> (A, BundleInstanceV1) {
    let adapter = A::new();
    let mut instance = adapter.create().expect("create");
    execute(&adapter, &mut instance, 7, 0, &[1, 2, 3], false).expect("prefill");
    instance.commit().expect("commit prefill");
    (adapter, instance)
}

fn terminal<A: Adapter>(adapter: &A, mut instance: BundleInstanceV1) {
    assert!(instance.is_poisoned());
    let journal = adapter.device().journal.borrow().clone();
    let releases = adapter.release_count();
    let callbacks = adapter.callback_count();
    let poisoned = Err(BundleAbiError::Poisoned);
    assert_eq!(execute(adapter, &mut instance, 7, 3, &[4], false), poisoned);
    assert_eq!(instance.commit(), poisoned);
    assert_eq!(instance.commit_accepted(&[kept(7, 1)]), poisoned);
    assert_eq!(instance.cancel(), poisoned);
    assert_eq!(instance.retire_request(7), poisoned);
    adapter.assert_poisoned_prefix_calls(&mut instance);
    assert_eq!(instance.destroy(), poisoned);
    assert_eq!(
        adapter.callback_count(),
        callbacks,
        "poisoned client invoked a host callback"
    );
    drop(instance);
    assert_eq!(
        adapter.callback_count(),
        callbacks,
        "poisoned drop invoked a host callback"
    );
    assert_eq!(adapter.release_count(), releases);
    assert_eq!(
        *adapter.device().journal.borrow(),
        journal,
        "terminal client must issue no services or release resources"
    );
}

// Compare rollback against a fresh instance whose transaction never saw a
// refusal. The journal includes every copy's source, destination and length.
fn refusal_then_cancel<A: Adapter>(records: Option<&[BundleKeptRowsV1]>) -> Vec<String> {
    let (adapter, mut instance) = prepared::<A>();
    execute(&adapter, &mut instance, 7, 3, &[4, 5, 6], true).unwrap();
    if let Some(records) = records {
        let journal = adapter.device().journal.borrow().clone();
        assert_eq!(
            adapter.raw_commit(instance.handle(), records),
            BundleStatusV1::InvalidArgument
        );
        assert_eq!(*adapter.device().journal.borrow(), journal);
    }
    let start = adapter.device().journal.borrow().len();
    instance.cancel().expect("cancel after raw refusal");
    let trace = adapter.device().journal.borrow()[start..].to_vec();
    match adapter.cancellation_policy() {
        CancellationPolicy::PreserveCommitted => {
            execute(&adapter, &mut instance, 7, 3, &[9], false)
                .expect("cancel preserves committed frontier");
        }
        CancellationPolicy::InvalidateTouched => {
            assert!(execute(&adapter, &mut instance, 7, 3, &[9], false).is_err());
            instance.retire_request(7).unwrap();
            execute(&adapter, &mut instance, 7, 0, &[1, 2, 3], false).unwrap();
        }
    }
    instance.commit().unwrap();
    instance.destroy().unwrap();
    adapter.assert_released();
    trace
}

fn allocation_indices(
    allocations: usize,
    shard: usize,
    shards: usize,
) -> impl Iterator<Item = usize> {
    assert!(
        shards > 0 && shards <= allocations,
        "each shard must exercise a failure"
    );
    assert!(shard < shards, "shard index is out of range");
    (shard..allocations).step_by(shards)
}

/// Exercise a disjoint part of the allocation sweep. Every failure still
/// creates a fresh adapter and instance and checks the original cutoff and
/// release invariants. One shard keeps the complete original sweep.
#[doc(hidden)]
pub fn allocation_release_shard<S: Startup, A: Adapter<Startup = S>>(
    shard: usize,
    shards: usize,
) -> usize {
    let adapter = A::new();
    let mut instance = adapter.create().expect("create");
    let allocations = adapter.device().buffer_order.borrow().len();
    assert!(allocations > 0);
    instance.destroy().expect("destroy");
    adapter.assert_released();
    let mut cases = 1;
    for index in allocation_indices(allocations, shard, shards) {
        let failed = A::new();
        failed
            .device()
            .fail_allocate
            .set(Some(index.try_into().unwrap()));
        assert_eq!(
            failed.create().err(),
            Some(BundleAbiError::Status(BundleStatusV1::Panic)),
            "allocation {index}"
        );
        assert_eq!(
            failed.device().buffer_order.borrow().len(),
            index,
            "first failure cutoff"
        );
        assert_eq!(failed.device().fail_allocate.get(), None, "fault consumed");
        assert_eq!(failed.device().allocation_failures.get(), 1);
        failed.assert_released();
        cases += 1;
    }
    println!(
        "AllocationRelease: shard={shard}/{shards} failures={} allocations={allocations}",
        cases - 1
    );
    cases
}

/// Return the number of independently exercised cases. Model tests report it
/// for each scenario, including every allocation failure index. Only
/// `suite!` calls it, beside the suite's `serving_startup` tests of the
/// adapter's startup `S`.
#[doc(hidden)]
pub fn run<S: Startup, A: Adapter<Startup = S>>(scenario: Scenario) -> usize {
    let cases = match scenario {
        Scenario::AllocationRelease => allocation_release_shard::<S, A>(0, 1),
        Scenario::MalformedCommit => {
            let (adapter, mut instance) = prepared::<A>();
            execute(&adapter, &mut instance, 7, 3, &[4, 5, 6], true).expect("verify");
            let journal = adapter.device().journal.borrow().clone();
            let invalid = [
                vec![],
                vec![kept(8, 1)],
                vec![kept(7, 0)],
                vec![kept(7, 4)],
                vec![kept(7, 1), kept(7, 1)],
            ];
            for records in &invalid {
                assert_eq!(
                    adapter.raw_commit(instance.handle(), records),
                    BundleStatusV1::InvalidArgument
                );
                assert_eq!(
                    *adapter.device().journal.borrow(),
                    journal,
                    "raw refusal must not touch device state"
                );
            }
            instance
                .commit_accepted(&[kept(7, 2)])
                .expect("raw refusals leave transaction intact");
            assert!(
                execute(&adapter, &mut instance, 7, 6, &[7], false).is_err(),
                "offered frontier must not have committed"
            );
            execute(&adapter, &mut instance, 7, 5, &[7], false).expect("accepted frontier");
            instance.commit().unwrap();
            instance.destroy().unwrap();
            let baseline = refusal_then_cancel::<A>(None);
            for records in &invalid {
                assert_eq!(
                    refusal_then_cancel::<A>(Some(records)),
                    baseline,
                    "refusal must preserve the no-refusal rollback trace"
                );
            }
            invalid.len() * 2
        }
        Scenario::PoisonedClient => {
            // Exercise each terminal method independently. Raw callback retry
            // coverage above does not authorize retry through this client.
            for method in 0..3 {
                let (adapter, mut instance) = prepared::<A>();
                if method < 2 {
                    execute(&adapter, &mut instance, 7, 3, &[4, 5, 6], true).unwrap();
                    let result = if method == 0 {
                        instance.commit()
                    } else {
                        instance.commit_accepted(&[kept(8, 1)])
                    };
                    assert!(result.is_err());
                } else {
                    // A prefill continuation saves state that cancel must restore.
                    execute(&adapter, &mut instance, 7, 3, &[4, 5], false).unwrap();
                    adapter.arm_cancel_failure();
                    assert!(instance.cancel().is_err());
                    adapter.assert_cancel_failure_consumed();
                }
                terminal(&adapter, instance);
            }
            3
        }
        Scenario::Cancellation => {
            let (adapter, mut instance) = prepared::<A>();
            execute(&adapter, &mut instance, 7, 3, &[4], false).unwrap();
            instance.cancel().unwrap();
            let retry = execute(&adapter, &mut instance, 7, 3, &[4], false);
            match adapter.cancellation_policy() {
                CancellationPolicy::PreserveCommitted => {
                    retry.unwrap();
                    instance.commit().unwrap();
                }
                CancellationPolicy::InvalidateTouched => {
                    assert!(retry.is_err());
                    instance.retire_request(7).unwrap();
                    execute(&adapter, &mut instance, 7, 0, &[1, 2, 3], false).unwrap();
                    instance.commit().unwrap();
                }
            }
            instance.destroy().unwrap();
            1
        }
        Scenario::SlotReuse => {
            let (adapter, mut instance) = prepared::<A>();
            assert!(
                execute(&adapter, &mut instance, 8, 0, &[9, 10, 11], false).is_err(),
                "the only slot is occupied"
            );
            instance.retire_request(7).unwrap();
            assert!(
                execute(&adapter, &mut instance, 7, 3, &[4], false).is_err(),
                "retired frontier must be absent"
            );
            execute(&adapter, &mut instance, 8, 0, &[9, 10, 11], false).unwrap();
            instance.commit().unwrap();
            assert!(
                execute(&adapter, &mut instance, 7, 3, &[4], false).is_err(),
                "old identity cannot inherit the reused slot"
            );
            execute(&adapter, &mut instance, 8, 3, &[12], false).unwrap();
            instance.commit().unwrap();
            instance.destroy().unwrap();
            1
        }
    };
    assert!(cases > 0);
    cases
}

/// A bundle's conformance suite, invoked once per bundle crate (its
/// inventory test requires it): a `serving_startup` module with one test per
/// profile of the startup adapter, `test: "profile",` (the profile's name
/// with `_` for every other character), and a `profiles` test that checks
/// the list and the adapter's configurations against profiles.json; then
/// one test per scenario line, `name: Adapter => Scenario;`, optionally
/// grouped in `mod name { .. }`. `AllocationReleaseShards` partitions the
/// complete allocation sweep into 16 tests with fresh instances at every
/// failure index. Each adapter's `Startup` must be the
/// suite's, so each profile starts once per suite in any test runner,
/// including one process per test, and the profiles start in parallel.
///
/// ```ignore
/// conformance::suite! {
///     startup: Served {
///         tp4_ar: "tp4-ar",
///         tp4_mtp: "tp4-mtp",
///     }
///     allocation_release: Model => AllocationRelease;
///     mod bf16 {
///         cancellation_policy: Model<BF16> => Cancellation;
///     }
/// }
/// ```
#[macro_export]
macro_rules! conformance_suite {
    (
        startup: $startup:ty { $($profile:ident: $name:literal),+ $(,)? }
        $($tests:tt)*
    ) => {
        mod serving_startup {
            use super::*;

            #[test]
            fn profiles() {
                $crate::fake_host::conformance::startup_profiles::<$startup>(&[
                    $((stringify!($profile), $name)),+
                ]);
            }
            $(
                #[test]
                fn $profile() {
                    $crate::fake_host::conformance::serving_startup_of::<$startup>($name);
                }
            )+
        }
        mod route_sweep {
            use super::*;
            $(
                #[test]
                fn $profile() {
                    $crate::fake_host::conformance::route_guard::sweep::<$startup>(
                        $name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")), env!("CARGO_PKG_NAME"),
                    );
                }
            )+
        }
        $crate::__conformance_scenarios! { $startup; $($tests)* }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __conformance_scenarios {
    ($startup:ty;) => {};
    ($startup:ty; mod $module:ident { $($tests:tt)* } $($rest:tt)*) => {
        mod $module {
            use super::*;
            $crate::__conformance_scenarios! { $startup; $($tests)* }
        }
        $crate::__conformance_scenarios! { $startup; $($rest)* }
    };
    ($startup:ty; $test:ident: $adapter:ty => AllocationReleaseShards; $($rest:tt)*) => {
        mod $test {
            use super::*;
            $crate::__conformance_allocation_shards! {
                $startup, $adapter;
                shard_00 = 0, shard_01 = 1, shard_02 = 2, shard_03 = 3,
                shard_04 = 4, shard_05 = 5, shard_06 = 6, shard_07 = 7,
                shard_08 = 8, shard_09 = 9, shard_10 = 10, shard_11 = 11,
                shard_12 = 12, shard_13 = 13, shard_14 = 14, shard_15 = 15
            }
        }
        $crate::__conformance_scenarios! { $startup; $($rest)* }
    };
    ($startup:ty; $test:ident: $adapter:ty => $scenario:ident; $($rest:tt)*) => {
        #[test]
        fn $test() {
            let scenario = $crate::fake_host::conformance::Scenario::$scenario;
            let cases = $crate::fake_host::conformance::run::<$startup, $adapter>(scenario);
            println!("{scenario:?}: {cases} cases");
        }
        $crate::__conformance_scenarios! { $startup; $($rest)* }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __conformance_allocation_shards {
    ($startup:ty, $adapter:ty; $($name:ident = $index:expr),+ $(,)?) => {
        const SHARDS: &[usize] = &[$($index),+];
        $(
            #[test]
            fn $name() {
                assert!(SHARDS.iter().copied().eq(0..SHARDS.len()), "shards must cover every residue");
                let cases = $crate::fake_host::conformance::allocation_release_shard::<$startup, $adapter>(
                    $index, SHARDS.len(),
                );
                println!("AllocationRelease {}::{} shard={}/{} cases={cases}",
                         module_path!(), stringify!($name), $index, SHARDS.len());
            }
        )+
    };
}

#[cfg(test)]
mod allocation_partition_tests {
    use super::allocation_indices;

    #[test]
    fn every_failure_index_runs_once_including_uneven_partitions() {
        for allocations in [1, 2, 7, 16, 17, 31, 32, 33, 808, 1616, 3232] {
            for shards in [1, 2, 4, 16]
                .into_iter()
                .filter(|&count| count <= allocations)
            {
                let mut covered = Vec::new();
                for shard in 0..shards {
                    let cases: Vec<_> = allocation_indices(allocations, shard, shards).collect();
                    assert!(!cases.is_empty());
                    covered.extend(cases);
                }
                covered.sort_unstable();
                assert_eq!(covered, (0..allocations).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn empty_or_invalid_shards_are_refused() {
        for (allocations, shard, shards) in [(0, 0, 1), (4, 0, 0), (3, 0, 4), (4, 4, 4)] {
            assert!(
                std::panic::catch_unwind(|| allocation_indices(allocations, shard, shards).count())
                    .is_err()
            );
        }
    }
}
