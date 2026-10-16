//! The shared ABI entry bodies behind a probe bundle exported through the
//! ABI macro, over the host's own resource bookkeeping
//! (`HostServicesV1Backend<FakeHostDeviceV1>`).

use std::cell::RefCell;
use std::rc::Rc;

use bundle_runtime::entry::{self, Bundle, Failed};
use bundle_runtime::pipeline::Collectives;
use bundle_runtime::resources::{Held, status};
use bundle_runtime::table::{InstanceTable, Live};
use runtime_bundle_abi::{
    BundleAbiError, BundleApiV1, BundleHandleV1, BundleKeptRowsV1, BundleStatusV1,
    ExecuteResultSinkV1, HostServicesTableV1, HostServicesV1,
};
use runtime_host::fake::{FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

/// Holds one context, one stream and one event. Its configuration names how create
/// fails: `refuse` before acquiring anything, `fail` after, `fail-graph` after
/// also acquiring an event that stands in for a graph captured last.
struct Probe {
    services: HostServicesV1,
    held: Vec<Held>,
    /// Released by `release_graphs`, before `held`.
    graphs: Vec<Held>,
    collectives: Collectives,
    commits: u32,
}

thread_local! {
    static INSTANCES: RefCell<InstanceTable<Probe>> = const { RefCell::new(InstanceTable::new()) };
}

impl Live for Probe {
    fn collectives(&self) -> &Collectives {
        &self.collectives
    }
}

impl Bundle for Probe {
    fn instances() -> &'static std::thread::LocalKey<RefCell<InstanceTable<Self>>> {
        &INSTANCES
    }

    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>> {
        if configuration == b"refuse" {
            return Err(BundleStatusV1::InvalidArgument.into());
        }
        let mut probe = Self {
            services,
            held: Vec::new(),
            graphs: Vec::new(),
            collectives: Collectives::default(),
            commits: 0,
        };
        let acquired = probe.services.select_context(0, 0).and_then(|context| {
            probe.held.push(Held::Context(context));
            let stream = probe.services.create_stream(context, 0)?;
            probe.held.push(Held::Stream(stream));
            let event = probe.services.create_event(context)?;
            probe.held.push(Held::Event(event));
            if configuration == b"fail-graph" {
                let graph = probe.services.create_event(context)?;
                probe.graphs.push(Held::Event(graph));
            }
            Ok(())
        });
        match acquired {
            Err(error) => Err(Failed::releasing(probe, status(error))),
            Ok(()) if configuration == b"fail" || configuration == b"fail-graph" => {
                Err(Failed::releasing(probe, BundleStatusV1::Unsupported))
            }
            Ok(()) => Ok(probe),
        }
    }

    fn resources(&mut self) -> (&HostServicesV1, &mut Vec<Held>) {
        (&self.services, &mut self.held)
    }

    fn release_graphs(&mut self) -> Result<(), BundleAbiError> {
        bundle_runtime::resources::release_all(&self.services, &mut self.graphs)
    }

    fn execute(&mut self, _request: &[u8], _sink: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
        BundleStatusV1::Unsupported
    }

    fn commit(&mut self) -> BundleStatusV1 {
        self.commits += 1;
        BundleStatusV1::Success
    }

    fn commit_accepted(&mut self, _records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
        self.commit()
    }

    /// Stands in for a submission that cannot be drained.
    fn cancel(&mut self) -> BundleStatusV1 {
        self.collectives.poison();
        BundleStatusV1::HostError
    }

    fn retire_request(&mut self, _request_id: u64) -> BundleStatusV1 {
        BundleStatusV1::Success
    }
}

runtime_bundle_abi::export_bundle_api_v1!(
    PROBE_API_V1,
    entry::create::<Probe>,
    entry::execute::<Probe>,
    entry::commit::<Probe>,
    entry::cancel::<Probe>,
    entry::destroy::<Probe>,
);

type Backend = HostServicesV1Backend<FakeHostDeviceV1>;

fn backend() -> Rc<Backend> {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap());
    Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config))
}

/// Creates a probe over `backend`; the handle outlives the wrapper, whose
/// own finalization is not under test.
fn create(backend: &Rc<Backend>, configuration: &[u8]) -> Result<BundleHandleV1, BundleAbiError> {
    let api = BundleApiV1::new(PROBE_API_V1).unwrap();
    let instance = api.create(HostServicesTableV1::new(Rc::clone(backend)), configuration)?;
    let handle = instance.handle();
    std::mem::forget(instance);
    Ok(handle)
}

#[test]
fn calls_reach_the_instance_and_destroy_releases_what_it_holds() {
    let backend = backend();
    let handle = create(&backend, b"").unwrap();
    assert_eq!(backend.live_resource_count(), 3);
    assert_eq!(entry::commit::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!(
        entry::commit_accepted::<Probe>(handle, &[]),
        BundleStatusV1::Success
    );
    let commits = INSTANCES.with(|table| table.borrow_mut().get_mut(handle).unwrap().commits);
    assert_eq!(commits, 2);
    assert_eq!(entry::destroy::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(entry::commit::<Probe>(handle), BundleStatusV1::StaleHandle);
    assert_eq!(entry::destroy::<Probe>(handle), BundleStatusV1::StaleHandle);
}

#[test]
fn a_failed_create_gives_back_what_it_acquired() {
    let backend = backend();
    assert!(create(&backend, b"refuse").is_err());
    assert!(create(&backend, b"fail").is_err());
    assert_eq!(backend.live_resource_count(), 0);
    INSTANCES.with(|table| {
        assert!(
            table
                .borrow_mut()
                .get_mut(BundleHandleV1::from_parts(1, 1))
                .is_none()
        )
    });
}

/// A failed create whose graphs were captured after its other resources
/// releases the graphs first (the host releases newest first), so nothing
/// stays behind for a sweep.
#[test]
fn a_failed_create_releases_its_graphs_before_what_it_holds() {
    let backend = backend();
    assert!(create(&backend, b"fail-graph").is_err());
    assert_eq!((retained(), backend.live_resource_count()), (0, 0));
}

#[test]
fn a_poisoned_instance_refuses_calls_and_keeps_its_resources() {
    let backend = backend();
    let handle = create(&backend, b"").unwrap();
    assert_eq!(entry::cancel::<Probe>(handle), BundleStatusV1::HostError);
    for status in [
        entry::commit::<Probe>(handle),
        entry::cancel::<Probe>(handle),
        entry::retire_request::<Probe>(handle, 7),
        entry::destroy::<Probe>(handle),
    ] {
        assert_eq!(status, BundleStatusV1::InvalidState);
    }
    assert_eq!(backend.live_resource_count(), 3);
}

/// Refuses `refusal` on `backend` until `allow`.
fn refuse(backend: &Backend, refusal: FakeRefusalV1) {
    backend.with_device_mut(|device| device.refuse(refusal));
}

fn allow(backend: &Backend, refusal: FakeRefusalV1) {
    backend.with_device_mut(|device| device.allow(refusal));
}

/// The instances whose release failed, waiting for a retry.
fn retained() -> usize {
    INSTANCES.with(|table| table.borrow().unreleased_count())
}

/// A failed create whose release fails keeps the instance with what it
/// still holds (newest first: the stream went back, the context did not).
/// Each later create retries it until the release succeeds, without
/// releasing the event twice.
#[test]
fn a_create_retries_a_failed_creates_release_until_it_succeeds() {
    let backend = backend();
    refuse(&backend, FakeRefusalV1::ReleaseStream);
    assert!(create(&backend, b"fail").is_err());
    assert_eq!((retained(), backend.live_resource_count()), (1, 2));
    // The sweep fails again: the instance stays, beside the new one whose
    // release failed the same way.
    assert!(create(&backend, b"fail").is_err());
    assert_eq!((retained(), backend.live_resource_count()), (2, 3));
    allow(&backend, FakeRefusalV1::ReleaseStream);
    let handle = create(&backend, b"").unwrap();
    assert_eq!((retained(), backend.live_resource_count()), (0, 3));
    assert_eq!(entry::destroy::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!(backend.live_resource_count(), 0);
}

/// Destroy retries a failed create's release before it looks up its
/// handle, and keeps a live instance whose own release fails, holding only
/// what did not go back, for a later destroy.
#[test]
fn a_destroy_retries_retained_releases_and_keeps_an_instance_it_cannot_release() {
    let backend = backend();
    let handle = create(&backend, b"").unwrap();
    refuse(&backend, FakeRefusalV1::ReleaseStream);
    assert!(create(&backend, b"fail").is_err());
    assert_eq!((retained(), backend.live_resource_count()), (1, 4));
    allow(&backend, FakeRefusalV1::ReleaseStream);
    let stale = BundleHandleV1::from_parts(handle.generation() + 1, handle.slot());
    assert_eq!(entry::destroy::<Probe>(stale), BundleStatusV1::StaleHandle);
    assert_eq!((retained(), backend.live_resource_count()), (0, 3));
    // The event goes back and the stream does not: the instance stays live
    // with its context and stream.
    refuse(&backend, FakeRefusalV1::ReleaseStream);
    assert_ne!(entry::destroy::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!(backend.live_resource_count(), 2);
    assert_eq!(entry::commit::<Probe>(handle), BundleStatusV1::Success);
    allow(&backend, FakeRefusalV1::ReleaseStream);
    assert_eq!(entry::destroy::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!(backend.live_resource_count(), 0);
}

/// A table in use (a call re-entered from a host callback) refuses create
/// with `InvalidState` before it acquires anything, and destroy before it
/// retries a retained release.
#[test]
fn a_reentered_table_refuses_create_and_destroy() {
    let backend = backend();
    let handle = create(&backend, b"").unwrap();
    refuse(&backend, FakeRefusalV1::ReleaseStream);
    assert!(create(&backend, b"fail").is_err());
    allow(&backend, FakeRefusalV1::ReleaseStream);
    let live = backend.live_resource_count();
    INSTANCES.with(|table| {
        let _busy = table.borrow();
        assert_eq!(
            entry::destroy::<Probe>(handle),
            BundleStatusV1::InvalidState
        );
        assert_eq!(entry::commit::<Probe>(handle), BundleStatusV1::InvalidState);
        assert_eq!(
            create(&backend, b"").err(),
            Some(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
    });
    assert_eq!((retained(), backend.live_resource_count()), (1, live));
    assert_eq!(entry::destroy::<Probe>(handle), BundleStatusV1::Success);
    assert_eq!((retained(), backend.live_resource_count()), (0, 0));
}
