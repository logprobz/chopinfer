//! The ABI entry bodies of a bundle, over one thread's instance table.
//!
//! A bundle implements [`Bundle`] for its instance and exports these bodies
//! (`create::<Instance>`, ...) through `runtime_bundle_abi`'s export macro,
//! which decodes the native arguments and contains panics. The bodies refuse
//! stale handles, re-entered tables and poisoned instances
//! ([`crate::table::usable`]), release what a failed create acquired, and
//! keep any instance whose release failed for a later create or destroy to
//! retry. The bundle keeps everything a call does once its instance is
//! found.

use std::cell::RefCell;
use std::thread::LocalKey;

use runtime_bundle_abi::{
    BundleAbiError, BundleCreateResultV1, BundleHandleV1, BundleKeptRowsV1,
    BundlePrefixStateInfoV1, BundleStatusV1, ExecuteResultSinkV1, HostServicesV1,
    PrefixStateRequestV1,
};

use crate::prefix::{self, PrefixInstance};
use crate::resources::{Held, release_all, status};
use crate::table::{InstanceTable, Live, usable, with_instance};

/// Why create failed, and the instance whose held resources it must give
/// back when it got that far.
pub struct Failed<T> {
    pub status: BundleStatusV1,
    pub instance: Option<Box<T>>,
}

impl<T> Failed<T> {
    /// A failure after `instance` began to hold resources.
    pub fn releasing(instance: T, status: BundleStatusV1) -> Self {
        Self {
            status,
            instance: Some(Box::new(instance)),
        }
    }
}

impl<T> From<BundleStatusV1> for Failed<T> {
    fn from(status: BundleStatusV1) -> Self {
        Self {
            status,
            instance: None,
        }
    }
}

/// A bundle's instance behind the shared entry bodies.
pub trait Bundle: Live + Sized + 'static {
    /// This thread's instances.
    fn instances() -> &'static LocalKey<RefCell<InstanceTable<Self>>>;
    /// Builds an instance from the host's create record.
    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>>;
    /// The host services and the resources the instance holds from them,
    /// newest last.
    fn resources(&mut self) -> (&HostServicesV1, &mut Vec<Held>);
    /// Releases the instance's graphs, before its held resources.
    fn release_graphs(&mut self) -> Result<(), BundleAbiError>;
    fn execute(&mut self, request: &[u8], sink: &mut ExecuteResultSinkV1) -> BundleStatusV1;
    fn commit(&mut self) -> BundleStatusV1;
    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> BundleStatusV1;
    fn cancel(&mut self) -> BundleStatusV1;
    fn retire_request(&mut self, request_id: u64) -> BundleStatusV1;
}

/// Releases `instance`'s graphs, then its held resources, as destroy does
/// (graphs captured after them would block their release, newest first);
/// true when all of them went back.
fn released<T: Bundle>(instance: &mut T) -> bool {
    if instance.release_graphs().is_err() {
        return false;
    }
    let (services, held) = instance.resources();
    release_all(services, held).is_ok()
}

/// Releases `instance`, or keeps it in `table` for [`sweep`] to retry.
fn abandon<T: Bundle>(table: &mut InstanceTable<T>, mut instance: T) {
    if !released(&mut instance) {
        table.unreleased.push(instance);
    }
}

/// Retries the release of the instances a failed release left behind,
/// newest first: a host releases its resources in reverse order of
/// acquisition, so an older instance's release waits for a newer one's.
fn sweep<T: Bundle>(table: &mut InstanceTable<T>) {
    for index in (0..table.unreleased.len()).rev() {
        if released(&mut table.unreleased[index]) {
            table.unreleased.remove(index);
        }
    }
}

pub fn create<T: Bundle>(services: HostServicesV1, configuration: &[u8]) -> BundleCreateResultV1 {
    let _routes =
        crate::routes::logging_enabled().then(|| crate::routes::RouteCounter::begin(true));
    T::instances().with(|instances| {
        let Ok(mut table) = instances.try_borrow_mut() else {
            return BundleCreateResultV1::failure(BundleStatusV1::InvalidState);
        };
        sweep(&mut table);
        match T::create(services, configuration) {
            Ok(instance) => match table.insert(Box::new(instance)) {
                Ok(handle) => BundleCreateResultV1::success(handle),
                Err(instance) => {
                    abandon(&mut table, *instance);
                    BundleCreateResultV1::failure(BundleStatusV1::ResourceExhausted)
                }
            },
            Err(failed) => {
                if let Some(instance) = failed.instance {
                    abandon(&mut table, *instance);
                }
                BundleCreateResultV1::failure(failed.status)
            }
        }
    })
}

/// Account for every execution export, including negotiated prefill callbacks.
pub fn with_route_counter<T: Bundle>(
    instance: &mut T,
    execute: impl FnOnce(&mut T) -> BundleStatusV1,
) -> BundleStatusV1 {
    let counter = (crate::routes::logging_enabled() && !crate::routes::active())
        .then(|| crate::routes::RouteCounter::begin(true));
    let status = execute(instance);
    if let Some(counts) = counter.and_then(crate::routes::RouteCounter::finish) {
        let mut record = counts.json();
        record["schema"] = "chopinfer.route.v1".into();
        let (services, _) = instance.resources();
        let _ = services.log(record.to_string().as_bytes());
    }
    status
}

pub fn execute<T: Bundle>(
    handle: BundleHandleV1,
    request: &[u8],
    sink: &mut ExecuteResultSinkV1,
) -> BundleStatusV1 {
    T::instances().with(|instances| {
        with_instance(instances, handle, |instance| {
            with_route_counter(instance, |instance| instance.execute(request, sink))
        })
    })
}

pub fn commit<T: Bundle>(handle: BundleHandleV1) -> BundleStatusV1 {
    T::instances().with(|instances| with_instance(instances, handle, T::commit))
}

pub fn commit_accepted<T: Bundle>(
    handle: BundleHandleV1,
    records: &[BundleKeptRowsV1],
) -> BundleStatusV1 {
    T::instances().with(|instances| {
        with_instance(instances, handle, |instance| {
            instance.commit_accepted(records)
        })
    })
}

pub fn cancel<T: Bundle>(handle: BundleHandleV1) -> BundleStatusV1 {
    T::instances().with(|instances| with_instance(instances, handle, T::cancel))
}

pub fn retire_request<T: Bundle>(handle: BundleHandleV1, request_id: u64) -> BundleStatusV1 {
    T::instances().with(|instances| {
        with_instance(instances, handle, |instance| {
            instance.retire_request(request_id)
        })
    })
}

pub fn destroy<T: Bundle>(handle: BundleHandleV1) -> BundleStatusV1 {
    T::instances().with(|instances| {
        let Ok(mut table) = instances.try_borrow_mut() else {
            return BundleStatusV1::InvalidState;
        };
        sweep(&mut table);
        // A poisoned instance keeps its resources: releasing them could wait
        // on streams stuck at a collective.
        let instance = match usable(&mut table, handle) {
            Ok(instance) => instance,
            Err(status) => return status,
        };
        if let Err(error) = instance.release_graphs() {
            return status(error);
        }
        let (services, held) = instance.resources();
        if let Err(error) = release_all(services, held) {
            return status(error);
        }
        table.remove(handle);
        BundleStatusV1::Success
    })
}

pub fn describe_prefix_state<T: Bundle + PrefixInstance>(
    handle: BundleHandleV1,
) -> BundlePrefixStateInfoV1 {
    T::instances().with(|instances| prefix::describe(instances, handle))
}

pub fn capture_prefix_state<T: Bundle + PrefixInstance>(
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    T::instances().with(|instances| prefix::capture(instances, handle, request))
}

pub fn restore_prefix_state<T: Bundle + PrefixInstance>(
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    T::instances().with(|instances| prefix::restore(instances, handle, request))
}

pub fn restore_prefix_state_in_slot<T: Bundle + PrefixInstance>(
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
    slot: u64,
) -> BundleStatusV1 {
    T::instances().with(|instances| prefix::restore_in_slot(instances, handle, request, slot))
}
