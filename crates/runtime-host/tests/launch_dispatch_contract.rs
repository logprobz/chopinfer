//! How a launch's flags decide the way a backend dispatches it.
//!
//! A launch attribute is not decoration: a kernel launched without the
//! attribute its route asked for behaves differently from one launched with it.
//! The decision therefore lives in one model-neutral place, every backend reads
//! it from there, and a bit no version defines is refused rather than dropped.
//!
//! The CUDA arm of that decision needs a device, so what is checked here is the
//! decision itself and the backend that can run anywhere. The repository
//! contract holds the CUDA backend to routing both arms.

use runtime_bundle_abi::{
    BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION, BUNDLE_LAUNCH_FLAGS_V1_KNOWN,
    BundleStatusV1, LaunchClusterV1, LaunchConfigV1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{HostDeviceBackendV1, LaunchDispatchV1, launch_cluster_v1, launch_dispatch_v1};

const MODULE: &[u8] = b"fake-module-image";
const SYMBOL: &[u8] = b"fake_entry_v1";

fn config(flags: u32) -> LaunchConfigV1 {
    LaunchConfigV1 {
        flags,
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        block_x: 1,
        block_y: 1,
        block_z: 1,
        shared_memory_bytes: 0,
        cluster: LaunchClusterV1::NONE,
    }
}

/// One launch over a grid that divides into whole clusters of `cluster`.
fn clustered(cluster: LaunchClusterV1) -> LaunchConfigV1 {
    LaunchConfigV1 {
        grid_x: 8,
        grid_y: 8,
        grid_z: 8,
        cluster,
        ..config(0)
    }
}

#[test]
fn an_unflagged_launch_is_plain_and_a_flagged_one_asks_for_the_attribute() {
    assert_eq!(
        launch_dispatch_v1(config(0)),
        Ok(LaunchDispatchV1::Plain),
        "a launch with no flags asks for no attribute"
    );
    assert_eq!(
        launch_dispatch_v1(config(
            BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION
        )),
        Ok(LaunchDispatchV1::ProgrammaticStreamSerialization)
    );
}

#[test]
fn a_flag_no_version_defines_is_refused_rather_than_dropped() {
    for unknown in [
        BUNDLE_LAUNCH_FLAGS_V1_KNOWN + 1,
        BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION | 0x8000_0000,
        u32::MAX,
    ] {
        assert_eq!(
            launch_dispatch_v1(config(unknown)),
            Err(BundleStatusV1::InvalidArgument),
            "launch flags {unknown:#010x} must be refused"
        );
    }
}

/// One backend with a context, a stream, a module, and its one symbol.
struct HarnessV1 {
    device: FakeHostDeviceV1,
    stream: runtime_host::DeviceObjectV1,
    symbol: runtime_host::DeviceObjectV1,
}

fn harness() -> HarnessV1 {
    let mut device = FakeHostDeviceV1::new();
    let context = device.create_context(0, 0).expect("a fake context");
    let stream = device.create_stream(context, 0).expect("a fake stream");
    let module = device.load_module(context, MODULE).expect("a fake module");
    let symbol = device
        .resolve_symbol(module, SYMBOL)
        .expect("a fake symbol");
    HarnessV1 {
        device,
        stream,
        symbol,
    }
}

#[test]
fn a_backend_records_the_dispatch_each_launch_asked_for() {
    let mut harness = harness();
    for (flags, expected) in [
        (0, LaunchDispatchV1::Plain),
        (
            BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
            LaunchDispatchV1::ProgrammaticStreamSerialization,
        ),
    ] {
        // This backend runs no kernel, so it refuses the launch after deciding
        // how a backend that could run one would have dispatched it.
        assert_eq!(
            harness
                .device
                .launch_symbol(harness.stream, harness.symbol, config(flags), &[]),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(
            harness.device.refused_launch_dispatches().last(),
            Some(&expected),
            "flags {flags:#010x} must dispatch as {expected:?}"
        );
    }
    assert_eq!(harness.device.refused_launches().len(), 2);
}

/// What a launch's cluster dimensions ask a backend for.
///
/// The one-block default asks for nothing, so it is not a cluster launch and
/// every backend runs it the way it always did. Any other shape is a cluster a
/// backend either performs exactly or refuses, and the decision of which shape
/// was asked for lives in one model-neutral place, next to the flag decision.
#[test]
fn the_one_block_default_asks_for_no_cluster_and_any_other_shape_is_exact() {
    assert_eq!(
        launch_cluster_v1(config(0)),
        Ok(None),
        "one block in every dimension is the launch that asks for no cluster"
    );
    assert_eq!(
        launch_cluster_v1(clustered(LaunchClusterV1 { x: 2, y: 2, z: 1 })),
        Ok(Some(LaunchClusterV1 { x: 2, y: 2, z: 1 }))
    );
}

/// A backend with no device performs no cluster, and says so.
///
/// Flattening a requested cluster into an ordinary launch would run a different
/// kernel program: a cluster launch is what a kernel's distributed shared
/// memory and cluster synchronization are defined against. So a backend that
/// cannot form one refuses the launch instead.
#[test]
fn a_backend_that_cannot_form_a_cluster_refuses_rather_than_flattening_it() {
    let mut harness = harness();
    assert_eq!(
        harness.device.launch_symbol(
            harness.stream,
            harness.symbol,
            clustered(LaunchClusterV1 { x: 2, y: 1, z: 1 }),
            &[],
        ),
        Err(BundleStatusV1::Unsupported),
        "a host-memory backend forms no cluster and must say so"
    );
    // Nothing was recorded as launched, so no caller can read this as a launch
    // that happened without the cluster it asked for.
    assert!(harness.device.refused_launches().is_empty());
}

/// A cluster shape no launch could take never reaches a backend at all.
#[test]
fn a_cluster_shape_the_grid_does_not_admit_is_an_invalid_argument() {
    let mut harness = harness();
    let mut ragged = clustered(LaunchClusterV1 { x: 3, y: 1, z: 1 });
    ragged.grid_x = 8;
    assert_eq!(
        launch_cluster_v1(ragged),
        Err(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(
        harness
            .device
            .launch_symbol(harness.stream, harness.symbol, ragged, &[]),
        Err(BundleStatusV1::InvalidArgument),
        "a grid that does not divide into whole clusters is not a launch"
    );
    assert!(harness.device.refused_launches().is_empty());
}

#[test]
fn an_undefined_flag_never_reaches_a_backend_s_launch_path() {
    let mut harness = harness();
    assert_eq!(
        harness.device.launch_symbol(
            harness.stream,
            harness.symbol,
            config(BUNDLE_LAUNCH_FLAGS_V1_KNOWN + 1),
            &[],
        ),
        Err(BundleStatusV1::InvalidArgument),
        "an undefined flag is an invalid argument, not an unsupported launch"
    );
    // The refusal came before the launch was recorded at all.
    assert!(harness.device.refused_launches().is_empty());
    assert!(harness.device.refused_launch_dispatches().is_empty());
}
