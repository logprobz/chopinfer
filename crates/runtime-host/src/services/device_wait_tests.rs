//! What a traced step counts as device waits when a bundle blocks on the
//! device through the host services, called through the C ABI table a bundle
//! holds. The fake device runs each launched graph for an emulated time, so a
//! host wait blocks for as long as a device would hold it. The step trace's
//! `boundary_us` runs from the previous step's last device wait ending to
//! the step's first one starting; these tests compute it the same way.

use super::*;
use crate::fake::FakeHostDeviceV1;
use runtime_bundle_abi::{
    BundleEventHandleV1, BundleGraphHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    HostServicesTableV1, HostServicesV1,
};
use runtime_engine::profile::{self, DeviceWaits};
use std::rc::Rc;
use std::time::{Duration, Instant};

/// How long a target graph runs.
const TARGET: Duration = Duration::from_millis(40);
/// How long a successor drafter graph runs.
const SUCCESSOR: Duration = Duration::from_millis(5);
/// Host work between two steps: commit, event delivery and scheduling.
const BETWEEN: Duration = Duration::from_millis(1);
/// Bytes of one rank's saved status.
const STATUS: u64 = 16;

type Backend = Rc<HostServicesV1Backend<FakeHostDeviceV1>>;

struct Rank {
    context: BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    status: DeviceBufferSpanV1,
    ready: BundleEventHandleV1,
    target: BundleGraphHandleV1,
    successor: BundleGraphHandleV1,
}

fn backend(ranks: usize) -> Backend {
    let topology = HostTopologyV1::new((0..ranks as i32).collect()).unwrap();
    Rc::new(HostServicesV1Backend::new(
        FakeHostDeviceV1::new(),
        HostInstanceConfigV1::new(topology),
    ))
}

/// Each rank's stream, status, boundary event and two graphs; rank zero also
/// holds the boundary packet every rank's status is gathered into.
fn ranks(host: &HostServicesV1, count: usize) -> (Vec<Rank>, DeviceBufferSpanV1) {
    let contexts = (0..count)
        .map(|rank| host.select_context(rank as u32, rank as i32).unwrap())
        .collect::<Vec<_>>();
    let pools = contexts
        .iter()
        .map(|&context| host.select_memory_pool(context).unwrap())
        .collect::<Vec<_>>();
    // Rank zero's stream pulls every other rank's status.
    for rank in 1..count {
        host.acquire_peer_access(contexts[0], contexts[rank])
            .unwrap();
        host.set_memory_pool_peer_access(pools[rank], contexts[0], true)
            .unwrap();
    }
    let mut ranks = Vec::new();
    let mut packet = None;
    for (rank, (&context, &pool)) in contexts.iter().zip(&pools).enumerate() {
        let stream = host.create_stream(context, 0).unwrap();
        let status = host
            .allocate_device_buffer(pool, stream, STATUS, 16)
            .unwrap();
        host.copy_host_to_device(
            stream,
            DeviceBufferSpanV1::new(status, 0, STATUS),
            &[rank as u8 + 1; STATUS as usize],
        )
        .unwrap();
        if rank == 0 {
            let bytes = STATUS * count as u64;
            let buffer = host
                .allocate_device_buffer(pool, stream, bytes, 16)
                .unwrap();
            packet = Some(DeviceBufferSpanV1::new(buffer, 0, bytes));
        }
        let graph = || {
            host.begin_graph_capture(stream).unwrap();
            host.end_graph_capture(stream).unwrap()
        };
        ranks.push(Rank {
            context,
            stream,
            status: DeviceBufferSpanV1::new(status, 0, STATUS),
            ready: host.create_event(context).unwrap(),
            target: graph(),
            successor: graph(),
        });
    }
    (ranks, packet.unwrap())
}

fn launch_time(backend: &Backend, time: Duration) {
    backend.with_device_mut(|device| device.set_launch_time(time));
}

/// One step's device waits and when its blocking readback was called and
/// returned.
struct Step {
    waits: DeviceWaits,
    readback: (Instant, Instant),
    launched: Instant,
}

/// A multi-rank drafting decode step whose main wait is one blocking
/// readback: every rank runs the target, rank zero joins each rank's saved
/// status into one packet and reads it back with one blocking copy, then
/// every rank queues the successor drafter, and the step's results are
/// collected without further waiting.
fn boundary_step(
    backend: &Backend,
    host: &HostServicesV1,
    ranks: &[Rank],
    packet: DeviceBufferSpanV1,
) -> Step {
    let step = profile::TraceTotals::begin();
    launch_time(backend, TARGET);
    let launched = Instant::now();
    for rank in ranks {
        host.launch_graph(rank.stream, rank.target).unwrap();
    }
    for rank in ranks {
        host.record_event(rank.ready, rank.stream).unwrap();
    }
    let lead = &ranks[0];
    for rank in &ranks[1..] {
        host.wait_event(lead.stream, rank.ready).unwrap();
    }
    for (index, rank) in ranks.iter().enumerate() {
        let destination = DeviceBufferSpanV1::new(packet.buffer, index as u64 * STATUS, STATUS);
        if index == 0 {
            host.copy_device_to_device(lead.stream, destination, rank.status)
        } else {
            host.copy_peer_to_peer(
                lead.stream,
                lead.context,
                destination,
                rank.context,
                rank.status,
            )
        }
        .unwrap();
    }
    let mut output = vec![0; packet.length as usize];
    let called = Instant::now();
    host.copy_device_to_host(lead.stream, &mut output, packet)
        .unwrap();
    let returned = Instant::now();
    let expected = (0..ranks.len())
        .flat_map(|rank| [rank as u8 + 1; STATUS as usize])
        .collect::<Vec<_>>();
    assert_eq!(output, expected, "the packet holds every rank's status");
    launch_time(backend, SUCCESSOR);
    host.record_event(lead.ready, lead.stream).unwrap();
    for rank in ranks {
        host.launch_graph(rank.stream, rank.successor).unwrap();
    }
    host.synchronize_event(lead.ready).unwrap();
    let waits = profile::device_waits();
    drop(step);
    Step {
        waits,
        readback: (called, returned),
        launched,
    }
}

/// `boundary_us` as the step trace computes it.
fn boundary(previous: &DeviceWaits, next: &DeviceWaits) -> Duration {
    next.first_started
        .unwrap()
        .saturating_duration_since(previous.last_ended.unwrap())
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

#[test]
fn a_gathered_boundary_readback_is_the_steps_device_wait() {
    let backend = backend(4);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let (ranks, packet) = ranks(host, 4);
    let first = boundary_step(&backend, host, &ranks, packet);
    std::thread::sleep(BETWEEN);
    let second = boundary_step(&backend, host, &ranks, packet);
    for step in [&first, &second] {
        let (called, returned) = step.readback;
        // The readback blocks until every rank's target has finished, and
        // that time is the step's device wait. Its window opens at the
        // readback, not at the step's final collection.
        let started = step.waits.first_started.expect("the readback waited");
        assert!(called <= started && started <= returned);
        let remaining = (step.launched + TARGET).saturating_duration_since(started);
        assert!(!remaining.is_zero() && returned - started >= remaining);
        assert!(step.waits.micros >= micros(remaining));
        assert!(step.waits.micros <= micros(returned - called) + 1_000);
    }
    // Between the steps the device was idle at most from the first step's
    // collection to the second step's readback: the host work between them
    // and the second step's launches, never the target it then ran.
    let between = boundary(&first.waits, &second.waits);
    assert!(between >= BETWEEN);
    assert!(between < second.readback.1 - second.launched);
    assert!(between < TARGET);
}

/// A step that reads its tokens back with a blocking copy, as a drafting
/// bundle reading back its proposals does, counts that copy as its device
/// wait. One that synchronizes first counts the wait once: the readback after
/// it adds only its own copy.
#[test]
fn a_token_readback_counts_once_with_or_without_a_synchronization_first() {
    for synchronize in [false, true] {
        let backend = backend(1);
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let host = table.caller();
        let (ranks, packet) = ranks(host, 1);
        let rank = &ranks[0];
        let mut steps = Vec::new();
        for _ in 0..2 {
            let step = profile::TraceTotals::begin();
            launch_time(&backend, TARGET);
            let launched = Instant::now();
            host.launch_graph(rank.stream, rank.target).unwrap();
            if synchronize {
                host.record_event(rank.ready, rank.stream).unwrap();
                host.synchronize_event(rank.ready).unwrap();
            }
            let mut tokens = vec![0; packet.length as usize];
            host.copy_device_to_host(rank.stream, &mut tokens, packet)
                .unwrap();
            let returned = Instant::now();
            let waits = profile::device_waits();
            drop(step);
            let started = waits.first_started.expect("the step waited");
            assert!(started >= launched && started < launched + TARGET);
            assert!(waits.micros >= micros(launched + TARGET - started));
            assert!(waits.micros <= micros(returned - launched));
            steps.push(waits);
            std::thread::sleep(BETWEEN);
        }
        let between = boundary(&steps[0], &steps[1]);
        assert!(between >= BETWEEN && between < TARGET, "{between:?}");
    }
}

/// Outside a traced step a blocking readback still blocks and records
/// nothing.
#[test]
fn an_untraced_readback_records_nothing() {
    let backend = backend(1);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let (ranks, packet) = ranks(host, 1);
    let step = profile::TraceTotals::begin();
    drop(step);
    let before = profile::device_waits();
    launch_time(&backend, SUCCESSOR);
    let launched = Instant::now();
    host.launch_graph(ranks[0].stream, ranks[0].target).unwrap();
    let mut bytes = vec![0; packet.length as usize];
    host.copy_device_to_host(ranks[0].stream, &mut bytes, packet)
        .unwrap();
    assert!(launched.elapsed() >= SUCCESSOR);
    assert_eq!(profile::device_waits(), before);
}

/// With step overlap (`CHOPINFER_STEP_OVERLAP=1`) the events queued after a
/// step are sent at the next blocking wait after a launch. In a boundary step
/// that is the gathered readback, while the target still runs, rather than
/// the final collection after the successor drafter is queued; the step's
/// wait window opens where the deferred work starts.
#[test]
fn deferred_delivery_runs_at_the_readback_while_the_target_runs() {
    let backend = backend(4);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let (ranks, packet) = ranks(host, 4);
    let delivered = Rc::new(std::cell::Cell::new(None));
    let job = Rc::clone(&delivered);
    crate::overlap::defer(move || job.set(Some(Instant::now())));
    let step = boundary_step(&backend, host, &ranks, packet);
    assert_eq!(crate::overlap::pending(), 0);
    let delivered = delivered.get().expect("the step delivered the queued work");
    let (called, returned) = step.readback;
    assert!(called <= delivered && delivered < returned);
    assert!(
        delivered < step.launched + TARGET,
        "the target was still running"
    );
    let started = step.waits.first_started.unwrap();
    assert!(called <= started && started <= delivered);
}
