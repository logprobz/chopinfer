//! Host resources a bundle instance holds, released in reverse order.

use runtime_bundle_abi::{
    BundleAbiError, BundleAuthenticatedComponentHandleV1, BundleCompiledImageHandleV1,
    BundleContextHandleV1, BundleCudaModuleHandleV1, BundleCudaSymbolHandleV1,
    BundleDeviceBufferHandleV1, BundleEventHandleV1, BundleMemoryPoolHandleV1,
    BundlePeerAccessHandleV1, BundleRankArenaHandleV1, BundleStateHandleV1, BundleStatusV1,
    BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};

/// One host resource to release, or a pool peer grant to revoke.
#[derive(Clone, Copy, Debug)]
pub enum Held {
    Image(BundleCompiledImageHandleV1),
    Component(BundleAuthenticatedComponentHandleV1),
    Context(BundleContextHandleV1),
    Pool(BundleMemoryPoolHandleV1),
    Stream(BundleStreamHandleV1),
    Event(BundleEventHandleV1),
    Arena(BundleRankArenaHandleV1),
    Buffer(BundleDeviceBufferHandleV1),
    Peer(BundlePeerAccessHandleV1),
    PoolPeer(BundleMemoryPoolHandleV1, BundleContextHandleV1),
    Module(BundleCudaModuleHandleV1),
    Symbol(BundleCudaSymbolHandleV1),
    State(BundleStateHandleV1),
}

pub fn release(services: &HostServicesV1, held: Held) -> Result<(), BundleAbiError> {
    match held {
        Held::Image(value) => services.release_compiled_image(value),
        Held::Component(value) => services.release_authenticated_component(value),
        Held::Context(value) => services.release_context(value),
        Held::Pool(value) => services.release_memory_pool(value),
        Held::Stream(value) => services.release_stream(value),
        Held::Event(value) => services.release_event(value),
        Held::Arena(value) => services.release_rank_arena(value),
        Held::Buffer(value) => services.release_device_buffer(value),
        Held::Peer(value) => services.release_peer_access(value),
        Held::PoolPeer(pool, peer) => services.set_memory_pool_peer_access(pool, peer, false),
        Held::Module(value) => services.release_cuda_module(value),
        Held::Symbol(value) => services.release_cuda_symbol(value),
        Held::State(value) => services.release_state(value),
    }
}

/// Releases `held` newest first. A failure leaves the failed resource and
/// everything older in `held`, so a later sweep can retry.
pub fn release_all(services: &HostServicesV1, held: &mut Vec<Held>) -> Result<(), BundleAbiError> {
    while let Some(value) = held.last().copied() {
        release(services, value)?;
        held.pop();
    }
    Ok(())
}

thread_local! {
    /// Bytes [`allocate_span`] allocated on this thread, by pool.
    static ALLOCATED: std::cell::RefCell<std::collections::BTreeMap<u64, u64>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// The bytes [`allocate_span`] has allocated from `pool` on this thread so
/// far, released buffers included. A bundle takes the difference across a
/// step of its create to attribute that step's device memory to each rank.
pub fn allocated_bytes(pool: BundleMemoryPoolHandleV1) -> u64 {
    ALLOCATED.with(|allocated| allocated.borrow().get(&pool.raw).copied().unwrap_or(0))
}

fn tally(pool: BundleMemoryPoolHandleV1, bytes: u64) {
    ALLOCATED.with(|allocated| {
        let mut allocated = allocated.borrow_mut();
        let total = allocated.entry(pool.raw).or_default();
        *total = total.saturating_add(bytes);
    });
}

/// Allocates one 256-byte aligned device buffer and records it in `held`.
pub fn allocate_span(
    services: &HostServicesV1,
    pool: BundleMemoryPoolHandleV1,
    stream: BundleStreamHandleV1,
    bytes: u64,
    held: &mut Vec<Held>,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    let buffer = services
        .allocate_device_buffer(pool, stream, bytes, 256)
        .inspect_err(|error| {
            // The host's status alone does not say which buffer it refused.
            let _ = services
                .log(format!("allocate_device_buffer of {bytes} bytes failed: {error}").as_bytes());
        })?;
    held.push(Held::Buffer(buffer));
    tally(pool, bytes);
    Ok(DeviceBufferSpanV1::new(buffer, 0, bytes))
}

/// Copies `bytes` to the start of `buffer`, refusing more than it holds.
pub fn upload_bytes(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    buffer: DeviceBufferSpanV1,
    bytes: &[u8],
) -> Result<(), BundleAbiError> {
    if bytes.len() as u64 > buffer.length {
        return Err(BundleAbiError::InvalidRange(
            "input exceeds its prepared buffer",
        ));
    }
    services.copy_host_to_device(
        stream,
        DeviceBufferSpanV1::new(buffer.buffer, buffer.offset, bytes.len() as u64),
        bytes,
    )
}

/// The ABI status a host service error reports.
pub fn status(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        BundleAbiError::Status(value) => value,
        BundleAbiError::MissingHostService(_) => BundleStatusV1::Unsupported,
        _ => BundleStatusV1::HostError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocated_bytes_accumulate_per_pool() {
        let [first, second] = [
            BundleMemoryPoolHandleV1::from_parts(1, 41),
            BundleMemoryPoolHandleV1::from_parts(1, 42),
        ];
        let before = [allocated_bytes(first), allocated_bytes(second)];
        tally(first, 256);
        tally(first, 1);
        tally(second, 7);
        assert_eq!(allocated_bytes(first) - before[0], 257);
        assert_eq!(allocated_bytes(second) - before[1], 7);
        assert_eq!(
            allocated_bytes(BundleMemoryPoolHandleV1::from_parts(1, 43)),
            0
        );
    }
}
