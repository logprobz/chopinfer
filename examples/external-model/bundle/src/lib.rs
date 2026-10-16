//! A CPU-only stateful bundle. It exercises the ABI, not model quality.
#![deny(unsafe_code)]
use example_contract as model;
use runtime_bundle_abi::{
    BundleContextHandleV1, BundleCreateResultV1, BundleHandleV1, BundleStatusV1 as Status,
    ExecuteResultSinkV1, HostServicesV1,
};
use runtime_bundle_sdk::{
    BatchCodecV1, BatchReaderV1, StreamTransactionPhaseV1, StreamTransactionV1,
    authenticated_component_v1,
};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[cfg(feature = "precision-exercise")]
mod precision_note;
#[cfg(feature = "precision-exercise")]
pub use example_contract::precision as precision_contract;
#[cfg(feature = "precision-exercise")]
pub mod precision;
#[cfg(feature = "precision-exercise")]
#[path = "../precision_program.rs"]
pub mod precision_program;
// The target's kernel selections and implementations; nothing outside it
// names them.
#[cfg(any(feature = "cuda-probe", feature = "precision-exercise"))]
mod sm120;
// The precision package's weights are a planned component: the host packs
// them from the checkpoint by this plan (see `example_contract::weight_plan`).
#[cfg(feature = "precision-exercise")]
runtime_bundle_abi::export_weight_plan_v1!(example_contract::weight_plan::plan);

// The chat frontend: the model's own renderer (`example_contract::frontend`)
// and the shared parsers, over this package's frontend component
// (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(example_contract::frontend::ExampleFormat);

/// The kernels an instance prepares the example's ops on. The target's
/// module supplies them (`Kernels::new` in `sm120`).
#[cfg(feature = "cuda-probe")]
struct Kernels {
    backend: Box<dyn kernel_core::backend::Backend>,
    own: Box<dyn model::ops::OwnOps>,
}

struct Pending {
    states: BTreeMap<u64, u64>,
    streams: StreamTransactionV1<()>,
}
struct Instance {
    services: HostServicesV1,
    context: BundleContextHandleV1,
    seed: u64,
    states: BTreeMap<u64, u64>,
    pending: Option<Pending>,
}
#[derive(Default)]
struct Instances {
    next: u32,
    live: BTreeMap<u64, Instance>,
}
thread_local! { static INSTANCES: RefCell<Instances> = RefCell::new(Instances::default()); }

fn configuration(bytes: &[u8]) -> Result<i32, Status> {
    let mut r = BatchReaderV1::new("external example configuration", bytes);
    let bad = |_| Status::InvalidArgument;
    fn id<'a>(r: &mut BatchReaderV1<'a>) -> Result<&'a [u8], Status> {
        let n = r.count("identifier").map_err(|_| Status::InvalidArgument)?;
        r.take(n).map_err(|_| Status::InvalidArgument)
    }
    r.expect_magic(b"chopbundle-config-v1").map_err(bad)?;
    if id(&mut r)? != model::TOPOLOGY.as_bytes() || r.u32().map_err(bad)? != 1 {
        return Err(Status::Unsupported);
    }
    let ordinal = r.u32().map_err(bad)? as i32;
    if id(&mut r)? != model::KV.as_bytes() {
        return Err(Status::Unsupported);
    }
    for expected in [32, 32, 8, 1, 0] {
        if r.u64().map_err(bad)? != expected {
            return Err(Status::Unsupported);
        }
    }
    if id(&mut r)? != model::STATE.as_bytes() {
        return Err(Status::Unsupported);
    }
    for expected in [64, 64] {
        if r.u64().map_err(bad)? != expected {
            return Err(Status::Unsupported);
        }
    }
    for _ in 0..r.count("roles").map_err(bad)? {
        id(&mut r)?;
    }
    if r.remaining() != 0 {
        let items = r.u32().map_err(bad)?;
        if items == 0 || items > model::ITEMS {
            return Err(Status::Unsupported);
        }
    }
    r.finish().map_err(bad)?;
    Ok(ordinal)
}

fn create(services: HostServicesV1, bytes: &[u8]) -> BundleCreateResultV1 {
    #[cfg(feature = "omit-retirement")]
    if let Some(path) = std::env::var_os("EXTERNAL_MODEL_CREATE_MARKER") {
        if std::fs::write(path, b"created").is_err() {
            return BundleCreateResultV1::failure(Status::HostError);
        }
    }
    let result = (|| {
        let ordinal = configuration(bytes)?;
        // The synthetic program launches no kernel, so it keeps no prepared
        // op; preparing them refuses kernels that cannot serve the example.
        #[cfg(feature = "cuda-probe")]
        {
            let kernels = Kernels::new();
            model::ops::Ops::prepare(&*kernels.backend, &*kernels.own)
                .map_err(|_| Status::Unsupported)?;
        }
        // The host opens only the authenticated component selected by its role.
        let seed = authenticated_component_v1(&services, model::SEED_ROLE.as_bytes())
            .map_err(|_| Status::HostError)?;
        let seed = u64::from_le_bytes(
            seed.bytes()
                .try_into()
                .map_err(|_| Status::InvalidArgument)?,
        );
        INSTANCES.with(|cell| {
            let mut instances = cell.try_borrow_mut().map_err(|_| Status::InvalidState)?;
            let next = instances
                .next
                .checked_add(1)
                .ok_or(Status::ResourceExhausted)?;
            let context = services
                .select_context(0, ordinal)
                .map_err(|_| Status::HostError)?;
            let handle = BundleHandleV1::from_parts(next, 1);
            instances.next = next;
            instances.live.insert(
                handle.raw,
                Instance {
                    services,
                    context,
                    seed,
                    states: BTreeMap::new(),
                    pending: None,
                },
            );
            Ok(handle)
        })
    })();
    match result {
        Ok(handle) => BundleCreateResultV1::success(handle),
        Err(status) => BundleCreateResultV1::failure(status),
    }
}

fn with_instance(
    handle: BundleHandleV1,
    f: impl FnOnce(&mut Instance) -> Result<(), Status>,
) -> Status {
    let result = INSTANCES.with(|cell| {
        let mut instances = cell.try_borrow_mut().map_err(|_| Status::InvalidState)?;
        f(instances
            .live
            .get_mut(&handle.raw)
            .ok_or(Status::StaleHandle)?)
    });
    result.err().unwrap_or(Status::Success)
}

fn execute(handle: BundleHandleV1, bytes: &[u8], sink: &mut ExecuteResultSinkV1) -> Status {
    with_instance(handle, |instance| {
        if instance.pending.is_some() {
            return Err(Status::InvalidState);
        }
        let batch = BatchCodecV1::decode(bytes).map_err(|_| Status::InvalidArgument)?;
        if batch.items().is_empty() || batch.items().len() > model::ITEMS as usize {
            return Err(Status::Unsupported);
        }
        let mut next = instance.states.clone();
        let mut rows = Vec::new();
        for item in batch.items() {
            if item.is_verify() || item.bundle_drafts() || item.input_tokens().len() > 512 {
                return Err(Status::Unsupported);
            }
            let state = next.entry(item.request_id()).or_insert(instance.seed);
            for &token in item.input_tokens() {
                if token as usize >= model::VOCAB {
                    return Err(Status::InvalidArgument);
                }
                *state = model::fold(*state, token);
                if item.prompt_rows() {
                    rows.push((item.request_id(), model::logits(*state)));
                }
            }
            if item.expects_logits() && !item.prompt_rows() {
                rows.push((item.request_id(), model::logits(*state)));
            }
        }
        if next.len() > model::ITEMS as usize {
            return Err(Status::ResourceExhausted);
        }
        let streams = StreamTransactionV1::acquire(
            0,
            &[0],
            vec![(0, ())],
            |_, kind| instance.services.create_stream(instance.context, kind),
            |_, stream| instance.services.release_stream(stream),
        )
        .map_err(|_| Status::HostError)?;
        instance.pending = Some(Pending {
            states: next,
            streams,
        });
        for (request, row) in rows {
            sink.emit(request, &row).map_err(|_| Status::HostError)?;
        }
        Ok(())
    })
}

fn finish(instance: &mut Instance, commit: bool) -> Result<(), Status> {
    let pending = instance.pending.as_mut().ok_or(Status::InvalidState)?;
    let sync = |_, stream| instance.services.synchronize_stream(stream);
    let release = |_, stream| instance.services.release_stream(stream);
    if pending.streams.evidence().phase == StreamTransactionPhaseV1::Quarantined {
        pending.streams.retry_close(sync, release)
    } else {
        pending.streams.close(sync, release)
    }
    .map_err(|_| Status::HostError)?;
    let pending = instance.pending.take().ok_or(Status::InvalidState)?;
    if commit {
        instance.states = pending.states;
    }
    Ok(())
}
fn commit(handle: BundleHandleV1) -> Status {
    with_instance(handle, |i| finish(i, true))
}
fn cancel(handle: BundleHandleV1) -> Status {
    with_instance(handle, |i| finish(i, false))
}
fn destroy(handle: BundleHandleV1) -> Status {
    let status = with_instance(handle, |i| {
        if i.pending.is_some() {
            finish(i, false)?;
        }
        i.services
            .release_context(i.context)
            .map_err(|_| Status::HostError)
    });
    if status.is_success() {
        INSTANCES.with(|t| {
            t.borrow_mut().live.remove(&handle.raw);
        });
    }
    status
}
#[cfg(not(feature = "omit-retirement"))]
#[cfg(not(feature = "omit-retirement"))]
fn retire(handle: BundleHandleV1, request: u64) -> Status {
    with_instance(handle, |i| {
        if i.pending.is_some() {
            return Err(Status::InvalidState);
        }
        i.states.remove(&request);
        Ok(())
    })
}
#[cfg(not(feature = "omit-retirement"))]
runtime_bundle_abi::export_bundle_api_v1!(with_request_retirement API, EXTENDED,
    create, execute, commit, cancel, destroy, retire);
#[cfg(feature = "omit-retirement")]
runtime_bundle_abi::export_bundle_api_v1!(API, create, execute, commit, cancel, destroy);

#[cfg(all(test, feature = "sm120-helpers"))]
mod tests {
    #[test]
    fn shared_index_parser_rejects_unaligned_images() {
        let bytes = br#"{"schema":"example-index-v1","rank":0,"image":"rank0.img","image_bytes":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","views":[]}"#;
        assert!(bundle_runtime::index::parse(bytes, 0, "example-index-v1").is_err());
    }
}
