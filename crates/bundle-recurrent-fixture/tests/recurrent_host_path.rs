//! The recurrent execution path, driven end to end over host services.
//!
//! Nothing here reaches into the fixture: the host builds its own service table,
//! the fixture is entered through the same V1 API table its native module
//! exports, and the only things that cross are bytes and opaque handles. The
//! device work happens in host memory, so this proves the ownership, ordering,
//! transaction, and refusal behaviour of the path without a GPU.

use std::collections::BTreeMap;
use std::rc::Rc;

// Reuse the frontend's synthetic declaration for the stream decoder contract.
#[allow(dead_code)]
#[path = "../../frontend-format/tests/common/mod.rs"]
mod frontend_fixture;

use bundle_recurrent_fixture::{
    BUNDLE_RECURRENT_FIXTURE_API_V1, RECURRENT_FRONTEND_DECLARATION_V1, RECURRENT_FRONTEND_ROLE_V1,
    RECURRENT_GEOMETRY_ALIGNMENT_V1, RECURRENT_GEOMETRY_BYTES_V1, RECURRENT_GEOMETRY_OFFSET_V1,
    RECURRENT_GEOMETRY_VIEW_V1, RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_MAXIMUM_TOKENS_V1,
    RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1, RECURRENT_PROGRAM_IMAGE_BYTES_V1,
    RECURRENT_PROGRAM_ROLE_V1, RECURRENT_STATE_ALIGNMENT_V1, RECURRENT_STATE_BYTES_V1,
    RECURRENT_STATE_LAYOUT_ID_V1, RECURRENT_STATE_SEED_OFFSET_V1, RECURRENT_STATE_SEED_VIEW_V1,
    RECURRENT_STEP_OPERATION_ID_V1, RECURRENT_STEP_OPERATION_V1, RECURRENT_TOPOLOGY_ID_V1,
    RECURRENT_VOCABULARY_V1, destroy_v1, fold_tokens_v1, logits_row_v1, program_image_bytes_v1,
    recurrent_generic_copy_contract_v1, recurrent_kv_layout_v1, recurrent_topology_v1,
    state_seed_lanes_v1, unreleased_resource_count_v1,
};
use runtime_bundle::{
    GENERIC_COPY_F32_OPERATION_V1, GENERIC_COPY_F32_PRIMITIVE_V1, generic_copy_f32_bytes_v1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleApiV1, BundleInstanceV1, BundleStatusV1, HostServiceResultV1,
    HostServicesTableV1, LogitsRowSinkV1, OwnedLogitsRowV1, ResultSinkLimitsV1,
};
use runtime_host::fake::{FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{
    CompiledImageDescriptionV1, CompiledImageViewV1, CompiledRankImageV1, CompiledViewSpanV1,
    DeclaredKvContractV1, DeclaredStateContractV1, HostGenericCatalogV1, HostInstanceConfigV1,
    HostServicesV1Backend, HostTopologyV1,
};
use runtime_kernels::{KernelImplementationTierV1, KernelOpaqueIdV1, KernelResolverV1};

/// How the program component divides for the one rank this fixture serves.
fn program_image_description() -> CompiledImageDescriptionV1 {
    let mut views = BTreeMap::new();
    views.insert(
        RECURRENT_GEOMETRY_VIEW_V1.to_vec(),
        CompiledImageViewV1::new(
            RECURRENT_GEOMETRY_ALIGNMENT_V1,
            BTreeMap::from([(
                0,
                CompiledViewSpanV1 {
                    offset: RECURRENT_GEOMETRY_OFFSET_V1,
                    byte_length: RECURRENT_GEOMETRY_BYTES_V1,
                },
            )]),
        ),
    );
    views.insert(
        RECURRENT_STATE_SEED_VIEW_V1.to_vec(),
        CompiledImageViewV1::new(
            RECURRENT_STATE_ALIGNMENT_V1,
            BTreeMap::from([(
                0,
                CompiledViewSpanV1 {
                    offset: RECURRENT_STATE_SEED_OFFSET_V1,
                    byte_length: RECURRENT_STATE_BYTES_V1,
                },
            )]),
        ),
    );
    CompiledImageDescriptionV1::new(
        RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1,
        vec![CompiledRankImageV1 {
            offset: 0,
            byte_length: RECURRENT_PROGRAM_IMAGE_BYTES_V1,
        }],
        views,
    )
}

/// The declaration a host builds from this fixture's own manifest records.
fn instance_config() -> HostInstanceConfigV1 {
    let topology = HostTopologyV1::new(vec![0]).expect("the TP1 test topology is valid");
    let page_layout = recurrent_kv_layout_v1()
        .expect("the declared KV layout is valid")
        .declared_page_layout(&recurrent_topology_v1().expect("the declared topology is valid"))
        .expect("the declared KV layout describes this topology");
    HostInstanceConfigV1::new(topology)
        .with_kv_contract(DeclaredKvContractV1::new(
            RECURRENT_KV_LAYOUT_ID_V1.as_bytes().to_vec(),
            page_layout,
        ))
        .with_state_contract(DeclaredStateContractV1::new(
            RECURRENT_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            RECURRENT_STATE_BYTES_V1,
            RECURRENT_STATE_ALIGNMENT_V1,
        ))
        .with_compiled_image(
            RECURRENT_PROGRAM_ROLE_V1.as_bytes(),
            program_image_bytes_v1(),
            program_image_description(),
        )
        .expect("the program component is valid")
        .with_component(
            RECURRENT_FRONTEND_ROLE_V1.as_bytes(),
            RECURRENT_FRONTEND_DECLARATION_V1.as_bytes().to_vec(),
        )
        .expect("the frontend component is valid")
}

fn services() -> (
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    HostServicesTableV1,
    Vec<u8>,
) {
    let config = instance_config();
    let configuration = config.encode_v1(RECURRENT_TOPOLOGY_ID_V1.as_bytes());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    (backend, table, configuration)
}

fn api() -> BundleApiV1 {
    BundleApiV1::new(BUNDLE_RECURRENT_FIXTURE_API_V1).expect("the exported V1 table is valid")
}

fn created() -> (
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    BundleInstanceV1,
) {
    let (backend, table, configuration) = services();
    let instance = api()
        .create(table, &configuration)
        .expect("the fixture accepts the TP1 instance the host configured");
    (backend, instance)
}

/// One prefill of `tokens.len()` tokens, in the host's own wire form.
fn batch_request(request_id: u64, tokens: &[u32], decode: bool) -> Vec<u8> {
    let mut request = Vec::new();
    request.extend_from_slice(b"chopbatch-v1");
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&request_id.to_le_bytes());
    request.push(u8::from(decode));
    request.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    request.push(1);
    request.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        request.extend_from_slice(&token.to_le_bytes());
    }
    // One prepared append per item, which this fixture reads and appends to
    // nothing: its transaction is the recurrent state, not a page table.
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&0_u64.to_le_bytes());
    request.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    request.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&0_u64.to_le_bytes());
    request
}

/// A sink that would accept a row, so a refusal cannot be blamed on capacity.
struct CollectingSink {
    rows: Vec<OwnedLogitsRowV1>,
}

impl CollectingSink {
    fn new() -> Self {
        Self { rows: Vec::new() }
    }
}

impl LogitsRowSinkV1 for CollectingSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(4, 1 << 20, 1 << 22)
    }

    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.rows.push(row);
        Ok(())
    }
}

fn logged(backend: &HostServicesV1Backend<FakeHostDeviceV1>) -> Vec<String> {
    backend.with_device(|device| {
        device
            .logs()
            .iter()
            .map(|line| String::from_utf8_lossy(line).into_owned())
            .collect()
    })
}

/// Where one diagnostic first appears, so two of them can be ordered.
fn index_of(logs: &[String], needle: &str) -> usize {
    logs.iter()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no diagnostic mentions '{needle}': {logs:?}"))
}

fn one_row(sink: &CollectingSink) -> &OwnedLogitsRowV1 {
    let [row] = sink.rows.as_slice() else {
        panic!(
            "this step scheduled exactly one row, and got {}",
            sink.rows.len()
        );
    };
    row
}

/// Every resource this fixture takes is given back, newest first.
#[test]
fn one_instance_acquires_its_whole_topology_and_gives_every_resource_back() {
    let (backend, mut instance) = created();

    // One context, one pool, one stage stream, one compiled image, one rank
    // arena, three device buffers, and the persistent state.
    assert_eq!(backend.live_resource_count(), 9);
    let logs = logged(&backend);
    assert!(
        logs.iter().any(
            |line| line.contains("persistent state instead of a KV cache")
                && line.contains(RECURRENT_KV_LAYOUT_ID_V1)
        ),
        "the instance must say it allocates no KV pool: {logs:?}"
    );
    assert!(
        logs.iter()
            .any(|line| line.contains("64-wide vocabulary read from its own program image")),
        "the instance must report the width it read from the package: {logs:?}"
    );

    let handle = instance.handle();
    instance
        .destroy()
        .expect("destroy releases in reverse order");

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
    // The fixture's own generation check refuses the handle it just retired,
    // whatever the caller's state machine believes.
    assert_eq!(
        bundle_recurrent_fixture::commit_v1(handle),
        BundleStatusV1::StaleHandle
    );
    assert_eq!(
        bundle_recurrent_fixture::destroy_v1(handle),
        BundleStatusV1::StaleHandle
    );
}

/// One execution performs both declared operations, in the declared order, and
/// the row it emits is what the second one produced.
#[test]
fn one_execution_runs_both_declared_operations_in_order_and_emits_the_folded_row() {
    let (backend, mut instance) = created();
    let mut sink = CollectingSink::new();
    let tokens = [3_u32, 5, 8];

    instance
        .execute(&batch_request(7, &tokens, false), &mut sink)
        .expect("the fixture runs one recurrent step");

    let row = one_row(&sink);
    assert_eq!(row.request_id(), 7);
    assert_eq!(row.logit_count(), u64::from(RECURRENT_VOCABULARY_V1));
    let expected = logits_row_v1(
        &fold_tokens_v1(&state_seed_lanes_v1(), &tokens),
        RECURRENT_VOCABULARY_V1,
    );
    assert_eq!(row.logit_bits(), expected.as_slice());

    // The host ran the custom operation the manifest declares, over the two
    // operands the fixture named, and nothing else.
    let operations = backend.with_device(|device| device.custom_operations().to_vec());
    let [(id, lent)] = operations.as_slice() else {
        panic!(
            "one step runs one custom operation, and ran {}",
            operations.len()
        );
    };
    assert_eq!(id.as_slice(), RECURRENT_STEP_OPERATION_ID_V1);
    assert_eq!(lent.len(), 2);
    assert_eq!(lent[0].length, tokens.len() as u64 * 4);
    assert_eq!(lent[1].length, u64::from(RECURRENT_VOCABULARY_V1) * 4);

    // Both operations ran, and the generic copy ran after the step whose output
    // it moves. That order is what makes the emitted row the copy's result.
    let logs = logged(&backend);
    let custom = index_of(&logs, "ran bundle custom operation");
    let generic = index_of(&logs, "ran host generic operation");
    assert!(
        custom < generic,
        "the step must run before the copy that moves its output: {logs:?}"
    );
    assert!(logs[custom].contains(RECURRENT_STEP_OPERATION_V1));
    assert!(logs[generic].contains(GENERIC_COPY_F32_OPERATION_V1));
    assert!(
        logs[generic].contains(GENERIC_COPY_F32_PRIMITIVE_V1),
        "the copy must name the host primitive it went through"
    );

    // Nothing was launched: this fixture computes through its own operation and
    // the host's own copy, not through a kernel the host would have to launch.
    assert!(backend.with_device(|device| device.refused_launches().is_empty()));

    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// What this host resolves and what this module calls are one operation.
///
/// The manifest requires a primitive by name, the host catalog offers a
/// candidate for one exact contract, and the module calls a device-to-device
/// copy. Nothing in those three sentences forces them to be the same thing, so
/// this checks that they are: the resolved candidate is the one this host offers
/// under the primitive the fixture named, it moves exactly the bytes the
/// declared contract describes, and refusing that one service is what takes the
/// row away.
#[test]
fn the_resolved_generic_primitive_is_the_copy_service_this_module_calls() {
    let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");
    let declared =
        recurrent_generic_copy_contract_v1().expect("this fixture's declared copy is valid");
    let resolver = KernelResolverV1::new(catalog.candidates().expect("the catalog is offerable"))
        .expect("the catalog declares without conflict");

    let resolution = resolver
        .resolve(
            &declared
                .semantic_key()
                .expect("the declared copy has a key"),
        )
        .expect("this host serves the width this fixture requires");
    assert_eq!(resolution.tier(), KernelImplementationTierV1::Generic);
    assert_eq!(
        catalog
            .primitive_of(resolution.implementation_id())
            .map(KernelOpaqueIdV1::as_str),
        Some(GENERIC_COPY_F32_PRIMITIVE_V1),
        "the resolved candidate must be the one this host offers under the primitive the \
         manifest names"
    );

    let (backend, mut instance) = created();
    let mut sink = CollectingSink::new();
    instance
        .execute(&batch_request(21, &[1, 2, 3], false), &mut sink)
        .expect("the fixture runs one recurrent step");

    // The row a caller received is the destination of a copy of exactly the
    // width the resolved contract describes.
    assert_eq!(
        one_row(&sink).logit_bits().len() as u64,
        generic_copy_f32_bytes_v1(RECURRENT_VOCABULARY_V1)
    );
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains(GENERIC_COPY_F32_PRIMITIVE_V1)),
        "the executed copy must name the primitive that resolved: {:?}",
        logged(&backend)
    );
    instance.commit().expect("the step commits");

    // Refusing the one host service that performs it is what removes the row.
    // A module calling some other service would still emit one here.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::CopyDeviceToDevice));
    let mut refused = CollectingSink::new();
    instance
        .execute(&batch_request(22, &[1, 2, 3], false), &mut refused)
        .expect_err("a host that will not copy cannot produce this fixture's row");

    assert!(refused.rows.is_empty());
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A host that will not perform the declared copy emits no row at all.
///
/// The custom operation still runs, so the fixture has a complete row in hand.
/// It is the operation this package does not implement that decides whether a
/// caller ever sees one.
#[test]
fn a_refused_generic_copy_emits_no_row_and_leaves_nothing_pending() {
    let (backend, mut instance) = created();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::CopyDeviceToDevice));
    let mut sink = CollectingSink::new();

    let refusal = instance
        .execute(&batch_request(11, &[2, 4], false), &mut sink)
        .expect_err("a host with no copy cannot produce this fixture's row");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::HostError));
    assert!(sink.rows.is_empty(), "a refused copy emits no row");
    // The custom operation did run, so the refusal is the generic one and the
    // diagnostic says which operation and which primitive it was.
    assert_eq!(
        backend.with_device(|device| device.custom_operations().len()),
        1
    );
    assert!(
        logged(&backend).iter().any(|line| {
            line.contains("could not run host generic operation")
                && line.contains(GENERIC_COPY_F32_PRIMITIVE_V1)
        }),
        "the refusal must name the primitive the host did not provide: {:?}",
        logged(&backend)
    );

    // Nothing is pending, so the instance takes the next step once the host can
    // perform the copy again, and the state never advanced past the refusal.
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::CopyDeviceToDevice));
    instance
        .execute(&batch_request(12, &[2, 4], false), &mut sink)
        .expect("the instance runs once the host can copy again");
    assert_eq!(
        one_row(&sink).logit_bits(),
        logits_row_v1(
            &fold_tokens_v1(&state_seed_lanes_v1(), &[2, 4]),
            RECURRENT_VOCABULARY_V1
        )
        .as_slice(),
        "the refused step must have left the committed state untouched"
    );
    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// Commit advances the persistent state; cancel leaves it exactly as it was.
#[test]
fn repeated_transactions_advance_the_state_and_a_cancelled_one_rolls_back() {
    let (backend, mut instance) = created();
    let first = [1_u32, 2];
    let second = [9_u32];

    let mut sink = CollectingSink::new();
    instance
        .execute(&batch_request(1, &first, false), &mut sink)
        .expect("the first step runs");
    instance.commit().expect("the first step commits");

    // A committed step is the state the next one reads.
    let after_first = fold_tokens_v1(&state_seed_lanes_v1(), &first);
    let mut cancelled = CollectingSink::new();
    instance
        .execute(&batch_request(2, &second, true), &mut cancelled)
        .expect("the second step runs");
    assert_eq!(
        one_row(&cancelled).logit_bits(),
        logits_row_v1(
            &fold_tokens_v1(&after_first, &second),
            RECURRENT_VOCABULARY_V1
        )
        .as_slice()
    );
    instance.cancel().expect("the second step is cancelled");

    // The cancelled step changed nothing, so replaying it produces the same row.
    let mut replay = CollectingSink::new();
    instance
        .execute(&batch_request(3, &second, true), &mut replay)
        .expect("the replayed step runs");
    assert_eq!(
        one_row(&replay).logit_bits(),
        one_row(&cancelled).logit_bits(),
        "a cancelled step must leave the persistent state untouched"
    );
    instance.commit().expect("the replayed step commits");

    // This one was committed, so the step after it starts somewhere else.
    let mut third = CollectingSink::new();
    instance
        .execute(&batch_request(4, &second, true), &mut third)
        .expect("the third step runs");
    assert_ne!(
        one_row(&third).logit_bits(),
        one_row(&replay).logit_bits(),
        "a committed step must advance the persistent state"
    );
    assert_eq!(
        one_row(&third).logit_bits(),
        logits_row_v1(
            &fold_tokens_v1(&fold_tokens_v1(&after_first, &second), &second),
            RECURRENT_VOCABULARY_V1
        )
        .as_slice()
    );

    // A destroy needs the open step closed first, so the caller closes it.
    instance.cancel().expect("the third step is cancelled");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("rolled back the step of request 2")),
        "a cancel must say which step it rolled back"
    );
}

/// A destroy with a step still pending is refused, and rolls nothing back.
///
/// V1 allows at most one transaction per instance, and closing it is the
/// caller's call to make. A destroy that rolled one back would decide the
/// outcome of a transaction the caller still owns and report as released an
/// instance whose step nobody chose to abandon.
///
/// The safe wrapper refuses a pending destroy before the callback runs, so the
/// native entry is called directly here: this is a rule that entry owes on its
/// own, to whatever drives the C ABI without that wrapper. The refusal is
/// repeated, because a rule that only holds the first time is not one.
///
/// What proves nothing was rolled back is the commit afterwards: it succeeds,
/// and the state it publishes is the one that step computed, which a rollback
/// would have discarded.
#[test]
fn a_native_destroy_with_a_step_pending_is_refused_and_leaves_it_to_the_caller() {
    let (backend, mut instance) = created();
    let first = [4_u32, 5];
    let second = [7_u32];

    let mut sink = CollectingSink::new();
    instance
        .execute(&batch_request(1, &first, false), &mut sink)
        .expect("the step runs");
    let handle = instance.handle();

    for attempt in 0..2 {
        assert_eq!(
            destroy_v1(handle),
            BundleStatusV1::InvalidState,
            "attempt {attempt} must be refused while a step is pending"
        );
    }
    assert!(
        matches!(
            instance.destroy(),
            Err(BundleAbiError::Lifecycle("destroy"))
        ),
        "the wrapper refuses it too, before the callback"
    );
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("refused to destroy an instance with the step of request 1")),
        "a refused destroy must say what it refused"
    );

    // The step is still the caller's, and committing it publishes exactly what
    // it computed: a destroy that had rolled it back would leave the seed here.
    instance.commit().expect("the pending step still commits");
    let after_first = fold_tokens_v1(&state_seed_lanes_v1(), &first);
    let mut next = CollectingSink::new();
    instance
        .execute(&batch_request(2, &second, true), &mut next)
        .expect("the next step runs");
    assert_eq!(
        one_row(&next).logit_bits(),
        logits_row_v1(
            &fold_tokens_v1(&after_first, &second),
            RECURRENT_VOCABULARY_V1
        )
        .as_slice(),
        "the committed step is the state the next one reads"
    );

    instance.cancel().expect("the next step is cancelled");
    instance.destroy().expect("a closed instance destroys");
    assert_eq!(backend.live_resource_count(), 0);
}

/// The same refusal, closed with a cancel instead of a commit.
///
/// A cancel after a refused destroy has to reach the same step the destroy left
/// alone. What proves it did is the replay: the state is untouched, so the step
/// after the cancel produces the row the cancelled one produced.
#[test]
fn a_cancel_after_a_refused_destroy_reaches_the_step_that_destroy_left_open() {
    let (backend, mut instance) = created();
    let tokens = [2_u32, 3];

    let mut sink = CollectingSink::new();
    instance
        .execute(&batch_request(1, &tokens, false), &mut sink)
        .expect("the step runs");
    assert_eq!(destroy_v1(instance.handle()), BundleStatusV1::InvalidState);

    instance.cancel().expect("the pending step still cancels");
    let mut replay = CollectingSink::new();
    instance
        .execute(&batch_request(2, &tokens, false), &mut replay)
        .expect("the replayed step runs");
    assert_eq!(
        one_row(&replay).logit_bits(),
        one_row(&sink).logit_bits(),
        "a cancelled step must leave the persistent state untouched"
    );

    instance.cancel().expect("the replayed step is cancelled");
    instance.destroy().expect("a closed instance destroys");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A batch this fixture cannot run leaves nothing open and no row behind.
#[test]
fn a_batch_this_fixture_cannot_run_is_refused_with_nothing_left_pending() {
    let (backend, mut instance) = created();
    let before = backend.live_resource_count();
    let too_many = (0..=RECURRENT_MAXIMUM_TOKENS_V1).collect::<Vec<_>>();

    for (request, reason) in [
        (batch_request(1, &[], false), "no tokens"),
        (
            batch_request(2, &too_many, false),
            "more tokens than declared",
        ),
        (
            batch_request(3, &[1, 2], true),
            "a decode of more than one token",
        ),
        (b"chopbatch-v1".to_vec(), "a truncated request"),
    ] {
        let mut sink = CollectingSink::new();

        let refusal = instance
            .execute(&request, &mut sink)
            .expect_err(&format!("{reason} must be refused"));

        assert_eq!(
            refusal,
            BundleAbiError::Status(BundleStatusV1::InvalidArgument),
            "{reason}"
        );
        assert!(sink.rows.is_empty(), "{reason} must emit no row");
        assert_eq!(backend.live_resource_count(), before, "{reason}");
    }

    // A refused execution left no transaction open, so the instance still runs.
    let mut sink = CollectingSink::new();
    instance
        .execute(&batch_request(9, &[4], false), &mut sink)
        .expect("the instance is still usable after a refusal");
    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A host that refuses the custom operation gets no row and no open transaction.
#[test]
fn a_refused_custom_operation_produces_no_row_and_leaves_nothing_pending() {
    let (backend, mut instance) = created();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::RunCustomOperation));
    let mut sink = CollectingSink::new();

    let refusal = instance
        .execute(&batch_request(5, &[7], false), &mut sink)
        .expect_err("a refused operation cannot produce a row");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::HostError));
    assert!(sink.rows.is_empty());
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("custom operation produced no result")),
        "the refusal must say the operation produced nothing"
    );

    // Allowing it again makes the same instance run, so nothing was left open.
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::RunCustomOperation));
    instance
        .execute(&batch_request(6, &[7], false), &mut sink)
        .expect("the instance runs once the host allows the operation again");
    assert_eq!(sink.rows.len(), 1);
    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A refused release keeps exactly what the host still owns, and a retry
/// resumes from there rather than losing it.
#[test]
fn a_refused_release_is_retryable_and_retains_every_resource_until_it_succeeds() {
    let (backend, mut instance) = created();
    let acquired = backend.live_resource_count();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseStream));

    let refusal = instance
        .destroy()
        .expect_err("a host that will not take the stream back refuses the destroy");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::HostError));
    // The releases before the stream succeeded and are gone; the stream and the
    // two the host must still take after it are all the fixture is holding.
    assert_eq!(backend.live_resource_count(), 3, "acquired {acquired}");

    // The wrapper poisons after a refused destroy, so the retry names the
    // handle directly, which is what whoever owns the native handle would do.
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseStream));
    assert_eq!(
        bundle_recurrent_fixture::destroy_v1(instance.handle()),
        BundleStatusV1::Success
    );
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
}

/// A host this fixture does not serve costs it no resource at all.
#[test]
fn a_topology_this_fixture_does_not_serve_is_refused_before_anything_is_acquired() {
    let topology = HostTopologyV1::new(vec![0, 1]).expect("a two-rank topology is valid");
    let page_layout = recurrent_kv_layout_v1()
        .expect("the declared KV layout is valid")
        .declared_page_layout(&recurrent_topology_v1().expect("the declared topology is valid"))
        .expect("the declared KV layout describes this topology");
    let config = HostInstanceConfigV1::new(topology)
        .with_kv_contract(DeclaredKvContractV1::new(
            RECURRENT_KV_LAYOUT_ID_V1.as_bytes().to_vec(),
            page_layout,
        ))
        .with_state_contract(DeclaredStateContractV1::new(
            RECURRENT_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            RECURRENT_STATE_BYTES_V1,
            RECURRENT_STATE_ALIGNMENT_V1,
        ));
    let configuration = config.encode_v1(b"tp2");
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));

    let refusal = api()
        .create(table, &configuration)
        .err()
        .expect("this fixture serves one topology only");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::Unsupported));
    assert_eq!(backend.live_resource_count(), 0);
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("serves topology tp1")),
        "the refusal must name the topology this fixture serves"
    );
}

/// A package with no program component has nothing for this fixture to read.
#[test]
fn a_package_without_the_declared_program_component_is_refused() {
    let topology = HostTopologyV1::new(vec![0]).expect("the TP1 test topology is valid");
    let config =
        HostInstanceConfigV1::new(topology).with_state_contract(DeclaredStateContractV1::new(
            RECURRENT_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            RECURRENT_STATE_BYTES_V1,
            RECURRENT_STATE_ALIGNMENT_V1,
        ));
    let configuration = config.encode_v1(RECURRENT_TOPOLOGY_ID_V1.as_bytes());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));

    let refusal = api()
        .create(table, &configuration)
        .err()
        .expect("this fixture reads its geometry out of the program component");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::NotFound));
    assert_eq!(backend.live_resource_count(), 0);
}

mod request_state {
    use super::*;
    use runtime_bundle_abi::BundleKeptRowsV1;
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    fn created_requests() -> (
        Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
        BundleInstanceV1,
    ) {
        let (backend, table, configuration) = services();
        let api = BundleApiV1::new_with_speculation(
            bundle_recurrent_fixture::BUNDLE_RECURRENT_FIXTURE_API_WITH_SPECULATION_V1,
        )
        .unwrap();
        (backend, api.create(table, &configuration).unwrap())
    }

    fn batch(id: u64, slot: u64, previous: u64, tokens: &[u32], verify: bool) -> Vec<u8> {
        let item = if verify {
            BatchItemV1::verify(id, tokens.len() as u32, false, tokens.to_vec())
        } else {
            BatchItemV1::new(id, false, tokens.len() as u32, true, tokens.to_vec())
        };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            vec![BatchAppendV1::new(
                previous,
                tokens.len() as u64,
                previous + tokens.len() as u64,
                vec![0, 1, 2, 3],
                None,
            )],
        )
        .with_state_slots(vec![slot])
        .encode()
        .unwrap()
    }

    #[test]
    fn checkpoints_select_every_kept_prefix_and_cancel_preserves_committed_bytes() {
        for kept in 1..=3 {
            let (_backend, mut instance) = created_requests();
            let mut sink = CollectingSink::new();
            instance
                .execute(&batch(1, 0, 0, &[2, 3], false), &mut sink)
                .unwrap();
            instance.commit().unwrap();
            let committed = fold_tokens_v1(&state_seed_lanes_v1(), &[2, 3]);
            let mut verified = CollectingSink::new();
            instance
                .execute(&batch(1, 0, 2, &[4, 5, 6], true), &mut verified)
                .unwrap();
            assert_eq!(verified.rows.len(), 3);
            for (index, row) in verified.rows.iter().enumerate() {
                assert_eq!(
                    row.logit_bits(),
                    logits_row_v1(
                        &fold_tokens_v1(&committed, &[4, 5, 6][..=index]),
                        RECURRENT_VOCABULARY_V1
                    )
                );
            }
            instance.cancel().unwrap();
            let mut replay = CollectingSink::new();
            instance
                .execute(&batch(1, 0, 2, &[4, 5, 6], true), &mut replay)
                .unwrap();
            assert_eq!(verified.rows, replay.rows);
            instance
                .commit_accepted(&[BundleKeptRowsV1 {
                    struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                    reserved: 0,
                    request_id: 1,
                    kept_rows: kept as u32,
                    reserved_tail: 0,
                }])
                .unwrap();
            let mut next = CollectingSink::new();
            instance
                .execute(&batch(1, 0, 2 + kept as u64, &[7], false), &mut next)
                .unwrap();
            let expected = fold_tokens_v1(
                &fold_tokens_v1(&committed, &[4, 5, 6][..kept as usize]),
                &[7],
            );
            assert_eq!(
                next.rows[0].logit_bits(),
                logits_row_v1(&expected, RECURRENT_VOCABULARY_V1)
            );
            instance.cancel().unwrap();
            instance.retire_request(1).unwrap();
            let mut reused = CollectingSink::new();
            instance
                .execute(&batch(2, 0, 0, &[7], false), &mut reused)
                .unwrap();
            assert_eq!(
                reused.rows[0].logit_bits(),
                logits_row_v1(
                    &fold_tokens_v1(&state_seed_lanes_v1(), &[7]),
                    RECURRENT_VOCABULARY_V1
                )
            );
            instance.cancel().unwrap();
            instance.retire_request(2).unwrap();
            instance.destroy().unwrap();
        }
    }
    use runtime_engine::{
        DraftSource, Engine, EngineConfig, EngineEventKind, EngineRequest, ExecutorError,
        ModelBatchItem, ModelExecutor, ModelTransaction, SpeculationConfig, SpeculationMode,
        StepStatus,
    };
    use runtime_host::{
        BatchEmissionsV1, BatchWireCapabilityV1, BundleExecutionV1, BundleModelExecutorV1,
        BundleTransactionStateV1, DeviceLogitsSamplerV1,
    };
    use runtime_kv::{KvGeometry, PreparedAppendBatch};
    use runtime_model_api::ModelVocabulary;
    use runtime_sampling::SamplingParameters;
    use runtime_scheduler::{
        AdmissionPolicy, BatchPhase, OptimisticAdmission, RequestId, SchedulerConfig,
    };
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;

    struct Direct {
        instance: BundleInstanceV1,
        verifies: Rc<Cell<usize>>,
        slot_restores: Rc<Cell<usize>>,
    }

    struct BoundedSink {
        drafts: Vec<runtime_bundle_abi::OwnedDraftTokensV1>,
        rows: Vec<OwnedLogitsRowV1>,
        limits: ResultSinkLimitsV1,
    }
    impl LogitsRowSinkV1 for BoundedSink {
        fn limits(&self) -> ResultSinkLimitsV1 {
            self.limits
        }
        fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
            self.rows.push(row);
            Ok(())
        }
        fn emit_draft_tokens(
            &mut self,
            drafts: runtime_bundle_abi::OwnedDraftTokensV1,
        ) -> HostServiceResultV1<()> {
            self.drafts.push(drafts);
            Ok(())
        }
    }

    impl Direct {
        fn collect(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
        ) -> Result<BoundedSink, BundleAbiError> {
            let batch = BatchRequestV1::decode(request).unwrap();
            assert!(batch.state_slots().is_some());
            self.verifies.set(
                self.verifies.get() + batch.items().iter().filter(|item| item.is_verify()).count(),
            );
            let mut sink = BoundedSink {
                drafts: Vec::new(),
                rows: Vec::new(),
                limits,
            };
            self.instance.execute(request, &mut sink)?;
            Ok(sink)
        }
    }

    impl BundleExecutionV1 for Direct {
        fn execute_batch(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
        ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
            Ok(self.collect(request, limits)?.rows)
        }
        fn execute_with_sampling(
            &mut self,
            request: &[u8],
            limits: ResultSinkLimitsV1,
            _items: &[ModelBatchItem],
            _sampler: Option<Rc<dyn DeviceLogitsSamplerV1>>,
        ) -> Result<BatchEmissionsV1, BundleAbiError> {
            let sink = self.collect(request, limits)?;
            Ok(BatchEmissionsV1::from_rows_and_drafts(
                sink.rows,
                sink.drafts,
            ))
        }

        fn commit(&mut self) -> Result<(), BundleAbiError> {
            self.instance.commit()
        }
        fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
            self.instance.commit_accepted(records)
        }
        fn cancel(&mut self) -> Result<(), BundleAbiError> {
            self.instance.cancel()
        }
        fn destroy(&mut self) -> Result<(), BundleAbiError> {
            self.instance.destroy()
        }
        fn is_poisoned(&self) -> bool {
            self.instance.is_poisoned()
        }
        fn retire_request(&mut self, id: u64) -> Result<(), BundleAbiError> {
            self.instance.retire_request(id)
        }
        fn capture_prefix_state(
            &mut self,
            id: u64,
            tokens: u64,
            pages: &[u64],
        ) -> Result<(), BundleAbiError> {
            self.instance.capture_prefix_state(id, tokens, pages)
        }
        fn restore_prefix_state(
            &mut self,
            id: u64,
            tokens: &[u32],
            pages: &[u64],
        ) -> Result<(), BundleAbiError> {
            self.instance.restore_prefix_state(id, tokens, pages)
        }
        fn supports_prefix_state_slot(&self) -> bool {
            self.instance.supports_prefix_state_slot()
        }
        fn restore_prefix_state_in_slot(
            &mut self,
            id: u64,
            tokens: &[u32],
            pages: &[u64],
            slot: u64,
        ) -> Result<(), BundleAbiError> {
            self.instance
                .restore_prefix_state_in_slot(id, tokens, pages, slot)?;
            // Only restores the bundle accepted: a refused one runs cold.
            self.slot_restores.set(self.slot_restores.get() + 1);
            Ok(())
        }
        fn supports_speculation(&self) -> bool {
            true
        }
        fn supports_request_retirement(&self) -> bool {
            true
        }
    }

    /// Requests holding a state slot, as the engine admitted and retired them.
    type LiveSlots = Rc<RefCell<BTreeSet<RequestId>>>;

    struct Drafting {
        inner: BundleModelExecutorV1<Direct>,
        wrong: usize,
        live: LiveSlots,
        restores: Rc<Cell<usize>>,
        restored: BTreeSet<RequestId>,
        restore_phases: Rc<RefCell<Vec<BatchPhase>>>,
    }

    impl ModelExecutor for Drafting {
        fn kv_offload_info_v1(&self) -> Option<runtime_engine::KvOffloadInfo> {
            self.inner.kv_offload_info_v1()
        }
        fn offload_pages_v1(
            &mut self,
            plan: runtime_kv::KvTransferPlan,
        ) -> runtime_engine::KvTransferSubmitResult {
            self.inner.offload_pages_v1(plan)
        }
        fn reload_pages_v1(
            &mut self,
            plan: runtime_kv::KvTransferPlan,
        ) -> runtime_engine::KvTransferSubmitResult {
            self.inner.reload_pages_v1(plan)
        }
        fn poll_kv_transfer_v1(
            &mut self,
            id: runtime_kv::KvTransferId,
        ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
            self.inner.poll_kv_transfer_v1(id)
        }
        fn wait_kv_transfer_v1(
            &mut self,
            id: runtime_kv::KvTransferId,
        ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
            self.inner.wait_kv_transfer_v1(id)
        }
        fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
            self.inner.drain_kv_transfers_v1()
        }
        type TransactionState<'a>
            = BundleTransactionStateV1<'a, Direct>
        where
            Self: 'a;
        fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
            self.inner.admit_request_v1(id)?;
            self.live.borrow_mut().insert(id);
            Ok(())
        }
        fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
            self.inner.retire_request_v1(id)?;
            self.live.borrow_mut().remove(&id);
            Ok(())
        }
        fn execute<'a>(
            &'a mut self,
            items: &[ModelBatchItem],
            plan: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            for item in items {
                if self.restored.remove(&item.request_id()) {
                    self.restore_phases.borrow_mut().push(item.phase());
                }
            }
            self.inner.execute(items, plan)
        }
        fn capture_prefix_state_v1(
            &mut self,
            id: RequestId,
            tokens: usize,
            pages: &[runtime_kv::BlockId],
        ) -> Result<(), ExecutorError> {
            self.inner.capture_prefix_state_v1(id, tokens, pages)
        }
        fn restore_prefix_state_v1(
            &mut self,
            id: RequestId,
            prefix: &[u32],
            pages: &[runtime_kv::BlockId],
        ) -> Result<(), ExecutorError> {
            self.inner.restore_prefix_state_v1(id, prefix, pages)?;
            self.restores.set(self.restores.get() + 1);
            self.restored.insert(id);
            Ok(())
        }
        fn draft_tokens_v1(
            &mut self,
            _id: RequestId,
            history: &[u32],
            maximum: usize,
        ) -> Result<Vec<u32>, ExecutorError> {
            let mut lanes = fold_tokens_v1(&state_seed_lanes_v1(), history);
            let mut drafts = Vec::new();
            for index in 0..maximum {
                let bytes = logits_row_v1(&lanes, RECURRENT_VOCABULARY_V1);
                let logits: Vec<_> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let mut best = 0;
                for token in 1..logits.len() {
                    if logits[token] > logits[best] {
                        best = token;
                    }
                }
                let token = (best as u32 + u32::from(self.wrong != 0 && index % self.wrong == 0))
                    % RECURRENT_VOCABULARY_V1;
                drafts.push(token);
                lanes = fold_tokens_v1(&lanes, &[token]);
            }
            Ok(drafts)
        }
    }

    struct Opened {
        backend: Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
        engine: Engine<Drafting>,
        verifies: Rc<Cell<usize>>,
        slot_restores: Rc<Cell<usize>>,
        live: LiveSlots,
        restores: Rc<Cell<usize>>,
        restore_phases: Rc<RefCell<Vec<BatchPhase>>>,
    }

    fn open_engine(speculative: bool, wrong: usize, prefix: bool) -> Opened {
        open_admitted_engine(speculative, wrong, prefix, AdmissionPolicy::Reserved, 64)
    }

    fn open_admitted_engine(
        speculative: bool,
        wrong: usize,
        prefix: bool,
        admission: AdmissionPolicy,
        pages: usize,
    ) -> Opened {
        open_admitted_engine_with_source(
            speculative,
            wrong,
            prefix,
            admission,
            pages,
            DraftSource::Host,
        )
    }

    fn open_admitted_engine_with_source(
        speculative: bool,
        wrong: usize,
        prefix: bool,
        admission: AdmissionPolicy,
        pages: usize,
        source: DraftSource,
    ) -> Opened {
        open_swap_engine(
            speculative,
            wrong,
            prefix,
            admission,
            pages,
            source,
            (false, false),
        )
    }

    fn open_swap_engine(
        speculative: bool,
        wrong: usize,
        prefix: bool,
        admission: AdmissionPolicy,
        pages: usize,
        source: DraftSource,
        swap: (bool, bool),
    ) -> Opened {
        let (offload, force) = swap;
        let (backend, instance) = if prefix || offload {
            created_prefix()
        } else {
            created_requests()
        };
        let draft =
            bundle_recurrent_fixture::recurrent_bundle_draft_v1(std::path::Path::new("Cargo.toml"))
                .unwrap();
        let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
        let pool = runtime_host::recurrent::StateSlotPoolV1::new(&manifest, 2, 4).unwrap();
        let verifies = Rc::new(Cell::new(0));
        let slot_restores = Rc::new(Cell::new(0));
        let inner = BundleModelExecutorV1::new_with_batch_wire(
            Direct {
                instance,
                verifies: verifies.clone(),
                slot_restores: slot_restores.clone(),
            },
            RECURRENT_VOCABULARY_V1 as usize,
            8,
            BatchWireCapabilityV1::V3Speculative,
        )
        .with_state_slots(pool);
        let inner = if offload {
            let arena = backend
                .initialize_kv_offload(
                    runtime_host::offload::KvOffloadOptionsV1 {
                        budget_bytes: 32768,
                        minimum_tokens: 1,
                        bytes_per_second: None,
                    },
                    32 * 1024 * 1024 * 1024,
                )
                .unwrap()
                .unwrap();
            inner.with_kv_offload(
                backend.clone(),
                runtime_engine::KvOffloadInfo {
                    host_page_capacity: arena.pages_per_rank,
                    minimum_tokens: 1,
                    snapshot_pages: 3,
                },
            )
        } else {
            inner
        };
        let vocab = ModelVocabulary::new(RECURRENT_VOCABULARY_V1 as usize).unwrap();
        let mut config = EngineConfig::new(
            SchedulerConfig::new(2, 16, 8, 3, 1).unwrap(),
            KvGeometry::new(8, pages).unwrap(),
            vocab,
            64.min(pages * 8),
            16,
        )
        .unwrap()
        .with_admission(admission);
        if speculative {
            config = config.with_speculation(
                SpeculationConfig::new(3, 8, SpeculationMode::Fixed, source).unwrap(),
            );
        }
        if prefix {
            config = config.with_prefix_cache(3);
        }
        if force {
            config = config.with_forced_swaps();
        }
        let live = Rc::new(RefCell::new(BTreeSet::new()));
        let restores = Rc::new(Cell::new(0));
        let restore_phases = Rc::new(RefCell::new(Vec::new()));
        let engine = Engine::new(
            config,
            Drafting {
                inner,
                wrong,
                live: live.clone(),
                restores: restores.clone(),
                restored: BTreeSet::new(),
                restore_phases: restore_phases.clone(),
            },
        )
        .unwrap();
        Opened {
            backend,
            engine,
            verifies,
            slot_restores,
            live,
            restores,
            restore_phases,
        }
    }

    fn run_exact(
        speculative: bool,
        wrong: usize,
        temperature: f64,
    ) -> BTreeMap<RequestId, Vec<EngineEventKind>> {
        run_admitted(
            open_engine(speculative, wrong, false),
            speculative,
            temperature,
            64,
        )
        .0
    }

    /// Runs the exactness workload and returns its events and preemptions.
    fn run_admitted(
        opened: Opened,
        speculative: bool,
        temperature: f64,
        pages: usize,
    ) -> (BTreeMap<RequestId, Vec<EngineEventKind>>, usize) {
        run_admitted_with_pinned_neighbor(opened, speculative, temperature, pages, false)
    }

    fn run_admitted_with_pinned_neighbor(
        opened: Opened,
        speculative: bool,
        temperature: f64,
        pages: usize,
        pin_first: bool,
    ) -> (BTreeMap<RequestId, Vec<EngineEventKind>>, usize) {
        let Opened {
            mut engine,
            verifies,
            live,
            ..
        } = opened;
        let vocab = ModelVocabulary::new(RECURRENT_VOCABULARY_V1 as usize).unwrap();
        let mut output: BTreeMap<RequestId, Vec<EngineEventKind>> = BTreeMap::new();
        for index in 0..8 {
            let sampling = SamplingParameters::new(temperature)
                .unwrap()
                .with_top_logprobs(3)
                .unwrap()
                .with_penalties(0.2, 0.1, 1.1)
                .unwrap();
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(index + 1),
                        vec![2, 3, 4, index as u32 + 1, 2],
                        if pin_first && index == 0 { 48 } else { 12 },
                        sampling,
                        index * 97 + 3,
                        if index % 2 == 0 && !(pin_first && index == 0) {
                            vec![5]
                        } else {
                            vec![]
                        },
                        vocab,
                    )
                    .map(|request| {
                        if pin_first && index == 0 {
                            request.with_prompt_logprobs(0)
                        } else {
                            request
                        }
                    })
                    .unwrap(),
                )
                .unwrap();
        }
        for _ in 0..1000 {
            let status = engine.step().unwrap();
            engine.audit().unwrap();
            for event in engine.drain_events() {
                output
                    .entry(event.request_id())
                    .or_default()
                    .push(event.kind().clone());
            }
            if status == StepStatus::Idle {
                let stats = engine.kv_stats();
                assert_eq!(stats.used_blocks(), 0);
                assert_eq!(stats.free_blocks() + stats.cached_blocks(), pages);
                assert!(live.borrow().is_empty());
                if speculative {
                    assert!(verifies.get() > 0);
                }
                let metrics = engine.metrics().prometheus_compatibility("fixture");
                let preemptions = metrics
                    .lines()
                    .find(|line| line.starts_with("vllm:num_preemptions_total{"))
                    .and_then(|line| line.rsplit(' ').next())
                    .and_then(|value| value.parse().ok())
                    .unwrap();
                return (output, preemptions);
            }
        }
        panic!("recurrent exactness run did not finish");
    }

    /// A preempted request retires its recurrent state slot and recomputes
    /// the state from its tokens, or restores it from the anchor its
    /// preemption captured, so every event matches a run without preemption.
    #[test]
    fn preempted_requests_recompute_recurrent_state_exactly() {
        for temperature in [0.0, 0.7] {
            let expected = run_exact(false, 0, temperature);
            for speculative in [false, true] {
                for prefix in [false, true] {
                    for (admission, pages) in [
                        (
                            AdmissionPolicy::Optimistic(OptimisticAdmission::new(0, 3).unwrap()),
                            3,
                        ),
                        (
                            AdmissionPolicy::Optimistic(
                                OptimisticAdmission::DEFAULT
                                    .with_forced_preemption(4)
                                    .unwrap(),
                            ),
                            64,
                        ),
                    ] {
                        let opened = open_admitted_engine(speculative, 1, prefix, admission, pages);
                        let restores = opened.restores.clone();
                        let restore_phases = opened.restore_phases.clone();
                        let (output, preemptions) =
                            run_admitted(opened, speculative, temperature, pages);
                        // With room for anchors, recomputes restore the
                        // state their preemptions captured.
                        assert_eq!(
                            restores.get() > 0,
                            prefix && pages == 64,
                            "{temperature} {speculative} {prefix} {admission:?}"
                        );
                        assert!(
                            preemptions > 0,
                            "{temperature} {speculative} {prefix} {admission:?}"
                        );
                        assert_eq!(
                            output, expected,
                            "{temperature} {speculative} {prefix} {admission:?}"
                        );
                        for phase in restore_phases.borrow().iter() {
                            assert!(!matches!(phase, BatchPhase::Prefill { .. }));
                        }
                        if speculative && prefix && pages == 64 {
                            assert!(
                                restore_phases
                                    .borrow()
                                    .iter()
                                    .any(|phase| matches!(phase, BatchPhase::Verify { .. }))
                            );
                        }
                        assert_event_bits(&output, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn host_swaps_preserve_recurrent_event_bits() {
        for prefix in [false, true] {
            for source in [DraftSource::Host, DraftSource::Bundle] {
                for speculative in [false, true] {
                    for temperature in [0.0, 0.7] {
                        let expected = run_exact(false, 0, temperature);
                        for (pages, force) in [(64, true), (8, false)] {
                            let admission = AdmissionPolicy::Optimistic(if force {
                                OptimisticAdmission::DEFAULT
                                    .with_forced_preemption(4)
                                    .unwrap()
                            } else {
                                OptimisticAdmission::new(0, 3).unwrap()
                            });
                            let opened = open_swap_engine(
                                speculative,
                                0,
                                prefix,
                                admission,
                                pages,
                                source,
                                (true, force),
                            );
                            let metrics = opened.engine.metrics();
                            let phases = opened.restore_phases.clone();
                            let expected = if force {
                                expected.clone()
                            } else {
                                run_admitted_with_pinned_neighbor(
                                    open_engine(false, 0, false),
                                    false,
                                    temperature,
                                    64,
                                    true,
                                )
                                .0
                            };
                            let (actual, preemptions) = run_admitted_with_pinned_neighbor(
                                opened,
                                speculative,
                                temperature,
                                pages,
                                !force,
                            );
                            assert_event_bits(&actual, &expected);
                            assert!(preemptions > 0);
                            let stats = metrics.snapshot().offload;
                            assert!(stats.copies[0] > 0, "{stats:?}");
                            assert!(stats.copies[1] > 0, "{stats:?}");
                            assert_eq!(stats.used_pages, 0);
                            assert_eq!(stats.swapped_requests, 0);
                            assert!(
                                phases
                                    .borrow()
                                    .iter()
                                    .all(|p| !matches!(p, BatchPhase::Prefill { .. }))
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bundle_drafted_verify_is_the_first_step_after_an_exact_restore() {
        for temperature in [0.0, 0.7] {
            let expected = run_exact(false, 0, temperature);
            let opened = open_admitted_engine_with_source(
                true,
                0,
                true,
                AdmissionPolicy::Optimistic(
                    OptimisticAdmission::DEFAULT
                        .with_forced_preemption(4)
                        .unwrap(),
                ),
                64,
                DraftSource::Bundle,
            );
            let phases = opened.restore_phases.clone();
            let (actual, preemptions) = run_admitted(opened, true, temperature, 64);
            assert!(preemptions > 0);
            assert!(!phases.borrow().is_empty());
            assert!(
                phases
                    .borrow()
                    .iter()
                    .all(|phase| matches!(phase, BatchPhase::Verify { .. }))
            );
            assert_event_bits(&actual, &expected);
        }
    }

    fn assert_event_bits(
        actual: &BTreeMap<RequestId, Vec<EngineEventKind>>,
        expected: &BTreeMap<RequestId, Vec<EngineEventKind>>,
    ) {
        assert_eq!(actual, expected);
        for (id, events) in actual {
            for (actual, expected) in events.iter().zip(&expected[id]) {
                if let (
                    EngineEventKind::Token {
                        logprob: a,
                        top_logprobs: at,
                        ..
                    },
                    EngineEventKind::Token {
                        logprob: b,
                        top_logprobs: bt,
                        ..
                    },
                ) = (actual, expected)
                {
                    assert_eq!(a.to_bits(), b.to_bits());
                    for (a, b) in at.iter().zip(bt) {
                        assert_eq!(a.logprob().to_bits(), b.logprob().to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn a_stop_string_crossing_a_restore_keeps_seeded_grammar_output_and_usage() {
        use runtime_frontend::{
            BundleParserSourceV1, CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1,
            DecoderLimitsV1, FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
        };
        use runtime_grammar::{
            CompiledGrammar, ConstraintState, TokenVocabulary, VocabularyToken, regex_grammar,
        };
        use std::sync::Arc;

        // E is emitted immediately before forced preemption, N and D after
        // restore. The decoder must retain E across the request's retirement.
        let bytes = |token: u32| match token {
            0..=9 => b'0' + token as u8,
            10 => b'E',
            11 => b'N',
            12 => b'D',
            _ => b'x',
        };
        let vocabulary = Arc::new(
            TokenVocabulary::new(
                64,
                (0..64)
                    .map(|token| VocabularyToken {
                        bytes: vec![bytes(token)],
                        special: false,
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let grammar = Arc::new(
            CompiledGrammar::new(&regex_grammar("[0-9]{3}END[0-9]{6}").unwrap(), vocabulary)
                .unwrap(),
        );
        let descriptor = Arc::new(
            FrontendDescriptorV1::from_component_bytes(frontend_fixture::KEY_VALUE.as_bytes())
                .unwrap(),
        );
        for temperature in [0.0, 0.7] {
            let run = |prefix, speculative, wrong, forced, swap| {
                let admission = if forced {
                    AdmissionPolicy::Optimistic(
                        OptimisticAdmission::new(0, 1)
                            .unwrap()
                            .with_forced_preemption(4)
                            .unwrap(),
                    )
                } else {
                    AdmissionPolicy::Reserved
                };
                let Opened {
                    mut engine,
                    live,
                    restores,
                    ..
                } = open_swap_engine(
                    speculative,
                    wrong,
                    prefix,
                    admission,
                    64,
                    DraftSource::Host,
                    (swap, swap),
                );
                let mut decoder = CompletionDecoderV1::with_client_stops(
                    descriptor.clone(),
                    PromptModeV1::Chat,
                    DecoderLimitsV1::declared(&descriptor),
                    &["END".to_owned()],
                )
                .unwrap();
                // The completion parses through the test fixture format's
                // export over the parser fixture, entered in process.
                decoder.set_parser_source(
                    BundleParserSourceV1::from_export(
                        &frontend_format::chat_format_export_v1::<
                            frontend_format::test_support::FixtureFormatV1,
                        >(),
                        frontend_fixture::KEY_VALUE.as_bytes(),
                    )
                    .unwrap(),
                );
                let id = RequestId::new(1);
                engine
                    .submit(
                        EngineRequest::new(
                            id,
                            vec![2, 3, 4, 5, 6],
                            12,
                            SamplingParameters::new(temperature)
                                .unwrap()
                                .with_top_logprobs(3)
                                .unwrap(),
                            97,
                            vec![],
                            ModelVocabulary::new(64).unwrap(),
                        )
                        .unwrap()
                        .with_constraint(ConstraintState::new(grammar.clone(), None, vec![]))
                        .unwrap(),
                    )
                    .unwrap();
                let mut updates = Vec::new();
                type TokenBits = (u32, u64, Vec<(u32, u64)>);
                let mut tokens: Vec<TokenBits> = Vec::new();
                let mut stopped = false;
                for _ in 0..100 {
                    let restores_before = restores.get();
                    let emitted_before = tokens.len();
                    let status = engine.step().unwrap();
                    engine.audit().unwrap();
                    if restores.get() > restores_before {
                        assert_eq!(emitted_before, 4, "restore must split END after E");
                        assert_eq!(tokens.last().unwrap().0, 10);
                    }
                    for event in engine.drain_events() {
                        if stopped {
                            continue;
                        }
                        if let EngineEventKind::Token {
                            token_id,
                            logprob,
                            top_logprobs,
                        } = event.kind()
                        {
                            let next = decoder
                                .push_engine_token(GeneratedTokenV1::new(
                                    tokens.len() as u64,
                                    200 + *token_id,
                                    vec![bytes(*token_id)],
                                ))
                                .unwrap();
                            tokens.push((
                                *token_id,
                                logprob.to_bits(),
                                top_logprobs
                                    .iter()
                                    .map(|top| (top.token_id(), top.logprob().to_bits()))
                                    .collect::<Vec<_>>(),
                            ));
                            stopped = next
                                .iter()
                                .any(|update| matches!(update, CompletionUpdateV1::Finished(_)));
                            updates.extend(next);
                            if stopped {
                                engine.cancel(id).unwrap();
                            }
                        }
                    }
                    if status == StepStatus::Idle {
                        assert!(stopped);
                        assert!(live.borrow().is_empty());
                        assert_eq!(engine.kv_stats().used_blocks(), 0);
                        assert_eq!(restores.get() > 0, (prefix || swap) && forced);
                        let finished = updates
                            .iter()
                            .find_map(|update| match update {
                                CompletionUpdateV1::Finished(value) => Some(value),
                                _ => None,
                            })
                            .unwrap();
                        assert_eq!(finished.reason, CompletionFinishV1::Stop);
                        assert_eq!(finished.generated_token_count, 6);
                        assert_eq!(tokens.len(), 6);
                        assert!(tokens[..3].iter().all(|token| token.0 < 10));
                        return (tokens, updates);
                    }
                }
                panic!("stop-string run did not finish");
            };
            let expected = run(false, false, 0, false, false);
            for prefix in [false, true] {
                // Fixed drafts exercise both accepted and rejected verify rows.
                for (speculative, wrong) in [(false, 0), (true, 0), (true, 1), (true, 2)] {
                    assert_eq!(run(prefix, speculative, wrong, true, false), expected);
                    assert_eq!(run(prefix, speculative, wrong, true, true), expected);
                }
            }
        }
    }

    #[test]
    fn recurrent_fixture_speculation_is_bitwise_exact_for_greedy_and_seeded_sampling() {
        for temperature in [0.0, 0.7, 1.3] {
            let expected = run_exact(false, 0, temperature);
            assert_eq!(expected.len(), 8);
            for wrong in [0, 1, 2] {
                assert_eq!(run_exact(true, wrong, temperature), expected);
            }
        }
    }

    fn request(id: u64, prompt: Vec<u32>, output: usize) -> EngineRequest {
        EngineRequest::new(
            RequestId::new(id),
            prompt,
            output,
            SamplingParameters::new(0.0).unwrap(),
            7,
            vec![],
            ModelVocabulary::new(RECURRENT_VOCABULARY_V1 as usize).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn cancellation_and_execution_errors_return_slots_for_later_admission() {
        let Opened {
            backend,
            mut engine,
            live,
            ..
        } = open_engine(false, 0, false);
        for id in 1..=3 {
            engine.submit(request(id, vec![2, 3, 4], 12)).unwrap();
        }
        engine.step().unwrap();
        engine.cancel(RequestId::new(3)).unwrap(); // queued, no state slot
        engine.cancel(RequestId::new(1)).unwrap(); // admitted and committed
        backend.with_device_mut(|device| device.refuse(FakeRefusalV1::CopyDeviceToDevice));
        assert!(engine.step().is_err());
        backend.with_device_mut(|device| device.allow(FakeRefusalV1::CopyDeviceToDevice));
        let _ = engine.drain_events();
        for id in 4..=6 {
            engine.submit(request(id, vec![2, 3], 3)).unwrap();
        }
        for _ in 0..100 {
            let status = engine.step().unwrap();
            let _ = engine.drain_events();
            engine.audit().unwrap();
            if status == StepStatus::Idle {
                assert_eq!(engine.kv_stats().free_blocks(), 64);
                assert!(live.borrow().is_empty());
                return;
            }
        }
        panic!("replacement requests did not finish");
    }

    fn created_prefix() -> (
        Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
        BundleInstanceV1,
    ) {
        let config = instance_config().with_serving_kv(runtime_host::ServingKvPolicyV1 {
            sampling_vocabulary: None,
            maximum_context_tokens: None,
            state_reserved_bytes: 1024,
            fraction_millionths: 900_000,
            maximum_pages: Some(64),
        });
        let configuration = config.encode_v1(RECURRENT_TOPOLOGY_ID_V1.as_bytes());
        let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
        let table = HostServicesTableV1::new(backend.clone());
        let api = BundleApiV1::new_with_prefix_state_slot(
            bundle_recurrent_fixture::BUNDLE_RECURRENT_FIXTURE_API_WITH_PREFIX_STATE_SLOT_V1,
        )
        .unwrap();
        (backend, api.create(table, &configuration).unwrap())
    }

    fn snapshot_bytes(
        backend: &Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
        pages: &[u64],
    ) -> Vec<u8> {
        let table = HostServicesTableV1::new(backend.clone());
        let host = table.caller();
        let context = host.select_context(0, 0).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        let pool = host.resolve_serving_kv_pool(0).unwrap();
        let mut bytes = Vec::new();
        for page in pages {
            let mut row = vec![0; 32];
            host.copy_device_to_host(
                stream,
                &mut row,
                runtime_bundle_abi::DeviceBufferSpanV1::new(
                    pool.buffer,
                    pool.offset + page * 32,
                    32,
                ),
            )
            .unwrap();
            host.synchronize_stream(stream).unwrap();
            bytes.extend_from_slice(&row);
        }
        host.release_stream(stream).unwrap();
        host.release_context(context).unwrap();
        bytes
    }

    #[test]
    fn prefix_snapshots_restore_identical_bytes_and_resume_the_same_logits() {
        let (backend, mut instance) = created_prefix();
        let prefix = [2, 3, 4, 5];
        let mut sink = CollectingSink::new();
        instance
            .execute(&batch(1, 0, 0, &prefix, false), &mut sink)
            .unwrap();
        instance.commit().unwrap();
        assert_eq!(instance.describe_prefix_state().unwrap(), 3);
        instance.capture_prefix_state(1, 4, &[10, 11, 12]).unwrap();
        let captured = snapshot_bytes(&backend, &[10, 11, 12]);
        assert_eq!(
            &captured[..64],
            bundle_recurrent_fixture::lanes_to_bytes_v1(&fold_tokens_v1(
                &state_seed_lanes_v1(),
                &prefix
            ))
        );
        instance.retire_request(1).unwrap();
        backend.with_device_mut(|device| device.refuse(FakeRefusalV1::SynchronizeStream));
        assert!(
            instance
                .restore_prefix_state(2, &prefix, &[10, 11, 12])
                .is_err()
        );
        backend.with_device_mut(|device| device.allow(FakeRefusalV1::SynchronizeStream));
        // A refused restore keeps no request state, so this retry can succeed.
        instance
            .restore_prefix_state(2, &prefix, &[10, 11, 12])
            .unwrap();
        instance.capture_prefix_state(2, 4, &[20, 21, 22]).unwrap();
        assert_eq!(snapshot_bytes(&backend, &[20, 21, 22]), captured);
        let mut resumed = CollectingSink::new();
        instance
            .execute(&batch(2, 0, 4, &[6], false), &mut resumed)
            .unwrap();
        assert_eq!(
            resumed.rows[0].logit_bits(),
            logits_row_v1(
                &fold_tokens_v1(&state_seed_lanes_v1(), &[2, 3, 4, 5, 6]),
                RECURRENT_VOCABULARY_V1
            )
        );
        instance.cancel().unwrap();
        instance.capture_prefix_state(2, 4, &[20, 21, 22]).unwrap();
        assert_eq!(snapshot_bytes(&backend, &[20, 21, 22]), captured);
        instance.retire_request(2).unwrap();
        instance.destroy().unwrap();
    }

    #[test]
    fn a_state_slot_restore_binds_the_slot_before_the_first_batch() {
        let (backend, mut instance) = created_prefix();
        assert!(instance.supports_prefix_state_slot());
        let prefix = [2, 3, 4, 5];
        let mut sink = CollectingSink::new();
        instance
            .execute(&batch(1, 0, 0, &prefix, false), &mut sink)
            .unwrap();
        instance.commit().unwrap();
        instance.capture_prefix_state(1, 4, &[10, 11, 12]).unwrap();
        let captured = snapshot_bytes(&backend, &[10, 11, 12]);
        // Slot 0 still belongs to request 1.
        assert_eq!(
            instance.restore_prefix_state_in_slot(2, &prefix, &[10, 11, 12], 0),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
        instance
            .restore_prefix_state_in_slot(2, &prefix, &[10, 11, 12], 1)
            .unwrap();
        // The restored state is bound to slot 1: a batch naming slot 0 is
        // refused, and one naming slot 1 resumes the prefix.
        let mut refused = CollectingSink::new();
        assert!(
            instance
                .execute(&batch(2, 0, 4, &[6], false), &mut refused)
                .is_err()
        );
        let mut resumed = CollectingSink::new();
        instance
            .execute(&batch(2, 1, 4, &[6], false), &mut resumed)
            .unwrap();
        assert_eq!(
            resumed.rows[0].logit_bits(),
            logits_row_v1(
                &fold_tokens_v1(&state_seed_lanes_v1(), &[2, 3, 4, 5, 6]),
                RECURRENT_VOCABULARY_V1
            )
        );
        instance.cancel().unwrap();
        instance.capture_prefix_state(2, 4, &[20, 21, 22]).unwrap();
        assert_eq!(snapshot_bytes(&backend, &[20, 21, 22]), captured);
        instance.retire_request(1).unwrap();
        instance.retire_request(2).unwrap();
        instance.destroy().unwrap();
    }

    #[test]
    fn concurrent_hits_each_restore_into_their_own_admitted_slot() {
        // Five requests share a 16-token prefix and wait for two slots. Each
        // hit is admitted with a slot, restores into it and releases it at
        // retirement, however admission settles around the shared pages.
        fn run(prefix: bool) -> (Vec<EngineEventKind>, usize, usize) {
            let Opened {
                mut engine,
                slot_restores,
                live,
                ..
            } = open_engine(false, 0, prefix);
            let shared = (0..16).map(|at| at * 3 % 60 + 2).collect::<Vec<u32>>();
            let drive = |engine: &mut Engine<Drafting>, output: &mut Vec<_>, cached: &mut usize| {
                for _ in 0..400 {
                    let status = engine.step().unwrap();
                    engine.audit().unwrap();
                    for event in engine.drain_events() {
                        match event.kind() {
                            EngineEventKind::PromptCached { tokens } => *cached += tokens,
                            kind => output.push(kind.clone()),
                        }
                    }
                    if status == StepStatus::Idle {
                        return;
                    }
                }
                panic!("requests did not finish");
            };
            let (mut output, mut cached) = (Vec::new(), 0);
            engine.submit(request(1, shared.clone(), 2)).unwrap();
            drive(&mut engine, &mut output, &mut cached);
            for id in 2..=6 {
                let mut prompt = shared.clone();
                prompt.extend([id as u32 + 3, 7, id as u32]);
                engine.submit(request(id, prompt, 4)).unwrap();
            }
            drive(&mut engine, &mut output, &mut cached);
            assert!(live.borrow().is_empty());
            (output, cached, slot_restores.get())
        }
        let (cold, _, _) = run(false);
        let (warm, cached, restores) = run(true);
        assert_eq!(restores, 5, "every hit restored into its slot");
        assert_eq!(cached, 5 * 16);
        assert_eq!(warm, cold);
    }

    #[test]
    fn anchors_evicted_under_pressure_leave_outputs_equal_to_cold_runs() {
        // Twelve 20-token prompts, each followed by a turn that extends it:
        // every anchor takes 3 prompt pages and 3 snapshot pages of the 64,
        // so later anchors evict earlier ones, and the last round asks for
        // the first prompts again.
        fn run(prefix: bool) -> (Vec<EngineEventKind>, usize, u64) {
            let Opened {
                mut engine,
                slot_restores,
                live,
                ..
            } = open_engine(false, 0, prefix);
            let prompt = |index: u32| (0..20).map(|at| (index * 7 + at * 3) % 60 + 2).collect();
            let turns = (0..12)
                .chain(0..3)
                .flat_map(|index: u32| {
                    let first: Vec<u32> = prompt(index);
                    let mut next = first.clone();
                    next.extend([index + 2, 5]);
                    [first, next]
                })
                .collect::<Vec<_>>();
            let mut output = Vec::new();
            let mut cached = 0;
            for (id, turn) in turns.into_iter().enumerate() {
                engine.submit(request(id as u64 + 1, turn, 3)).unwrap();
                for _ in 0..100 {
                    let status = engine.step().unwrap();
                    engine.audit().unwrap();
                    for event in engine.drain_events() {
                        match event.kind() {
                            EngineEventKind::PromptCached { tokens } => cached += tokens,
                            kind => output.push(kind.clone()),
                        }
                    }
                    if status == StepStatus::Idle {
                        break;
                    }
                }
            }
            assert!(live.borrow().is_empty());
            assert_eq!(slot_restores.get() > 0, prefix);
            (
                output,
                cached,
                engine.kv_metrics().snapshot().prefix_cache_evictions,
            )
        }
        let (cold, _, _) = run(false);
        let (warm, cached, evictions) = run(true);
        assert!(cached > 0);
        assert!(evictions > 0, "the pool never evicted an anchor");
        assert_eq!(warm, cold);
    }

    #[test]
    fn anchors_restore_recurrent_state_before_speculative_execution() {
        fn run(prefix: bool) -> (Vec<EngineEventKind>, usize) {
            let Opened {
                mut engine,
                verifies,
                slot_restores,
                live,
                ..
            } = open_engine(true, 0, prefix);
            engine
                .submit(request(1, vec![2, 3, 4, 5, 6, 7, 8, 9], 1))
                .unwrap();
            for _ in 0..20 {
                let status = engine.step().unwrap();
                let _ = engine.drain_events();
                if status == StepStatus::Idle {
                    break;
                }
            }
            engine
                .submit(request(2, vec![2, 3, 4, 5, 6, 7, 8, 9, 10], 12))
                .unwrap();
            let mut output = Vec::new();
            let mut cached = 0;
            for _ in 0..100 {
                let status = engine.step().unwrap();
                engine.audit().unwrap();
                for event in engine.drain_events() {
                    match event.kind() {
                        EngineEventKind::PromptCached { tokens } => cached += tokens,
                        kind => output.push(kind.clone()),
                    }
                }
                if status == StepStatus::Idle {
                    assert!(verifies.get() > 0);
                    assert!(live.borrow().is_empty());
                    // The host restores through the state-slot tail.
                    assert_eq!(slot_restores.get(), usize::from(prefix));
                    return (output, cached);
                }
            }
            panic!("prefix request did not finish");
        }
        let (cold, _) = run(false);
        let (warm, cached) = run(true);
        assert_eq!(cached, 8);
        assert_eq!(warm, cold);
    }
}
