//! The TP4-shaped execution path, driven end to end over host services.
//!
//! Nothing here reaches into the fixture: the host builds its own service table,
//! the fixture is entered through the same V1 API table its native module
//! exports, and the only things that cross are bytes and opaque handles. The
//! device work happens in host memory, so this proves the ownership, ordering,
//! transaction, and refusal behaviour of the path without a GPU.
//!
//! What makes it worth proving separately from the other two fixtures is the
//! shape. Four ranks, one PrefillChunk followed by one DecodeStep over a rank-local
//! paged KV pool, a 154,880-wide row on each, and the shared sampler consuming
//! that row. None of the values in it are TP4 arithmetic and none of them are
//! claimed to be.

mod host_fixture;

use std::rc::Rc;

use bundle_tp4_fixture::{
    BUNDLE_TP4_FLASH_API_V1, BUNDLE_TP4_FLASH_API_WITH_PREFIX_CACHE_V1,
    BUNDLE_TP4_FLASH_API_WITH_SPECULATION_V1, TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
    TP4_FIXTURE_KV_LAYOUT_ID_V1, TP4_FIXTURE_KV_PAGE_TOKENS_V1, TP4_FIXTURE_MARKER_ROLE_V1,
    TP4_FIXTURE_PEAK_TOKEN_ID_V1, TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1, TP4_FIXTURE_PROGRAM_ROLE_V1,
    TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_ROW_INPUT_BYTES_V1, TP4_FIXTURE_ROW_OPERANDS_V1,
    TP4_FIXTURE_ROW_OPERATION_ID_V1, TP4_FIXTURE_ROW_OPERATION_V1,
    TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1, TP4_FIXTURE_STATE_ALIGNMENT_V1,
    TP4_FIXTURE_STATE_BYTES_V1, TP4_FIXTURE_STATE_LAYOUT_ID_V1, TP4_FIXTURE_TOPOLOGY_ID_V1,
    fold_tokens_v1, history_to_bytes_v1, program_image_digests_v1, row_logits_v1,
    state_seed_history_v1, synthetic_row_v1, tp4_fixture_kv_layout_v1, tp4_fixture_row_contract_v1,
    tp4_fixture_topology_v1, tp4_fixture_vocabulary_v1, unreleased_resource_count_v1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleApiV1, BundleInstanceV1, BundleKeptRowsV1, BundleStatusV1,
    HostServiceResultV1, HostServicesTableV1, LogitsRowSinkV1, OwnedDraftTokensV1,
    OwnedLogitsRowV1, ResultSinkLimitsV1,
};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
use runtime_host::fake::{FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{
    DeclaredKvContractV1, DeclaredStateContractV1, HostInstanceConfigV1, HostServicesV1Backend,
    HostTopologyV1,
};

use host_fixture::{
    instance_config_with_program, program_component_bytes, program_image_description, tp4_ordinals,
};
use runtime_sampling::{SampleContext, SamplingParameters, SamplingState, sample};

fn instance_config() -> HostInstanceConfigV1 {
    host_fixture::instance_config()
}

fn services() -> (
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    HostServicesTableV1,
    Vec<u8>,
) {
    let config = instance_config();
    let configuration = config.encode_v1(TP4_FIXTURE_TOPOLOGY_ID_V1.as_bytes());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    (backend, table, configuration)
}

fn api() -> BundleApiV1 {
    BundleApiV1::new(BUNDLE_TP4_FLASH_API_V1).expect("the exported V1 table is valid")
}

fn created() -> (
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    BundleInstanceV1,
) {
    let (backend, table, configuration) = services();
    let instance = api()
        .create(table, &configuration)
        .expect("the fixture accepts the TP4 instance the host configured");
    (backend, instance)
}

/// The 32 tokens one PrefillChunk carries, as a batch a host could schedule.
fn prefill_tokens() -> Vec<u32> {
    (0..TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1)
        .map(|index| 100 + index)
        .collect()
}

/// One scheduled batch in the host's own wire form.
///
/// `previous_tokens` is how far the sequence had already got when the host
/// prepared this append, and `page_table` is the complete ordered logical block
/// table that append commits to. Both are what `runtime-host` encodes from a
/// prepared append, so a batch built here is one the engine could have built.
struct ScheduledBatchV1 {
    request_id: u64,
    tokens: Vec<u32>,
    decode: bool,
    expects_logits: bool,
    previous_tokens: u64,
    page_table: Vec<u64>,
}

impl ScheduledBatchV1 {
    fn prefill(request_id: u64, tokens: &[u32], previous_tokens: u64) -> Self {
        Self {
            request_id,
            tokens: tokens.to_vec(),
            decode: false,
            expects_logits: true,
            previous_tokens,
            page_table: identity_page_table(previous_tokens + tokens.len() as u64),
        }
    }

    fn decode(request_id: u64, token: u32, previous_tokens: u64) -> Self {
        Self {
            request_id,
            tokens: vec![token],
            decode: true,
            expects_logits: true,
            previous_tokens,
            page_table: identity_page_table(previous_tokens + 1),
        }
    }

    /// The same batch, scheduled as a chunk the engine expects no row from.
    fn without_row(mut self) -> Self {
        self.expects_logits = false;
        self
    }

    /// The same batch, over a placement the host chose rather than the identity.
    fn with_page_table(mut self, page_table: &[u64]) -> Self {
        self.page_table = page_table.to_vec();
        self
    }

    fn encode(&self) -> Vec<u8> {
        let mut request = Vec::new();
        request.extend_from_slice(b"chopbatch-v1");
        request.extend_from_slice(&1_u32.to_le_bytes());
        request.extend_from_slice(&self.request_id.to_le_bytes());
        request.push(u8::from(self.decode));
        request.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        request.push(u8::from(self.expects_logits));
        request.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for token in &self.tokens {
            request.extend_from_slice(&token.to_le_bytes());
        }
        request.extend_from_slice(&1_u32.to_le_bytes());
        request.extend_from_slice(&self.previous_tokens.to_le_bytes());
        request.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        request.extend_from_slice(&(self.previous_tokens + self.tokens.len() as u64).to_le_bytes());
        request.extend_from_slice(&(self.page_table.len() as u32).to_le_bytes());
        for page in &self.page_table {
            request.extend_from_slice(&page.to_le_bytes());
        }
        request
    }
}

/// The page table a host prepares for a sequence of this length.
///
/// One entry per logical page the sequence occupies once the append commits,
/// each placed at the physical page of the same ordinal.
fn identity_page_table(new_tokens: u64) -> Vec<u64> {
    (0..new_tokens.div_ceil(u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1))).collect()
}

fn prefill(request_id: u64) -> Vec<u8> {
    ScheduledBatchV1::prefill(request_id, &prefill_tokens(), 0).encode()
}

fn decode(request_id: u64, token: u32, previous_tokens: u64) -> Vec<u8> {
    ScheduledBatchV1::decode(request_id, token, previous_tokens).encode()
}

/// A sink that would accept a row, so a refusal cannot be blamed on capacity.
struct CollectingSink {
    rows: Vec<OwnedLogitsRowV1>,
    drafts: Vec<OwnedDraftTokensV1>,
}

impl CollectingSink {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            drafts: Vec::new(),
        }
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

    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.drafts.push(drafts);
        Ok(())
    }
}

#[test]
fn verify_rows_and_prefix_commit_leave_only_kept_state() {
    let (backend, table, configuration) = services();
    let api = BundleApiV1::new_with_speculation(BUNDLE_TP4_FLASH_API_WITH_SPECULATION_V1).unwrap();
    let mut instance = api.create(table, &configuration).unwrap();
    let initial = prefill_tokens();
    let mut sink = CollectingSink::new();
    instance.execute(&prefill(77), &mut sink).unwrap();
    instance.commit().unwrap();
    let previous = initial.len() as u64;
    let verify = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(
            77,
            4,
            true,
            vec![TP4_FIXTURE_PEAK_TOKEN_ID_V1],
        )],
        vec![BatchAppendV1::new(
            previous,
            4,
            previous + 4,
            identity_page_table(previous + 4),
            None,
        )],
    )
    .encode()
    .unwrap();
    let mut sink = CollectingSink::new();
    instance.execute(&verify, &mut sink).unwrap();
    assert_eq!(sink.drafts.len(), 1);
    assert_eq!(sink.drafts[0].tokens, [TP4_FIXTURE_PEAK_TOKEN_ID_V1; 3]);
    assert_eq!(sink.rows.len(), 4);
    let history = fold_tokens_v1(state_seed_history_v1(), &initial);
    for count in 1..=4 {
        assert_eq!(
            sink.rows[count - 1].logit_bits(),
            expected_row(
                fold_tokens_v1(history, &vec![TP4_FIXTURE_PEAK_TOKEN_ID_V1; count - 1]),
                &[TP4_FIXTURE_PEAK_TOKEN_ID_V1],
            )
        );
    }
    instance
        .commit_accepted(&[BundleKeptRowsV1 {
            struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
            reserved: 0,
            request_id: 77,
            kept_rows: 2,
            reserved_tail: 0,
        }])
        .unwrap();
    let mut next = CollectingSink::new();
    instance
        .execute(
            &decode(77, TP4_FIXTURE_PEAK_TOKEN_ID_V1, previous + 2),
            &mut next,
        )
        .unwrap();
    let kept_history = fold_tokens_v1(history, &[TP4_FIXTURE_PEAK_TOKEN_ID_V1; 2]);
    assert_eq!(
        one_row(&next).logit_bits(),
        expected_row(kept_history, &[TP4_FIXTURE_PEAK_TOKEN_ID_V1])
    );
    instance.cancel().unwrap();
    instance.destroy().unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

/// The bytes of every state-sized allocation, in allocation order: the
/// per-rank history the fixture keeps outside paged KV.
fn state_bytes(backend: &HostServicesV1Backend<FakeHostDeviceV1>) -> Vec<Vec<u8>> {
    backend.with_device(|device| {
        device
            .buffer_objects()
            .into_iter()
            .filter_map(|buffer| device.buffer_bytes(buffer))
            .filter(|bytes| bytes.len() as u64 == TP4_FIXTURE_STATE_BYTES_V1)
            .map(<[u8]>::to_vec)
            .collect()
    })
}

/// A second chat turn restored from the first turn's prompt-end anchor holds
/// the same state bytes as a cold prefill of that prompt, after the first
/// turn decoded past the anchor, and continues with the same rows.
#[test]
fn a_restored_turn_matches_a_cold_prefill_of_its_prefix_byte_for_byte() {
    let api =
        BundleApiV1::new_with_prefix_cache(BUNDLE_TP4_FLASH_API_WITH_PREFIX_CACHE_V1).unwrap();
    let turn_one = prefill_tokens();
    let boundary = turn_one.len() as u64;
    let answer = [11_u32, 12];
    // Turn two repeats turn one, its answer and a new user message.
    let turn_two_tail = [11_u32, 12, 21, 22, 23, 24, 25, 26];
    // The pool's last page; no sequence below uses it.
    let snapshot = [3_u64];

    let (warm_backend, table, configuration) = services();
    let mut warm = api.create(table, &configuration).unwrap();
    let mut sink = CollectingSink::new();
    warm.execute(
        &ScheduledBatchV1::prefill(1, &turn_one, 0)
            .with_page_table(&[0])
            .encode(),
        &mut sink,
    )
    .unwrap();
    warm.commit().unwrap();
    assert_eq!(warm.describe_prefix_state().unwrap(), 1);
    warm.capture_prefix_state(1, boundary, &snapshot).unwrap();
    for (index, token) in answer.iter().enumerate() {
        let mut sink = CollectingSink::new();
        warm.execute(
            &ScheduledBatchV1::decode(1, *token, boundary + index as u64)
                .with_page_table(&[0, 1])
                .encode(),
            &mut sink,
        )
        .unwrap();
        warm.commit().unwrap();
    }
    warm.retire_request(1).unwrap();
    let decoded = state_bytes(&warm_backend);
    warm.restore_prefix_state(2, &turn_one, &snapshot).unwrap();

    let (cold_backend, table, configuration) = services();
    let mut cold = api.create(table, &configuration).unwrap();
    let mut sink = CollectingSink::new();
    cold.execute(
        &ScheduledBatchV1::prefill(3, &turn_one, 0)
            .with_page_table(&[0])
            .encode(),
        &mut sink,
    )
    .unwrap();
    cold.commit().unwrap();

    let history = fold_tokens_v1(state_seed_history_v1(), &turn_one);
    let restored = state_bytes(&warm_backend);
    assert!(
        restored
            .iter()
            .any(|bytes| *bytes == history_to_bytes_v1(history)),
        "the restored state is the fold of the prefix: {restored:?}"
    );
    assert_eq!(restored, state_bytes(&cold_backend));
    assert_ne!(restored, decoded, "the restore rewound the decoded answer");

    let mut warm_rows = CollectingSink::new();
    warm.execute(
        &ScheduledBatchV1::prefill(2, &turn_two_tail, boundary)
            .with_page_table(&[0, 2])
            .encode(),
        &mut warm_rows,
    )
    .unwrap();
    warm.commit().unwrap();
    let mut cold_rows = CollectingSink::new();
    cold.execute(
        &ScheduledBatchV1::prefill(3, &turn_two_tail, boundary)
            .with_page_table(&[0, 1])
            .encode(),
        &mut cold_rows,
    )
    .unwrap();
    cold.commit().unwrap();
    assert_eq!(
        one_row(&warm_rows).logit_bits(),
        one_row(&cold_rows).logit_bits()
    );
    assert_eq!(
        one_row(&warm_rows).logit_bits(),
        expected_row(history, &turn_two_tail)
    );
    assert_eq!(state_bytes(&warm_backend), state_bytes(&cold_backend));

    for (mut instance, backend, request) in [(warm, warm_backend, 2), (cold, cold_backend, 3)] {
        instance.retire_request(request).unwrap();
        instance.destroy().unwrap();
        assert_eq!(backend.live_resource_count(), 0);
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

fn one_row(sink: &CollectingSink) -> &OwnedLogitsRowV1 {
    let [row] = sink.rows.as_slice() else {
        panic!(
            "this step scheduled exactly one row, and got {}",
            sink.rows.len()
        );
    };
    row
}

/// The row this fixture is expected to emit for one batch and one history.
fn expected_row(history: u64, tokens: &[u32]) -> Vec<u8> {
    synthetic_row_v1(
        &program_image_digests_v1(),
        history,
        tokens,
        tp4_fixture_vocabulary_v1(),
    )
}

/// Every resource this fixture takes is given back, newest first.
#[test]
fn one_instance_acquires_its_whole_topology_and_gives_every_resource_back() {
    let (backend, mut instance) = created();

    // One compiled image, four ranks of context, pool, stage stream, and rank
    // arena, two device buffers, the committed history, and the rank-local KV
    // pool with the one request every execution appends against.
    let acquired = backend.live_resource_count();
    assert!(
        acquired > 4 * TP4_FIXTURE_RANK_COUNT_V1 as usize,
        "a TP4 instance holds at least one resource per rank, and holds {acquired}"
    );
    let logs = logged(&backend);
    assert!(
        logs.iter()
            .any(|line| line.contains("154880-wide vocabulary read from its own program image")),
        "the instance must report the width it read from the package: {logs:?}"
    );
    assert!(
        logs.iter()
            .any(|line| line.contains("for topology tp4 across 4 ranks")),
        "the instance must say it serves TP4's TP4 topology: {logs:?}"
    );
    assert!(
        !logs.iter().any(|line| line.contains("tp1")),
        "nothing this fixture logs may imply TP1 is the TP4 target: {logs:?}"
    );
    assert!(
        logs.iter().any(|line| {
            line.contains("computes no TP4 arithmetic")
                && line.contains("qualifies no NVFP4 route")
                && line.contains("implements no TP4 operation")
                && line.contains("executes no authenticated payload")
        }),
        "the instance must say what it does not do: {logs:?}"
    );
    for (rank, digest) in program_image_digests_v1().into_iter().enumerate() {
        assert!(
            logs.iter()
                .any(|line| line.contains(&format!("rank {rank} {digest}"))),
            "the instance must name every rank image it authenticated: {logs:?}"
        );
    }

    let handle = instance.handle();
    instance
        .destroy()
        .expect("destroy releases in reverse order");

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
    // The fixture's own generation check refuses the handle it just retired,
    // whatever the caller's state machine believes.
    assert_eq!(
        bundle_tp4_fixture::commit_v1(handle),
        BundleStatusV1::StaleHandle
    );
    assert_eq!(
        bundle_tp4_fixture::destroy_v1(handle),
        BundleStatusV1::StaleHandle
    );
}

/// One PrefillChunk and one DecodeStep each emit one full-width finite row, and the
/// shared sampler reads them.
#[test]
fn a_prefill_chunk_then_a_decode_step_each_emit_one_finite_full_vocabulary_row() {
    let (backend, mut instance) = created();
    let tokens = prefill_tokens();
    let seed = state_seed_history_v1();

    let mut prefilled = CollectingSink::new();
    instance
        .execute(&prefill(7), &mut prefilled)
        .expect("the fixture runs one PrefillChunk");

    let row = one_row(&prefilled);
    assert_eq!(row.request_id(), 7);
    assert_eq!(row.logit_count(), u64::from(tp4_fixture_vocabulary_v1()));
    assert_eq!(row.logit_count(), 154_880);
    assert_eq!(row.logit_bits(), expected_row(seed, &tokens).as_slice());

    // The shared sampler is what consumes a row, so it is what says the row is
    // a finite one of exactly the declared width. It refuses either otherwise.
    let logits = row_logits_v1(row.logit_bits());
    let sampled = sample(
        &SamplingParameters::new(0.0).expect("greedy sampling parameters are valid"),
        &logits,
        SampleContext::new(&tokens, 0, &[]),
        &mut SamplingState::new(1),
    )
    .expect("the shared sampler accepts this full-vocabulary row");
    assert_eq!(sampled.token_id(), TP4_FIXTURE_PEAK_TOKEN_ID_V1);
    assert!(sampled.logprob().is_finite());

    instance.commit().expect("the PrefillChunk commits");

    // The committed prefill is the history the decode reads, so its row is a
    // different one and it is the one this fixture declares for that history.
    let after_prefill = fold_tokens_v1(seed, &tokens);
    let mut decoded = CollectingSink::new();
    instance
        .execute(
            &decode(8, TP4_FIXTURE_PEAK_TOKEN_ID_V1, tokens.len() as u64),
            &mut decoded,
        )
        .expect("the fixture runs one DecodeStep");

    let decoded_row = one_row(&decoded);
    assert_eq!(decoded_row.request_id(), 8);
    assert_eq!(decoded_row.logit_count(), 154_880);
    assert_eq!(
        decoded_row.logit_bits(),
        expected_row(after_prefill, &[TP4_FIXTURE_PEAK_TOKEN_ID_V1]).as_slice()
    );
    assert_ne!(
        decoded_row.logit_bits(),
        row.logit_bits(),
        "a committed prefill has to move the row the decode after it emits"
    );

    // Both phases ran through the one bundle-private operation the manifest
    // declares, over the three operands this fixture named, and nothing else.
    let operations = backend.with_device(|device| device.custom_operations().to_vec());
    assert_eq!(operations.len(), 2, "one operation per execution");
    for (id, lent) in &operations {
        assert_eq!(id.as_slice(), TP4_FIXTURE_ROW_OPERATION_ID_V1);
        assert_eq!(lent.len(), TP4_FIXTURE_ROW_OPERANDS_V1);
        assert_eq!(lent[0].length, TP4_FIXTURE_ROW_INPUT_BYTES_V1);
        assert_eq!(lent[1].length, TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1);
        assert_eq!(lent[2].length, u64::from(tp4_fixture_vocabulary_v1()) * 4);
    }
    assert!(
        u64::from(TP4_FIXTURE_DECODE_STEP_TOKENS_V1) * 4 < TP4_FIXTURE_ROW_INPUT_BYTES_V1,
        "a DecodeStep carries fewer tokens than the operand it is lent, which is what the count \
         operand is there to say"
    );

    let logs = logged(&backend);
    assert!(
        logs.iter().any(|line| {
            line.contains(TP4_FIXTURE_ROW_OPERATION_V1)
                && line.contains("PrefillChunk of 32 tokens")
        }),
        "the prefill must name the private operation it ran: {logs:?}"
    );
    assert!(
        logs.iter().any(|line| {
            line.contains(TP4_FIXTURE_ROW_OPERATION_V1) && line.contains("DecodeStep of 1 tokens")
        }),
        "the decode must name the private operation it ran: {logs:?}"
    );
    // Nothing was launched: this fixture computes through its own operation, not
    // through a kernel the host would have to launch.
    assert!(backend.with_device(|device| device.refused_launches().is_empty()));

    instance.commit().expect("the DecodeStep commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A cancelled execution rolls its KV append back and publishes no history, so
/// replaying it produces the same row and a commit after it does not.
#[test]
fn a_cancelled_execution_replays_identically_and_a_committed_one_advances() {
    let (backend, mut instance) = created();
    let tokens = prefill_tokens();
    let seed = state_seed_history_v1();

    let mut prefilled = CollectingSink::new();
    instance
        .execute(&prefill(1), &mut prefilled)
        .expect("the PrefillChunk runs");
    instance.commit().expect("the PrefillChunk commits");
    let after_prefill = fold_tokens_v1(seed, &tokens);

    let mut cancelled = CollectingSink::new();
    instance
        .execute(&decode(2, 11, tokens.len() as u64), &mut cancelled)
        .expect("the DecodeStep runs");
    assert_eq!(
        one_row(&cancelled).logit_bits(),
        expected_row(after_prefill, &[11]).as_slice()
    );
    instance.cancel().expect("the DecodeStep is cancelled");

    // The cancelled step changed neither half of the transaction, so replaying
    // it against the same placement produces exactly the same row.
    let mut replay = CollectingSink::new();
    instance
        .execute(&decode(3, 11, tokens.len() as u64), &mut replay)
        .expect("the replayed DecodeStep runs");
    assert_eq!(
        one_row(&replay).logit_bits(),
        one_row(&cancelled).logit_bits(),
        "a cancelled execution must leave the committed history untouched"
    );
    instance.commit().expect("the replayed DecodeStep commits");

    // This one was committed, so the step after it starts somewhere else.
    let mut third = CollectingSink::new();
    instance
        .execute(&decode(4, 11, tokens.len() as u64 + 1), &mut third)
        .expect("the third step runs");
    assert_ne!(
        one_row(&third).logit_bits(),
        one_row(&replay).logit_bits(),
        "a committed execution must advance the history the next row is read from"
    );
    assert_eq!(
        one_row(&third).logit_bits(),
        expected_row(fold_tokens_v1(after_prefill, &[11]), &[11]).as_slice()
    );

    instance.cancel().expect("the third step is cancelled");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("rolled back the execution of request 2")),
        "a cancel must say which execution it rolled back"
    );
}

/// Destroying an instance with an execution still open is refused, and the
/// refusal resolves nothing.
///
/// Rolling back on the caller's behalf would report a transaction as resolved
/// that its owner never resolved. The ABI wrapper never asks for this, because
/// it finalizes first; what reaches it is whoever owns the native handle, so the
/// handle is what this drives.
#[test]
fn destroying_an_instance_with_an_open_execution_is_refused_and_changes_nothing() {
    let (backend, mut instance) = created();
    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(21), &mut sink)
        .expect("the PrefillChunk runs");
    assert_eq!(sink.rows.len(), 1);
    let held = backend.live_resource_count();

    assert_eq!(
        bundle_tp4_fixture::destroy_v1(instance.handle()),
        BundleStatusV1::InvalidState
    );

    // Nothing was released and nothing was resolved.
    assert_eq!(backend.live_resource_count(), held);
    assert!(
        logged(&backend).iter().any(|line| {
            line.contains("refused a destroy while the execution of request 21 is still open")
                && line.contains("rolled nothing back")
        }),
        "the refusal must say it resolved nothing: {:?}",
        logged(&backend)
    );

    // The execution is still the caller's to finish, and finishing it works.
    instance.cancel().expect("the open execution still cancels");
    instance
        .destroy()
        .expect("a destroy after the caller finalized succeeds");
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
}

/// The same refusal leaves a commit reachable, not just a cancel.
#[test]
fn a_destroy_refused_for_an_open_execution_still_permits_its_commit() {
    let (backend, mut instance) = created();
    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(22), &mut sink)
        .expect("the PrefillChunk runs");

    assert_eq!(
        bundle_tp4_fixture::destroy_v1(instance.handle()),
        BundleStatusV1::InvalidState
    );

    instance
        .commit()
        .expect("the open execution still commits after a refused destroy");
    // The commit landed, so the next row is read from the advanced history.
    let mut next = CollectingSink::new();
    instance
        .execute(&decode(23, 11, prefill_tokens().len() as u64), &mut next)
        .expect("the DecodeStep runs");
    assert_eq!(
        one_row(&next).logit_bits(),
        expected_row(
            fold_tokens_v1(state_seed_history_v1(), &prefill_tokens()),
            &[11]
        )
        .as_slice()
    );
    instance.cancel().expect("the DecodeStep is cancelled");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A batch this fixture cannot run leaves nothing open and no row behind.
#[test]
fn a_batch_this_fixture_cannot_run_is_refused_with_nothing_left_pending() {
    let (backend, mut instance) = created();
    let before = backend.live_resource_count();
    let too_wide = (0..=TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1).collect::<Vec<_>>();

    for (request, reason) in [
        (
            ScheduledBatchV1::prefill(1, &too_wide, 0).encode(),
            "a prefill wider than the declared chunk",
        ),
        (
            ScheduledBatchV1::prefill(2, &[], 0).encode(),
            "a prefill carrying no tokens",
        ),
        (
            ScheduledBatchV1 {
                decode: true,
                ..ScheduledBatchV1::prefill(6, &[1, 2], 0)
            }
            .encode(),
            "a decode of more than one token",
        ),
        (
            ScheduledBatchV1::prefill(3, &prefill_tokens(), 0)
                .with_page_table(&[99])
                .encode(),
            "a page this pool does not have",
        ),
        (
            ScheduledBatchV1::prefill(4, &prefill_tokens(), 0)
                .with_page_table(&[])
                .encode(),
            "an append the host reserved no page for",
        ),
        (b"chopbatch-v1".to_vec(), "a truncated request"),
    ] {
        let mut sink = CollectingSink::new();

        let refusal = instance
            .execute(&request, &mut sink)
            .expect_err(&format!("{reason} must be refused"));

        assert!(
            matches!(refusal, BundleAbiError::Status(_)),
            "{reason} must be refused with a status, and answered {refusal:?}"
        );
        assert!(sink.rows.is_empty(), "{reason} must emit no row");
        assert_eq!(backend.live_resource_count(), before, "{reason}");
    }

    // A refused execution left no transaction open, so the instance still runs
    // and its next commit is the first one it makes.
    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(9), &mut sink)
        .expect("the instance is still usable after a refusal");
    assert_eq!(
        one_row(&sink).logit_bits(),
        expected_row(state_seed_history_v1(), &prefill_tokens()).as_slice(),
        "a refused execution must not have advanced the committed history"
    );
    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A host that refuses the private operation gets no row and no open append.
#[test]
fn a_refused_private_operation_produces_no_row_and_leaves_nothing_pending() {
    let (backend, mut instance) = created();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::RunCustomOperation));
    let mut sink = CollectingSink::new();

    let refusal = instance
        .execute(&prefill(5), &mut sink)
        .expect_err("a refused operation cannot produce a row");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::HostError));
    assert!(sink.rows.is_empty());
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains("private operation produced no result")),
        "the refusal must say the operation produced nothing"
    );

    // Allowing it again makes the same instance run, so nothing was left open
    // and the append the refused execution opened was rolled back.
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::RunCustomOperation));
    instance
        .execute(&prefill(6), &mut sink)
        .expect("the instance runs once the host allows the operation again");
    assert_eq!(sink.rows.len(), 1);
    assert_eq!(
        one_row(&sink).logit_bits(),
        expected_row(state_seed_history_v1(), &prefill_tokens()).as_slice()
    );
    instance.commit().expect("the step commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A prompt longer than one chunk drains: the engine schedules a chunk that
/// expects no row, this fixture runs it and emits none, and the tail after it
/// reads the history that chunk committed.
#[test]
fn a_prefill_chunk_that_expects_no_row_runs_and_commits_without_one() {
    let (backend, mut instance) = created();
    let chunk = prefill_tokens();
    let tail = [7_u32];
    let seed = state_seed_history_v1();

    let mut chunked = CollectingSink::new();
    instance
        .execute(
            &ScheduledBatchV1::prefill(31, &chunk, 0)
                .without_row()
                .encode(),
            &mut chunked,
        )
        .expect("a non-final prefill chunk runs");

    assert!(
        chunked.rows.is_empty(),
        "a chunk the engine expects no row from must emit none"
    );
    assert!(
        backend.with_device(|device| device.custom_operations().is_empty()),
        "a step with no row to emit must not run the operation that produces one"
    );
    instance.commit().expect("the chunk commits");

    // The tail is the step that emits, and it reads the history the chunk left.
    let mut tailed = CollectingSink::new();
    instance
        .execute(
            &ScheduledBatchV1::prefill(31, &tail, chunk.len() as u64).encode(),
            &mut tailed,
        )
        .expect("the authenticated tail runs");

    assert_eq!(
        one_row(&tailed).logit_bits(),
        expected_row(fold_tokens_v1(seed, &chunk), &tail).as_slice(),
        "the tail must read the history the chunk committed"
    );
    assert_eq!(
        backend.with_device(|device| device.custom_operations().len()),
        1,
        "the one operation call belongs to the one step that emitted a row"
    );
    instance.commit().expect("the tail commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// Each token goes into the page its own logical index names, so a decode past
/// the first page reaches the second entry of the table and not the first.
#[test]
fn a_token_is_placed_in_the_page_its_logical_index_names() {
    let (backend, mut instance) = created();
    let full_page = u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1);

    // Everything this prefill carries is inside logical page zero, so a table
    // naming only that page is enough for it.
    let mut prefilled = CollectingSink::new();
    instance
        .execute(
            &ScheduledBatchV1::prefill(41, &prefill_tokens(), 0)
                .with_page_table(&[0])
                .encode(),
            &mut prefilled,
        )
        .expect("a prefill inside the first page runs against a one-page table");
    assert_eq!(prefilled.rows.len(), 1);
    instance.commit().expect("the prefill commits");

    // The decode after it sits at the first position of logical page one, so it
    // is the second entry of the table this fixture has to resolve.
    let mut refused = CollectingSink::new();
    let refusal = instance
        .execute(
            &ScheduledBatchV1::decode(42, 11, full_page)
                .with_page_table(&[0, 99])
                .encode(),
            &mut refused,
        )
        .expect_err("the page this token's logical index names is not in the pool");
    assert!(matches!(refusal, BundleAbiError::Status(_)));
    assert!(refused.rows.is_empty());

    let mut decoded = CollectingSink::new();
    instance
        .execute(
            &ScheduledBatchV1::decode(43, 11, full_page)
                .with_page_table(&[0, 1])
                .encode(),
            &mut decoded,
        )
        .expect("the same decode runs once its logical page is one the pool has");
    assert_eq!(decoded.rows.len(), 1);
    instance.commit().expect("the decode commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// An append that crosses a page boundary is placed in both pages, and one that
/// has nowhere to put the tokens past the boundary is refused before it emits.
#[test]
fn an_append_that_crosses_a_page_boundary_is_placed_in_both_pages() {
    let (backend, mut instance) = created();
    // The sequence stops two tokens short of the first page's end, so this
    // append writes two tokens into page zero and two into page one.
    let previous = u64::from(TP4_FIXTURE_KV_PAGE_TOKENS_V1) - 2;
    let crossing = [41_u32, 42, 43, 44];

    let mut sink = CollectingSink::new();
    instance
        .execute(
            &ScheduledBatchV1::prefill(51, &crossing, previous).encode(),
            &mut sink,
        )
        .expect("an append across a page boundary runs");
    assert_eq!(sink.rows.len(), 1);
    instance.cancel().expect("the crossing append is cancelled");

    for (page_table, reason) in [
        (
            vec![0_u64],
            "a table that stops before the page the later tokens need",
        ),
        (vec![0, 99], "a later page this pool does not have"),
    ] {
        let mut refused = CollectingSink::new();

        let refusal = instance
            .execute(
                &ScheduledBatchV1::prefill(52, &crossing, previous)
                    .with_page_table(&page_table)
                    .encode(),
                &mut refused,
            )
            .expect_err(&format!("{reason} must be refused"));

        assert!(
            matches!(refusal, BundleAbiError::Status(_)),
            "{reason} must be refused with a status, and answered {refusal:?}"
        );
        assert!(refused.rows.is_empty(), "{reason} must emit no row");
    }

    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A KV commit the host refuses after the history was published puts the
/// history back, keeps the execution pending, and replays identically.
///
/// The ABI wrapper poisons an instance whose commit failed and calls nothing
/// else on it, so what a refused commit leaves behind is only reachable through
/// the native handle, and the handle is what this drives.
#[test]
fn a_kv_commit_refused_after_the_history_was_published_restores_it() {
    let (backend, mut instance) = created();
    let tokens = prefill_tokens();
    let seed = state_seed_history_v1();

    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(61), &mut sink)
        .expect("the PrefillChunk runs");
    let row = one_row(&sink).logit_bits().to_vec();

    // Zeroing the page this append reserved is the first device work the commit
    // does, so refusing it refuses the commit after the history was published.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ZeroDeviceBuffer));
    assert_eq!(
        bundle_tp4_fixture::commit_v1(instance.handle()),
        BundleStatusV1::HostError
    );
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ZeroDeviceBuffer));
    assert!(
        logged(&backend).iter().any(|line| {
            line.contains("could not commit its KV append")
                && line.contains("the execution stays pending")
        }),
        "the refusal must say the execution is still open: {:?}",
        logged(&backend)
    );

    // It is still pending, so nothing else may run on this instance.
    let mut second = CollectingSink::new();
    let refusal = api()
        .execute_handle(instance.handle(), &prefill(62), &mut second)
        .expect_err("an instance with a pending execution runs nothing else");
    assert_eq!(
        refusal,
        BundleAbiError::Status(BundleStatusV1::InvalidState)
    );
    assert!(second.rows.is_empty());

    // Cancelling it succeeds, and the replay after it produces exactly the row
    // the first attempt did, which is only true if the history was put back.
    instance.cancel().expect("the pending execution cancels");
    let mut replay = CollectingSink::new();
    instance
        .execute(&prefill(63), &mut replay)
        .expect("the replay runs");
    assert_eq!(
        one_row(&replay).logit_bits(),
        row.as_slice(),
        "a refused commit must leave the committed history where it was"
    );
    assert_eq!(
        one_row(&replay).logit_bits(),
        expected_row(seed, &tokens).as_slice()
    );

    instance.commit().expect("the replay commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
}

/// A history publication the host refuses leaves the committed history where it
/// was, and the execution is still the caller's to finish.
///
/// The refused commit is made through the native handle for the same reason as
/// the test above: the wrapper poisons an instance whose commit failed.
#[test]
fn a_history_publication_the_host_refuses_leaves_the_committed_history_alone() {
    let (backend, mut instance) = created();
    let tokens = prefill_tokens();
    let seed = state_seed_history_v1();

    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(71), &mut sink)
        .expect("the PrefillChunk runs");
    let row = one_row(&sink).logit_bits().to_vec();

    // Publishing the history is a copy this fixture waits on, so refusing the
    // wait refuses the publication itself.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::SynchronizeStream));
    assert_eq!(
        bundle_tp4_fixture::commit_v1(instance.handle()),
        BundleStatusV1::HostError
    );
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::SynchronizeStream));
    assert!(
        logged(&backend).iter().any(|line| {
            line.contains("could not publish its committed history")
                && line.contains("the execution stays pending")
        }),
        "the refusal must say the execution is still open: {:?}",
        logged(&backend)
    );

    instance.cancel().expect("the pending execution cancels");
    let mut replay = CollectingSink::new();
    instance
        .execute(&prefill(72), &mut replay)
        .expect("the replay runs");
    assert_eq!(
        one_row(&replay).logit_bits(),
        row.as_slice(),
        "a refused publication must leave the committed history where it was"
    );
    assert_eq!(
        one_row(&replay).logit_bits(),
        expected_row(seed, &tokens).as_slice()
    );

    instance.commit().expect("the replay commits");
    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(unreleased_resource_count_v1(), 0);
}

/// The private operation is lent exactly the shape its contract declares,
/// whatever the batch carries.
#[test]
fn the_private_operation_is_lent_exactly_the_shape_its_contract_declares() {
    let (backend, mut instance) = created();
    let contract = tp4_fixture_row_contract_v1().expect("the row contract is valid");
    let [tokens, count, logits] = contract.tensors() else {
        panic!(
            "this operation reads a fixed-capacity token operand and its count, and writes one row"
        );
    };
    let declared = [
        tokens.dimensions()[0] * 4,
        count.dimensions()[0] * 4,
        logits.dimensions()[0] * 4,
    ];

    // One PrefillChunk and one DecodeStep, whose token counts differ by 31.
    let mut sink = CollectingSink::new();
    instance
        .execute(&prefill(81), &mut sink)
        .expect("the PrefillChunk runs");
    instance.commit().expect("the PrefillChunk commits");
    instance
        .execute(&decode(82, 11, prefill_tokens().len() as u64), &mut sink)
        .expect("the DecodeStep runs");
    instance.commit().expect("the DecodeStep commits");

    let operations = backend.with_device(|device| device.custom_operations().to_vec());
    assert_eq!(operations.len(), 2, "one operation per emitted row");
    for (id, lent) in &operations {
        assert_eq!(id.as_slice(), TP4_FIXTURE_ROW_OPERATION_ID_V1);
        assert_eq!(
            lent.iter().map(|range| range.length).collect::<Vec<_>>(),
            declared,
            "the lend must be the declared shape, whatever the batch carries"
        );
    }
    assert_eq!(
        declared[0],
        u64::from(TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1) * 4,
        "the token operand is one fixed-capacity PrefillChunk of identifiers"
    );

    instance.destroy().expect("the instance is released");
    assert_eq!(backend.live_resource_count(), 0);
}

/// A package that does not declare itself this fixture is refused outright.
#[test]
fn a_package_without_the_fixture_marker_role_is_refused_before_anything_is_acquired() {
    let topology = HostTopologyV1::new(tp4_ordinals()).expect("the TP4 test topology is valid");
    let page_layout = tp4_fixture_kv_layout_v1()
        .expect("the declared KV layout is valid")
        .declared_page_layout(&tp4_fixture_topology_v1().expect("the declared topology is valid"))
        .expect("the declared KV layout describes this topology");
    let config = HostInstanceConfigV1::new(topology)
        .with_kv_contract(DeclaredKvContractV1::new(
            TP4_FIXTURE_KV_LAYOUT_ID_V1.as_bytes().to_vec(),
            page_layout,
        ))
        .with_state_contract(DeclaredStateContractV1::new(
            TP4_FIXTURE_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            TP4_FIXTURE_STATE_BYTES_V1,
            TP4_FIXTURE_STATE_ALIGNMENT_V1,
        ))
        .with_compiled_image(
            TP4_FIXTURE_PROGRAM_ROLE_V1.as_bytes(),
            program_component_bytes(),
            program_image_description(),
        )
        .expect("the program component is valid");
    let configuration = config.encode_v1(TP4_FIXTURE_TOPOLOGY_ID_V1.as_bytes());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));

    let refusal = api()
        .create(table, &configuration)
        .err()
        .expect("a package that is not this fixture is refused");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::Unsupported));
    assert_eq!(backend.live_resource_count(), 0);
    assert!(
        logged(&backend)
            .iter()
            .any(|line| line.contains(TP4_FIXTURE_MARKER_ROLE_V1)),
        "the refusal must name the role that gates this module: {:?}",
        logged(&backend)
    );
}

/// A rank image this build cannot regenerate is refused, and the row is never
/// computed against bytes the host did not verify.
#[test]
fn a_rank_image_this_build_did_not_produce_is_refused() {
    // One rank's image is corrupted inside the component the host uploads, so
    // three ranks authenticate and the fourth does not.
    let mut tampered = program_component_bytes();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    let config = instance_config_with_program(tampered);
    let configuration = config.encode_v1(TP4_FIXTURE_TOPOLOGY_ID_V1.as_bytes());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));

    let refusal = api()
        .create(table, &configuration)
        .err()
        .expect("an image this build cannot regenerate is refused");

    assert_eq!(refusal, BundleAbiError::Status(BundleStatusV1::Unsupported));
    assert_eq!(backend.live_resource_count(), 0);
    assert!(
        logged(&backend)
            .iter()
            .any(|line| { line.contains("rank 3 program image this build cannot") }),
        "the refusal must name the rank whose image is not this fixture's: {:?}",
        logged(&backend)
    );
}

/// A host this fixture does not serve costs it no resource at all.
#[test]
fn a_topology_this_fixture_does_not_serve_is_refused_before_anything_is_acquired() {
    let topology = HostTopologyV1::new(vec![0]).expect("a one-rank topology is valid");
    let config =
        HostInstanceConfigV1::new(topology).with_state_contract(DeclaredStateContractV1::new(
            TP4_FIXTURE_STATE_LAYOUT_ID_V1.as_bytes().to_vec(),
            TP4_FIXTURE_STATE_BYTES_V1,
            TP4_FIXTURE_STATE_ALIGNMENT_V1,
        ));
    let configuration = config.encode_v1(b"tp1");
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
            .any(|line| line.contains("serves topology tp4")),
        "the refusal must name the topology this fixture serves"
    );
}
