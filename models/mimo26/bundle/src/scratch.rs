//! Fixed rank buffers for the largest MiMo prefill chunk and decode graph.

use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};
use std::ops::{Deref, DerefMut};

use crate::topology::Tp;

use model_mimo26_contract::resources::{BufferLayout, CommBuffers, ScratchBuffers, ScratchLayout};
pub use model_mimo26_contract::resources::{
    COMM_BUCKETS, HIDDEN_SLACK_ROWS, comm_inbox_bytes, hidden_epoch_bytes, hidden_flag_bytes,
};
#[cfg(test)]
use model_mimo26_contract::resources::{PADDED_ROUTES_FLASH, ROUTES_FLASH, TOKENS};

pub fn bind_hidden(
    layout: BufferLayout,
    allocate: &mut impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    let span = allocate(layout.allocation_bytes())?;
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset,
        layout.span_bytes(),
    ))
}
/// The span of a rank's comm inbox that comm state `index` of
/// `COMM_BUCKETS` owns. Collectives of different states follow each other
/// with no barrier across ranks (a verify's target state then its drafter's,
/// a prefill chunk then the step's decode), and a rank that finished one call
/// may write its next while a peer still reads the previous one; the
/// protocols keep that safe only within a state. So no two states share
/// inbox bytes.
pub fn comm_inbox(inbox: DeviceBufferSpanV1, index: usize) -> DeviceBufferSpanV1 {
    let (offset, length) = model_mimo26_contract::resources::comm_inbox_range(inbox.length, index);
    DeviceBufferSpanV1::new(inbox.buffer, inbox.offset + offset, length)
}

pub type CommState = CommBuffers<DeviceBufferSpanV1>;

#[derive(Clone)]
pub struct RankScratch {
    /// Read-only GA and SWA tables, installed before KV sizing.
    pub rope_tables: Option<[DeviceBufferSpanV1; 2]>,
    pub buffers: ScratchBuffers<DeviceBufferSpanV1>,
}

impl Deref for RankScratch {
    type Target = ScratchBuffers<DeviceBufferSpanV1>;
    fn deref(&self) -> &Self::Target {
        &self.buffers
    }
}

impl DerefMut for RankScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffers
    }
}

impl RankScratch {
    /// Buffers of one rank at `tp`'s per-rank shapes.
    /// `exchange_through`: comm buckets up to this many rows get logits exchange state.
    pub fn allocate(
        exchange_through: u32,
        tp: Tp,
        allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        Ok(Self {
            buffers: bundle_runtime::scratch::bind(
                ScratchLayout::new(tp, exchange_through)?,
                allocate,
            )?,
            rope_tables: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::*;

    use super::*;
    use crate::contract::RING_TOKENS;

    #[test]
    fn allocation_layout_matches_order_handles_spans_and_failures() {
        for tp in crate::topology::all() {
            for exchange in [0, 1, 2, 31, 32, 33, 127, 128, 255, 256, 257, 8192] {
                let expected = ScratchLayout::new(tp, exchange).unwrap().allocations();
                let mut allocated = Vec::new();
                let scratch = RankScratch::allocate(exchange, tp, |bytes| {
                    allocated.push(bytes);
                    Ok(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1::from_parts(1, allocated.len() as u32),
                        4096,
                        bytes,
                    ))
                })
                .unwrap();
                assert_eq!(
                    allocated,
                    expected
                        .iter()
                        .map(|size| size.allocation_bytes())
                        .collect::<Vec<_>>()
                );
                let mut spans = vec![
                    scratch.qkv,
                    scratch.q,
                    scratch.k_new,
                    scratch.v_new,
                    scratch.attention,
                    scratch.hidden_partial,
                    scratch.dense_gate,
                    scratch.router_logits,
                    scratch.topk_ids,
                    scratch.topk_weights,
                    scratch.expert_counts,
                    scratch.expert_offsets,
                    scratch.sorted_rows,
                    scratch.padded_rows,
                    scratch.gathered,
                    scratch.routed_gate,
                    scratch.routed_down,
                    scratch.combined,
                    scratch.comm_inbox,
                ];
                for state in &scratch.comm_state {
                    spans.extend([
                        state.hidden_flags,
                        state.hidden_epoch,
                        state.gather_rows_flags,
                        state.gather_rows_epoch,
                    ]);
                    if let Some(exchange) = state.exchange_rows {
                        spans.extend([exchange.flags, exchange.epoch]);
                    }
                }
                spans.extend([
                    scratch.active_experts,
                    scratch.all_experts,
                    scratch.unit_scale,
                    scratch.gemm_semaphores,
                    scratch.attention_fold_counters,
                    scratch.route_ticket,
                ]);
                assert_eq!(spans.len(), expected.len());
                for (index, (span, size)) in spans.iter().zip(&expected).enumerate() {
                    assert_eq!(span.buffer.slot(), index as u32 + 1);
                    assert_eq!((span.offset, span.length), (4096, size.span_bytes()));
                }
                for fail_at in 0..expected.len() {
                    let mut calls = 0;
                    let result = RankScratch::allocate(exchange, tp, |bytes| {
                        let index = calls;
                        calls += 1;
                        if index == fail_at {
                            Err(BundleAbiError::InvalidRange("allocation probe"))
                        } else {
                            Ok(DeviceBufferSpanV1::new(
                                BundleDeviceBufferHandleV1::from_parts(1, 1),
                                0,
                                bytes,
                            ))
                        }
                    });
                    assert!(matches!(
                        result,
                        Err(BundleAbiError::InvalidRange("allocation probe"))
                    ));
                    assert_eq!(calls, fail_at + 1);
                }
            }
        }
    }

    #[test]
    fn scratch_covers_routed_and_vocabulary_capacities() {
        for tp in Tp::SUPPORTED {
            scratch_covers_capacities_at(tp);
        }
    }

    fn scratch_covers_capacities_at(tp: Tp) {
        let ranks = tp.ranks();
        let mut bytes = Vec::new();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let scratch = RankScratch::allocate(0, tp, |length| {
            bytes.push(length);
            Ok(DeviceBufferSpanV1::new(buffer, 0, length))
        })
        .unwrap();
        assert_eq!(bytes.len(), 65);
        // The parallel route's arrival count is its own allocation, so no GEMM
        // semaphore or other buffer shares it.
        assert_eq!(
            scratch.route_ticket.length,
            model_mimo26_contract::sm120::moe::PARALLEL_ROUTE_TICKET_BYTES
        );
        assert_eq!(
            bytes.last().copied(),
            Some(model_mimo26_contract::sm120::moe::PARALLEL_ROUTE_TICKET_BYTES)
        );
        assert!(
            scratch
                .comm_state
                .iter()
                .all(|state| state.exchange_rows.is_none())
        );
        let mut exchange_bytes = Vec::new();
        let exchanged = RankScratch::allocate(crate::contract::DECODE_ENTRY_ROWS, tp, |length| {
            exchange_bytes.push(length);
            Ok(DeviceBufferSpanV1::new(buffer, 0, length))
        })
        .unwrap();
        // Two more buffers for each of the six decode buckets.
        assert_eq!(exchange_bytes.len(), 65 + 12);
        assert!(exchanged.comm_state[..6].iter().all(|state| {
            state.exchange_rows.is_some_and(|exchange| {
                exchange.flags.length == crate::sm120::comm::EXCHANGE_ROWS_FLAG_BYTES
                    && exchange.epoch.length == crate::sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES
            })
        }));
        // Wide verify buckets and the prefill state have none.
        assert!(
            exchanged.comm_state[6..]
                .iter()
                .all(|state| state.exchange_rows.is_none())
        );
        // Exchanged verify rows add the three wide verify buckets.
        let mut verify_bytes = Vec::new();
        let verified = RankScratch::allocate(crate::contract::VERIFY_TOKENS, tp, |length| {
            verify_bytes.push(length);
            Ok(DeviceBufferSpanV1::new(buffer, 0, length))
        })
        .unwrap();
        assert_eq!(verify_bytes.len(), 65 + 18);
        assert!(
            verified.comm_state[..9]
                .iter()
                .all(|state| state.exchange_rows.is_some())
        );
        assert!(verified.comm_state[9].exchange_rows.is_none());
        assert_eq!(scratch.gathered.length, PADDED_ROUTES_FLASH * 4096 * 2);
        assert_eq!(scratch.routed_down.length, ROUTES_FLASH * 4096 * 2);
        assert_eq!(scratch.comm_state.len(), COMM_BUCKETS.len());
        assert_eq!(scratch.comm_inbox.length, comm_inbox_bytes(tp));
        // No more than the fixed states and a prefill state of 128 MiB.
        assert!(scratch.comm_inbox.length <= (96 + 128) << 20, "{ranks}");
        // Every comm state owns a disjoint span that holds its protocols.
        let mut end = scratch.comm_inbox.offset;
        for (index, &bucket) in COMM_BUCKETS.iter().enumerate() {
            let span = comm_inbox(scratch.comm_inbox, index);
            assert_eq!(span.offset, end);
            end = span.offset + span.length;
            let push = crate::sm120::comm::push_buffers(ranks, bucket.min(32)).unwrap();
            let mut needed = push.inbox_bytes;
            if bucket > 32 {
                let phased = crate::sm120::comm::phased_buffers(
                    ranks,
                    u64::from(bucket) * 4096,
                    kernel_core::contract::DType::Bf16,
                )
                .unwrap();
                needed = needed.max(phased.inbox_bytes);
            }
            assert!(needed <= span.length, "bucket {bucket} at {ranks} ranks");
            // The planner's sizes for every row count the state takes.
            let state = crate::sm120::comm::fused_state_buffers(ranks, bucket).unwrap();
            let comm = &scratch.comm_state[index];
            assert!(
                state.inbox_bytes <= span.length,
                "bucket {bucket} at {ranks} ranks"
            );
            assert!(
                state.flag_bytes <= comm.hidden_flags.length,
                "bucket {bucket} at {ranks} ranks"
            );
            assert!(
                state.epoch_bytes <= comm.hidden_epoch.length,
                "bucket {bucket} at {ranks} ranks"
            );
        }
        assert_eq!(end, scratch.comm_inbox.offset + scratch.comm_inbox.length);
        let prefill = COMM_BUCKETS.len() - 1;
        assert_eq!(
            scratch.comm_state[prefill].gather_rows_flags.length,
            crate::sm120::comm::GATHER_ROWS_FLAG_BYTES
        );
        assert_eq!(
            scratch.comm_state[prefill].gather_rows_epoch.length,
            crate::sm120::comm::GATHER_ROWS_EPOCH_BYTES
        );
        assert_eq!(scratch.all_experts.length, 257 * 4);
        assert!(scratch.qkv.length >= TOKENS * tp.qkv_rows(false) * 2);
        assert!(scratch.qkv.length >= TOKENS * tp.qkv_rows(true) * 2);
        if tp == Tp::TP4 {
            assert!(
                scratch
                    .comm_state
                    .iter()
                    .zip(COMM_BUCKETS)
                    .all(|(state, bucket)| state.hidden_flags.length == 64 * u64::from(bucket))
            );
        }
        assert!(u64::from(crate::contract::DECODE_BATCH) <= TOKENS);
        assert_eq!(RING_TOKENS, 128);
    }
    // Independent formulas and protocol sizes, kept separate from ScratchLayout.
    fn allocation_expectations(tp: Tp, exchange: u32) -> Vec<(&'static str, u64, u64)> {
        let g = tp.geometry;
        let tokens = 8192;
        let hidden = u64::from(g.hidden());
        let experts = u64::from(g.experts());
        let routes = tokens * u64::from(g.top_k());
        let padded = routes + experts * 63;
        let q = [true, false].map(|global| tp.q_heads_for(global) * tp.qk_dim(global));
        let k = [true, false].map(|global| tp.kv_heads(global) * tp.qk_dim(global));
        let v = [true, false].map(|global| tp.kv_heads(global) * tp.value_dim(global));
        let a = [true, false].map(|global| tp.q_heads_for(global) * tp.value_dim(global));
        let full = |name, bytes| (name, bytes, bytes);
        let slack = |name| (name, 12288 * hidden * 2, tokens * hidden * 2);
        let mut expected = vec![
            full(
                "qkv",
                tokens * (q[0] + k[0] + v[0]).max(q[1] + k[1] + v[1]) * 2,
            ),
            full("q", tokens * q[0].max(q[1]) * 2),
            full("k_new", tokens * k[0].max(k[1]) * 2),
            full("v_new", tokens * v[0].max(v[1]) * 2),
            full("attention", tokens * a[0].max(a[1]) * 2),
            slack("hidden_partial"),
            full("dense_gate", tokens * tp.dense_intermediate() * 2),
            full("router_logits", tokens * experts * 4),
            full("topk_ids", routes * 4),
            full("topk_weights", routes * 4),
            full("expert_counts", experts * 4),
            full("expert_offsets", (experts + 1) * 4),
            full("sorted_rows", padded * 4),
            full("padded_rows", 4),
            full("gathered", padded * hidden * 2),
            full("routed_gate", routes * tp.expert_intermediate() * 2),
            full("routed_down", routes * hidden * 2),
            slack("combined"),
            full("comm_inbox", 168 << 20),
        ];
        for bucket in [1, 2, 4, 8, 16, 32, 64, 128, 256, 8192] {
            expected.extend([
                full(
                    "hidden_flags",
                    16 * u64::from(tp.ranks()) * u64::from(bucket),
                ),
                full("hidden_epoch", 4 * u64::from(bucket)),
                full("gather_rows_flags", 288),
                full("gather_rows_epoch", 32),
            ]);
            if bucket <= exchange {
                expected.extend([full("exchange.flags", 512), full("exchange.epoch", 32)]);
            }
        }
        let heads = [true, false]
            .map(|global| Tp::of_geometry(g, 2).unwrap().kv_heads(global))
            .into_iter()
            .max()
            .unwrap();
        expected.extend([
            full("active_experts", (experts + 1) * 4),
            full("all_experts", (experts + 1) * 4),
            full("unit_scale", 4),
            full("gemm_semaphores", tokens * hidden / 128 * 4),
            full("attention_fold_counters", tokens * heads * 4),
            full("route_ticket", 4),
        ]);
        expected
    }

    fn observed_fields(s: &RankScratch) -> Vec<DeviceBufferSpanV1> {
        let mut spans = vec![
            s.qkv,
            s.q,
            s.k_new,
            s.v_new,
            s.attention,
            s.hidden_partial,
            s.dense_gate,
            s.router_logits,
            s.topk_ids,
            s.topk_weights,
            s.expert_counts,
            s.expert_offsets,
            s.sorted_rows,
            s.padded_rows,
            s.gathered,
            s.routed_gate,
            s.routed_down,
            s.combined,
            s.comm_inbox,
        ];
        for state in &s.comm_state {
            spans.extend([
                state.hidden_flags,
                state.hidden_epoch,
                state.gather_rows_flags,
                state.gather_rows_epoch,
            ]);
            if let Some(exchange) = state.exchange_rows {
                spans.extend([exchange.flags, exchange.epoch]);
            }
        }
        spans.extend([
            s.active_experts,
            s.all_experts,
            s.unit_scale,
            s.gemm_semaphores,
            s.attention_fold_counters,
            s.route_ticket,
        ]);
        spans
    }

    #[test]
    fn every_scratch_field_and_inbox_alias_matches_independent_sizes() {
        for tp in crate::topology::all() {
            for exchange in [0, 1, 2, 31, 32, 33, 127, 128, 255, 256, 257, 8192] {
                let expected = allocation_expectations(tp, exchange);
                let mut physical = Vec::new();
                let scratch = RankScratch::allocate(exchange, tp, |bytes| {
                    let index = physical.len();
                    physical.push(bytes);
                    Ok(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1::from_parts(7, index as u32 + 1),
                        4096 + index as u64 * 256,
                        bytes,
                    ))
                })
                .unwrap();
                assert!(
                    scratch.rope_tables.is_none(),
                    "tables are installed by create"
                );
                let spans = observed_fields(&scratch);
                assert_eq!(spans.len(), expected.len());
                assert_eq!(physical.len(), expected.len());
                for (index, ((name, allocated, visible), span)) in
                    expected.iter().zip(&spans).enumerate()
                {
                    assert_eq!(physical[index], *allocated, "{tp:?} {exchange} {name}");
                    assert_eq!(
                        *span,
                        DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_parts(7, index as u32 + 1),
                            4096 + index as u64 * 256,
                            *visible,
                        ),
                        "{tp:?} {exchange} {name}"
                    );
                }
                let mut offset = scratch.comm_inbox.offset;
                for (index, length) in [8, 8, 8, 8, 8, 8, 16, 16, 16, 72]
                    .map(|mib| mib << 20)
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(
                        comm_inbox(scratch.comm_inbox, index),
                        DeviceBufferSpanV1::new(scratch.comm_inbox.buffer, offset, length)
                    );
                    offset += length;
                }
                assert_eq!(
                    offset,
                    scratch.comm_inbox.offset + scratch.comm_inbox.length
                );
            }
        }
    }

    #[derive(Default)]
    struct AllocationHost {
        calls: std::cell::Cell<usize>,
        fail_at: usize,
        allocated: std::cell::RefCell<Vec<(BundleDeviceBufferHandleV1, u64)>>,
        released: std::cell::RefCell<Vec<BundleDeviceBufferHandleV1>>,
    }

    impl HostServicesBackendV1 for AllocationHost {
        fn allocate_device_buffer(
            &self,
            _pool: BundleMemoryPoolHandleV1,
            _stream: BundleStreamHandleV1,
            bytes: u64,
            alignment: u64,
        ) -> HostServiceResultV1<BundleDeviceBufferHandleV1> {
            assert_eq!(alignment, 256);
            let index = self.calls.get();
            self.calls.set(index + 1);
            if index == self.fail_at {
                return Err(BundleStatusV1::HostError);
            }
            let handle = BundleDeviceBufferHandleV1::from_parts(7, index as u32 + 1);
            self.allocated.borrow_mut().push((handle, bytes));
            Ok(handle)
        }

        fn release_device_buffer(
            &self,
            buffer: BundleDeviceBufferHandleV1,
        ) -> HostServiceResultV1<()> {
            self.released.borrow_mut().push(buffer);
            Ok(())
        }

        bundle_runtime::unused_host_services! {
            log(_message: &[u8]) -> ();
            select_context(_rank: u32,
        _device_ordinal: i32,) -> BundleContextHandleV1;
            release_context(_context: BundleContextHandleV1) -> ();
            select_memory_pool(_context: BundleContextHandleV1,) -> BundleMemoryPoolHandleV1;
            set_memory_pool_peer_access(_pool: BundleMemoryPoolHandleV1,
        _peer_context: BundleContextHandleV1,
        _enabled: bool,) -> ();
            release_memory_pool(_pool: BundleMemoryPoolHandleV1) -> ();
            open_compiled_image(_component_role: &[u8],) -> BundleCompiledImageHandleV1;
            upload_rank_arena(_context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        _rank: u32,) -> BundleRankArenaHandleV1;
            release_rank_arena(_arena: BundleRankArenaHandleV1) -> ();
            resolve_compiled_image_view(_image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        _rank: u32,
        _view_id: &[u8],) -> CompiledImageSpanV1;
            release_compiled_image(_image: BundleCompiledImageHandleV1) -> ();
            acquire_peer_access(_owner_context: BundleContextHandleV1,
        _peer_context: BundleContextHandleV1,) -> BundlePeerAccessHandleV1;
            release_peer_access(_peer_access: BundlePeerAccessHandleV1) -> ();
            create_stream(_context: BundleContextHandleV1,
        _stream_kind: u32,) -> BundleStreamHandleV1;
            select_stage_stream(_context: BundleContextHandleV1,
        _rank: u32,
        _stage_id: &[u8],
        _stream_kind: u32,) -> BundleStreamHandleV1;
            synchronize_stream(_stream: BundleStreamHandleV1) -> ();
            release_stream(_stream: BundleStreamHandleV1) -> ();
            create_event(_context: BundleContextHandleV1,) -> BundleEventHandleV1;
            record_event(_event: BundleEventHandleV1,
        _stream: BundleStreamHandleV1,) -> ();
            wait_event(_stream: BundleStreamHandleV1,
        _event: BundleEventHandleV1,) -> ();
            synchronize_event(_event: BundleEventHandleV1) -> ();
            release_event(_event: BundleEventHandleV1) -> ();
            copy_host_to_device(_stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: &[u8],) -> ();
            copy_device_to_host(_stream: BundleStreamHandleV1,
        _destination: &mut [u8],
        _source: DeviceBufferSpanV1,) -> ();
            copy_device_to_device(_stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: DeviceBufferSpanV1,) -> ();
            copy_peer_to_peer(_stream: BundleStreamHandleV1,
        _destination_context: BundleContextHandleV1,
        _destination: DeviceBufferSpanV1,
        _source_context: BundleContextHandleV1,
        _source: DeviceBufferSpanV1,) -> ();
            zero_device_buffer(_stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,) -> ();
            write_device_pointer_table(_stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _slots: &[Option<DeviceBufferSpanV1>],) -> ();
            load_cuda_module(_context: BundleContextHandleV1,
        _module_bytes: &[u8],) -> BundleCudaModuleHandleV1;
            resolve_cuda_symbol(_module: BundleCudaModuleHandleV1,
        _symbol_name: &[u8],) -> BundleCudaSymbolHandleV1;
            configure_cuda_symbol(_symbol: BundleCudaSymbolHandleV1,
        _dynamic_shared_memory_bytes: u32,) -> ();
            launch_cuda_symbol(_stream: BundleStreamHandleV1,
        _symbol: BundleCudaSymbolHandleV1,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],) -> ();
            release_cuda_symbol(_symbol: BundleCudaSymbolHandleV1) -> ();
            release_cuda_module(_module: BundleCudaModuleHandleV1) -> ();
            allocate_physical_kv(_request: &PhysicalKvRequestV1,) -> BundlePhysicalKvHandleV1;
            create_kv_request(_kv: BundlePhysicalKvHandleV1,
        _quota_tokens: u64,) -> BundleKvRequestHandleV1;
            prepare_kv(_request: BundleKvRequestHandleV1,
        _operation: &[u8],) -> BundleKvTransactionHandleV1;
            commit_kv(_transaction: BundleKvTransactionHandleV1) -> ();
            cancel_kv(_transaction: BundleKvTransactionHandleV1) -> ();
            resolve_kv_view(_request: BundleKvRequestHandleV1,
        _rank: u32,
        _view_id: &[u8],) -> DeviceBufferSpanV1;
            release_kv_request(_request: BundleKvRequestHandleV1) -> ();
            release_physical_kv(_kv: BundlePhysicalKvHandleV1) -> ();
            allocate_state(_request: &StateRequestV1) -> BundleStateHandleV1;
            resolve_state_view(_state: BundleStateHandleV1,
        _rank: u32,) -> DeviceBufferSpanV1;
            release_state(_state: BundleStateHandleV1) -> ();
            create_collective(_request: &CollectiveRequestV1,) -> BundleCollectiveHandleV1;
            execute_collective(_collective: BundleCollectiveHandleV1,
        _stream: BundleStreamHandleV1,
        _operation: &[u8],
        _arguments: &[KernelArgumentV1],) -> ();
            release_collective(_collective: BundleCollectiveHandleV1) -> ();
            run_custom_operation(_context: BundleContextHandleV1,
        _stream: BundleStreamHandleV1,
        _completion: CustomCompletionV1,
        _operation_id: &[u8],
        _operands: &[CustomOperandV1],
        _operation: &CustomOperationV1<'_>,) -> i32;        }
    }

    #[test]
    fn every_allocation_failure_releases_exactly_the_successful_prefix() {
        use bundle_runtime::resources::{allocate_span, release_all};
        for tp in crate::topology::all() {
            for exchange in [0, 1, 2, 31, 32, 33, 127, 128, 255, 256, 257, 8192] {
                let expected = allocation_expectations(tp, exchange);
                // Include the successful end so teardown of a complete scratch is pinned too.
                for fail_at in 0..=expected.len() {
                    let host = std::rc::Rc::new(AllocationHost {
                        fail_at,
                        ..Default::default()
                    });
                    let table = HostServicesTableV1::new(std::rc::Rc::clone(&host));
                    let services = table.caller();
                    let mut held = Vec::new();
                    let result = RankScratch::allocate(exchange, tp, |bytes| {
                        allocate_span(
                            services,
                            BundleMemoryPoolHandleV1::from_parts(1, 1),
                            BundleStreamHandleV1::from_parts(1, 1),
                            bytes,
                            &mut held,
                        )
                    });
                    if fail_at < expected.len() {
                        assert!(matches!(
                            result,
                            Err(BundleAbiError::Status(BundleStatusV1::HostError))
                        ));
                        assert_eq!(host.calls.get(), fail_at + 1);
                    } else {
                        assert!(result.is_ok());
                        assert_eq!(host.calls.get(), expected.len());
                    }
                    assert_eq!(held.len(), fail_at);
                    let allocated = host.allocated.borrow();
                    assert_eq!(allocated.len(), fail_at);
                    for (index, &(handle, bytes)) in allocated.iter().enumerate() {
                        assert_eq!(
                            handle,
                            BundleDeviceBufferHandleV1::from_parts(7, index as u32 + 1)
                        );
                        assert_eq!(bytes, expected[index].1);
                    }
                    release_all(services, &mut held).unwrap();
                    assert!(held.is_empty());
                    assert_eq!(
                        *host.released.borrow(),
                        allocated
                            .iter()
                            .rev()
                            .map(|&(handle, _)| handle)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
