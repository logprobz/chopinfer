//! A logical zero-filled mapped table with bounded gather allocations, or
//! one whose rows' bytes follow their ids (`MappedHost::with_row_ids`).
//! All other services use the ordinary fake host, including span checks.
//!
//! The fake gather writes its rows when a consumer stream waits, but the
//! host's copies of a gather's ids and rows run asynchronously from its
//! enqueue. So every device access between the two is checked against the
//! pending spans (`MappedHost::touch`): a launch, a graph replay (through
//! the spans its captured launches named), a copy, a zeroing, a pointer
//! table and the spans it names, a collective or a custom operation that
//! touches a pending gather's ids before it completes, or a plane's
//! destination rows before its stream waits, is a race (`FakeHost::races`).
use super::*;
use runtime_bundle_abi::{BundleMappedGatherHandleV1, MappedGatherPlaneV1, MappedGatherStatsV1};
use std::collections::BTreeMap;

/// Services backed by the model fake device, with optional encoder support.
pub(in crate::fake_host_tests) trait MappedServices:
    HostServicesBackendV1
{
    fn fake(&self) -> &FakeHost;
}
impl MappedServices for Host {
    fn fake(&self) -> &FakeHost {
        self
    }
}

struct Gather {
    producer: BundleStreamHandleV1,
    indices: DeviceBufferSpanV1,
    planes: Vec<MappedGatherPlaneV1>,
    pending: Option<u64>,
    /// The ids the enqueue read, with `row_ids`.
    ids: Vec<u32>,
    waited: std::collections::BTreeSet<u64>,
    waited_at: usize,
}

/// Device spans the race check follows (`MappedHost::touch`).
#[derive(Default)]
struct Spans {
    /// The spans each open capture's launches touch, by capturing stream.
    capturing: BTreeMap<u32, Vec<DeviceBufferSpanV1>>,
    /// The spans each captured graph's launches touch, by graph.
    graphs: BTreeMap<u64, Vec<DeviceBufferSpanV1>>,
    /// Pointer tables on the device by (buffer, offset): their length and
    /// the spans their slots name, which an access of the table touches.
    tables: BTreeMap<(u64, u64), (u64, Vec<DeviceBufferSpanV1>)>,
}

fn overlaps(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> bool {
    a.buffer.raw == b.buffer.raw && a.offset < b.offset + b.length && b.offset < a.offset + a.length
}

/// The device spans a launch's arguments name.
fn argument_spans(arguments: &[KernelArgumentV1]) -> Vec<DeviceBufferSpanV1> {
    arguments
        .iter()
        .flat_map(|argument| match argument {
            KernelArgumentV1::DeviceBuffer(span) => vec![*span],
            KernelArgumentV1::TensorMap(map) => vec![map.buffer],
            KernelArgumentV1::DevicePointerArray(slots) => {
                slots.iter().flatten().copied().collect()
            }
            _ => Vec::new(),
        })
        .collect()
}

pub(in crate::fake_host_tests) struct MappedHost {
    host: Rc<dyn MappedServices>,
    gathers: std::cell::RefCell<std::collections::BTreeMap<u64, Gather>>,
    spans: std::cell::RefCell<Spans>,
    peer_copies: bool,
    delayed_completion: bool,
    completed_at: std::cell::Cell<usize>,
    fail_gather: std::cell::Cell<Option<(&'static str, usize)>>,
    row_ids: bool,
}
impl MappedHost {
    pub(in crate::fake_host_tests) fn new<H: MappedServices + 'static>(host: Rc<H>) -> Self {
        Self {
            host,
            gathers: Default::default(),
            spans: Default::default(),
            peer_copies: false,
            delayed_completion: false,
            completed_at: Default::default(),
            fail_gather: Default::default(),
            row_ids: false,
        }
    }

    /// Each gathered row's bytes follow its id, and a prepared gather reads
    /// the ids the bundle uploaded when it is enqueued, as the host's id
    /// copy does; ids a kernel wrote read as row 0. A gather of the wrong
    /// rows then lands other bytes than the synchronous gather.
    pub(in crate::fake_host_tests) fn with_row_ids<H: MappedServices + 'static>(
        host: Rc<H>,
    ) -> Self {
        Self {
            row_ids: true,
            ..Self::new(host)
        }
    }

    /// This host, failing the `after`-th later call of `stage` (`enqueue`,
    /// `wait` or `complete`).
    pub(in crate::fake_host_tests) fn failing(self, stage: &'static str, after: usize) -> Self {
        self.fail_gather.set(Some((stage, after)));
        self
    }

    /// Every pending gather's spans that no other device access may touch:
    /// its ids until it completes, and each plane's destination rows until
    /// the plane's stream waits for them.
    fn pending(&self) -> Vec<(DeviceBufferSpanV1, String)> {
        let mut pending = Vec::new();
        for (handle, gather) in self.gathers.borrow().iter() {
            let Some(rows) = gather.pending else {
                continue;
            };
            pending.push((gather.indices, format!("gather {handle} ids")));
            for plane in &gather.planes {
                if !gather.waited.contains(&plane.stream.raw) {
                    let rows_span = DeviceBufferSpanV1::new(
                        plane.destination.buffer,
                        plane.destination.offset,
                        rows * plane.row_bytes,
                    );
                    let view = String::from_utf8_lossy(&plane.view_id);
                    pending.push((
                        rows_span,
                        format!("gather {handle} rank {} {view}", plane.rank),
                    ));
                }
            }
        }
        pending
    }

    /// A device access of `spans` on `stream`, with the spans a pointer
    /// table among them names: kept for the graph an open capture builds,
    /// or else checked against every pending gather.
    fn touch(
        &self,
        what: &str,
        stream: BundleStreamHandleV1,
        spans: impl IntoIterator<Item = DeviceBufferSpanV1>,
    ) {
        let mut touched: Vec<_> = spans.into_iter().collect();
        let mut recorded = self.spans.borrow_mut();
        let named: Vec<_> = touched
            .iter()
            .flat_map(|&span| {
                recorded
                    .tables
                    .range((span.buffer.raw, 0)..=(span.buffer.raw, u64::MAX))
                    .filter(move |&(&(_, offset), &(length, _))| {
                        overlaps(span, DeviceBufferSpanV1::new(span.buffer, offset, length))
                    })
                    .flat_map(|(_, (_, slots))| slots.iter().copied())
            })
            .collect();
        touched.extend(named);
        if let Some(capture) = self.host.fake().capture_of(stream.slot()) {
            recorded
                .capturing
                .entry(capture)
                .or_default()
                .extend(touched);
            return;
        }
        drop(recorded);
        self.check(what, stream, &touched);
    }

    /// Records each of `touched` that a pending gather holds as a race.
    fn check(&self, what: &str, stream: BundleStreamHandleV1, touched: &[DeviceBufferSpanV1]) {
        let pending = self.pending();
        if pending.is_empty() {
            return;
        }
        let fake = self.host.fake();
        for &span in touched {
            for (held, name) in &pending {
                if overlaps(span, *held) {
                    let line = fake.journal.borrow().len();
                    fake.races.borrow_mut().push(format!(
                        "journal line {line}: {what} on stream {} touches {span:?}, {name} {held:?}",
                        stream.slot()
                    ));
                }
            }
        }
    }

    /// The uploaded ids `indices` holds, the first `rows` of them.
    fn uploaded_ids(&self, indices: DeviceBufferSpanV1, rows: u64) -> Vec<u32> {
        let uploads = self.host.fake().uploads.borrow();
        let bytes = rows as usize * 4;
        uploads
            .range((indices.buffer.raw, 0)..=(indices.buffer.raw, indices.offset))
            .next_back()
            .and_then(|(&(_, offset), uploaded)| {
                let at = (indices.offset - offset) as usize;
                uploaded.get(at..at + bytes)
            })
            .map_or_else(
                || vec![0; rows as usize],
                |ids| {
                    ids.chunks_exact(4)
                        .map(|id| u32::from_le_bytes(id.try_into().unwrap()))
                        .collect()
                },
            )
    }

    /// A stream wait queues a dependency. Completion becomes observable
    /// only after a later host readback, as at the consolidated status boundary.
    pub(in crate::fake_host_tests) fn with_delayed_completion(host: Rc<Host>) -> Self {
        Self {
            delayed_completion: true,
            ..Self::new(host)
        }
    }

    pub(in crate::fake_host_tests) fn with_gather_failure(
        host: Rc<Host>,
        stage: &'static str,
        after: usize,
    ) -> Self {
        Self {
            fail_gather: std::cell::Cell::new(Some((stage, after))),
            ..Self::with_delayed_completion(host)
        }
    }

    fn gather_failure(&self, stage: &str) -> HostServiceResultV1<()> {
        if let Some((selected, after)) = self.fail_gather.get()
            && selected == stage
        {
            if after == 0 {
                self.fail_gather.set(None);
                self.host
                    .fake()
                    .note(format!("async mapped failure {stage}"));
                return Err(BundleStatusV1::HostError);
            }
            self.fail_gather.set(Some((selected, after - 1)));
        }
        Ok(())
    }

    #[cfg(not(feature = "diagnostic-taps"))]
    pub(in crate::fake_host_tests) fn fail_gather_after(&self, stage: &'static str, after: usize) {
        self.fail_gather.set(Some((stage, after)));
    }

    /// Journal peer copies as the ordinary fake host does (`peer-copy`
    /// lines), so the prefill DMA audit can order them.
    pub(in crate::fake_host_tests) fn with_peer_copies(host: Rc<Host>) -> Self {
        Self {
            peer_copies: true,
            ..Self::new(host)
        }
    }
    fn emulate_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        if self.host.fake().capturing_on_device_of(stream.slot()) {
            return Err(BundleStatusV1::InvalidState);
        }
        let g = self.host.fake().shape.geometry;
        let e = &g.shape.engram;
        let table = e.layers.iter().enumerate().find_map(|(index, layer)| {
            ["weight", "scale"].into_iter().find_map(|part| {
                (gather.view_id == format!("layers.{layer}.engram.embed.{part}").as_bytes())
                    .then_some((
                        e.rows[index],
                        if part == "weight" {
                            e.head_dim
                        } else {
                            e.head_dim / 32
                        },
                    ))
            })
        });
        let (rows, width) = table.ok_or(BundleStatusV1::NotFound)?;
        if gather.image.raw == 0
            || gather.rank >= self.host.fake().shape.ranks()
            || gather.row_bytes != u64::from(width)
            || gather.rows.is_empty()
            || gather.rows.iter().any(|&row| row >= rows)
            || (gather.rows.len() as u64).checked_mul(gather.row_bytes) != Some(destination.length)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.host.fake().note(format!(
            "mapped gather {} {}",
            gather.rank,
            String::from_utf8_lossy(gather.view_id)
        ));
        let bytes = if self.row_ids {
            gather
                .rows
                .iter()
                .flat_map(|&row| {
                    (0..gather.row_bytes).map(move |at| {
                        (row.wrapping_mul(2_654_435_761).rotate_left(at as u32 % 32) as u8)
                            ^ at as u8
                    })
                })
                .collect()
        } else {
            vec![0; destination.length as usize]
        };
        self.host.copy_host_to_device(stream, destination, &bytes)
    }
}

macro_rules! forward {
    ($name:ident($($arg:ident: $ty:ty),*) -> $result:ty) => {
        fn $name(&self, $($arg: $ty),*) -> HostServiceResultV1<$result> {
            self.host.$name($($arg),*)
        }
    };
}
impl HostServicesBackendV1 for MappedHost {
    forward!(log(message: &[u8]) -> ());
    forward!(select_context(rank: u32, device_ordinal: i32) -> BundleContextHandleV1);
    forward!(release_context(context: BundleContextHandleV1) -> ());
    forward!(select_memory_pool(context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1);
    forward!(release_memory_pool(pool: BundleMemoryPoolHandleV1) -> ());
    forward!(allocate_device_buffer(pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, byte_length: u64, alignment: u64) -> BundleDeviceBufferHandleV1);
    forward!(release_device_buffer(buffer: BundleDeviceBufferHandleV1) -> ());
    forward!(open_compiled_image(component_role: &[u8]) -> BundleCompiledImageHandleV1);
    forward!(upload_rank_arena(context: BundleContextHandleV1, pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, image: BundleCompiledImageHandleV1, rank: u32) -> BundleRankArenaHandleV1);
    forward!(release_rank_arena(arena: BundleRankArenaHandleV1) -> ());
    forward!(resolve_compiled_image_view(image: BundleCompiledImageHandleV1, arena: BundleRankArenaHandleV1, rank: u32, view_id: &[u8]) -> CompiledImageSpanV1);
    forward!(release_compiled_image(image: BundleCompiledImageHandleV1) -> ());
    forward!(open_authenticated_component(component_role: &[u8]) -> BundleAuthenticatedComponentHandleV1);
    forward!(inspect_authenticated_component(component: BundleAuthenticatedComponentHandleV1) -> AuthenticatedComponentMetadataV1);
    forward!(read_authenticated_component(component: BundleAuthenticatedComponentHandleV1, role_destination: &mut [u8], bytes_destination: &mut [u8]) -> ());
    forward!(release_authenticated_component(component: BundleAuthenticatedComponentHandleV1) -> ());
    forward!(acquire_peer_access(owner_context: BundleContextHandleV1, peer_context: BundleContextHandleV1) -> BundlePeerAccessHandleV1);
    forward!(release_peer_access(peer_access: BundlePeerAccessHandleV1) -> ());
    forward!(create_stream(context: BundleContextHandleV1, stream_kind: u32) -> BundleStreamHandleV1);
    forward!(select_stage_stream(context: BundleContextHandleV1, rank: u32, stage_id: &[u8], stream_kind: u32) -> BundleStreamHandleV1);
    forward!(synchronize_stream(stream: BundleStreamHandleV1) -> ());
    forward!(release_stream(stream: BundleStreamHandleV1) -> ());
    forward!(create_event(context: BundleContextHandleV1) -> BundleEventHandleV1);
    forward!(record_event(event: BundleEventHandleV1, stream: BundleStreamHandleV1) -> ());
    forward!(wait_event(stream: BundleStreamHandleV1, event: BundleEventHandleV1) -> ());
    forward!(synchronize_event(event: BundleEventHandleV1) -> ());
    forward!(release_event(event: BundleEventHandleV1) -> ());
    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.touch("upload", stream, [destination]);
        self.host.copy_host_to_device(stream, destination, source)
    }
    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.touch("readback", stream, [source]);
        self.host.copy_device_to_host(stream, destination, source)?;
        self.completed_at
            .set(self.host.fake().journal.borrow().len());
        Ok(())
    }
    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.touch("device copy", stream, [destination, source]);
        self.host.copy_device_to_device(stream, destination, source)
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.touch("peer copy", stream, [destination, source]);
        let issuer = self.host.fake().stream_ranks.borrow()[&stream.slot()] + 1;
        assert!(
            issuer == destination_context.slot() || issuer == source_context.slot(),
            "a peer copy must use an endpoint's stream"
        );
        if !self.peer_copies && issuer != destination_context.slot() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        assert_ne!(destination_context, source_context);
        assert!(!self.host.fake().capturing_on_device_of(stream.slot()));
        self.host.fake().note(format!(
            "peer results {} {}",
            source_context.slot(),
            destination_context.slot()
        ));
        if self.peer_copies {
            return self.host.copy_peer_to_peer(
                stream,
                destination_context,
                destination,
                source_context,
                source,
            );
        }
        self.host.copy_device_to_device(stream, destination, source)
    }
    fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.touch("zeroing", stream, [destination]);
        self.host.zero_device_buffer(stream, destination)
    }
    fn write_device_pointer_table(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        slots: &[Option<DeviceBufferSpanV1>],
    ) -> HostServiceResultV1<()> {
        self.touch("pointer table", stream, [destination]);
        self.spans.borrow_mut().tables.insert(
            (destination.buffer.raw, destination.offset),
            (
                destination.length,
                slots.iter().flatten().copied().collect(),
            ),
        );
        self.host
            .write_device_pointer_table(stream, destination, slots)
    }
    forward!(load_cuda_module(context: BundleContextHandleV1, module_bytes: &[u8]) -> BundleCudaModuleHandleV1);
    forward!(resolve_cuda_symbol(module: BundleCudaModuleHandleV1, symbol_name: &[u8]) -> BundleCudaSymbolHandleV1);
    forward!(configure_cuda_symbol(symbol: BundleCudaSymbolHandleV1, dynamic_shared_memory_bytes: u32) -> ());
    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        self.touch("launch", stream, argument_spans(arguments));
        self.host
            .launch_cuda_symbol(stream, symbol, config, arguments)
    }
    forward!(release_cuda_symbol(symbol: BundleCudaSymbolHandleV1) -> ());
    forward!(release_cuda_module(module: BundleCudaModuleHandleV1) -> ());
    forward!(allocate_physical_kv(request: &PhysicalKvRequestV1) -> BundlePhysicalKvHandleV1);
    forward!(create_kv_request(kv: BundlePhysicalKvHandleV1, quota_tokens: u64) -> BundleKvRequestHandleV1);
    forward!(prepare_kv(request: BundleKvRequestHandleV1, operation: &[u8]) -> BundleKvTransactionHandleV1);
    forward!(commit_kv(transaction: BundleKvTransactionHandleV1) -> ());
    forward!(cancel_kv(transaction: BundleKvTransactionHandleV1) -> ());
    forward!(resolve_kv_view(request: BundleKvRequestHandleV1, rank: u32, view_id: &[u8]) -> DeviceBufferSpanV1);
    forward!(release_kv_request(request: BundleKvRequestHandleV1) -> ());
    forward!(release_physical_kv(kv: BundlePhysicalKvHandleV1) -> ());
    forward!(acquire_serving_kv(layout_id: &[u8]) -> u64);
    forward!(resolve_serving_kv_pool(rank: u32) -> DeviceBufferSpanV1);
    forward!(query_speculation() -> BundleRawSpeculationV1);
    forward!(query_batch_limits() -> BundleRawBatchLimitsV1);
    forward!(query_cluster_residency(symbol: BundleCudaSymbolHandleV1, config: LaunchConfigV1) -> BundleRawClusterResidencyV1);
    forward!(allocate_state(request: &StateRequestV1) -> BundleStateHandleV1);
    forward!(resolve_state_view(state: BundleStateHandleV1, rank: u32) -> DeviceBufferSpanV1);
    forward!(release_state(state: BundleStateHandleV1) -> ());
    forward!(create_collective(request: &CollectiveRequestV1) -> BundleCollectiveHandleV1);
    fn execute_collective(
        &self,
        collective: BundleCollectiveHandleV1,
        stream: BundleStreamHandleV1,
        operation: &[u8],
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        self.touch("collective", stream, argument_spans(arguments));
        self.host
            .execute_collective(collective, stream, operation, arguments)
    }
    forward!(release_collective(collective: BundleCollectiveHandleV1) -> ());
    fn run_custom_operation(
        &self,
        context: BundleContextHandleV1,
        stream: BundleStreamHandleV1,
        completion: CustomCompletionV1,
        operation_id: &[u8],
        operands: &[CustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostServiceResultV1<i32> {
        self.touch(
            "custom operation",
            stream,
            operands.iter().map(|operand| operand.span),
        );
        self.host.run_custom_operation(
            context,
            stream,
            completion,
            operation_id,
            operands,
            operation,
        )
    }
    forward!(begin_graph_capture(stream: BundleStreamHandleV1) -> ());
    fn end_graph_capture(
        &self,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<BundleGraphHandleV1> {
        let captured = self
            .spans
            .borrow_mut()
            .capturing
            .remove(&stream.slot())
            .unwrap_or_default();
        let graph = self.host.end_graph_capture(stream)?;
        self.spans.borrow_mut().graphs.insert(graph.raw, captured);
        Ok(graph)
    }
    fn launch_graph(
        &self,
        stream: BundleStreamHandleV1,
        graph: BundleGraphHandleV1,
    ) -> HostServiceResultV1<()> {
        let spans = self
            .spans
            .borrow()
            .graphs
            .get(&graph.raw)
            .cloned()
            .unwrap_or_default();
        self.touch("graph replay", stream, spans);
        self.host.launch_graph(stream, graph)
    }
    fn release_graph(&self, graph: BundleGraphHandleV1) -> HostServiceResultV1<()> {
        self.spans.borrow_mut().graphs.remove(&graph.raw);
        self.host.release_graph(graph)
    }
    forward!(set_memory_pool_peer_access(pool: BundleMemoryPoolHandleV1, peer_context: BundleContextHandleV1, enabled: bool) -> ());
    forward!(upload_rank_arenas(uploads: &[RankArenaUploadV1]) -> Vec<BundleRankArenaHandleV1>);
    fn create_mapped_gather(
        &self,
        producer: BundleStreamHandleV1,
        indices: DeviceBufferSpanV1,
        planes: &[MappedGatherPlaneV1],
    ) -> HostServiceResultV1<BundleMappedGatherHandleV1> {
        self.host.fake().check_span(indices);
        if planes.is_empty() || self.host.fake().capturing_on_device_of(producer.slot()) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for plane in planes {
            self.host.fake().check_span(plane.destination);
            if plane.row_bytes == 0
                || !plane.destination.length.is_multiple_of(plane.row_bytes)
                || plane.index_offset + plane.destination.length / plane.row_bytes * 4
                    > indices.length
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
        let rank = *self
            .host
            .fake()
            .stream_ranks
            .borrow()
            .get(&producer.slot())
            .ok_or(BundleStatusV1::InvalidArgument)?;
        // An event proxy gives this fake gather the existing checked LIFO
        // resource lifetime. Production owns a dedicated gather handle.
        let event = self
            .host
            .create_event(BundleContextHandleV1::from_parts(1, rank + 1))?;
        let gather = BundleMappedGatherHandleV1::from_raw(event.raw);
        self.gathers.borrow_mut().insert(
            gather.raw,
            Gather {
                producer,
                indices,
                planes: planes.to_vec(),
                pending: None,
                ids: Vec::new(),
                waited: Default::default(),
                waited_at: 0,
            },
        );
        self.host
            .fake()
            .note(format!("async mapped create {}", gather.slot()));
        Ok(gather)
    }
    fn enqueue_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
        rows: u64,
    ) -> HostServiceResultV1<()> {
        self.gather_failure("enqueue")?;
        let indices = self
            .gathers
            .borrow()
            .get(&handle.raw)
            .ok_or(BundleStatusV1::StaleHandle)?
            .indices;
        let ids = if self.row_ids {
            self.uploaded_ids(indices, rows.min(indices.length / 4))
        } else {
            Vec::new()
        };
        let mut gathers = self.gathers.borrow_mut();
        let gather = gathers
            .get_mut(&handle.raw)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if gather.pending.is_some()
            || self
                .host
                .fake()
                .capturing_on_device_of(gather.producer.slot())
        {
            return Err(BundleStatusV1::InvalidState);
        }
        if rows == 0
            || gather
                .planes
                .iter()
                .any(|p| rows > p.destination.length / p.row_bytes)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        gather.pending = Some(rows);
        gather.ids = ids;
        gather.waited.clear();
        self.host.fake().note(format!(
            "async mapped ids {} {}",
            gather.indices.length, rows
        ));
        Ok(())
    }
    fn wait_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        self.gather_failure("wait")?;
        let (rows, planes, ids) = {
            let mut gathers = self.gathers.borrow_mut();
            let gather = gathers
                .get_mut(&handle.raw)
                .ok_or(BundleStatusV1::StaleHandle)?;
            let rows = gather.pending.ok_or(BundleStatusV1::InvalidState)?;
            if self.host.fake().capturing_on_device_of(stream.slot())
                || !gather.waited.insert(stream.raw)
            {
                return Err(BundleStatusV1::InvalidState);
            }
            let planes = gather
                .planes
                .iter()
                .filter(|plane| plane.stream == stream)
                .cloned()
                .collect::<Vec<_>>();
            if planes.is_empty() {
                return Err(BundleStatusV1::InvalidArgument);
            }
            (rows, planes, gather.ids.clone())
        };
        self.host
            .fake()
            .note(format!("async mapped wait {}", stream.slot()));
        for plane in planes {
            let destination = DeviceBufferSpanV1::new(
                plane.destination.buffer,
                plane.destination.offset,
                rows * plane.row_bytes,
            );
            let at = (plane.index_offset / 4) as usize;
            let rows_of = if self.row_ids {
                ids.get(at..at + rows as usize)
                    .ok_or(BundleStatusV1::InvalidArgument)?
                    .to_vec()
            } else {
                vec![0; rows as usize]
            };
            self.emulate_mapped_rows(
                stream,
                destination,
                MappedRowGatherV1 {
                    image: plane.image,
                    rank: plane.rank,
                    view_id: &plane.view_id,
                    row_bytes: plane.row_bytes,
                    rows: &rows_of,
                },
            )?;
        }
        self.gathers
            .borrow_mut()
            .get_mut(&handle.raw)
            .expect("live gather")
            .waited_at = self.host.fake().journal.borrow().len();
        Ok(())
    }
    fn complete_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<Option<MappedGatherStatsV1>> {
        self.gather_failure("complete")?;
        let mut gathers = self.gathers.borrow_mut();
        let gather = gathers
            .get_mut(&handle.raw)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if gather.pending.is_none() {
            return Err(BundleStatusV1::InvalidState);
        }
        if (self.delayed_completion && self.completed_at.get() <= gather.waited_at)
            || gather
                .planes
                .iter()
                .any(|plane| !gather.waited.contains(&plane.stream.raw))
        {
            return Ok(None);
        }
        gather.pending = None;
        self.host.fake().note("async mapped complete".into());
        Ok(Some(MappedGatherStatsV1::default()))
    }
    fn release_mapped_gather(&self, handle: BundleMappedGatherHandleV1) -> HostServiceResultV1<()> {
        if !self.gathers.borrow().contains_key(&handle.raw) {
            return Err(BundleStatusV1::StaleHandle);
        }
        self.host
            .release_event(BundleEventHandleV1::from_raw(handle.raw))?;
        self.gathers.borrow_mut().remove(&handle.raw);
        self.host
            .fake()
            .note(format!("async mapped release {}", handle.slot()));
        Ok(())
    }
    fn gather_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        self.host.fake().count_mapped_gather();
        self.emulate_mapped_rows(stream, destination, gather)
    }
}
