//! Fake-host tests of MTP speculation, with a CPU model of what the state
//! kernels write.
//!
//! The model follows provenance, not values: every linear-state and PLE
//! entry holds the tokens its state has absorbed, and every final-X row of
//! the target or the drafter carries the (slot, position, token) it was
//! computed for. Copies move both, the recurrent kernels append their rows'
//! tokens, and the drafter's input rows are checked against the row they
//! claim to read. Two runs whose committed entries hold the same tokens hold
//! the same state, since the kernels are deterministic in their inputs.

use super::*;

/// Byte address of a model entry: buffer slot and offset.
type Address = (u32, u64);

/// One launch as a graph captured it: stream, symbol, grid and arguments.
pub(super) type Captured = (u32, String, [u32; 3], Vec<KernelArgumentV1>);

/// The row a final-X row was computed for; `drafter` for the drafter's own
/// rows, whose token is the next one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Tag {
    pub slot: u32,
    pub position: u32,
    pub token: u32,
    pub drafter: bool,
}

/// One drafter input row as `mtp_gather` read it, and what the drafter
/// program then ran it as.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DrafterRow {
    pub source: Tag,
    pub slot: u32,
    pub position: u32,
    pub token: u32,
}

/// The fake host's device model: bytes of small buffers, token histories
/// of state entries and tags of X rows.
#[derive(Default)]
pub(super) struct Sim {
    /// Written prefixes of buffers of at most `TRACKED` bytes, by slot.
    /// The unwritten tail reads as zero without allocating the logical extent.
    memory: BTreeMap<u32, Vec<u8>>,
    pub(super) histories: BTreeMap<Address, Vec<u32>>,
    tags: BTreeMap<Address, Tag>,
    /// Per stream: the rows of the running program.
    programs: BTreeMap<u32, Program>,
    /// Interleaved prefill parts share a stream but have distinct input arenas.
    program_parts: BTreeMap<u32, Option<usize>>,
    suspended: BTreeMap<(u32, Option<usize>), Program>,
    /// Captured launches per graph.
    graphs: BTreeMap<u32, Vec<Captured>>,
    /// Spans the model needs per stream (rank): final X, and the PLE
    /// staging input of each phase.
    pub spans: BTreeMap<u32, ModelSpans>,
    pub drafter_rows: Vec<DrafterRow>,
    /// Offload tests additionally model QSA raw keys, positions and block keys.
    pub(super) track_qsa: bool,
    pub(super) qsa_blocks: Vec<(u32, Vec<u32>)>,
    /// The topology of the modeled instance; TP4 unless a test sets it.
    pub topology: Option<Shape>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModelSpans {
    pub x_start: DeviceBufferSpanV1,
    pub x_final: DeviceBufferSpanV1,
    pub ple_input: [DeviceBufferSpanV1; 3],
    pub parts: Option<[u32; 2]>,
}

#[derive(Clone, Debug, Default)]
struct Program {
    row_offset: u64,
    tokens: Vec<u32>,
    /// (slot, position) per row, once a full layer wrote its index keys.
    rows: Vec<(u32, u32)>,
    gathered: Vec<Tag>,
    heads: Vec<u32>,
}

pub(super) const TRACKED: u64 = 64 << 20;

/// The token the fake drafter predicts at position `position`: every draft
/// is right when the sequence follows this rule.
pub(super) fn truth(position: u64) -> u32 {
    (1000 + 7 * position as u32) % VOCAB
}

#[test]
fn simulated_bytes_keep_small_transfers_small() {
    let first = BundleDeviceBufferHandleV1::from_parts(1, 1);
    let second = BundleDeviceBufferHandleV1::from_parts(1, 2);
    let extents = BTreeMap::from([(first.slot(), TRACKED), (second.slot(), TRACKED)]);
    let span = |buffer, offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
    let mut sim = Sim::default();
    assert_eq!(sim.read(span(first, TRACKED - 16, 16), &extents), [0; 16]);
    let mut expected = [0_u8; 16];
    expected[2..10].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    sim.write(span(first, 2, 8), &expected[2..10], &extents);
    // An overlapping copy reads its source before writing the destination.
    expected.copy_within(2..10, 4);
    sim.copy(span(first, 4, 8), span(first, 2, 8), &extents);
    assert_eq!(sim.read(span(first, 0, 16), &extents), expected);
    sim.copy(span(second, 3, 16), span(first, 0, 16), &extents);
    assert_eq!(sim.read(span(second, 3, 16), &extents), expected);
    sim.write(span(first, 4, 8), &[0; 8], &extents);
    expected[4..12].fill(0);
    assert_eq!(sim.read(span(first, 0, 16), &extents), expected);
    assert_eq!(sim.read(span(second, TRACKED - 16, 16), &extents), [0; 16]);
    // Two 64 MiB logical buffers need only the prefixes actually written.
    assert!(sim.memory.values().map(Vec::capacity).sum::<usize>() <= 64 * 1024);
}

#[test]
fn released_fake_buffers_drop_bytes_and_preserve_live_buffers() {
    let host = FakeHost::default();
    let pool = BundleMemoryPoolHandleV1::from_parts(1, 1);
    let stream = BundleStreamHandleV1::from_parts(1, 1);
    let first = host
        .allocate_device_buffer(pool, stream, 1024, 256)
        .unwrap();
    let second = host
        .allocate_device_buffer(pool, stream, 1024, 256)
        .unwrap();
    for buffer in [first, second] {
        host.sim.borrow_mut().write(
            DeviceBufferSpanV1 {
                buffer,
                offset: 0,
                length: 1024,
            },
            &[7; 1024],
            &host.extents.borrow(),
        );
    }
    host.release_device_buffer(first).unwrap();
    assert!(!host.sim.borrow().memory.contains_key(&first.slot()));
    assert!(!host.extents.borrow().contains_key(&first.slot()));
    assert_eq!(host.sim.borrow().memory[&second.slot()], [7; 1024]);
    host.release_device_buffer(second).unwrap();
    assert!(host.sim.borrow().memory.is_empty());
}

impl Sim {
    /// All requests have retired. Keep graph commands, spans and resident
    /// bytes: the bundle may upload only deltas to their block tables.
    pub(super) fn finish_route_case(&mut self) {
        self.histories.clear();
        self.tags.clear();
        self.programs.clear();
        self.program_parts.clear();
        self.suspended.clear();
        self.drafter_rows.clear();
        self.qsa_blocks.clear();
    }

    pub(super) fn release_graph(&mut self, graph: u32) {
        // The ABI pins the host, so graph-owned arguments must follow the
        // graph handle's lifetime rather than the simulator's lifetime.
        self.graphs.remove(&graph);
    }

    pub(super) fn release_buffer(&mut self, slot: u32) {
        self.memory.remove(&slot);
        self.histories.retain(|&(buffer, _), _| buffer != slot);
        self.tags.retain(|&(buffer, _), _| buffer != slot);
    }

    fn tracked(span: DeviceBufferSpanV1, extents: &BTreeMap<u32, u64>) -> bool {
        let Some(&extent) = extents.get(&span.buffer.slot()).filter(|&&n| n <= TRACKED) else {
            return false;
        };
        assert!(span.offset <= extent && span.length <= extent - span.offset);
        true
    }

    pub fn write(&mut self, span: DeviceBufferSpanV1, bytes: &[u8], extents: &BTreeMap<u32, u64>) {
        if !Self::tracked(span, extents) {
            return;
        }
        let end = (span.offset + span.length) as usize;
        let memory = self.memory.entry(span.buffer.slot()).or_default();
        if memory.is_empty() {
            *memory = vec![0; end];
        } else if memory.len() < end {
            memory.resize(end, 0);
        }
        memory[span.offset as usize..end].copy_from_slice(bytes);
    }

    pub fn read(&self, span: DeviceBufferSpanV1, extents: &BTreeMap<u32, u64>) -> Vec<u8> {
        let tracked = Self::tracked(span, extents);
        let mut bytes = vec![0; span.length as usize];
        if tracked && let Some(memory) = self.memory.get(&span.buffer.slot()) {
            let start = span.offset as usize;
            let end = ((span.offset + span.length) as usize).min(memory.len());
            if start < end {
                bytes[..end - start].copy_from_slice(&memory[start..end]);
            }
        }
        bytes
    }

    fn word(&mut self, span: DeviceBufferSpanV1, index: u64, extents: &BTreeMap<u32, u64>) -> i32 {
        let bytes = self.read(
            DeviceBufferSpanV1::new(span.buffer, span.offset + 4 * index, 4),
            extents,
        );
        i32::from_le_bytes(bytes.try_into().unwrap())
    }

    fn put_word(
        &mut self,
        span: DeviceBufferSpanV1,
        index: u64,
        value: i32,
        extents: &BTreeMap<u32, u64>,
    ) {
        self.write(
            DeviceBufferSpanV1::new(span.buffer, span.offset + 4 * index, 4),
            &value.to_le_bytes(),
            extents,
        );
    }

    /// Moves the model entries of `[source, source + length)` to
    /// `destination`, as a byte copy moves the state they describe.
    pub fn copy(
        &mut self,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
        extents: &BTreeMap<u32, u64>,
    ) {
        let bytes = self.read(source, extents);
        self.write(destination, &bytes, extents);
        let slot = source.buffer.slot();
        let range = (slot, source.offset)..(slot, source.offset + source.length);
        let histories = self
            .histories
            .range(range.clone())
            .map(|(&(_, offset), history)| (offset - source.offset, history.clone()))
            .collect::<Vec<_>>();
        let tags = self
            .tags
            .range(range)
            .map(|(&(_, offset), &tag)| (offset - source.offset, tag))
            .collect::<Vec<_>>();
        let target = destination.buffer.slot();
        let target_range =
            (target, destination.offset)..(target, destination.offset + destination.length);
        let stale = self
            .histories
            .range(target_range.clone())
            .map(|(&key, _)| key)
            .collect::<Vec<_>>();
        stale.into_iter().for_each(|key| {
            self.histories.remove(&key);
        });
        let stale = self
            .tags
            .range(target_range)
            .map(|(&key, _)| key)
            .collect::<Vec<_>>();
        stale.into_iter().for_each(|key| {
            self.tags.remove(&key);
        });
        for (offset, history) in histories {
            self.histories
                .insert((target, destination.offset + offset), history);
        }
        for (offset, tag) in tags {
            self.tags.insert((target, destination.offset + offset), tag);
        }
    }

    fn entry(span: DeviceBufferSpanV1, index: i64, bytes: u64) -> Address {
        (span.buffer.slot(), span.offset + index as u64 * bytes)
    }

    fn history(&self, address: Address) -> Vec<u32> {
        self.histories.get(&address).cloned().unwrap_or_default()
    }

    pub fn history_at(&self, span: DeviceBufferSpanV1, index: i64, bytes: u64) -> Vec<u32> {
        self.history(Self::entry(span, index, bytes))
    }

    pub fn capture(&mut self, graph: u32, launches: Vec<Captured>) {
        self.graphs.insert(graph, launches);
    }

    pub fn replay(&mut self, graph: u32, extents: &BTreeMap<u32, u64>) {
        for (stream, name, grid, arguments) in self.graphs[&graph].clone() {
            self.launch(stream, &name, grid, &arguments, extents);
        }
    }

    /// Applies one launch to the model.
    pub fn launch(
        &mut self,
        stream: u32,
        name: &str,
        grid: [u32; 3],
        arguments: &[KernelArgumentV1],
        extents: &BTreeMap<u32, u64>,
    ) {
        let span = |index: usize| match &arguments[index] {
            KernelArgumentV1::DeviceBuffer(span) => *span,
            other => panic!("{name} argument {index} is {other:?}"),
        };
        let scalar = |index: usize| match &arguments[index] {
            KernelArgumentV1::ScalarU32(value) => u64::from(*value),
            KernelArgumentV1::ScalarU64(value) => *value,
            other => panic!("{name} argument {index} is {other:?}"),
        };
        let Some(&spans) = self.spans.get(&stream) else {
            return;
        };
        let shape = self.topology.unwrap_or(Shape::TP4);
        // A topology's per-rank entries behave as TP4's of the same name.
        let name = name.replacen(
            shape.tag(),
            model_qwen38_contract::geometry::FLASH.kernel_tag,
            1,
        );
        let name = name.replacen(
            shape.geometry.kernel_tag,
            model_qwen38_contract::geometry::FLASH.kernel_tag,
            1,
        );
        let name = name.as_str();
        let part = spans.parts.and_then(|parts| {
            arguments
                .iter()
                .filter_map(|argument| match argument {
                    KernelArgumentV1::DeviceBuffer(span) => {
                        parts.iter().position(|&arena| span.buffer.slot() == arena)
                    }
                    _ => None,
                })
                .next_back()
        });
        let previous = self.program_parts.get(&stream).copied().flatten();
        if previous != part {
            self.program_parts.insert(stream, part);
            let program = self.programs.remove(&stream).unwrap_or_default();
            self.suspended.insert((stream, previous), program);
            let program = self.suspended.remove(&(stream, part)).unwrap_or_default();
            self.programs.insert(stream, program);
        }
        match name {
            // The token embedding starts a program (the prefill head's row
            // gather passes a live width, not the vocabulary).
            "chop_row_embedding_qwen38_v1" if scalar(5) == u64::from(shape.vocab()) => {
                let live = self.word(span(3), 0, extents) as u64;
                let tokens = (0..live)
                    .map(|row| self.word(span(1), row, extents) as u32)
                    .collect();
                let gathered = self
                    .programs
                    .get(&stream)
                    .map(|program| program.gathered.clone())
                    .unwrap_or_default();
                self.programs.insert(
                    stream,
                    Program {
                        tokens,
                        gathered,
                        ..Program::default()
                    },
                );
            }
            // The stream expansion, the shared broadcast.
            "chop_row_copy_broadcast_streams_v1" => {
                let output = span(1);
                assert_eq!(output.buffer, spans.x_start.buffer);
                self.programs.entry(stream).or_default().row_offset =
                    (output.offset - spans.x_start.offset) / crate::scratch::wide_row_bytes(shape);
            }
            // The drafter's head rows.
            "chop_row_embedding_qwen38_v1" => {
                let live = self.word(span(3), 0, extents) as u64;
                let heads = (0..live)
                    .map(|row| self.word(span(1), row, extents) as u32)
                    .collect();
                self.programs.entry(stream).or_default().heads = heads;
            }
            // The owner-routed key write of context-parallel groups takes the
            // served key write's arguments first, so it runs the same rows and
            // updates the replicated ring alike; its group rank and size then
            // place each completed block's key on the rank owning its page
            // only (`owned_index_key_write`): physical page p lives on group
            // rank `p % size` at local table entry `p / size`.
            "chop_qsa_qwen38_index_key_write_v1"
            | "chop_owned_writes_qwen38_index_key_write_v1" => {
                let (cp_rank, cp_size) = if name == "chop_owned_writes_qwen38_index_key_write_v1" {
                    (scalar(19), scalar(20))
                } else {
                    (0, 1)
                };
                let live = self.word(span(6), 0, extents) as u64;
                let rows = (0..live)
                    .map(|row| {
                        let item = self.word(span(4), row, extents) as u64;
                        (
                            self.word(span(7), item, extents) as u32,
                            self.word(span(5), row, extents) as u32,
                        )
                    })
                    .collect::<Vec<_>>();
                let program = self.programs.entry(stream).or_default().clone();
                if self.track_qsa {
                    // Match indexer_key_write: complete a block using this
                    // launch's rows plus earlier raw keys and positions from
                    // the ring, then overwrite the ring's last sixteen rows.
                    for (row, &(slot, position)) in rows.iter().enumerate() {
                        if position % shape.qsa_block() != shape.qsa_block() - 1 {
                            continue;
                        }
                        let mut block = Vec::new();
                        for p in position + 1 - shape.qsa_block()..=position {
                            if let Some(r) = rows[..=row].iter().position(|&key| key == (slot, p)) {
                                block.extend([p, program.tokens[r]]);
                            } else {
                                let cell = u64::from(slot * QSA_RING + p % QSA_RING);
                                let key = self.history(Self::entry(
                                    span(8),
                                    cell as i64,
                                    u64::from(shape.index_dim()) * BF16_BYTES,
                                ));
                                let rope = self.history(Self::entry(span(9), cell as i64, 12));
                                assert_eq!(
                                    rope,
                                    vec![p],
                                    "QSA must restore the position of raw key {p}"
                                );
                                assert_eq!(key.len(), 2, "QSA must restore raw key {p}");
                                assert_eq!(key[0], p);
                                block.extend(key);
                            }
                        }
                        let physical = u64::from(position) / PAGE_TOKENS;
                        if physical % cp_size != cp_rank {
                            continue;
                        }
                        let item = self.word(span(4), row as u64, extents) as u64;
                        let page =
                            self.word(span(14), item * scalar(16) + physical / cp_size, extents)
                                as u64;
                        let offset = page * scalar(17)
                            + u64::from(position % PAGE_TOKENS as u32 / shape.qsa_block())
                                * u64::from(shape.index_dim())
                                * BF16_BYTES;
                        self.histories
                            .insert(Self::entry(span(15), offset as i64, 1), block.clone());
                        self.qsa_blocks.push((stream, block));
                    }
                    for (row, &(slot, position)) in rows.iter().enumerate() {
                        let cell = i64::from(slot * QSA_RING + position % QSA_RING);
                        self.histories.insert(
                            Self::entry(span(8), cell, u64::from(shape.index_dim()) * BF16_BYTES),
                            vec![position, program.tokens[row]],
                        );
                        self.histories
                            .insert(Self::entry(span(9), cell, 12), vec![position]);
                        // An anchored item's anchor ring keeps its rows of
                        // the sixteen positions before its anchor end.
                        let item = self.word(span(4), row as u64, extents) as u64;
                        let anchor = self.word(span(10), item, extents);
                        let end = self.word(span(11), item, extents);
                        if anchor >= 0 && (end - QSA_RING as i32..end).contains(&(position as i32))
                        {
                            let cell = i64::from(anchor as u32 * QSA_RING + position % QSA_RING);
                            self.histories.insert(
                                Self::entry(
                                    span(12),
                                    cell,
                                    u64::from(shape.index_dim()) * BF16_BYTES,
                                ),
                                vec![position, program.tokens[row]],
                            );
                            self.histories
                                .insert(Self::entry(span(13), cell, 12), vec![position]);
                        }
                    }
                }
                if program.rows == rows {
                    return;
                }
                let drafter = !program.gathered.is_empty();
                for (row, &(slot, position)) in rows.iter().enumerate() {
                    let token = program.tokens[row];
                    if drafter {
                        self.drafter_rows.push(DrafterRow {
                            source: program.gathered[row],
                            slot,
                            position,
                            token,
                        });
                    }
                    self.tags.insert(
                        Self::entry(
                            spans.x_final,
                            row as i64 + program.row_offset as i64,
                            crate::scratch::wide_row_bytes(shape),
                        ),
                        Tag {
                            slot,
                            position,
                            token,
                            drafter,
                        },
                    );
                }
                let program = self.programs.get_mut(&stream).unwrap();
                program.rows = rows;
                program.gathered.clear();
            }
            "chop_qwen38_mtp_gather_v1" => {
                let live = self.word(span(7), 0, extents) as u64;
                let mut gathered = Vec::new();
                for row in 0..live {
                    let source = self.word(span(2), row, extents);
                    let address = if source >= 0 {
                        Self::entry(
                            span(0),
                            i64::from(source),
                            crate::scratch::wide_row_bytes(shape),
                        )
                    } else {
                        Self::entry(
                            span(1),
                            -1 - i64::from(source),
                            crate::scratch::wide_row_bytes(shape),
                        )
                    };
                    gathered.push(
                        *self
                            .tags
                            .get(&address)
                            .unwrap_or_else(|| panic!("drafter row {row} reads an untagged X row")),
                    );
                    let token = self.word(span(3), row, extents);
                    if token >= 0 {
                        let draft = self.word(span(4), token as u64, extents);
                        self.put_word(span(6), row, draft, extents);
                    }
                }
                self.programs.entry(stream).or_default().gathered = gathered;
            }
            "chop_qwen38_draft_argmax_final_v1" => {
                let live = self.word(span(5), 0, extents) as u64;
                let program = self.programs[&stream].clone();
                for head in 0..live {
                    let (_, position) = program.rows[program.heads[head as usize] as usize];
                    let draft = truth(u64::from(position) + 2) as i32;
                    let slot = self.word(span(1), head, extents);
                    self.put_word(span(3), slot as u64, draft, extents);
                    let target = self.word(span(2), head, extents);
                    if target >= 0 {
                        self.put_word(span(4), target as u64, draft, extents);
                    }
                }
            }
            "chop_qwen38_copy_entries_v1" | "chop_qwen38_copy_entries" => {
                let live = self.word(span(4), 0, extents) as u64;
                let (entry, stride) = (scalar(5), scalar(6));
                for item in 0..live.min(u64::from(grid[0])) {
                    let from = self.word(span(2), item, extents);
                    if from < 0 {
                        continue;
                    }
                    let to = self.word(span(3), item, extents);
                    for plane in 0..u64::from(grid[1]) {
                        let at = |span: DeviceBufferSpanV1, index: i32| {
                            DeviceBufferSpanV1::new(
                                span.buffer,
                                span.offset + plane * stride + index as u64 * entry,
                                entry,
                            )
                        };
                        self.copy(at(span(1), to), at(span(0), from), extents);
                    }
                }
            }
            "chop_qwen38_ple_stage_v1" => {
                let live = self.word(span(4), 0, extents) as u64;
                for item in 0..live {
                    let slot = self.word(span(2), item, extents);
                    let history = if self.word(span(3), item, extents) > 0 {
                        self.history_at(span(0), i64::from(slot), shape.ple_entry_bytes())
                    } else {
                        Vec::new()
                    };
                    self.histories.insert(
                        Self::entry(span(1), item as i64, shape.ple_entry_bytes()),
                        history,
                    );
                }
            }
            // One launch per item: the item is the staged entry's index, and
            // its rows follow the earlier items' rows.
            "chop_row_qwen38_ple_conv_prefill_v1" | "chop_row_qwen38_ple_conv_decode_v1" => {
                let state = span(2);
                let phase = spans
                    .ple_input
                    .iter()
                    .position(|input| {
                        input.buffer == state.buffer
                            && (input.offset..input.offset + input.length).contains(&state.offset)
                    })
                    .expect("PLE state in a staging buffer");
                let staged = spans.ple_input[phase];
                let item = (state.offset - staged.offset) / shape.ple_entry_bytes();
                let q_len = span(5);
                let first = DeviceBufferSpanV1::new(q_len.buffer, q_len.offset - 4 * item, 4);
                let start = (0..item)
                    .map(|index| self.word(first, index, extents) as usize)
                    .sum::<usize>();
                let count = self.word(q_len, 0, extents) as usize;
                let output = span(4);
                self.ple_rows(
                    stream,
                    (state.buffer.slot(), state.offset),
                    (output.buffer.slot(), output.offset),
                    start,
                    count,
                );
            }
            // Every decode item at once: item i has q_len[i] rows (zero or
            // one) at row i, and its staged entry `entry` elements apart.
            "chop_row_qwen38_ple_conv_items_decode_v1" => {
                let stride = scalar(6) * BF16_BYTES;
                let (state, output) = (span(2), span(4));
                for item in 0..u64::from(grid[0]) {
                    let count = self.word(span(5), item, extents) as usize;
                    if count == 0 {
                        continue;
                    }
                    self.ple_rows(
                        stream,
                        (state.buffer.slot(), state.offset + item * stride),
                        (output.buffer.slot(), output.offset + item * stride),
                        item as usize,
                        count,
                    );
                }
            }
            "chop_qwen38_ple_conv_verify_v1" => {
                let live = self.word(span(7), 0, extents) as u64;
                for item in 0..live {
                    let start = self.word(span(5), item, extents) as usize;
                    let count = self.word(span(6), item, extents) as usize;
                    self.ple_rows(
                        stream,
                        Self::entry(span(2), item as i64, shape.ple_entry_bytes()),
                        Self::entry(span(4), item as i64, shape.ple_entry_bytes()),
                        start,
                        count,
                    );
                }
            }
            "chop_qwen38_ple_anchor_v1" => {
                let live = self.word(span(8), 0, extents) as u64;
                let tokens = self.programs[&stream].tokens.clone();
                for item in 0..live {
                    let slot = self.word(span(6), item, extents);
                    if slot < 0 {
                        continue;
                    }
                    let start = self.word(span(4), item, extents) as usize;
                    let rows = (self.word(span(7), item, extents)
                        - self.word(span(5), item, extents))
                        as usize;
                    let mut history =
                        self.history_at(span(2), item as i64, shape.ple_entry_bytes());
                    history.extend_from_slice(&tokens[start..start + rows]);
                    self.histories.insert(
                        Self::entry(span(3), i64::from(slot), shape.ple_entry_bytes()),
                        history,
                    );
                }
            }
            "chop_qwen38_ple_checkpoints_v1" | "chop_qwen38_ple_checkpoints_indexed_v1" => {
                let live = self.word(span(6), 0, extents) as u64;
                let tokens = self.programs[&stream].tokens.clone();
                for item in 0..live {
                    let start = self.word(span(4), item, extents) as usize;
                    let count = self.word(span(5), item, extents) as usize;
                    let mut history =
                        self.history_at(span(2), item as i64, shape.ple_entry_bytes());
                    for row in 0..count {
                        history.push(tokens[start + row]);
                        let entry = if name.contains("indexed") {
                            i64::from(self.word(
                                span(7),
                                item * u64::from(MAX_VERIFY_ROWS) + row as u64,
                                extents,
                            ))
                        } else {
                            (item * u64::from(MAX_VERIFY_ROWS) + row as u64) as i64
                        };
                        if name.contains("indexed") && (entry < 0 || entry as u64 >= scalar(8)) {
                            continue;
                        }
                        self.histories.insert(
                            Self::entry(span(3), entry, shape.ple_entry_bytes()),
                            history.clone(),
                        );
                    }
                }
            }
            "chop_linear_qwen38_conv_verify_v1" | "chop_linear_qwen38_delta_verify_v1" => {
                let (state, base) = if name.contains("conv") {
                    (span(3), 5)
                } else {
                    (span(7), 9)
                };
                let bytes = if name.contains("conv") {
                    shape.conv_entry_bytes
                } else {
                    shape.delta_entry_bytes
                };
                let live = self.word(span(base + 3), 0, extents) as u64;
                let tokens = self.programs[&stream].tokens.clone();
                for item in 0..live {
                    let start = self.word(span(base), item, extents) as usize;
                    let end = self.word(span(base), item + 1, extents) as usize;
                    let committed = self.word(span(base + 1), item, extents);
                    let mut history = if committed >= 0 {
                        self.history_at(state, i64::from(committed), bytes)
                    } else {
                        Vec::new()
                    };
                    for (offset, &token) in tokens[start..end].iter().enumerate() {
                        history.push(token);
                        let checkpoint = self.word(
                            span(base + 2),
                            item * u64::from(MAX_VERIFY_ROWS) + offset as u64,
                            extents,
                        );
                        self.histories.insert(
                            Self::entry(state, i64::from(checkpoint), bytes),
                            history.clone(),
                        );
                    }
                }
            }
            "chop_linear_qwen38_conv_prefill_v1" | "chop_linear_qwen38_delta_prefill_scan_v1" => {
                let bytes = if name.contains("conv") {
                    shape.conv_entry_bytes
                } else {
                    shape.delta_entry_bytes
                };
                let state = span(3);
                let live = self.word(span(9), 0, extents) as u64;
                let tokens = self.programs[&stream].tokens.clone();
                for item in 0..live {
                    let start = self.word(span(5), item, extents) as usize;
                    let end = self.word(span(5), item + 1, extents) as usize;
                    let working = self.word(span(6), item, extents);
                    let mut history = if self.word(span(7), item, extents) != 0 {
                        self.history_at(state, i64::from(working), bytes)
                    } else {
                        Vec::new()
                    };
                    // Match the prefill kernels: an anchor entry keeps the
                    // state after the last whole 64-row chunk before a
                    // partial one.
                    let anchor = self.word(span(8), item, extents);
                    let chunk = PREFILL_GRANULE as usize;
                    if anchor >= 0 && (end - start) % chunk != 0 && end - start > chunk {
                        let mut kept = history.clone();
                        kept.extend_from_slice(
                            &tokens[start..start + (end - start) / chunk * chunk],
                        );
                        self.histories
                            .insert(Self::entry(state, i64::from(anchor), bytes), kept);
                    }
                    history.extend_from_slice(&tokens[start..end]);
                    self.histories
                        .insert(Self::entry(state, i64::from(working), bytes), history);
                }
            }
            _ => {}
        }
    }

    fn ple_rows(
        &mut self,
        stream: u32,
        input: Address,
        output: Address,
        start: usize,
        count: usize,
    ) {
        let mut history = self.history(input);
        history.extend_from_slice(&self.programs[&stream].tokens[start..start + count]);
        self.histories.insert(output, history);
    }
}

/// A fake host for a package with the MTP drafter, serving `drafts` drafts
/// per verify item.
fn drafter_host(drafts: u32) -> Rc<FakeHost> {
    drafter_host_at(drafts, Shape::TP4)
}

/// A drafter host of `shape`'s topology, with the n-gram table where its
/// package must place it.
pub(super) fn drafter_host_at(drafts: u32, shape: Shape) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::at(shape));
    host.ngram_host.set(ngram_in_host_memory(shape));
    host.drafter.set(true);
    host.speculation.set((drafts, MAX_VERIFY_TOKENS));
    host
}

fn create_drafter(host: &Rc<FakeHost>) -> BundleInstanceV1 {
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &configuration_at(host.shape(), host.layout(), &[]),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    host.model(instance.handle());
    instance
}

/// A verify item the bundle drafts: the bonus token and `drafts` rows.
pub(super) fn drafted(
    request: u64,
    slot: u64,
    previous: u64,
    bonus: u32,
    drafts: u32,
) -> (BatchItemV1, BatchAppendV1, u64) {
    let rows = u64::from(drafts) + 1;
    let end = previous + rows;
    let pages = (0..end.div_ceil(PAGE_TOKENS))
        .map(|page| slot * 64 + page)
        .collect();
    (
        BatchItemV1::verify(request, rows as u32, true, vec![bonus]),
        BatchAppendV1::new(previous, rows, end, pages, None),
        slot,
    )
}

pub(super) fn plain(
    request: u64,
    slot: u64,
    previous: u64,
    inputs: &[u32],
) -> (BatchItemV1, BatchAppendV1, u64) {
    let count = inputs.len() as u64;
    let end = previous + count;
    let pages = (0..end.div_ceil(PAGE_TOKENS))
        .map(|page| slot * 64 + page)
        .collect();
    (
        BatchItemV1::new(
            request,
            previous > 0 && count == 1,
            count as u32,
            true,
            inputs.to_vec(),
        ),
        BatchAppendV1::new(previous, count, end, pages, None),
        slot,
    )
}

pub(super) fn batch(items: Vec<(BatchItemV1, BatchAppendV1, u64)>) -> BatchRequestV1 {
    let (items, rest): (Vec<_>, Vec<_>) = items
        .into_iter()
        .map(|(item, append, slot)| (item, (append, slot)))
        .unzip();
    let (appends, slots): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends).with_state_slots(slots)
}

/// The limits of the sink of draft tokens and views a step emitted.
pub(super) struct SpeculativeLimits;

impl SinkLimits for SpeculativeLimits {
    fn limits() -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(8192, u64::from(VOCAB) * 4, 64 * u64::from(VOCAB) * 4)
    }
}

/// Draft tokens and views a step emitted.
pub(super) type SpeculativeSink = RecordingSink<SpeculativeLimits>;

pub(super) fn execute(
    host: &FakeHost,
    instance: &mut BundleInstanceV1,
    request: &BatchRequestV1,
) -> SpeculativeSink {
    let mut sink = SpeculativeSink::default();
    instance
        .execute(&request.encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    sink
}

pub(super) fn kept(records: &[(u64, u32)]) -> Vec<BundleKeptRowsV1> {
    records
        .iter()
        .map(|&(request_id, kept_rows)| BundleKeptRowsV1 {
            struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
            reserved: 0,
            request_id,
            kept_rows,
            reserved_tail: 0,
        })
        .collect()
}

#[test]
fn without_the_drafter_the_host_drafts_and_with_it_the_bundle_does() {
    // The package without the drafter verifies host drafts.
    let host = Rc::new(FakeHost::default());
    host.speculation.set((3, MAX_VERIFY_TOKENS));
    // The gathered layout: one rank-0 view per verify item.
    let mut instance = crate::instance::with_row_owners(false, || {
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
            .expect("fake-host create")
    });
    host.model(instance.handle());
    execute(&host, &mut instance, &batch(vec![plain(10, 0, 0, &PROMPT)]));
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let host_drafted = |previous: u64, tokens: Vec<u32>| {
        let rows = tokens.len() as u64;
        let end = previous + rows;
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(10, rows as u32, false, tokens)],
            vec![BatchAppendV1::new(previous, rows, end, vec![0], None)],
        )
        .with_state_slots(vec![0])
    };
    let sink = execute(&host, &mut instance, &host_drafted(3, vec![4, 5, 6, 7]));
    assert!(sink.drafts.is_empty());
    assert_eq!(sink.views[0].raw().row_count, 4);
    assert!(
        !launched(&host, before)
            .iter()
            .any(|name| name.contains("mtp"))
    );
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    // Every state holds the prompt and the two kept rows.
    assert!(
        committed_state(&host, &instance, 0)
            .iter()
            .all(|history| *history == [1, 2, 3, 4, 5])
    );
    // A bundle-drafted item needs the drafter.
    let mut sink = SpeculativeSink::default();
    let bundle_drafted = batch(vec![drafted(10, 0, 5, 6, 3)]);
    assert!(
        instance
            .execute(&bundle_drafted.encode().unwrap(), &mut sink)
            .is_err()
    );
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    // The drafter package serves without speculation, running no drafter.
    let host = Rc::new(FakeHost::default());
    host.drafter.set(true);
    let mut instance = create_drafter(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    assert!(!launched(&host, 0).iter().any(|name| name.contains("mtp")));
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    // With the drafter, the host may not bring its own drafts.
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    execute(&host, &mut instance, &batch(vec![plain(10, 0, 0, &PROMPT)]));
    instance.commit().unwrap();
    let mut sink = SpeculativeSink::default();
    assert!(
        instance
            .execute(&host_drafted(3, vec![4, 5]).encode().unwrap(), &mut sink)
            .is_err()
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|log| log.contains("drafts its MTP drafter wrote"))
    );
}

/// Rank 0's committed linear-state entries of `slot` (convolution, then
/// delta rule, per linear layer) and its PLE entry, as token histories.
pub(super) fn committed_state(
    host: &FakeHost,
    instance: &BundleInstanceV1,
    slot: u32,
) -> Vec<Vec<u32>> {
    committed_state_on_rank(host, instance, slot, 0)
}

pub(super) fn committed_state_on_rank(
    host: &FakeHost,
    instance: &BundleInstanceV1,
    slot: u32,
    rank: usize,
) -> Vec<Vec<u32>> {
    let handle = instance.handle();
    let (entry, pool) = crate::instance::committed_entry(handle, slot).expect("owned slot");
    let [conv, delta, ple] = if rank == 0 {
        crate::instance::pools(handle)
    } else {
        let [conv, delta, ple, _, _] = crate::instance::state_views(handle, rank);
        [conv, delta, ple]
    };
    let sim = host.sim.borrow();
    let mut state = Vec::new();
    let shape = host.shape();
    for (span, bytes) in [
        (conv, shape.conv_entry_bytes),
        (delta, shape.delta_entry_bytes),
    ] {
        for layer in 0..shape.linear_layers() {
            let plane = DeviceBufferSpanV1::new(
                span.buffer,
                span.offset + layer * pool * bytes,
                pool * bytes,
            );
            state.push(sim.history_at(plane, i64::from(entry), bytes));
        }
    }
    state.push(sim.history_at(
        ple,
        i64::from(crate::instance::ple_entry(handle, slot)),
        shape.ple_entry_bytes(),
    ));
    state
}

/// Every drafter row read the target X of its own position (or, chained,
/// the drafter's row before it) and the token after it.
pub(super) fn assert_drafter_rows(host: &FakeHost, sequences: &BTreeMap<u32, Vec<u32>>) {
    let rows = std::mem::take(&mut host.sim.borrow_mut().drafter_rows);
    assert!(!rows.is_empty());
    for row in rows {
        let sequence = &sequences[&row.slot];
        let position = row.position as usize;
        assert_eq!(row.source.slot, row.slot, "{row:?}");
        if row.source.drafter {
            // A chained row: the previous drafter row, and its draft.
            assert_eq!(row.source.position + 1, row.position, "{row:?}");
            assert_eq!(row.source.token, truth(u64::from(row.position)), "{row:?}");
            assert_eq!(row.token, truth(u64::from(row.position) + 1), "{row:?}");
        } else {
            assert_eq!(row.source.position, row.position, "{row:?}");
            assert_eq!(row.source.token, sequence[position], "{row:?}");
            assert_eq!(row.token, sequence[position + 1], "{row:?}");
        }
    }
}

pub(super) fn truths(range: std::ops::Range<u64>) -> Vec<u32> {
    range.map(truth).collect()
}

#[test]
fn interleaved_target_and_drafter_graphs_release_in_reverse_capture_order() {
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 1, 0, &truths(0..5))]),
    );
    instance.commit().unwrap();
    // Different target buckets interleave with drafter graphs. Reversing
    // the order of whole caches cannot satisfy the host's resource stack.
    for (position, drafts) in [(5, 3), (6, 0), (7, 1), (8, 2)] {
        let item = if drafts == 0 {
            plain(10, 1, position, &[truth(position)])
        } else {
            drafted(10, 1, position, truth(position), drafts)
        };
        execute(&host, &mut instance, &batch(vec![item]));
        if drafts == 0 {
            instance.commit().unwrap();
        } else {
            instance.commit_accepted(&kept(&[(10, 1)])).unwrap();
        }
    }
    instance.destroy().unwrap();
    let journal = host.journal.borrow();
    let graph_ids = |prefix: &str| {
        journal
            .iter()
            .filter(|line| line.starts_with(prefix))
            .map(|line| {
                line.split_whitespace()
                    .nth(2)
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let mut captured = graph_ids("captured ");
    assert!(captured.len() > 12, "must capture multiple graph buckets");
    captured.reverse();
    assert_eq!(graph_ids("release graph "), captured);
    assert!(host.sim.borrow().graphs.is_empty());
    assert!(host.graph_nodes.borrow().is_empty());
    assert!(host.graph_collectives.borrow().is_empty());
    assert_eq!(host.released.get(), host.acquired.get());
}

#[test]
fn drafted_steps_emit_drafts_before_logits_and_commit_the_kept_rows() {
    let host = drafter_host(3);
    crate::pipeline::take_recorded();
    // The gathered layout: one rank-0 view per verify item.
    let mut instance = crate::instance::with_row_owners(false, || create_drafter(&host));
    let prompt = truths(0..5);
    execute(&host, &mut instance, &batch(vec![plain(10, 1, 0, &prompt)]));
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let sink = execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
    );
    // The drafter rows 4, 5 and 6 predict tokens 6, 7 and 8.
    assert_eq!(sink.draft_tokens(), [(10, truths(6..9))]);
    assert_eq!(sink.views.len(), 1);
    assert_eq!(sink.views[0].raw().row_count, 4);
    let names = launched(&host, before);
    let ranks = RANKS as usize;
    // One catch-up and first-draft run, then two chained runs, all of one
    // row: the first captures the drafter's bucket-1 graph and the chained
    // runs replay it, as the host's model saw (three drafts above).
    assert_eq!(count(&names, "chop_qwen38_mtp_gather_v1"), ranks);
    assert_eq!(count(&names, "chop_qwen38_draft_argmax_final_v1"), ranks);
    // Each rank's best reaches every rank.
    assert_eq!(
        count(&names, "chop_comm_gather_rows_bf16_v1"),
        ranks * (4 + 4 + 1)
    );
    // The target verifies the four rows in one bucket.
    assert_eq!(count(&names, "chop_qwen38_ngram_verify_v1"), ranks);
    assert_eq!(
        count(&names, "chop_qwen38_ple_checkpoints_indexed_v1"),
        ranks
    );
    assert_eq!(
        count(&names, "chop_linear_qwen38_delta_verify_v1"),
        36 * ranks
    );
    // The target's verify logits are F32; the drafter's argmax reads its
    // one-row BF16 head.
    assert_eq!(count(&names, "chop_gemm_qwen38_head_f32_m4_v1"), ranks);
    assert_eq!(count(&names, "chop_gemm_qwen38_head_m4_v1"), 0);
    assert!(count(&names, "chop_gemm_qwen38_head_m1_v1") >= ranks);
    assert_eq!(count(&names, "chop_gemm_qwen38_head_f32_m1_v1"), 0);
    assert_eq!(
        sink.views[0].raw().element_type,
        runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
    );
    // Verify rows and the drafter's graph runs are decode-like: they reduce
    // through the decode all-reduce and mix through the fused up mix. The
    // target's 96 sublayers and the drafter's two each reduce once.
    assert_eq!(count(&names, "chop_comm_qwen38_allreduce_hc_v1"), 0);
    let target = super::decode_comm(4, 4);
    let draft = super::decode_comm(4, 1);
    if target == draft {
        assert_eq!(count(&names, target), ranks * (96 + 2));
    } else {
        assert_eq!(count(&names, target), ranks * 96);
        assert_eq!(count(&names, draft), ranks * 2);
    }
    assert_eq!(count(&names, "chop_row_qwen38_hc_bottleneck_v1"), 0);
    // The verify item needs its kept rows, at most its four; a refused
    // commit changes nothing (the raw entries keep the wrapper pending).
    let handle = instance.handle();
    assert_eq!(
        bundle_runtime::entry::commit::<crate::instance::Instance>(handle),
        BundleStatusV1::InvalidState
    );
    for refused in [&[(10, 5)][..], &[(10, 0)], &[(11, 2)], &[]] {
        assert_eq!(
            bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(
                handle,
                &kept(refused)
            ),
            BundleStatusV1::InvalidArgument
        );
    }
    let previous = crate::instance::committed_entry(handle, 1).unwrap().0;
    let before_commit = host.launches.borrow().len();
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    let committed = crate::instance::committed_entry(handle, 1).unwrap().0;
    assert_ne!(committed, previous);
    // Only retained X copies at a verify commit.
    for (_, name, grid) in &host.launches.borrow()[before_commit..] {
        if name.contains("copy_entries") {
            assert_eq!(grid[1], 1, "commit copied linear layer planes");
            assert_eq!(grid[2], 2, "commit copied PLE state");
        }
    }

    let mut sequences = BTreeMap::new();
    sequences.insert(1, truths(0..16));
    assert_drafter_rows(&host, &sequences);
    // The next step catches row 5 up and drafts from row 6.
    let sink = execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 7, truth(7), 1)]),
    );
    assert_eq!(sink.draft_tokens(), [(10, truths(8..9))]);
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    assert_drafter_rows(&host, &sequences);
    assert_eq!(committed_state(&host, &instance, 1)[0], truths(0..9));
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    super::export_rows("mtp-drafted", &[&host]);
}

/// Speculative steps with partial accepts, a cancel, a one-row decode item
/// among verify items, a stretch of plain decodes (the drafter catches up
/// when its retained rows run out) and a prefill beside verify items leave
/// the same committed state as one decode step per token.
#[test]
fn verify_rollback_leaves_the_state_autoregressive_steps_leave() {
    for shape in Shape::ALL {
        verify_rollback_matches_autoregressive_steps(shape);
    }
}

fn verify_rollback_matches_autoregressive_steps(shape: Shape) {
    let spec_host = drafter_host_at(3, shape);
    let mut spec = create_drafter(&spec_host);
    let ar_host = Rc::new(FakeHost::at(shape));
    ar_host.ngram_host.set(ngram_in_host_memory(shape));
    let mut ar = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&ar_host)),
            &configuration_at(shape, ar_host.layout(), &[]),
        )
        .expect("fake-host create");
    ar_host.model(ar.handle());
    let slots = BTreeMap::from([(10, 2), (11, 5), (12, 7)]);
    let prompts = BTreeMap::from([(10, 6), (11, 3), (12, 4)]);
    let mut lengths = BTreeMap::new();

    // Prefill 10 and 11.
    let prefill = |request: u64| plain(request, slots[&request], 0, &truths(0..prompts[&request]));
    execute(
        &spec_host,
        &mut spec,
        &batch(vec![prefill(10), prefill(11)]),
    );
    spec.commit().unwrap();
    lengths.insert(10, 6);
    lengths.insert(11, 3);
    // Both draft; the first attempt is cancelled.
    let both = |lengths: &BTreeMap<u64, u64>, drafts: [u32; 2]| {
        batch(
            [10, 11]
                .into_iter()
                .zip(drafts)
                .map(|(request, count)| {
                    let length = lengths[&request];
                    if count == 0 {
                        plain(request, slots[&request], length, &[truth(length)])
                    } else {
                        drafted(request, slots[&request], length, truth(length), count)
                    }
                })
                .collect(),
        )
    };
    let first = both(&lengths, [3, 1]);
    execute(&spec_host, &mut spec, &first);
    spec.cancel().unwrap();
    for request in [10, 11] {
        spec.retire_request(request).unwrap();
    }
    execute(
        &spec_host,
        &mut spec,
        &batch(vec![prefill(10), prefill(11)]),
    );
    spec.commit().unwrap();
    let sink = execute(&spec_host, &mut spec, &first);
    assert_eq!(
        sink.draft_tokens(),
        [(10, truths(7..10)), (11, truths(4..5))]
    );
    spec.commit_accepted(&kept(&[(10, 4), (11, 1)])).unwrap();
    *lengths.get_mut(&10).unwrap() += 4;
    *lengths.get_mut(&11).unwrap() += 1;
    // 10 drafts beside 11's one-row decode, keeping only its bonus row.
    execute(&spec_host, &mut spec, &both(&lengths, [2, 0]));
    spec.commit_accepted(&kept(&[(10, 1)])).unwrap();
    for length in lengths.values_mut() {
        *length += 1;
    }
    // Twelve plain decode steps.
    for _ in 0..12 {
        let step = batch(
            [10, 11]
                .into_iter()
                .map(|request| {
                    let length = lengths[&request];
                    plain(request, slots[&request], length, &[truth(length)])
                })
                .collect(),
        );
        execute(&spec_host, &mut spec, &step);
        spec.commit().unwrap();
        for length in lengths.values_mut() {
            *length += 1;
        }
    }
    execute(&spec_host, &mut spec, &both(&lengths, [3, 3]));
    spec.commit_accepted(&kept(&[(10, 3), (11, 4)])).unwrap();
    *lengths.get_mut(&10).unwrap() += 3;
    *lengths.get_mut(&11).unwrap() += 4;
    // A prefill beside a drafting item and a decode item.
    let length = |request: u64| lengths[&request];
    let mixed = batch(vec![
        prefill(12),
        drafted(10, 2, length(10), truth(length(10)), 3),
        plain(11, 5, length(11), &[truth(length(11))]),
    ]);
    execute(&spec_host, &mut spec, &mixed);
    spec.commit_accepted(&kept(&[(10, 2)])).unwrap();
    *lengths.get_mut(&10).unwrap() += 2;
    *lengths.get_mut(&11).unwrap() += 1;
    lengths.insert(12, 4);
    let sequences = slots
        .values()
        .map(|&slot| (slot as u32, truths(0..64)))
        .collect::<BTreeMap<_, _>>();
    assert_drafter_rows(&spec_host, &sequences);

    // The same tokens, one per decode step.
    run(
        &ar_host,
        &mut ar,
        "prefill",
        step(&[(10, 2, 0, &truths(0..6)), (11, 5, 0, &truths(0..3))]),
    );
    ar.commit().unwrap();
    run(
        &ar_host,
        &mut ar,
        "prefill 12",
        step(&[(12, 7, 0, &truths(0..4))]),
    );
    ar.commit().unwrap();
    let mut ar_lengths = BTreeMap::from([(10, 6), (11, 3)]);
    while ar_lengths
        .iter()
        .any(|(request, &length)| length < lengths[request])
    {
        let items = ar_lengths
            .iter()
            .filter(|&(request, &length)| length < lengths[request])
            .map(|(&request, &length)| (request, slots[&request], length, vec![truth(length)]))
            .collect::<Vec<_>>();
        let items = items
            .iter()
            .map(|(request, slot, length, tokens)| (*request, *slot, *length, tokens.as_slice()))
            .collect::<Vec<_>>();
        run(&ar_host, &mut ar, "decode", step(&items));
        ar.commit().unwrap();
        for (request, ..) in &items {
            *ar_lengths.get_mut(request).unwrap() += 1;
        }
    }
    for (&request, &slot) in &slots {
        let expected = truths(0..lengths[&request]);
        let spec_state = committed_state(&spec_host, &spec, slot as u32);
        let ar_state = committed_state(&ar_host, &ar, slot as u32);
        assert_eq!(spec_state, ar_state, "request {request}");
        assert!(
            spec_state.iter().all(|history| *history == expected),
            "request {request}"
        );
    }

    // Retirement frees slot 2 for a request with other tokens; nothing of
    // request 10 survives in its state.
    spec.retire_request(10).unwrap();
    let prompt = [5, 6, 7];
    execute(
        &spec_host,
        &mut spec,
        &batch(vec![plain(13, 2, 0, &prompt)]),
    );
    spec.commit().unwrap();
    for (length, bonus) in [(3, 8), (4, 9)] {
        execute(
            &spec_host,
            &mut spec,
            &batch(vec![drafted(13, 2, length, bonus, 1)]),
        );
        // The fake drafter's draft is wrong for this sequence.
        spec.commit_accepted(&kept(&[(13, 1)])).unwrap();
    }
    assert!(
        committed_state(&spec_host, &spec, 2)
            .iter()
            .all(|history| *history == [5, 6, 7, 8, 9])
    );
    assert_drafter_rows(&spec_host, &BTreeMap::from([(2, vec![5, 6, 7, 8, 9, 0])]));
}

/// A continuing prefill after a decode stretch left
/// retained rows behind. Its drafter rows must fit a prefill-sized run, and
/// the state after it must equal one token at a time.
#[test]
fn a_full_continuing_prefill_after_decodes_fits_the_drafter() {
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 0, 0, &truths(0..3))]),
    );
    instance.commit().unwrap();
    // Seven decodes: frontier 2, length 10, eight rows outstanding.
    for length in 3..10 {
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 0, length, &[truth(length)])]),
        );
        instance.commit().unwrap();
    }
    let chunk = truths(10..10 + u64::from(PREFILL_TOKENS));
    let sink = execute(&host, &mut instance, &batch(vec![plain(10, 0, 10, &chunk)]));
    assert_eq!(sink.views.len(), 1);
    instance.commit().unwrap();
    let length = 10 + u64::from(PREFILL_TOKENS);
    let expected = truths(0..length);
    assert!(
        committed_state(&host, &instance, 0)
            .iter()
            .all(|history| *history == expected)
    );
    assert_drafter_rows(&host, &BTreeMap::from([(0, truths(0..length + 1))]));
    // Drafting resumes from the chunk's last row.
    let sink = execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 0, length, truth(length), 1)]),
    );
    assert_eq!(sink.draft_tokens(), [(10, vec![truth(length + 1)])]);
}

/// Refuses the drafts, so the step fails after the drafter ran.
struct RejectDrafts;

impl LogitsRowSinkV1 for RejectDrafts {
    fn limits(&self) -> ResultSinkLimitsV1 {
        SpeculativeLimits::limits()
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::HostError)
    }

    fn emit_draft_tokens(&mut self, _drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::HostError)
    }
}

/// When a drafted step fails and the
/// drafter ring's restore then fails on one rank, the step stays open and
/// cancellable (execute returns the retained-transaction success), commit is
/// refused, cancel restores every rank, and the identical step then runs
/// and commits the state one token at a time leaves. With a working
/// restore the failed step is refused outright and leaves nothing open.
#[test]
fn a_failed_drafter_rollback_leaves_a_cancellable_step() {
    for rank in [None, Some(0), Some(1), Some(2), Some(3)] {
        let host = drafter_host(3);
        let mut instance = create_drafter(&host);
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 0, 0, &truths(0..3))]),
        );
        instance.commit().unwrap();
        let step = batch(vec![drafted(10, 0, 3, truth(3), 3)]);
        host.fail_restore_rank.set(rank);
        let result = instance.execute(&step.encode().unwrap(), &mut RejectDrafts);
        assert_eq!(host.fail_restore_rank.get(), None, "{rank:?}");
        if rank.is_none() {
            assert!(result.is_err());
        } else {
            result.unwrap();
            assert_eq!(
                bundle_runtime::entry::commit::<crate::instance::Instance>(instance.handle()),
                BundleStatusV1::InvalidState
            );
            let before = host.copies.borrow().len();
            instance.cancel().unwrap();
            // Every rank's drafter ring is restored again.
            assert_eq!(host.copies.borrow().len() - before, RANKS as usize);
        }
        instance.retire_request(10).unwrap();
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 0, 0, &truths(0..3))]),
        );
        instance.commit().unwrap();
        let sink = execute(&host, &mut instance, &step);
        assert_eq!(sink.draft_tokens(), [(10, truths(4..7))]);
        instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
        assert!(
            committed_state(&host, &instance, 0)
                .iter()
                .all(|history| *history == truths(0..7)),
            "{rank:?}"
        );
        instance.retire_request(10).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

/// The QSA ring's declared cancel save for a step that only drafts: the
/// drafter's layer goes to the adjacent shadow, and cancel copies it back.
#[test]
fn a_cancelled_drafting_step_restores_the_declared_drafter_layer() {
    use bundle_runtime::state::{Cancel, Extent, Region, Save, Writer};
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    let handle = instance.handle();
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 0, 0, &truths(0..3))]),
    );
    instance.commit().unwrap();
    let state = model_qwen38_contract::state::request_state(Shape::TP4, &KV_FORMATS[1], true);
    let ring = &state.buffers[3];
    let drafter_layer = Region::Layers {
        first: FULL_LAYERS,
        count: 1,
    };
    assert_eq!(
        ring.cancel,
        Cancel::Restore(vec![
            Save {
                writer: Writer::Prefill,
                region: Region::All,
            },
            Save {
                writer: Writer::Drafter,
                region: drafter_layer,
            },
        ])
    );
    let extent = Extent {
        served: host.served(),
        bucket: host.served().next_power_of_two(),
        verify_rows: 4,
    };
    let (offset, length) = ring.region(extent, drafter_layer).unwrap();
    let (separate, shadow) = ring.shadow_offset(extent, offset).unwrap();
    assert!(!separate);
    let span = |rank: usize, at: u64| {
        let view = crate::instance::state_views(handle, rank)[3];
        DeviceBufferSpanV1::new(view.buffer, view.offset + at, length)
    };
    let ranks = RANKS as usize;
    let fill = |seed: u8| {
        for rank in 0..ranks {
            let bytes = (0..length)
                .map(|at| (at as u8).wrapping_mul(29) ^ seed ^ rank as u8)
                .collect::<Vec<_>>();
            host.sim
                .borrow_mut()
                .write(span(rank, offset), &bytes, &host.extents.borrow());
        }
    };
    let ring_copies = |from: usize| {
        host.copies.borrow()[from..]
            .iter()
            .copied()
            .filter(|(source, _)| source.length == length)
            .collect::<Vec<_>>()
    };
    fill(0x5a);
    let committed = (0..ranks)
        .map(|rank| host.device_bytes(span(rank, offset)))
        .collect::<Vec<_>>();
    let before = host.copies.borrow().len();
    execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 0, 3, truth(3), 3)]),
    );
    let saves = (0..ranks)
        .map(|rank| (span(rank, offset), span(rank, shadow)))
        .collect::<Vec<_>>();
    assert_eq!(ring_copies(before), saves);
    fill(0xc3);
    let before = host.copies.borrow().len();
    instance.cancel().unwrap();
    assert_eq!(
        ring_copies(before),
        saves
            .iter()
            .map(|&(live, saved)| (saved, live))
            .collect::<Vec<_>>()
    );
    for (rank, committed) in committed.iter().enumerate() {
        assert!(
            host.device_bytes(span(rank, offset)) == *committed,
            "rank {rank}"
        );
    }
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
}

/// Prefix snapshots with the drafter: a capture after a prefill commit and
/// one at a decode boundary whose drafter frontier lags (as a preemption
/// captures), each restored for another request, which then drafts and
/// commits exactly as the source would have.
#[test]
fn a_restored_request_resumes_drafting_from_the_snapshot() {
    for shape in Shape::ALL {
        for fp8_kv in [false, true] {
            restored_request_resumes_drafting_at(shape, fp8_kv);
        }
    }
}

/// The same prefix hits on the FP8 KV layout with the drafter (decision
/// FP8 KV): its smaller pages give the snapshot 58 of them, every copy moves
/// the state bytes through those pages, and the restored requests draft and
/// commit as with BF16 KV.
#[test]
fn a_prefix_hit_with_fp8_kv_and_the_drafter_resumes_drafting() {
    let host = restored_request_resumes_drafting(true, 58);
    let copies = host
        .journal
        .borrow()
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && launch_symbol(line) == model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED
        })
        .map(|line| {
            assert!(
                line.contains(&format!("ScalarU64({PAGE_BYTES_FP8_MTP})")),
                "{line}"
            );
        })
        .count();
    // Two captures and two restores, one copy per rank each.
    assert_eq!(copies, 4 * RANKS as usize);
}

/// Captures after a prefill and after a verify commit, restored for two
/// other requests that then draft; returns the host for its journal.
fn restored_request_resumes_drafting(fp8_kv: bool, expected_pages: u64) -> Rc<FakeHost> {
    let host = restored_request_resumes_drafting_at(Shape::TP4, fp8_kv);
    assert_eq!(
        super::expected_snapshot_pages(4, host.layout().1, true),
        expected_pages
    );
    host
}

/// The restore scenario below at `shape`'s topology; the snapshot takes the
/// pages the documented per-rank state sizes give.
fn restored_request_resumes_drafting_at(shape: Shape, fp8_kv: bool) -> Rc<FakeHost> {
    let host = drafter_host_at(3, shape);
    host.fp8_kv.set(fp8_kv);
    let expected_pages = super::expected_snapshot_pages(shape.ranks(), host.layout().1, true);
    let mut instance = create_drafter(&host);
    host.backing.set(true);
    let pages = instance.describe_prefix_state().unwrap();
    assert_eq!(pages, expected_pages);
    let snapshot = |first: u64| (first..first + pages).collect::<Vec<_>>();
    // Request 10 prefills 70 tokens, a prompt end inside a QSA block.
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 1, 0, &truths(0..70))]),
    );
    instance.commit().unwrap();
    instance
        .capture_prefix_state(10, 70, &snapshot(300))
        .unwrap();
    // It drafts and keeps three rows: its drafter frontier stays at 70.
    execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 70, truth(70), 3)]),
    );
    instance.commit_accepted(&kept(&[(10, 3)])).unwrap();
    instance
        .capture_prefix_state(10, 73, &snapshot(500))
        .unwrap();
    instance.retire_request(10).unwrap();
    let sequences = BTreeMap::from([(1, truths(0..96))]);
    assert_drafter_rows(&host, &sequences);

    for (request, length, first) in [(11, 73, 500), (12, 70, 300)] {
        instance
            .restore_prefix_state_in_slot(request, &truths(0..length), &snapshot(first), 1)
            .unwrap();
        assert!(
            committed_state(&host, &instance, 1)
                .iter()
                .all(|history| *history == truths(0..length)),
            "request {request}"
        );
        // The first step drafts from the restored rows; the second keeps
        // drafting from where the first commit left the frontier.
        let sink = execute(
            &host,
            &mut instance,
            &batch(vec![drafted(request, 1, length, truth(length), 3)]),
        );
        assert_eq!(
            sink.draft_tokens(),
            [(request, truths(length + 1..length + 4))]
        );
        instance.commit_accepted(&kept(&[(request, 2)])).unwrap();
        let sink = execute(
            &host,
            &mut instance,
            &batch(vec![drafted(request, 1, length + 2, truth(length + 2), 3)]),
        );
        assert_eq!(
            sink.draft_tokens(),
            [(request, truths(length + 3..length + 6))]
        );
        instance.commit_accepted(&kept(&[(request, 4)])).unwrap();
        assert_drafter_rows(&host, &sequences);
        assert!(
            committed_state(&host, &instance, 1)
                .iter()
                .all(|history| *history == truths(0..length + 6)),
            "request {request}"
        );
        instance.retire_request(request).unwrap();
    }
    // A capture while a step is open, or of a length the request did not
    // commit, is refused.
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(13, 1, 0, &truths(0..70))]),
    );
    assert!(
        instance
            .capture_prefix_state(13, 70, &snapshot(700))
            .is_err()
    );
    instance.commit().unwrap();
    assert_eq!(
        instance.capture_prefix_state(13, 69, &snapshot(700)),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    // Two captures and two restores: one copy per rank each, of this
    // topology's entry; the draft reductions use its rank count.
    let journal = host.journal.borrow().clone();
    let copy = crate::launch::per_rank(
        model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED,
        shape,
    );
    let launches = |symbol: &str| {
        journal
            .iter()
            .filter(|line| line.starts_with("launch ") && launch_symbol(line) == symbol)
            .count()
    };
    assert_eq!(launches(&copy), 4 * shape.ranks() as usize);
    let partial = crate::launch::per_rank(
        model_qwen38_contract::sm120::owned::DRAFT_ARGMAX_PARTIAL,
        shape,
    );
    assert!(launches(&partial) > 0);
    assert_eq!(launches(&partial) % shape.ranks() as usize, 0);
    instance.retire_request(13).unwrap();
    instance.destroy().unwrap();
    host
}

/// The host copies only the page holding a shared prefix's last row before
/// an append (the package declares last-page rewrites), so a drafter row a
/// continuation writes below it would land in a full page an anchor shares
/// with its hits. A prefill ending on a page boundary (B = 64, frontier 63)
/// and one ending inside a page (B = 70, frontier 69) restore for any
/// request, and their continuations write only rows of the last page. A
/// verify commit whose frontier lags into the previous page (B = 66,
/// frontier 62), a preemption's private anchor, restores only for the
/// request that captured it, whose continuation rewrites rows 62 and 63 of
/// its own pages and then drafts from them.
#[test]
fn drafter_rows_below_the_last_page_restore_only_for_their_own_request() {
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    host.backing.set(true);
    let pages = instance.describe_prefix_state().unwrap();
    let snapshot = |first: u64| (first..first + pages).collect::<Vec<_>>();
    let mut captured = Vec::new();
    // (request, prompt, drafted rows kept after it, snapshot pages).
    for (request, prompt, kept_rows, first) in
        [(10, 64, 0, 500), (11, 62, 4, 550), (12, 70, 0, 600)]
    {
        let slot = 1;
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(request, slot, 0, &truths(0..prompt))]),
        );
        instance.commit().unwrap();
        let mut length = prompt;
        if kept_rows > 0 {
            execute(
                &host,
                &mut instance,
                &batch(vec![drafted(request, slot, length, truth(length), 3)]),
            );
            instance
                .commit_accepted(&kept(&[(request, kept_rows)]))
                .unwrap();
            length += u64::from(kept_rows);
        }
        instance
            .capture_prefix_state(request, length, &snapshot(first))
            .unwrap();
        captured.push((request, length, first));
        instance.retire_request(request).unwrap();
    }
    host.sim.borrow_mut().drafter_rows.clear();
    for &(owner, length, first) in &captured {
        let tail = (length - 1) / PAGE_TOKENS * PAGE_TOKENS;
        // Another request: every snapshot whose frontier sits in its last
        // page; the lagging one is refused and leaves the slot free.
        let other = owner + 10;
        let restored =
            instance.restore_prefix_state_in_slot(other, &truths(0..length), &snapshot(first), 1);
        if length == 66 {
            assert_eq!(
                restored,
                Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
            );
            assert_eq!(crate::instance::owner(instance.handle(), 1), None);
        } else {
            restored.unwrap();
            execute(
                &host,
                &mut instance,
                &batch(vec![drafted(other, 1, length, truth(length), 3)]),
            );
            instance.commit_accepted(&kept(&[(other, 1)])).unwrap();
            let rows = std::mem::take(&mut host.sim.borrow_mut().drafter_rows);
            assert!(!rows.is_empty());
            for row in rows {
                assert!(
                    u64::from(row.position) >= tail,
                    "anchor {length}: drafter row {} is below its last page",
                    row.position
                );
            }
            instance.retire_request(other).unwrap();
        }
        // The capturing request itself restores and drafts from the
        // snapshot, rewriting its rows from the frontier on.
        instance
            .restore_prefix_state_in_slot(owner, &truths(0..length), &snapshot(first), 1)
            .unwrap();
        let sink = execute(
            &host,
            &mut instance,
            &batch(vec![drafted(owner, 1, length, truth(length), 3)]),
        );
        assert_eq!(
            sink.draft_tokens(),
            [(owner, truths(length + 1..length + 4))]
        );
        instance.commit_accepted(&kept(&[(owner, 4)])).unwrap();
        let lowest = host
            .sim
            .borrow()
            .drafter_rows
            .iter()
            .map(|row| u64::from(row.position))
            .min()
            .unwrap();
        assert_eq!(lowest, if length == 66 { 62 } else { length - 1 });
        assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..length + 8))]));
        assert!(
            committed_state(&host, &instance, 1)
                .iter()
                .all(|history| *history == truths(0..length + 4)),
            "anchor {length}"
        );
        instance.retire_request(owner).unwrap();
    }
}

fn fresh_prompt_catch_up(tokens: u64) {
    let host = drafter_host(3);
    // The gathered layout: one rank-0 view per verify item.
    let mut instance = crate::instance::with_row_owners(false, || create_drafter(&host));
    let prompt = truths(0..tokens);
    execute(&host, &mut instance, &batch(vec![plain(10, 1, 0, &prompt)]));
    instance.commit().unwrap();
    let sink = execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, tokens, truth(tokens), 3)]),
    );
    assert_eq!(sink.draft_tokens(), [(10, truths(tokens + 1..tokens + 4))]);
    assert_eq!(sink.views[0].raw().row_count, 4);
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..tokens + 5))]));
    assert_eq!(
        committed_state(&host, &instance, 1)[0],
        truths(0..tokens + 2)
    );
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

#[test]
fn fresh_prompt_3() {
    fresh_prompt_catch_up(3);
}
#[test]
fn fresh_prompt_8() {
    fresh_prompt_catch_up(8);
}
#[test]
fn fresh_prompt_9() {
    fresh_prompt_catch_up(9);
}
#[test]
fn fresh_prompt_20() {
    fresh_prompt_catch_up(20);
}
#[test]
fn fresh_prompt_39() {
    fresh_prompt_catch_up(39);
}
#[test]
fn fresh_prompt_40() {
    fresh_prompt_catch_up(40);
}

#[test]
fn fresh_prompt_10() {
    fresh_prompt_catch_up(10);
}
#[test]
fn fresh_prompt_33() {
    fresh_prompt_catch_up(33);
}
#[test]
fn fresh_prompt_34() {
    fresh_prompt_catch_up(34);
}

/// A prefill, a drafted verify step with a partial accept and a cancelled
/// one, then a plain decode, as the journal of a drafter package.
fn drafter_journal(fp8_kv: bool, scales: Option<&'static [[f32; 2]]>) -> Vec<String> {
    let host = drafter_host(3);
    host.fp8_kv.set(fp8_kv);
    let mut instance = create_drafter(&host);
    if let Some(scales) = scales {
        crate::instance::force_kv_scales(instance.handle(), scales);
    }
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 1, 0, &truths(0..5))]),
    );
    instance.commit().unwrap();
    execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
    );
    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
    execute(
        &host,
        &mut instance,
        &batch(vec![drafted(10, 1, 7, truth(7), 1)]),
    );
    instance.cancel().unwrap();
    instance.retire_request(10).unwrap();
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 1, 0, &truths(0..7))]),
    );
    instance.commit().unwrap();
    execute(
        &host,
        &mut instance,
        &batch(vec![plain(10, 1, 7, &[truth(7)])]),
    );
    instance.commit().unwrap();
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    host.journal.take()
}

/// With the drafter, the FP8 layout quantizes only the main
/// full layers: verify rows and prefills use their FP8 entries, and the MTP
/// drafter's layer keeps its BF16 KV write and attention, at scale 1.0, on
/// BF16 planes after the FP8 layers. Every other journal line is the BF16
/// drafter package's.
#[test]
fn fp8_kv_with_the_drafter_keeps_the_mtp_layer_bf16() {
    let bf16 = drafter_journal(false, None);
    let fp8 = drafter_journal(true, None);
    let changed = fp8_changes(&bf16, &fp8);
    assert!(changed.contains("chop_qsa_qwen38_attn_fp8_decode_v1"));
    assert!(changed.contains("chop_attn_qwen38_qk_norm_rope_kv_fp8_v1"));
    let scaled = drafter_journal(true, Some(&DISTINCT_SCALES));
    let seen = assert_layer_storage(&scaled, &KV_FORMATS[3]);
    for symbol in [
        "chop_attn_qwen38_qk_norm_rope_kv_bf16_v1",
        "chop_qsa_qwen38_attn_bf16_decode_v1",
    ] {
        assert!(
            seen.contains(&(symbol.to_owned(), MTP_LAYER)),
            "the MTP layer never ran {symbol}"
        );
    }
    let main_layers = seen
        .iter()
        .filter(|(symbol, _)| symbol == "chop_qsa_qwen38_attn_fp8_decode_v1")
        .count();
    assert_eq!(main_layers, FULL_LAYERS as usize);
}

/// Serialize the actual ring keys and positions written by index_key_write,
/// in slot-relative order, on every rank. Copies and rollback move this
/// provenance through the same byte spans as the device state.
pub(super) fn offload_ring_state(
    host: &FakeHost,
    instance: &BundleInstanceV1,
    slot: u32,
) -> Vec<Vec<u32>> {
    let sim = host.sim.borrow();
    let mut state: Vec<Vec<u32>> = Vec::new();
    for rank in 0..host.shape().ranks.count() {
        let ring = crate::instance::state_views(instance.handle(), rank)[3];
        for piece in super::host_table(host, slot).descriptors {
            // The ring's keys and positions.
            if piece.buffer != 3 {
                continue;
            }
            let start = ring.offset + piece.offset;
            let entries = sim
                .histories
                .range((ring.buffer.slot(), start)..(ring.buffer.slot(), start + piece.length));
            state.push(
                entries
                    .flat_map(|(&(_, offset), history)| {
                        std::iter::once((offset - start) as u32).chain(history.iter().copied())
                    })
                    .collect(),
            );
        }
    }
    assert!(
        state
            .iter()
            .take(2 * FULL_LAYERS as usize)
            .all(|piece| !piece.is_empty()),
        "QSA ring provenance must exist before continuation"
    );
    state
}

/// QSA block keys in the independently relocated serving pages. The sparse
/// model tracks block-key provenance, not the numerical attention K/V.
fn offload_block_keys(host: &FakeHost, slot: u64, tokens: u64) -> Vec<Vec<u32>> {
    let sim = host.sim.borrow();
    let mut state = Vec::new();
    for rank in 0..host.shape().ranks.count() {
        for page in 0..tokens.div_ceil(PAGE_TOKENS) {
            let start = (slot * 64 + page) * host.layout().1;
            let entries = sim.histories.range(
                (KV_SLOT + rank as u32, start)..(KV_SLOT + rank as u32, start + host.layout().1),
            );
            let keys: Vec<u32> = entries
                .flat_map(|(&(_, offset), history)| {
                    std::iter::once((offset - start) as u32).chain(history.iter().copied())
                })
                .collect();
            assert!(
                !keys.is_empty(),
                "each serving page must contain QSA block keys"
            );
            state.push(keys);
        }
    }
    state
}

/// GPU M5 checks numerical logits. These CPU rows consume recurrent state,
/// QSA keys/positions and the MTP retained rows on every rank.
#[test]
fn offload_decode_and_verify_boundaries_resume_in_another_slot_bitwise() {
    for drafter in [false, true] {
        for verify in [false, true] {
            for (fp8, draft_next) in [(false, false), (false, true), (true, false), (true, true)] {
                let host = drafter_host(3);
                host.drafter.set(drafter);
                host.fp8_kv.set(fp8);
                // Target and MTP state follow the served count: resume in
                // the highest of 40 slots.
                let destination_slot = 39;
                host.served.set(Some(40));
                let mut instance = create(&host);
                host.model(instance.handle());
                host.backing.set(true);
                host.sim.borrow_mut().track_qsa = true;
                execute(
                    &host,
                    &mut instance,
                    &batch(vec![plain(10, 1, 0, &truths(0..70))]),
                );
                instance.commit().unwrap();
                let boundary = if verify {
                    let mut item = drafted(10, 1, 70, truth(70), 3);
                    if !drafter {
                        item.0 = BatchItemV1::verify(10, 4, false, truths(70..74));
                    }
                    execute(&host, &mut instance, &batch(vec![item]));
                    instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
                    72
                } else {
                    execute(
                        &host,
                        &mut instance,
                        &batch(vec![plain(10, 1, 70, &[truth(70)])]),
                    );
                    instance.commit().unwrap();
                    71
                };
                let pages = super::snapshot_pages(&host, 700);
                instance.capture_prefix_state(10, boundary, &pages).unwrap();
                // The host tier moves serving pages separately from the
                // bundle snapshot. Destination pages differ from the source,
                // including the partial tail and the drafter's block keys.
                for rank in 0..host.shape().ranks.count() {
                    let pool = host.resolve_serving_kv_pool(rank as u32).unwrap();
                    for page in 0..boundary.div_ceil(PAGE_TOKENS) {
                        let at = |slot: u64| {
                            DeviceBufferSpanV1::new(
                                pool.buffer,
                                pool.offset + (slot * 64 + page) * host.layout().1,
                                host.layout().1,
                            )
                        };
                        host.sim
                            .borrow_mut()
                            .copy(at(7), at(1), &host.extents.borrow());
                    }
                }
                instance
                    .restore_prefix_state_in_slot(
                        11,
                        &truths(0..boundary),
                        &pages,
                        destination_slot,
                    )
                    .unwrap();
                for rank in 0..host.shape().ranks.count() {
                    assert_eq!(
                        committed_state_on_rank(&host, &instance, 1, rank),
                        committed_state_on_rank(&host, &instance, destination_slot as u32, rank)
                    );
                }
                assert_eq!(
                    offload_ring_state(&host, &instance, 1),
                    offload_ring_state(&host, &instance, destination_slot as u32),
                    "restored QSA keys and positions must equal the committed ring"
                );
                assert_eq!(
                    offload_block_keys(&host, 1, boundary),
                    offload_block_keys(&host, 7, boundary)
                );
                let mut rows = Vec::new();
                for (request, slot, page_group) in [(10, 1, 1), (11, destination_slot, 7)] {
                    host.sim.borrow_mut().drafter_rows.clear();
                    host.sim.borrow_mut().qsa_blocks.clear();
                    // Exercise ordinary decode and a separate immediate draft:
                    // plain decode can leave retained X unused for several steps.
                    let mut item = if draft_next {
                        let mut item = drafted(request, page_group, boundary, truth(boundary), 3);
                        if !drafter {
                            item.0 = BatchItemV1::verify(
                                request,
                                4,
                                false,
                                truths(boundary..boundary + 4),
                            );
                        }
                        item
                    } else {
                        plain(request, page_group, boundary, &[truth(boundary)])
                    };
                    // Keep serving page IDs independent of the state slot.
                    item.2 = slot;
                    let sink = execute(&host, &mut instance, &batch(vec![item]));
                    assert_eq!(
                        sink.views
                            .iter()
                            .map(|view| view.raw().row_count)
                            .sum::<u32>(),
                        if draft_next { 4 } else { 1 }
                    );
                    let sim = host.sim.borrow();
                    let mut row = Vec::new();
                    for (stream, block) in &sim.qsa_blocks {
                        row.push(*stream);
                        row.extend(block);
                    }
                    if draft_next {
                        assert!(
                            !row.is_empty(),
                            "verify continuation must complete QSA blocks"
                        );
                    }
                    if drafter && draft_next {
                        assert!(!sim.drafter_rows.is_empty());
                        assert!(
                            sim.drafter_rows.iter().any(|row| !row.source.drafter
                                && u64::from(row.source.position) < boundary)
                        );
                        for read in &sim.drafter_rows {
                            let p = u64::from(read.position);
                            if read.source.drafter {
                                assert_eq!(read.source.position + 1, read.position);
                                assert_eq!(read.source.token, truth(p));
                            } else {
                                assert_eq!(read.source.position, read.position);
                                assert_eq!(read.source.token, truth(p));
                            }
                            assert_eq!(read.token, truth(p + 1));
                            row.extend([
                                read.source.position,
                                read.source.token,
                                read.source.drafter as u32,
                                read.position,
                                read.token,
                            ]);
                        }
                    }
                    drop(sim);
                    if draft_next {
                        instance.commit_accepted(&kept(&[(request, 1)])).unwrap();
                    } else {
                        instance.commit().unwrap();
                    }
                    for rank in 0..host.shape().ranks.count() {
                        let state = committed_state_on_rank(&host, &instance, slot as u32, rank);
                        assert_eq!(state.len(), 2 * LINEAR_LAYERS as usize + 1);
                        assert!(
                            state
                                .iter()
                                .all(|history| *history == truths(0..boundary + 1))
                        );
                        row.extend(state.into_iter().flatten());
                    }
                    row.extend(
                        offload_ring_state(&host, &instance, slot as u32)
                            .into_iter()
                            .flatten(),
                    );
                    row.extend(
                        offload_block_keys(&host, page_group, boundary + 1)
                            .into_iter()
                            .flatten(),
                    );
                    rows.push(
                        row.into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    );
                }
                assert_eq!(
                    rows[0], rows[1],
                    "drafter={drafter} verify={verify} fp8={fp8} draft_next={draft_next}"
                );
                instance.destroy().unwrap();
            }
        }
    }
}

#[test]
fn owned_verify_splits_items_across_ranks_without_changing_drafts_or_commits() {
    for drafts in [vec![2_u32, 5, 1], vec![1; 16]] {
        let mut results = Vec::new();
        for enabled in [false, true] {
            let host = drafter_host(*drafts.iter().max().unwrap());
            let mut instance = crate::instance::with_row_owners(enabled, || create_drafter(&host));
            let prompt = truths(0..5);
            execute(
                &host,
                &mut instance,
                &batch(
                    (0..drafts.len())
                        .map(|i| plain(10 + i as u64, i as u64, 0, &prompt))
                        .collect(),
                ),
            );
            instance.commit().unwrap();
            let before = host.launches.borrow().len();
            let sink = execute(
                &host,
                &mut instance,
                &batch(
                    drafts
                        .iter()
                        .enumerate()
                        .map(|(i, &drafts)| drafted(10 + i as u64, i as u64, 5, truth(5), drafts))
                        .collect(),
                ),
            );
            let rows = drafts.iter().map(|drafts| drafts + 1).sum::<u32>();
            let block = rows.next_power_of_two().div_ceil(RANKS);
            let mut first = 0;
            for view in &sink.views {
                let view = view.raw();
                assert_eq!(view.first_item_index, first);
                assert_eq!(view.rank, if enabled { first / block } else { 0 });
                assert_eq!(
                    view.row_count,
                    if enabled {
                        (rows - first).min(block)
                    } else {
                        rows
                    }
                );
                first += view.row_count;
            }
            assert_eq!(first, rows);
            let names = launched(&host, before);
            assert_eq!(
                count(&names, "chop_comm_exchange_rows_f32_v1"),
                if enabled { RANKS as usize } else { 0 }
            );
            let kept_rows = drafts
                .iter()
                .enumerate()
                .map(|(i, _)| (10 + i as u64, 2))
                .collect::<Vec<_>>();
            instance.commit_accepted(&kept(&kept_rows)).unwrap();
            let states = (0..drafts.len())
                .map(|i| committed_state(&host, &instance, i as u32))
                .collect::<Vec<_>>();
            results.push((sink.draft_tokens(), states));
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
fn hc_precompute_covers_target_verify_and_the_drafter_graph() {
    let host = drafter_host(1);
    crate::pipeline::take_recorded();
    let mut instance = crate::instance::with_hc_precompute(true, || create_drafter(&host));
    let prompt = truths(0..5);
    execute(
        &host,
        &mut instance,
        &batch(
            (0..16)
                .map(|slot| plain(10 + slot, slot, 0, &prompt))
                .collect(),
        ),
    );
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let sink = execute(
        &host,
        &mut instance,
        &batch(
            (0..16)
                .map(|slot| drafted(10 + slot, slot, 5, truth(5), 1))
                .collect(),
        ),
    );
    assert_eq!(sink.drafts.len(), 16);
    let names = launched(&host, before);
    assert_eq!(
        count(&names, "chop_gemm_qwen38_hc_in_precompute_c32_v1"),
        96 * RANKS as usize
    );
    assert_eq!(
        count(&names, "chop_gemm_qwen38_hc_in_precompute_c16_v1"),
        2 * RANKS as usize
    );
    super::hc::check_bottleneck_bindings(&host, before);
    instance
        .commit_accepted(&kept(
            &(0..16).map(|slot| (10 + slot, 1)).collect::<Vec<_>>(),
        ))
        .unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    super::export_rows("mtp-hc-precompute", &[&host]);
}

/// Small served instances opt into wide target graphs and retain only
/// accepted history across repeated K2/K3 steps.
#[test]
fn c16_c32_deeper_drafts_use_opt_in_wide_graphs() {
    // Separate processes keep process-start geometry out of parallel tests.
    const CHILD: &str = "CHOPINFER_TEST_WIDE_ROWS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for value in ["0", "1"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "fake_host_tests::speculation::c16_c32_deeper_drafts_use_opt_in_wide_graphs",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("CHOPINFER_SPECULATIVE_WIDE_ROWS", value)
                .status()
                .unwrap();
            assert!(status.success(), "wide rows {value}");
        }
        return;
    }
    let wide = std::env::var("CHOPINFER_SPECULATIVE_WIDE_ROWS").as_deref() == Ok("1");
    // Short histories need one page each. The general fixture helper's
    // 64-page stride runs beyond this fake host's pool at C32.
    let compact = |(item, append, slot): (BatchItemV1, BatchAppendV1, u64)| {
        let append = BatchAppendV1::new(
            append.previous_tokens(),
            append.additional_tokens(),
            append.new_tokens(),
            vec![slot],
            None,
        );
        (item, append, slot)
    };
    for served in [16_u32, 32] {
        let items = Items::new(served).unwrap();
        assert_eq!(
            crate::scratch::Served::new(items).rows,
            if wide { 256 } else { 32 }
        );
        for depth in [2, 3] {
            let rows = served * (depth + 1);
            assert_eq!(
                bundle_runtime::graphs::verify_bucket_within::<Qwen>(rows, items),
                wide.then_some(rows.next_power_of_two())
            );
            if !wide {
                continue;
            }
            let host = drafter_host(depth);
            host.served.set(Some(served));
            host.speculation.set((depth, rows));
            crate::pipeline::take_recorded();
            let mut instance = create(&host);
            host.model(instance.handle());
            execute(
                &host,
                &mut instance,
                &batch(
                    (0..served)
                        .map(|slot| {
                            compact(plain(
                                u64::from(slot) + 10,
                                u64::from(slot),
                                0,
                                &truths(0..5),
                            ))
                        })
                        .collect(),
                ),
            );
            instance.commit().unwrap();
            let mut lengths = vec![5_u64; served as usize];
            for iteration in 0..3 {
                let sink = execute(
                    &host,
                    &mut instance,
                    &batch(
                        (0..served)
                            .map(|slot| {
                                let length = lengths[slot as usize];
                                compact(drafted(
                                    u64::from(slot) + 10,
                                    u64::from(slot),
                                    length,
                                    truth(length),
                                    depth,
                                ))
                            })
                            .collect(),
                    ),
                );
                assert_eq!(sink.drafts.len(), served as usize);
                assert_eq!(
                    sink.views
                        .iter()
                        .map(|view| view.raw().row_count)
                        .sum::<u32>(),
                    rows
                );
                let accepted = (0..served)
                    .map(|slot| (u64::from(slot) + 10, 1 + (slot + iteration) % (depth + 1)))
                    .collect::<Vec<_>>();
                instance.commit_accepted(&kept(&accepted)).unwrap();
                for (slot, &(_, count)) in accepted.iter().enumerate() {
                    lengths[slot] += u64::from(count);
                    assert!(
                        committed_state(&host, &instance, slot as u32)
                            .iter()
                            .all(|history| *history == truths(0..lengths[slot]))
                    );
                }
            }
            assert!(
                host.journal
                    .borrow()
                    .iter()
                    .any(|line| line.starts_with("captured "))
            );
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            super::export_rows(&format!("mtp-wide-c{served}-k{depth}"), &[&host]);
        }
    }
}

/// High slot IDs, bounded drafting, AR siblings, partial acceptance, and
/// retained rows surviving the wide eager catch-up path.
#[test]
fn wide_mtp_steps_keep_every_slots_committed_history() {
    fn wide_plain(
        request: u64,
        slot: u64,
        previous: u64,
        inputs: &[u32],
    ) -> (BatchItemV1, BatchAppendV1, u64) {
        let count = inputs.len() as u64;
        (
            BatchItemV1::new(
                request,
                previous > 0 && count == 1,
                count as u32,
                true,
                inputs.to_vec(),
            ),
            BatchAppendV1::new(previous, count, previous + count, vec![slot], None),
            slot,
        )
    }
    fn wide_drafted(
        request: u64,
        slot: u64,
        previous: u64,
        bonus: u32,
        drafts: u32,
    ) -> (BatchItemV1, BatchAppendV1, u64) {
        let count = u64::from(drafts) + 1;
        (
            BatchItemV1::verify(request, count as u32, true, vec![bonus]),
            BatchAppendV1::new(previous, count, previous + count, vec![slot], None),
            slot,
        )
    }
    for served in [33_u32, 64, 128, 256] {
        let host = drafter_host(3);
        host.served.set(Some(served));
        host.speculation.set((3, 256));
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_served(Shape::TP4, host.layout(), &[], served),
            )
            .unwrap();
        host.model(instance.handle());
        execute(
            &host,
            &mut instance,
            &batch(
                (0..served)
                    .map(|slot| wide_plain(u64::from(slot) + 10, u64::from(slot), 0, &truths(0..3)))
                    .collect(),
            ),
        );
        instance.commit().unwrap();
        let mut lengths = vec![3_u64; served as usize];
        for iteration in 0..10 {
            // Leave room for the AR siblings inside the 256-row limit.
            let drafting = 32.min((256 - served) / 3);
            let mut keep = Vec::new();
            let step = batch(
                (0..served)
                    .map(|slot| {
                        let request = u64::from(slot) + 10;
                        let length = lengths[slot as usize];
                        if iteration < 2 && slot >= served - drafting {
                            let rows = 1 + slot % 4;
                            keep.push((request, rows));
                            wide_drafted(request, u64::from(slot), length, truth(length), 3)
                        } else {
                            wide_plain(request, u64::from(slot), length, &[truth(length)])
                        }
                    })
                    .collect(),
            );
            execute(&host, &mut instance, &step);
            if keep.is_empty() {
                instance.commit().unwrap();
            } else {
                instance.commit_accepted(&kept(&keep)).unwrap();
            }
            for slot in 0..served {
                let rows = keep
                    .iter()
                    .find(|&&(id, _)| id == u64::from(slot) + 10)
                    .map_or(1, |&(_, n)| n);
                lengths[slot as usize] += u64::from(rows);
                assert!(
                    committed_state(&host, &instance, slot)
                        .iter()
                        .all(|history| *history == truths(0..lengths[slot as usize])),
                    "{served} {slot} {iteration}"
                );
            }
        }
        instance.destroy().unwrap();
    }
}

/// The host's in-place failure path: a request whose sampling fails commits
/// one placeholder row with its step and then retires, while its drafting
/// and AR siblings keep their committed state and continue, and its slot
/// starts a fresh request.
#[test]
fn a_failed_request_retires_after_its_step_commits_without_touching_siblings() {
    let host = drafter_host(3);
    let mut instance = create_drafter(&host);
    execute(
        &host,
        &mut instance,
        &batch(vec![
            plain(10, 0, 0, &truths(0..3)),
            plain(11, 1, 0, &truths(0..3)),
            plain(12, 2, 0, &truths(0..3)),
        ]),
    );
    instance.commit().unwrap();
    execute(
        &host,
        &mut instance,
        &batch(vec![
            drafted(10, 0, 3, truth(3), 3),
            plain(11, 1, 3, &[truth(3)]),
            drafted(12, 2, 3, truth(3), 3),
        ]),
    );
    // Request 10's sampling failed: it commits its placeholder row.
    instance
        .commit_accepted(&kept(&[(10, 1), (12, 3)]))
        .unwrap();
    instance.retire_request(10).unwrap();
    for (slot, length) in [(1, 4), (2, 6)] {
        assert!(
            committed_state(&host, &instance, slot)
                .iter()
                .all(|history| *history == truths(0..length)),
            "{slot}"
        );
    }
    execute(
        &host,
        &mut instance,
        &batch(vec![
            plain(20, 0, 0, &truths(0..2)),
            plain(11, 1, 4, &[truth(4)]),
            plain(12, 2, 6, &[truth(6)]),
        ]),
    );
    instance.commit().unwrap();
    for (slot, length) in [(0, 2), (1, 5), (2, 7)] {
        assert!(
            committed_state(&host, &instance, slot)
                .iter()
                .all(|history| *history == truths(0..length)),
            "{slot}"
        );
    }
    instance.destroy().unwrap();
}

/// CONC-QWEN-STATE lifecycle above 32 served sequences, with fewer live
/// requests than slots so the highest slots draft. Requests have distinct
/// histories; drafting membership, kept rows and item order change every
/// step; a retired slot starts a new request with another history in a
/// mixed step; and a request is captured, retired and restored into
/// another high slot, where it drafts again. Every step drafts, and every
/// live slot's committed linear and PLE state is exactly its request's
/// history.
#[test]
fn wide_mtp_lifecycle_keeps_each_slots_own_history() {
    /// A token of `request`'s own history at `position`.
    fn token(request: u64, position: u64) -> u32 {
        ((request * 7919 + position * 104_729) % 200_000 + 11) as u32
    }
    struct Live {
        request: u64,
        slot: u64,
        /// Serving pages: independent of the state slot, as a host's are.
        group: u64,
        history: Vec<u32>,
    }
    fn item(live: &Live, drafts: u32) -> (BatchItemV1, BatchAppendV1, u64) {
        let length = live.history.len() as u64;
        let bonus = token(live.request, length);
        let mut item = if drafts > 0 {
            drafted(live.request, live.group, length, bonus, drafts)
        } else {
            plain(live.request, live.group, length, &[bonus])
        };
        item.2 = live.slot;
        item
    }
    fn prefill(request: u64, slot: u64, group: u64, length: u64) -> Live {
        Live {
            request,
            slot,
            group,
            history: (0..length)
                .map(|position| token(request, position))
                .collect(),
        }
    }
    for served in [33_u64, 64, 128, 256] {
        let host = drafter_host(3);
        host.served.set(Some(served as u32));
        host.backing.set(true);
        let mut instance = create(&host);
        host.model(instance.handle());
        let check = |instance: &BundleInstanceV1, live: &[Live]| {
            for entry in live {
                assert!(
                    committed_state(&host, instance, entry.slot as u32)
                        .iter()
                        .all(|history| *history == entry.history),
                    "served {served}: request {} in slot {}",
                    entry.request,
                    entry.slot
                );
            }
        };
        // Twelve requests: the eight highest slots, the lowest, and three
        // in between, with prompts of 9 to 20 tokens.
        let slots =
            (0..8)
                .map(|index| served - 1 - index)
                .chain([0, 5, served / 2, served / 2 + 1]);
        let mut live: Vec<Live> = slots
            .enumerate()
            .map(|(index, slot)| prefill(100 + index as u64, slot, index as u64, 9 + index as u64))
            .collect();
        let step = |instance: &mut BundleInstanceV1, items| execute(&host, instance, &batch(items));
        step(
            &mut instance,
            live.iter()
                .map(|entry| {
                    let (item, append, _) = plain(entry.request, entry.group, 0, &entry.history);
                    (item, append, entry.slot)
                })
                .collect(),
        );
        instance.commit().unwrap();
        check(&instance, &live);
        let mut snapshot = None;
        for iteration in 0..8_u64 {
            // Membership rotates: a third of the requests draft each step.
            let drafting = |index: usize| (index as u64 + iteration) % 3 == 0;
            let mut order: Vec<usize> = (0..live.len()).collect();
            if iteration % 2 == 1 {
                order.reverse();
            }
            let mut items: Vec<_> = order
                .iter()
                .map(|&index| item(&live[index], if drafting(index) { 3 } else { 0 }))
                .collect();
            // Iteration 3 also starts request 200, with its own history, in
            // the slot request 100 left.
            let fresh = (iteration == 3).then(|| prefill(200, served - 1, 12, 17));
            if let Some(fresh) = &fresh {
                // A step's prefill items come before its decode items.
                let (item, append, _) = plain(fresh.request, fresh.group, 0, &fresh.history);
                items.insert(0, (item, append, fresh.slot));
            }
            let sink = step(&mut instance, items);
            let drafts: BTreeMap<u64, Vec<u32>> = sink.draft_tokens().into_iter().collect();
            let mut keep = Vec::new();
            for (index, entry) in live.iter_mut().enumerate() {
                let length = entry.history.len() as u64;
                entry.history.push(token(entry.request, length));
                if drafting(index) {
                    let proposed = &drafts[&entry.request];
                    assert_eq!(proposed.len(), 3);
                    // Partial rejection: keep 1 to 4 rows.
                    let rows = 1 + (index as u64 + iteration / 3) % 4;
                    entry
                        .history
                        .extend_from_slice(&proposed[..rows as usize - 1]);
                    keep.push((entry.request, rows as u32));
                }
            }
            assert_eq!(drafts.len(), keep.len(), "served {served}");
            assert!(!keep.is_empty());
            // Kept rows follow the verify items' order in the step.
            keep.sort_by_key(|&(request, _)| {
                order
                    .iter()
                    .position(|&index| live[index].request == request)
            });
            instance.commit_accepted(&kept(&keep)).unwrap();
            if let Some(fresh) = fresh {
                live.push(fresh);
            }
            check(&instance, &live);
            if iteration == 2 {
                // Request 100 finishes: its slot is free for request 200.
                let done = live.remove(0);
                instance.retire_request(done.request).unwrap();
            }
            if iteration == 4 {
                // Preempt request 101 (the second highest slot): capture its
                // committed state, then retire it.
                let index = live.iter().position(|entry| entry.request == 101).unwrap();
                let entry = live.remove(index);
                // The last pages of the pool, clear of every request's.
                let count = instance.describe_prefix_state().unwrap();
                let pages = super::snapshot_pages(&host, super::KV_PAGES - count);
                instance
                    .capture_prefix_state(entry.request, entry.history.len() as u64, &pages)
                    .unwrap();
                instance.retire_request(entry.request).unwrap();
                snapshot = Some((entry, pages));
            }
            if iteration == 5 {
                // It resumes in a free high slot and drafts on the next steps.
                let (mut entry, pages) = snapshot.take().unwrap();
                let slot = served - 9;
                instance
                    .restore_prefix_state_in_slot(entry.request, &entry.history, &pages, slot)
                    .unwrap();
                entry.slot = slot;
                check(&instance, std::slice::from_ref(&entry));
                live.push(entry);
            }
        }
        assert!(live.iter().any(|entry| entry.request == 101));
        assert!(live.iter().any(|entry| entry.request == 200));
        for entry in &live {
            instance.retire_request(entry.request).unwrap();
        }
        instance.destroy().unwrap();
    }
}

/// A prompt-ending prefill of 150 tokens that keeps its state at its anchor
/// (128, the last multiple of 64) captures there after its one step, as a
/// chunk ending at 128 would: the linear entries, PLE entry, rings and the
/// drafter's frontier row 127 of the anchor item's spares. A request
/// restored from it (into the source's slot, which the drafter-row check
/// follows) holds the state of those 128 tokens, completes the
/// drafter's QSA block from the kept ring and drafts from the kept X, with
/// or without the drafter (the target alone keeps no X), also on an MTP
/// package serving without speculation (a ring with the drafter's layer
/// that no drafter writes), at every topology.
/// Only the committed length and the kept anchor capture; the next step's
/// begin drops the anchor.
#[test]
fn a_prefill_keeps_its_anchor_state_for_a_capture_inside_it() {
    for shape in Shape::ALL {
        for (drafter, mtp_layout) in [(true, true), (false, false), (false, true)] {
            let host = drafter_host_at(3, shape);
            host.drafter.set(mtp_layout);
            if !drafter {
                host.speculation.set((0, 0));
            }
            let mut instance = if drafter {
                create_drafter(&host)
            } else {
                let instance = create(&host);
                host.model(instance.handle());
                instance
            };
            host.backing.set(true);
            host.sim.borrow_mut().track_qsa = true;
            let pages = instance.describe_prefix_state().unwrap();
            let snapshot = |first: u64| (first..first + pages).collect::<Vec<_>>();
            let prompt = truths(0..150);
            let (item, append, slot) = plain(10, 1, 0, &prompt);
            execute(
                &host,
                &mut instance,
                &batch(vec![(item.with_anchor(true), append, slot)]),
            );
            instance.commit().unwrap();
            assert_eq!(
                instance.capture_prefix_state(10, 64, &snapshot(700)),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
                "{shape:?}: only the kept anchor captures inside the step"
            );
            instance
                .capture_prefix_state(10, 128, &snapshot(300))
                .unwrap_or_else(|error| panic!("{shape:?} {drafter}: {error:?}"));
            instance
                .capture_prefix_state(10, 150, &snapshot(500))
                .unwrap();
            instance.retire_request(10).unwrap();
            for (request, length, first) in [(11, 128, 300), (12, 150, 500)] {
                instance
                    .restore_prefix_state_in_slot(request, &truths(0..length), &snapshot(first), 1)
                    .unwrap();
                let state = committed_state(&host, &instance, 1);
                assert!(
                    state.iter().all(|history| *history == truths(0..length)),
                    "{:?} {drafter}: request {request}: {:?}",
                    shape.ranks,
                    state.iter().map(Vec::len).collect::<Vec<_>>()
                );
                let item = if drafter {
                    drafted(request, 1, length, truth(length), 3)
                } else {
                    plain(request, 1, length, &truths(length..length + 5))
                };
                let sink = execute(&host, &mut instance, &batch(vec![item]));
                if drafter {
                    assert_eq!(
                        sink.draft_tokens(),
                        [(request, truths(length + 1..length + 4))]
                    );
                    instance.commit_accepted(&kept(&[(request, 4)])).unwrap();
                } else {
                    instance.commit().unwrap();
                }
                assert!(
                    committed_state(&host, &instance, 1)
                        .iter()
                        .all(|history| *history == truths(0..length + 4 + u64::from(!drafter))),
                    "{shape:?} {drafter}: request {request} continues"
                );
                instance.retire_request(request).unwrap();
            }
            if drafter {
                let sequences = BTreeMap::from([(1, truths(0..160))]);
                assert_drafter_rows(&host, &sequences);
            }
            // A later step drops the anchor its predecessor kept.
            let (item, append, slot) = plain(13, 1, 0, &prompt);
            execute(
                &host,
                &mut instance,
                &batch(vec![(item.with_anchor(true), append, slot)]),
            );
            instance.commit().unwrap();
            execute(
                &host,
                &mut instance,
                &batch(vec![plain(13, 1, 150, &[truth(150)])]),
            );
            instance.commit().unwrap();
            assert_eq!(
                instance.capture_prefix_state(13, 128, &snapshot(700)),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
            );
            instance.retire_request(13).unwrap();
            instance.destroy().unwrap();
        }
    }
}

// Raw callbacks isolate read-only ledger validation. The supported client
// poisons on any terminal callback error, including a validation refusal.
#[test]
fn raw_ledger_refusal_preserves_state() {
    for shape in [Shape::TP2, Shape::TP4] {
        let host = drafter_host_at(3, shape);
        let mut instance = create_drafter(&host);
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 1, 0, &truths(0..5))]),
        );
        instance.commit().unwrap();
        execute(
            &host,
            &mut instance,
            &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
        );
        crate::instance::set_test_tail_window(instance.handle(), Some(1));
        let launches = host.launches.borrow().clone();
        assert_eq!(
            bundle_runtime::entry::commit_accepted::<crate::instance::Instance>(
                instance.handle(),
                &kept(&[(10, 2)])
            ),
            BundleStatusV1::InvalidArgument
        );
        assert_eq!(*host.launches.borrow(), launches);
        instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
        for rank in 0..shape.ranks() as usize {
            for history in committed_state_on_rank(&host, &instance, 1, rank) {
                assert_eq!(history, truths(0..9));
            }
        }
        execute(
            &host,
            &mut instance,
            &batch(vec![drafted(10, 1, 9, truth(9), 3)]),
        );
        instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
        assert_drafter_rows(&host, &BTreeMap::from([(1, truths(0..16))]));
        instance.retire_request(10).unwrap();
        instance.destroy().unwrap();
    }
}

/// Retained-X copies follow the CPU drafter commit. Their failure leaves
/// unknown native state, so only terminal poisoning is promised.
#[test]
fn late_commit_failure_poisons_and_blocks_further_work() {
    let pending = || {
        let host = drafter_host_at(3, Shape::TP4);
        let mut instance = create_drafter(&host);
        execute(
            &host,
            &mut instance,
            &batch(vec![plain(10, 1, 0, &truths(0..5))]),
        );
        instance.commit().unwrap();
        execute(
            &host,
            &mut instance,
            &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
        );
        (host, instance)
    };
    let (control, mut committed) = pending();
    let before = control.launches.borrow().len();
    committed.commit_accepted(&kept(&[(10, 2)])).unwrap();
    let launches = control.launches.borrow()[before..].to_vec();
    let symbol = model_qwen38_contract::sm120::owned::COPY_ENTRIES;
    assert_eq!(launches.last().unwrap().1, symbol);
    let copies = launches
        .iter()
        .filter(|(_, name, _)| name == symbol)
        .count();
    // The last COPY_ENTRIES copies retained X on the final rank, after
    // state.commit. Checkpoint ownership stays unchanged until all copies succeed.
    assert_eq!(copies, Shape::TP4.ranks() as usize);
    committed.destroy().unwrap();

    let (host, mut instance) = pending();
    let handle = instance.handle();
    let before_linear = crate::instance::committed_entry(handle, 1).unwrap();
    let before_ple = crate::instance::ple_entry(handle, 1);
    let before_state = committed_state(&host, &instance, 1);
    *host.fail_symbol_launch.borrow_mut() = Some((symbol.into(), (copies - 1) as u32));
    assert_eq!(
        instance.commit_accepted(&kept(&[(10, 2)])),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert!(
        host.fail_symbol_launch.borrow().is_none(),
        "retained-row copy reached the fault"
    );
    assert!(instance.is_poisoned());
    assert_eq!(
        crate::instance::committed_entry(handle, 1).unwrap(),
        before_linear
    );
    assert_eq!(crate::instance::ple_entry(handle, 1), before_ple);
    assert_eq!(committed_state(&host, &instance, 1), before_state);
    let journal = host.journal.borrow().clone();
    let released = host.released.get();
    let request = batch(vec![drafted(10, 1, 7, truth(7), 3)])
        .encode()
        .unwrap();
    assert_eq!(
        instance.execute(&request, &mut SpeculativeSink::default()),
        Err(BundleAbiError::Poisoned)
    );
    assert_eq!(instance.commit(), Err(BundleAbiError::Poisoned));
    assert_eq!(
        instance.commit_accepted(&kept(&[(10, 2)])),
        Err(BundleAbiError::Poisoned)
    );
    assert_eq!(instance.cancel(), Err(BundleAbiError::Poisoned));
    assert_eq!(instance.retire_request(10), Err(BundleAbiError::Poisoned));
    assert_eq!(
        instance.capture_prefix_state(10, 5, &[]),
        Err(BundleAbiError::Poisoned)
    );
    assert_eq!(
        instance.restore_prefix_state(10, &truths(0..5), &[]),
        Err(BundleAbiError::Poisoned)
    );
    assert_eq!(instance.destroy(), Err(BundleAbiError::Poisoned));
    assert_eq!(*host.journal.borrow(), journal);
    assert_eq!(host.released.get(), released);
}

#[test]
fn indexed_commits_cancel_capture_restore_and_reuse_every_accept_count() {
    for rows in 2..=MAX_VERIFY_ROWS {
        for accepted in 0..=rows {
            let host = Rc::new(FakeHost::at(Shape::TP2));
            host.ngram_host.set(ngram_in_host_memory(Shape::TP2));
            host.speculation.set((rows - 1, MAX_VERIFY_TOKENS));
            host.backing.set(true);
            let mut instance = create_drafter(&host);
            let handle = instance.handle();
            // Logical state slots and KV page IDs are independent. The
            // fixture's page pool is smaller than slot * 64 at this slot.
            let plain_at = |request, previous, inputs: &[u32]| {
                let (item, _, slot) = plain(request, 31, previous, inputs);
                let count = inputs.len() as u64;
                (
                    item,
                    BatchAppendV1::new(previous, count, previous + count, vec![0], None),
                    slot,
                )
            };
            execute(
                &host,
                &mut instance,
                &batch(vec![plain_at(10, 0, &truths(0..5))]),
            );
            instance.commit().unwrap();
            let original = crate::instance::committed_entry(handle, 31).unwrap();
            let original_ple = crate::instance::ple_entry(handle, 31);
            let request = |previous| {
                BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    vec![BatchItemV1::verify(
                        10,
                        rows,
                        false,
                        truths(previous..previous + u64::from(rows)),
                    )],
                    vec![BatchAppendV1::new(
                        previous,
                        u64::from(rows),
                        previous + u64::from(rows),
                        vec![0],
                        None,
                    )],
                )
                .with_state_slots(vec![31])
            };
            execute(&host, &mut instance, &request(5));
            if accepted == 0 {
                instance.cancel().unwrap();
                let [conv, delta, ple, _, _] = crate::instance::state_views(handle, 0);
                let sim = host.sim.borrow();
                for (span, bytes) in [
                    (conv, host.shape().conv_entry_bytes),
                    (delta, host.shape().delta_entry_bytes),
                ] {
                    for layer in 0..LINEAR_LAYERS {
                        let plane = DeviceBufferSpanV1::new(
                            span.buffer,
                            span.offset + layer * original.1 * bytes,
                            original.1 * bytes,
                        );
                        assert_eq!(
                            sim.history_at(plane, i64::from(original.0), bytes),
                            truths(0..5)
                        );
                    }
                }
                assert_eq!(
                    sim.history_at(ple, i64::from(original_ple), PLE_ENTRY_BYTES),
                    truths(0..5)
                );
                drop(sim);
                instance.retire_request(10).unwrap();
                execute(
                    &host,
                    &mut instance,
                    &batch(vec![plain_at(11, 0, &truths(0..3))]),
                );
                instance.commit().unwrap();
                assert!(
                    committed_state(&host, &instance, 31)
                        .iter()
                        .all(|state| *state == truths(0..3))
                );
            } else {
                let begin = host.launches.borrow().len();
                instance.commit_accepted(&kept(&[(10, accepted)])).unwrap();
                assert!(
                    host.launches.borrow()[begin..].is_empty(),
                    "state-only commit launched a copy"
                );
                assert_ne!(
                    crate::instance::committed_entry(handle, 31).unwrap().0,
                    original.0
                );
                assert_ne!(crate::instance::ple_entry(handle, 31), original_ple);
                let count = 5 + u64::from(accepted);
                assert!(
                    committed_state(&host, &instance, 31)
                        .iter()
                        .all(|state| *state == truths(0..count))
                );
                let pages = snapshot_pages(&host, 32);
                instance.capture_prefix_state(10, count, &pages).unwrap();
                // Replaying the verify graph writes only the recycled workspace.
                let before_cancel = committed_state(&host, &instance, 31);
                let ple_before_cancel = crate::instance::ple_entry(handle, 31);
                execute(&host, &mut instance, &request(count));
                instance.cancel().unwrap();
                assert_eq!(committed_state(&host, &instance, 31), before_cancel);
                assert_eq!(crate::instance::ple_entry(handle, 31), ple_before_cancel);
                instance.retire_request(10).unwrap();
                instance
                    .restore_prefix_state_in_slot(11, &truths(0..count), &pages, 31)
                    .unwrap();
                assert!(
                    committed_state(&host, &instance, 31)
                        .iter()
                        .all(|state| *state == truths(0..count))
                );
                execute(
                    &host,
                    &mut instance,
                    &batch(vec![plain_at(11, count, &[truth(count)])]),
                );
                instance.commit().unwrap();
                assert!(
                    committed_state(&host, &instance, 31)
                        .iter()
                        .all(|state| *state == truths(0..count + 1))
                );
            }
            instance.retire_request(11).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.released.get(), host.acquired.get());
        }
    }
}
