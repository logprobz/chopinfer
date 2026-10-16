//! State-sensitive CPU rows for snapshot continuation. This models byte
//! movement and slot selection, not transformer arithmetic (GPU M5).
use super::*;
use crate::prefix::Drafter;
use crate::topology::Tp;

const PREFILL_END: u64 = 1022;
type Captured = (String, Vec<KernelArgumentV1>);

pub(super) struct Probe {
    /// The instance's ring: its slot count places every slot's pieces.
    ring: crate::ring::RingGeometry,
    words: BTreeMap<(u64, u64), u64>,
    views: BTreeMap<u32, Vec<KernelArgumentV1>>,
    capturing: BTreeMap<u32, Vec<Captured>>,
    graphs: BTreeMap<u32, Vec<Captured>>,
    gathered: Vec<(u32, u64, u64)>,
    required_until: u64,
    rows: Vec<Vec<u64>>,
    /// The slot of each row.
    row_slots: Vec<u32>,
}

fn span(args: &[KernelArgumentV1], index: usize) -> DeviceBufferSpanV1 {
    match args[index] {
        KernelArgumentV1::DeviceBuffer(span) => span,
        _ => panic!("buffer argument {index}"),
    }
}
fn scalar(args: &[KernelArgumentV1], index: usize) -> u64 {
    match args[index] {
        KernelArgumentV1::ScalarU32(n) => n.into(),
        KernelArgumentV1::ScalarU64(n) => n,
        _ => panic!("scalar argument {index}"),
    }
}
fn uploaded(host: &Package, span: DeviceBufferSpanV1) -> Vec<u8> {
    let uploads = host.latest_uploads.borrow();
    let ((_, offset), bytes) = uploads
        .range((span.buffer.raw, 0)..=(span.buffer.raw, span.offset))
        .next_back()
        .unwrap();
    let start = (span.offset - offset) as usize;
    bytes[start..start + span.length as usize].to_vec()
}
fn indexed_word(host: &Package, span: DeviceBufferSpanV1, index: u64) -> u32 {
    word(
        host,
        DeviceBufferSpanV1::new(span.buffer, span.offset + 4 * index, 4),
    )
}

fn provenance(stream: u32, position: u64) -> u64 {
    (u64::from(stream) << 32) | (position + 1)
}

fn word(host: &Package, span: DeviceBufferSpanV1) -> u32 {
    u32::from_le_bytes(
        uploaded(host, DeviceBufferSpanV1::new(span.buffer, span.offset, 4))
            .try_into()
            .unwrap(),
    )
}

impl Probe {
    /// A probe of a TP4 instance whose ring holds `slots` slots.
    fn new(slots: u32) -> Self {
        Self {
            ring: crate::ring::RingGeometry::new(Tp::TP4, slots),
            words: BTreeMap::new(),
            views: BTreeMap::new(),
            capturing: BTreeMap::new(),
            graphs: BTreeMap::new(),
            gathered: Vec::new(),
            required_until: 0,
            rows: Vec::new(),
            row_slots: Vec::new(),
        }
    }

    /// The pieces the copy kernel moves for `slot`, in snapshot order
    /// (`prefix_copy` in `cuda/mimo_ring.cuh`): every layer's K then V slot
    /// entry, then the drafter piece the launch binds.
    fn pieces(
        ring: crate::ring::RingGeometry,
        args: &[KernelArgumentV1],
        slot: u32,
    ) -> Vec<DeviceBufferSpanV1> {
        let state = span(args, 0);
        let slot = u64::from(slot);
        let tokens = crate::contract::RING_TOKENS;
        let (k_slot, v_slot) = (tokens * ring.k_entry, tokens * ring.v_entry);
        let mut pieces = (0..crate::contract::SWA_LAYERS)
            .flat_map(|layer| {
                [
                    (layer * ring.layer + slot * k_slot, k_slot),
                    (layer * ring.layer + ring.k_layer + slot * v_slot, v_slot),
                ]
            })
            .map(|(offset, length)| {
                DeviceBufferSpanV1::new(state.buffer, state.offset + offset, length)
            })
            .collect::<Vec<_>>();
        if scalar(args, 2) > 0 {
            pieces.push(span(args, 1));
        }
        pieces
    }

    pub fn copy(&mut self, host: &Package, stream: u32, args: &[KernelArgumentV1]) {
        let pages = uploaded(host, span(args, 4))
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let page_bytes = scalar(args, 6);
        let pool = span(args, 3);
        let to_pages = scalar(args, 8) != 0;
        self.views.insert(stream, args.to_vec());
        let mut at = 0;
        for piece in Self::pieces(self.ring, args, scalar(args, 7) as u32) {
            let mut done = 0;
            while done < piece.length {
                let count = (page_bytes - at % page_bytes).min(piece.length - done);
                let paged = DeviceBufferSpanV1::new(
                    pool.buffer,
                    pool.offset + pages[(at / page_bytes) as usize] * page_bytes + at % page_bytes,
                    count,
                );
                let part = DeviceBufferSpanV1::new(piece.buffer, piece.offset + done, count);
                let (from, to) = if to_pages {
                    (part, paged)
                } else {
                    (paged, part)
                };
                let values = self
                    .words
                    .range((from.buffer.raw, from.offset)..(from.buffer.raw, from.offset + count))
                    .map(|(&(_, offset), &value)| (offset - from.offset, value))
                    .collect::<Vec<_>>();
                self.words.retain(|&(buffer, offset), _| {
                    buffer != to.buffer.raw || !(to.offset..to.offset + count).contains(&offset)
                });
                for (offset, value) in values {
                    self.words
                        .insert((to.buffer.raw, to.offset + offset), value);
                }
                at += count;
                done += count;
            }
        }
    }

    fn seed(&mut self) {
        for (&rank, args) in &self.views {
            // The captured slot's pieces.
            let slot = scalar(args, 7) as u32;
            for (index, piece) in Self::pieces(self.ring, args, slot).into_iter().enumerate() {
                // Layout sentinels cover the ring and DFlash pieces. MTP rows
                // must come from the retained-row kernel, never a post-commit fill.
                if scalar(args, 2) == crate::prefix::MTP_SLOT_ROWS * crate::prefix::MTP_HIDDEN_BYTES
                    && index == crate::contract::SWA_LAYERS as usize * 2
                {
                    continue;
                }
                // Nonzero sentinels through every piece and at both ends.
                for offset in (0..piece.length).step_by(4096).chain([piece.length - 8]) {
                    self.words.insert(
                        (piece.buffer.raw, piece.offset + offset),
                        (u64::from(rank) << 48) ^ ((index as u64 + 1) << 32) ^ offset ^ 0x1234,
                    );
                }
            }
        }
    }

    fn row(&mut self, host: &Package, stream: u32, args: &[KernelArgumentV1]) {
        let Some(views) = self.views.get(&stream) else {
            return;
        };
        assert_eq!(word(host, span(args, 4)), 1);
        let slot = word(host, span(args, 1));
        let mut row = vec![u64::from(word(host, span(args, 3)))];
        for piece in Self::pieces(self.ring, views, slot) {
            row.extend(
                self.words
                    .range(
                        (piece.buffer.raw, piece.offset)
                            ..(piece.buffer.raw, piece.offset + piece.length),
                    )
                    .flat_map(|(&(_, offset), &value)| [offset - piece.offset, value]),
            );
        }
        assert!(row.len() > 100, "the row must depend on nonempty state");
        self.rows.push(row);
        self.row_slots.push(slot);
    }

    fn execute(&mut self, host: &Package, stream: u32, name: &str, args: &[KernelArgumentV1]) {
        match name {
            // The offload probes run at TP4.
            _ if name
                == crate::sm120::ring_entries::ring_backup_symbol(Tp::TP4, self.ring.slots) =>
            {
                self.row(host, stream, args)
            }
            _ if name == crate::retain::retain_rows_symbol(Tp::TP4)
                && scalar(args, 10) == crate::prefix::MTP_SLOT_ROWS =>
            {
                // One marker per actual hidden row, at the same circular cell
                // the kernel writes. The position comes from the launch's row
                // metadata, not the test's expected committed boundary.
                let history = span(args, 1);
                for row in 0..u64::from(word(host, span(args, 6))) {
                    let item = u64::from(indexed_word(host, span(args, 3), row));
                    let slot = indexed_word(host, span(args, 2), item);
                    let position = u64::from(indexed_word(host, span(args, 5), item)) + row
                        - u64::from(indexed_word(host, span(args, 4), item));
                    let offset =
                        crate::retain::cell(slot, position, scalar(args, 10)) * scalar(args, 9) * 2;
                    self.words.insert(
                        (history.buffer.raw, history.offset + offset),
                        provenance(stream, position),
                    );
                }
            }
            _ => {}
        }
    }

    pub fn hidden_copy(&mut self, stream: u32, source: DeviceBufferSpanV1) {
        let Some(args) = self.views.get(&stream) else {
            return;
        };
        let slot_bytes = crate::prefix::MTP_SLOT_ROWS * crate::prefix::MTP_HIDDEN_BYTES;
        if scalar(args, 2) != slot_bytes || source.buffer != span(args, 1).buffer {
            return;
        }
        let base = span(args, 1).offset - scalar(args, 7) * slot_bytes;
        let slot = (source.offset - base) / slot_bytes;
        // MTP catch-up copies committed hidden rows to its input with D2D.
        // Read the markers from that source span, before any target writes.
        for (_, &value) in self.words.range(
            (source.buffer.raw, source.offset)..(source.buffer.raw, source.offset + source.length),
        ) {
            let position = (value & u64::from(u32::MAX)) - 1;
            assert!(
                position < self.required_until,
                "catch-up read rejected row {position}"
            );
        }
        for position in PREFILL_END..self.required_until {
            let offset = base
                + crate::retain::cell(slot as u32, position, crate::prefix::MTP_SLOT_ROWS)
                    * crate::prefix::MTP_HIDDEN_BYTES;
            if !(source.offset..source.offset + source.length).contains(&offset) {
                continue;
            }
            let actual = self.words.get(&(source.buffer.raw, offset)).copied();
            assert_eq!(
                actual,
                Some(provenance(stream, position)),
                "MTP catch-up must read retained boundary row {position} in slot {slot}"
            );
            self.gathered.push((stream, position, actual.unwrap()));
        }
    }

    pub fn launch(&mut self, host: &Package, stream: u32, name: &str, args: &[KernelArgumentV1]) {
        if let Some(captures) = self.capturing.get_mut(&stream) {
            captures.push((name.into(), args.to_vec()));
        } else {
            self.execute(host, stream, name, args);
        }
    }
    pub fn begin(&mut self, stream: u32) {
        self.capturing.insert(stream, Vec::new());
    }
    pub fn end(&mut self, stream: u32, graph: u32) {
        self.graphs
            .insert(graph, self.capturing.remove(&stream).unwrap());
    }
    pub fn replay(&mut self, host: &Package, stream: u32, graph: u32) {
        for (name, args) in self.graphs[&graph].clone() {
            self.execute(host, stream, &name, &args);
        }
    }
}

/// Request 7 captures its snapshot in the second highest slot and request 8
/// continues it in the highest: slots 30 and 31 at 32 served sequences, 98
/// and 99 in a 128-slot ring at 100, and 254 and 255 at 256, whose DFlash
/// history offsets pass 2^31 bytes. Resident one-token prefills hold the
/// lower slots. MTP serves at most 32.
#[test]
fn decode_and_verify_snapshots_continue_in_another_slot_with_identical_cpu_rows() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    for (drafter, served) in [
        (Drafter::None, 32),
        (Drafter::Mtp, 32),
        (Drafter::Dflash, 32),
        (Drafter::None, 100),
        (Drafter::Dflash, 100),
        (Drafter::None, 256),
        (Drafter::Dflash, 256),
    ] {
        for verify in [false, true] {
            // FP8 pages change only the snapshot's page count, which the
            // slot count does not affect.
            for fp8 in [false, true]
                .into_iter()
                .filter(|&fp8| served == 32 || !fp8)
            {
                let host = Rc::new(FakeHost::default());
                host.served.set(served);
                host.fp8_kv.set(fp8);
                if drafter == Drafter::Dflash {
                    host.pool_pages.set(Some(256));
                }
                host.speculation.set(Some((7, 32)));
                host.dflash.set(drafter == Drafter::Dflash);
                let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
                    .unwrap();
                let mut instance = api
                    .create(
                        HostServicesTableV1::new(Rc::clone(&host)),
                        &configuration_serving(
                            Tp::TP4,
                            host.kv(),
                            drafter != Drafter::None,
                            served,
                        ),
                    )
                    .unwrap();
                // Resident one-token sequences hold slots 0 to served - 3.
                let residents = u64::from(served) - 2;
                let hold = BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    (0..residents)
                        .map(|request| BatchItemV1::new(1000 + request, false, 1, true, vec![5]))
                        .collect(),
                    (0..residents)
                        .map(|_| BatchAppendV1::new(0, 1, 1, vec![127], None))
                        .collect(),
                );
                instance
                    .execute(&hold.encode().unwrap(), &mut DraftSink::default())
                    .unwrap();
                instance.commit().unwrap();
                let step = |request, start: u64, tokens: Vec<u32>, rows: u32, verify: bool| {
                    let item = if verify {
                        BatchItemV1::verify(request, rows, drafter != Drafter::None, tokens)
                    } else {
                        BatchItemV1::new(request, start > 0, rows, true, tokens)
                    };
                    BatchRequestV1::new(
                        BatchWireVersionV1::V3,
                        vec![item],
                        vec![BatchAppendV1::new(
                            start,
                            u64::from(rows),
                            start + u64::from(rows),
                            (0..(start + u64::from(rows)).div_ceil(PAGE_TOKENS)).collect(),
                            None,
                        )],
                    )
                    .encode()
                    .unwrap()
                };
                *host.offload.borrow_mut() = Some(Probe::new(host.ring_slots()));
                let mut tokens = (0..PREFILL_END as u32).collect::<Vec<u32>>();
                instance
                    .execute(
                        &step(7, 0, tokens.clone(), 1022, false),
                        &mut DraftSink::default(),
                    )
                    .unwrap();
                instance.commit().unwrap();
                *host.readback_tokens.borrow_mut() = vec![1022, 1023, 1024, 1025];
                let input = if verify && drafter == Drafter::None {
                    vec![1022, 1023, 1024, 1025]
                } else {
                    vec![1022]
                };
                instance
                    .execute(
                        &step(7, 1022, input, if verify { 4 } else { 1 }, verify),
                        &mut DraftSink::default(),
                    )
                    .unwrap();
                if verify {
                    instance.commit_accepted(&[kept(7, 2)]).unwrap();
                    tokens.extend([1022, 1023]);
                } else {
                    instance.commit().unwrap();
                    tokens.push(1022);
                }
                let boundary = tokens.len() as u64;
                let pages = (30..30 + instance.describe_prefix_state().unwrap())
                    .rev()
                    .collect::<Vec<_>>();
                // Resolve the device views and fill the copy-layout sentinels.
                // MTP hidden rows are left exactly as execution wrote them.
                instance.capture_prefix_state(7, boundary, &pages).unwrap();
                host.offload.borrow_mut().as_mut().unwrap().seed();
                instance.capture_prefix_state(7, boundary, &pages).unwrap();
                assert!(
                    instance
                        .capture_prefix_state(7, boundary + 1, &pages)
                        .is_err()
                );
                instance.restore_prefix_state(8, &tokens, &pages).unwrap();
                // Request 7 still owns slot served - 2, so request 8 must
                // decode in slot served - 1.
                for request in [7, 8] {
                    instance
                        .execute(
                            &step(request, boundary, vec![42], 1, false),
                            &mut DraftSink::default(),
                        )
                        .unwrap();
                    instance.commit().unwrap();
                }
                let probe = host.offload.borrow();
                let rows = &probe.as_ref().unwrap().rows;
                assert_eq!(rows.len(), 2 * RANKS as usize);
                assert_eq!(
                    rows[..RANKS as usize],
                    rows[RANKS as usize..],
                    "{drafter:?} served={served} verify={verify} fp8={fp8}"
                );
                let slots = &probe.as_ref().unwrap().row_slots;
                assert!(
                    slots[..RANKS as usize]
                        .iter()
                        .all(|&slot| slot == served - 2)
                );
                assert!(
                    slots[RANKS as usize..]
                        .iter()
                        .all(|&slot| slot == served - 1)
                );
                drop(probe);
                if drafter == Drafter::Mtp {
                    // The very next drafted verify must consume the committed
                    // decode/verify hidden rows, including those restored into
                    // the other slot. Rejection left tentative rows beyond the
                    // boundary, which catch-up must not read.
                    host.offload.borrow_mut().as_mut().unwrap().required_until = boundary + 1;
                    let mut reads = Vec::new();
                    for request in [7, 8] {
                        host.offload.borrow_mut().as_mut().unwrap().gathered.clear();
                        instance
                            .execute(
                                &step(request, boundary + 1, vec![43], 4, true),
                                &mut DraftSink::default(),
                            )
                            .unwrap();
                        let probe = host.offload.borrow();
                        let gathered = &probe.as_ref().unwrap().gathered;
                        for stream in probe.as_ref().unwrap().views.keys() {
                            for position in PREFILL_END..=boundary {
                                assert!(
                                    gathered.contains(&(
                                        *stream,
                                        position,
                                        provenance(*stream, position)
                                    )),
                                    "catch-up omitted rank {stream} committed row {position}: {gathered:?}, verify={verify} request={request}"
                                );
                            }
                        }
                        reads.push(gathered.clone());
                        drop(probe);
                        instance.commit_accepted(&[kept(request, 1)]).unwrap();
                    }
                    // Rebuilding a cold drafter can split catch-up into a
                    // different set of launches. Compare the consumed rows.
                    for rows in &mut reads {
                        rows.sort_unstable();
                    }
                    assert_eq!(reads[0], reads[1]);
                }
                instance.destroy().unwrap();
            }
        }
    }
}
