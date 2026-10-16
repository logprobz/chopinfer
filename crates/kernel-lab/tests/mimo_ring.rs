//! The MiMo bundle's SWA ring backup copies exactly the entries a phase will
//! overwrite: the last min(q_len, 128) positions of each item, every layer,
//! K and V, from the working half into the shadow half. A verify step's
//! restore then copies its rejected positions back, leaving the ring as if
//! only the kept rows had been written.

use bundle_mimo26::topology::{Tp, ring_threads};
use kernel_lab::launch::simple_config;
use model_mimo26_contract::own_ops::ring::{
    Direction, RingBackup, RingLayout, RingPrefixCopy, RingRestore, ring_backup, ring_prefix_copy,
    ring_restore,
};
use runtime_cuda::{KernelArgument, LaunchConfig, Module};

const RING_TOKENS: u64 = 128;
const SLOTS: u64 = 32;
const LAYERS: u64 = 39;

/// A rank's SWA ring at one rank count and slot count, and the kernels that
/// serve it: the lab probe at TP4 and 32 slots, the bundle's own cubins
/// otherwise (other slot counts from its `ring_slots` cubin).
#[derive(Clone, Copy)]
struct Ring {
    tp: Tp,
    slots: u64,
    k_entry: u64,
    v_entry: u64,
    threads: u32,
}

impl Ring {
    fn new(tp: Tp) -> Self {
        Self::with_slots(tp, SLOTS)
    }
    fn with_slots(tp: Tp, slots: u64) -> Self {
        let heads = tp.kv_heads(false);
        Self {
            tp,
            slots,
            k_entry: heads * 192 * 2,
            v_entry: heads * 128 * 2,
            threads: ring_threads(tp),
        }
    }
    fn k_layer(self) -> u64 {
        self.slots * RING_TOKENS * self.k_entry
    }
    fn layer(self) -> u64 {
        self.k_layer() + self.slots * RING_TOKENS * self.v_entry
    }
    fn working(self) -> u64 {
        LAYERS * self.layer()
    }
    /// The ring as the contract states it.
    fn layout(self) -> RingLayout {
        RingLayout::of(self.tp, self.slots as u32)
    }
    fn module(self, session: &kernel_lab::Session) -> kernel_lab::LabResult<Module> {
        if self.tp == Tp::TP4 && self.slots == SLOTS {
            return session.module("probe_sm120");
        }
        let suffix = if self.slots == SLOTS { "" } else { "_slots" };
        let name = format!(
            "{}_ring{suffix}",
            bundle_mimo26::topology::kernel_tag(self.tp)
        );
        session.load_cubin(
            bundle_mimo26::embedded_cubin(&name).ok_or(format!("missing cubin {name}"))?,
        )
    }
    fn symbol(self, name: &str) -> String {
        if self.tp == Tp::TP4 && self.slots == SLOTS {
            format!("chop_probe_mimo_ring_{name}_v1")
        } else {
            bundle_mimo26::sm120::ring_entries::ring_symbol(
                self.tp,
                &format!("ring_{name}"),
                self.slots as u32,
            )
        }
    }
}

/// A device column of `values`, zero padded to at least 32 entries.
fn column_buffer(
    session: &kernel_lab::Session,
    values: &[i32],
) -> kernel_lab::LabResult<runtime_cuda::DeviceBuffer> {
    let entries = values.len().max(32);
    let buffer = session.allocate(4 * entries)?;
    let mut bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    bytes.resize(4 * entries, 0);
    buffer
        .copy_from_bytes(&bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// One phase's items as the bundle packs them: (slot, q_len, prefix_len).
fn run_case(
    session: &kernel_lab::Session,
    ring: Ring,
    items: &[(i32, i32, i32)],
    span: u32,
) -> kernel_lab::LabResult<()> {
    let (working, layer_bytes, k_layer) = (ring.working(), ring.layer(), ring.k_layer());
    let state_bytes = (2 * working) as usize;
    let state = session.allocate(state_bytes)?;
    // Working bytes carry their own offset; the shadow starts as 0xff.
    let mut host = vec![0xff_u8; state_bytes];
    for (index, word) in host[..working as usize].chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&((index as u32).wrapping_mul(2_654_435_761)).to_le_bytes());
    }
    state
        .copy_from_bytes(&host, &session.stream)
        .map_err(|e| e.to_string())?;
    let column = |values: &[i32]| column_buffer(session, values);
    let slot_ids = items.iter().map(|item| item.0).collect::<Vec<_>>();
    let q_lens = items.iter().map(|item| item.1).collect::<Vec<_>>();
    let prefix_lens = items.iter().map(|item| item.2).collect::<Vec<_>>();
    let slots = column(&slot_ids)?;
    let q_len = column(&q_lens)?;
    let prefix = column(&prefix_lens)?;
    let live = column(&[items.len() as i32])?;
    let module = ring.module(session)?;
    let function = module
        .function(&ring.symbol("backup"))
        .map_err(|e| e.to_string())?;
    // Planned for more items than are live, as a decode bucket is.
    let planned = items.len().next_power_of_two() as u32;
    function
        .launch(
            &session.stream,
            simple_config(
                (planned * span, LAYERS as u32, 1),
                (ring.threads, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?,
            &[
                KernelArgument::Buffer(&state),
                KernelArgument::Buffer(&slots),
                KernelArgument::Buffer(&q_len),
                KernelArgument::Buffer(&prefix),
                KernelArgument::Buffer(&live),
                KernelArgument::U32(span),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut device = vec![0_u8; state_bytes];
    state
        .copy_to_bytes(&mut device, &session.stream)
        .map_err(|e| e.to_string())?;

    // The cells the host bookkeeping saves and cancel restores.
    let mut expected = host.clone();
    ring_backup(
        &RingBackup {
            layout: ring.layout(),
            span,
        },
        &mut expected,
        &slot_ids,
        &q_lens,
        &prefix_lens,
        items.len() as u32,
        planned,
    )
    .map_err(|e| e.0.to_string())?;
    if device[..working as usize] != host[..working as usize] {
        return Err("the backup wrote the working half".into());
    }
    for layer in 0..LAYERS {
        for (plane, entry) in [(0, ring.k_entry), (k_layer, ring.v_entry)] {
            for cell in 0..ring.slots * RING_TOKENS {
                let at = (layer * layer_bytes + plane + cell * entry) as usize;
                let shadow = &device[working as usize + at..][..entry as usize];
                let saved = &expected[working as usize + at..][..entry as usize];
                if shadow != saved {
                    return Err(format!(
                        "layer {layer} plane {plane} cell {cell}: shadow differs (touched {})",
                        saved != &host[working as usize + at..][..entry as usize]
                    ));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn ring_backup_saves_exactly_the_touched_entries() {
    kernel_lab::gpu_test(1 << 30, |session| {
        for tp in Tp::SUPPORTED {
            let ring = Ring::new(tp);
            // Decode: one position per item, including ring wrap at 128.
            run_case(session, ring, &[(0, 1, 0), (5, 1, 127), (31, 1, 128)], 1)?;
            // Prefill: a chunk longer than the ring, a short one crossing
            // the wrap, and a first chunk.
            run_case(session, ring, &[(3, 130, 0), (7, 5, 126), (9, 64, 0)], 128)?;
        }
        Ok(())
    })
    .expect("MiMo ring backup");
}

/// Rings of other slot counts (MiMo above 32 sequences): the backup, restore and snapshot
/// entries of the `ring_slots` cubin address their wider (or narrower)
/// planes, including slots past 32 and batches of more than 32 items.
#[test]
fn ring_entries_of_other_slot_counts_address_their_planes() {
    // The 256-slot TP4 ring state alone takes 3.3 GB.
    kernel_lab::gpu_test(5 << 30, |session| {
        for (tp, slots) in [(Tp::TP4, 8), (Tp::TP4, 64), (Tp::TP4, 256), (Tp::TP2, 64)] {
            let ring = Ring::with_slots(tp, slots);
            let last = slots as i32 - 1;
            run_case(
                session,
                ring,
                &[(0, 1, 0), (last / 2, 1, 127), (last, 1, 128)],
                1,
            )?;
            run_case(session, ring, &[(last, 130, 0), (1, 5, 126)], 128)?;
            // A decode bucket of every slot.
            let every = (0..slots as i32)
                .map(|slot| (slot, 1, slot * 7))
                .collect::<Vec<_>>();
            run_case(session, ring, &every, 1)?;
            for kept in [1, 5, 8] {
                run_prefix_commit(
                    session,
                    ring,
                    &[
                        (0, 8, 124, kept),
                        (last, 8, 300, 9 - kept),
                        (last / 2, 1, 40, 1),
                    ],
                )?;
            }
            let page_bytes = bundle_mimo26::topology::ga_page_bytes(tp, false);
            for pair in [(last as u64, 0), (0, last as u64)] {
                run_prefix_copy(session, ring, page_bytes, 0, None, pair)?;
            }
        }
        Ok(())
    })
    .expect("MiMo ring entries of other slot counts");
}

/// Uploads per-item columns, zero padded to at least 32 entries each.
fn columns(
    session: &kernel_lab::Session,
    values: &[Vec<i32>],
) -> kernel_lab::LabResult<Vec<runtime_cuda::DeviceBuffer>> {
    values
        .iter()
        .map(|values| column_buffer(session, values))
        .collect()
}

/// Every (layer, plane, byte range) of one ring cell in the working half.
fn cell_ranges(ring: Ring, cell: u64) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    for layer in 0..LAYERS {
        for (plane, entry) in [(0, ring.k_entry), (ring.k_layer(), ring.v_entry)] {
            let at = (layer * ring.layer() + plane + cell * entry) as usize;
            ranges.push(at..at + entry as usize);
        }
    }
    ranges
}

/// One verify step per item: (slot, rows, prefix, kept). Backs up, writes
/// every row's cells, restores the rejected tail, and compares the working
/// half with a ring that only ever received the kept rows.
fn run_prefix_commit(
    session: &kernel_lab::Session,
    ring: Ring,
    items: &[(i32, i32, i32, i32)],
) -> kernel_lab::LabResult<()> {
    const SPAN: u32 = 8;
    let working = ring.working();
    let state_bytes = (2 * working) as usize;
    let state = session.allocate(state_bytes)?;
    let mut before = vec![0xff_u8; state_bytes];
    for (index, word) in before[..working as usize].chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&((index as u32).wrapping_mul(2_654_435_761)).to_le_bytes());
    }
    state
        .copy_from_bytes(&before, &session.stream)
        .map_err(|e| e.to_string())?;
    let module = ring.module(session)?;
    let planned = items.len().next_power_of_two() as u32;
    let config = || {
        simple_config(
            (planned * SPAN, LAYERS as u32, 1),
            (ring.threads, 1, 1),
            0,
            LaunchConfig::NO_CLUSTER,
            true,
        )
    };
    let backup_columns = [
        items.iter().map(|item| item.0).collect::<Vec<_>>(),
        items.iter().map(|item| item.1).collect(),
        items.iter().map(|item| item.2).collect(),
        vec![items.len() as i32],
    ];
    let backup = columns(session, &backup_columns)?;
    module
        .function(&ring.symbol("backup"))
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            config()?,
            &[
                KernelArgument::Buffer(&state),
                KernelArgument::Buffer(&backup[0]),
                KernelArgument::Buffer(&backup[1]),
                KernelArgument::Buffer(&backup[2]),
                KernelArgument::Buffer(&backup[3]),
                KernelArgument::U32(SPAN),
            ],
        )
        .map_err(|e| e.to_string())?;
    // The step writes a new value into every row's cell, as the ring write
    // does; the value depends on the absolute position.
    let mut written = vec![0_u8; state_bytes];
    state
        .copy_to_bytes(&mut written, &session.stream)
        .map_err(|e| e.to_string())?;
    // The oracles' state: the backup, the same writes, then the restore.
    let mut expected = before.clone();
    ring_backup(
        &RingBackup {
            layout: ring.layout(),
            span: SPAN,
        },
        &mut expected,
        &backup_columns[0],
        &backup_columns[1],
        &backup_columns[2],
        items.len() as u32,
        planned,
    )
    .map_err(|e| e.0.to_string())?;
    let mut kept_only = before[..working as usize].to_vec();
    for &(slot, rows, prefix, kept) in items {
        for row in 0..rows {
            let position = (prefix + row) as u64;
            let cell = slot as u64 * RING_TOKENS + position % RING_TOKENS;
            for range in cell_ranges(ring, cell) {
                let value = (position as u8).wrapping_mul(17).wrapping_add(1);
                written[range.clone()].fill(value);
                expected[range.clone()].fill(value);
                if row < kept {
                    kept_only[range].fill(value);
                }
            }
        }
    }
    state
        .copy_from_bytes(&written, &session.stream)
        .map_err(|e| e.to_string())?;
    let restore_columns = [
        items.iter().map(|item| item.0).collect::<Vec<_>>(),
        items.iter().map(|item| item.2 + item.3).collect(),
        items.iter().map(|item| item.1 - item.3).collect(),
        vec![items.len() as i32],
    ];
    let restore = columns(session, &restore_columns)?;
    ring_restore(
        &RingRestore {
            layout: ring.layout(),
            span: SPAN,
        },
        &mut expected,
        &restore_columns[0],
        &restore_columns[1],
        &restore_columns[2],
        items.len() as u32,
        planned,
    )
    .map_err(|e| e.0.to_string())?;
    // A ring that only ever received the kept rows.
    if expected[..working as usize] != kept_only[..] {
        return Err("the oracles' working half differs from the kept rows".into());
    }
    module
        .function(&ring.symbol("restore"))
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            config()?,
            &[
                KernelArgument::Buffer(&state),
                KernelArgument::Buffer(&restore[0]),
                KernelArgument::Buffer(&restore[1]),
                KernelArgument::Buffer(&restore[2]),
                KernelArgument::Buffer(&restore[3]),
                KernelArgument::U32(SPAN),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut device = vec![0_u8; state_bytes];
    state
        .copy_to_bytes(&mut device, &session.stream)
        .map_err(|e| e.to_string())?;
    if device[working as usize..] != written[working as usize..] {
        return Err("the restore wrote the shadow half".into());
    }
    if device[..working as usize] != expected[..working as usize] {
        let first = device
            .iter()
            .zip(&expected)
            .position(|(left, right)| left != right)
            .unwrap_or(0);
        return Err(format!("working half differs at byte {first}"));
    }
    Ok(())
}

#[test]
fn ring_restore_leaves_only_the_kept_rows_written() {
    kernel_lab::gpu_test(1 << 30, |session| {
        for tp in Tp::SUPPORTED {
            for kept in 1..=8 {
                // A step crossing the ring wrap, one far past it, and a
                // decode item that rejects nothing.
                run_prefix_commit(
                    session,
                    Ring::new(tp),
                    &[(0, 8, 124, kept), (6, 8, 300, 9 - kept), (31, 1, 40, 1)],
                )?;
            }
        }
        Ok(())
    })
    .expect("MiMo ring prefix commit");
}

fn patterned(bytes: usize, seed: u32) -> Vec<u8> {
    let mut host = vec![0_u8; bytes];
    for (index, word) in host.chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&((index as u32 ^ seed).wrapping_mul(2_654_435_761)).to_le_bytes());
    }
    host
}

/// Captures `from` into scattered pool pages, checks the pages hold its
/// pieces end to end and no other page or byte changed, then restores the
/// pages into slot `to` and checks only that slot's pieces changed, to
/// `from`'s bytes.
///
/// Every byte of every slot and page starts distinct, so the other slots
/// stand for live sequences and the other pages for other anchors and KV.
/// The kernel copies whole slots, so a ring or drafter window whose write
/// cursor wrapped is carried position for position.
fn run_prefix_copy(
    session: &kernel_lab::Session,
    ring: Ring,
    page_bytes: u64,
    drafter: u64,
    expected_pages: Option<u64>,
    (from, to): (u64, u64),
) -> kernel_lab::LabResult<()> {
    const POOL_PAGES: u64 = 128;
    let total = ring.layout().slot_bytes() + drafter;
    let page_count = total.div_ceil(page_bytes);
    let op = |direction| RingPrefixCopy {
        layout: ring.layout(),
        drafter_bytes: drafter,
        page_bytes,
        pages: page_count,
        direction,
    };
    if expected_pages.is_some_and(|expected| page_count != expected || total % page_bytes == 0) {
        return Err(format!(
            "{total} bytes take {page_count} pages of {page_bytes}, expected {expected_pages:?} with a partial last page"
        ));
    }
    // Distinct, out of order pages: 7 is coprime to the pool size.
    let pages = (0..page_count)
        .map(|index| (7 * index + 3) % POOL_PAGES)
        .collect::<Vec<_>>();
    let mut hosts = [
        patterned(ring.working() as usize, 1),
        patterned(((from.max(to) + 1) * drafter).max(16) as usize, 2),
    ];
    let pool_host = patterned((POOL_PAGES * page_bytes) as usize, 3);
    let upload = |bytes: &[u8]| -> kernel_lab::LabResult<runtime_cuda::DeviceBuffer> {
        let buffer = session.allocate(bytes.len())?;
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let buffers = hosts
        .iter()
        .map(|host| upload(host))
        .collect::<kernel_lab::LabResult<Vec<_>>>()?;
    let pool = upload(&pool_host)?;
    let table = upload(
        &pages
            .iter()
            .flat_map(|page| page.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let module = ring.module(session)?;
    let function = module
        .function(&ring.symbol("prefix_copy"))
        .map_err(|e| e.to_string())?;
    let launch = |slot: u64, to_pages: bool| -> kernel_lab::LabResult<()> {
        let history = if drafter > 0 {
            KernelArgument::Range {
                buffer: &buffers[1],
                offset: (slot * drafter) as usize,
                byte_len: drafter as usize,
            }
        } else {
            KernelArgument::NullBuffer
        };
        function
            .launch(
                &session.stream,
                simple_config(
                    ((total / 16).div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&buffers[0]),
                    history,
                    KernelArgument::U64(drafter),
                    KernelArgument::Buffer(&pool),
                    KernelArgument::Buffer(&table),
                    KernelArgument::U32(page_count as u32),
                    KernelArgument::U64(page_bytes),
                    KernelArgument::U32(slot as u32),
                    KernelArgument::U32(u32::from(to_pages)),
                ],
            )
            .map_err(|e| e.to_string())
    };
    let read = |buffer: &runtime_cuda::DeviceBuffer, bytes: usize| {
        let mut device = vec![0_u8; bytes];
        buffer
            .copy_to_bytes(&mut device, &session.stream)
            .map_err(|e| e.to_string())
            .map(|()| device)
    };

    let piece = |slot: u64| (slot * drafter) as usize..((slot + 1) * drafter) as usize;
    launch(from, true)?;
    let mut expected_pool = pool_host.clone();
    ring_prefix_copy(
        &op(Direction::Capture),
        &mut hosts[0].clone(),
        &mut hosts[1][piece(from)].to_vec(),
        &mut expected_pool,
        &pages,
        from as u32,
    )
    .map_err(|e| e.0.to_string())?;
    if read(&pool, pool_host.len())? != expected_pool {
        return Err(format!(
            "capture pages differ (page bytes {page_bytes}, drafter {drafter}, slot {from})"
        ));
    }

    launch(to, false)?;
    let [ring_host, drafter_host] = &mut hosts;
    ring_prefix_copy(
        &op(Direction::Restore),
        ring_host,
        &mut drafter_host[piece(to)],
        &mut expected_pool,
        &pages,
        to as u32,
    )
    .map_err(|e| e.0.to_string())?;
    for (index, (buffer, host)) in buffers.iter().zip(&hosts).enumerate() {
        let device = read(buffer, host.len())?;
        if device != *host {
            let first = device
                .iter()
                .zip(host)
                .position(|(left, right)| left != right)
                .unwrap_or(0);
            return Err(format!(
                "restore buffer {index} differs at byte {first} (page bytes {page_bytes}, drafter {drafter}, slots {from} to {to})"
            ));
        }
    }
    Ok(())
}

#[test]
fn prefix_copy_round_trips_a_slot_through_scattered_pages() {
    // Per-slot drafter storage as the bundle allocates it: 136 retained MTP
    // hidden rows of 4096 BF16 values, and 1031 DFlash feature rows of 5120
    // BF16 values (the read windows plus eight slack rows).
    const MTP: u64 = 136 * 4096 * 2;
    const DFLASH: u64 = 1031 * 5120 * 2;
    const BF16_PAGE: u64 = 368_640;
    const FP8_PAGE: u64 = 184_320;
    kernel_lab::gpu_test(1 << 30, |session| {
        // The snapshot sizes the host prints at startup, each ending in a
        // partial page, between middle and edge slots in both directions.
        for (page_bytes, drafter, pages) in [
            (BF16_PAGE, MTP, 21),
            (BF16_PAGE, DFLASH, 46),
            (BF16_PAGE, 0, 18),
            (FP8_PAGE, MTP, 41),
            (FP8_PAGE, DFLASH, 92),
            (FP8_PAGE, 0, 35),
        ] {
            for slots in [(5, 9), (SLOTS - 1, 0), (0, SLOTS - 1)] {
                run_prefix_copy(
                    session,
                    Ring::new(Tp::TP4),
                    page_bytes,
                    drafter,
                    Some(pages),
                    slots,
                )?;
            }
        }
        // TP2 rings through their own GA pages, BF16 and FP8.
        {
            let tp = Tp::TP2;
            for fp8 in [false, true] {
                let page_bytes = bundle_mimo26::topology::ga_page_bytes(tp, fp8);
                for drafter in [0, MTP, DFLASH] {
                    run_prefix_copy(session, Ring::new(tp), page_bytes, drafter, None, (5, 9))?;
                }
            }
        }
        Ok(())
    })
    .expect("MiMo prefix snapshot copy");
}
