//! The Qwen3.8 bundle's own rows (`models/qwen38/bundle/cuda/`): the PLE
//! residual, PLE state staging, indexed state copies and the prefix
//! snapshot copy against the contract's own-op oracles (`own_ops::ple`, `own_ops::copies`). Every case plans more rows or items
//! than are live, as a decode bucket does, and checks that the dead ones
//! stay untouched. The prefix snapshot copy round-trips one slot's state
//! through scattered pool pages at the serving geometry, over pools of 32
//! to 256 served slots and live rings of 32 to 256 bucket items.
//!
//! `CHOPINFER_QWEN38_RANKS` (2 or 4, default 4) selects the topology of
//! the linear state entries: the prefix copy of TP2 is the probe
//! entry built with the bundle's entry sizes of that rank count, in
//! `probe_{tag}` as `chop_probe_{tag}_prefix_copy_v1`, and the state copies
//! move entries of that size. The other rows do not depend on the rank
//! count.

use kernel_lab::launch::simple_config;
use model_qwen38_contract::{own_ops::copies, own_ops::ple};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "common/pool_offsets.rs"]
mod pool_offsets;

const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const PLE_CONV_BYTES: usize = 9 * HIDDEN * STREAMS * 2;
const PLE_ENTRY_BYTES: usize = PLE_CONV_BYTES + 128;
const EOS: u32 = 248_044;
const CHUNK: usize = 16 * 1024;
/// The probe rows' PLE state entry.
const LAYOUT: ple::PleLayout = ple::PleLayout {
    wide: (HIDDEN * STREAMS) as u32,
    conv_rows: 9,
    eos: EOS,
};
const SENTINEL: u8 = 0xa5;

/// Live items of the wide decode steps, each planned over its power-of-two
/// bucket.
const WIDE_ITEMS: [usize; 6] = [33, 64, 100, 128, 255, 256];
/// (pool slots, ring slots) of the wide prefix copies: served sequences and
/// the largest graph bucket's items.
const WIDE_SLOTS: [(usize, usize); 4] = [(33, 64), (100, 128), (255, 256), (256, 256)];

/// Deterministic bytes.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

/// Finite BF16 values in [-4, 4).
fn bf16_values(count: usize, seed: u32) -> Vec<u16> {
    noise(4 * count, seed)
        .chunks_exact(4)
        .map(|word| {
            let unit = u32::from_le_bytes(word.try_into().unwrap()) as f32 / 4_294_967_296.0;
            to_bf16(8.0 * unit - 4.0)
        })
        .collect()
}

/// Round to nearest even, as the kernels do.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn halves(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload(session: &kernel_lab::Session, bytes: &[u8]) -> kernel_lab::LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn words(session: &kernel_lab::Session, values: &[i32]) -> kernel_lab::LabResult<DeviceBuffer> {
    let mut bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    bytes.resize(bytes.len().max(4 * 32), 0);
    upload(session, &bytes)
}

fn download(
    session: &kernel_lab::Session,
    buffer: &DeviceBuffer,
    len: usize,
) -> kernel_lab::LabResult<Vec<u8>> {
    let mut bytes = vec![0_u8; len];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// Launches `symbol` of the base cubin `base`, or of this tree's probe when
/// `base` is `None`.
fn launch(
    session: &kernel_lab::Session,
    base: Option<&Module>,
    symbol: &str,
    grid: (u32, u32, u32),
    args: &[KernelArgument<'_>],
) -> kernel_lab::LabResult<()> {
    match base {
        Some(module) => launch_on(session, module, symbol, grid, args),
        None => launch_on(session, &session.module("probe_sm120")?, symbol, grid, args),
    }
}

fn launch_on(
    session: &kernel_lab::Session,
    module: &Module,
    symbol: &str,
    grid: (u32, u32, u32),
    args: &[KernelArgument<'_>],
) -> kernel_lab::LabResult<()> {
    module
        .function(symbol)
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            simple_config(grid, (256, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
            args,
        )
        .map_err(|e| e.to_string())
}

/// The builds a row runs in: this tree's probe (`None`) first, then the
/// base cubin when there is one.
fn builds(base: Option<&Module>) -> Vec<Option<&Module>> {
    std::iter::once(None).chain(base.map(Some)).collect()
}

/// Base versus tip: the base cubin's output bytes must be the tip's.
fn same_as_tip(tip: &[u8], base: &[u8], what: &str) -> kernel_lab::LabResult<()> {
    if tip == base {
        return Ok(());
    }
    let first = tip
        .iter()
        .zip(base)
        .position(|(a, b)| a != b)
        .unwrap_or(tip.len().min(base.len()));
    Err(format!(
        "{what}: the base cubin differs from the tip at byte {first}"
    ))
}

/// The PLE residual over `planned` rows, `live` of them live. With `base`,
/// the base cubin must then give the tip's bytes.
fn ple_residual(
    session: &kernel_lab::Session,
    planned: usize,
    live: usize,
    base: Option<&Module>,
) -> kernel_lab::LabResult<()> {
    let count = planned * STREAMS * HIDDEN;
    let [x, gated, conv] = [2, 3, 4].map(|seed| bf16_values(count, seed));
    let buffers = [&x, &gated, &conv].map(|values| upload(session, &halves(values)));
    let [x_buffer, gated_buffer, conv_buffer] = buffers;
    let (x_buffer, gated_buffer, conv_buffer) = (x_buffer?, gated_buffer?, conv_buffer?);
    let live_rows = words(session, &[live as i32])?;
    let mut expected = vec![u16::from_le_bytes([SENTINEL; 2]); count];
    let op = ple::PleResidual {
        hidden: HIDDEN as u32,
        streams: STREAMS as u32,
    };
    ple::ple_residual(
        &op,
        &x,
        &gated,
        &conv,
        &mut expected,
        live as u32,
        planned as u32,
    )
    .map_err(|e| format!("ple_residual oracle: {}", e.0))?;
    let mut tip: Option<Vec<u8>> = None;
    for build in builds(base) {
        let out = upload(session, &vec![SENTINEL; 2 * count])?;
        launch(
            session,
            build,
            "chop_probe_qwen38_ple_residual_v1",
            (planned as u32, STREAMS as u32, 1),
            &[
                KernelArgument::Buffer(&x_buffer),
                KernelArgument::Buffer(&gated_buffer),
                KernelArgument::Buffer(&conv_buffer),
                KernelArgument::Buffer(&out),
                KernelArgument::Buffer(&live_rows),
            ],
        )?;
        let device = download(session, &out, 2 * count)?;
        if let Some(tip) = &tip {
            same_as_tip(
                tip,
                &device,
                &format!("ple_residual at {live} of {planned} rows"),
            )?;
            continue;
        }
        for (index, &expected) in expected.iter().enumerate() {
            let got = u16::from_le_bytes([device[2 * index], device[2 * index + 1]]);
            if got != expected {
                return Err(format!(
                    "ple_residual element {index}: got {got:#06x}, expected {expected:#06x}"
                ));
            }
        }
        tip = Some(device);
    }
    Ok(())
}

/// PLE staging of `items` ((slot, prefix length) each) over `planned`
/// items from a pool of `pool_slots` entries. With `base`, the base cubin
/// must then give the tip's bytes.
fn ple_stage(
    session: &kernel_lab::Session,
    items: &[(i32, i32)],
    planned: usize,
    pool_slots: usize,
    base: Option<&Module>,
) -> kernel_lab::LabResult<()> {
    let pool = noise(pool_slots * PLE_ENTRY_BYTES, 5);
    let pool_buffer = upload(session, &pool)?;
    let staging_bytes = planned * PLE_ENTRY_BYTES;
    let slot_ids = words(
        session,
        &items.iter().map(|item| item.0).collect::<Vec<_>>(),
    )?;
    let prefix_len = words(
        session,
        &items.iter().map(|item| item.1).collect::<Vec<_>>(),
    )?;
    let live_items = words(session, &[items.len() as i32])?;
    let mut expected = vec![SENTINEL; staging_bytes];
    ple::ple_stage(
        &ple::PleStage { layout: LAYOUT },
        &pool,
        &mut expected,
        &items.iter().map(|item| item.0).collect::<Vec<_>>(),
        &items.iter().map(|item| item.1).collect::<Vec<_>>(),
        items.len() as u32,
        planned as u32,
    )
    .map_err(|e| format!("ple_stage oracle: {}", e.0))?;
    let mut tip: Option<Vec<u8>> = None;
    for build in builds(base) {
        let staging = upload(session, &vec![SENTINEL; staging_bytes])?;
        launch(
            session,
            build,
            "chop_probe_qwen38_ple_stage_v1",
            (planned as u32, PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32, 1),
            &[
                KernelArgument::Buffer(&pool_buffer),
                KernelArgument::Buffer(&staging),
                KernelArgument::Buffer(&slot_ids),
                KernelArgument::Buffer(&prefix_len),
                KernelArgument::Buffer(&live_items),
            ],
        )?;
        let device = download(session, &staging, staging_bytes)?;
        if let Some(tip) = &tip {
            same_as_tip(
                tip,
                &device,
                &format!("ple_stage at {} of {planned} items", items.len()),
            )?;
            continue;
        }
        for item in 0..planned {
            let entry = item * PLE_ENTRY_BYTES..(item + 1) * PLE_ENTRY_BYTES;
            if device[entry.clone()] != expected[entry] {
                return Err(format!(
                    "ple_stage item {item} of {} live, {planned} planned differs",
                    items.len()
                ));
            }
        }
        tip = Some(device);
    }
    Ok(())
}

/// The layout a state copy moves: `planes` planes of `entries` entries of
/// `entry_bytes`.
#[derive(Clone, Copy, Debug)]
struct Planes {
    planes: usize,
    entries: usize,
    entry_bytes: usize,
}

/// One copy: `(src_id, dst_id)` per live item over `planned` items, in
/// place (`src == dst`) or between buffers. With `base`, the base cubin
/// must then give the tip's bytes.
fn copy_entries(
    session: &kernel_lab::Session,
    items: &[(i32, i32)],
    planned: usize,
    layout: Planes,
    in_place: bool,
    base: Option<&Module>,
) -> kernel_lab::LabResult<()> {
    let Planes {
        planes,
        entries,
        entry_bytes,
    } = layout;
    let plane_stride = entries * entry_bytes;
    let bytes = planes * plane_stride;
    let source = noise(bytes, 6 + entry_bytes as u32);
    let target = if in_place {
        source.clone()
    } else {
        vec![SENTINEL; bytes]
    };
    let src_ids = words(
        session,
        &items.iter().map(|item| item.0).collect::<Vec<_>>(),
    )?;
    let dst_ids = words(
        session,
        &items.iter().map(|item| item.1).collect::<Vec<_>>(),
    )?;
    let live = items.len();
    let live_items = words(session, &[live as i32])?;
    let mut expected = target.clone();
    copies::copy_entries(
        &copies::CopyEntries {
            entry_bytes: entry_bytes as u64,
            planes: planes as u32,
            plane_stride: plane_stride as u64,
        },
        &source,
        &mut expected,
        &items.iter().map(|item| item.0).collect::<Vec<_>>(),
        &items.iter().map(|item| item.1).collect::<Vec<_>>(),
        live as u32,
        planned as u32,
    )
    .map_err(|e| format!("copy_entries oracle: {}", e.0))?;
    let what = format!(
        "copy_entries ({entry_bytes} bytes, {planes} planes, in place {in_place}) at {live} of \
         {planned} items"
    );
    let mut tip: Option<Vec<u8>> = None;
    for build in builds(base) {
        let src = upload(session, &source)?;
        let dst = if in_place {
            None
        } else {
            Some(upload(session, &target)?)
        };
        let dst_ref = dst.as_ref().unwrap_or(&src);
        launch(
            session,
            build,
            "chop_probe_qwen38_copy_entries_v1",
            (
                planned as u32,
                planes as u32,
                entry_bytes.div_ceil(CHUNK) as u32,
            ),
            &[
                KernelArgument::Buffer(&src),
                KernelArgument::Buffer(dst_ref),
                KernelArgument::Buffer(&src_ids),
                KernelArgument::Buffer(&dst_ids),
                KernelArgument::Buffer(&live_items),
                KernelArgument::U64(entry_bytes as u64),
                KernelArgument::U64(plane_stride as u64),
            ],
        )?;
        let device = download(session, dst_ref, bytes)?;
        if let Some(tip) = &tip {
            same_as_tip(tip, &device, &what)?;
            continue;
        }
        if device != expected {
            let first = device
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!("{what} differs at byte {first}"));
        }
        tip = Some(device);
    }
    Ok(())
}

/// One decode step's PLE staging and state copies over `count` live items,
/// planned over the power-of-two bucket, for `slots` served slots (a power
/// of two). Each item has its own slot; every fifth item starts a sequence
/// and every seventh has no committed linear state. The PLE commit copies
/// every item's staged entry back to its slot; the linear state copies move
/// committed to working entries (two per slot) in every layer plane of the
/// convolution pool and one plane of the delta-rule pool.
fn item_rows(
    session: &kernel_lab::Session,
    count: usize,
    slots: usize,
    base: Option<&Module>,
) -> kernel_lab::LabResult<()> {
    assert!(slots.is_power_of_two() && count <= slots);
    let bucket = count.next_power_of_two();
    // Distinct slots: the stride is odd and `slots` a power of two.
    let slot = |item: usize| ((37 * item + 11) % slots) as i32;
    let staged = (0..count)
        .map(|item| {
            let prefix = if item % 5 == 1 { 0 } else { item % 13 + 1 };
            (slot(item), prefix as i32)
        })
        .collect::<Vec<_>>();
    ple_stage(session, &staged, bucket, slots, base)?;
    let commit = (0..count)
        .map(|item| (item as i32, slot(item)))
        .collect::<Vec<_>>();
    let ple = Planes {
        planes: 1,
        entries: slots,
        entry_bytes: PLE_ENTRY_BYTES,
    };
    copy_entries(session, &commit, bucket, ple, false, base)?;
    let linear = (0..count)
        .map(|item| {
            let (entry, parity) = (2 * slot(item), (item % 2) as i32);
            let committed = if item % 7 == 3 { -1 } else { entry + parity };
            (committed, entry + 1 - parity)
        })
        .collect::<Vec<_>>();
    let (conv_entry, delta_entry) = state_entries()?;
    for (planes, entry_bytes) in [(LINEAR_LAYERS, conv_entry), (1, delta_entry)] {
        let layout = Planes {
            planes,
            entries: 2 * slots,
            entry_bytes,
        };
        copy_entries(session, &linear, bucket, layout, true, base)?;
    }
    Ok(())
}

// The serving geometries of the prefix snapshot (bundle-qwen38 `prefix.rs`).
const LINEAR_LAYERS: usize = 36;
const RING_KEYS: usize = 16 * 128 * 2;
const RING_POSITIONS: usize = 16 * 3 * 4;
const RETAINED_SLOT: usize = 8 * 4 * HIDDEN * 2;
const HEADER: usize = 16;

/// Bytes of one linear layer's convolution and delta-rule state entries at
/// the selected topology (TP4: 15,360 and 786,432).
fn state_entries() -> kernel_lab::LabResult<(usize, usize)> {
    let shape = kernel_lab::qwen38_shape()?;
    Ok((
        shape.conv_entry_bytes() as usize,
        shape.delta_entry_bytes() as usize,
    ))
}

/// One instance's snapshot: pool entries per slot, ring layers, page bytes,
/// whether the drafter's retained rows follow, and the slots of the pools
/// (the served sequences) and of the live ring (the largest graph bucket's
/// items).
#[derive(Clone, Copy, Debug)]
struct Geometry {
    entries: usize,
    checkpoints: usize,
    ring_layers: usize,
    page_bytes: usize,
    retained: bool,
    slots: usize,
    ring_slots: usize,
}

impl Geometry {
    fn pool_entries(self) -> usize {
        self.slots * self.entries + self.checkpoints
    }

    /// Bytes of one ring layer: every ring slot's keys, then their
    /// positions.
    fn ring_layer(self) -> usize {
        self.ring_slots * (RING_KEYS + RING_POSITIONS)
    }
}

/// Autoregressive, and MTP at two drafts (three verify rows; three drafts
/// take five entries, which only changes the stride but needs 4.5 GB), at
/// 32 served slots and a 32-item ring.
const TARGET: Geometry = Geometry {
    entries: 2,
    checkpoints: 0,
    ring_layers: 12,
    page_bytes: 835_584,
    retained: false,
    slots: 32,
    ring_slots: 32,
};
const DRAFTED: Geometry = Geometry {
    entries: 4,
    checkpoints: 0,
    ring_layers: 13,
    page_bytes: 905_216,
    retained: true,
    slots: 32,
    ring_slots: 32,
};

/// Host bytes of one range of a snapshot buffer that is uploaded and
/// checked.
#[derive(Clone)]
struct Window {
    offset: usize,
    bytes: Vec<u8>,
}

/// The window bytes of snapshot piece `piece`.
fn piece(windows: &mut [Window], piece: copies::Piece) -> &mut [u8] {
    window(windows, piece.offset as usize, piece.length as usize)
}

/// The window holding `[offset, offset + len)`.
fn window(windows: &mut [Window], offset: usize, len: usize) -> &mut [u8] {
    let found = windows
        .iter_mut()
        .find(|w| w.offset <= offset && offset + len <= w.offset + w.bytes.len())
        .expect("snapshot piece inside a checked window");
    let at = offset - found.offset;
    &mut found.bytes[at..at + len]
}

/// Checked ranges of each snapshot buffer: all of it, or when `windowed`
/// (TP2, a lean topology, or a pool of more than 32 slots) for the
/// convolution and delta-rule pools, the `from` and `to` pool entries and
/// their neighbours in every layer. TP2's drafted delta-rule pool is 7.2 GB
/// and TP4's at 256 slots 14.5 GB; their windows are at most 340 MB.
fn windows(
    geometry: Geometry,
    state: (usize, usize),
    sizes: &[usize; 5],
    touched: [(usize, usize); 2],
    windowed: bool,
) -> Vec<Vec<Window>> {
    let pool_entries = geometry.pool_entries();
    let mut indices = touched
        .iter()
        .flat_map(|&(slot, entry)| {
            let index = geometry.entries * slot + entry;
            [index.wrapping_sub(1), index, index + 1]
        })
        .filter(|&index| index < pool_entries)
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    (0..sizes.len())
        .map(|buffer| {
            let ranges = match buffer {
                0 | 1 if windowed => {
                    let entry_bytes = if buffer == 0 { state.0 } else { state.1 };
                    (0..LINEAR_LAYERS)
                        .flat_map(|layer| {
                            indices.iter().map(move |&index| {
                                ((layer * pool_entries + index) * entry_bytes, entry_bytes)
                            })
                        })
                        .collect()
                }
                _ => vec![(0, sizes[buffer])],
            };
            ranges
                .into_iter()
                .map(|(offset, len)| Window {
                    offset,
                    bytes: vec![SENTINEL; len],
                })
                .collect()
        })
        .collect()
}

/// Captures entry `from.1` of slot `from.0` into scattered pool pages,
/// checks the pages hold its pieces end to end and nothing else changed (the
/// last page's slack included), then restores the pages into entry `to.1`
/// of slot `to.0` and checks only that entry changed, to the source's bytes.
/// Lean runs and pools of more than 32 slots check the large pools'
/// `windows` only.
///
/// With `base` (a base cubin's 13-parameter prefix copy, whose pools and
/// ring hold 32 slots), the same capture and restore then run on the base
/// from the same initial bytes and must give the same pages and restored
/// buffers: the reference the tip matched bit for bit, so the base's bytes
/// are the tip's.
fn prefix_copy(
    session: &kernel_lab::Session,
    geometry: Geometry,
    from: (usize, usize),
    to: (usize, usize),
    base: Option<&Module>,
) -> kernel_lab::LabResult<()> {
    prefix_copy_indices(session, geometry, from, to, base, None)
}

fn prefix_copy_indices(
    session: &kernel_lab::Session,
    geometry: Geometry,
    from: (usize, usize),
    to: (usize, usize),
    base: Option<&Module>,
    ple: Option<(usize, usize, usize)>,
) -> kernel_lab::LabResult<()> {
    assert!(
        base.is_none() || (geometry.slots == 32 && geometry.ring_slots == 32),
        "the base prefix copy holds 32 slots"
    );
    let ranks = kernel_lab::qwen38_ranks()?;
    let state = state_entries()?;
    // TP2's snapshots take 70 pages; 72 is coprime to 11 as 48 is.
    let pool_pages: usize = if ranks == 2 { 72 } else { 48 };
    let pool_entries = geometry.pool_entries();
    let sizes = [
        LINEAR_LAYERS * pool_entries * state.0,
        LINEAR_LAYERS * pool_entries * state.1,
        ple.map_or(geometry.slots, |(_, _, entries)| entries) * PLE_ENTRY_BYTES,
        geometry.ring_layers * geometry.ring_layer(),
        if geometry.retained {
            geometry.slots * RETAINED_SLOT
        } else {
            16
        },
    ];
    // The contract's snapshot of this geometry, and the slot, pool entry
    // and PLE entry a copy of `(slot, entry)` moves.
    let snapshot = copies::Snapshot {
        linear_layers: LINEAR_LAYERS as u32,
        conv_entry_bytes: state.0 as u64,
        delta_entry_bytes: state.1 as u64,
        pool_entries: pool_entries as u32,
        ple_entry_bytes: PLE_ENTRY_BYTES as u64,
        ple_entries: (sizes[2] / PLE_ENTRY_BYTES) as u32,
        ring_layers: geometry.ring_layers as u32,
        ring_slots: geometry.ring_slots as u32,
        ring_key_bytes: RING_KEYS as u64,
        ring_position_bytes: RING_POSITIONS as u64,
        retained_bytes: if geometry.retained {
            RETAINED_SLOT as u64
        } else {
            0
        },
    };
    let held = |(slot, entry): (usize, usize)| copies::Held {
        slot: slot as u32,
        index: (geometry.entries * slot + entry) as u32,
        ple_entry: match ple {
            Some((source, _, _)) if (slot, entry) == from => source,
            Some((_, destination, _)) => destination,
            None => slot,
        } as u32,
    };
    let mut window_sizes = sizes;
    window_sizes[2] = geometry.slots * PLE_ENTRY_BYTES;
    let mut hosts = windows(
        geometry,
        state,
        &window_sizes,
        [from, to],
        kernel_lab::qwen38_lean()? || geometry.slots > 32,
    );
    if let Some((source, destination, entries)) = ple {
        assert!(base.is_none() && source < entries && destination < entries);
        for entry in [source, destination] {
            hosts[2].push(Window {
                offset: (entry - 1) * PLE_ENTRY_BYTES,
                bytes: vec![SENTINEL; 3 * PLE_ENTRY_BYTES],
            });
        }
    }
    // Noise in the captured slot's pieces, a sentinel elsewhere.
    for piece in snapshot.pieces(held(from)) {
        let (buffer, offset, length) = (
            piece.buffer as usize,
            piece.offset as usize,
            piece.length as usize,
        );
        window(&mut hosts[buffer], offset, length)
            .copy_from_slice(&noise(length, (buffer * 1000 + offset / 16) as u32));
    }
    let pool_host = vec![
        if ple.is_some() {
            kernel_lab::poison::POISON
        } else {
            0xee_u8
        };
        pool_pages * geometry.page_bytes
    ];
    let total = snapshot.bytes() as usize;
    let page_count = (total + HEADER).div_ceil(geometry.page_bytes);
    // (without, with the drafter) of each topology.
    let expected_pages = match ranks {
        2 => (70, 65),
        _ => (35, 33),
    };
    assert_eq!(
        page_count,
        if geometry.retained {
            expected_pages.1
        } else {
            expected_pages.0
        }
    );
    assert!(page_count <= pool_pages);
    // Distinct, out of order pages: 11 is coprime to the pool size.
    let pages = (0..page_count)
        .map(|index| (11 * index + 3) % pool_pages)
        .collect::<Vec<_>>();
    let table = upload(
        session,
        &pages
            .iter()
            .flat_map(|page| (*page as u64).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    // The pages a capture of `from` writes, and the state a restore of
    // them into `to` leaves.
    let op = copies::PrefixCopy {
        snapshot,
        page_bytes: geometry.page_bytes as u64,
        pages: page_count as u64,
        direction: copies::Direction::Capture,
    };
    let page_ids = pages.iter().map(|&page| page as u64).collect::<Vec<_>>();
    let mut expected_pool = pool_host.clone();
    copies::prefix_capture(
        &op,
        held(from),
        |at| Ok(piece(&mut hosts[at.buffer as usize], at).to_vec()),
        &page_ids,
        &mut expected_pool,
    )
    .map_err(|e| format!("prefix capture oracle: {}", e.0))?;
    let mut expected = hosts.clone();
    copies::prefix_restore(&op, held(to), &page_ids, &expected_pool, |at, bytes| {
        piece(&mut expected[at.buffer as usize], at).copy_from_slice(bytes);
        Ok(())
    })
    .map_err(|e| format!("prefix restore oracle: {}", e.0))?;
    let tag = kernel_lab::qwen38_profile()?.tag();
    let tip = if ple.is_some() {
        session
            .context
            .load_module(&owned_cubins::load(&format!("{tag}_bundle"))?)
            .map_err(|e| e.to_string())?
    } else {
        session.module(&if ranks == 4 {
            "probe_sm120".to_string()
        } else {
            format!("probe_{tag}")
        })?
    };
    let mut symbol = format!("chop_probe_{tag}_prefix_copy_v1");
    if ple.is_some() {
        symbol = symbol
            .replacen("chop_probe_", "chop_", 1)
            .replace("_v1", "_indexed_v1");
    }
    let mut runs = vec![(&tip, "tip")];
    runs.extend(base.map(|module| (module, "base")));
    for (module, build) in runs {
        let buffers = hosts
            .iter()
            .zip(sizes)
            .map(|(windows, size)| {
                let buffer = session.allocate(size)?;
                for w in windows {
                    buffer
                        .copy_from_bytes_at(w.offset, &w.bytes, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                session.stream.synchronize().map_err(|e| e.to_string())?;
                Ok(buffer)
            })
            .collect::<kernel_lab::LabResult<Vec<_>>>()?;
        let pool = upload(session, &pool_host)?;
        let run = |(slot, entry): (usize, usize), to_pages: bool, pdl: bool| {
            let mut args = vec![
                KernelArgument::Buffer(&buffers[0]),
                KernelArgument::Buffer(&buffers[1]),
                KernelArgument::Buffer(&buffers[2]),
                KernelArgument::Buffer(&buffers[3]),
                if geometry.retained {
                    KernelArgument::Buffer(&buffers[4])
                } else {
                    KernelArgument::NullBuffer
                },
                KernelArgument::Buffer(&pool),
                KernelArgument::Buffer(&table),
            ];
            if build == "tip" {
                // The tip addresses the linear entry directly, over planes
                // of every pool entry.
                args.extend([
                    KernelArgument::U32(geometry.ring_layers as u32),
                    KernelArgument::U64(geometry.page_bytes as u64),
                    KernelArgument::U32(slot as u32),
                    KernelArgument::U32((slot * geometry.entries + entry) as u32),
                    KernelArgument::U32(u32::from(to_pages)),
                    KernelArgument::U32(pool_entries as u32),
                    KernelArgument::U32(geometry.ring_slots as u32),
                ]);
            } else {
                // The base's slots are its compiled 32.
                args.extend([
                    KernelArgument::U32(geometry.entries as u32),
                    KernelArgument::U32(geometry.ring_layers as u32),
                    KernelArgument::U64(geometry.page_bytes as u64),
                    KernelArgument::U32(slot as u32),
                    KernelArgument::U32(entry as u32),
                    KernelArgument::U32(u32::from(to_pages)),
                ]);
            }
            if let Some((source, destination, entries)) = ple {
                args.extend([
                    KernelArgument::U32(if (slot, entry) == from {
                        source
                    } else {
                        destination
                    } as u32),
                    KernelArgument::U32(entries as u32),
                ]);
            }
            module
                .function(&symbol)
                .map_err(|e| e.to_string())?
                .launch(
                    &session.stream,
                    simple_config(
                        ((total / 16).div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        pdl,
                    )?,
                    &args,
                )
                .map_err(|e| e.to_string())
        };

        let graph = if ple.is_some() {
            Some(session.capture(|_| {
                run(from, true, true)?;
                run(to, false, true)
            })?)
        } else {
            None
        };
        for cycle in 0..if ple.is_some() { 4 } else { 1 } {
            if cycle > 0 {
                for (buffer, windows) in buffers.iter().zip(&hosts) {
                    for w in windows {
                        buffer
                            .copy_from_bytes_at(w.offset, &w.bytes, &session.stream)
                            .map_err(|e| e.to_string())?;
                    }
                }
                kernel_lab::poison::poison(session, &[&pool])?;
            }
            if cycle >= 2 {
                graph
                    .as_ref()
                    .unwrap()
                    .launch(&session.stream)
                    .map_err(|e| e.to_string())?;
            } else {
                run(from, true, cycle == 0)?;
            }

            if download(session, &pool, pool_host.len())? != expected_pool {
                return Err(format!(
                    "{build} prefix capture pages differ ({geometry:?}, {from:?} to {to:?})"
                ));
            }
            if cycle < 2 {
                run(to, false, cycle == 0)?;
            }
            for (index, buffer) in buffers.iter().enumerate() {
                for w in &expected[index] {
                    let mut got = vec![0_u8; w.bytes.len()];
                    buffer
                        .copy_to_bytes_at_async(w.offset, &mut got, &session.stream)
                        .map_err(|e| e.to_string())?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    if got != w.bytes {
                        return Err(format!(
                            "{build} prefix restore differs in buffer {index} ({geometry:?}, {from:?} \
                         to {to:?})"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// PLE state entries past the 32-bit lines (`pool_offsets`): `ple_stage`
/// reads committed entries whose slot ids alternate between group A and
/// group B of the 184,448-byte entry, beside fresh items, and an in-place
/// `copy_entries` over one PLE plane moves entries across the groups. The
/// staging and every pool entry hold the expected bytes and equal the
/// low-id run's bit for bit, and no other pool byte changes.
#[test]
fn ple_state_high_ids_match_low_ids() {
    kernel_lab::gpu_test(pool_offsets::need_bytes(5 << 30), |session| {
        let entry = PLE_ENTRY_BYTES as u64;
        let ids = pool_offsets::high_ids(entry, 1, 4);
        let contents: Vec<Vec<u8>> = (0..8).map(|i| noise(PLE_ENTRY_BYTES, 40 + i)).collect();
        // (pool entry, prefix length): items with a prefix stage entries 0
        // to 3; items without one stage a fresh state.
        let items: [(usize, i32); 6] = [(0, 5), (1, 17), (4, 0), (2, 1), (3, 300), (5, 0)];
        // (source entry or -1, destination entry): entries 0 to 3 onto
        // entries 4 to 7, each into the other group, then a skipped item.
        let moves: [(i32, usize); 5] = [(0, 5), (1, 4), (2, 7), (3, 6), (-1, 4)];
        let fresh = LAYOUT.fresh();
        let planned = 8_usize;
        let mut results = Vec::new();
        for (placement, placed) in [
            ("low", (0..8).collect::<Vec<u64>>()),
            ("high", ids.alternating(8)),
        ] {
            let pool_bytes = (placed.iter().copied().max().unwrap_or(0) + 1) * entry;
            let mut pool = pool_offsets::SentinelPool::new(session, pool_bytes)?;
            for (i, &id) in placed.iter().enumerate() {
                pool.put(session, id * entry, &contents[i])?;
            }
            let at = |e: usize| i32::try_from(placed[e]).expect("an id below i32::MAX");
            let slot_ids = words(
                session,
                &items.iter().map(|&(e, _)| at(e)).collect::<Vec<_>>(),
            )?;
            let prefix_len = words(session, &items.iter().map(|&(_, p)| p).collect::<Vec<_>>())?;
            let live_items = words(session, &[items.len() as i32])?;
            let staging = upload(session, &vec![SENTINEL; planned * PLE_ENTRY_BYTES])?;
            launch(
                session,
                None,
                "chop_probe_qwen38_ple_stage_v1",
                (planned as u32, PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32, 1),
                &[
                    KernelArgument::Buffer(&pool.buffer),
                    KernelArgument::Buffer(&staging),
                    KernelArgument::Buffer(&slot_ids),
                    KernelArgument::Buffer(&prefix_len),
                    KernelArgument::Buffer(&live_items),
                ],
            )?;
            let staged = download(session, &staging, planned * PLE_ENTRY_BYTES)?;
            for (item, got) in staged.chunks_exact(PLE_ENTRY_BYTES).enumerate() {
                let ok = match items.get(item) {
                    Some(&(e, prefix)) if prefix > 0 => got == &contents[e][..],
                    Some(_) => got == &fresh[..],
                    None => got.iter().all(|&byte| byte == SENTINEL),
                };
                if !ok {
                    return Err(format!("{placement} ple_stage item {item} differs"));
                }
            }
            let src_ids = words(
                session,
                &moves
                    .iter()
                    .map(|&(from, _)| if from < 0 { -1 } else { at(from as usize) })
                    .collect::<Vec<_>>(),
            )?;
            let dst_ids = words(
                session,
                &moves.iter().map(|&(_, to)| at(to)).collect::<Vec<_>>(),
            )?;
            let live_moves = words(session, &[moves.len() as i32])?;
            launch(
                session,
                None,
                "chop_probe_qwen38_copy_entries_v1",
                (planned as u32, 1, PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32),
                &[
                    KernelArgument::Buffer(&pool.buffer),
                    KernelArgument::Buffer(&pool.buffer),
                    KernelArgument::Buffer(&src_ids),
                    KernelArgument::Buffer(&dst_ids),
                    KernelArgument::Buffer(&live_moves),
                    KernelArgument::U64(entry),
                    KernelArgument::U64(pool_bytes),
                ],
            )?;
            let mut after = Vec::new();
            for (i, &id) in placed.iter().enumerate() {
                let expected = moves
                    .iter()
                    .find(|&&(from, to)| from >= 0 && to == i)
                    .map_or(&contents[i], |&(from, _)| &contents[from as usize]);
                let got = pool.read(session, id * entry, entry)?;
                if &got != expected {
                    return Err(format!(
                        "{placement} copy_entries entry {i} at id {id} differs"
                    ));
                }
                after.push(got);
            }
            let scanned = pool.check_untouched(session, placement)?;
            results.push((staged, after, scanned, placed, pool_bytes));
        }
        let (low, high) = (&results[0], &results[1]);
        pool_offsets::same("ple", "staging", &low.0, &high.0)?;
        pool_offsets::same("ple", "pool entries", &low.1.concat(), &high.1.concat())?;
        pool_offsets::receipt(
            session,
            "ple",
            "ple_stage_and_copy_entries",
            json!({
                "entry_bytes": entry,
                "ids": ids.describe(),
                "entry_ids": high.3,
                "highest_offset": high.3.iter().copied().max().unwrap_or(0) * entry,
                "pool_bytes": high.4,
                "staging_bitwise": true,
                "entries_bitwise": true,
                "sentinel_bytes_checked": high.2,
            }),
        )
    })
    .expect("PLE state at high slot and entry ids");
}

#[test]
fn qwen38_prefix_copy_round_trips_a_slot_through_scattered_pages() {
    // TP2's drafted delta-rule pool alone is 7.2 GB.
    let need = match kernel_lab::qwen38_ranks() {
        Ok(2) => 9 << 30,
        _ => 4 << 30,
    };
    kernel_lab::gpu_test(need, |session| {
        // Both committed parities, and the first and last slots.
        prefix_copy(session, TARGET, (5, 1), (9, 0), None)?;
        prefix_copy(session, TARGET, (0, 0), (31, 0), None)?;
        prefix_copy(session, TARGET, (31, 1), (0, 0), None)?;
        // MTP: four entries per slot, the drafter's ring layer and the
        // retained rows, committed entries past the first two.
        prefix_copy(session, DRAFTED, (5, 3), (9, 0), None)?;
        prefix_copy(session, DRAFTED, (31, 2), (0, 0), None)?;
        session.stream.synchronize().map_err(|e| e.to_string())
    })
    .expect("Qwen3.8 prefix snapshot copy");
}

/// Qwen pools of up to 256 served slots and live rings of up to the
/// 256-item bucket's slots, through the prefix copy's `slots` and
/// `ring_slots`. Each wide geometry round-trips from the first slot to the
/// last, from the last to a middle one and from the middle one to the
/// first, over both committed parities; the drafter's geometry does the
/// same at 32 slots. The wide pools check windows around the touched
/// entries in every layer, the ring, PLE pool and page pool in full.
#[test]
fn qwen38_prefix_copy_wide_round_trips_a_slot_through_scattered_pages() {
    // At 256 slots TP4's delta-rule pool is 36 x 512 entries of 786,432
    // bytes (14.5 GB) and the rest 0.6 GB; TP2 doubles the entries.
    let need = match kernel_lab::qwen38_ranks() {
        Ok(2) => 30 << 30,
        _ => 16 << 30,
    };
    kernel_lab::gpu_test(need, |session| {
        let mut round_trips = Vec::new();
        let geometries = WIDE_SLOTS
            .map(|(slots, ring_slots)| Geometry {
                slots,
                ring_slots,
                ..TARGET
            })
            .into_iter()
            .chain([DRAFTED]);
        for geometry in geometries {
            let (last, middle) = (geometry.slots - 1, geometry.slots / 2);
            let top = geometry.entries - 1;
            for (from, to) in [
                ((0, 0), (last, 1)),
                ((last, top), (middle, 0)),
                ((middle, 1), (0, top)),
            ] {
                prefix_copy(session, geometry, from, to, None)?;
                round_trips.push(json!({
                    "slots": geometry.slots,
                    "ring_slots": geometry.ring_slots,
                    "entries": geometry.entries,
                    "from": [from.0, from.1],
                    "to": [to.0, to.1],
                }));
            }
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let receipt = BTreeMap::from([
            (
                "check".into(),
                json!("qwen38_prefix_copy_wide_round_trips_a_slot_through_scattered_pages"),
            ),
            ("qwen38_ranks".into(), json!(kernel_lab::qwen38_ranks()?)),
            ("round_trips".into(), json!(round_trips)),
            ("bitwise".into(), json!(true)),
        ]);
        session.receipt("qwen38-prefix-copy-wide", receipt)?;
        Ok(())
    })
    .expect("Qwen3.8 wide prefix snapshot copy");
}

#[test]
fn qwen38_bundle_rows_match_cpu_references() {
    kernel_lab::gpu_test(1 << 30, |session| {
        ple_residual(session, 16, 11, None)?;
        // (slot, prefix): continuing, fresh, continuing by one token; one
        // planned item is dead.
        ple_stage(session, &[(2, 5), (0, 0), (3, 1)], 4, 4, None)?;
        // The bundle's linear-state copies: committed to working entries of
        // one pool, 36 layer planes of 64 entries (convolution and delta
        // rule, the selected topology's entry sizes), with an item that has
        // no committed entry.
        let items = [(0, 1), (-1, 3), (6, 7), (60, 13)];
        let planned = items.len().next_power_of_two() + 1;
        let (conv_entry, delta_entry) = state_entries()?;
        for (planes, entry_bytes) in [(36, conv_entry), (3, delta_entry)] {
            let layout = Planes {
                planes,
                entries: 64,
                entry_bytes,
            };
            copy_entries(session, &items, planned, layout, true, None)?;
        }
        // The PLE commit: staged entries back to their slots.
        let commit = [(0, 5), (1, 2), (2, 0)];
        let ple = Planes {
            planes: 1,
            entries: 8,
            entry_bytes: PLE_ENTRY_BYTES,
        };
        copy_entries(
            session,
            &commit,
            commit.len().next_power_of_two() + 1,
            ple,
            false,
            None,
        )?;
        session.stream.synchronize().map_err(|e| e.to_string())
    })
    .expect("Qwen3.8 bundle rows");
}

/// Qwen decode buckets of up to 256 items. PLE staging, the PLE
/// commit over every served item and the linear state copies at 33, 64,
/// 100, 128, 255 and 256 live items, each launched over its power-of-two
/// bucket with the live count on the device, against 256 served slots: the
/// padded items must leave their destinations untouched.
#[test]
fn qwen38_bundle_rows_wide_items_match_cpu_references() {
    // The largest copy: TP2's delta-rule plane of 512 entries, 0.8 GB.
    kernel_lab::gpu_test(2 << 30, |session| {
        for count in WIDE_ITEMS {
            item_rows(session, count, 256, None)?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let receipt = BTreeMap::from([
            (
                "check".into(),
                json!("qwen38_bundle_rows_wide_items_match_cpu_references"),
            ),
            ("qwen38_ranks".into(), json!(kernel_lab::qwen38_ranks()?)),
            ("live_items".into(), json!(WIDE_ITEMS)),
            ("slots".into(), json!(256)),
            (
                "rows".into(),
                json!(["ple_stage", "copy_entries ple", "copy_entries linear"]),
            ),
            ("bitwise".into(), json!(true)),
        ]);
        session.receipt("qwen38-bundle-rows-wide", receipt)?;
        Ok(())
    })
    .expect("Qwen3.8 wide bundle rows");
}

/// Keep arbitrary accepted checkpoints, including the high served slots,
/// and snapshot only the committed entry with the checkpoint-pool stride.
#[test]
fn qwen38_checkpoint_pool_copies_and_snapshots() {
    kernel_lab::gpu_test(24 << 30, |session| {
        for slots in [33, 64, 128, 256] {
            let checkpoints = 32 * 4;
            // Copy two representative layer planes at real entry widths;
            // snapshot tests exercise the full 36-layer stride.
            for entry_bytes in [state_entries()?.0, state_entries()?.1] {
                let items = (0..32)
                    .map(|i| ((slots + i * 4 + i % 4) as i32, (slots - 1 - i) as i32))
                    .collect::<Vec<_>>();
                copy_entries(
                    session,
                    &items,
                    32,
                    Planes {
                        planes: 2,
                        entries: slots + checkpoints,
                        entry_bytes,
                    },
                    true,
                    None,
                )?;
            }
            let geometry = Geometry {
                entries: 1,
                checkpoints,
                slots,
                ring_slots: slots.next_power_of_two(),
                ..DRAFTED
            };
            prefix_copy(session, geometry, (slots - 1, 0), (0, 0), None)?;
        }
        session.receipt(
            "qwen38-checkpoint-pool-copies",
            BTreeMap::from([("bitwise".into(), json!(true))]),
        )?;
        Ok(())
    })
    .expect("checkpoint commit copies and single-entry snapshots preserve exact bytes");
}

#[test]
fn prefix_copy_uses_independent_entry_indices() {
    kernel_lab::gpu_test(8 << 30, |session| {
        prefix_copy_indices(
            session,
            Geometry {
                slots: 2,
                ring_slots: 2,
                ..TARGET
            },
            (0, 1),
            (1, 0),
            None,
            Some((30_001, 34_001, 34_003)),
        )
    })
    .expect(
        "indirect prefix copies preserve bytes across high entry offsets, PDL and graph replay",
    );
}

#[path = "common/owned_cubins.rs"]
mod owned_cubins;
