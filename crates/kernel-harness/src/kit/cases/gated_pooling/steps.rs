//! Group crossings, holes inside an item, and verify checkpoints compared
//! with one-row decode from the exact state before each accepted row.

use super::*;
use crate::kit::Decoded;
use crate::kit::decode::{Compared, DecodeLaunch, Verified};

const ITEMS: usize = 2;
const STATES: usize = 16;
const POS: [i32; 6] = [1, 2, -1, 0, -1, 1];
const BATCH: [i32; 6] = [0, 0, 0, 1, 1, 1];
// The final four slots are reserved for incoming state and its high-ID
// permutation. Checkpoints remain disjoint in both layouts.
const SOURCES: [i32; 2] = [13, 15];

fn checkpoint_rows(query: GatedPoolingQuery) -> usize {
    match query.phase {
        RecurrentPhase::Verify { max_rows } => max_rows as usize,
        _ => 0,
    }
}

fn state_ranges(
    query: GatedPoolingQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<std::ops::Range<usize>>> {
    let width = query.width as usize;
    let positions = images.i32s("op.positions")?;
    let items = images.i32s("op.row_batch")?;
    let slots = images.i32s("op.tail.slot_ids")?;
    let max_rows = checkpoint_rows(query);
    let checkpoints = if max_rows > 0 {
        images.i32s("op.tail.checkpoint_ids")?
    } else {
        vec![]
    };
    let mut targets = Vec::new();
    let mut tails = [None; ITEMS];
    for row in 0..live as usize {
        let position = positions[row];
        if position < 0 {
            continue;
        }
        let item = items[row] as usize;
        tails[item] = (position % 2 == 0).then_some(slots[item]);
        if max_rows > 0 && position % 2 == 0 {
            let start = (0..=row)
                .find(|&r| items[r] == items[row])
                .ok_or("missing item start")?;
            targets.push(checkpoints[item * max_rows + row - start]);
        }
    }
    if max_rows == 0 {
        targets.extend(tails.into_iter().flatten());
    }
    Ok(targets
        .into_iter()
        .map(|id| {
            let at = id as usize * 2 * width * 4;
            at..at + 2 * width * 4
        })
        .collect())
}

/// A first odd-position row reads the incoming tail, even if a later
/// even-position row replaces that same entry before this launch ends.
fn updated_tail_ranges(
    query: GatedPoolingQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<std::ops::Range<usize>>> {
    if checkpoint_rows(query) > 0 {
        return Ok(vec![]);
    }
    let positions = images.i32s("op.positions")?;
    let items = images.i32s("op.row_batch")?;
    let slots = images.i32s("op.tail.slot_ids")?;
    let stride = 2 * query.width as usize * 4;
    let mut seen = [false; ITEMS];
    let mut incoming = Vec::new();
    for row in 0..live as usize {
        if positions[row] < 0 {
            continue;
        }
        let item = items[row] as usize;
        if !seen[item] && positions[row] % 2 == 1 {
            let at = slots[item] as usize * stride;
            incoming.push(at..at + stride);
        }
        seen[item] = true;
    }
    Ok(state_ranges(query, images, live)?
        .into_iter()
        .filter(|range| incoming.contains(range))
        .collect())
}

fn case(
    query: GatedPoolingQuery,
    prepared: Prepared<GatedPoolingBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    if !matches!(
        query.phase,
        RecurrentPhase::Prefill | RecurrentPhase::Decode | RecurrentPhase::Verify { max_rows: 6 }
    ) {
        return Err("grouped pooling case needs in-place or six-row verify".into());
    }
    let width = query.width as usize;
    let base = |d: usize| if d % 3 == 0 { -8.0 } else { 8.0 };
    let mut keys = vec![0.0; CAPACITY as usize * width];
    let mut gates = keys.clone();
    let mut state = vec![0.0; STATES * 2 * width];
    for id in 0..STATES {
        for d in 0..width {
            let held = if (id + d) % 3 == 0 { -8.0 } else { 8.0 };
            state[id * 2 * width + d] = held + (id % 7) as f32 / 2048.0;
            state[id * 2 * width + width + d] = (id % 5) as f32 / 4.0;
        }
    }
    for row in 0..CAPACITY as usize {
        for d in 0..width {
            keys[row * width + d] = base(d) + ((row + d) % 5) as f32 / 1024.0;
            gates[row * width + d] = (d % 7) as f32 / 4.0;
            // The incoming member dominates exactly in both F32 and f64.
            // Its slot-specific signs make stale source IDs change exact
            // pooled output bits, not only a numeric norm reference.
            if row == 0 {
                gates[row * width + d] = -1000.0;
            }
        }
    }
    let weights: Vec<u16> = (0..width)
        .map(|d| f32_to_bf16(0.5 + (d % 5) as f32 / 4.0))
        .collect();
    let max_rows = checkpoint_rows(query);
    let roles = ObservedBuffers {
        op: GatedPoolingBuffers {
            key: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&keys)),
            gate: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&gates)),
            norm_weight: Spec::input(DType::Bf16, (Rows::Fixed(1), width), bf16_bytes(&weights)),
            positions: Spec::ids((Rows::Live, 1), POS.to_vec(), &[], Holes::LastLive),
            row_batch: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&BATCH)),
            live_rows: Spec::live(),
            tail: GatedPoolingState {
                state: Spec::state_updating(
                    DType::F32,
                    (Rows::Fixed(STATES), 2 * width),
                    f32_bytes(&state),
                    Write::Map(Box::new(move |images, live| {
                        let updated = updated_tail_ranges(query, images, live)?;
                        WriteMap::of(DType::F32).byte_ranges(
                            state_ranges(query, images, live)?
                                .into_iter()
                                .filter(|range| !updated.contains(range)),
                        )
                    })),
                    move |images, live| updated_tail_ranges(query, images, live),
                ),
                entries: STATES as u32,
                slot_ids: Spec::ids(
                    (Rows::Fixed(ITEMS), 1),
                    SOURCES.to_vec(),
                    &[("op.tail.state", 2 * width * 4)],
                    Holes::Never,
                ),
                checkpoint_ids: (max_rows > 0).then(|| {
                    // Source slots own the moving state-pool probe. The first
                    // twelve entries are fixed checkpoint destinations.
                    Spec::input(
                        DType::I32,
                        (Rows::Fixed(ITEMS), max_rows),
                        i32_bytes(&(0..(ITEMS * max_rows) as i32).collect::<Vec<_>>()),
                    )
                }),
            },
            output: Spec::output_with(
                DType::F32,
                (Rows::Live, width),
                output_map(DType::F32, width, 4),
            ),
            output_positions: Spec::output(DType::I32, (Rows::Live, 1)).filled_with(&[0x5a]),
        },
        pooled: Spec::output_with(
            DType::F32,
            (Rows::Live, width),
            output_map(DType::F32, width, 4),
        ),
    };
    let mut info = prepared.info().clone();
    info.workspace = 0;
    let mut result = OpCase::with_counts(
        Box::new(Observed { op: prepared, info }),
        CAPACITY,
        roles,
        |rows| Counts::rows(rows).with_items(ITEMS as u32, 0),
    )?
    .oracle(move |images, live| oracle(query, images, live))
    .live_counts(vec![0, 1, 2, 3, 5, 6])
    .no_write(
        "op.tail.state",
        1,
        "the first row completes its incoming group",
    );
    if max_rows > 0 {
        result = result
            .item_rows("two items of three token rows, with an internal hole")
            .batch_images(move |images, live| batch_images(query, images, live));
    }
    Ok(result)
}

fn batch_images(
    query: GatedPoolingQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<crate::kit::batch::Comparison>> {
    if live != CAPACITY {
        return Err("item batch probe needs both three-row items".into());
    }
    let width = query.width as usize;
    let positions = images.i32s("op.positions")?;
    let checkpoints = images.i32s("op.tail.checkpoint_ids")?;
    let mut probes = Vec::new();
    for item in 0..ITEMS {
        let other = 1 - item;
        let mut changed = images.clone();
        for path in ["op.key", "op.gate"] {
            let mut values = images.f32s(path)?;
            for value in &mut values[other * 3 * width..(other + 1) * 3 * width] {
                *value = -*value;
            }
            changed = changed.with(path, f32_bytes(&values));
        }
        let mut compared = Vec::new();
        for row in item * 3..(item + 1) * 3 {
            compared.push(("op.output_positions".into(), row * 4..(row + 1) * 4));
            if positions[row] < 0 {
                continue;
            }
            if positions[row] % 2 == 1 {
                for (path, bytes) in [("op.output", width * 4), ("pooled", width * 4)] {
                    compared.push((path.into(), row * bytes..(row + 1) * bytes));
                }
            } else {
                let id = checkpoints[item * checkpoint_rows(query) + row - item * 3] as usize;
                compared.push((
                    "op.tail.state".into(),
                    id * 2 * width * 4..(id + 1) * 2 * width * 4,
                ));
            }
        }
        probes.push(crate::kit::batch::Comparison {
            label: format!("item {item}"),
            images: changed,
            compared,
        });
    }
    Ok(probes)
}

fn oracle(
    query: GatedPoolingQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let width = query.width as usize;
    let keys = images.f32s("op.key")?;
    let gates = images.f32s("op.gate")?;
    let source = images.f32s("op.tail.state")?;
    let mut state = images.bytes("op.tail.state")?.to_vec();
    let slots = images.i32s("op.tail.slot_ids")?;
    let positions = images.i32s("op.positions")?;
    let items = images.i32s("op.row_batch")?;
    let weights: Vec<f64> = images
        .u16s("op.norm_weight")?
        .into_iter()
        .map(|w| f64::from(bf16_to_f32(w)))
        .collect();
    let max_rows = checkpoint_rows(query);
    let checkpoints = if max_rows > 0 {
        images.i32s("op.tail.checkpoint_ids")?
    } else {
        vec![]
    };
    let n = CAPACITY as usize * width;
    let mut pooled = vec![0.0f32; n];
    let mut output = OracleOutput::new(vec![0.0; n], vec![0.0; n], width).map_err(|e| e.0)?;
    let mut output_positions = vec![-1; CAPACITY as usize];
    let mut previous = [None; ITEMS];
    let mut pending = [None; ITEMS];
    let write = |state: &mut [u8], id: i32, row: usize| {
        let target = id as usize * 2 * width * 4;
        state[target..target + width * 4]
            .copy_from_slice(&f32_bytes(&keys[row * width..(row + 1) * width]));
        state[target + width * 4..target + 2 * width * 4]
            .copy_from_slice(&f32_bytes(&gates[row * width..(row + 1) * width]));
    };
    for row in 0..live as usize {
        let position = positions[row];
        if position < 0 {
            continue;
        }
        let item = items[row] as usize;
        if position % 2 == 0 {
            pending[item] = Some(row);
            if max_rows > 0 {
                let first = (0..=row)
                    .find(|&r| items[r] == items[row])
                    .ok_or("missing item")?;
                write(&mut state, checkpoints[item * max_rows + row - first], row);
            }
        } else {
            pending[item] = None;
            output_positions[row] = position / 2;
            let (earlier_key, earlier_gate) = if let Some(prev) = previous[item] {
                if positions[prev] != position - 1 {
                    return Err("fixture has a nonconsecutive group".into());
                }
                (
                    &keys[prev * width..(prev + 1) * width],
                    &gates[prev * width..(prev + 1) * width],
                )
            } else {
                let at = slots[item] as usize * 2 * width;
                (&source[at..at + width], &source[at + width..at + 2 * width])
            };
            let group_keys: Vec<f32> = earlier_key
                .iter()
                .chain(&keys[row * width..(row + 1) * width])
                .copied()
                .collect();
            let group_gates: Vec<f32> = earlier_gate
                .iter()
                .chain(&gates[row * width..(row + 1) * width])
                .copied()
                .collect();
            let reference =
                gated_pooling_oracle::pool_without_position(query, &group_keys, &group_gates)
                    .map_err(|e| e.0)?;
            if reference.values.iter().any(|&v| f64::from(v as f32) != v) {
                return Err("grouped fixture must have an exact F32 pool".into());
            }
            let norm = gated_pooling_oracle::norm_after_pool(
                query,
                &reference
                    .values
                    .iter()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
                &weights,
            )
            .map_err(|e| e.0)?;
            for d in 0..width {
                let at = row * width + d;
                pooled[at] = reference.values[d] as f32;
                output.values[at] = norm.values[d];
                output.sum_abs_products[at] = norm.sum_abs_products[d];
                output.reduction_lengths[at] = norm.reduction_lengths[d];
            }
        }
        previous[item] = Some(row);
    }
    if max_rows == 0 {
        for item in 0..ITEMS {
            if let Some(row) = pending[item] {
                write(&mut state, slots[item], row);
            }
        }
    }
    Ok(vec![
        ("pooled".into(), Expected::Bits(f32_bytes(&pooled))),
        ("op.output".into(), Expected::Values(output)),
        (
            "op.output_positions".into(),
            Expected::Bits(i32_bytes(&output_positions)),
        ),
        ("op.tail.state".into(), Expected::Bits(state)),
    ])
}

fn decoded(
    query: GatedPoolingQuery,
    prepared: Prepared<GatedPoolingBuffers<View>>,
) -> LabResult<Decoded<'static>> {
    let decode_case = case(query, prepared)?;
    let width = query.width as usize;
    Ok(Decoded::new(
        decode_case,
        |images, max| {
            if max < 3 {
                return Err("the verify case holds three rows per item".into());
            }
            Ok(images.clone())
        },
        move |verified: &Verified<'_>, base: &Images| {
            let positions = verified.images.i32s("op.positions")?;
            let items = verified.images.i32s("op.row_batch")?;
            let slots = verified.images.i32s("op.tail.slot_ids")?;
            let checkpoints = verified.images.i32s("op.tail.checkpoint_ids")?;
            let committed = verified.images.bytes("op.tail.state")?;
            let left = verified
                .results
                .get("op.tail.state")
                .ok_or("verify returned no state")?;
            let mut previous = [None; ITEMS];
            let mut launches = Vec::new();
            for row in 0..verified.live as usize {
                let item = items[row] as usize;
                let start = (0..=row)
                    .find(|&r| items[r] == items[row])
                    .ok_or("missing item")?;
                let from = previous[item].map_or(slots[item], |prev: usize| {
                    checkpoints[item * 6 + prev - start]
                });
                let state = if previous[item].is_some() {
                    left.as_slice()
                } else {
                    committed
                };
                let mut held = base.with("op.tail.state", state.to_vec()).with(
                    "op.norm_weight",
                    verified.images.bytes("op.norm_weight")?.to_vec(),
                );
                let mut decode_slots = vec![0; ITEMS];
                decode_slots[item] = from;
                decode_slots[1 - item] = if from == 0 { 1 } else { 0 };
                held = held.with("op.tail.slot_ids", i32_bytes(&decode_slots));
                for (path, bytes) in [
                    ("op.key", width * 4),
                    ("op.gate", width * 4),
                    ("op.positions", 4),
                    ("op.row_batch", 4),
                ] {
                    let mut image = base.bytes(path)?.to_vec();
                    image[..bytes].copy_from_slice(
                        &verified.images.bytes(path)?[row * bytes..(row + 1) * bytes],
                    );
                    held = held.with(path, image);
                }
                let mut compared = vec![Compared {
                    verify: ("op.output_positions".into(), row * 4..row * 4 + 4),
                    decode: ("op.output_positions".into(), 0..4),
                }];
                if positions[row] >= 0 {
                    if positions[row] % 2 == 0 {
                        let to = checkpoints[item * 6 + row - start] as usize * 2 * width * 4;
                        let from = from as usize * 2 * width * 4;
                        compared.push(Compared {
                            verify: ("op.tail.state".into(), to..to + 2 * width * 4),
                            decode: ("op.tail.state".into(), from..from + 2 * width * 4),
                        });
                    } else {
                        for (path, bytes) in [("pooled", width * 4), ("op.output", width * 4)] {
                            compared.push(Compared {
                                verify: (path.into(), row * bytes..(row + 1) * bytes),
                                decode: (path.into(), 0..bytes),
                            });
                        }
                    }
                    previous[item] = Some(row);
                }
                launches.push(DecodeLaunch {
                    label: format!("item {item}, row {row}"),
                    images: held,
                    compared,
                });
            }
            Ok(launches)
        },
    ))
}

pub fn family(entries: &[Entry]) -> Vec<Required<'static>> {
    entries
        .iter()
        .filter(|e| {
            matches!(
                e.query().phase,
                RecurrentPhase::Decode | RecurrentPhase::Verify { max_rows: 6 }
            )
        })
        .map(|entry| {
            let query = entry.query();
            let required = Required::new(
                format!(
                    "grouped F32 pooling {:?} width {}",
                    query.phase, query.width
                ),
                Need::rows(CAPACITY),
                move |backend, need| backend.gated_pooling(&query, need),
                move |prepared| case(query, prepared),
            )
            .required();
            if matches!(query.phase, RecurrentPhase::Verify { .. }) {
                required.with_decode(move |backend| {
                    let decode = GatedPoolingQuery {
                        phase: RecurrentPhase::Decode,
                        ..query
                    };
                    let prepared = backend.gated_pooling(&decode, Need::rows(CAPACITY))?;
                    Ok(decoded(decode, prepared))
                })
            } else {
                required
            }
        })
        .collect()
}

/// Prefill across every split of the drawn rows, including a split inside
/// a group or beside a hole. The first launch's raw state is downloaded and
/// restored as bytes for the next launch, as a snapshot would preserve it.
pub fn prefill_splits(
    backend: &dyn kernel_core::backend::Backend,
    modules: &dyn crate::kit::Modules,
    session: &crate::Session,
    entries: &[Entry],
) -> LabResult<serde_json::Value> {
    let functions = modules.load(session)?;
    let mut checked = 0;
    let mut prepared_count = 0;
    for entry in entries
        .iter()
        .filter(|e| e.query().phase == RecurrentPhase::Prefill)
    {
        let query = entry.query();
        let op = backend
            .gated_pooling(&query, Need::rows(CAPACITY))
            .map_err(|e| e.to_string())?;
        let case = case(query, op)?;
        case.validate()?;
        prepared_count += 1;
        let images = case.images();
        let positions = images.i32s("op.positions")?;
        let whole = case.launch(session, functions.as_ref(), &images, CAPACITY)?;
        let reference = oracle(query, &images, CAPACITY)?;
        let width = query.width as usize;
        for (path, expected) in reference {
            if let Expected::Values(reference) = expected {
                let actual = &whole[&path];
                for (row, &position) in positions.iter().enumerate().take(CAPACITY as usize) {
                    if position < 0 || position % 2 == 0 {
                        continue;
                    }
                    let range = row * width..(row + 1) * width;
                    let actual: Vec<_> = actual[range.start * 4..range.end * 4]
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .collect();
                    let reference = OracleOutput {
                        values: reference.values[range.clone()].to_vec(),
                        sum_abs_products: reference.sum_abs_products[range.clone()].to_vec(),
                        reduction_lengths: reference.reduction_lengths[range].to_vec(),
                    };
                    kernel_core::oracle::check_numeric(&actual, &reference, DType::F32)
                        .map_err(|e| format!("prefill f64 bound: {e:?}"))?;
                }
            } else if let Expected::Bits(expected) = expected {
                if path == "pooled" {
                    for (row, &position) in positions.iter().enumerate().take(CAPACITY as usize) {
                        if position < 0 || position % 2 == 0 {
                            continue;
                        }
                        let range = row * width * 4..(row + 1) * width * 4;
                        if whole[&path][range.clone()] != expected[range] {
                            return Err("prefill pool is not its exact f64 reference".into());
                        }
                    }
                } else if whole[&path] != expected {
                    return Err(format!("prefill exact {path} differs"));
                }
            }
        }
        for split in 1..CAPACITY as usize {
            let first = case.select(&images, &(0..split).collect::<Vec<_>>())?;
            let first = case.launch(session, functions.as_ref(), &first, split as u32)?;
            let next = case
                .select(&images, &(split..CAPACITY as usize).collect::<Vec<_>>())?
                .with("op.tail.state", first["op.tail.state"].clone());
            let next = case.launch(session, functions.as_ref(), &next, CAPACITY - split as u32)?;
            for (row, &position) in positions.iter().enumerate().take(CAPACITY as usize) {
                let (result, local) = if row < split {
                    (&first, row)
                } else {
                    (&next, row - split)
                };
                for (path, bytes) in [
                    ("op.output_positions", 4),
                    ("pooled", width * 4),
                    ("op.output", width * 4),
                ] {
                    if path != "op.output_positions" && (position < 0 || position % 2 == 0) {
                        continue;
                    }
                    if whole[path][row * bytes..(row + 1) * bytes]
                        != result[path][local * bytes..(local + 1) * bytes]
                    {
                        return Err(format!(
                            "prefill width {width}, split {split}, row {row}: {path} differs"
                        ));
                    }
                }
            }
            for range in state_ranges(query, &images, CAPACITY)? {
                if whole["op.tail.state"][range.clone()] != next["op.tail.state"][range] {
                    return Err(format!(
                        "prefill width {width}, split {split}: live F32 tail differs"
                    ));
                }
            }
            checked += 1;
        }
    }
    if prepared_count == 0 || checked == 0 {
        return Err("no prefill pooling entries were exercised".into());
    }
    Ok(
        serde_json::json!({"prepared": prepared_count, "splits": checked, "f64_pool": true, "raw_tail_restore": true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;

    fn verify_case() -> (GatedPoolingQuery, OpCase<'static>) {
        let query = GatedPoolingQuery {
            group: 2,
            width: 128,
            norm_eps_bits: 1e-20f32.to_bits(),
            phase: RecurrentPhase::Verify { max_rows: 6 },
            capacity: CAPACITY,
        };
        let entry = Entry::new(query, "chop_item_batch_test").unwrap();
        let backend = Sm120::new().with_gated_pooling([entry]);
        let prepared = backend.gated_pooling(&query, Need::rows(CAPACITY)).unwrap();
        (query, case(query, prepared).unwrap())
    }

    #[test]
    fn poison_keeps_incoming_tails_that_the_launch_reads_before_replacing() {
        for phase in [RecurrentPhase::Decode, RecurrentPhase::Prefill] {
            let query = GatedPoolingQuery {
                phase,
                ..verify_case().0
            };
            let backend =
                Sm120::new()
                    .with_gated_pooling([Entry::new(query, "chop_tail_update_test").unwrap()]);
            let prepared = backend.gated_pooling(&query, Need::rows(CAPACITY)).unwrap();
            let fixture = case(query, prepared).unwrap();
            fixture.validate().unwrap();
            let images = fixture.case.images();
            let stride = 2 * query.width as usize * 4;
            for live in [2, 3, 5, 6] {
                let poisoned = fixture.case.poison_state_maps(&images, live).unwrap();
                let state = poisoned.bytes("op.tail.state").unwrap();
                let before = images.bytes("op.tail.state").unwrap();
                let incoming = SOURCES[0] as usize * stride;
                assert!(
                    state[incoming..incoming + stride] == before[incoming..incoming + stride],
                    "{phase:?} live {live}: incoming tail was poisoned before its read"
                );
                if live == 5 {
                    let target = SOURCES[1] as usize * stride;
                    assert!(
                        state[target..target + stride].iter().all(|&b| b == 0xff),
                        "a tail overwritten without a read must still be poisoned"
                    );
                }
                assert_eq!(
                    fixture
                        .case
                        .state_writes("op.tail.state", &images, live)
                        .unwrap(),
                    state_ranges(query, &images, live).unwrap()
                );
            }
        }
    }

    #[test]
    fn item_batch_mapping_rejects_unwritten_or_changed_comparison_bytes() {
        let (_, valid) = verify_case();
        valid.validate().unwrap();

        let (query, unwritten) = verify_case();
        let unwritten = unwritten.batch_images(move |images, live| {
            let mut probes = batch_images(query, images, live)?;
            // Item zero's third row is a hole and writes no latent.
            probes[0].compared = vec![("op.output".into(), 2 * 128 * 4..3 * 128 * 4)];
            Ok(probes)
        });
        assert!(
            unwritten
                .validate()
                .unwrap_err()
                .contains("compares unwritten")
        );

        let (query, changed) = verify_case();
        let changed = changed.batch_images(move |images, live| {
            let mut probes = batch_images(query, images, live)?;
            // This is the modified neighbor's output, not item zero's.
            probes[0].compared = vec![("pooled".into(), 5 * 128 * 4..6 * 128 * 4)];
            Ok(probes)
        });
        assert!(
            changed
                .validate()
                .unwrap_err()
                .contains("oracle changes compared")
        );
    }
}
