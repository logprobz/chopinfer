//! Native query, window-write and key-list gates over poisoned replay.
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, DType, Refusal};
use kernel_core::latent_attention::{LatentWindowBuffers, LatentWindowStorage};
use kernel_core::latent_glue::{
    LatentQueryRope, LatentQueryRopeBuffers, LatentWindowKeyBuffers, LatentWindowKeys,
    LatentWindowWrite, LatentWindowWriteBuffers, WindowAddressing,
};
use kernel_core::latent_glue_oracle as oracle;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32s(images: &Images, path: &str) -> LabResult<Vec<i32>> {
    Ok(images
        .bytes(path)?
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn table(capacity: u32, tail: usize) -> Vec<f32> {
    (0..capacity as usize)
        .flat_map(|row| {
            (0..tail / 2).flat_map(move |pair| {
                let angle = ((row * 17 + pair) as f64 + 0.3) * 0.271;
                [angle.cos() as f32, angle.sin() as f32]
            })
        })
        .collect()
}
fn latent_values(capacity: u32, width: usize) -> Vec<f32> {
    (0..capacity as usize * width)
        .map(|i| {
            let block = i / 32;
            let maximum = [0.0, 1e-6, 448.03125, 449.0, 896.1, 0.5, 1.0039][block % 7];
            if i % 32 == 0 {
                maximum
            } else {
                maximum * ((i % 19) as f32 - 9.0) / 23.0
            }
        })
        .collect()
}
fn rotated_state<'a>(query: LatentQueryRope, values: Vec<f32>) -> Spec<'a> {
    let (heads, width, tail) = (
        query.heads as usize,
        query.latent as usize,
        query.rotary as usize,
    );
    Spec::state_updating(
        DType::F32,
        (Rows::Live, heads * width),
        f32_bytes(&values),
        Write::Map(Box::new(|_, _| WriteMap::of(DType::F32).none())),
        move |_, live| {
            Ok((0..live as usize * heads)
                .map(|head| (head * width + width - tail) * 4..(head + 1) * width * 4)
                .collect())
        },
    )
}

pub fn query_rope<'a>(query: LatentQueryRope, capacity: u32) -> Required<'a> {
    Required::new(
        format!("latent query F32 RoPE heads{}", query.heads),
        Need::rows(capacity),
        move |backend, need| backend.latent_query_rope(&query, need),
        move |prepared| query_case(query, capacity, prepared),
    )
    .required()
}

pub fn query_case<'a>(
    query: LatentQueryRope,
    capacity: u32,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    query_case_with_table(
        query,
        capacity,
        prepared,
        table(capacity, query.rotary as usize),
    )
}

/// The model supplies its pinned, row-packed F32 cosine/sine table.
pub fn query_case_with_table<'a>(
    query: LatentQueryRope,
    capacity: u32,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
    cos_sin: Vec<f32>,
) -> LabResult<OpCase<'a>> {
    query_case_with_values(
        query,
        capacity,
        prepared,
        cos_sin,
        latent_values(capacity, query.heads as usize * query.latent as usize),
    )
}

fn query_case_with_values<'a>(
    query: LatentQueryRope,
    capacity: u32,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
    cos_sin: Vec<f32>,
    values: Vec<f32>,
) -> LabResult<OpCase<'a>> {
    query_case_with_status(
        query,
        capacity,
        prepared,
        cos_sin,
        values,
        vec![0; capacity as usize],
    )
}

fn query_case_with_status<'a>(
    query: LatentQueryRope,
    capacity: u32,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
    cos_sin: Vec<f32>,
    values: Vec<f32>,
    initial: Vec<u32>,
) -> LabResult<OpCase<'a>> {
    query_case_status_offset(query, capacity, prepared, cos_sin, values, initial, 0)
}

fn query_case_status_offset<'a>(
    query: LatentQueryRope,
    capacity: u32,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
    cos_sin: Vec<f32>,
    values: Vec<f32>,
    initial: Vec<u32>,
    offset: usize,
) -> LabResult<OpCase<'a>> {
    if cos_sin.len() != capacity as usize * query.rotary as usize {
        return Err("query fixture table shape mismatch".into());
    }
    let width = query.heads as usize * query.latent as usize;
    let tail = query.rotary as usize;
    let roles = LatentQueryRopeBuffers {
        status: if offset == 0 {
            super::block_quantize::status_role(&initial)
        } else {
            Spec::state_updating(
                DType::I32,
                (Rows::Fixed(1), initial.len()),
                initial.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Write::Map(Box::new(|_, _| WriteMap::of(DType::I32).none())),
                move |_, live| {
                    Ok((offset..offset + live as usize)
                        .map(|r| 4 * r..4 * r + 4)
                        .collect())
                },
            )
        },
        status_row_offset: offset as u64,
        query: rotated_state(query, values),
        native_q: Spec::output(DType::Bf16, (Rows::Live, width)),
        cos_sin: Spec::input(DType::F32, (Rows::Live, tail), f32_bytes(&cos_sin)),
        live_rows: Spec::live(),
    };
    Ok(
        OpCase::new(prepared, capacity, roles)?.oracle(move |images, live| {
            let mut whole = images.f32s("query")?;
            let mut status = i32s(images, "status")?
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>();
            let (rotated, mut native, flags) = oracle::query_rope_status(
                query,
                &whole[..live as usize * width],
                &images.f32s("cos_sin")?[..live as usize * tail],
                &status[offset..offset + live as usize],
            )
            .map_err(|e| e.to_string())?;
            status[offset..offset + live as usize].copy_from_slice(&flags);
            whole[..rotated.len()].copy_from_slice(&rotated);
            native.resize(capacity as usize * width, 0);
            Ok(vec![
                ("query".into(), Expected::Bits(f32_bytes(&whole))),
                (
                    "status".into(),
                    Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ),
                (
                    "native_q".into(),
                    Expected::Bits(native.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ),
            ])
        }),
    )
}

/// Separate offset replay uses fixed guard words, outside the row-property suite.
pub fn query_status_offset_case<'a>(
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let query = LatentQueryRope {
        heads: 3,
        latent: 512,
        rotary: 64,
    };
    let mut values = latent_values(8, 3 * 512);
    values[2 * 512 + 17] = f32::NAN;
    let mut initial = vec![0x5a5a5a5a; 13];
    initial[3..11].fill(0);
    initial[5] = 0x80;
    query_case_status_offset(query, 8, prepared, table(8, 64), values, initial, 3)
}

pub fn query_status_case<'a>(
    query: LatentQueryRope,
    prepared: Prepared<LatentQueryRopeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let width = query.heads as usize * 512;
    let mut values = latent_values(8, width);
    values[width - 1] = f32::NAN;
    values[width + 17] = f32::INFINITY;
    values[2 * width + 5] = f32::MAX;
    let mut angles = table(8, 64);
    angles[6 * 64] = f32::NAN;
    query_case_with_status(
        query,
        8,
        prepared,
        angles,
        values,
        vec![0, 0, 0, 0x80, 2, 0, 0, 0],
    )
}

pub fn query_status<'a>(heads: u32) -> Required<'a> {
    let query = LatentQueryRope {
        heads,
        latent: 512,
        rotary: 64,
    };
    Required::new(
        format!("query status heads{heads}"),
        Need::rows(8),
        move |b, n| b.latent_query_rope(&query, n),
        move |op| query_status_case(query, op),
    )
    .required()
}

fn pairing_values(query: LatentQueryRope) -> Vec<f32> {
    let mut values = vec![0.0; query.heads as usize * query.latent as usize];
    for head in values.chunks_exact_mut(query.latent as usize) {
        head[448] = 1.0;
        head[480] = 2.0;
    }
    values
}

/// Exact quarter-turn inputs separate adjacent pairs from split-half pairs.
pub fn query_pairing_regression<'a>() -> Required<'a> {
    let query = LatentQueryRope {
        heads: 3,
        latent: 512,
        rotary: 64,
    };
    Required::new(
        "query adjacent-pair regression",
        Need::rows(2),
        move |b, n| b.latent_query_rope(&query, n),
        move |op| {
            query_case_with_values(
                query,
                2,
                op,
                [0.0, 1.0].repeat(64),
                pairing_values(query).repeat(2),
            )
        },
    )
    .required()
}

#[cfg(test)]
mod pairing_tests {
    use super::*;
    #[test]
    fn query_replay_moves_state_with_tables_and_preserves_signed_zero_bits() {
        use crate::kit::{Outcome, properties};
        use kernel_core::backend::Backend;
        use kernel_core::properties::{Invariance, Property, Scope};
        use kernels_sm120::backend::Sm120;
        use kernels_sm120::latent::glue::{Entry, Kernel};
        for heads in [1, 3, 64] {
            let query = LatentQueryRope {
                heads,
                latent: 512,
                rotary: 64,
            };
            let backend = Sm120::new().with_latent_glue(Entry {
                kernel: Kernel::QueryRope,
                capacity: 17,
                symbol: "query".into(),
            });
            let case = query_case(
                query,
                17,
                backend.latent_query_rope(&query, Need::rows(17)).unwrap(),
            )
            .unwrap();
            let images = case.images();
            let claims = Invariance::any_rows(17);
            let run = |images: &Images, live| case.oracle_bits(images, live);
            let written = |name: &str, images: &Images, live| case.state_ranges(name, images, live);
            let checks = properties::check(
                claims,
                &claims.claims(),
                &case.leaves,
                &images,
                17,
                &run,
                &written,
            );
            assert_eq!(checks.len(), 3);
            assert!(
                checks
                    .iter()
                    .all(|c| matches!(c.outcome, Outcome::Passed(_))),
                "heads {heads}: {checks:?}"
            );
            let selected = case.select(&images, &[16, 1, 16]).unwrap();
            let row_bytes = heads as usize * 512 * 4;
            assert_eq!(
                &selected.bytes("query").unwrap()[..row_bytes],
                &images.bytes("query").unwrap()[16 * row_bytes..17 * row_bytes]
            );
            let roles = case.row_roles(Scope::Entry).unwrap();
            assert!(roles.inputs.iter().any(|(name, _)| name == "query"));
            assert!(roles.outputs.iter().any(|(name, _)| name == "query"));
            let wrong_sign = |images: &Images, live| {
                let mut result = run(images, live)?;
                if live > 1 {
                    result.get_mut("query").unwrap()[448 * 4 + 3] ^= 0x80;
                }
                Ok(result)
            };
            let checks = properties::check(
                claims,
                &[Property::Rows, Property::Chunk],
                &case.leaves,
                &images,
                17,
                &wrong_sign,
                &written,
            );
            assert!(
                checks
                    .iter()
                    .all(|c| matches!(c.outcome, Outcome::Failed(_))),
                "signed-zero mutation escaped: {checks:?}"
            );
        }
    }

    #[test]
    fn quarter_turn_distinguishes_adjacent_and_split_half_pairs() {
        let query = LatentQueryRope {
            heads: 3,
            latent: 512,
            rotary: 64,
        };
        let values = pairing_values(query);
        let (rotated, _) = oracle::query_rope_f32(query, &values, &[0.0, 1.0].repeat(32)).unwrap();
        for head in 0..query.heads as usize {
            let at = head * 512;
            assert_eq!(rotated[at + 448], 0.0);
            assert_eq!(rotated[at + 449], 1.0);
            assert_eq!(rotated[at + 480], 0.0);
            assert_eq!(rotated[at + 481], 2.0);
            let split_half = values[at + 448] * 0.0 - values[at + 480] * 1.0;
            assert_eq!(split_half, -2.0);
            assert_ne!(rotated[at + 448], split_half);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum KeyMutation {
    Outside,
    Stale,
}
struct KeyControl {
    inner: Prepared<LatentWindowKeyBuffers<View>>,
    mutation: KeyMutation,
}
impl PreparedOp<LatentWindowKeyBuffers<View>> for KeyControl {
    fn info(&self) -> &OpInfo {
        self.inner.info()
    }
    fn launches(
        &self,
        b: &LatentWindowKeyBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        for launch in &mut launches {
            let Some(Argument::U32(slots)) = launch.arguments.0.get_mut(7) else {
                return Err(Refusal("window key control lacks its slot argument"));
            };
            *slots = match self.mutation {
                KeyMutation::Outside => *slots + 1,
                KeyMutation::Stale => *slots - 1,
            };
        }
        Ok(launches)
    }
}

pub fn window_keys<'a>(query: LatentWindowKeys, capacity: u32) -> Required<'a> {
    key_required(query, capacity, None)
}
pub fn window_key_control<'a>(
    query: LatentWindowKeys,
    capacity: u32,
    mutation: KeyMutation,
) -> Required<'a> {
    key_required(query, capacity, Some(mutation))
}
fn key_required<'a>(
    query: LatentWindowKeys,
    capacity: u32,
    mutation: Option<KeyMutation>,
) -> Required<'a> {
    Required::new(format!("latent window keys {:?} {mutation:?}", query.addressing), Need::rows(capacity), move |backend, need| {
        let inner = backend.latent_window_keys(&query, need)?;
        if let Some(mutation) = mutation {
            if !matches!(query.addressing, WindowAddressing::Circular { slots } if slots > query.window && slots < i32::MAX as u32) {
                return Err(Refused::Rule("key controls require a circular window with headroom".into()));
            }
            Ok(Box::new(KeyControl { inner, mutation }) as Prepared<LatentWindowKeyBuffers<View>>)
        } else { Ok(inner) }
    }, move |prepared| key_case(query, capacity, prepared)).required()
}

fn key_case<'a>(
    query: LatentWindowKeys,
    capacity: u32,
    prepared: Prepared<LatentWindowKeyBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    key_case_data(query, capacity, prepared, false)
}
fn key_case_data<'a>(
    query: LatentWindowKeys,
    capacity: u32,
    prepared: Prepared<LatentWindowKeyBuffers<View>>,
    holes: bool,
) -> LabResult<OpCase<'a>> {
    let staged = matches!(query.addressing, WindowAddressing::Staged { .. });
    let long = query.addressing.slots() >= query.window;
    let mut firsts = if long { vec![128, 257] } else { vec![0, 0] };
    let mut batches = (0..capacity).map(|r| (r % 2) as i32).collect::<Vec<_>>();
    let mut positions = (0..capacity)
        .map(|row| {
            if !staged {
                [0, 1, 127, 128, 135, 136, 257, i32::MAX][row as usize % 8]
            } else if long {
                firsts[row as usize % 2] + query.window as i32 - 1
                    + (row / 2 % (query.addressing.slots() - query.window + 1)) as i32
            } else {
                (row % query.addressing.slots()) as i32
            }
        })
        .collect::<Vec<_>>();
    if holes {
        positions[2] = -2;
        if staged {
            batches[3] = 99;
            batches[4] = -2;
            firsts[1] = -2;
        }
    }
    let roles = LatentWindowKeyBuffers {
        positions: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&positions)),
        row_batch: staged.then(|| Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&batches))),
        first_positions: staged
            .then(|| Spec::input(DType::I32, (Rows::Fixed(1), 2), i32_bytes(&firsts))),
        sequences: 2,
        key_lists: Spec::output_with(
            DType::I32,
            (Rows::Live, query.window as usize),
            Write::Map(Box::new(move |images, live| {
                let batches = if staged {
                    i32s(images, "row_batch")?
                } else {
                    vec![0; capacity as usize]
                };
                WriteMap::of(DType::I32).byte_ranges(
                    (0..live as usize)
                        .filter(|&row| !staged || (0..2).contains(&batches[row]))
                        .map(|row| {
                            row * query.window as usize * 4..(row + 1) * query.window as usize * 4
                        })
                        .collect::<Vec<_>>(),
                )
            })),
        )
        .filled_with(&[0x80]),
        live_rows: Spec::live(),
    };
    Ok(
        OpCase::new(prepared, capacity, roles)?.oracle(move |images, live| {
            let positions = i32s(images, "positions")?;
            let (batch, first) = if staged {
                (i32s(images, "row_batch")?, i32s(images, "first_positions")?)
            } else {
                (vec![], vec![])
            };
            Ok(vec![(
                "key_lists".into(),
                Expected::Judge(Box::new(move |bytes| {
                    let slots = bytes
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    for row in 0..live as usize {
                        if staged && !(0..2).contains(&batch[row]) {
                            continue;
                        }
                        let first = if !staged || positions[row] < 0 {
                            0
                        } else {
                            first[batch[row] as usize]
                        };
                        oracle::check_window_keys(
                            query,
                            positions[row],
                            first,
                            slots
                                .get(row * query.window as usize..(row + 1) * query.window as usize)
                                .ok_or("short window key output")?,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })),
            )])
        }),
    )
}

fn checked_window(
    query: LatentWindowWrite,
    images: &Images,
    live: u32,
) -> LabResult<oracle::WindowStatusRows> {
    let prior = i32s(images, "status")?
        .iter()
        .map(|&v| v as u32)
        .collect::<Vec<_>>();
    oracle::window_rows_status(
        query,
        &images.f32s("input")?[..live as usize * query.latent as usize],
        &images.u16s("weight")?,
        &images.f32s("cos_sin")?[..live as usize * query.rotary as usize],
        &prior[..live as usize],
    )
    .map_err(|e| e.to_string())
}

fn plane_map(
    query: LatentWindowWrite,
    images: &Images,
    live: u32,
    stride: usize,
    width: usize,
) -> LabResult<WriteMap> {
    let positions = i32s(images, "positions")?;
    let batches = i32s(images, "row_batch")?;
    let last = i32s(images, "last_positions")?;
    let first = if matches!(query.addressing, WindowAddressing::Staged { .. }) {
        i32s(images, "first_positions")?
    } else {
        vec![0; last.len()]
    };
    let (_, _, flags) = checked_window(query, images, live)?;
    let mut ranges = Vec::new();
    for row in 0..live as usize {
        let (p, b) = (positions[row], batches[row] as usize);
        if flags[row] != 0 || b >= last.len() || p < 0 || p > last[b] {
            continue;
        }
        if matches!(query.addressing, WindowAddressing::Circular { slots } if i64::from(p) + i64::from(slots) <= i64::from(last[b]))
        {
            continue;
        }
        let Ok(slot) = query.addressing.slot(p, first[b]) else {
            continue;
        };
        let slot = slot as usize;
        let begin = b * stride + slot * width;
        ranges.push(begin..begin + width);
    }
    ranges.sort_by_key(|r| r.start);
    WriteMap::of(DType::U8).byte_ranges(ranges)
}

pub fn window_write<'a>(query: LatentWindowWrite, capacity: u32, first: i32) -> Required<'a> {
    Required::new(
        format!("latent window write {:?} rows{capacity}", query.addressing),
        Need::rows(capacity),
        move |backend, need| backend.latent_window_write(&query, need),
        move |prepared| write_case(query, capacity, first, prepared),
    )
    .required()
}

pub fn write_case<'a>(
    query: LatentWindowWrite,
    capacity: u32,
    first: i32,
    prepared: Prepared<LatentWindowWriteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    write_case_with_table(
        query,
        capacity,
        first,
        prepared,
        table(capacity, query.rotary as usize),
    )
}

/// The model supplies the F32 table rows for the fixture's absolute token
/// positions, as the native window writer consumes them.
pub fn write_case_with_table<'a>(
    query: LatentWindowWrite,
    capacity: u32,
    first: i32,
    prepared: Prepared<LatentWindowWriteBuffers<View>>,
    cos_sin: Vec<f32>,
) -> LabResult<OpCase<'a>> {
    write_case_impl(
        query,
        capacity,
        first,
        prepared,
        cos_sin,
        WindowFixture::Varying,
    )
}

#[derive(Clone, Copy)]
enum WindowFixture {
    Varying,
    F32Boundary,
    InvalidData,
    Status,
}
fn write_case_impl<'a>(
    query: LatentWindowWrite,
    capacity: u32,
    first: i32,
    prepared: Prepared<LatentWindowWriteBuffers<View>>,
    mut cos_sin: Vec<f32>,
    fixture: WindowFixture,
) -> LabResult<OpCase<'a>> {
    if cos_sin.len() != capacity as usize * query.rotary as usize {
        return Err("window fixture table shape mismatch".into());
    }
    let (width, tail, slots) = (
        query.latent as usize,
        query.rotary as usize,
        query.addressing.slots() as usize,
    );
    let staged = matches!(query.addressing, WindowAddressing::Staged { .. });
    let starts = [first, first + if staged { 257 } else { 0 }];
    let mut batches = (0..capacity)
        .map(|row| (row % 2) as i32)
        .collect::<Vec<_>>();
    let mut positions = (0..capacity)
        .map(|row| starts[row as usize % 2] + (row / 2) as i32)
        .collect::<Vec<_>>();
    let lasts = (0..2)
        .map(|b| {
            positions
                .iter()
                .zip(&batches)
                .filter_map(|(&p, &batch)| (batch == b).then_some(p))
                .max()
                .unwrap_or(starts[b as usize])
        })
        .collect::<Vec<_>>();
    // Put a retained row first so every nonempty live prefix exercises
    // the store. Keep its table tied to the same absolute position. The
    // displaced stale row remains in the full batch to test non-aliasing.
    if !staged && i64::from(positions[0]) + slots as i64 <= i64::from(lasts[0]) {
        let retained = positions
            .iter()
            .zip(&batches)
            .position(|(&position, &batch)| batch == 0 && position == lasts[0])
            .expect("the first sequence has a final row");
        positions.swap(0, retained);
        for channel in 0..tail {
            cos_sin.swap(channel, retained * tail + channel);
        }
    }
    let (code_stride, scale_stride) = (slots * width + 66, slots * width / 32 + 7);
    let mut input = latent_values(capacity, width);
    // A tiny row exposes the norm epsilon; a zero row checks the quantizer's
    // scale convention on zero blocks. Other rows vary token, channel and magnitude.
    input[..width].fill(1e-11);
    if capacity > 1 {
        input[width..2 * width].fill(0.0);
    }

    let mut weight = (0..width)
        .map(|i| kernel_core::codec::f32_to_bf16(0.7 + (i % 19) as f32 / 23.0))
        .collect::<Vec<_>>();
    if matches!(fixture, WindowFixture::F32Boundary) {
        input.fill(1.0);
        input[0] = 1.00375;
        weight.fill(kernel_core::codec::f32_to_bf16(1.0));
        weight[0] = kernel_core::codec::f32_to_bf16(1.0625);
    }
    if matches!(fixture, WindowFixture::InvalidData) {
        positions[2] = -2;
        positions[3] = lasts[1] + 1;
        batches[4] = 99;
        batches[5] = -2;
        positions[6] = starts[0] - 1;
        positions[7] = starts[1] + slots as i32;
    }
    let mut initial = vec![0u32; capacity as usize];
    if matches!(fixture, WindowFixture::Status) {
        input[width] = f32::NAN;
        input[2 * width] = f32::NAN;
        positions[2] = -1;
        input[3 * width] = f32::INFINITY;
        initial[4] = 0x80;
        initial[5] = 2;
    }
    let roles = LatentWindowWriteBuffers {
        status: super::block_quantize::status_role(&initial),
        status_row_offset: 0,
        input: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&input)),
        weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), width),
            weight.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        latent: Spec::output(DType::F32, (Rows::Live, width)),
        cos_sin: Spec::input(DType::F32, (Rows::Live, tail), f32_bytes(&cos_sin)),
        positions: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&positions)),
        row_batch: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&batches)),
        first_positions: staged
            .then(|| Spec::input(DType::I32, (Rows::Fixed(1), 2), i32_bytes(&starts))),
        last_positions: Spec::input(DType::I32, (Rows::Fixed(1), 2), i32_bytes(&lasts)),
        window: LatentWindowBuffers {
            codes: Spec::state(
                DType::U8,
                (Rows::Fixed(1), 2 * code_stride),
                vec![0xa5; 2 * code_stride],
                Write::Map(Box::new(move |images, live| {
                    plane_map(query, images, live, code_stride, width)
                })),
            ),
            scales: Spec::state(
                DType::U8,
                (Rows::Fixed(1), 2 * scale_stride),
                vec![0xff; 2 * scale_stride],
                Write::Map(Box::new(move |images, live| {
                    plane_map(query, images, live, scale_stride, width / 32)
                })),
            ),
            // The writer addresses sequences by row; it reads neither the
            // attention's item slots nor its key lists.
            item_slots: Spec::untouched(4),
            items: 0,
            sequences: 2,
            code_stride: code_stride as u64,
            scale_stride: scale_stride as u64,
            storage: if slots == 128 {
                LatentWindowStorage::Ring128
            } else {
                LatentWindowStorage::Staged {
                    slots: slots as u32,
                }
            },
            key_lists: Spec::untouched(4),
            key_stride: 1,
            budget: 0,
        },
        quantized_codes: Spec::output(DType::U8, (Rows::Live, width)),
        quantized_scales: Spec::output(DType::U8, (Rows::Live, width / 32)),
        live_rows: Spec::live(),
    };
    let quantizer = prepared
        .info()
        .symbols
        .get(1)
        .ok_or("window write has no shared quantizer stage")?
        .0
        .clone();
    let producer = prepared
        .info()
        .symbols
        .first()
        .ok_or("window write has no normalization stage")?
        .0
        .clone();
    let case = OpCase::new(prepared, capacity, roles)?
        .oracle(move |images, live| {
            let (mut latent, encoded, flags) = checked_window(query, images, live)?;
            let mut status = i32s(images, "status")?
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>();
            status[..live as usize].copy_from_slice(&flags);
            if live > 0 && matches!(fixture, WindowFixture::F32Boundary) {
                assert_eq!(
                    (encoded[0].values[0], encoded[0].scale_bytes[0]),
                    (0x79, 119)
                );
            }
            latent.resize(capacity as usize * width, 0.0);
            let mut codes = images.bytes("window.codes")?.to_vec();
            let mut scales = images.bytes("window.scales")?.to_vec();
            let first = if staged {
                i32s(images, "first_positions")?
            } else {
                vec![0; 2]
            };
            let batch = i32s(images, "row_batch")?
                .iter()
                .map(|&b| b as u32)
                .collect::<Vec<_>>();
            let mut positions = i32s(images, "positions")?;
            for (position, flag) in positions.iter_mut().zip(&flags) {
                if *flag != 0 {
                    *position = -1;
                }
            }
            oracle::store_window_rows(
                query,
                &encoded,
                &positions[..live as usize],
                &batch[..live as usize],
                &first,
                &i32s(images, "last_positions")?,
                &mut codes,
                &mut scales,
                code_stride,
                scale_stride,
            )
            .map_err(|e| e.to_string())?;
            let mut qcodes = encoded
                .iter()
                .flat_map(|r| r.values.clone())
                .collect::<Vec<_>>();
            let mut qscales = encoded
                .iter()
                .flat_map(|r| r.scale_bytes.clone())
                .collect::<Vec<_>>();
            qcodes.resize(capacity as usize * width, 0);
            qscales.resize(capacity as usize * width / 32, 0);
            Ok(vec![
                ("latent".into(), Expected::Bits(f32_bytes(&latent))),
                (
                    "status".into(),
                    Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ),
                ("window.codes".into(), Expected::Bits(codes)),
                ("window.scales".into(), Expected::Bits(scales)),
                ("quantized_codes".into(), Expected::Bits(qcodes)),
                ("quantized_scales".into(), Expected::Bits(qscales)),
            ])
        })
        .omission_guard(&producer, "latent")
        .omission_guard(&quantizer, "quantized_codes");
    Ok(case)
}

pub fn window_status<'a>(query: LatentWindowWrite) -> Required<'a> {
    Required::new(
        format!("window status {:?}", query.addressing),
        Need::rows(8),
        move |b, n| b.latent_window_write(&query, n),
        move |op| write_case_impl(query, 8, 0, op, table(8, 64), WindowFixture::Status),
    )
    .required()
}

pub fn window_f32_regression<'a>(query: LatentWindowWrite) -> Required<'a> {
    Required::new(
        "F32 projection boundary",
        Need::rows(2),
        move |b, n| b.latent_window_write(&query, n),
        move |op| {
            write_case_impl(
                query,
                2,
                0,
                op,
                [1.0, 0.0].repeat(64),
                WindowFixture::F32Boundary,
            )
        },
    )
    .required()
}
pub fn window_invalid_data<'a>(query: LatentWindowWrite) -> Required<'a> {
    Required::new(
        "window invalid device metadata",
        Need::rows(8),
        move |b, n| b.latent_window_write(&query, n),
        move |op| write_case_impl(query, 8, 130, op, table(8, 64), WindowFixture::InvalidData),
    )
    .required()
}

struct Mutant<B> {
    inner: Prepared<B>,
    info: OpInfo,
    symbol: String,
}
impl<B> PreparedOp<B> for Mutant<B> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, b: &B, workspace: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        launches
            .first_mut()
            .ok_or(Refusal("mutant has no launch"))?
            .symbol
            .clone_from(&self.symbol);
        Ok(launches)
    }
}
fn mutant<B: 'static>(
    inner: Prepared<B>,
    mutation: kernels_sm120::latent::glue::Mutation,
) -> Prepared<B> {
    let symbol = mutation.symbol();
    let mut info = inner.info().clone();
    info.symbols[0].0.clone_from(&symbol);
    Box::new(Mutant {
        inner,
        info,
        symbol,
    })
}
pub fn device_mutant<'a>(mutation: kernels_sm120::latent::glue::Mutation) -> Required<'a> {
    use kernels_sm120::latent::glue::Mutation;
    let q = LatentQueryRope {
        heads: 3,
        latent: 512,
        rotary: 64,
    };
    let keys = LatentWindowKeys {
        window: 128,
        addressing: WindowAddressing::Circular { slots: 136 },
    };
    let write = LatentWindowWrite {
        latent: 512,
        rotary: 64,
        epsilon_bits: 1e-20_f32.to_bits(),
        addressing: keys.addressing,
    };
    match mutation {
        Mutation::DropQueryFlag => Required::new(
            "dropped query flag",
            Need::rows(8),
            move |b, n| b.latent_query_rope(&q, n).map(|op| mutant(op, mutation)),
            move |op| query_status_case(q, op),
        )
        .required(),
        Mutation::DropWindowFlag => Required::new(
            "dropped window flag",
            Need::rows(8),
            move |b, n| {
                b.latent_window_write(&write, n)
                    .map(|op| mutant(op, mutation))
            },
            move |op| write_case_impl(write, 8, 0, op, table(8, 64), WindowFixture::Status),
        )
        .required(),

        Mutation::SplitHalf => Required::new(
            format!("mutant {mutation:?}"),
            Need::rows(1),
            move |b, n| b.latent_query_rope(&q, n).map(|op| mutant(op, mutation)),
            move |op| query_case_with_values(q, 1, op, [0.0, 1.0].repeat(32), pairing_values(q)),
        )
        .required(),
        Mutation::WindowEdge => Required::new(
            format!("mutant {mutation:?}"),
            Need::rows(17),
            move |b, n| {
                b.latent_window_keys(&keys, n)
                    .map(|op| mutant(op, mutation))
            },
            move |op| key_case(keys, 17, op),
        )
        .required(),
        Mutation::InputRound => Required::new(
            format!("mutant {mutation:?}"),
            Need::rows(1),
            move |b, n| {
                b.latent_window_write(&write, n)
                    .map(|op| mutant(op, mutation))
            },
            move |op| {
                write_case_impl(
                    write,
                    1,
                    0,
                    op,
                    [1.0, 0.0].repeat(32),
                    WindowFixture::F32Boundary,
                )
            },
        )
        .required(),
        _ => Required::new(
            format!("mutant {mutation:?}"),
            Need::rows(17),
            move |b, n| b.latent_query_rope(&q, n).map(|op| mutant(op, mutation)),
            move |op| query_case(q, 17, op),
        )
        .required(),
    }
}

pub fn key_invalid_data<'a>(query: LatentWindowKeys) -> Required<'a> {
    Required::new(
        "window key device holes",
        Need::rows(8),
        move |b, n| b.latent_window_keys(&query, n),
        move |op| key_case_data(query, 8, op, true),
    )
    .required()
}
