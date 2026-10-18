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
fn rotated_state<'a>(query: LatentQueryRope, capacity: u32) -> Spec<'a> {
    let (heads, width, tail) = (
        query.heads as usize,
        query.latent as usize,
        query.rotary as usize,
    );
    Spec::state_updating(
        DType::F32,
        (Rows::Live, heads * width),
        f32_bytes(&latent_values(capacity, heads * width)),
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
    if cos_sin.len() != capacity as usize * query.rotary as usize {
        return Err("query fixture table shape mismatch".into());
    }
    let width = query.heads as usize * query.latent as usize;
    let tail = query.rotary as usize;
    let roles = LatentQueryRopeBuffers {
        query: rotated_state(query, capacity),
        native_q: Spec::output(DType::Bf16, (Rows::Live, width)),
        cos_sin: Spec::input(DType::F32, (Rows::Live, tail), f32_bytes(&cos_sin)),
        live_rows: Spec::live(),
    };
    Ok(
        OpCase::new(prepared, capacity, roles)?.oracle(move |images, live| {
            let mut whole = images.f32s("query")?;
            let (rotated, mut native) = oracle::query_rope_f32(
                query,
                &whole[..live as usize * width],
                &images.f32s("cos_sin")?[..live as usize * tail],
            )
            .map_err(|e| e.to_string())?;
            whole[..rotated.len()].copy_from_slice(&rotated);
            native.resize(capacity as usize * width, 0);
            Ok(vec![
                ("query".into(), Expected::Bits(f32_bytes(&whole))),
                (
                    "native_q".into(),
                    Expected::Bits(native.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ),
            ])
        }),
    )
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
    let staged = matches!(query.addressing, WindowAddressing::Staged { .. });
    let long = query.addressing.slots() >= query.window;
    let firsts = if long { vec![128, 257] } else { vec![0, 0] };
    let batches = (0..capacity).map(|r| (r % 2) as i32).collect::<Vec<_>>();
    let positions = (0..capacity)
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
    let roles = LatentWindowKeyBuffers {
        positions: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&positions)),
        row_batch: staged.then(|| Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&batches))),
        first_positions: staged
            .then(|| Spec::input(DType::I32, (Rows::Fixed(1), 2), i32_bytes(&firsts))),
        sequences: 2,
        key_lists: Spec::output(DType::I32, (Rows::Live, query.window as usize))
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
    let mut ranges = Vec::new();
    for row in 0..live as usize {
        let (p, b) = (positions[row], batches[row] as usize);
        if matches!(query.addressing, WindowAddressing::Circular { slots } if i64::from(p) + i64::from(slots) <= i64::from(last[b]))
        {
            continue;
        }
        let slot = query
            .addressing
            .slot(p, first[b])
            .map_err(|e| e.to_string())? as usize;
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
    let batches = (0..capacity)
        .map(|row| (row % 2) as i32)
        .collect::<Vec<_>>();
    let positions = (0..capacity)
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
    let (code_stride, scale_stride) = (slots * width + 66, slots * width / 32 + 7);
    let mut input = latent_values(capacity, width);
    // A tiny row exposes the norm epsilon; a zero row checks the quantizer's
    // scale convention on zero blocks. Other rows vary token, channel and magnitude.
    input[..width].fill(1e-11);
    if capacity > 1 {
        input[width..2 * width].fill(0.0);
    }

    let roles = LatentWindowWriteBuffers {
        input: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            input
                .iter()
                .flat_map(|&v| kernel_core::codec::f32_to_bf16(v).to_le_bytes())
                .collect(),
        ),
        weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), width),
            (0..width)
                .flat_map(|i| {
                    kernel_core::codec::f32_to_bf16(0.7 + (i % 19) as f32 / 23.0).to_le_bytes()
                })
                .collect(),
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
    let mut case = OpCase::new(prepared, capacity, roles)?
        .oracle(move |images, live| {
            let (mut latent, encoded) = oracle::window_rows(
                query,
                &images.u16s("input")?[..live as usize * width],
                &images.u16s("weight")?,
                &images.f32s("cos_sin")?[..live as usize * tail],
            )
            .map_err(|e| e.to_string())?;
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
            oracle::store_window_rows(
                query,
                &encoded,
                &i32s(images, "positions")?[..live as usize],
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
                ("window.codes".into(), Expected::Bits(codes)),
                ("window.scales".into(), Expected::Bits(scales)),
                ("quantized_codes".into(), Expected::Bits(qcodes)),
                ("quantized_scales".into(), Expected::Bits(qscales)),
            ])
        })
        .omission_guard(&quantizer, "quantized_codes");
    if let WindowAddressing::Circular { slots } = query.addressing {
        for live in crate::poison::live_counts(capacity)
            .into_iter()
            .filter(|&n| n > 0)
        {
            if positions[..live as usize]
                .iter()
                .zip(&batches)
                .all(|(&p, &b)| i64::from(p) + i64::from(slots) <= i64::from(lasts[b as usize]))
            {
                for plane in ["window.codes", "window.scales"] {
                    case = case.no_write(
                        plane,
                        live,
                        "all live rows precede the retained ring tail selected by last_positions",
                    );
                }
            }
        }
    }
    Ok(case)
}
