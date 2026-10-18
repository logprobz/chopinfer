//! Poisoned bitwise window-prefix gathers, including retained-ring holes.
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::latent_attention::{LatentWindowBuffers, LatentWindowStorage};
use kernel_core::latent_window_stage::{
    LatentWindowStage, LatentWindowStageBuffers, PlaneImages, gather_bits,
};
use kernels_sm120::latent::window_stage::Mutation;

pub const QUERY: LatentWindowStage = LatentWindowStage {
    kv: kernel_core::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
    latent: 512,
    window: 128,
    ring_slots: 136,
    staged_slots: 256,
};
const RC: usize = 136 * 512 + 66;
const RS: usize = 136 * 16 + 23;
pub const SC: usize = 256 * 512 + 34;
pub const SS: usize = 256 * 16 + 11;
#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    Short,
    ExactWindow,
    Wrap,
    TwoSequences,
    SplitFirst,
    SplitSecond,
    Invalid,
}
impl Scenario {
    pub const ALL: [Self; 7] = [
        Self::Short,
        Self::ExactWindow,
        Self::Wrap,
        Self::TwoSequences,
        Self::SplitFirst,
        Self::SplitSecond,
        Self::Invalid,
    ];
    pub fn capacity(self) -> u32 {
        if matches!(self, Self::Invalid) { 8 } else { 2 }
    }
    fn metadata(self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let (first, last) = match self {
            Self::Short => ([3, 0], [-1, 999, 2]),
            Self::ExactWindow => ([128, 127], [126, 999, 127]),
            Self::Wrap => ([140, 137], [136, 999, 139]),
            Self::TwoSequences => ([260, 7], [6, 999, 259]),
            Self::SplitFirst => ([140, 7], [6, 999, 139]),
            Self::SplitSecond => ([172, 39], [38, 999, 171]),
            Self::Invalid => {
                return (
                    vec![140, 7, -2, 8, 8, 8, 8, 8],
                    vec![2, 0, 2, -2, 3, 1, 0, 2],
                    vec![200, -1, 139],
                );
            }
        };
        (first.to_vec(), vec![2, 0], last.to_vec())
    }
}
fn ints(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn read(images: &Images, name: &str) -> LabResult<Vec<i32>> {
    Ok(images
        .bytes(name)?
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
pub fn payload(position: i32, column: usize, scale: bool) -> u8 {
    if scale {
        // These scales keep every finite E4M3 code exact and finite in BF16.
        ((position as usize * 7 + column * 11) % 13 + 121) as u8
    } else {
        ((position as usize * 13 + column * 19) % 127) as u8
    }
}
fn ring(last: &[i32]) -> (Vec<u8>, Vec<u8>) {
    let (mut c, mut s) = (vec![0xff; last.len() * RC], vec![0xff; last.len() * RS]);
    for (slot, &end) in last.iter().enumerate() {
        for p in (end - 135).max(0)..=end {
            for col in 0..512 {
                c[slot * RC + p as usize % 136 * 512 + col] = payload(p, col, false);
            }
            for col in 0..16 {
                s[slot * RS + p as usize % 136 * 16 + col] = payload(p, col, true);
            }
        }
    }
    (c, s)
}
fn map(images: &Images, live: u32, stride: usize, width: usize) -> LabResult<WriteMap> {
    let first = read(images, "first_positions")?;
    let slots = read(images, "ring.item_slots")?;
    let last = read(images, "last_positions")?;
    let mut ranges = Vec::new();
    for item in 0..live as usize {
        let Ok(slot) = usize::try_from(slots[item]) else {
            continue;
        };
        if slot >= last.len() || first[item] < 0 || last[slot] < 0 {
            continue;
        }
        let begin = (i64::from(first[item]) - 127).max(0);
        let retained = (i64::from(last[slot]) - 135).max(0);
        for p in begin.max(retained)..i64::from(first[item]).min(i64::from(last[slot]) + 1) {
            let at = item * stride + (p - begin) as usize * width;
            ranges.push(at..at + width);
        }
    }
    WriteMap::of(DType::U8).byte_ranges(ranges)
}
pub fn case<'a>(
    scenario: Scenario,
    op: Prepared<LatentWindowStageBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    case_with_format(QUERY.kv, scenario, op)
}

pub fn case_with_format<'a>(
    kv: kernel_core::latent_attention::LatentKvFormat,
    scenario: Scenario,
    op: Prepared<LatentWindowStageBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (first, slots, last) = scenario.metadata();
    let capacity = scenario.capacity();
    let (mut c, mut s) = ring(&last);
    let bf16 = kv == kernel_core::latent_attention::LatentKvFormat::Bf16;
    let (rc, rs, sc, ss, width, scale_width) = if bf16 {
        let rc = 136 * 1024 + 66;
        let mut products = vec![0xff; last.len() * rc];
        for (sequence, &end) in last.iter().enumerate() {
            for p in (end - 135).max(0)..=end {
                let slot = p as usize % 136;
                let encoded = kernel_core::latent_attention_oracle::EncodedLatent {
                    values: c[sequence * RC + slot * 512..sequence * RC + (slot + 1) * 512]
                        .to_vec(),
                    scales: vec![],
                    scale_bytes: s[sequence * RS + slot * 16..sequence * RS + (slot + 1) * 16]
                        .to_vec(),
                };
                products[sequence * rc + slot * 1024..sequence * rc + (slot + 1) * 1024]
                    .copy_from_slice(
                        &kernel_core::latent_glue_oracle::exact_window_bf16(&encoded, 512)
                            .map_err(|e| e.to_string())?,
                    );
            }
        }
        c = products;
        // A same-sized unread plane lets the swapped-plane control stay in bounds.
        s = vec![0xff; c.len()];
        (rc, 0, 256 * 1024 + 34, 11, 1024, 0)
    } else {
        (RC, RS, SC, SS, 512, 16)
    };
    let roles = LatentWindowStageBuffers {
        ring: LatentWindowBuffers {
            codes: Spec::input(DType::U8, (Rows::Fixed(1), c.len()), c),
            scales: Spec::input(DType::U8, (Rows::Fixed(1), s.len()), s),
            item_slots: Spec::input(DType::I32, (Rows::Live, 1), ints(&slots)),
            items: capacity,
            sequences: 3,
            code_stride: rc as u64,
            scale_stride: rs as u64,
            storage: LatentWindowStorage::Staged { slots: 136 },
            key_lists: Spec::untouched(4),
            key_stride: 0,
            budget: 0,
        },
        staged: LatentWindowBuffers {
            codes: Spec::state(
                DType::U8,
                (Rows::Live, sc),
                vec![0xa5; capacity as usize * sc],
                Write::Map(Box::new(move |images, live| map(images, live, sc, width))),
            ),
            scales: Spec::state(
                DType::U8,
                (Rows::Live, ss),
                vec![0xbd; capacity as usize * ss],
                Write::Map(Box::new(move |images, live| {
                    map(images, live, ss, scale_width)
                })),
            ),
            item_slots: Spec::untouched(4),
            items: 0,
            sequences: capacity,
            code_stride: sc as u64,
            scale_stride: ss as u64,
            storage: LatentWindowStorage::Staged { slots: 256 },
            key_lists: Spec::untouched(4),
            key_stride: 0,
            budget: 0,
        },
        first_positions: Spec::input(DType::I32, (Rows::Live, 1), ints(&first)),
        last_positions: Spec::input(DType::I32, (Rows::Fixed(1), 3), ints(&last)),
        live_items: Spec::live(),
    };
    let mut case = OpCase::new(op, capacity, roles)?.oracle(move |images, live| {
        let (c, s) = gather_bits(
            LatentWindowStage { kv, ..QUERY },
            PlaneImages {
                codes: images.bytes("ring.codes")?,
                scales: images.bytes("ring.scales")?,
                code_stride: rc,
                scale_stride: rs,
                sequences: 3,
            },
            PlaneImages {
                codes: images.bytes("staged.codes")?,
                scales: images.bytes("staged.scales")?,
                code_stride: sc,
                scale_stride: ss,
                sequences: capacity as usize,
            },
            &read(images, "ring.item_slots")?,
            &read(images, "first_positions")?,
            &read(images, "last_positions")?,
            live as usize,
        )
        .map_err(|e| e.to_string())?;
        Ok(vec![
            ("staged.codes".into(), Expected::Bits(c)),
            ("staged.scales".into(), Expected::Bits(s)),
        ])
    });
    if bf16 {
        for live in crate::poison::live_counts(capacity)
            .into_iter()
            .filter(|&live| live > 0)
        {
            case = case.no_write(
                "staged.scales",
                live,
                "BF16 staging has no scale plane; this guard must remain untouched",
            );
        }
    }
    Ok(case)
}
pub fn required<'a>(scenario: Scenario) -> Required<'a> {
    Required::new(
        format!("window stage {scenario:?}"),
        Need::rows(scenario.capacity()),
        move |b, n| b.latent_window_stage(&QUERY, n),
        move |op| case(scenario, op),
    )
    .required()
}
struct Mutant {
    inner: Prepared<LatentWindowStageBuffers<View>>,
    info: OpInfo,
    symbol: String,
}
impl PreparedOp<LatentWindowStageBuffers<View>> for Mutant {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowStageBuffers<View>,
        w: View,
        c: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, w, c)?;
        launches[0].symbol.clone_from(&self.symbol);
        Ok(launches)
    }
}
pub fn mutant<'a>(mutation: Mutation) -> Required<'a> {
    let scenario = if mutation == Mutation::UnclampedStart {
        Scenario::Short
    } else {
        Scenario::Wrap
    };
    Required::new(
        format!("window stage mutant {mutation:?}"),
        Need::rows(scenario.capacity()),
        move |b, n| {
            let inner = b.latent_window_stage(&QUERY, n)?;
            let symbol = mutation.symbol();
            let mut info = inner.info().clone();
            info.symbols[0].0.clone_from(&symbol);
            Ok(Box::new(Mutant {
                inner,
                info,
                symbol,
            }) as Prepared<LatentWindowStageBuffers<View>>)
        },
        move |op| case(scenario, op),
    )
    .required()
}
