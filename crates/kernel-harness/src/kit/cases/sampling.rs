//! Cases of the host's device sampler (`kernel_core::sampling`) through the
//! kit's host adapter ([`crate::kit::host`]): every served path the host
//! dispatches (simple, top-logprob, greedy with and without one top
//! logprob, filtered) over F32 and BF16 logits, each row judged by the
//! host's CPU sampler (`runtime_sampling::oracle::sample_row`) with
//! [`kernel_core::sampling::check`]: the same token and top IDs, logprobs
//! within [`kernel_core::sampling::LOGPROB_TOLERANCE`].
//!
//! Rows hold [`LIVE`] live logits (three tiles of 1,024, the last partial)
//! of a [`STRIDE`]-element stride in a [`WIDTH`]-wide sampler; the elements
//! past the live width are NaN, which fails any row that reads them. Logits
//! lie on a 1/128 grid, so raw logits tie and the ties break by token ID.
//! Each path's rows meet its dispatch condition and mix the settings it
//! serves: greedy and tempered rows with their own seeds and counters; on
//! the filtered path, top-k, top-p and min-p filters, penalties over count
//! lists, logit biases, EOS masks naming the row's best token, and
//! constraint masks whose bits past the live width are set. Every row
//! samples (no row fails), so the output's write map holds each live row's
//! token, status and logprob and its first `top_count` top entries, at
//! 4-byte granularity: a write of one more top ID fails.
//!
//! The ops claim no row property: the contract fixes tokens and top IDs
//! but bounds logprobs, so no row's bits are claimed. Each conformance run
//! still judges every row against its own oracle at row counts 1, 6, 11
//! and 12, both sides of the eight-row switch from 256 to 64 threads.
//! Partition ops launch up to four rows under a 16-row gathered dispatch,
//! as an owned subview of a gathered view does (64 threads).

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::host::{self, SamplingPath};
use crate::kit::{OpCase, Required, Spec};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::ngram_oracle::splitmix64;
use kernel_core::sampling::{
    BIAS_BYTES, COUNT_BYTES, ListSpan, OUTPUT_BYTES, RowOutcome, RowRecord, RowSettings,
    SamplingBuffers, SamplingRequirement, SamplingRow, SamplingScratch, TokenBias, TokenCount,
};

/// The rows every sampling op serves: past eight, so the simple partials
/// switch from 256 threads to 64 between row counts.
pub const ROWS: u32 = 12;

/// The sampler's widest row.
pub const WIDTH: u32 = 4096;

/// The live logits of each row.
pub const LIVE: u32 = 2600;

/// Elements between rows of logits.
pub const STRIDE: usize = 2688;

/// One row's inputs past its logits.
#[derive(Clone, Debug)]
struct Row {
    settings: RowSettings,
    counts: Vec<TokenCount>,
    biases: Vec<TokenBias>,
    eos: Vec<u32>,
    /// `LIVE.div_ceil(32)` words, or none to admit every token.
    mask: Option<Vec<u32>>,
}

impl Row {
    fn plain(settings: RowSettings) -> Self {
        Self {
            settings,
            counts: Vec::new(),
            biases: Vec::new(),
            eos: Vec::new(),
            mask: None,
        }
    }

    fn at<'r>(&'r self, logits: &'r [f32]) -> SamplingRow<'r> {
        SamplingRow {
            logits,
            settings: self.settings,
            counts: &self.counts,
            biases: &self.biases,
            eos: &self.eos,
            mask: self.mask.as_deref(),
        }
    }
}

/// Row `row`'s live logits: multiples of 1/128 in [-8, 8], rounded to
/// BF16 for BF16 rows.
fn logits(row: usize, dtype: DType) -> Vec<f32> {
    (0..LIVE as u64)
        .map(|token| {
            let hash = splitmix64(0x6c6f_6769_7473 ^ (row as u64) << 32 ^ token);
            let value = ((hash % 2049) as i32 - 1024) as f32 / 128.0;
            match dtype {
                DType::Bf16 => bf16_to_f32(f32_to_bf16(value)),
                _ => value,
            }
        })
        .collect()
}

/// The lowest token with the largest logit.
fn best(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (token, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = token;
        }
    }
    best as u32
}

/// `count` distinct live tokens of row `row` from `salt`, ascending.
fn tokens(row: usize, salt: u64, count: usize) -> Vec<u32> {
    let mut tokens = std::collections::BTreeSet::new();
    let mut at = 0;
    while tokens.len() < count {
        tokens.insert((splitmix64(salt << 40 ^ (row as u64) << 20 ^ at) % u64::from(LIVE)) as u32);
        at += 1;
    }
    tokens.into_iter().collect()
}

/// A constraint mask admitting the tokens `admits` keeps, with every bit
/// past the live width set.
fn mask(admits: impl Fn(u32) -> bool) -> Vec<u32> {
    (0..LIVE.div_ceil(32))
        .map(|word| {
            (0..32).fold(0_u32, |bits, bit| {
                let token = word * 32 + bit;
                if token >= LIVE || admits(token) {
                    bits | 1 << bit
                } else {
                    bits
                }
            })
        })
        .collect()
}

/// The first `count` rows of `path`, each meeting the path's dispatch
/// condition.
fn rows(path: SamplingPath, dtype: DType, count: usize) -> Vec<Row> {
    (0..count)
        .map(|row| {
            let seed = splitmix64(0x7365_6564 ^ row as u64);
            let generated = 11 + 3 * row as u64;
            let tempered = RowSettings::new(0.5 + 0.25 * (row % 5) as f64, seed, generated);
            let greedy = RowSettings::new(0.0, seed, generated);
            match path {
                SamplingPath::Simple => Row::plain(if row % 3 == 1 { greedy } else { tempered }),
                SamplingPath::Top => Row::plain(RowSettings {
                    top_logprobs: [20, 1, 5, 0, 3, 7][row % 6],
                    ..if row % 3 == 1 { greedy } else { tempered }
                }),
                SamplingPath::Greedy { top } => Row::plain(RowSettings {
                    top_logprobs: u32::from(top && row % 2 == 0),
                    ..greedy
                }),
                SamplingPath::Filtered => filtered(row, dtype, tempered, greedy),
            }
        })
        .collect()
}

/// Row `row` of the filtered path, one of four kinds; the first, which a
/// one-row launch samples alone, takes every filter and list, chosen so
/// that the filtered path's null phase sees each stage it drops
/// ([`kept_without`]).
fn filtered(row: usize, dtype: DType, tempered: RowSettings, greedy: RowSettings) -> Row {
    let values = logits(row, dtype);
    let best = best(&values);
    let counts = tokens(row, 1, 24)
        .into_iter()
        .chain([best])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(at, token_id)| TokenCount {
            token_id,
            prompt_seen: at % 2 == 0,
            output_count: (at % 3) as u32 + u32::from(at % 2 == 1),
        })
        .collect::<Vec<_>>();
    let biases = tokens(row, 2, 6)
        .into_iter()
        .zip([2.5, -1.75, 8.0, -8.0, 0.625, 3.0])
        .map(|(token_id, bias)| TokenBias { token_id, bias })
        .collect::<Vec<_>>();
    let eos = tokens(row, 3, 4)
        .into_iter()
        .chain([best])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    match row % 4 {
        0 => {
            let admits = |token: u32| token % 5 != 4;
            // Each tile's best admitted raw logit is an EOS token, which
            // the row never samples.
            let eos = eos
                .into_iter()
                .chain(values.chunks(1024).enumerate().map(|(tile, chunk)| {
                    let admitted = chunk
                        .iter()
                        .enumerate()
                        .map(|(at, &value)| ((tile * 1024 + at) as u32, value))
                        .filter(|&(token, _)| admits(token));
                    admitted
                        .fold((u32::MAX, f32::NEG_INFINITY), |best, (token, value)| {
                            if value > best.1 { (token, value) } else { best }
                        })
                        .0
                }))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            let mut inputs = Row {
                settings: RowSettings {
                    temperature: 0.9,
                    top_k: 50,
                    top_p: 0.9,
                    min_p: 0.02,
                    repetition_penalty: 1.2,
                    frequency_penalty: 0.2,
                    presence_penalty: 0.3,
                    top_logprobs: 4,
                    mask_eos: true,
                    ..tempered
                },
                counts,
                biases,
                eos,
                mask: Some(mask(admits)),
            };
            inputs.settings.seed = (0..1024)
                .map(|at| splitmix64(tempered.seed ^ at << 48))
                .find(|&seed| {
                    let row = Row {
                        settings: RowSettings {
                            seed,
                            ..inputs.settings
                        },
                        ..inputs.clone()
                    };
                    kept_without(&row, &values)
                })
                .expect("a seed whose sample the cutoffs change");
            inputs
        }
        1 => Row {
            settings: RowSettings {
                temperature: 1.3,
                top_p: 0.85,
                repetition_penalty: 0.8,
                frequency_penalty: -0.4,
                presence_penalty: 0.5,
                ..tempered
            },
            counts,
            biases,
            eos: Vec::new(),
            mask: Some(mask(|_| true)),
        },
        2 => Row {
            settings: RowSettings {
                repetition_penalty: 1.5,
                frequency_penalty: 0.75,
                top_logprobs: 2,
                mask_eos: true,
                ..greedy
            },
            counts,
            biases,
            eos,
            mask: Some(mask(|token| token % 3 != 2)),
        },
        _ => Row {
            settings: RowSettings {
                temperature: 0.7,
                top_k: 7,
                min_p: 0.1,
                top_logprobs: 20,
                ..tempered
            },
            counts: Vec::new(),
            biases: Vec::new(),
            eos: Vec::new(),
            mask: Some(mask(|token| token % 3 != 1)),
        },
    }
}

/// Whether the row's top-k and top-p cutoffs change the token it samples:
/// the filtered path without its histogram or resolve stage keeps every
/// min-p survivor. (Without its Gumbel stage it takes a tile's best
/// admitted raw logit, an EOS token of the row; without its first stage
/// it reads poison.)
fn kept_without(row: &Row, logits: &[f32]) -> bool {
    let uncut = Row {
        settings: RowSettings {
            top_k: 0,
            top_p: 1.0,
            ..row.settings
        },
        ..row.clone()
    };
    let token = |row: &Row| match runtime_sampling::oracle::sample_row(&row.at(logits)) {
        Ok(Ok(sampled)) => Some(sampled.token_id),
        _ => None,
    };
    matches!((token(row), token(&uncut)), (Some(cut), Some(all)) if cut != all)
}

/// The metadata records and the flat count, bias and EOS lists of `rows`,
/// each list at least the 16, 16 and 8 bytes the record asks for.
fn encode(rows: &[Row]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut meta, mut counts, mut biases, mut eos) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in rows {
        let span = |offset: usize, len: usize| ListSpan {
            offset: offset as u64,
            len: len as u32,
        };
        let listed = if row.settings.mask_eos {
            &row.eos[..]
        } else {
            &[]
        };
        meta.extend(row.settings.encode(
            span(counts.len() / COUNT_BYTES, row.counts.len()),
            span(biases.len() / BIAS_BYTES, row.biases.len()),
            span(eos.len() / 4, listed.len()),
        ));
        counts.extend(row.counts.iter().flat_map(TokenCount::encode));
        biases.extend(row.biases.iter().flat_map(TokenBias::encode));
        eos.extend(listed.iter().flat_map(|token| token.to_le_bytes()));
    }
    counts.resize(counts.len().max(COUNT_BYTES), 0);
    biases.resize(biases.len().max(BIAS_BYTES), 0);
    eos.resize(eos.len().max(8), 0);
    (meta, counts, biases, eos)
}

/// The logits buffer: each row's live logits, then NaN to the stride.
fn logits_image(dtype: DType, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|row| {
            let mut values = logits(row, dtype);
            values.resize(STRIDE, f32::NAN);
            values
        })
        .flat_map(|value| match dtype {
            DType::Bf16 => f32_to_bf16(value).to_le_bytes().to_vec(),
            _ => value.to_le_bytes().to_vec(),
        })
        .collect()
}

/// Each row's live logits as the case's `logits` image holds them.
fn live_logits(images: &Images, dtype: DType, live: u32) -> LabResult<Vec<Vec<f32>>> {
    let values = match dtype {
        DType::Bf16 => images
            .u16s("logits")?
            .into_iter()
            .map(bf16_to_f32)
            .collect(),
        _ => images.f32s("logits")?,
    };
    Ok((0..live as usize)
        .map(|row| values[row * STRIDE..row * STRIDE + LIVE as usize].to_vec())
        .collect())
}

/// The record bytes the sampler writes at `live` rows of `tops` top
/// entries each.
fn record_map(tops: &[usize], live: u32) -> LabResult<WriteMap> {
    let mut ranges = Vec::new();
    for (row, &top) in tops.iter().take(live as usize).enumerate() {
        let base = row * OUTPUT_BYTES;
        ranges.push(base..base + 16);
        ranges.push(base + 16..base + 16 + 4 * top);
        ranges.push(base + 96..base + 96 + 8 * top);
    }
    WriteMap::of(DType::F32).byte_ranges(ranges)
}

/// The case of `path` over the first `capacity` of its rows of
/// `requirement`'s logits.
pub fn case<'a>(
    requirement: SamplingRequirement,
    path: SamplingPath,
    capacity: u32,
    prepared: Prepared<SamplingBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let dtype = requirement.dtype;
    let rows = rows(path, dtype, capacity as usize);
    let (meta, counts, biases, eos) = encode(&rows);
    let masks = rows.iter().any(|row| row.mask.is_some()).then(|| {
        rows.iter()
            .flat_map(|row| row.mask.clone().unwrap_or_else(|| mask(|_| true)))
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>()
    });
    let tops = rows
        .iter()
        .enumerate()
        .map(|(row, inputs)| inputs.at(&logits(row, dtype)).top_count())
        .collect::<Vec<_>>();
    let sized = |path: &str| {
        prepared
            .info()
            .bytes_of(path)
            .map(|bytes| Spec::scratch(bytes as usize))
    };
    let words = LIVE.div_ceil(32) as usize;
    let roles = SamplingBuffers {
        logits: Spec::input(
            dtype,
            (Rows::Live, STRIDE),
            logits_image(dtype, capacity as usize),
        ),
        metadata: Spec::input(
            DType::U8,
            (Rows::Live, meta.len() / capacity as usize),
            meta,
        ),
        counts: Spec::input(
            DType::U8,
            (Rows::Fixed(counts.len() / COUNT_BYTES), COUNT_BYTES),
            counts,
        ),
        biases: Spec::input(
            DType::U8,
            (Rows::Fixed(biases.len() / BIAS_BYTES), BIAS_BYTES),
            biases,
        ),
        eos: Spec::input(DType::U8, (Rows::Fixed(eos.len() / 4), 4), eos),
        masks: masks.map(|masks| Spec::input(DType::U8, (Rows::Live, words * 4), masks)),
        // Records as 4-byte elements, the write map's granularity.
        output: Spec::output_with(
            DType::F32,
            (Rows::Live, OUTPUT_BYTES / 4),
            Write::Map(Box::new(move |_, live| record_map(&tops, live))),
        ),
        scratch: SamplingScratch {
            partials: sized("scratch.partials").ok_or("the sampler sizes no partials")?,
            filter: sized("scratch.filter").unwrap_or_else(|| Spec::untouched(16)),
            adjusted: sized("scratch.adjusted").unwrap_or_else(|| Spec::untouched(16)),
            order: sized("scratch.order").unwrap_or_else(|| Spec::untouched(16)),
        },
        live_width: LIVE,
        row_stride: STRIDE as u64,
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        if live == 0 {
            return Ok(Vec::new());
        }
        let logits = live_logits(images, dtype, live)?;
        let outcomes = rows
            .iter()
            .zip(&logits)
            .enumerate()
            .map(|(index, (row, logits))| {
                runtime_sampling::oracle::sample_row(&row.at(logits))
                    .map_err(|refusal| format!("row {index}: {}", refusal.0))
            })
            .collect::<LabResult<Vec<RowOutcome>>>()?;
        let rows = rows.clone();
        let judge = move |bytes: &[u8]| -> LabResult<()> {
            for (index, ((row, logits), outcome)) in
                rows.iter().zip(&logits).zip(&outcomes).enumerate()
            {
                let record = bytes
                    .get(index * OUTPUT_BYTES..(index + 1) * OUTPUT_BYTES)
                    .and_then(|record| record.try_into().ok())
                    .ok_or_else(|| format!("no record of row {index}"))?;
                kernel_core::sampling::check(&row.at(logits), outcome, &RowRecord::decode(record))
                    .map_err(|refusal| format!("row {index}: {}", refusal.0))?;
            }
            Ok(())
        };
        Ok(vec![(
            "output".to_owned(),
            Expected::Judge(Box::new(judge)),
        )])
    };
    let case = OpCase::new(prepared, capacity, roles)?.oracle(oracle);
    Ok(match path {
        SamplingPath::Filtered => {
            let collect = match dtype {
                DType::F32 => runtime_kernels::SAMPLING_FILTERED_F32_SYMBOLS[2],
                _ => runtime_kernels::SAMPLING_FILTERED_BF16_SYMBOLS[2],
            };
            case.redundant_launch(
                collect,
                "resolve sorts every survivor itself when the collected candidates do not \
                 match the histogram's window, so the cutoff is exact without collect",
            )
        }
        _ => case,
    })
}

/// The rows of a gathered view whose dispatch the [`PARTITION`] rows of an
/// owned subview keep: past eight, so the host's simple partials use 64
/// threads where a view of the subview's rows alone would use 256.
pub const GATHERED: u32 = 16;

/// The rows of a partitioned view.
pub const PARTITION: u32 = 4;

/// Every served path over F32 and BF16 logits, over [`ROWS`] rows, and the
/// simple and greedy paths over [`PARTITION`] rows of a [`GATHERED`]-row
/// dispatch.
pub fn family<'a>() -> Vec<Required<'a>> {
    let mut ops = Vec::new();
    for dtype in [DType::F32, DType::Bf16] {
        let requirement = SamplingRequirement {
            dtype,
            width: WIDTH,
        };
        for path in [
            SamplingPath::Simple,
            SamplingPath::Top,
            SamplingPath::Greedy { top: false },
            SamplingPath::Greedy { top: true },
            SamplingPath::Filtered,
        ] {
            ops.push(Required::new(
                format!(
                    "sampling {} {dtype:?} live {LIVE} of {WIDTH} rows {ROWS}",
                    path.name()
                ),
                Need::rows(ROWS),
                move |_, need| host::sampling(requirement, path, None, need),
                move |prepared| case(requirement, path, ROWS, prepared),
            ));
        }
        for path in [SamplingPath::Simple, SamplingPath::Greedy { top: false }] {
            ops.push(Required::new(
                format!(
                    "sampling {} {dtype:?} live {LIVE} of {WIDTH} rows {PARTITION} of a \
                     {GATHERED}-row dispatch",
                    path.name()
                ),
                Need::rows(PARTITION),
                move |_, need| host::sampling(requirement, path, Some(GATHERED), need),
                move |prepared| case(requirement, path, PARTITION, prepared),
            ));
        }
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::sampling::Sampled;

    fn sampled(path: SamplingPath, dtype: DType) -> Vec<(Row, Sampled)> {
        rows(path, dtype, ROWS as usize)
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let logits = logits(index, dtype);
                let outcome = runtime_sampling::oracle::sample_row(&row.at(&logits))
                    .expect("a row inside the contract")
                    .expect("every row samples");
                (row, outcome)
            })
            .collect()
    }

    /// Each path's rows meet the host's dispatch condition for it and
    /// every row samples.
    #[test]
    fn rows_meet_their_path() {
        for dtype in [DType::F32, DType::Bf16] {
            for path in [
                SamplingPath::Simple,
                SamplingPath::Top,
                SamplingPath::Greedy { top: false },
                SamplingPath::Greedy { top: true },
            ] {
                let rows = sampled(path, dtype);
                for (row, _) in &rows {
                    let settings = row.settings;
                    assert!(row.counts.is_empty() && row.biases.is_empty());
                    assert!(!settings.mask_eos && row.mask.is_none());
                    assert_eq!(
                        (settings.top_k, settings.top_p, settings.min_p),
                        (0, 1.0, 0.0)
                    );
                    if let SamplingPath::Greedy { .. } = path {
                        assert!(settings.temperature == 0.0 && settings.top_logprobs <= 1);
                    }
                }
                let top = rows.iter().any(|(row, _)| row.settings.top_logprobs > 0);
                let greedy = rows.iter().all(|(row, _)| row.settings.temperature == 0.0);
                let tempered = rows.iter().any(|(row, _)| row.settings.temperature > 0.0);
                match path {
                    SamplingPath::Simple => assert!(!top && tempered && !greedy),
                    SamplingPath::Top => assert!(top && tempered),
                    SamplingPath::Greedy { top: wanted } => assert!(greedy && top == wanted),
                    SamplingPath::Filtered => unreachable!(),
                }
            }
            sampled(SamplingPath::Filtered, dtype);
        }
    }

    /// The filtered rows' settings change what they sample: without its
    /// filters, lists and masks, each kind of row samples another token or
    /// reports other top entries, and the EOS mask removes the row's best
    /// token.
    #[test]
    fn filtered_settings_matter() {
        for dtype in [DType::F32, DType::Bf16] {
            for (index, (row, outcome)) in sampled(SamplingPath::Filtered, dtype)
                .into_iter()
                .enumerate()
                .take(4)
            {
                let logits = logits(index, dtype);
                let plain = Row::plain(RowSettings {
                    top_logprobs: row.settings.top_logprobs,
                    ..RowSettings::new(
                        row.settings.temperature,
                        row.settings.seed,
                        row.settings.generated,
                    )
                });
                let unfiltered = runtime_sampling::oracle::sample_row(&plain.at(&logits))
                    .expect("a plain row")
                    .expect("a plain row samples");
                assert_ne!(
                    (outcome.token_id, &outcome.top),
                    (unfiltered.token_id, &unfiltered.top),
                    "row {index}"
                );
                if row.settings.mask_eos {
                    assert!(row.eos.contains(&best(&logits)));
                    assert_ne!(outcome.token_id, best(&logits), "row {index}");
                }
            }
        }
    }

    /// The rows a one-row launch samples see every filtered stage the null
    /// phase drops: the cutoffs change the token, and each tile's best
    /// admitted raw logit, which the chain without its Gumbel stage
    /// samples, is an EOS token the row cannot sample.
    #[test]
    fn the_first_filtered_row_sees_each_stage() {
        for dtype in [DType::F32, DType::Bf16] {
            for (index, (row, outcome)) in sampled(SamplingPath::Filtered, dtype)
                .into_iter()
                .enumerate()
                .step_by(4)
            {
                let logits = logits(index, dtype);
                assert!(kept_without(&row, &logits), "row {index}");
                for (tile, chunk) in logits.chunks(1024).enumerate() {
                    let at = (0..chunk.len())
                        .filter(|&at| row.at(&logits).admits((tile * 1024 + at) as u32))
                        .fold(None::<usize>, |best, at| match best {
                            Some(best) if chunk[best] >= chunk[at] => Some(best),
                            _ => Some(at),
                        })
                        .unwrap();
                    let token = (tile * 1024 + at) as u32;
                    assert!(row.eos.contains(&token), "row {index} tile {tile}");
                    assert_ne!(outcome.token_id, token, "row {index}");
                }
            }
        }
    }

    /// The write map holds one more top ID only where the row reports it.
    #[test]
    fn records_map_the_reported_top_entries() {
        let map = record_map(&[1], 1).unwrap();
        let ranges = map.ranges().to_vec();
        assert_eq!(ranges, vec![0..20, 96..104]);
    }
}
