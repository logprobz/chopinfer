//! The Qwen3.8 bundle's speculation rows: the verify n-gram and PLE
//! convolution against one per-sequence launch per item, and the PLE state
//! after each verify row and at a prefill anchor and the MTP drafter's
//! input, norms, residual and draft argmax, all against the contract's
//! own-op oracles (`own_ops::{ngram, ple, mtp}`): bitwise, except the
//! convolution rows and the norms, within the oracles' bounds. Every case
//! plans more items or rows than are live.
//!
//! `CHOPINFER_QWEN38_RANKS` (2 or 4, default 4) selects the topology of
//! the rows that depend on it: the verify n-gram (the rank's hash heads)
//! and the draft argmax (the vocabulary shard and the rank count). TP4's
//! are in `probe_sm120`; TP2's are the same probe entries built
//! with the bundle's defines of that rank count, in `probe_{tag}` as
//! `chop_probe_{tag}_*`. The other rows do not depend on the rank count.
//!
//! Run on one GPU in a kernel-lab window:
//! `CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --test qwen38_speculation_rows`.

use kernel_core::ngram_oracle as ng;
use kernel_lab::launch::simple_config;
use model_qwen38_contract::own_ops::{ItemForm, item_rows, mtp, ngram, ple};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};

#[path = "common/norm_weights.rs"]
mod norm_weight_checks;
#[path = "common/pool_offsets.rs"]
mod pool_offsets;
const MTP_NORMS: &str = "chop_probe_qwen38_mtp_norms_v1";
const BASE_CHECKPOINTS: &str = "chop_probe_qwen38_ple_checkpoints_v1";
const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const WIDE: usize = HIDDEN * STREAMS;
const PLE_CONV_BYTES: usize = 9 * WIDE * 2;
const PLE_ENTRY_BYTES: usize = PLE_CONV_BYTES + 128;
const VERIFY_ROWS: usize = 8;
const EOS: u32 = 248_044;
const CHUNK: usize = 16 * 1024;
const SENTINEL: u8 = 0xa5;
/// The probe rows' PLE state entry.
const LAYOUT: ple::PleLayout = ple::PleLayout {
    wide: WIDE as u32,
    conv_rows: 9,
    eos: EOS,
};
const VERIFY: ItemForm = ItemForm::Verify {
    rows: VERIFY_ROWS as u32,
};

/// `bytes` as little-endian words.
fn words_of<T>(bytes: &[u8], word: impl Fn([u8; 4]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(4)
        .map(|chunk| word(chunk.try_into().unwrap()))
        .collect()
}

/// `bytes` as BF16 values.
fn values_of(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

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

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
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

fn halves(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn le_words<T: Copy>(values: &[T], bytes: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|&value| bytes(value)).collect()
}

fn upload(session: &kernel_lab::Session, bytes: &[u8]) -> kernel_lab::LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn words(session: &kernel_lab::Session, values: &[i32]) -> kernel_lab::LabResult<DeviceBuffer> {
    let mut bytes = le_words(values, |value| value.to_le_bytes().to_vec());
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

fn launch(
    session: &kernel_lab::Session,
    module: &str,
    symbol: &str,
    grid: (u32, u32, u32),
    args: &[KernelArgument<'_>],
) -> kernel_lab::LabResult<()> {
    session
        .module(module)?
        .function(symbol)
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            simple_config(grid, (256, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
            args,
        )
        .map_err(|e| e.to_string())
}

fn range(buffer: &DeviceBuffer, offset: usize, byte_len: usize) -> KernelArgument<'_> {
    KernelArgument::Range {
        buffer,
        offset,
        byte_len,
    }
}

/// The `probe_sm120` symbol of TP4's bundle row `name`.
fn probe(name: &str) -> String {
    format!("chop_probe_qwen38_{name}_v1")
}

/// The probe cubin and symbol of a rank-dependent row, `name` as in
/// `chop_probe_qwen38_{name}_v1`, at the selected topology.
fn rank_probe(name: &str) -> kernel_lab::LabResult<(String, String)> {
    let tag = kernel_lab::qwen38_profile()?.tag();
    let module = if tag == "qwen38" {
        "probe_sm120".to_string()
    } else {
        format!("probe_{tag}")
    };
    Ok((module, format!("chop_probe_{tag}_{name}_v1")))
}

/// The selected topology's row cubin: TP4's holds every row, TP2's
/// holds its n-gram rows.
fn row_module(session: &kernel_lab::Session) -> kernel_lab::LabResult<runtime_cuda::Module> {
    session.module(&format!("row_{}", kernel_lab::qwen38_profile()?.tag()))
}

fn launch_row(
    session: &kernel_lab::Session,
    module: &runtime_cuda::Module,
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

/// Four live verify items of 3, 1, 8 and 2 rows (eight planned), their
/// tokens and staged PLE entries with every n-gram history case.
struct Items {
    rows: Vec<usize>,
    cu: Vec<i32>,
    tokens: Vec<u32>,
    staged: Vec<u8>,
}

fn items() -> Items {
    let rows = vec![3, 1, 8, 2];
    let mut cu = vec![0];
    for &count in &rows {
        cu.push(cu.last().unwrap() + count as i32);
    }
    let total = *cu.last().unwrap() as usize;
    let tokens = (0..total)
        .map(|i| if i % 5 == 2 { EOS } else { 100 + i as u32 })
        .collect();
    let mut staged = Vec::new();
    for (item, history) in [[EOS, EOS], [19, EOS], [EOS, 23], [19, 23]]
        .iter()
        .enumerate()
    {
        let mut entry = halves(&bf16_values(9 * WIDE, 40 + item as u32));
        entry.extend_from_slice(&history[0].to_le_bytes());
        entry.extend_from_slice(&history[1].to_le_bytes());
        entry.resize(PLE_ENTRY_BYTES, 0);
        staged.extend(entry);
    }
    Items {
        rows,
        cu,
        tokens,
        staged,
    }
}

const PLANNED_ITEMS: u32 = 8;

/// `ngram_verify` at rank 1 (TP4) or the last rank; lean runs (TP2)
/// at every rank.
fn verify_ngram(session: &kernel_lab::Session, items: &Items) -> kernel_lab::LabResult<()> {
    let ranks = kernel_lab::qwen38_ranks()? as usize;
    if kernel_lab::qwen38_lean()? {
        (0..ranks).try_for_each(|rank| verify_ngram_rank(session, items, rank))
    } else {
        verify_ngram_rank(session, items, if ranks == 4 { 1 } else { ranks - 1 })
    }
}

fn verify_ngram_rank(
    session: &kernel_lab::Session,
    items: &Items,
    rank: usize,
) -> kernel_lab::LabResult<()> {
    let mut tables = ng::ngram_tables(248_064, 3, 8, 17, 0, 1, 1).map_err(|e| e.to_string())?;
    for offset in &mut tables.offsets {
        *offset += 1_u64 << 33;
    }
    // The rank's `local` hash heads, head `rank * local + y`.
    let local = kernel_lab::qwen38_shape()?.ngram_heads() as usize;
    let start = tables.offsets[rank * local];
    let table_rows = tables.sizes[rank * local..(rank + 1) * local]
        .iter()
        .sum::<u64>();
    let codes = (0..table_rows as usize * 160)
        .map(|i| (i % 127) as u8 | if i % 2 == 0 { 0 } else { 128 })
        .collect::<Vec<_>>();
    let u64s = |values: &[u64]| le_words(values, |value| value.to_le_bytes().to_vec());
    let [multipliers, sizes, offsets] = [&tables.multipliers, &tables.sizes, &tables.offsets]
        .map(|values| upload(session, &u64s(values)));
    let (multipliers, sizes, offsets) = (multipliers?, sizes?, offsets?);
    let table = upload(session, &codes)?;
    let scale = upload(session, &to_bf16(0.125).to_le_bytes())?;
    let tokens = upload(
        session,
        &le_words(&items.tokens, |value| value.to_le_bytes().to_vec()),
    )?;
    let staged = upload(session, &items.staged)?;
    let total = items.tokens.len();
    let head_row = local * 160;
    let output_bytes = total * head_row * 2;
    let ids_bytes = total * local * 8;
    let (verify_module, verify_symbol) = rank_probe("ngram_verify")?;
    let row_symbol = format!("chop_row_{}_ngram_v1", kernel_lab::qwen38_profile()?.tag());
    let history_bytes = items.rows.len() * PLE_ENTRY_BYTES;
    let run = |verify: bool| -> kernel_lab::LabResult<[Vec<u8>; 3]> {
        let output = upload(session, &vec![SENTINEL; output_bytes])?;
        let row_ids = upload(session, &vec![SENTINEL; ids_bytes])?;
        let history_out = upload(session, &vec![SENTINEL; history_bytes])?;
        let cu = words(session, &items.cu)?;
        let q_len = words(
            session,
            &items
                .rows
                .iter()
                .map(|&rows| rows as i32)
                .collect::<Vec<_>>(),
        )?;
        if verify {
            let live = words(session, &[items.rows.len() as i32])?;
            launch(
                session,
                &verify_module,
                &verify_symbol,
                (VERIFY_ROWS as u32, local as u32, PLANNED_ITEMS),
                &[
                    KernelArgument::Buffer(&tokens),
                    range(&staged, PLE_CONV_BYTES, items.staged.len() - PLE_CONV_BYTES),
                    KernelArgument::Buffer(&multipliers),
                    KernelArgument::Buffer(&sizes),
                    KernelArgument::Buffer(&offsets),
                    KernelArgument::Buffer(&table),
                    KernelArgument::Buffer(&scale),
                    KernelArgument::Buffer(&output),
                    KernelArgument::Buffer(&row_ids),
                    range(&history_out, PLE_CONV_BYTES, history_bytes - PLE_CONV_BYTES),
                    KernelArgument::Buffer(&cu),
                    KernelArgument::Buffer(&q_len),
                    KernelArgument::Buffer(&live),
                    KernelArgument::U32(rank as u32),
                    KernelArgument::U64(start),
                    KernelArgument::U64(table_rows),
                    KernelArgument::U64((PLE_ENTRY_BYTES / 4) as u64),
                ],
            )?;
        } else {
            let module = row_module(session)?;
            for (item, &rows) in items.rows.iter().enumerate() {
                let first = items.cu[item] as usize;
                let entry = item * PLE_ENTRY_BYTES + PLE_CONV_BYTES;
                launch_row(
                    session,
                    &module,
                    &row_symbol,
                    (rows as u32, local as u32, 1),
                    &[
                        range(&tokens, 4 * first, 4 * rows),
                        range(&staged, entry, 8),
                        KernelArgument::Buffer(&multipliers),
                        KernelArgument::Buffer(&sizes),
                        KernelArgument::Buffer(&offsets),
                        KernelArgument::Buffer(&table),
                        KernelArgument::Buffer(&scale),
                        range(&output, first * head_row * 2, rows * head_row * 2),
                        range(&row_ids, first * local * 8, rows * local * 8),
                        range(&history_out, entry, 8),
                        range(&q_len, 4 * item, 4),
                        KernelArgument::U32(rank as u32),
                        KernelArgument::U64(start),
                        KernelArgument::U64(table_rows),
                    ],
                )?;
            }
        }
        Ok([
            download(session, &output, output_bytes)?,
            download(session, &row_ids, ids_bytes)?,
            download(session, &history_out, history_bytes)?,
        ])
    };
    let (verify, reference) = (run(true)?, run(false)?);
    for (name, (got, expected)) in ["output", "row ids", "history"]
        .iter()
        .zip(verify.iter().zip(&reference))
    {
        if got != expected {
            return Err(format!(
                "ngram_verify rank {rank} {name} differs from per-item launches"
            ));
        }
    }
    // Both are the contract's n-gram of every verify item.
    let op = ngram::Ngram {
        orders: 3,
        heads_per_order: 8,
        head_width: 160,
        local_heads: local as u32,
        rank: rank as u32,
        eos: EOS,
        entry_words: (PLE_ENTRY_BYTES / 4) as u64,
        writes: ngram::NgramWrites::Gather,
        items: VERIFY,
    };
    let q_len = items
        .rows
        .iter()
        .map(|&rows| rows as i32)
        .collect::<Vec<_>>();
    let oracle = ngram::ngram_launch(
        &op,
        &tables,
        &items.tokens,
        &words_of(&items.staged[PLE_CONV_BYTES..], u32::from_le_bytes),
        &q_len,
        Some(&items.cu),
        Some(items.rows.len() as u32),
        PLANNED_ITEMS,
        ngram::Shard {
            start,
            rows: table_rows,
            codes: Some(&codes),
            scale: to_bf16(0.125),
        },
    )
    .map_err(|e| format!("ngram oracle: {}", e.0))?;
    let [output, row_ids, history] = &verify;
    for (at, item) in &oracle.written {
        let rows = at.start as usize..(at.start + u64::from(at.rows)) as usize;
        let entry = at.item as usize * PLE_ENTRY_BYTES + PLE_CONV_BYTES;
        let ids = row_ids[rows.start * local * 8..rows.end * local * 8]
            .chunks_exact(8)
            .map(|id| u64::from_le_bytes(id.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values_of(&output[rows.start * head_row * 2..rows.end * head_row * 2]) != item.output
            || ids != item.row_ids
            || words_of(&history[entry..entry + 8], u32::from_le_bytes) != item.history
        {
            return Err(format!(
                "ngram_verify rank {rank} item {} differs from the oracle",
                at.item
            ));
        }
    }
    Ok(())
}

fn verify_ple_conv(session: &kernel_lab::Session, items: &Items) -> kernel_lab::LabResult<()> {
    let total = items.tokens.len();
    let (input_values, weight_values) = (bf16_values(total * WIDE, 50), bf16_values(WIDE * 4, 51));
    let input = upload(session, &halves(&input_values))?;
    let weight = upload(session, &halves(&weight_values))?;
    let staged = upload(session, &items.staged)?;
    let history_bytes = items.rows.len() * PLE_ENTRY_BYTES;
    let run = |verify: bool| -> kernel_lab::LabResult<[Vec<u8>; 2]> {
        let output = upload(session, &vec![SENTINEL; total * WIDE * 2])?;
        let state_out = upload(session, &vec![SENTINEL; history_bytes])?;
        let cu = words(session, &items.cu)?;
        let q_len = words(
            session,
            &items
                .rows
                .iter()
                .map(|&rows| rows as i32)
                .collect::<Vec<_>>(),
        )?;
        if verify {
            let live = words(session, &[items.rows.len() as i32])?;
            launch(
                session,
                "probe_sm120",
                &probe("ple_conv_verify"),
                (VERIFY_ROWS as u32, (WIDE / 256) as u32, PLANNED_ITEMS),
                &[
                    KernelArgument::Buffer(&input),
                    KernelArgument::Buffer(&weight),
                    KernelArgument::Buffer(&staged),
                    KernelArgument::Buffer(&output),
                    KernelArgument::Buffer(&state_out),
                    KernelArgument::Buffer(&cu),
                    KernelArgument::Buffer(&q_len),
                    KernelArgument::Buffer(&live),
                    KernelArgument::U64((PLE_ENTRY_BYTES / 2) as u64),
                ],
            )?;
        } else {
            // TP4's row at every rank count.
            let module = session.module("row_qwen38")?;
            for (item, &rows) in items.rows.iter().enumerate() {
                let first = items.cu[item] as usize;
                launch_row(
                    session,
                    &module,
                    "chop_row_qwen38_ple_conv_prefill_v1",
                    (rows as u32, (WIDE / 256) as u32, 1),
                    &[
                        range(&input, first * WIDE * 2, rows * WIDE * 2),
                        KernelArgument::Buffer(&weight),
                        range(&staged, item * PLE_ENTRY_BYTES, PLE_CONV_BYTES),
                        range(&output, first * WIDE * 2, rows * WIDE * 2),
                        range(&state_out, item * PLE_ENTRY_BYTES, PLE_CONV_BYTES),
                        range(&q_len, 4 * item, 4),
                    ],
                )?;
            }
        }
        Ok([
            download(session, &output, total * WIDE * 2)?,
            download(session, &state_out, history_bytes)?,
        ])
    };
    let (verify, reference) = (run(true)?, run(false)?);
    if verify != reference {
        return Err("ple_conv_verify differs from per-item launches".into());
    }
    // Each item's rows within the contract's bounds, its state exact.
    let op = ple::PleConv {
        layout: LAYOUT,
        taps: 4,
        dilation: 3,
        items: VERIFY,
    };
    let [output, state_out] = &verify;
    let q_len = items
        .rows
        .iter()
        .map(|&rows| rows as i32)
        .collect::<Vec<_>>();
    let live = Some(items.rows.len() as u32);
    for at in item_rows(VERIFY, &q_len, Some(&items.cu), live, PLANNED_ITEMS)
        .map_err(|e| e.0.to_string())?
    {
        let rows = at.start as usize * WIDE..(at.start as usize + at.rows as usize) * WIDE;
        let entry = at.item as usize * PLE_ENTRY_BYTES;
        let staged = values_of(&items.staged[entry..entry + PLE_CONV_BYTES]);
        let oracle = ple::ple_conv_item(
            &op,
            &input_values[rows.clone()],
            &weight_values,
            &staged,
            at.rows,
        )
        .map_err(|e| format!("ple_conv oracle: {}", e.0))?;
        oracle
            .output
            .check_bf16(&values_of(&output[2 * rows.start..2 * rows.end]))
            .map_err(|failure| {
                format!(
                    "ple_conv_verify item {} outside the oracle: {failure:?}",
                    at.item
                )
            })?;
        if values_of(&state_out[entry..entry + PLE_CONV_BYTES]) != oracle.state {
            return Err(format!(
                "ple_conv_verify item {} state differs from the oracle",
                at.item
            ));
        }
    }
    Ok(())
}

/// The state after row j is the per-sequence kernels' output state for a
/// chunk ending there: the last nine convolution inputs and the two last
/// n-gram ids.
fn ple_checkpoints(session: &kernel_lab::Session, items: &Items) -> kernel_lab::LabResult<()> {
    let total = items.tokens.len();
    let input_values = bf16_values(total * WIDE, 60);
    let input = upload(session, &halves(&input_values))?;
    let tokens = upload(
        session,
        &le_words(&items.tokens, |value| value.to_le_bytes().to_vec()),
    )?;
    let staged = upload(session, &items.staged)?;
    let bytes = PLANNED_ITEMS as usize * VERIFY_ROWS * PLE_ENTRY_BYTES;
    let checkpoints = upload(session, &vec![SENTINEL; bytes])?;
    let cu = words(session, &items.cu)?;
    let q_len = words(
        session,
        &items
            .rows
            .iter()
            .map(|&rows| rows as i32)
            .collect::<Vec<_>>(),
    )?;
    let live = words(session, &[items.rows.len() as i32])?;
    launch(
        session,
        "probe_sm120",
        BASE_CHECKPOINTS,
        (
            VERIFY_ROWS as u32,
            PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32,
            PLANNED_ITEMS,
        ),
        &[
            KernelArgument::Buffer(&input),
            KernelArgument::Buffer(&tokens),
            KernelArgument::Buffer(&staged),
            KernelArgument::Buffer(&checkpoints),
            KernelArgument::Buffer(&cu),
            KernelArgument::Buffer(&q_len),
            KernelArgument::Buffer(&live),
        ],
    )?;
    let device = download(session, &checkpoints, bytes)?;
    // The base entry writes checkpoint item * 8 + row.
    let mut expected = vec![SENTINEL; bytes];
    let ids = (0..PLANNED_ITEMS as i32 * VERIFY_ROWS as i32).collect::<Vec<_>>();
    ple::ple_checkpoints(
        &ple::PleCheckpoints {
            layout: LAYOUT,
            verify_rows: VERIFY_ROWS as u32,
        },
        &input_values,
        &items.tokens,
        &items.staged,
        &mut expected,
        &items.cu,
        &items
            .rows
            .iter()
            .map(|&rows| rows as i32)
            .collect::<Vec<_>>(),
        &ids,
        items.rows.len() as u32,
        PLANNED_ITEMS,
    )
    .map_err(|e| format!("ple_checkpoints oracle: {}", e.0))?;
    for (index, (got, want)) in device
        .chunks_exact(PLE_ENTRY_BYTES)
        .zip(expected.chunks_exact(PLE_ENTRY_BYTES))
        .enumerate()
    {
        if got != want {
            let (item, row) = (index / VERIFY_ROWS, index % VERIFY_ROWS);
            return Err(format!("ple_checkpoints entry {item}/{row} differs"));
        }
    }
    Ok(())
}

fn mtp_gather(session: &kernel_lab::Session) -> kernel_lab::LabResult<()> {
    let row_bytes = WIDE * 2;
    let last = noise(6 * row_bytes, 70);
    let retained = noise(16 * row_bytes, 71);
    let (last_buffer, retained_buffer) = (upload(session, &last)?, upload(session, &retained)?);
    // Rows: last row 4, retained entry 9, last row 0, retained entry 0.
    let sources = [4, -10, 0, -1];
    let token_sources = [-1, 2, -1, 0];
    let drafts = [700, 701, 702];
    let planned = 8;
    let x = upload(session, &vec![SENTINEL; planned * row_bytes])?;
    let token_ids = words(session, &[11, 12, 13, 14, 15, 16, 17, 18])?;
    let [sources_buffer, token_buffer, drafts_buffer] =
        [&sources[..], &token_sources, &drafts].map(|values| words(session, values));
    let (sources_buffer, token_buffer, drafts_buffer) =
        (sources_buffer?, token_buffer?, drafts_buffer?);
    let live = words(session, &[sources.len() as i32])?;
    launch(
        session,
        "probe_sm120",
        &probe("mtp_gather"),
        (planned as u32, 1, 1),
        &[
            KernelArgument::Buffer(&last_buffer),
            KernelArgument::Buffer(&retained_buffer),
            KernelArgument::Buffer(&sources_buffer),
            KernelArgument::Buffer(&token_buffer),
            KernelArgument::Buffer(&drafts_buffer),
            KernelArgument::Buffer(&x),
            KernelArgument::Buffer(&token_ids),
            KernelArgument::Buffer(&live),
        ],
    )?;
    let device = download(session, &x, planned * row_bytes)?;
    let mut expected = vec![SENTINEL; planned * row_bytes];
    let mut expected_tokens = vec![11, 12, 13, 14, 15, 16, 17, 18];
    mtp::mtp_gather(
        &mtp::MtpGather { wide: WIDE as u32 },
        &last,
        &retained,
        &sources,
        &token_sources,
        &drafts,
        &mut expected,
        &mut expected_tokens,
        sources.len() as u32,
        planned as u32,
    )
    .map_err(|e| format!("mtp_gather oracle: {}", e.0))?;
    for (row, (got, want)) in device
        .chunks_exact(row_bytes)
        .zip(expected.chunks_exact(row_bytes))
        .enumerate()
    {
        if got != want {
            return Err(format!("mtp_gather row {row} differs"));
        }
    }
    let tokens = words_of(&download(session, &token_ids, 4 * 8)?, i32::from_le_bytes);
    if tokens != expected_tokens {
        return Err(format!("mtp_gather tokens {tokens:?}"));
    }
    Ok(())
}

fn mtp_norms_and_combine(session: &kernel_lab::Session) -> kernel_lab::LabResult<()> {
    let (planned, live) = (4, 3);
    let x = bf16_values(planned * WIDE, 80);
    let embed = bf16_values(planned * HIDDEN, 81);
    let hidden_weight = bf16_values(WIDE, 82);
    let embed_weight = bf16_values(HIDDEN, 83);
    let buffers = [&x, &embed, &hidden_weight, &embed_weight].map(|v| upload(session, &halves(v)));
    let [x_buffer, embed_buffer, hidden_buffer, weight_buffer] = buffers;
    let (x_buffer, embed_buffer, hidden_buffer, weight_buffer) =
        (x_buffer?, embed_buffer?, hidden_buffer?, weight_buffer?);
    let x_normed = upload(session, &vec![SENTINEL; planned * WIDE * 2])?;
    let embed_normed = upload(session, &vec![SENTINEL; planned * HIDDEN * 2])?;
    let live_rows = words(session, &[live as i32])?;
    launch(
        session,
        "probe_sm120",
        MTP_NORMS,
        (planned as u32, 1, 1),
        &[
            KernelArgument::Buffer(&x_buffer),
            KernelArgument::Buffer(&hidden_buffer),
            KernelArgument::Buffer(&embed_buffer),
            KernelArgument::Buffer(&weight_buffer),
            KernelArgument::Buffer(&x_normed),
            KernelArgument::Buffer(&embed_normed),
            KernelArgument::Buffer(&live_rows),
        ],
    )?;
    // Each live row within one BF16 rounding of the F64 norms.
    let [hidden_norm, embed_norm] = mtp::mtp_norms(
        &mtp::MtpNorms {
            hidden: HIDDEN as u32,
            streams: STREAMS as u32,
            eps: 1e-6,
        },
        &x,
        &hidden_weight,
        &embed,
        &embed_weight,
        live as u32,
    )
    .map_err(|e| format!("mtp_norms oracle: {}", e.0))?;
    let got = download(session, &x_normed, planned * WIDE * 2)?;
    hidden_norm
        .check_bf16(&values_of(&got[..live * WIDE * 2]))
        .map_err(|failure| format!("mtp_norms hidden: {failure:?}"))?;
    if got[live * WIDE * 2..].iter().any(|&byte| byte != SENTINEL) {
        return Err("mtp_norms wrote a dead row".into());
    }
    let got = download(session, &embed_normed, planned * HIDDEN * 2)?;
    embed_norm
        .check_bf16(&values_of(&got[..live * HIDDEN * 2]))
        .map_err(|failure| format!("mtp_norms embedding: {failure:?}"))?;
    // The combine: each stream plus the token row, rounded once.
    let out = upload(session, &vec![SENTINEL; planned * WIDE * 2])?;
    launch(
        session,
        "probe_sm120",
        &probe("mtp_combine"),
        (planned as u32, STREAMS as u32, 1),
        &[
            KernelArgument::Buffer(&x_buffer),
            KernelArgument::Buffer(&embed_buffer),
            KernelArgument::Buffer(&out),
            KernelArgument::Buffer(&live_rows),
        ],
    )?;
    let got = values_of(&download(session, &out, planned * WIDE * 2)?);
    let mut expected = vec![u16::from_le_bytes([SENTINEL; 2]); planned * WIDE];
    mtp::mtp_combine(
        &mtp::MtpCombine {
            hidden: HIDDEN as u32,
            streams: STREAMS as u32,
        },
        &x,
        &embed,
        &mut expected,
        live as u32,
        planned as u32,
    )
    .map_err(|e| format!("mtp_combine oracle: {}", e.0))?;
    if let Some(index) = (0..planned * WIDE).find(|&index| got[index] != expected[index]) {
        return Err(format!(
            "mtp_combine element {index}: {:#06x} vs {:#06x}",
            got[index], expected[index]
        ));
    }
    Ok(())
}

/// Each rank's best of its shard, then the global best of every row: the
/// highest logit, the lowest id on a tie, across and within ranks. Lean
/// runs (TP2) add two rows at the shard edge between ranks `edge` and
/// `edge + 1`: a tie of the lower rank's last id with the upper rank's
/// first, and a single winner at the upper rank's first id.
fn draft_argmax(session: &kernel_lab::Session) -> kernel_lab::LabResult<()> {
    let lean = kernel_lab::qwen38_lean()?;
    let (planned, live) = if lean { (6, 5) } else { (4, 3) };
    let ranks = kernel_lab::qwen38_ranks()? as usize;
    let shard_vocab = kernel_lab::qwen38_shape()?.vocab_shard() as usize;
    // Four words per rank and row.
    let row_bests = ranks * 16;
    let mut shards = (0..ranks as u32)
        .map(|rank| bf16_values(planned * shard_vocab, 90 + rank))
        .collect::<Vec<_>>();
    let top = to_bf16(9.0);
    // Row 0: a tie inside rank `ranks / 2`'s shard. Row 1: ranks `low` and
    // `high` tie, the lower rank's id winning with the higher local index
    // (TP4: ranks 1 and 3). Row 2: one maximum on rank 0.
    let middle = ranks / 2;
    let (low, high) = if ranks == 2 { (0, 1) } else { (1, ranks - 1) };
    let single = shard_vocab - 1080;
    shards[middle][100] = top;
    shards[middle][70] = top;
    shards[low][shard_vocab + 5] = top;
    shards[high][shard_vocab + 1] = top;
    shards[0][2 * shard_vocab + single] = top;
    let mut expected = vec![middle * shard_vocab + 70, low * shard_vocab + 5, single];
    if lean {
        let edge = ranks / 2 - 1;
        shards[edge][3 * shard_vocab + shard_vocab - 1] = top;
        shards[edge + 1][3 * shard_vocab] = top;
        shards[edge + 1][4 * shard_vocab] = top;
        expected.extend([
            edge * shard_vocab + shard_vocab - 1,
            (edge + 1) * shard_vocab,
        ]);
    }
    let (partial_module, partial_symbol) = rank_probe("draft_argmax_partial")?;
    let (final_module, final_symbol) = rank_probe("draft_argmax_final")?;
    let bests = upload(session, &vec![0; planned * row_bests])?;
    let live_rows = words(session, &[live as i32])?;
    let mut local = Vec::new();
    for (rank, shard) in shards.iter().enumerate() {
        let logits = upload(session, &halves(shard))?;
        let best = upload(session, &vec![SENTINEL; planned * 16])?;
        launch(
            session,
            &partial_module,
            &partial_symbol,
            (planned as u32, 1, 1),
            &[
                KernelArgument::Buffer(&logits),
                KernelArgument::Buffer(&best),
                KernelArgument::Buffer(&live_rows),
                KernelArgument::U32(rank as u32),
            ],
        )?;
        let got = words_of(&download(session, &best, planned * 16)?, u32::from_le_bytes);
        let mut want = vec![u32::from_le_bytes([SENTINEL; 4]); planned * 4];
        mtp::draft_argmax_partial(
            &mtp::DraftArgmaxPartial {
                shard_vocab: shard_vocab as u32,
                rank: rank as u32,
            },
            shard,
            &mut want,
            live as u32,
            planned as u32,
        )
        .map_err(|e| format!("draft argmax oracle: {}", e.0))?;
        if got != want {
            return Err(format!(
                "draft argmax rank {rank} bests differ from the oracle"
            ));
        }
        local.push(download(session, &best, planned * 16)?);
    }
    // The exchange lays every rank's best side by side per row.
    let mut exchanged = vec![0_u8; planned * row_bests];
    for row in 0..live {
        for (rank, best) in local.iter().enumerate() {
            let at = row * row_bests + rank * 16;
            exchanged[at..at + 16].copy_from_slice(&best[row * 16..row * 16 + 16]);
        }
    }
    bests
        .copy_from_bytes(&exchanged, &session.stream)
        .map_err(|e| e.to_string())?;
    // Lean rows 3 and 4 draft into slots 4 and 3; row 3 also feeds verify
    // token 7.
    let (slots, targets): (&[i32], &[i32]) = if lean {
        (&[2, 0, 1, 4, 3], &[5, -1, 1, 7, -1])
    } else {
        (&[2, 0, 1], &[5, -1, 1])
    };
    // The oracle's drafts and verify tokens from the exchanged bests.
    let (mut oracle_drafts, mut oracle_tokens) = (vec![-7; planned], vec![-9; 8]);
    mtp::draft_argmax_final(
        &mtp::DraftArgmaxFinal {
            ranks: ranks as u32,
        },
        &words_of(&exchanged, u32::from_le_bytes),
        slots,
        targets,
        &mut oracle_drafts,
        &mut oracle_tokens,
        live as u32,
        planned as u32,
    )
    .map_err(|e| format!("draft argmax oracle: {}", e.0))?;
    let slots = words(session, slots)?;
    let targets = words(session, targets)?;
    let drafts = words(session, &vec![-7; planned])?;
    let token_ids = words(session, &[-9; 8])?;
    launch(
        session,
        &final_module,
        &final_symbol,
        (planned as u32, 1, 1),
        &[
            KernelArgument::Buffer(&bests),
            KernelArgument::Buffer(&slots),
            KernelArgument::Buffer(&targets),
            KernelArgument::Buffer(&drafts),
            KernelArgument::Buffer(&token_ids),
            KernelArgument::Buffer(&live_rows),
        ],
    )?;
    let read = |buffer: &DeviceBuffer, count: usize| -> kernel_lab::LabResult<Vec<i32>> {
        Ok(download(session, buffer, 4 * count)?
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect())
    };
    let expected = expected.iter().map(|&id| id as i32).collect::<Vec<_>>();
    let drafts = read(&drafts, planned)?;
    let mut want_drafts = vec![expected[1], expected[2], expected[0]];
    let mut want_tokens = vec![-9, expected[2], -9, -9, -9, expected[0], -9, -9];
    if lean {
        want_drafts.extend([expected[4], expected[3]]);
        want_tokens[7] = expected[3];
    }
    want_drafts.push(-7);
    assert_eq!(
        (&oracle_drafts, &oracle_tokens),
        (&want_drafts, &want_tokens),
        "the oracle drafts the hand-derived ids"
    );
    if drafts != want_drafts {
        return Err(format!(
            "draft argmax drafts {drafts:?}, expected {want_drafts:?}"
        ));
    }
    let tokens = read(&token_ids, 8)?;
    if tokens != want_tokens {
        return Err(format!("draft argmax verify tokens {tokens:?}"));
    }
    Ok(())
}

#[test]
fn qwen38_speculation_rows_match_references() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let items = items();
        verify_ngram(session, &items)?;
        verify_ple_conv(session, &items)?;
        ple_checkpoints(session, &items)?;
        mtp_gather(session)?;
        mtp_norms_and_combine(session)?;
        draft_argmax(session)?;
        session.stream.synchronize().map_err(|e| e.to_string())
    })
    .expect("Qwen3.8 speculation rows");
}

#[test]
fn indexed_checkpoint_entries_match_copy_bits() {
    kernel_lab::gpu_test(pool_offsets::need_bytes(5 << 30), |session| {
        let name = format!(
            "{}_speculation",
            BASE_CHECKPOINTS.split('_').nth(2).unwrap()
        );
        let module = session
            .context
            .load_module(&owned_cubins::load(&name)?)
            .map_err(|e| e.to_string())?;
        for rows in 1..=VERIFY_ROWS {
            let input = upload(session, &halves(&bf16_values(rows * WIDE, 901)))?;
            let tokens = words(
                session,
                &(0..rows as i32).map(|row| 100 + row).collect::<Vec<_>>(),
            )?;
            let committed = items().staged[..PLE_ENTRY_BYTES].to_vec();
            let staged = upload(session, &committed)?;
            let cu = words(session, &[0, rows as i32])?;
            let q_len = words(session, &[rows as i32])?;
            let live = words(session, &[1])?;
            let baseline = upload(
                session,
                &vec![kernel_lab::poison::POISON; VERIFY_ROWS * PLE_ENTRY_BYTES],
            )?;
            launch(
                session,
                "probe_sm120",
                BASE_CHECKPOINTS,
                (
                    VERIFY_ROWS as u32,
                    PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32,
                    1,
                ),
                &[
                    KernelArgument::Buffer(&input),
                    KernelArgument::Buffer(&tokens),
                    KernelArgument::Buffer(&staged),
                    KernelArgument::Buffer(&baseline),
                    KernelArgument::Buffer(&cu),
                    KernelArgument::Buffer(&q_len),
                    KernelArgument::Buffer(&live),
                ],
            )?;
            let reference = download(session, &baseline, VERIFY_ROWS * PLE_ENTRY_BYTES)?;
            let high = pool_offsets::high_ids(PLE_ENTRY_BYTES as u64, 1, 4);
            let mut placements = vec![(1..=VERIFY_ROWS as u64).rev().collect::<Vec<_>>()];
            if rows == VERIFY_ROWS {
                placements.push(high.alternating(VERIFY_ROWS));
            }
            for ids in placements {
                let mut pool = pool_offsets::SentinelPool::new(
                    session,
                    (ids.iter().max().unwrap() + 1) * PLE_ENTRY_BYTES as u64,
                )?;
                pool.put(session, 0, &committed)?;
                let checkpoint_ids = words(session, &pool_offsets::i32_ids(&ids))?;
                for &id in ids.iter().take(rows) {
                    pool.mark(id * PLE_ENTRY_BYTES as u64, PLE_ENTRY_BYTES as u64);
                }
                let symbol = BASE_CHECKPOINTS
                    .replacen("chop_probe_", "chop_", 1)
                    .replace("_v1", "_indexed_v1");
                let launch = |pdl| {
                    module
                        .function(&symbol)
                        .map_err(|e| e.to_string())?
                        .launch(
                            &session.stream,
                            simple_config(
                                (
                                    VERIFY_ROWS as u32,
                                    PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32,
                                    1,
                                ),
                                (256, 1, 1),
                                0,
                                LaunchConfig::NO_CLUSTER,
                                pdl,
                            )?,
                            &[
                                KernelArgument::Buffer(&input),
                                KernelArgument::Buffer(&tokens),
                                KernelArgument::Buffer(&staged),
                                KernelArgument::Buffer(&pool.buffer),
                                KernelArgument::Buffer(&cu),
                                KernelArgument::Buffer(&q_len),
                                KernelArgument::Buffer(&live),
                                KernelArgument::Buffer(&checkpoint_ids),
                                KernelArgument::U64(ids.iter().max().unwrap() + 1),
                            ],
                        )
                        .map_err(|e| e.to_string())
                };
                let fresh_ids = words(session, &pool_offsets::i32_ids(&ids))?;
                let mut descriptor = Vec::new();
                descriptor.extend_from_slice(
                    &fresh_ids
                        .device_address_at(0, VERIFY_ROWS * 4)
                        .map_err(|e| e.to_string())?
                        .to_ne_bytes(),
                );
                descriptor.extend_from_slice(
                    &checkpoint_ids
                        .device_address_at(0, VERIFY_ROWS * 4)
                        .map_err(|e| e.to_string())?
                        .to_ne_bytes(),
                );
                descriptor.extend_from_slice(&(VERIFY_ROWS as u32).to_ne_bytes());
                descriptor.extend_from_slice(&0_u32.to_ne_bytes());
                let descriptors = upload(session, &descriptor)?;
                let eviction = upload(session, &[0; 16])?;
                let checksum = upload(session, &[kernel_lab::poison::POISON; 128 * 4])?;
                let producer_module = session.module("probe_pdl_prefetch_sm120")?;
                let produce = || {
                    producer_module
                        .function("chop_probe_pdl_prefetch_producer_v1")
                        .map_err(|e| e.to_string())?
                        .launch(
                            &session.stream,
                            simple_config(
                                (1, 1, 1),
                                (128, 1, 1),
                                0,
                                LaunchConfig::NO_CLUSTER,
                                true,
                            )?,
                            &[
                                KernelArgument::Buffer(&eviction),
                                KernelArgument::Buffer(&checksum),
                                KernelArgument::U64(1),
                                KernelArgument::Buffer(&descriptors),
                                KernelArgument::U32(1),
                                KernelArgument::Buffer(&live),
                                KernelArgument::U32(1),
                                KernelArgument::U64(15_000),
                            ],
                        )
                        .map_err(|e| e.to_string())
                };
                let delayed_graph = session.capture(|_| {
                    produce()?;
                    launch(true)
                })?;
                let graph = session.capture(|_| launch(true))?;
                for cycle in 0..6 {
                    // Poison destinations between launches, preserving committed input.
                    for &id in ids.iter().take(rows) {
                        pool.buffer
                            .copy_from_bytes_at(
                                id as usize * PLE_ENTRY_BYTES,
                                &vec![kernel_lab::poison::POISON; PLE_ENTRY_BYTES],
                                &session.stream,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    kernel_lab::poison::poison(session, &[&checksum])?;
                    match cycle {
                        0 | 1 => launch(cycle == 0)?,
                        2 | 3 => graph.launch(&session.stream).map_err(|e| e.to_string())?,
                        4 => {
                            produce()?;
                            launch(true)?;
                        }
                        _ => delayed_graph
                            .launch(&session.stream)
                            .map_err(|e| e.to_string())?,
                    }
                    for kept in 0..=rows {
                        let (entry, expected) = if kept == 0 {
                            (0, committed.as_slice())
                        } else {
                            (
                                ids[kept - 1],
                                &reference[(kept - 1) * PLE_ENTRY_BYTES..kept * PLE_ENTRY_BYTES],
                            )
                        };
                        if pool.read(
                            session,
                            entry * PLE_ENTRY_BYTES as u64,
                            PLE_ENTRY_BYTES as u64,
                        )? != expected
                        {
                            return Err(format!(
                                "indexed checkpoint differs for R{rows} accept{kept} replay{cycle}"
                            ));
                        }
                    }
                    pool.check_untouched(session, "indexed checkpoints")?;
                }
            }
        }
        // Invalid IDs must leave the entire small pool unchanged.
        let bytes = vec![kernel_lab::poison::POISON; 2 * PLE_ENTRY_BYTES];
        let pool = upload(session, &bytes)?;
        let input = upload(session, &halves(&bf16_values(WIDE, 19)))?;
        let tokens = words(session, &[7])?;
        let staged = upload(session, &items().staged[..PLE_ENTRY_BYTES])?;
        let cu = words(session, &[0, 1])?;
        let lengths = words(session, &[1])?;
        let live = words(session, &[1])?;
        let symbol = BASE_CHECKPOINTS
            .replacen("chop_probe_", "chop_", 1)
            .replace("_v1", "_indexed_v1");
        for invalid in [-1, 2, i32::MAX] {
            let ids = words(session, &[invalid; VERIFY_ROWS])?;
            for pdl in [false, true] {
                module
                    .function(&symbol)
                    .map_err(|e| e.to_string())?
                    .launch(
                        &session.stream,
                        simple_config(
                            (
                                VERIFY_ROWS as u32,
                                PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32,
                                1,
                            ),
                            (256, 1, 1),
                            0,
                            LaunchConfig::NO_CLUSTER,
                            pdl,
                        )?,
                        &[
                            KernelArgument::Buffer(&input),
                            KernelArgument::Buffer(&tokens),
                            KernelArgument::Buffer(&staged),
                            KernelArgument::Buffer(&pool),
                            KernelArgument::Buffer(&cu),
                            KernelArgument::Buffer(&lengths),
                            KernelArgument::Buffer(&live),
                            KernelArgument::Buffer(&ids),
                            KernelArgument::U64(2),
                        ],
                    )
                    .map_err(|e| e.to_string())?;
                if download(session, &pool, bytes.len())? != bytes {
                    return Err("invalid checkpoint ID modified the pool".into());
                }
            }
        }
        Ok(())
    })
    .expect("indexed checkpoint commits match copied bytes");
}

/// The PLE anchor (the bundle's speculation cubin, which no lab cubin
/// holds) against the contract's oracle: each anchored prefill item's spare
/// entry holds the state after its row before the anchor, and nothing else
/// changes. Items of 5, 1 and 9 rows follow prefixes of 10, 0 and 3: the
/// first anchors at 13 into spare 1, the second has no anchor, the third
/// anchors after its last row into spare 0; a fourth planned item is dead.
#[test]
fn anchors_keep_the_state_before_the_anchor() {
    kernel_lab::gpu_test(64 << 20, |session| {
        let name = format!(
            "{}_speculation",
            BASE_CHECKPOINTS.split('_').nth(2).unwrap()
        );
        let module = session
            .context
            .load_module(&owned_cubins::load(&name)?)
            .map_err(|e| e.to_string())?;
        let symbol = BASE_CHECKPOINTS
            .replacen("chop_probe_", "chop_", 1)
            .replace("ple_checkpoints", "ple_anchor");
        let (cu, prefix, slots, ends) = ([0, 5, 6], [10, 0, 3], [1, -1, 0], [13, 0, 12]);
        let total = 15;
        let input_values = bf16_values(total * WIDE, 77);
        let tokens = (0..total as u32)
            .map(|i| if i % 4 == 1 { EOS } else { 300 + i })
            .collect::<Vec<_>>();
        let staged = items().staged;
        let mut expected = vec![SENTINEL; 2 * PLE_ENTRY_BYTES];
        ple::ple_anchor(
            &ple::PleAnchor { layout: LAYOUT },
            &input_values,
            &tokens,
            &staged,
            &mut expected,
            &cu,
            &prefix,
            &slots,
            &ends,
            3,
            4,
        )
        .map_err(|e| format!("ple_anchor oracle: {}", e.0))?;
        let input = upload(session, &halves(&input_values))?;
        let token_buffer = upload(
            session,
            &le_words(&tokens, |value| value.to_le_bytes().to_vec()),
        )?;
        let staged = upload(session, &staged)?;
        let anchors = upload(session, &vec![SENTINEL; 2 * PLE_ENTRY_BYTES])?;
        let tables = [&cu[..], &prefix, &slots, &ends].map(|values| words(session, values));
        let [cu, prefix, slots, ends] = tables;
        let (cu, prefix, slots, ends) = (cu?, prefix?, slots?, ends?);
        let live = words(session, &[3])?;
        module
            .function(&symbol)
            .map_err(|e| e.to_string())?
            .launch(
                &session.stream,
                simple_config(
                    (PLE_ENTRY_BYTES.div_ceil(CHUNK) as u32, 4, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&input),
                    KernelArgument::Buffer(&token_buffer),
                    KernelArgument::Buffer(&staged),
                    KernelArgument::Buffer(&anchors),
                    KernelArgument::Buffer(&cu),
                    KernelArgument::Buffer(&prefix),
                    KernelArgument::Buffer(&slots),
                    KernelArgument::Buffer(&ends),
                    KernelArgument::Buffer(&live),
                ],
            )
            .map_err(|e| e.to_string())?;
        if download(session, &anchors, 2 * PLE_ENTRY_BYTES)? != expected {
            return Err("ple_anchor spares differ from the oracle".into());
        }
        Ok(())
    })
    .expect("PLE anchors match the oracle");
}

#[path = "common/owned_cubins.rs"]
mod owned_cubins;

#[test]
fn norm_weights_legacy_consumers() {
    kernel_lab::gpu_test(64 << 20, |session| {
        norm_weight_checks::legacy_rows(session, MTP_NORMS)?;
        norm_weight_checks::legacy_comm(session)
    })
    .unwrap();
}
