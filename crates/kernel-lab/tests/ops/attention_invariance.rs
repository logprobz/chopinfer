//! The dense attention invariance adapter. A prefill entry (a variant's
//! partial or direct entry and its merge) runs one seeded prompt in chunks
//! after the cached prefix of each earlier chunk (`chunks::prefill`); its
//! batch units are items beside other items of one step. An exact verify
//! rows entry (`rows`, `rows_shared`, `rows_merge`) runs one verify item of
//! R rows, and each row alone is the one-row decode step at its position
//! (`exact_rows::one_row`), whose keys hold the rows before it.
//!
//! Every compared launch writes a poisoned BF16 output (the helpers
//! allocate and poison it, and the split workspace, per step), and each
//! output is checked afterwards (`written_rows`): every live token row is
//! written and finite, and an exact rows step's rows past its tokens still
//! hold the poison. A prefill step's output holds exactly its rows. None of
//! the compared entries writes a KV cache: the row engine writes paged rows
//! before attention and ring rows after it, and only the decode
//! `_rope_v1`/`_rope_table_v1` entries, which claim no invariance, fold that
//! write in (attention README).

use super::chunks::{self, Prompt};
use super::exact_rows;
use super::*;
use crate::invariance::{Adapter, Element, Units, written};
use kernels_sm120::invariance::{Declared, OpKind};

/// Rows of a chunk case: rows past 4,096 keys split theirs into three
/// segments.
const PROMPT: u32 = 5000;

/// The variant whose entries include `symbol`, and whether `symbol` is one
/// of its exact verify rows entries (with the shared-key one).
fn variant_of(symbol: &str) -> LabResult<(Variant, Option<bool>)> {
    let v = attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| v.declared_symbols().iter().any(|(s, _)| s == symbol))
        .ok_or_else(|| format!("{symbol}: no attention variant"))?;
    let rows = if symbol == v.rows_shared_symbol() {
        Some(true)
    } else if symbol == v.rows_symbol() || symbol == v.rows_merge_symbol() {
        Some(false)
    } else {
        None
    };
    Ok((v, rows))
}

#[test]
fn claimed_lab_attention_entries_resolve_to_variants() {
    let variants = (kernel_lab::lab_shapes::CATALOG.attention)();
    assert_eq!(variants.len(), 70);
    let inventory = crate::invariance::inventory();
    for variant in variants {
        let mut claimed = 0;
        for (symbol, invariance) in variant.declared_symbols() {
            let entry = inventory.iter().find(|e| e.symbol == symbol).unwrap();
            assert_eq!(entry.invariance, invariance);
            if symbol == variant.rope_symbol() || symbol == variant.rope_table_symbol() {
                assert!(invariance.claims().is_empty());
                continue;
            }
            assert!(
                invariance
                    .claims()
                    .contains(&kernels_sm120::invariance::Property::Batch)
            );
            if variant.query.phase == AttentionPhase::Prefill {
                assert!(
                    invariance
                        .claims()
                        .contains(&kernels_sm120::invariance::Property::Chunk)
                );
            }
            let (resolved, rows) = variant_of(&symbol).unwrap();
            assert_eq!(resolved.name, variant.name);
            assert!(rows.is_none());
            claimed += 1;
        }
        assert_eq!(claimed, 2, "{} partial and merge", variant.name);
    }
}

/// Checks an attention output launched on poison: its first `live` token
/// rows of `row_bytes` BF16 bytes (`q_heads * v_dim` values each) are
/// written and finite, every later byte still holds the poison. The map
/// follows the attention README: each token row of a step writes its
/// heads' outputs (direct, or through its merge), and rows past the step's
/// tokens are never written (`attention_exact_rows_equal_one_row_decode`
/// checks the same padding). The finite check also rejects a merge over
/// unwritten (poisoned) partials.
pub(super) fn written_rows(
    label: &str,
    bytes: &[u8],
    live: usize,
    row_bytes: usize,
) -> Result<(), String> {
    written(
        label,
        bytes,
        &WriteMap::of(DType::Bf16).byte_prefix(live * row_bytes)?,
        Element::Bf16,
    )
}

/// BF16 bits as bytes.
fn bytes_of(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_ne_bytes()).collect()
}

/// A verify item's prefix for `seed`: around the ring size, past a page,
/// and long enough for several stream-K CTAs per row.
fn prefix(seed: u64) -> usize {
    [0, 7, 126, 129, 1017, 12_020, 33_000][(seed % 7) as usize]
}

/// Each row of one verify item of `seeds.len()` rows at `prefix(seeds[0])`
/// through the rows entry, or, `alone`, through one-row decode steps. The
/// rows entry's output holds `EXACT_ROWS` rows at least, poisoned before
/// the launch, of which the item's are live; each one-row step's output is
/// one row, poisoned by `exact_rows::one_row`.
fn verify_rows(
    session: &Session,
    v: Variant,
    shared: bool,
    seeds: &[u64],
    alone: bool,
) -> LabResult<Units> {
    let cubin = module(session, v.profile)?;
    let len = seeds.len();
    let items = [exact_rows::Item {
        prefix: prefix(seeds[0]),
        len,
    }];
    let mut step = exact_rows::step(session, v, &items, seeds[0])?;
    let width = v.query.q_heads as usize * v.query.v_dim as usize;
    let row_bytes = width * 2;
    if alone {
        return exact_rows::positions(&items)
            .into_iter()
            .enumerate()
            .map(|(token, (item, position))| {
                let (mut buffers, arguments) =
                    exact_rows::one_row(session, v, &step, item, token, position)?;
                let one = v
                    .plan_for_bucket(1, exact_rows::counts(&items, 1), arguments)
                    .map_err(|e| e.to_string())?;
                exact_rows::run(session, &cubin, &one, &mut buffers)?;
                let bytes = poison::read(session, &buffers[&OUT])?;
                written_rows(
                    &format!("{} one-row output of token {token}", v.name),
                    &bytes,
                    1,
                    row_bytes,
                )?;
                Ok(bytes)
            })
            .collect();
    }
    let planned = exact_rows::plan_exact(
        v,
        shared,
        len.next_power_of_two().max(2) as u32,
        exact_rows::counts(&items, len),
        step.arguments.clone(),
    )?;
    poison::poison(session, &[&step.buffers[&OUT]])?;
    exact_rows::run(session, &cubin, &planned, &mut step.buffers)?;
    let bytes = poison::read(session, &step.buffers[&OUT])?;
    written_rows(
        &format!("{} rows output", planned.partial_symbol),
        &bytes,
        len,
        row_bytes,
    )?;
    Ok(bytes[..len * row_bytes]
        .chunks(row_bytes)
        .map(<[u8]>::to_vec)
        .collect())
}

#[derive(Default)]
pub(crate) struct Attention;

impl Adapter for Attention {
    fn kind(&self) -> OpKind {
        OpKind::Attention
    }

    fn prompt_rows(&self, _entry: &Declared) -> u32 {
        PROMPT
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        let (v, _) = variant_of(&entry.symbol)?;
        let cubin = module(session, v.profile)?;
        let prompt = Prompt::new(&v, rows as usize, 0x6368_756e_6b73);
        let ends = ends.iter().map(|&end| end as usize).collect::<Vec<_>>();
        let bits = chunks::prefill(session, &cubin, v, &prompt, &prompt, &ends, false)?;
        let row_bytes = (v.query.q_heads * v.query.v_dim) as usize * 2;
        // Each chunk's step holds a poisoned output of exactly the chunk's
        // rows (`chunks::prepare_step`), which `prefill` returns in full:
        // every chunk's rows must be written.
        let bytes = bytes_of(&bits);
        written_rows(
            &format!("{} chunk outputs", entry.symbol),
            &bytes,
            ends.last().copied().unwrap_or(0),
            row_bytes,
        )?;
        Ok(bytes.chunks(row_bytes).map(<[u8]>::to_vec).collect())
    }

    /// Exact verify rows entries: the rows of one item. Phase entries:
    /// items of one step, each its rows' outputs.
    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        let (v, rows) = variant_of(&entry.symbol)?;
        if let Some(shared) = rows {
            return verify_rows(session, v, shared, seeds, false);
        }
        let cubin = module(session, v.profile)?;
        // Each item continues a cached prefix of its own length. Paged
        // decode rows continue long, unequal prefixes, so a step's units
        // outnumber its CTAs and a plan that cut across rows would couple
        // them (decode.cuh `piece_units`).
        let paged_decode =
            v.query.phase == AttentionPhase::Decode && v.layout == CacheLayout::Paged;
        let len = match v.query.phase {
            AttentionPhase::Decode => 1,
            AttentionPhase::Verify => 3,
            AttentionPhase::Prefill => 300,
        };
        let prefix = |seed: u64| {
            if paged_decode {
                1500 + (seed % 14_000) as usize
            } else {
                100 + (seed % 300) as usize
            }
        };
        let prompts = seeds
            .iter()
            .map(|&seed| {
                if paged_decode {
                    Prompt::decoding(&v, prefix(seed) + len, seed, prefix(seed))
                } else {
                    Prompt::new(&v, 700, seed)
                }
            })
            .collect::<Vec<_>>();
        let items = prompts
            .iter()
            .zip(seeds)
            .map(|(prompt, &seed)| chunks::Item {
                prompt,
                prefix: prefix(seed),
                len,
            })
            .collect::<Vec<_>>();
        let step = chunks::prepare_step(session, &cubin, v, &items)?;
        step.launch(session, &session.stream)?;
        // The step's poisoned output holds exactly its items' rows, which
        // `outputs` returns in full, item by item: every row is live.
        let outputs = step.outputs(session)?;
        let row_bytes = (v.query.q_heads * v.query.v_dim) as usize * 2;
        written_rows(
            &format!("{} step output", entry.symbol),
            &bytes_of(&outputs.concat()),
            items.iter().map(|item| item.len).sum(),
            row_bytes,
        )?;
        Ok(outputs.iter().map(|bits| bytes_of(bits)).collect())
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let (v, rows) = variant_of(&entry.symbol)?;
        let shared = rows.ok_or_else(|| format!("{}: not a rows entry", entry.symbol))?;
        verify_rows(&sessions[0], v, shared, seeds, true)
    }
}

/// The output checks the adapter runs after every launch reject an output
/// left poisoned in full or in part, a NaN from a merge over unwritten
/// partials and a write into an exact rows step's padding rows, and accept
/// live rows written in full (no GPU).
#[test]
fn attention_write_checks_reject_unwritten_outputs() {
    let (one, nan) = (0x3f80_u16.to_ne_bytes(), 0x7fc0_u16.to_ne_bytes());
    // Three live token rows of two BF16 values, padded to five rows.
    let row = 4;
    let check = |bytes: &[u8]| written_rows("output", bytes, 3, row);
    let mut output = vec![POISON; 5 * row];
    output[..3 * row].copy_from_slice(&one.repeat(6));
    assert!(check(&output).is_ok());
    // A launch that returned before writing: both arms would compare equal
    // poison.
    assert!(check(&vec![POISON; 5 * row]).is_err());
    // A live value left poisoned.
    let mut hole = output.clone();
    hole[2 * row + 2..3 * row].fill(POISON);
    assert!(check(&hole).is_err());
    // A live value merged from poisoned partials.
    let mut not_finite = output.clone();
    not_finite[row..row + 2].copy_from_slice(&nan);
    assert!(check(&not_finite).is_err());
    // A padding row written.
    let mut padding = output.clone();
    padding[4 * row..4 * row + 2].copy_from_slice(&one);
    assert!(check(&padding).is_err());
    // A prefill step's output, all live: written in full, and with a hole.
    let live = &output[..3 * row];
    assert!(written_rows("step", live, 3, row).is_ok());
    assert!(written_rows("step", &hole[..3 * row], 3, row).is_err());
}
