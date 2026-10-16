//! The QSA invariance adapter. Every entry runs the bundle's prefill chain
//! over a `Fixture` (key write, indexer query, block scores, top-k and
//! sparse attention) for sequences that continue a cached prefix
//! `write_history` wrote, and compares its own stage: the query rows, the
//! block keys and ring a key write leaves, the score rows of the complete
//! blocks, the selected blocks or the attention rows. Every token's inputs
//! (raw key, index projection, RoPE position, K and V, attention query) come
//! from its sequence's seed and its position only (`token`), so neither a
//! split nor the other sequences of a launch change them.
//!
//! A chunk case prefills one prompt after `PREFIX` cached tokens chunk by
//! chunk, each chunk continuing the ring, block keys and KV of the chunks
//! before it; every row selects among more than `TOP` blocks. Batch units
//! are sequences of one launch, each over its own prefix. Prefill attention
//! takes one split, so its merge claims nothing. The tie case runs the
//! top-k on written scores against the declared rule;
//! the anchor case runs key writes that keep anchor rings in spare entries.
//!
//! A rows case runs Qwen3.8's exact verify step (`EXACT_VERIFY_ROWS`) over
//! the decode chain: one item of R rows ending at each `VERIFY_ENDS`
//! context, launched together at the bundle's verify bucket (key write,
//! index query, decode block scores, top-k, decode attention and its
//! merge), against the same rows as R one-row decode steps in order, each
//! writing its key before it scores, selects and attends (the AR sequence).
//! Both arms load the same fixture, so every input matches.
//!
//! Every launch of the chain starts with poisoned outputs (query, scores,
//! selection, attention and the split partials) and poisoned write-only
//! key-write state (the ring rows and block keys it writes without reading,
//! the anchor rings), and is checked afterwards against the write maps of
//! sparse_indexer.cuh and sparse.cuh (`output_maps`, `key_maps`,
//! `anchor_rows`): every live element is written, and finite where it is
//! arithmetic; rows past the live count, rows without scores and every other
//! state byte keep their image.

use super::*;
use crate::invariance::{Adapter, Compared, Element, Units, written, written_over};
use kernel_core::properties::Anchors;
use kernel_lab::poison::POISON;
use kernels_sm120::invariance::{Declared, OpKind};
use model_qwen38_contract::sm120::{attention as model_attention, config as model_config};
use std::ops::Range;

/// The largest context of every fixture: the plans, the score stride and
/// the cos/sin table are the same in every case.
const CONTEXT: usize = 6144;
/// Cached tokens before a chunk case's prompt.
const PREFIX: usize = 4096;
/// Contexts a rows case's verify item ends at: the first sparse row (2,052
/// tokens) after dense ones, and a block completed inside the item before
/// a row on a new page (4,097).
const VERIFY_ENDS: [usize; 2] = [2052, 4097];
/// Prompt rows of each batch unit, after 2,100 to 5,099 cached tokens.
const UNIT_ROWS: usize = 96;
/// The bundle's prefill granule: anchored prompts start on a multiple and
/// keep their anchor at the last multiple before their end.
const ANCHOR_GRANULE: usize = 64;
/// How far in after a sequence's first launched row the anchor case anchors
/// for a plain spare claim, which covers every anchor: inside the first
/// ring of rows.
const NEAR_ANCHOR: usize = 4;
/// Bytes of a ring (or anchor ring) row: its raw BF16 key, and its three
/// RoPE positions.
const KEY_ROW: usize = INDEX_DIM * 2;
const ROPE_ROW: usize = 12;

/// The chain stage an entry belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stage {
    Query,
    KeyWrite,
    Scores,
    TopK,
    Attention,
}

/// The stage of `entry` (an attention entry or its merge), the profile and
/// KV dtype whose chain runs it, and the phase of that chain, `None` for
/// the indexer entries every phase runs.
fn stage_of(entry: &Declared) -> LabResult<(Stage, Profile, bool, Option<AttentionPhase>)> {
    let symbol = entry.symbol.as_str();
    let found = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| e.symbol() == symbol || e.merge_symbol().as_deref() == Some(symbol))
        .ok_or_else(|| format!("{symbol}: no QSA entry"))?;
    let (stage, phase) = match (found.kind, &found.query) {
        (Kind::Query, _) => (Stage::Query, None),
        (Kind::KeyWrite, _) => (Stage::KeyWrite, None),
        (Kind::TopK, _) => (Stage::TopK, None),
        // Decode has one score row; prefill tiles have 16 to 64 rows.
        // Verify uses by-run scoring and declares no score invariance.
        (
            Kind::Scores {
                rows,
                by_run: false,
            },
            _,
        ) => (
            Stage::Scores,
            Some(if rows == 1 {
                AttentionPhase::Decode
            } else {
                AttentionPhase::Prefill
            }),
        ),
        (
            Kind::Attention,
            SparseAttentionQuery::SparseAttention {
                phase: phase @ (AttentionPhase::Prefill | AttentionPhase::Decode),
                ..
            },
        ) => (Stage::Attention, Some(*phase)),
        _ => return Err(format!("{symbol}: not a QSA prefill or decode entry")),
    };
    let fp8 = matches!(
        found.query,
        SparseAttentionQuery::SparseAttention {
            kv_dtype: DType::Fp8E4M3,
            ..
        }
    );
    Ok((stage, found.profile, fp8, phase))
}

/// TP4's indexer cubin and `profile`'s attention cubin.
fn modules(session: &Session, profile: Profile) -> LabResult<Modules> {
    if let Some((profile, heads, _)) =
        kernel_lab::lab_shapes::indexer_shapes().find(|s| s.0 == profile)
    {
        return Modules::for_topology(session, lab_indexer_topology(profile, heads));
    }
    if kernel_lab::lab_shapes::sparse_attention_profiles().any(|p| p == profile) {
        return Modules::for_topology(session, lab_attention_topology(profile));
    }
    let shape = model_config::profile_shape(profile)
        .ok_or_else(|| format!("{}: not a Qwen3.8 profile", profile.tag()))?;
    let attention = (profile != model_config::PROFILE)
        .then(|| session.module(&format!("sparse_attention_{}_sm120", profile.tag())))
        .transpose()?;
    Ok(Modules {
        topology: Topology {
            geometry: model_attention::sparse_geometry(profile).ok_or("missing sparse geometry")?,
            indexer_profile: model_config::PROFILE,
            profile,
            ranks: shape.ranks,
            heads: shape.q_heads() as usize,
            lean: false,
        },
        indexer: module(session)?,
        attention,
    })
}

/// A token's generator: its sequence's seed and its position only.
fn token(seed: u64, position: usize) -> Rng {
    let mut z = seed ^ (position as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    Rng((z ^ (z >> 31)) | 1)
}

/// One sequence: its seed, cached tokens and prompt rows.
#[derive(Clone, Copy)]
struct Item {
    seed: u64,
    prefix: usize,
    rows: usize,
}

/// The anchor rings a key write keeps: each sequence's spare entry (-1 for
/// none) and anchor end, on the host and the device, and the spare entries'
/// keys and RoPE positions.
struct Spare {
    named: Vec<i32>,
    anchor_ends: Vec<i32>,
    slots: DeviceBuffer,
    ends: DeviceBuffer,
    keys: DeviceBuffer,
    positions: DeviceBuffer,
}

/// Each prompt row's output of every stage: the indexer query, the scores
/// of its complete blocks, its selected blocks and its attention.
#[derive(Default)]
struct Rows {
    query: Units,
    scores: Units,
    selected: Units,
    out: Units,
}

/// One step of `items` over a fixture holding their history: a prefill,
/// or a decode-chain step whose `fixture.prepared` the case sets.
struct Run {
    fixture: Fixture,
    kernels: Kernels,
    /// Every prompt row; `launch` uploads them range by range.
    all: Step,
    stage: Stage,
}

impl Run {
    /// `items` over the chain of `phase` (prefill or decode), which must
    /// run `entry`.
    fn new(
        session: &Session,
        entry: &Declared,
        items: &[Item],
        phase: AttentionPhase,
    ) -> LabResult<Self> {
        let (stage, profile, fp8, own) = stage_of(entry)?;
        if own.is_some_and(|own| own != phase) {
            return Err(format!(
                "{}: a {own:?} entry, which the {phase:?} chain does not run",
                entry.symbol
            ));
        }
        let modules = modules(session, profile)?;
        let case = Case {
            name: "qsa_invariance".into(),
            phase,
            fp8,
            rounded: false,
            sequences: items.iter().map(|i| (i.prefix, i.rows)).collect(),
            pattern: Pattern::Random,
        };
        let rows = items.iter().map(|i| i.rows).sum::<usize>();
        let capacity = Capacity {
            // A decode step's buffers hold its whole bucket (`Capacity::of`).
            rows: if phase == AttentionPhase::Decode {
                rows.max(prepared_rows(phase, rows) as usize)
            } else {
                rows
            },
            sequences: items.len(),
            pages: items.len() * CONTEXT.div_ceil(PAGE) + 3,
            context: CONTEXT,
        };
        let mut fixture = Fixture::allocate(session, &case, modules.topology, capacity)?;
        fixture.load(session, &case)?;
        let kernels = Kernels::load(&modules, &case)?;
        let all = fill(session, &mut fixture, items)?;
        fixture.write_history(session, &kernels)?;
        Ok(Self {
            fixture,
            kernels,
            all,
            stage,
        })
    }

    /// Runs the chain over prompt rows `range` (whole sequences, or one
    /// sequence's rows after its earlier ones) at `fixture.prepared` rows,
    /// keeping anchor rings in `spare`, checks every buffer the chain writes
    /// and appends the rows' outputs to `rows`. The outputs, the split
    /// partials, the ring rows and block keys the key write writes without
    /// reading and the anchor rings start poisoned.
    fn launch(
        &mut self,
        session: &Session,
        range: Range<usize>,
        spare: Option<&Spare>,
        rows: &mut Rows,
    ) -> LabResult<()> {
        let heads = self.fixture.topology.heads;
        let head_dim = self.fixture.topology.geometry.head_dim as usize;
        let index_heads = self.fixture.topology.geometry.index_heads as usize;
        let top = self.fixture.topology.geometry.top_blocks as usize;
        let width = (index_heads + 1) * INDEX_DIM;
        let all = &self.all;
        self.fixture.step = Step {
            rows: range.len(),
            row_batch: all.row_batch[range.clone()].to_vec(),
            positions: all.positions[range.clone()].to_vec(),
            rope: all.rope[range.clone()].to_vec(),
            projection: all.projection[range.start * width..range.end * width].to_vec(),
            q: all.q[range.start * heads * head_dim..range.end * heads * head_dim].to_vec(),
        };
        let f = &self.fixture;
        let kernels = &self.kernels;
        let stream = &session.stream;
        f.upload_step(session)?;
        poison::write(session, &f.d_q, 0, &bytes_u16(&f.step.q))?;
        poison::poison(
            session,
            &[
                &f.d_q_index,
                &f.d_scores,
                &f.d_selected,
                &f.d_out,
                &f.d_workspace,
            ],
        )?;
        let slots = f.sequences.iter().map(|s| s.slot).collect::<Vec<_>>();
        let spans = spans(&f.step.row_batch, &f.step.positions, &slots);
        let pages = f
            .sequences
            .iter()
            .map(|s| s.pages.as_slice())
            .collect::<Vec<_>>();
        let keys = key_maps(
            &spans,
            &pages,
            f.plane_offsets[2] as usize,
            f.block_stride as usize,
        );
        let rows_of = |bytes: usize| {
            keys.ring
                .iter()
                .map(|&row| row * bytes..(row + 1) * bytes)
                .collect::<Vec<_>>()
        };
        poison::fill(session, &f.d_ring, &rows_of(KEY_ROW), &[POISON])?;
        poison::fill(session, &f.d_ring_positions, &rows_of(ROPE_ROW), &[POISON])?;
        poison::fill(session, &f.d_block_keys, &keys.blocks, &[POISON])?;
        let before = KeyState::read(session, f)?;
        match spare {
            Some(spare) => {
                poison::poison(session, &[&spare.keys, &spare.positions])?;
                key_write(f, stream, kernels, spare)?;
            }
            None => {
                poison::poison(session, &[&f.d_anchor_ring])?;
                f.launch_key_write(stream, kernels, f.prepared)?;
            }
        }
        f.launch_query(stream, kernels)?;
        f.launch_scores(stream, kernels)?;
        f.launch_topk(stream, kernels)?;
        f.launch_attention(stream, kernels)?;
        let query = poison::read(session, &f.d_q_index)?;
        let scores = poison::read(session, &f.d_scores)?;
        let selected = poison::read(session, &f.d_selected)?;
        let out = poison::read(session, &f.d_out)?;
        let workspace = poison::read(session, &f.d_workspace)?;
        let label = |error: String| format!("prompt rows {range:?}: {error}");
        let maps = output_maps(
            &f.step.positions,
            heads,
            head_dim,
            index_heads,
            f.score_stride as usize,
            f.workspace_partials(kernels),
            top,
        )?;
        check_outputs(&maps, &query, &scores, &selected, &out, &workspace).map_err(label)?;
        check_key_write(&keys, &before, &KeyState::read(session, f)?).map_err(label)?;
        match spare {
            Some(spare) => {
                let entries = anchor_rows(&spans, &spare.named, &spare.anchor_ends);
                check_anchors(
                    &entries,
                    &poison::read(session, &spare.keys)?,
                    &poison::read(session, &spare.positions)?,
                )
                .map_err(label)?;
            }
            // No sequence names an anchor entry.
            None => poison::check_within(
                "anchor ring",
                &poison::read(session, &f.d_anchor_ring)?,
                &[],
            )
            .map_err(label)?,
        }
        let (query_row, out_row) = (index_heads * INDEX_DIM * 2, heads * head_dim * 2);
        let stride = f.score_stride as usize * 4;
        for (row, &position) in f.step.positions.iter().enumerate() {
            let complete = (position as usize + 1) / BLOCK;
            // Rows of at most `TOP` blocks keep them all without scores.
            let scored = if complete > top { complete } else { 0 };
            rows.query
                .push(query[row * query_row..][..query_row].to_vec());
            rows.scores
                .push(scores[row * stride..][..scored * 4].to_vec());
            rows.selected
                .push(selected[row * top * 4..][..complete.min(top) * 4].to_vec());
            rows.out.push(out[row * out_row..][..out_row].to_vec());
        }
        Ok(())
    }

    /// Each sequence's key-write state: its block keys in block order, then
    /// its slot's ring keys and ring positions.
    fn state(&self, session: &Session) -> LabResult<Vec<Units>> {
        let f = &self.fixture;
        let blocks = poison::read(session, &f.d_block_keys)?;
        let ring = poison::read(session, &f.d_ring)?;
        let ring_positions = poison::read(session, &f.d_ring_positions)?;
        let (key_rows, rope_rows) = (RING * INDEX_DIM * 2, RING * 12);
        Ok(f.sequences
            .iter()
            .map(|s| {
                let slot = s.slot as usize;
                let mut units = (0..s.total() / BLOCK)
                    .map(|block| {
                        let page = s.pages[block * BLOCK / PAGE] as usize;
                        let at =
                            page * f.block_stride as usize + block % (PAGE / BLOCK) * INDEX_DIM * 2;
                        blocks[at..at + INDEX_DIM * 2].to_vec()
                    })
                    .collect::<Units>();
                units.push(ring[slot * key_rows..][..key_rows].to_vec());
                units.push(ring_positions[slot * rope_rows..][..rope_rows].to_vec());
                units
            })
            .collect())
    }

    /// The rows of the run's own stage.
    fn own(&self, rows: Rows) -> Units {
        match self.stage {
            Stage::Query => rows.query,
            Stage::Scores => rows.scores,
            Stage::TopK => rows.selected,
            Stage::Attention => rows.out,
            Stage::KeyWrite => Vec::new(),
        }
    }
}

/// Replaces every token's inputs of the loaded fixture with `token`'s: raw
/// keys, RoPE positions (H and W offset by the seed), K and V in their
/// pages, and the prompt rows' index projection and attention query.
/// Returns the prompt rows.
fn fill(session: &Session, fixture: &mut Fixture, items: &[Item]) -> LabResult<Step> {
    let heads = fixture.topology.heads;
    let head_dim = fixture.topology.geometry.head_dim as usize;
    let index_heads = fixture.topology.geometry.index_heads as usize;
    let element = fixture.element();
    let (k_stride, v_stride) = (fixture.k_stride as usize, fixture.v_stride as usize);
    // Per plane, the FP8 codes whose scaled values lie in [-2, 2].
    let codes = FP8_SCALES.map(|scale| {
        (0..=255_u8)
            .filter(|&code| {
                let value = SmallFloat::E4M3.decode_f64(code) * f64::from(scale);
                value.is_finite() && value.abs() <= 2.0
            })
            .collect::<Vec<u8>>()
    });
    let kv = |rng: &mut Rng, plane: usize| -> Vec<u8> {
        if element == 1 {
            let codes = &codes[plane];
            (0..head_dim)
                .map(|_| codes[(rng.next() % codes.len() as u64) as usize])
                .collect()
        } else {
            (0..head_dim)
                .flat_map(|_| rng.bf16(2.0).to_ne_bytes())
                .collect()
        }
    };
    let mut k_pages = poison::read(session, &fixture.d_k)?;
    let mut v_pages = poison::read(session, &fixture.d_v)?;
    let mut step = Step::default();
    for (b, (sequence, item)) in fixture.sequences.iter_mut().zip(items).enumerate() {
        let (dh, dw) = ((item.seed % 3) as i32, (2 * (item.seed % 2)) as i32);
        sequence.keys.clear();
        sequence.rope.clear();
        sequence.k.clear();
        sequence.v.clear();
        for t in 0..sequence.total() {
            let mut rng = token(item.seed, t);
            let key = (0..INDEX_DIM).map(|_| rng.bf16(2.0)).collect::<Vec<_>>();
            let rope = [t as i32, t as i32 + dh, t as i32 + dw];
            let (k, v) = (kv(&mut rng, 0), kv(&mut rng, 1));
            let page = sequence.pages[t / PAGE] as usize;
            let within = (t % PAGE) * head_dim * element;
            k_pages[page * k_stride + within..][..k.len()].copy_from_slice(&k);
            v_pages[page * v_stride + within..][..v.len()].copy_from_slice(&v);
            if t >= sequence.cached {
                step.row_batch.push(b as i32);
                step.positions.push(t as i32);
                step.rope.push(rope);
                step.projection
                    .extend((0..index_heads * INDEX_DIM).map(|_| rng.bf16(2.0)));
                step.projection.extend_from_slice(&key);
                step.q.extend((0..heads * head_dim).map(|_| rng.bf16(2.0)));
            }
            sequence.keys.extend(key);
            sequence.rope.push(rope);
            sequence.k.extend(k);
            sequence.v.extend(v);
        }
    }
    step.rows = step.row_batch.len();
    poison::write(session, &fixture.d_k, 0, &k_pages)?;
    poison::write(session, &fixture.d_v, 0, &v_pages)?;
    Ok(step)
}

/// `Fixture::launch_key_write` that also keeps the anchor rings of `spare`.
fn key_write(f: &Fixture, stream: &Stream, kernels: &Kernels, spare: &Spare) -> LabResult<()> {
    let (entry, function) = &kernels.key_write;
    let plan = entry.plan(f.prepared, 1).map_err(|e| e.0.to_string())?;
    function
        .launch(
            stream,
            config(plan)?,
            &[
                KernelArgument::Buffer(&f.d_projection),
                KernelArgument::Buffer(&f.d_k_norm),
                KernelArgument::Buffer(&f.d_rope),
                KernelArgument::Buffer(&f.d_cos_sin),
                KernelArgument::Buffer(&f.d_row_batch),
                KernelArgument::Buffer(&f.d_positions),
                KernelArgument::Buffer(&f.d_live),
                KernelArgument::Buffer(&f.d_slots),
                KernelArgument::Buffer(&f.d_ring),
                KernelArgument::Buffer(&f.d_ring_positions),
                KernelArgument::Buffer(&spare.slots),
                KernelArgument::Buffer(&spare.ends),
                KernelArgument::Buffer(&spare.keys),
                KernelArgument::Buffer(&spare.positions),
                KernelArgument::Buffer(&f.d_tables),
                plane(&f.d_block_keys, f.plane_offsets[2]),
                KernelArgument::U32(f.max_pages),
                KernelArgument::U64(f.block_stride),
                KernelArgument::U32(f.topology.geometry.eps.to_bits()),
            ],
        )
        .map_err(|e| e.to_string())?;
    kernel_lab::launched(&entry.symbol());
    Ok(())
}

/// One sequence's rows of a launch: its index in the batch, its ring slot
/// and the positions [first, end) of its rows, which the packed rows lay
/// out contiguously.
#[derive(Clone, Copy, Debug)]
struct Span {
    batch: usize,
    slot: usize,
    first: usize,
    end: usize,
}

/// The sequences of the packed rows `row_batch` at `positions`, with the
/// ring slots `slots` of the batch.
fn spans(row_batch: &[i32], positions: &[i32], slots: &[i32]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    for (&batch, &position) in row_batch.iter().zip(positions) {
        let (batch, position) = (batch as usize, position as usize);
        match spans.last_mut() {
            Some(span) if span.batch == batch => span.end = position + 1,
            _ => spans.push(Span {
                batch,
                slot: slots[batch] as usize,
                first: position,
                end: position + 1,
            }),
        }
    }
    spans
}

/// The bytes of the score buffer the block scores write for rows at
/// `positions` (sparse_indexer.cuh `block_scores`, SPARSE.md step 3): every
/// complete block of a row with more than `top` of them, F32 at
/// `score_stride` per row; rows of at most `top` blocks skip scoring.
fn scores_map(positions: &[i32], score_stride: usize, top: usize) -> LabResult<WriteMap> {
    WriteMap::of(DType::F32).byte_ranges(positions.iter().enumerate().filter_map(
        |(row, &position)| {
            let complete = (position as usize + 1) / BLOCK;
            (complete > top).then(|| row * score_stride * 4..(row * score_stride + complete) * 4)
        },
    ))
}

/// The selections the top-k writes for rows at `positions`
/// (sparse_indexer.cuh `block_topk`): the first `min(complete, top)` I32
/// entries of each row's `top`.
fn selection_map(positions: &[i32], top: usize) -> LabResult<WriteMap> {
    WriteMap::of(DType::I32).byte_ranges(positions.iter().enumerate().map(|(row, &position)| {
        let complete = (position as usize + 1) / BLOCK;
        row * top * 4..(row * top + complete.min(top)) * 4
    }))
}

/// The bytes the chain writes into its outputs for the live rows.
struct OutputMaps {
    query: WriteMap,
    scores: WriteMap,
    selected: WriteMap,
    out: WriteMap,
    workspace: WriteMap,
}

/// The output maps of live rows at `positions` (SPARSE.md, sparse_indexer.cuh
/// and sparse.cuh): each row's indexer query (`INDEX_HEADS` BF16 rows of
/// `INDEX_DIM`) and attention (`heads` BF16 rows of `head_dim`), its scores
/// (`scores_map`) and selections (`selection_map`); rows past the live count
/// stay untouched. With more than one split the attention writes every live
/// row's F32 partials, `partials` bytes packed by row (sparse.cuh
/// `partial_index`; an empty partial holds a -inf maximum), and the merge
/// the output; with one split it writes the output and no partial.
fn output_maps(
    positions: &[i32],
    heads: usize,
    head_dim: usize,
    index_heads: usize,
    score_stride: usize,
    partials: usize,
    top: usize,
) -> LabResult<OutputMaps> {
    let rows = positions.len();
    Ok(OutputMaps {
        query: WriteMap::of(DType::Bf16).prefix(rows * index_heads * INDEX_DIM)?,
        scores: scores_map(positions, score_stride, top)?,
        selected: selection_map(positions, top)?,
        out: WriteMap::of(DType::Bf16).prefix(rows * heads * head_dim)?,
        workspace: WriteMap::of(DType::F32).byte_prefix(partials)?,
    })
}

/// `written_over`, or with an empty map (a launch without row scores, a
/// step that completes no block), every byte kept its image.
fn written_where(
    label: &str,
    before: &[u8],
    after: &[u8],
    map: &WriteMap,
    element: Element,
) -> Result<(), String> {
    if map.written_bytes() > 0 {
        return written_over(label, before, after, map, element);
    }
    if before.len() != after.len() {
        return Err(format!(
            "{label}: {} bytes after the launch, {} before",
            after.len(),
            before.len()
        ));
    }
    match before.iter().zip(after).position(|(was, is)| was != is) {
        Some(at) => Err(format!(
            "{label}: byte {at} changed from {:#04x} to {:#04x}; the launch writes nothing there",
            before[at], after[at]
        )),
        None => Ok(()),
    }
}

/// Checks the chain's outputs after a launch that found them poisoned:
/// each element of `maps` written, and finite but for the selections and
/// the partials (whose empty ones hold a -inf maximum); every other byte
/// still poison.
fn check_outputs(
    maps: &OutputMaps,
    query: &[u8],
    scores: &[u8],
    selected: &[u8],
    out: &[u8],
    workspace: &[u8],
) -> Result<(), String> {
    for (label, bytes, map, element) in [
        ("query", query, &maps.query, Element::Bf16),
        ("scores", scores, &maps.scores, Element::F32),
        ("selection", selected, &maps.selected, Element::Raw(4)),
        ("attention", out, &maps.out, Element::Bf16),
        (
            "attention workspace",
            workspace,
            &maps.workspace,
            Element::Raw(4),
        ),
    ] {
        written_where(label, &vec![POISON; bytes.len()], bytes, map, element)?;
    }
    Ok(())
}

/// The key-write state a launch writes (sparse_indexer.cuh
/// `indexer_key_write`, SPARSE.md step 2), in rows of its slots' rings
/// (`slot * RING + position % RING`) and byte ranges of the block keys.
#[derive(Debug, Default)]
struct KeyMaps {
    /// Ring rows of each sequence's last `RING` positions that the launch
    /// writes without reading: poisoned before it.
    ring: Vec<usize>,
    /// Ring rows the leader reads (the earlier tokens of a block begun before
    /// the launch that the launch completes, and that block's first RoPE
    /// position) and then rewrites: never poisoned, and must change.
    rewritten: Vec<usize>,
    /// The block-key rows of every block the launch completes (the leader
    /// pools one begun before the launch), poisoned before it.
    blocks: Vec<Range<usize>>,
}

/// The key-write maps of `spans`: each sequence's physical pages `pages`
/// (by batch) place its block keys at `base + page * stride` plus the
/// block's row of its page.
fn key_maps(spans: &[Span], pages: &[&[i32]], base: usize, stride: usize) -> KeyMaps {
    let mut maps = KeyMaps::default();
    for span in spans {
        let ring_row = |position: usize| span.slot * RING + position % RING;
        let start = span.first - span.first % BLOCK;
        let read = if start < span.first && start + BLOCK - 1 < span.end {
            (start..span.first).map(ring_row).collect()
        } else {
            Vec::new()
        };
        for row in (span.end.saturating_sub(RING).max(span.first)..span.end).map(ring_row) {
            if read.contains(&row) {
                maps.rewritten.push(row);
            } else {
                maps.ring.push(row);
            }
        }
        for position in (span.first..span.end).filter(|position| position % BLOCK == BLOCK - 1) {
            let block = position / BLOCK;
            let page = pages[span.batch][block * BLOCK / PAGE] as usize;
            let at = base + page * stride + block % (PAGE / BLOCK) * INDEX_DIM * 2;
            maps.blocks.push(at..at + INDEX_DIM * 2);
        }
    }
    maps
}

/// The key-write state around a launch: ring keys, ring RoPE positions and
/// block keys.
#[derive(Clone)]
struct KeyState {
    ring: Vec<u8>,
    positions: Vec<u8>,
    blocks: Vec<u8>,
}

impl KeyState {
    fn read(session: &Session, f: &Fixture) -> LabResult<Self> {
        Ok(Self {
            ring: poison::read(session, &f.d_ring)?,
            positions: poison::read(session, &f.d_ring_positions)?,
            blocks: poison::read(session, &f.d_block_keys)?,
        })
    }
}

/// Checks a ring of `row_bytes` rows against its image `before` the launch:
/// each row of `rewritten` changed, each of `written` (poisoned before) is
/// written, and every other byte kept its image.
fn ring_written(
    label: &str,
    before: &[u8],
    after: &[u8],
    row_bytes: usize,
    element: Element,
    written: &[usize],
    rewritten: &[usize],
) -> Result<(), String> {
    let mut kept = before.to_vec();
    for &row in rewritten {
        let range = row * row_bytes..(row + 1) * row_bytes;
        if range.end > before.len().min(after.len()) {
            return Err(format!("{label}: row {row} lies outside the ring"));
        }
        if before[range.clone()] == after[range.clone()] {
            return Err(format!(
                "{label}: row {row}, read and rewritten, kept its bytes; the launch did not write it"
            ));
        }
        kept[range.clone()].copy_from_slice(&after[range]);
    }
    let map = element.write_map()?.byte_ranges(
        written
            .iter()
            .map(|&row| row * row_bytes..(row + 1) * row_bytes),
    )?;
    written_where(label, &kept, after, &map, element)
}

/// Checks the key-write state after a launch against `maps` and its image
/// `before` the launch: ring keys (BF16) and ring positions (I32) by
/// `ring_written`, block keys (BF16) written where `maps.blocks` says and
/// unchanged elsewhere.
fn check_key_write(maps: &KeyMaps, before: &KeyState, after: &KeyState) -> Result<(), String> {
    ring_written(
        "ring keys",
        &before.ring,
        &after.ring,
        KEY_ROW,
        Element::Bf16,
        &maps.ring,
        &maps.rewritten,
    )?;
    ring_written(
        "ring positions",
        &before.positions,
        &after.positions,
        ROPE_ROW,
        Element::Raw(4),
        &maps.ring,
        &maps.rewritten,
    )?;
    written_where(
        "block keys",
        &before.blocks,
        &after.blocks,
        &WriteMap::of(DType::Bf16).byte_ranges(maps.blocks.iter().cloned())?,
        Element::Bf16,
    )
}

/// The anchor-ring rows a key write over `spans` writes
/// (sparse_indexer.cuh `indexer_key_write`): for a sequence naming entry
/// `named[batch]` >= 0, the rows `entry * RING + p % RING` of its positions
/// p in [anchor_end - RING, anchor_end) that lie in its rows of the launch.
fn anchor_rows(spans: &[Span], named: &[i32], anchor_ends: &[i32]) -> Vec<usize> {
    spans
        .iter()
        .filter(|span| named[span.batch] >= 0)
        .flat_map(|span| {
            let (entry, end) = (named[span.batch] as usize, anchor_ends[span.batch] as usize);
            (span.first.max(end.saturating_sub(RING))..end.min(span.end))
                .map(move |position| entry * RING + position % RING)
        })
        .collect()
}

/// Checks the anchor rings after a launch that found them poisoned: the
/// rows `rows` of the keys (BF16) and RoPE positions (I32) written, every
/// other byte, such as an entry no sequence names, still poison.
fn check_anchors(rows: &[usize], keys: &[u8], positions: &[u8]) -> Result<(), String> {
    written(
        "anchor ring keys",
        keys,
        &WriteMap::of(DType::Bf16).rows(INDEX_DIM, rows.iter().copied())?,
        Element::Bf16,
    )?;
    written(
        "anchor ring positions",
        positions,
        &WriteMap::of(DType::I32).rows(ROPE_ROW / 4, rows.iter().copied())?,
        Element::Raw(4),
    )
}

/// The declared tie rule: the `top` highest scores, equal scores by lower
/// block, listed by ascending block (all blocks when there are fewer).
fn lower_index_top(scores: &[f32], top: usize) -> Vec<i32> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(top);
    order.sort_unstable();
    order.into_iter().map(|b| b as i32).collect()
}

/// Each row of one verify item of `seeds.len()` rows (seeded by
/// `seeds[0]`) per `VERIFY_ENDS` context over the decode chain, item by
/// item: launched together at the bundle's verify bucket (2 or 4 rows, the
/// same 64 attention splits as AR), or, `alone`, as one-row decode steps in
/// order (bucket 1). A key write gives each item's block keys and ring
/// instead.
fn verify_rows(
    session: &Session,
    entry: &Declared,
    seeds: &[u64],
    alone: bool,
) -> LabResult<Units> {
    let rows = seeds.len();
    let mut units = Units::new();
    for end in VERIFY_ENDS {
        let item = Item {
            seed: seeds[0],
            prefix: end - rows,
            rows,
        };
        let mut run = Run::new(session, entry, &[item], AttentionPhase::Decode)?;
        let mut out = Rows::default();
        if alone {
            run.fixture.prepared = 1;
            for row in 0..rows {
                run.launch(session, row..row + 1, None, &mut out)?;
            }
        } else {
            run.fixture.prepared = (rows as u32).next_power_of_two().max(2);
            run.launch(session, 0..rows, None, &mut out)?;
        }
        if run.stage == Stage::KeyWrite {
            units.extend(run.state(session)?.remove(0));
        } else {
            // A row of at most `TOP` blocks writes no scores (the write
            // check holds its row untouched), so its unit compares nothing
            // and is left out; both arms have the same such rows.
            units.extend(run.own(out).into_iter().filter(|unit| !unit.is_empty()));
        }
    }
    Ok(units)
}

#[derive(Default)]
pub(crate) struct Sparse;

impl Adapter for Sparse {
    fn kind(&self) -> OpKind {
        OpKind::SparseAttention
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        let item = Item {
            seed: 0x6368_756e_6b73,
            prefix: PREFIX,
            rows: rows as usize,
        };
        let mut run = Run::new(session, entry, &[item], AttentionPhase::Prefill)?;
        let mut out = Rows::default();
        let mut start = 0;
        for &end in ends {
            run.launch(session, start..end as usize, None, &mut out)?;
            start = end as usize;
        }
        if run.stage == Stage::KeyWrite {
            return Ok(run.state(session)?.remove(0));
        }
        Ok(run.own(out))
    }

    /// Each unit is a sequence of `UNIT_ROWS` rows over a prefix of its own
    /// length: its rows' outputs, or a key write's state. The lengths follow
    /// the units' places, not their seeds: `Fixture::load` draws the shared
    /// norm weights after per-token values whose count the lengths set.
    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        let items = seeds
            .iter()
            .enumerate()
            .map(|(place, &seed)| Item {
                seed,
                prefix: 2100 + 731 * place,
                rows: UNIT_ROWS,
            })
            .collect::<Vec<_>>();
        let mut run = Run::new(session, entry, &items, AttentionPhase::Prefill)?;
        let mut out = Rows::default();
        run.launch(session, 0..run.all.rows, None, &mut out)?;
        if run.stage == Stage::KeyWrite {
            return Ok(run
                .state(session)?
                .into_iter()
                .map(|units| units.concat())
                .collect());
        }
        Ok(run
            .own(out)
            .chunks(UNIT_ROWS)
            .map(|rows| rows.concat())
            .collect())
    }

    /// The rows of one verify item launched together (`verify_rows`).
    fn rows(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        verify_rows(&sessions[0], entry, seeds, false)
    }

    /// The same rows as one-row decode steps in order (`verify_rows`).
    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        verify_rows(&sessions[0], entry, seeds, true)
    }

    /// The top-k on written scores with exact ties, against the declared
    /// rule. Rows with at most `top` blocks keep them all; past that the
    /// kernel keeps the blocks above the `top`-th largest score T, then the
    /// lowest-index blocks equal to T, written by ascending block.
    fn ties(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let session = &sessions[0];
        let (stage, profile, ..) = stage_of(entry)?;
        if stage != Stage::TopK {
            return Err(format!("{}: not the top-k", entry.symbol));
        }
        let modules = modules(session, profile)?;
        let geometry = modules.topology.geometry;
        let stride = geometry.max_blocks() as usize;
        let top = geometry.top_blocks as usize;
        let mut rng = Rng(0x7135_5eed);
        let rows: Vec<(&str, Vec<f32>)> = vec![
            (
                "eight levels",
                (0..2000).map(|_| (rng.next() % 8) as f32 / 8.0).collect(),
            ),
            (
                "even blocks above odd ones",
                (0..2000)
                    .map(|b| if b % 2 == 0 { 1.0 } else { 0.5 })
                    .collect(),
            ),
            ("all equal", vec![0.5; 3000]),
            ("all zero", vec![0.0; 513]),
            // The tied bin outgrows the gathered bin.
            (
                "a few above 19,990 equal",
                (0..20_000)
                    .map(|b| if b % 2001 == 7 { 2.0 } else { 1.0 })
                    .collect(),
            ),
            (
                "64 levels over every block",
                (0..65_536)
                    .map(|_| (rng.next() % 64) as f32 / 16.0)
                    .collect(),
            ),
            (
                "fewer blocks than the top",
                (0..300.min(top - 1))
                    .map(|_| (rng.next() % 4) as f32)
                    .collect(),
            ),
        ];
        let mut scores = vec![0_f32; rows.len() * stride];
        for (row, (_, values)) in rows.iter().enumerate() {
            scores[row * stride..][..values.len()].copy_from_slice(values);
        }
        let positions = rows
            .iter()
            .map(|(_, values)| (values.len() * BLOCK - 1) as i32)
            .collect::<Vec<_>>();
        let d_scores = upload(session, &bytes_f32(&scores))?;
        let d_positions = upload(session, &bytes_i32(&positions))?;
        let d_live = upload(session, &(rows.len() as u32).to_ne_bytes())?;
        let d_selected = upload(session, &vec![POISON; rows.len() * top * 4])?;
        launch_entry(
            session,
            &modules.indexer,
            &modules.topology.indexer("block_topk"),
            rows.len() as u32,
            geometry.max_context,
            &[
                KernelArgument::Buffer(&d_scores),
                KernelArgument::Buffer(&d_positions),
                KernelArgument::Buffer(&d_live),
                KernelArgument::Buffer(&d_selected),
                KernelArgument::U32(stride as u32),
            ],
        )?;
        let selected = poison::read(session, &d_selected)?;
        written(
            "tie selection",
            &selected,
            &selection_map(&positions, top)?,
            Element::Raw(4),
        )?;
        Ok(rows
            .iter()
            .enumerate()
            .map(|(row, (label, values))| {
                let expected = bytes_i32(&lower_index_top(values, top));
                Compared {
                    label: format!("ties: {label}"),
                    got: selected[row * top * 4..][..expected.len()].to_vec(),
                    expected,
                }
            })
            .collect())
    }

    /// Sequences starting on a multiple of `ANCHOR_GRANULE` with unaligned
    /// ends, all but the last anchored: four at their last whole granule,
    /// as the bundle plans anchors, and two at the nearest anchor the claim
    /// covers (past its span, or `NEAR_ANCHOR` rows in for a plain spare
    /// claim), one inside a longer prompt and one ending there, whose spare
    /// rings reach back into their cached prefixes. Each spare entry must hold the ring a prefill
    /// ending at the anchor leaves; the outputs and every slot's ring and
    /// block keys must equal those of the prefill without anchors, and the
    /// spare entry no sequence names must keep its poison. Every launch
    /// checks its anchor rings (`Run::launch`, `check_anchors`): the named
    /// entries' rows written, every other byte still poison.
    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let session = &sessions[0];
        if stage_of(entry)?.0 != Stage::KeyWrite {
            return Err(format!("{}: not the key write", entry.symbol));
        }
        let span = match entry.invariance.anchors {
            Anchors::SpareAfter { rows } => rows as usize,
            Anchors::Spare => NEAR_ANCHOR,
            other => {
                return Err(format!(
                    "{}: keeps no spare entry ({other:?})",
                    entry.symbol
                ));
            }
        };
        let last = |rows: usize| Some(rows / ANCHOR_GRANULE * ANCHOR_GRANULE);
        // (prefix, rows, rows before the anchor)
        let shapes = [
            (4096, 69, last(69)),
            (2048, 130, last(130)),
            (1024, 455, last(455)),
            (0, 200, last(200)),
            (5120, 40, Some(span)),
            (5632, span, Some(span)),
            (3072, 100, None),
        ];
        let items = |truncated: bool| {
            shapes
                .iter()
                .enumerate()
                .map(|(i, &(prefix, rows, anchor))| Item {
                    seed: 0xa4c0 + i as u64,
                    prefix,
                    rows: match anchor {
                        Some(anchor) if truncated => anchor,
                        _ => rows,
                    },
                })
                .collect::<Vec<_>>()
        };
        let mut spares = 0;
        let slots = shapes
            .iter()
            .map(|&(_, _, anchor)| {
                spares += i32::from(anchor.is_some());
                if anchor.is_some() { spares - 1 } else { -1 }
            })
            .collect::<Vec<_>>();
        let ends = shapes
            .iter()
            .map(|&(prefix, rows, anchor)| (prefix + anchor.unwrap_or(rows)) as i32)
            .collect::<Vec<_>>();
        // One more spare entry than the anchored sequences name.
        let entries = spares as usize + 1;
        let (key_rows, rope_rows) = (RING * INDEX_DIM * 2, RING * 12);
        let spare = Spare {
            slots: upload(session, &bytes_i32(&slots))?,
            ends: upload(session, &bytes_i32(&ends))?,
            keys: upload(session, &vec![POISON; entries * key_rows])?,
            positions: upload(session, &vec![POISON; entries * rope_rows])?,
            named: slots.clone(),
            anchor_ends: ends.clone(),
        };
        let prefill = |items: &[Item], spare: Option<&Spare>| -> LabResult<(Rows, Vec<Units>)> {
            let mut run = Run::new(session, entry, items, AttentionPhase::Prefill)?;
            let mut rows = Rows::default();
            run.launch(session, 0..run.all.rows, spare, &mut rows)?;
            Ok((rows, run.state(session)?))
        };
        let (with, with_state) = prefill(&items(false), Some(&spare))?;
        let (without, without_state) = prefill(&items(false), None)?;
        let (_, truncated) = prefill(&items(true), None)?;
        let keys = download(session, &spare.keys, entries * key_rows)?;
        let positions = download(session, &spare.positions, entries * rope_rows)?;
        let mut compared = [
            ("query", with.query, without.query),
            ("score", with.scores, without.scores),
            ("selection", with.selected, without.selected),
            ("attention", with.out, without.out),
        ]
        .into_iter()
        .map(|(name, got, expected)| Compared {
            label: format!("{name} rows with anchors"),
            got: got.concat(),
            expected: expected.concat(),
        })
        .collect::<Vec<_>>();
        for (q, (got, expected)) in with_state.iter().zip(&without_state).enumerate() {
            compared.push(Compared {
                label: format!("sequence {q} block keys and ring with anchors"),
                got: got.concat(),
                expected: expected.concat(),
            });
        }
        let spare_of = |index: usize| {
            [
                keys[index * key_rows..][..key_rows].to_vec(),
                positions[index * rope_rows..][..rope_rows].to_vec(),
            ]
        };
        for (q, &slot) in slots.iter().enumerate().filter(|&(_, &slot)| slot >= 0) {
            let ring = &truncated[q][truncated[q].len() - 2..];
            for (part, (got, expected)) in ["keys", "positions"]
                .into_iter()
                .zip(spare_of(slot as usize).into_iter().zip(ring))
            {
                compared.push(Compared {
                    label: format!("sequence {q} anchor ring {part}"),
                    got,
                    expected: expected.clone(),
                });
            }
        }
        let unnamed = spare_of(spares as usize).concat();
        compared.push(Compared {
            label: "the unnamed spare entry".into(),
            expected: vec![POISON; unnamed.len()],
            got: unnamed,
        });
        Ok(compared)
    }
}

#[test]
fn claimed_sparse_entries_resolve_to_stages() {
    let entries: Vec<_> = crate::invariance::inventory()
        .into_iter()
        .filter(|entry| {
            entry.op == OpKind::SparseAttention && !entry.invariance.claims().is_empty()
        })
        .collect();
    assert!(!entries.is_empty());
    let mut profiles = Vec::new();
    for entry in entries {
        for property in entry.invariance.claims() {
            let (_, profile, _, _) = stage_of(&entry)
                .unwrap_or_else(|error| panic!("{} {property:?}: {error}", entry.symbol));
            profiles.push(profile);
        }
    }
    assert!(profiles.contains(&model_config::PROFILE));
    for (profile, _, _) in kernel_lab::lab_shapes::indexer_shapes() {
        assert!(profiles.contains(&profile), "{}", profile.tag());
    }
    for profile in kernel_lab::lab_shapes::sparse_attention_profiles() {
        assert!(profiles.contains(&profile), "{}", profile.tag());
    }
}

/// The adapter's write maps and checks reject outputs and state the launch
/// left unwritten, NaN, or written outside its map, and accept a correct
/// image (no GPU).
#[test]
fn sparse_output_maps_match_every_lab_head_width() {
    for profile in kernel_lab::lab_shapes::sparse_attention_profiles() {
        let geometry = kernel_lab::lab_shapes::sparse_geometry(profile, 4);
        let maps = output_maps(
            &[2051, 4096],
            geometry.q_heads as usize,
            geometry.head_dim as usize,
            geometry.index_heads as usize,
            1088,
            64,
            geometry.top_blocks as usize,
        )
        .unwrap();
        assert_eq!(
            maps.out.written_bytes(),
            2 * geometry.q_heads as usize * geometry.head_dim as usize * 2
        );
    }
}

#[test]
fn sparse_write_checks_reject_unwritten_outputs() {
    let one = 0x3f80_u16.to_ne_bytes();
    let nan = 0x7fc0_u16.to_ne_bytes();
    let poisoned = |bytes: &[u8]| vec![POISON; bytes.len()];
    for head_dim in [128, 256] {
        // Two live rows of three: 511 complete blocks (no scores) and 513.
        let (positions, heads, stride, partials) = ([2046, 2051], 1, 520, 64);
        let maps = output_maps(
            &positions,
            heads,
            head_dim,
            INDEX_HEADS,
            stride,
            partials,
            TOP,
        )
        .unwrap();
        let fill = |bytes: usize, map: &WriteMap, element: &[u8]| {
            let mut image = vec![POISON; bytes];
            for range in map.ranges() {
                for at in range.clone().step_by(element.len()) {
                    image[at..at + element.len()].copy_from_slice(element);
                }
            }
            image
        };
        let query = fill(3 * INDEX_HEADS * INDEX_DIM * 2, &maps.query, &one);
        let scores = fill(3 * stride * 4, &maps.scores, &1.5_f32.to_ne_bytes());
        let selected = fill(3 * TOP * 4, &maps.selected, &7_i32.to_ne_bytes());
        let out = fill(3 * heads * head_dim * 2, &maps.out, &one);
        // An empty partial's -inf maximum is written.
        let workspace = fill(96, &maps.workspace, &f32::NEG_INFINITY.to_ne_bytes());
        let check = |query: &[u8], scores: &[u8], selected: &[u8], out: &[u8], workspace: &[u8]| {
            check_outputs(&maps, query, scores, selected, out, workspace)
        };
        assert!(check(&query, &scores, &selected, &out, &workspace).is_ok());
        // Nothing written: every buffer still all poison.
        assert!(check(&poisoned(&query), &scores, &selected, &out, &workspace).is_err());
        assert!(check(&query, &poisoned(&scores), &selected, &out, &workspace).is_err());
        assert!(check(&query, &scores, &poisoned(&selected), &out, &workspace).is_err());
        assert!(check(&query, &scores, &selected, &poisoned(&out), &workspace).is_err());
        assert!(check(&query, &scores, &selected, &out, &poisoned(&workspace)).is_err());
        // A hole: the second row's last selection, a partial word.
        let mut hole = selected.clone();
        hole[(TOP + TOP - 1) * 4..(2 * TOP) * 4].fill(POISON);
        assert!(check(&query, &scores, &hole, &out, &workspace).is_err());
        let mut hole = workspace.clone();
        hole[60..64].fill(POISON);
        assert!(check(&query, &scores, &selected, &out, &hole).is_err());
        // NaN in the arithmetic outputs.
        let mut bad = out.clone();
        bad[2..4].copy_from_slice(&nan);
        assert!(check(&query, &scores, &selected, &bad, &workspace).is_err());
        let mut bad = query.clone();
        bad[..2].copy_from_slice(&nan);
        assert!(check(&bad, &scores, &selected, &out, &workspace).is_err());
        let mut bad = scores.clone();
        bad[(stride + 512) * 4..(stride + 513) * 4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(check(&query, &bad, &selected, &out, &workspace).is_err());
        // Writes outside the maps: the third (padding) row, scores of the row
        // without them, a selection past the row's blocks, a partial past the
        // live rows'.
        let mut pad = out.clone();
        pad[2 * head_dim * 2..2 * head_dim * 2 + 2].copy_from_slice(&one);
        assert!(check(&query, &scores, &selected, &pad, &workspace).is_err());
        let mut pad = query.clone();
        pad[2 * INDEX_HEADS * INDEX_DIM * 2] = 0;
        assert!(check(&pad, &scores, &selected, &out, &workspace).is_err());
        let mut pad = scores.clone();
        pad[..4].copy_from_slice(&1.5_f32.to_ne_bytes());
        assert!(check(&query, &pad, &selected, &out, &workspace).is_err());
        let mut pad = selected.clone();
        pad[511 * 4..512 * 4].copy_from_slice(&7_i32.to_ne_bytes());
        assert!(check(&query, &scores, &pad, &out, &workspace).is_err());
        let mut pad = workspace.clone();
        pad[64..68].fill(0);
        assert!(check(&query, &scores, &selected, &out, &pad).is_err());
    }
    // Key writes: sequence 0 (ring slot 1, page 1) continues at 18 to 34,
    // completing block 4, begun before the launch, from ring rows 0 and 1,
    // which it then rewrites; sequence 1 (slot 0, page 0) writes 0 to 5.
    let spans = spans(
        &[[0; 16], [1; 16]].concat()[..21],
        &(18..34).chain(0..5).collect::<Vec<_>>(),
        &[1, 0],
    );
    let pages: [&[i32]; 2] = [&[1], &[0]];
    let stride = BLOCK_PAGE_STRIDE as usize;
    let maps = key_maps(&spans, &pages, 0, stride);
    assert_eq!(maps.rewritten, [RING, RING + 1]);
    assert_eq!(maps.ring.len(), 14 + 5);
    assert_eq!(maps.blocks.len(), 4 + 1);
    // The history: every ring row and the block keys of earlier blocks hold
    // data; the launch poisons the rows it only writes.
    let mut before = KeyState {
        ring: vec![0x11; 2 * RING * KEY_ROW],
        positions: vec![0x11; 2 * RING * ROPE_ROW],
        blocks: vec![POISON; 2 * stride],
    };
    before.blocks[stride..stride + 4 * KEY_ROW].fill(0x11);
    for &row in &maps.ring {
        before.ring[row * KEY_ROW..][..KEY_ROW].fill(POISON);
        before.positions[row * ROPE_ROW..][..ROPE_ROW].fill(POISON);
    }
    let mut after = before.clone();
    for &row in maps.ring.iter().chain(&maps.rewritten) {
        after.ring[row * KEY_ROW..][..KEY_ROW].copy_from_slice(&one.repeat(INDEX_DIM));
        after.positions[row * ROPE_ROW..][..ROPE_ROW].copy_from_slice(&[0x22; 12]);
    }
    for range in &maps.blocks {
        after.blocks[range.clone()].copy_from_slice(&one.repeat(INDEX_DIM));
    }
    assert!(check_key_write(&maps, &before, &after).is_ok());
    // Nothing written.
    assert!(check_key_write(&maps, &before, &before).is_err());
    // A hole in a write-only ring row, a ring position, a block key.
    let mut hole = after.clone();
    hole.ring[(RING + 5) * KEY_ROW..][..2].fill(POISON);
    assert!(check_key_write(&maps, &before, &hole).is_err());
    let mut hole = after.clone();
    hole.positions[(RING + 5) * ROPE_ROW..][..4].fill(POISON);
    assert!(check_key_write(&maps, &before, &hole).is_err());
    let mut hole = after.clone();
    hole.blocks[maps.blocks[0].end - 2..maps.blocks[0].end].fill(POISON);
    assert!(check_key_write(&maps, &before, &hole).is_err());
    // A rewritten ring row left as it was.
    let mut kept = after.clone();
    kept.ring[RING * KEY_ROW..][..KEY_ROW].fill(0x11);
    assert!(check_key_write(&maps, &before, &kept).is_err());
    // A NaN block key.
    let mut bad = after.clone();
    bad.blocks[maps.blocks[1].start..][..2].copy_from_slice(&nan);
    assert!(check_key_write(&maps, &before, &bad).is_err());
    // Writes outside the maps: a ring row of slot 0 past position 5, a
    // block key of a block not completed, a history block key.
    let mut stray = after.clone();
    stray.ring[10 * KEY_ROW..][..2].copy_from_slice(&one);
    assert!(check_key_write(&maps, &before, &stray).is_err());
    let mut stray = after.clone();
    stray.blocks[KEY_ROW..][..2].copy_from_slice(&one);
    assert!(check_key_write(&maps, &before, &stray).is_err());
    let mut stray = after.clone();
    stray.blocks[stride..][..2].copy_from_slice(&one);
    assert!(check_key_write(&maps, &before, &stray).is_err());
    // A write-only row that was not poisoned before the launch.
    let mut unpoisoned = before.clone();
    unpoisoned.ring[(RING + 5) * KEY_ROW..][..KEY_ROW].fill(0x11);
    assert!(check_key_write(&maps, &unpoisoned, &after).is_err());

    // Anchor rings: sequence 0 names entry 1 with its anchor at 32, so the
    // rows of positions 18 to 31; entry 0 (named by no sequence) and the
    // rows of positions 16 and 17 stay poisoned.
    let rows = anchor_rows(&spans, &[1, -1], &[32, 0]);
    assert_eq!(rows, (RING + 2..2 * RING).collect::<Vec<_>>());
    let mut keys = vec![POISON; 2 * RING * KEY_ROW];
    let mut ropes = vec![POISON; 2 * RING * ROPE_ROW];
    for &row in &rows {
        keys[row * KEY_ROW..][..KEY_ROW].copy_from_slice(&one.repeat(INDEX_DIM));
        ropes[row * ROPE_ROW..][..ROPE_ROW].copy_from_slice(&[0x22; 12]);
    }
    assert!(check_anchors(&rows, &keys, &ropes).is_ok());
    assert!(check_anchors(&rows, &poisoned(&keys), &ropes).is_err());
    let mut hole = ropes.clone();
    hole[(RING + 3) * ROPE_ROW..][..4].fill(POISON);
    assert!(check_anchors(&rows, &keys, &hole).is_err());
    let mut bad = keys.clone();
    bad[(RING + 3) * KEY_ROW..][..2].copy_from_slice(&nan);
    assert!(check_anchors(&rows, &bad, &ropes).is_err());
    // The unnamed entry written.
    let mut unnamed = keys.clone();
    unnamed[..2].copy_from_slice(&one);
    assert!(check_anchors(&rows, &unnamed, &ropes).is_err());
}

#[test]
fn kept_block_count_controls_score_and_selection_maps() {
    for top in [256, 512] {
        let positions = [(top * BLOCK - 1) as i32, ((top + 1) * BLOCK - 1) as i32];
        let scores = scores_map(&positions, 1024, top).unwrap();
        assert_eq!(scores.written_bytes(), (top + 1) * 4);
        assert_eq!(scores.ranges().len(), 1);
        assert_eq!(scores.ranges()[0], 4096..4096 + (top + 1) * 4);
        let selected = selection_map(&positions, top).unwrap();
        assert_eq!(selected.written_bytes(), 2 * top * 4);
        assert_eq!(
            lower_index_top(&vec![1.0; top + 1], top),
            (0..top as i32).collect::<Vec<_>>()
        );
    }
}
