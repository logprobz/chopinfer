//! Qwen3.8 Flash's linear-attention shapes, capacities and entry selection.

use kernel_build::EntryParam;
use kernel_core::contract::{ConvActivation, GateActivation, RecurrentPhase, RecurrentQuery};
use kernels_sm120::Profile;
use kernels_sm120::invariance::{Anchors, Chunk, Invariance};
use kernels_sm120::linear::{CHUNK, Entry, Stage, conv_params, delta_params};

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}

/// Qwen3.8 Flash per-rank TP4 geometry: 4 of 16 key heads and 12 of 48 value
/// heads, both 128 wide; the convolution covers q, k and v (2560 channels).
/// Other rank counts take `qwen38::RankShape`'s.
pub const QWEN38_KEY_HEADS: u32 = crate::sm120::config::RankShape { ranks: 4 }.linear_key_heads();
pub const QWEN38_VALUE_HEADS: u32 =
    crate::sm120::config::RankShape { ranks: 4 }.linear_value_heads();
pub const QWEN38_HEAD_DIM: u32 = crate::sm120::config::LINEAR_HEAD;
pub const QWEN38_CONV_CHANNELS: u32 = (2 * QWEN38_KEY_HEADS + QWEN38_VALUE_HEADS) * QWEN38_HEAD_DIM;
/// Prefill rows per launch, decode sequences and verify rows per sequence.
/// Decode and verify run one CTA column per sequence, so their capacity is
/// the package maximum of 256 sequences (CONC-QWEN); 32 and fewer plan as
/// before.
pub const QWEN38_PREFILL_ROWS: u32 = crate::contract::PREFILL_TOKENS;
pub const QWEN38_DECODE_SEQUENCES: u32 = crate::contract::DECODE_BATCH;
pub const QWEN38_VERIFY_ROWS: u32 = crate::contract::MAX_VERIFY_ROWS;

/// The claims of a convolution or delta-rule verify entry: an R-row verify
/// launch of one sequence, R at most `qwen38::EXACT_VERIFY_ROWS` (the MTP
/// drafter's K3 steps), gives every row the output, and every checkpoint the
/// state, that R one-row decode steps (`conv_decode`, `delta_decode`) give
/// in turn, each continuing the slot state the one before left. With more
/// than one row, one sequence's rows and checkpoints also do not depend on
/// the other sequences of the launch.
const fn verify_invariance() -> Invariance {
    Invariance::one_row(crate::sm120::config::EXACT_VERIFY_ROWS)
}

/// The claims of a convolution or single-stage delta-rule entry of `phase`.
fn recurrent_invariance(phase: RecurrentPhase) -> Invariance {
    match phase {
        RecurrentPhase::Prefill => Invariance::PREFILL.with_anchors(Anchors::Spare),
        RecurrentPhase::Verify { .. } => verify_invariance(),
        RecurrentPhase::Decode => Invariance::NONE,
    }
}

fn phase_name(phase: RecurrentPhase) -> &'static str {
    match phase {
        RecurrentPhase::Prefill => "prefill",
        RecurrentPhase::Decode => "decode",
        RecurrentPhase::Verify { .. } => "verify",
    }
}

/// Contract keys and ABI of every linear-attention instantiation: TP4's,
/// then TP2's (`qwen38_catalog`).
pub fn catalog() -> Vec<Entry> {
    crate::sm120::PROFILES
        .into_iter()
        .flat_map(qwen38_catalog)
        .collect()
}

pub fn cubin_name(profile: Profile) -> String {
    if super::geometry::owner(profile).is_some_and(|owner| owner.geometry.pinned_names)
        && profile.ranks() == Some(4)
    {
        format!("linear_{}_flash_sm120", profile.tag())
    } else {
        format!("linear_{}_sm120", profile.tag())
    }
}

/// The linear-attention entries of one Qwen3.8 profile at its per-rank key
/// and value heads, symbols `chop_linear_{tag}_...`. Empty for other
/// profiles.
pub fn qwen38_catalog(profile: Profile) -> Vec<Entry> {
    let Some(shape) = super::geometry::shape(profile) else {
        return Vec::new();
    };
    let tag = profile.tag();
    let key_heads = shape.linear_key_heads();
    let value_heads = shape.linear_value_heads();
    let phases = [
        (RecurrentPhase::Prefill, QWEN38_PREFILL_ROWS),
        (RecurrentPhase::Decode, QWEN38_DECODE_SEQUENCES),
        (
            RecurrentPhase::Verify {
                max_rows: QWEN38_VERIFY_ROWS,
            },
            QWEN38_DECODE_SEQUENCES * QWEN38_VERIFY_ROWS,
        ),
    ];
    let chunks = Invariance::PREFILL.with_chunk(Chunk::Granule(CHUNK));
    let mut items = Vec::new();
    for (phase, capacity) in phases {
        items.push(Entry {
            profile,
            symbol: format!("chop_linear_{tag}_conv_{}_v1", phase_name(phase)),
            query: RecurrentQuery::CausalConv1d {
                channels: shape.linear_qkv(),
                kernel: shape.geometry.conv_taps(),
                dilation: 1,
                activation: ConvActivation::Silu,
                phase,
                capacity,
            },
            stage: Stage::Single,
            params: conv_params(phase),
            invariance: recurrent_invariance(phase),
        });
        let query = RecurrentQuery::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim: shape.geometry.linear_head(),
            value_dim: shape.geometry.linear_head(),
            phase,
            capacity,
        };
        let stages: &[(Stage, &str)] = if phase == RecurrentPhase::Prefill {
            &[(Stage::Prepare, "_prepare"), (Stage::Scan, "_scan")]
        } else {
            &[(Stage::Single, "")]
        };
        for &(stage, suffix) in stages {
            items.push(Entry {
                profile,
                symbol: format!("chop_linear_{tag}_delta_{}{suffix}_v1", phase_name(phase)),
                query,
                stage,
                params: delta_params(phase, stage),
                invariance: match stage {
                    Stage::Prepare => chunks,
                    Stage::Scan => chunks.with_anchors(Anchors::Spare),
                    Stage::Single => recurrent_invariance(phase),
                },
            });
        }
    }
    items.push(Entry {
        profile,
        symbol: format!("chop_linear_{tag}_gated_norm_v1"),
        query: RecurrentQuery::GatedRmsNorm {
            width: shape.linear_value(),
            head_dim: shape.geometry.linear_head(),
            eps_bits: shape.geometry.eps().to_bits(),
            gate: GateActivation::Sigmoid,
            capacity: QWEN38_PREFILL_ROWS,
        },
        stage: Stage::Single,
        params: vec![
            ptr("input", "const uint16_t"),
            ptr("gate", "const uint16_t"),
            u32_arg("gate_stride"),
            ptr("weight", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        // Row-local: every row has its one-row bits in a launch of up to the
        // capacity, so in any chunk split and beside any other rows.
        invariance: Invariance::any_rows(QWEN38_PREFILL_ROWS),
    });
    items
}
