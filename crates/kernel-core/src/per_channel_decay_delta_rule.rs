//! The per-channel-decay delta rule: a gated delta rule whose forget gate
//! decays every key channel of every head separately, with its short causal
//! convolutions. A sibling of `contract::RecurrentQuery::GatedDeltaRule`
//! (one decay per head). States live in the slots the runtime assigns to
//! requests and advance per `contract::RecurrentPhase`.
//!
//! The output then goes through `contract::RecurrentQuery::GatedRmsNorm`
//! with the sigmoid gate (width `heads * head_dim`, per-head norm, the BF16
//! output of the gate's low-rank projection), which this module does not
//! repeat.
//!
//! Chunk invariance (docs/numerics.md, "Prefill chunk invariance") is
//! exactness a model needs, not math: a prefill whose chunks start at
//! absolute multiples of its chunk length gives output and final state bits
//! that depend only on the tokens and their positions, never on how the
//! host splits a prompt into calls that end on multiples of that length. A
//! backend reports the length as its `Chunk::Granule` guarantee and a model
//! requests it through `backend::Need`; verify rows equal to decode rows
//! likewise (`Rows::OneRow`).
//!
//! Field classes: every query field is model math (shapes, the conv weight
//! binding, the decay gate, the phase and the capacity). The chunk length
//! and every tile, split and thread count are the backend's choices; head
//! dims and tap counts a backend cannot run are its refusals.

use crate::contract::{
    NumericalContract, NumericalContractId, RecurrentPhase, Refusal, Validate, valid_phase,
};

/// The prefill chunk length today's kernels run, chunks starting at
/// absolute multiples of it. It is a backend choice that models rely on as
/// exactness: a model's prefill granule must be a multiple of the chunk
/// granule its backend guarantees.
pub const DELTA_RULE_CHUNK: u32 = 64;

/// Most heads one entry runs.
pub const MAX_DELTA_RULE_HEADS: u32 = 1024;

/// Most bytes of one slot's state, so `checkpointed_state_bytes` fits a u64
/// for every u32 row count.
pub const MAX_STATE_BYTES: u64 = u32::MAX as u64;

/// The forget gate's form. With `x = f32(f) + dt_bias` per channel (f the
/// BF16 output of the gate's low-rank projection `[heads * head_dim]`,
/// `dt_bias` F32 `[heads * head_dim]`) and `a = exp(A_log[head])` (F32
/// `[heads]`), the per-step log decay g, all F32:
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DecayGate {
    /// `g = lower_bound * sigmoid(a * x)`, in `(lower_bound, 0)`: -5 for
    /// the served model. F32 bits of a finite negative bound.
    BoundedSigmoid { lower_bound_bits: u32 },
    /// `g = -a * softplus(x)` with PyTorch's softplus (x above 20 passes
    /// through, else `log(1 + exp(x))`), unbounded below.
    Softplus,
}

/// How the q, k and v convolutions bind their weights. The math is the
/// same depthwise convolution over `3 * heads * head_dim` channels, q then
/// k then v.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QkvConvs {
    /// One weight `[3 * heads * head_dim, kernel]`.
    Fused,
    /// Three weights `[heads * head_dim, kernel]`, q, k and v.
    Separate,
}

/// Per-channel-decay delta rule contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PerChannelDecayDeltaRuleQuery {
    /// Depthwise causal convolution without bias over the BF16 q, k and v
    /// projections `[rows, 3 * heads * head_dim]`, then SiLU: each output
    /// is `silu(sum_t w[t] * f32(x[t]))` with the BF16 input widened to F32,
    /// the weight in F32 (a BF16 checkpoint weight widens exactly), FMA from
    /// zero in tap order oldest first, SiLU in F32, rounded once to BF16.
    /// Tap 0 multiplies the input `kernel - 1` rows back. The slot state is
    /// the previous `kernel - 1` BF16 inputs `[kernel - 1, 3 * heads *
    /// head_dim]`, oldest first.
    ShortConv {
        heads: u32,
        head_dim: u32,
        kernel: u32,
        convs: QkvConvs,
        phase: RecurrentPhase,
        capacity: u32,
    },
    /// The recurrence per head over the convolved BF16 q, k, v `[rows,
    /// heads, head_dim]`, the BF16 beta logits `[rows, heads]` and the BF16
    /// gate projection output, all F32 unless stated:
    ///
    /// - `q = f32(q) / sqrt(sum(q^2) + 1e-6)`, then `q *= head_dim^-0.5`
    ///   (the scale after the L2 norm); k likewise without the scale;
    /// - `beta = bf16(sigmoid(f32(b)))`, rounded to BF16 as the reference
    ///   computes it;
    /// - g per channel from `gate`;
    /// - with state `S[v][k]` (`[heads, head_dim, head_dim]`, value-major,
    ///   the transpose of a `[k][v]` reference layout): `S[v][k] *= exp(g[k])`,
    ///   `u[v] = sum_k S[v][k] k[k]`, `delta[v] = (v[v] - u[v]) * beta`,
    ///   `S[v][k] += k[k] * delta[v]`, `o[v] = sum_k S[v][k] q[k]`;
    /// - the output `[rows, heads, head_dim]` rounds once to BF16.
    ///
    /// Decode and verify run that recurrence row by row in one fixed order,
    /// so a verify row's output and checkpoint equal decoding that row.
    /// Prefill runs chunks of rows starting at absolute multiples of the
    /// chunk length (the backend's, `DELTA_RULE_CHUNK` for today's
    /// kernels) in any form equal to the recurrence in exact
    /// arithmetic, with one constraint: decay enters only as differences
    /// within a chunk. With `G_i` the F32 running sum of g from the chunk's
    /// first row, the factors are `exp(G_i - G_j)` for `j <= i`, `exp(G_i)`
    /// and `exp(G_last - G_j)`, every exponent at most zero. A cumulative
    /// decay reaches -320 per chunk at a bound of -5 and is unbounded with
    /// softplus, so a factor `exp(-G)` would overflow F32. Chunk products
    /// follow the 128-K chain rule.
    DeltaRule {
        heads: u32,
        head_dim: u32,
        gate: DecayGate,
        phase: RecurrentPhase,
        capacity: u32,
    },
}

/// One per-channel-decay op a model prepares: its short convolution or its
/// delta rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PerChannelDecayOp {
    pub query: PerChannelDecayDeltaRuleQuery,
    /// Whether every launch plans as for the op's prepared items (the
    /// `items` its backend method takes), whatever items it serves, so a
    /// launch does not depend on the step's items; otherwise a launch plans
    /// for `Counts::items`. Either way the per-sequence arrays
    /// (`recurrent::RecurrentSlots`) need hold only `Counts::items`
    /// sequences, which bound the launch's device-live sequences: the
    /// kernels read a per-sequence entry only for a live sequence. A host
    /// that hands one launch the entries of some of its sequences (a
    /// prefill in row parts, each part's entries from its own offset in
    /// shared arrays) passes those sequences.
    pub plan_at_capacity: bool,
}

impl From<PerChannelDecayDeltaRuleQuery> for PerChannelDecayOp {
    /// The op of `query` that plans for each launch's items.
    fn from(query: PerChannelDecayDeltaRuleQuery) -> Self {
        PerChannelDecayOp {
            query,
            plan_at_capacity: false,
        }
    }
}

impl PerChannelDecayDeltaRuleQuery {
    /// Elements of one slot's state: F32 for the delta rule, BF16 for the
    /// convolution. The host reserves `rows + 1` copies per slot
    /// (docs/numerics.md#recurrent-state).
    ///
    /// The size helpers are exact for a validated query (a state of at
    /// most `MAX_STATE_BYTES`), at any u32 row count. On a query that fails
    /// validation they saturate at `u64::MAX` rather than wrap, so a
    /// capacity plan never undercounts.
    pub fn state_elements(&self) -> u64 {
        let product = |factors: &[u64]| {
            factors
                .iter()
                .fold(1_u64, |total, &factor| total.saturating_mul(factor))
        };
        match *self {
            Self::ShortConv {
                heads,
                head_dim,
                kernel,
                ..
            } => product(&[
                u64::from(kernel.saturating_sub(1)),
                3,
                u64::from(heads),
                u64::from(head_dim),
            ]),
            Self::DeltaRule {
                heads, head_dim, ..
            } => product(&[u64::from(heads), u64::from(head_dim), u64::from(head_dim)]),
        }
    }

    /// Bytes of one slot's state.
    pub fn state_bytes(&self) -> u64 {
        let element = match self {
            Self::ShortConv { .. } => 2,
            Self::DeltaRule { .. } => 4,
        };
        self.state_elements().saturating_mul(element)
    }

    /// Bytes a slot reserves for the committed state and one checkpoint per
    /// row an item may run before a commit.
    pub fn checkpointed_state_bytes(&self, rows: u32) -> u64 {
        self.state_bytes().saturating_mul(u64::from(rows) + 1)
    }
}

impl Validate for PerChannelDecayDeltaRuleQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let (heads, head_dim, phase, capacity) = match *self {
            Self::ShortConv {
                heads,
                head_dim,
                kernel,
                phase,
                capacity,
                ..
            } => {
                if kernel < 2 {
                    return Err(Refusal("short convolution kernel must be at least 2"));
                }
                (heads, head_dim, phase, capacity)
            }
            Self::DeltaRule {
                heads,
                head_dim,
                gate,
                phase,
                capacity,
            } => {
                if let DecayGate::BoundedSigmoid { lower_bound_bits } = gate {
                    let bound = f32::from_bits(lower_bound_bits);
                    if !bound.is_finite() || bound >= 0.0 {
                        return Err(Refusal(
                            "the bounded decay gate needs a finite negative lower bound",
                        ));
                    }
                }
                (heads, head_dim, phase, capacity)
            }
        };
        if heads == 0
            || heads > MAX_DELTA_RULE_HEADS
            || !matches!(head_dim, 64 | 128)
            || capacity == 0
            || !valid_phase(phase)
        {
            return Err(Refusal(
                "invalid per-channel-decay delta rule heads, head dim, phase or capacity",
            ));
        }
        if self.state_bytes() > MAX_STATE_BYTES {
            return Err(Refusal(
                "a per-channel-decay delta rule slot state exceeds MAX_STATE_BYTES",
            ));
        }
        Ok(())
    }
}

fn phase_name(phase: RecurrentPhase) -> &'static str {
    match phase {
        RecurrentPhase::Prefill => "prefill",
        RecurrentPhase::Decode => "decode",
        RecurrentPhase::Verify { .. } => "verify",
    }
}

impl NumericalContract for PerChannelDecayDeltaRuleQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match *self {
            Self::ShortConv { phase, .. } => format!(
                "recurrent.causal-conv1d-depthwise-silu-f32-weight.{}.v1",
                phase_name(phase)
            ),
            Self::DeltaRule { gate, phase, .. } => {
                let gate = match gate {
                    DecayGate::BoundedSigmoid { .. } => "bounded-sigmoid",
                    DecayGate::Softplus => "softplus",
                };
                format!(
                    "recurrent.per-channel-decay-delta-rule-{gate}.{}.v1",
                    phase_name(phase)
                )
            }
        };
        Ok(NumericalContractId::new(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHASES: [RecurrentPhase; 3] = [
        RecurrentPhase::Prefill,
        RecurrentPhase::Decode,
        RecurrentPhase::Verify { max_rows: 4 },
    ];

    fn rule(
        head_dim: u32,
        gate: DecayGate,
        phase: RecurrentPhase,
    ) -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads: 16,
            head_dim,
            gate,
            phase,
            capacity: 8192,
        }
    }

    fn conv(convs: QkvConvs, phase: RecurrentPhase) -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: 16,
            head_dim: 128,
            kernel: 4,
            convs,
            phase,
            capacity: 8192,
        }
    }

    const BOUNDED: DecayGate = DecayGate::BoundedSigmoid {
        lower_bound_bits: 0xc0a0_0000, // -5.0
    };

    #[test]
    fn served_and_oracle_only_forms_validate_in_every_phase() {
        assert_eq!(f32::from_bits(0xc0a0_0000), -5.0);
        for phase in PHASES {
            for head_dim in [64, 128] {
                for gate in [BOUNDED, DecayGate::Softplus] {
                    rule(head_dim, gate, phase).numerical_contract_id().unwrap();
                }
            }
            for convs in [QkvConvs::Fused, QkvConvs::Separate] {
                conv(convs, phase).numerical_contract_id().unwrap();
            }
        }
        assert_ne!(
            rule(128, BOUNDED, RecurrentPhase::Decode)
                .numerical_contract_id()
                .unwrap(),
            rule(128, DecayGate::Softplus, RecurrentPhase::Decode)
                .numerical_contract_id()
                .unwrap()
        );
        assert_eq!(
            rule(128, BOUNDED, RecurrentPhase::Prefill)
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "recurrent.per-channel-decay-delta-rule-bounded-sigmoid.prefill.v1"
        );
    }

    #[test]
    fn state_sizes_are_exact() {
        let state = rule(128, BOUNDED, RecurrentPhase::Decode);
        assert_eq!(state.state_elements(), 16 * 128 * 128);
        assert_eq!(state.state_bytes(), 1_048_576);
        let conv = conv(QkvConvs::Fused, RecurrentPhase::Decode);
        assert_eq!(conv.state_elements(), 3 * 3 * 16 * 128);
        assert_eq!(conv.state_bytes(), 36_864);
        // 34 layers per rank: 35.2 MiB per slot, reserved (rows + 1) times.
        let per_slot = 34 * (state.state_bytes() + conv.state_bytes());
        assert_eq!(per_slot, 36_904_960);
        assert_eq!(state.checkpointed_state_bytes(4), 5 * 1_048_576);
        assert_eq!(
            rule(64, BOUNDED, RecurrentPhase::Decode).state_bytes(),
            262_144
        );
    }

    #[test]
    fn invalid_gate_and_geometry_refuse_by_name() {
        let positive = DecayGate::BoundedSigmoid {
            lower_bound_bits: 5.0_f32.to_bits(),
        };
        assert_eq!(
            rule(128, positive, RecurrentPhase::Decode).validate(),
            Err(Refusal(
                "the bounded decay gate needs a finite negative lower bound"
            ))
        );
        assert_eq!(
            rule(256, BOUNDED, RecurrentPhase::Decode).validate(),
            Err(Refusal(
                "invalid per-channel-decay delta rule heads, head dim, phase or capacity"
            ))
        );
        assert!(
            rule(128, BOUNDED, RecurrentPhase::Verify { max_rows: 0 })
                .validate()
                .is_err()
        );
        let PerChannelDecayDeltaRuleQuery::ShortConv {
            heads,
            head_dim,
            convs,
            phase,
            capacity,
            ..
        } = conv(QkvConvs::Separate, RecurrentPhase::Prefill)
        else {
            unreachable!()
        };
        assert_eq!(
            PerChannelDecayDeltaRuleQuery::ShortConv {
                heads,
                head_dim,
                kernel: 1,
                convs,
                phase,
                capacity,
            }
            .validate(),
            Err(Refusal("short convolution kernel must be at least 2"))
        );
    }

    #[test]
    fn checkpoint_reservation_cannot_overflow() {
        let heads = |heads| PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads,
            head_dim: 128,
            gate: BOUNDED,
            phase: RecurrentPhase::Prefill,
            capacity: 262_143,
        };
        // The largest valid state at the largest row count is exact.
        let largest = heads(MAX_DELTA_RULE_HEADS);
        largest.validate().unwrap();
        assert_eq!(largest.state_bytes(), 1 << 26);
        assert_eq!(largest.checkpointed_state_bytes(u32::MAX), 1 << 58);
        // The reviewed case (2^30 heads) and the first head count above the
        // bound are refused, and their sizes saturate instead of wrapping.
        for count in [MAX_DELTA_RULE_HEADS + 1, 1 << 30] {
            assert_eq!(
                heads(count).validate(),
                Err(Refusal(
                    "invalid per-channel-decay delta rule heads, head dim, phase or capacity"
                ))
            );
        }
        assert_eq!(heads(1 << 30).checkpointed_state_bytes(262_143), u64::MAX);
        let conv = PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: MAX_DELTA_RULE_HEADS,
            head_dim: 128,
            kernel: 8,
            convs: QkvConvs::Fused,
            phase: RecurrentPhase::Prefill,
            capacity: 1,
        };
        conv.validate().unwrap();
        assert_eq!(
            conv.checkpointed_state_bytes(u32::MAX),
            7 * 3 * 1024 * 128 * 2 * (1 << 32)
        );
        // Taps are bounded by the state's bytes, not by a kernel's limit.
        let taps = |kernel| PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: MAX_DELTA_RULE_HEADS,
            head_dim: 128,
            kernel,
            convs: QkvConvs::Fused,
            phase: RecurrentPhase::Prefill,
            capacity: 1,
        };
        taps(5462).validate().unwrap();
        assert!(taps(5462).state_bytes() <= MAX_STATE_BYTES);
        assert_eq!(
            taps(5463).validate(),
            Err(Refusal(
                "a per-channel-decay delta rule slot state exceeds MAX_STATE_BYTES"
            ))
        );
        assert_eq!(taps(u32::MAX).checkpointed_state_bytes(u32::MAX), u64::MAX);
    }
}
