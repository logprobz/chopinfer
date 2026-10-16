//! MiMo projection identities and generated capability inventories.

use crate::Tp;
use crate::backend::{Kernels, Selected};
use crate::serving::{DECODE_ENTRY_ROWS, PREFILL_TOKENS};
use kernel_core::contract::{DType, Epilogues, Refusal, ScaleMode, SwigluColumns, WeightFormat};
use kernel_core::precision::capability::{
    CapabilityDocument, CapabilityInventory, OperationCapability,
};
use kernel_core::precision::{ExecutionPhase, MatrixGeometry, OperationUse};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// A dense text projection a MiMo GEMM entry computes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    /// The token embedding: a BF16 table row per token ID, on every rank.
    Embedding,
    /// Attention's fused `qkv_proj` of a global-attention layer, or of a
    /// sliding-window one: the rank's Q heads, then its K and V heads.
    Qkv {
        global: bool,
    },
    /// Attention `o_proj`.
    Output,
    /// A sliding-window layer's `o_proj` where its K differs from the
    /// global layers' (geometries without pinned entry names).
    SlidingOutput,
    /// MTP's `eh_proj` over the normalized embedding and hidden state, where
    /// it is not the `o_proj` shape (geometries without pinned entry names).
    MtpEh,
    /// The MoE router's logits, in F32.
    Router,
    /// `lm_head`, with BF16 logits, or F32 for scored prompt rows.
    Head {
        f32: bool,
    },
    /// The dense MLP's gate and up projections, fused with SwiGLU.
    MlpGateUp,
    MlpDown,
    /// A MoE layer's routed experts' gate and up projections, fused with
    /// SwiGLU, over every expert of the layer.
    ExpertGateUp,
    /// The routed experts' down projection, weighting and combining the
    /// routes.
    ExpertDown,
    /// A DFlash drafter projection over the drafter image, BF16 throughout.
    Draft(Draft),
}

/// The DFlash drafter's projections: the context feature's `fc`, the
/// fused five-layer context K/V, and each draft layer's fused QKV,
/// `o_proj`, fused gate and up, and down.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Draft {
    Fc,
    ContextKv,
    Qkv,
    Output,
    GateUp,
    Down,
}

impl Projection {
    pub const ALL: [Self; 11] = [
        Self::Embedding,
        Self::Qkv { global: true },
        Self::Qkv { global: false },
        Self::Output,
        Self::Router,
        Self::Head { f32: false },
        Self::Head { f32: true },
        Self::MlpGateUp,
        Self::MlpDown,
        Self::ExpertGateUp,
        Self::ExpertDown,
    ];

    /// The projections `geometry` compiles: geometries without pinned entry
    /// names add their own sliding-window output and MTP EH entries.
    pub fn of(geometry: &crate::geometry::Geometry) -> Vec<Self> {
        let mut all = Self::ALL.to_vec();
        if !geometry.pinned_names {
            all.extend([Self::SlidingOutput, Self::MtpEh]);
        }
        all
    }

    /// The entry name in the GEMM catalog.
    fn entry(self) -> &'static str {
        match self {
            Self::Embedding => "embedding_full",
            Self::Qkv { global: true } => "qkv_ga",
            Self::Qkv { global: false } => "qkv_swa",
            Self::Output => "o",
            Self::SlidingOutput => "swa_o",
            Self::MtpEh => "mtp_eh",
            Self::Router => "router",
            Self::Head { f32: false } => "head",
            Self::Head { f32: true } => "head_f32",
            Self::MlpGateUp => "mlp_gate_up",
            Self::MlpDown => "mlp_down",
            Self::ExpertGateUp => "experts_gate_up",
            Self::ExpertDown => "experts_down",
            Self::Draft(Draft::Fc) => "dflash_fc",
            Self::Draft(Draft::ContextKv) => "dflash_context_kv",
            Self::Draft(Draft::Qkv) => "dflash_qkv",
            Self::Draft(Draft::Output) => "dflash_o",
            Self::Draft(Draft::GateUp) => "dflash_gate_up",
            Self::Draft(Draft::Down) => "dflash_down",
        }
    }

    /// The source weight tensors of one projection, by suffix of the
    /// projection's module: one, or gate then up.
    pub fn operands(self) -> &'static [&'static str] {
        match self {
            Self::MlpGateUp | Self::ExpertGateUp | Self::Draft(Draft::GateUp) => {
                &["gate_proj", "up_proj"]
            }
            Self::ExpertDown => &["down_proj"],
            _ => &[""],
        }
    }

    /// The weight encoding its kernels read at `tp`. A rank's QKV is one
    /// source chunk at TP4, with its 128-row block scales; at other rank
    /// counts its heads come from several chunks, with one scale per 64 rows.
    pub fn weight(self, tp: Tp) -> WeightFormat {
        match self {
            Self::Qkv { .. } if tp.ranks() != 4 => WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
            Self::Qkv { .. } | Self::MlpGateUp | Self::MlpDown => {
                WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
            }
            Self::ExpertGateUp | Self::ExpertDown => WeightFormat::Mxfp4,
            _ => WeightFormat::Bf16,
        }
    }

    pub fn output(self) -> DType {
        match self {
            Self::Router | Self::Head { f32: true } => DType::F32,
            _ => DType::Bf16,
        }
    }

    pub fn swiglu(self) -> Option<SwigluColumns> {
        matches!(
            self,
            Self::MlpGateUp | Self::ExpertGateUp | Self::Draft(Draft::GateUp)
        )
        .then_some(SwigluColumns::ConcatenatedHalves)
    }

    /// One rank's matrix per operand at `tp`, from the rank shape.
    pub fn geometry(self, tp: Tp) -> MatrixGeometry {
        let hidden = tp.geometry.hidden() as u64;
        let (rows, k) = match self {
            Self::Embedding => (tp.geometry.vocab() as u64, hidden),
            Self::Qkv { global } => (tp.qkv_rows(global), hidden),
            Self::Output => (hidden, tp.o_proj_k()),
            Self::SlidingOutput => (hidden, tp.o_proj_k_for(false)),
            Self::MtpEh => (hidden, tp.mtp_eh_k()),
            Self::Router => (tp.geometry.experts() as u64, hidden),
            Self::Head { .. } => (tp.vocab(), hidden),
            Self::MlpGateUp => (tp.dense_intermediate(), hidden),
            Self::MlpDown => (hidden, tp.dense_intermediate()),
            Self::ExpertGateUp => (tp.expert_intermediate(), hidden),
            Self::ExpertDown => (hidden, tp.expert_intermediate()),
            Self::Draft(draft) => {
                let drafter = tp.geometry.dflash();
                let (hidden, head) = (drafter.hidden_size as u64, drafter.head_dim as u64);
                let kv = tp.dflash_kv_heads() * head;
                match draft {
                    Draft::Fc => (hidden, tp.dflash_fc_k()),
                    Draft::ContextKv => (drafter.num_hidden_layers as u64 * 2 * kv, hidden),
                    Draft::Qkv => (tp.dflash_qkv_rows(), hidden),
                    Draft::Output => (hidden, tp.dflash_q_heads() * head),
                    Draft::GateUp => (tp.dflash_intermediate(), hidden),
                    Draft::Down => (hidden, tp.dflash_intermediate()),
                }
            }
        };
        MatrixGeometry { rows, k }
    }

    /// Whether it looks rows up by token ID rather than projecting
    /// activations.
    pub fn lookup(self) -> bool {
        self == Self::Embedding
    }

    /// Whether it is grouped over a MoE layer's routed experts.
    pub fn routed(self) -> bool {
        matches!(self, Self::ExpertGateUp | Self::ExpertDown)
    }

    /// Its kernel's epilogues: SwiGLU after a fused gate and up, and the
    /// routes' weighted combine after the experts' down projection.
    pub fn epilogues(self) -> Epilogues {
        Epilogues {
            swiglu: self.swiglu(),
            weighted_moe_combine: self == Self::ExpertDown,
            ..Default::default()
        }
    }

    /// The implementation name of its offers at `tp`.
    pub fn implementation(self, tp: Tp) -> String {
        let kind = if self.routed() {
            "grouped"
        } else if self.lookup() {
            "lookup"
        } else {
            "dense"
        };
        format!("mimo.{kind}.{}.tp{}.v1", self.entry(), tp.ranks())
    }
}

/// The compiled offers of each served declared projection of `geometry`:
/// the one table from declaration IDs to MiMo's kernels. The head offers
/// BF16 and F32 logits; MTP and its adapters run the sliding-window and
/// dense entries.
pub fn served(
    geometry: &crate::geometry::Geometry,
    projection: &str,
) -> Option<&'static [Projection]> {
    let pinned = geometry.pinned_names;
    Some(match projection {
        "embedding" => &[Projection::Embedding],
        "ga.qkv" => &[Projection::Qkv { global: true }],
        "swa.qkv" | "mtp.qkv" | "mtp.attention-adapter.qkv" => &[Projection::Qkv { global: false }],
        "ga.output" => &[Projection::Output],
        "swa.output" | "mtp.output" | "mtp.attention-adapter.output" if !pinned => {
            &[Projection::SlidingOutput]
        }
        "mtp.eh" if !pinned => &[Projection::MtpEh],
        "swa.output" | "mtp.output" | "mtp.attention-adapter.output" | "mtp.eh" => {
            &[Projection::Output]
        }
        "dense.mlp.gate-up" | "mtp-mlp.mlp.gate-up" | "mtp.mlp-adapter.gate-up" => {
            &[Projection::MlpGateUp]
        }
        "dense.down" | "mtp-mlp.down" | "mtp.mlp-adapter.down" => &[Projection::MlpDown],
        "moe.experts.gate-up" => &[Projection::ExpertGateUp],
        "moe.experts.down" => &[Projection::ExpertDown],
        "router" => &[Projection::Router],
        "head" => &[
            Projection::Head { f32: false },
            Projection::Head { f32: true },
        ],
        "dflash.fc" => &[Projection::Draft(Draft::Fc)],
        "dflash.context-kv" => &[Projection::Draft(Draft::ContextKv)],
        "dflash.qkv" => &[Projection::Draft(Draft::Qkv)],
        "dflash.output" => &[Projection::Draft(Draft::Output)],
        "dflash.mlp.gate-up" => &[Projection::Draft(Draft::GateUp)],
        "dflash.down" => &[Projection::Draft(Draft::Down)],
        _ => return None,
    })
}

/// Every phase and token capacity MiMo's dense projections run: prefill, the
/// package's decode graph buckets and its verify buckets (powers of two up
/// to its verify rows).
pub fn uses() -> Vec<OperationUse> {
    use bundle_runtime::geometry::Items;
    let package = Items::package::<crate::serving::Mimo>();
    let verify_rows = bundle_runtime::graphs::verify_tokens::<crate::serving::Mimo>(package);
    let graph = |phase, capacity| OperationUse {
        phase,
        capacity,
        graph: true,
    };
    let mut uses = vec![OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: PREFILL_TOKENS,
        graph: false,
    }];
    uses.extend(
        package
            .buckets()
            .map(|bucket| graph(ExecutionPhase::Decode, bucket)),
    );
    uses.extend((0..=verify_rows.ilog2()).map(|power| graph(ExecutionPhase::Verify, 1 << power)));
    uses
}

/// Select the generated offer with the retained implementation identity.
/// Drafter projections, which the generated inventory does not list, offer
/// their catalog entry.
pub fn capability(
    projection: Projection,
    tp: Tp,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let implementation = projection.implementation(tp);
    if let Projection::Draft(_) = projection {
        let mut capability = Selected::drafter_capability(projection, tp, usage)?;
        capability.implementation = implementation;
        return Ok(capability);
    }
    inventory_for(tp.geometry)?
        .document()
        .operations
        .iter()
        .find(|offer| offer.implementation == implementation && offer.usage == usage)
        .cloned()
        .ok_or(Refusal("missing generated projection capability"))
}

/// Every offer of every projection at every supported rank count.
pub fn document_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<CapabilityDocument, Refusal> {
    Selected::capabilities(geometry)
        .map_err(|_| Refusal("invalid generated projection capability document"))
}

/// The offers of `geometry` as an inventory to bind source operations
/// against. Every stage is an entry of the MiMo cubins the bundle builds
/// from the same catalog; the inventory's identity is the SHA-256 of its
/// stage names.
pub fn inventory_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static CapabilityInventory, Refusal> {
    static INVENTORIES: OnceLock<Vec<Result<CapabilityInventory, Refusal>>> = OnceLock::new();
    let index = crate::geometry::GEOMETRIES
        .iter()
        .position(|g| g.id == geometry.id)
        .ok_or(Refusal("unlisted geometry"))?;
    INVENTORIES.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                let document = document_for(g)?;
                let available = document.entries();
                let mut digest = Sha256::new();
                for stage in &available {
                    digest.update((stage.len() as u64).to_le_bytes());
                    digest.update(stage.as_bytes());
                }
                CapabilityInventory::new(document, available, format!("{:x}", digest.finalize()))
                    .map_err(|_| Refusal("invalid MiMo capability inventory"))
            })
            .collect()
    })[index]
        .as_ref()
        .map_err(|error| *error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The uses are the serving selection's buckets: prefill, every decode
    /// bucket of the package maximum and every verify bucket, wide ones
    /// included, each offered at its own phase and capacity.
    #[test]
    fn uses_cover_every_serving_bucket() {
        use crate::serving::Mimo;
        use bundle_runtime::graphs::{decode_bucket, verify_bucket};
        let served = |bucket: fn(u32) -> Option<u32>| -> BTreeSet<u32> {
            (1..=4096).filter_map(bucket).collect()
        };
        let declared = |phase| -> BTreeSet<u32> {
            uses()
                .into_iter()
                .filter(|usage| usage.phase == phase)
                .map(|usage| usage.capacity)
                .collect()
        };
        let decode = served(decode_bucket::<Mimo>);
        let verify = served(verify_bucket::<Mimo>);
        assert!(decode.contains(&256) && verify.contains(&256));
        assert_eq!(declared(ExecutionPhase::Decode), decode);
        assert_eq!(declared(ExecutionPhase::Verify), verify);
        assert_eq!(
            declared(ExecutionPhase::Prefill),
            BTreeSet::from([PREFILL_TOKENS])
        );
        for capacity in [64, 128, 256] {
            for phase in [ExecutionPhase::Decode, ExecutionPhase::Verify] {
                let usage = OperationUse {
                    phase,
                    capacity,
                    graph: true,
                };
                let offer = capability(Projection::Output, Tp::TP4, usage).unwrap();
                assert_eq!(offer.usage, usage);
                assert_eq!(offer.stages, ["chop_gemm_mimo_o_prefill_v1"]);
            }
        }
    }
}

/// The generated offers from the SM120 catalogs, and the catalog offers of
/// the drafter projections.
pub mod sm120;
