//! The checkpoint tensor catalog, generated from the geometry by layer type
//! and count: KDA and latent attention mixers, dense and MoE MLPs, mHC
//! mixing per sublayer, the MTP layers, the vision tower, and the embedding
//! and head. Each tensor records its rank placement and memory group. The
//! checkpoint's quantization layout fixes how each projection is stored.
use crate::geometry::{Geometry, LayerKind, MlpKind};
use crate::kernels::FP8_BLOCK;
use crate::quantization::{Encoding, FP8_BLOCK128, Layout};

/// K values per NVFP4 and MXFP8 weight scale.
pub const NVFP4_GROUP: u32 = 16;
const MXFP8_GROUP: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    F8E4M3,
    Bf16,
    F32,
    /// Packed E2M1 pairs and E8M0 scales.
    U8,
}

impl DType {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F8E4M3 | Self::U8 => 1,
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
    /// The safetensors header spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F8E4M3 => "F8_E4M3",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::U8 => "U8",
        }
    }
}

/// The per-rank memory budget a tensor counts toward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    /// Routed experts of the main layers.
    Routed,
    /// Every other main-layer tensor, and the final norm.
    Other,
    /// The MTP layers, routed experts included.
    Mtp,
    Vision,
    EmbedHead,
}

/// How a tensor is placed across the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Every rank holds the whole tensor.
    Replicated,
    /// Each rank holds an equal contiguous slice of dimension `dim` (of the
    /// weight, and of its FP8 scale in block units).
    Split { dim: usize },
    /// Routed expert `expert`'s tensor, split as [`Placement::Split`] is:
    /// every rank holds an equal contiguous slice of dimension `dim` of
    /// every expert (its intermediate rows of the gate and up projections,
    /// columns of the down projection).
    Expert { expert: u32, dim: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub group: Group,
    pub placement: Placement,
    /// Present in the checkpoint and never read (an NVFP4 export's static
    /// `input_scale`, since activations stay BF16).
    pub ignored: bool,
}

impl TensorSpec {
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.bytes()
    }
}

/// Rows of an mHC scale tensor: the pre, post and combination scales.
const HC_SCALES: u64 = 3;

const TEXT: &str = "model.language_model";
const VISION: &str = "model.visual";

struct Builder {
    tensors: Vec<TensorSpec>,
    group: Group,
    layout: Layout,
}

impl Builder {
    fn push(&mut self, name: String, dtype: DType, shape: &[u32], placement: Placement) {
        self.tensors.push(TensorSpec {
            name,
            dtype,
            shape: shape.iter().map(|&d| u64::from(d)).collect(),
            group: self.group,
            placement,
            ignored: false,
        });
    }
    fn bf16(&mut self, name: String, shape: &[u32], placement: Placement) {
        self.push(name, DType::Bf16, shape, placement);
    }
    fn norm(&mut self, name: String, width: u32) {
        self.bf16(name, &[width], Placement::Replicated);
    }
    /// A projection `[rows, columns]` in the layout's encoding for its
    /// class, with its scales.
    fn projection(&mut self, stem: &str, rows: u32, columns: u32, placement: Placement) {
        let l = &self.layout;
        let encoding = match (placement, self.group) {
            (Placement::Expert { .. }, Group::Mtp) => l.mtp_routed,
            (Placement::Expert { .. }, _) => l.routed,
            _ if crate::quantization::dense(stem) => l.dense,
            _ => l.projections,
        };
        // A projection's NVFP4 global and activation scales are scalars
        // every rank holds, as every placement splits or replicates it.
        let scalar = Placement::Replicated;
        let weight = format!("{stem}.weight");
        let scale = format!("{stem}.weight_scale");
        match encoding {
            Encoding::Bf16 => self.bf16(weight, &[rows, columns], placement),
            Encoding::Fp8Block128 => {
                self.push(weight, DType::F8E4M3, &[rows, columns], placement);
                self.push(
                    format!("{stem}.weight_scale_inv"),
                    DType::F32,
                    &[rows.div_ceil(FP8_BLOCK), columns.div_ceil(FP8_BLOCK)],
                    placement,
                );
            }
            Encoding::Nvfp4 => {
                self.push(weight, DType::U8, &[rows, columns / 2], placement);
                let groups = columns / NVFP4_GROUP;
                self.push(scale, DType::F8E4M3, &[rows, groups], placement);
                self.push(format!("{stem}.weight_scale_2"), DType::F32, &[], scalar);
                self.push(format!("{stem}.input_scale"), DType::F32, &[], scalar);
                self.tensors.last_mut().unwrap().ignored = true;
            }
            Encoding::Mxfp8 => {
                self.push(weight, DType::F8E4M3, &[rows, columns], placement);
                let groups = columns / MXFP8_GROUP;
                self.push(scale, DType::U8, &[rows, groups], placement);
            }
        }
    }

    fn hyper_connections(&mut self, g: &Geometry, prefix: &str) {
        for sublayer in ["attn", "ffn"] {
            let stem = format!("{prefix}.hc_{sublayer}");
            self.tensors.push(TensorSpec {
                name: format!("{stem}_base"),
                dtype: self.layout.hc_vectors,
                shape: vec![u64::from(g.hc_mix())],
                group: self.group,
                placement: Placement::Replicated,
                ignored: false,
            });
            self.bf16(
                format!("{stem}_fn"),
                &[g.hc_mix(), g.hc_width()],
                Placement::Replicated,
            );
            self.tensors.push(TensorSpec {
                name: format!("{stem}_scale"),
                dtype: self.layout.hc_vectors,
                shape: vec![HC_SCALES],
                group: self.group,
                placement: Placement::Replicated,
                ignored: false,
            });
        }
    }

    fn linear_attention(&mut self, g: &Geometry, prefix: &str) {
        let s = &g.shape;
        let (hidden, width, dim) = (s.hidden, g.linear_width(), s.linear.head_dim);
        let rows = Placement::Split { dim: 0 };
        let a = format!("{prefix}.self_attn");
        for name in ["q", "k", "v"] {
            self.bf16(format!("{a}.{name}_proj.weight"), &[width, hidden], rows);
            self.push(
                format!("{a}.{name}_conv1d.weight"),
                self.layout.kda_conv,
                &[width, 1, s.linear.conv_taps],
                rows,
            );
        }
        self.push(format!("{a}.A_log"), DType::F32, &[s.linear.heads], rows);
        self.push(format!("{a}.dt_bias"), DType::F32, &[width], rows);
        self.bf16(
            format!("{a}.b_proj.weight"),
            &[s.linear.heads, hidden],
            rows,
        );
        for gate in ["f", "g"] {
            self.bf16(
                format!("{a}.{gate}_a_proj.weight"),
                &[dim, hidden],
                Placement::Replicated,
            );
            self.bf16(format!("{a}.{gate}_b_proj.weight"), &[width, dim], rows);
        }
        self.norm(format!("{a}.o_norm.weight"), dim);
        self.bf16(
            format!("{a}.o_proj.weight"),
            &[hidden, width],
            Placement::Split { dim: 1 },
        );
    }

    fn latent_attention(&mut self, g: &Geometry, prefix: &str) {
        let s = &g.shape;
        let l = &s.latent;
        let a = format!("{prefix}.self_attn");
        let replicated = Placement::Replicated;
        self.projection(&format!("{a}.q_a_proj"), l.q_rank, s.hidden, replicated);
        self.norm(format!("{a}.q_a_layernorm.weight"), l.q_rank);
        self.projection(
            &format!("{a}.q_b_proj"),
            l.heads * g.qk_head(),
            l.q_rank,
            Placement::Split { dim: 0 },
        );
        self.projection(
            &format!("{a}.kv_a_proj_with_mqa"),
            l.kv_rank + l.qk_rope,
            s.hidden,
            replicated,
        );
        self.norm(format!("{a}.kv_a_layernorm.weight"), l.kv_rank);
        self.bf16(
            format!("{a}.kv_b_proj.weight"),
            &[g.kv_up_rows(), l.kv_rank],
            Placement::Split { dim: 0 },
        );
        self.projection(
            &format!("{a}.o_proj"),
            s.hidden,
            l.heads * l.v_head,
            Placement::Split { dim: 1 },
        );
        let i = format!("{a}.indexer");
        let x = &s.indexer;
        self.bf16(
            format!("{i}.wq_b.weight"),
            &[x.heads * x.dim, l.q_rank],
            replicated,
        );
        self.bf16(format!("{i}.wk.weight"), &[x.dim, s.hidden], replicated);
        self.norm(format!("{i}.k_norm.weight"), x.dim);
        self.norm(format!("{i}.k_norm.bias"), x.dim);
        self.bf16(
            format!("{i}.weights_proj.weight"),
            &[x.heads, s.hidden],
            replicated,
        );
        self.bf16(
            format!("{i}.index_kpool_compress_ape"),
            &[x.pool, x.dim],
            replicated,
        );
        self.bf16(
            format!("{i}.index_kpool_compress_gate"),
            &[x.dim, s.hidden],
            replicated,
        );
    }

    fn dense_mlp(&mut self, g: &Geometry, prefix: &str) {
        let (hidden, inner) = (g.shape.hidden, g.shape.dense.intermediate);
        let m = format!("{prefix}.mlp");
        self.projection(
            &format!("{m}.gate_proj"),
            inner,
            hidden,
            Placement::Split { dim: 0 },
        );
        self.projection(
            &format!("{m}.up_proj"),
            inner,
            hidden,
            Placement::Split { dim: 0 },
        );
        self.projection(
            &format!("{m}.down_proj"),
            hidden,
            inner,
            Placement::Split { dim: 1 },
        );
    }

    /// The router, the routed experts (in `routed`, which the main layers
    /// count separately) and the shared experts.
    fn moe(&mut self, g: &Geometry, prefix: &str, routed: Group) {
        let s = &g.shape;
        let (hidden, inner) = (s.hidden, s.moe.intermediate);
        let m = format!("{prefix}.mlp");
        self.bf16(
            format!("{m}.gate.weight"),
            &[s.moe.experts, hidden],
            Placement::Replicated,
        );
        self.push(
            format!("{m}.gate.e_score_correction_bias"),
            DType::F32,
            &[s.moe.experts],
            Placement::Replicated,
        );
        let group = std::mem::replace(&mut self.group, routed);
        for expert in 0..s.moe.experts {
            let e = format!("{m}.experts.{expert}");
            let rows = Placement::Expert { expert, dim: 0 };
            let columns = Placement::Expert { expert, dim: 1 };
            self.projection(&format!("{e}.gate_proj"), inner, hidden, rows);
            self.projection(&format!("{e}.up_proj"), inner, hidden, rows);
            self.projection(&format!("{e}.down_proj"), hidden, inner, columns);
        }
        self.group = group;
        let shared = s.moe.shared * inner;
        if shared > 0 {
            let e = format!("{m}.shared_experts");
            self.projection(
                &format!("{e}.gate_proj"),
                shared,
                hidden,
                Placement::Split { dim: 0 },
            );
            self.projection(
                &format!("{e}.up_proj"),
                shared,
                hidden,
                Placement::Split { dim: 0 },
            );
            self.projection(
                &format!("{e}.down_proj"),
                hidden,
                shared,
                Placement::Split { dim: 1 },
            );
        }
    }

    fn vision(&mut self, g: &Geometry) {
        let v = &g.shape.vision;
        let replicated = Placement::Replicated;
        let p = format!("{VISION}.patch_embed.proj");
        self.bf16(
            format!("{p}.weight"),
            &[v.hidden, v.channels, v.temporal_patch, v.patch, v.patch],
            replicated,
        );
        self.bf16(format!("{p}.bias"), &[v.hidden], replicated);
        for block in 0..v.depth {
            let b = format!("{VISION}.blocks.{block}");
            self.norm(format!("{b}.norm1.weight"), v.hidden);
            self.norm(format!("{b}.norm2.weight"), v.hidden);
            for (name, rows) in [("qkv", 3 * v.hidden), ("proj", v.hidden)] {
                self.bf16(
                    format!("{b}.attn.{name}.weight"),
                    &[rows, v.hidden],
                    replicated,
                );
                if g.semantics.vision_attention_bias {
                    self.bf16(format!("{b}.attn.{name}.bias"), &[rows], replicated);
                }
            }
            self.norm(format!("{b}.attn.q_norm.weight"), g.vision_head_dim());
            self.norm(format!("{b}.attn.k_norm.weight"), g.vision_head_dim());
            for (name, rows, columns) in [
                ("gate_proj", v.intermediate, v.hidden),
                ("up_proj", v.intermediate, v.hidden),
                ("down_proj", v.hidden, v.intermediate),
            ] {
                self.bf16(
                    format!("{b}.mlp.{name}.weight"),
                    &[rows, columns],
                    replicated,
                );
                self.bf16(format!("{b}.mlp.{name}.bias"), &[rows], replicated);
            }
        }
        self.norm(format!("{VISION}.post_layernorm.weight"), v.hidden);
        self.bf16(
            format!("{VISION}.downsample.weight"),
            &[v.out, v.hidden, v.merge, v.merge],
            replicated,
        );
        self.bf16(format!("{VISION}.downsample.bias"), &[v.out], replicated);
        let m = format!("{VISION}.merger");
        self.bf16(format!("{m}.proj.weight"), &[v.out, v.out], replicated);
        self.norm(format!("{m}.post_projection_norm.weight"), v.out);
        self.norm(format!("{m}.post_projection_norm.bias"), v.out);
        let inner = v.projection_intermediate;
        self.bf16(format!("{m}.gate_proj.weight"), &[inner, v.out], replicated);
        self.bf16(format!("{m}.up_proj.weight"), &[inner, v.out], replicated);
        self.bf16(format!("{m}.down_proj.weight"), &[v.out, inner], replicated);
    }
}

/// Every tensor of `g`'s base block-FP8 checkpoint, in layer order.
pub fn catalog(g: &Geometry) -> Vec<TensorSpec> {
    checkpoint_catalog(g, &FP8_BLOCK128)
}

/// Every tensor of a checkpoint of `g` stored in `layout`, in layer order.
pub fn checkpoint_catalog(g: &Geometry, layout: &Layout) -> Vec<TensorSpec> {
    let s = &g.shape;
    let mut b = Builder {
        tensors: Vec::new(),
        group: Group::EmbedHead,
        layout: *layout,
    };
    let vocab = Placement::Split { dim: 0 };
    b.bf16(
        format!("{TEXT}.embed_tokens.weight"),
        &[s.vocab, s.hidden],
        vocab,
    );
    b.bf16("lm_head.weight".into(), &[s.vocab, s.hidden], vocab);
    b.group = Group::Other;
    b.norm(format!("{TEXT}.norm.weight"), s.hidden);
    for layer in 0..g.layers() {
        let prefix = format!("{TEXT}.layers.{layer}");
        b.norm(format!("{prefix}.input_layernorm.weight"), s.hidden);
        b.norm(
            format!("{prefix}.post_attention_layernorm.weight"),
            s.hidden,
        );
        b.hyper_connections(g, &prefix);
        match g.layer_kind(layer) {
            LayerKind::Linear => b.linear_attention(g, &prefix),
            LayerKind::Sparse => b.latent_attention(g, &prefix),
        }
        match g.mlp_kind(layer) {
            MlpKind::Dense => b.dense_mlp(g, &prefix),
            MlpKind::Moe => b.moe(g, &prefix, Group::Routed),
        }
    }
    b.group = Group::Mtp;
    for layer in g.layers()..g.layers() + g.mtp_layers() {
        let prefix = format!("{TEXT}.layers.{layer}");
        b.bf16(
            format!("{prefix}.eh_proj.weight"),
            &[s.hidden, 2 * s.hidden],
            Placement::Replicated,
        );
        for norm in ["enorm", "hnorm", "shared_head.norm"] {
            b.norm(format!("{prefix}.{norm}.weight"), s.hidden);
        }
        b.norm(format!("{prefix}.input_layernorm.weight"), s.hidden);
        b.norm(
            format!("{prefix}.post_attention_layernorm.weight"),
            s.hidden,
        );
        b.latent_attention(g, &prefix);
        b.moe(g, &prefix, Group::Mtp);
    }
    b.group = Group::Vision;
    b.vision(g);
    b.tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;
    use crate::quantization::{NVFP4_DENSE_ROUTED, NVFP4_ROUTED};
    use std::collections::BTreeMap;

    /// A tensor family: the name with each layer, block and expert index
    /// replaced by `{}`.
    pub(crate) fn family(name: &str) -> String {
        name.split('.')
            .map(|part| {
                if part.bytes().all(|c| c.is_ascii_digit()) {
                    "{}"
                } else {
                    part
                }
            })
            .collect::<Vec<_>>()
            .join(".")
    }

    #[test]
    fn flash_catalog_totals() {
        let tensors = catalog(&FLASH);
        assert_eq!(tensors.len(), 76_108);
        assert_eq!(
            tensors.iter().map(TensorSpec::bytes).sum::<u64>(),
            328_326_771_576
        );
        let mut names = std::collections::BTreeSet::new();
        assert!(tensors.iter().all(|t| names.insert(t.name.as_str())));
    }

    /// The committed families were read from the pinned checkpoint's headers
    /// (`fixtures/tensor-families.json`).
    #[test]
    fn flash_catalog_matches_the_checkpoint_families() {
        assert_families(
            &catalog(&FLASH),
            include_str!("../fixtures/tensor-families.json"),
        );
    }

    #[test]
    fn nvfp4_catalog_totals() {
        let tensors = checkpoint_catalog(&FLASH, &NVFP4_ROUTED);
        assert_eq!(tensors.len(), 148_498);
        assert_eq!(
            tensors.iter().map(TensorSpec::bytes).sum::<u64>(),
            198_042_331_512
        );
        // Only the NVFP4 experts' static input scales are ignored.
        let ignored = tensors.iter().filter(|t| t.ignored).collect::<Vec<_>>();
        assert_eq!(ignored.len(), 42 * 288 * 3);
        assert!(ignored.iter().all(|t| t.name.ends_with(".input_scale")
            && t.dtype == DType::F32
            && t.shape.is_empty()));
    }

    #[test]
    fn nvidia_nvfp4_catalog_totals_and_families() {
        let tensors = checkpoint_catalog(&FLASH, &NVFP4_DENSE_ROUTED);
        assert_eq!(tensors.len(), 147_661);
        assert_eq!(
            tensors.iter().map(TensorSpec::bytes).sum::<u64>(),
            204_419_110_596
        );
        // The NVFP4 routed experts' and dense MLPs' static input scales.
        let ignored = tensors.iter().filter(|t| t.ignored).count();
        assert_eq!(ignored, 42 * 288 * 3 + 3 * 3);
        assert_families(
            &tensors,
            include_str!("../fixtures/nvidia-nvfp4-tensor-families.json"),
        );
    }

    /// Read from the local-inference-lab NVFP4 export's headers
    /// (`fixtures/nvfp4-tensor-families.json`).
    #[test]
    fn nvfp4_catalog_matches_the_export_families() {
        assert_families(
            &checkpoint_catalog(&FLASH, &NVFP4_ROUTED),
            include_str!("../fixtures/nvfp4-tensor-families.json"),
        );
    }

    fn assert_families(tensors: &[TensorSpec], fixture: &str) {
        let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let mut expected = BTreeMap::new();
        for row in fixture.as_array().unwrap() {
            let shape = row["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect::<Vec<_>>();
            let key = (row["name"].as_str().unwrap().to_owned(), shape);
            let previous = expected.insert(
                key,
                (
                    row["dtype"].as_str().unwrap().to_owned(),
                    row["count"].as_u64().unwrap(),
                ),
            );
            assert!(previous.is_none());
        }
        let mut observed = BTreeMap::<_, (String, u64)>::new();
        for tensor in tensors {
            let entry = observed
                .entry((family(&tensor.name), tensor.shape.clone()))
                .or_insert_with(|| (tensor.dtype.name().to_owned(), 0));
            assert_eq!(entry.0, tensor.dtype.name(), "{}", tensor.name);
            entry.1 += 1;
        }
        assert_eq!(observed, expected);
    }
}
