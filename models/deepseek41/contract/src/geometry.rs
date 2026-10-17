//! Geometry generated from the supported execution configurations
//! (`models/deepseek41/geometries`, and the directories
//! `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS` lists). Every field is fixed at
//! compile time: kernel instances and the tensor catalog derive from it.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub id: &'static str,
    /// Whether the geometry comes from `models/deepseek41/geometries`.
    pub base: bool,
    pub kernel_tag: &'static str,
    pub shape: Shape,
    pub rules: &'static str,
    pub document: &'static str,
}

/// How a layer finds its compressed keys and its top-k selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerRole {
    /// Sliding-window attention only (compress ratio 0).
    Window,
    /// Owns a compressor, index keys and an indexer.
    Full,
    /// Reads the latest source's compressed KV and top-k selection.
    Reuse,
    /// Reads the latest source's compressed KV and index keys, and selects
    /// with its own index query and weights.
    Reindex,
}

const fn count(length: usize) -> u32 {
    assert!(length <= u32::MAX as usize, "geometry list exceeds u32");
    length as u32
}

const fn contains(ids: &[u32], id: u32) -> bool {
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == id {
            return true;
        }
        i += 1;
    }
    false
}

impl Geometry {
    /// Backbone layers, without the DSpark stages.
    pub const fn layers(&self) -> u32 {
        self.shape.layers
    }
    pub const fn dspark_stages(&self) -> u32 {
        self.shape.dspark.stages
    }
    /// The compress ratio of backbone layer or DSpark stage `layer`
    /// (stages follow the backbone).
    pub const fn compress_ratio(&self, layer: u32) -> u32 {
        self.shape.compress_ratios[layer as usize]
    }
    pub const fn layer_role(&self, layer: u32) -> LayerRole {
        if self.compress_ratio(layer) == 0 {
            LayerRole::Window
        } else if contains(self.shape.csa.kv_sources, layer) {
            LayerRole::Full
        } else if contains(self.shape.csa.index_sources, layer) {
            LayerRole::Reindex
        } else {
            LayerRole::Reuse
        }
    }
    pub const fn is_engram_layer(&self, layer: u32) -> bool {
        contains(self.shape.engram.layers, layer)
    }
    pub const fn engram_layers(&self) -> u32 {
        count(self.shape.engram.layers.len())
    }
    /// Width of the hyper-connection residual: one hidden row per stream.
    pub const fn hc_width(&self) -> u32 {
        self.shape.hc.streams * self.shape.hidden
    }
    /// Rows of each mHC mixing projection: pre and post per stream, plus the
    /// stream-by-stream combination.
    pub const fn hc_mix(&self) -> u32 {
        (2 + self.shape.hc.streams) * self.shape.hc.streams
    }
    /// Latent channels without RoPE.
    pub const fn nope_dim(&self) -> u32 {
        self.shape.attention.head_dim - self.shape.attention.rope_dim
    }
    /// Input width of each output group's `wo_a` block.
    pub const fn o_group_width(&self) -> u32 {
        self.shape.attention.heads * self.shape.attention.head_dim / self.shape.attention.o_groups
    }
    pub const fn context_tokens(&self) -> u64 {
        self.shape.context as u64
    }
}

include!(concat!(env!("OUT_DIR"), "/geometries.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_layer_schedule() {
        assert_eq!(GEOMETRIES[0], FLASH);
        const { assert!(FLASH.base) };
        assert_eq!(FLASH.kernel_tag, "deepseek41");
        assert_eq!(FLASH.layers(), 40);
        assert_eq!(FLASH.dspark_stages(), 3);
        for layer in 0..43 {
            let expected = match layer {
                0 | 1 | 40.. => LayerRole::Window,
                2 | 8 | 14 | 20 => LayerRole::Full,
                24 | 28 | 32 | 36 => LayerRole::Reindex,
                _ => LayerRole::Reuse,
            };
            assert_eq!(FLASH.layer_role(layer), expected, "layer {layer}");
            let ratio = match layer {
                2..=19 => 2,
                20..=39 => 1,
                _ => 0,
            };
            assert_eq!(FLASH.compress_ratio(layer), ratio, "layer {layer}");
        }
        assert_eq!(FLASH.shape.candidates.source, 20);
        assert!(FLASH.is_engram_layer(1) && FLASH.is_engram_layer(14));
        assert_eq!(FLASH.engram_layers(), 2);
        assert_eq!(FLASH.shape.engram.rows, &[384_006_168, 384_016_682]);
        assert_eq!(FLASH.hc_width(), 20480);
        assert_eq!(FLASH.hc_mix(), 24);
        assert_eq!(FLASH.nope_dim(), 448);
        assert_eq!(FLASH.o_group_width(), 4096);
        assert_eq!(FLASH.context_tokens(), 1 << 20);
        assert_eq!(FLASH.shape.eps, 1e-20);
        assert_eq!(FLASH.shape.hc.eps, 1e-6);
        assert_eq!(FLASH.shape.moe.routed_scale, 1.5);
        assert_eq!(FLASH.shape.rope.compress_theta, 160000.0);
        assert_eq!(FLASH.shape.dspark.targets, &[37, 38, 39]);
        assert_eq!(FLASH.shape.tokens.image, 129264);
        // The checkpoint's architecture first; a geometry recipe appends its own.
        assert_eq!(ARCHITECTURE_ALIASES[0], "DeepseekV41ForCausalLM");
    }

    #[test]
    fn the_engram_layout_reproduces_the_checkpoint_rows() {
        let s = FLASH.shape.engram;
        let rows = crate::structure::engram_rows(
            s.layers.len(),
            s.vocab.into(),
            s.max_ngram.into(),
            s.heads.into(),
        );
        assert_eq!(
            rows,
            s.rows.iter().map(|&r| u64::from(r)).collect::<Vec<_>>()
        );
    }
}
