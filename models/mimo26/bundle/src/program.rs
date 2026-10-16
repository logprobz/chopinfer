//! Ordered rank program for one decode or prefill phase.
//!
//! Lowering each stage to an engine plan stays in its engine adapter. The
//! schedule keeps the model's residual, cache write, and routing dependencies
//! explicit so graph capture and eager prefill use the same ordering.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

#[cfg(test)]
use crate::contract::GA_LAYERS;

#[cfg(test)]
pub const LAYERS: u32 = V26_FLASH.layers();
#[cfg(test)]
pub const GLOBAL_LAYERS: [u32; GA_LAYERS as usize] = {
    let mut layers = [0; GA_LAYERS as usize];
    let mut layer = 0;
    let mut ordinal = 0;
    while layer < LAYERS {
        if V26_FLASH.is_global(layer) {
            layers[ordinal] = layer;
            ordinal += 1;
        }
        layer += 1;
    }
    layers
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    /// Saves the SWA ring entries this phase will overwrite, for cancel.
    RingBackup,
    Embedding,
    /// Prefill replaces placeholder embeddings from the media arena.
    MediaInject,
    InputNorm,
    Qkv(u32),
    RopeAndPagedWrite(u32),
    RopeForSlidingWindow(u32),
    AttentionPartial(u32),
    AttentionMerge(u32),
    SlidingWindowWrite(u32),
    AttentionOutput(u32),
    AttentionReduceNorm(u32),
    DenseGateUp(u32),
    DenseDown(u32),
    RouterProjection(u32),
    RouterTop8(u32),
    PermuteCount(u32),
    PermuteScan(u32),
    PermuteScatter(u32),
    GroupedGateUp(u32),
    GroupedDown(u32),
    CombineExperts(u32),
    MlpReduceNorm(u32),
    FinalReduceNorm,
    /// Prefill gathers the rows that need logits before the vocabulary head.
    GatherHeadRows,
    LmHead,
    GatherLogits,
}

impl Stage {
    /// Whether the stage runs a declared projection of the precision
    /// component rather than a selected entry.
    #[cfg(test)]
    pub(crate) const fn projection(self) -> bool {
        matches!(
            self,
            Self::Embedding
                | Self::Qkv(_)
                | Self::AttentionOutput(_)
                | Self::DenseGateUp(_)
                | Self::DenseDown(_)
                | Self::RouterProjection(_)
                | Self::GroupedGateUp(_)
                | Self::GroupedDown(_)
                | Self::LmHead
        )
    }

    /// Whether the stage runs an op the instance prepared through its
    /// backend (`crate::ops`) rather than a selected entry.
    #[cfg(test)]
    pub(crate) const fn prepared(self) -> bool {
        matches!(
            self,
            Self::InputNorm
                | Self::AttentionReduceNorm(_)
                | Self::MlpReduceNorm(_)
                | Self::FinalReduceNorm
                | Self::RouterTop8(_)
                | Self::PermuteCount(_)
                | Self::PermuteScan(_)
                | Self::PermuteScatter(_)
                | Self::CombineExperts(_)
        )
    }

    pub const fn layer(self) -> Option<u32> {
        match self {
            Self::Qkv(layer)
            | Self::RopeAndPagedWrite(layer)
            | Self::RopeForSlidingWindow(layer)
            | Self::AttentionPartial(layer)
            | Self::AttentionMerge(layer)
            | Self::SlidingWindowWrite(layer)
            | Self::AttentionOutput(layer)
            | Self::AttentionReduceNorm(layer)
            | Self::DenseGateUp(layer)
            | Self::DenseDown(layer)
            | Self::RouterProjection(layer)
            | Self::RouterTop8(layer)
            | Self::PermuteCount(layer)
            | Self::PermuteScan(layer)
            | Self::PermuteScatter(layer)
            | Self::GroupedGateUp(layer)
            | Self::GroupedDown(layer)
            | Self::CombineExperts(layer)
            | Self::MlpReduceNorm(layer) => Some(layer),
            Self::RingBackup
            | Self::Embedding
            | Self::MediaInject
            | Self::InputNorm
            | Self::FinalReduceNorm
            | Self::GatherHeadRows
            | Self::LmHead
            | Self::GatherLogits => None,
        }
    }
}

#[cfg(test)]
pub fn stages() -> Vec<Stage> {
    stages_for(crate::topology::Tp::TP4)
}

pub fn stages_for(tp: crate::topology::Tp) -> Vec<Stage> {
    let mut result = Vec::with_capacity(700);
    result.extend([
        Stage::RingBackup,
        Stage::Embedding,
        Stage::MediaInject,
        Stage::InputNorm,
    ]);
    for layer in 0..tp.geometry.layers() {
        result.push(Stage::Qkv(layer));
        if tp.geometry.is_global(layer) {
            result.push(Stage::RopeAndPagedWrite(layer));
        } else {
            result.push(Stage::RopeForSlidingWindow(layer));
        }
        result.push(Stage::AttentionPartial(layer));
        result.push(Stage::AttentionMerge(layer));
        if !tp.geometry.is_global(layer) {
            result.push(Stage::SlidingWindowWrite(layer));
        }
        result.push(Stage::AttentionOutput(layer));
        result.push(Stage::AttentionReduceNorm(layer));
        if layer < tp.geometry.dense_layer_end() {
            result.push(Stage::DenseGateUp(layer));
            result.push(Stage::DenseDown(layer));
        } else {
            result.extend([
                Stage::RouterProjection(layer),
                Stage::RouterTop8(layer),
                Stage::PermuteCount(layer),
                Stage::PermuteScan(layer),
                Stage::PermuteScatter(layer),
                Stage::GroupedGateUp(layer),
                Stage::GroupedDown(layer),
                Stage::CombineExperts(layer),
            ]);
        }
        result.push(if layer + 1 == tp.geometry.layers() {
            Stage::FinalReduceNorm
        } else {
            Stage::MlpReduceNorm(layer)
        });
    }
    result.extend([Stage::GatherHeadRows, Stage::LmHead, Stage::GatherLogits]);
    result
}

#[cfg(test)]
pub fn for_each<E>(mut stage: impl FnMut(Stage) -> Result<(), E>) -> Result<(), E> {
    for item in stages() {
        stage(item)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::contract::SWA_LAYERS;

    use super::*;

    #[test]
    fn forty_eight_layers_preserve_cache_and_residual_order() {
        let stages = stages();
        assert_eq!(stages.first(), Some(&Stage::RingBackup));
        let position = |wanted| stages.iter().position(|stage| *stage == wanted).unwrap();
        assert_eq!(position(Stage::MediaInject), position(Stage::Embedding) + 1);
        assert_eq!(position(Stage::InputNorm), position(Stage::MediaInject) + 1);
        assert_eq!(stages.last(), Some(&Stage::GatherLogits));
        assert_eq!(
            stages
                .iter()
                .filter(|stage| matches!(stage, Stage::RopeAndPagedWrite(_)))
                .count(),
            GA_LAYERS as usize
        );
        assert_eq!(
            stages
                .iter()
                .filter(|stage| matches!(stage, Stage::SlidingWindowWrite(_)))
                .count(),
            SWA_LAYERS as usize
        );
        for layer in 0..LAYERS {
            let at = |wanted| stages.iter().position(|stage| *stage == wanted).unwrap();
            assert!(at(Stage::Qkv(layer)) < at(Stage::AttentionPartial(layer)));
            assert!(at(Stage::AttentionPartial(layer)) < at(Stage::AttentionMerge(layer)));
            if GLOBAL_LAYERS.contains(&layer) {
                assert!(at(Stage::RopeAndPagedWrite(layer)) < at(Stage::AttentionPartial(layer)));
            } else {
                assert!(at(Stage::AttentionMerge(layer)) < at(Stage::SlidingWindowWrite(layer)));
            }
            assert!(at(Stage::AttentionOutput(layer)) < at(Stage::AttentionReduceNorm(layer)));
            let mlp_finish = if layer + 1 == LAYERS {
                Stage::FinalReduceNorm
            } else {
                Stage::MlpReduceNorm(layer)
            };
            assert!(at(Stage::AttentionReduceNorm(layer)) < at(mlp_finish));
        }
        let mut walked = 0;
        for_each::<()>(|_| {
            walked += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(walked, stages.len());
    }
}
