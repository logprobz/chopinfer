//! The tensor-parallel weight plan: which slice of each catalog tensor every
//! rank holds, and the bytes each rank holds per memory group.
//!
//! KDA and latent attention split by heads, the latent, query-down, indexer
//! and router projections are replicated, the dense and shared MLPs and
//! every routed expert split by intermediate column, the embedding and head
//! split by vocabulary row, and the vision tower is replicated. No rank
//! holds a tensor the catalog marks ignored.
use crate::catalog::{DType, Group, NVFP4_GROUP, Placement, TensorSpec};
use crate::geometry::Geometry;
use crate::kernels::FP8_BLOCK;

/// One rank's part of a catalog tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankTensor<'a> {
    pub spec: &'a TensorSpec,
    pub shape: Vec<u64>,
    /// First index of the rank's slice along the split dimension.
    pub start: u64,
}

impl RankTensor<'_> {
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.spec.dtype.bytes()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupBytes {
    pub routed: u64,
    pub other: u64,
    pub mtp: u64,
    pub vision: u64,
    pub embed_head: u64,
}

impl GroupBytes {
    pub fn total(&self) -> u64 {
        self.routed + self.other + self.mtp + self.vision + self.embed_head
    }
    fn add(&mut self, group: Group, bytes: u64) {
        *match group {
            Group::Routed => &mut self.routed,
            Group::Other => &mut self.other,
            Group::Mtp => &mut self.mtp,
            Group::Vision => &mut self.vision,
            Group::EmbedHead => &mut self.embed_head,
        } += bytes;
    }
}

#[derive(Clone, Debug)]
pub struct RankPlan<'a> {
    pub rank: u32,
    pub tensors: Vec<RankTensor<'a>>,
    pub bytes: GroupBytes,
}

/// The part of `spec` that `rank` of `ranks` holds, if any.
pub fn rank_tensor<'a>(
    g: &Geometry,
    spec: &'a TensorSpec,
    ranks: u32,
    rank: u32,
) -> Result<Option<RankTensor<'a>>, String> {
    let ranks64 = u64::from(ranks);
    if spec.ignored {
        return Ok(None);
    }
    match spec.placement {
        Placement::Replicated => Ok(Some(RankTensor {
            spec,
            shape: spec.shape.clone(),
            start: 0,
        })),
        Placement::Expert { expert, .. } if expert >= g.shape.moe.experts => Err(format!(
            "{}: expert {expert} of {} experts",
            spec.name, g.shape.moe.experts
        )),
        Placement::Split { dim } | Placement::Expert { dim, .. } => {
            let total = spec.shape[dim];
            let part = total / ranks64;
            // A weight's slice holds whole scale groups: 128 x 128 blocks of
            // block FP8, and along K, groups of 16 NVFP4 values (8 bytes of
            // E2M1 pairs). Scales split in their own units.
            let weight = spec.name.ends_with(".weight");
            let granule = match (spec.dtype, dim) {
                (DType::F8E4M3, _) if weight => u64::from(FP8_BLOCK),
                (DType::U8, 1) if weight => u64::from(NVFP4_GROUP / 2),
                _ => 1,
            };
            if total % ranks64 != 0 || part % granule != 0 {
                return Err(format!(
                    "{}: dimension {dim} of {total} does not split across {ranks} ranks",
                    spec.name
                ));
            }
            let mut shape = spec.shape.clone();
            shape[dim] = part;
            Ok(Some(RankTensor {
                spec,
                shape,
                start: part * u64::from(rank),
            }))
        }
    }
}

/// Every rank's tensors and group bytes at `ranks`.
pub fn plan<'a>(
    g: &Geometry,
    catalog: &'a [TensorSpec],
    ranks: u32,
) -> Result<Vec<RankPlan<'a>>, String> {
    crate::admission::check_ranks(ranks)?;
    (0..ranks)
        .map(|rank| {
            let mut bytes = GroupBytes::default();
            let mut tensors = Vec::new();
            for spec in catalog {
                if let Some(part) = rank_tensor(g, spec, ranks, rank)? {
                    bytes.add(spec.group, part.bytes());
                    tensors.push(part);
                }
            }
            Ok(RankPlan {
                rank,
                tensors,
                bytes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{self, catalog};
    use crate::geometry::FLASH;
    use crate::quantization::{NVFP4_DENSE_ROUTED, NVFP4_ROUTED};

    #[test]
    fn flash_tp4_bytes_per_rank() {
        let tensors = catalog(&FLASH);
        let ranks = plan(&FLASH, &tensors, 4).unwrap();
        let expected = GroupBytes {
            routed: 76_120_031_232,
            other: 3_546_141_048,
            mtp: 1_942_986_368,
            vision: 1_127_254_016,
            embed_head: 634_388_480,
        };
        for rank in &ranks {
            assert_eq!(rank.bytes, expected, "rank {}", rank.rank);
        }
        // 77.65 GiB per GPU.
        assert_eq!(expected.total(), 83_370_801_144);
        let gib = |bytes: u64| (bytes as f64 / f64::from(1 << 30) * 100.0).round() / 100.0;
        assert_eq!(
            [
                gib(expected.routed),
                gib(expected.other),
                gib(expected.mtp),
                gib(expected.vision),
                gib(expected.embed_head),
                gib(expected.total())
            ],
            [70.89, 3.3, 1.81, 1.05, 0.59, 77.65]
        );
        // Every rank holds a quarter of every routed expert: the gate and up
        // projections' intermediate rows, the down projection's columns.
        let experts = tensors
            .iter()
            .filter(|spec| matches!(spec.placement, Placement::Expert { .. }))
            .count();
        for rank in &ranks {
            let parts = rank
                .tensors
                .iter()
                .filter(|part| matches!(part.spec.placement, Placement::Expert { .. }))
                .collect::<Vec<_>>();
            assert_eq!(parts.len(), experts, "rank {}", rank.rank);
            for part in parts {
                let Placement::Expert { dim, .. } = part.spec.placement else {
                    unreachable!()
                };
                assert_eq!(
                    part.shape[dim] * 4,
                    part.spec.shape[dim],
                    "{}",
                    part.spec.name
                );
                assert_eq!(part.start, part.shape[dim] * u64::from(rank.rank));
            }
        }
        let part = |name: &str| {
            let name = format!("model.language_model.layers.5.mlp.experts.7.{name}");
            let part = ranks[2]
                .tensors
                .iter()
                .find(|t| t.spec.name == name)
                .unwrap();
            (part.shape.clone(), part.start)
        };
        assert_eq!(part("gate_proj.weight"), (vec![512, 4096], 1024));
        assert_eq!(part("gate_proj.weight_scale_inv"), (vec![4, 32], 8));
        assert_eq!(part("down_proj.weight"), (vec![4096, 512], 1024));
        assert_eq!(part("down_proj.weight_scale_inv"), (vec![32, 4], 8));
    }

    #[test]
    fn nvfp4_tp4_bytes_per_rank() {
        let tensors = catalog::checkpoint_catalog(&FLASH, &NVFP4_ROUTED);
        let ranks = plan(&FLASH, &tensors, 4).unwrap();
        let expected = GroupBytes {
            routed: 42_807_211_776,
            other: 4_269_481_848,
            mtp: 2_036_906_624,
            vision: 1_127_254_016,
            embed_head: 634_388_480,
        };
        for rank in &ranks {
            assert_eq!(rank.bytes, expected, "rank {}", rank.rank);
        }
        // 47.38 GiB per GPU.
        assert_eq!(expected.total(), 50_875_242_744);
        // No rank holds an ignored input scale.
        assert!(
            ranks
                .iter()
                .flat_map(|rank| &rank.tensors)
                .all(|part| !part.spec.ignored)
        );
    }

    #[test]
    fn nvidia_nvfp4_tp4_bytes_per_rank() {
        let tensors = catalog::checkpoint_catalog(&FLASH, &NVFP4_DENSE_ROUTED);
        let ranks = plan(&FLASH, &tensors, 4).unwrap();
        let expected = GroupBytes {
            routed: 42_807_211_776,
            other: 4_108_356_768,
            mtp: 3_792_222_848,
            vision: 1_127_254_016,
            embed_head: 634_388_480,
        };
        for rank in &ranks {
            assert_eq!(rank.bytes, expected, "rank {}", rank.rank);
        }
        // 48.87 GiB per GPU.
        assert_eq!(expected.total(), 52_469_433_888);
        // The row-split NVFP4 down projection: whole groups of 16 K values
        // per rank in its codes and scales, its global on every rank.
        let part = |name: &str| {
            let name = format!("model.language_model.layers.1.mlp.down_proj.{name}");
            let part = ranks[1]
                .tensors
                .iter()
                .find(|t| t.spec.name == name)
                .unwrap();
            (part.shape.clone(), part.start)
        };
        assert_eq!(part("weight"), (vec![4096, 1536], 1536));
        assert_eq!(part("weight_scale"), (vec![4096, 192], 192));
        assert_eq!(part("weight_scale_2"), (vec![], 0));
        assert!(
            ranks[1]
                .tensors
                .iter()
                .all(|t| !t.spec.name.ends_with(".input_scale"))
        );
    }

    #[test]
    fn slices_tile_their_tensor() {
        let tensors = catalog(&FLASH);
        let ranks = plan(&FLASH, &tensors, 4).unwrap();
        let name = "model.language_model.layers.3.self_attn.q_b_proj.weight_scale_inv";
        let starts = ranks
            .iter()
            .map(|rank| {
                let part = rank.tensors.iter().find(|t| t.spec.name == name).unwrap();
                (part.start, part.shape.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            starts,
            [0, 32, 64, 96].map(|start| (start, vec![32, 12])).to_vec()
        );
        assert!(plan(&FLASH, &tensors, 2).is_err());
    }
}
