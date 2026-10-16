//! Declared comm shapes and profile tags. Cubin entries keep their shared ABI.

use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_core::contract::{
    CompileKey, CompileKnobs, DType, InstantiationSpec, MAX_COMM_RANKS, OpFamily, Refusal,
    StaticQuery, Target,
};

/// Rank counts the catalog lists. The cubin takes `ranks` at run time, so
/// one build serves them all; the keys record the shapes per rank count.
pub const CATALOG_RANKS: [u32; 3] = [2, 4, 8];

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub instantiation: InstantiationSpec,
    /// The symbol's declaration from `declared_entries`; a shape adds none.
    pub invariance: Invariance,
}

/// One declared row shape; neither its family name nor width selects a profile.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub profile: Profile,
    pub family: &'static str,
    pub width: u64,
    pub dtype: DType,
}

impl Shape {
    /// The shape's instantiation of the transport entry `symbol` at `ranks`
    /// ranks and `rows` rows, with the symbol's declaration.
    pub fn entry(self, ranks: u32, rows: u64, symbol: &str) -> Entry {
        Entry {
            profile: self.profile,
            instantiation: InstantiationSpec {
                family: OpFamily::Comm,
                key: CompileKey {
                    query: StaticQuery::Comm {
                        family: self.family,
                        ranks,
                        elements_capacity: self.width * rows,
                        dtype: self.dtype,
                    },
                    // Transport tile size, independent of the declared row width.
                    knobs: CompileKnobs {
                        tile_m: 1,
                        tile_n: 256,
                        tile_k: 4096,
                        stages: 1,
                        swizzle: 0,
                        warps: 8,
                        split_k_capacity: 1,
                    },
                    dense: None,
                    target: Target::Sm120f,
                },
                symbol: symbol.into(),
            },
            invariance: super::declared_entries()
                .into_iter()
                .find(|declared| declared.symbol == symbol)
                .map_or(Invariance::NONE, |declared| declared.invariance),
        }
    }
}

const STANDARD_SHAPES: [Shape; 2] = [
    Shape {
        profile: Profile::StdDenseMoe,
        family: "std_dense_moe_hidden",
        width: 4096,
        dtype: DType::Bf16,
    },
    Shape {
        profile: Profile::StdDenseMoe,
        family: "std_dense_moe_hidden",
        width: 8192,
        dtype: DType::Bf16,
    },
];

/// Every rank count's tagged instantiations in `catalogs`: at each rank
/// count, each model's shapes, then the standard profile's.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    let mut entries = Vec::new();
    for ranks in CATALOG_RANKS {
        for model in catalogs.models() {
            entries.extend((model.comm_shapes)(ranks));
        }
        entries.extend(own_catalog_for(ranks).expect("catalog rank count"));
    }
    entries
}

/// The standard profile's instantiations at `ranks` ranks. Each model
/// declares its own widths and sharding (`ModelCatalog::comm_shapes`); a
/// new profile does not inherit another profile's row shape.
pub fn own_catalog_for(ranks: u32) -> Result<Vec<Entry>, Refusal> {
    if ranks == 0 || ranks as usize > MAX_COMM_RANKS {
        return Err(Refusal("comm catalog rank count"));
    }
    let mut entries = Vec::new();
    for shape in STANDARD_SHAPES {
        for rows in [1, 32, 8192] {
            let symbol = if shape.width * rows * 2 <= 262144 {
                "chop_comm_allreduce_oneshot_bf16_v1"
            } else {
                "chop_comm_reduce_scatter_bf16_v1"
            };
            entries.push(shape.entry(ranks, rows, symbol));
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_profile_and_shape_do_not_follow_names_or_widths() {
        for width in [2560, 4096, 8192] {
            let shape = Shape {
                profile: Profile::StdDenseMoe,
                family: "other_hidden",
                width,
                dtype: DType::F32,
            };
            let entry = shape.entry(2, 3, "chop_comm_allgather_f32_v1");
            assert_eq!(entry.profile, Profile::StdDenseMoe);
            assert_eq!(
                entry.instantiation.key.query,
                StaticQuery::Comm {
                    family: "other_hidden",
                    ranks: 2,
                    elements_capacity: 3 * width,
                    dtype: DType::F32,
                }
            );
        }
    }

    #[test]
    fn catalog_profiles_cover_the_declared_rank_shapes() {
        for ranks in CATALOG_RANKS {
            let entries = own_catalog_for(ranks).unwrap();
            assert_eq!(entries.len(), 6);
            assert!(entries.iter().all(|e| e.profile == Profile::StdDenseMoe));
        }
        assert!(own_catalog_for(0).is_err());
    }
}
