//! GLM's context-parallel entries at group sizes 2 and 4
//! (`kernels_sm120::context_parallel`): the partial merge of the rank's own
//! latent heads, the key list localization and the localization of a step's
//! appended positions, plus the shared comm engine's F32 all-gather that the
//! group runs over its own members (`context_parallel::group`). Both cubins
//! gate on their own lock family, [`FAMILY`]. The phased group gather
//! (`context_parallel::phased`), which moves the large group blocks one
//! destination at a time, has its own cubin in [`PHASED_FAMILY`].

use kernel_build::CubinSpec;
use kernels_sm120::context_parallel::{self, group, phased};

use super::engines::{PROFILE, latent_geometry};
use crate::contract::{PAGE_TOKENS, PREFILL_TOKENS, Shape};

/// The lock family of both context-parallel cubins.
pub const FAMILY: &str = "context_parallel";

/// The lock family of the phased group gather's cubin.
pub const PHASED_FAMILY: &str = "context_parallel_phased";

/// The group barrier: an all-gather of one word.
pub const ALLGATHER_SYMBOL: &str = group::ALLGATHER_SYMBOL;

/// The group all-gather of the absorbed queries and of the indexer's
/// per-rank selections, and the exchange of the partial blocks to their
/// owners.
pub const PHASED_SYMBOL: &str = phased::SYMBOL;

/// The merge and localization geometry of `shape`: the rank's latent heads
/// over the latent width, the latent attention's key list and pages, and
/// the prefill capacity.
pub fn geometry(shape: Shape) -> context_parallel::Geometry {
    let latent = latent_geometry(shape);
    context_parallel::Geometry {
        heads: latent.heads,
        width: latent.latent,
        page_size: PAGE_TOKENS as u32,
        list_width: latent.max_keys,
        capacity: PREFILL_TOKENS,
    }
}

pub fn entries(shape: Shape) -> Vec<context_parallel::Entry> {
    context_parallel::entries(PROFILE, geometry(shape))
}

/// Every embedded context-parallel symbol: the entries, the group
/// all-gather and the phased group gather.
pub fn symbols(shape: Shape) -> Vec<String> {
    entries(shape)
        .iter()
        .map(context_parallel::Entry::symbol)
        .chain([ALLGATHER_SYMBOL.to_owned(), PHASED_SYMBOL.to_owned()])
        .collect()
}

/// The names of the entries' cubin, the group all-gather's and the phased
/// group gather's.
pub fn cubin_names() -> [String; 3] {
    [
        format!("context_parallel_{}_sm120", PROFILE.tag()),
        format!("comm_group_{}_sm120", PROFILE.tag()),
        format!("comm_group_phased_{}_sm120", PROFILE.tag()),
    ]
}

/// Whether cubin `name` is one of [`cubin_names`]: an instance loads them
/// only at a context-parallel size above 1, but for the group all-gather's,
/// which it also loads with the MTP drafter, whose argmax gathers its pairs
/// through the all-gather (`crate::draft_argmax`).
pub fn is_cubin(name: &str) -> bool {
    cubin_names().iter().any(|cubin| cubin == name)
}

/// The entries' cubin and the group all-gather's, both in [`FAMILY`], then
/// the phased group gather's in [`PHASED_FAMILY`].
pub fn cubin_specs(shape: Shape) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let entries = context_parallel::cubin_spec(PROFILE, &entries(shape))
        .map_err(|refusal| format!("context-parallel cubin: {refusal}"))?;
    let mut gather = kernels_sm120::comm::cubin_spec();
    let [_, name, phased_name] = cubin_names();
    gather.name = name;
    gather
        .entries
        .retain(|entry| entry.symbol == ALLGATHER_SYMBOL);
    if gather.entries.len() != 1 {
        return Err(format!("the comm engine has no {ALLGATHER_SYMBOL}"));
    }
    Ok(vec![
        (FAMILY, entries),
        (FAMILY, gather),
        (PHASED_FAMILY, phased::cubin_spec(phased_name)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_follow_the_latent_geometry() {
        let shape = Shape::TP4;
        let g = geometry(shape);
        assert_eq!(
            (g.heads, g.width, g.page_size, g.list_width, g.capacity),
            (16, 512, 64, 2051, 8192)
        );
        let names = symbols(shape);
        assert_eq!(
            names,
            [
                "chop_context_parallel_glm5next_partial_merge2_v1",
                "chop_context_parallel_glm5next_localize_keys2_v1",
                "chop_context_parallel_glm5next_localize_positions2_v1",
                "chop_context_parallel_glm5next_partial_merge4_v1",
                "chop_context_parallel_glm5next_localize_keys4_v1",
                "chop_context_parallel_glm5next_localize_positions4_v1",
                "chop_comm_allgather_f32_v1",
                "chop_comm_group_phased_f32_v1",
            ]
        );
        let specs = cubin_specs(shape).unwrap();
        assert_eq!(
            specs.iter().map(|(family, _)| *family).collect::<Vec<_>>(),
            [FAMILY, FAMILY, PHASED_FAMILY]
        );
        assert_eq!(specs[0].1.entries.len(), 6);
        assert!(specs.iter().all(|(_, spec)| is_cubin(&spec.name)));
        assert_eq!(specs[1].1.entries.len(), 1);
        assert_eq!(specs[2].1.entries.len(), 1);
        // The comm cubin keeps its own entries; the all-gather lives here
        // only.
        let all = super::super::engines::cubin_specs(shape).unwrap();
        let holders = all
            .iter()
            .filter(|(_, spec)| {
                spec.entries
                    .iter()
                    .any(|entry| entry.symbol == ALLGATHER_SYMBOL)
            })
            .count();
        assert_eq!(holders, 1);
    }
}
