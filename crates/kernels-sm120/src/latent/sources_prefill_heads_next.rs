//! Shared-head modes with profiling or an exact single-split output epilogue.
pub use super::Mode;
use super::*;

pub const WORDS: usize = 14;
pub const STAGES: [&str; 6] = ["setup", "gather", "qk", "softmax", "pv", "epilogue"];

pub fn symbol(entry: &sources::Entry, asynchronous: bool, mode: Mode) -> String {
    format!(
        "chop_latent_sources_{}_head_{}_{}_v1",
        entry.name,
        if asynchronous { "async" } else { "shared" },
        mode.name()
    )
}

pub fn cubin_spec(
    profile: Profile,
    entries: &[sources::Entry],
    mode: Mode,
) -> Result<CubinSpec, Refusal> {
    let mut spec = super::cubin_spec(profile, &[])?;
    spec.name = format!(
        "latent_sources_prefill_head_{}_{}_sm120",
        mode.name(),
        profile.tag()
    );
    spec.headers = vec!["#include \"latent/sources_prefill_heads_async.cuh\"".into()];
    spec.entries = entries
        .iter()
        .map(|e| head_entry(e, e.name.ends_with("_decoded"), false, mode, false))
        .collect::<Result<_, _>>()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    #[test]
    fn finish_shared_layout_is_disjoint_and_fixed_r_stores_visit_every_bank() {
        let mut addresses = std::collections::BTreeSet::new();
        for group in 0..2 {
            for warp in 0..4 {
                for r in 0..2 {
                    for m in 0..8 {
                        for half in 0..2 {
                            let mut banks = std::collections::BTreeSet::new();
                            for lane in 0..32 {
                                let (g, c) = (lane / 4, lane % 4);
                                let at = (group * 8 + 2 * c + r) * 516
                                    + 4
                                    + warp * 128
                                    + m * 16
                                    + g
                                    + half * 8;
                                assert!(addresses.insert(at));
                                banks.insert(at % 32);
                            }
                            assert_eq!(banks.len(), 32);
                        }
                    }
                }
            }
        }
        assert_eq!(addresses.len(), 16 * 512);
        assert!((addresses.last().unwrap() + 1) * 4 <= 64 * 512 * 2);
        for head in 0..16 {
            for metadata in 0..4 {
                assert!(!addresses.contains(&(head * 516 + metadata)));
            }
        }
    }
}
