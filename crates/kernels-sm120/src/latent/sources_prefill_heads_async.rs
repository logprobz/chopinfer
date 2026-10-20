//! Direct BF16 copies and an ordered consumed-half PV pipeline.
use super::*;
use kernel_core::latent_attention::LatentKvFormat;

pub fn symbol(reference: &sources::Entry, pipeline: bool) -> String {
    format!(
        "chop_latent_sources_{}_head_{}_v1",
        reference.name,
        if pipeline { "pipeline" } else { "async" }
    )
}
pub fn cubin_spec(
    profile: Profile,
    entries: &[sources::Entry],
    pipeline: bool,
) -> Result<CubinSpec, Refusal> {
    for e in entries {
        if !matches!(
            e.query,
            LatentAttentionQuery::Attention {
                kv: LatentKvFormat::Bf16,
                ..
            }
        ) {
            return Err(Refusal(
                "asynchronous shared-head staging requires BF16 keys",
            ));
        }
    }
    let mut spec = super::cubin_spec(profile, entries)?;
    spec.name = format!(
        "latent_sources_prefill_head_{}_{}_sm120",
        if pipeline { "pipeline" } else { "async" },
        profile.tag()
    );
    spec.headers = vec!["#include \"latent/sources_prefill_heads_async.cuh\"".into()];
    for (entry, reference) in spec.entries.iter_mut().zip(entries) {
        entry.symbol = symbol(reference, pipeline);
        entry.body = entry.body.replace(
            "::head_shared_attention<",
            &format!("::head_async_attention<{pipeline},"),
        );
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    #[test]
    fn asynchronous_heads_keep_the_score_and_softmax_source() {
        let shared = include_str!("../../cuda/latent/sources_prefill_heads.cuh");
        let asynchronous = include_str!("../../cuda/latent/sources_prefill_heads_async.cuh");
        let prefix = |s: &str| {
            s.split("// QK: one chain")
                .nth(1)
                .unwrap()
                .split("// PV: one chain")
                .next()
                .unwrap()
                .lines()
                .map(str::trim)
                .take_while(|l| *l != "if constexpr (Pipeline) {")
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(prefix(shared), prefix(asynchronous));
        assert!(asynchronous.contains("if (kt == 1 && more)"));
        assert!(asynchronous.contains("attention::cp_async_wait<0>();"));
        let staging = include_str!("../../cuda/latent/sources_prefill_staging.cuh");
        assert!(staging.contains("tile_offset<Keys>(key, dim, phase)"));
        assert!(staging.contains("attention::cp_async_commit();"));
    }
}
