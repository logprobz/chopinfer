//! BF16 storage of exact native latent operands, paired with native entries.
use kernel_build::CubinSpec;
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::sources::{self, Entry};
use kernels_sm120::{Profile, ProfileNames};

pub const FAMILY: &str = "latent_sources_bf16";
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-native-bf16",
    tag: "latent_native_bf16",
});
pub const REQUIRED: usize = 18;

pub fn native_catalog() -> Vec<Entry> {
    let mut entries = crate::latent_sources_pages::catalog();
    entries.extend(crate::latent_sources::catalog().into_iter().filter(|e| {
        matches!(
            e.query,
            LatentAttentionQuery::Attention {
                latent: 512,
                second_fp4_keys: false,
                ..
            }
        )
    }));
    assert_eq!(entries.len(), REQUIRED);
    entries
}

pub fn paired(entry: &Entry) -> Entry {
    let mut query = entry.query;
    let LatentAttentionQuery::Attention {
        kv,
        phase,
        second_fp4_keys,
        ..
    } = &mut query
    else {
        unreachable!()
    };
    *kv = LatentKvFormat::Bf16;
    let phase = *phase;
    let entries = if *second_fp4_keys {
        sources::paged_entries(PROFILE, query)
    } else {
        sources::entries(PROFILE, query)
    }
    .unwrap();
    entries
        .into_iter()
        .find(|e| matches!(e.query, LatentAttentionQuery::Attention { phase: p, .. } if p == phase))
        .unwrap()
}

pub fn catalog() -> Vec<Entry> {
    native_catalog().iter().map(paired).collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &["latent_sources_latent_native_bf16_sm120"],
        vec![sources::cubin_spec(PROFILE, &catalog()).unwrap()],
    )
}
