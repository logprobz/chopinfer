//! Qualification instances of every shared-head mode and key staging path.
use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::{sources, sources_prefill::heads, sources_prefill_cache};
use kernels_sm120::{Profile, ProfileNames};

pub const FAMILY: &str = "latent_head_modes";
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-head-modes",
    tag: "head_modes",
});
pub const ROWS: u32 = 8192;

pub fn entries() -> Vec<sources::Entry> {
    let mut entries = Vec::new();
    for heads in [64, 32, 16] {
        for page in [64, 32] {
            let query = LatentAttentionQuery::Attention {
                heads,
                latent: 512,
                page_size: page,
                max_keys: 640,
                softmax_scale_bits: (1.0 / 512.0_f32.sqrt()).to_bits(),
                kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
                sink: true,
                tail_rope_dims: 64,
                second_fp4_keys: true,
                output: LatentAttentionOutput::Normalized,
                phase: AttentionPhase::Prefill,
                capacity: ROWS,
            };
            let mut entry = sources::entries(PROFILE, query).unwrap().remove(0);
            entry.name.push_str(&format!("_p{page}"));
            entries.push(entry);
        }
    }
    let decoded = entries
        .iter()
        .map(sources_prefill_cache::decoded_entry)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    entries.extend(decoded);
    entries
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let decoded = &entries[6..];
    vec![
        heads::cubin_spec(PROFILE, &entries).unwrap(),
        heads::asynchronous::cubin_spec(PROFILE, decoded, false).unwrap(),
        heads::asynchronous::cubin_spec(PROFILE, decoded, true).unwrap(),
        heads::next::cubin_spec(PROFILE, &entries, heads::Mode::Profile).unwrap(),
        heads::next::cubin_spec(PROFILE, &entries, heads::Mode::Finish).unwrap(),
        heads::empty::cubin_spec(PROFILE, &entries).unwrap(),
    ]
}
