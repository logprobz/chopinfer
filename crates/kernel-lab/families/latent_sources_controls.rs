//! Native attention numerical witnesses and a per-split sink control.

use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::sources::{self, Entry};
use kernels_sm120::{Profile, ProfileNames};

pub const FAMILY: &str = "latent_sources_controls";
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-native-controls",
    tag: "latent_native_controls",
});
pub const CUBIN: &str = "latent_sources_latent_native_controls_sm120";
pub const REQUIRED: usize = 2;

pub fn catalog() -> Vec<Entry> {
    let entries = sources::entries(
        PROFILE,
        LatentAttentionQuery::Attention {
            heads: 4,
            latent: 512,
            max_keys: 256,
            softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
            page_size: 16,
            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
            phase: AttentionPhase::Prefill,
            output: LatentAttentionOutput::Normalized,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: true,
            capacity: 4,
        },
    )
    .expect("native controls geometry");
    let entries = entries
        .into_iter()
        .filter(|entry| {
            !matches!(
                entry.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Verify,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), REQUIRED);
    entries
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = catalog();
    let mut spec = sources::cubin_spec(PROFILE, &entries).expect("native controls cubin");
    let split = entries
        .iter()
        .find(|e| e.splits > 1)
        .expect("decode control");
    spec.entries
        .push(split.sink_per_split_entry_spec().expect("sink control"));
    crate::declared(&[CUBIN], vec![spec])
}
