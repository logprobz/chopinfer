//! Native window and paged compressed attention at shared generic shapes.

use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::sources::{self, Entry};
use kernels_sm120::{Profile, ProfileNames};

pub const FAMILY: &str = "latent_sources";
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-native-shared",
    tag: "latent_native_shared",
});
pub const CUBIN: &str = "latent_sources_latent_native_shared_sm120";
pub const REQUIRED: usize = 24;

pub fn catalog() -> Vec<Entry> {
    let mut all = Vec::new();
    for (heads, latent) in [(4, 128), (16, 512)] {
        for second in [false, true] {
            for output in [
                LatentAttentionOutput::Normalized,
                LatentAttentionOutput::Partial,
            ] {
                all.extend(
                    sources::entries(
                        PROFILE,
                        LatentAttentionQuery::Attention {
                            heads,
                            latent,
                            max_keys: if second { 256 } else { 128 },
                            softmax_scale_bits: (latent as f32).powf(-0.5).to_bits(),
                            page_size: 16,
                            kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
                            phase: AttentionPhase::Prefill,
                            output,
                            sink: true,
                            tail_rope_dims: 64,
                            second_fp4_keys: second,
                            capacity: 4,
                        },
                    )
                    .expect("native latent lab geometry"),
                );
            }
        }
    }
    assert_eq!(all.len(), REQUIRED);
    all
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![sources::cubin_spec(PROFILE, &catalog()).expect("native latent lab cubin")],
    )
}
