//! Exact BF16 window storage and its paired attention consumer.
use kernel_build::CubinSpec;
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::{glue, sources, window_stage};
use kernels_sm120::{Profile, ProfileNames};
pub const FAMILY: &str = "latent_window_bf16";
pub const TRUNCATE: &str = "chop_window_bf16_truncate_mutant_v1";
pub const SWAP: &str = "chop_window_bf16_planes_mutant_v1";
pub const STAGE_SWAP: &str = "chop_window_bf16_stage_planes_mutant_v1";
pub fn store() -> glue::Entry {
    glue::Entry {
        kernel: glue::Kernel::WindowStoreBf16,
        capacity: 320,
        symbol: "chop_window_store_bf16_v1".into(),
    }
}
pub fn stage() -> window_stage::Entry {
    window_stage::Entry {
        kv: LatentKvFormat::Bf16,
        capacity: 16,
        symbol: "chop_window_stage_bf16_v1".into(),
        mutation: None,
    }
}
const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "window-bf16-chain",
    tag: "window_bf16_chain",
});
pub fn attention() -> Vec<sources::Entry> {
    let mut query = crate::latent_window_chain::query();
    if let LatentAttentionQuery::Attention { kv, .. } = &mut query {
        *kv = LatentKvFormat::Bf16;
    }
    sources::entries(PROFILE, query)
        .unwrap()
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: kernel_core::contract::AttentionPhase::Prefill,
                    ..
                }
            )
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut store = glue::cubin_spec("latent_window_bf16_store_sm120", &[store()]);
    for (symbol, mutation) in [(TRUNCATE, 1), (SWAP, 2)] {
        let mut entry = store.entries[0].clone();
        entry.symbol = symbol.into();
        entry.body = entry
            .body
            .replace("store_bf16<>", &format!("store_bf16<{mutation}>"));
        if mutation == 1 {
            entry.params.push(kernel_build::EntryParam::DevicePointer {
                name: "unquantized".into(),
                cxx_type: "const float".into(),
            });
            entry.body = entry
                .body
                .replace("status_row_offset);", "status_row_offset,unquantized);");
        }
        store.entries.push(entry);
    }
    let mut stage = window_stage::cubin_spec("latent_window_bf16_stage_sm120", &[stage()]);
    // Pointer replacement in the test supplies a full-size unread plane.
    let mut swap = stage.entries[0].clone();
    swap.symbol = STAGE_SWAP.into();
    stage.entries.push(swap);
    vec![
        store,
        stage,
        sources::cubin_spec(PROFILE, &attention()).unwrap(),
    ]
}
