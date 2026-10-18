//! Compressed page geometries with independent lock records and address controls.
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernels_sm120::latent::sources::{self, Entry};
use kernels_sm120::{Profile, ProfileNames};

pub const FAMILY: &str = "latent_sources_pages";
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-native-pages", tag: "latent_native_pages",
});
pub const REQUIRED: usize = 12;

pub fn catalog() -> Vec<Entry> {
    let mut all = Vec::new();
    for page_size in [32, 64] {
        for output in [LatentAttentionOutput::Normalized, LatentAttentionOutput::Partial] {
            all.extend(sources::paged_entries(PROFILE, LatentAttentionQuery::Attention {
                heads: 4, latent: 512, max_keys: 256,
                softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
                page_size, kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
                phase: AttentionPhase::Prefill, output, sink: true,
                tail_rope_dims: 64, second_fp4_keys: true, capacity: 4,
            }).expect("native compressed page geometry"));
        }
    }
    assert_eq!(all.len(), REQUIRED);
    all
}

pub fn mutants(entry: &Entry) -> Vec<EntrySpec> {
    let correct = entry.entry_specs().expect("native source entry").remove(0);
    let start = correct.body.find("attention<").unwrap() + "attention<".len();
    let end = correct.body.find(">(q,").unwrap();
    let traits = &correct.body[start..end];
    [("wrong_page", 64, 64, 0), ("scales64", 32, 64, 0), ("next_page", 32, 32, 1)]
        .into_iter().map(|(name, rows, scales, bias)| {
            let mut spec = correct.clone();
            spec.symbol = format!("chop_latent_sources_{}_{name}_v1", entry.name);
            spec.body.replace_range(start..end, &format!("chop::latent::page_controls::Geometry<{traits},{rows},{scales},{bias}>"));
            spec
        }).collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = catalog();
    let mut specs = Vec::new();
    for page in [32, 64] {
        let page_entries = entries.iter().filter(|e| matches!(e.query,
            LatentAttentionQuery::Attention { page_size, .. } if page_size == page)).cloned().collect::<Vec<_>>();
        specs.push(sources::paged_cubin_spec(PROFILE, page, &page_entries).unwrap());
    }
    let control = entries.iter().find(|e| matches!(e.query, LatentAttentionQuery::Attention {
        page_size: 32, phase: AttentionPhase::Decode, output: LatentAttentionOutput::Normalized, .. })).unwrap();
    let mut spec = sources::paged_cubin_spec(PROFILE, 32, std::slice::from_ref(control)).unwrap();
    spec.name = "latent_sources_latent_native_pages_mutants_sm120".into();
    spec.headers.push("#include \"latent/page_controls.cuh\"".into());
    spec.entries = mutants(control);
    specs.push(spec);
    crate::declared(&["latent_sources_latent_native_pages_p32_sm120", "latent_sources_latent_native_pages_p64_sm120", "latent_sources_latent_native_pages_mutants_sm120"], specs)
}
