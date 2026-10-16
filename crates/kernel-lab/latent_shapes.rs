//! The latent attention lab shape, built only by kernel-lab: 16 local heads
//! (64 heads at TP4), latent width 512, query and value head widths 256,
//! 64-token pages, softmax scale 1/16 and key lists of 2048 selected tokens
//! plus a 3-token incomplete pool tail. Its one cubin builds in its own lock
//! family, `kernels.lock.d/latent_attention.lock`.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::DType;
use kernels_sm120::latent::{Entry, Geometry, Query, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "latent-h16-l512",
    tag: "latent_h16_l512",
});

pub const GEOMETRY: Geometry = Geometry {
    heads: 16,
    latent: 512,
    page_size: 64,
    max_keys: 2051,
    softmax_scale: 0.0625,
    nope: 256,
    v_dim: 256,
};

/// The lock family of the latent attention cubin.
pub const FAMILY: &str = "latent_attention";

/// Latent widths of the lab-only KV write entries (`kv_write_{dtype}_w{width}`):
/// the narrowest, two that are not powers of two, and the widest.
pub const WRITE_WIDTHS: [u32; 4] = [128, 384, 640, 1024];

/// The served entries, F32-output forms of the two projections (named
/// `{name}_f32`) that expose their accumulation to the 128-K test, and KV
/// writes at [`WRITE_WIDTHS`].
pub fn catalog() -> Vec<Entry> {
    let mut all = entries(PROFILE, GEOMETRY);
    let f32_forms = all
        .iter()
        .filter_map(|entry| match entry.query {
            Query::HeadGemm(query) => Some(Entry {
                name: format!("{}_f32", entry.name),
                query: Query::HeadGemm(kernel_core::latent_attention::HeadBatchedGemmQuery {
                    output: DType::F32,
                    ..query
                }),
                ..entry.clone()
            }),
            Query::Latent(_) => None,
        })
        .collect::<Vec<_>>();
    all.extend(f32_forms);
    for width in WRITE_WIDTHS {
        let geometry = Geometry {
            latent: width,
            ..GEOMETRY
        };
        all.extend(
            entries(PROFILE, geometry)
                .into_iter()
                .filter(|entry| {
                    matches!(
                        entry.query,
                        Query::Latent(
                            kernel_core::latent_attention::LatentAttentionQuery::KvWrite { .. }
                        )
                    )
                })
                .map(|entry| Entry {
                    name: format!("{}_w{width}", entry.name),
                    ..entry
                }),
        );
    }
    all
}

/// A lab-only PDL producer, `chop_latent_{tag}_pdl_copy_v1(src, dst, words,
/// spin)`: it releases its dependents at entry, spins for `spin` clock
/// cycles, then copies `words` u32 words from `src` to `dst`. A dependent
/// that reads `dst` before its PDL wait sees stale bytes.
pub fn pdl_copy_symbol() -> String {
    format!("chop_latent_{}_pdl_copy_v1", PROFILE.tag())
}

fn pdl_copy_entry() -> EntrySpec {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    EntrySpec::new(
        pdl_copy_symbol(),
        vec![
            pointer("src", "const uint32_t"),
            pointer("dst", "uint32_t"),
            EntryParam::U64 {
                name: "words".into(),
            },
            EntryParam::U64 {
                name: "spin".into(),
            },
        ],
        "chop::ptx::griddep_launch_dependents();\
         const long long start = clock64();\
         while (clock64() - start < (long long)spin) {}\
         for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;\
              i += uint64_t(gridDim.x) * blockDim.x) dst[i] = src[i];"
            .into(),
        256,
        1,
    )
}

/// The lab's latent attention cubin.
pub fn cubin() -> CubinSpec {
    let mut spec = cubin_spec(PROFILE, &catalog()).expect("the lab's latent entries are valid");
    spec.entries.push(pdl_copy_entry());
    spec
}

/// The cubin's spec name, for `Session::module`.
pub fn cubin_name() -> String {
    format!("latent_attention_{}_sm120", PROFILE.tag())
}
