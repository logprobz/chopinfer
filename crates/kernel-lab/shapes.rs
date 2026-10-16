//! Synthetic head shapes built only by kernel-lab, never by a bundle.
use kernel_core::contract::{AttentionPhase, DType};
use kernels_sm120::attention::{self, CacheLayout, Shape, gated, sparse};
use kernels_sm120::invariance::{Batch, Invariance};
use kernels_sm120::{ModelCatalog, Profile, ProfileNames};

/// Lab geometries with separate profiles for new kernel entries.
pub const ROW_PROFILES: [Profile; 6] = [
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d128-r32",
        tag: "lab_shapes_d128_r32",
    }),
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d128-r64",
        tag: "lab_shapes_d128_r64",
    }),
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d256-r32",
        tag: "lab_shapes_d256_r32",
    }),
    // Keep the original dense cubin byte-identical when adding gated rows.
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d128-r32-gated",
        tag: "lab_shapes_d128_r32_gated",
    }),
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d256-r32-fold",
        tag: "lab_shapes_d256_r32_fold",
    }),
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-d128-r64-g16",
        tag: "lab_shapes_d128_r64_g16",
    }),
];

/// Head-128 sparse attention at one and two n8 tiles.
pub const SPARSE_PROFILES: [Profile; 2] = [ROW_PROFILES[1], ROW_PROFILES[5]];

/// Indexer-only instances: head count and rotary width are part of each tag.
pub const INDEXER_SHAPES: [(Profile, u32, u32); 5] = [
    (
        Profile::synthetic(ProfileNames {
            name: "lab-shapes-i4-r32",
            tag: "lab_shapes_i4_r32",
        }),
        4,
        32,
    ),
    (
        Profile::synthetic(ProfileNames {
            name: "lab-shapes-i8-r32",
            tag: "lab_shapes_i8_r32",
        }),
        8,
        32,
    ),
    (
        Profile::synthetic(ProfileNames {
            name: "lab-shapes-i8-r64",
            tag: "lab_shapes_i8_r64",
        }),
        8,
        64,
    ),
    (
        Profile::synthetic(ProfileNames {
            name: "lab-shapes-i16-r32",
            tag: "lab_shapes_i16_r32",
        }),
        16,
        32,
    ),
    (
        Profile::synthetic(ProfileNames {
            name: "lab-shapes-i16-r64",
            tag: "lab_shapes_i16_r64",
        }),
        16,
        64,
    ),
];

/// All profiles owned by this lab-only catalog.
const ORIGINAL_PROFILES: [Profile; 11] = [
    ROW_PROFILES[0],
    ROW_PROFILES[1],
    ROW_PROFILES[2],
    ROW_PROFILES[3],
    ROW_PROFILES[4],
    INDEXER_SHAPES[0].0,
    INDEXER_SHAPES[1].0,
    INDEXER_SHAPES[2].0,
    INDEXER_SHAPES[3].0,
    INDEXER_SHAPES[4].0,
    ROW_PROFILES[5],
];

/// Row, dense tile and fold instances covered by the lab gates.
#[derive(Clone, Copy)]
pub struct QualificationShape {
    pub profile: Profile,
    pub head_dim: u32,
    pub rotary: u32,
    pub q_heads: u32,
    names: [[&'static str; 3]; 2],
}

macro_rules! qualification_shapes {
    ($table:ident, $prefix:literal; $(($dim:literal, $rotary:literal, $heads:literal)),* $(,)?) => {
        pub const $table: &[QualificationShape] = &[$(
            QualificationShape {
                profile: Profile::synthetic(ProfileNames {
                    name: concat!("lab-shapes-", $prefix, "-d", $dim, "-r", $rotary, "-g", $heads),
                    tag: concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads),
                }),
                head_dim: $dim,
                rotary: $rotary,
                q_heads: $heads,
                names: [
                    [concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_bf16_prefill"),
                     concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_bf16_decode"),
                     concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_bf16_verify")],
                    [concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_fp8_prefill"),
                     concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_fp8_decode"),
                     concat!("lab_shapes_", $prefix, "_d", $dim, "_r", $rotary, "_g", $heads, "_fp8_verify")],
                ],
            },
        )*];
    };
}

qualification_shapes! { QUALIFICATION_SHAPES, "w7";
    (128, 32, 1),
    (128, 32, 2),
    (128, 32, 3),
    (128, 32, 4),
    (128, 32, 5),
    (128, 32, 6),
    (128, 32, 7),
    (128, 32, 8),
    (128, 32, 9),
    (128, 32, 10),
    (128, 32, 11),
    (128, 32, 12),
    (128, 32, 13),
    (128, 32, 14),
    (128, 32, 15),
    (128, 32, 16),
    (128, 64, 12),
    (128, 64, 16),
    (256, 32, 1),
    (256, 32, 2),
    (256, 32, 3),
    (256, 32, 5),
    (256, 32, 7),
    (256, 32, 8),
    (256, 32, 10),
    (256, 32, 12),
    (256, 32, 14),
    (256, 32, 16),
}

qualification_shapes! { NEW_ROW_SHAPES, "w7d";
    (128, 64, 2), (128, 64, 4), (128, 64, 8),
    (256, 64, 8), (256, 64, 16),
}

#[derive(Clone, Copy)]
pub struct SparseShape {
    pub profile: Profile,
    pub head_dim: u32,
    pub q_heads: u32,
    pub top_blocks: u32,
    pub index_heads: u32,
    pub rotary: u32,
    pub indexer: bool,
    pub attention: bool,
}

macro_rules! sparse_shapes {
    ($(($dim:literal, $q:literal, $top:literal, $heads:literal, $rotary:literal, $indexer:literal, $attention:literal)),* $(,)?) => {
        pub const NEW_SPARSE_SHAPES: &[SparseShape] = &[$(
            SparseShape {
                profile: Profile::synthetic(ProfileNames {
                    name: concat!("lab-shapes-w7d-d", $dim, "-g", $q, "-k", $top, "-i", $heads, "-r", $rotary),
                    tag: concat!("lab_shapes_w7d_d", $dim, "_g", $q, "_k", $top, "_i", $heads, "_r", $rotary),
                }),
                head_dim: $dim, q_heads: $q, top_blocks: $top,
                index_heads: $heads, rotary: $rotary,
                indexer: $indexer, attention: $attention,
            },
        )*];
    };
}

sparse_shapes! {
    (128, 2, 256, 4, 64, false, true),
    (128, 4, 256, 4, 64, false, true),
    (128, 8, 256, 4, 64, false, true),
    (128, 16, 256, 4, 64, false, true),
    (128, 2, 512, 4, 64, false, true),
    (128, 4, 512, 4, 64, false, true),
    (128, 8, 512, 4, 64, false, true),
    (128, 12, 512, 4, 64, false, true),
    (256, 6, 256, 4, 64, true, true),
    (256, 8, 256, 4, 64, false, true),
    (256, 12, 256, 4, 64, false, true),
    (256, 16, 256, 4, 64, false, true),
    (256, 8, 512, 4, 64, false, true),
    (256, 16, 512, 4, 64, false, true),
    (256, 6, 256, 4, 32, true, false),
    (256, 6, 256, 8, 64, true, false),
    (256, 6, 256, 16, 32, true, false),
    (256, 6, 256, 16, 64, true, false),
}

#[allow(dead_code)] // Also included by the build script.
pub fn indexer_shapes() -> impl Iterator<Item = (Profile, u32, u32)> {
    INDEXER_SHAPES.into_iter().chain(
        NEW_SPARSE_SHAPES
            .iter()
            .filter(|s| s.indexer)
            .map(|s| (s.profile, s.index_heads, s.rotary)),
    )
}

#[allow(dead_code)] // Also included by the build script.
pub fn sparse_attention_profiles() -> impl Iterator<Item = Profile> {
    SPARSE_PROFILES
        .into_iter()
        .chain(WIDE_SPARSE_PROFILES)
        .chain(
            NEW_SPARSE_SHAPES
                .iter()
                .filter(|s| s.attention)
                .map(|s| s.profile),
        )
}

/// The two sparse instances built by the widened fixture at TP4 and TP2.
pub const WIDE_SPARSE_PROFILES: [Profile; 2] = [
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-w7-wide-tp4",
        tag: "lab_shapes_w7_wide_tp4",
    }),
    Profile::synthetic(ProfileNames {
        name: "lab-shapes-w7-wide-tp2",
        tag: "lab_shapes_w7_wide_tp2",
    }),
];

pub const PROFILES: [Profile;
    11 + QUALIFICATION_SHAPES.len() + 2 + NEW_ROW_SHAPES.len() + NEW_SPARSE_SHAPES.len()] = {
    let mut profiles = [ORIGINAL_PROFILES[0];
        11 + QUALIFICATION_SHAPES.len() + 2 + NEW_ROW_SHAPES.len() + NEW_SPARSE_SHAPES.len()];
    let mut i = 0;
    while i < ORIGINAL_PROFILES.len() {
        profiles[i] = ORIGINAL_PROFILES[i];
        i += 1;
    }
    let mut j = 0;
    while j < QUALIFICATION_SHAPES.len() {
        profiles[i] = QUALIFICATION_SHAPES[j].profile;
        i += 1;
        j += 1;
    }
    profiles[i] = WIDE_SPARSE_PROFILES[0];
    profiles[i + 1] = WIDE_SPARSE_PROFILES[1];
    i += 2;
    let mut j = 0;
    while j < NEW_ROW_SHAPES.len() {
        profiles[i] = NEW_ROW_SHAPES[j].profile;
        i += 1;
        j += 1;
    }
    let mut j = 0;
    while j < NEW_SPARSE_SHAPES.len() {
        profiles[i] = NEW_SPARSE_SHAPES[j].profile;
        i += 1;
        j += 1;
    }
    profiles
};

/// Row profiles with the new per-rank counts, including both Q warp counts.
#[allow(dead_code)] // Also included by the build script.
pub fn row_profiles() -> impl Iterator<Item = Profile> {
    ROW_PROFILES.into_iter().chain(
        QUALIFICATION_SHAPES
            .iter()
            .chain(NEW_ROW_SHAPES)
            .map(|s| s.profile),
    )
}

/// Gated geometry of each lab profile, including the separately compiled folds.
pub fn gated_config(profile: Profile) -> gated::Config {
    if let Some(s) = QUALIFICATION_SHAPES
        .iter()
        .chain(NEW_ROW_SHAPES)
        .find(|s| s.profile == profile)
    {
        return gated::Config {
            q_heads: s.q_heads,
            head_dim: s.head_dim,
            rotary: s.rotary,
            theta: if s.head_dim == 128 && s.rotary == 32 {
                5e6
            } else {
                1e7
            },
            sections: if s.rotary == 32 {
                [6, 5, 5]
            } else {
                [11, 11, 10]
            },
            ..gated_config(ROW_PROFILES[0])
        };
    }
    let index = ROW_PROFILES
        .iter()
        .position(|&p| p == profile)
        .expect("lab shape profile");
    let rotary = if [1, 5].contains(&index) { 64 } else { 32 };
    gated::Config {
        q_heads: if index == 5 { 16 } else { 6 },
        kv_heads: 1,
        head_dim: if [0, 1, 3, 5].contains(&index) {
            128
        } else {
            256
        },
        rotary,
        theta: 1e7,
        sections: if rotary == 32 {
            [6, 5, 5]
        } else {
            [11, 11, 10]
        },
        eps: 1e-6,
        page: 64,
        max_rows: 8192,
    }
}

/// Sparse geometry with an explicit indexer head count.
pub fn sparse_geometry(profile: Profile, index_heads: u32) -> sparse::Geometry {
    if let Some(s) = NEW_SPARSE_SHAPES.iter().find(|s| s.profile == profile) {
        return sparse::Geometry {
            head_dim: s.head_dim,
            q_heads: s.q_heads,
            top_blocks: s.top_blocks,
            index_heads: s.index_heads,
            rotary_dim: s.rotary,
            mrope: if s.rotary == 32 {
                [6, 5, 5]
            } else {
                [11, 11, 10]
            },
            softmax_scale: 1.0 / (s.head_dim as f32).sqrt(),
            ..sparse_geometry(ROW_PROFILES[1], s.index_heads)
        };
    }
    if let Some(index) = WIDE_SPARSE_PROFILES.iter().position(|&p| p == profile) {
        return sparse::Geometry {
            index_heads: 8,
            q_heads: if index == 0 { 6 } else { 12 },
            head_dim: 128,
            theta: 5e6,
            top_blocks: 256,
            softmax_scale: 1.0 / 128_f32.sqrt(),
            ..sparse_geometry(ROW_PROFILES[0], 8)
        };
    }
    let g = if let Some(&(_, _, rotary)) = INDEXER_SHAPES.iter().find(|s| s.0 == profile) {
        gated::Config {
            rotary,
            sections: if rotary == 32 {
                [6, 5, 5]
            } else {
                [11, 11, 10]
            },
            ..gated_config(ROW_PROFILES[2])
        }
    } else {
        gated_config(profile)
    };
    sparse::Geometry {
        index_heads,
        index_dim: 128,
        rotary_dim: g.rotary,
        theta: g.theta,
        mrope: g.sections,
        eps: g.eps,
        block: 4,
        ring: 16,
        top_blocks: 512,
        max_context: 262144,
        q_heads: g.q_heads,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
        page_size: g.page,
        softmax_scale: 1.0 / (g.head_dim as f32).sqrt(),
    }
}

const ATTENTION_NAMES: [[&str; 3]; 2] = [
    [
        "lab_shapes_d128_r32_bf16_prefill",
        "lab_shapes_d128_r32_bf16_decode",
        "lab_shapes_d128_r32_bf16_verify",
    ],
    [
        "lab_shapes_d128_r32_fp8_prefill",
        "lab_shapes_d128_r32_fp8_decode",
        "lab_shapes_d128_r32_fp8_verify",
    ],
];

fn attention_entries() -> Vec<attention::Variant> {
    let mut entries = Vec::new();
    // Dense attention has no rotary parameter. Head 256 is already covered
    // by the production catalog; build head 128 only once.
    let profile = ROW_PROFILES[0];
    let g = gated_config(profile);
    let names = ATTENTION_NAMES;
    for (dtype, names) in [(DType::Bf16, names[0]), (DType::Fp8E4M3, names[1])] {
        for (phase, name) in [
            AttentionPhase::Prefill,
            AttentionPhase::Decode,
            AttentionPhase::Verify,
        ]
        .into_iter()
        .zip(names)
        {
            let mut entry = attention::variant(
                name,
                profile,
                phase,
                Shape {
                    q_heads: g.q_heads,
                    kv_heads: g.kv_heads,
                    qk_dim: g.head_dim,
                    v_dim: g.head_dim,
                    page_size: g.page,
                    kv_dtype: dtype,
                    layout: CacheLayout::Paged,
                    window_left: None,
                    sink: false,
                    softcap: None,
                    causal: true,
                },
            );
            entry.query.max_context_capacity = 262144;
            entry.invariance = if phase == AttentionPhase::Prefill {
                Invariance::PREFILL
            } else {
                Invariance {
                    batch: Batch::Independent,
                    ..Invariance::NONE
                }
            };
            entries.push(entry);
        }
    }
    for (profile, names) in [
        (
            ROW_PROFILES[1],
            [
                "lab_shapes_d128_r64_bf16_decode",
                "lab_shapes_d128_r64_fp8_decode",
            ],
        ),
        (
            ROW_PROFILES[3],
            [
                "lab_shapes_d128_r32_gated_bf16_decode",
                "lab_shapes_d128_r32_gated_fp8_decode",
            ],
        ),
        (
            ROW_PROFILES[4],
            [
                "lab_shapes_d256_r32_fold_bf16_decode",
                "lab_shapes_d256_r32_fold_fp8_decode",
            ],
        ),
    ] {
        let g = gated_config(profile);
        for (dtype, name) in [(DType::Bf16, names[0]), (DType::Fp8E4M3, names[1])] {
            let mut entry = attention::variant(
                name,
                profile,
                AttentionPhase::Decode,
                Shape {
                    q_heads: g.q_heads,
                    kv_heads: g.kv_heads,
                    qk_dim: g.head_dim,
                    v_dim: g.head_dim,
                    page_size: g.page,
                    kv_dtype: dtype,
                    layout: CacheLayout::Paged,
                    window_left: None,
                    sink: false,
                    softcap: None,
                    causal: true,
                },
            );
            entry.query.max_context_capacity = 262144;
            entry.gated_fold = Some(g.fold());
            // The ordinary decode and merge entries read an already written
            // cache. The folded entries keep their no-claim declaration.
            entry.invariance = Invariance {
                batch: Batch::Independent,
                ..Invariance::NONE
            };
            entries.push(entry);
        }
    }
    for s in QUALIFICATION_SHAPES.iter().chain(NEW_ROW_SHAPES) {
        // Direct dense is selected only at TP4, with at most eight Q heads.
        // Head-256 groups two and four fail the FP8 prefill stack audit.
        if s.q_heads > 8 || (s.head_dim == 256 && [2, 4].contains(&s.q_heads)) {
            continue;
        }
        let g_profile = s.profile;
        let g = gated_config(s.profile);
        for (dtype, names) in [(DType::Bf16, s.names[0]), (DType::Fp8E4M3, s.names[1])] {
            for (phase, name) in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ]
            .into_iter()
            .zip(names)
            {
                // The new head-256 instances need folded decode, while their
                // prefill and verify paths are outside this qualification.
                if ((g.head_dim == 256 || NEW_ROW_SHAPES.iter().any(|s| s.profile == g_profile))
                    && phase != AttentionPhase::Decode)
                    || (phase == AttentionPhase::Verify && !g.q_heads.is_multiple_of(2))
                {
                    continue;
                }
                let mut entry = attention::variant(
                    name,
                    s.profile,
                    phase,
                    Shape {
                        q_heads: g.q_heads,
                        kv_heads: g.kv_heads,
                        qk_dim: g.head_dim,
                        v_dim: g.head_dim,
                        page_size: g.page,
                        kv_dtype: dtype,
                        layout: CacheLayout::Paged,
                        window_left: None,
                        sink: false,
                        softcap: None,
                        causal: true,
                    },
                );
                entry.query.max_context_capacity = 262144;
                entry.invariance = if phase == AttentionPhase::Prefill {
                    Invariance::PREFILL
                } else {
                    Invariance {
                        batch: Batch::Independent,
                        ..Invariance::NONE
                    }
                };
                if phase == AttentionPhase::Decode {
                    entry.gated_fold = Some(g.fold());
                }
                entries.push(entry);
            }
        }
    }
    entries
}

/// Profile that owns the norm/KV and output-gate rows for a folded variant.
#[allow(dead_code)] // Also included by the build script.
pub fn row_profile(profile: Profile) -> Profile {
    if profile == ROW_PROFILES[4] {
        ROW_PROFILES[2]
    } else {
        profile
    }
}

fn gated_entries() -> Vec<gated::Entry> {
    [ROW_PROFILES[1], ROW_PROFILES[2], ROW_PROFILES[3]]
        .into_iter()
        .chain(
            QUALIFICATION_SHAPES
                .iter()
                .chain(NEW_ROW_SHAPES)
                .map(|s| s.profile),
        )
        .flat_map(|profile| gated::entries(profile, gated_config(profile)))
        .collect()
}

fn sparse_entries() -> Vec<sparse::Entry> {
    use kernels_sm120::attention::qualification::EXACT_VERIFY_ROWS;
    let mut entries: Vec<_> = INDEXER_SHAPES
        .into_iter()
        .flat_map(|(profile, heads, _)| {
            sparse::indexer_entries(profile, sparse_geometry(profile, heads), EXACT_VERIFY_ROWS)
        })
        .collect();
    // The unchanged rotary-64 indexer is shared with the existing catalog.
    // Attention itself does not consume rotary parameters.
    for profile in SPARSE_PROFILES {
        entries.extend(sparse::attention_entries(
            profile,
            sparse_geometry(profile, 4),
            EXACT_VERIFY_ROWS,
        ));
    }
    entries.extend(sparse::indexer_entries(
        WIDE_SPARSE_PROFILES[0],
        sparse_geometry(WIDE_SPARSE_PROFILES[0], 8),
        EXACT_VERIFY_ROWS,
    ));
    for profile in WIDE_SPARSE_PROFILES {
        entries.extend(sparse::attention_entries(
            profile,
            sparse_geometry(profile, 8),
            EXACT_VERIFY_ROWS,
        ));
    }
    for s in NEW_SPARSE_SHAPES {
        let g = sparse_geometry(s.profile, s.index_heads);
        if s.indexer {
            entries.extend(sparse::indexer_entries(s.profile, g, EXACT_VERIFY_ROWS));
        }
        if s.attention {
            entries.extend(sparse::attention_entries(s.profile, g, EXACT_VERIFY_ROWS));
        }
    }
    entries
}

/// Unique lab instances, excluded from every production bundle catalog.
pub static CATALOG: ModelCatalog = ModelCatalog {
    name: "lab-shapes",
    profiles: &PROFILES,
    attention: attention_entries,
    gated: gated_entries,
    sparse: sparse_entries,
    ..ModelCatalog::EMPTY
};
