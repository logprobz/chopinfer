//! The projections, forms and lab tiles the verify-row tests cover.
use bundle_deepseek41::sm120::{
    f32_prefill, f32_split, f32_verify, f32_verify_rows, head_last, projection_reuse,
};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::f32_projection::{
    Entry, cached_spans,
    rows::{self, Candidate, Tile},
};
use model_deepseek41_contract::geometry::FLASH;
use std::path::Path;

pub const FAMILY: &str = "f32_verify_rows_lab";
pub const CUBIN: &str = "f32_verify_rows_lab";
/// The package's verify rows: 32 sequences of six rows.
pub const CAPACITY: u32 = 192;
pub const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_vector_v1",
    "chop_lab_mma_rate_v1",
];

/// One rank's exact F32 projection of Flash TP4 verify steps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub name: &'static str,
    pub n: u32,
    pub k: u32,
}

/// The rank's vocabulary head, the MoE router, a KV source's compressor
/// (`wkv` and `wgate`), and an index source's `weights_proj` and `wk`.
pub fn shapes() -> [Shape; 5] {
    let s = &FLASH.shape;
    [
        Shape {
            name: "head",
            n: s.vocab / 4,
            k: s.hidden,
        },
        Shape {
            name: "router",
            n: s.moe.experts,
            k: s.hidden,
        },
        Shape {
            name: "compressor",
            n: s.attention.head_dim,
            k: s.hidden,
        },
        Shape {
            name: "index_weights",
            n: s.indexer.heads,
            k: s.hidden,
        },
        Shape {
            name: "index_key",
            n: s.indexer.dim,
            k: s.attention.head_dim,
        },
    ]
}

pub fn shape(name: &str) -> Option<Shape> {
    shapes().into_iter().find(|s| s.name == name)
}

const fn tile(f: u32, r: u32, wf: u32, wr: u32, stages: u32, plane_k: u32, slices: u32) -> Tile {
    Tile {
        features: f,
        rows: r,
        warp_features: wf,
        warp_rows: wr,
        stages,
        plane_k,
        slices,
    }
}

/// The lab tiles of `s`: whole-K feature tiles for the head, and K slices
/// that spread the small projections over more CTAs.
pub fn tiles(s: Shape) -> Vec<Tile> {
    if s.n >= 4096 {
        vec![
            tile(128, 64, 32, 32, 2, 64, 1),
            tile(128, 64, 32, 32, 4, 32, 1),
            tile(128, 64, 32, 32, 3, 32, 1),
            tile(64, 64, 32, 32, 3, 64, 1),
            tile(64, 64, 32, 32, 5, 32, 1),
            tile(256, 32, 32, 32, 2, 64, 1),
            tile(256, 32, 32, 32, 4, 32, 1),
            tile(64, 96, 32, 32, 3, 32, 1),
            tile(128, 64, 16, 64, 2, 64, 1),
            // Decode rows: as many CTAs as one wave of two to four per SM
            // holds, each streaming its feature lines over the whole K.
            tile(96, 8, 32, 8, 3, 64, 1),
            tile(64, 8, 16, 8, 5, 32, 1),
            tile(48, 8, 16, 8, 5, 32, 1),
            tile(128, 8, 32, 8, 2, 64, 1),
            tile(32, 8, 16, 8, 4, 64, 1),
            tile(96, 16, 32, 16, 2, 64, 1),
            tile(64, 16, 16, 16, 4, 32, 1),
            tile(96, 32, 32, 32, 3, 32, 1),
            tile(64, 32, 32, 32, 3, 32, 1),
        ]
    } else {
        vec![
            tile(16, 16, 16, 8, 4, 64, 1),
            tile(16, 16, 16, 8, 4, 64, 4),
            tile(16, 16, 16, 8, 4, 64, 8),
            tile(32, 32, 16, 16, 4, 64, 4),
            tile(32, 32, 16, 16, 4, 64, 8),
            tile(64, 32, 32, 16, 3, 64, 8),
            tile(32, 64, 16, 32, 3, 64, 8),
            tile(16, 32, 16, 16, 4, 64, 5),
            tile(64, 64, 32, 16, 2, 64, 10),
            tile(32, 32, 16, 16, 3, 128, 8),
        ]
    }
}

/// The lab's name for `s`'s projection (an uncached entry at the verify
/// capacity), which stems the candidate symbols.
pub fn lab_entry(s: Shape) -> Entry {
    Entry::new(
        GemmQuery {
            m_capacity: CAPACITY,
            n: s.n,
            k: s.k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        OperationUse {
            phase: ExecutionPhase::Verify,
            capacity: CAPACITY,
            graph: false,
        },
        &format!("chop_lab_verify_f32_n{}_k{}", s.n, s.k),
    )
    .unwrap()
}

/// The lab tiles of `s` its K admits: a slice holds whole 128-K chains.
pub fn candidates(s: Shape) -> Vec<Candidate> {
    tiles(s)
        .into_iter()
        .filter(|t| (s.k / 128).is_multiple_of(t.slices))
        .map(|t| Candidate::new(lab_entry(s), t, CAPACITY).unwrap())
        .collect()
}

/// The registered split entry of `s` with the bundle's serving routes:
/// the prepared op of the served backend at every row count.
pub fn served(s: Shape) -> Entry {
    let entry = f32_split::entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| e.query().n == s.n && e.query().k == s.k)
        .expect("registered projection");
    let entry = f32_prefill::serving_entry(&FLASH, entry).unwrap();
    let entry = projection_reuse::head(&FLASH, 4, entry).unwrap();
    head_last::head(&FLASH, 4, entry).unwrap()
}

/// The registered split entry of `s` alone, the reference of every form.
pub fn split(s: Shape) -> Entry {
    f32_split::entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| e.query().n == s.n && e.query().k == s.k)
        .expect("registered projection")
}

pub fn spans(s: Shape) -> Option<cached_spans::Candidate> {
    f32_verify::entries()
        .unwrap()
        .into_iter()
        .find(|c| c.entry().query().n == s.n && c.entry().query().k == s.k)
}

/// The row tiles the bundle registers for `s` at every site, with the
/// rows each serves; none for a projection that keeps its other routes.
pub fn packed(s: Shape) -> Vec<(Candidate, [u32; 2])> {
    f32_verify_rows::SITES
        .into_iter()
        .flat_map(|site| f32_verify_rows::tiles(site, s.n, s.k).unwrap())
        .map(|t| (t.candidate, t.rows))
        .collect()
}

/// The lab cubin: every candidate and the timing probes.
pub fn spec() -> CubinSpec {
    let all: Vec<_> = shapes().into_iter().flat_map(candidates).collect();
    let mut spec = rows::cubin_spec(CUBIN, &all).unwrap();
    spec.sources_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    spec.include_dirs = vec!["crates/kernels-sm120/cuda".into()];
    spec.headers
        .push("#include \"crates/kernel-lab/cuda/probes/probe.cuh\"".into());
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let words = || EntryParam::U64 {
        name: "words".into(),
    };
    spec.entries.extend([
        EntrySpec::new(
            "chop_probe_gate_v1".into(),
            vec![ptr("flag", "const uint32_t")],
            "chop::probe::gate(flag);".into(),
            32,
            1,
        ),
        EntrySpec::new(
            "chop_probe_flush_v1".into(),
            vec![ptr("buffer", "uint32_t"), words()],
            "chop::probe::flush(buffer,words);".into(),
            256,
            1,
        ),
        EntrySpec::new(
            "chop_probe_read_vector_v1".into(),
            vec![
                ptr("input", "const uint4"),
                ptr("checksum", "unsigned int"),
                words(),
                EntryParam::U32 {
                    name: "repeats".into(),
                },
            ],
            "chop::probe::bandwidth_read_vector(input, checksum, words, repeats);".into(),
            256,
            1,
        ),
        // Independent BF16 MMAs from registers: the tensor rate ceiling.
        EntrySpec::new(
            "chop_lab_mma_rate_v1".into(),
            vec![
                ptr("output", "float"),
                EntryParam::U32 {
                    name: "iterations".into(),
                },
            ],
            "chop::ptx::A4 a; chop::ptx::B2 b; chop::ptx::D4 d[24]{}; \
             for (int i = 0; i < 4; ++i) a.v[i] = 0x3f803f80u ^ (threadIdx.x & 1u); \
             b.v[0] = b.v[1] = 0x3c003c00u; \
             for (uint32_t it = 0; it < iterations; ++it) { \
             _Pragma(\"unroll\") for (int j = 0; j < 24; ++j) chop::ptx::mma_bf16(d[j], a, b); } \
             float sum = 0.f; for (int j = 0; j < 24; ++j) sum += d[j].v[0] + d[j].v[3]; \
             if (sum == 1234.5f) output[threadIdx.x] = sum;"
                .into(),
            256,
            1,
        ),
    ]);
    spec
}

#[test]
fn cpu_lab_tiles_plan_every_timed_row_count() {
    let spec = spec();
    for s in shapes() {
        let reference = split(s);
        assert!(reference.uses_fused_split());
        for c in candidates(s) {
            for symbol in c.symbols() {
                assert!(spec.entries.iter().any(|e| e.symbol == symbol));
            }
            for rows in [1, 32, 33, 48, 64, 65, 96, 128, 160, 192] {
                c.plans(rows).unwrap();
            }
        }
        // Every timed verify row count above the decode bucket takes the
        // ordered split route on the served registration.
        let served = served(s);
        for rows in 33..=CAPACITY {
            assert_eq!(
                served.entry_for_rows(rows).symbol(),
                reference.symbol(),
                "{} rows={rows}",
                s.name
            );
        }
    }
    assert!(spans(shape("head").unwrap()).is_some());
    assert!(spans(shape("router").unwrap()).is_some());
    // The bundle's tiles are lab tiles, each at its site's bound.
    for s in shapes() {
        for site in f32_verify_rows::SITES {
            for t in f32_verify_rows::tiles(site, s.n, s.k).unwrap() {
                let c = t.candidate;
                assert!(candidates(s).iter().any(|lab| lab.tile() == c.tile()));
                assert_eq!(c.capacity(), site.bound);
            }
        }
    }
    assert_eq!(packed(shape("index_weights").unwrap()).len(), 0);
    assert_eq!(packed(shape("compressor").unwrap()).len(), 2);
}
