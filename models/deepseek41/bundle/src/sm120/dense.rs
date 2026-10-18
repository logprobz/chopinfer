//! The dense FP8 projections' input quantizers the bundle registers on
//! SM120 and its build compiles: the ceil E4M3/32 row quantizer
//! (`kernels_sm120::row::block_quantize`) at each input a dense projection
//! quantizes itself (`ops::dense::quantized_inputs`): the F32 `q_norm` and
//! `wo_a` outputs, rounded to BF16 before their amax, the BF16 attention
//! output and the BF16 decoded Engram rows. The routed MoE's registration quantizes the layer
//! input. The fold GEMM itself is the dense engine's registration.

use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::BlockQuantizeQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::block_quantize::{self, Entry};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "dense_quantize";

/// The name of `g`'s quantizer cubin.
pub fn cubin_name(g: &Geometry) -> String {
    format!("dense_{}_quantize_sm120", g.kernel_tag)
}

/// The inputs the dense projections quantize at `ranks` ranks, each width
/// and type: the query rank (both `wq_b`, F32), the rank's attention heads
/// (`wo_a`, BF16), its `wo_a` outputs (`wo_b`, F32) and, with Engram
/// layers, the decoded table rows (`engram.wkv`, BF16).
pub fn inputs(g: &Geometry, ranks: u32) -> Vec<(u32, DType)> {
    let a = &g.shape.attention;
    let e = &g.shape.engram;
    let mut inputs = Vec::new();
    let engram =
        (g.engram_layers() > 0).then_some(((e.max_ngram - 1) * e.heads * e.head_dim, DType::Bf16));
    for input in [
        (a.q_rank, DType::F32),
        (a.heads / ranks * a.head_dim, DType::Bf16),
        (a.o_groups / ranks * a.o_rank, DType::F32),
    ]
    .into_iter()
    .chain(engram)
    {
        if !inputs.contains(&input) {
            inputs.push(input);
        }
    }
    inputs
}

/// The quantizers of `g` at `ranks` ranks that neither the routed MoE nor
/// the latent glue registers: SM120 refuses a query two entries serve.
pub fn quantizers(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    check_ranks(g, ranks)?;
    let registered: Vec<BlockQuantizeQuery> = super::moe::quantizers(g, ranks)?
        .into_iter()
        .chain(super::latent_glue::quantizers())
        .map(|entry| entry.query)
        .collect();
    inputs(g, ranks)
        .into_iter()
        .map(|(width, input)| BlockQuantizeQuery {
            input,
            ..moe::quantizer(width)
        })
        .filter(|query| !registered.contains(query))
        .map(|query| {
            let input = match query.input {
                DType::F32 => "_f32",
                _ => "",
            };
            Entry::new(
                query,
                PREFILL_TOKENS,
                format!(
                    "chop_{}_dense_ceil{input}_w{}_v1",
                    g.kernel_tag, query.width
                ),
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

/// Each geometry's quantizer cubin, with every supported rank count's
/// widths.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries: Vec<Entry> = Vec::new();
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(g, ranks).is_err() {
                continue;
            }
            for entry in quantizers(g, ranks)? {
                if !entries.iter().any(|e| e.symbol == entry.symbol) {
                    entries.push(entry);
                }
            }
        }
        specs.push((
            FAMILY,
            block_quantize::cubin_spec(&cubin_name(g), &entries).map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(quantizers(g, ranks)?
        .into_iter()
        .fold(backend, Sm120::with_block_quantize))
}
