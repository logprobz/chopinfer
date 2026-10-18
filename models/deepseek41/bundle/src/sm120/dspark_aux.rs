//! Shared row kernels needed by the drafter, in its own lock family.

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::BlockQuantizeQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::{argmax, block_quantize, rounding};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "dspark_aux";

use super::dspark_entries::rounding_symbol;

fn pair(g: &Geometry) -> rounding::PartialRounding {
    let width = g.shape.dspark.markov_rank;
    rounding::PartialRounding {
        width,
        round: rounding::entry(
            rounding::Rounding::Round,
            width,
            rounding_symbol(g, rounding::Rounding::Round),
        ),
        widen: rounding::entry(
            rounding::Rounding::Widen,
            width,
            rounding_symbol(g, rounding::Rounding::Widen),
        ),
        invariance: rounding::shared_invariance(),
    }
}

fn embedding(g: &Geometry, ranks: u32) -> kernels_sm120::row::RowInstantiation {
    let mut entry = kernels_sm120::row::embedding(
        super::rows::profile(g),
        &format!("dspark_markov_{}_tp{ranks}", g.kernel_tag),
        g.shape.vocab / ranks,
        g.shape.dspark.markov_rank,
        true,
    );
    entry.invariance = kernels_sm120::row::row_local();
    entry
}

fn quantizers(g: &Geometry) -> Result<Vec<block_quantize::Entry>, String> {
    [
        (
            g.shape.hidden * g.shape.dspark.targets.len() as u32,
            DType::Bf16,
        ),
        (g.shape.hidden, DType::F32),
    ]
    .into_iter()
    .map(|(width, input)| {
        block_quantize::Entry::new(
            BlockQuantizeQuery {
                input,
                ..model_deepseek41_contract::moe::quantizer(width)
            },
            PREFILL_TOKENS,
            format!(
                "chop_{}_dspark_quantize_w{width}_{input:?}_v1",
                g.kernel_tag
            )
            .to_lowercase(),
        )
        .map_err(|error| error.to_string())
    })
    .collect()
}

fn selection() -> Vec<argmax::Entry> {
    vec![
        argmax::Entry::new(argmax::Pattern::PartialF32, "chop_dspark_argmax_f32_v1"),
        argmax::Entry::new(argmax::Pattern::Final, "chop_dspark_argmax_final_v1"),
    ]
}

pub(super) fn feature_entries(g: &Geometry) -> Vec<kernels_sm120::sinkhorn_hc::Entry> {
    kernels_sm120::sinkhorn_hc::entries(
        super::rows::profile(g),
        kernels_sm120::sinkhorn_hc::Geometry {
            mix: model_deepseek41_contract::hc::mix(g),
            capacity: PREFILL_TOKENS,
        },
        false,
    )
    .into_iter()
    .filter(|entry| entry.op == kernels_sm120::sinkhorn_hc::Op::Head)
    .collect()
}

pub fn geometry_cubins(g: &Geometry) -> [String; 4] {
    ["rows", "quantize", "rounding", "features"]
        .map(|kind| format!("dspark_{kind}_{}_sm120", g.kernel_tag))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = vec![(
        FAMILY,
        argmax::cubin_spec("dspark_select_sm120", &selection()),
    )];
    let mut gather = kernels_sm120::comm::cubin_spec();
    gather.name = "dspark_gather_sm120".into();
    gather
        .entries
        .retain(|entry| entry.symbol == "chop_comm_allgather_f32_v1");
    specs.push((FAMILY, gather));
    for g in geometries {
        let [rows, quantize, round, features] = geometry_cubins(g);
        // The drafter's F32 norm is the main head's (`rows::final_norm`).
        let mut spec = CubinSpec {
            name: rows,
            sources_root: kernels_sm120::cuda_source_root(),
            include_dirs: vec![],
            headers: vec!["#include \"row/row.cuh\"".into()],
            defines: vec![],
            entries: vec![],
        };
        for &ranks in SUPPORTED_RANKS {
            if model_deepseek41_contract::admission::check_ranks(g, ranks).is_err() {
                continue;
            }
            let entry = embedding(g, ranks);
            spec.entries.push(EntrySpec::new(
                entry.symbol,
                entry.params,
                entry.body,
                entry.threads,
                1,
            ));
        }
        specs.push((FAMILY, spec));
        specs.push((
            FAMILY,
            block_quantize::cubin_spec(&quantize, &quantizers(g)?).map_err(|e| e.to_string())?,
        ));
        specs.push((FAMILY, rounding::cubin_spec(round, &[pair(g)])));
        specs.push((
            FAMILY,
            kernels_sm120::sinkhorn_hc::cubin_spec(features, &feature_entries(g)),
        ));
    }
    Ok(specs)
}

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    backend = backend
        .with_sharded_embedding(embedding(g, ranks))
        .with_partial_rounding(pair(g));
    let existing = super::dense::quantizers(g, ranks)?
        .into_iter()
        .chain(super::moe::quantizers(g, ranks)?)
        .chain(super::latent_glue::quantizers())
        .map(|entry| entry.query)
        .collect::<Vec<_>>();
    for entry in quantizers(g)? {
        if !existing.contains(&entry.query) {
            backend = backend.with_block_quantize(entry);
        }
    }
    for entry in selection() {
        backend = backend.with_row_argmax(entry);
    }
    Ok(backend.with_sinkhorn(feature_entries(g)))
}
