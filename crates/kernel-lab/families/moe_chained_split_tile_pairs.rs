//! Lab-only tile-pair projections of the split
//! (`chained::Schedule::TilePairs`, all 288 experts the rank's,
//! intermediate 512): the routed FP8 prefill gate/up and down at each swept
//! row-pair threshold the split does not serve, the deliberate defects of
//! `chop::moe::tile_pairs::Mutation`, and the captured served routes the
//! comparisons and timings read. Shared with `tests/moe_chained.rs`
//! (bitwise sweep, producer and mutant checks) and
//! `tests/projection_latency.rs` (prefill timings).

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Schedule, Weights, split};

pub const FAMILY: &str = "moe_chained_split_tile_pairs";

/// The candidates' cubin prefix: `{PREFIX}_fp8block_tp`.
pub const PREFIX: &str = "chop_moe_chain_lab_split";

/// The mutants' cubin.
pub const MUTANTS_CUBIN: &str = "chop_moe_chain_lab_split_tp_mutants";

/// `chop::moe::tile_pairs::Mutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 5] = ["none", "drop_wait", "late_wait", "odd_tail", "long_chain"];

/// The swept row-pair thresholds: experts with at least this many 64-row
/// tiles pair them under one weight tile; 0 pairs no rows (every CTA a
/// column pair). The split serves 2 for its gate/up and 0 for its down
/// (`split::GATE_UP_MIN_PAIR_TILES`, `split::DOWN_MIN_PAIR_TILES`).
pub const MIN_PAIR_TILES: [u32; 3] = [2, 4, 0];

/// The row-pair threshold of the mutants.
pub const MUTANT_MIN_PAIR_TILES: u32 = 2;

/// The split's routed FP8 prefill (64-row) column projections: gate/up,
/// then down.
pub fn references() -> [Projection; 2] {
    [true, false].map(|gate_up| {
        split::catalog()
            .into_iter()
            .find(|p| p.row_tile == 64 && p.weights() == Weights::Fp8Block128 && p.gate_up() == gate_up)
            .expect("the split's routed FP8 prefill projections are cataloged")
    })
}

/// The row-pair threshold of a lab projection.
pub fn min_pair_tiles(projection: &Projection) -> u32 {
    match projection.schedule {
        Schedule::TilePairs { min_pair_tiles } => min_pair_tiles,
        _ => panic!("{} is not a tile-pair projection", projection.symbol),
    }
}

/// [`references`] at each threshold of [`MIN_PAIR_TILES`], as gate/up and
/// down pairs.
pub fn candidate_pairs() -> Vec<[Projection; 2]> {
    let references = references();
    MIN_PAIR_TILES
        .iter()
        .map(|&min| {
            references.clone().map(|reference| {
                reference
                    .tile_pairs(min)
                    .expect("tile pairs take the split's routed prefill projections")
            })
        })
        .collect()
}

/// Every candidate projection.
pub fn projections() -> Vec<Projection> {
    candidate_pairs().into_iter().flatten().collect()
}

/// The candidates the split does not serve, which the lab builds: the
/// served ones (the gate/up at 2 and the down at 0) are the split family's
/// (`moe_chained_split`), whose symbols they share.
pub fn lab_projections() -> Vec<Projection> {
    let served = split::forms();
    projections()
        .into_iter()
        .filter(|p| served.iter().all(|s| s.symbol != p.symbol))
        .collect()
}

/// The mutated projections: [`references`] at [`MUTANT_MIN_PAIR_TILES`].
pub fn mutant_projections() -> [Projection; 2] {
    references().map(|reference| {
        reference
            .tile_pairs(MUTANT_MIN_PAIR_TILES)
            .expect("tile pairs take the split's routed prefill projections")
    })
}

/// The symbol of a mutant projection's instance with `mutation`.
pub fn mutant_symbol(mutation: &str, gate_up: bool) -> String {
    let projection = if gate_up { "gateup" } else { "down" };
    format!("chop_lab_split_tp_{mutation}_{projection}_v1")
}

/// The lab candidates' cubin ([`lab_projections`]).
pub fn candidates_cubin() -> CubinSpec {
    let mut specs = chained::projection_cubin_specs(PREFIX, &lab_projections());
    assert_eq!(specs.len(), 1, "one tile-pair cubin");
    specs.remove(0)
}

/// The mutants' cubin: the mutant projections' entries once per mutation,
/// each body given its mutation as the kernel's last template argument.
pub fn mutants_cubin() -> CubinSpec {
    let mut cubin = chained::projection_cubin_specs(MUTANTS_CUBIN, &mutant_projections())
        .pop()
        .expect("one tile-pair cubin");
    cubin.name = MUTANTS_CUBIN.into();
    let base = std::mem::take(&mut cubin.entries);
    for mutation in MUTATIONS {
        for (entry, gate_up) in base.iter().zip([true, false]) {
            let mut entry = entry.clone();
            assert_eq!(
                entry.body.matches(">(input").count(),
                1,
                "{}",
                entry.symbol
            );
            entry.symbol = mutant_symbol(mutation, gate_up);
            entry.body = entry.body.replace(
                ">(input",
                &format!(",chop::moe::tile_pairs::Mutation::{mutation}>(input"),
            );
            cubin.entries.push(entry);
        }
    }
    cubin
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &["chop_moe_chain_lab_split_fp8block_tp", MUTANTS_CUBIN],
        vec![candidates_cubin(), mutants_cubin()],
    )
}

/// The directory of a captured served prefill's routes, read by the lab's
/// real-routing cases: `L<NN>.router.ids.r0.bin` per MoE layer, u32
/// `[rows, top_k]` global expert ids in the router's order.
pub const ROUTE_CAPTURE_ENV: &str = "CHOPINFER_MOE_ROUTE_CAPTURE";

/// One captured layer's routes.
pub struct CapturedLayer {
    pub name: String,
    /// `[rows * top_k]` global expert ids.
    pub ids: Vec<i32>,
}

/// The captured layers of [`ROUTE_CAPTURE_ENV`] in name order, or `None`
/// when it is unset; an unreadable, empty or ragged capture is an error.
pub fn captured_layers() -> Result<Option<Vec<CapturedLayer>>, String> {
    let Some(dir) = std::env::var_os(ROUTE_CAPTURE_ENV) else {
        return Ok(None);
    };
    let top_k = split::SHAPES.top_k as usize;
    let experts = split::SHAPES.experts;
    let mut layers = Vec::new();
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{dir:?}: {e}"))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".router.ids.r0.bin"))
        else {
            continue;
        };
        let bytes = std::fs::read(&path).map_err(|e| format!("{path:?}: {e}"))?;
        if bytes.is_empty() || !bytes.len().is_multiple_of(4 * top_k) {
            return Err(format!("{path:?}: not whole rows of {top_k} u32 ids"));
        }
        let ids = bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect::<Vec<_>>();
        if ids.iter().any(|&id| id >= experts) {
            return Err(format!("{path:?}: an id is not below {experts}"));
        }
        layers.push(CapturedLayer {
            name: name.to_owned(),
            ids: ids.into_iter().map(|id| id as i32).collect(),
        });
    }
    if layers.is_empty() {
        return Err(format!("{dir:?} holds no router id capture"));
    }
    layers.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Some(layers))
}

/// `tokens` captured token rows from layer `first` on: its rows, then the
/// next layers' (wrapping), for counts above one capture's prompt.
pub fn adjacent_rows(layers: &[CapturedLayer], first: usize, tokens: usize) -> Vec<i32> {
    let top_k = split::SHAPES.top_k as usize;
    let mut ids = Vec::with_capacity(tokens * top_k);
    let mut layer = first;
    while ids.len() < tokens * top_k {
        let take = (tokens * top_k - ids.len()).min(layers[layer % layers.len()].ids.len());
        ids.extend_from_slice(&layers[layer % layers.len()].ids[..take]);
        layer += 1;
    }
    ids
}

/// `tokens` token rows of layer `layer` alone: its capture repeated, so a
/// longer prompt keeps that layer's expert skew.
pub fn repeated_rows(layers: &[CapturedLayer], layer: usize, tokens: usize) -> Vec<i32> {
    let top_k = split::SHAPES.top_k as usize;
    let ids = &layers[layer].ids;
    ids.iter().copied().cycle().take(tokens * top_k).collect()
}
