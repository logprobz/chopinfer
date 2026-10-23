//! The TP4 backbone gate/up bank stored in 64-row by 128-K row tiles
//! (`RoutedLayout::Tiles`, `row-tiles-64x128-groups`). Each entry a served
//! path launches on that bank compiles as its row-tiled build, under the
//! tiled symbol, in place of the row-major entry in its served module, and
//! the routed ops launch it there (`moe_row_tiles_op`, `moe_direct`):
//! - the absorbing gate queue (Decode-32, Verify-32), the direct gate small
//!   and wide queues and the resident8 gate (the other direct buckets);
//! - the streamed module's gate small and wide queues (the other decode and
//!   verify rows up to 64);
//! - the column kernel's 8192-token gate/up (verify above 64 rows);
//! - the native prefill's fused ceil and scanned gate/up tiles.
//!
//! Each reads only that bank (384 experts, 1152 gate/up rows each at TP4),
//! so no other rank count, bank or projection loses an entry. Entries no
//! served path launches on the bank keep their row-major build; a launch of
//! one over the tiled bank is refused.
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::chained::{
    absorb, bandwidth,
    floor::{Entry, Stage},
    streamed, streamed_resident,
};
use kernels_sm120::moe::native_grouped::{self as native, Variant};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use model_deepseek41_contract::weight_plan::RoutedLayout;
use std::collections::BTreeMap;

/// The rank count whose served plan may store the backbone gate/up bank in
/// tiles, and that bank's experts.
pub const RANKS: u32 = 4;
pub const EXPERTS: u32 = 384;

/// Each entry a served path launches on the TP4 backbone gate/up bank, by
/// its row-major symbol, with its row-tiled build, when `layout` stores the
/// bank in tiles; none for rows.
pub fn builds_for(
    g: &Geometry,
    layout: RoutedLayout,
) -> Result<BTreeMap<String, EntrySpec>, String> {
    let mut out = BTreeMap::new();
    if layout != RoutedLayout::Tiles || !super::moe::streamed_qualified(g, RANKS) {
        return Ok(out);
    }
    let mut add = |rows: String, tiled: Result<EntrySpec, kernel_core::contract::Refusal>| {
        let tiled = tiled.map_err(|e| format!("{rows}: {e}"))?;
        match out.insert(rows.clone(), tiled) {
            Some(_) => Err(format!("{rows} has two row-tiled builds")),
            None => Ok(()),
        }
    };
    let direct: Vec<Entry> = super::moe_direct_kernels::entries(g)?
        .into_iter()
        .filter(|e| e.experts() == EXPERTS)
        .collect();
    for e in &direct {
        add(
            absorb::entry(e, Stage::GateSmall).symbol,
            absorb::tiled_entry(e),
        )?;
        for s in [Stage::GateSmall, Stage::GateWide] {
            add(
                bandwidth::entry(e, s, bandwidth::Loads::Direct).symbol,
                bandwidth::tiled_entry(e, s),
            )?;
        }
        add(
            streamed_resident::symbol(&e.gate),
            streamed_resident::tiled_entry(&e.gate),
        )?;
    }
    let backbone = |p: &kernels_sm120::moe::chained::quantized::Projection| {
        p.gate_up() && p.query.projection.gemm.grouped.unwrap().experts == EXPERTS
    };
    for p in super::moe::streamed_projections(g)?
        .iter()
        .filter(|p| backbone(p))
    {
        for stage in streamed::tiled_stages(p) {
            add(streamed::symbol(p, stage), streamed::tiled_entry(p, stage))?;
        }
    }
    for p in super::moe::projections(g, RANKS)?
        .iter()
        .filter(|p| backbone(p) && p.tokens_capacity == PREFILL_TOKENS)
    {
        add(p.symbol.clone(), p.tiled_entry())?;
    }
    if g.id == FLASH.id {
        let shape = super::moe_native_entries::shape(GatedMlpProjection::GateUp);
        let launched = native_symbols();
        let entries: Vec<EntrySpec> = [
            super::moe_native_entries::ceil_cubins(),
            super::moe_native_entries::lean_cubins(),
        ]
        .concat()
        .into_iter()
        .flat_map(|spec| spec.entries)
        .filter(|e| launched.contains(&e.symbol))
        .collect();
        if entries.len() != launched.len() {
            return Err("the served native cubins lack a launched gate/up entry".into());
        }
        for e in entries {
            add(e.symbol.clone(), native::row_tiles_entry(&shape, &e))?;
        }
    }
    Ok(out)
}

/// [`builds_for`] the layout the served plan stores at TP4.
pub fn builds(g: &Geometry) -> Result<BTreeMap<String, EntrySpec>, String> {
    builds_for(g, RoutedLayout::served(RANKS))
}

/// Each row-major symbol of [`builds`] with its tiled build's symbol.
pub fn symbols(g: &Geometry) -> Result<BTreeMap<String, String>, String> {
    Ok(builds(g)?
        .into_iter()
        .map(|(rows, tiled)| (rows, tiled.symbol))
        .collect())
}

/// `specs` with each entry `builds` names replaced in place by its build.
/// Every build replaces exactly one entry, so a renamed or missing served
/// entry fails rather than leaving the row-major build to read tiles.
pub fn serve<F>(
    specs: &mut [(F, CubinSpec)],
    builds: &BTreeMap<String, EntrySpec>,
) -> Result<(), String> {
    let mut replaced = BTreeMap::new();
    for (_, spec) in specs.iter_mut() {
        for entry in &mut spec.entries {
            if let Some(tiled) = builds.get(&entry.symbol) {
                *replaced.entry(entry.symbol.clone()).or_insert(0) += 1;
                *entry = tiled.clone();
            }
        }
    }
    for symbol in builds.keys() {
        if replaced.get(symbol) != Some(&1) {
            return Err(format!(
                "{symbol} is served {} times; its row-tiled build replaces exactly one entry",
                replaced.get(symbol).copied().unwrap_or(0)
            ));
        }
    }
    Ok(())
}

/// Which native gate/up compute entries serving launches: the fused ceil
/// tiles (steps of at most `FUSED_CEIL_ROWS` rows) and the scanned wide
/// tiles above them.
pub fn native_symbols() -> [String; 2] {
    let shape = super::moe_native_entries::shape(GatedMlpProjection::GateUp);
    [
        super::moe_native_entries::ceil()
            .tiles_symbol(&shape, &super::moe_native_entries::GATE_UP_TILE),
        native::tune_symbol(
            &shape,
            &super::moe_native_entries::tile(GatedMlpProjection::GateUp),
            super::moe_native_entries::UNFUSED_LOOP,
            Variant::Serving,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With rows the map is empty; with tiles it names one build per entry
    /// a served path launches on the backbone gate/up bank, each differing
    /// from its row-major entry only in its symbol and the tiles flag of its
    /// configuration, with the same ABI and bounds.
    #[test]
    fn row_tiled_builds_differ_from_the_served_entries_only_in_their_weight_addresses() {
        assert!(builds_for(&FLASH, RoutedLayout::Rows).unwrap().is_empty());
        let builds = builds_for(&FLASH, RoutedLayout::Tiles).unwrap();
        let mut specs = served_specs();
        let rows: BTreeMap<String, EntrySpec> = specs
            .iter()
            .flat_map(|(_, s)| s.entries.iter())
            .filter(|e| builds.contains_key(&e.symbol))
            .map(|e| (e.symbol.clone(), e.clone()))
            .collect();
        assert_eq!(rows.len(), builds.len());
        let kinds = |of: &dyn Fn(&str) -> bool| builds.keys().filter(|s| of(s)).count();
        // Absorbing, direct small and wide, resident8; streamed t8 small,
        // t64 small and wide; the column t8192; native ceil and scanned.
        assert_eq!(
            [
                kinds(&|s| s.contains("_moe_absorb_")),
                kinds(&|s| s.contains("_bandwidth_direct_")),
                kinds(&|s| s.ends_with("_resident8_v1")),
                kinds(&|s| s.ends_with("_stream_small_v1") || s.ends_with("_stream_wide_v1")),
                kinds(&|s| s.contains("_a8ceil_before_gateup_e384k6_n1152_k5120_r64_t8192_")),
                kinds(&|s| s.starts_with("chop_moe_native_gateup_")),
            ],
            [1, 2, 1, 3, 1, 2]
        );
        assert_eq!(builds.len(), 10);
        for (symbol, tiled) in &builds {
            let row = &rows[symbol];
            assert_ne!(&tiled.symbol, symbol);
            assert_eq!(
                (&tiled.params, tiled.max_threads, tiled.min_blocks),
                (&row.params, row.max_threads, row.min_blocks),
                "{symbol}"
            );
            // Each build adds the tiles flag (`,true`) to its row-major
            // body's kernel instantiations, and nothing else.
            let flags = |body: &str| body.matches(",true").count();
            assert!(flags(&tiled.body) > flags(&row.body), "{symbol}");
            assert_eq!(
                tiled.body.replace(",true", ""),
                row.body.replace(",true", ""),
                "{symbol}"
            );
        }
        let before = specs.clone();
        serve(&mut specs, &builds).unwrap();
        // `EntrySpec` has no equality; its debug form states every field.
        let form = |e: &EntrySpec| format!("{e:?}");
        for ((family, spec), (prior, old)) in specs.iter().zip(&before) {
            assert_eq!((family, &spec.name), (prior, &old.name));
            assert_eq!(spec.entries.len(), old.entries.len());
            for (new, old) in spec.entries.iter().zip(&old.entries) {
                let expected = builds.get(&old.symbol).unwrap_or(old);
                assert_eq!(form(new), form(expected));
            }
        }
        // A second pass finds no row-major entry to replace.
        assert!(serve(&mut specs, &builds).is_err());
        let mut extra = builds.clone();
        extra.insert("absent_v1".into(), builds.values().next().unwrap().clone());
        assert!(serve(&mut before.clone(), &extra).is_err());
    }

    /// The native symbols are the gate/up tiles the native wrapper launches.
    #[test]
    fn native_builds_are_the_tiles_native_prefill_launches() {
        let builds = builds_for(&FLASH, RoutedLayout::Tiles).unwrap();
        for symbol in native_symbols() {
            assert!(builds.contains_key(&symbol), "{symbol}");
        }
        assert_eq!(
            builds
                .keys()
                .filter(|s| s.starts_with("chop_moe_native_"))
                .count(),
            2
        );
    }

    /// The bundle's served MoE modules (the build script's specs).
    fn served_specs() -> Vec<(&'static str, CubinSpec)> {
        let mut specs = super::super::moe_direct_kernels::cubins(&[FLASH]).unwrap();
        specs.extend(super::super::moe::cubins(&[FLASH]).unwrap());
        specs.extend(super::super::moe::streamed_cubins(&[FLASH]).unwrap());
        use super::super::moe_native_entries as entries;
        for (family, cubins) in [
            (entries::FAMILY, entries::cubins()),
            (entries::CEIL_FAMILY, entries::ceil_cubins()),
            (entries::WIDE_FAMILY, entries::wide_cubins()),
            (entries::LEAN_FAMILY, entries::lean_cubins()),
        ] {
            specs.extend(cubins.into_iter().map(|s| (family, s)));
        }
        specs
    }
}
