//! Kernel schedule tuning tables keyed by geometry, with checked
//! inheritance from the nearest tuned geometry.
//!
//! A geometry's table is `tuning/<id>/schedules.json` in the shared
//! `chopinfer-tuning-v1` form (`kernel_build::tuning`): rows keyed by
//! operation, weight format, per-rank shape and token bucket. Flash's rows
//! are hand-picked defaults until an offline sweep replaces them under
//! `kernel_build::tuning::accepts`. A geometry without a table tries the
//! tuned geometries in increasing distance in hidden width, expert count,
//! expert intermediate width and top-k, taking the first row for the same
//! operation and bucket whose knobs are legal at its own shape; otherwise
//! it uses the model default. Every choice for an untuned geometry carries
//! a status naming its source, and needs its own numerical and performance
//! evidence before qualification.
//!
//! Only the grouped MoE engine has a schedule choice today, its row tile.
//! The other engines derive their schedules from the bucket.
use crate::capacity::PREFILL_TOKENS;
use crate::geometry::{GEOMETRIES, Geometry};
use crate::kernels::KernelProfile;
use kernel_build::tuning::{Key, Knob, Knobs, Table};
use std::collections::BTreeMap;
use std::sync::OnceLock;

include!(concat!(env!("OUT_DIR"), "/tuning.rs"));

/// Weight format of the FP8 checkpoint's routed experts.
pub const MOE_WEIGHTS: &str = "fp8_block128";

/// Row tiles the grouped MoE engine compiles.
pub const MOE_ROW_TILES: [u32; 3] = [16, 32, 64];

/// Tokens of the grouped MoE decode bucket (its single-launch route).
pub const MOE_DECODE_TOKENS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MoeProjection {
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MoePhase {
    Decode,
    Prefill,
}

impl MoePhase {
    pub const fn bucket(self) -> u32 {
        match self {
            Self::Decode => MOE_DECODE_TOKENS,
            Self::Prefill => PREFILL_TOKENS,
        }
    }
}

/// The schedule chosen for one key, and for an untuned geometry where it
/// came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub schedule: String,
    pub knobs: Knobs,
    /// `None` for a geometry's own row.
    pub status: Option<String>,
}

fn tables() -> &'static BTreeMap<&'static str, Table> {
    static TABLES_BY_ID: OnceLock<BTreeMap<&'static str, Table>> = OnceLock::new();
    TABLES_BY_ID.get_or_init(|| {
        TABLES
            .iter()
            .map(|&(id, text)| {
                let table = Table::parse(text)
                    .unwrap_or_else(|e| panic!("tuning/{id}/schedules.json: {e}"));
                (id, table)
            })
            .collect()
    })
}

/// The geometry's own table, if it is tuned.
pub fn table(g: &Geometry) -> Option<&'static Table> {
    tables().get(g.id)
}

/// The distance `nearest` orders tuned geometries by: hidden width, expert
/// count, expert intermediate width and top-k.
fn distance(g: &Geometry, c: &Geometry) -> u64 {
    let (a, b) = (&g.shape, &c.shape);
    u64::from(a.hidden.abs_diff(b.hidden))
        + u64::from(a.moe.experts.abs_diff(b.moe.experts))
        + u64::from(a.moe.intermediate.abs_diff(b.moe.intermediate))
        + u64::from(a.moe.top_k.abs_diff(b.moe.top_k))
}

/// Tuned geometries, the geometry itself first when tuned, then by
/// distance and id.
pub fn nearest(g: &Geometry) -> Vec<&'static Geometry> {
    kernel_build::tuning::nearest_first(
        Some(g.id),
        GEOMETRIES.iter().filter(|c| table(c).is_some()),
        |c| c.id,
        |c| distance(g, c),
    )
}

/// Where an untuned geometry's decision may come from: a tuned geometry
/// with its table, by ID, or (`None`) the model default.
type Source = Option<(&'static Geometry, &'static Table)>;

/// Every tuned geometry as a source, by ID.
fn sources() -> &'static BTreeMap<String, Source> {
    static SOURCES: OnceLock<BTreeMap<String, Source>> = OnceLock::new();
    SOURCES.get_or_init(|| {
        GEOMETRIES
            .iter()
            .filter_map(|g| Some((g.id.to_owned(), Some((g, table(g)?)))))
            .collect()
    })
}

fn op(projection: MoeProjection, phase: MoePhase) -> String {
    let projection = match projection {
        MoeProjection::GateUp => "gate_up",
        MoeProjection::Down => "down",
    };
    let phase = match phase {
        MoePhase::Decode => "decode",
        MoePhase::Prefill => "prefill",
    };
    format!("grouped_moe.{projection}.{phase}")
}

/// The table key of a grouped MoE projection at `ranks`.
pub fn moe_key(g: &Geometry, ranks: u32, projection: MoeProjection, phase: MoePhase) -> Key {
    let profile = KernelProfile::new(g, ranks);
    Key {
        op: op(projection, phase),
        weights: MOE_WEIGHTS.into(),
        shape: BTreeMap::from([
            ("experts".into(), profile.local_experts),
            ("hidden".into(), profile.hidden),
            ("intermediate".into(), profile.expert_intermediate),
            ("top_k".into(), profile.top_k),
        ]),
        bucket: phase.bucket(),
    }
}

/// Whether a grouped MoE row runs at `key`: the chained schedule with one
/// compiled row tile no wider than the bucket.
fn moe_legal(key: &Key, schedule: &str, knobs: &Knobs) -> bool {
    schedule == "chained"
        && knobs.len() == 1
        && knobs
            .get("row_tile")
            .and_then(|knob| knob.count())
            .is_some_and(|tile| MOE_ROW_TILES.contains(&tile) && tile <= key.bucket)
}

/// The model default: the decode bucket's smallest tile, prefill's widest.
fn moe_default(phase: MoePhase) -> Knobs {
    let tile = match phase {
        MoePhase::Decode => MOE_ROW_TILES[0],
        MoePhase::Prefill => MOE_ROW_TILES[2],
    };
    Knobs::from([("row_tile".into(), Knob::Count(tile))])
}

/// The grouped MoE schedule of `g` at `ranks`: its own row when tuned
/// (refused if illegal), else the nearest tuned geometry's legal row for the
/// same operation and bucket, else the model default
/// (`kernel_build::tuning::build_nearest` over `sources`, one decision per
/// operation, so an untuned geometry's operations may come from different
/// sources).
pub fn moe(
    g: &Geometry,
    ranks: u32,
    projection: MoeProjection,
    phase: MoePhase,
) -> Result<Decision, String> {
    let key = moe_key(g, ranks, projection, phase);
    if let Some(own) = table(g) {
        let row = own.require(&key).map_err(|e| format!("{}: {e}", g.id))?;
        if !moe_legal(&key, &row.schedule, &row.knobs) {
            return Err(format!("{}: illegal tuning row for {key}", g.id));
        }
        return Ok(Decision {
            schedule: row.schedule.clone(),
            knobs: row.knobs.clone(),
            status: None,
        });
    }
    let (decision, _) = kernel_build::tuning::build_nearest(
        g.id,
        "grouped MoE",
        sources(),
        |id| sources()[id].map_or(u64::MAX, |(c, _)| distance(g, c)),
        || None,
        |source: &Source| match source {
            Some((candidate, table)) => {
                let source = moe_key(candidate, ranks, projection, phase);
                match table.get(&source) {
                    Some(row) if moe_legal(&key, &row.schedule, &row.knobs) => Ok(Decision {
                        schedule: row.schedule.clone(),
                        knobs: row.knobs.clone(),
                        status: Some(format!("untuned: inherited {} row {source}", candidate.id)),
                    }),
                    _ => Err(format!("no legal row for {key}")),
                }
            }
            None => Ok(Decision {
                schedule: "chained".into(),
                knobs: moe_default(phase),
                status: Some(format!("untuned: model default for {key}")),
            }),
        },
    )?;
    Ok(decision)
}

/// Every grouped MoE decision of `g` at `ranks`, by operation key.
pub fn moe_decisions(g: &Geometry, ranks: u32) -> Result<BTreeMap<String, Decision>, String> {
    let mut decisions = BTreeMap::new();
    for projection in [MoeProjection::GateUp, MoeProjection::Down] {
        for phase in [MoePhase::Decode, MoePhase::Prefill] {
            decisions.insert(op(projection, phase), moe(g, ranks, projection, phase)?);
        }
    }
    Ok(decisions)
}

/// The labels of every untuned choice of every geometry at its served
/// ranks, as `<geometry>/tp<ranks>/<op>: <status>`; a tuned geometry has
/// none.
pub fn entry_status() -> Result<BTreeMap<String, String>, String> {
    let mut labels = BTreeMap::new();
    for g in GEOMETRIES {
        for &ranks in crate::kernels::SERVED_RANKS {
            for (op, decision) in moe_decisions(g, ranks)? {
                if let Some(status) = decision.status {
                    labels.insert(format!("{}/tp{ranks}/{op}", g.id), status);
                }
            }
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;

    #[test]
    fn flash_has_a_canonical_table_covering_its_moe_projections() {
        let own = table(&FLASH).expect("flash is tuned");
        assert_eq!(own.render(), TABLES[0].1);
        assert_eq!(own.rows().len(), 4);
        let decisions = moe_decisions(&FLASH, 4).unwrap();
        let tiles = decisions
            .iter()
            .map(|(op, d)| {
                assert_eq!((d.schedule.as_str(), &d.status), ("chained", &None));
                (op.as_str(), d.knobs["row_tile"].count().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tiles,
            [
                ("grouped_moe.down.decode", 16),
                ("grouped_moe.down.prefill", 64),
                ("grouped_moe.gate_up.decode", 16),
                ("grouped_moe.gate_up.prefill", 64),
            ]
        );
    }

    #[test]
    fn every_tuned_table_covers_exactly_its_own_keys() {
        for g in GEOMETRIES.iter().filter(|g| table(g).is_some()) {
            for &ranks in crate::kernels::SERVED_RANKS {
                let decisions = moe_decisions(g, ranks).unwrap();
                assert!(decisions.values().all(|d| d.status.is_none()));
                assert_eq!(table(g).unwrap().rows().len(), decisions.len());
            }
        }
    }

    #[test]
    fn tuned_geometries_label_nothing_and_untuned_ones_name_their_source() {
        let labels = entry_status().unwrap();
        for g in GEOMETRIES {
            let own = labels
                .iter()
                .filter(|(name, _)| name.starts_with(&format!("{}/", g.id)))
                .collect::<Vec<_>>();
            if table(g).is_some() {
                assert!(own.is_empty(), "{own:?}");
            } else {
                assert_eq!(own.len(), 4 * crate::kernels::SERVED_RANKS.len());
                assert!(
                    own.iter()
                        .all(|(_, status)| status.starts_with("untuned: "))
                );
            }
        }
    }

    #[test]
    fn an_untuned_geometry_inherits_the_nearest_legal_row_or_the_default() {
        let mut other = FLASH;
        other.id = "other";
        other.shape.moe.experts = 128;
        assert_eq!(nearest(&other)[0].id, "flash");
        let decision = moe(&other, 4, MoeProjection::Down, MoePhase::Decode).unwrap();
        assert_eq!(decision.knobs["row_tile"], Knob::Count(16));
        assert_eq!(
            decision.status.unwrap(),
            "untuned: inherited flash row grouped_moe.down.decode fp8_block128 experts=288,hidden=4096,intermediate=512,top_k=8 t32"
        );
        // A row tile wider than the bucket is not inherited.
        let key = moe_key(&other, 4, MoeProjection::Down, MoePhase::Decode);
        let wide = Knobs::from([("row_tile".into(), Knob::Count(64))]);
        assert!(!moe_legal(&key, "chained", &wide));
        assert!(!moe_legal(&key, "ring", &moe_default(MoePhase::Decode)));
        assert!(moe_legal(&key, "chained", &moe_default(MoePhase::Decode)));
    }
}
