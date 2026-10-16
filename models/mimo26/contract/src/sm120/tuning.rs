//! Geometry-keyed measurements and checked inheritance.

use super::{gemm::RankProjection, geometry::GeometryModel};
use kernel_build::tuning::{Row, Table};
use kernel_core::contract::{DType, DenseKernel, SwigluColumns, Target};
use kernels_sm120::{
    gemm,
    moe::{self, Variant, template},
};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

pub fn moe_tables() -> &'static BTreeMap<String, moe::Tuning> {
    static TABLES: OnceLock<BTreeMap<String, moe::Tuning>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let values: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(include_str!("../../tuning/moe.json"))
                .expect("geometry-keyed MoE tuning");
        values
            .into_iter()
            .map(|(id, value)| {
                (
                    id,
                    moe::Tuning::parse(&value.to_string()).expect("MoE tuning"),
                )
            })
            .collect()
    })
}

fn statuses() -> &'static Mutex<BTreeMap<String, String>> {
    static STATUS: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    STATUS.get_or_init(Default::default)
}

pub fn record(owner: GeometryModel, symbol: String, status: String) {
    if !moe_tables().contains_key(owner.geometry.id) {
        let mut statuses = statuses().lock().unwrap();
        statuses
            .entry(symbol)
            .and_modify(|previous| {
                if !previous.split("; ").any(|label| label == status) {
                    previous.push_str("; ");
                    previous.push_str(&status);
                }
            })
            .or_insert(status);
    }
}

pub fn entry_status() -> BTreeMap<String, String> {
    // Catalog construction records only actual tunable entries.
    super::gemm::catalog();
    super::moe::catalog();
    statuses().lock().unwrap().clone()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleTable {
    schema: String,
    target: kernel_build::tuning::Target,
    rows: Vec<Row>,
}

pub fn measured_schedules() -> Result<Vec<(String, Table)>, String> {
    let values: BTreeMap<String, ScheduleTable> =
        serde_json::from_str(include_str!("../../tuning/schedules.json"))
            .map_err(|e| e.to_string())?;
    values
        .into_iter()
        .map(|(id, value)| {
            if value.schema != kernel_build::tuning::SCHEMA
                || value.rows.windows(2).any(|rows| rows[0].key >= rows[1].key)
            {
                return Err(format!(
                    "geometry {id}: schedule schema or key order is invalid"
                ));
            }
            let mut table = Table::new(value.target);
            for row in value.rows {
                table
                    .upsert(row)
                    .map_err(|e| format!("geometry {id}: {e}"))?;
            }
            template::parse_table(&table.render())
                .map(|table| (id, table))
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn schedules(variants: &[Variant]) -> Result<Table, String> {
    schedules_with_tables(variants, &measured_schedules()?)
}

fn schedules_with_tables(
    variants: &[Variant],
    tables: &[(String, Table)],
) -> Result<Table, String> {
    template::resolve_table(
        variants,
        tables,
        |variant| super::geometry::owner(variant.profile).map(|owner| owner.geometry.id),
        |variant, status| {
            if let Some(owner) = super::geometry::owner(variant.profile) {
                record(*owner, variant.symbol.clone(), status);
            }
        },
    )
}

/// Keep a single decode reduction across every exact-verify bucket.
pub fn dense(
    owner: GeometryModel,
    ranks: u32,
    projection: &mut RankProjection,
) -> Result<(), String> {
    let (name, n, k, weight, swiglu, segments, decode) = *projection;
    let validate = |m, kernel| {
        let mut query = gemm::bf16_query(
            m,
            n,
            k,
            if name == "head_f32" {
                DType::F32
            } else {
                DType::Bf16
            },
        );
        query.weight = weight;
        if swiglu {
            query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        }
        gemm::instantiate(gemm::DenseRequest {
            query,
            kernel,
            target: Target::Sm120f,
            symbol: format!("chop_gemm_{}_{}_probe_v1", owner.profile(ranks).tag(), name),
            knobs: None,
        })
        .map(|_| ())
    };
    let buckets = [1, 2, 4, 8, 16, 32];
    let selected = [decode, gemm::WHOLE_K_DECODE].into_iter().find(|kernel| buckets.into_iter().all(|m| validate(m, *kernel).is_ok())).ok_or_else(|| format!("geometry {} TP{ranks} role {name} N={n} K={k}: no legal dense decode for buckets {buckets:?}", owner.geometry.id))?;
    // K segments use whole 128-column blocks; defaults retain one segment.
    let segments = [segments, 1]
        .into_iter()
        .find(|segments| {
            k.is_multiple_of(u32::from(*segments) * 128)
                && validate(
                    8192,
                    DenseKernel::FragmentPrefill {
                        segments: *segments,
                    },
                )
                .is_ok()
        })
        .ok_or_else(|| {
            format!(
                "geometry {} TP{ranks} role {name} N={n} K={k}: no legal dense prefill",
                owner.geometry.id
            )
        })?;
    let tag = owner.profile(ranks).tag();
    let source = crate::geometry::V26_FLASH.id;
    for bucket in buckets {
        record(
            owner,
            format!("chop_gemm_{tag}_{name}_m{bucket}_v1"),
            if selected == decode {
                format!("inherited {source} dense row TP{ranks}/{name}")
            } else {
                "engine default whole-K decode".into()
            },
        );
    }
    record(
        owner,
        format!("chop_gemm_{tag}_{name}_prefill_v1"),
        if segments == projection.5 {
            format!("inherited {source} dense row TP{ranks}/{name}")
        } else {
            "engine default single-segment prefill".into()
        },
    );
    projection.5 = segments;
    projection.6 = selected;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_measured_schedule_refusal_does_not_inherit() {
        let variants = super::super::moe::catalog_for(super::super::geometry::MODELS[0]).unwrap();
        let variant = variants.iter().find(|v| template::covers(v)).unwrap();
        let key = template::key(variant);
        let mut tables = measured_schedules().unwrap();
        let table = &mut tables
            .iter_mut()
            .find(|(id, _)| id == "v26-flash")
            .unwrap()
            .1;
        let mut row = table.require(&key).unwrap().clone();
        row.knobs
            .insert("warps".into(), kernel_build::tuning::Knob::Count(3));
        table.upsert(row).unwrap();
        let error = schedules_with_tables(&variants, &tables).unwrap_err();
        assert!(
            error.contains("geometry v26-flash measured schedule"),
            "{error}"
        );
        assert!(error.contains(&variant.symbol), "{error}");
    }
}
