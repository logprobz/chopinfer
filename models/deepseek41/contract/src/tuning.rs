//! Flash schedule choices. Missing optional rows use the model's baseline.
//! Required descriptor rows are checked at load; other geometries inherit no rows.
use crate::geometry::{FLASH, Geometry};
use kernel_build::tuning::{Row, Table, TuningError};
use std::collections::BTreeMap;
use std::sync::OnceLock;

// Family, its names and the embedded texts come from one build-time list.
include!(concat!(env!("OUT_DIR"), "/tuning.rs"));

mod descriptors;
mod sites;

const SHAPE_KEYS: [&str; 8] = [
    "n", "k", "groups", "paired", "ranks", "experts", "top_k", "width",
];

fn parse_family(family: Family, text: &str) -> Result<Table, TuningError> {
    let table = Table::parse(text)?;
    for row in table.rows() {
        let mut parts = row.key.op.split('.');
        if parts.next() != Some(family.name())
            || parts.next().is_none_or(str::is_empty)
            || parts.next().is_none_or(str::is_empty)
            || parts.next().is_some()
        {
            return Err(TuningError(format!(
                "{}: expected {}.<site>.<phase>",
                row.key,
                family.name()
            )));
        }
        if row
            .key
            .shape
            .keys()
            .any(|name| !SHAPE_KEYS.contains(&name.as_str()))
        {
            return Err(TuningError(format!(
                "{}: unsupported tuning shape key",
                row.key
            )));
        }
        if matches!(
            row.key.op.as_str(),
            "latent.staging.decode" | "latent.staging.verify"
        ) && !matches!(row.key.rows, Some([lo, hi]) if lo == hi)
        {
            return Err(TuningError(format!(
                "{}: staging requires a singleton launch row",
                row.key
            )));
        }
        sites::validate(row)?;
    }
    if family == Family::DenseFold {
        descriptors::validate(&table)?;
    }
    Ok(table)
}

fn tables() -> &'static BTreeMap<Family, Table> {
    static PARSED: OnceLock<BTreeMap<Family, Table>> = OnceLock::new();
    PARSED.get_or_init(|| {
        TABLES
            .iter()
            .map(|&(family, text)| {
                let table = parse_family(family, text)
                    .unwrap_or_else(|e| panic!("tuning/flash/{}.json: {e}", family.name()));
                (family, table)
            })
            .collect()
    })
}

fn table_from<'a>(
    g: &Geometry,
    family: Family,
    tables: &'a BTreeMap<Family, Table>,
) -> Option<&'a Table> {
    if g.id != FLASH.id {
        return None;
    }
    tables.get(&family)
}

/// Flash's own family table, with no inheritance for another geometry.
pub fn table(g: &Geometry, family: Family) -> Option<&'static Table> {
    table_from(g, family, tables())
}

fn select_from<'a>(
    table: &'a Table,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
    rows: u32,
) -> Option<&'a Row> {
    if shape.len() > SHAPE_KEYS.len() {
        return None;
    }
    let mut sorted = [("", 0); SHAPE_KEYS.len()];
    let sorted = &mut sorted[..shape.len()];
    sorted.copy_from_slice(shape);
    sorted.sort_unstable_by_key(|&(name, _)| name);
    table.select_sorted(op, weights, sorted, bucket, rows)
}

/// The schedule for this exact operation, format, shape and bucket at
/// `rows` launch rows. Missing optional admissions use the baseline; required
/// descriptor rows have already been checked when loading the table.
/// Shape order is immaterial; duplicate names return `None`. Lookup allocates
/// nothing once the embedded tables have been initialized.
pub fn select(
    g: &Geometry,
    family: Family,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
    rows: u32,
) -> Option<&'static Row> {
    select_from(table(g, family)?, op, weights, shape, bucket, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_build::tuning::{Key, Target};

    fn fixture(family: Family) -> Table {
        let mut table = Table::new(Target {
            compute_capability: 120,
            sm_count: 188,
        });
        table
            .upsert(Row {
                key: Key {
                    op: format!("{}.site.decode", family.name()),
                    weights: "f32".into(),
                    shape: [("n".into(), 32), ("k".into(), 16), ("ranks".into(), 4)].into(),
                    bucket: 8,
                    rows: Some([2, 5]),
                },
                schedule: "measured".into(),
                knobs: BTreeMap::new(),
            })
            .unwrap();
        Table::parse(&table.render()).unwrap()
    }

    #[test]
    fn presence_only_sites_refuse_unknown_schedules_and_knobs() {
        let source = TABLES
            .iter()
            .find(|(f, _)| *f == Family::Indexer)
            .unwrap()
            .1;
        let full = Table::parse(source).unwrap();
        let original = full
            .rows()
            .iter()
            .find(|r| r.key.op == "indexer.cached.prefill")
            .unwrap();
        for mode in 0..3 {
            let mut row = original.clone();
            match mode {
                0 => row.schedule = "unknown".into(),
                1 => {
                    row.knobs
                        .insert("ignored".into(), kernel_build::tuning::Knob::Count(1));
                }
                _ => row.key.op = "indexer.unknown.prefill".into(),
            }
            let mut table = Table::new(full.target);
            table.upsert(row.clone()).unwrap();
            let error = parse_family(Family::Indexer, &table.render()).unwrap_err();
            assert!(error.0.contains(&row.key.op), "{error}");
        }
    }

    #[test]
    fn missing_required_descriptor_names_its_row() {
        let source = TABLES
            .iter()
            .find(|(f, _)| *f == Family::DenseFold)
            .unwrap()
            .1;
        let full = Table::parse(source).unwrap();
        for narrow in [false, true] {
            let mut table = Table::new(full.target);
            for mut row in full.rows().iter().cloned() {
                if row.key.op == "dense_fold.status_wide.verify" {
                    if !narrow {
                        continue;
                    }
                    row.key.rows = Some([32, 63]);
                }
                table.upsert(row).unwrap();
            }
            let error = parse_family(Family::DenseFold, &table.render()).unwrap_err();
            assert!(error.0.contains("dense_fold.status_wide.verify"), "{error}");
            assert!(error.0.contains("row 64"), "{error}");
        }
    }

    #[test]
    fn every_site_checks_schedule_and_knob_schema() {
        let mut sites_by_name = BTreeMap::new();
        for &(_, text) in TABLES {
            for row in Table::parse(text).unwrap().rows() {
                sites_by_name.insert(row.key.op.clone(), row.clone());
            }
        }
        assert!(sites_by_name.len() > 100);
        for original in sites_by_name.values() {
            sites::validate(original).unwrap();
            let mut row = original.clone();
            row.schedule = "unsupported".into();
            assert!(sites::validate(&row).is_err(), "{}", row.key);
            let mut row = original.clone();
            row.knobs
                .insert("unused".into(), kernel_build::tuning::Knob::Count(1));
            assert!(sites::validate(&row).is_err(), "{}", row.key);
            for (name, value) in &original.knobs {
                let mut row = original.clone();
                row.knobs.remove(name);
                assert!(sites::validate(&row).is_err(), "{} missing {name}", row.key);
                row.knobs.insert(
                    name.clone(),
                    match value {
                        kernel_build::tuning::Knob::Count(_) => {
                            kernel_build::tuning::Knob::Flag(false)
                        }
                        kernel_build::tuning::Knob::Flag(_) => kernel_build::tuning::Knob::Count(0),
                    },
                );
                assert!(
                    sites::validate(&row).is_err(),
                    "{} mistyped {name}",
                    row.key
                );
            }
        }
    }

    #[test]
    fn prefill_norm_strategies_keep_both_implemented_choices() {
        let full =
            Table::parse(TABLES.iter().find(|(f, _)| *f == Family::Norm).unwrap().1).unwrap();
        for (op, schedules) in [
            ("norm.input_strategy.prefill", ["cached", "recompute"]),
            ("norm.query_strategy.prefill", ["cached", "reread"]),
        ] {
            let mut row = full.rows().iter().find(|r| r.key.op == op).unwrap().clone();
            for schedule in schedules {
                row.schedule = schedule.into();
                sites::validate(&row).unwrap();
            }
        }
    }

    #[test]
    fn site_knobs_refuse_unimplemented_catalog_values() {
        use kernel_build::tuning::Knob::{Count, Flag};
        for (family, op, knob, value) in [
            (
                Family::Latent,
                "latent.staging.decode",
                "stage_keys",
                Count(32),
            ),
            (
                Family::Latent,
                "latent.staging.verify",
                "head_group",
                Count(1),
            ),
            (
                Family::Latent,
                "latent.cache.decode",
                "head_group",
                Count(8),
            ),
            (
                Family::DensePrefill,
                "dense_prefill.matrix.prefill",
                "stages",
                Count(4),
            ),
            (
                Family::DensePrefill,
                "dense_prefill.compact.prefill",
                "matrix_fetch",
                Flag(true),
            ),
            (
                Family::DenseFold,
                "dense_fold.split.decode",
                "finish",
                Count(0),
            ),
            (Family::F32, "f32.blocked.prefill", "tile", Count(64)),
            (Family::Hc, "hc.expand_mix.prefill", "chains", Count(16)),
            (Family::Norm, "norm.query_cached.decode", "warps", Count(4)),
        ] {
            let full = Table::parse(TABLES.iter().find(|(f, _)| *f == family).unwrap().1).unwrap();
            let mut row = full.rows().iter().find(|r| r.key.op == op).unwrap().clone();
            row.knobs.insert(knob.into(), value);
            let error = sites::validate(&row).unwrap_err();
            assert!(error.0.contains(op), "{error}");
        }
    }

    #[test]
    fn staging_ranges_are_refused_with_the_row_name() {
        let text = TABLES.iter().find(|(f, _)| *f == Family::Latent).unwrap().1;
        let widened = text.replace(
            "\"rows\": [8, 8], \"schedule\": \"staged\"",
            "\"rows\": [8, 9], \"schedule\": \"staged\"",
        );
        assert_ne!(text, widened);
        let error = parse_family(Family::Latent, &widened).unwrap_err();
        assert!(error.0.contains("latent.staging.decode"), "{error}");
        assert!(error.0.contains("8..=9"), "{error}");
    }

    #[test]
    fn every_family_has_one_canonical_flash_table() {
        use std::{collections::BTreeSet, fs, path::Path};
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tuning/flash");
        let files = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<BTreeSet<_>>();
        let names = Family::ALL
            .iter()
            .map(|family| format!("{}.json", family.name()).into())
            .collect::<BTreeSet<_>>();
        assert_eq!(files, names);
        assert_eq!(files.len(), Family::ALL.len());
        assert_eq!(TABLES.len(), Family::ALL.len());
        assert_eq!(tables().len(), Family::ALL.len());
        for family in Family::ALL {
            let table = table(&FLASH, family).unwrap();
            let text = TABLES.iter().find(|(f, _)| *f == family).unwrap().1;
            assert_eq!(
                fs::read_to_string(directory.join(format!("{}.json", family.name()))).unwrap(),
                text
            );
            assert_eq!(table.render(), text);
            assert_eq!(table.target.compute_capability, 120);
            assert_eq!(table.target.sm_count, 188);
        }
    }

    #[test]
    fn family_names_and_shape_keys_are_checked() {
        for family in Family::ALL {
            let valid = fixture(family).render();
            let op = format!("{}.site.decode", family.name());
            for invalid in [
                "wrong.site.decode".to_string(),
                format!("{}.site", family.name()),
                format!("{}..decode", family.name()),
                format!("{}.site.decode.extra", family.name()),
            ] {
                let error = parse_family(family, &valid.replace(&op, &invalid)).unwrap_err();
                assert!(error.0.contains("expected"), "{error}");
            }
            assert!(
                parse_family(family, &valid.replace("\"ranks\"", "\"runtime_flag\""))
                    .unwrap_err()
                    .0
                    .contains("unsupported tuning shape key")
            );
        }
    }

    #[test]
    fn other_geometries_do_not_inherit_flash_tables() {
        let untuned = Geometry {
            id: "untuned",
            ..FLASH
        };
        let fixtures = Family::ALL
            .into_iter()
            .map(|family| (family, fixture(family)))
            .collect();
        for family in Family::ALL {
            let own = table_from(&FLASH, family, &fixtures).unwrap();
            let op = format!("{}.site.decode", family.name());
            assert!(
                select_from(own, &op, "f32", &[("ranks", 4), ("n", 32), ("k", 16)], 8, 2).is_some()
            );
            assert!(table_from(&untuned, family, &fixtures).is_none());
            assert!(table(&untuned, family).is_none());
        }
    }

    #[test]
    fn missing_rows_use_the_baseline() {
        for family in Family::ALL {
            let op = format!("{}.site.decode", family.name());
            let shape = [("ranks", 4), ("n", 32), ("k", 16)];
            let fixture = fixture(family);
            for rows in 0..=9 {
                let chosen = select_from(&fixture, &op, "f32", &shape, 8, rows)
                    .map(|row| row.schedule.as_str());
                assert_eq!(chosen, (2..=5).contains(&rows).then_some("measured"));
            }
            assert!(
                select_from(
                    &fixture,
                    &op,
                    "f32",
                    &[("k", 16), ("n", 32), ("ranks", 4), ("ranks", 4)],
                    8,
                    2
                )
                .is_none()
            );
            assert!(select_from(&fixture, &op, "f32", &[("n", 32); 9], 8, 2).is_none());
            assert!(
                select_from(
                    &fixture,
                    &op,
                    "f32",
                    &[("k", 16), ("n", 31), ("ranks", 4)],
                    8,
                    2
                )
                .is_none()
            );
            assert!(select_from(&fixture, &op, "f32", &shape, 16, 2).is_none());
            assert!(select_from(&fixture, &op, "bf16", &shape, 8, 2).is_none());
        }
    }
}
