//! Flash schedule choices. A missing row uses the model's baseline;
//! other geometries never inherit Flash's rows.
use crate::geometry::{FLASH, Geometry};
use kernel_build::tuning::{Row, Table, TuningError};
use std::collections::BTreeMap;
use std::sync::OnceLock;

// Family, its names and the embedded texts come from one build-time list.
include!(concat!(env!("OUT_DIR"), "/tuning.rs"));

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
/// `rows` launch rows. `None` means the model's baseline schedule.
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
        parse_family(family, &table.render()).unwrap()
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
