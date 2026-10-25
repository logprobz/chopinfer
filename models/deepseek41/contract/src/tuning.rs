//! Flash schedule choices. A missing row uses the model's baseline;
//! other geometries never inherit Flash's rows.
use crate::geometry::{FLASH, Geometry};
use kernel_build::tuning::{Row, Table};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Schedule families with independent tuning files and coverage tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    DenseFold,
    Reuse,
    DensePrefill,
    F32,
    Moe,
    Hc,
    Norm,
    Latent,
    Indexer,
}

impl Family {
    pub const ALL: [Self; 9] = [
        Self::DenseFold,
        Self::Reuse,
        Self::DensePrefill,
        Self::F32,
        Self::Moe,
        Self::Hc,
        Self::Norm,
        Self::Latent,
        Self::Indexer,
    ];

    /// The filename stem and operation prefix.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DenseFold => "dense_fold",
            Self::Reuse => "reuse",
            Self::DensePrefill => "dense_prefill",
            Self::F32 => "f32",
            Self::Moe => "moe",
            Self::Hc => "hc",
            Self::Norm => "norm",
            Self::Latent => "latent",
            Self::Indexer => "indexer",
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/tuning.rs"));

fn tables() -> &'static BTreeMap<Family, Table> {
    static PARSED: OnceLock<BTreeMap<Family, Table>> = OnceLock::new();
    PARSED.get_or_init(|| {
        TABLES
            .iter()
            .map(|&(family, text)| {
                let table = Table::parse(text)
                    .unwrap_or_else(|e| panic!("tuning/flash/{}.json: {e}", family.name()));
                (family, table)
            })
            .collect()
    })
}

/// Flash's own family table, with no inheritance for another geometry.
pub fn table(g: &Geometry, family: Family) -> Option<&'static Table> {
    if g.id != FLASH.id {
        return None;
    }
    tables().get(&family)
}

/// The schedule for this exact operation, format, shape and bucket at
/// `rows` launch rows. `None` means the model's baseline schedule.
pub fn select(
    g: &Geometry,
    family: Family,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
    rows: u32,
) -> Option<&'static Row> {
    let table = table(g, family)?;
    let shape = shape
        .iter()
        .map(|&(name, size)| (name.to_owned(), size))
        .collect();
    table.select(op, weights, &shape, bucket, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_has_one_canonical_flash_table() {
        assert_eq!(TABLES.len(), Family::ALL.len());
        assert_eq!(tables().len(), Family::ALL.len());
        for family in Family::ALL {
            let table = table(&FLASH, family).unwrap();
            let text = TABLES.iter().find(|(f, _)| *f == family).unwrap().1;
            assert_eq!(table.render(), text);
            assert_eq!(table.target.compute_capability, 120);
            assert_eq!(table.target.sm_count, 188);
        }
    }

    #[test]
    fn other_geometries_do_not_inherit_flash_tables() {
        let untuned = Geometry {
            id: "untuned",
            ..FLASH
        };
        for family in Family::ALL {
            assert!(table(&untuned, family).is_none());
            assert!(
                select(
                    &untuned,
                    family,
                    "hc.mix.decode",
                    "f32",
                    &[("width", 4096)],
                    8,
                    1
                )
                .is_none()
            );
        }
    }

    #[test]
    fn missing_rows_use_the_baseline() {
        for family in Family::ALL {
            assert!(
                select(
                    &FLASH,
                    family,
                    "missing.site.decode",
                    "f32",
                    &[("width", 4096)],
                    8,
                    1
                )
                .is_none()
            );
        }
    }
}
