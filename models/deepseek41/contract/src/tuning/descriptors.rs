//! Rows needed to construct cubin descriptors even when no launch selects them.
use kernel_build::tuning::{Row, Table, TuningError};

const REQUIRED: &[(&str, u32)] = &[
    ("dense_fold.status.decode", 32),
    ("dense_fold.status_wide.verify", 64),
    ("dense_fold.attention_q_a_status.verify", 64),
    ("dense_fold.attention_o_a_status.verify", 64),
    ("dense_fold.attention_o_b_status.verify", 64),
];

fn descriptor<'a>(
    table: &'a Table,
    op: &str,
    bucket: u32,
    rows: u32,
) -> Result<&'a Row, TuningError> {
    super::select_from(table, op, "mxfp8", &[("ranks", 4)], bucket, rows).ok_or_else(|| {
        TuningError(format!(
            "missing required tuning row: {op} mxfp8 ranks=4 bucket={bucket} row {rows}"
        ))
    })
}

pub(super) fn validate(table: &Table) -> Result<(), TuningError> {
    for &(op, bucket) in REQUIRED {
        descriptor(table, op, bucket, bucket)?;
    }
    for row in table.rows().iter().filter(|row| {
        row.key.weights == "mxfp8"
            && row.key.shape == [("ranks".into(), 4)].into()
            && matches!(
                (row.key.op.as_str(), row.key.bucket),
                ("dense_fold.status.decode", 32) | ("dense_fold.status_tails.decode", 64)
            )
    }) {
        let tile = row.knobs["tile_rows"].count().unwrap();
        if tile > 32 {
            continue;
        }
        let required = descriptor(table, "dense_fold.status.decode", 32, tile)?;
        if row.key.op == "dense_fold.status.decode" && row.knobs != required.knobs {
            return Err(TuningError(format!(
                "{}: must match status descriptor row {tile}",
                row.key
            )));
        }
        for site in [
            "attention_q_a",
            "attention_kv",
            "attention_o_a",
            "attention_o_b",
        ] {
            if row.key.op == "dense_fold.status_tails.decode" && site == "attention_kv" {
                continue;
            }
            descriptor(
                table,
                &format!("dense_fold.{site}_cluster.decode"),
                32,
                tile,
            )?;
        }
    }
    for row in table.rows() {
        if row.key.weights != "mxfp8"
            || row.key.shape != [("ranks".into(), 4)].into()
            || row.key.bucket != 64
        {
            continue;
        }
        let op = match row.key.op.as_str() {
            "dense_fold.status_wide.verify" | "dense_fold.status_wide_tails.decode" => {
                "dense_fold.status_wide.verify"
            }
            "dense_fold.attention_q_a_status.verify"
            | "dense_fold.attention_o_a_status.verify"
            | "dense_fold.attention_o_b_status.verify" => row.key.op.as_str(),
            _ => continue,
        };
        let required = descriptor(table, op, 64, 64)?;
        if row.schedule != required.schedule || row.knobs != required.knobs {
            return Err(TuningError(format!(
                "{}: must match descriptor {op} row 64",
                row.key
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{Family, TABLES, parse_family};

    #[test]
    fn every_required_descriptor_refuses_deletion_or_excluded_launch_row() {
        let text = TABLES
            .iter()
            .find(|(f, _)| *f == Family::DenseFold)
            .unwrap()
            .1;
        let full = Table::parse(text).unwrap();
        for &(op, bucket) in REQUIRED {
            for narrow in [false, true] {
                let mut table = Table::new(full.target);
                for mut row in full.rows().iter().cloned() {
                    if row.key.op == op
                        && row.key.bucket == bucket
                        && row.key.rows.unwrap_or([1, bucket])[1] == bucket
                    {
                        if !narrow {
                            continue;
                        }
                        let [lo, _] = row.key.rows.unwrap_or([1, bucket]);
                        row.key.rows = Some([lo.min(bucket - 1), bucket - 1]);
                    }
                    table.upsert(row).unwrap();
                }
                let error = parse_family(Family::DenseFold, &table.render()).unwrap_err();
                assert!(error.0.contains(op), "{error}");
                assert!(error.0.contains(&format!("row {bucket}")), "{error}");
            }
        }
    }

    #[test]
    fn retained_status_routes_require_their_tile_descriptors() {
        let text = TABLES
            .iter()
            .find(|(f, _)| *f == Family::DenseFold)
            .unwrap()
            .1;
        let full = Table::parse(text).unwrap();
        for op in [
            "dense_fold.status.decode",
            "dense_fold.attention_q_a_cluster.decode",
        ] {
            let mut table = Table::new(full.target);
            for row in full.rows().iter().cloned() {
                if row.key.op == op && row.key.rows == Some([16, 16]) {
                    continue;
                }
                table.upsert(row).unwrap();
            }
            let error = parse_family(Family::DenseFold, &table.render()).unwrap_err();
            assert!(
                error.0.contains(op) && error.0.contains("row 16"),
                "{error}"
            );
        }
    }

    #[test]
    fn wide_tail_payload_must_match_its_required_descriptor() {
        let text = TABLES
            .iter()
            .find(|(f, _)| *f == Family::DenseFold)
            .unwrap()
            .1;
        let mut table = Table::parse(text).unwrap();
        let mut tail = table
            .rows()
            .iter()
            .find(|r| r.key.op == "dense_fold.status_wide_tails.decode")
            .unwrap()
            .clone();
        tail.knobs
            .insert("tile_n".into(), kernel_build::tuning::Knob::Count(32));
        table.upsert(tail).unwrap();
        let error = parse_family(Family::DenseFold, &table.render()).unwrap_err();
        assert!(
            error
                .0
                .contains("must match descriptor dense_fold.status_wide.verify row 64"),
            "{error}"
        );
    }
}
