//! Audit matrix-load substitutions while keeping every other launch exact.
use super::shared_overlap::journal_spans;
use super::*;

fn run(rows: u32, kv: KvLayout, drafter: bool, reference: bool) -> Vec<String> {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, kv);
    host.drafter.set(drafter);
    if drafter {
        host.speculation.set(Some((5, 32)));
    }
    let record = crate::config::tests::record(&shape, kv, 1, false);
    let services = HostServicesTableV1::new(Rc::new(dspark::MappedHost::new(Rc::clone(&host))));
    let mut instance = api().create(services, &record).unwrap();
    let start = host.device.journal.borrow().len();
    instance
        .execute(
            &prompt(7, 0, 0, u64::from(rows), true),
            &mut Views::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    let lines = host.device.journal.borrow()[start..].to_vec();
    assert_launches_fit_configured_shared(&host.device.journal.borrow(), &shape.topology());
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    if let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK41_MATRIX_JOURNAL_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        let name = format!(
            "rows{rows}-{:?}-draft{drafter}-reference{reference}.journal",
            kv.storage
        );
        std::fs::write(root.join(&name), lines.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &root, &name);
    }
    lines
}
fn args(line: &str) -> &str {
    line.split_once(" } [").unwrap().1
}
/// A launch's symbol without the packed suffix: both arms read packed codes.
fn symbol(line: &str) -> &str {
    let symbol = line.split_whitespace().nth(2).unwrap();
    crate::sm120::test_symbols::row_major_fold_symbol(symbol).unwrap_or(symbol)
}
fn compute_spans(line: &str) -> Vec<(u64, u64, u64)> {
    let mut spans = journal_spans(line);
    if line.contains("_prefill_tma_") && !line.contains("_split_tail_") {
        // TMA descriptors repeat A and W before the pointer operands.
        assert_eq!(spans.len(), 9);
        assert_eq!(spans[0], spans[2]);
        assert_eq!(spans[1], spans[3]);
        spans.drain(..2);
        spans.swap(1, 2);
    }
    spans
}

/// The reference keeps full TMA tiles and writes the remainder through exact
/// rows. Check those rows against the matrix launch's complete operands.
fn audit_fold_tail(
    admit: &str,
    compute: &str,
    matrix: &str,
    base: &str,
    rows: u64,
    offset: u64,
    count: u64,
) -> Result<(), String> {
    let full = compute_spans(matrix);
    let admission = journal_spans(admit);
    let tail = journal_spans(compute);
    let dimensions = base
        .strip_prefix("chop_deepseek41_mxfp8_fold_m8192_")
        .ok_or("unknown matrix projection")?
        .split('_')
        .take(3)
        .collect::<Vec<_>>()
        .join("_");
    let name = admit
        .split_whitespace()
        .nth(2)
        .ok_or("missing tail symbol")?;
    let stem = name
        .strip_suffix("_admit_v1")
        .ok_or("missing tail admission")?;
    if !stem.starts_with(&format!(
        "chop_deepseek41_dense_decode_{dimensions}_single_"
    )) || compute.split_whitespace().nth(2) != Some(&format!("{stem}_packed_v1"))
        || [admit, compute]
            .iter()
            .any(|line| line.split_whitespace().nth(1) != matrix.split_whitespace().nth(1))
        || !args(admit).ends_with(&format!("ScalarU64({offset}), ScalarU32({count})]"))
        || !args(compute).contains(&format!("ScalarU32({count}),"))
        || full.len() != 7
        || admission.len() != 5
        || tail.len() != 7
    {
        return Err("tail fold identity, counts or stream changed".into());
    }
    let slice = |index: usize| -> Result<_, String> {
        let (buffer, start, bytes) = full[index];
        if !bytes.is_multiple_of(rows) {
            return Err("matrix operand has no integral row stride".into());
        }
        let stride = bytes / rows;
        Ok((buffer, start + offset * stride, count * stride))
    };
    let (a, scales, output) = (slice(0)?, slice(1)?, slice(4)?);
    let scratch = admission[2];
    if admission != [a, scales, scratch, full[5], full[6]]
        || tail != [a, scales, full[2], full[3], output, scratch, full[5]]
        || full.iter().any(|&(buffer, start, bytes)| {
            buffer == scratch.0 && start < scratch.1 + scratch.2 && scratch.1 < start + bytes
        })
    {
        return Err("tail fold operands or scratch changed".into());
    }
    Ok(())
}

fn audit(before: &[String], after: &[String]) -> Result<usize, String> {
    let old: Vec<_> = before.iter().filter(|l| l.starts_with("launch ")).collect();
    let new: Vec<_> = after.iter().filter(|l| l.starts_with("launch ")).collect();
    let (mut i, mut j, mut changed) = (0, 0, 0);
    while i < old.len() && j < new.len() {
        if old[i] == new[j] {
            i += 1;
            j += 1;
            continue;
        }
        let Some((base, _)) = symbol(new[j]).split_once("_matrix_") else {
            return Err(format!(
                "unexpected launch change {} / {}",
                symbol(old[i]),
                symbol(new[j])
            ));
        };
        if !symbol(new[j]).ends_with("_admit_v1")
            || !symbol(old[i]).starts_with(base)
            || !symbol(old[i]).ends_with("_admit_v1")
            || args(old[i]) != args(new[j])
            || old[i].split_whitespace().nth(1) != new[j].split_whitespace().nth(1)
        {
            return Err("admission operands or stream changed".into());
        }
        i += 1;
        j += 1;
        if i == old.len()
            || j == new.len()
            || !symbol(old[i]).starts_with(base)
            || !symbol(new[j]).starts_with(base)
            || symbol(new[j]).ends_with("_admit_v1")
            || compute_spans(old[i]) != compute_spans(new[j])
            || old[i].split("ScalarU64(").last() != new[j].split("ScalarU64(").last()
            || old[i].split_whitespace().nth(1) != new[j].split_whitespace().nth(1)
        {
            return Err("compute operands, counts or stream changed".into());
        }
        let rows = old[i]
            .split("ScalarU32(")
            .last()
            .and_then(|text| text.split(')').next())
            .and_then(|text| text.parse::<u64>().ok())
            .ok_or("missing prefill row count")?;
        if rows % 128 != 0 && symbol(old[i]).contains("_prefill_tma_") {
            let mut offset = rows / 128 * 128;
            while offset < rows {
                if i + 2 >= old.len() {
                    return Err("missing tail fold".into());
                }
                let count = (rows - offset).min(64);
                audit_fold_tail(old[i + 1], old[i + 2], new[j], base, rows, offset, count)?;
                i += 2;
                offset += count;
            }
        } else if symbol(old[i]).ends_with("_split_full_v1") {
            i += 1;
            if i == old.len()
                || !symbol(old[i]).starts_with(base)
                || !symbol(old[i]).ends_with("_split_tail_v1")
                || compute_spans(old[i]) != compute_spans(new[j])
            {
                return Err("split tail operands changed".into());
            }
        }
        i += 1;
        j += 1;
        changed += 1;
    }
    if i != old.len() || j != new.len() {
        return Err("launches missing".into());
    }
    let synchronization = |lines: &[String]| {
        lines
            .iter()
            .filter(|l| {
                l.starts_with("record ") || l.starts_with("wait ") || l.starts_with("synchronize ")
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    if synchronization(before) != synchronization(after) {
        return Err("dependency ordering changed".into());
    }
    Ok(changed)
}

#[test]
fn preserve_operands_and_dependencies_in_ar_and_dspark_prefill() {
    crate::sm120::prefill_tma_scales::with_reference(|| {
        for (kv, drafter) in [
            (KV_LAYOUTS[0], false),
            (KV_LAYOUTS[1], false),
            (KV_LAYOUTS[0], true),
        ] {
            for rows in [192, 2496, 8192] {
                let before =
                    crate::sm120::prefill_matrix::with_reference(|| run(rows, kv, drafter, true));
                let after = run(rows, kv, drafter, false);
                let changed = audit(&before, &after).unwrap();
                assert!(
                    changed >= 4 * 4,
                    "four projections on each TP4 rank must change"
                );
                let mut bad = after.clone();
                let at = bad
                    .iter()
                    .position(|l| {
                        l.starts_with("launch ")
                            && l.contains("_matrix_")
                            && !l.contains("_admit_v1")
                    })
                    .unwrap();
                bad[at] = bad[at].replacen("offset: ", "offset: 1", 1);
                assert!(
                    audit(&before, &bad).is_err(),
                    "changed operand escaped the audit"
                );
                let mut missing = after.clone();
                missing.remove(at - 1);
                assert!(
                    audit(&before, &missing).is_err(),
                    "missing admission escaped the audit"
                );
                if let Some(tail) = before.iter().position(|line| {
                    line.starts_with("launch ")
                        && line.contains("dense_decode_")
                        && line.contains("_admit_v1 ")
                }) {
                    let mut bad = before.clone();
                    bad[tail] = bad[tail].replacen("ScalarU64(", "ScalarU64(1", 1);
                    assert!(audit(&bad, &after).is_err(), "tail status offset escaped");
                    let mut missing = before.clone();
                    missing.remove(tail + 1);
                    assert!(
                        audit(&missing, &after).is_err(),
                        "missing tail fold escaped"
                    );
                }
            }
        }
    });
}

#[test]
fn retain_the_qualified_cubin_and_load_only_flash_tp4() {
    use crate::sm120::prefill_matrix::{CUBIN, enabled};
    let cubin = crate::CUBINS.iter().find(|c| c.name == CUBIN).unwrap();
    assert_eq!(
        kernel_build::digest(cubin.bytes),
        // The qualified tiles' packed twins (kernel-lab
        // mxfp8_prefill_matrix_packed: bitwise equal to the row-major tiles).
        "7d4fb44917e658735acb5e8bc302c3f7243c45cdd2bad52421049b53083cc46a"
    );
    for shape in crate::shape::topologies() {
        assert_eq!(
            crate::Kernels::loads(CUBIN, &shape),
            enabled(shape.geometry, shape.ranks())
        );
    }
}
