//! Audit the measured staged-scale substitutions and preserve other launches.
use super::shared_overlap::journal_spans;
use super::*;

fn run(rows: u32, kv: KvLayout, drafter: bool, reference: bool) -> Vec<String> {
    run_case(rows, kv, drafter, reference, "")
}
fn run_case(rows: u32, kv: KvLayout, drafter: bool, reference: bool, prefix: &str) -> Vec<String> {
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
    if let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK41_TMA_SCALES_JOURNAL_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        let name = format!(
            "{prefix}rows{rows}-{:?}-draft{drafter}-reference{reference}.journal",
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
fn symbol(line: &str) -> &str {
    line.split_whitespace().nth(2).unwrap()
}
fn compute_spans(line: &str) -> Result<Vec<(u64, u64, u64)>, String> {
    let mut spans = journal_spans(line);
    if line.contains("_tma_matrix_")
        || line.contains("_tma_scales_")
        || line.contains("_prefill_tma_")
    {
        // TMA descriptors repeat A and W before the pointer operands.
        if spans.len() != 9 || spans[0] != spans[2] || spans[1] != spans[3] {
            return Err("TMA descriptors differ from pointer operands".into());
        }
        spans.drain(..2);
        spans.swap(1, 2);
    }
    Ok(spans)
}
fn audit(before: &[String], after: &[String]) -> Result<usize, String> {
    let old: Vec<_> = before.iter().filter(|l| l.starts_with("launch ")).collect();
    let new: Vec<_> = after.iter().filter(|l| l.starts_with("launch ")).collect();
    if old.len() != new.len() {
        return Err("launch count changed".into());
    }
    let mut changed = 0;
    for (a, b) in old.iter().zip(&new) {
        if a == b {
            continue;
        }
        let prior = symbol(a)
            .split_once("_tma_matrix_")
            .or_else(|| symbol(a).split_once("_matrix_"))
            .or_else(|| {
                // The full-row Engram baseline uses the native TMA entry.
                symbol(a)
                    .contains("_n25600_k6144_g1_")
                    .then(|| symbol(a).split_once("_prefill_tma_"))
                    .flatten()
            })
            .map(|(base, _)| base);
        let next = symbol(b).split_once("_tma_scales_").map(|(base, _)| base);
        if prior.is_none()
            || prior != next
            || a.split_whitespace().nth(1) != b.split_whitespace().nth(1)
        {
            return Err("unqualified launch change".into());
        }
        if symbol(a).ends_with("_admit_v1") {
            if !symbol(b).ends_with("_admit_v1") || args(a) != args(b) {
                return Err("admission changed".into());
            }
        } else {
            if symbol(b).ends_with("_admit_v1")
                || compute_spans(a)? != compute_spans(b)?
                || a.split("ScalarU64(").last() != b.split("ScalarU64(").last()
            {
                return Err("compute operands or scalars changed".into());
            }
            changed += 1;
        }
    }
    let sync = |rows: &[String]| {
        rows.iter()
            .filter(|l| {
                l.starts_with("record ") || l.starts_with("wait ") || l.starts_with("synchronize ")
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    if sync(before) != sync(after) {
        return Err("dependencies changed".into());
    }
    Ok(changed)
}
#[test]
fn preserve_exact_operands_and_dependencies_for_staged_scale_buckets() {
    for (kv, drafter) in [
        (KV_LAYOUTS[0], false),
        (KV_LAYOUTS[1], false),
        (KV_LAYOUTS[0], true),
    ] {
        for rows in [2496, 4096, 8192] {
            let before =
                crate::sm120::prefill_tma_scales::with_reference(|| run(rows, kv, drafter, true));
            let after = run(rows, kv, drafter, false);
            let count = audit(&before, &after).unwrap();
            if rows == 2496 {
                assert_eq!(count, 0);
                continue;
            }
            assert!(count >= 16);
            let at = after
                .iter()
                .position(|l| {
                    l.starts_with("launch ")
                        && l.contains("_tma_scales_")
                        && !l.contains("_admit_v1")
                })
                .unwrap();
            let mut bad = after.clone();
            bad[at] = bad[at].replacen("offset: ", "offset: 1", 1);
            assert!(audit(&before, &bad).is_err());
            let mut missing = after.clone();
            missing.remove(at - 1);
            assert!(audit(&before, &missing).is_err());
        }
    }
}
#[test]
fn retain_the_qualified_module_only_for_flash_tp4() {
    let name = crate::sm120::prefill_tma_scales::CUBIN;
    let cubin = crate::CUBINS.iter().find(|c| c.name == name).unwrap();
    assert_eq!(
        kernel_build::digest(cubin.bytes),
        "3b91575bc78d3d40065e2f18291e596745cfcfdf8690bd097051a3cc099868fa"
    );
    for shape in crate::shape::topologies() {
        assert_eq!(
            crate::Kernels::loads(name, &shape),
            shape.geometry.id == FLASH.id && shape.ranks() == 4
        );
    }
}

#[test]
fn engram_changes_only_both_full_row_projection_sites() {
    for (kv, drafter) in [
        (KV_LAYOUTS[0], false),
        (KV_LAYOUTS[1], false),
        (KV_LAYOUTS[0], true),
    ] {
        for rows in [4096, 8192] {
            let before = crate::sm120::prefill_tma_scales::with_engram_reference(|| {
                run_case(rows, kv, drafter, true, "engram-")
            });
            let after = run_case(rows, kv, drafter, false, "engram-");
            assert_eq!(
                audit(&before, &after).unwrap(),
                if rows == 8192 { 8 } else { 0 }
            );
            let selected = |line: &&String| {
                line.starts_with("launch ")
                    && line.contains("_n25600_k6144_g1_")
                    && line.contains("_tma_scales_s2_v1 ")
            };
            assert_eq!(before.iter().filter(selected).count(), 0);
            assert_eq!(
                after.iter().filter(selected).count(),
                if rows == 8192 { 8 } else { 0 }
            );
            if rows == 8192 {
                let at = after.iter().position(|line| selected(&line)).unwrap();
                let mut corrupt = after.clone();
                corrupt[at] = corrupt[at].replacen("offset: ", "offset: 1", 1);
                assert!(audit(&before, &corrupt).is_err());
                let mut missing = after.clone();
                missing.remove(at - 1);
                assert!(audit(&before, &missing).is_err());
            }
        }
    }
}
