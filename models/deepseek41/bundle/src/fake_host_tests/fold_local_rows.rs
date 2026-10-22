//! Audit the measured row-local folds: each replaces only its projection's
//! fold in the decode and verify buckets `fold_local_rows::SERVED` names,
//! on the same operands, and every other launch is retained.
use super::shared_overlap::journal_spans;
use super::*;
use crate::sm120::fold_local_rows::SERVED;
fn run(kv: KvLayout, reference: bool) -> Vec<String> {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, kv);
    if kv.storage == KvStorage::Native {
        host.speculation.set(Some((5, 32)));
    }
    let mut instance = create(&host, 32);
    let mut records = Vec::new();
    let mut ends = [1_u64; 32];
    for slot in 0..32 {
        let count = if slot == 0 { 32 } else { 1 };
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, count, true),
            &mut Views::default(),
        );
        if slot == 0 {
            records.push("case prefill 32".into());
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            ends[0] = 32;
        }
    }
    for count in [8, 16, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..count)
            .map(|i| (i + 1, i, ends[i as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        records.push(format!("case decode {count}"));
        records.extend_from_slice(&host.device.journal.borrow()[from..]);
        for end in &mut ends[..count as usize] {
            *end += 1;
        }
    }
    if kv.storage == KvStorage::Native {
        // Four rows per item: verify 32 on slots 0..8, then verify 16 on
        // slots 8..12.
        for (first, count) in [(0_u64, 8_u64), (8, 4)] {
            let from = host.device.journal.borrow().len();
            let tokens = vec![7; 4];
            let items = (first..first + count)
                .map(|i| (i + 1, i, ends[i as usize], tokens.as_slice()))
                .collect::<Vec<_>>();
            instance
                .execute(&verify(&items), &mut Views::default())
                .unwrap();
            instance
                .commit_accepted(
                    &(first + 1..=first + count)
                        .map(|i| kept(i, 4))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            records.push(format!("case verify {}", count * 4));
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            for end in &mut ends[first as usize..(first + count) as usize] {
                *end += 4;
            }
        }
    }
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    if let Some(root) = std::env::var_os("CHOPINFER_LOCAL_ROWS_JOURNALS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        let name = format!("local.{:?}.reference{reference}.journal", kv.storage);
        std::fs::write(root.join(&name), records.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &root, &name);
    }
    records
}
fn symbol(line: &str) -> &str {
    line.split_whitespace().nth(2).unwrap_or("")
}
/// The replaced projections of each case, by (case, projection).
fn audit(before: &[String], after: &[String]) -> Result<Vec<(String, String)>, String> {
    let (mut i, mut j) = (0, 0);
    let mut case = String::new();
    let mut changed = Vec::new();
    // Launches the replacements have added to the open capture.
    let mut nodes = 0_i64;
    while i < before.len() && j < after.len() {
        let (a, b) = (&before[i], &after[j]);
        if a == b {
            if let Some(name) = a.strip_prefix("case ") {
                case = name.to_owned();
            }
            if a.starts_with("capture ") {
                nodes = 0;
            }
            i += 1;
            j += 1;
            continue;
        }
        // A capture closes with its node count, which the replacements alone
        // change.
        let count = |line: &str| {
            line.strip_prefix("captured ")
                .and_then(|rest| rest.rsplit_once(' '))
                .map(|(head, count)| (head.to_owned(), count.parse::<i64>().unwrap()))
        };
        if let (Some((graph, old)), Some((same, new))) = (count(a), count(b)) {
            if graph != same || new != old + nodes {
                return Err(format!("a capture changed more than its folds: {a}\n{b}"));
            }
            i += 1;
            j += 1;
            continue;
        }
        let ring = after.get(j + 1).ok_or("a row-local fold lacks its ring")?;
        let shape = symbol(b)
            .strip_prefix("chop_fold_local_")
            .and_then(|s| s.split("_g1_f32_local32_").next())
            .ok_or_else(|| format!("changed an unrelated operation: {a}\n{b}"))?;
        if !b.starts_with("launch ")
            || !ring.starts_with("launch ")
            || symbol(b) != format!("{}_admit_v1", symbol(ring))
        {
            return Err(format!("a row-local fold is not admit then ring: {b}"));
        }
        let (kind, rows) = case.split_once(' ').unwrap();
        let rows: u32 = rows.parse().unwrap();
        let Some(&(id, _, _, _, tile_n, stages)) = SERVED
            .iter()
            .find(|s| s.1 == rows && format!("n{}_k{}", s.2, s.3) == shape && kind != "prefill")
        else {
            return Err(format!("no row-local fold serves {case}: {b}"));
        };
        let x = if stages == 0 {
            String::new()
        } else {
            format!("_s{stages}x")
        };
        if !symbol(ring).ends_with(&format!("_local32_n{tile_n}{x}_v1")) {
            return Err(format!("{id} at {case} runs another ring: {ring}"));
        }
        // The reference fold of the same shape that the pair replaces.
        let start = i;
        let mut old = Vec::new();
        while i < before.len()
            && before[i].starts_with("launch ")
            && symbol(&before[i]).contains(&format!("_{shape}_g1_"))
        {
            old.extend(journal_spans(&before[i]));
            i += 1;
        }
        if old.is_empty() {
            return Err(format!("{id} at {case} replaces no fold: {a}"));
        }
        nodes += 2 - (i - start) as i64;
        // Every operand but the 256-byte workspace is one the fold passed;
        // the workspace starts where the fold's did.
        for span in journal_spans(b).into_iter().chain(journal_spans(ring)) {
            let found = if span.2 == 256 {
                old.iter().any(|o| (o.0, o.1) == (span.0, span.1))
            } else {
                old.contains(&span)
            };
            if !found {
                return Err(format!("{id} at {case} binds another operand: {span:?}"));
            }
        }
        changed.push((case.clone(), id.to_owned()));
        j += 2;
    }
    if i != before.len() || j != after.len() {
        return Err("operation count changed".into());
    }
    Ok(changed)
}
#[test]
fn local_rows_change_only_measured_decode_and_verify_projections() {
    for kv in KV_LAYOUTS {
        // The cluster folds, audited by `fold_cluster`, take over WO_b's
        // rings at 16 and 32 rows; both sides run without them.
        let (before, after) = crate::sm120::fold_cluster::with_reference(|| {
            (
                crate::sm120::fold_local_rows::with_reference(|| run(kv, true)),
                run(kv, false),
            )
        });
        let changed = audit(&before, &after).unwrap();
        let mut expected = Vec::new();
        let mut cases = vec!["decode 16", "decode 32"];
        if kv.storage == KvStorage::Native {
            cases.extend(["verify 32", "verify 16"]);
        }
        // Every backbone layer on each of the four ranks, but not the
        // drafter's layer.
        for case in cases {
            let rows: u32 = case.split_once(' ').unwrap().1.parse().unwrap();
            for id in SERVED.iter().filter(|s| s.1 == rows).map(|s| s.0) {
                expected.push((
                    (case.to_owned(), id.to_owned()),
                    4 * FLASH.layers() as usize,
                ));
            }
        }
        let mut counted = std::collections::BTreeMap::new();
        for key in changed {
            *counted.entry(key).or_insert(0) += 1;
        }
        expected.sort();
        assert_eq!(counted.into_iter().collect::<Vec<_>>(), expected, "{kv:?}");
        let at = after
            .iter()
            .position(|l| l.contains(" chop_fold_local_n5120_k2048_g1_f32_local32_n32_s5x_v1 "))
            .unwrap();
        let mut bad = after.clone();
        bad[at] = bad[at].replacen("offset: ", "offset: 1", 1);
        assert!(audit(&before, &bad).is_err());
        let mut bad = after.clone();
        bad.remove(at - 1);
        assert!(audit(&before, &bad).is_err());
        let mut bad = after.clone();
        for line in &mut bad[at - 1..=at] {
            *line = line.replace("_s5x_v1", "_s3x_v1");
        }
        assert!(audit(&before, &bad).is_err());
    }
}
#[test]
fn local_rows_retain_the_measured_cubin_only_at_flash_tp4() {
    let cubin = crate::CUBINS
        .iter()
        .find(|c| c.name == crate::sm120::fold_local_rows::CUBIN)
        .unwrap();
    assert_eq!(
        kernel_build::digest(cubin.bytes),
        "20d4fdcf84ab9697b07dd9f4a9464db650e78f6718424d565d626a5bef425c42"
    );
    for shape in crate::shape::topologies() {
        assert_eq!(
            crate::Kernels::loads(cubin.name, &shape),
            shape.geometry.id == FLASH.id && shape.ranks() == 4
        );
    }
}
