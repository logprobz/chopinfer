//! Audit the measured query substitution and retain every other phase.
use super::*;
const OLD: &str = "chop_deepseek41_mxfp8_ticket_n8192_k1280_g1_f32_v2";
const NEW: &str = "chop_fold_local_n8192_k1280_g1_f32_local32_n32_v1";
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
            records.push("case prefill32".into());
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            ends[0] = 32;
        }
    }
    for count in [8, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..count)
            .map(|i| (i + 1, i, ends[i as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        records.push(format!("case decode{count}"));
        records.extend_from_slice(&host.device.journal.borrow()[from..]);
        for end in &mut ends[..count as usize] {
            *end += 1;
        }
    }
    if kv.storage == KvStorage::Native {
        let from = host.device.journal.borrow().len();
        let tokens = vec![7; 4];
        let items = (0..8)
            .map(|i| (i + 1, i, ends[i as usize], tokens.as_slice()))
            .collect::<Vec<_>>();
        instance
            .execute(&verify(&items), &mut Views::default())
            .unwrap();
        instance
            .commit_accepted(&(1..=8).map(|i| kept(i, 4)).collect::<Vec<_>>())
            .unwrap();
        records.push("case verify32".into());
        records.extend_from_slice(&host.device.journal.borrow()[from..]);
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
fn audit(before: &[String], after: &[String]) -> Result<usize, String> {
    if before.len() != after.len() {
        return Err("operation count changed".into());
    }
    let mut c32 = false;
    let mut changed = 0;
    for (a, b) in before.iter().zip(after) {
        if a.starts_with("case ") {
            c32 = a == "case decode32";
        }
        if a == b {
            continue;
        }
        if !c32 || !a.starts_with("launch ") {
            return Err("changed an unrelated operation".into());
        }
        let symbol = a.split_whitespace().nth(2).unwrap();
        let expected = if symbol == format!("{OLD}_initialize_v2") {
            a.replace(&format!("{OLD}_initialize_v2"), &format!("{NEW}_admit_v1"))
                .replace("length: 10487040", "length: 256")
        } else if symbol == OLD {
            a.replace(OLD, NEW)
                .replace(
                    "grid_x: 128, grid_y: 2, grid_z: 3",
                    "grid_x: 256, grid_y: 1, grid_z: 1",
                )
                .replace("block_x: 256", "block_x: 128")
                .replace("length: 10487040", "length: 256")
                .replace(", ScalarU32(4)]", ", ScalarU32(0)]")
        } else {
            return Err("changed another projection".into());
        };
        if &expected != b {
            return Err(format!(
                "projection operands or geometry changed: {expected}\n{b}"
            ));
        }
        changed += 1;
    }
    Ok(changed)
}
#[test]
fn local_rows_change_only_backbone_decode32_query_calls() {
    for kv in KV_LAYOUTS {
        let before = crate::sm120::fold_local_rows::with_reference(|| run(kv, true));
        let after = run(kv, false);
        assert_eq!(audit(&before, &after).unwrap(), 320);
        assert_eq!(
            after
                .iter()
                .filter(|l| l.starts_with("launch ") && l.contains(&format!(" {NEW} ")))
                .count(),
            160
        );
        let at = after
            .iter()
            .position(|l| l.contains(&format!(" {NEW} ")))
            .unwrap();
        let mut bad = after.clone();
        bad[at] = bad[at].replacen("offset: ", "offset: 1", 1);
        assert!(audit(&before, &bad).is_err());
        let mut bad = after.clone();
        bad.remove(at - 1);
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
        "ec7cd1ebd9ab40f9f5532d82468cea3b879a89ccac49d8f180f5b66ec9de7756"
    );
    for shape in crate::shape::topologies() {
        assert_eq!(
            crate::Kernels::loads(cubin.name, &shape),
            shape.geometry.id == FLASH.id && shape.ranks() == 4
        );
    }
}
