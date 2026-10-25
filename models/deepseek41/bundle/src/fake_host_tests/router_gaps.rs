//! Full lifecycle journals for the separately proposed small router admissions.
use super::*;
use std::path::Path;

fn save(root: &Path, name: &str, host: &FakeHost, from: usize) {
    std::fs::write(
        root.join(format!("{name}.journal")),
        host.device.journal.borrow()[from..].join("\n"),
    )
    .unwrap();
}

fn export(root: &Path) {
    std::fs::create_dir(root).unwrap();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.speculation.set(Some((5, 64)));
    let mut instance = create(&host, 32);
    for slot in 0..32 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    save(root, "create-prefill", &host, 0);
    let mut ends = [1u64; 32];
    for rows in [1u32, 2, 4, 8, 16, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..rows)
            .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        for end in &mut ends[..rows as usize] {
            *end += 1;
        }
        save(root, &format!("decode-{rows}"), &host, from);
    }
    for rows in (2u32..=32).chain([33, 48, 64]) {
        let count = rows.div_ceil(6);
        let tokens = (0..count)
            .map(|slot| vec![7; (rows / count + u32::from(slot < rows % count)) as usize])
            .collect::<Vec<_>>();
        let items = tokens
            .iter()
            .enumerate()
            .map(|(slot, tokens)| (slot as u64 + 1, slot as u64, ends[slot], tokens.as_slice()))
            .collect::<Vec<_>>();
        let from = host.device.journal.borrow().len();
        instance
            .execute(&verify(&items), &mut Views::default())
            .unwrap();
        instance
            .commit_accepted(
                &(0..u64::from(count))
                    .map(|slot| kept(slot + 1, 1))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for end in &mut ends[..count as usize] {
            *end += 1;
        }
        save(root, &format!("verify-{rows}"), &host, from);
    }
    let from = host.device.journal.borrow().len();
    for request in 1..=32 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    save(root, "destroy", &host, from);
}

#[test]
#[ignore = "writes audit journals to CHOPINFER_ROUTER_GAP_JOURNALS outside the repository"]
fn export_router_gap_lifecycles() {
    let root = std::path::PathBuf::from(
        std::env::var_os("CHOPINFER_ROUTER_GAP_JOURNALS").expect("set external journal output"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    assert!(
        !root.starts_with(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .canonicalize()
                .unwrap()
        )
    );
    crate::sm120::moe_router_finish::with_base_admission(|| export(&root.join("reference")));
    export(&root.join("candidate"));
    let mut unchanged = 0;
    for path in std::fs::read_dir(root.join("reference")).unwrap() {
        let path = path.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        let verify_tail = name
            .strip_prefix("verify-")
            .and_then(|n| n.strip_suffix(".journal"))
            .and_then(|n| n.parse::<u32>().ok())
            .is_some_and(|rows| matches!(rows, 2 | 3 | 5 | 7 | 9..=15 | 17..=31));
        if ["decode-2.journal", "decode-4.journal", "verify-5.journal"].contains(&name)
            || verify_tail
        {
            continue;
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(root.join("candidate").join(name)).unwrap(),
            "unrelated lifecycle changed: {name}"
        );
        unchanged += 1;
    }
    assert_eq!(unchanged, 14);
    println!(
        "ROUTER_GAP_LIFECYCLES total=42 unchanged={unchanged} candidates={} resources_balanced=true PASS",
        42 - unchanged
    );
}
