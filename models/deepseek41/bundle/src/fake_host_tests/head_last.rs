//! Audited head row routing with the same inventory on both comparison arms.
use super::*;

#[test]
fn qualified_module_and_row_policy_are_explicit() {
    use crate::sm120::{f32_head, head_last};
    let module = crate::CUBINS
        .iter()
        .find(|m| m.name == head_last::CUBIN)
        .unwrap();
    assert_eq!(
        kernel_build::digest(module.bytes),
        "5d641744eda4253538b0bb7ca23f723d0a28ca3171744c745baeee6c4a950014"
    );
    for shape in crate::shape::topologies() {
        let enabled = head_last::enabled(shape.geometry, shape.ranks());
        assert_eq!(crate::Kernels::loads(head_last::CUBIN, &shape), enabled);
        let entry = f32_head::entry(shape.geometry, shape.ranks())
            .unwrap()
            .with_split_decode();
        let entry = head_last::head(shape.geometry, shape.ranks(), entry).unwrap();
        for rows in 1..=256 {
            assert_eq!(
                entry.decode_last(rows).is_some(),
                enabled && [1, 8, 32].contains(&rows)
            );
        }
    }
}

/// One prefill step of `items` new one-token prompts from request `first`,
/// in slots 0 onward, each scoring its row: a head of `items` rows.
pub(super) fn prompts(first: u64, items: u32) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let slots = 0..u64::from(items);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        slots
            .clone()
            .map(|slot| BatchItemV1::new(first + slot, false, 1, true, vec![7]))
            .collect(),
        slots
            .clone()
            .map(|slot| BatchAppendV1::new(0, 1, 1, vec![slot * 8], None))
            .collect(),
    )
    .with_state_slots(slots.collect())
    .encode()
    .unwrap()
}

/// Decode and verify heads run the head's row tiles, so the measured last
/// producers serve the prefill heads of 1, 8 and 32 rows: prefills of one
/// to 32 one-token prompts.
fn run_case(reference: bool) -> Vec<String> {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 32);
    let mut first = 1;
    for rows in [1u32, 2, 6, 8, 9, 16, 24, 31, 32] {
        let from = host.device.journal.borrow().len();
        step(&mut instance, &prompts(first, rows), &mut Views::default());
        let selected = !reference && [1, 8, 32].contains(&rows);
        assert_eq!(
            host.device.journal.borrow()[from..]
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_last_m"))
                .count(),
            if selected { 8 } else { 0 },
            "rows={rows}"
        );
        for request in first..first + u64::from(rows) {
            instance.retire_request(request).unwrap();
        }
        first += u64::from(rows);
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let lines = host.device.journal.borrow().clone();
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_HEAD_LAST_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = if reference {
            "head-reference.journal"
        } else {
            "head-serving.journal"
        };
        std::fs::write(dir.join(name), lines.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, name);
    }
    lines
}

#[test]
fn measured_heads_preserve_other_launches_and_dependencies() {
    let before = crate::sm120::head_last::with_reference(|| run_case(true));
    let after = run_case(false);
    let retained = |lines: Vec<String>| {
        lines
            .into_iter()
            .filter(|line| {
                line.starts_with("launch ") && !line.contains("_head_f32_m")
                    || line.starts_with("record ")
                    || line.starts_with("wait ")
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        retained(before),
        retained(after),
        "all other arithmetic and cross-stream dependencies remain unchanged"
    );
}

/// Every AR decode graph runs the head's decode row tiles at its capacity,
/// and no other head route.
#[test]
fn ar_decode_heads_run_in_row_tiles() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 32);
    for slot in 0..32 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    let mut ends = [1u64; 32];
    let stem = "chop_deepseek41_decode_f32_n32320_k5120_rows";
    for (rows, tag) in [
        (1u32, "f96_r8_w32x8_s3_k64"),
        (2, "f96_r8_w32x8_s3_k64"),
        (4, "f96_r8_w32x8_s3_k64"),
        (8, "f96_r8_w32x8_s3_k64"),
        (16, "f96_r16_w32x16_s2_k64"),
        (32, "f256_r32_w32x32_s2_k64"),
    ] {
        let from = host.device.journal.borrow().len();
        let items = (0..rows)
            .map(|i| (u64::from(i) + 1, u64::from(i), ends[i as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        let lines = host.device.journal.borrow();
        let mut heads: Vec<_> = lines[from..]
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains("n32320_k5120"))
            .map(|line| line.split_whitespace().nth(2).unwrap().to_owned())
            .collect();
        heads.sort();
        let mut expected: Vec<_> = (0..4)
            .flat_map(|_| [format!("{stem}_prepare_v1"), format!("{stem}_{tag}_v1")])
            .collect();
        expected.sort();
        assert_eq!(heads, expected, "rows={rows}");
        for end in &mut ends[..rows as usize] {
            *end += 1;
        }
    }
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
