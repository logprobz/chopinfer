//! Compare routed and reference journals with the same loaded module catalog.
use super::*;

fn projection(line: &str) -> bool {
    line.starts_with("launch ")
        && (line.contains("chop_deepseek41_mxfp8_")
            || line.contains("chop_deepseek41_dense_decode_")
            || line.contains("head_f32_") && line.contains("_decode_v1"))
}
fn run_case(reference: bool) -> Vec<String> {
    crate::sm120::head_last::with_reference(|| run_without_last(reference))
}
/// Decode and verify heads run the head's row tiles, so the reuse head
/// serves the prefill heads of 9 to 32 rows: prefills of one to 32
/// one-token prompts.
fn run_without_last(reference: bool) -> Vec<String> {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 32);
    let mut records = Vec::new();
    let mut first = 1;
    for rows in [1u32, 8, 9, 16, 24, 32] {
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &super::head_last::prompts(first, rows),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[from..].to_vec();
        let changed: Vec<_> = lines
            .iter()
            .filter(|l| {
                l.starts_with("launch ") && (l.contains("_reuse32_") || l.contains("r2s3w0f0"))
            })
            .collect();
        if reference || rows <= 8 {
            assert!(changed.is_empty(), "rows={rows}");
        } else {
            assert_eq!(
                changed.iter().filter(|l| l.contains("_head_f32_")).count(),
                4,
                "rows={rows}: one reuse head per rank"
            );
        }
        records.push(format!("case rows={rows}"));
        records.extend(lines);
        for request in first..first + u64::from(rows) {
            instance.retire_request(request).unwrap();
        }
        first += u64::from(rows);
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_PROJECTION_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = if reference {
            "projection-reference.journal"
        } else {
            "projection-serving.journal"
        };
        std::fs::write(dir.join(name), records.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, name);
    }
    records
}

#[test]
fn reuse_preserves_non_projection_launches_and_bindings() {
    let before = crate::sm120::projection_reuse::with_reference(|| run_case(true));
    let after = run_case(false);
    let retained = |lines: Vec<String>| {
        let mut all = false;
        let mut kept = Vec::new();
        for line in lines {
            if let Some(case) = line.strip_prefix("case rows=") {
                all = case
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
                    <= 8;
                kept.push(line);
            } else if line.starts_with("launch ") && (all || !projection(&line)) {
                kept.push(line);
            }
        }
        kept
    };
    assert_eq!(
        retained(before),
        retained(after),
        "all other kernels, head classifiers, reducers and their spans remain exact"
    );
}

#[test]
fn modules_retain_the_gpu_qualified_bytes() {
    let name = "f32_decode_reuse_deepseek41_sm120";
    let expected = "5822024aceddb2e6064580c326b8281518289b2884cbf93b07200bc8982727de";
    let module = crate::CUBINS.iter().find(|c| c.name == name).unwrap();
    assert_eq!(kernel_build::digest(module.bytes), expected);
}

#[test]
fn reuse_modules_load_only_for_flash_tp4() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        assert_eq!(
            crate::Kernels::loads(crate::sm120::projection_reuse::HEAD_CUBIN, &shape),
            ranks == Ranks::TP4
        );
    }
}
