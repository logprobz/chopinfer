//! Reconstruct each changed launch from the qualified exact-row operation.
use super::*;
use crate::ops::{Bind, Lowered, dense, exact};
use bundle_runtime::dispatch::{arguments, launch_config};
use kernel_core::backend::Counts;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;

fn span((raw, offset, length): (u64, u64, u64)) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(raw), offset, length)
}
fn line(stream: &str, call: &Lowered) -> String {
    let launch = &call.launch;
    assert!(!call.collective);
    format!(
        "launch {stream} {} {:?} {:?}",
        launch.symbol,
        launch_config(launch.plan).unwrap(),
        arguments(&launch.arguments, &launch.bindings).unwrap()
    )
}
fn audit(before: &[String], after: &[String], ranks: Ranks) -> Result<usize, String> {
    // Verify fragments have no acquisition records and must remain identical.
    if before == after {
        return Ok(0);
    }
    if ranks != Ranks::TP4 {
        return if before == after {
            Ok(0)
        } else {
            Err("unqualified topology changed".into())
        };
    }
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let mut workspaces = BTreeMap::<String, DeviceBufferSpanV1>::new();
    for row in before
        .iter()
        .filter(|l| l.starts_with("launch ") && l.contains("_cached_rows_v1 "))
    {
        let stream = row.split_whitespace().nth(1).unwrap();
        let view = span(
            *shared_overlap::journal_spans(row)
                .last()
                .ok_or("no workspace")?,
        );
        if let Some(old) = workspaces.get_mut(stream) {
            if (old.buffer, old.offset) != (view.buffer, view.offset) {
                return Err("workspace owner changed".into());
            }
            old.length = old.length.max(view.length);
        } else {
            workspaces.insert(stream.into(), view);
        }
    }
    let asks = crate::ops::Asks::default();
    let _ = dense::DenseOps::prepare(kernels.backend.as_ref(), &shape, &asks).unwrap();
    let shared_bytes = asks
        .asked()
        .iter()
        .filter(|a| {
            a.what.starts_with("dense.moe.shared-") || a.what.starts_with("dense.shared-pair")
        })
        .map(|a| a.workspace)
        .max()
        .unwrap();
    if !before
        .iter()
        .any(|row| row == &format!("allocate {shared_bytes} 256"))
    {
        return Err("shared workspace allocation is absent".into());
    }
    let mut shared = BTreeMap::new();
    for row in before.iter().filter(|row| {
        row.starts_with("launch ")
            && row.contains("_n576_k5120_g1_pair_")
            && row.contains("_initialize_v3 ")
    }) {
        let spans = shared_overlap::journal_spans(row);
        if spans.len() != 5 {
            return Err("shared admission operands changed".into());
        }
        let mut workspace = span(spans[2]);
        workspace.length = shared_bytes;
        if let Some(old) = shared.insert(spans[4].0, workspace)
            && old != workspace
        {
            return Err("shared workspace owner changed".into());
        }
    }
    let mut expected = Vec::new();
    let mut changed = 0;
    for row in before {
        let words = row.split_whitespace().collect::<Vec<_>>();
        let projection = (words.first() == Some(&"launch"))
            .then(|| {
                dense::projections(&shape).into_iter().find(|d| {
                    let q = d.query;
                    words[2]
                        == format!(
                            "chop_deepseek41_mxfp8_fold_m8192_n{}_k{}_g{}_f32_v2",
                            q.n, q.k, q.groups
                        )
                })
            })
            .flatten();
        let Some(projection) = projection else {
            expected.push(row.clone());
            continue;
        };
        let views = shared_overlap::journal_spans(row)
            .into_iter()
            .map(span)
            .collect::<Vec<_>>();
        if views.len() != 7 {
            return Err("legacy fold operands changed".into());
        }
        let rows = views[0].length / u64::from(projection.query.activation_row_stride);
        if !(1..=u64::from(dense::short_prefill::MAX_ROWS)).contains(&rows) {
            expected.push(row.clone());
            continue;
        }
        let rows = rows as u32;
        let workspace = if projection.id.starts_with("moe.shared-") {
            *shared
                .get(&views[6].buffer.raw)
                .ok_or("missing shared workspace")?
        } else {
            *workspaces
                .get(words[1])
                .ok_or("missing declared workspace")?
        };
        let prepare = |small| -> Result<Vec<Lowered>, String> {
            let q = if small {
                crate::Kernels::dense_decode_query(projection.query, rows)
            } else {
                projection.query
            };
            let op = kernels
                .backend
                .mxfp8_fold_gemm(&q, exact::need(q.m_capacity, exact::every_phase(rows)))
                .map_err(|e| e.to_string())?;
            let mut bind = Bind::default();
            let b = Mxfp8FoldGemmBuffers {
                activations: bind.view(views[0]),
                activation_scales: bind.view(views[1]),
                weights: bind.view(views[2]),
                weight_scales: bind.view(views[3]),
                output: bind.view(views[4]),
                live_rows: bind.view(views[5]),
                status: bind.view(views[6]),
                status_row_offset: 0,
            };
            bind.launch_within(&op, &b, workspace, Counts::rows(rows))
                .map_err(|e| e.to_string())
        };
        let old = prepare(false)?;
        if old.len() != 1 || line(words[1], &old[0]) != *row {
            return Err("legacy launch contract changed".into());
        }
        for call in prepare(true)? {
            expected.push(line(words[1], &call));
        }
        changed += 1;
    }
    if expected != after {
        let first = expected
            .iter()
            .zip(after)
            .position(|(a, b)| a != b)
            .unwrap_or(expected.len().min(after.len()));
        return Err(format!(
            "unexpected journal change at {first}: {:?} / {:?}",
            expected.get(first),
            after.get(first)
        ));
    }
    Ok(changed)
}

#[test]
#[ignore = "requires staged before/after journals in CHOPINFER_SHORT_PREFILL_JOURNALS"]
fn short_prefill_journals_preserve_every_other_operation() {
    let root =
        std::path::PathBuf::from(std::env::var_os("CHOPINFER_SHORT_PREFILL_JOURNALS").unwrap());
    let mut checked = 0;
    for dir in ["lifecycle", "lifecycle/exact", "verify"] {
        for item in std::fs::read_dir(root.join("before").join(dir)).unwrap() {
            let path = item.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("journal") {
                continue;
            }
            let name = path.file_name().unwrap().to_str().unwrap();
            let a = std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let b = std::fs::read_to_string(root.join("after").join(dir).join(name))
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let ranks = if name.contains("tp4") {
                Ranks::TP4
            } else {
                Ranks::TP2
            };
            let count = audit(&a, &b, ranks).unwrap();
            if name.contains("tp4") && dir != "verify" {
                assert!(count > 0);
            }
            let mut bad = b.clone();
            let at = bad.iter().position(|s| s.starts_with("launch ")).unwrap();
            bad[at] = bad[at].replacen("offset: ", "offset: 1", 1);
            assert!(audit(&a, &bad, ranks).is_err());
            eprintln!("SHORT_PREFILL_AUDIT {dir}/{name} replacements={count}");
            checked += 1;
        }
    }
    assert_eq!(checked, 14);
}
