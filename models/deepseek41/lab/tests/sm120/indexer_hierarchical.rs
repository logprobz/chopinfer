//! Ordered hierarchical selection against the qualified guarded scorer.
use super::*;
use kernel_harness::eviction::EvictionMode;
use kernels_sm120::pooled_indexer::position_hierarchical as hierarchy;

pub(super) const MODULE: &str = "position_hierarchical_deepseek41_sm120";

#[test]
fn cpu_group_union_preserves_exact_keys_and_order() {
    fn retain(input: &[u64], k: usize) -> Vec<u64> {
        let mut ranked: Vec<_> = input.iter().copied().filter(|&v| v != 0).collect();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        let cutoff = ranked.get(k - 1).copied().unwrap_or(0);
        input
            .iter()
            .copied()
            .filter(|&v| v != 0 && v >= cutoff)
            .collect()
    }
    let mut seed = 0xa72683_u32;
    for count in [0_usize, 1, 31, 32, 64, 2048, 2049, 8192, 32768, 131072] {
        for k in [1, 128, 511, 512] {
            for ties in [false, true] {
                let keys: Vec<_> = (0..count)
                    .map(|i| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        (u64::from(if ties { 1 } else { seed }) << 32) | u64::from(!(i as u32))
                    })
                    .collect();
                let span = count.div_ceil(32).max(1);
                let locals: Vec<_> = keys.chunks(span).map(|s| retain(s, k)).collect();
                let groups: Vec<_> = locals
                    .chunks(8)
                    .flat_map(|g| retain(&g.concat(), k))
                    .collect();
                assert_eq!(retain(&keys, k), retain(&groups, k));
            }
        }
    }
    let keys: Vec<_> = (0_u32..16384)
        .map(|i| (1_u64 << 32) | u64::from(!i))
        .collect();
    let wrong: Vec<_> = keys
        .chunks(4096)
        .flat_map(|group| retain(group, 128))
        .collect();
    assert_ne!(
        retain(&keys, 512),
        retain(&wrong, 512),
        "under-retaining groups must fail"
    );
}

#[test]
fn cpu_edge_scores_keep_signed_zero_and_poisoned_tails() {
    for layer in [2, 20, 24] {
        let width = 131136 / FLASH.compress_ratio(layer);
        let extent = if layer == 24 { 16384 } else { width };
        for case in [21, 28, 30, 31] {
            let label = serde_json::json!({"rows":2,"layer":layer,"offset":131072,"case":case});
            let bytes = edge_scores(&label, u64::from(extent) * 8);
            let words: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if case == 21 {
                assert_eq!(&words[..2], &[0, 0x80000000]);
            }
            if case == 30 {
                assert!(words.iter().all(|&v| v == 0x7fc12345));
            }
            if case == 31 {
                let visible = 1 / FLASH.compress_ratio(layer);
                assert_eq!(
                    words.iter().filter(|&&v| v == 1_f32.to_bits()).count(),
                    visible as usize * 2
                );
            }
            if case == 28 {
                assert!(words.contains(&0x7fc12345));
            }
        }
    }
}
pub(super) fn top_k(case: u32) -> u32 {
    match case {
        21 => 128,
        22 => 1,
        23 => 511,
        _ => 512,
    }
}
fn adapt_calls(calls: &mut Vec<Launch>, intermediate: View) {
    let mut final_call = calls.pop().unwrap();
    assert!(
        final_call.symbol.ends_with("_live_merge_v1")
            || final_call.symbol.ends_with("_valid_merge_v1")
    );
    final_call
        .arguments
        .0
        .push(Argument::DeviceBuffer(intermediate.handle));
    let mut group = final_call.clone();
    group.symbol = hierarchy::symbol(&group.symbol, false);
    group.plan.grid[0] = hierarchy::GROUPS;
    final_call.symbol = hierarchy::symbol(&final_call.symbol, true);
    calls.extend([group, final_call]);
}
pub(super) fn adapt(arena: &mut Arena<'_>, calls: &mut Vec<Launch>, rows: u32) -> LabResult<View> {
    let intermediate = arena.output(hierarchy::workspace_bytes(rows, 512) as usize)?;
    adapt_calls(calls, intermediate);
    Ok(intermediate)
}

#[test]
fn cpu_hierarchy_keeps_original_keys_abi_and_scratch_disjoint() {
    use kernels_sm120::pooled_indexer::{position_live, position_valid};
    for entry in crate::indexer_entries::entries(&FLASH) {
        let old = position_live::dense_specs(&entry, false).unwrap().remove(2);
        for new in hierarchy::dense_specs(&entry).unwrap() {
            assert_eq!(new.params[..old.params.len()], old.params);
            assert_eq!(new.params.len(), old.params.len() + 1);
            assert!(
                new.body
                    .contains("chop::position_hierarchical::dense_select")
            );
        }
    }
    for entry in crate::indexer_entries::block_entries(&FLASH) {
        let old = position_valid::compact_specs(&entry, false)
            .unwrap()
            .remove(2);
        for new in hierarchy::compact_specs(&entry).unwrap() {
            assert_eq!(new.params[..old.params.len()], old.params);
            assert_eq!(new.params.len(), old.params.len() + 1);
            assert!(
                new.body
                    .contains("chop::position_hierarchical::compact_select")
            );
        }
    }
    for rows in [1, 2, 4, 6, 8, 16, 32, 8192] {
        let b = cpu_buffers(rows, 1, 131136);
        let query = indexer::selection(&FLASH, 20, rows).unwrap();
        for k in [1, 128, 511, 512] {
            let op = guard::guarded(true)
                .position_indexer(&query, k, 131136, Need::rows(rows))
                .unwrap();
            let old = op
                .launches(&b, v(99, op.info().workspace), Counts::rows(rows))
                .unwrap();
            let mut new = old.clone();
            let scratch = v(100, hierarchy::workspace_bytes(rows, 512));
            adapt_calls(&mut new, scratch);
            assert_eq!(old[..old.len() - 1], new[..new.len() - 2]);
            assert_eq!(new[new.len() - 2].plan.grid, [4, rows, 1]);
            assert_eq!(new.last().unwrap().plan, old.last().unwrap().plan);
            for call in &new[new.len() - 2..] {
                assert_eq!(
                    call.arguments.0[..call.arguments.0.len() - 1],
                    old.last().unwrap().arguments.0
                );
                assert_eq!(call.arguments.0.last(), Some(&Argument::DeviceBuffer(100)));
            }
        }
    }
}

fn edge_scores(label: &serde_json::Value, bytes: u64) -> Vec<u8> {
    let rows = label["rows"].as_u64().unwrap() as usize;
    let layer = label["layer"].as_u64().unwrap() as u32;
    let ratio = FLASH.compress_ratio(layer) as usize;
    let extent = bytes as usize / rows / 4;
    let offset = label["offset"].as_u64().unwrap() as usize;
    let case = label["case"].as_u64().unwrap();
    let mut seed = 0xa87312_u32;
    (0..rows * extent)
        .flat_map(|at| {
            let row = at / extent;
            let slot = at % extent;
            let count = match case {
                30 => 0,
                31 => 1 / ratio,
                _ => (offset + row + 1) / ratio,
            };
            let kept = count.div_ceil(8).min(2048);
            let index = if layer == 24 && kept > 0 && slot / 8 + 1 == kept {
                (count.div_ceil(8) - 1) * 8 + slot % 8
            } else {
                slot
            };
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let score = match case {
                20 => 1.0,
                21 => {
                    if slot % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                23 => {
                    if slot % 1024 == 1 {
                        f32::NAN
                    } else {
                        1.0
                    }
                }
                24 => {
                    if slot % 1024 == 1 {
                        f32::INFINITY
                    } else {
                        1.0
                    }
                }
                25 => f32::MAX,
                26 => f32::NEG_INFINITY,
                28 | 30 | 31 => {
                    if index >= count {
                        f32::from_bits(0x7fc12345)
                    } else {
                        1.0
                    }
                }
                29 => {
                    if (index / (64 / ratio)) % 7 == 0 {
                        f32::NAN
                    } else {
                        1.0
                    }
                }
                32 => -((slot % 103 + 1) as f32),
                _ => (seed as i32) as f32 / 65536.0,
            };
            score.to_le_bytes()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exercise<B>(
    session: &Session,
    arena: &mut Arena<'_>,
    b: &B,
    planes: [View; 3],
    op: Prepared<B>,
    rows: u32,
    label: serde_json::Value,
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
) -> LabResult<()> {
    let workspace = arena.output(op.info().workspace as usize)?;
    let mut old = op
        .launches(b, workspace, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    guard_fast::remap(&mut old);
    let mut new = old.clone();
    let intermediate = adapt(arena, &mut new, rows)?;
    for call in old.iter().chain(&new) {
        guard_fast::load(session, modules, &call.symbol)?;
    }
    let case = label["case"].as_u64().unwrap_or(0);
    let custom = (case >= 20).then(|| edge_scores(&label, planes[0].bytes));
    let mut baseline = None;
    for (arm, calls) in [&old, &new].into_iter().enumerate() {
        let start = if custom.is_some() { 2 } else { 0 };
        let graph = session.capture(|_| fire(session, arena, modules, &calls[start..]))?;
        for _ in 0..2 {
            arena.poison(&[planes[0], planes[1], planes[2], workspace, intermediate])?;
            if let Some(data) = &custom {
                arena.write(planes[0], data)?;
            }
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let output = [
                arena.read(planes[0])?,
                arena.read(planes[1])?,
                arena.read(planes[2])?,
            ];
            if let Some(expected) = &baseline {
                assert_eq!(&output, expected, "hierarchical arm={arm} {label}");
            } else {
                baseline = Some(output);
            }
        }
    }
    let mut negative_mismatches = 0;
    if case == 20 {
        let mut mutant = new.clone();
        let at = mutant.len() - 2;
        mutant[at].symbol.push_str("_negative_v1");
        arena.poison(&[planes[1], planes[2], workspace, intermediate])?;
        arena.write(planes[0], custom.as_ref().unwrap())?;
        fire(session, arena, modules, &mutant[2..])?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let observed = arena.read(planes[2])?;
        negative_mismatches = baseline.as_ref().unwrap()[2]
            .chunks_exact(4)
            .zip(observed.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            negative_mismatches > 0,
            "group-pruning negative control escaped"
        );
    }
    if matches!(case, 22 | 30 | 31) {
        assert!(
            arena.read(intermediate)?.iter().all(|&v| v == 0xa5),
            "small-list route read/wrote the intermediate"
        );
    }
    println!(
        "HIER_CASE {}",
        serde_json::json!({"shape":label,"mismatches":0,"negative_mismatches":negative_mismatches,
        "negative_control_rejected":case==20,"intermediate_bytes":intermediate.bytes,"old_launches":old.len(),"new_launches":new.len()})
    );
    if case == 0 {
        for stage in ["merge", "select", "all"] {
            let graphs = [&old, &new]
                .into_iter()
                .enumerate()
                .map(|(arm, calls)| {
                    let start = match stage {
                        "merge" => calls.len() - if arm == 0 { 1 } else { 2 },
                        "select" => 2,
                        _ => 0,
                    };
                    session.capture(|_| fire(session, arena, modules, &calls[start..]))
                })
                .collect::<LabResult<Vec<_>>>()?;
            let measured =
                kernel_harness::with_timing_label(&format!("hier-{label}-{stage}"), || {
                    kernel_harness::time_ab(
                        session,
                        2,
                        5,
                        [EvictionMode::Write; 2],
                        |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                    )
                })?;
            println!(
                "HIER_TIMING {}",
                serde_json::json!({"shape":label,"stage":stage,"exclusive":true,
                "median_ms":measured.arms.map(|a|a.median_ms),"paired_ratio":measured.paired_ratio})
            );
        }
    }
    session.receipt(
        "indexer-hierarchical",
        BTreeMap::from([
            ("shape".into(), label),
            ("mismatches".into(), serde_json::json!(0)),
        ]),
    )?;
    Ok(())
}

#[test]
#[ignore = "requires sealed captures and CHOPINFER_INDEX_GUARD_ARTIFACTS on idle GPUs"]
fn gpu_real_selections_and_cost() {
    guard::real_variant(true, true);
}
#[test]
#[ignore = "requires CHOPINFER_INDEX_GUARD_ARTIFACTS and idle GPUs"]
fn gpu_matrix_and_edges() {
    guard::matrix_variant(true, true);
}
#[test]
#[ignore = "requires sealed captures and CHOPINFER_INDEX_GUARD_ARTIFACTS on idle GPUs"]
fn gpu_poisoned_failures() {
    captured_comparisons_variant(guard::guarded, true, true);
}
