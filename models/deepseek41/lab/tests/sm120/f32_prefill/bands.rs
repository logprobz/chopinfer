//! Packed-row sweeps of existing feature tiles against the baseline kernel.
use super::*;
use model_deepseek41_contract::geometry::FLASH;

#[path = "../prefill_sweep_rows.rs"]
mod coverage;

const SHAPES: [(u32, u32); 4] = [(512, 5120), (32, 5120), (128, 512), (384, 5120)];
const TIMED: [u32; 12] = [
    257, 319, 446, 789, 1023, 1024, 1803, 4095, 4096, 4542, 7754, 8192,
];
const ROUTER_TIMED: [u32; 12] = [257, 261, 266, 271, 277, 282, 288, 293, 298, 304, 309, 319];

fn rows(n: u32, k: u32, timed: bool) -> Vec<u32> {
    if (n, k) == (384, 5120) {
        if timed {
            ROUTER_TIMED.to_vec()
        } else {
            (257..=319).collect()
        }
    } else if timed {
        TIMED.to_vec()
    } else {
        let mut all = coverage::rows(257);
        all.extend(TIMED);
        all.sort_unstable();
        all.dedup();
        all
    }
}

fn entry(n: u32, k: u32) -> Entry {
    registration::baseline_entries(&FLASH)
        .unwrap()
        .into_iter()
        .find(|e| (e.query().n, e.query().k) == (n, k))
        .expect("declared band shape")
}

fn operands(n: u32, k: u32, outliers: bool) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut b) = random_operands(8192, n, k, 123_457);
    for (index, row) in a.chunks_exact_mut(k as usize).enumerate() {
        if index % 61 == 11 {
            row.fill(0.0);
            continue;
        }
        if outliers {
            if index % 41 == 9 && index % 37 != 3 && index % 53 != 7 {
                for value in row.iter_mut().skip(1).step_by(2) {
                    *value *= 2f32.powi(-40);
                }
            }
            if index % 37 == 3 {
                row[index % k as usize] = 1e15;
            }
            if index % 53 == 7 {
                row[(index * 7) % k as usize] = f32::from_bits(0x7fc1_2345);
            }
            if index % 47 == 13 {
                row[k as usize - 1] = f32::from_bits(1);
            }
        }
    }
    if outliers {
        for column in n / 2..(n / 2 + 16).min(n) {
            for value in &mut b[(column * k) as usize..((column + 1) * k) as usize] {
                *value *= 1e-20;
            }
        }
    }
    (a, b)
}

#[test]
fn cpu_band_sweep_covers_shapes_boundaries_and_fallbacks() {
    let mut cases = Vec::new();
    for (n, k) in SHAPES {
        let old = entry(n, k);
        assert!(!old.uses_cached_weights());
        let checked = rows(n, k, false);
        let timed = rows(n, k, true);
        assert_eq!(timed.len(), 12);
        assert!(timed.iter().all(|r| checked.contains(r)));
        if n == 384 {
            assert_eq!(checked, (257..=319).collect::<Vec<_>>());
        } else {
            assert!(checked.len() > 1000);
        }
        for tile in [32, 64] {
            let candidate = old.clone().with_prefill_tiles(tile).unwrap();
            assert!(fold_cubin_for_symbol(candidate.symbol()).is_some());
            for count in [257, 319, 789, 1023, 1024, 4095, 4096, 8192] {
                let op = prepared(&candidate, count);
                assert_eq!(
                    op.info().workspace,
                    candidate.prefill_workspace_bytes(count)
                );
            }
        }
        cases.push(serde_json::json!({"n":n,"k":k,"rows":checked,"timed":timed,"patterns":["general","outliers"],"tiles":[32,64],"comparisons_per_case":8}));
    }
    println!(
        "F32_BAND_MATRIX {}",
        serde_json::json!({"shapes":cases,"bands":[[257,1023],[1024,4095],[4096,8192]],"router_band":[257,319],"winning_ratio":0.97,"baseline":"unchanged_simt"})
    );
}

#[test]
#[ignore = "requires staged F32 cubins, CHOPINFER_F32_BAND_SHAPE and an assigned idle GPU"]
fn gpu_band_sweep() {
    let shape = std::env::var("CHOPINFER_F32_BAND_SHAPE").expect("band shape");
    let (n, k) = shape.split_once('x').expect("n x k");
    let (n, k) = (n.parse::<u32>().unwrap(), k.parse::<u32>().unwrap());
    assert!(SHAPES.contains(&(n, k)));
    let worker = Shard::from_env().unwrap();
    assert_eq!(worker.count, 2);
    let tile = [32, 64][worker.ordinal];
    let phase = std::env::var("CHOPINFER_PREFILL_SWEEP_PHASE").expect("phase");
    assert!(["check", "time"].contains(&phase.as_str()));
    let timed = phase == "time";
    kernel_harness::gpu_test_with(timing_cubins(), 4 << 30, |session| -> LabResult<()> {
        check_poison_expansion(session)?;
        let old = entry(n, k);
        let candidate = std::fs::read(artifacts().join(format!("{}.cubin", registration::cubin_name(&FLASH, tile)))).map_err(|e| e.to_string())?;
        let baseline = fold_cubin_for_symbol(old.symbol()).ok_or("missing baseline")?.1;
        let selected = rows(n, k, timed);
        let patterns: &[bool] = if timed { &[false] } else { &[false, true] };
        let mut count = 0;
        for &outliers in patterns {
            let (a, b) = operands(n, k, outliers);
            let weights = ComparisonWeights::prefill(session, &old, tile, &b, [baseline, &candidate])?;
            let pattern = if outliers { "outliers" } else { "general" };
            for &r in &selected {
                let case = HostCase::new(format!("band_n{n}_k{k}_tile{tile}_{pattern}"));
                compare(session, &old, r, &a[..(r * k) as usize], &weights, &case, timed)?;
                count += 1;
                println!("F32_BAND_BITS {}", serde_json::json!({"n":n,"k":k,"worker":worker.ordinal,"tile":tile,"phase":phase,"pattern":pattern,"rows":r,"comparisons":8,"whole_output":true,"bitwise":true}));
            }
        }
        assert_eq!(count, selected.len() * patterns.len());
        println!("F32_BAND_DONE {}", serde_json::json!({"n":n,"k":k,"worker":worker.ordinal,"tile":tile,"phase":phase,"cases":count,"comparisons":8*count,"bitwise":true}));
        Ok(())
    }).unwrap().expect("band sweep GPU comparison must execute");
}
