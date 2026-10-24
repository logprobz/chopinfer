//! Every candidate row is compared with the unchanged baseline SIMT kernel.
use super::*;
const K: u32 = 5120;
const N: u32 = 384;
const CAPACITY: u32 = 8192;

#[path = "../prefill_sweep_rows.rs"]
mod coverage;
#[path = "../router_norm/fixtures.rs"]
mod fixtures;

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        0x7fc0
    } else {
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
}
fn widen(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

fn run(s: &Session) -> LabResult<()> {
    if n() != N as usize {
        return Err("router row sweep requires 384 features".into());
    }
    let worker: usize = std::env::var("CHOPINFER_PROJECTION_ZERO_WORKER")
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|_| "worker")?;
    if worker > 1 {
        return Err("router row sweep requires one of two workers".into());
    }
    let phase = std::env::var("CHOPINFER_PREFILL_SWEEP_PHASE").map_err(|e| e.to_string())?;
    if !["check", "time"].contains(&phase.as_str()) {
        return Err("row sweep phase must be check or time".into());
    }
    let timed = phase == "time";
    let pattern = if !timed && worker == 1 {
        fixtures::Pattern::Outliers
    } else {
        fixtures::Pattern::Clean
    };
    let f = Functions::load(s)?;
    let bytes = |v: Vec<u16>| v.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let x = data::normalized(
        s,
        &f,
        &bytes(fixtures::input_rows(pattern)),
        &bytes(fixtures::norm_weight()),
        model_deepseek41_contract::rows::norm_quantizers(&FLASH)[0]
            .norm
            .epsilon_bits,
    )?;
    let weights = fixtures::router_weight();
    let rows = if timed {
        coverage::TIMED.to_vec()
    } else {
        coverage::rows(320)
    };
    for &count in &rows {
        compare_with_reference(
            s,
            &f,
            &x[..count as usize * K as usize],
            &weights,
            count,
            32,
            &format!("row_sweep_{pattern:?}"),
            timed,
            family::entry(),
        )?;
        println!(
            "PROJECTION_ZERO_SWEEP_BITS {}",
            json!({
                "worker":worker,"phase":phase,"rows":count,"pattern":format!("{pattern:?}"),
                "reference":family::entry().symbol(),"comparisons":8,"whole_output":true,"bitwise":true
            })
        );
    }
    if !timed {
        for mutation in ["omit_chain", "force_certificate"] {
            negative(s, &f, mutation)?;
        }
    }
    println!(
        "PROJECTION_ZERO_SWEEP_DONE {}",
        json!({
            "worker":worker,"phase":phase,"cases":rows.len(),"comparisons":rows.len()*8,
            "negative":if timed {0} else {2},"bitwise":true
        })
    );
    Ok(())
}

#[test]
fn row_sweep_uses_the_baseline_and_declares_its_exact_coverage() {
    let counts = coverage::rows(320);
    assert!(!family::entry().uses_prefill_tiles());
    assert!(!family::entry().uses_cached_weights());
    assert_eq!(coverage::TIMED.len(), 12);
    println!(
        "PROJECTION_ZERO_SWEEP_DRY {}",
        json!({
            "rows":counts,"timed":coverage::TIMED,"workers":2,"patterns":["Clean","Outliers"],
            "comparisons_per_case":8,"negative_per_worker":2,"reference":"baseline_simt"
        })
    );
}

#[test]
#[ignore = "requires CHOPINFER_PROJECTION_ZERO_ROOT and an assigned idle GPU"]
fn projection_row_sweep_gpu() {
    gpu_run(run);
}
