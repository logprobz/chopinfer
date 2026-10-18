//! Whole-machine installed-package DSpark checks. Build before the GPU job.
//! This fixture forces commit depths; it does not claim vendor acceptance parity.
use runtime_bundle_abi::*;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[path = "dspark_lifecycle/device.rs"]
mod device;
use device::Fixture;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    bundle: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
}

fn maximum(row: &[u8]) -> Result<u32> {
    if row.is_empty() || !row.len().is_multiple_of(4) {
        return Err("invalid F32 logits row".into());
    }
    let mut best = (0, f32::NEG_INFINITY);
    for (token, word) in row.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(word.try_into()?);
        if !value.is_finite() {
            return Err(format!("non-finite logit at token {token}").into());
        }
        if value > best.1 {
            best = (token as u32, value);
        }
    }
    Ok(best.0)
}

fn same(config: &Config, label: &str, actual: &[Vec<u8>], expected: &[Vec<u8>]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: row counts differ").into());
    }
    for (row, (a, b)) in actual.iter().zip(expected).enumerate() {
        if a != b {
            std::fs::write(config.output.join(format!("{label}-{row}-actual.bin")), a)?;
            std::fs::write(config.output.join(format!("{label}-{row}-expected.bin")), b)?;
            return Err(format!("{label}: logits row {row} differs bitwise").into());
        }
    }
    Ok(())
}

fn gpu_guard() -> Result<()> {
    let selected = std::env::var("CHOPINFER_KERNEL_GPUS")?;
    let ordinals = selected.split(',').map(str::trim).collect::<Vec<_>>();
    if ordinals != ["0", "1", "2", "3"] {
        return Err("this test requires the job's four GPUs, numbered 0,1,2,3".into());
    }
    Ok(())
}

#[test]
#[ignore = "requires an installed checkpoint package and four idle GPUs"]
fn dspark_gpu_lifecycle() -> Result<()> {
    gpu_guard()?;
    let config: Config = serde_json::from_reader(std::fs::File::open(std::env::var(
        "CHOPINFER_DEEPSEEK41_DSPARK_TEST",
    )?)?)?;
    std::fs::create_dir(&config.output)?;
    let mut fixture = Fixture::new(&config)?;
    // Valid text IDs cross the main ring wrap and are independent of a
    // frontend renderer. Greedy serving prompts run separately in the job.
    let prompt = (100..241).collect::<Vec<_>>();
    let source = fixture.sequence()?;
    fixture.execute(source, 1, 0, &prompt, true, 0)?;
    fixture.commit(prompt.len())?;
    let pages = fixture.capture(1, prompt.len())?;
    // Keep the source KV pages pinned while preempting its bundle slot.
    fixture.instance.retire_request(1)?;
    let child = fixture.restore(source, 2, 0, &prompt, &pages)?;
    let full = fixture.execute(child, 2, 0, &[7], false, 5)?;
    fixture.commit(6)?;
    fixture.retire(2, child)?;
    let natural_accepted = full
        .drafts
        .iter()
        .zip(&full.rows)
        .map(|(&token, row)| Ok(token == maximum(row)?))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .take_while(|&matched| matched)
        .count();
    std::fs::write(
        config.output.join("smoke.json"),
        serde_json::to_vec_pretty(&json!({
            "passed":true,"prompt_rows":prompt.len(),"draft_ids":full.drafts,
            "verify_rows":full.rows.len(),"greedy_acceptance":natural_accepted,
        }))?,
    )?;
    eprintln!(
        "DSpark smoke passed: create, wrapped prefill, snapshot restore, five drafts, six verify rows and commit"
    );
    let mut budgets = Vec::new();
    for budget in 1..=4 {
        let request = 10 + u64::from(budget);
        let child = fixture.restore(source, request, 0, &prompt, &pages)?;
        let capture = fixture.execute(child, request, 0, &[7], false, budget)?;
        if capture.drafts != full.drafts[..budget as usize] {
            return Err(format!("K={budget}: draft prefix differs").into());
        }
        same(
            &config,
            &format!("budget-{budget}"),
            &capture.rows,
            &full.rows[..budget as usize + 1],
        )?;
        fixture.commit(1)?;
        fixture.retire(request, child)?;
        budgets.push(budget);
    }
    for accepted in 0..=5 {
        let request = 20 + accepted as u64;
        let child = fixture.restore(source, request, 0, &prompt, &pages)?;
        let capture = fixture.execute(child, request, 0, &[7], false, 5)?;
        fixture.commit(accepted + 1)?;
        let mut committed = vec![7];
        committed.extend_from_slice(&capture.drafts[..accepted]);
        let mut last = Vec::new();
        for token in [11, 12] {
            last = fixture.execute(child, request, 0, &[token], false, 0)?.rows;
            fixture.commit(1)?;
            committed.push(token);
        }
        // The next drafted step must catch up every accepted and AR row.
        let resumed = fixture.execute(child, request, 0, &[13], false, 5)?;
        fixture.cancel()?;
        fixture.retire(request, child)?;
        let reference = fixture.restore(source, 40, 1, &prompt, &pages)?;
        let mut reference_last = Vec::new();
        for token in committed {
            reference_last = fixture.execute(reference, 40, 1, &[token], false, 0)?.rows;
            fixture.commit(1)?;
        }
        same(
            &config,
            &format!("depth-{accepted}-ar"),
            &last,
            &reference_last,
        )?;
        let ar_drafted = fixture.execute(reference, 40, 1, &[13], false, 5)?;
        if resumed.drafts != ar_drafted.drafts {
            return Err(format!("depth {accepted}: catch-up draft differs").into());
        }
        same(
            &config,
            &format!("depth-{accepted}-catch-up"),
            &resumed.rows,
            &ar_drafted.rows,
        )?;
        fixture.commit(1)?;
        fixture.retire(40, reference)?;
        eprintln!(
            "DSpark depth {accepted}: partial commit, K=0 catch-up, cancellation and replay match"
        );
    }
    // Reuse the cancelled slot for another prompt, then compare an unused slot.
    let mut reused = Vec::new();
    for slot in [0, 3] {
        let request = 50 + slot;
        let handle = fixture.sequence()?;
        fixture.execute(handle, request, slot, &[211, 212, 213], true, 0)?;
        fixture.commit(3)?;
        let capture = fixture.execute(handle, request, slot, &[7], false, 5)?;
        if slot == 0 {
            reused = capture.rows
        } else {
            same(&config, "slot-reuse", &capture.rows, &reused)?
        }
        fixture.cancel()?;
        fixture.retire(request, handle)?;
    }
    fixture.finish(source)?;
    std::fs::write(
        config.output.join("receipt.json"),
        serde_json::to_vec_pretty(&json!({
            "schema":"chopinfer.deepseek41.dspark-lifecycle.v1", "passed":true,
            "geometry":"flash", "tensor_parallel":4, "forced_acceptance_depths":[0,1,2,3,4,5],
            "budgets_against_full_block":budgets,"k_zero_intervals":true,"cancellation":true,
            "prefix_restore":true,"preemption":true,"ring_wrap":true,"slot_reuse":true,
            "comparison":"bitwise F32 target logits and exact draft IDs", "live_resources":0,
            "greedy_acceptance":natural_accepted,
        }))?,
    )?;
    Ok(())
}

#[test]
fn greedy_comparison_refuses_nonfinite_rows_and_breaks_ties_by_id() {
    let row = [1_f32, 3.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(maximum(&row).unwrap(), 1);
    assert!(maximum(&f32::NAN.to_le_bytes()).is_err());
    assert!(maximum(&[]).is_err());
}
