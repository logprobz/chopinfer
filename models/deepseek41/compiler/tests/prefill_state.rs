//! Raw long-context prefill and snapshot comparison for an installed package.
use runtime_bundle_abi::*;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[path = "prefill_state/device.rs"]
mod device;
use device::Fixture;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    bundle: PathBuf,
    checkpoint: PathBuf,
    tokens: PathBuf,
    output: PathBuf,
    reference: Option<PathBuf>,
}

fn record(config: &Config, label: &str, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    if bytes.is_empty() {
        return Err(format!("{label}: empty output").into());
    }
    let name = format!("{label}.bin");
    std::fs::File::create_new(config.output.join(&name))?.write_all(bytes)?;
    if let Some(reference) = &config.reference {
        let expected = std::fs::read(reference.join(name))?;
        if bytes != expected {
            let first = bytes.iter().zip(&expected).position(|(a, b)| a != b);
            return Err(format!(
                "{label}: bytes differ at {first:?}, lengths {} and {}",
                bytes.len(),
                expected.len()
            )
            .into());
        }
    }
    Ok(())
}

fn logits(config: &Config, label: &str, capture: device::Capture) -> Result<()> {
    let width = model_deepseek41_contract::geometry::FLASH.shape.vocab as usize;
    if !capture.drafts.is_empty() || capture.rows.len() != 1 || capture.rows[0].len() != 4 * width {
        return Err(format!("{label}: incomplete AR vocabulary row").into());
    }
    for word in capture.rows[0].chunks_exact(4) {
        if !f32::from_le_bytes(word.try_into()?).is_finite() {
            return Err(format!("{label}: non-finite F32 logit").into());
        }
    }
    record(config, label, &capture.rows[0])
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_PREFILL_STATE_TEST and four idle GPUs"]
fn prefill_state_gpu() -> Result<()> {
    if std::env::var("CHOPINFER_KERNEL_GPUS")? != "0,1,2,3" {
        return Err("the fixture requires four exclusive GPUs numbered 0,1,2,3".into());
    }
    let config: Config = serde_json::from_reader(std::fs::File::open(std::env::var(
        "CHOPINFER_DEEPSEEK41_PREFILL_STATE_TEST",
    )?)?)?;
    let ids: Vec<u32> = serde_json::from_reader(std::fs::File::open(&config.tokens)?)?;
    let width = model_deepseek41_contract::geometry::FLASH.shape.vocab;
    if ids.len() != 32768 || ids.iter().any(|&id| id >= width) {
        return Err("fixture needs exactly 32768 vocabulary token IDs".into());
    }
    std::fs::create_dir(&config.output)?;
    let mut fixture = Fixture::new(&config)?;
    let source = fixture.sequence()?;
    let mut snapshots = Vec::new();
    for start in [0, 8192, 16384] {
        let end = start + 8192;
        logits(
            &config,
            &format!("prefill-{end}"),
            fixture.execute(source, 1, 0, &ids[start..end], true, 0)?,
        )?;
        fixture.commit(8192)?;
        if end == 8192 {
            let pages = fixture.capture(1, end, 0)?;
            fixture.record_state(&config, "prefix-8192", &pages)?;
            let child = fixture.restore(source, 2, 1, &ids[..end], &pages)?;
            logits(
                &config,
                "restored-8193",
                fixture.execute(child, 2, 1, &[7], false, 0)?,
            )?;
            fixture.commit(1)?;
            fixture.retire(2, child)?;
            snapshots.push(end);
        }
    }
    let pages = fixture.capture(1, 24576, 0)?;
    fixture.record_state(&config, "prefix-24576", &pages)?;
    snapshots.push(24576);
    let odd = fixture.restore(source, 3, 1, &ids[..24576], &pages)?;
    logits(
        &config,
        "prefill-32767",
        fixture.execute(odd, 3, 1, &ids[24576..32767], true, 0)?,
    )?;
    fixture.commit(8191)?;
    let odd_pages = fixture.capture(3, 32767, 1)?;
    fixture.record_state(&config, "prefix-32767", &odd_pages)?;
    snapshots.push(32767);
    let odd_child = fixture.restore(odd, 4, 2, &ids[..32767], &odd_pages)?;
    logits(
        &config,
        "restored-32768",
        fixture.execute(odd_child, 4, 2, &[7], false, 0)?,
    )?;
    fixture.commit(1)?;
    fixture.retire(4, odd_child)?;
    fixture.retire(3, odd)?;
    logits(
        &config,
        "prefill-32768",
        fixture.execute(source, 1, 0, &ids[24576..], true, 0)?,
    )?;
    fixture.commit(8192)?;
    let pages = fixture.capture(1, 32768, 0)?;
    fixture.record_state(&config, "prefix-32768", &pages)?;
    snapshots.push(32768);
    let full_child = fixture.restore(source, 5, 1, &ids, &pages)?;
    logits(
        &config,
        "restored-32769",
        fixture.execute(full_child, 5, 1, &[7], false, 0)?,
    )?;
    fixture.commit(1)?;
    fixture.retire(5, full_child)?;
    fixture.finish(source)?;
    let mut files = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(&config.output)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "bin") {
            files.insert(
                entry.file_name().to_string_lossy().into_owned(),
                entry.metadata()?.len(),
            );
        }
    }
    if files.len() != 40 || files.values().any(|&bytes| bytes == 0) {
        return Err("incomplete snapshot and logits coverage".into());
    }
    std::fs::write(
        config.output.join("receipt.json"),
        serde_json::to_vec_pretty(&json!({
            "passed":true, "negotiated_chunk":8192, "snapshot_tokens":snapshots, "logits_rows":8, "vocabulary":width,
            "rank_snapshots":16, "rank_kv_pages":16, "files":files, "live_resources":0,
            "reference":config.reference, "mode":std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP")?,
            "dma":std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_DMA").ok(),
        }))?,
    )?;
    Ok(())
}

#[test]
fn raw_comparison_rejects_changed_missing_and_empty_bytes() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("prefill-state-comparison-{unique}"));
    std::fs::create_dir(&root).unwrap();
    let config = |name: &str, reference: Option<PathBuf>| {
        let output = root.join(name);
        std::fs::create_dir(&output).unwrap();
        Config {
            bundle: PathBuf::new(),
            checkpoint: PathBuf::new(),
            tokens: PathBuf::new(),
            output,
            reference,
        }
    };
    let base = config("base", None);
    record(&base, "row", &(-0.0_f32).to_le_bytes()).unwrap();
    let equal = config("equal", Some(base.output.clone()));
    record(&equal, "row", &(-0.0_f32).to_le_bytes()).unwrap();
    let different = config("different", Some(base.output.clone()));
    assert!(record(&different, "row", &0.0_f32.to_le_bytes()).is_err());
    assert!(record(&different, "missing", &[1]).is_err());
    assert!(record(&different, "empty", &[]).is_err());
    assert!(record(&equal, "row", &(-0.0_f32).to_le_bytes()).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
