//! Opt-in comparisons against frozen production cubins, with guarded capacity.
use super::*;
use sha2::{Digest, Sha256};

pub(super) const CAPACITY: usize = 8192;
// Full then partial launches reuse the same allocations. The 2047-token hot
// cases also give persistent CTAs full items followed by partial items.
pub(super) const TOKENS: [usize; 12] = [2048, 2047, 1025, 1024, 1021, 129, 128, 127, 17, 16, 15, 1];
const SENTINEL: u16 = 0x7fa5;

pub(super) fn enabled() -> bool {
    std::env::var_os("CHOPINFER_MOE_BOUNDARY_BASELINE").is_some()
}

pub(super) fn baseline(
    session: &Session,
    v: &Variant,
    counts: RuntimeCounts,
    expected: &str,
) -> LabResult<(Function, Value)> {
    let name = moe::cubin_specs(kernel_lab::catalogs(), Some(v.profile))
        .into_iter()
        .find(|spec| spec.entries.iter().any(|entry| entry.symbol == v.symbol))
        .ok_or("missing boundary cubin")?
        .name;
    let root = PathBuf::from(std::env::var("CHOPINFER_MOE_BOUNDARY_BASELINE").map_err(err)?);
    let bytes = fs::read(root.join(format!("{name}.cubin"))).map_err(err)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if hash != expected {
        return Err(format!("{name}: baseline cubin {hash} != {expected}"));
    }
    // The loader may read an external frozen directory. Require those bytes
    // to equal this executable's embedded production artifact.
    let candidate_root = PathBuf::from(std::env::var("CHOPINFER_MOE_CUBIN_DIR").map_err(err)?);
    let candidate = fs::read(candidate_root.join(format!("{name}.cubin"))).map_err(err)?;
    if candidate != kernel_lab::cubin(&name)? {
        return Err(format!(
            "{name}: external candidate differs from embedded cubin"
        ));
    }
    let function = session
        .context
        .load_module(&bytes)
        .map_err(err)?
        .function(&v.symbol)
        .map_err(err)?;
    let plan = v.plan(counts).map_err(err)?;
    if plan.grid != [188, 1, 1] {
        return Err("boundary fixture requires the production 188-CTA grid".into());
    }
    function
        .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
        .map_err(err)?;
    Ok((
        function,
        json!({"name": name, "baseline_sha256": hash,
        "candidate_sha256": format!("{:x}", Sha256::digest(&candidate))}),
    ))
}

/// Fills every capacity element with the NaN sentinel.
pub(super) fn poison(session: &Session, buffers: &[&DeviceBuffer]) -> LabResult<()> {
    if buffers.is_empty() {
        return Err("no boundary buffers".into());
    }
    for buffer in buffers {
        kernel_lab::poison::fill_all(session, buffer, &SENTINEL.to_ne_bytes())?;
    }
    Ok(())
}

/// Download every capacity element. Live local rows must be finite, and all
/// other rows must retain the NaN sentinel. Keep live bytes for exact comparison.
pub(super) fn checked_output(
    session: &Session,
    buffer: &DeviceBuffer,
    columns: usize,
    live_mask: &[bool],
) -> LabResult<Vec<u8>> {
    let mut bytes = kernel_lab::poison::read(session, buffer)?;
    validate_output(&bytes, columns, live_mask)?;
    bytes.truncate(live_mask.len() * columns * 2);
    Ok(bytes)
}

fn validate_output(bytes: &[u8], columns: usize, live_mask: &[bool]) -> LabResult<()> {
    use kernel_lab::poison::{Fill, WriteMap, check};
    if columns == 0
        || bytes.len() % (columns * 2) != 0
        || live_mask.len() > bytes.len() / (columns * 2)
    {
        return Err("invalid boundary output shape".into());
    }
    let row_bytes = columns * 2;
    let live = || (0..live_mask.len()).filter(|&row| live_mask[row]);
    let map = WriteMap::of(DType::Bf16).rows(columns, live())?;
    check(
        "boundary output",
        bytes,
        &map,
        Fill::Pattern(&SENTINEL.to_ne_bytes()),
    )?;
    for row in live() {
        let values = &bytes[row * row_bytes..(row + 1) * row_bytes];
        for (col, pair) in values.chunks_exact(2).enumerate() {
            let bits = u16::from_ne_bytes(pair.try_into().unwrap());
            if !bf16_to_f32(bits).is_finite() {
                return Err(format!(
                    "row {row} col {col}: {bits:04x}, expected finite written output"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn compare(expected: &[Vec<u8>], actual: &[Vec<u8>], arm: &str) -> LabResult<()> {
    if expected.len() != 3 || actual.len() != 3 {
        return Err("boundary comparison requires three stages".into());
    }
    for (stage, (a, b)) in expected.iter().zip(actual).enumerate() {
        if a != b {
            return Err(format!(
                "{arm} stage {stage}: bitwise mismatch at byte {:?}",
                a.iter().zip(b).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}

pub(super) fn record(
    session: &Session,
    label: &str,
    tokens: usize,
    counts: &[usize],
    identity: &Value,
    outputs: &[Vec<u8>],
    feature_tiles: usize,
) -> LabResult<()> {
    let full_items: usize = counts.iter().map(|c| c / 128 * feature_tiles).sum();
    let partial_items: usize = counts.iter().filter(|&&c| c % 128 != 0).count() * feature_tiles;
    let full_then_partial = full_items >= 188 && partial_items > 0;
    if [2047, 1025].contains(&tokens) && label.ends_with("hot-local") && !full_then_partial {
        return Err("fixture did not exercise full-to-partial persistent CTA reuse".into());
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(label));
    receipt.insert("live_tokens".into(), json!(tokens));
    receipt.insert("capacity_tokens".into(), json!(CAPACITY));
    receipt.insert("expert_counts".into(), json!(counts));
    receipt.insert("cubins".into(), identity.clone());
    receipt.insert(
        "full_then_partial_cta_reuse".into(),
        json!(full_then_partial),
    );
    receipt.insert(
        "schedule".into(),
        prefill_diagnostics::schedule_grid(counts, feature_tiles, 188),
    );
    receipt.insert(
        "arms".into(),
        json!([
            "baseline-pdl",
            "candidate-pdl",
            "candidate-graph-1",
            "candidate-graph-2",
            "candidate-serial"
        ]),
    );
    receipt.insert(
        "live_sha256".into(),
        json!(
            outputs
                .iter()
                .map(|b| format!("{:x}", Sha256::digest(b)))
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert("bitwise_and_capacity_sentinels_passed".into(), json!(true));
    let path = session.receipt("moe_prefill_boundary", receipt)?;
    eprintln!("boundary case passed {label} t{tokens}: {}", path.display());
    Ok(())
}

#[test]
fn boundary_guards_reject_stale_or_missing_writes() {
    let bytes = |values: &[u16]| {
        values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let valid = bytes(&[0x3f80, SENTINEL, SENTINEL]);
    assert!(validate_output(&valid, 1, &[true, false]).is_ok());
    assert!(validate_output(&bytes(&[SENTINEL; 3]), 1, &[true, false]).is_err());
    assert!(validate_output(&bytes(&[0x3f80, 0, SENTINEL]), 1, &[true, false]).is_err());
    assert!(validate_output(&bytes(&[0x3f80, SENTINEL, 0]), 1, &[true, false]).is_err());
    let expected = vec![valid; 3];
    assert!(compare(&expected, &expected, "same").is_ok());
    let mut actual = expected.clone();
    actual[0][0] ^= 1;
    assert!(compare(&expected, &actual, "one bit").is_err());
}

/// The failed GPU run accepted 2048 then refused 2047 before kernel launch:
/// live tokens were incorrectly used as the prepared row bucket. Check all
/// row plans on the actual 8192-token allocation before another GPU slot.
#[test]
fn boundary_row_plans_accept_partial_live_counts() {
    let symbols = [
        "chop_row_router_mimo_sigmoid256_v1",
        "chop_row_permute_tile_count_mimo_b64_v1",
        "chop_row_permute_tile_prefix_mimo_b64_v1",
        "chop_row_permute_tile_scan_mimo_b64_v1",
        "chop_row_permute_tile_scatter_mimo_b64_v1",
    ];
    let counts = |tokens| RuntimeCounts {
        m: tokens,
        batch: 0,
        context: 0,
        routed_rows: tokens * 8,
        split_k: 1,
    };
    let router = row_entry(symbols[0]).unwrap();
    assert!(router.plan(2047, counts(2047)).is_err());
    for symbol in symbols.iter().map(|s| s.to_string()).chain(std::iter::once(
        combine_entry(model_mimo26_contract::sm120::config::PROFILE).unwrap(),
    )) {
        let entry = row_entry(&symbol).unwrap();
        for live in TOKENS {
            let plan = entry.plan(CAPACITY as u32, counts(live as u32)).unwrap();
            assert_eq!(
                plan.grid,
                entry
                    .plan(CAPACITY as u32, counts(CAPACITY as u32))
                    .unwrap()
                    .grid
            );
        }
    }
    for operation in [
        "router",
        "permute_count",
        "permute_prefix",
        "permute_scan",
        "permute_scatter",
        "combine",
    ] {
        let entry = model_qwen38_contract::sm120::moe::route_entry_for(
            model_qwen38_contract::sm120::config::PROFILE,
            operation,
        )
        .unwrap();
        for live in TOKENS {
            let plan = entry.plan(CAPACITY as u32, live as u32).unwrap();
            assert_eq!(
                plan.grid,
                entry.plan(CAPACITY as u32, CAPACITY as u32).unwrap().grid
            );
        }
    }
}

#[test]
fn boundary_route_intermediate_contract_cpu() {
    use kernel_lab::poison::{self, Fill, WriteMap};
    for words in [8, 128, 129, 256, 257, 1] {
        let map = WriteMap::of(DType::F32).prefix(words).unwrap();
        let mut bytes = vec![0x80; (words + 1) * 4];
        assert!(poison::check("stale routing", &bytes, &map, Fill::Pattern(&[0x80])).is_err());
        for element in bytes[..words * 4].chunks_exact_mut(4) {
            element.copy_from_slice(&(-1_i32).to_ne_bytes());
        }
        assert!(poison::check("written metadata", &bytes, &map, Fill::Pattern(&[0x80])).is_ok());
        bytes[words * 4] = 0;
        assert!(poison::check("routing tail", &bytes, &map, Fill::Pattern(&[0x80])).is_err());
    }
}
