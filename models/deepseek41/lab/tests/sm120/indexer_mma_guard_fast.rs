//! Lab-only entry substitution, with the guarded serving plans unchanged.
use super::*;
use kernels_sm120::pooled_indexer::position_mma_guard_fast as fast;

const MODULE: &str = "position_mma_guard_fast_deepseek41_sm120";
pub(super) fn remap(calls: &mut [Launch]) {
    let mut scores = 0;
    for call in calls {
        if call.symbol.ends_with("_mma_fp4_guard_v1") {
            call.symbol = fast::score_symbol(&call.symbol);
            scores += 1;
        }
    }
    assert_eq!(
        scores, 1,
        "each planned op must replace exactly one score launch"
    );
}
pub(super) fn module_name(symbol: &str) -> LabResult<&'static str> {
    if symbol.ends_with("_mma_fp4_guard_fast_v1")
        || symbol.ends_with("_mma_fp4_guard_fast_v1_counts_v1")
    {
        Ok(MODULE)
    } else {
        Ok(bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol)
            .ok_or("missing served cubin")?
            .0)
    }
}
pub(super) fn load(
    session: &Session,
    modules: &mut BTreeMap<String, runtime_cuda::Module>,
    symbol: &str,
) -> LabResult<()> {
    let name = module_name(symbol)?;
    if modules.contains_key(name) {
        return Ok(());
    }
    let data = if name == MODULE {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_INDEX_GUARD_ARTIFACTS")
                .ok_or("staged candidate required")?,
        );
        bytes(&root.join(format!("{MODULE}.cubin")))?
    } else {
        bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol)
            .ok_or("missing served cubin")?
            .1
            .to_vec()
    };
    modules.insert(name.into(), session.load_cubin(&data)?);
    Ok(())
}
pub(super) fn diagnose(
    session: &Session,
    arena: &mut Arena<'_>,
    modules: &BTreeMap<String, runtime_cuda::Module>,
    calls: &[Launch],
) -> LabResult<[u32; 4]> {
    fire(session, arena, modules, &calls[..1])?;
    let counts = arena.input(&[0; 16])?;
    let mut call = calls[1].clone();
    call.symbol.push_str("_counts_v1");
    call.arguments.0.push(Argument::DeviceBuffer(counts.handle));
    fire(session, arena, modules, &[call])?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let data = arena.read(counts)?;
    Ok(std::array::from_fn(|i| {
        u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap())
    }))
}

#[test]
fn cpu_candidate_keeps_abi_grid_workspace_and_selection() {
    guard::compact_matrix_cpu(true);
    use kernels_sm120::pooled_indexer::position_mma_guard;
    for entry in crate::indexer_entries::entries(&FLASH) {
        let old = position_mma_guard::dense_specs(&entry).unwrap().remove(1);
        let new = fast::score(old.clone());
        assert_eq!(old.params, new.params);
        assert_eq!(new.symbol, fast::score_symbol(&old.symbol));
        assert_eq!(
            new.body
                .replace("position_mma_guard_fast::", "position_mma_guard::"),
            old.body
        );
    }
    for entry in crate::indexer_entries::block_entries(&FLASH) {
        let old = position_mma_guard::compact_specs(&entry).unwrap().remove(1);
        let new = fast::score(old.clone());
        assert_eq!(old.params, new.params);
        assert_eq!(
            new.body
                .replace("position_mma_guard_fast::", "position_mma_guard::"),
            old.body
        );
    }
    for layer in [2, 20] {
        for rows in [1, 2, 4, 6, 8, 16, 32, 8192] {
            let width = 131136 / FLASH.compress_ratio(layer);
            let b = cpu_buffers(rows, FLASH.compress_ratio(layer), width);
            let query = indexer::selection(&FLASH, layer, rows).unwrap();
            let op = guard::guarded(true)
                .position_indexer(&query, 512, width, Need::rows(rows))
                .unwrap();
            let old = op
                .launches(&b, v(99, op.info().workspace), Counts::rows(rows))
                .unwrap();
            let mut new = old.clone();
            remap(&mut new);
            assert_eq!(new[0], old[0]);
            assert_eq!(new[2..], old[2..]);
            new[1].symbol = old[1].symbol.clone();
            assert_eq!(new, old);
        }
    }
}

#[test]
#[ignore = "requires sealed captures and CHOPINFER_INDEX_GUARD_ARTIFACTS on idle GPUs"]
fn gpu_capture_failures_match_simt() {
    captured_comparisons_with(guard::guarded, true);
}
#[test]
#[ignore = "requires sealed captures and CHOPINFER_INDEX_GUARD_ARTIFACTS on idle GPUs"]
fn gpu_real_guard_frequency_and_cost() {
    guard::real(true);
}
#[test]
#[ignore = "requires CHOPINFER_INDEX_GUARD_ARTIFACTS and idle GPUs"]
fn gpu_decode_prefill_and_exponent_edges() {
    guard::matrix(true);
}
