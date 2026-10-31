//! The cached decode producer that classifies its own rows and resets a
//! consumer's arrival ticket (`gemm/exact_f32_classified.cuh`), for a
//! consumer that would otherwise launch `classify_rows` and the reset
//! before it. Its chains, partial layout and arithmetic are the cached
//! `_decode` stage's; only the CTA grouping of the chains changes, so one
//! portable cluster covers every K of a token block.
use super::{Entry, pointer};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

/// Chains per CTA: 40 chains of K 5120 in a cluster of eight CTAs.
pub const WARPS: u32 = 5;

pub fn symbol(entry: &Entry) -> String {
    cached(entry)
        .map(|c| c.stage_symbol("_decode_classified"))
        .unwrap_or_default()
}

fn cached(entry: &Entry) -> Result<Entry, Refusal> {
    let q = entry.query();
    if q.weight != WeightFormat::Bf16 || q.k % 4 != 0 {
        return Err(Refusal(
            "a classifying decode needs BF16 weights and K in whole vectors",
        ));
    }
    if q.k.div_ceil(128).div_ceil(WARPS) > 8 {
        return Err(Refusal(
            "a classifying decode covers K in one portable cluster",
        ));
    }
    Ok(if entry.uses_cached_weights() {
        entry.clone()
    } else {
        entry.clone().with_decode()
    })
}

/// The cached decode's grid with its chains in CTAs of [`WARPS`] warps,
/// one cluster per token block and feature tile.
pub fn plan(entry: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let cached = cached(entry)?;
    let mut plan = cached.decode_plan(rows)?;
    let ctas = cached.query().k.div_ceil(128).div_ceil(WARPS);
    plan.grid[2] = ctas;
    plan.block = [WARPS * 32, 1, 1];
    plan.cluster = [1, 1, ctas];
    crate::backend::check_plan(&plan)?;
    Ok(plan)
}

/// The cached `_decode` stage's parameters followed by the ticket.
pub fn entry_spec(entry: &Entry) -> Result<EntrySpec, Refusal> {
    let cached = cached(entry)?;
    let decode = cached
        .entry_specs()
        .into_iter()
        .find(|s| s.symbol == cached.stage_symbol("_decode"))
        .ok_or(Refusal(
            "a classifying decode needs the cached decode stage",
        ))?;
    let mut params = decode.params;
    params.push(pointer("arrived", "uint32_t"));
    let q = cached.query();
    Ok(EntrySpec::new(
        symbol(entry),
        params,
        format!(
            "chop::exact_f32::decode_classified<{},{},{WARPS}>(activation,weight,workspace+32,live_rows,rows,weight_domain,arrived);",
            q.n, q.k
        ),
        WARPS * 32,
        1,
    ))
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = super::cubin_spec(name, entries)?;
    spec.entries = entries.iter().map(entry_spec).collect::<Result<_, _>>()?;
    spec.headers = vec!["#include \"gemm/exact_f32_classified.cuh\"".into()];
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery};
    use kernel_core::precision::{ExecutionPhase, OperationUse};

    fn router(k: u32, weight: WeightFormat) -> Entry {
        Entry::new(
            GemmQuery {
                m_capacity: 8192,
                n: 384,
                k,
                weight,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            },
            "chop_test_dense_f32_m8192_n384_k5120_v1",
        )
        .unwrap()
    }

    #[test]
    fn classified_decode_keeps_the_decode_chains_and_abi_in_one_cluster() {
        let entry = router(5120, WeightFormat::Bf16).with_decode();
        let decode = entry
            .entry_specs()
            .into_iter()
            .find(|s| s.symbol == entry.stage_symbol("_decode"))
            .unwrap();
        let spec = entry_spec(&entry).unwrap();
        assert_eq!(spec.symbol, symbol(&entry));
        assert!(spec.symbol.ends_with("_cached_decode_classified_v1"));
        assert_eq!(spec.params[..decode.params.len()], decode.params[..]);
        assert_eq!(spec.params.len(), decode.params.len() + 1);
        assert!(
            spec.body
                .contains("decode_classified<384,5120,5>(activation,weight,workspace+32,")
        );
        for rows in [1, 6, 8, 9, 32] {
            let old = entry.decode_plan(rows).unwrap();
            let new = plan(&entry, rows).unwrap();
            assert_eq!(old.grid[..2], new.grid[..2]);
            // Ten CTAs of four chains become eight of five: 40 chains.
            assert_eq!(old.grid[2] * old.block[0] / 32, 40);
            assert_eq!(
                (new.grid[2], new.block[0], new.cluster),
                (8, 160, [1, 1, 8])
            );
            assert!(new.pdl);
        }
        assert!(plan(&entry, 33).is_err());
        assert_eq!(symbol(&router(5120, WeightFormat::Bf16)), spec.symbol);
        assert!(entry_spec(&router(5120, WeightFormat::F32)).is_err());
        assert!(entry_spec(&router(5122, WeightFormat::Bf16)).is_err());
        assert!(entry_spec(&router(8192, WeightFormat::Bf16)).is_err());
        let cubin = cubin_spec("classified", &[entry]).unwrap();
        assert_eq!(cubin.entries.len(), 1);
        assert_eq!(
            (&cubin.entries[0].symbol, &cubin.entries[0].body),
            (&spec.symbol, &spec.body)
        );
        assert_eq!(
            cubin.headers,
            ["#include \"gemm/exact_f32_classified.cuh\""]
        );
    }
}
