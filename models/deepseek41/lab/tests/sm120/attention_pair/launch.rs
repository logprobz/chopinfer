//! Configure the same per-symbol limits that serving gets from prepared ops.
use super::*;
use kernel_core::backend::{Backend, Launch, Need};
use kernel_core::contract::LaunchPlan;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};

const DEFAULT_SHARED: u32 = 48 * 1024;

fn entries() -> LabResult<Vec<Entry>> {
    let mut entries = Vec::new();
    for geometry in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(geometry, ranks).is_ok() {
                entries.extend(latent::entries(geometry, ranks)?);
                entries.extend(candidate::entries(geometry, ranks)?);
            }
        }
    }
    entries.push(state_probe::pair()?.1);
    Ok(entries)
}

fn prepared_limits(entry: &Entry) -> LabResult<Vec<(String, u32)>> {
    let op = kernels_sm120::backend::Sm120::new()
        .with_latent_sources(entry.clone())
        .latent_attention_sources(&entry.query, Need::rows(1))
        .map_err(|e| e.to_string())?;
    Ok(op.info().symbols.clone())
}

pub fn limits() -> LabResult<BTreeMap<String, u32>> {
    let mut limits = BTreeMap::new();
    for entry in entries()? {
        for (symbol, bytes) in prepared_limits(&entry)? {
            let limit = limits.entry(symbol).or_insert(0);
            *limit = (*limit).max(bytes);
        }
    }
    for entry in decode_candidate::entries(geometry())? {
        limits.insert(
            entry.symbol(),
            entry
                .plan(1)
                .map_err(|e| e.to_string())?
                .dynamic_shared_bytes,
        );
    }
    for entry in cache_catalog::entries(geometry())? {
        limits.insert(
            entry.symbol(),
            entry
                .plan(1)
                .map_err(|e| e.to_string())?
                .dynamic_shared_bytes,
        );
    }
    for entry in grouped_cache::entries(geometry())? {
        limits.insert(
            entry.symbol(),
            entry
                .plan(4)
                .map_err(|e| e.to_string())?
                .dynamic_shared_bytes,
        );
    }
    Ok(limits)
}

pub fn configure(
    limits: &BTreeMap<String, u32>,
    mut set: impl FnMut(&str, u32) -> LabResult<()>,
) -> LabResult<BTreeMap<String, u32>> {
    let mut configured = BTreeMap::new();
    for (symbol, &bytes) in limits {
        if bytes > DEFAULT_SHARED {
            set(symbol, bytes)?;
            configured.insert(symbol.clone(), bytes);
        }
    }
    Ok(configured)
}

pub(super) fn check(
    symbol: &str,
    plan: LaunchPlan,
    configured: &BTreeMap<String, u32>,
) -> LabResult<()> {
    let limit = configured.get(symbol).copied().unwrap_or(DEFAULT_SHARED);
    if plan.dynamic_shared_bytes > limit {
        return Err(format!(
            "{symbol}: launch needs {} shared bytes but the function is configured for {limit}",
            plan.dynamic_shared_bytes
        ));
    }
    kernel_harness::launch::config(plan)?;
    Ok(())
}

fn description(call: &Launch) -> Value {
    json!({"symbol": call.symbol, "grid": call.plan.grid, "block": call.plan.block,
        "dynamic_shared_bytes": call.plan.dynamic_shared_bytes, "cluster": call.plan.cluster,
        "pdl": call.plan.pdl})
}

impl Modules {
    pub fn trace(&self, call: &Launch) -> LabResult<()> {
        let mut fields = description(call);
        fields["configured_shared_bytes"] = json!(
            self.configured
                .get(&call.symbol)
                .copied()
                .unwrap_or(DEFAULT_SHARED)
        );
        eprintln!("ATTENTION_LAUNCH {fields}");
        self.function(&call.symbol)?;
        check(&call.symbol, call.plan, &self.configured)
    }

    pub fn launch(
        &self,
        session: &Session,
        stream: &runtime_cuda::Stream,
        call: &Launch,
        bindings: &kernel_harness::launch::Bindings<'_>,
    ) -> LabResult<()> {
        self.trace(call)?;
        kernel_harness::launch::launch(
            &session.context,
            self.function(&call.symbol)?,
            stream,
            call.plan,
            &call.arguments,
            bindings,
        )
        .map_err(|e| format!("launch {} failed: {e}", description(call)))
    }
}

#[test]
fn cpu_source_launches_require_function_shared_memory_opt_in() {
    let limits = limits().unwrap();
    let mut attributes = BTreeMap::new();
    let configured = configure(&limits, |symbol, bytes| {
        assert!(attributes.insert(symbol.to_owned(), bytes).is_none());
        Ok(())
    })
    .unwrap();
    assert_eq!(configured, attributes);
    let mut checked = 0;
    for entry in entries().unwrap() {
        let metadata = prepared_limits(&entry).unwrap();
        for rows in [1, 4, 6, 8, 32, 48, 192, entry.capacity()] {
            if rows > entry.capacity() {
                continue;
            }
            for (index, plan) in [entry.plan(rows).unwrap(), entry.merge_plan(rows).unwrap()]
                .into_iter()
                .enumerate()
            {
                let (symbol, shared) = &metadata[index];
                assert!(plan.dynamic_shared_bytes <= *shared);
                assert!(
                    plan.block.iter().product::<u32>()
                        <= entry.entry_specs().unwrap()[index].max_threads
                );
                assert_eq!(plan.cluster, [1, 1, 1]);
                check(symbol, plan, &configured).unwrap();
                if plan.dynamic_shared_bytes > DEFAULT_SHARED {
                    assert!(check(symbol, plan, &BTreeMap::new()).is_err());
                    assert_eq!(attributes[symbol], *shared);
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 100 && !attributes.is_empty());
    assert!(configure(&limits, |_, _| Err("attribute refused".into())).is_err());
    println!(
        "ATTENTION_LAUNCH_CONTRACT_OK plans={checked} configured={}",
        attributes.len()
    );
}
