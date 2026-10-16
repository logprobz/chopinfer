//! The SM120 module loading MiMo's instances do: the dynamic shared memory
//! each compiled entry requests.

use kernel_core::contract::Refusal;

/// Dynamic shared memory requested by a compiled MiMo entry of any topology,
/// for an instance serving `items`: attention entries prepared for its
/// largest bucket (verify item tables grow above 32 items).
pub fn dynamic_shared_bytes(
    symbol: &str,
    items: bundle_runtime::geometry::Items,
) -> Result<u32, Refusal> {
    for entry in crate::sm120::launch::dense::mimo_entries() {
        if entry.symbol == symbol {
            return kernels_sm120::gemm::capacity_shared_bytes(&entry.key);
        }
    }
    for variant in
        crate::topology::all().flat_map(|tp| crate::sm120::launch::moe::mimo_variants(tp).iter())
    {
        if variant.symbol == symbol {
            return variant.capacity_shared_bytes();
        }
        if variant.reduction_symbol().as_deref() == Some(symbol) {
            return Ok(0);
        }
    }
    for variant in crate::topology::all()
        .flat_map(|tp| crate::sm120::launch::attention::mimo_variants(tp).iter())
    {
        let variant = if items.capacity() > variant.query.max_batch_capacity {
            variant.with_batch_capacity(items.capacity())?
        } else {
            *variant
        };
        if let Some(bytes) = variant.dynamic_shared_bytes(symbol) {
            return Ok(bytes);
        }
    }
    for geometry in model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .filter(|g| g.encoders)
    {
        if let Some(entry) = model_mimo26_contract::sm120::encoders::kernels_for(geometry)
            .map_err(|_| Refusal("MiMo encoder declarations are invalid"))?
            .iter()
            .find(|entry| entry.symbol == symbol)
        {
            return Ok(entry.dynamic_shared_bytes().unwrap_or(0));
        }
    }
    if let Some(bytes) = kernels_sm120::gemm::mid::dynamic_shared_bytes(symbol) {
        return Ok(bytes);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_embedded_symbol_has_valid_shared_memory_request() {
        for cubin in crate::CUBINS {
            for symbol in cubin.symbols {
                for served in [1, 32, 256] {
                    let items = bundle_runtime::geometry::Items::new(served).unwrap();
                    assert!(
                        super::dynamic_shared_bytes(symbol, items).expect("shared memory plan")
                            <= 101_376
                    );
                }
            }
        }
    }
}
