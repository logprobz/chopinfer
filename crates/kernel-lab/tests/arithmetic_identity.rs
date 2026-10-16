//! Every catalog entry of the kernel base and every model declares its
//! arithmetic (`kernel_lab::catalogs`).
use kernels_sm120::attention::sparse;
use kernels_sm120::{gemm, moe};

/// Every dense, fused GEMM, medium-M, grouped and sparse attention catalog
/// entry's cubin declares its arithmetic.
#[test]
fn every_catalog_entry_declares_its_arithmetic() {
    let cubins: Vec<_> = gemm::declared_cubin_specs(kernel_lab::catalogs(), None)
        .into_iter()
        .chain(gemm::fused::declared_cubin_specs(
            kernel_lab::catalogs(),
            None,
        ))
        .chain(gemm::mid::declared_cubin_spec_for(
            kernel_lab::catalogs(),
            "gemm_mid_sm120",
            &kernel_lab::catalogs().profiles(),
        ))
        .chain(moe::declared_cubin_specs(kernel_lab::catalogs(), None))
        .chain(sparse::declared_cubin_specs(kernel_lab::catalogs(), None))
        .collect();
    let declared = |symbol: &str| {
        cubins
            .iter()
            .find_map(|cubin| cubin.arithmetic(symbol))
            .unwrap_or_else(|| panic!("{symbol} declares no arithmetic"))
            .to_owned()
    };
    for symbol in gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .map(|v| v.symbol.clone())
        .chain(
            gemm::fused::catalog(kernel_lab::catalogs())
                .into_iter()
                .map(|e| e.symbol),
        )
        .chain(
            gemm::mid::catalog(kernel_lab::catalogs())
                .into_iter()
                .map(|e| e.kernel.symbol()),
        )
        .chain(
            moe::catalog(kernel_lab::catalogs())
                .into_iter()
                .map(|v| v.symbol),
        )
    {
        declared(&symbol);
    }
    for entry in sparse::catalog(kernel_lab::catalogs()) {
        if let Some(merge) = entry.merge_symbol() {
            assert!(declared(&entry.symbol()).contains("splits_from_rows=[(1, "));
            assert!(declared(&merge).starts_with("merge; "));
        }
    }
    let grouped = moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .next()
        .unwrap();
    assert!(declared(&grouped.symbol).contains("route_weight=AfterDown"));
}
