//! Router-in-norm at Flash TP4's 8192-row prefill steps: the MoE-input
//! norm certifies each row's router operands (`rmsnorm_prefill` with
//! `certify`) and the router's certified projection
//! (`f32_projection::certified`) reads the norm's BF16 rows and flags, in
//! place of the zero-component prepare and projection (`router_prefill`).
//! Every output is the served pair's bit for bit
//! (models/deepseek41/lab/router_norm.md); the lab measured the pair at
//! 0.880 of the served norm, prepare and projection at 8192 rows. The
//! projection runs 64-expert tiles (`router_norm_entries::FEATURES`), bit
//! for bit the 32-expert ones, which the router tile pair measured at 0.77
//! of them on captured layers 14 and 39 (models/deepseek41/lab/
//! router_wide.md). The norm writes no widened rows: the projection's SIMT
//! path widens the BF16 rows itself, bit for bit, and the narrow pair
//! measured the norm and router at 0.89 of the step with the widened write
//! (models/deepseek41/lab/router_narrow.md). Other row counts keep the
//! served path.
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, Refusal};
use kernels_sm120::f32_projection::certified;
use model_deepseek41_contract::geometry::Geometry;

use super::f32_zero;
pub use super::router_norm_entries::{
    CUBIN, FAMILY, FEATURES, TILE_ROWS, cubins, norm, router, router_symbol,
};

/// The one qualified and measured step size.
pub const ROWS: u32 = 8192;

/// Whether a prefill step of `rows` rows of `g` at `ranks` ranks runs the
/// router in the norm: Flash TP4's zero-component router at 8192 rows.
pub fn applies(g: &Geometry, ranks: u32, rows: u32) -> bool {
    rows == ROWS
        && f32_zero::selected(g, ranks, g.shape.moe.experts)
        && f32_zero::ROWS.contains(&rows)
}

/// The certifying norm's launch: the served norm launch `served` (its
/// one launch, of the served cached entry, writing no widened rows) with
/// the certificate written to `certified`.
pub fn norm_launch(served: Vec<Launch>, certified: View) -> Result<Launch, Refusal> {
    let [mut launch] = <[Launch; 1]>::try_from(served)
        .map_err(|_| Refusal("router-in-norm needs the one served norm launch"))?;
    let entry = norm();
    if launch.symbol != entry.reference.symbol.replace("_certified_v1", "_v1")
        || launch.rows != ROWS
        || launch.plan != entry.plan(ROWS)?
    {
        return Err(Refusal("router-in-norm needs the served cached norm"));
    }
    if certified.handle == 0 || certified.bytes < u64::from(ROWS) * 4 {
        return Err(Refusal("router-in-norm certificate rows are missing"));
    }
    // Arguments: input, weight, normed, widened, ...
    if launch.arguments.0.get(3) != Some(&Argument::NullDevicePointer) {
        return Err(Refusal("router-in-norm norm must not write widened rows"));
    }
    launch.symbol = entry.reference.symbol;
    launch
        .arguments
        .0
        .push(Argument::DeviceBuffer(certified.handle));
    Ok(launch)
}

/// The router views of the certified projection.
pub(crate) use crate::ops::exact_f32::CertifiedRouterViews as RouterViews;

pub(crate) fn router_launch(v: &RouterViews) -> Result<Launch, Refusal> {
    let e = router();
    let q = e.query();
    let rows = u64::from(ROWS);
    v.normed.covers(rows * u64::from(q.k) * 2)?;
    v.certified.covers(rows * 4)?;
    v.weight.covers(u64::from(q.n) * u64::from(q.k) * 2)?;
    v.output.covers(rows * u64::from(q.n) * 4)?;
    v.live_rows.covers(4)?;
    v.weight_domain.covers(e.weight_domain_bytes())?;
    let written = [v.output.handle];
    let read = [
        v.normed.handle,
        v.certified.handle,
        v.weight.handle,
        v.live_rows.handle,
        v.weight_domain.handle,
    ];
    if read.contains(&0) || written.contains(&0) || read.contains(&v.output.handle) {
        return Err(Refusal(
            "router-in-norm operands are missing or alias the logits",
        ));
    }
    let ptr = |v: View| Argument::DeviceBuffer(v.handle);
    Ok(Launch {
        symbol: router_symbol(),
        rows: ROWS,
        plan: certified::wide_plan(&e, TILE_ROWS, FEATURES, ROWS)?,
        arguments: ArgumentList(vec![
            // The SIMT path widens `normed`; no widened rows exist.
            Argument::NullDevicePointer,
            ptr(v.normed),
            ptr(v.certified),
            ptr(v.weight),
            ptr(v.output),
            ptr(v.live_rows),
            Argument::U32(ROWS),
            ptr(v.weight_domain),
        ]),
        collective: false,
        persistent: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm120::norm_prefill;
    use kernels_sm120::f32_projection::zero_prefill as zero;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn binds_flash_tp4_at_8192_rows_only() {
        for rows in [1, 2048, 4096, 8191, 8192] {
            assert_eq!(applies(&FLASH, 4, rows), rows == 8192);
            assert!(!applies(&FLASH, 2, rows));
        }
    }

    #[test]
    fn keeps_the_served_plans_and_arguments() {
        // The zero path's rows and block; half its feature grid, and only
        // the shared memory the 64-expert tiles stage.
        let e = router();
        let wide = certified::wide_plan(&e, TILE_ROWS, FEATURES, ROWS).unwrap();
        let zero = zero::plan(&e, TILE_ROWS, ROWS).unwrap();
        assert_eq!((wide.block, wide.grid[1]), (zero.block, zero.grid[1]));
        assert_eq!((wide.grid[0], zero.grid[0]), (6, 12));
        assert_eq!(wide.dynamic_shared_bytes, 24_576);
        assert_eq!(
            router_symbol(),
            "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_certified_m32_f64_normed_v1"
        );
        let served = norm_prefill::entries()
            .unwrap()
            .into_iter()
            .find(|e| e.cache_groups)
            .unwrap();
        assert_eq!(norm().plan(ROWS).unwrap(), served.plan(ROWS).unwrap());
        assert_eq!(
            norm().reference.symbol,
            "chop_prefill_norm_ceil_deepseek41_cached_certified_v1"
        );
        let spec = &cubins(&[FLASH]).unwrap()[0];
        let symbols: Vec<_> = spec.entries.iter().map(|e| e.symbol.clone()).collect();
        assert_eq!(symbols, [router_symbol(), norm().reference.symbol]);
        // The norm launch gains only the certificate pointer.
        let launch = Launch {
            symbol: served.reference.symbol.clone(),
            rows: ROWS,
            plan: served.plan(ROWS).unwrap(),
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::NullDevicePointer,
            ]),
            collective: false,
            persistent: false,
        };
        let certified = View {
            handle: 9,
            bytes: 8192 * 4,
        };
        let out = norm_launch(vec![launch.clone()], certified).unwrap();
        assert_eq!(out.symbol, norm().reference.symbol);
        assert_eq!(out.arguments.0[..4], launch.arguments.0[..]);
        assert_eq!(out.arguments.0[4..], [Argument::DeviceBuffer(9)]);
        // A norm still writing its widened rows is refused.
        let mut widening = launch.clone();
        widening.arguments.0[3] = Argument::DeviceBuffer(4);
        assert!(norm_launch(vec![widening], certified).is_err());
        assert!(norm_launch(vec![launch.clone(), launch.clone()], certified).is_err());
        assert!(
            norm_launch(
                vec![Launch {
                    rows: 2048,
                    ..launch.clone()
                }],
                certified
            )
            .is_err()
        );
        assert!(
            norm_launch(
                vec![launch],
                View {
                    handle: 0,
                    ..certified
                }
            )
            .is_err()
        );
    }
}
