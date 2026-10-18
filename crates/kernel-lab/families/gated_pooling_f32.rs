//! F32 pair pooling, exact tail storage and masked F32 RMSNorm.

use kernel_build::CubinSpec;
use kernel_core::contract::RecurrentPhase;
use kernel_core::gated_pooling::GatedPoolingQuery;
use kernels_sm120::gated_pooling::{Entry, cubin_spec};

pub const FAMILY: &str = "gated_pooling_f32";
pub const CUBIN: &str = "gated_pooling_f32_v1";

pub fn entries() -> Vec<Entry> {
    [128, 512]
        .into_iter()
        .flat_map(|width| {
            [
                ("prefill", RecurrentPhase::Prefill),
                ("decode", RecurrentPhase::Decode),
                ("verify", RecurrentPhase::Verify { max_rows: 6 }),
            ]
            .into_iter()
            .map(move |(phase_name, phase)| {
                Entry::new(
                    GatedPoolingQuery {
                        group: 2,
                        width,
                        norm_eps_bits: 1e-20f32.to_bits(),
                        phase,
                        capacity: 12,
                    },
                    &format!("chop_pool_f32_{phase_name}_w{width}"),
                )
                .expect("lab pooling geometry")
            })
        })
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![cubin_spec(CUBIN, &entries()).expect("lab pooling entries")],
    )
}
