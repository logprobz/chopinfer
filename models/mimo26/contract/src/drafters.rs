//! Exact MTP and DFlash materialized views.

use crate::geometry::V26_FLASH as GEOMETRY;

/// BF16 bytes of the DFlash mask embedding.
pub const MASK_BYTES: usize = GEOMETRY.dflash().hidden_size as usize * 2;

use crate::Tp;

pub type MtpField = (&'static str, Vec<u64>, &'static str, &'static str);

/// MTP layer tensors at `tp`: name suffix, per-rank shape, dtype, layout.
/// The attention is SWA-shaped (8 KV heads) and the MLP is layer 0's.
pub fn mtp_fields(tp: Tp) -> [MtpField; 14] {
    const FRAGMENTS: &str = "bf16_mma_fragments_16x16_v1";
    const ROW_MAJOR: &str = "bf16_row_major_v1";
    const FP8: &str = "fp8_e4m3_mma_fragments_16x32_v1";
    const SCALE: &str = "fp8_f32_block_scale_128x128_v1";
    let dense = tp.dense_intermediate();
    [
        (
            "eh_proj.weight",
            vec![tp.geometry.hidden() as u64, tp.mtp_eh_k()],
            "BF16",
            FRAGMENTS,
        ),
        (
            "enorm.weight",
            vec![tp.geometry.hidden() as u64],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "hnorm.weight",
            vec![tp.geometry.hidden() as u64],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "input_layernorm.weight",
            vec![tp.geometry.hidden() as u64],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "pre_mlp_layernorm.weight",
            vec![tp.geometry.hidden() as u64],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "final_layernorm.weight",
            vec![tp.geometry.hidden() as u64],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "self_attn.attention_sink_bias",
            vec![tp.q_heads_for(false)],
            "BF16",
            ROW_MAJOR,
        ),
        (
            "self_attn.o_proj.weight",
            vec![tp.geometry.hidden() as u64, tp.o_proj_k_for(false)],
            "BF16",
            FRAGMENTS,
        ),
        (
            "self_attn.qkv_proj.weight",
            vec![tp.qkv_rows(false), tp.geometry.hidden() as u64],
            "F8_E4M3",
            FP8,
        ),
        (
            "self_attn.qkv_proj.weight_scale_inv",
            vec![tp.qkv_scale_rows(false), tp.geometry.hidden() as u64 / 128],
            "F32",
            tp.qkv_scale_layout(),
        ),
        (
            "mlp.gate_up_proj.weight",
            vec![2 * dense, tp.geometry.hidden() as u64],
            "F8_E4M3",
            FP8,
        ),
        (
            "mlp.gate_up_proj.weight_scale_inv",
            vec![2 * dense / 128, tp.geometry.hidden() as u64 / 128],
            "F32",
            SCALE,
        ),
        (
            "mlp.down_proj.weight",
            vec![tp.geometry.hidden() as u64, dense],
            "F8_E4M3",
            FP8,
        ),
        (
            "mlp.down_proj.weight_scale_inv",
            vec![tp.geometry.hidden() as u64 / 128, dense / 128],
            "F32",
            SCALE,
        ),
    ]
}

const FRAGMENTS: &str = "bf16_mma_fragments_16x16_v1";
const ROW_MAJOR: &str = "bf16_row_major_v1";

pub type DflashField = (&'static str, Vec<u64>, &'static str);

/// DFlash tensors outside the five layers: name, per-rank shape, layout.
pub fn model_fields(tp: Tp) -> [DflashField; 5] {
    [
        (
            "fc.weight",
            vec![
                u64::from(tp.geometry.dflash().hidden_size),
                tp.dflash_fc_k(),
            ],
            FRAGMENTS,
        ),
        (
            "hidden_norm.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size)],
            ROW_MAJOR,
        ),
        (
            "norm.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size)],
            ROW_MAJOR,
        ),
        (
            "context_kv_proj.weight",
            vec![
                tp.geometry.dflash().num_hidden_layers as u64
                    * 2
                    * tp.dflash_kv_heads()
                    * tp.geometry.dflash().head_dim as u64,
                u64::from(tp.geometry.dflash().hidden_size),
            ],
            FRAGMENTS,
        ),
        (
            "mask_embedding.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size)],
            ROW_MAJOR,
        ),
    ]
}

/// DFlash tensors of each layer, after `layers.{layer}.`.
pub fn layer_fields(tp: Tp) -> [DflashField; 9] {
    let intermediate = tp.dflash_intermediate();
    [
        (
            "input_layernorm.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size)],
            ROW_MAJOR,
        ),
        (
            "post_attention_layernorm.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size)],
            ROW_MAJOR,
        ),
        (
            "self_attn.qkv_proj.weight",
            vec![
                tp.dflash_qkv_rows(),
                u64::from(tp.geometry.dflash().hidden_size),
            ],
            FRAGMENTS,
        ),
        (
            "self_attn.q_norm.weight",
            vec![tp.geometry.dflash().head_dim as u64],
            ROW_MAJOR,
        ),
        (
            "self_attn.k_norm.weight",
            vec![tp.geometry.dflash().head_dim as u64],
            ROW_MAJOR,
        ),
        (
            "self_attn.o_proj.weight",
            vec![
                u64::from(tp.geometry.dflash().hidden_size),
                tp.dflash_q_heads() * tp.geometry.dflash().head_dim as u64,
            ],
            FRAGMENTS,
        ),
        (
            "self_attn.attention_sink_bias",
            vec![tp.dflash_q_heads()],
            ROW_MAJOR,
        ),
        (
            "mlp.gate_up_proj.weight",
            vec![
                2 * intermediate,
                u64::from(tp.geometry.dflash().hidden_size),
            ],
            FRAGMENTS,
        ),
        (
            "mlp.down_proj.weight",
            vec![u64::from(tp.geometry.dflash().hidden_size), intermediate],
            FRAGMENTS,
        ),
    ]
}

/// Every view name of a complete DFlash rank image with its geometry.
pub fn fields(tp: Tp) -> impl Iterator<Item = (String, Vec<u64>, &'static str)> {
    model_fields(tp)
        .into_iter()
        .map(|(name, shape, layout)| (name.to_owned(), shape, layout))
        .chain(
            (0..tp.geometry.dflash().num_hidden_layers).flat_map(move |layer| {
                layer_fields(tp)
                    .into_iter()
                    .map(move |(suffix, shape, layout)| {
                        (format!("layers.{layer}.{suffix}"), shape, layout)
                    })
            }),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TP4 tables every existing drafter image was checked against.
    #[test]
    fn tp4_drafter_shapes_are_unchanged() {
        let mtp = mtp_fields(Tp::TP4);
        let shapes: Vec<_> = mtp.iter().map(|field| field.1.clone()).collect();
        assert_eq!(
            shapes,
            [
                vec![4096, 2048],
                vec![4096],
                vec![4096],
                vec![4096],
                vec![4096],
                vec![4096],
                vec![16],
                vec![4096, 2048],
                vec![3712, 4096],
                vec![29, 32],
                vec![8192, 4096],
                vec![64, 32],
                vec![4096, 4096],
                vec![32, 32],
            ]
        );
        assert!(
            mtp.iter()
                .all(|field| field.3 != crate::tp::ROW_SCALE_LAYOUT)
        );
        let dflash: Vec<_> = model_fields(Tp::TP4)
            .into_iter()
            .chain(layer_fields(Tp::TP4))
            .map(|field| field.1)
            .collect();
        assert_eq!(
            dflash,
            [
                vec![4096, 5120],
                vec![4096],
                vec![4096],
                vec![2560, 4096],
                vec![4096],
                vec![4096],
                vec![4096],
                vec![2560, 4096],
                vec![128],
                vec![128],
                vec![4096, 2048],
                vec![16],
                vec![8192, 4096],
                vec![4096, 4096],
            ]
        );
    }
}
