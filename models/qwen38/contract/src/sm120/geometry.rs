//! Geometry-owned profile identities and CUDA dimensions.

use crate::geometry::*;
use kernels_sm120::{Model, Profile};

#[derive(Clone, Copy, Debug)]
pub struct GeometryModel {
    pub attention_names: [&'static str; 6],
    pub geometry: Geometry,
    pub model: Model,
    pub hc: kernels_sm120::gemm::fused::HcGeometry,
    pub transport: kernels_sm120::comm::hyper_connection::HcTransport,
    pub decode: kernels_sm120::comm::hyper_connection_decode::HcDecodeTransport,
    /// The cubins and entries of the F32 norm-weight forms.
    pub f32_norms: F32NormNames,
}

/// Names of the consumers' F32 norm-weight forms beside the BF16 ones: the
/// all-reduce transports with the F32 inject-norm epilogue, and the grouped
/// norm row. Their arithmetic is the BF16 forms' apart from the weight load.
#[derive(Clone, Copy, Debug)]
pub struct F32NormNames {
    pub cubin: &'static str,
    pub symbol: &'static str,
    pub scatter_symbol: &'static str,
    pub gather_symbol: &'static str,
    pub decode_cubin: &'static str,
    pub wide_cubin: &'static str,
    pub symbols_v2: [&'static str; 3],
    pub symbols_v3: [&'static str; 3],
    pub wide_symbols: [&'static str; 3],
    pub row_cubin: &'static str,
    pub row_symbol: &'static str,
}

impl GeometryModel {
    /// The epilogue shape whose weight is stored as `weight`.
    fn shape_for(
        &self,
        weight: kernel_core::contract::DType,
    ) -> Result<kernels_sm120::comm::hyper_connection::HcShape, kernel_core::contract::Refusal>
    {
        let shape = self.transport.shape;
        Ok(kernels_sm120::comm::hyper_connection::HcShape {
            epilogue: shape.epilogue.with_norm_weight_dtype(weight)?,
            ..shape
        })
    }

    /// The prefill all-reduce whose next norm weight is stored as `weight`:
    /// the BF16 transport unchanged, or its F32 form.
    pub fn transport_for(
        &self,
        weight: kernel_core::contract::DType,
    ) -> Result<kernels_sm120::comm::hyper_connection::HcTransport, kernel_core::contract::Refusal>
    {
        if weight == kernel_core::contract::DType::Bf16 {
            return Ok(self.transport);
        }
        let names = self.f32_norms;
        Ok(kernels_sm120::comm::hyper_connection::HcTransport {
            shape: self.shape_for(weight)?,
            cubin: names.cubin,
            symbol: names.symbol,
            scatter_symbol: names.scatter_symbol,
            gather_symbol: names.gather_symbol,
            ..self.transport
        })
    }

    /// The decode and wide all-reduces whose next norm weight is stored as
    /// `weight`.
    pub fn decode_for(
        &self,
        weight: kernel_core::contract::DType,
    ) -> Result<
        kernels_sm120::comm::hyper_connection_decode::HcDecodeTransport,
        kernel_core::contract::Refusal,
    > {
        if weight == kernel_core::contract::DType::Bf16 {
            return Ok(self.decode);
        }
        let names = self.f32_norms;
        Ok(
            kernels_sm120::comm::hyper_connection_decode::HcDecodeTransport {
                shape: self.shape_for(weight)?,
                cubin: names.decode_cubin,
                wide_cubin: names.wide_cubin,
                symbols_v2: names.symbols_v2,
                symbols_v3: names.symbols_v3,
                wide_symbols: names.wide_symbols,
                ..self.decode
            },
        )
    }

    /// The per-stream grouped norm over F32 weights: one 256-thread CTA per
    /// (row, stream) group, with the BF16 row's strided sum and
    /// shuffle-down tree.
    pub fn f32_norm_row(
        &self,
    ) -> Result<kernels_sm120::comm::epilogue::GroupedRmsNorm, kernel_core::contract::Refusal> {
        if self.geometry.eps().to_bits() != kernels_sm120::comm::epilogue::HC_INJECT_NORM_EPS_BITS {
            return Err(kernel_core::contract::Refusal(
                "unsupported grouped norm epsilon",
            ));
        }
        let row = kernels_sm120::comm::epilogue::GroupedRmsNorm {
            reduction: kernels_sm120::comm::epilogue::NormReduction::ShuffleDown,
            weight_dtype: kernel_core::contract::DType::F32,
            group_width: self.geometry.hidden(),
            groups: self.geometry.streams(),
        };
        row.validate()?;
        Ok(row)
    }

    /// Whether all HC producers required by the padded/precomputed path are qualified.
    pub fn hc_precompute(self) -> bool {
        super::gemm::fused_for(&self)
            .iter()
            .filter(|entry| kernels_sm120::gemm::fused::hc_precompute_spec(entry).is_ok())
            .count()
            == 4
    }
}

/// Validate cluster settings before a build starts generating any catalog entries.
pub fn validate_catalogs() -> Result<(), String> {
    for owner in MODELS {
        for ranks in [2, 4] {
            super::gemm::cluster_shapes(
                GeometryShape {
                    geometry: owner.geometry,
                    ranks,
                },
                ranks == 4,
            )?;
        }
    }
    Ok(())
}

pub fn require_base_geometries(consumer: &str) -> Result<(), String> {
    require_base_set(consumer, GEOMETRIES)
}

pub fn require_base_set(consumer: &str, geometries: &[Geometry]) -> Result<(), String> {
    let extras = geometries
        .iter()
        .filter(|g| !g.base)
        .map(|g| g.id)
        .collect::<Vec<_>>();
    if extras.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{consumer} refuses fixture geometries: {} (unset CHOPINFER_QWEN38_GEOMETRY_DIRS)",
            extras.join(", ")
        ))
    }
}

macro_rules! geometry_model {
    ($id:literal, $tag:literal, $geometry:expr, $config:expr) => {{
        const G: Geometry = $geometry;
        const EPILOGUE: kernels_sm120::comm::Epilogue = kernels_sm120::comm::Epilogue {
            width: G.hidden(),
            streams: G.streams(),
            ..super::comm::EPILOGUE
        };
        const SHAPE: kernels_sm120::comm::hyper_connection::HcShape =
            kernels_sm120::comm::hyper_connection::HcShape {
                epilogue: EPILOGUE,
                input: kernels_sm120::comm::input::BF16,
            };
        // Only pinned geometries compile distinct decode v3 entries; the
        // others alias v3 to their v2 transport.
        const SYMBOLS_V3: [&str; 3] = if G.pinned_names {
            super::comm::decode::SYMBOLS_V3
        } else {
            [
                concat!("chop_comm_", $tag, "_allreduce_hc_decode_r2_v2"),
                concat!("chop_comm_", $tag, "_allreduce_hc_decode_r4_v2"),
                concat!("chop_comm_", $tag, "_allreduce_hc_decode_r8_v2"),
            ]
        };
        const F32_SYMBOLS_V2: [&str; 3] = [
            concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r2_v2"),
            concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r4_v2"),
            concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r8_v2"),
        ];
        const F32_SYMBOLS_V3: [&str; 3] = if G.pinned_names {
            [
                concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r2_v3"),
                concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r4_v3"),
                concat!("chop_comm_", $tag, "_allreduce_hc_f32_decode_r8_v3"),
            ]
        } else {
            F32_SYMBOLS_V2
        };
        GeometryModel {
            attention_names: [
                concat!($tag, "_bf16_prefill"),
                concat!($tag, "_bf16_decode"),
                concat!($tag, "_bf16_verify"),
                concat!($tag, "_fp8_prefill"),
                concat!($tag, "_fp8_decode"),
                concat!($tag, "_fp8_verify"),
            ],
            geometry: G,
            model: Model::new(
                &[
                    (
                        4,
                        kernels_sm120::ProfileNames {
                            name: concat!("qwen38-", $id),
                            tag: $tag,
                        },
                    ),
                    (
                        2,
                        kernels_sm120::ProfileNames {
                            name: concat!("qwen38-", $id, "-tp2"),
                            tag: concat!($tag, "_tp2"),
                        },
                    ),
                ],
                crate::geometry::MODEL_CONFIGS[$config],
            ),
            hc: kernels_sm120::gemm::fused::HcGeometry {
                width: G.hidden(),
                streams: G.streams(),
                rank: G.hc_rank(),
                up_mix_symbol: concat!("chop_gemm_", $tag, "_hc_up_mix_decode_v1"),
            },
            transport: kernels_sm120::comm::hyper_connection::HcTransport {
                shape: SHAPE,
                eps_bits: G.eps().to_bits(),
                cubin: concat!("comm_", $tag),
                symbol: concat!("chop_comm_", $tag, "_allreduce_hc_v1"),
                scatter_symbol: concat!("chop_comm_", $tag, "_reduce_scatter_v2"),
                gather_symbol: concat!("chop_comm_", $tag, "_allgather_hc_v2"),
            },
            decode: kernels_sm120::comm::hyper_connection_decode::HcDecodeTransport {
                shape: SHAPE,
                eps_bits: G.eps().to_bits(),
                cubin: concat!("comm_", $tag, "_decode"),
                wide_cubin: concat!("comm_", $tag, "_decode_wide"),
                symbols_v2: [
                    concat!("chop_comm_", $tag, "_allreduce_hc_decode_r2_v2"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_decode_r4_v2"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_decode_r8_v2"),
                ],
                symbols_v3: SYMBOLS_V3,
                wide_symbols: [
                    concat!("chop_comm_", $tag, "_allreduce_hc_wide_r2_v1"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_wide_r4_v1"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_wide_r8_v1"),
                ],
            },
            f32_norms: F32NormNames {
                cubin: concat!("comm_", $tag, "_f32"),
                symbol: concat!("chop_comm_", $tag, "_allreduce_hc_f32_v1"),
                scatter_symbol: concat!("chop_comm_", $tag, "_reduce_scatter_f32_v2"),
                gather_symbol: concat!("chop_comm_", $tag, "_allgather_hc_f32_v2"),
                decode_cubin: concat!("comm_", $tag, "_f32_decode"),
                wide_cubin: concat!("comm_", $tag, "_f32_decode_wide"),
                symbols_v2: F32_SYMBOLS_V2,
                symbols_v3: F32_SYMBOLS_V3,
                wide_symbols: [
                    concat!("chop_comm_", $tag, "_allreduce_hc_f32_wide_r2_v1"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_f32_wide_r4_v1"),
                    concat!("chop_comm_", $tag, "_allreduce_hc_f32_wide_r8_v1"),
                ],
                row_cubin: concat!("row_", $tag, "_norm_f32"),
                row_symbol: concat!("chop_row_", $tag, "_hc_norm_f32_v1"),
            },
        }
    }};
}

include!(concat!(env!("OUT_DIR"), "/kernel_profiles.rs"));

const fn profiles() -> [Profile; MODELS.len() * 2] {
    let mut profiles = [MODELS[0].model.at(4); MODELS.len() * 2];
    let mut i = 0;
    while i < MODELS.len() {
        profiles[2 * i] = MODELS[i].model.at(4);
        profiles[2 * i + 1] = MODELS[i].model.at(2);
        i += 1;
    }
    profiles
}

pub fn owner(profile: Profile) -> Option<&'static GeometryModel> {
    MODELS
        .iter()
        .find(|entry| profile.ranks_of(entry.model).is_some())
}

pub fn model(geometry: &Geometry) -> &'static GeometryModel {
    MODELS
        .iter()
        .find(|entry| entry.geometry.id == geometry.id)
        .expect("listed geometry")
}

pub fn shape(profile: Profile) -> Option<GeometryShape> {
    let owner = owner(profile)?;
    Some(GeometryShape {
        geometry: owner.geometry,
        ranks: profile.ranks_of(owner.model)?,
    })
}

/// A served topology's SM120 kernel profile and row checks.
impl crate::contract::Shape {
    /// The kernel profile of this topology's per-rank entries; the
    /// rank-invariant entries are `config::PROFILE`'s.
    pub fn profile(self) -> Profile {
        model(self.geometry)
            .model
            .at(if self.ranks() == 2 { 2 } else { 4 })
    }

    /// Validate the row geometry with the same builders the catalog compiles.
    pub fn validate_rows(self) -> Result<(), kernel_core::contract::Refusal> {
        use kernels_sm120::row::hyper_connection::{self as row, Ngram, Streams};
        let profile = self.profile();
        let streams = Streams {
            streams: self.streams(),
            width: self.hidden(),
            rank: self.hc_rank(),
            eps_bits: self.eps().to_bits(),
            capacity: crate::contract::PREFILL_TOKENS,
        };
        row::hc_entries(profile, streams)?;
        row::ple_gate_entry(profile, streams)?;
        row::ple_conv_entry(
            profile,
            streams,
            kernel_core::contract::RecurrentPhase::Prefill,
        )?;
        let ngram = row::ngram_entry(
            profile,
            Ngram {
                orders: self.ngram_size(),
                heads_per_order: self.ngram_heads_per_order(),
                head_width: self.ngram_head_width(),
                capacity: crate::contract::PREFILL_TOKENS,
            },
            self.ngram_local_heads,
            self.ngram_local_heads,
        )?;
        row::ngram_decode_entry(&ngram, self.ngram_local_heads)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryShape {
    pub geometry: Geometry,
    pub ranks: u32,
}

impl GeometryShape {
    pub fn profile(self) -> Profile {
        model(&self.geometry)
            .model
            .try_at(self.ranks)
            .expect("supported ranks")
    }
    pub fn shape(self) -> kernel_core::rank_shape::RankShape {
        self.profile().rank_shape().expect("model profile")
    }
    pub fn q_heads(self) -> u32 {
        self.geometry.attention_q_heads() / self.ranks
    }
    pub fn q_proj_rows(self) -> u32 {
        self.q_heads() * 2 * self.geometry.head_dim()
    }
    pub fn kv_heads(self) -> u32 {
        (self.geometry.attention_kv_heads() / self.ranks).max(1)
    }
    pub fn attn_row(self) -> u32 {
        self.q_proj_rows() + 2 * self.kv_heads() * self.geometry.head_dim()
    }
    pub fn attn_value(self) -> u32 {
        self.q_heads() * self.geometry.head_dim()
    }
    pub fn linear_key_heads(self) -> u32 {
        self.geometry.linear_key_heads() / self.ranks
    }
    pub fn linear_value_heads(self) -> u32 {
        self.geometry.linear_value_heads() / self.ranks
    }
    pub fn linear_qkv(self) -> u32 {
        (2 * self.linear_key_heads() + self.linear_value_heads()) * self.geometry.linear_head()
    }
    pub fn linear_value(self) -> u32 {
        self.linear_value_heads() * self.geometry.linear_value_head()
    }
    pub fn lin_ab_stride(self) -> u32 {
        (2 * self.linear_value_heads()).next_multiple_of(8)
    }
    pub fn local_experts(self) -> u32 {
        self.geometry.experts() / self.ranks
    }
    pub fn shared_padded(self) -> u32 {
        (self.geometry.shared_expert_intermediate() / self.ranks).next_multiple_of(128)
    }
    pub fn vocab_shard(self) -> u32 {
        self.geometry.vocab() / self.ranks
    }
    pub fn ngram_heads(self) -> u32 {
        self.geometry.ngram_heads() / self.ranks
    }
    pub fn conv_entry_bytes(self) -> u64 {
        u64::from(self.geometry.conv_taps() - 1) * u64::from(self.linear_qkv()) * 2
    }
    pub fn delta_entry_bytes(self) -> u64 {
        u64::from(self.linear_value_heads())
            * u64::from(self.geometry.linear_head())
            * u64::from(self.geometry.linear_value_head())
            * 4
    }
}

pub fn hc_defines(g: Geometry) -> Vec<(String, String)> {
    [
        ("CHOP_QWEN38_HIDDEN", g.hidden()),
        ("CHOP_QWEN38_STREAMS", g.streams()),
        ("CHOP_QWEN38_HC_RANK", g.hc_rank()),
    ]
    .map(|(name, value)| (name.into(), value.to_string()))
    .into()
}

pub fn ple_defines(g: Geometry) -> Vec<(String, String)> {
    [
        ("CHOP_QWEN38_PLE_CONV_TAPS", g.ple_conv_taps()),
        ("CHOP_QWEN38_PLE_CONV_DILATION", g.ple_conv_dilation()),
        (
            "CHOP_QWEN38_PLE_CONV_ROWS",
            (g.ple_conv_taps() - 1) * g.ple_conv_dilation(),
        ),
    ]
    .map(|(name, value)| (name.into(), value.to_string()))
    .into()
}

pub fn ngram_defines(g: Geometry) -> Vec<(String, String)> {
    [
        ("CHOP_QWEN38_NGRAM_EOS", g.eos_token_id()),
        ("CHOP_QWEN38_NGRAM_SIZE", g.ngram_orders()),
        ("CHOP_QWEN38_NGRAM_HEADS", g.ngram_heads()),
        (
            "CHOP_QWEN38_NGRAM_HEADS_PER_ORDER",
            g.ngram_heads_per_order(),
        ),
        ("CHOP_QWEN38_NGRAM_HEAD_WIDTH", g.ngram_head_width()),
        ("CHOP_QWEN38_NGRAM_DEFAULT_LOCAL_HEADS", g.ngram_heads() / 4),
    ]
    .map(|(name, value)| (name.into(), value.to_string()))
    .into()
}

impl GeometryShape {
    pub fn bundle_defines(self) -> Vec<(String, String)> {
        let g = self.geometry;
        let mut defines = vec![
            (
                "CHOP_QWEN38_LINEAR_LAYERS".into(),
                (u64::from(g.layers()) - g.full_layers()).to_string(),
            ),
            (
                "CHOP_QWEN38_MAX_VERIFY_ROWS".into(),
                crate::contract::MAX_VERIFY_ROWS.to_string(),
            ),
            (
                "CHOP_QWEN38_RING_SLOT_KEY_BYTES".into(),
                format!(
                    "{}ull",
                    u64::from(crate::contract::QSA_RING) * u64::from(g.index_dim()) * 2
                ),
            ),
            (
                "CHOP_QWEN38_RING_SLOT_POSITION_BYTES".into(),
                format!("{}ull", u64::from(crate::contract::QSA_RING) * 3 * 4),
            ),
            (
                "CHOP_QWEN38_CONV_ENTRY_BYTES".into(),
                format!("{}ull", self.conv_entry_bytes()),
            ),
            (
                "CHOP_QWEN38_DELTA_ENTRY_BYTES".into(),
                format!("{}ull", self.delta_entry_bytes()),
            ),
        ];
        defines.extend(hc_defines(g));
        defines.extend(ple_defines(g));
        defines.extend(ngram_defines(g));
        defines
    }
    pub fn speculation_defines(self) -> Vec<(String, String)> {
        let mut defines = vec![
            ("CHOP_QWEN38_RANKS".into(), self.ranks.to_string()),
            (
                "CHOP_QWEN38_NGRAM_LOCAL_HEADS".into(),
                self.ngram_heads().to_string(),
            ),
            (
                "CHOP_QWEN38_SHARD_VOCAB".into(),
                self.vocab_shard().to_string(),
            ),
        ];
        defines.extend(self.bundle_defines());
        defines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listed_profiles_resolve_their_own_geometry() {
        assert_eq!(MODELS.len(), GEOMETRIES.len());
        assert_eq!(MODELS[0].model, super::super::config::MODEL);
        for model in MODELS {
            for profile in model.model.profiles() {
                let shape = shape(profile).unwrap();
                assert_eq!(shape.geometry, model.geometry);
                shape.shape().validate().unwrap();
                assert_eq!(shape.shape().config().hidden, model.geometry.hidden());
                assert_eq!(owner(profile).unwrap().hc.width, model.geometry.hidden());
                assert!(model.hc.up_mix_symbol.contains(model.geometry.kernel_tag));
            }
        }
    }

    /// The contract's neutral topology facts are the SM120 profiles': the
    /// tag, the configuration values and the served rank counts.
    #[test]
    fn shapes_name_their_profiles_tag_and_configuration() {
        assert_eq!(
            crate::contract::SERVED_RANKS,
            [4, 2].map(|ranks| MODELS[0].model.at(ranks).ranks().unwrap())
        );
        for model in MODELS {
            assert_eq!(
                crate::geometry::model_config(&model.geometry),
                model.model.config()
            );
            for ranks in crate::contract::SERVED_RANKS {
                let shape = crate::contract::Shape::new(
                    GEOMETRIES
                        .iter()
                        .find(|g| g.id == model.geometry.id)
                        .unwrap(),
                    crate::contract::Ranks::new(ranks).unwrap(),
                );
                assert_eq!(shape.tag(), shape.profile().tag());
                assert_eq!(
                    shape.model_config(),
                    shape.profile().rank_shape().unwrap().config()
                );
            }
        }
    }
}
