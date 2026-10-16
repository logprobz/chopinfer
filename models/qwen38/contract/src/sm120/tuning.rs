//! Geometry-owned measured knobs and checked inheritance.

use crate::geometry::{GEOMETRIES, Geometry};
use kernels_sm120::gemm::fused::cluster_decode_shape;

#[derive(Clone, Debug)]
pub struct Decision<T> {
    pub value: T,
    pub status: Option<String>,
}

pub fn measured(g: Geometry) -> bool {
    g.tuning.is_some()
}

pub fn nearest(g: Geometry) -> Vec<Geometry> {
    kernel_build::tuning::nearest_first(
        Some(g.id),
        GEOMETRIES.iter().copied().filter(|g| measured(*g)),
        |candidate| candidate.id,
        |candidate| {
            u64::from(g.hidden().abs_diff(candidate.hidden()))
                + u64::from(g.experts().abs_diff(candidate.experts()))
                + u64::from(
                    g.expert_intermediate()
                        .abs_diff(candidate.expert_intermediate()),
                )
                + u64::from(g.top_k().abs_diff(candidate.top_k()))
        },
    )
}

fn status(g: Geometry, source: Option<Geometry>, key: &str) -> Option<String> {
    if measured(g) {
        None
    } else {
        Some(match source {
            Some(source) => format!("untuned: inherited {} row {key}", source.id),
            None => format!("untuned: model default {key}"),
        })
    }
}

pub fn dense(g: Geometry) -> &'static serde_json::Value {
    static TABLES: std::sync::OnceLock<std::collections::BTreeMap<&str, serde_json::Value>> =
        std::sync::OnceLock::new();
    &TABLES.get_or_init(|| {
        GEOMETRIES
            .iter()
            .filter_map(|g| {
                g.tuning.map(|data| {
                    (
                        g.id,
                        serde_json::from_str(data.dense).expect("dense tuning JSON"),
                    )
                })
            })
            .collect()
    })[g.id]
}

fn count(g: Geometry, section: &str, name: &str, k: u32, granule: u32) -> Decision<u32> {
    let key = format!("{section}.{name}");
    for candidate in nearest(g) {
        if let Some(value) = dense(candidate)[section][name]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            && value > 0
            && (candidate.id == g.id
                || value
                    .checked_mul(granule)
                    .is_some_and(|n| k.is_multiple_of(n)))
        {
            return Decision {
                value,
                status: status(g, Some(candidate), &key),
            };
        }
    }
    Decision {
        value: 1,
        status: status(g, None, &key),
    }
}

pub fn prefill_decision(g: Geometry, name: &str, k: u32) -> Decision<u32> {
    count(g, "prefill_segments", name, k, 64)
}

pub fn prefill(g: Geometry, name: &str, k: u32) -> u8 {
    u8::try_from(prefill_decision(g, name, k).value).expect("prefill segment count fits u8")
}

pub fn skinny_decision(g: Geometry, name: &str, k: u32) -> Decision<u32> {
    count(g, "skinny_splits", name, k, 256)
}

pub fn skinny(g: Geometry, name: &str, k: u32) -> u32 {
    skinny_decision(g, name, k).value
}

pub fn cluster_decisions(
    g: Geometry,
    name: &str,
    ranks: u32,
    k: u32,
) -> Result<[Decision<u32>; 2], String> {
    let key = if name == "out" && ranks == 2 {
        "out_tp2"
    } else {
        name
    };
    let groups = [[1, 2, 4], [8, 16, 32]];
    let mut decisions = Vec::new();
    for (index, buckets) in groups.into_iter().enumerate() {
        let legal = |blocks| {
            (1..=8).contains(&blocks)
                && buckets
                    .into_iter()
                    .all(|m| cluster_decode_shape(k, m, blocks).is_some())
        };
        let inherited = nearest(g).into_iter().find_map(|candidate| {
            let blocks = dense(candidate)["cluster_warp_blocks"][key][index]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())?;
            legal(blocks).then_some((blocks, candidate))
        });
        let row = format!("cluster_warp_blocks.{key}[{index}]");
        let decision = if let Some((value, source)) = inherited {
            Decision {
                value,
                status: status(g, Some(source), &row),
            }
        } else {
            let value = (1..=8).find(|&blocks| legal(blocks)).ok_or_else(|| format!(
                "geometry {} role {name}: cluster decode K={k} has no legal warp-block count in 1..=8 for buckets {buckets:?} (at most 8 CTAs and 101376 shared bytes)", g.id
            ))?;
            Decision {
                value,
                status: status(
                    g,
                    None,
                    &format!("{row}: smallest legal warp-block count {value}"),
                ),
            }
        };
        decisions.push(decision);
    }
    Ok(decisions.try_into().expect("two bucket groups"))
}

pub fn cluster(g: Geometry, name: &str, ranks: u32, k: u32) -> Result<(u32, u32), String> {
    let [small, large] = cluster_decisions(g, name, ranks, k)?;
    Ok((small.value, large.value))
}

/// Report only entries whose knobs were chosen for an unmeasured geometry.
pub fn entry_status() -> std::collections::BTreeMap<String, String> {
    use super::geometry::{GeometryShape, MODELS};
    let mut statuses = super::moe::tuning_status().clone();
    let mut insert = |symbol: String, decision: Decision<u32>| {
        if let Some(status) = decision.status {
            statuses.insert(symbol, status);
        }
    };
    for owner in MODELS {
        let g = owner.geometry;
        if measured(g) {
            continue;
        }
        for profile in owner.model.profiles() {
            let ranks = profile.ranks_of(owner.model).unwrap();
            let shape = GeometryShape { geometry: g, ranks };
            let tag = profile.tag();
            let mut dense = super::dense::shapes_for(shape);
            dense.extend([
                (
                    "shared_gate_up",
                    0,
                    g.hidden(),
                    kernel_core::contract::DType::Bf16,
                    0,
                ),
                (
                    "shared_down",
                    0,
                    shape.shared_padded(),
                    kernel_core::contract::DType::Bf16,
                    0,
                ),
            ]);
            for (name, _, k, ..) in dense {
                insert(
                    format!("chop_gemm_{tag}_{name}_prefill_v1"),
                    prefill_decision(g, name, k),
                );
            }
            insert(
                format!("chop_gemm_{tag}_lin_ab_v1"),
                skinny_decision(g, "lin_ab", g.hidden()),
            );
            if ranks == 4 {
                insert(
                    format!("chop_gemm_{tag}_hc_inject_v1"),
                    skinny_decision(g, "hc_inject", g.hidden() * g.streams()),
                );
            }
            for (name, _, k, ..) in
                super::gemm::cluster_shapes(shape, ranks == 4).expect("validated cluster catalog")
            {
                let choices =
                    cluster_decisions(g, name, ranks, k).expect("validated cluster settings");
                for bucket in crate::contract::DECODE_ENTRY_BUCKETS {
                    insert(
                        format!("chop_gemm_{tag}_{name}_c{bucket}_v1"),
                        choices[usize::from(bucket > 4)].clone(),
                    );
                }
            }
        }
    }
    statuses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_fallback_covers_every_bucket() {
        let g = Geometry {
            id: "untuned",
            tuning: None,
            ..crate::geometry::FLASH
        };
        for hidden in (256..=8192).step_by(256) {
            let k = 4 * hidden;
            let expected = [[1, 2, 4], [8, 16, 32]].map(|buckets| {
                (1..=8).find(|&wb| {
                    buckets
                        .into_iter()
                        .all(|m| cluster_decode_shape(k, m, wb).is_some())
                })
            });
            match cluster(g, "hc_in", 4, k) {
                Ok(blocks) => {
                    assert!(expected.iter().all(Option::is_some));
                    for m in crate::contract::DECODE_ENTRY_BUCKETS {
                        let wb = if m <= 4 { blocks.0 } else { blocks.1 };
                        assert!(
                            cluster_decode_shape(k, m, wb).is_some(),
                            "hidden={hidden}, bucket={m}, warp_blocks={wb}"
                        );
                    }
                }
                Err(error) => {
                    assert!(expected.iter().any(Option::is_none));
                    assert!(error.contains("hc_in") && error.contains(&format!("K={k}")));
                }
            }
        }
        assert_eq!(cluster(g, "hc_in", 4, 2304 * 4).unwrap(), (3, 3));
        assert_eq!(cluster(g, "hc_in", 4, 5120 * 4).unwrap(), (4, 5));
        assert!(cluster(g, "hc_in", 4, 2816 * 4).is_err());
    }

    #[test]
    fn missing_cluster_row_searches_instead_of_inventing_measured_knobs() {
        let g = Geometry {
            id: "untuned",
            tuning: None,
            ..crate::geometry::FLASH
        };
        let choices = cluster_decisions(g, "new_role", 4, 4096).unwrap();
        assert!(
            choices
                .iter()
                .all(|d| d.status.as_ref().unwrap().contains("model default"))
        );
        assert!(
            skinny_decision(g, "lin_ab", 1024)
                .status
                .unwrap()
                .contains("model default")
        );
        assert!(
            prefill_decision(g, "lin_qkv", 1024)
                .status
                .unwrap()
                .contains("inherited flash row prefill_segments.lin_qkv")
        );
    }
}
