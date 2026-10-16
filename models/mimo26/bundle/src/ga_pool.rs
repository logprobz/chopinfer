//! Layer-local K and V views into the host-owned serving page pool, over the
//! shared layer-interleaved page view.

use bundle_runtime::pages::{PageGeometry, layer_planes};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::contract::{BF16_BYTES, PAGE_TOKENS};
#[cfg(test)]
use crate::contract::{GA_LAYERS, QK_DIM};
use crate::topology::Tp;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerPages {
    pub k: DeviceBufferSpanV1,
    pub v: DeviceBufferSpanV1,
    pub k_page_stride_bytes: u64,
    pub v_page_stride_bytes: u64,
    pub page_capacity: u64,
}

/// The K and V scale bits of global layer `layer` from a format's per-layer
/// scales; an empty table, or a layer outside it, is 1.0.
pub fn scale_bits_for(tp: Tp, scales: &[[f32; 2]], layer: u32) -> [u32; 2] {
    let [k, v] = (tp.geometry.is_global(layer) && layer < tp.geometry.layers())
        .then(|| tp.geometry.global_ordinal(layer))
        .and_then(|ordinal| scales.get(ordinal))
        .copied()
        .unwrap_or([1.0, 1.0]);
    [k.to_bits(), v.to_bits()]
}

pub fn layer_pages(
    pool: DeviceBufferSpanV1,
    layer: u32,
    fp8: bool,
    tp: Tp,
) -> Result<LayerPages, &'static str> {
    if layer >= tp.geometry.layers() || !tp.geometry.is_global(layer) {
        return Err("MiMo layer has no global-attention pages");
    }
    let ordinal = tp.geometry.global_ordinal(layer) as u64;
    let element = if fp8 { 1 } else { BF16_BYTES };
    let heads = tp.kv_heads(true);
    let k_bytes = PAGE_TOKENS * heads * tp.geometry.head_dim() as u64 * element;
    let v_bytes = PAGE_TOKENS * heads * tp.geometry.value_head_dim() as u64 * element;
    let geometry = PageGeometry {
        layers: tp.geometry.global_layers() as u64,
        planes: &[k_bytes, v_bytes],
    };
    if geometry.page_bytes() != crate::topology::ga_page_bytes(tp, fp8) {
        return Err("MiMo serving KV pool has the wrong page geometry");
    }
    let planes = layer_planes(pool, geometry, ordinal)?;
    Ok(LayerPages {
        k: planes.planes[0],
        v: planes.planes[1],
        k_page_stride_bytes: planes.page_stride_bytes,
        v_page_stride_bytes: planes.page_stride_bytes,
        page_capacity: planes.page_capacity,
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;
    use crate::contract::{GA_PAGE_BYTES_BF16, GA_PAGE_BYTES_FP8};

    #[test]
    fn scales_and_planes_use_the_selected_layer_ordinal_and_heads() {
        use model_mimo26_contract::geometry::{
            FeedForwardKind, Geometry, Layer, LayerKind, V26_FLASH,
        };
        const GEOMETRY: Geometry = {
            let mut g = V26_FLASH;
            g.shape.text.layers = &[
                Layer {
                    attention: LayerKind::Global,
                    feed_forward: FeedForwardKind::Dense,
                },
                Layer {
                    attention: LayerKind::Sliding,
                    feed_forward: FeedForwardKind::Moe,
                },
                Layer {
                    attention: LayerKind::Global,
                    feed_forward: FeedForwardKind::Moe,
                },
            ];
            g.shape.global.kv_heads = 8;
            g
        };
        let tp = Tp::of_geometry(&GEOMETRY, 4).unwrap();
        let scales = [[1.25_f32, 1.5_f32], [2.0_f32, 2.5_f32]];
        for (ordinal, layer) in [0, 2].into_iter().enumerate() {
            assert_eq!(
                scale_bits_for(tp, &scales, layer),
                scales[ordinal].map(f32::to_bits)
            );
            let page = crate::topology::ga_page_bytes(tp, false);
            let pool =
                DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 64, page * 3);
            let planes = layer_pages(pool, layer, false, tp).unwrap();
            let k_bytes = PAGE_TOKENS * tp.kv_heads(true) * tp.geometry.head_dim() as u64 * 2;
            assert_eq!(planes.k.offset, 64 + ordinal as u64 * page / 2);
            assert_eq!(planes.v.offset, planes.k.offset + k_bytes);
        }
    }

    #[test]
    fn layer_planes_advance_inside_each_host_page() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for (fp8, page_bytes, k_bytes) in [
            (false, GA_PAGE_BYTES_BF16, PAGE_TOKENS * QK_DIM * BF16_BYTES),
            (true, GA_PAGE_BYTES_FP8, PAGE_TOKENS * QK_DIM),
        ] {
            let pool = DeviceBufferSpanV1::new(buffer, 4096, 128 * page_bytes);
            let first = layer_pages(pool, 0, fp8, Tp::TP4).unwrap();
            let second = layer_pages(pool, 5, fp8, Tp::TP4).unwrap();
            assert_eq!(first.k.offset, 4096);
            assert_eq!(first.v.offset, 4096 + k_bytes);
            assert_eq!(second.k.offset, 4096 + page_bytes / GA_LAYERS);
            assert_eq!(first.k_page_stride_bytes, page_bytes);
            assert_eq!(first.v_page_stride_bytes, page_bytes);
            assert_eq!(second.page_capacity, 128);
            assert!(layer_pages(pool, 1, fp8, Tp::TP4).is_err());
        }
    }

    #[test]
    fn pages_hold_the_rank_ga_heads() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let page_bytes = crate::topology::ga_page_bytes(Tp::TP2, false);
        let pool = DeviceBufferSpanV1::new(buffer, 0, 16 * page_bytes);
        let pages = layer_pages(pool, 0, false, Tp::TP2).unwrap();
        assert_eq!(pages.v.offset, 2 * PAGE_TOKENS * QK_DIM * BF16_BYTES);
        assert!(layer_pages(pool, 0, false, Tp::TP4).is_ok());
        assert_eq!(
            page_bytes,
            2 * crate::topology::ga_page_bytes(Tp::TP4, false)
        );
    }
}
