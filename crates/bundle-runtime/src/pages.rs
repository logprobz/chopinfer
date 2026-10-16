//! Layer-interleaved views into the host-owned serving page pool.
//!
//! A serving page holds one page of tokens for every layer that keeps paged
//! state, layer after layer. Within a layer, its planes (K, V, and whatever
//! else the model pages) follow each other. A kernel addresses plane `p` of
//! layer ordinal `l` in page `n` at that plane's start plus `n * page_bytes`.

use runtime_bundle_abi::DeviceBufferSpanV1;

/// Bytes of each plane in one page of one layer, and how many layers a page
/// holds.
#[derive(Clone, Copy, Debug)]
pub struct PageGeometry<'a> {
    pub layers: u64,
    pub planes: &'a [u64],
}

impl PageGeometry<'_> {
    pub fn layer_bytes(&self) -> u64 {
        self.planes.iter().sum()
    }

    pub fn page_bytes(&self) -> u64 {
        self.layers * self.layer_bytes()
    }
}

/// One layer's planes: each span starts at the plane in page zero and runs
/// to the pool's end, so every page is inside it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerPlanes {
    pub planes: Vec<DeviceBufferSpanV1>,
    pub page_stride_bytes: u64,
    pub page_capacity: u64,
}

/// The planes of layer `ordinal` (counted among the paged layers only).
pub fn layer_planes(
    pool: DeviceBufferSpanV1,
    geometry: PageGeometry<'_>,
    ordinal: u64,
) -> Result<LayerPlanes, &'static str> {
    if ordinal >= geometry.layers {
        return Err("serving KV pool has the wrong page geometry");
    }
    planes_at(
        pool,
        geometry.page_bytes(),
        ordinal * geometry.layer_bytes(),
        geometry.planes,
    )
}

/// The planes of a layer that starts `layer_offset` bytes into every page
/// of `page_bytes`, for pages whose layers differ in size (FP8 KV contract:
/// FP8 layers beside a BF16 one).
pub fn planes_at(
    pool: DeviceBufferSpanV1,
    page_bytes: u64,
    layer_offset: u64,
    planes: &[u64],
) -> Result<LayerPlanes, &'static str> {
    let layer_end = planes
        .iter()
        .try_fold(layer_offset, |end, &bytes| end.checked_add(bytes));
    if page_bytes == 0
        || layer_end.is_none_or(|end| end > page_bytes)
        || pool.length < page_bytes
        || !pool.length.is_multiple_of(page_bytes)
    {
        return Err("serving KV pool has the wrong page geometry");
    }
    let end = pool
        .offset
        .checked_add(pool.length)
        .ok_or("serving KV pool end overflow")?;
    let mut offset = pool
        .offset
        .checked_add(layer_offset)
        .ok_or("serving KV plane offset overflow")?;
    let mut spans = Vec::with_capacity(planes.len());
    for &bytes in planes {
        spans.push(DeviceBufferSpanV1::new(pool.buffer, offset, end - offset));
        offset = offset
            .checked_add(bytes)
            .ok_or("serving KV plane offset overflow")?;
    }
    Ok(LayerPlanes {
        planes: spans,
        page_stride_bytes: page_bytes,
        page_capacity: pool.length / page_bytes,
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;

    #[test]
    fn planes_advance_inside_each_page() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let geometry = PageGeometry {
            layers: 3,
            planes: &[100, 60, 20],
        };
        let pool = DeviceBufferSpanV1::new(buffer, 4096, 10 * 540);
        let second = layer_planes(pool, geometry, 1).unwrap();
        assert_eq!(second.page_stride_bytes, 540);
        assert_eq!(second.page_capacity, 10);
        assert_eq!(
            second
                .planes
                .iter()
                .map(|span| span.offset)
                .collect::<Vec<_>>(),
            [4096 + 180, 4096 + 280, 4096 + 340]
        );
        assert!(
            second
                .planes
                .iter()
                .all(|span| span.offset + span.length == 4096 + 5400)
        );
        assert!(layer_planes(pool, geometry, 3).is_err());
        // A last, larger layer after two of the geometry's.
        let last = planes_at(pool, 540, 360, &[100, 80]).unwrap();
        assert_eq!(
            last.planes
                .iter()
                .map(|span| span.offset)
                .collect::<Vec<_>>(),
            [4096 + 360, 4096 + 460]
        );
        assert!(planes_at(pool, 540, 360, &[100, 81]).is_err());
        let short = DeviceBufferSpanV1::new(buffer, 0, 541);
        assert!(layer_planes(short, geometry, 0).is_err());
    }
}
