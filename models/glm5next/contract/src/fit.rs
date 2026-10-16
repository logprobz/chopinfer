//! Fit checks: which tensor-parallel sizes `glm5next.v1` serves, and what a
//! TP4 device holds before the serving KV pool is sized.
//!
//! Only TP4 is served. The block-FP8 checkpoint does not fit at TP2: each
//! GPU would hold half the routed experts, more weight bytes than a device
//! has. The NVFP4 checkpoint's TP2 weights fit, and TP8 would fit, but both
//! are out of scope for this package. Each is refused by name before
//! loading. The weights are those of the checkpoint's quantization layout.
use crate::catalog::checkpoint_catalog;
use crate::contract::{Shape, TOPOLOGY_ID};
use crate::geometry::Geometry;
use crate::quantization::{FP8_BLOCK128, Layout};
use crate::weight_plan::rank_tensor;

pub use model_compiler_sdk::device::HOST_RESERVE_BYTES;

const GIB: f64 = (1u64 << 30) as f64;

/// What one TP4 rank holds before the serving KV pool, by formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fit {
    pub weight_bytes: u64,
    /// State slots times the committed copy and its verify checkpoints.
    pub state_bytes: u64,
    /// Device bytes left after weights, state and the host reserve.
    pub free_bytes: u64,
}

/// Weight bytes rank 0 of `ranks` holds of the block-FP8 checkpoint.
pub fn weight_bytes(g: &Geometry, ranks: u32) -> Result<u64, String> {
    checkpoint_weight_bytes(g, &FP8_BLOCK128, ranks)
}

/// Weight bytes rank 0 of `ranks` holds of a checkpoint stored in `layout`:
/// every catalog tensor's share, MTP and vision included.
pub fn checkpoint_weight_bytes(g: &Geometry, layout: &Layout, ranks: u32) -> Result<u64, String> {
    checkpoint_catalog(g, layout)
        .iter()
        .try_fold(0u64, |total, spec| {
            Ok(total + rank_tensor(g, spec, ranks, 0)?.map_or(0, |part| part.bytes()))
        })
}

/// Refuse every tensor-parallel size but 4 for the block-FP8 checkpoint.
pub fn check_ranks(g: &Geometry, ranks: u32, device_bytes: u64) -> Result<(), String> {
    check_checkpoint_ranks(g, &FP8_BLOCK128, ranks, device_bytes)
}

/// Refuse every tensor-parallel size but 4 for a checkpoint stored in
/// `layout`, naming why.
pub fn check_checkpoint_ranks(
    g: &Geometry,
    layout: &Layout,
    ranks: u32,
    device_bytes: u64,
) -> Result<(), String> {
    match ranks {
        4 => Ok(()),
        8 => Err(format!(
            "glm5next.v1 serves TP4 only: TP8 serving is out of scope; serve {TOPOLOGY_ID}"
        )),
        2 => {
            let weights = checkpoint_weight_bytes(g, layout, 2)?;
            if weights <= device_bytes {
                return Err(format!(
                    "glm5next.v1 serves TP4 only: TP2 serving is out of scope; serve {TOPOLOGY_ID}"
                ));
            }
            Err(format!(
                "glm5next.v1 does not fit at TP2: each GPU would hold {:.2} GiB of weights, \
                 above the device's {:.2} GiB; serve {TOPOLOGY_ID}",
                weights as f64 / GIB,
                device_bytes as f64 / GIB
            ))
        }
        _ => Err(format!(
            "unsupported tensor parallel size {ranks}: glm5next.v1 serves TP4 ({TOPOLOGY_ID})"
        )),
    }
}

/// [`check_checkpoint`] for the block-FP8 checkpoint.
pub fn check(
    shape: Shape,
    device_bytes: u64,
    slots: u32,
    mtp: bool,
    rows: u32,
) -> Result<Fit, String> {
    check_checkpoint(shape, &FP8_BLOCK128, device_bytes, slots, mtp, rows)
}

/// Check that `slots` state slots of `rows` verify rows each fit beside the
/// weights of a checkpoint stored in `layout` on a device of
/// `device_bytes`, and what remains for KV pages and working buffers.
pub fn check_checkpoint(
    shape: Shape,
    layout: &Layout,
    device_bytes: u64,
    slots: u32,
    mtp: bool,
    rows: u32,
) -> Result<Fit, String> {
    check_checkpoint_ranks(shape.geometry, layout, shape.ranks(), device_bytes)?;
    let weight_bytes = checkpoint_weight_bytes(shape.geometry, layout, shape.ranks())?;
    let state_bytes = u64::from(slots) * shape.state_slot_bytes(mtp, rows);
    let used = weight_bytes + state_bytes + HOST_RESERVE_BYTES;
    let free_bytes = device_bytes.checked_sub(used).ok_or_else(|| {
        format!(
            "glm5next.v1 does not fit: {:.2} GiB of weights and {:.2} GiB of state for {slots} \
             slots of {rows} rows exceed the device's {:.2} GiB",
            weight_bytes as f64 / GIB,
            state_bytes as f64 / GIB,
            device_bytes as f64 / GIB
        )
    })?;
    Ok(Fit {
        weight_bytes,
        state_bytes,
        free_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;

    /// 97,887 MiB, an SM120 workstation GPU.
    const DEVICE: u64 = 97_887 << 20;

    #[test]
    fn tp2_and_tp8_are_refused_by_name() {
        assert_eq!(
            check_ranks(&FLASH, 2, DEVICE).unwrap_err(),
            "glm5next.v1 does not fit at TP2: each GPU would hold 153.69 GiB of weights, \
             above the device's 95.59 GiB; serve glm5next-tp4"
        );
        assert_eq!(
            check_ranks(&FLASH, 8, DEVICE).unwrap_err(),
            "glm5next.v1 serves TP4 only: TP8 serving is out of scope; serve glm5next-tp4"
        );
        assert!(
            check_ranks(&FLASH, 1, DEVICE)
                .unwrap_err()
                .starts_with("unsupported")
        );
        assert_eq!(check_ranks(&FLASH, 4, DEVICE), Ok(()));
        // The NVFP4 checkpoint's 93.07 GiB per GPU at TP2 fit the device;
        // TP2 is refused by policy, not by size.
        let nvfp4 = &crate::quantization::NVFP4_ROUTED;
        assert!(checkpoint_weight_bytes(&FLASH, nvfp4, 2).unwrap() <= DEVICE);
        assert_eq!(
            check_checkpoint_ranks(&FLASH, nvfp4, 2, DEVICE).unwrap_err(),
            "glm5next.v1 serves TP4 only: TP2 serving is out of scope; serve glm5next-tp4"
        );
        assert_eq!(check_checkpoint_ranks(&FLASH, nvfp4, 4, DEVICE), Ok(()));
    }

    #[test]
    fn tp4_fits_32_slots_with_three_drafts() {
        let fit = check(Shape::TP4, DEVICE, 32, true, 4).unwrap();
        assert_eq!(fit.weight_bytes, 83_370_801_144);
        assert_eq!(fit.state_bytes, 32 * Shape::TP4.state_slot_bytes(true, 4));
        assert_eq!(
            fit.free_bytes,
            DEVICE - fit.weight_bytes - fit.state_bytes - HOST_RESERVE_BYTES
        );
        assert!(
            check(Shape::TP4, DEVICE, 512, true, 6)
                .unwrap_err()
                .starts_with("glm5next.v1 does not fit: ")
        );
    }
}
