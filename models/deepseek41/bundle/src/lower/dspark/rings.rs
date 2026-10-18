//! Checked state views and copies of committed rings into draft scratch.

use super::ops::slice;
use super::{BLOCK_ROWS, Window};
use crate::buffers::StateSpans;
use crate::dspark::features::Copy;
use crate::shape::Shape;
use bundle_runtime::state::Extent;
use model_deepseek41_contract::state::{self, KvStorage};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

pub fn committed(
    shape: &Shape,
    extent: Extent,
    spans: &StateSpans,
    stage: u32,
) -> Result<Window, String> {
    let g = shape.geometry;
    if stage >= g.dspark_stages() || extent.bucket == 0 {
        return Err("DSpark ring stage or slot extent is invalid".into());
    }
    let declaration = state::request_state(g, KvStorage::Native, true);
    let ring = declaration
        .buffers
        .iter()
        .find(|b| b.name == "dspark.ring")
        .ok_or("DSpark ring is not declared")?;
    let span = spans
        .named("dspark.ring")
        .ok_or("DSpark state has no committed ring")?;
    let codes = ring.planes[0];
    let scales = ring.planes[1];
    let slots = u64::from(extent.bucket);
    let stride = codes
        .checked_add(scales)
        .and_then(|n| n.checked_mul(slots))
        .ok_or("DSpark ring stage stride overflows")?;
    let base = stride
        .checked_mul(u64::from(stage))
        .ok_or("DSpark ring stage offset overflows")?;
    let code_bytes = codes
        .checked_mul(slots)
        .ok_or("DSpark ring codes overflow")?;
    let scale_bytes = scales
        .checked_mul(slots)
        .ok_or("DSpark ring scales overflow")?;
    Ok(Window {
        codes: slice(span, base, code_bytes)?,
        scales: slice(
            span,
            base.checked_add(code_bytes)
                .ok_or("DSpark scale offset overflows")?,
            scale_bytes,
        )?,
        sequences: extent.bucket,
        cells: g.shape.attention.window,
        code_stride: codes,
        scale_stride: scales,
    })
}

fn overlaps(a: Span, b: Span) -> Result<bool, String> {
    let a_end = a
        .offset
        .checked_add(a.length)
        .ok_or("DSpark ring address overflows")?;
    let b_end = b
        .offset
        .checked_add(b.length)
        .ok_or("DSpark staged address overflows")?;
    Ok(a.buffer == b.buffer && a.offset < b_end && b.offset < a_end)
}

/// Preserve physical slot order, exactly as WindowBlockVisibility indexes
/// it. Cells outside the live main range are copied but the key lists omit
/// them, and no draft write can modify the durable source ring.
pub fn stage(main: Window, draft: Window, slots: &[u32], latent: u32) -> Result<Vec<Copy>, String> {
    if latent == 0
        || !latent.is_multiple_of(32)
        || main.cells.checked_add(BLOCK_ROWS) != Some(draft.cells)
        || slots.len() > draft.sequences as usize
        || slots.iter().any(|&slot| slot >= main.sequences)
    {
        return Err("DSpark ring staging has invalid cells, slots or width".into());
    }
    for source in [main.codes, main.scales] {
        for target in [draft.codes, draft.scales] {
            if overlaps(source, target)? {
                return Err("DSpark draft storage overlaps the committed ring".into());
            }
        }
    }
    if overlaps(main.codes, main.scales)? || overlaps(draft.codes, draft.scales)? {
        return Err("DSpark cache code and scale planes overlap".into());
    }
    let mut copies = Vec::new();
    for (item, &slot) in slots.iter().enumerate() {
        for (source, target, source_stride, target_stride, width) in [
            (
                main.codes,
                draft.codes,
                main.code_stride,
                draft.code_stride,
                latent,
            ),
            (
                main.scales,
                draft.scales,
                main.scale_stride,
                draft.scale_stride,
                latent / 32,
            ),
        ] {
            let bytes = u64::from(main.cells) * u64::from(width);
            if source_stride < bytes || target_stride < u64::from(draft.cells) * u64::from(width) {
                return Err("DSpark cache stride omits cells".into());
            }
            let from = u64::from(slot)
                .checked_mul(source_stride)
                .ok_or("DSpark source offset overflows")?;
            let to = (item as u64)
                .checked_mul(target_stride)
                .ok_or("DSpark draft offset overflows")?;
            copies.push(Copy {
                source: slice(source, from, bytes)?,
                destination: slice(target, to, bytes)?,
            });
        }
    }
    Ok(copies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;
    fn window(id: u32, cells: u32, sequences: u32) -> Window {
        let codes = u64::from(cells) * 512;
        let scales = codes / 32;
        Window {
            codes: Span::new(
                BundleDeviceBufferHandleV1::from_parts(1, id),
                0,
                codes * u64::from(sequences),
            ),
            scales: Span::new(
                BundleDeviceBufferHandleV1::from_parts(1, id + 1),
                0,
                scales * u64::from(sequences),
            ),
            cells,
            sequences,
            code_stride: codes,
            scale_stride: scales,
        }
    }
    #[test]
    fn staging_copies_the_physical_window_and_never_the_provisional_tail() {
        let main = window(1, 128, 4);
        let draft = window(3, 133, 2);
        let copies = stage(main, draft, &[3, 1], 512).unwrap();
        assert_eq!(copies.len(), 4);
        assert_eq!(copies[0].source.offset, 3 * 128 * 512);
        assert_eq!(copies[0].destination.offset, 0);
        assert_eq!(copies[2].source.offset, 128 * 512);
        assert_eq!(copies[2].destination.offset, 133 * 512);
        assert_eq!(copies[2].destination.length, 128 * 512);
        assert!(
            stage(
                main,
                Window {
                    codes: main.codes,
                    ..draft
                },
                &[0],
                512
            )
            .is_err()
        );
        assert!(stage(main, draft, &[4], 512).is_err());
        assert!(
            stage(
                main,
                Window {
                    code_stride: 128 * 512,
                    ..draft
                },
                &[0],
                512
            )
            .is_err()
        );
    }
}
