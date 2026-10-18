//! Key-list poison replay over missing prefixes and invalid metadata.
use crate::{
    LabResult,
    conformance::{Expected, Images, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::contract::DType;
use kernel_core::latent_staged_window_keys::{
    self as oracle, LatentStagedWindowKeyBuffers, LatentStagedWindowKeys,
};
pub const QUERY: LatentStagedWindowKeys = LatentStagedWindowKeys {
    window: 128,
    ring_slots: 136,
    staged_slots: 256,
};
fn ints(x: &[i32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn read(i: &Images, n: &str) -> LabResult<Vec<i32>> {
    Ok(i.bytes(n)?
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
pub fn required<'a>() -> Required<'a> {
    Required::new(
        "staged keys exclude gathered holes",
        Need::rows(8),
        |b, n| b.latent_staged_window_keys(&QUERY, n),
        case,
    )
    .required()
}
fn case<'a>(op: Prepared<LatentStagedWindowKeyBuffers<View>>) -> LabResult<OpCase<'a>> {
    let roles = LatentStagedWindowKeyBuffers {
        positions: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            ints(&[8, 100, 140, 171, -2, 8, 8, 8]),
        ),
        row_items: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            ints(&[0, 1, 2, 2, 0, -2, 99, 3]),
        ),
        chunk_first: Spec::input(DType::I32, (Rows::Fixed(1), 4), ints(&[8, 100, 140, 9])),
        chunk_last: Spec::input(DType::I32, (Rows::Fixed(1), 4), ints(&[8, 103, 171, 8])),
        item_slots: Spec::input(DType::I32, (Rows::Fixed(1), 4), ints(&[0, 0, 1, -2])),
        ring_last: Spec::input(DType::I32, (Rows::Fixed(1), 2), ints(&[200, 139])),
        items: 4,
        sequences: 2,
        key_lists: Spec::output(DType::I32, (Rows::Live, 128)).filled_with(&[0x80]),
        live_rows: Spec::live(),
    };
    Ok(OpCase::new(op, 8, roles)?.oracle(|images, live| {
        let mut keys = oracle::keys(
            QUERY,
            &read(images, "positions")?[..live as usize],
            &read(images, "row_items")?[..live as usize],
            &read(images, "chunk_first")?,
            &read(images, "chunk_last")?,
            &read(images, "item_slots")?,
            &read(images, "ring_last")?,
        )
        .map_err(|e| e.to_string())?;
        keys.resize(8 * 128, 0);
        Ok(vec![("key_lists".into(), Expected::Bits(ints(&keys)))])
    }))
}
