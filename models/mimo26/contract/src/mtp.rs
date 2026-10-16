//! Validate the MTP rank image before aliasing its layer 0 into target adapters.

use crate::index::RankIndex;

pub const IMAGE_ROLE: &[u8] = b"mimo.drafter-rank-images.v1";

use crate::Tp;
pub(crate) use crate::drafters::mtp_fields;

pub fn validate(index: &RankIndex, tp: Tp) -> Result<(), &'static str> {
    if index.rank >= tp.ranks()
        || index.views.len() != tp.geometry.mtp_layers() as usize * mtp_fields(tp).len()
    {
        return Err("MTP image must contain every configured layer completely");
    }
    let mut extents = Vec::new();
    let fields = mtp_fields(tp);
    for layer in 0..tp.geometry.mtp_layers() {
        for (suffix, shape, dtype, layout) in &fields {
            let view = index
                .views
                .get(&format!("model.mtp.layers.{layer}.{suffix}"))
                .ok_or("MTP tensor missing")?;
            let bytes = shape.iter().product::<u64>()
                * match *dtype {
                    "F32" => 4,
                    "BF16" => 2,
                    _ => 1,
                };
            if view.shape != *shape
                || view.dtype != *dtype
                || view.layout_id != *layout
                || view.bytes != bytes
            {
                return Err("MTP tensor geometry or layout mismatch");
            }
            let end = view
                .offset
                .checked_add(bytes)
                .filter(|&end| end <= index.image_bytes)
                .ok_or("MTP tensor exceeds image")?;
            extents.push((view.offset, end));
        }
    }
    extents.sort_unstable();
    if extents.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("MTP tensor extents overlap");
    }
    Ok(())
}
