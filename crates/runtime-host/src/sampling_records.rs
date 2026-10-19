//! The existing 256-byte sampler record, decoded identically for either result path.
use crate::sampling_workspace::OUTPUT_BYTES;
use runtime_bundle_abi::BundleStatusV1;
use runtime_engine::{ModelBatchItem, SampledRow};
use runtime_sampling::TokenLogprob;
type SamplingResult<T> = Result<T, BundleStatusV1>;
fn row_mask(item: &ModelBatchItem) -> Option<&[u32]> {
    item.sampling()?.constraint.as_ref()?.mask()
}
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("bounded record"),
    )
}
fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounded record"),
    )
}
/// The sampler's row status when no token survives the row's masks
/// (`cuda/sampling/sampling.cuh`).
const EMPTY_DISTRIBUTION: u32 = 2;

pub(crate) fn decode_output(
    bytes: &[u8],
    items: &[&ModelBatchItem],
    live: usize,
) -> SamplingResult<Vec<SampledRow>> {
    if items.len().checked_mul(OUTPUT_BYTES) != Some(bytes.len()) || live == 0 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut sampled = Vec::with_capacity(items.len());
    for (row, item) in items.iter().enumerate() {
        let record = &bytes[row * OUTPUT_BYTES..(row + 1) * OUTPUT_BYTES];
        match read_u32(record, 4) {
            0 => {}
            // No token survives the row's masks: the request's own failure,
            // as host sampling reports it. The engine fails that request
            // and commits the step's others.
            EMPTY_DISTRIBUTION => {
                sampled.push(SampledRow::failed(
                    item.request_id(),
                    runtime_sampling::SamplingError::AllTokensMasked,
                ));
                continue;
            }
            // A non-finite logit or malformed row parameters.
            _ => return Err(BundleStatusV1::HostError),
        }
        let work = item.sampling().ok_or(BundleStatusV1::InvalidArgument)?;
        // The kernel ranks admitted tokens; fill the masked tail on the host.
        let admitted = row_mask(item).map_or(live, |mask| {
            mask.iter().map(|word| word.count_ones() as usize).sum()
        });
        let top_count = work.parameters.top_logprobs().min(live).min(admitted);
        if top_count > 20 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut top = Vec::with_capacity(top_count);
        for index in 0..top_count {
            top.push(TokenLogprob::new(
                read_u32(record, 16 + index * 4),
                read_f64(record, 96 + index * 8),
            ));
        }
        if let Some(mask) = row_mask(item) {
            let remaining = work.parameters.top_logprobs().min(live) - top.len();
            top.extend(
                (0..live)
                    .filter(|token| mask[token / 32] & (1 << (token % 32)) == 0)
                    .take(remaining)
                    .map(|token| TokenLogprob::new(token as u32, f64::NEG_INFINITY)),
            );
        }
        sampled.push(SampledRow::new(
            item.request_id(),
            read_u32(record, 0),
            read_f64(record, 8),
            top,
        ));
    }
    Ok(sampled)
}
