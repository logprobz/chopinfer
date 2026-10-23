//! Census scaffolding of a plan that stores encoded scale planes
//! (`chop_bundle_weight_census_v1`, `kernel_core::precision::scale_offset`).
//!
//! Whether a section may be encoded follows from the operand's admitted
//! format, never from a model declaration: one matrix's untiled 1-byte local
//! scales, UE8M0 for MXFP4 and MXFP8, E4M3 for NVFP4. The census measures
//! each raw section; [`bank_width`] then picks the width that stores a bank
//! smallest, or keeps it raw when no width is smaller.
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::scale_offset::{self, ScaleCode, WIDTHS};
use kernel_core::weight_transform::{PackedOperand, PackedSection, ScaleEncoding};
use runtime_bundle_abi::weight_plan::{
    CENSUS_SCALE_ROW_OFFSET_V1, CensusItemV1, WeightCensusV1, WeightPlanRequestV1, WeightRecordV1,
};
use sha2::{Digest, Sha256};

/// The code `operand`'s local scales encode as, or `None` when its section
/// cannot be encoded.
#[must_use]
pub fn scale_code(operand: &PackedOperand) -> Option<ScaleCode> {
    let plan = &operand.plan;
    if plan.experts != 1 || plan.projections != 1 || plan.scale_width() != 1 || plan.tiled_scales()
    {
        return None;
    }
    match (plan.format, plan.local_scale_dtype) {
        (WeightFormat::Mxfp4 | WeightFormat::Mxfp8, Some(DType::Ue8M0)) => Some(ScaleCode::Ue8m0),
        (WeightFormat::Nvfp4, Some(DType::Fp8E4M3 | DType::Ue4M3)) => Some(ScaleCode::E4m3),
        _ => None,
    }
}

/// The `[rows, groups]` of `operand`'s raw local-scale section.
#[must_use]
pub fn scale_shape(operand: &PackedOperand) -> (u64, u64) {
    let (rows, groups) = operand.plan.scale_shape();
    (rows as u64, groups as u64)
}

/// The census item measuring `record`, a raw local-scale record of an
/// operand [`scale_code`] admits, placed nowhere.
pub fn scale_item(operand: &PackedOperand, record: WeightRecordV1) -> Result<CensusItemV1, String> {
    if operand.section != PackedSection::LocalScales
        || operand.local_encoding.is_some()
        || scale_code(operand).is_none()
    {
        return Err("a census measures raw local-scale sections an encoding admits".into());
    }
    let (rows, groups) = scale_shape(operand);
    if rows.checked_mul(groups) != Some(record.byte_length) {
        return Err("a local-scale record differs from its scale grid".into());
    }
    Ok(CensusItemV1 {
        kind: CENSUS_SCALE_ROW_OFFSET_V1,
        rows,
        groups,
        record: WeightRecordV1 {
            arena_offset: 0,
            ..record
        },
    })
}

/// The SHA-256 of `census` as the bundle emits it.
#[must_use]
pub fn document_sha256(census: &WeightCensusV1) -> [u8; 32] {
    Sha256::digest(census.encode()).into()
}

/// The counts `request` carries for `census`, this bundle's census of the
/// request's rank, in item order: `None` when the host measured nothing
/// (it skipped the census), refused when they answer another census.
pub fn counts<'a>(
    request: &'a WeightPlanRequestV1,
    census: &WeightCensusV1,
) -> Result<Option<&'a [[u64; 3]]>, String> {
    if request.census.is_empty() {
        return Ok(None);
    }
    let measured = request
        .census_of(census.rank)
        .ok_or_else(|| format!("the request's census has no rank {}", census.rank))?;
    if measured.document_sha256 != document_sha256(census)
        || measured.counts.len() != census.items.len()
    {
        return Err(format!(
            "rank {}'s census counts answer another census than this plan's",
            census.rank
        ));
    }
    Ok(Some(&measured.counts))
}

/// The width at which a bank of matrices, each `(rows, groups)` with its
/// measured counts, plus its offset table, stores fewest bytes, or `None`
/// when no width stores fewer than the raw bank.
pub fn bank_width(matrices: &[((u64, u64), [u64; 3])]) -> Result<Option<u32>, String> {
    let raw: u64 = matrices
        .iter()
        .map(|((rows, groups), _)| rows * groups)
        .sum();
    let mut best: Option<(u64, u32)> = None;
    for (slot, width) in WIDTHS.iter().enumerate() {
        let blocks = matrices
            .iter()
            .map(|((rows, groups), counts)| {
                scale_offset::encoded_bytes(*rows, *groups, *width, counts[slot])
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let bytes = scale_offset::bank_layout(&blocks)
            .map_err(|e| e.to_string())?
            .bytes;
        if bytes < raw && best.is_none_or(|(least, _)| bytes < least) {
            best = Some((bytes, *width));
        }
    }
    Ok(best.map(|(_, width)| width))
}

/// The encoding of one matrix of `code` at `width`, from its counts.
#[must_use]
pub fn encoding(code: ScaleCode, width: u32, counts: [u64; 3]) -> ScaleEncoding {
    let slot = WIDTHS
        .iter()
        .position(|w| *w == width)
        .expect("a width of the codec");
    ScaleEncoding {
        code,
        width,
        exceptions: counts[slot],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::SwigluColumns;
    use kernel_core::precision::layout::PackingPlan;
    use kernel_core::precision::{DecodeOrder, WeightLayout};
    use runtime_bundle_abi::weight_plan::CensusRankV1;

    fn operand(format: WeightFormat, dtype: DType) -> PackedOperand {
        PackedOperand {
            local_encoding: None,
            plan: PackingPlan {
                layout: WeightLayout {
                    id: "row-major-groups".into(),
                    version: 1,
                },
                format,
                experts: 1,
                projections: 1,
                rows: 4,
                k: 64,
                columns: SwigluColumns::ConcatenatedHalves,
                local_scale_dtype: Some(dtype),
            },
            section: PackedSection::LocalScales,
            decode_order: DecodeOrder::NativeBlockScaledMma,
            matrices: vec![],
        }
    }

    #[test]
    fn eligibility_follows_the_admitted_format() {
        assert_eq!(
            scale_code(&operand(WeightFormat::Mxfp4, DType::Ue8M0)),
            Some(ScaleCode::Ue8m0)
        );
        assert_eq!(
            scale_code(&operand(WeightFormat::Mxfp8, DType::Ue8M0)),
            Some(ScaleCode::Ue8m0)
        );
        assert_eq!(
            scale_code(&operand(WeightFormat::Nvfp4, DType::Fp8E4M3)),
            Some(ScaleCode::E4m3)
        );
        assert_eq!(
            scale_code(&operand(
                WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32),
                DType::F32
            )),
            None
        );
        let mut bank = operand(WeightFormat::Mxfp4, DType::Ue8M0);
        bank.plan.experts = 2;
        assert_eq!(scale_code(&bank), None);
    }

    #[test]
    fn banks_take_the_smallest_width_or_stay_raw() {
        // Few exceptions: one bit per scale wins.
        assert_eq!(bank_width(&[((1024, 160), [10, 0, 0])]).unwrap(), Some(1));
        // Many 1-bit exceptions: two bits win.
        assert_eq!(
            bank_width(&[((1024, 160), [40_000, 5, 0])]).unwrap(),
            Some(2)
        );
        // A tiny matrix cannot pay for its table and bases.
        assert_eq!(bank_width(&[((2, 2), [0, 0, 0])]).unwrap(), None);
        // Every width loses to raw when exceptions abound.
        assert_eq!(bank_width(&[((64, 64), [4096, 4096, 4096])]).unwrap(), None);
    }

    #[test]
    fn counts_must_answer_this_census() {
        let census = WeightCensusV1 {
            ranks: 2,
            rank: 1,
            items: vec![],
        };
        let mut request = WeightPlanRequestV1 {
            ranks: 2,
            rank: 1,
            ..Default::default()
        };
        assert_eq!(counts(&request, &census).unwrap(), None);
        request.census = vec![CensusRankV1 {
            rank: 1,
            document_sha256: document_sha256(&census),
            counts: vec![],
        }];
        assert_eq!(counts(&request, &census).unwrap(), Some(&[][..]));
        request.census[0].document_sha256[0] ^= 1;
        assert!(counts(&request, &census).is_err());
        request.census[0].rank = 0;
        assert!(counts(&request, &census).is_err());
    }
}
