//! Raw encoded rows reach the store below the serving quantizer's amax floor.
use super::*;
use kernel_core::contract::DType;
use kernel_core::latent_attention::{LatentWindowBuffers, LatentWindowStorage};
use kernel_core::latent_attention_oracle::EncodedLatent;
use kernel_core::latent_glue_oracle::exact_window_bf16;
use kernel_core::precision::quantizer::UNREPRESENTABLE_OUTPUT;
use kernel_harness::conformance::{Expected, Images, Rows, Write};
use kernel_harness::kit::{OpCase, Spec};
use kernel_harness::poison::WriteMap;
use std::ops::Range;
const CAPACITY: u32 = 8;
const STRIDE: usize = CAPACITY as usize * 1024 + 32;
struct StoreOnly {
    inner: Prepared<LatentWindowWriteBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<LatentWindowWriteBuffers<View>> for StoreOnly {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &LatentWindowWriteBuffers<View>,
        w: View,
        c: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // Preserve the backend's ABI and validation, then isolate the final
        // store so already-quantized subnormals are not raised to the amax floor.
        let mut calls = self.inner.launches(b, w, c)?;
        Ok(vec![
            calls.pop().ok_or(Refusal("window write has no store"))?,
        ])
    }
}
fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn read(images: &Images, name: &str) -> LabResult<Vec<i32>> {
    Ok(images
        .bytes(name)?
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
struct ExpectedRow {
    position: usize,
    bytes: Vec<u8>,
}
fn rows(images: &Images, live: u32) -> LabResult<(Vec<u32>, Vec<ExpectedRow>)> {
    let mut status = read(images, "status")?
        .iter()
        .map(|&v| v as u32)
        .collect::<Vec<_>>();
    let positions = read(images, "positions")?;
    let codes = images.bytes("quantized_codes")?;
    let scales = images.bytes("quantized_scales")?;
    let mut written = Vec::new();
    for row in 0..live as usize {
        let Ok(position) = usize::try_from(positions[row]) else {
            continue;
        };
        if position >= CAPACITY as usize {
            continue;
        }
        let encoded = EncodedLatent {
            values: codes[row * 512..(row + 1) * 512].to_vec(),
            scales: vec![],
            scale_bytes: scales[row * 16..(row + 1) * 16].to_vec(),
        };
        match exact_window_bf16(&encoded, 512) {
            Ok(bytes) if status[row] == 0 => written.push(ExpectedRow { position, bytes }),
            Ok(_) => {}
            Err(_) => status[row] |= UNREPRESENTABLE_OUTPUT,
        }
    }
    Ok((status, written))
}
fn ranges(images: &Images, live: u32) -> LabResult<Vec<Range<usize>>> {
    Ok(rows(images, live)?
        .1
        .into_iter()
        .map(|r| r.position * 1024..(r.position + 1) * 1024)
        .collect())
}
fn case(op: Prepared<LatentWindowWriteBuffers<View>>) -> LabResult<OpCase<'static>> {
    let n = CAPACITY as usize;
    let mut codes = vec![0; n * 512];
    let mut scales = vec![127; n * 16];
    for group in 0..16 {
        scales[group] = if group % 2 == 0 { 3 } else { 6 };
        for col in 0..32 {
            codes[group * 32 + col] = if group % 2 == 0 {
                [0x01, 0x81, 0x00, 0x80, 0x02, 0x82][col % 6]
            } else {
                [0x0f, 0x8f, 0x10, 0x90][col % 4]
            };
        }
    }
    for (row, code, scale) in [
        (1, 0x01, 2),
        (2, 0x7f, 127),
        (3, 0x7e, 254),
        (4, 0x38, 255),
        (5, 0x01, 3),
        (6, 0x01, 2),
        (7, 0xff, 255),
    ] {
        codes[row * 512..(row + 1) * 512].fill(code);
        scales[row * 16..(row + 1) * 16].fill(scale);
    }
    let mut positions = (0..CAPACITY as i32).collect::<Vec<_>>();
    positions[7] = -2; // A hole distinguishable from the producer poison (-1).
    let mut initial = vec![0; n];
    initial[5] = 1;
    initial[6] = 1;
    let roles = LatentWindowWriteBuffers {
        input: Spec::untouched(n * 512 * 4),
        weight: Spec::untouched(512 * 2),
        latent: Spec::untouched(n * 512 * 4),
        cos_sin: Spec::untouched(n * 64 * 4),
        positions: Spec::input(DType::I32, (Rows::Live, 1), ints(&positions)),
        row_batch: Spec::input(DType::I32, (Rows::Live, 1), ints(&vec![0; n])),
        first_positions: Some(Spec::input(DType::I32, (Rows::Fixed(1), 1), ints(&[0]))),
        last_positions: Spec::input(
            DType::I32,
            (Rows::Fixed(1), 1),
            ints(&[CAPACITY as i32 - 1]),
        ),
        window: LatentWindowBuffers {
            codes: Spec::state(
                DType::U8,
                (Rows::Fixed(1), STRIDE),
                vec![0xff; STRIDE],
                Write::Map(Box::new(|images, live| {
                    WriteMap::of(DType::U8).byte_ranges(ranges(images, live)?)
                })),
            ),
            scales: Spec::untouched(4),
            item_slots: Spec::untouched(4),
            items: 0,
            sequences: 1,
            code_stride: STRIDE as u64,
            scale_stride: 0,
            storage: LatentWindowStorage::Staged { slots: CAPACITY },
            key_lists: Spec::untouched(4),
            key_stride: 0,
            budget: 0,
        },
        quantized_codes: Spec::input(DType::U8, (Rows::Live, 512), codes),
        quantized_scales: Spec::input(DType::U8, (Rows::Live, 16), scales),
        status: kit::cases::block_quantize::status_role(&initial),
        status_row_offset: 0,
        live_rows: Spec::live(),
    };
    Ok(OpCase::new(op, CAPACITY, roles)?.oracle(|images, live| {
        let (status, rows) = rows(images, live)?;
        let mut output = images.bytes("window.codes")?.to_vec();
        for row in rows {
            output[row.position * 1024..(row.position + 1) * 1024].copy_from_slice(&row.bytes);
        }
        Ok(vec![
            ("window.codes".into(), Expected::Bits(output)),
            (
                "status".into(),
                Expected::Bits(status.iter().flat_map(|s| s.to_le_bytes()).collect()),
            ),
        ])
    }))
}
fn prepare(
    b: &dyn Backend,
    n: Need,
) -> Result<Prepared<LatentWindowWriteBuffers<View>>, kernel_core::backend::Refused> {
    let inner = b.latent_window_write(&query(WindowAddressing::Staged { slots: CAPACITY }), n)?;
    let mut info = inner.info().clone();
    info.symbols = vec![info.symbols.last().expect("store symbol").clone()];
    Ok(Box::new(StoreOnly { inner, info }))
}
pub(super) fn required() -> kit::Required<'static> {
    kit::Required::new(
        "BF16 raw window subnormals and refusal boundary",
        Need::rows(CAPACITY),
        prepare,
        case,
    )
    .required()
}
#[test]
fn raw_boundary_oracle_pins_subnormals_refusals_prior_flags_and_holes() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    let report = kit::conform(&backend(), &modules, None, &[required()]).unwrap();
    assert_eq!(report.prepared().len(), 1);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    let case = case(prepare(&backend(), Need::rows(CAPACITY)).unwrap()).unwrap();
    let images = case.images();
    let expected = case.oracle_bits(&images, CAPACITY).unwrap();
    let bits = |i: usize| {
        u16::from_le_bytes(
            expected["window.codes"][2 * i..2 * i + 2]
                .try_into()
                .unwrap(),
        )
    };
    assert_eq!(
        [bits(0), bits(1), bits(2), bits(3)],
        [0x0001, 0x8001, 0, 0x8000]
    );
    assert_eq!(
        [bits(32), bits(33), bits(34), bits(35)],
        [0x0078, 0x8078, 0x0080, 0x8080]
    );
    assert_eq!(expected["status"], ints(&[0, 4, 4, 4, 4, 1, 5, 0]));
    assert!(expected["window.codes"][1024..].iter().all(|&v| v == 0xff));
}
