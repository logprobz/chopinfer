//! Two ordered quantizers accumulate different row errors in one status plane.
use crate::{
    LabResult,
    conformance::{Expected, Rows, Write},
    kit::{OpCase, Required, Spec},
    poison::WriteMap,
};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput,
};
use kernel_core::properties::Invariance;

kernel_core::scratch_record! {
    #[derive(Clone)]
    pub struct Buffers {
        pub input_first: value,
        pub input_second: value,
        pub output_first: value,
        pub output_second: value,
        pub scales_first: value,
        pub scales_second: value,
        pub status: value,
        pub live_rows: value,
    }
}
struct Sequence {
    inner: Prepared<BlockQuantizeBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Sequence {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = Vec::new();
        for (input, output, scales) in [
            (b.input_first, b.output_first, b.scales_first),
            (b.input_second, b.output_second, b.scales_second),
        ] {
            launches.extend(self.inner.launches(
                &BlockQuantizeBuffers {
                    status_row_offset: 0,
                    input,
                    output,
                    scales: Some(scales),
                    status: b.status,
                    live_rows: b.live_rows,
                },
                workspace,
                counts,
            )?);
        }
        Ok(launches)
    }
}

pub fn required<'a>() -> Required<'a> {
    let query = BlockQuantizeQuery {
        width: 128,
        input: DType::F32,
        format: BlockQuantization::E2M1E4M3Scale16,
        output: QuantizeOutput::Codes,
    };
    Required::new(
        "shared quantizer row status",
        Need::rows(16).with_exact(Invariance::any_rows(16)),
        move |backend, need| {
            let inner = backend.block_quantize(&query, need)?;
            let mut info = inner.info().clone();
            info.bytes.clear();
            Ok(Box::new(Sequence { inner, info }) as Prepared<Buffers<View>>)
        },
        move |op| case(query, op),
    )
    .required()
}

fn case<'a>(query: BlockQuantizeQuery, op: Prepared<Buffers<View>>) -> LabResult<OpCase<'a>> {
    let width = query.width as usize;
    let mut first = vec![1_f32; 16 * width];
    let mut second = vec![1_f32; 16 * width];
    for row in 0..16 {
        match row % 4 {
            0 => {
                first[row * width] = f32::NAN;
                second[row * width + width - 1] = 2704.;
            }
            1 => first[row * width + width - 1] = 2704.,
            2 => second[row * width] = f32::NEG_INFINITY,
            _ => {}
        }
    }
    let values = |v: &[f32]| {
        Spec::input(
            DType::F32,
            (Rows::Live, width),
            v.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    };
    let roles = Buffers {
        input_first: values(&first),
        input_second: values(&second),
        output_first: Spec::output(DType::U8, (Rows::Live, width / 2)),
        output_second: Spec::output(DType::U8, (Rows::Live, width / 2)),
        scales_first: Spec::output(DType::U8, (Rows::Live, width / 16)),
        scales_second: Spec::output(DType::U8, (Rows::Live, width / 16)),
        status: Spec::state_updating(
            DType::I32,
            (Rows::Live, 1),
            vec![0; 16 * 4],
            Write::Map(Box::new(|_, _| WriteMap::of(DType::I32).none())),
            |_, live| Ok((0..live as usize).map(|r| r * 4..r * 4 + 4).collect()),
        ),
        live_rows: Spec::live(),
    };
    Ok(OpCase::new(op, 16, roles)?
        .oracle(move |images, live| {
            let prior = images
                .bytes("status")?
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            let (first, scales_first, status) =
                super::expected_rows(query, &images.f32s("input_first")?, &prior, live)?;
            let (second, scales_second, status) =
                super::expected_rows(query, &images.f32s("input_second")?, &status, live)?;
            Ok(vec![
                ("output_first".into(), Expected::Bits(first)),
                ("scales_first".into(), Expected::Bits(scales_first)),
                ("output_second".into(), Expected::Bits(second)),
                ("scales_second".into(), Expected::Bits(scales_second)),
                (
                    "status".into(),
                    Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
                ),
            ])
        })
        .live_counts(vec![0, 1, 3, 15, 16]))
}
