//! Two ordered SwiGLU launches share sticky status and retain earlier failures.
use crate::{
    LabResult,
    conformance::{Expected, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::gated_mlp::quantized::{SiluMulQuantizeBuffers, SiluMulQuantizeQuery};
use kernel_core::properties::Invariance;

kernel_core::scratch_record! {
    #[derive(Clone)]
    pub struct Buffers {
        pub input_first: value,
        pub input_second: value,
        pub up: value,
        pub output_first: value,
        pub output_second: value,
        pub scales_first: value,
        pub scales_second: value,
        pub status: value,
        pub live_rows: value,
    }
}
struct Sequence {
    inner: Prepared<SiluMulQuantizeBuffers<View>>,
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
                &SiluMulQuantizeBuffers {
                    status_row_offset: 0,
                    gate: input,
                    up: b.up,
                    codes: output,
                    scales,
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
    let query = kernels_sm120::row::silu_mul_quantize::lab_entries()
        .remove(1)
        .query;
    Required::new(
        "shared SwiGLU row status",
        Need::rows(16).with_exact(Invariance::any_rows(16)),
        move |backend, need| {
            let inner = backend.silu_mul_quantize(&query, need)?;
            let mut info = inner.info().clone();
            info.bytes.clear();
            Ok(Box::new(Sequence { inner, info }) as Prepared<Buffers<View>>)
        },
        move |op| case(query, op),
    )
    .required()
}

fn case<'a>(query: SiluMulQuantizeQuery, op: Prepared<Buffers<View>>) -> LabResult<OpCase<'a>> {
    let width = query.width as usize;
    let mut first = vec![1_f32; 16 * width];
    let mut second = vec![1_f32; 16 * width];
    for row in 0..16 {
        match row % 4 {
            0 => {
                first[row * width] = f32::NAN;
                second[row * width + width - 1] = 2.0;
            }
            1 => first[row * width + width - 1] = f32::INFINITY,
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
        up: values(&vec![3.0; 16 * width]),
        input_first: values(&first),
        input_second: values(&second),
        output_first: Spec::output(DType::U8, (Rows::Live, width)),
        output_second: Spec::output(DType::U8, (Rows::Live, width)),
        scales_first: Spec::output(DType::U8, (Rows::Live, width / 32)),
        scales_second: Spec::output(DType::U8, (Rows::Live, width / 32)),
        status: super::super::block_quantize::status_role(&[0; 16]),
        live_rows: Spec::live(),
    };
    Ok(OpCase::new(op, 16, roles)?
        .oracle(move |images, live| {
            let prior = images
                .bytes("status")?
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            let live = live as usize;
            let up = images.f32s("up")?;
            let first = super::outcomes_with_status(
                query,
                &images.f32s("input_first")?[..live * width],
                &up[..live * width],
                &prior[..live],
            )?;
            let second = super::outcomes_with_status(
                query,
                &images.f32s("input_second")?[..live * width],
                &up[..live * width],
                &first.status,
            )?;
            let mut status = prior;
            status[..live].copy_from_slice(&second.status);
            let mut expected = Vec::new();
            for (suffix, outcome) in [("first", first), ("second", second)] {
                expected.extend(
                    super::judges(outcome, live)
                        .into_iter()
                        .map(|(name, value)| {
                            let name = if name == "codes" { "output" } else { &name };
                            (format!("{name}_{suffix}"), value)
                        }),
                );
            }
            expected.push((
                "status".into(),
                Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
            ));
            Ok(expected)
        })
        .live_counts(vec![0, 1, 3, 15, 16]))
}
