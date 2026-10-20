//! Execute the selected arm through the same prepared backend op as serving.
use super::{OFFSET, registration};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    properties::Invariance,
    rmsnorm_quantize::RmsNormQuantizeBuffers,
};
use kernel_harness::LabResult;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;

pub fn plan(rows: u32, views: &[View]) -> LabResult<Launch> {
    let backend = bundle_deepseek41::sm120::norm_quantize::register(Sm120::new(), &FLASH)?;
    let backend = registration::register(backend, &FLASH)?;
    let entry = registration::selected()?;
    let op = backend
        .rmsnorm_quantize(
            &entry.reference.query,
            Need::rows(8192).with_exact(Invariance::any_rows(8192)),
        )
        .map_err(|e| format!("{e:?}"))?;
    let mut launches = op
        .launches(
            &RmsNormQuantizeBuffers {
                input: views[0],
                weight: views[1],
                normed: views[2],
                widened: None,
                codes: views[3],
                scales: views[4],
                live_rows: views[5],
                status: views[6],
                status_row_offset: OFFSET as u64,
            },
            View {
                handle: 0,
                bytes: 0,
            },
            Counts::rows(rows),
        )
        .map_err(|e| e.to_string())?;
    if launches.len() != 1 {
        return Err("prepared query norm must issue one launch".into());
    }
    Ok(launches.remove(0))
}

#[test]
fn served_plan_and_every_argument_match_the_qualified_recipe() {
    use kernel_core::contract::Argument as A;
    let views = (1..=7)
        .map(|handle| View {
            handle,
            bytes: 1 << 30,
        })
        .collect::<Vec<_>>();
    let entry = registration::selected().unwrap();
    for rows in [1, 64, 8128, 8192] {
        let actual = plan(rows, &views).unwrap();
        assert_eq!(actual.symbol, entry.reference.symbol);
        assert_eq!(actual.plan, entry.plan(rows).unwrap());
        assert_eq!(
            actual.arguments.0,
            vec![
                A::DeviceBuffer(1),
                A::DeviceBuffer(2),
                A::DeviceBuffer(3),
                A::NullDevicePointer,
                A::DeviceBuffer(4),
                A::DeviceBuffer(5),
                A::DeviceBuffer(6),
                A::DeviceBuffer(7),
                A::U64(OFFSET as u64),
                A::U32(entry.reference.query.norm.epsilon_bits),
            ]
        );
        assert!(!actual.collective && !actual.persistent);
    }
}
