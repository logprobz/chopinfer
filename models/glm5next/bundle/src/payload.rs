//! Payload admission at create (`model_glm5next_contract::payload`): after
//! the embedded modules load, each rank runs one admission launch per
//! placed scale view of its image into a report buffer (an own op,
//! `own_ops::state::Admission`), then reads the reports back. Create refuses the image on any report, naming the rank,
//! the view and the lowest invalid element.

use std::collections::BTreeMap;

use bundle_runtime::acquire::RankHandles;
use bundle_runtime::resources::{Held, allocate_span};
use kernel_core::backend::{Counts, Need};
use model_glm5next_contract::direct::Stored;
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::own_ops::state::{self, AdmissionBuffers};
use model_glm5next_contract::payload::{Admission, NONE, admissions};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};

/// The first refusal among `reports` (one word per admission, in order):
/// the view and the lowest invalid element.
pub(crate) fn refusal(admissions: &[Admission], reports: &[u8]) -> Option<String> {
    admissions
        .iter()
        .zip(reports.chunks_exact(4))
        .find_map(|(admission, word)| {
            let first = u32::from_le_bytes(word.try_into().unwrap());
            (first != NONE).then(|| {
                format!(
                    "{} element {first} fails its {:?} admission",
                    admission.view, admission.check
                )
            })
        })
}

/// One launch of `admission`'s op on the own kernels `own` over its placed
/// views into `report`.
fn launch(
    services: &HostServicesV1,
    (_, _, stream): RankHandles,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    views: &BTreeMap<String, Span>,
    own: &dyn OwnOps,
    admission: &Admission,
    report: Span,
) -> Result<(), String> {
    let view = |name: &str| {
        views
            .get(name)
            .copied()
            .ok_or_else(|| format!("no placed view {name}"))
    };
    let op = state::Admission {
        rule: admission.check,
        count: admission.count,
    };
    let prepared = own
        .admission(&op, Need::rows(1))
        .map_err(|refused| format!("{}: {refused}", admission.view))?;
    let spans = AdmissionBuffers {
        codes: view(&admission.view)?,
        scales: admission.scales.as_deref().map(view).transpose()?,
        first: report,
    };
    let launches = crate::ops::own::launch(&prepared, spans, Counts::rows(1))
        .map_err(|refusal| format!("{}: {refusal}", admission.view))?;
    for launch in crate::ops::launches(launches) {
        crate::submit::launch(
            services,
            stream,
            symbols,
            &launch.symbol,
            launch.plan,
            &launch.arguments,
            &launch.bindings,
        )
        .map_err(|error| format!("{}: {error:?}", admission.view))?;
    }
    Ok(())
}

/// Admits every rank's placed scale views (`views`, by name), as its
/// verified layout `stored` lists them, with the rank's loaded `symbols`,
/// on the own kernels `own`.
pub(crate) fn admit(
    services: &HostServicesV1,
    own: &dyn OwnOps,
    ranks: &[RankHandles],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    views: &[BTreeMap<String, Span>],
    stored: &[Vec<Stored>],
    held: &mut Vec<Held>,
) -> Result<(), BundleAbiError> {
    let refuse = |message: String| {
        let _ = services.log(format!("GLM 5.3 Flash payload admission: {message}").as_bytes());
        BundleAbiError::InvalidRange("glm5next payload admission")
    };
    for (rank, (((&handles, symbols), views), stored)) in
        ranks.iter().zip(symbols).zip(views).zip(stored).enumerate()
    {
        let admissions = admissions(stored).map_err(refuse)?;
        if admissions.is_empty() {
            continue;
        }
        let (_, pool, stream) = handles;
        let bytes = 4 * admissions.len() as u64;
        let reports = allocate_span(services, pool, stream, bytes, held)?;
        services.copy_host_to_device(stream, reports, &vec![0xff; bytes as usize])?;
        for (index, admission) in admissions.iter().enumerate() {
            let report = Span::new(reports.buffer, reports.offset + 4 * index as u64, 4);
            launch(services, handles, symbols, views, own, admission, report)
                .map_err(|error| refuse(format!("rank {rank}: {error}")))?;
        }
        services.synchronize_stream(stream)?;
        let mut read = vec![0; bytes as usize];
        services.copy_device_to_host(stream, &mut read, reports)?;
        services.synchronize_stream(stream)?;
        if let Some(message) = refusal(&admissions, &read) {
            return Err(refuse(format!("rank {rank}: {message}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::payload::Check;

    #[test]
    fn a_lowered_report_names_its_view_and_element() {
        let admission = |view: &str, check| Admission {
            check,
            view: view.into(),
            scales: None,
            count: 16,
        };
        let admissions = [
            admission("a.weight_scale_inv", Check::F32Scales),
            admission("b.weight_scale", Check::Ue4m3Scales),
        ];
        let reports = |words: [u32; 2]| {
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>()
        };
        assert_eq!(refusal(&admissions, &reports([NONE, NONE])), None);
        assert_eq!(
            refusal(&admissions, &reports([NONE, 5])).unwrap(),
            "b.weight_scale element 5 fails its Ue4m3Scales admission"
        );
        assert!(
            refusal(&admissions, &reports([0, 5]))
                .unwrap()
                .starts_with("a.weight_scale_inv element 0")
        );
    }
}
