//! Initialize qualified folds once, after weight upload and before graph capture.
use crate::ops::fold_stream::{self, Storage};
use bundle_runtime::resources::{Held, allocate_span};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleMemoryPoolHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};
use std::collections::BTreeMap;

#[allow(clippy::too_many_arguments)]
pub(super) fn initialize(
    target: &crate::ops::Ops,
    drafter: Option<&crate::dspark::device::Device>,
    services: &HostServicesV1,
    pool: BundleMemoryPoolHandleV1,
    stream: BundleStreamHandleV1,
    resolved: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    views: &mut BTreeMap<String, Span>,
    held: &mut Vec<Held>,
) -> Result<(), BundleAbiError> {
    let weights: Vec<_> = views
        .iter()
        .filter(|(name, _)| name.ends_with(".weight"))
        .map(|(name, &span)| (name.clone(), span))
        .collect();
    for (name, weight) in weights {
        let ops = if name.starts_with("mtp.") {
            drafter.map(|d| &d.ops)
        } else {
            Some(target)
        };
        let mut selected = ops
            .and_then(|ops| ops.dense.as_ref())
            .map(|dense| dense.stream_weight_ops(&name))
            .unwrap_or_default();
        if name == "mtp.0.main_proj.weight"
            && let Some(op) = drafter.and_then(|d| d.prepared.main_stream_initializer())
        {
            selected.push((false, op, fold_stream::allocation_bytes(op, false)));
        }
        for (paired, op, [a, b, state, workspace]) in selected {
            let second = if paired {
                let prefix = name
                    .strip_suffix("w1.weight")
                    .ok_or(BundleAbiError::InvalidRange("paired fold weight name"))?;
                *views
                    .get(&format!("{prefix}w3.weight"))
                    .ok_or(BundleAbiError::InvalidRange("paired fold up weight"))?
            } else {
                Span::new(
                    runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                    0,
                    0,
                )
            };
            // Each operation owns all writable storage, including workspace.
            // Row buckets share it only through ordered graph submissions.
            let storage = Storage {
                packed: [
                    allocate_span(services, pool, stream, a, held)?,
                    if paired {
                        allocate_span(services, pool, stream, b, held)?
                    } else {
                        Span::new(
                            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                            0,
                            0,
                        )
                    },
                ],
                state: allocate_span(services, pool, stream, state, held)?,
                workspace: if workspace > 0 {
                    allocate_span(services, pool, stream, workspace, held)?
                } else {
                    Span::new(
                        runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                        0,
                        0,
                    )
                },
            };
            let launches = fold_stream::initialize(op, [weight, second], storage, paired).map_err(
                |error| {
                    let _ = services.log(error.to_string().as_bytes());
                    BundleAbiError::InvalidRange("packed fold initialization")
                },
            )?;
            for lowered in launches {
                let launch = lowered.launch;
                bundle_runtime::dispatch::launch(
                    services,
                    stream,
                    resolved,
                    &launch.symbol,
                    launch.plan,
                    &launch.arguments,
                    &launch.bindings,
                )
                .map_err(|error| {
                    let _ =
                        services.log(format!("packed fold initialization: {error:?}").as_bytes());
                    BundleAbiError::InvalidRange("packed fold initialization")
                })?;
            }
            for (role, span) in [
                ("packed0", storage.packed[0]),
                ("packed1", storage.packed[1]),
                ("state", storage.state),
                ("workspace", storage.workspace),
            ] {
                if span.length > 0 {
                    views.insert(fold_stream::name(&name, paired, role), span);
                }
            }
        }
    }
    Ok(())
}
