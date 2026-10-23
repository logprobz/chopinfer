//! Initialize qualified folds once, after weight upload and before graph capture.
//!
//! A fold operation's state and workspace may alias across the ordered
//! layers of one projection, paired mode and implementation
//! ([`ScratchKey`]). Each rank keeps the target and DSpark (`mtp.*`) apart
//! and the shared expert's side lane apart from the main lane. Row buckets
//! keep their independent maximum bounds, and packed weights stay private
//! to each weight and operation. Only the stream and direct families share:
//!
//! - `mxfp8_fold_stream.cuh`'s entries wait before `run_task` touches
//!   mutable storage, and the last arrivals reset every completed and
//!   deferred ticket and the ready, done and pending flags. The one-row
//!   dual fold (`mxfp8_fold_dual.cuh`) binds the stream storage of its two
//!   projections and runs the same `run_task` after its own wait.
//! - `mxfp8_fold_direct.cuh` waits before its task, and its last arrivals
//!   clear the leaf and root counters and the admission flags.
//! - Create-time prepare is ordered on the rank stream without PDL, clears
//!   the full state span (the maximum over row buckets), and create
//!   synchronizes before any consumer runs. Workspace need not end zero:
//!   flags and partials are written before they are read.
//!
//! Every other family (ticket, prefetch, split, reuse, cluster, local rows)
//! keeps private storage or binds ordinary backend workspace. Successive
//! layers are ordered on one stream, or by the fork and join events of
//! `shared_overlap` on the side lanes. The fake-host fold tests check the
//! allocations and replay each step journal for unordered aliases; neither
//! is GPU evidence.
use crate::ops::fold_stream::{self, Storage};
use bundle_runtime::resources::{Held, allocate_span};
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, BundleMemoryPoolHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ScratchKey {
    drafter: bool,
    side_lane: bool,
    projection: String,
    paired: bool,
    implementation: String,
}

/// Only the audited stream/direct families reset all persistent state and
/// wait before mutable accesses. Unknown operations keep private storage.
fn scratch_key(weight: &str, paired: bool, implementation: &str) -> Option<ScratchKey> {
    if !["sm120.mxfp8-fold.stream.v1:", "sm120.mxfp8-fold.direct.v1:"]
        .iter()
        .any(|prefix| implementation.starts_with(prefix))
    {
        return None;
    }
    let mut parts = weight.splitn(3, '.');
    let drafter = match parts.next()? {
        "layers" => false,
        "mtp" => true,
        _ => return None,
    };
    parts.next()?.parse::<u32>().ok()?;
    let projection = parts.next()?;
    Some(ScratchKey {
        drafter,
        side_lane: projection.starts_with("ffn.shared_experts."),
        projection: projection.into(),
        paired,
        implementation: implementation.into(),
    })
}

/// Whether weight `name` can lower a decode or verify bucket of `rows`.
/// The drafter's blocks always lower whole five-row blocks
/// (`lower::dspark::block_stages`); only its catch-up KV projection runs
/// other decode counts. Target weights lower every bucket.
fn lowers_bucket(name: &str, rows: u32) -> bool {
    !name.starts_with("mtp.")
        || name.ends_with(".attn.wkv.weight")
        || rows.is_multiple_of(crate::dspark::state::BLOCK_ROWS)
}

/// The bytes one rank's folds allocated: their packed weight copies, their
/// state and their workspace.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FoldBytes {
    pub packed: u64,
    pub state: u64,
    pub workspace: u64,
}

fn empty() -> Span {
    Span::new(
        runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
        0,
        0,
    )
}

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
) -> Result<FoldBytes, BundleAbiError> {
    let mut bytes = FoldBytes::default();
    // This cache belongs to one rank. Layers and row buckets submit in
    // order; fork/join events also order successive uses on the side lane.
    // Target and drafter never share mutable storage (module documentation).
    let mut scratch = BTreeMap::<ScratchKey, [Span; 2]>::new();
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
            .map(|dense| dense.stream_weight_ops_within(&name, |rows| lowers_bucket(&name, rows)))
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
                empty()
            };
            let packed = [
                allocate_span(services, pool, stream, a, held)?,
                if paired {
                    allocate_span(services, pool, stream, b, held)?
                } else {
                    empty()
                },
            ];
            let key = scratch_key(&name, paired, &op.info().implementation);
            let shared = key.as_ref().and_then(|key| scratch.get(key)).copied();
            let [state_span, workspace_span] = if let Some(spans) = shared {
                // stream_weight_ops already takes independent maxima over
                // every row bucket of this operation, for every layer.
                if spans[0].length != state || spans[1].length != workspace {
                    return Err(BundleAbiError::InvalidRange("shared fold scratch bounds"));
                }
                spans
            } else {
                let spans = [
                    allocate_span(services, pool, stream, state, held)?,
                    if workspace > 0 {
                        allocate_span(services, pool, stream, workspace, held)?
                    } else {
                        empty()
                    },
                ];
                if let Some(key) = key {
                    scratch.insert(key, spans);
                }
                bytes.state += spans[0].length;
                bytes.workspace += spans[1].length;
                spans
            };
            bytes.packed += packed[0].length + packed[1].length;
            let storage = Storage {
                packed,
                state: state_span,
                workspace: workspace_span,
            };
            // Prepare is non-PDL and ordered on the rank stream. Each call
            // packs this weight and clears the full shared state span; create
            // synchronizes before any graph consumer can run.
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
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_fold_storage_covers_only_the_buckets_it_lowers() {
        assert!(lowers_bucket("layers.3.attn.wq_a.weight", 1));
        assert!(lowers_bucket("mtp.1.attn.wq_a.weight", 10));
        assert!(!lowers_bucket("mtp.1.attn.wq_a.weight", 8));
        assert!(!lowers_bucket("mtp.2.ffn.shared_experts.w2.weight", 1));
        assert!(lowers_bucket("mtp.1.attn.wkv.weight", 3));
    }

    #[test]
    fn fold_scratch_keys_keep_images_lanes_pairs_and_ops_separate() {
        let op = "sm120.mxfp8-fold.stream.v1:entry";
        let a = scratch_key("layers.0.attn.wq_a.weight", false, op).unwrap();
        assert_eq!(
            Some(a.clone()),
            scratch_key("layers.39.attn.wq_a.weight", false, op)
        );
        for (weight, paired, implementation) in [
            ("mtp.0.attn.wq_a.weight", false, op),
            ("layers.0.ffn.shared_experts.w2.weight", false, op),
            ("layers.0.attn.wq_a.weight", true, op),
            (
                "layers.0.attn.wq_a.weight",
                false,
                "sm120.mxfp8-fold.direct.v1:entry",
            ),
        ] {
            assert_ne!(Some(a.clone()), scratch_key(weight, paired, implementation));
        }
        let side = scratch_key("layers.0.ffn.shared_experts.w2.weight", false, op).unwrap();
        assert!(side.side_lane);
        assert_eq!(
            Some(side),
            scratch_key("layers.39.ffn.shared_experts.w2.weight", false, op)
        );
        assert!(scratch_key("layers.0.attn.wq_a.weight", false, "unreviewed").is_none());
        assert!(scratch_key("other.0.attn.wq_a.weight", false, op).is_none());
    }
}
