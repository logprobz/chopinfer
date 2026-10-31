//! The `attention_merge_ceil` candidate's launch (`latent_merge_ceil`): one
//! layer's served attention launches with the merge replaced by its
//! quantizing twin, which keeps the merge's plan, arguments and bindings
//! and adds WO_a's input codes, scales and row status.
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::{Argument, Refusal};
use kernel_core::latent_attention::LatentAttentionQuery;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::latent_merge_ceil;
use crate::ops::Lowered;

impl crate::Kernels {
    /// Whether `shape` runs the quantizing merges: the candidate option on
    /// a topology that has them.
    pub(crate) fn merge_ceil_serves(shape: &crate::shape::Shape) -> bool {
        shape.attention_merge_ceil && latent_merge_ceil::serves(shape.geometry, shape.ranks())
    }

    /// `attention`, one layer's served attention launches, with its one
    /// merge replaced by the quantizing twin that writes WO_a's input codes
    /// and scales to `codes` and `scales` and ORs each row's flags into
    /// `status` at row offset zero, as WO_a's sliced quantizer of those rows
    /// does with the merge's output.
    pub(crate) fn merge_ceil_launch(
        mut attention: Vec<Lowered>,
        codes: DeviceBufferSpanV1,
        scales: DeviceBufferSpanV1,
        status: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let entries = latent_merge_ceil::entries();
        let mut merges = attention.iter().enumerate().filter_map(|(at, lowered)| {
            entries
                .iter()
                .find(|e| e.merge_symbol() == lowered.launch.symbol)
                .map(|entry| (at, entry))
        });
        let (Some((at, entry)), None) = (merges.next(), merges.next()) else {
            return Err(Refusal(
                "a quantizing merge replaces exactly one served merge with a twin",
            ));
        };
        let LatentAttentionQuery::Attention { heads, latent, .. } = entry.query else {
            return Err(Refusal("a quantizing merge needs an attention query"));
        };
        let merge = attention.remove(at);
        let launch = merge.launch;
        let rows = launch.rows;
        if merge.collective
            || launch.plan != entry.merge_plan(rows)?
            || launch.arguments.0.len() != entry.merge_params()?.len()
        {
            return Err(Refusal(
                "a quantizing merge keeps the served merge's plan and arguments",
            ));
        }
        let columns = u64::from(rows) * u64::from(heads) * u64::from(latent);
        if codes.length < columns
            || scales.length < columns / 32
            || status.length < u64::from(rows) * 4
        {
            return Err(Refusal(
                "a quantizing merge needs codes, scales and status for its rows",
            ));
        }
        let mut bindings = launch.bindings;
        let mut arguments = launch.arguments;
        for view in [codes, scales, status] {
            arguments
                .0
                .push(Argument::DeviceBuffer(bindings.bind(view)));
        }
        arguments.0.push(Argument::U64(0));
        attention.insert(
            at,
            Lowered {
                collective: false,
                launch: PreparedLaunch {
                    symbol: entry.merge_ceil_symbol(),
                    rows,
                    plan: launch.plan,
                    arguments,
                    bindings,
                },
            },
        );
        Ok(attention)
    }
}
