//! A lab candidate in the candidate arm: each served launch of the 9- to
//! 64-row direct MoE (`moe_small_l2::served`) becomes its twin under one
//! knob set, with the same arguments and launch. `CHOPINFER_MOE_WIDE_SWAP`
//! names the knob set's tag and `CHOPINFER_MOE_WIDE_LAB_CUBIN` the staged
//! lab cubin, which must match its lock; the cubin alone also provides the
//! timing harness's read flush. Both arms otherwise load the same exported
//! graphs.
#[allow(dead_code)]
#[path = "../../../sm120/moe_small_l2.rs"]
pub mod moe_small_l2;

use super::recorded::Arm;
use kernel_harness::LabResult;
use std::path::Path;

pub struct Lab {
    /// The staged lab cubin, checked against its lock.
    pub cubin: Vec<u8>,
    /// The knob set whose twins replace the candidate arm's served launches.
    pub swap: Option<String>,
    served: Vec<String>,
}

impl Lab {
    pub fn environment() -> LabResult<Option<Self>> {
        let path = std::env::var("CHOPINFER_MOE_WIDE_LAB_CUBIN").ok();
        let swap = std::env::var("CHOPINFER_MOE_WIDE_SWAP").ok();
        match (path, swap) {
            (None, None) => Ok(None),
            (None, Some(_)) => {
                Err("CHOPINFER_MOE_WIDE_SWAP needs CHOPINFER_MOE_WIDE_LAB_CUBIN".into())
            }
            (Some(path), swap) => {
                Self::new(&std::fs::read(path).map_err(|e| e.to_string())?, swap).map(Some)
            }
        }
    }

    pub fn new(cubin: &[u8], swap: Option<String>) -> LabResult<Self> {
        if let Some(tag) = &swap
            && !moe_small_l2::VARIANTS.iter().any(|(t, _)| t == tag)
        {
            return Err(format!("unknown lab knob set {tag}"));
        }
        let lock: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("kernels.lock.d")
                    .join(format!("{}.lock", moe_small_l2::FAMILY)),
            )
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let locked = lock
            .as_array()
            .and_then(|rows| rows.iter().find(|r| r["name"] == moe_small_l2::CUBIN))
            .and_then(|r| r["cubin_sha256"].as_str())
            .ok_or("the lab cubin has no lock")?;
        if kernel_build::digest(cubin) != locked {
            return Err("the staged lab cubin differs from its lock".into());
        }
        let served = moe_small_l2::served(&moe_small_l2::entry()?)?
            .map(|e| e.symbol)
            .to_vec();
        Ok(Self {
            cubin: cubin.to_vec(),
            swap,
            served,
        })
    }

    /// Renames every served launch of `arm` to its twin under the knob set;
    /// returns how many. The down finisher reads the layout its queue wrote,
    /// and a folding twin queue writes one value per route, so the finisher's
    /// twin replaces it only where the served down queue is replaced too.
    pub fn apply(&self, arm: &mut Arm) -> usize {
        let Some(tag) = &self.swap else {
            return 0;
        };
        let down_queue = arm
            .prefix
            .iter()
            .chain(&arm.calls)
            .any(|c| c.symbol == self.served[2]);
        let mut swapped = 0;
        for call in arm.prefix.iter_mut().chain(arm.calls.iter_mut()) {
            if let Some(at) = self.served.iter().position(|s| *s == call.symbol)
                && (at != 3 || down_queue)
            {
                call.symbol = moe_small_l2::candidate_symbol(&call.symbol, tag);
                swapped += 1;
            }
        }
        swapped
    }
}
