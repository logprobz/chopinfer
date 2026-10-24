//! Calibrated drafter confidence for balanced plans, as a profile's
//! `speculation.confidence` and `--speculation-confidence` give it. With it,
//! balanced plans reserve every item's drafts up to its limits and a package
//! that allocates verify drafts verifies the ones its drafter is most
//! confident in within the width the plan would verify without confidence.

use runtime_bundle::BundleManifestV1;
use runtime_engine::VerifyAllocation;
use serde::{Deserialize, Serialize};

/// The written calibration: the package `confidence_contract` it was fitted
/// for, the drafts every item verifies before confidence ranks the rest, and
/// `[scale, offset]` per draft position, from the first, for greedy and for
/// sampled items. A raw confidence `s` at a position means
/// `sigmoid(scale * s + offset)`, the chance its draft is accepted when every
/// earlier one was.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Calibration {
    pub contract: String,
    pub floor: usize,
    pub greedy: Vec<[f32; 2]>,
    pub sampled: Vec<[f32; 2]>,
}

impl Calibration {
    /// Both kinds cover the same 1 to 16 positions with finite values, the
    /// floor is at most the positions, and the contract is a valid ID.
    pub(crate) fn check(&self) -> Result<VerifyConfidence, String> {
        runtime_bundle::BundleOpaqueId::new(self.contract.as_str())
            .map_err(|error| format!("the confidence contract is invalid: {error}"))?;
        let pairs = |pairs: &[[f32; 2]]| -> Vec<(f32, f32)> {
            pairs
                .iter()
                .map(|&[scale, offset]| (scale, offset))
                .collect()
        };
        let allocation =
            VerifyAllocation::new(self.floor, &pairs(&self.greedy), &pairs(&self.sampled))
                .map_err(|_| {
                    "the confidence calibration needs 1 to 16 finite greedy and sampled \
                     [scale, offset] pairs, as many of each, and a floor of at most that many"
                        .to_owned()
                })?;
        Ok(VerifyConfidence {
            contract: self.contract.clone(),
            allocation,
        })
    }
}

/// A checked calibration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyConfidence {
    contract: String,
    allocation: VerifyAllocation,
}

impl VerifyConfidence {
    /// Draft positions the calibration covers.
    pub(crate) fn positions(&self) -> usize {
        self.allocation.positions()
    }

    /// The allocation serving uses with `manifest`, or why it serves without
    /// one: the package must draft, allocate verify drafts, and name this
    /// calibration's confidence contract, with device results selected.
    pub(crate) fn resolve(
        &self,
        manifest: &BundleManifestV1,
        device_results: bool,
    ) -> Result<VerifyAllocation, String> {
        if !device_results {
            return Err("the package returns host values".to_owned());
        }
        if !crate::executor::verify_allocation_capability_v1(manifest) {
            return Err(format!(
                "the package does not declare {}",
                runtime_bundle_abi::VERIFY_ALLOCATION_CAPABILITY_V1
            ));
        }
        let declared = manifest
            .speculation()
            .filter(|declaration| declaration.bundle_drafter())
            .and_then(|declaration| declaration.confidence_contract());
        match declared {
            Some(contract) if contract == self.contract => Ok(self.allocation),
            Some(contract) => Err(format!(
                "the calibration was fitted for {}, the package's drafter follows {contract}",
                self.contract
            )),
            None => Err("the package names no confidence contract".to_owned()),
        }
    }
}

/// `--speculation-confidence`: `off`, or a calibration as JSON.
pub(crate) fn parse(text: &str) -> Result<Option<VerifyConfidence>, String> {
    if text == "off" {
        return Ok(None);
    }
    serde_json::from_str::<Calibration>(text)
        .map_err(|error| format!("the confidence calibration is not valid JSON: {error}"))?
        .check()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> Calibration {
        Calibration {
            contract: "fixture.drafter-confidence.v1".into(),
            floor: 1,
            greedy: vec![[1.5, -0.25], [1.0, 0.0]],
            sampled: vec![[0.75, 0.5], [2.0, -1.0]],
        }
    }

    #[test]
    fn calibrations_parse_from_json_or_off() {
        assert_eq!(parse("off"), Ok(None));
        let text = serde_json::to_string(&calibration()).unwrap();
        let checked = parse(&text).unwrap().unwrap();
        assert_eq!(checked.positions(), 2);
        assert_eq!(checked.allocation.floor(), 1);
        assert_eq!(
            checked
                .allocation
                .pair(runtime_engine::AllocationKind::Sampled, 1),
            (2.0, -1.0)
        );
        assert_eq!(calibration().check(), Ok(checked));
    }

    #[test]
    fn malformed_calibrations_are_refused() {
        let mut uneven = calibration();
        uneven.sampled.pop();
        let mut deep_floor = calibration();
        deep_floor.floor = 3;
        let mut empty = calibration();
        empty.greedy.clear();
        empty.sampled.clear();
        let mut contract = calibration();
        contract.contract = "has space".into();
        for bad in [uneven, deep_floor, empty, contract] {
            assert!(bad.check().is_err(), "{bad:?} accepted");
        }
        for text in [
            "",
            "{}",
            r#"{"contract":"a","floor":0,"greedy":[[1,0]],"sampled":[[1,0]],"extra":1}"#,
            r#"{"contract":"a","floor":0,"greedy":[[1]],"sampled":[[1,0]]}"#,
        ] {
            assert!(parse(text).is_err(), "{text} accepted");
        }
        assert!(parse(r#"{"contract":"a","floor":0,"greedy":[[1,0]],"sampled":[[1,0]]}"#).is_ok());
    }
}
