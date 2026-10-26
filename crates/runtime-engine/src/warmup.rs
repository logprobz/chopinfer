//! Shapes that startup warmup drives through the engine; the shared rule
//! lives in `runtime_bundle_sdk::warmup`, so a bundle's CPU conformance
//! scenario drives the same shapes.
use crate::{DraftSource, SpeculationConfig, SpeculationMode};
pub use runtime_bundle_sdk::warmup::{
    AllocatedShape, VerifyShape, VerifyWarmup, context_ladder, verify_shapes,
};

/// One step the engine executed during startup warmup: its rows by phase
/// and the draft tokens its verify items accepted. Warmup reads these to
/// end a verify wave once it has run what the wave is for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StartupStep {
    pub prefill_tokens: usize,
    pub decode_rows: usize,
    pub verify_items: usize,
    pub verify_rows: usize,
    pub accepted_tokens: usize,
}

/// The verify steps with an allocation that waves of fresh unpinned
/// requests reach, up to `sequences` requests: one wave for each distinct
/// width and reserved rows, which is all an executor's allocated graphs may
/// key on, at the fewest requests that reach it. `item_rows` are the rows
/// one verify item may take ([`VerifyWarmup::item_rows`]); the engine plans
/// only their draft counts. Empty unless `config` allocates.
#[must_use]
pub fn allocated_shapes(
    config: SpeculationConfig,
    sequences: usize,
    item_rows: &[usize],
) -> Vec<AllocatedShape> {
    let (Some(allocation), SpeculationMode::Balanced(depths), DraftSource::Bundle) =
        (config.verify_allocation(), config.mode(), config.source())
    else {
        return Vec::new();
    };
    let widths: Vec<_> = config
        .verify_widths()
        .widths()
        .iter()
        .map(|&n| usize::from(n))
        .collect();
    runtime_bundle_sdk::warmup::allocated_shapes(
        sequences,
        &runtime_bundle_sdk::warmup::AllocationWarmup {
            maximum_draft_tokens: config.maximum_draft_tokens(),
            step_rows: config.maximum_verify_tokens(),
            drafting_items: config.maximum_draft_items(),
            item_rows,
            widths: &widths,
            floor: allocation.floor(),
        },
        |items| depths.at(items),
    )
}

#[cfg(test)]
mod tests {
    use super::{AllocatedShape, allocated_shapes};
    use crate::{
        DraftDepths, DraftSource, SpeculationConfig, SpeculationMode, VerifyAllocation,
        VerifyWidths,
    };

    use crate::speculation::{AcceptanceEstimate, PlanCandidate, SpeculationPolicy};
    use runtime_scheduler::DraftBudget;

    // Keep the engine planner as an independent oracle for the SDK's fresh waves.
    fn policy_shapes(
        config: SpeculationConfig,
        sequences: usize,
        item_rows: &[usize],
    ) -> Vec<AllocatedShape> {
        if config.verify_allocation().is_none() {
            return Vec::new();
        }
        // The executor's rounding of a draft count to its row buckets.
        let round = |count: usize| {
            item_rows
                .iter()
                .copied()
                .filter(|&rows| rows <= count + 1)
                .max()
                .unwrap_or(1)
                .saturating_sub(1)
                .min(count)
        };
        let acceptance = AcceptanceEstimate::initial();
        let mut shapes: Vec<AllocatedShape> = Vec::new();
        for items in 2..=sequences {
            let Some(spare) = config.maximum_verify_tokens().checked_sub(items) else {
                break;
            };
            // As the scheduler offers a fresh request with output left for its
            // deepest drafts, and the engine plans and rounds them.
            let candidate = PlanCandidate {
                maximum_draft_tokens: config.maximum_draft_tokens(),
                acceptance: &acceptance,
                fresh: true,
                remaining_tokens: usize::MAX,
            };
            let mut policy = SpeculationPolicy::new(config).with_draft_rounding(round);
            let drafts = policy.plan(&vec![candidate; items], DraftBudget::new(spare, spare));
            let Some(allocation) = policy.take_allocation() else {
                continue;
            };
            let reserved_rows = items + drafts.iter().map(|&count| round(count)).sum::<usize>();
            let width = allocation.width.min(reserved_rows);
            if width > items
                && reserved_rows > width
                && !shapes
                    .iter()
                    .any(|shape| (shape.width, shape.reserved_rows) == (width, reserved_rows))
            {
                shapes.push(AllocatedShape {
                    items,
                    width,
                    reserved_rows,
                });
            }
        }
        shapes
    }

    fn config(capacity: usize, floor: usize) -> SpeculationConfig {
        SpeculationConfig::new(
            5,
            capacity,
            SpeculationMode::Balanced(DraftDepths::parse("1:3").unwrap()),
            DraftSource::Bundle,
        )
        .unwrap()
        .with_verify_widths(VerifyWidths::parse("6,8,16,32").unwrap())
        .with_verify_allocation(Some(
            VerifyAllocation::new(floor, &[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap(),
        ))
    }

    fn triples(shapes: &[AllocatedShape]) -> Vec<(usize, usize, usize)> {
        shapes
            .iter()
            .map(|shape| (shape.items, shape.width, shape.reserved_rows))
            .collect()
    }

    #[test]
    fn allocated_waves_reach_each_width_and_reservation_once() {
        let rows: Vec<usize> = (1..=6).collect();
        // Depth 3 within 48 rows: each item reserves its deepest drafts the
        // capacity holds. 12 and 24 items repeat 8 items' 32 of 48 rows, and
        // a floor of one draft fills 16 items' 32 rows, so none of those
        // runs. Past 24 items the plan's rows are all the capacity holds.
        assert_eq!(
            triples(&allocated_shapes(config(48, 1), 32, &rows)),
            [
                (2, 8, 12),
                (3, 8, 18),
                (4, 16, 24),
                (5, 16, 30),
                (6, 16, 36),
                (7, 16, 42),
                (8, 32, 48),
                (9, 32, 45),
                (10, 32, 40),
                (11, 32, 44),
                (13, 32, 39),
                (14, 32, 42),
                (17, 32, 34),
                (18, 32, 36),
                (19, 32, 38),
                (23, 32, 46),
            ]
        );
        assert_eq!(
            triples(&allocated_shapes(config(48, 1), 4, &rows)),
            [(2, 8, 12), (3, 8, 18), (4, 16, 24)]
        );
        // Only draft counts with a row bucket are reserved: one or three
        // drafts, which 2 and 4 items already plan.
        assert_eq!(
            triples(&allocated_shapes(config(48, 1), 4, &[1, 2, 4])),
            [(3, 8, 12)]
        );
        let plain = config(48, 1).with_verify_allocation(None);
        assert!(allocated_shapes(plain, 32, &rows).is_empty());
    }

    #[test]
    fn allocated_waves_keep_the_served_capacity_and_depth_schedule() {
        let config = SpeculationConfig::new(
            5,
            64,
            SpeculationMode::Balanced(DraftDepths::parse("1:5,8:3,16:3,32:0").unwrap()),
            DraftSource::Bundle,
        )
        .unwrap()
        .with_verify_widths(VerifyWidths::parse("6,8,16,32,64").unwrap())
        .with_verify_allocation(config(64, 1).verify_allocation());
        assert_eq!(
            triples(&allocated_shapes(config, 32, &[2, 3, 4, 5, 6])),
            [
                (2, 8, 12),
                (3, 16, 18),
                (4, 16, 24),
                (5, 16, 30),
                (6, 32, 36),
                (7, 32, 42),
                (8, 32, 48),
                (9, 32, 54),
                (10, 32, 60),
                (11, 32, 55),
                (13, 32, 52),
                (14, 32, 56),
            ]
        );
    }

    #[test]
    fn shared_allocated_warmup_matches_engine_planning() {
        let mut cases = 0;
        for capacity in [8, 17, 32, 48, 64] {
            for floor in [0, 1, 3, 5] {
                for schedule in ["1:3", "1:5,8:3,16:3,32:0", "3:2,9:0"] {
                    for rows in [&[1, 2, 3, 4, 5, 6][..], &[1, 2, 4], &[2, 6]] {
                        for limit in [None, Some(0), Some(1), Some(3)] {
                            for widths in [
                                VerifyWidths::NONE,
                                VerifyWidths::parse("6,8,16,32,64").unwrap(),
                            ] {
                                let config = SpeculationConfig::new(
                                    5,
                                    capacity,
                                    SpeculationMode::Balanced(
                                        DraftDepths::parse(schedule).unwrap(),
                                    ),
                                    DraftSource::Bundle,
                                )
                                .unwrap()
                                .with_verify_widths(widths)
                                .with_maximum_draft_items(limit)
                                .with_verify_allocation(
                                    config(capacity, floor).verify_allocation(),
                                );
                                assert_eq!(
                                    allocated_shapes(config, 32, rows),
                                    policy_shapes(config, 32, rows),
                                    "{config:?}, rows={rows:?}"
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 1440);
    }
}
