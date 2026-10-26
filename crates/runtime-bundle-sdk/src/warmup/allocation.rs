//! Allocated warmup shapes for fresh, equally ranked requests in a balanced plan.

use super::AllocatedShape;

/// The profile and package limits that determine allocated warmup shapes.
pub struct AllocationWarmup<'a> {
    pub maximum_draft_tokens: usize,
    pub step_rows: usize,
    pub drafting_items: Option<usize>,
    pub item_rows: &'a [usize],
    pub widths: &'a [usize],
    pub floor: usize,
}

/// One wave per distinct allocated width and reservation, at the fewest
/// requests that reach it. Fresh requests have identical acceptance estimates,
/// so spare draft increments go to them in request order. `depth` gives the
/// balanced schedule's limit at each concurrency, or no limit.
pub fn allocated_shapes(
    sequences: usize,
    warmup: &AllocationWarmup<'_>,
    depth: impl Fn(usize) -> Option<usize>,
) -> Vec<AllocatedShape> {
    let mut extras: Vec<_> = warmup
        .item_rows
        .iter()
        .copied()
        .filter(|&rows| rows > 1 && rows - 1 <= warmup.maximum_draft_tokens)
        .map(|rows| rows - 1)
        .collect();
    extras.sort_unstable();
    extras.dedup();
    let mut shapes: Vec<AllocatedShape> = Vec::new();
    for items in 2..=sequences.min(warmup.step_rows) {
        let drafting = items.min(warmup.drafting_items.unwrap_or(items));
        if drafting < 2 {
            continue;
        }
        let spare = warmup.step_rows - items;
        let limit = depth(items).unwrap_or(warmup.maximum_draft_tokens);
        let plan = |budget: usize| {
            let mut drafts = vec![0; items];
            let mut left = budget;
            for &next in extras.iter().filter(|&&next| next <= limit) {
                let increment = next - drafts[0];
                let count = drafting.min(left / increment);
                for draft in &mut drafts[..count] {
                    *draft = next;
                }
                left -= count * increment;
                if count < drafting {
                    break;
                }
            }
            drafts
        };
        let mut drafts = plan(spare);
        let rows = items + drafts.iter().sum::<usize>();
        if let Some(width) = warmup
            .widths
            .iter()
            .copied()
            .filter(|width| (items..=rows).contains(width))
            .max()
            .filter(|&width| width < rows)
        {
            drafts = plan(width - items);
        }
        let width = items + drafts.iter().sum::<usize>();
        if width == items {
            continue;
        }
        let mut reserved = drafts.clone();
        for &next in &extras {
            let candidate: Vec<_> = drafts
                .iter()
                .enumerate()
                .map(|(index, &draft)| if index < drafting { next.max(draft) } else { 0 })
                .collect();
            if candidate.iter().sum::<usize>() > spare {
                break;
            }
            reserved = candidate;
        }
        let reserved_rows = items + reserved.iter().sum::<usize>();
        let floor = (0..=warmup.floor)
            .rev()
            .find(|&floor| items + reserved.iter().map(|&d| d.min(floor)).sum::<usize>() <= width)
            .unwrap_or(0);
        let floor_rows = items + reserved.iter().map(|&d| d.min(floor)).sum::<usize>();
        if reserved_rows > width
            && floor_rows < width
            && reserved.iter().filter(|&&drafts| drafts > floor).count() >= 2
            && !shapes
                .iter()
                .any(|s| (s.width, s.reserved_rows) == (width, reserved_rows))
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
