//! Draft counts from per-request acceptance and a fixed step-cost table.
//!
//! The table comes from a calibration of one bundle on one machine, so a
//! plan depends only on the batch, each request's verified outcomes and the
//! table: never on measured step times. Acceptance compares drafts with the
//! target's own samples, so no plan changes the tokens a request emits.

use super::MAX_DRAFT_TOKENS;

/// Concurrencies one table may describe.
pub const MAX_STEP_COSTS: usize = 16;

/// Step time at one concurrency: a step of `items` items, `drafting` of
/// which draft, with `rows` decode-phase rows in all, takes
/// `base_us + drafting_us * drafting + drafting_sq_us * drafting^2 +
/// row_us * (rows - items)` microseconds. `base_us` is the step with no
/// drafts, so a plan without drafts is the AR step of this engine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StepCost {
    pub items: u16,
    pub base_us: u32,
    pub drafting_us: u32,
    pub drafting_sq_us: u32,
    pub row_us: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepCostsError {
    Empty,
    TooMany {
        maximum: usize,
    },
    /// Entries must have positive, strictly increasing item counts and a
    /// positive base cost.
    Invalid,
    Syntax,
}

/// Step costs of every calibrated concurrency, by increasing items. Other
/// concurrencies interpolate each coefficient linearly between the nearest
/// entries; outside them the nearest entry applies.
///
/// Each distinct table is kept for the life of the process, so the engine's
/// settings stay small and `Copy`. A process builds a handful at most. The
/// default table is empty: a placeholder no plan may use.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct StepCosts {
    entries: &'static [StepCost],
}

impl std::fmt::Debug for StepCosts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.entries()).finish()
    }
}

/// Every table built so far.
static TABLES: std::sync::Mutex<Vec<&'static [StepCost]>> = std::sync::Mutex::new(Vec::new());

impl StepCosts {
    pub fn new(entries: &[StepCost]) -> Result<Self, StepCostsError> {
        if entries.is_empty() {
            return Err(StepCostsError::Empty);
        }
        if entries.len() > MAX_STEP_COSTS {
            return Err(StepCostsError::TooMany {
                maximum: MAX_STEP_COSTS,
            });
        }
        if entries
            .iter()
            .any(|entry| entry.items == 0 || entry.base_us == 0)
            || entries
                .windows(2)
                .any(|pair| pair[0].items >= pair[1].items)
        {
            return Err(StepCostsError::Invalid);
        }
        let mut tables = TABLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entries = match tables.iter().find(|table| **table == entries) {
            Some(table) => *table,
            None => {
                let table: &'static [StepCost] = Box::leak(entries.into());
                tables.push(table);
                table
            }
        };
        Ok(Self { entries })
    }

    /// Parses `items:base:drafting:drafting_sq:row` entries separated by
    /// commas, all in whole microseconds, for example
    /// `1:15900:4400:0:780,8:28000:4400:0:780`.
    pub fn parse(text: &str) -> Result<Self, StepCostsError> {
        let entries = text
            .split(',')
            .map(|entry| {
                let fields = entry
                    .split(':')
                    .map(|field| field.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| StepCostsError::Syntax)?;
                let [items, base_us, drafting_us, drafting_sq_us, row_us] = fields[..] else {
                    return Err(StepCostsError::Syntax);
                };
                Ok(StepCost {
                    items: u16::try_from(items).map_err(|_| StepCostsError::Invalid)?,
                    base_us,
                    drafting_us,
                    drafting_sq_us,
                    row_us,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(&entries)
    }

    pub fn entries(&self) -> &'static [StepCost] {
        self.entries
    }

    /// Estimated microseconds of a step of `items` items, `drafting` of them
    /// drafting, and `rows` decode-phase rows.
    pub fn estimate(&self, items: usize, drafting: usize, rows: usize) -> f64 {
        let entries = self.entries();
        let at = |entry: &StepCost| {
            [
                f64::from(entry.base_us),
                f64::from(entry.drafting_us),
                f64::from(entry.drafting_sq_us),
                f64::from(entry.row_us),
            ]
        };
        let upper = entries.partition_point(|entry| usize::from(entry.items) < items);
        let [base, linear, square, row] = if upper == 0 {
            at(&entries[0])
        } else if upper == entries.len() {
            at(&entries[entries.len() - 1])
        } else {
            let (low, high) = (&entries[upper - 1], &entries[upper]);
            let span = f64::from(high.items - low.items);
            let weight = (items as f64 - f64::from(low.items)) / span;
            let (low, high) = (at(low), at(high));
            std::array::from_fn(|index| low[index] + weight * (high[index] - low[index]))
        };
        let drafting = drafting as f64;
        base + linear * drafting
            + square * drafting * drafting
            + row * rows.saturating_sub(items) as f64
    }
}

/// One item's draft limit and its chance of reaching each position: the
/// product of the per-position acceptance up to it.
#[derive(Clone, Copy, Debug)]
pub struct TableCandidate<'a> {
    pub maximum_draft_tokens: usize,
    pub survival: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct TablePlan {
    pub drafts: Vec<usize>,
    pub expected_tokens: f64,
    pub estimated_us: f64,
}

/// The draft counts that maximize expected emitted tokens per estimated
/// step time, with at most `remaining` drafts and `maximum_drafting`
/// drafting items, using only counts `executable` holds. For each count of
/// drafting items and drafts, a dynamic program finds the most expected
/// tokens; the table prices every pair. Ties prefer fewer drafts, then
/// fewer drafting items, and resolve equal plans in a fixed item order, so
/// the plan is a function of its inputs alone.
pub fn plan(
    candidates: &[TableCandidate<'_>],
    costs: &StepCosts,
    remaining: usize,
    maximum_drafting: usize,
    executable: &[bool; MAX_DRAFT_TOKENS + 1],
) -> TablePlan {
    let items = candidates.len();
    let limits = candidates
        .iter()
        .map(|candidate| {
            candidate
                .maximum_draft_tokens
                .min(candidate.survival.len())
                .min(MAX_DRAFT_TOKENS)
        })
        .collect::<Vec<_>>();
    let drafts_cap = remaining.min(limits.iter().sum());
    let drafting_cap = maximum_drafting.min(limits.iter().filter(|&&limit| limit > 0).count());
    // best[d][r]: most expected accepted drafts with d drafting items and r
    // drafts among the items seen so far; choice[i][d][r]: item i's count.
    let width = drafts_cap + 1;
    let none = f64::NEG_INFINITY;
    let mut best = vec![none; (drafting_cap + 1) * width];
    best[0] = 0.0;
    let mut choice = vec![0_u8; items * (drafting_cap + 1) * width];
    for (index, candidate) in candidates.iter().enumerate() {
        let mut gains = vec![0.0; limits[index] + 1];
        for k in 1..gains.len() {
            gains[k] = gains[k - 1] + candidate.survival[k - 1];
        }
        let mut next = vec![none; best.len()];
        let chosen = &mut choice[index * best.len()..(index + 1) * best.len()];
        for d in 0..=drafting_cap {
            for r in 0..width {
                let mut value = none;
                let mut pick = 0;
                // k = 0 first: equal gains keep fewer drafts.
                for (k, gain) in gains.iter().enumerate().take(r + 1) {
                    if !executable[k] {
                        continue;
                    }
                    let (from_d, from_r) = match k {
                        0 => (d, r),
                        _ if d == 0 => continue,
                        _ => (d - 1, r - k),
                    };
                    let previous = best[from_d * width + from_r];
                    if previous > none && previous + gain > value {
                        value = previous + gain;
                        pick = k;
                    }
                }
                next[d * width + r] = value;
                chosen[d * width + r] = pick as u8;
            }
        }
        best = next;
    }
    let mut chosen_state = (0, 0);
    let mut chosen_rate = items as f64 / costs.estimate(items, 0, items);
    for r in 1..width {
        for d in 1..=drafting_cap.min(r) {
            let gain = best[d * width + r];
            if gain == none {
                continue;
            }
            let rate = (items as f64 + gain) / costs.estimate(items, d, items + r);
            // Strictly better only: ties keep fewer drafts and items.
            if rate > chosen_rate {
                chosen_rate = rate;
                chosen_state = (d, r);
            }
        }
    }
    let (mut d, mut r) = chosen_state;
    let mut drafts = vec![0; items];
    for index in (0..items).rev() {
        let k = usize::from(choice[index * best.len() + d * width + r]);
        drafts[index] = k;
        if k > 0 {
            d -= 1;
            r -= k;
        }
    }
    let (drafting, total) = chosen_state;
    TablePlan {
        expected_tokens: items as f64 + best[drafting * width + total].max(0.0),
        estimated_us: costs.estimate(items, drafting, items + total),
        drafts,
    }
}

#[cfg(test)]
mod tests;
