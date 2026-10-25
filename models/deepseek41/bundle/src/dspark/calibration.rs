//! Diagnostic JSON for confidence calibration, never an input to serving.

use super::execution::{Diagnostic, Outcome};
use super::state::BLOCK_ROWS;
use model_deepseek41_contract::own_ops::dspark::CONFIDENCE_CONTRACT;

// The model-neutral suffix selects the host's bounded routine-log queue.
pub const SCHEMA: &str = "chopinfer.deepseek41.drafter-confidence.v1";

/// The coordinator assigns a unique instance ID within the run, and a new
/// step ID for every execute attempt, including retries after cancellation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Context {
    pub instance_id: u64,
    pub step_id: u64,
}

/// Encode one completed lifecycle diagnostic. The current commit ABI proves
/// an acceptance prefix only; it supplies no evidence for rejection labels.
/// Refuse malformed outcomes instead of exporting misleading calibration data.
pub fn encode(context: Context, diagnostic: &Diagnostic) -> Result<String, &'static str> {
    if diagnostic.outcomes.contains(&Outcome::Rejected) {
        return Err(
            "confidence record does not support rejection labels without verifier evidence",
        );
    }
    let emitted = diagnostic
        .outcomes
        .iter()
        .position(|&outcome| outcome == Outcome::NotEmitted)
        .unwrap_or(BLOCK_ROWS as usize);
    if emitted == 0
        || diagnostic.outcomes[emitted..]
            .iter()
            .any(|&outcome| outcome != Outcome::NotEmitted)
    {
        return Err("confidence record requires a nonempty emitted prefix");
    }
    let computed = diagnostic.computed_positions as usize;
    if computed < emitted || computed > BLOCK_ROWS as usize {
        return Err("confidence computed depth does not cover emitted positions");
    }
    let prefix = &diagnostic.outcomes[..emitted];
    let completion = if prefix.iter().all(|&outcome| outcome == Outcome::Cancelled) {
        "cancelled"
    } else {
        let accepted = prefix
            .iter()
            .take_while(|&&outcome| outcome == Outcome::Accepted)
            .count();
        if prefix[accepted..]
            .iter()
            .any(|&outcome| outcome != Outcome::Unknown)
        {
            return Err("confidence record requires proven acceptances followed by unknowns");
        }
        "committed"
    };
    // Only integers and fixed strings enter the JSON. IDs are strings so
    // readers using IEEE-754 numbers cannot silently round a u64 identity.
    let outcomes = diagnostic.outcomes[..computed]
        .iter()
        .map(|outcome| match outcome {
            Outcome::Accepted => "\"accepted\"",
            Outcome::Unknown => "\"unknown\"",
            Outcome::Cancelled => "\"cancelled\"",
            Outcome::NotEmitted => "\"not_emitted\"",
            Outcome::Rejected => unreachable!("rejection was refused above"),
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        concat!(
            "{{\"schema\":\"{}\",\"confidence_contract\":\"{}\",",
            "\"instance_id\":\"{}\",\"step_id\":\"{}\",\"request_id\":\"{}\",",
            "\"main_rows\":{},\"computed_positions\":{},\"emitted_positions\":{},",
            "\"verify_rows\":{},\"completion\":\"{}\",",
            "\"acceptance_evidence\":\"retained_rows_lower_bound\",",
            "\"draft_ids\":{:?},\"confidence_f32_bits\":{:?},\"outcomes\":[{}]}}"
        ),
        SCHEMA,
        CONFIDENCE_CONTRACT,
        context.instance_id,
        context.step_id,
        diagnostic.request,
        diagnostic.main_rows,
        computed,
        emitted,
        emitted + 1,
        completion,
        &diagnostic.proposed[..computed],
        &diagnostic.confidence_bits[..computed],
        outcomes,
    ))
}

/// Call once from the coordinator's `Host::record_diagnostics`, passing
/// `|message| services.log(message)` as the sink. The schema selects bounded,
/// best-effort host logging with whole-message drops and drop accounting.
/// Custom sinks must also avoid waiting for output progress. Encoding or
/// logging failure must not change serving; calibration checks coverage.
pub fn record<E>(
    context: Context,
    diagnostics: &[Diagnostic],
    mut log: impl FnMut(&[u8]) -> Result<(), E>,
) {
    for diagnostic in diagnostics {
        if let Ok(message) = encode(context, diagnostic) {
            let _ = log(message.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests;
