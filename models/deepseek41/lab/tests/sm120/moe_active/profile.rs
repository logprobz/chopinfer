//! Eager routed projections for Nsight Compute. Device profiling is run
//! separately from graph parity and event timings because replay perturbs both.
use super::*;

pub(super) fn run(session: &Session) -> LabResult<()> {
    let bank = std::env::var("CHOPINFER_MOE_PROFILE_BANK")
        .map_err(|e| e.to_string())?
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    let rows = std::env::var("CHOPINFER_MOE_PROFILE_ROWS")
        .map_err(|e| e.to_string())?
        .parse::<u32>()
        .map_err(|e| e.to_string())?;
    if bank > 1 || ![1, 6, 8, 32, 48].contains(&rows) {
        return Err("invalid profile bank or rows".into());
    }
    let mut a = Arena {
        session: Some(session),
        next: 0,
        buffers: BTreeMap::new(),
    };
    let set = moe::expert_sets(&FLASH)[bank];
    let case = Case {
        set,
        rows,
        phase: if rows <= 32 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Verify
        },
        routing: Routing::Mixed,
    };
    let inputs = Inputs {
        input: a.filled(u64::from(rows) * 5120, 0x38)?,
        scales: a.filled(u64::from(rows) * 160, 124)?,
        live: a.bytes(&rows.to_le_bytes())?,
        logits: a.bytes(&case.logits())?,
        weights: banks(&mut a, set)?,
    };
    let shared = shared::weights(&mut a)?;
    let arms = [Mode::Old, Mode::BlockWarps, Mode::Streamed]
        .map(|m| arm(&mut a, case, m, inputs, shared))
        .into_iter()
        .collect::<LabResult<Vec<_>>>()?;
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    for arm in arms {
        for op in arm.launches {
            let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                .ok_or("profile cubin missing")?;
            let function = session
                .load_cubin(bytes)?
                .function(&op.symbol)
                .map_err(|e| e.to_string())?;
            launch(
                &session.context,
                &function,
                &session.stream,
                op.plan,
                &op.arguments,
                &bindings,
            )?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}
