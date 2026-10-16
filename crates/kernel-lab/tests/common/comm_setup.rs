//! CPU validation of collective test plans, entry names and buffer bindings.

use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Validate};
use kernel_lab::{LabResult, launch};
use std::collections::BTreeMap;

pub fn check(
    cubin: &str,
    symbol: &str,
    plan: LaunchPlan,
    args: &ArgumentList,
    buffer_bytes: &BTreeMap<u64, u64>,
) -> LabResult<()> {
    if !kernel_lab::comm_cubin(cubin)? || !kernel_lab::cubin_entries(cubin)?.contains(&symbol) {
        return Err(format!("{symbol} is absent from collective cubin {cubin}"));
    }
    launch::config(plan)?;
    args.validate().map_err(|e| e.to_string())?;
    let bound = |id: &u64| match buffer_bytes.get(id) {
        Some(&bytes) if bytes > 0 => Ok(()),
        _ => Err(format!("{symbol}: buffer {id} is absent or empty")),
    };
    for argument in &args.0 {
        match argument {
            Argument::DeviceBuffer(id) => bound(id)?,
            Argument::DevicePointerArray(ids) => {
                for id in ids {
                    bound(id)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
