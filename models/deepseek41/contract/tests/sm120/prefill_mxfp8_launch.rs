//! The comparison's function attributes and offline launch audit share one policy.
use kernel_core::backend::Launch;
use kernel_core::contract::{LaunchPlan, Validate};
use kernel_harness::LabResult;

pub const SHARED_ATTRIBUTE: &str = "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES";
const DEFAULT_SHARED: u32 = 48 * 1024;
const SM120_SHARED: u32 = 101_376;

// The callback is the actual driver setter on GPU and an attribute recorder
// in the CPU dry run, so omitting the opt-in fails the same launch audit.
pub fn configure_shared(bytes: u32, set: impl FnOnce(u32) -> LabResult<()>) -> LabResult<()> {
    if bytes != 0 {
        set(bytes)?;
    }
    Ok(())
}

pub fn audit(
    plan: LaunchPlan,
    max_threads: u32,
    static_shared: u32,
    applied_shared: Option<u32>,
) -> LabResult<()> {
    plan.validate().map_err(|e| e.to_string())?;
    kernel_harness::launch::config(plan)?;
    if plan.cluster != [1, 1, 1] {
        return Err(format!(
            "unexpected cluster {:?}: these entries use no cluster",
            plan.cluster
        ));
    }
    let threads: u32 = plan.block.iter().product();
    if threads > max_threads {
        return Err(format!(
            "block {:?} exceeds compiled max_threads={max_threads}",
            plan.block
        ));
    }
    let maximum = match applied_shared {
        Some(bytes)
            if bytes
                .checked_add(static_shared)
                .is_some_and(|n| n <= SM120_SHARED) =>
        {
            bytes
        }
        Some(bytes) => {
            return Err(format!(
                "{SHARED_ATTRIBUTE}={bytes} plus static_shared={static_shared} exceeds SM120 limit"
            ));
        }
        None => DEFAULT_SHARED.saturating_sub(static_shared),
    };
    if plan.dynamic_shared_bytes > maximum {
        return Err(format!(
            "dynamic_shared={} exceeds {SHARED_ATTRIBUTE}={applied_shared:?} (effective {maximum}, static_shared={static_shared})",
            plan.dynamic_shared_bytes
        ));
    }
    Ok(())
}

pub fn context(module: &str, launch: &Launch, configured_shared: u32) -> String {
    let plan = launch.plan;
    format!(
        "module={module} symbol={} rows={} grid={:?} block={:?} cluster={:?} dynamic_shared={} {SHARED_ATTRIBUTE}={configured_shared} CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION={}",
        launch.symbol,
        launch.rows,
        plan.grid,
        plan.block,
        plan.cluster,
        plan.dynamic_shared_bytes,
        plan.pdl
    )
}

#[test]
fn launch_audit_rejects_missing_attributes_and_invalid_geometry() {
    let entry = kernel_lab_families::mxfp8_prefill::entries().remove(0);
    let plan = kernels_sm120::gemm::mxfp8_prefill::plan(&entry, 128).unwrap();
    let shared = plan.dynamic_shared_bytes;
    let error = audit(plan, 384, 1056, None).unwrap_err();
    assert!(error.contains(SHARED_ATTRIBUTE) && error.contains("66560"));
    assert!(audit(plan, 384, 1056, Some(shared - 1)).is_err());
    assert!(audit(plan, 384, 1056, Some(SM120_SHARED)).is_err());
    assert!(
        audit(
            LaunchPlan {
                block: [385, 1, 1],
                ..plan
            },
            384,
            1056,
            Some(shared)
        )
        .is_err()
    );
    assert!(
        audit(
            LaunchPlan {
                block: [1, 1, 65],
                ..plan
            },
            384,
            1056,
            Some(shared)
        )
        .is_err()
    );
    assert!(
        audit(
            LaunchPlan {
                cluster: [2, 1, 1],
                ..plan
            },
            384,
            1056,
            Some(shared)
        )
        .is_err()
    );
    assert!(
        audit(
            LaunchPlan {
                cluster: [0, 1, 1],
                ..plan
            },
            384,
            1056,
            Some(shared)
        )
        .is_err()
    );
    assert!(
        audit(
            LaunchPlan {
                grid: [2, 1, 1],
                cluster: [2, 1, 1],
                ..plan
            },
            384,
            1056,
            Some(shared)
        )
        .is_err()
    );
    assert!(audit(plan, 384, 1056, Some(shared)).is_ok());
    let mut applied = None;
    configure_shared(shared, |bytes| {
        applied = Some(bytes);
        Ok(())
    })
    .unwrap();
    assert_eq!(applied, Some(shared));
    assert_eq!(
        configure_shared(shared, |_| Err("driver rejected attribute".into())).unwrap_err(),
        "driver rejected attribute"
    );
    configure_shared(0, |_| panic!("default kernel must not opt in")).unwrap();
}
