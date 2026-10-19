//! One selected GPU per job worker. Both timing arms run on that GPU.
use kernel_harness::{Cubins, LabResult, Session};
use serde_json::json;

pub(super) const FOLD: &str = "mxfp8_fold_stream::stream_fold_gpu";
pub(super) const AHEAD: &str = "mxfp8_fold_stream::ahead::stream_ahead_gpu";
pub(super) const SUBLAYER: &str = "mxfp8_fold_stream::sublayer::stream_sublayer_gpu";
const TESTS: [&str; 3] = [FOLD, AHEAD, SUBLAYER];

fn check(test: &str, ordinals: &[i32]) -> LabResult<()> {
    if !TESTS.contains(&test) || ordinals.len() != 1 {
        return Err(format!(
            "{test} requires exactly one GPU ordinal per worker, got {ordinals:?}"
        ));
    }
    Ok(())
}

pub(super) fn run(
    test: &str,
    cubins: Cubins,
    run: impl FnOnce(&Session) -> LabResult<()>,
) -> LabResult<Option<()>> {
    if let Some(ordinals) = kernel_harness::selected_ordinals()? {
        check(test, &ordinals)?;
    }
    kernel_harness::gpu_test_with(cubins, 3 << 30, run)
}

#[test]
fn worker_ordinal_dry_run() {
    // The job invokes this through the same worker script and environment
    // setup as its GPU tests. Parsing ordinals never opens a CUDA session.
    let ordinals = kernel_harness::selected_ordinals()
        .unwrap()
        .expect("the dry run must select a worker ordinal");
    for test in TESTS {
        check(test, &ordinals).unwrap();
        println!(
            "{}",
            json!({"test":test,"required_ordinals":1,"selected_ordinals":ordinals,"gpu_launched":false})
        );
    }
}

#[test]
fn worker_rejects_missing_or_multiple_ordinals() {
    for test in TESTS {
        for device in [0, 1] {
            assert!(check(test, &[device]).is_ok());
        }
        assert!(check(test, &[]).is_err());
        assert!(check(test, &[0, 1]).is_err());
    }
    assert!(check("unregistered test", &[0]).is_err());
}
