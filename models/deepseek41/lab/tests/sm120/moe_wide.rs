//! Complete wide MoE pairs keep explicit reference schedules and guarded spans.
mod device;
mod router;

#[test]
fn router64_cpu_inventory() {
    router::cpu_inventory().unwrap();
}

#[test]
#[ignore = "requires an allocated GPU pair"]
fn router64_gpu() {
    let worker: u32 = std::env::var("CHOPINFER_MOE_WIDE_WORKER")
        .expect("set worker 0 or 1")
        .parse()
        .unwrap();
    assert!(worker < 2);
    let checked = std::env::var("CHOPINFER_MOE_WIDE_MEMCHECK").as_deref() == Ok("1");
    if std::env::var("CHOPINFER_MOE_WIDE_DRY_RUN").as_deref() == Ok("1") {
        router::cpu_worker(worker, checked).unwrap();
        return;
    }
    let ran = kernel_harness::gpu_test(256 << 20, |s| router::run(s, worker, checked)).unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
