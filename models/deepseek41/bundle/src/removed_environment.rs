//! Warn about retired execution switches without refusing older launch environments.

const REMOVED: &[&str] = &[
    "CHOPINFER_DEEPSEEK41_AR_ENGRAM",
    "CHOPINFER_DEEPSEEK41_AR_STATUS_PACKET",
    "CHOPINFER_DEEPSEEK41_BF16_PARTIALS",
    "CHOPINFER_DEEPSEEK41_CEIL_SPLIT",
    "CHOPINFER_DEEPSEEK41_KV_OVERLAP",
    "CHOPINFER_DEEPSEEK41_NATIVE_MOE_PREFILL",
    "CHOPINFER_DEEPSEEK41_PREFILL_BF16_GATHER",
    "CHOPINFER_DEEPSEEK41_PREFILL_DMA",
    "CHOPINFER_DEEPSEEK41_PREFILL_DMA_CONSUME",
    "CHOPINFER_DEEPSEEK41_PREFILL_DMA_PRODUCE",
    "CHOPINFER_DEEPSEEK41_PREFILL_DMA_PUSH",
    "CHOPINFER_DEEPSEEK41_PREFILL_ENGRAM",
    "CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP",
    "CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE",
    "CHOPINFER_DEEPSEEK41_PREFILL_PLAN_CACHE_PLANS",
    "CHOPINFER_DEEPSEEK41_QUERY_WINDOW_TWIN",
    "CHOPINFER_DEEPSEEK41_SHARED_OVERLAP",
    "CHOPINFER_DSPARK_BATCHED_COMMIT",
    "CHOPINFER_DSPARK_BOUNDARY_PACKET",
    "CHOPINFER_DSPARK_ENGRAM_EARLY",
    "CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK",
    "CHOPINFER_DSPARK_INDEXED_RETENTION",
    "CHOPINFER_DSPARK_PREFETCH",
    "CHOPINFER_DSPARK_UNDRAFTED_STATUS",
];

pub(crate) fn warn(log: impl Fn(&str)) {
    for &name in REMOVED {
        if std::env::var_os(name).is_some() {
            log(&format!(
                "DeepSeek warning: {name} is ignored (removed environment switch)"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStringExt;

    #[test]
    fn startup_warns_once_per_removed_variable_without_reading_its_value() {
        const CHILD: &str = "CHOPINFER_C26_REMOVED_ENV_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "removed_environment::tests::startup_warns_once_per_removed_variable_without_reading_its_value", "--nocapture"])
                .env(CHILD, "1")
                .envs(REMOVED.iter().map(|&name| (name, "0")))
                .env("CHOPINFER_DEEPSEEK41_AR_ENGRAM", "")
                .env("CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK", std::ffi::OsString::from_vec(vec![255]))
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        assert!(REMOVED.windows(2).all(|pair| pair[0] < pair[1]));
        let shape = crate::shape::topologies()[0];
        let host = crate::fake_host_tests::FakeHost::new(shape, crate::config::KV_LAYOUTS[0]);
        let mut instance = crate::fake_host_tests::create(&host, 1);
        let logs = host.logs.borrow();
        let warnings = logs
            .iter()
            .filter(|line| line.starts_with("DeepSeek warning: "))
            .collect::<Vec<_>>();
        assert_eq!(warnings.len(), REMOVED.len());
        for (&name, line) in REMOVED.iter().zip(warnings) {
            assert_eq!(
                line,
                &format!("DeepSeek warning: {name} is ignored (removed environment switch)")
            );
        }
        drop(logs);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
