//! Host clocks around existing work; timing never submits or waits for device work.
use super::calibration::Context;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TargetTimes {
    pub lower_us: u64,
    pub upload_us: u64,
    pub submit_us: u64,
    pub wait_us: u64,
}

#[derive(Default)]
pub(crate) struct ProjectionTimes {
    pub lower_us: u64,
    pub upload_us: u64,
    pub submit_us: u64,
    pub status_us: u64,
}

#[derive(Default)]
pub(crate) struct CommitTimes {
    pub total_us: u64,
    pub retention_us: u64,
    pub projection: ProjectionTimes,
    pub wait_us: u64,
    pub passes: u32,
    pub rows: u64,
}

impl CommitTimes {
    pub fn record(&self, context: Context) -> String {
        format!(
            concat!(
                "{{\"schema\":\"chopinfer.deepseek41.commit-shape.v1\",",
                "\"instance_id\":\"{}\",\"step_id\":\"{}\",",
                "\"projection_passes\":{},\"projection_rows\":{},",
                "\"micros\":{{\"total\":{},\"retention\":{},",
                "\"projection_lower\":{},\"projection_upload\":{},",
                "\"projection_submit\":{},\"projection_status\":{},\"wait\":{}}}}}"
            ),
            context.instance_id,
            context.step_id,
            self.passes,
            self.rows,
            self.total_us,
            self.retention_us,
            self.projection.lower_us,
            self.projection.upload_us,
            self.projection.submit_us,
            self.projection.status_us,
            self.wait_us,
        )
    }
}

pub(crate) fn measure<T>(field: Option<&mut u64>, work: impl FnOnce() -> T) -> T {
    let started = field.as_ref().map(|_| std::time::Instant::now());
    let result = work();
    if let (Some(field), Some(started)) = (field, started) {
        *field =
            field.saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
    }
    result
}

pub(crate) fn finish(field: Option<&mut u64>, started: Option<std::time::Instant>) {
    if let (Some(field), Some(started)) = (field, started) {
        *field =
            field.saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
    }
}
