//! Bounded metadata reuse. Cached plans still pass ordinary image validation.

use super::{WeightPlanRequestV1, WeightPlanV1};

#[cfg(target_os = "linux")]
mod files;
#[cfg(target_os = "linux")]
pub(super) use files::Cache;

#[cfg(not(target_os = "linux"))]
pub(super) struct Cache;
#[cfg(not(target_os = "linux"))]
impl Cache {
    pub fn selected() -> Option<Self> {
        None
    }
    pub fn read(&self, _key: &str) -> Option<WeightPlanV1> {
        None
    }
    pub fn write_plan(&self, _key: &str, _plan: &WeightPlanV1) {}
}

fn key(request: &WeightPlanRequestV1) -> String {
    let mut bytes = request.encode();
    bytes.extend_from_slice(b"chopinfer-weight-plan-cache-v1\0");
    runtime_weights::sha256_bytes(&bytes)
}

fn matches_request(plan: &WeightPlanV1, request: &WeightPlanRequestV1) -> bool {
    // Mirror BundleLibraryV1::weight_plan, including the primary-image wildcard.
    (plan.ranks, plan.rank) == (request.ranks, request.rank)
        && (request.image_role.is_empty() || plan.image_role == request.image_role)
}

pub(super) fn prepare<T>(
    cache: Option<&Cache>,
    ranks: usize,
    mut request: impl FnMut(usize) -> WeightPlanRequestV1,
    mut generate: impl FnMut(&WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String>,
    mut validate: impl FnMut(&[WeightPlanV1]) -> Result<T, String>,
) -> Result<Option<(T, usize, usize)>, String> {
    let mut keys = Vec::new();
    if let Some(cache) = cache {
        let mut cached = Some(Vec::with_capacity(ranks));
        for rank in 0..ranks {
            // Keep only one copy of checkpoint headers and documents live.
            let request = request(rank);
            let key = key(&request);
            if cached.is_some() {
                match cache
                    .read(&key)
                    .filter(|plan| matches_request(plan, &request))
                {
                    Some(plan) => cached.as_mut().unwrap().push(plan),
                    None => cached = None,
                }
            }
            keys.push(key);
        }
        if let Some(plans) = cached {
            let records = plans.iter().map(|plan| plan.records.len()).sum();
            if let Ok(image) = validate(&plans) {
                return Ok(Some((image, records, ranks)));
            }
        }
    }
    let mut plans = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        let request = request(rank);
        match generate(&request)? {
            Some(plan) => plans.push(plan),
            None if rank == 0 => return Ok(None),
            None => {
                return Err(format!(
                    "{}: the bundle declined rank {rank} only",
                    request.image_role
                ));
            }
        }
    }
    let records = plans.iter().map(|plan| plan.records.len()).sum();
    let image = validate(&plans)?;
    if let Some(cache) = cache {
        // Encode and release one rank at a time, after ordinary validation.
        for (key, plan) in keys.iter().zip(plans) {
            cache.write_plan(key, &plan);
        }
    }
    Ok(Some((image, records, 0)))
}

/// Conservative wire-size bound without allocating an encoding. Destructure
/// every field so adding payload fields requires updating this estimate.
#[cfg(target_os = "linux")]
fn encoded_bound(plan: &WeightPlanV1) -> usize {
    use runtime_bundle_abi::weight_plan::{
        NamedBytesV1, PlannedViewV1, RecordSourceV1, WeightRecordV1,
    };
    let WeightPlanV1 {
        ranks: _,
        rank: _,
        image_role,
        alignment: _,
        arena_bytes: _,
        views,
        components,
        records,
    } = plan;
    let mut bytes = 256_usize.saturating_add(image_role.len());
    for PlannedViewV1 {
        id,
        alignment: _,
        placement: _,
        offset: _,
        byte_length: _,
    } in views
    {
        bytes = bytes.saturating_add(128).saturating_add(id.len());
    }
    for NamedBytesV1 {
        name,
        bytes: payload,
    } in components
    {
        bytes = bytes
            .saturating_add(128)
            .saturating_add(name.len())
            .saturating_add(payload.len());
    }
    for WeightRecordV1 {
        arena_offset: _,
        byte_length: _,
        transform: _,
        params,
        sources,
        axis: _,
        ranges,
        stored_shape,
    } in records
    {
        bytes = bytes
            .saturating_add(128)
            .saturating_add(params.len())
            .saturating_add(ranges.len().saturating_mul(16))
            .saturating_add(stored_shape.len().saturating_mul(8));
        for RecordSourceV1 { file: _, tensor } in sources {
            bytes = bytes.saturating_add(128).saturating_add(tensor.len());
        }
    }
    bytes
}

#[cfg(test)]
mod tests;
