//! Passive snapshots at selected lifecycle boundaries, outside timed windows.
use super::Instance;
use crate::dspark::{execution::Input, staging::Staging, state::State};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

struct Plan {
    before_draft: BTreeSet<u64>,
    after_commit: BTreeSet<u64>,
    requests: Vec<u64>,
    max_bytes: u64,
}

struct Observation<'a> {
    boundary: &'static str,
    step: u64,
    state: &'a State,
    staging: &'a Staging,
    active: &'a [u64],
    noise: Value,
}

impl Plan {
    fn parse(value: Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("transition capture plan must be an object")?;
        if object.len() != 5
            || value["version"] != 1
            || object.keys().any(|key| {
                ![
                    "version",
                    "before_draft",
                    "after_commit",
                    "requests",
                    "max_bytes",
                ]
                .contains(&key.as_str())
            })
        {
            return Err("unsupported transition capture plan".into());
        }
        let numbers = |key: &str, maximum: usize| -> Result<Vec<u64>, String> {
            let values = value[key]
                .as_array()
                .ok_or("transition capture list is absent")?;
            if values.len() > maximum {
                return Err("transition capture list exceeds its bound".into());
            }
            let numbers = values
                .iter()
                .map(|v| {
                    v.as_u64().filter(|&n| n > 0).ok_or_else(|| {
                        "transition capture ids must be positive integers".to_owned()
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if numbers.iter().collect::<BTreeSet<_>>().len() != numbers.len() {
                return Err("duplicate transition capture id".into());
            }
            Ok(numbers)
        };
        let before_draft = numbers("before_draft", 8)?.into_iter().collect();
        let after_commit = numbers("after_commit", 8)?.into_iter().collect();
        let requests = numbers("requests", 32)?;
        let max_bytes = value["max_bytes"]
            .as_u64()
            .filter(|&n| n > 0 && n <= 64 << 30)
            .ok_or("transition capture budget must be in 1..=64 GiB")?;
        let plan = Self {
            before_draft,
            after_commit,
            requests,
            max_bytes,
        };
        if plan.requests.is_empty()
            || (plan.before_draft.is_empty() && plan.after_commit.is_empty())
            || plan
                .before_draft
                .iter()
                .chain(&plan.after_commit)
                .any(|&step| step > 1024)
        {
            return Err("transition capture plan is empty or exceeds the step bound".into());
        }
        Ok(plan)
    }

    fn read(root: &Path) -> Result<Option<Self>, String> {
        let path = root.join("transitions.json");
        let metadata = match std::fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            result => result.map_err(|error| error.to_string())?,
        };
        if !metadata.is_file() || metadata.len() > 64 << 10 {
            return Err("transition capture plan must be a regular file of at most 64 KiB".into());
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .map_err(|error| error.to_string())?
            .take((64 << 10) + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        if bytes.len() > 64 << 10 {
            return Err("transition capture plan grew past its bound".into());
        }
        Self::parse(serde_json::from_slice(&bytes).map_err(|error| error.to_string())?).map(Some)
    }
}

impl Instance {
    pub(super) fn capture_before_draft(
        &self,
        state: &State,
        staging: &Staging,
        inputs: &[Input],
    ) -> Result<(), String> {
        let Some(root) = self.active_dspark_state_capture() else {
            return Ok(());
        };
        let Some(plan) = Plan::read(root)? else {
            return Ok(());
        };
        let step = self
            .dspark
            .as_ref()
            .ok_or("capture has no drafter")?
            .diagnostic
            .step_id;
        if !plan.before_draft.contains(&step) {
            return Ok(());
        }
        let requests: Vec<_> = inputs
            .iter()
            .filter(|input| input.rows.kind.drafts() > 0)
            .map(|input| input.rows.request)
            .collect();
        let noise: Vec<_> = inputs
            .iter()
            .map(|input| {
                json!({
                    "request": input.rows.request, "rows": format!("{:?}", input.rows),
                    "tokens": input.tokens, "seed": input.noise.seed,
                    "generated": input.noise.generated, "temperature_bits": input.noise.temperature,
                })
            })
            .collect();
        self.capture_transition(
            &plan,
            root,
            Observation {
                boundary: "before-draft",
                step,
                state,
                staging,
                active: &requests,
                noise: json!(noise),
            },
        )
    }

    pub(in crate::instance) fn capture_transition_commit(
        &self,
        committed: &[bundle_runtime::ledger::Committed],
    ) -> Result<(), String> {
        let Some(root) = self.active_dspark_state_capture() else {
            return Ok(());
        };
        let Some(plan) = Plan::read(root)? else {
            return Ok(());
        };
        let Some(device) = self.dspark.as_ref() else {
            return Ok(());
        };
        let step = device.diagnostic.step_id;
        if !plan.after_commit.contains(&step) {
            return Ok(());
        }
        let runtime = device
            .runtime
            .as_ref()
            .ok_or("capture lifecycle is borrowed")?;
        let staging = device
            .staging
            .as_ref()
            .ok_or("capture staging is borrowed")?;
        let requests: Vec<_> = committed.iter().map(|item| item.request).collect();
        self.capture_transition(
            &plan,
            root,
            Observation {
                boundary: "after-commit",
                step,
                state: runtime.state(),
                staging,
                active: &requests,
                noise: Value::Null,
            },
        )
    }

    fn capture_transition(
        &self,
        plan: &Plan,
        root: &Path,
        observation: Observation<'_>,
    ) -> Result<(), String> {
        let Observation {
            boundary,
            step,
            state,
            staging,
            active,
            noise,
        } = observation;
        if plan
            .requests
            .iter()
            .any(|request| !active.contains(request))
        {
            return Err("selected transition request is absent from this boundary".into());
        }
        let directory = root.join(format!("transition-{boundary}-{step}"));
        let metadata = serde_json::to_vec_pretty(&json!({
            "schema": "chopinfer.deepseek41.transition-capture.v1",
            "step": step, "boundary": boundary, "requests": plan.requests,
            "state_debug": format!("{state:?}"),
            "staging": staging.capture_metadata(&plan.requests), "sampling_inputs": noise,
        }))
        .map_err(|error| error.to_string())?;
        let next = self
            .transition_capture_bytes
            .get()
            .checked_add(metadata.len() as u64)
            .filter(|&next| next <= plan.max_bytes)
            .ok_or("DSpark transition capture exceeds its byte budget")?;
        self.transition_capture_bytes.set(next);
        std::fs::create_dir(&directory).map_err(|error| error.to_string())?;
        for &request in &plan.requests {
            let context = state
                .context(request)
                .ok_or("captured request has no committed context")?;
            self.capture_complete_state_with_context(
                request,
                context.end,
                Some(context),
                &directory,
                Some((&self.transition_capture_bytes, plan.max_bytes)),
            )?;
        }
        std::fs::write(directory.join("host.json"), metadata).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_plans_refuse_unbounded_or_ambiguous_captures() {
        let good = json!({"version":1, "before_draft":[272,273], "after_commit":[271,272,273],
            "requests":[1,2], "max_bytes": 1_u64 << 30});
        assert!(Plan::parse(good.clone()).is_ok());
        for (key, value) in [
            ("version", json!(2)),
            ("requests", json!([])),
            ("requests", json!([1, 1])),
            ("before_draft", json!([1025])),
            ("after_commit", json!([0])),
            ("max_bytes", json!(0)),
            ("max_bytes", json!(65_u64 << 30)),
            ("surplus", json!(true)),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            assert!(Plan::parse(bad).is_err(), "{key}");
        }
    }
}
