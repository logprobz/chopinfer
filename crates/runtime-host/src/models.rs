//! `chopinfer models`: the installed families and their serving profiles,
//! the profile `serve --model` picks for a checkpoint, and the `serve
//! --model` options that serve another of the family's profiles. Each prints
//! one JSON document.
use crate::command::HostCommandErrorV1;
use crate::preparation::InstalledHelper;
use crate::profiles::{Profile, Profiles, Status};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) const MODELS_USAGE_V1: &str = "models --json | models pick --model DIR --json | models command --model DIR \
     --tensor-parallel-size N --profile NAME [-- SERVE-OPTIONS]";

fn usage(message: impl std::fmt::Display) -> HostCommandErrorV1 {
    HostCommandErrorV1::Usage(format!("{message}\nusage: {MODELS_USAGE_V1}"))
}

fn failure(message: impl std::fmt::Display) -> HostCommandErrorV1 {
    HostCommandErrorV1::Run(message.to_string())
}

pub(crate) fn run_v1(arguments: &[String]) -> Result<String, HostCommandErrorV1> {
    match arguments.split_first() {
        Some((json, [])) if json == "--json" => catalog_v1(),
        Some((verb, rest)) if verb == "pick" => pick_v1(rest),
        Some((verb, rest)) if verb == "command" => command_v1(rest),
        _ => Err(usage("models needs --json, pick or command")),
    }
}

/// The family's serving profiles, as its helper's `serving-profiles` prints
/// them.
fn serving_profiles(helper: &InstalledHelper) -> Result<Profiles, String> {
    let bytes = helper.serving_profiles().map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err(
            "it prints no serving profiles; install the helper that matches this chopinfer"
                .to_owned(),
        );
    }
    Profiles::parse(&bytes).map_err(|e| e.to_string())
}

/// `models --json`: every installed family with its profiles, and the
/// helpers that could not list theirs.
fn catalog_v1() -> Result<String, HostCommandErrorV1> {
    let helpers = crate::preparation::installed_helpers().map_err(failure)?;
    let (mut models, mut errors) = (Vec::new(), Vec::new());
    for helper in &helpers {
        match catalog_entry(helper) {
            Ok(model) => models.push(model),
            Err(message) => errors.push(json!({"helper": helper.name(), "message": message})),
        }
    }
    Ok(json!({"schema": "chopinfer.models.v1", "models": models, "errors": errors}).to_string())
}

fn catalog_entry(helper: &InstalledHelper) -> Result<Value, String> {
    let model = helper.model().map_err(|e| e.to_string())?;
    let file = serving_profiles(helper)?;
    let profiles = file
        .profiles
        .iter()
        .map(|(name, profile)| profile_entry(name, profile))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "model": model,
        "helper": helper.name(),
        "served_model_name": file.served_model_name,
        "profiles": profiles,
    }))
}

/// A profile as `models --json` lists it: its settings are what `serve
/// --model` resolves from it without explicit options.
pub(crate) fn profile_entry(name: &str, profile: &Profile) -> Result<Value, String> {
    let command = crate::profiles::direct_command(&crate::profiles::direct_defaults(profile))
        .map_err(|e| format!("profile {name}: {e}"))?;
    Ok(json!({
        "name": name,
        "status": match profile.status {
            Status::Qualified => "qualified",
            Status::Experimental => "experimental",
        },
        "tensor_parallel": profile.tensor_parallel,
        "drafter": profile.speculation.drafter,
        "speculative_tokens": profile.speculation.tokens,
        "max_sequences": profile.max_sequences,
        "qualified_max_sequences": profile.qualified_max_sequences,
        "qualification": profile.qualification,
        "settings": crate::command::effective_settings_v1(&command),
        "model_options": profile.model_option_texts(),
    }))
}

/// The value of `--flag` in `words`, given once.
fn take<'a>(
    words: &'a [String],
    flag: &str,
    seen: &mut Option<&'a str>,
    index: &mut usize,
) -> Result<(), HostCommandErrorV1> {
    let value = words
        .get(*index + 1)
        .ok_or_else(|| usage(format!("{flag} requires a value")))?;
    if seen.replace(value).is_some() {
        return Err(usage(format!("{flag} is given more than once")));
    }
    *index += 2;
    Ok(())
}

/// `models pick --model DIR --json`: the model whose helper accepts the
/// checkpoint, and for each rank count its profiles serve, the profile
/// `serve --model` picks (or its refusal) and how each other profile at
/// that count is served over it, or why it cannot be.
fn pick_v1(arguments: &[String]) -> Result<String, HostCommandErrorV1> {
    let (mut model, mut json) = (None, false);
    let mut index = 0;
    while let Some(word) = arguments.get(index) {
        match word.as_str() {
            "--model" => take(arguments, "--model", &mut model, &mut index)?,
            "--json" if !json => {
                json = true;
                index += 1;
            }
            _ => return Err(usage(format!("models pick does not accept {word}"))),
        }
    }
    let model = model.ok_or_else(|| usage("models pick needs --model DIR"))?;
    if !json {
        return Err(usage("models pick prints JSON: add --json"));
    }
    let executable = std::env::current_exe().map_err(failure)?;
    pick_document(Path::new(model), &executable)
}

/// The helper beside `executable` that `serve --model` selects for the
/// checkpoint directory `checkpoint`, at the default rank count or else at
/// the first other count an installed family declares, with the refusal of
/// each count it was refused at. When none accepts it, the refusal at the
/// default count.
fn checkpoint_family(
    checkpoint: &Path,
    executable: &Path,
) -> Result<(InstalledHelper, BTreeMap<u32, String>), String> {
    let default = crate::preparation::DEFAULT_TENSOR_PARALLEL_SIZE;
    let select = |ranks| {
        crate::preparation::checkpoint_helper_with_executable(checkpoint, ranks, executable)
            .map_err(|e| e.to_string())
    };
    let mut refusals = BTreeMap::new();
    match select(default) {
        Ok(helper) => return Ok((helper, refusals)),
        Err(refusal) => refusals.insert(default, refusal),
    };
    let declared: BTreeSet<u32> = crate::preparation::installed_helpers_with_executable(executable)
        .unwrap_or_default()
        .iter()
        .filter_map(|helper| serving_profiles(helper).ok())
        .flat_map(|file| file.profiles.into_values())
        .filter_map(|profile| u32::try_from(profile.tensor_parallel).ok())
        .filter(|&ranks| ranks != default)
        .collect();
    for ranks in declared {
        match select(ranks) {
            Ok(helper) => return Ok((helper, refusals)),
            Err(refusal) => refusals.insert(ranks, refusal),
        };
    }
    Err(refusals.remove(&default).unwrap_or_default())
}

/// The `models pick` document of the checkpoint directory `checkpoint`,
/// over the helpers installed beside `executable`.
fn pick_document(checkpoint: &Path, executable: &Path) -> Result<String, HostCommandErrorV1> {
    let document = |model: Option<&str>, refusal: Option<String>, sizes: Vec<Value>| {
        json!({
            "schema": "chopinfer.models-pick.v1",
            "model": model,
            "refusal": refusal,
            "sizes": sizes,
        })
        .to_string()
    };
    let (helper, refusals) = match checkpoint_family(checkpoint, executable) {
        Ok(found) => found,
        Err(refusal) => return Ok(document(None, Some(refusal), Vec::new())),
    };
    let name = helper.model().map_err(failure)?.to_owned();
    let file = match serving_profiles(&helper) {
        Ok(file) => file,
        Err(error) => {
            return Ok(document(
                Some(&name),
                Some(format!("{}: {error}", helper.name())),
                Vec::new(),
            ));
        }
    };
    let sizes: BTreeSet<usize> = file
        .profiles
        .values()
        .map(|profile| profile.tensor_parallel)
        .collect();
    let sizes = sizes
        .into_iter()
        .map(|size| {
            // A count every helper refused keeps serve's refusal.
            let picked = u32::try_from(size)
                .map_err(|e| e.to_string())
                .and_then(|ranks| match refusals.get(&ranks) {
                    Some(refusal) => Err(refusal.clone()),
                    None => helper.profile(checkpoint, ranks).map_err(|e| e.to_string()),
                });
            pick_size(picked, &file, size)
        })
        .collect();
    Ok(document(Some(&name), None, sizes))
}

/// One rank count of `models pick`, where the helper `picked` a profile or
/// refused the checkpoint.
fn pick_size(
    picked: Result<runtime_bundle::preparation::DirectProfile, String>,
    file: &Profiles,
    size: usize,
) -> Value {
    let at_size = || {
        file.profiles
            .iter()
            .filter(move |(_, profile)| profile.tensor_parallel == size)
    };
    let unavailable =
        |name: &str, reason: &str| json!({"name": name, "available": false, "reason": reason});
    let picked = picked.and_then(|picked| {
        crate::profiles::direct_entry(&picked)
            .map(|entry| (picked.name, entry))
            .map_err(|e| e.to_string())
    });
    let (pick, entry) = match picked {
        Ok(picked) => picked,
        Err(refusal) => {
            let reason = format!("serve --model refuses TP{size} for this checkpoint: {refusal}");
            let profiles: Vec<Value> = at_size()
                .map(|(name, _)| unavailable(name, &reason))
                .collect();
            return json!({
                "tensor_parallel": size,
                "pick": null,
                "refusal": refusal,
                "profiles": profiles,
            });
        }
    };
    let picked_entry = json!({
        "name": pick,
        "available": true,
        "arguments": [],
        "inherited_model_options": [],
    });
    let mut profiles = Vec::new();
    if !file.profiles.contains_key(&pick) {
        profiles.push(picked_entry.clone());
    }
    for (name, target) in at_size() {
        if *name == pick {
            profiles.push(picked_entry.clone());
            continue;
        }
        profiles.push(match crate::profiles::translate(&entry, target) {
            Ok(translation) => json!({
                "name": name,
                "available": true,
                "arguments": translation.arguments,
                "inherited_model_options": translation.inherited_model_options,
            }),
            Err(reason) => unavailable(name, &reason),
        });
    }
    json!({
        "tensor_parallel": size,
        "pick": pick,
        "refusal": null,
        "profiles": profiles,
    })
}

/// Flags the caller sets per instance, which `models command` leaves out.
pub(crate) const INSTANCE_OPTIONS: &[&str] = &["--listen", "--devices"];

/// `models command --model DIR --tensor-parallel-size N --profile NAME [--
/// SERVE-OPTIONS]`: the `serve --model` arguments that serve the family's
/// profile NAME with the explicit options, checked as `serve --model` checks
/// them before it prepares a package.
fn command_v1(arguments: &[String]) -> Result<String, HostCommandErrorV1> {
    let (mut model, mut ranks, mut profile) = (None, None, None);
    let mut index = 0;
    let mut explicit: &[String] = &[];
    while let Some(word) = arguments.get(index) {
        match word.as_str() {
            "--model" => take(arguments, "--model", &mut model, &mut index)?,
            "--tensor-parallel-size" => {
                take(arguments, "--tensor-parallel-size", &mut ranks, &mut index)?
            }
            "--profile" => take(arguments, "--profile", &mut profile, &mut index)?,
            "--" => {
                explicit = &arguments[index + 1..];
                break;
            }
            _ => return Err(usage(format!("models command does not accept {word}"))),
        }
    }
    let model = model.ok_or_else(|| usage("models command needs --model DIR"))?;
    let ranks_text = ranks.ok_or_else(|| usage("models command needs --tensor-parallel-size N"))?;
    let ranks: u32 = ranks_text
        .parse()
        .ok()
        .filter(|&ranks| ranks > 0)
        .ok_or_else(|| usage("--tensor-parallel-size takes a positive count"))?;
    let profile = profile.ok_or_else(|| usage("models command needs --profile NAME"))?;
    let checkpoint = Path::new(model);
    let helper = crate::preparation::checkpoint_helper(checkpoint, ranks).map_err(failure)?;
    let picked = helper.profile(checkpoint, ranks).map_err(failure)?;
    let words = command_arguments(
        &picked,
        &|| serving_profiles(&helper),
        (model, ranks),
        profile,
        explicit,
    )?;
    Ok(json!({"schema": "chopinfer.models-command.v1", "arguments": words}).to_string())
}

/// The `serve --model` arguments of `models command` for the checkpoint
/// directory `model` at `ranks`, where the helper `picked` a profile and
/// `file` reads the family's profiles.
fn command_arguments(
    picked: &runtime_bundle::preparation::DirectProfile,
    file: &dyn Fn() -> Result<Profiles, String>,
    (model, ranks): (&str, u32),
    profile: &str,
    explicit: &[String],
) -> Result<Vec<String>, HostCommandErrorV1> {
    if let Some(flag) = explicit
        .iter()
        .find(|word| INSTANCE_OPTIONS.contains(&word.as_str()))
    {
        return Err(usage(format!(
            "models command leaves {flag} to each instance: --listen is added per instance and \
             CUDA_VISIBLE_DEVICES selects the devices"
        )));
    }
    let mut words = vec![
        "serve".to_owned(),
        "--model".to_owned(),
        model.to_owned(),
        "--tensor-parallel-size".to_owned(),
        ranks.to_string(),
    ];
    if profile == picked.name {
        words.extend_from_slice(explicit);
    } else {
        let entry = crate::profiles::direct_entry(picked)?;
        let file = file().map_err(failure)?;
        let target = file.get(profile)?;
        if target.tensor_parallel != ranks as usize {
            return Err(usage(format!(
                "profile {profile} serves {} ranks, not {ranks}",
                target.tensor_parallel
            )));
        }
        crate::profiles::translatable(&entry, target)
            .map_err(|reason| failure(format!("profile {profile} is unavailable: {reason}")))?;
        // Preparation options stay with `serve --model`; the others are
        // serve options over the profile.
        let served = crate::profiles::apply_defaults(
            &crate::profiles::direct_defaults(target),
            &crate::profiles::serving_options(explicit),
        );
        let translation = crate::profiles::translate_options(&entry, &served)
            .map_err(|reason| failure(format!("profile {profile} is unavailable: {reason}")))?;
        words.extend(crate::profiles::preparation_options(explicit));
        words.extend(translation.arguments);
    }
    // The checks `serve --model` makes before it prepares anything: its own
    // parse, then the parse of its package's serve with the profile.
    crate::parse_host_command_v1(&words)?;
    let resolution = crate::profiles::direct_arguments(
        picked,
        ranks,
        &crate::profiles::serving_options(&words[1..]),
        Path::new("/direct-package"),
    )?;
    crate::parse_host_command_v1(&resolution.arguments)?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family's profiles at four ranks: autoregressive, a layer drafter,
    /// and two balanced block drafter profiles with a TBT budget and model
    /// options; and one at two ranks.
    fn family() -> Value {
        let profile = |speculation: Value, extra: Value| {
            let mut entry = json!({
                "bundle_id": "fixture.v1", "topology": "fixture-tp4", "tensor_parallel": 4,
                "devices_required": 4, "devices": [0, 1, 2, 3], "speculation": speculation,
                "max_sequences": 32, "qualified_max_sequences": 64,
                "qualification": "CPU fixture only", "kv_cache_dtype": "auto",
                "prefix_cache": true, "max_batch_tokens": 8192, "prefill_chunk_tokens": 2048,
                "queue_depth": null, "gpu_memory_utilization": 1.0,
                "graph_memory_reserve_mib": 320, "status": "qualified",
                "evidence": ["fixture"],
            });
            for (field, value) in extra.as_object().unwrap() {
                entry[field] = value.clone();
            }
            entry
        };
        let none = json!({"mode": "fixed", "tokens": 0, "draft_cost": "per-token",
                          "drafter": "none"});
        let block = |depths: &str, widths: &str| {
            json!({"mode": "balanced", "tokens": 5, "draft_cost": "per-block",
                   "drafter": "block", "depths": depths, "widths": widths, "rows": 64,
                   "graph_rows": 64})
        };
        let tbt = json!({"prefill_budget": "tbt", "decode_tbt_target_ms": 750,
                         "model_options": {"chunk": "8192", "draft_items": "12"},
                         "qualified_max_sequences": null, "status": "experimental"});
        json!({
            "schema": "chopinfer-serve-profiles-v2", "model": "fixture",
            "served_model_name": "fixture",
            "profiles": {
                "tp4-ar": profile(none.clone(), json!({})),
                "tp4-layer": profile(
                    json!({"mode": "adaptive", "tokens": 3, "draft_cost": "per-token",
                           "drafter": "layer"}),
                    json!({"graph_memory_reserve_mib": 384}),
                ),
                "tp4-block": profile(block("1:5,8:3,16:0", "6,8,16,32"), tbt.clone()),
                "tp4-block-wide": profile(block("1:5,32:0", "8,64"), tbt),
                "tp2-ar": profile(none, json!({"topology": "fixture-tp2", "tensor_parallel": 2,
                    "devices_required": 2, "devices": [0, 1]})),
            }
        })
    }

    fn profiles() -> Profiles {
        Profiles::parse(family().to_string().as_bytes()).unwrap()
    }

    fn picked(name: &str) -> runtime_bundle::preparation::DirectProfile {
        runtime_bundle::preparation::DirectProfile {
            name: name.to_owned(),
            profile: family()["profiles"][name].clone(),
        }
    }

    fn command(pick: &str, profile: &str, explicit: &[&str]) -> Result<Vec<String>, String> {
        let explicit: Vec<String> = explicit.iter().map(|word| (*word).to_owned()).collect();
        command_arguments(
            &picked(pick),
            &|| Ok(profiles()),
            ("/checkpoint", 4),
            profile,
            &explicit,
        )
        .map_err(|e| e.to_string())
    }

    fn count(words: &[String], flag: &str) -> usize {
        words.iter().filter(|word| *word == flag).count()
    }

    /// Each profile at the pick's rank count is the pick, serves over it, or
    /// says why it cannot; a refused rank count refuses all its profiles.
    #[test]
    fn a_pick_lists_each_profile_at_its_rank_count() {
        let size = pick_size(Ok(picked("tp4-layer")), &profiles(), 4);
        assert_eq!(size["tensor_parallel"], 4);
        assert_eq!(size["pick"], "tp4-layer");
        assert!(size["refusal"].is_null());
        let entries = size["profiles"].as_array().unwrap();
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["tp4-ar", "tp4-block", "tp4-block-wide", "tp4-layer"]
        );
        let entry = |name: &str| entries.iter().find(|e| e["name"] == name).unwrap();
        assert_eq!(
            entry("tp4-layer"),
            &json!({"name": "tp4-layer", "available": true, "arguments": [],
                    "inherited_model_options": []})
        );
        assert_eq!(entry("tp4-ar")["available"], true);
        assert!(
            entry("tp4-ar")["arguments"]
                .as_array()
                .unwrap()
                .contains(&json!("--speculative-tokens"))
        );
        assert_eq!(entry("tp4-block")["available"], false);
        assert!(
            entry("tp4-block")["reason"]
                .as_str()
                .unwrap()
                .contains("drafts with block")
        );
        let size = pick_size(Ok(picked("tp4-block")), &profiles(), 4);
        let entries = size["profiles"].as_array().unwrap();
        let ar = entries.iter().find(|e| e["name"] == "tp4-ar").unwrap();
        assert_eq!(
            ar["inherited_model_options"],
            json!(["chunk", "draft_items"])
        );
        let refused = pick_size(Err("needs a drafter".into()), &profiles(), 2);
        assert!(refused["pick"].is_null());
        assert_eq!(refused["refusal"], "needs a drafter");
        assert_eq!(
            refused["profiles"],
            json!([{"name": "tp2-ar", "available": false,
                    "reason": "serve --model refuses TP2 for this checkpoint: needs a drafter"}])
        );
    }

    /// An explicit option replaces the profile's flag of the same name, so
    /// the command carries it once.
    #[test]
    fn an_explicit_option_replaces_the_profiles_flag() {
        let words = command("tp4-layer", "tp4-ar", &["--max-sequences", "64"]).unwrap();
        assert_eq!(
            words[..5],
            [
                "serve",
                "--model",
                "/checkpoint",
                "--tensor-parallel-size",
                "4"
            ]
        );
        assert_eq!(count(&words, "--max-sequences"), 1);
        let at = words.iter().position(|w| w == "--max-sequences").unwrap();
        assert_eq!(words[at + 1], "64");
        assert_eq!(count(&words, "--devices"), 0);
        // The pick takes the options alone.
        let words = command("tp4-layer", "tp4-layer", &["--max-sequences", "64"]).unwrap();
        assert_eq!(words[5..], ["--max-sequences", "64"]);
        // Preparation options stay with serve --model.
        let words = command(
            "tp4-layer",
            "tp4-ar",
            &["--model-cache-dir", "/cache", "--load-threads", "2"],
        )
        .unwrap();
        assert!(
            words
                .windows(2)
                .any(|pair| pair == ["--model-cache-dir", "/cache"])
        );
        assert_eq!(count(&words, "--load-threads"), 1);
    }

    /// Zero speculative tokens drop a balanced profile's depth schedule and
    /// widths, over the pick and as the pick.
    #[test]
    fn zero_speculative_tokens_drop_the_balanced_schedule() {
        let words = command(
            "tp4-block",
            "tp4-block-wide",
            &["--speculative-tokens", "0"],
        )
        .unwrap();
        for absent in [
            "--speculation-depths",
            "--speculation-widths",
            "--speculative-rows",
            "--speculation-graph-rows",
        ] {
            assert_eq!(count(&words, absent), 0, "{absent}: {words:?}");
        }
        let words = command("tp4-block", "tp4-block", &["--speculative-tokens", "0"]).unwrap();
        assert_eq!(words[5..], ["--speculative-tokens", "0"]);
        let resolution = crate::profiles::direct_arguments(
            &picked("tp4-block"),
            4,
            &crate::profiles::serving_options(&words[1..]),
            Path::new("/direct-package"),
        )
        .unwrap();
        for absent in ["--speculation-depths", "--speculation-widths"] {
            assert_eq!(count(&resolution.arguments, absent), 0, "{absent}");
        }
        // Without the option the other balanced profile keeps its own.
        let words = command("tp4-block", "tp4-block-wide", &[]).unwrap();
        assert!(
            words
                .windows(2)
                .any(|pair| pair == ["--speculation-depths", "1:5,32:0"])
        );
    }

    /// A duplicate or invalid option, a per-instance flag, or a profile the
    /// pick cannot serve is refused before anything starts.
    #[test]
    fn a_command_that_serve_would_refuse_is_refused() {
        for (pick, profile, explicit, expected) in [
            (
                "tp4-layer",
                "tp4-ar",
                &["--max-sequences", "4", "--max-sequences", "8"][..],
                "--max-sequences is given more than once",
            ),
            (
                "tp4-layer",
                "tp4-layer",
                &["--max-sequences", "4", "--max-sequences", "8"],
                "--max-sequences is given more than once",
            ),
            (
                "tp4-layer",
                "tp4-ar",
                &["--prefix-cache", "maybe"],
                "--prefix-cache must be on or off",
            ),
            (
                "tp4-layer",
                "tp4-layer",
                &["--listen", "127.0.0.1:8311"],
                "leaves --listen",
            ),
            (
                "tp4-layer",
                "tp4-ar",
                &["--devices", "0,1,2,3"],
                "leaves --devices",
            ),
            ("tp4-layer", "tp4-block", &[], "drafts with block"),
            ("tp4-layer", "tp2-ar", &[], "serves 2 ranks"),
            ("tp4-layer", "tp4-missing", &[], "unknown profile"),
        ] {
            let error = command(pick, profile, explicit).unwrap_err();
            assert!(error.contains(expected), "{profile} {explicit:?}: {error}");
        }
    }

    /// `models --json` lists a profile with the settings `serve --model`
    /// resolves from it and its model options as option values.
    #[test]
    fn a_catalog_profile_carries_its_effective_settings() {
        let file = profiles();
        let entry = profile_entry("tp4-block", file.get("tp4-block").unwrap()).unwrap();
        let mut keys: Vec<&str> = entry
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "drafter",
                "max_sequences",
                "model_options",
                "name",
                "qualification",
                "qualified_max_sequences",
                "settings",
                "speculative_tokens",
                "status",
                "tensor_parallel"
            ]
        );
        assert_eq!(entry["status"], "experimental");
        assert_eq!(entry["drafter"], "block");
        assert_eq!(entry["speculative_tokens"], 5);
        assert!(entry["qualified_max_sequences"].is_null());
        assert_eq!(
            entry["model_options"],
            json!({"chunk": "8192", "draft_items": "12"})
        );
        let settings = &entry["settings"];
        assert_eq!(settings["prefill-budget"], "tbt");
        assert_eq!(settings["decode-tbt-target-ms"], "750");
        assert_eq!(settings["speculation"], "balanced");
        assert_eq!(settings["speculation-depths"], "1:5,8:3,16:0");
        assert_eq!(settings["speculative-rows"], "64");
        assert_eq!(settings["max-sequences"], "32");
        assert!(settings.get("devices").is_none());
        let ar = profile_entry("tp4-ar", file.get("tp4-ar").unwrap()).unwrap();
        assert_eq!(ar["status"], "qualified");
        assert_eq!(ar["qualified_max_sequences"], 64);
        assert_eq!(ar["settings"]["speculative-tokens"], "0");
        assert!(ar["settings"].get("speculation").is_none());
    }

    /// An install beside a host binary with one fake helper,
    /// `chopinfer-model-duo`, whose family serves the `family()` profiles
    /// `names` and accepts the checkpoint only at two ranks; it names the
    /// pick at each count it is asked about.
    fn duo(label: &str, names: &[&str]) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("models-pick-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(bin.join("chopinfer"), b"").unwrap();
        let mut document = family();
        document["profiles"]
            .as_object_mut()
            .unwrap()
            .retain(|name, _| names.contains(&name.as_str()));
        let selected = |name: &str| {
            json!({"result": "direct-profile-selected", "name": name,
                   "profile": family()["profiles"][name]})
        };
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = serving-profiles ]; then printf '%s' '{document}'; exit 0; fi\n\
             read -r request\ncase \"$request\" in\n\
             *'\"describe\"'*) echo '{{\"result\":\"description\",\"version\":2,\
             \"provider\":\"chopinfer-model-duo\",\"profiles\":[],\
             \"capabilities\":[\"prepare-direct\",\"direct-profile\"]}}' ;;\n\
             *'\"probe\"'*) case \"$request\" in\n\
             *'\"tensor_parallel_size\":2'*) echo '{{\"result\":\"probe\",\"compatible\":true,\
             \"reason\":\"compatible\"}}' ;;\n\
             *) echo '{{\"result\":\"probe\",\"compatible\":false,\
             \"reason\":\"duo needs two ranks\"}}' ;;\nesac ;;\n\
             *'\"tensor_parallel_size\":2'*) echo '{}' ;;\n\
             *'\"direct-profile\"'*) echo '{}' ;;\nesac\n",
            selected("tp2-ar"),
            selected("tp4-ar"),
        );
        let helper = bin.join("chopinfer-model-duo");
        std::fs::write(&helper, script).unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let checkpoint = root.join("checkpoint");
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::write(
            checkpoint.join("config.json"),
            r#"{"architectures": ["DuoForCausalLM"]}"#,
        )
        .unwrap();
        root
    }

    fn pick(root: &Path) -> Value {
        let document =
            pick_document(&root.join("checkpoint"), &root.join("bin/chopinfer")).unwrap();
        serde_json::from_str(&document).unwrap()
    }

    /// A family that serves a checkpoint only at a count other than the
    /// default is found at that count; a count every helper refused keeps
    /// serve's refusal though the helper would name a profile there.
    #[test]
    fn a_pick_finds_a_family_at_the_rank_counts_it_serves() {
        let root = duo("two", &["tp2-ar"]);
        let document = pick(&root);
        assert_eq!(document["model"], "duo", "{document}");
        assert!(document["refusal"].is_null());
        let sizes = document["sizes"].as_array().unwrap();
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0]["tensor_parallel"], 2);
        assert_eq!(sizes[0]["pick"], "tp2-ar");
        std::fs::remove_dir_all(root).unwrap();

        let root = duo("both", &["tp2-ar", "tp4-ar"]);
        let document = pick(&root);
        assert_eq!(document["model"], "duo", "{document}");
        let sizes = document["sizes"].as_array().unwrap();
        assert_eq!(sizes[0]["pick"], "tp2-ar");
        assert_eq!(sizes[1]["tensor_parallel"], 4);
        assert!(sizes[1]["pick"].is_null());
        let refusal = sizes[1]["refusal"].as_str().unwrap();
        assert!(
            refusal.contains("--tensor-parallel-size 4") && refusal.contains("duo needs two ranks"),
            "{refusal}"
        );
        assert_eq!(sizes[1]["profiles"][0]["available"], false);
        std::fs::remove_dir_all(root).unwrap();

        // Refused at every count: the refusal at the default count.
        let root = duo("none", &["tp4-ar"]);
        let document = pick(&root);
        assert!(document["model"].is_null());
        let refusal = document["refusal"].as_str().unwrap();
        assert!(
            refusal.contains("--tensor-parallel-size 4") && refusal.contains("duo needs two ranks"),
            "{refusal}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
