//! The checkpoint's declared chat frontend.

use model_mimo26_contract::encoders::Encoders;
use runtime_frontend_contract::FrontendDescriptorV1;

pub const COMPONENT: &[u8] = include_bytes!("frontend.json");

pub fn component_for_flash(
    identity: &crate::checkpoint::Identity,
    encoders: Encoders,
) -> crate::MimoResult<Vec<u8>> {
    crate::checkpoint::registry().component_for_flash(identity, encoders)
}

fn validate_vocabulary(
    size: u32,
    geometry: &model_mimo26_contract::geometry::Geometry,
) -> crate::MimoResult<()> {
    if size != geometry.vocab() {
        return Err(format!(
            "frontend vocabulary size {size} differs from geometry {} vocabulary {}",
            geometry.id,
            geometry.vocab()
        )
        .into());
    }
    Ok(())
}

fn merge(target: &mut serde_json::Value, patch: &serde_json::Value) {
    if let (Some(target), Some(patch)) = (target.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            merge(target.entry(key).or_insert(serde_json::Value::Null), value);
        }
    } else {
        *target = patch.clone();
    }
}

impl crate::checkpoint::Registry {
    pub fn component_for_flash(
        &self,
        identity: &crate::checkpoint::Identity,
        encoders: Encoders,
    ) -> crate::MimoResult<Vec<u8>> {
        self.component_for_geometry(
            identity,
            encoders,
            &model_mimo26_contract::geometry::V26_FLASH,
        )
    }

    pub fn component_for_geometry(
        &self,
        identity: &crate::checkpoint::Identity,
        encoders: Encoders,
        geometry: &'static model_mimo26_contract::geometry::Geometry,
    ) -> crate::MimoResult<Vec<u8>> {
        validate_audio_geometry(geometry, encoders)?;
        let identity = self.identity(self.name(identity)?)?;
        let template = std::str::from_utf8(COMPONENT).expect("frontend template UTF-8");
        let component: serde_json::Value =
            serde_json::from_slice(COMPONENT).expect("frontend template");
        let escaped = |value: &str| {
            let json = serde_json::to_string(value).expect("JSON string");
            json[1..json.len() - 1].to_owned()
        };
        let mut bytes = template
            .replace(
                component["tokenizer"]["repository"].as_str().unwrap(),
                &escaped(&identity.repository),
            )
            .replace(
                component["models"][0]["id"].as_str().unwrap(),
                &escaped(&identity.served_model_name),
            )
            .replace(
                component["tokenizer"]["revision"].as_str().unwrap(),
                &escaped(&identity.revision),
            )
            .into_bytes();
        let lock = self.source_lock(identity)?;
        let mut selected = component["tokenizer"].clone();
        selected["repository"] = identity.repository.clone().into();
        selected["revision"] = identity.revision.clone().into();
        for (field, name) in [
            ("definition", "tokenizer.json"),
            ("configuration", "tokenizer_config.json"),
        ] {
            let pin = lock.pin(name).ok_or("tokenizer source pin absent")?;
            selected[field]["byte_length"] = pin.byte_length.into();
            selected[field]["sha256"] = pin.sha256.clone().into();
        }
        let mut current: serde_json::Value = serde_json::from_slice(&bytes)?;
        let original = current.clone();
        if let Some(patch) = &identity.frontend {
            merge(&mut current, patch);
        }
        apply_geometry(&mut current, geometry)?;
        // Repository, revision and file pins always come from the registered lock.
        for field in ["repository", "revision", "definition", "configuration"] {
            current["tokenizer"][field] = selected[field].clone();
        }
        if current != original {
            bytes = serde_json::to_vec(&current)?;
        }
        let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes)?;
        validate_vocabulary(descriptor.vocabulary.size, geometry)?;
        if encoders == Encoders::ALL {
            return Ok(bytes);
        }
        let mut component: serde_json::Value = serde_json::from_slice(&bytes).expect("frontend");
        // The package serves exactly the selected encoders' kinds, and the
        // chat format labels the kinds it serves.
        if encoders.any() {
            component["multimodal"]["modalities"]
                .as_array_mut()
                .unwrap()
                .retain(|modality| {
                    encoders
                        .names()
                        .contains(&modality["kind"].as_str().unwrap())
                });
        } else {
            component.as_object_mut().unwrap().remove("multimodal");
        }
        Ok(serde_json::to_vec(&component)?)
    }
}

pub(crate) fn validate_audio_geometry(
    g: &model_mimo26_contract::geometry::Geometry,
    encoders: Encoders,
) -> crate::MimoResult<()> {
    if !encoders.audio {
        return Ok(());
    }
    let fixed = runtime_frontend_contract::audio_log_mel_v1_fixed_parameters();
    for (key, declared) in [
        ("sample_rate", u64::from(g.audio_tokenizer().sampling_rate)),
        (
            "maximum_seconds",
            model_mimo26_contract::encoders::maximum_audio_seconds(g)?,
        ),
        ("segment_frames", u64::from(g.audio().audio_segment_size)),
    ] {
        let expected = fixed[key].as_u64().ok_or("invalid v1 audio contract")?;
        if declared != expected {
            return Err(format!(
                "audio.log-mel.v1 requires {key}={expected}, geometry {} declares {declared}",
                g.id
            )
            .into());
        }
    }
    Ok(())
}

fn apply_geometry(
    component: &mut serde_json::Value,
    g: &model_mimo26_contract::geometry::Geometry,
) -> crate::MimoResult<()> {
    if g.id == model_mimo26_contract::geometry::V26_FLASH.id {
        return Ok(());
    }
    component["context"]["maximum_total_tokens"] = g.context_tokens().into();
    component["limits"]["maximum_generated_tokens"] = g.context_tokens().into();
    if component.get("multimodal").is_none() {
        return Ok(());
    }
    let v = g.vision();
    let a = g.audio();
    let t = g.audio_tokenizer();
    let columns = v.in_chans * v.temporal_patch_size * v.patch_size.pow(2);
    component["multimodal"]["arena"]["hidden"] = g.hidden().into();
    component["multimodal"]["encoders"][0]["payload_columns"] = columns.into();
    component["multimodal"]["encoders"][1]["payload_columns"] = t.n_mels.into();
    let limits = model_mimo26_contract::encoders::payload_limits(g)?;
    let maximum_seconds = model_mimo26_contract::encoders::maximum_audio_seconds(g)?;
    let frames = t.sampling_rate as u64 * maximum_seconds / t.hop_length as u64 + 1;
    let [vision_recipes, _] = model_mimo26_contract::encoders::spec::recipes_for(g)?;
    component["multimodal"]["encoders"][1]["maximum_payload_bytes"] = limits[1].into();
    for modality in component["multimodal"]["modalities"]
        .as_array_mut()
        .ok_or("frontend needs modalities")?
    {
        let kind = modality["kind"]
            .as_str()
            .ok_or("frontend needs modality kind")?
            .to_owned();
        let parameters = &mut modality["recipe"]["parameters"];
        if kind == "audio" {
            for (key, value) in [
                ("sample_rate", t.sampling_rate),
                ("maximum_seconds", maximum_seconds as u32),
                ("n_fft", t.nfft),
                ("window", t.window_size),
                ("hop", t.hop_length),
                ("mel_bins", t.n_mels),
                ("f_min", t.fmin),
                ("first_kernel", t.kernel_size),
                ("second_kernel", t.kernel_size),
                ("second_stride", t.stride_size),
                ("pool", t.avg_pooler),
                ("group", a.group_size),
                ("segment_frames", a.audio_segment_size),
            ] {
                parameters[key] = value.into();
            }
            parameters["f_max"] = t.fmax.unwrap_or(t.sampling_rate / 2).into();
            modality["maximum_tokens"] = frames
                .div_ceil(t.stride_size as u64 * t.avg_pooler as u64 * a.group_size as u64)
                .into();
        } else {
            let (image, pixels, temporal) = if kind == "video" {
                let model_mimo26_contract::encoders::spec::Recipe::VideoTimestampedGrids(recipe) =
                    &vision_recipes[1]
                else {
                    return Err("expected video recipe".into());
                };
                parameters["maximum_pixels_per_frame"] = recipe.maximum_pixels_per_frame.into();
                parameters["maximum_frames"] = recipe.maximum_frames.into();
                let pixels = parameters["maximum_pixels_per_frame"]
                    .as_u64()
                    .ok_or("video pixel limit")?;
                let count = parameters["maximum_frames"]
                    .as_u64()
                    .ok_or("video frame limit")?;
                (&mut parameters["image"], pixels, count)
            } else {
                let model_mimo26_contract::encoders::spec::Recipe::ImagePatchGrid(recipe) =
                    &vision_recipes[0]
                else {
                    return Err("expected image recipe".into());
                };
                parameters["maximum_pixels"] = recipe.maximum_pixels.into();
                let pixels = parameters["maximum_pixels"]
                    .as_u64()
                    .ok_or("image pixel limit")?;
                (parameters, pixels, 1)
            };
            image["patch"] = v.patch_size.into();
            image["temporal_patch"] = v.temporal_patch_size.into();
            image["merge"] = v.spatial_merge_size.into();
            image["factor"] = (v.patch_size * v.spatial_merge_size).into();
            let rows = pixels / v.patch_size.pow(2) as u64 * temporal;
            modality["maximum_tokens"] = (rows / v.spatial_merge_size.pow(2) as u64).into();
        }
    }
    component["multimodal"]["encoders"][0]["maximum_payload_bytes"] = limits[0].into();
    Ok(())
}

pub fn descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(
        &component_for_flash(crate::checkpoint::default_identity(), Encoders::ALL)
            .expect("registered frontend"),
    )
    .expect("the pinned frontend declaration is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_limits_and_payloads_follow_the_carried_geometry() {
        let mut geometry = model_mimo26_contract::geometry::V26_FLASH;
        geometry.id = "frontend-test";
        geometry.limits.context = 32768;
        geometry.shape.vision.patch_size = 32;
        geometry.shape.audio_tokenizer.n_mels = 64;
        let geometry = Box::leak(Box::new(geometry));
        let bytes = crate::checkpoint::registry()
            .component_for_geometry(
                crate::checkpoint::default_identity(),
                Encoders::ALL,
                geometry,
            )
            .unwrap();
        let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
        assert_eq!(descriptor.vocabulary.size, geometry.vocab());
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["context"]["maximum_total_tokens"], 32768);
        assert_eq!(value["limits"]["maximum_generated_tokens"], 32768);
        assert_eq!(value["multimodal"]["arena"]["hidden"], geometry.hidden());
        assert_eq!(value["multimodal"]["encoders"][0]["payload_columns"], 6144);
        assert_eq!(value["multimodal"]["encoders"][1]["payload_columns"], 64);
        assert_eq!(value["multimodal"]["modalities"][0]["maximum_tokens"], 256);
        assert_eq!(
            value["multimodal"]["modalities"][0]["recipe"]["parameters"]["factor"],
            64
        );
    }

    #[test]
    fn payload_policy_preserves_flash_and_scales_recipe_geometry() {
        let flash: serde_json::Value = serde_json::from_str(include_str!("frontend.json")).unwrap();
        for (temporal, hop, expected) in [
            (2, 240, [100_663_296, 30_720_512]),
            (1, 240, [100_663_296, 30_720_512]),
            (3, 240, [103_809_024, 30_720_512]),
        ] {
            let mut geometry = model_mimo26_contract::geometry::V26_FLASH;
            geometry.id = "payload-policy-test";
            geometry.shape.vision.temporal_patch_size = temporal;
            geometry.shape.audio_tokenizer.hop_length = hop;
            let mut component = flash.clone();
            apply_geometry(&mut component, &geometry).unwrap();
            assert_eq!(
                model_mimo26_contract::encoders::payload_limits(&geometry).unwrap(),
                expected
            );
            for (encoder, limit) in component["multimodal"]["encoders"]
                .as_array()
                .unwrap()
                .iter()
                .zip(expected)
            {
                assert_eq!(encoder["maximum_payload_bytes"], limit);
            }
            assert_eq!(
                component["multimodal"]["modalities"][1]["recipe"]["parameters"]["maximum_seconds"],
                600
            );
        }
        assert_eq!(
            model_mimo26_contract::encoders::payload_limits(
                &model_mimo26_contract::geometry::V26_FLASH
            )
            .unwrap(),
            [100_663_296, 30_720_512]
        );
    }

    #[test]
    fn frontend_vocabulary_must_equal_geometry() {
        let geometry = &model_mimo26_contract::geometry::V26_FLASH;
        validate_vocabulary(geometry.vocab(), geometry).unwrap();
        assert_eq!(
            validate_vocabulary(geometry.vocab() + 1, geometry)
                .unwrap_err()
                .to_string(),
            "frontend vocabulary size 152577 differs from geometry v26-flash vocabulary 152576"
        );
    }

    #[test]
    fn each_served_set_keeps_encoder_indices_and_filters_modalities() {
        for set in Encoders::SETS {
            let bytes = component_for_flash(crate::checkpoint::default_identity(), set).unwrap();
            let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
            assert_eq!(descriptor.multimodal.is_some(), set.any());
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            if let Some(media) = descriptor.multimodal {
                assert_eq!(media.encoders.len(), 2);
                let mut kinds = value["multimodal"]["modalities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| m["kind"].as_str().unwrap())
                    .collect::<Vec<_>>();
                kinds.sort();
                let mut expected = set.names();
                expected.sort();
                assert_eq!(kinds, expected);
            }
        }
    }

    #[test]
    fn default_frontend_preserves_template_bytes_and_uses_lock_pins() {
        assert_eq!(
            component_for_flash(crate::checkpoint::default_identity(), Encoders::ALL).unwrap(),
            COMPONENT
        );
        let template: serde_json::Value = serde_json::from_slice(COMPONENT).unwrap();
        for (_, identity) in crate::checkpoint::identities() {
            let expected = std::str::from_utf8(COMPONENT)
                .unwrap()
                .replace(
                    template["tokenizer"]["repository"].as_str().unwrap(),
                    &identity.repository,
                )
                .replace(
                    template["tokenizer"]["revision"].as_str().unwrap(),
                    &identity.revision,
                )
                .replace(
                    template["models"][0]["id"].as_str().unwrap(),
                    &identity.served_model_name,
                );
            assert_eq!(
                component_for_flash(identity, Encoders::ALL).unwrap(),
                expected.as_bytes()
            );
            let descriptor = FrontendDescriptorV1::from_component_bytes(
                &component_for_flash(identity, Encoders::ALL).unwrap(),
            )
            .unwrap();
            assert_eq!(
                runtime_frontend_contract::tokenizer_contract_v1(&descriptor).unwrap(),
                crate::tokenizer::tokenizer_contract_for_flash(identity).unwrap()
            );
        }
    }

    #[test]
    fn unregistered_frontend_returns_without_panicking() {
        let mut identity = crate::checkpoint::default_identity().clone();
        identity.revision = "unlisted".into();
        let result = std::panic::catch_unwind(|| component_for_flash(&identity, Encoders::ALL));
        assert!(
            result.is_ok(),
            "unregistered frontend must return an error without panicking"
        );
        assert!(result.unwrap().is_err());
    }

    /// Every encoder selection builds a component MiMo's export opens, and
    /// the format then renders exactly the media kinds the package serves.
    #[test]
    fn every_served_set_opens_under_the_chat_format_export() {
        use runtime_server::{
            FrontendChatRequestV1, FrontendContentPartV1, FrontendMediaPartKindV1,
            FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1,
            ServerFrontendV1,
        };
        let export = frontend_format::chat_format_export_v1::<
            model_mimo26_contract::frontend::Mimo26Format,
        >();
        let request = |kind, part_type: &str| FrontendChatRequestV1 {
            messages: vec![FrontendMessageV1 {
                role: FrontendRoleV1::User,
                content: Some("Describe it.".into()),
                parts: Some(vec![
                    FrontendContentPartV1::Media(FrontendMediaPartV1 {
                        kind,
                        part_type: part_type.into(),
                        source: "data:,".into(),
                        format: None,
                    }),
                    FrontendContentPartV1::Text("Describe it.".into()),
                ]),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tools: Vec::new(),
            }],
            continue_final_message: false,
            add_generation_prompt: true,
            tools: Vec::new(),
            tool_choice: FrontendToolChoiceV1::Auto,
            response_format: None,
            reasoning_effort: None,
            enable_thinking: None,
            thinking: None,
            nested_reasoning_effort: None,
            preserve_thinking: None,
            clear_thinking: None,
            add_vision_id: None,
            tool_objects: Vec::new(),
            chat_template_kwargs: None,
        };
        for set in Encoders::SETS {
            let bytes = component_for_flash(crate::checkpoint::default_identity(), set).unwrap();
            let frontend = runtime_frontend::BundleFrontendV1::from_export(&export, &bytes)
                .unwrap_or_else(|error| panic!("{:?}: {error}", set.names()));
            for (kind, name, part_type, label) in [
                (
                    FrontendMediaPartKindV1::Image,
                    "image",
                    "image_url",
                    "<|vision_start|>",
                ),
                (
                    FrontendMediaPartKindV1::Audio,
                    "audio",
                    "input_audio",
                    "<|mimo_audio_start|>",
                ),
                (
                    FrontendMediaPartKindV1::Video,
                    "video",
                    "video_url",
                    "<|vision_start|><|mimo_video_start|>",
                ),
            ] {
                let rendered = frontend.render_chat(request(kind, part_type));
                let label_text = format!("{:?} {name}", set.names());
                if set.names().contains(&name) {
                    let rendered = rendered.unwrap_or_else(|error| panic!("{label_text}: {error}"));
                    assert!(
                        rendered
                            .prompt
                            .starts_with(&format!("<|im_start|>user\n{label}")),
                        "{label_text}: {}",
                        rendered.prompt
                    );
                } else {
                    assert!(rendered.is_err(), "{label_text}");
                }
            }
        }
    }

    #[test]
    fn mimo_multimodal_descriptor_validates() {
        let descriptor = descriptor();
        let media = descriptor.multimodal.expect("MiMo media declaration");
        assert_eq!(media.arena.hidden, 4096);
        assert_eq!(media.encoders.len(), 2);
        assert_eq!(media.modalities.len(), 3);
    }
}

#[cfg(test)]
mod byte_probe {
    #[test]
    fn frontend_and_tokenizer_bytes() {
        for (name, identity) in crate::checkpoint::identities() {
            for set in model_mimo26_contract::encoders::Encoders::SETS {
                let bytes = super::component_for_flash(identity, set).unwrap();
                println!(
                    "{name} {:?} {}",
                    set.names(),
                    runtime_weights::sha256_bytes(&bytes)
                );
            }
            println!(
                "{name} contract {:?}",
                crate::tokenizer::tokenizer_contract_for_flash(identity).unwrap()
            );
        }
    }
}
