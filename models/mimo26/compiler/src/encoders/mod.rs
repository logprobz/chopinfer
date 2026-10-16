//! Replicated MiMo vision and audio encoder weights, planned as their own
//! image beside the text image.

pub mod catalog;
pub mod config;
pub mod image;

/// Select the encoders a validated checkpoint catalog provides and the
/// direct request serves. MiMo catalogs contain both towers or neither.
pub fn served(
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    geometry: &model_mimo26_contract::geometry::Geometry,
    catalog: &catalog::EncoderCatalog,
) -> crate::MimoResult<model_mimo26_contract::encoders::Encoders> {
    use runtime_bundle::preparation::Modality;
    let served = served_modalities(request, geometry, catalog)?;
    Ok(model_mimo26_contract::encoders::Encoders::of(
        served.contains(&Modality::Image),
        served.contains(&Modality::Video),
        served.contains(&Modality::Audio),
    ))
}

/// The concrete modalities selected from the checkpoint's encoder catalog.
pub fn served_modalities(
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    geometry: &model_mimo26_contract::geometry::Geometry,
    catalog: &catalog::EncoderCatalog,
) -> crate::MimoResult<std::collections::BTreeSet<runtime_bundle::preparation::Modality>> {
    use runtime_bundle::preparation::Modality;
    let mut provides = std::collections::BTreeSet::from([Modality::Text]);
    if !catalog.tensors.is_empty() {
        provides.extend([Modality::Image, Modality::Video, Modality::Audio]);
    }
    if !geometry.encoders
        && matches!(
            request.modalities,
            runtime_bundle::preparation::ModalitySelection::Auto
        )
    {
        provides.retain(|modality| *modality == Modality::Text);
    }
    let served = request.served_modalities(&provides)?;
    if !geometry.encoders && served.iter().any(|modality| *modality != Modality::Text) {
        return Err(format!(
            "geometry {} has no encoder kernels; serve --modalities text",
            geometry.id
        )
        .into());
    }
    crate::frontend::validate_audio_geometry(
        geometry,
        model_mimo26_contract::encoders::Encoders::of(
            false,
            false,
            served.contains(&Modality::Audio),
        ),
    )?;
    Ok(served)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_mimo26_contract::encoders::Encoders;
    use runtime_bundle::preparation::{ModalitySelection, PrepareDirectRequest};

    fn full_catalog() -> catalog::EncoderCatalog {
        let mut full = catalog::EncoderCatalog::default();
        full.tensors.insert(
            "visual.weight".into(),
            crate::catalog::TensorRecord {
                name: "visual.weight".into(),
                dtype: runtime_weights::DType::Bf16,
                shape: vec![1],
                file: "encoder.safetensors".into(),
                file_start: 0,
                bytes: 2,
            },
        );
        full
    }

    fn request(modalities: ModalitySelection) -> PrepareDirectRequest {
        PrepareDirectRequest {
            path: "checkpoint".into(),
            cache_root: "cache".into(),
            tensor_parallel_size: 2,
            activation_policy: None,
            modalities,
        }
    }

    #[test]
    fn unsupported_audio_is_refused_at_modality_selection() {
        let mut g = model_mimo26_contract::geometry::V26_FLASH;
        g.id = "unsupported-audio";
        g.shape.audio_tokenizer.hop_length = 160;
        let error = served(&request(ModalitySelection::All), &g, &full_catalog())
            .unwrap_err()
            .to_string();
        assert!(error.contains("audio.log-mel.v1 requires maximum_seconds=600"));
        assert!(error.contains("declares 400"));
    }

    /// The default package plans the vision tower for images and serves
    /// neither video nor audio; `all` plans every encoder and serves all
    /// three.
    #[test]
    fn the_default_package_serves_images_and_all_serves_every_modality() {
        use runtime_server::ServerFrontendV1;
        let served_media = |modalities| {
            let encoders = served(
                &request(modalities),
                &model_mimo26_contract::geometry::V26_FLASH,
                &full_catalog(),
            )
            .unwrap();
            let component = crate::frontend::component_for_flash(
                crate::checkpoint::default_identity(),
                encoders,
            )
            .unwrap();
            let frontend = runtime_frontend::BundleFrontendV1::from_export(
                &frontend_format::chat_format_export_v1::<
                    model_mimo26_contract::frontend::Mimo26Format,
                >(),
                &component,
            )
            .unwrap();
            let mut kinds = frontend
                .served_media()
                .into_iter()
                .map(|kind| kind.name())
                .collect::<Vec<_>>();
            kinds.sort();
            (encoders, kinds)
        };
        assert_eq!(
            served_media(ModalitySelection::Auto),
            (Encoders::of(true, false, false), vec!["image"])
        );
        assert_eq!(
            served_media(ModalitySelection::All),
            (Encoders::ALL, vec!["audio", "image", "video"])
        );
    }

    #[test]
    fn geometries_without_encoders_refuse_media_before_preparation() {
        let mut geometry = model_mimo26_contract::geometry::V26_FLASH;
        geometry.encoders = false;
        assert_eq!(
            served(
                &request(ModalitySelection::Auto),
                &geometry,
                &full_catalog()
            )
            .unwrap(),
            Encoders::NONE
        );
        for modalities in [
            ModalitySelection::All,
            ModalitySelection::parse("text,image").unwrap(),
            ModalitySelection::parse("text,video").unwrap(),
            ModalitySelection::parse("text,audio").unwrap(),
        ] {
            let error = served(&request(modalities), &geometry, &full_catalog()).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("has no encoder kernels; serve --modalities text")
            );
        }
        assert_eq!(
            served(
                &request(ModalitySelection::parse("text").unwrap()),
                &geometry,
                &full_catalog()
            )
            .unwrap(),
            Encoders::NONE
        );
    }

    #[test]
    fn served_sets_require_the_checkpoint_to_provide_them() {
        let empty = catalog::EncoderCatalog::default();
        let full = full_catalog();
        let mut request = request(ModalitySelection::Auto);
        // Auto serves text and image; all serves every encoder.
        assert_eq!(
            served(
                &request,
                &model_mimo26_contract::geometry::V26_FLASH,
                &empty
            )
            .unwrap(),
            Encoders::NONE
        );
        assert_eq!(
            served(&request, &model_mimo26_contract::geometry::V26_FLASH, &full).unwrap(),
            Encoders::of(true, false, false)
        );
        request.modalities = ModalitySelection::All;
        assert_eq!(
            served(
                &request,
                &model_mimo26_contract::geometry::V26_FLASH,
                &empty
            )
            .unwrap(),
            Encoders::NONE
        );
        assert_eq!(
            served(&request, &model_mimo26_contract::geometry::V26_FLASH, &full).unwrap(),
            Encoders::ALL
        );
        for set in Encoders::SETS {
            let names = std::iter::once("text")
                .chain(set.names())
                .collect::<Vec<_>>()
                .join(",");
            request.modalities = ModalitySelection::parse(&names).unwrap();
            assert_eq!(
                served(&request, &model_mimo26_contract::geometry::V26_FLASH, &full).unwrap(),
                set
            );
            if set.any() {
                assert!(
                    served(
                        &request,
                        &model_mimo26_contract::geometry::V26_FLASH,
                        &empty
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("--modalities names")
                );
            } else {
                assert_eq!(
                    served(
                        &request,
                        &model_mimo26_contract::geometry::V26_FLASH,
                        &empty
                    )
                    .unwrap(),
                    set
                );
            }
        }
    }
}
