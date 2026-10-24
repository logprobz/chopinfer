//! Fingerprint the same compacted byte ranges that rank uploads consume.

use super::*;
use crate::weight_keeper::CheckpointV1;
use crate::weight_keeper::content::{Identity, Image, Manifest, SCHEMA};

fn error(status: BundleStatusV1) -> String {
    format!(
        "weight content layout: {}",
        status.name().unwrap_or("invalid status")
    )
}

pub(super) fn layout_identity(
    image: &crate::direct::PlannedImageV1,
    description: &CompiledImageDescriptionV1,
    rank: u32,
) -> Result<String, String> {
    let layout = description.rank_layout(rank).map_err(error)?;
    let mut components = Vec::new();
    for component in image.components() {
        let mut bytes = component.bytes.clone();
        if let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && let Some(identity) = value.get("direct_identity")
        {
            // Rank indexes are fresh inputs to each borrower, not lent
            // memory. Normalize only their authenticated producer binding;
            // retain every layout, dtype, shape, offset and length field.
            let component_rank = value["rank"]
                .as_u64()
                .and_then(|rank| usize::try_from(rank).ok())
                .ok_or("content layout component has no rank")?;
            if identity.as_str()
                != Some(
                    image
                        .identity(component_rank)
                        .ok_or("content layout component rank is absent")?,
                )
                || value["image_bytes"].as_u64() != Some(image.arena_bytes(component_rank))
                || value.get("sha256").is_some()
            {
                return Err("content layout component has another producer binding".to_owned());
            }
            value
                .as_object_mut()
                .ok_or("content layout component is not an object")?
                .remove("direct_identity");
            bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
        }
        components.push(serde_json::json!({"name":component.name,"sha256":runtime_weights::sha256_bytes(&bytes)}));
    }
    let views: Vec<_> =
        description
            .views
            .iter()
            .filter_map(|(name, view)| {
                view.spans.get(&rank).map(|span| serde_json::json!({
            "name":name,"alignment":view.alignment,"host":view.host,"mapped":view.mapped,
            "offset":span.offset,"byte_length":span.byte_length,
        }))
            })
            .collect();
    let document = serde_json::json!({"schema":"chopinfer.weight-layout.v1",
        "image_role":image.image_role(),"rank":rank,"ranks":description.rank_count(),
        "alignment":description.alignment,"image_bytes":layout.image_bytes,
        "host_cuts":layout.cuts,"mapped_cuts":layout.mapped,"views":views,"components":components});
    Ok(runtime_weights::sha256_bytes(
        &serde_json::to_vec(&document).map_err(|error| error.to_string())?,
    ))
}

fn checkpoint(image: &crate::direct::PlannedImageV1) -> Result<CheckpointV1, String> {
    image.checkpoint().verify_unchanged()?;
    Ok(CheckpointV1 {
        identity: image.checkpoint().identity_hex(),
        stamps: image.checkpoint().stamps()?,
    })
}

pub(super) fn expected(
    manifest: &Manifest,
    image: &crate::direct::PlannedImageV1,
    description: &CompiledImageDescriptionV1,
    rank: u32,
) -> Result<Identity, String> {
    manifest.expected(
        &checkpoint(image)?,
        image.image_role(),
        rank,
        image
            .identity(rank as usize)
            .ok_or("content proof rank is absent")?,
        description.rank_layout(rank).map_err(error)?.device_bytes(),
        &layout_identity(image, description, rank)?,
    )
}

fn hash_rank(
    storage: &ComponentStorageV1,
    description: &CompiledImageDescriptionV1,
    rank: u32,
) -> Result<String, String> {
    let rank_image = description.rank_image(rank).map_err(error)?;
    let layout = description.rank_layout(rank).map_err(error)?;
    let segments: Vec<_> = layout
        .device_segments()
        .into_iter()
        .map(|(offset, length)| (rank_image.offset + offset, length))
        .collect();
    let length = layout.device_bytes();
    let mut hasher = runtime_weights::Sha256State::new();
    let mut buffer =
        vec![0; usize::try_from(length.min(64 << 20)).map_err(|error| error.to_string())?];
    let mut offset = 0;
    while offset < length {
        let count = usize::try_from((length - offset).min(buffer.len() as u64))
            .map_err(|error| error.to_string())?;
        if count == 0 || !read_segments_v1(storage, &segments, offset, &mut buffer[..count]) {
            return Err("cannot read complete produced weight content".to_owned());
        }
        hasher
            .update(&buffer[..count])
            .map_err(|error| error.to_string())?;
        offset += count as u64;
    }
    Ok(hasher.finalize())
}

impl HostInstanceConfigV1 {
    pub(crate) fn with_weight_content(mut self, manifest: Arc<Manifest>) -> Result<Self, String> {
        manifest.validate()?;
        let mut found = std::collections::BTreeSet::new();
        for record in self.components.values() {
            let (ComponentStorageV1::Planned(image), Some(description)) =
                (&record.storage, &record.description)
            else {
                continue;
            };
            for rank in 0..description.rank_count() as u32 {
                expected(&manifest, image, description, rank)?;
                found.insert((image.image_role().to_owned(), rank));
            }
        }
        let declared = manifest
            .images
            .iter()
            .map(|image| (image.image_role.clone(), image.rank))
            .collect();
        if found != declared {
            return Err("content manifest image coverage differs from the fresh plan".to_owned());
        }
        self.weight_content = Some(manifest);
        Ok(self)
    }

    /// CPU-only planning output. No device backend, context or GPU allocation is
    /// constructed. Full hashes are prepared before a queued serving job.
    pub(crate) fn weight_identity(&self, content: bool) -> Result<serde_json::Value, String> {
        let mut bound = None;
        let mut images = Vec::new();
        for record in self.components.values() {
            let (ComponentStorageV1::Planned(image), Some(description)) =
                (&record.storage, &record.description)
            else {
                continue;
            };
            let before = checkpoint(image)?;
            if bound.as_ref().is_some_and(|prior| prior != &before) {
                return Err("planned images name different checkpoints".to_owned());
            }
            bound = Some(before.clone());
            for rank in 0..description.rank_count() as u32 {
                let layout_sha256 = layout_identity(image, description, rank)?;
                let recipe_identity = image
                    .identity(rank as usize)
                    .ok_or("planned image rank missing")?
                    .to_owned();
                let byte_length = description.rank_layout(rank).map_err(error)?.device_bytes();
                images.push(Image {
                    image_role: image.image_role().to_owned(),
                    rank,
                    recipe_identity,
                    byte_length,
                    identity: Identity {
                        content_sha256: if content {
                            hash_rank(&record.storage, description, rank)?
                        } else {
                            String::new()
                        },
                        layout_sha256,
                    },
                });
            }
            if checkpoint(image)? != before {
                return Err("checkpoint changed while fingerprinting weights".to_owned());
            }
        }
        let checkpoint = bound.ok_or("the package plans no checkpoint images")?;
        if content {
            let manifest = Manifest {
                schema: SCHEMA.to_owned(),
                producer_sha256: crate::weight_keeper::content::producer_sha256()?,
                checkpoint,
                images,
            };
            manifest.validate()?;
            serde_json::to_value(manifest).map_err(|error| error.to_string())
        } else {
            Ok(
                serde_json::json!({"schema":"chopinfer.weight-recipes.v1","checkpoint":checkpoint,"images":images,
                "content_verified":false}),
            )
        }
    }
}
