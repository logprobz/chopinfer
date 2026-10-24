//! Content proofs for sharing produced rank bytes across implementation builds.
//!
//! A producer recipe remains the authority for a CPU proof. A keeper separately
//! hashes its actual device allocation. Only equal bytes and equal physical
//! layouts permit the content protocol to cross a recipe identity boundary.

use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use super::CheckpointV1;

pub(crate) const SCHEMA: &str = "chopinfer.weight-content.v1";
pub(crate) const PROTOCOL: &str = "chopinfer.weight-keeper.content.v2";
const LIMIT: u64 = 1 << 20;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Identity {
    pub(crate) content_sha256: String,
    pub(crate) layout_sha256: String,
}

impl Identity {
    pub(crate) fn valid(&self) -> bool {
        [&self.content_sha256, &self.layout_sha256]
            .iter()
            .all(|value| digest(value))
    }
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Image {
    pub(crate) image_role: String,
    pub(crate) rank: u32,
    pub(crate) recipe_identity: String,
    pub(crate) byte_length: u64,
    pub(crate) identity: Identity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) schema: String,
    pub(crate) producer_sha256: String,
    pub(crate) checkpoint: CheckpointV1,
    pub(crate) images: Vec<Image>,
}

impl Manifest {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let mut keys = std::collections::BTreeSet::new();
        if self.schema != SCHEMA
            || !digest(&self.producer_sha256)
            || !digest(&self.checkpoint.identity)
            || self.checkpoint.stamps.is_empty()
            || self.images.is_empty()
            || self.images.iter().any(|image| {
                image.image_role.is_empty()
                    || image.byte_length == 0
                    || !digest(&image.recipe_identity)
                    || !image.identity.valid()
                    || !keys.insert((&image.image_role, image.rank))
            })
        {
            return Err("invalid weight content manifest".to_owned());
        }
        Ok(())
    }

    pub(crate) fn expected(
        &self,
        checkpoint: &CheckpointV1,
        role: &str,
        rank: u32,
        recipe: &str,
        bytes: u64,
        layout: &str,
    ) -> Result<Identity, String> {
        self.validate()?;
        if self.producer_sha256 != producer_sha256()? {
            return Err("weight content proof belongs to another executable".to_owned());
        }
        if &self.checkpoint != checkpoint {
            return Err("weight content proof names another checkpoint state".to_owned());
        }
        let image = self
            .images
            .iter()
            .find(|image| image.image_role == role && image.rank == rank)
            .ok_or("weight content proof is missing a rank image")?;
        if image.recipe_identity != recipe
            || image.byte_length != bytes
            || image.identity.layout_sha256 != layout
        {
            return Err(
                "weight content proof does not match the fresh recipe and layout".to_owned(),
            );
        }
        Ok(image.identity.clone())
    }
}

/// Bind a CPU proof to the exact executable that produced its bytes, not only
/// source text. Different compiler settings must not reuse an unchecked proof.
pub(crate) fn producer_sha256() -> Result<String, String> {
    static DIGEST: OnceLock<Result<String, String>> = OnceLock::new();
    DIGEST
        .get_or_init(|| {
            let path = std::env::current_exe().map_err(|error| error.to_string())?;
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)
                .map_err(|error| error.to_string())?;
            let before = file.metadata().map_err(|error| error.to_string())?;
            if !before.is_file() {
                return Err("the executing host is not a regular file".to_owned());
            }
            let digest =
                runtime_weights::sha256_reader(&mut file).map_err(|error| error.to_string())?;
            let after = file.metadata().map_err(|error| error.to_string())?;
            if (
                before.dev(),
                before.ino(),
                before.len(),
                before.mtime(),
                before.mtime_nsec(),
                before.ctime(),
                before.ctime_nsec(),
            ) != (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            ) {
                return Err("executing host changed while fingerprinting it".to_owned());
            }
            Ok(digest)
        })
        .clone()
}

pub(crate) fn read(path: &Path) -> Result<Manifest, String> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| format!("weight content manifest: {error}"))?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file()
        || metadata.len() > LIMIT
        || metadata.uid() != super::fds::effective_uid()
        || metadata.mode() & 0o022 != 0
    {
        return Err("weight content manifest must be a bounded regular file owned by this user and not writable by others".to_owned());
    }
    let mut bytes = Vec::new();
    file.take(LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > LIMIT {
        return Err("weight content manifest grew beyond its bound".to_owned());
    }
    let value: Manifest = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    value.validate()?;
    Ok(value)
}

static SELECTED: OnceLock<Arc<Manifest>> = OnceLock::new();
static IDENTIFY: OnceLock<bool> = OnceLock::new();

pub(crate) fn select(path: &Path) -> Result<(), String> {
    let manifest = Arc::new(read(path)?);
    if let Some(prior) = SELECTED.get() {
        return if prior == &manifest {
            Ok(())
        } else {
            Err("another weight content manifest is already selected".to_owned())
        };
    }
    SELECTED
        .set(manifest)
        .map_err(|_| "weight content manifest was selected concurrently".to_owned())
}

pub(crate) fn selected() -> Option<Arc<Manifest>> {
    SELECTED.get().cloned()
}

pub(crate) fn select_identify(content: bool) -> Result<(), String> {
    IDENTIFY
        .set(content)
        .map_err(|_| "weight identity mode is already selected".to_owned())
}

pub(crate) fn identifying() -> Option<bool> {
    IDENTIFY.get().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    fn manifest() -> Manifest {
        Manifest {
            schema: SCHEMA.to_owned(),
            producer_sha256: producer_sha256().unwrap(),
            checkpoint: CheckpointV1 {
                identity: "a".repeat(64),
                stamps: vec!["fixed".to_owned()],
            },
            images: vec![Image {
                image_role: "fixture".to_owned(),
                rank: 0,
                recipe_identity: "b".repeat(64),
                byte_length: 64,
                identity: Identity {
                    content_sha256: "c".repeat(64),
                    layout_sha256: "d".repeat(64),
                },
            }],
        }
    }

    #[test]
    fn proof_requires_original_recipe_fresh_layout_and_checkpoint_stamps() {
        let proof = manifest();
        assert!(
            proof
                .expected(
                    &proof.checkpoint,
                    "fixture",
                    0,
                    &"b".repeat(64),
                    64,
                    &"d".repeat(64)
                )
                .is_ok()
        );
        for (recipe, layout, length) in [
            ("e".repeat(64), "d".repeat(64), 64),
            ("b".repeat(64), "e".repeat(64), 64),
            ("b".repeat(64), "d".repeat(64), 65),
        ] {
            assert!(
                proof
                    .expected(&proof.checkpoint, "fixture", 0, &recipe, length, &layout)
                    .is_err()
            );
        }
        let mut other_producer = proof.clone();
        other_producer.producer_sha256 = "e".repeat(64);
        assert!(
            other_producer
                .expected(
                    &proof.checkpoint,
                    "fixture",
                    0,
                    &"b".repeat(64),
                    64,
                    &"d".repeat(64)
                )
                .is_err()
        );
        let mut changed = proof.checkpoint.clone();
        changed.stamps.push("changed".to_owned());
        assert!(
            proof
                .expected(&changed, "fixture", 0, &"b".repeat(64), 64, &"d".repeat(64))
                .is_err()
        );
        let mut duplicate = proof.clone();
        duplicate.images.push(duplicate.images[0].clone());
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn proof_file_refuses_symlinks_and_other_writers() {
        let directory =
            std::env::temp_dir().join(format!("keeper-content-proof-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let file = directory.join("proof.json");
        std::fs::write(&file, serde_json::to_vec(&manifest()).unwrap()).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read(&file).is_ok());
        let link = directory.join("link");
        symlink(&file, &link).unwrap();
        assert!(read(&link).is_err());
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(read(&file).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
