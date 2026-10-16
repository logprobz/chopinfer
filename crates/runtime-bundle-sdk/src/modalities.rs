//! The modalities a package serves by default. `--modalities auto` is the
//! host's default and every serving profile's: the host prepares a direct
//! package with the modalities it selects (runtime-bundle
//! `PrepareDirectRequest::served_modalities`), and the package then names
//! the encoders of those modalities in every create record. A bundle's CPU
//! conformance tests create each profile's instance with the same rule.

/// The modalities `auto` serves, of those a checkpoint provides.
pub const AUTO: [&str; 2] = ["text", "image"];

/// The modalities `auto` serves of those a checkpoint `provides`, in their
/// order.
pub fn auto_served<'a>(provides: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    provides
        .into_iter()
        .filter(|modality| AUTO.contains(modality))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_serves_text_and_image_of_what_a_checkpoint_provides() {
        assert_eq!(auto_served(["text"]), ["text"]);
        assert_eq!(auto_served(["text", "image"]), ["text", "image"]);
        assert_eq!(
            auto_served(["text", "image", "video", "audio"]),
            ["text", "image"]
        );
    }
}
