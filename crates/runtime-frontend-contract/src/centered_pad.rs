//! Validation shared by descriptor loading and host media preparation.
use crate::{FrontendMediaLayoutV1, FrontendMediaModalityV1, FrontendMediaRecipeV1};

/// Largest image or canvas the host accepts, bounded to 256 MiB of RGB.
pub const MAXIMUM_DECODED_IMAGE_PIXELS_V1: u64 = 89_478_485;

fn integer(value: &serde_json::Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        let number = value.as_f64()?;
        (number.is_finite() && number >= 0.0 && number < u64::MAX as f64 && number.fract() == 0.0)
            .then_some(number as u64)
    })
}

impl FrontendMediaRecipeV1 {
    /// Checks the parameters consumed by `image.centered-pad.v1`, including
    /// conversion to the f32 normalization used during preparation.
    pub fn validate_centered_pad_v1(&self) -> Result<(), String> {
        let parameters = &self.parameters;
        let known = [
            "alpha_background",
            "patch",
            "downsample",
            "minimum_pixels",
            "maximum_tokens",
            "pad",
            "value_divisor",
            "mean",
            "std",
            "maximum_aspect_ratio",
        ];
        let object = parameters
            .as_object()
            .ok_or("invalid centered pad recipe")?;
        if let Some(key) = object.keys().find(|key| !known.contains(&key.as_str())) {
            return Err(format!("{} has no parameter {key}", self.tag));
        }
        if !matches!(
            parameters["alpha_background"].as_str(),
            Some("white" | "none")
        ) {
            return Err(format!(
                "{} requires alpha_background \"white\" (composite) or \"none\" (discard)",
                self.tag
            ));
        }
        if let Some(ratio) = parameters.get("maximum_aspect_ratio")
            && !ratio.as_f64().is_some_and(|r| r.is_finite() && r >= 1.0)
        {
            return Err(format!(
                "{} requires maximum_aspect_ratio to be a finite number of at least 1",
                self.tag
            ));
        }
        let at = |key: &str| integer(&parameters[key]).ok_or("invalid centered pad recipe");
        let patch = at("patch")?;
        let downsample = at("downsample")?;
        let minimum = at("minimum_pixels")?;
        let maximum = at("maximum_tokens")?;
        if patch == 0
            || downsample == 0
            || maximum < 4
            || minimum > 1 << 53
            || patch
                .checked_mul(downsample)
                .and_then(|cell| cell.checked_mul(maximum))
                .is_none_or(|side| side > u64::from(u32::MAX))
        {
            return Err("invalid centered pad recipe".into());
        }
        if !parameters["pad"].as_array().is_some_and(|pad| {
            pad.len() == 3 && pad.iter().all(|v| v.as_u64().is_some_and(|v| v <= 255))
        }) {
            return Err("invalid centered pad color".into());
        }
        let finite =
            |value: &serde_json::Value| value.as_f64().map(|v| v as f32).filter(|v| v.is_finite());
        let vector = |key: &str, nonzero: bool| {
            parameters[key].as_array().is_some_and(|v| {
                v.len() == 3
                    && v.iter()
                        .all(|v| finite(v).is_some_and(|v| !nonzero || v != 0.0))
            })
        };
        if !finite(&parameters["value_divisor"]).is_some_and(|v| v > 0.0)
            || !vector("mean", false)
            || !vector("std", true)
        {
            return Err("invalid image normalization".into());
        }
        Ok(())
    }
}

impl FrontendMediaModalityV1 {
    /// The delimited span must fit the modality and every possible canvas
    /// must fit the host's decoded pixel limit.
    pub fn validate_centered_pad_v1(&self) -> Result<(), String> {
        self.recipe.validate_centered_pad_v1()?;
        if self.layout != Some(FrontendMediaLayoutV1::LearnedDelimitersV1) {
            return Err("the token budget counts learned delimiter rows: the modality must declare the learned-delimiters.v1 layout".into());
        }
        let parameters = &self.recipe.parameters;
        let maximum = integer(&parameters["maximum_tokens"]).expect("validated integer");
        if maximum > u64::from(self.maximum_tokens) {
            return Err(format!(
                "maximum_tokens {maximum} exceeds the modality's {} tokens",
                self.maximum_tokens
            ));
        }
        let cell = u128::from(integer(&parameters["patch"]).expect("validated integer"))
            * u128::from(integer(&parameters["downsample"]).expect("validated integer"));
        let canvas = u128::from(maximum - 3) * cell * cell;
        if canvas > u128::from(MAXIMUM_DECODED_IMAGE_PIXELS_V1) {
            return Err(format!(
                "a canvas of up to {canvas} pixels exceeds the host's {MAXIMUM_DECODED_IMAGE_PIXELS_V1} decoded pixels"
            ));
        }
        Ok(())
    }
}
