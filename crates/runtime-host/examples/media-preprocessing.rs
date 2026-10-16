//! Export served frontend preprocessing for comparison with a reference.
use runtime_host::{TOKEN_BUDGET_TAG_V1, audio_recipe_v1, image_recipe_v1, token_budget_recipe_v1};
use runtime_media::{DecodedAudio, MediaLimits, RgbImage, audio, recipe};
use serde_json::{Value, json};
use std::error::Error;
use std::path::Path;

fn rows(root: &Path, name: &str, values: &[f32]) -> Result<String, Box<dyn Error>> {
    let file = format!("{name}.f32");
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(root.join(&file), bytes)?;
    Ok(file)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: media-preprocessing FRONTEND REFERENCE_DIRECTORY OUTPUT_DIRECTORY".into(),
        );
    }
    let frontend: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let source = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    let reference: Value = serde_json::from_slice(&std::fs::read(source.join("reference.json"))?)?;
    let mut cases = Vec::new();
    for case in reference["cases"].as_array().ok_or("missing cases")? {
        let name = case["name"].as_str().ok_or("missing name")?;
        let kind = case["kind"].as_str().ok_or("missing kind")?;
        let media = frontend.get("multimodal").unwrap_or(&frontend);
        let modality = media["modalities"]
            .as_array()
            .ok_or("missing modalities")?
            .iter()
            .find(|item| item["kind"] == kind)
            .ok_or("unsupported kind")?;
        let limits = MediaLimits {
            maximum_bytes: 128 << 20,
            maximum_pixels: 32 << 20,
            maximum_audio_frames: 24_000 * 600,
            maximum_frames: 32,
            maximum_tokens: modality["maximum_tokens"]
                .as_u64()
                .ok_or("missing token limit")?,
            maximum_aspect_ratio: 200.0,
        };
        let file = case["source"]["file"].as_str().ok_or("missing source")?;
        if Path::new(file).file_name().and_then(|s| s.to_str()) != Some(file) {
            return Err("source must be a filename".into());
        }
        let bytes = std::fs::read(source.join(file))?;
        let parameters = &modality["recipe"]["parameters"];
        match kind {
            "image" => {
                let shape = &case["source"]["shape"];
                let image = RgbImage::new(
                    u32::try_from(shape[1].as_u64().ok_or("missing width")?)?,
                    u32::try_from(shape[0].as_u64().ok_or("missing height")?)?,
                    bytes,
                    limits,
                )?;
                // The integer stage is the same grid unscaled and unnormalized.
                let (grid, raw) = if modality["recipe"]["tag"] == TOKEN_BUDGET_TAG_V1 {
                    let recipe = token_budget_recipe_v1(parameters)
                        .map_err(|e| format!("image recipe: {e:?}"))?;
                    let unscaled = recipe::TokenBudgetRecipe {
                        value_scale: 1.0,
                        mean: [0.0; 3],
                        std: [1.0; 3],
                        ..recipe
                    };
                    (
                        recipe::padded_token_budget_grid(&image, recipe, limits)?,
                        recipe::padded_token_budget_grid(&image, unscaled, limits)?,
                    )
                } else {
                    let recipe =
                        image_recipe_v1(parameters).map_err(|e| format!("image recipe: {e:?}"))?;
                    let unscaled = recipe::ImageRecipe {
                        value_scale: 1.0,
                        mean: [0.0; 3],
                        std: [1.0; 3],
                        ..recipe
                    };
                    (
                        recipe::image_grid(&image, recipe, limits)?,
                        recipe::image_grid(&image, unscaled, limits)?,
                    )
                };
                let file = rows(output, name, &grid.rows)?;
                let integer = rows(output, &format!("{name}.integer"), &raw.rows)?;
                cases.push(json!({"name": name, "grid": grid.grid,
                    "stages": {"patches": {"file": file, "integer": integer,
                        "shape": [grid.rows.len() / grid.columns, grid.columns]}}}));
            }
            "audio" => {
                if bytes.len() % 4 != 0 {
                    return Err("invalid F32 samples".into());
                }
                let samples = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                    .collect::<Vec<_>>();
                let sample_rate =
                    u32::try_from(case["sample_rate"].as_u64().ok_or("missing sample rate")?)?;
                let recipe =
                    audio_recipe_v1(parameters).map_err(|e| format!("audio recipe: {e:?}"))?;
                let resampled = audio::resample(&samples, sample_rate, recipe, limits)?;
                let mel = audio::log_mel(
                    &DecodedAudio {
                        samples,
                        sample_rate,
                    },
                    recipe,
                    limits,
                )?;
                cases.push(json!({"name": name, "stages": {
                    "resampled": {"file": rows(output, &format!("{name}.resampled"), &resampled)?,
                        "shape": [resampled.len()]},
                    "mel": {"file": rows(output, &format!("{name}.mel"), &mel.values)?,
                        "shape": [mel.frames, mel.bins]}}}));
            }
            _ => return Err("unsupported input kind".into()),
        }
    }
    std::fs::write(
        output.join("candidate.json"),
        serde_json::to_vec_pretty(&json!({"cases": cases}))?,
    )?;
    Ok(())
}
