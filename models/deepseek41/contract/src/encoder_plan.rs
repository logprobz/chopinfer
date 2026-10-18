//! The direct weight plan of the replicated encoder image
//! (`chop.weight-plan.v1`, image role `encoder::views::IMAGE_ROLE`): each
//! view of `encoder::views::layout` built from the checkpoint's BF16 vision
//! and aligner tensors as the declaration maps them, at its offset, on every
//! rank alike, and the rank's encoder index naming the plan's identity.
//!
//! A tensor stores by `Copy`, the patch weight's 588 columns zero-padded to
//! the view's 640. The aligner's first weight takes its columns permuted
//! from the vendor's channel-major 3 x 3 unfold (`c * 9 + p`) to the
//! position-major order the space-to-depth merge writes (`p * 1024 + c`),
//! which the shared set cannot express: it runs as this bundle's own
//! transform (`transform`, exported as `chop_bundle_weight_transform_v1`).
//!
//! The tensors also validate the encoder's `external_tensors` for the text
//! plan's binding (`validated`).

use std::collections::{BTreeMap, BTreeSet};

use bundle_runtime::index::encoder::Identity;
use encoder_spec::Declaration;
use encoder_spec::weights::{Source, TensorInfo, Transform};
use kernel_core::weight_transform::{BUNDLE_TRANSFORM_FIRST, WeightTransform};
use model_compiler_sdk::weight_plan::{RequestFiles, plan_identity};
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use serde_json::Value;

use crate::encoder::views::{self, Stored};

/// The owner the projection declaration names for the encoder's tensors.
pub const OWNER: &str = "encoder-spec";

/// The columns of a `factor x factor` space-to-depth unfold permuted from
/// channel-major (`c * f * f + p`) to position-major (`p * channels + c`),
/// row by row. Parameters: rows, columns, factor and element bytes, each a
/// little-endian `u64`.
pub const SPACE_TO_DEPTH_COLUMNS: u32 = BUNDLE_TRANSFORM_FIRST;

const CONFIG: &str = "config.json";
const INDEX: &str = "model.safetensors.index.json";
/// The encoder image's arena alignment: what a stream-ordered CUDA pool
/// guarantees every allocation. Views stay at 4096-byte offsets inside the
/// image (`views::layout_for`).
pub const ALIGNMENT: u64 = 256;

fn words(params: &[u8]) -> Result<[u64; 4], String> {
    if params.len() != 32 {
        return Err(format!("{} parameter bytes, not 32", params.len()));
    }
    let word = |at: usize| u64::from_le_bytes(params[at * 8..at * 8 + 8].try_into().unwrap());
    Ok([word(0), word(1), word(2), word(3)])
}

fn encode(words: [u64; 4]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// This bundle's weight transform: the whole output of transform `id` of
/// the record's source, written from byte `offset` of it into `output`.
pub fn transform(
    id: u32,
    params: &[u8],
    sources: &[&[u8]],
    offset: u64,
    output: &mut [u8],
) -> Result<(), String> {
    if id != SPACE_TO_DEPTH_COLUMNS {
        return Err(format!("no weight transform {id:#x}"));
    }
    let [rows, columns, factor, element] = words(params)?;
    let [source] = sources else {
        return Err(format!("{} sources, not one", sources.len()));
    };
    let row = columns
        .checked_mul(element)
        .filter(|&row| rows.checked_mul(row) == Some(source.len() as u64))
        .ok_or("the source is not rows x columns elements")?;
    let cell = factor
        .checked_mul(factor)
        .filter(|&cell| cell > 1 && columns % cell == 0)
        .ok_or_else(|| format!("{columns} columns have no {factor} x {factor} unfold"))?;
    let (row, element) = (row as usize, element as usize);
    let (cell, channels) = (cell as usize, (columns / cell) as usize);
    let mut whole = vec![0; source.len()];
    for (from, to) in source.chunks_exact(row).zip(whole.chunks_exact_mut(row)) {
        for channel in 0..channels {
            for position in 0..cell {
                let at = (channel * cell + position) * element;
                let into = (position * channels + channel) * element;
                to[into..into + element].copy_from_slice(&from[at..at + element]);
            }
        }
    }
    let start = usize::try_from(offset).map_err(|e| e.to_string())?;
    let part = whole
        .get(start..start + output.len())
        .ok_or("output past the transform's bytes")?;
    output.copy_from_slice(part);
    Ok(())
}

/// One checkpoint tensor as its safetensors header states it.
#[derive(Clone, Debug)]
struct Header {
    file: String,
    dtype: String,
    shape: Vec<u64>,
}

/// The header of every tensor in `names`: each from the file the weight
/// map assigns it, which must hold it; without a weight map, from the one
/// file that holds it.
fn headers(
    request: &WeightPlanRequestV1,
    names: &BTreeSet<String>,
) -> Result<BTreeMap<String, Header>, String> {
    let designated: Option<BTreeMap<String, String>> = match request.document(INDEX) {
        Some(bytes) => {
            let index: Value =
                serde_json::from_slice(bytes).map_err(|e| format!("{INDEX}: {e}"))?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| format!("{INDEX} has no weight_map"))?;
            let mut designated = BTreeMap::new();
            for name in names {
                if let Some(file) = map.get(name) {
                    let file = file
                        .as_str()
                        .ok_or_else(|| format!("{INDEX}: {name} names no file"))?;
                    designated.insert(name.clone(), file.to_owned());
                }
            }
            Some(designated)
        }
        None => None,
    };
    let mut headers: BTreeMap<String, Header> = BTreeMap::new();
    for file in &request.files {
        let header: Value = serde_json::from_slice(&file.header)
            .map_err(|e| format!("{} header: {e}", file.name))?;
        let Value::Object(header) = header else {
            return Err(format!("{}: header is not an object", file.name));
        };
        for (name, tensor) in header {
            if !names.contains(&name) {
                continue;
            }
            if let Some(designated) = &designated
                && designated.get(&name) != Some(&file.name)
            {
                continue;
            }
            let shape = tensor["shape"]
                .as_array()
                .and_then(|shape| shape.iter().map(Value::as_u64).collect())
                .ok_or_else(|| format!("{name}: no shape"))?;
            let header = Header {
                file: file.name.clone(),
                dtype: tensor["dtype"].as_str().unwrap_or_default().to_owned(),
                shape,
            };
            if let Some(earlier) = headers.insert(name.clone(), header) {
                return Err(format!(
                    "encoder tensor {name} is in both {} and {} and no weight map assigns it",
                    earlier.file, file.name
                ));
            }
        }
    }
    for (name, file) in designated.iter().flatten() {
        if !headers.contains_key(name) {
            return Err(format!(
                "encoder tensor {name} is not in {file}, the file the weight map assigns it"
            ));
        }
    }
    Ok(headers)
}

/// The checkpoint shape view `view` is stored from: its own, except that
/// zero-padded columns come from the payload columns the padding widens
/// (the patch projection's weight, the one padded tensor).
fn source_shape(view: &Stored, declaration: &Declaration) -> Result<Vec<u64>, String> {
    let Source::Tensor(tensor) = &view.source else {
        return Err(format!("{}: not a single tensor", view.name));
    };
    match tensor.transforms[..] {
        [] | [Transform::SpaceToDepthCols(_)] => Ok(view.shape.clone()),
        [Transform::PadCols(_)] => {
            let input_linear = view.stage.is_some_and(|stage| {
                declaration.stages[..stage]
                    .iter()
                    .all(|earlier| earlier.op.id() == "pad.zero-columns.v1")
            });
            if !input_linear {
                return Err(format!("{}: padded columns of no payload", view.name));
            }
            Ok(vec![view.shape[0], u64::from(declaration.input.columns)])
        }
        ref other => Err(format!("{}: no plan for transforms {other:?}", view.name)),
    }
}

/// Checks the checkpoint's encoder tensors against `declaration`: each one
/// its layout stores is present with exactly the shape it is stored from,
/// the declaration binds to them, and every one is BF16.
fn check(declaration: &Declaration, headers: &BTreeMap<String, Header>) -> Result<(), String> {
    for view in views::layout_for(declaration)? {
        let header = headers
            .get(&view.name)
            .ok_or_else(|| format!("the checkpoint has no encoder tensor {}", view.name))?;
        let shape = source_shape(&view, declaration)?;
        if header.shape != shape {
            return Err(format!(
                "encoder tensor {} has shape {:?}, the declaration reads {shape:?}",
                view.name, header.shape
            ));
        }
        if header.dtype != "BF16" {
            return Err(format!(
                "encoder tensor {} is {}, not BF16",
                view.name, header.dtype
            ));
        }
    }
    declaration
        .bind(&|name: &str| {
            headers.get(name).map(|header| TensorInfo {
                dtype: header.dtype.clone(),
                shape: header.shape.clone(),
                file: None,
            })
        })
        .map_err(|error| format!("the checkpoint's encoder: {}", error.0))
}

/// A checkpoint tensor's dtype and shape by name, `None` for one the
/// checkpoint does not have.
pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<(String, Vec<u64>)>;

/// The tensors the encoder validates as their owner ([`OWNER`]) for the
/// projection declaration's `external_tensors`: every one the declaration
/// reads, after they check against it as `lookup` states them. An error
/// when the checkpoint holds an encoder that differs from the declaration.
pub fn validated(
    lookup: Lookup,
) -> Result<BTreeSet<String>, String> {
    let declaration = &crate::encoder::declarations().map_err(|e| e.0)?[0];
    let names = declaration.tensors();
    let mut headers = BTreeMap::new();
    for name in &names {
        if let Some((dtype, shape)) = lookup(name) {
            headers.insert(
                name.clone(),
                Header {
                    file: String::new(),
                    dtype,
                    shape,
                },
            );
        }
    }
    check(declaration, &headers)?;
    Ok(names)
}

/// The tensors the encoder owns in the checkpoint `config` describes, as
/// `lookup` states them ([`validated`]), for a geometry whose vision config
/// the declaration pins. Another geometry's encoder is not served (a
/// synthetic geometry's smaller tower): its tensors are the checkpoint's
/// names under the encoder's prefixes, `names`, and no encoder image is
/// planned for it.
pub fn owned<'a>(
    config: &Value,
    names: impl IntoIterator<Item = &'a String>,
    lookup: Lookup,
) -> Result<BTreeSet<String>, String> {
    let declaration = &crate::encoder::declarations().map_err(|e| e.0)?[0];
    if declaration.check_config(CONFIG, config).is_ok() {
        return validated(lookup);
    }
    Ok(names
        .into_iter()
        .filter(|name| {
            ["vision.", "aligner.", "image_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
        .cloned()
        .collect())
}

/// The records of one view.
fn records(
    view: &Stored,
    headers: &BTreeMap<String, Header>,
    files: &RequestFiles,
) -> Result<Vec<WeightRecordV1>, String> {
    let Source::Tensor(tensor) = &view.source else {
        return Err(format!("{}: not a single tensor", view.name));
    };
    let header = headers
        .get(&view.name)
        .ok_or_else(|| format!("the checkpoint has no {}", view.name))?;
    let sources = vec![files.source(&view.name, &header.file)?];
    let (id, params) = match tensor.transforms[..] {
        [Transform::SpaceToDepthCols(factor)] => {
            let [rows, columns] = header.shape[..] else {
                return Err(format!("{}: not a matrix", view.name));
            };
            (
                SPACE_TO_DEPTH_COLUMNS,
                encode([rows, columns, u64::from(factor), 2]),
            )
        }
        // A copy, zero-padded along the columns to the view's shape.
        [] | [Transform::PadCols(_)] => {
            let copy = WeightTransform::Copy;
            (copy.id(), copy.encode_params().map_err(|e| e.to_string())?)
        }
        ref other => return Err(format!("{}: no plan for transforms {other:?}", view.name)),
    };
    Ok(vec![WeightRecordV1 {
        arena_offset: view.offset,
        byte_length: view.bytes(),
        transform: id,
        params,
        sources,
        axis: None,
        ranges: Vec::new(),
        stored_shape: view.shape.clone(),
    }])
}

/// The plan of the encoder image for the request's rank, after the
/// checkpoint's config is admitted and pins the declaration's vision
/// config.
pub fn plan(request: &WeightPlanRequestV1) -> Result<WeightPlanV1, String> {
    let config: Value = serde_json::from_slice(
        request
            .document(CONFIG)
            .ok_or_else(|| format!("the checkpoint has no {CONFIG}"))?,
    )
    .map_err(|e| format!("{CONFIG}: {e}"))?;
    let geometry = crate::admission::admit(&config)?;
    crate::admission::check_ranks(geometry, request.ranks)?;
    if request.rank >= request.ranks {
        return Err(format!(
            "rank {} is past the {} ranks",
            request.rank, request.ranks
        ));
    }
    let declaration = &crate::encoder::declarations().map_err(|e| e.0)?[0];
    declaration.check_config(CONFIG, &config).map_err(|e| e.0)?;
    let headers = headers(request, &declaration.tensors())?;
    check(declaration, &headers)?;
    let layout = views::layout_for(declaration)?;
    let files = RequestFiles::new(request);
    let mut records = Vec::new();
    for view in &layout {
        records.extend(self::records(view, &headers, &files)?);
    }
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: std::str::from_utf8(views::IMAGE_ROLE)
            .expect("ASCII role")
            .into(),
        alignment: ALIGNMENT,
        arena_bytes: views::end(&layout),
        views: Vec::new(),
        components: Vec::new(),
        records,
    };
    let identity = plan_identity(&plan, request);
    let index = views::index(request.rank, Identity::Direct(&identity))?;
    let bytes = serde_json::to_vec(&index).map_err(|e| e.to_string())?;
    // The bundle's own admission of the planned index.
    views::parse(&bytes, request.rank)?;
    plan.components = vec![NamedBytesV1 {
        name: views::index_role(request.rank),
        bytes,
    }];
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(params: [u64; 4], source: &[u8]) -> Result<Vec<u8>, String> {
        let mut whole = vec![0xee; source.len()];
        transform(
            SPACE_TO_DEPTH_COLUMNS,
            &encode(params),
            &[source],
            0,
            &mut whole,
        )?;
        // Any part equals the whole's bytes there.
        let mut part = vec![0; 5];
        transform(
            SPACE_TO_DEPTH_COLUMNS,
            &encode(params),
            &[source],
            3,
            &mut part,
        )?;
        assert_eq!(part, whole[3..8]);
        Ok(whole)
    }

    #[test]
    fn space_to_depth_columns_move_each_position_ahead_of_the_channels() {
        // One row of 2 channels x 2 x 2 positions, one-byte elements:
        // channel-major c * 4 + p to position-major p * 2 + c.
        let source = [0, 1, 2, 3, 10, 11, 12, 13];
        assert_eq!(
            run([1, 8, 2, 1], &source).unwrap(),
            [0, 10, 1, 11, 2, 12, 3, 13]
        );
        // Two-byte elements of 3 channels in 3 x 3 cells move whole, row by
        // row: position-major column p * 3 + c holds channel-major c * 9 + p.
        let source: Vec<u8> = (0..2 * 27 * 2).map(|x| x as u8).collect();
        let out = run([2, 27, 3, 2], &source).unwrap();
        let element = |row: usize, column: usize| &source[(row * 27 + column) * 2..][..2];
        for row in 0..2 {
            for channel in 0..3 {
                for position in 0..9 {
                    let at = (row * 27 + position * 3 + channel) * 2;
                    assert_eq!(&out[at..at + 2], element(row, channel * 9 + position));
                }
            }
        }
        assert!(run([1, 6, 2, 1], &source[..6]).is_err());
        assert!(run([1, 8, 1, 1], &source[..8]).is_err());
        assert!(run([2, 8, 2, 1], &source[..8]).is_err());
        assert!(transform(7, &encode([1, 8, 2, 1]), &[&source[..8]], 0, &mut [0; 8]).is_err());
    }

    /// The permuted aligner columns and the merge's gather agree: the
    /// weight column the merge's block `p`, channel `c` meets is the
    /// vendor's unfold column of that channel and position.
    #[test]
    fn the_permuted_columns_meet_the_merged_rows() {
        let (channels, factor) = (4_usize, 3_u64);
        let cell = (factor * factor) as usize;
        // One weight row whose value is its vendor column.
        let source: Vec<u8> = (0..channels * cell).map(|x| x as u8).collect();
        let out = run([1, source.len() as u64, factor, 1], &source).unwrap();
        // A 3 x 3 grid: the merge's one cell gathers patch rows ky * 3 + kx
        // into blocks p = ky * 3 + kx, as F.unfold orders its kernel
        // positions.
        let indices = encoder_spec::geometry::space_to_depth_indices([1, 3, 3], factor).unwrap();
        for (block, &patch) in indices.iter().enumerate() {
            assert_eq!(patch as usize, block);
            for channel in 0..channels {
                let merged = block * channels + channel;
                assert_eq!(out[merged] as usize, channel * cell + block);
            }
        }
    }

    #[test]
    fn the_layout_reads_every_tensor_at_its_checkpoint_shape() {
        let declaration = &crate::encoder::declarations().unwrap()[0];
        let lookup = |name: &str| -> Option<(String, Vec<u64>)> {
            let view = views::layout()
                .unwrap()
                .into_iter()
                .find(|v| v.name == name)?;
            Some(("BF16".into(), source_shape(&view, declaration).unwrap()))
        };
        let names = validated(&lookup).unwrap();
        assert_eq!(names, declaration.tensors());
        let patch = |shape: Vec<u64>, dtype: &'static str| {
            move |name: &str| {
                if name == "vision.patch_embed.proj.weight" {
                    return Some((dtype.to_owned(), shape.clone()));
                }
                lookup(name)
            }
        };
        assert!(validated(&patch(vec![1024, 588], "BF16")).is_ok());
        // Padded columns read exactly the payload's; a padded or other
        // dtype source is refused.
        for (shape, dtype) in [
            (vec![1024, 640], "BF16"),
            (vec![1024, 600], "BF16"),
            (vec![1024, 588], "F32"),
        ] {
            assert!(validated(&patch(shape, dtype)).is_err());
        }
        let missing = |name: &str| (name != "image_end").then(|| lookup(name)).flatten();
        assert!(validated(&missing).is_err());
    }
}
