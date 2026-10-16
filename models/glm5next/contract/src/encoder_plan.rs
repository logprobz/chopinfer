//! The direct weight plan of the replicated encoder image
//! (`chop.weight-plan.v1`, image role `encoder::views::IMAGE_ROLE`): each
//! view of `encoder::views::layout` built from the checkpoint's BF16
//! vision tensors as the declaration's weight mapping states, at its
//! offset, on every rank alike, and the rank's encoder index naming the
//! plan's identity.
//!
//! A tensor read as it is stores by `Copy` (a flattened one keeps its
//! bytes); a concatenation is one `Copy` per part at adjacent offsets; the
//! zero bias of a GEMM the checkpoint gives none is `Inline` zeros. Two
//! layouts the shared set cannot express run as this bundle's own
//! transforms (`transform`, exported as `chop_bundle_weight_transform_v1`):
//! rows copied into wider zero-padded rows (the patch weight's 1176
//! columns into 1216), and the column permutation of a space-to-depth
//! unfold (the downsample's channel-major columns to position-major).

use std::collections::BTreeMap;

use encoder_spec::Declaration;
use encoder_spec::weights::{Source, TensorInfo, Transform};
use kernel_core::weight_transform::{BUNDLE_TRANSFORM_FIRST, WeightTransform};
use model_compiler_sdk::weight_plan::{RequestFiles, plan_identity};
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use serde_json::Value;

use bundle_runtime::index::encoder::Identity;

use crate::encoder::views::{self, Content, Stored};

/// Rows of `columns` elements copied into rows of `padded`, zeros after.
/// Parameters: rows, columns, padded and element bytes, each a `u64`.
pub const PADDED_ROWS: u32 = BUNDLE_TRANSFORM_FIRST;
/// The columns of a `factor x factor` space-to-depth unfold permuted from
/// channel-major (`c * f * f + p`) to position-major (`p * channels + c`),
/// row by row. Parameters: rows, columns, factor and element bytes.
pub const SPACE_TO_DEPTH_COLUMNS: u32 = BUNDLE_TRANSFORM_FIRST + 1;

const CHECKPOINT_INDEX: &str = "model.safetensors.index.json";
/// The encoder image's arena alignment: what a stream-ordered CUDA pool
/// guarantees every allocation (qwen38's and mimo26's images ask the same).
/// The arena follows the text rank image in each rank's pool, so a larger
/// one is met only by chance; the host refused 4096 on a GPU. Views stay at
/// 4096-byte offsets inside the image (`views::layout_for`).
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

/// This bundle's weight transforms: the whole output of transform `id` of
/// the record's sources, written from byte `offset` of it into `output`.
pub fn transform(
    id: u32,
    params: &[u8],
    sources: &[&[u8]],
    offset: u64,
    output: &mut [u8],
) -> Result<(), String> {
    let [rows, columns, third, element] = words(params)?;
    let [source] = sources else {
        return Err(format!("{} sources, not one", sources.len()));
    };
    let row = columns
        .checked_mul(element)
        .filter(|&row| rows.checked_mul(row) == Some(source.len() as u64))
        .ok_or("the source is not rows x columns elements")?;
    let (row, element) = (row as usize, element as usize);
    let whole = match id {
        PADDED_ROWS => {
            if third < columns {
                return Err(format!("{columns} columns do not pad to {third}"));
            }
            let padded = third as usize * element;
            let mut whole = vec![0; rows as usize * padded];
            for (from, to) in source.chunks_exact(row).zip(whole.chunks_exact_mut(padded)) {
                to[..row].copy_from_slice(from);
            }
            whole
        }
        SPACE_TO_DEPTH_COLUMNS => {
            let cell = third.checked_mul(third).filter(|&cell| cell > 1);
            let Some(cell) = cell.filter(|cell| columns % cell == 0) else {
                return Err(format!(
                    "{columns} columns have no {third} x {third} unfold"
                ));
            };
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
            whole
        }
        _ => return Err(format!("no weight transform {id:#x}")),
    };
    let start = usize::try_from(offset).map_err(|e| e.to_string())?;
    let part = whole
        .get(start..start + output.len())
        .ok_or("output past the transform's bytes")?;
    output.copy_from_slice(part);
    Ok(())
}

/// One checkpoint tensor as a safetensors header states it.
#[derive(Clone, Debug)]
struct Header {
    file: String,
    dtype: String,
    shape: Vec<u64>,
}

/// The header of every tensor under `prefix`: each from the file the
/// weight map assigns it, which must hold it, whatever another file also
/// holds; without a weight map, from the one file that holds it.
fn headers(
    request: &WeightPlanRequestV1,
    prefix: &str,
) -> Result<BTreeMap<String, Header>, String> {
    let designated: Option<BTreeMap<String, String>> = match request.document(CHECKPOINT_INDEX) {
        Some(bytes) => {
            let index = crate::admission::document(CHECKPOINT_INDEX, bytes, bytes.len())?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| format!("{CHECKPOINT_INDEX} has no weight_map"))?;
            let designated = map
                .iter()
                .filter(|(name, _)| name.starts_with(prefix))
                .map(|(name, file)| {
                    let file = file
                        .as_str()
                        .ok_or_else(|| format!("{CHECKPOINT_INDEX}: {name} names no file"))?;
                    Ok((name.clone(), file.to_owned()))
                })
                .collect::<Result<_, String>>()?;
            Some(designated)
        }
        None => None,
    };
    let mut headers: BTreeMap<String, Header> = BTreeMap::new();
    for file in &request.files {
        if designated
            .as_ref()
            .is_some_and(|designated| !designated.values().any(|name| *name == file.name))
        {
            continue;
        }
        let Value::Object(header) = crate::admission::document(
            &format!("{}: header", file.name),
            &file.header,
            file.header.len(),
        )?
        else {
            return Err(format!("{}: header is not an object", file.name));
        };
        for (name, tensor) in header {
            if !name.starts_with(prefix) {
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
                    "vision tensor {name} is in both {} and {} and no weight map assigns it",
                    earlier.file, file.name
                ));
            }
        }
    }
    for (name, file) in designated.iter().flatten() {
        if !headers.contains_key(name) {
            return Err(format!(
                "vision tensor {name} is not in {file}, the file the weight map assigns it"
            ));
        }
    }
    Ok(headers)
}

/// The checkpoint shape of a vision tensor by name, `None` for a tensor
/// the checkpoint does not have.
pub type Expected<'a> = &'a dyn Fn(&str) -> Option<Vec<u64>>;

/// Checks the checkpoint's vision tensors against `declaration`: every
/// tensor its mapping reads is present with exactly its own checkpoint
/// shape (`expected`) before any transform reshapes it, every role gets
/// exactly the shape and element it reads, and every one is BF16.
fn check(
    declaration: &Declaration,
    headers: &BTreeMap<String, Header>,
    expected: Expected,
) -> Result<(), String> {
    for name in declaration.tensors() {
        let header = headers
            .get(&name)
            .ok_or_else(|| format!("the checkpoint has no vision tensor {name}"))?;
        let shape = expected(&name)
            .ok_or_else(|| format!("vision tensor {name} is not in the checkpoint's catalog"))?;
        if header.shape != shape {
            return Err(format!(
                "vision tensor {name} has shape {:?}, the checkpoint's catalog {shape:?}",
                header.shape
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
        .map_err(|error| format!("the checkpoint's vision tower: {}", error.0))?;
    for name in declaration.tensors() {
        if headers[&name].dtype != "BF16" {
            return Err(format!(
                "vision tensor {name} is {}, not BF16",
                headers[&name].dtype
            ));
        }
    }
    Ok(())
}

/// The admitted checkpoint's catalog shape of every tensor.
fn catalog_shapes(request: &WeightPlanRequestV1) -> Result<BTreeMap<String, Vec<u64>>, String> {
    let matched = crate::admission::admit_bytes(
        request
            .document("config.json")
            .ok_or("the checkpoint has no config.json")?,
    )?;
    Ok(crate::catalog::catalog(matched.geometry)
        .into_iter()
        .map(|spec| (spec.name, spec.shape))
        .collect())
}

/// Checks the vision tower of the checkpoint `request` describes against
/// the admitted checkpoint's catalog and the vision declaration, as a
/// package that plans the encoder image requires.
pub fn check_tower(request: &WeightPlanRequestV1) -> Result<(), String> {
    let declaration = &crate::encoder::declarations().map_err(|e| e.0)?[0];
    let shapes = catalog_shapes(request)?;
    check(
        declaration,
        &headers(request, &declaration.checkpoint.prefix)?,
        &|name| shapes.get(name).cloned(),
    )
}

/// The records of one view.
fn records(
    view: &Stored,
    declaration: &Declaration,
    headers: &BTreeMap<String, Header>,
    files: &RequestFiles,
) -> Result<Vec<WeightRecordV1>, String> {
    let record = |offset, bytes, transform: u32, params, sources, stored_shape| WeightRecordV1 {
        arena_offset: offset,
        byte_length: bytes,
        transform,
        params,
        sources,
        axis: None,
        ranges: Vec::new(),
        stored_shape,
    };
    let source = match (&view.content, &view.source) {
        (Content::Zeros, _) => {
            let zeros = WeightTransform::Inline(vec![0; view.bytes() as usize]);
            return Ok(vec![record(
                view.offset,
                view.bytes(),
                zeros.id(),
                zeros.encode_params().map_err(|e| e.to_string())?,
                Vec::new(),
                view.shape.clone(),
            )]);
        }
        (Content::Role, Some(source)) => source,
        (Content::Role, None) => return Err(format!("{} has no source", view.name)),
    };
    let tensor = |name: &str| -> Result<(String, &Header), String> {
        let name = match view.block {
            Some(block) => name.replace("{i}", &block.to_string()),
            None => name.to_owned(),
        };
        let name = format!("{}{name}", declaration.checkpoint.prefix);
        let header = headers
            .get(&name)
            .ok_or_else(|| format!("the checkpoint has no {name}"))?;
        Ok((name, header))
    };
    let copy = WeightTransform::Copy;
    let (copy_id, copy_params) = (copy.id(), copy.encode_params().map_err(|e| e.to_string())?);
    let mut out = Vec::new();
    match source {
        Source::Tensor(part) => {
            let (name, header) = tensor(&part.tensor)?;
            let sources = vec![files.source(&name, &header.file)?];
            let rows = header.shape.first().copied().unwrap_or(1);
            let columns = header.shape.iter().skip(1).product::<u64>();
            let bundle = |id, third| {
                record(
                    view.offset,
                    view.bytes(),
                    id,
                    encode([rows, columns, third, 2]),
                    sources.clone(),
                    view.shape.clone(),
                )
            };
            out.push(match part.transforms[..] {
                [] | [Transform::Flatten] => record(
                    view.offset,
                    view.bytes(),
                    copy_id,
                    copy_params,
                    sources.clone(),
                    header.shape.clone(),
                ),
                [Transform::Flatten, Transform::PadCols(padded)] => bundle(PADDED_ROWS, padded),
                [Transform::Flatten, Transform::SpaceToDepthCols(factor)] => {
                    bundle(SPACE_TO_DEPTH_COLUMNS, u64::from(factor))
                }
                ref other => {
                    return Err(format!("{name}: no plan for transforms {other:?}"));
                }
            });
        }
        Source::Concat(concat) => {
            let mut offset = view.offset;
            for part in &concat.concat_rows {
                let Source::Tensor(part) = part else {
                    return Err(format!("{}: a nested concatenation", view.name));
                };
                if !part.transforms.is_empty() {
                    return Err(format!("{}: a transformed part", view.name));
                }
                let (name, header) = tensor(&part.tensor)?;
                let bytes = 2 * header.shape.iter().product::<u64>();
                out.push(record(
                    offset,
                    bytes,
                    copy_id,
                    copy_params.clone(),
                    vec![files.source(&name, &header.file)?],
                    header.shape.clone(),
                ));
                offset += bytes;
            }
        }
        Source::ConcatRange(_) | Source::Zeros(_) => {
            return Err(format!("{}: no plan for its source", view.name));
        }
    }
    let planned: u64 = out.iter().map(|record| record.byte_length).sum();
    if planned != view.bytes() {
        return Err(format!(
            "{}: the records store {planned} bytes, the view holds {}",
            view.name,
            view.bytes()
        ));
    }
    Ok(out)
}

/// The plan of rank `request.rank`'s encoder image of `declaration`, after
/// checking the checkpoint's vision tower against it and each tensor's
/// own shape against `expected`.
pub fn plan_for(
    request: &WeightPlanRequestV1,
    declaration: &Declaration,
    expected: Expected,
) -> Result<WeightPlanV1, String> {
    let headers = headers(request, &declaration.checkpoint.prefix)?;
    check(declaration, &headers, expected)?;
    let layout = views::layout_for(declaration)?;
    let files = RequestFiles::new(request);
    let mut records = Vec::new();
    for view in &layout {
        records.extend(self::records(view, declaration, &headers, &files)?);
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
    let index = views::index_for(request.rank, &layout, Identity::Direct(&identity));
    let bytes = serde_json::to_vec(&index).map_err(|e| e.to_string())?;
    // The bundle's own admission of the planned index.
    views::parse_for(&bytes, request.rank, &layout)?;
    plan.components = vec![NamedBytesV1 {
        name: views::index_role(request.rank),
        bytes,
    }];
    Ok(plan)
}

/// The plan of the encoder image for the request's rank, after the
/// checkpoint's config is admitted.
pub fn plan(request: &WeightPlanRequestV1) -> Result<WeightPlanV1, String> {
    let shapes = catalog_shapes(request)?;
    crate::admission::check_ranks(request.ranks)?;
    plan_for(
        request,
        &crate::encoder::declarations().map_err(|e| e.0)?[0],
        &|name| shapes.get(name).cloned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(id: u32, params: [u64; 4], source: &[u8]) -> Result<Vec<u8>, String> {
        let [rows, _, third, element] = params;
        let bytes = match id {
            PADDED_ROWS => rows * third * element,
            _ => source.len() as u64,
        };
        let mut whole = vec![0xee; bytes as usize];
        transform(id, &encode(params), &[source], 0, &mut whole)?;
        // Any part equals the whole's bytes there.
        let mut part = vec![0; 5];
        transform(id, &encode(params), &[source], 3, &mut part)?;
        assert_eq!(part, whole[3..8]);
        Ok(whole)
    }

    #[test]
    fn padded_rows_copy_each_row_and_zero_the_rest() {
        let source: Vec<u8> = (1..=12).collect();
        // 2 rows of 3 two-byte elements into rows of 5.
        let out = run(PADDED_ROWS, [2, 3, 5, 2], &source).unwrap();
        assert_eq!(
            out,
            [
                1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0
            ]
        );
        assert!(run(PADDED_ROWS, [2, 3, 2, 2], &source).is_err());
        assert!(run(PADDED_ROWS, [3, 3, 5, 2], &source).is_err());
    }

    #[test]
    fn space_to_depth_columns_move_each_position_ahead_of_the_channels() {
        // One row of 2 channels x 2 x 2 positions, one-byte elements:
        // channel-major c * 4 + p to position-major p * 2 + c.
        let source = [0, 1, 2, 3, 10, 11, 12, 13];
        let out = run(SPACE_TO_DEPTH_COLUMNS, [1, 8, 2, 1], &source).unwrap();
        assert_eq!(out, [0, 10, 1, 11, 2, 12, 3, 13]);
        // Two-byte elements move whole, row by row.
        let source: Vec<u8> = (0..32).collect();
        let out = run(SPACE_TO_DEPTH_COLUMNS, [2, 8, 2, 2], &source).unwrap();
        let element = |row: usize, column: usize| &source[(row * 8 + column) * 2..][..2];
        for row in 0..2 {
            for channel in 0..2 {
                for position in 0..4 {
                    let at = (row * 8 + position * 2 + channel) * 2;
                    assert_eq!(&out[at..at + 2], element(row, channel * 4 + position));
                }
            }
        }
        assert!(run(SPACE_TO_DEPTH_COLUMNS, [1, 6, 2, 1], &source[..6]).is_err());
        assert!(run(SPACE_TO_DEPTH_COLUMNS, [1, 8, 1, 1], &source[..8]).is_err());
        assert!(transform(7, &encode([1, 8, 2, 1]), &[&source[..8]], 0, &mut [0; 8]).is_err());
    }

    /// The vision tensors of the checkpoint, from the tensor families.
    fn vision_header() -> serde_json::Map<String, Value> {
        let families: Value =
            serde_json::from_str(include_str!("../fixtures/tensor-families.json")).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        let mut add = |name: String, family: &Value| {
            let shape: Vec<u64> = family["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_u64().unwrap())
                .collect();
            let bytes = 2 * shape.iter().product::<u64>();
            header.insert(
                name,
                serde_json::json!({
                    "dtype": family["dtype"], "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        };
        for family in families.as_array().unwrap() {
            let name = family["name"].as_str().unwrap();
            if !name.starts_with("model.visual.") {
                continue;
            }
            if name.contains("{}") {
                for index in 0..family["count"].as_u64().unwrap() {
                    add(name.replace("{}", &index.to_string()), family);
                }
            } else {
                add(name.to_owned(), family);
            }
        }
        header
    }

    /// The text plan's test request with the vision tensors in its file.
    fn request(rank: u32) -> WeightPlanRequestV1 {
        let mut request = crate::direct::test_support::request(rank, |header| {
            header.extend(vision_header());
        });
        request.image_role = std::str::from_utf8(views::IMAGE_ROLE).unwrap().into();
        request
    }

    #[test]
    fn the_plan_stores_every_view_whole_on_every_rank() {
        let layout = views::layout().unwrap();
        for rank in 0..4 {
            let plan = plan(&request(rank)).unwrap();
            assert_eq!(plan.arena_bytes, views::image_bytes().unwrap());
            // The records tile each view exactly, in layout order.
            let mut records = plan.records.iter().peekable();
            for view in &layout {
                let mut at = view.offset;
                while let Some(record) = records.next_if(|record| record.arena_offset == at) {
                    at += record.byte_length;
                    if at == view.offset + view.bytes() {
                        break;
                    }
                }
                assert_eq!(at, view.offset + view.bytes(), "{}", view.name);
            }
            assert!(records.next().is_none());
            let [index] = &plan.components[..] else {
                panic!("one component")
            };
            assert_eq!(index.name, views::index_role(rank));
            let parsed = views::parse(&index.bytes, rank).unwrap();
            assert!(parsed.sha256.is_none() && parsed.direct_identity.is_some());
            // Two views take this bundle's transforms; the zero biases are
            // inline.
            let bundle: Vec<u32> = plan
                .records
                .iter()
                .map(|record| record.transform)
                .filter(|&id| id >= BUNDLE_TRANSFORM_FIRST)
                .collect();
            assert_eq!(bundle, [PADDED_ROWS, SPACE_TO_DEPTH_COLUMNS]);
            let inline = plan
                .records
                .iter()
                .filter(|record| record.transform == WeightTransform::Inline(vec![]).id())
                .count();
            assert_eq!(inline, 3);
        }
    }

    #[test]
    fn a_tower_that_differs_from_the_declaration_is_refused() {
        let wrong = |edit: fn(&mut serde_json::Map<String, Value>)| {
            let mut request = crate::direct::test_support::request(0, |header| {
                header.extend(vision_header());
                edit(header);
            });
            request.image_role = std::str::from_utf8(views::IMAGE_ROLE).unwrap().into();
            (
                plan(&request).unwrap_err(),
                check_tower(&request).unwrap_err(),
            )
        };
        let (error, _) = wrong(|header| {
            header.remove("model.visual.blocks.5.attn.k_norm.weight");
        });
        assert!(error.contains("blocks.5.attn.k_norm.weight"), "{error}");
        let (error, _) = wrong(|header| {
            header["model.visual.downsample.weight"]["shape"] =
                serde_json::json!([4096, 1024, 2, 3]);
        });
        assert!(
            error.contains("downsample.weight has shape [4096, 1024, 2, 3]"),
            "{error}"
        );
        let (error, _) = wrong(|header| {
            header["model.visual.merger.proj.weight"]["dtype"] = "F32".into();
        });
        assert!(
            error.contains("stages[6]") && error.contains("F32"),
            "{error}"
        );
        check_tower(&request(0)).unwrap();
    }

    const POST_NORM: &str = "model.visual.post_layernorm.weight";
    const DOWN: &str = "model.visual.merger.down_proj.weight";
    const OTHER: &str = "z-other.safetensors";

    /// The test request with a second shard holding the merger's down
    /// projection and a stale copy of the post norm, the index assigning
    /// the first to the new shard and the second to `model.safetensors`
    /// (edited by `assign`); `indexed` false drops the index.
    fn two_shards(
        assign: impl FnOnce(&mut serde_json::Map<String, Value>),
        indexed: bool,
    ) -> WeightPlanRequestV1 {
        let mut request = request(0);
        let first: serde_json::Map<String, Value> =
            serde_json::from_slice(&request.files[0].header).unwrap();
        let mut other = serde_json::Map::new();
        for name in [DOWN, POST_NORM] {
            other.insert(name.into(), first[name].clone());
        }
        let header = serde_json::to_vec(&other).unwrap();
        request
            .files
            .push(runtime_bundle_abi::weight_plan::CheckpointFileV1 {
                name: OTHER.into(),
                byte_length: 8 + header.len() as u64 + (64 << 20),
                data_offset: 8 + header.len() as u64,
                header,
            });
        let document = request
            .documents
            .iter_mut()
            .position(|document| document.name == CHECKPOINT_INDEX)
            .unwrap();
        if indexed {
            let mut index: Value =
                serde_json::from_slice(&request.documents[document].bytes).unwrap();
            let map = index["weight_map"].as_object_mut().unwrap();
            map.insert(DOWN.into(), OTHER.into());
            assign(map);
            request.documents[document].bytes = serde_json::to_vec(&index).unwrap();
        } else {
            request.documents.remove(document);
        }
        request
    }

    /// The file of the one record that reads `tensor`.
    fn source_file(plan: &WeightPlanV1, tensor: &str) -> u32 {
        let mut files = plan
            .records
            .iter()
            .flat_map(|record| &record.sources)
            .filter(|source| source.tensor == tensor)
            .map(|source| source.file);
        let file = files.next().unwrap();
        assert!(files.next().is_none());
        file
    }

    /// Each tensor is read from the shard the index assigns it, whatever
    /// another selected shard also holds; a tensor its shard lacks, and
    /// without an index a tensor in two shards, is refused.
    #[test]
    fn each_tensor_is_read_from_its_designated_shard() {
        let plan = plan(&two_shards(|_| {}, true)).unwrap();
        assert_eq!(source_file(&plan, POST_NORM), 0);
        assert_eq!(source_file(&plan, DOWN), 1);
        let misplaced = two_shards(
            |map| {
                map.insert(
                    "model.visual.blocks.3.attn.k_norm.weight".into(),
                    OTHER.into(),
                );
            },
            true,
        );
        for error in [
            super::plan(&misplaced).unwrap_err(),
            check_tower(&misplaced).unwrap_err(),
        ] {
            assert!(
                error.contains("blocks.3.attn.k_norm.weight") && error.contains(OTHER),
                "{error}"
            );
        }
        let ambiguous = two_shards(|_| {}, false);
        let error = super::plan(&ambiguous).unwrap_err();
        assert!(
            error.contains("is in both") && error.contains(OTHER),
            "{error}"
        );
    }

    /// A source tensor whose own axes differ from the checkpoint's, though
    /// flattening and padding give the role's shape, is refused: a patch
    /// weight of 14 x 13 patches (1092 columns, padded to 1216 alike) and a
    /// downsample whose axes flatten to the same 4096 columns in another
    /// order.
    #[test]
    fn source_axes_are_checked_before_their_transforms() {
        for (name, shape) in [
            (
                "model.visual.patch_embed.proj.weight",
                json!([1024, 3, 2, 14, 13]),
            ),
            (
                "model.visual.patch_embed.proj.weight",
                json!([1024, 2, 3, 14, 14]),
            ),
            ("model.visual.downsample.weight", json!([4096, 2, 2, 1024])),
        ] {
            let mut request = crate::direct::test_support::request(0, |header| {
                header.extend(vision_header());
                header[name]["shape"] = shape.clone();
            });
            request.image_role = std::str::from_utf8(views::IMAGE_ROLE).unwrap().into();
            for error in [
                plan(&request).unwrap_err(),
                check_tower(&request).unwrap_err(),
            ] {
                assert!(
                    error.contains(name) && error.contains(&shape.to_string().replace(',', ", ")),
                    "{shape}: {error}"
                );
            }
        }
    }
}
