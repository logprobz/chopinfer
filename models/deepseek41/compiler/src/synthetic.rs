//! A checkpoint of a compiled geometry with deterministic random weights:
//! every tensor the projection declaration (`contract/projections.json`)
//! names at that geometry, in the official export's dtypes and companion
//! scales, so the vendor reference and the bundle load the same weights
//! (the synthetic M0 run, `examples/geometry-checkpoint.rs`).
//!
//! Each tensor's values come from its own generator, seeded by its name, so
//! a tensor's bytes do not depend on the order or set of other tensors.
//! Values are finite and scaled like trained weights: projections about
//! `1 / sqrt(K)`, norms and Engram weights about 1. Block-scaled codes avoid
//! NaN codes and keep UE8M0 scales far from the MXFP4 overflow scales.

use kernel_core::precision::format::FormatId;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::projections;
use projection_spec::{DType, Dim, ProjectionKind, SideKind, TensorSource};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// The head's uniform bound times sqrt(hidden). Unit-RMS hidden rows give
/// logits of standard deviation HEAD_SCALE / sqrt(3), about 4.6, so the
/// next-token distribution is peaked (a mean entropy near 4 nats over the
/// flash vocabulary) rather than near uniform.
pub const HEAD_SCALE: f32 = 8.0;

/// How a tensor's values are drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fill {
    /// BF16 or F32 values uniform in `[-scale, scale]`.
    Uniform { scale: f32 },
    /// BF16 or F32 values uniform in `[1 - scale, 1 + scale]`.
    AroundOne { scale: f32 },
    /// E4M3 codes of magnitude 2^-6 to 2^2 (no NaN, no zero exponent).
    E4m3Codes,
    /// Packed E2M1 pairs, every code valid.
    E2m1Codes,
    /// UE8M0 scales `2^exponent`.
    Ue8m0 { exponent: i32 },
    /// UE8M0 scales `2^(exponent + d)`, `d` uniform in `[-spread, spread]`
    /// per scale, so neighbouring blocks differ.
    Ue8m0Varied { exponent: i32, spread: u32 },
}

/// One checkpoint tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    /// Safetensors dtype name.
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub fill: Fill,
}

impl Tensor {
    pub fn bytes(&self) -> u64 {
        let width = match self.dtype {
            "F32" => 4,
            "BF16" => 2,
            _ => 1,
        };
        self.shape.iter().product::<u64>() * width
    }
}

fn resolve(dims: &[Dim], keys: &projection_spec::Geometry) -> Result<Vec<u64>, String> {
    dims.iter()
        .map(|d| d.resolve(keys).map_err(|e| e.0))
        .collect()
}

/// The checkpoint names a source takes: every index of its stem, if any.
fn names(
    source: &TensorSource,
    stems: &BTreeMap<String, (String, Vec<u64>)>,
) -> Result<Vec<String>, String> {
    let Some(stem) = &source.stem else {
        return Ok(vec![source.tensor.clone()]);
    };
    let (template, indices) = stems
        .get(stem)
        .ok_or_else(|| format!("unknown layer stem {stem}"))?;
    Ok(indices
        .iter()
        .map(|i| {
            source
                .tensor
                .replace("{stem}", &template.replace("{i}", &i.to_string()))
        })
        .collect())
}

/// `log2(sqrt(k))`, rounded: the UE8M0 exponent that scales codes of
/// magnitude about 1 to `1 / sqrt(k)`.
fn half_log2(k: u64) -> i32 {
    ((k as f64).log2() / 2.0).round() as i32
}

/// Every tensor of a `geometry` checkpoint, sorted by name.
pub fn tensors(geometry: &Geometry) -> Result<Vec<Tensor>, String> {
    let keys = projections::geometry(geometry);
    let declaration = projections::declaration(geometry).map_err(|e| e.0)?;
    let mut stems = BTreeMap::new();
    for stem in &declaration.layer_stems {
        let indices = stem.indices.resolve(&keys).map_err(|e| e.0)?;
        stems.insert(stem.id.clone(), (stem.template.clone(), indices));
    }
    let mut out = Vec::new();
    for projection in &declaration.projections {
        let Some(source) = &projection.source else {
            continue;
        };
        let [rows, k] = resolve(&projection.shape, &keys)?[..] else {
            return Err(format!(
                "{}: a projection shape has two dims",
                projection.id
            ));
        };
        // A grouped dense projection's checkpoint tensor stacks its groups
        // along rows.
        let rows = match &projection.kind {
            ProjectionKind::GroupedDense { groups } => {
                rows * groups.resolve(&keys).map_err(|e| e.0)?
            }
            _ => rows,
        };
        let experts = match &projection.kind {
            ProjectionKind::ExpertBank { experts, .. } => {
                Some(experts.resolve(&keys).map_err(|e| e.0)?)
            }
            _ => None,
        };
        let lookup = matches!(projection.kind, ProjectionKind::LookupTable {});
        for stemmed in names(source, &stems)? {
            let mut members = vec![stemmed];
            if let Some(experts) = experts {
                members = members
                    .iter()
                    .flat_map(|n| (0..experts).map(move |e| n.replace("{expert}", &e.to_string())))
                    .collect();
            }
            if projection.operands.len() > 1 || source.tensor.contains("{operand}") {
                members = members
                    .iter()
                    .flat_map(|n| {
                        projection
                            .operands
                            .iter()
                            .map(move |o| n.replace("{operand}", o))
                    })
                    .collect();
            }
            for weight in members {
                let scale_name = weight
                    .strip_suffix(".weight")
                    .map(|stem| format!("{stem}.scale"))
                    .ok_or_else(|| format!("{weight}: a quantized weight ends in .weight"))?;
                match projection.format {
                    Some(FormatId::Bf16) => out.push(Tensor {
                        name: weight,
                        dtype: "BF16",
                        shape: vec![rows, k],
                        fill: Fill::Uniform {
                            scale: if lookup {
                                1.0
                            } else if projection.id == "head" {
                                HEAD_SCALE / (k as f32).sqrt()
                            } else {
                                1.0 / (k as f32).sqrt()
                            },
                        },
                    }),
                    Some(FormatId::Fp8Block32) => {
                        out.push(Tensor {
                            name: weight,
                            dtype: "F8_E4M3",
                            shape: vec![rows, k],
                            fill: Fill::E4m3Codes,
                        });
                        out.push(Tensor {
                            name: scale_name,
                            dtype: "F8_E8M0",
                            shape: vec![rows.div_ceil(32), k.div_ceil(32)],
                            fill: Fill::Ue8m0 {
                                exponent: -half_log2(k),
                            },
                        });
                    }
                    Some(FormatId::Mxfp4) => {
                        out.push(Tensor {
                            name: weight,
                            dtype: "I8",
                            shape: vec![rows, k / 2],
                            fill: Fill::E2m1Codes,
                        });
                        out.push(Tensor {
                            name: scale_name,
                            dtype: "F8_E8M0",
                            shape: vec![rows, k / 32],
                            fill: Fill::Ue8m0 {
                                exponent: -half_log2(k) - 1,
                            },
                        });
                    }
                    Some(FormatId::Mxfp8) => {
                        out.push(Tensor {
                            name: weight,
                            dtype: "F8_E4M3",
                            shape: vec![rows, k],
                            fill: Fill::E4m3Codes,
                        });
                        out.push(Tensor {
                            name: scale_name,
                            dtype: "F8_E8M0",
                            shape: vec![rows, k / 32],
                            fill: Fill::Ue8m0 { exponent: 0 },
                        });
                    }
                    other => {
                        return Err(format!(
                            "{}: no synthetic encoding for format {other:?}",
                            projection.id
                        ));
                    }
                }
            }
        }
    }
    for side in &declaration.side_tensors {
        let shape = resolve(&side.shape, &keys)?;
        let dtype = match side.dtypes.first() {
            Some(DType::F32) => "F32",
            Some(DType::Bf16) => "BF16",
            other => return Err(format!("{}: no synthetic dtype {other:?}", side.id)),
        };
        let fill = match side.kind {
            SideKind::Norm => Fill::AroundOne { scale: 0.1 },
            _ if side.id.ends_with("q_weight") || side.id.ends_with("k_weight") => {
                Fill::AroundOne { scale: 0.1 }
            }
            _ if shape.len() == 2 => Fill::Uniform {
                scale: 1.0 / (shape[1] as f32).sqrt(),
            },
            _ if side.id.ends_with("_scale") => Fill::AroundOne { scale: 0.5 },
            _ => Fill::Uniform { scale: 0.1 },
        };
        for name in names(&side.source, &stems)? {
            out.push(Tensor {
                name,
                dtype,
                shape: shape.clone(),
                fill,
            });
        }
    }
    for external in &declaration.external_tensors {
        let shape = resolve(&external.shape, &keys)?;
        if external.dtype != DType::Bf16 {
            return Err(format!(
                "{}: no synthetic dtype {:?}",
                external.id, external.dtype
            ));
        }
        let fill = if shape.len() == 2 {
            Fill::Uniform {
                scale: 1.0 / (shape[1] as f32).sqrt(),
            }
        } else if external.id.contains("norm") {
            Fill::AroundOne { scale: 0.1 }
        } else if external.id.ends_with("bias") {
            Fill::Uniform { scale: 0.1 }
        } else {
            Fill::Uniform { scale: 1.0 }
        };
        for name in names(&external.source, &stems)? {
            out.push(Tensor {
                name,
                dtype: "BF16",
                shape: shape.clone(),
                fill,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    if out.windows(2).any(|w| w[0].name == w[1].name) {
        return Err("the declaration names a tensor twice".into());
    }
    Ok(out)
}

/// The names of the encoder-owned tensors (`external_tensors`), which the
/// encoder declaration validates before binding.
pub fn external_names(geometry: &Geometry) -> Result<std::collections::BTreeSet<String>, String> {
    let keys = projections::geometry(geometry);
    let declaration = projections::declaration(geometry).map_err(|e| e.0)?;
    let mut stems = BTreeMap::new();
    for stem in &declaration.layer_stems {
        let indices = stem.indices.resolve(&keys).map_err(|e| e.0)?;
        stems.insert(stem.id.clone(), (stem.template.clone(), indices));
    }
    let mut out = std::collections::BTreeSet::new();
    for external in &declaration.external_tensors {
        out.extend(names(&external.source, &stems)?);
    }
    Ok(out)
}

/// SplitMix64 over a seed from the tensor's name (FNV-1a).
struct Values(u64);

impl Values {
    fn of(name: &str, seed: u64) -> Values {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        Values(hash)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1), from 24 random bits.
    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1 << 23) as f32 - 1.0
    }
}

fn bf16(value: f32) -> [u8; 2] {
    kernel_core::codec::f32_to_bf16(value).to_le_bytes()
}

/// The bytes a write of one tensor's values buffers before it flushes.
pub const CHUNK_BYTES: usize = 1 << 16;

/// Writes `tensor`'s values, in writes of at most [`CHUNK_BYTES`] plus one
/// value whatever the fill.
fn fill(tensor: &Tensor, seed: u64, out: &mut impl Write) -> std::io::Result<()> {
    let mut values = Values::of(&tensor.name, seed);
    let count = tensor.shape.iter().product::<u64>();
    let mut chunk = Vec::with_capacity(CHUNK_BYTES + 4);
    for _ in 0..count {
        match tensor.fill {
            Fill::Uniform { scale } => push(tensor, values.signed() * scale, &mut chunk),
            Fill::AroundOne { scale } => push(tensor, 1.0 + values.signed() * scale, &mut chunk),
            Fill::E4m3Codes => {
                let bits = values.next();
                // Exponent field 1..=9 (2^-6..2^2), any mantissa, any sign.
                let exponent = 1 + (bits % 9) as u8;
                chunk.push(((bits >> 8) as u8 & 0x80) | exponent << 3 | ((bits >> 16) as u8 & 7));
            }
            Fill::E2m1Codes => chunk.push(values.next() as u8),
            Fill::Ue8m0 { exponent } => chunk.push((127 + exponent) as u8),
            Fill::Ue8m0Varied { exponent, spread } => {
                let offset = (values.next() % (2 * u64::from(spread) + 1)) as i32 - spread as i32;
                chunk.push((127 + exponent + offset) as u8);
            }
        }
        if chunk.len() >= CHUNK_BYTES {
            out.write_all(&chunk)?;
            chunk.clear();
        }
    }
    out.write_all(&chunk)
}

/// Appends `value` in `tensor`'s dtype: F32, else BF16.
fn push(tensor: &Tensor, value: f32, chunk: &mut Vec<u8>) {
    match tensor.dtype {
        "F32" => chunk.extend(value.to_le_bytes()),
        _ => chunk.extend(bf16(value)),
    }
}

/// The single-file checkpoint's safetensors header (padded to 8 bytes, as
/// written) and its `model.safetensors.index.json`, with every tensor laid
/// out back to back in order, and the payload bytes.
pub fn header(tensors: &[Tensor]) -> Result<(Vec<u8>, Vec<u8>, u64), String> {
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut offset = 0_u64;
    for tensor in tensors {
        let end = offset + tensor.bytes();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({"dtype": tensor.dtype, "shape": tensor.shape,
                "data_offsets": [offset, end]}),
        );
        weight_map.insert(tensor.name.clone(), FILE.into());
        offset = end;
    }
    let mut header = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let index = serde_json::json!({"metadata": {"total_size": offset}, "weight_map": weight_map});
    let index = serde_json::to_vec_pretty(&index).map_err(|e| e.to_string())?;
    Ok((header, index, offset))
}

/// The one safetensors file a written checkpoint holds.
pub const FILE: &str = "model.safetensors";

/// Writes `tensors` at `dir` as one safetensors file with its index; each
/// tensor's values come from `seed` and its name. Returns the payload
/// bytes.
pub fn write(dir: &Path, tensors: &[Tensor], seed: u64) -> Result<u64, String> {
    let (header, index, offset) = header(tensors)?;
    let io = |e: std::io::Error| e.to_string();
    let mut file = BufWriter::new(File::create(dir.join(FILE)).map_err(io)?);
    file.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io)?;
    file.write_all(&header).map_err(io)?;
    for tensor in tensors {
        fill(tensor, seed, &mut file).map_err(io)?;
    }
    file.flush().map_err(io)?;
    std::fs::write(dir.join("model.safetensors.index.json"), index).map_err(io)?;
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    /// Records the largest write and refuses writes past `limit` bytes.
    struct Peak {
        largest: usize,
        written: usize,
        limit: usize,
    }

    impl Write for Peak {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.written >= self.limit {
                return Err(std::io::Error::other("enough"));
            }
            self.largest = self.largest.max(buf.len());
            self.written += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Every fill streams in chunks: a whole tensor of each mode writes all
    /// its bytes in writes of at most one chunk and a value, and Flash's
    /// largest quantized tensor (an Engram table of about 98 GB) starts
    /// writing without buffering itself.
    #[test]
    fn every_fill_streams_in_chunks() {
        let elements = 5 * CHUNK_BYTES as u64 + 3;
        for (dtype, mode) in [
            ("F32", Fill::Uniform { scale: 1.0 }),
            ("BF16", Fill::AroundOne { scale: 0.1 }),
            ("F8_E4M3", Fill::E4m3Codes),
            ("I8", Fill::E2m1Codes),
            ("F8_E8M0", Fill::Ue8m0 { exponent: -3 }),
            (
                "F8_E8M0",
                Fill::Ue8m0Varied {
                    exponent: -3,
                    spread: 2,
                },
            ),
        ] {
            let tensor = Tensor {
                name: "t".into(),
                dtype,
                shape: vec![elements],
                fill: mode,
            };
            let mut out = Peak {
                largest: 0,
                written: 0,
                limit: usize::MAX,
            };
            fill(&tensor, 7, &mut out).unwrap();
            assert_eq!(out.written as u64, tensor.bytes(), "{mode:?}");
            assert!(out.largest <= CHUNK_BYTES + 4, "{mode:?}: {}", out.largest);
        }
        let tensors = tensors(&FLASH).unwrap();
        let table = tensors
            .iter()
            .find(|t| t.name == "layers.14.engram.embed.weight")
            .unwrap();
        assert!(table.bytes() > 90 << 30, "{}", table.bytes());
        let mut out = Peak {
            largest: 0,
            written: 0,
            limit: 8 * CHUNK_BYTES,
        };
        assert!(fill(table, 7, &mut out).is_err());
        assert!(out.largest <= CHUNK_BYTES + 4, "{}", out.largest);
    }

    /// Varied UE8M0 scales stay within their spread and differ between
    /// neighbouring blocks.
    #[test]
    fn varied_scales_vary_within_their_spread() {
        let tensor = Tensor {
            name: "s".into(),
            dtype: "F8_E8M0",
            shape: vec![4096],
            fill: Fill::Ue8m0Varied {
                exponent: -4,
                spread: 3,
            },
        };
        let mut bytes = Vec::new();
        fill(&tensor, 1, &mut bytes).unwrap();
        assert!(bytes.iter().all(|&b| (120..=126).contains(&b)));
        assert!(bytes.windows(2).filter(|w| w[0] != w[1]).count() > 3000);
    }

    #[test]
    fn the_flash_catalog_is_the_official_checkpoints() {
        let tensors = tensors(&FLASH).unwrap();
        // The official revision's header count (checkpoint-files.json).
        assert_eq!(tensors.len(), 96_085);
        let find = |name: &str| tensors.iter().find(|t| t.name == name).unwrap();
        let wq_a = find("layers.9.attn.wq_a.scale");
        assert_eq!(
            (wq_a.dtype, wq_a.shape.as_slice()),
            ("F8_E8M0", &[40, 160][..])
        );
        let w2 = find("layers.3.ffn.experts.17.w2.weight");
        assert_eq!((w2.dtype, w2.shape.as_slice()), ("I8", &[5120, 1152][..]));
        let table = find("layers.14.engram.embed.scale");
        assert_eq!(table.shape, [384_016_682, 8]);
        let sink = find("mtp.2.attn.attn_sink");
        assert_eq!((sink.dtype, sink.shape.as_slice()), ("F32", &[64][..]));
        assert_eq!(find("vision.blocks.31.mlp.w1.weight").shape, [5632, 1024]);
    }

    #[test]
    fn the_flash_catalog_equals_the_official_headers() {
        let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT") else {
            eprintln!("CHOPINFER_DEEPSEEK_CHECKPOINT is unset: skipped");
            return;
        };
        let metadata = crate::checkpoint::metadata(Path::new(&root)).unwrap();
        let official: BTreeMap<_, _> = metadata
            .tensors
            .iter()
            .map(|(name, info)| (name.clone(), (info.dtype.clone(), info.shape.clone())))
            .collect();
        let generated: BTreeMap<_, _> = tensors(&FLASH)
            .unwrap()
            .into_iter()
            .map(|t| (t.name, (t.dtype.to_owned(), t.shape)))
            .collect();
        // Name the tensors that differ rather than print both catalogs.
        let differing: Vec<_> = official
            .keys()
            .chain(generated.keys())
            .filter(|name| official.get(*name) != generated.get(*name))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|name| (name, official.get(name), generated.get(name)))
            .collect();
        assert!(
            differing.is_empty(),
            "{} tensors differ (name, official, generated), first: {:?}",
            differing.len(),
            &differing[..differing.len().min(8)]
        );
    }

    #[test]
    fn the_head_is_scaled_for_peaked_logits() {
        let tensors = tensors(&FLASH).unwrap();
        let find = |name: &str| tensors.iter().find(|t| t.name == name).unwrap();
        let k = FLASH.shape.hidden as f32;
        assert_eq!(
            find("head.weight").fill,
            Fill::Uniform {
                scale: HEAD_SCALE / k.sqrt()
            }
        );
    }

    #[test]
    fn values_follow_the_name_not_the_order() {
        let tensor = Tensor {
            name: "layers.0.attn.wq_a.weight".into(),
            dtype: "F8_E4M3",
            shape: vec![64, 64],
            fill: Fill::E4m3Codes,
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        fill(&tensor, 7, &mut a).unwrap();
        fill(&tensor, 7, &mut b).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4096);
        // No NaN code (S.1111.111) and no zero exponent.
        assert!(a.iter().all(|&c| c & 0x7f != 0x7f && (c >> 3) & 0xf != 0));
        let mut c = Vec::new();
        fill(&tensor, 8, &mut c).unwrap();
        assert_ne!(a, c);
    }
}
