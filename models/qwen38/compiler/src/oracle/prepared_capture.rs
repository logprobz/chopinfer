//! Operation-isolated checks of real TP4 dumps against prepared weight operands.
//! Each projection consumes the captured BF16 input, including expert down's
//! rounded SwiGLU result. This is separate from the unrounded model chain.

use super::{linalg::Matrix, weights::Checkpoint};
use crate::Result;
use kernel_core::{contract::DType, oracle};
use model_qwen38_contract::{contract::*, precision::DenseRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range, path::Path};

#[derive(Deserialize)]
struct Entry {
    name: String,
    rank: usize,
    first_row: usize,
    rows: usize,
    cols: usize,
    dtype: String,
}

struct Capture {
    tensors: BTreeMap<(String, usize), (Entry, Vec<f64>)>,
    digests: BTreeMap<String, String>,
    items: serde_json::Value,
}

impl Capture {
    fn read(root: &Path) -> Result<Self> {
        let mut result = Self {
            tensors: BTreeMap::new(),
            digests: BTreeMap::new(),
            items: serde_json::from_slice(&std::fs::read(root.join("items.json"))?)?,
        };
        let v1 = super::tap_names::is_v1(root)?;
        for line in std::fs::read_to_string(root.join("index.jsonl"))?.lines() {
            let entry: Entry = serde_json::from_str(line)?;
            if entry.name.is_empty()
                || !entry
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._".contains(&c))
                || entry.rank >= 4
                || entry.rows == 0
                || entry.rows > PREFILL_TOKENS as usize
            {
                return Err("invalid TP4 capture entry".into());
            }
            let dtype = match entry.dtype.as_str() {
                "bf16" => runtime_weights::DType::Bf16,
                "f32" => runtime_weights::DType::F32,
                "u32" => runtime_weights::DType::I64,
                _ => return Err("unsupported capture dtype".into()),
            };
            let name = format!("{}.r{}.bin", entry.name, entry.rank);
            let bytes = std::fs::read(root.join(&name))?;
            let size = if entry.dtype == "bf16" { 2 } else { 4 };
            if entry
                .rows
                .checked_mul(entry.cols)
                .and_then(|n| n.checked_mul(size))
                != Some(bytes.len())
            {
                return Err(format!("{name}: capture byte count differs").into());
            }
            let values = if entry.dtype == "u32" {
                bytes
                    .chunks_exact(4)
                    .map(|v| f64::from(u32::from_le_bytes(v.try_into().unwrap())))
                    .collect()
            } else {
                super::trace::decode(&name, dtype, &bytes)?
            };
            result
                .digests
                .insert(name, runtime_weights::sha256_bytes(&bytes));
            // Taps go by their declared names; undeclared older ones are unused.
            let Some(tap) = super::tap_names::name(&entry.name, v1) else {
                continue;
            };
            if result
                .tensors
                .insert((tap, entry.rank), (entry, values))
                .is_some()
            {
                return Err("duplicate capture tensor".into());
            }
        }
        Ok(result)
    }

    fn entry(&self, name: &str, rank: usize) -> Result<&(Entry, Vec<f64>)> {
        self.tensors
            .get(&(name.into(), rank))
            .ok_or_else(|| format!("missing capture {name} rank {rank}").into())
    }

    fn row(
        &self,
        name: &str,
        rank: usize,
        row: usize,
        columns: Range<usize>,
        dtype: &str,
    ) -> Result<&[f64]> {
        let (entry, values) = self.entry(name, rank)?;
        if entry.dtype != dtype
            || row < entry.first_row
            || row >= entry.first_row + entry.rows
            || columns.end > entry.cols
            || columns.start > columns.end
        {
            return Err(format!("{name} rank {rank}: row, columns or dtype differs").into());
        }
        let start = (row - entry.first_row) * entry.cols;
        Ok(&values[start + columns.start..start + columns.end])
    }
}

#[derive(Default, Serialize)]
struct Statistics {
    rows: usize,
    elements: usize,
    max_abs_error: f64,
    max_error_over_bound: f64,
    minimum_cosine: Option<f64>,
    squared_error: f64,
    squared_reference: f64,
    relative_rms: Option<f64>,
    bound_violations: usize,
    cosine_violations: usize,
}

impl Statistics {
    fn add(
        &mut self,
        actual: &[f64],
        reference: &oracle::OracleOutput,
        enforce: bool,
    ) -> Result<()> {
        if enforce {
            oracle::check_numeric(actual, reference, DType::Bf16)
                .map_err(|e| format!("captured projection violates the declared bound: {e:?}"))?;
        }
        if actual.len() != reference.values.len()
            || actual
                .iter()
                .chain(&reference.values)
                .any(|v| !v.is_finite())
        {
            return Err("measurement requires matching finite vectors".into());
        }
        let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (i, (&a, &b)) in actual.iter().zip(&reference.values).enumerate() {
            let error = (a - b).abs();
            self.max_abs_error = self.max_abs_error.max(error);
            self.squared_error += error * error;
            self.squared_reference += b * b;
            self.bound_violations +=
                usize::from(error > oracle::numeric_bound(reference, i, DType::Bf16));
            self.max_error_over_bound = self.max_error_over_bound.max(
                error / oracle::numeric_bound(reference, i, DType::Bf16).max(f64::MIN_POSITIVE),
            );
            dot += a * b;
            aa += a * a;
            bb += b * b;
        }
        if aa > 0.0 && bb > 0.0 {
            let cosine = dot / (aa * bb).sqrt();
            if cosine < 0.9999 {
                self.cosine_violations += 1;
                if enforce {
                    return Err(format!("captured projection cosine {cosine} < 0.9999").into());
                }
            }
            self.minimum_cosine = Some(self.minimum_cosine.map_or(cosine, |old| old.min(cosine)));
        }
        self.relative_rms = if self.squared_reference > 0.0 {
            Some((self.squared_error / self.squared_reference).sqrt())
        } else {
            None
        };
        self.rows += 1;
        self.elements += actual.len();
        Ok(())
    }
}

fn reference(input: &[f64], weight: &Matrix, swiglu: bool) -> Result<oracle::OracleOutput> {
    Ok(oracle::gemm(
        input,
        &weight.data,
        oracle::GemmShape {
            m: 1,
            n: weight.rows,
            k: weight.cols,
            experts: 1,
        },
        &[],
        oracle::GemmEpilogue {
            residual: None,
            swiglu: swiglu.then_some(kernel_core::contract::SwigluColumns::ConcatenatedHalves),
        },
    )?)
}

fn weight_identity(weight: &Matrix) -> serde_json::Value {
    let mut hash = Sha256::new();
    for value in &weight.data {
        hash.update(value.to_le_bytes());
    }
    serde_json::json!({ "rows": weight.rows, "k": weight.cols,
        "decoded_f64_sha256": format!("{:x}", hash.finalize()) })
}

fn shard(
    checkpoint: &Checkpoint,
    name: &str,
    rank: usize,
    rows: usize,
    k: usize,
) -> Result<Matrix> {
    let full = checkpoint.projection(name)?;
    let mut record = checkpoint.record(name)?.clone();
    record.shape = checkpoint.logical_shape(name)?;
    let slices = record.rank_slices_for(&model_qwen38_contract::geometry::FLASH, rank, 4);
    let mut result = Matrix::new(rows, k, vec![0.0; rows * k]);
    let mut offset = 0;
    for (axis, range) in slices {
        let start = usize::try_from(range.start)?;
        let end = usize::try_from(range.end)?;
        match axis {
            0 if end <= full.rows && offset + end - start <= rows && full.cols == k => {
                result.data[offset * k..(offset + end - start) * k]
                    .copy_from_slice(&full.data[start * k..end * k]);
            }
            1 if end <= full.cols && offset + end - start <= k && full.rows == rows => {
                for r in 0..rows {
                    result.data[r * k + offset..r * k + offset + end - start]
                        .copy_from_slice(&full.row(r)[start..end]);
                }
            }
            _ => {
                return Err(
                    format!("{name}: logical rank slice differs from projection geometry").into(),
                );
            }
        }
        offset += end - start;
    }
    Ok(result)
}

fn dense_taps(role: DenseRole) -> (&'static str, &'static str, usize) {
    match role {
        DenseRole::LinearQkv => ("x.hc.attn", "x.gdn.qkv", 0),
        DenseRole::LinearZ => ("x.hc.attn", "x.gdn.z", 0),
        DenseRole::LinearA => ("x.hc.attn", "x.gdn.ab", 0),
        DenseRole::LinearB => ("x.hc.attn", "x.gdn.ab", Shape::TP4.value_heads as usize),
        DenseRole::AttentionQ => ("x.hc.attn", "x.attn.qkv", 0),
        DenseRole::AttentionK => (
            "x.hc.attn",
            "x.attn.qkv",
            (Shape::TP4.attn_value * 2) as usize,
        ),
        DenseRole::AttentionV => (
            "x.hc.attn",
            "x.attn.qkv",
            (Shape::TP4.attn_value * 2 + KV_HEADS * ATTN_HEAD) as usize,
        ),
        DenseRole::IndexQk => ("x.hc.attn", "x.indexer.qk", 0),
        // Linear attention's gated norm or full attention's gated output.
        DenseRole::Output | DenseRole::AttentionOutput => ("attn.core", "attn.out", 0),
        DenseRole::SharedGateUp => ("x.hc.mlp", "x.moe.shared_intermediate", 0),
        DenseRole::SharedDown => ("x.moe.shared_intermediate", "moe.shared", 0),
    }
}

/// Check selected complete rows of a single captured phase. Missing operands,
/// empty selections and shape differences refuse rather than reducing coverage.
pub fn compare(
    checkpoint: &Checkpoint,
    root: &Path,
    layers: Range<usize>,
    tail: usize,
    enforce: bool,
) -> Result<serde_json::Value> {
    if !checkpoint.has_prepared_precision()
        || tail == 0
        || layers.is_empty()
        || layers.end > LAYERS as usize
    {
        return Err(
            "prepared capture requires an execution component, nonempty layers and positive tail"
                .into(),
        );
    }
    let capture = Capture::read(root)?;
    let mut results = BTreeMap::<String, Statistics>::new();
    let mut selected = BTreeMap::new();
    let mut weight_identities = BTreeMap::new();
    for layer in layers {
        let name = |site: &str| format!("L{layer:02}.{site}");
        let (entry, _) = capture.entry(&name("x.hc.mlp"), 0)?;
        let end = entry.first_row + entry.rows;
        let rows = end - tail.min(entry.rows)..end;
        selected.insert(layer, rows.clone());
        for role in DenseRole::roles_for(Shape::TP4, layer as u32)? {
            let geometry = role.logical_geometry(Shape::TP4)?;
            let (n, k) = (geometry.rows as usize, geometry.k as usize);
            let (input, output, offset) = dense_taps(role);
            for rank in 0..4 {
                let mut weights = role
                    .operand_names(Shape::TP4, layer as u32)?
                    .iter()
                    .map(|name| shard(checkpoint, name, rank, n, k))
                    .collect::<Result<Vec<_>>>()?;
                let mut weight = weights.remove(0);
                for second in weights {
                    weight.rows += second.rows;
                    weight.data.extend(second.data);
                }
                weight_identities.insert(
                    format!("layer{layer:02}.{role:?}.rank{rank}"),
                    weight_identity(&weight),
                );
                for row in rows.clone() {
                    let input_rank = if matches!(input, "x.hc.attn" | "x.hc.mlp") {
                        0
                    } else {
                        rank
                    };
                    let x = capture.row(&name(input), input_rank, row, 0..k, "bf16")?;
                    let y = capture.row(&name(output), rank, row, offset..offset + n, "bf16")?;
                    let reference = reference(x, &weight, role == DenseRole::SharedGateUp)?;
                    results
                        .entry(format!("layer{layer:02}.{role:?}.rank{rank}"))
                        .or_default()
                        .add(y, &reference, enforce)?;
                }
            }
        }
        let mut experts = BTreeMap::<usize, (Matrix, Matrix)>::new();
        for row in rows {
            let ids = capture.row(&name("router.ids"), 0, row, 0..TOP_K as usize, "u32")?;
            let route_weights =
                capture.row(&name("router.weights"), 0, row, 0..TOP_K as usize, "f32")?;
            let logits = capture.row(&name("router.logits"), 0, row, 0..EXPERTS as usize, "f32")?;
            let selection = oracle::router(
                logits,
                None,
                TOP_K as usize,
                kernel_core::contract::RouterKind::SoftmaxTopK { renormalize: true },
            )?;
            if selection
                .expert_ids
                .iter()
                .map(|&e| e as f64)
                .collect::<Vec<_>>()
                != ids
            {
                return Err(
                    format!("layer{layer:02} row{row}: captured routing IDs differ").into(),
                );
            }
            for (route, &expert) in ids.iter().enumerate() {
                let expert = expert as usize;
                if expert >= EXPERTS as usize
                    || !route_weights[route].is_finite()
                    || route_weights[route] < 0.0
                {
                    return Err("invalid captured expert or route weight".into());
                }
                let rank = expert / Shape::TP4.local_experts as usize;
                if let std::collections::btree_map::Entry::Vacant(entry) = experts.entry(expert) {
                    let stem = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
                    let mut gate = checkpoint.projection(&format!("{stem}.gate_proj.weight"))?;
                    let up = checkpoint.projection(&format!("{stem}.up_proj.weight"))?;
                    gate.rows += up.rows;
                    gate.data.extend(up.data);
                    entry.insert((
                        gate,
                        checkpoint.projection(&format!("{stem}.down_proj.weight"))?,
                    ));
                }
                let (gate, down) = &experts[&expert];
                for (projection, weight) in [("gate_up", gate), ("down", down)] {
                    weight_identities
                        .entry(format!("layer{layer:02}.expert{expert}.{projection}"))
                        .or_insert_with(|| weight_identity(weight));
                }
                let x = capture.row(&name("x.hc.mlp"), 0, row, 0..HIDDEN as usize, "bf16")?;
                let intermediate = capture.row(
                    &name("x.moe.intermediate"),
                    rank,
                    row,
                    route * down.cols..(route + 1) * down.cols,
                    "bf16",
                )?;
                let y = capture.row(
                    &name("x.moe.routed"),
                    rank,
                    row,
                    route * down.rows..(route + 1) * down.rows,
                    "bf16",
                )?;
                results
                    .entry(format!("layer{layer:02}.expert{expert}.gate_up"))
                    .or_default()
                    .add(intermediate, &reference(x, gate, true)?, enforce)?;
                let mut reference = reference(intermediate, down, false)?;
                for value in &mut reference.values {
                    *value *= route_weights[route];
                }
                for value in &mut reference.sum_abs_products {
                    *value *= route_weights[route];
                }
                results
                    .entry(format!("layer{layer:02}.expert{expert}.down"))
                    .or_default()
                    .add(y, &reference, enforce)?;
            }
        }
    }
    checkpoint.verify_source()?;
    Ok(serde_json::json!({
        "schema": "chopinfer.qwen38.prepared-capture.v1",
        "precision": checkpoint.precision_provenance(),
        "items": capture.items, "input_sha256": capture.digests,
        "selected_rows": selected, "operations": results,
        "decoded_weights": weight_identities,
        "passed": results.values().all(|s| s.bound_violations == 0 && s.cosine_violations == 0),
        "contract": "f64 products over captured BF16 inputs and prepared decoded weights; half BF16 ulp + 2^-22*K*S, row cosine >= 0.9999; route IDs exact",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_arithmetic_rejects_a_changed_operand_and_preserves_rounding_boundary() {
        let w = Matrix::new(2, 2, vec![1.0, 0.5, -0.25, 1.0]);
        let input = [1.0078125, 0.50390625];
        let expected = reference(&input, &w, false).unwrap();
        let actual = expected
            .values
            .iter()
            .map(|v| oracle::bf16_round(*v))
            .collect::<Vec<_>>();
        let mut stats = Statistics::default();
        stats.add(&actual, &expected, true).unwrap();
        let wrong = reference(&[1.0, 0.5], &w, false).unwrap();
        assert!(stats.add(&actual, &wrong, true).is_err());
        let mut changed = actual;
        changed[0] += 0.125;
        assert!(stats.add(&changed, &expected, true).is_err());
        let mut measurement = Statistics::default();
        measurement.add(&changed, &expected, false).unwrap();
        assert!(measurement.bound_violations > 0);
        assert!(measurement.relative_rms.is_some_and(|v| v > 0.0));
    }
}
