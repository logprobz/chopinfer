//! Checkpoint tensor reads for the oracle, decoded to f64 exactly: BF16 values,
//! FP8 E4M3 codes times their BF16 inverse block scales, and I64 buffers.

use kernel_core::oracle::e4m3_table;
use std::path::{Path, PathBuf};

use kernel_core::codec::bf16_to_f32;
use runtime_weights::DType;

use super::linalg::Matrix;
use crate::Result;
use crate::catalog::TensorRecord;
use kernel_core::precision::{
    DecodeOrder, OperationKind,
    execution::{ActivationPolicy, PrecisionDocument},
};
use std::collections::BTreeMap;

pub const BLOCK: usize = 128;

pub fn e4m3(code: u8) -> f64 {
    e4m3_table()[code as usize]
}

pub fn bf16(bits: u16) -> f64 {
    f64::from(bf16_to_f32(bits))
}

/// `codes` [rows, k] times the BF16 inverse scale of each 128 x 128 block,
/// `scales` [ceil(rows / 128), ceil(k / 128)]. The product is exact in f64.
pub fn dequantize_fp8_blocks(codes: &[u8], rows: usize, k: usize, scales: &[u16]) -> Matrix {
    assert_eq!(codes.len(), rows * k, "FP8 weight shape");
    let scale_cols = k.div_ceil(BLOCK);
    assert_eq!(
        scales.len(),
        rows.div_ceil(BLOCK) * scale_cols,
        "FP8 scale shape"
    );
    let mut data = Vec::with_capacity(rows * k);
    for (row, codes) in codes.chunks_exact(k).enumerate() {
        let scale_row = &scales[row / BLOCK * scale_cols..(row / BLOCK + 1) * scale_cols];
        data.extend(
            codes
                .iter()
                .enumerate()
                .map(|(col, &code)| e4m3(code) * bf16(scale_row[col / BLOCK])),
        );
    }
    Matrix::new(rows, k, data)
}

use runtime_weights::read_exact_at;

/// One retained compatible checkpoint with its header catalog.
pub struct Checkpoint {
    root: PathBuf,
    source: crate::checkpoint::CheckpointSource,
    decode_orders: Option<BTreeMap<String, DecodeOrder>>,
    precision_provenance: Option<serde_json::Value>,
}

impl Checkpoint {
    /// Validates the config and every shard header against the family contract.
    pub fn open(root: &Path) -> Result<Self> {
        let checkpoint = Self::open_source(root)?;
        if checkpoint
            .source
            .catalog
            .precision
            .as_ref()
            .is_some_and(|s| !s.legacy_text_layout())
        {
            return Err("mixed checkpoint oracle requires its prepared precision component (--precision-component)".into());
        }
        Ok(checkpoint)
    }

    fn open_source(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
            source: crate::checkpoint::CheckpointSource::open(root, crate::catalog::DEFAULT_RANKS)?,
            decode_orders: None,
            precision_provenance: None,
        })
    }

    /// Use the declared decoded operands of a prepared execution. Activation
    /// quantization needs its own oracle and cannot be silently treated as A16.
    pub fn open_prepared(root: &Path, precision: &PrecisionDocument) -> Result<Self> {
        let mut checkpoint = Self::open_source(root)?;
        let source = checkpoint
            .source
            .catalog
            .precision
            .as_ref()
            .ok_or("missing precision source")?;
        crate::projection_program::SourceProgram::new(
            source,
            model_qwen38_contract::contract::Shape::TP4,
        )?
        .verify_recorded(precision)?;
        let mut orders = BTreeMap::new();
        for selection in &precision.operations {
            selection.require_policy(ActivationPolicy::Bf16)?;
            let order = selection.effective.arithmetic.decode_order;
            match selection.source.kind {
                OperationKind::Dense => {
                    for weight in &selection.source.weights {
                        orders.insert(weight.codes.tensor.clone(), order);
                    }
                }
                OperationKind::Grouped { .. } => {
                    let (prefix, projection) = selection
                        .source
                        .id
                        .rsplit_once('.')
                        .ok_or("invalid expert operation")?;
                    let projections: &[&str] = match projection {
                        "gate_up_proj" => &["gate_proj", "up_proj"],
                        "down_proj" => &["down_proj"],
                        _ => return Err("unknown expert projection in oracle contract".into()),
                    };
                    for expert in 0..model_qwen38_contract::contract::EXPERTS {
                        for projection in projections {
                            orders.insert(format!("{prefix}.{expert}.{projection}.weight"), order);
                        }
                    }
                }
                OperationKind::Lookup => {
                    let prefix = &selection.source.id;
                    if prefix.ends_with(".ngram_embedding") {
                        for shard in 0..128 {
                            orders.insert(format!("{prefix}.shard_{shard}.weight"), order);
                        }
                    } else {
                        orders.insert(format!("{prefix}.weight"), order);
                    }
                }
            }
        }
        checkpoint.precision_provenance = Some(serde_json::json!({
            "execution_sha256": runtime_weights::sha256_bytes(&serde_json::to_vec(precision)?),
            "choices": model_compiler_sdk::quantization::identity::report(precision)?,
            "weight_decoding": "the prepared operation's scale application order and weight rounding; reference-only matrix expansion",
            "activation_arithmetic": "the selected oracle workflow controls activation rounding; f64 reference arithmetic does not simulate a kernel's reduction order",
        }));
        checkpoint.decode_orders = Some(orders);
        Ok(checkpoint)
    }

    pub fn precision_provenance(&self) -> Option<&serde_json::Value> {
        self.precision_provenance.as_ref()
    }

    pub fn has_prepared_precision(&self) -> bool {
        self.decode_orders.is_some()
    }

    pub fn logical_shape(&self, name: &str) -> Result<Vec<u64>> {
        self.source
            .catalog
            .precision
            .as_ref()
            .and_then(|s| s.operands.get(name))
            .map_or_else(
                || self.record(name).map(|r| r.shape.clone()),
                |operand| Ok(operand.weight.logical_shape.clone()),
            )
    }

    /// Content provenance is independent of release and tokenizer labels.
    pub fn content_revision(&self) -> Result<String> {
        self.source
            .measure(&std::collections::BTreeMap::new())?
            .0
            .sha256()
    }

    pub fn verify_source(&self) -> Result<()> {
        self.source.files.verify()
    }

    fn range(&self, record: &TensorRecord, offset: u64, length: usize) -> Result<Vec<u8>> {
        let file = self.source.files.file(&record.file)?;
        file.verify()?;
        let mut bytes = vec![0; length];
        read_exact_at(file, &mut bytes, record.file_start + offset)?;
        file.verify()?;
        Ok(bytes)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn record(&self, name: &str) -> Result<&TensorRecord> {
        self.source
            .catalog
            .tensors
            .get(name)
            .ok_or_else(|| format!("checkpoint has no tensor {name}").into())
    }

    fn typed(&self, name: &str, dtype: DType) -> Result<&TensorRecord> {
        let record = self.record(name)?;
        if record.dtype != dtype {
            return Err(format!("{name} is {:?}, expected {dtype:?}", record.dtype).into());
        }
        Ok(record)
    }

    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let record = self.record(name)?;
        self.range(record, 0, record.bytes as usize)
    }

    /// Every element of a BF16 tensor.
    pub fn bf16(&self, name: &str) -> Result<Vec<f64>> {
        self.typed(name, DType::Bf16)?;
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    }

    pub fn bf16_bits(&self, name: &str) -> Result<Vec<u16>> {
        self.typed(name, DType::Bf16)?;
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    /// A BF16 tensor as [shape[0], product of the rest].
    pub fn matrix(&self, name: &str) -> Result<Matrix> {
        if self.record(name)?.dtype != DType::Bf16 {
            return self.projection(name);
        }
        let record = self.typed(name, DType::Bf16)?;
        let rows = record.shape[0] as usize;
        let cols = record.shape[1..].iter().product::<u64>() as usize;
        Ok(Matrix::new(rows, cols, self.bf16(name)?))
    }

    /// A prepared projection's exact MMA weight operands, expanded only for
    /// reference arithmetic. Production images remain packed.
    pub fn projection(&self, name: &str) -> Result<Matrix> {
        if self.record(name)?.dtype == DType::Bf16 {
            return self.matrix(name);
        }
        let order = match &self.decode_orders {
            Some(orders) => *orders
                .get(name)
                .ok_or_else(|| format!("tensor {name}: no prepared oracle operation"))?,
            None => return self.fp8_block(name),
        };
        let source = self
            .source
            .catalog
            .precision
            .as_ref()
            .ok_or("missing imported precision")?;
        let operand = source
            .operands
            .get(name)
            .ok_or_else(|| format!("tensor {name}: no imported operand"))?;
        if !operand.weight.transforms.is_empty() {
            return Err(format!("tensor {name}: oracle weight transforms are unsupported").into());
        }
        let loaded =
            model_compiler_sdk::quantization::payload::read_operand(operand.clone(), |span| {
                source.inventory.read(&self.source.files, span)
            })?;
        let matrix = loaded.matrix(0)?;
        let mut values = Vec::with_capacity(matrix.rows * matrix.k);
        for row in 0..matrix.rows {
            for k in 0..matrix.k {
                values.push(if order == DecodeOrder::BlockScaleAfterMma {
                    e4m3(matrix.codes[row * matrix.k + k])
                        * f64::from(
                            matrix
                                .local
                                .ok_or("missing block scales")?
                                .value(matrix.scale_index(row, k)?)
                                .map_err(|e| format!("tensor {name}: {e}"))?,
                        )
                } else {
                    matrix.oracle_weight(row, k, order)?
                });
            }
        }
        self.verify_source()?;
        Ok(Matrix::new(matrix.rows, matrix.k, values))
    }

    /// A packed table is read one requested row at a time through shared codecs.
    pub fn prepared_lookup_rows(&self, rows: &[(String, u64)]) -> Result<Vec<Vec<f64>>> {
        let orders = self
            .decode_orders
            .as_ref()
            .ok_or("lookup needs a prepared execution contract")?;
        let source = self
            .source
            .catalog
            .precision
            .as_ref()
            .ok_or("missing imported precision")?;
        let mut values = Vec::with_capacity(rows.len());
        for (name, row) in rows {
            let operand = source
                .operands
                .get(name)
                .ok_or_else(|| format!("tensor {name}: no imported lookup"))?;
            let order = *orders
                .get(name)
                .ok_or_else(|| format!("tensor {name}: no prepared lookup operation"))?;
            let mut result = model_compiler_sdk::quantization::payload::read_lookup_rows(
                operand,
                &[*row],
                order,
                |span| source.inventory.read_slice(&self.source.files, span),
            )?;
            values.push(result.pop().unwrap().into_iter().map(bf16).collect());
        }
        self.verify_source()?;
        Ok(values)
    }

    /// An FP8 block-quantized weight with its `_scale_inv` companion.
    pub fn fp8_block(&self, name: &str) -> Result<Matrix> {
        let record = self.typed(name, DType::F8E4M3)?;
        let (rows, k) = (record.shape[0] as usize, record.shape[1] as usize);
        let scales = self.bf16_bits(&format!("{name}_scale_inv"))?;
        Ok(dequantize_fp8_blocks(&self.bytes(name)?, rows, k, &scales))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;
    use kernel_core::moe_oracle::fp8_block_checkpoint_weight;

    #[test]
    fn alternate_oracle_and_frontend_identities_measure_weights_with_shared_tokenizers() {
        use model_compiler_sdk::source::{SourceFiles, TensorSpan};
        use std::collections::BTreeMap;
        let Ok(tokenizer) = std::env::var("CHOPINFER_QWEN38_TOKENIZER_DIR") else {
            return;
        };
        let root = std::env::temp_dir().join(format!("oracle-content-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut checkpoints = vec![];
        for (name, value) in [("first", 1_u8), ("second", 2_u8)] {
            let path = root.join(name);
            std::fs::create_dir(&path).unwrap();
            for file in ["tokenizer.json", "tokenizer_config.json"] {
                std::fs::write(
                    path.join(file),
                    crate::source::read(Path::new(&tokenizer), file, 32 * 1024 * 1024).unwrap(),
                )
                .unwrap();
            }
            let header = br#"{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
            let mut data = (header.len() as u64).to_le_bytes().to_vec();
            data.extend(header);
            data.extend([value; 4]);
            std::fs::write(path.join("weights"), data).unwrap();
            std::fs::write(
                path.join("index"),
                br#"{"weight_map":{"weight":"weights"}}"#,
            )
            .unwrap();
            let files = SourceFiles::capture(
                &path,
                &[
                    ("index", 4096),
                    ("tokenizer.json", 32 * 1024 * 1024),
                    ("tokenizer_config.json", 64 * 1024),
                ],
                &[],
                "index",
            )
            .unwrap();
            checkpoints.push(Checkpoint {
                root: path,
                decode_orders: None,
                precision_provenance: None,
                source: crate::checkpoint::CheckpointSource {
                    files,
                    catalog: crate::catalog::Catalog {
                        geometry: &model_qwen38_contract::geometry::FLASH,
                        tensors: BTreeMap::new(),
                        ranks: 4,
                        precision: None,
                    },
                    spans: BTreeMap::from([(
                        "weight".into(),
                        TensorSpan {
                            file: "weights".into(),
                            offset: 8 + header.len() as u64,
                            bytes: 4,
                        },
                    )]),
                },
            });
        }
        let a = checkpoints[0].content_revision().unwrap();
        let b = checkpoints[1].content_revision().unwrap();
        assert_ne!(a, b);
        assert_ne!(a, crate::REVISION);
        let first = crate::local_frontend::LocalFrontend::load(checkpoints[0].root(), &a).unwrap();
        let second = crate::local_frontend::LocalFrontend::load(checkpoints[1].root(), &b).unwrap();
        let first: serde_json::Value = serde_json::from_slice(&first.component).unwrap();
        let second: serde_json::Value = serde_json::from_slice(&second.component).unwrap();
        assert_eq!(first["tokenizer"], second["tokenizer"]);
        assert_eq!(first["models"][0]["id"], format!("local-{a}"));
        assert_eq!(second["models"][0]["id"], format!("local-{b}"));
        assert_eq!(first["tokenizer"]["repository"], "local");
        std::fs::write(checkpoints[0].root().join("weights"), b"changed source").unwrap();
        assert!(checkpoints[0].content_revision().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fp8_blocks_match_the_kernel_core_decoding() {
        let (rows, k) = (130, 260);
        let codes = (0..rows * k)
            .map(|i| (i * 37 % 251) as u8 & 0x7e)
            .collect::<Vec<_>>();
        let scales = (0..2 * 3)
            .map(|i| f32_to_bf16(0.001 * (i + 1) as f32))
            .collect::<Vec<_>>();
        let expected = fp8_block_checkpoint_weight(&codes, rows, k, &scales)
            .unwrap()
            .decoded();
        assert_eq!(
            dequantize_fp8_blocks(&codes, rows, k, &scales).data,
            expected
        );
    }
}
