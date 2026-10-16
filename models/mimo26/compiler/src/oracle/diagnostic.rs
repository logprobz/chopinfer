//! Opt-in offline layer and matched-operand operation evidence.
use super::*;
use runtime_bundle::diagnostic::Inventory;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;

pub const BOUNDARIES: [&str; 10] = [
    "input_norm",
    "qkv",
    "rope_q",
    "rope_k",
    "scaled_v",
    "attention",
    "attention_residual",
    "post_norm",
    "moe_total",
    "final",
];
pub const COSINE: &str = "mimo26.layer-cosine.v1";
pub const NUMERIC: &str = "gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1";
const POLICY: &[u8] = include_bytes!("../../../diagnostic-contracts.json");

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn bits_hash(bits: &[u16]) -> String {
    hash(
        &bits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub fn history_identity(history: &KvHistory) -> Value {
    let digest = |rows: &[Vec<f64>]| {
        hash(
            &rows
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    json!({"rows":history.keys.len(),"keys_sha256":digest(&history.keys),"values_sha256":digest(&history.values)})
}

impl LayerOracle {
    pub fn diagnostic_identity(&self) -> MimoResult<Value> {
        self.verifier
            .as_ref()
            .ok_or("oracle was not loaded with verified provenance")?
            .lock()
            .map_err(|_| "checkpoint verifier poisoned")?
            .diagnostic_identity()
    }

    /// Decode the actual uploaded fragments, then require equality to pinned operands.
    pub fn verify_qkv_operand(
        &self,
        rank: usize,
        packed: &[u8],
        scales: &[u8],
    ) -> MimoResult<Value> {
        use kernel_core::transform::{
            Layout, Packed, Transform,
            fragment::{FragmentWeightFormat, FragmentWeightLayout},
        };
        if rank >= CHECKPOINT_TP {
            return Err("QKV requires ranks 0 through 3".into());
        }
        self.diagnostic_identity()?;
        let rows = self.qkv.chunk_rows;
        let layout = FragmentWeightLayout {
            rows,
            k: HIDDEN,
            format: FragmentWeightFormat::Fp8E4M3,
        };
        let codes = layout.unpack(&Packed {
            bytes: packed.to_vec(),
            layout: Layout::FragmentMajor(layout),
        })?;
        let first = rank * rows * HIDDEN;
        let scale_count = rows.div_ceil(128) * (HIDDEN / 128);
        let decoded_scales = f32s(scales);
        if codes != self.qkv.codes[first..first + rows * HIDDEN]
            || scales.len() != scale_count * 4
            || decoded_scales != self.qkv.scales[rank * scale_count..(rank + 1) * scale_count]
        {
            return Err(
                "loaded QKV weight or scale differs from verified checkpoint operand".into(),
            );
        }
        let mut digest = Sha256::new();
        for row in 0..rows {
            for k in 0..HIDDEN {
                let value = SmallFloat::E4M3.decode_f64(codes[row * HIDDEN + k])
                    * f64::from(decoded_scales[(row / 128) * (HIDDEN / 128) + k / 128]);
                if !value.is_finite() {
                    return Err("nonfinite decoded QKV weight".into());
                }
                digest.update(value.to_le_bytes());
            }
        }
        Ok(
            json!({"shape":[rows,HIDDEN],"dtype":"F64","sha256":format!("{:x}",digest.finalize()),
            "recipe":"mimo-tp4-fragment-e4m3-f32-v1"}),
        )
    }

    pub fn qkv_rank_numeric(
        &self,
        input: &[f64],
        rank: usize,
    ) -> MimoResult<kernel_core::oracle::OracleOutput> {
        if rank >= CHECKPOINT_TP
            || input.is_empty()
            || !input.len().is_multiple_of(HIDDEN)
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("QKV requires finite complete normalized rows and a TP4 rank".into());
        }
        let mut values = Vec::new();
        let mut sums = Vec::new();
        for x in input.chunks_exact(HIDDEN) {
            values.extend(self.qkv_rank_direct(x, rank)?);
            for row in rank * self.qkv.chunk_rows..(rank + 1) * self.qkv.chunk_rows {
                let scale_row = self.qkv.grid().scale_row(row)?;
                let s = x
                    .iter()
                    .enumerate()
                    .map(|(k, value)| {
                        (value
                            * SmallFloat::E4M3.decode_f64(self.qkv.codes[row * HIDDEN + k])
                            * f64::from(self.qkv.scales[scale_row * (HIDDEN / 128) + k / 128]))
                        .abs()
                    })
                    .sum::<f64>();
                if !s.is_finite() {
                    return Err("nonfinite QKV bound operand".into());
                }
                sums.push(s);
            }
        }
        Ok(kernel_core::oracle::OracleOutput::new(
            values, sums, HIDDEN,
        )?)
    }
}

pub fn boundary(trace: &LayerTrace, name: &str, rank: usize, width: usize) -> MimoResult<Vec<f64>> {
    let (value, shard) = match name {
        "input_norm" => (&trace.input_norm, false),
        "qkv" => (&trace.qkv, true),
        "rope_q" => (&trace.q, true),
        "rope_k" => (&trace.k, true),
        "scaled_v" => (&trace.v, true),
        "attention" => (&trace.attention, true),
        "attention_residual" => (&trace.attention_residual, false),
        "post_norm" => (&trace.post_norm, false),
        "moe_total" => (&trace.moe, false),
        "final" => (&trace.final_output, false),
        _ => return Err("unknown layer boundary".into()),
    };
    let start = if shard { rank * width } else { 0 };
    Ok(value
        .get(start..start + width)
        .ok_or("boundary shape mismatch")?
        .to_vec())
}

pub fn widths(global: bool) -> [usize; 10] {
    use model_mimo26_contract::tp::{HIDDEN, QK_DIM, V_DIM};
    let tp = model_mimo26_contract::Tp::TP4;
    [
        HIDDEN as usize,
        tp.qkv_rows(global) as usize,
        (tp.q_heads() * QK_DIM) as usize,
        (tp.kv_heads(global) * QK_DIM) as usize,
        (tp.kv_heads(global) * V_DIM) as usize,
        tp.o_proj_k() as usize,
        HIDDEN as usize,
        HIDDEN as usize,
        HIDDEN as usize,
        HIDDEN as usize,
    ]
}

/// Both arms declare identical scope before launch. Sealing is an offline step.
pub struct Capture {
    root: PathBuf,
    inventory: Option<Inventory>,
    bytes: u64,
}
impl Capture {
    pub fn begin(root: &Path, scope: Value, execution: &str) -> MimoResult<Self> {
        std::fs::create_dir(root)?;
        let binary = model_compiler_sdk::source_file::SourceFile::open(&std::env::current_exe()?)?;
        let description = json!({"capture_id":format!("{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()),
            "scope":scope,"limits":{"tensor_bytes":2_u64<<30,"metadata_bytes":16_u64<<20},
            "producer":{"implementation":"mimo-layer-harness","implementation_sha256":runtime_weights::hash_range(&binary,0,binary.metadata()?.len())?,
                "source_sha256":hash(include_bytes!("../oracle.rs")),"execution":execution},
            "limitations":["Seeded hidden-input layer chain; no full-model token claim.","Router near-tie substitutions are reported, not independent router correctness."],
            "reproduction":["cargo","test","-p","kernel-lab","--locked","--offline","--test","mimo_layers","real_weight_layers","--","--exact"]});
        Ok(Self {
            root: root.into(),
            inventory: Some(Inventory::begin(root, description)?),
            bytes: 0,
        })
    }
    pub fn record(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: &str,
        raw: &[u8],
    ) -> MimoResult<()> {
        self.bytes = self
            .bytes
            .checked_add(raw.len() as u64)
            .ok_or("capture size overflow")?;
        if self.bytes > 2_u64 << 30 {
            return Err("capture byte budget exhausted".into());
        }
        let file = format!("{name}.bin");
        let mut output = std::fs::File::create_new(self.root.join(&file))?;
        output.write_all(raw)?;
        output.sync_all()?;
        self.inventory.as_mut().ok_or("capture already completed")?.record(json!({"name":name,"shape":shape,"dtype":dtype,"file":file,"offset":0,"bytes":raw.len()}))?;
        Ok(())
    }
    pub fn f64(&mut self, name: &str, rows: usize, values: &[f64]) -> MimoResult<()> {
        if rows == 0 || !values.len().is_multiple_of(rows) {
            return Err("incomplete tensor rows".into());
        }
        self.record(
            name,
            &[rows, values.len() / rows],
            "F64",
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    pub fn bf16(&mut self, name: &str, rows: usize, values: &[u16]) -> MimoResult<()> {
        if rows == 0 || !values.len().is_multiple_of(rows) {
            return Err("incomplete tensor rows".into());
        }
        self.record(
            name,
            &[rows, values.len() / rows],
            "BF16",
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    pub fn finish(mut self, identity: Value) -> MimoResult<()> {
        Ok(self
            .inventory
            .take()
            .ok_or("capture already completed")?
            .finish(&self.root, identity)?)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.inventory.is_some() {
            let _ = runtime_bundle::diagnostic::write_new(
                &self.root.join("aborted.json"),
                &json!({"completion":"incomplete","reason":"producer stopped before completion; retain the original scope"}),
            );
        }
    }
}

pub fn scope(
    layer: usize,
    global: bool,
    input: &[u16],
    position: usize,
    lineage: &[String],
    isolated: bool,
) -> Value {
    let rows = input.len() / HIDDEN;
    let mut probes = Vec::new();
    for (i, name) in BOUNDARIES.iter().enumerate() {
        for rank in 0..4 {
            if isolated && *name != "qkv" {
                continue;
            }
            let mut probe = json!({"name":format!("layer{layer:02}.{name}.r{rank}"),"shape":[rows,widths(global)[i]],"rank":rank});
            if isolated {
                probe["dependencies"] = json!([
                    format!("layer{layer:02}.input_norm.r{rank}"),
                    format!("layer{layer:02}.qkv.weight.r{rank}")
                ]);
            }
            probes.push(probe);
        }
    }
    json!({"model":"mimo26","evidence":if isolated {"op-isolated"} else {"chained"},
        "contract":if isolated {NUMERIC} else {COSINE},"policy_sha256":hash(POLICY),"topology":{"ranks":4,"fold":1},
        "settings":{"hidden_rounding":"bf16","fp8_kv":false,"fp8_linear":false,"oracle_tp_partials":null,
            "router":"checked-near-ties","history":"sequential-calls-from-fresh-zero"},
        "inputs":{"kind":"hidden-bf16","hidden_sha256":bits_hash(input),
            "state":{"kind":"fresh-zero-sequence","prior_inputs":lineage},
            "rows":(0..rows).map(|row|json!({"row":row,"request":0,"position":position+row,
                "phase":if position==0 {"prefill"} else {"decode"},"padding":false})).collect::<Vec<_>>()},"probes":probes})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rust_layer_exports_compare_with_existing_cosine_contract() {
        let root = std::env::temp_dir().join(format!("mimo-compare-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let pins: Value =
            serde_json::from_str(include_str!("../../../checkpoint-files.json")).unwrap();
        let checkpoint = crate::checkpoint::default_identity();
        let identity = json!({"checkpoint":{"repository":checkpoint.repository,"revision":checkpoint.revision,
            "source_lock_sha256":hash(&serde_json::to_vec(&pins["mopd"]).unwrap())},"prepared_weights":{"kind":"none"},"fixture":true});
        let input = vec![0x3f80; HIDDEN];
        let mut requested = scope(1, false, &input, 0, &[], false);
        requested["probes"] = json!([{"name":"layer01.input_norm.r0","shape":[1,HIDDEN],"rank":0},
            {"name":"layer01.final.r0","shape":[1,HIDDEN],"rank":0}]);
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap();
        let cli = repo.join("tools/dev/diagnostics.py");
        for (name, bad) in [("reference", false), ("matching", false), ("changed", true)] {
            let dir = root.join(name);
            let mut capture = Capture::begin(&dir, requested.clone(), "cpu-fixture").unwrap();
            capture.bf16("input.hidden", 1, &input).unwrap();
            if name == "reference" {
                capture
                    .f64("layer01.input_norm.r0", 1, &vec![1.; HIDDEN])
                    .unwrap();
                capture
                    .f64("layer01.final.r0", 1, &vec![1.; HIDDEN])
                    .unwrap();
            } else {
                capture.bf16("layer01.input_norm.r0", 1, &input).unwrap();
                capture
                    .bf16(
                        "layer01.final.r0",
                        1,
                        &vec![if bad { 0xbf80 } else { 0x3f80 }; HIDDEN],
                    )
                    .unwrap();
            }
            capture.finish(identity.clone()).unwrap();
            let output = std::process::Command::new("python3")
                .arg("-B")
                .arg(&cli)
                .args(["seal", "--mode", "oracle", "--candidate"])
                .arg(&dir)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        for (name, code) in [("matching", 0), ("changed", 1)] {
            let output = std::process::Command::new("python3")
                .arg("-B")
                .arg(&cli)
                .args(["compare", "--mode", "oracle", "--reference"])
                .arg(root.join("reference"))
                .arg("--candidate")
                .arg(root.join(name))
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(code),
                "{} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let value: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["coverage"]["gated"], 2);
            if code == 1 {
                assert_eq!(value["first"]["name"], "layer01.final.r0");
                assert_eq!(value["first"]["bound"], 0.9999);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn history_and_export_keep_hidden_inputs_and_order() {
        let mut history = KvHistory::default();
        let empty = history_identity(&history);
        history.keys.push(vec![1., 2.]);
        history.values.push(vec![3., 4.]);
        assert_ne!(history_identity(&history), empty);
        let root = std::env::temp_dir().join(format!("mimo-attachment-{}", std::process::id()));
        let input = vec![0x3f80; HIDDEN];
        let mut scope = scope(1, false, &input, 0, &[], false);
        scope["probes"] = json!([{"name":"layer01.input_norm.r0","shape":[1,HIDDEN]},
            {"name":"layer01.final.r0","shape":[1,HIDDEN]}]);
        let mut capture = Capture::begin(&root, scope, "cpu-fixture").unwrap();
        capture.bf16("layer01.input_norm.r0", 1, &input).unwrap();
        assert!(capture.finish(json!({"fixture":true})).is_err());
        let value: Value =
            serde_json::from_slice(&std::fs::read(root.join("producer.json")).unwrap()).unwrap();
        assert_eq!(value["completion"], "incomplete");
        assert_eq!(value["missing"], json!(["layer01.final.r0"]));
        assert_eq!(
            value["description"]["scope"]["inputs"]["kind"],
            "hidden-bf16"
        );
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap();
        let output = std::process::Command::new("python3")
            .arg("-B")
            .arg(repo.join("tools/dev/diagnostics.py"))
            .args(["seal", "--mode", "oracle", "--candidate"])
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let output = std::process::Command::new("python3")
            .arg("-B")
            .arg(repo.join("tools/dev/diagnostics.py"))
            .args(["compare", "--mode", "oracle", "--reference"])
            .arg(&root)
            .arg("--candidate")
            .arg(&root)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3));
        std::fs::remove_dir_all(root).unwrap();
    }
}
