//! MTP layer 0 reference, sharing the target's SWA and dense MLP arithmetic.

use super::*;
use crate::catalog::DrafterKind;

pub struct MtpOracle {
    enorm: Vec<u16>,
    hnorm: Vec<u16>,
    eh: Vec<u16>,
    final_norm: Vec<u16>,
    layer: LayerOracle,
}

impl MtpOracle {
    pub fn load(root: &Path) -> MimoResult<Self> {
        let catalog = Catalog::load_drafter(root, DrafterKind::Mtp)?;
        let read =
            |suffix: &str| read_tensor(root, &catalog, &format!("model.mtp.layers.0.{suffix}"));
        let fp8 = |stem: &str, rows, cols, chunk_rows| -> MimoResult<Fp8Matrix> {
            Ok(Fp8Matrix {
                rows,
                cols,
                chunk_rows,
                codes: read(&format!("{stem}.weight"))?,
                scales: f32s(&read(&format!("{stem}.weight_scale_inv"))?),
            })
        };
        let enorm = u16s(&read("enorm.weight")?);
        let hnorm = u16s(&read("hnorm.weight")?);
        let eh = u16s(&read("eh_proj.weight")?);
        let final_norm = u16s(&read("final_layernorm.weight")?);
        let input_norm = u16s(&read("input_layernorm.weight")?);
        let post_norm = u16s(&read("pre_mlp_layernorm.weight")?);
        let qkv = fp8(
            "self_attn.qkv_proj",
            model_mimo26_contract::tp::checkpoint_qkv_rows(false) as usize,
            HIDDEN,
            model_mimo26_contract::tp::checkpoint_qkv_chunk_rows(false) as usize,
        )?;
        let o = u16s(&read("self_attn.o_proj.weight")?);
        let sink = u16s(&read("self_attn.attention_sink_bias")?)
            .into_iter()
            .map(|v| f64::from(bf16_to_f32(v)))
            .collect();
        let mlp = FeedForward::Dense {
            gate: fp8(
                "mlp.gate_proj",
                model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                HIDDEN,
                model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
            )?,
            up: fp8(
                "mlp.up_proj",
                model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                HIDDEN,
                model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
            )?,
            down: fp8(
                "mlp.down_proj",
                HIDDEN,
                model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                HIDDEN,
            )?,
        };
        Ok(Self {
            enorm,
            hnorm,
            eh,
            final_norm,
            layer: LayerOracle {
                root: root.to_path_buf(),
                catalog,
                layer: 0,
                global: false,
                input_norm,
                post_norm,
                qkv,
                o,
                sink: Some(sink),
                mlp,
                verifier: None,
            },
        })
    }

    /// Pair each target final-norm row with the following token's embedding.
    /// For chaining, pass the previous MTP output as `hidden`. History holds
    /// only this MTP layer's keys and values, starting at position zero.
    /// `tp_partials` is `LayerOracle::run_with_partials`'s, for the EH
    /// projection too.
    pub fn run(
        &mut self,
        hidden: &[u16],
        embeddings: &[u16],
        history: &mut KvHistory,
        tp_partials: Option<usize>,
    ) -> MimoResult<Vec<u16>> {
        let joined = normalized_pairs(hidden, embeddings, &self.hnorm, &self.enorm)?;
        let projected = row_parallel(
            &Bf16Rows {
                bits: &self.eh,
                cols: HIDDEN * 2,
            },
            &joined,
            tp_partials,
        );
        let input = projected
            .into_iter()
            .flatten()
            .map(|v| f32_to_bf16(v as f32))
            .collect::<Vec<_>>();
        let (output, _) = self
            .layer
            .run_with_partials(&input, history, None, tp_partials)?;
        Ok(HF_NORM.apply_bf16(&output, &self.final_norm, HIDDEN)?)
    }
}

fn normalized_pairs(
    hidden: &[u16],
    embeddings: &[u16],
    hnorm: &[u16],
    enorm: &[u16],
) -> MimoResult<Vec<Vec<f64>>> {
    if hidden.is_empty() || hidden.len() != embeddings.len() || !hidden.len().is_multiple_of(HIDDEN)
    {
        return Err("MTP hidden and embedding rows must have matching nonempty shapes".into());
    }
    let hidden = HF_NORM.apply_bf16(hidden, hnorm, HIDDEN)?;
    let embeddings = HF_NORM.apply_bf16(embeddings, enorm, HIDDEN)?;
    Ok(embeddings
        .chunks_exact(HIDDEN)
        .zip(hidden.chunks_exact(HIDDEN))
        .map(|(e, h)| {
            e.iter()
                .chain(h)
                .map(|&v| f64::from(bf16_to_f32(v)))
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CHOPINFER_MIMO_CHECKPOINT"]
    fn checkpoint_layer_chunk_matches_single_rows() {
        let root = std::env::var_os("CHOPINFER_MIMO_CHECKPOINT").expect("checkpoint path");
        let mut oracle = MtpOracle::load(Path::new(&root)).unwrap();
        let hidden = (0..HIDDEN * 2)
            .map(|i| f32_to_bf16(((i % 127) as f32 - 63.0) / 64.0))
            .collect::<Vec<_>>();
        let embeddings = (0..HIDDEN * 2)
            .map(|i| f32_to_bf16(((i % 79) as f32 - 39.0) / 40.0))
            .collect::<Vec<_>>();
        let mut chunk_history = KvHistory::default();
        let chunk = oracle
            .run(&hidden, &embeddings, &mut chunk_history, Some(4))
            .unwrap();
        let mut row_history = KvHistory::default();
        let mut rows = Vec::new();
        for (h, e) in hidden
            .chunks_exact(HIDDEN)
            .zip(embeddings.chunks_exact(HIDDEN))
        {
            rows.extend(oracle.run(h, e, &mut row_history, Some(4)).unwrap());
        }
        assert_eq!(chunk, rows);
        assert_eq!(chunk_history.keys, row_history.keys);
        assert_eq!(chunk_history.values, row_history.values);
        assert!(chunk.iter().all(|&bits| bf16_to_f32(bits).is_finite()));
        assert!(chunk.iter().any(|&bits| bf16_to_f32(bits).abs() > 0.1));
    }

    #[test]
    fn pairs_normalize_each_input_before_embedding_first_concatenation() {
        let h = vec![f32_to_bf16(-2.0); 2 * HIDDEN];
        let e = vec![f32_to_bf16(3.0); 2 * HIDDEN];
        let hw = vec![f32_to_bf16(0.5); HIDDEN];
        let ew = vec![f32_to_bf16(2.0); HIDDEN];
        let rows = normalized_pairs(&h, &e, &hw, &ew).unwrap();
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(&row[..HIDDEN], vec![2.0; HIDDEN]);
            assert_eq!(&row[HIDDEN..], vec![-0.5; HIDDEN]);
        }
        assert!(normalized_pairs(&h, &e[..HIDDEN], &hw, &ew).is_err());
        assert!(normalized_pairs(&[], &[], &hw, &ew).is_err());
    }
}
