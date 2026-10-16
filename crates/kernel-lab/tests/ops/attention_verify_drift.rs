//! Row 7 of an eight-row verify item against the decode row of the same
//! token over the same keys and values. The decode fixture's query and
//! caches are copied into the verify fixture: paged layers share the whole
//! cache; ring layers take the decode ring and the eight newest ring rows
//! as the verify item's new K and V (the slots they overwrite lie outside
//! row 7's window).

use super::{
    Case, K, K_NEW, OUT, Q, V, V_NEW, bytes, functions, module, named, prepare, put_bytes,
};
use kernel_core::codec::bf16_to_f32;
use kernel_lab::gpu_test;
use kernels_sm120::attention::CacheLayout;
use serde_json::json;
use std::collections::BTreeMap;

use crate::verify_drift_stats as stats;

#[test]
fn attention_verify_drift_row7() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        for (decode_name, verify_name) in [
            ("mimo_ga_bf16_decode", "mimo_ga_bf16_verify"),
            ("mimo_ga_fp8_decode", "mimo_ga_fp8_verify"),
            ("mimo_swa_decode", "mimo_swa_verify"),
        ] {
            let decode_variant = named(decode_name)?;
            let verify_variant = named(verify_name)?;
            let cubin = module(session, decode_variant.profile)?;
            for prefix in [0_usize, 120, 1017, 4095] {
                let decode_case = Case {
                    variant: decode_variant,
                    batch: 1,
                    rows: 1,
                    prefix: prefix + 7,
                    mixed: false,
                };
                let verify_case = Case {
                    variant: verify_variant,
                    batch: 1,
                    rows: 8,
                    prefix,
                    mixed: false,
                };
                let decode = prepare(session, decode_case)?;
                let mut verify = prepare(session, verify_case)?;
                let query = decode_variant.query;
                let (qh, kh) = (query.q_heads as usize, query.kv_heads as usize);
                let (qd, vd) = (query.qk_dim as usize, query.v_dim as usize);
                let k_cache = bytes(session, &decode.buffers[&K])?;
                let v_cache = bytes(session, &decode.buffers[&V])?;
                if k_cache.len() != verify.buffers[&K].byte_len()
                    || v_cache.len() != verify.buffers[&V].byte_len()
                {
                    return Err(format!("{decode_name} p{prefix}: cache sizes differ"));
                }
                if decode_variant.layout == CacheLayout::Ring {
                    let ring = query.page_size as usize;
                    let (k_row, v_row) = (kh * qd * 2, kh * vd * 2);
                    let mut k_new = Vec::new();
                    let mut v_new = Vec::new();
                    for position in prefix..prefix + 8 {
                        let slot = position % ring;
                        k_new.extend_from_slice(&k_cache[slot * k_row..(slot + 1) * k_row]);
                        v_new.extend_from_slice(&v_cache[slot * v_row..(slot + 1) * v_row]);
                    }
                    verify.buffers.insert(K_NEW, put_bytes(session, &k_new)?);
                    verify.buffers.insert(V_NEW, put_bytes(session, &v_new)?);
                }
                verify.buffers.insert(K, put_bytes(session, &k_cache)?);
                verify.buffers.insert(V, put_bytes(session, &v_cache)?);
                let q_row = bytes(session, &decode.buffers[&Q])?;
                let mut q_all = bytes(session, &verify.buffers[&Q])?;
                q_all[7 * qh * qd * 2..8 * qh * qd * 2].copy_from_slice(&q_row[..qh * qd * 2]);
                verify.buffers.insert(Q, put_bytes(session, &q_all)?);

                let (partial, merge) = functions(&cubin, decode_variant, &decode.planned)?;
                decode.launch(session, &partial, &merge)?;
                let (partial, merge) = functions(&cubin, verify_variant, &verify.planned)?;
                verify.launch(session, &partial, &merge)?;
                let width = qh * vd;
                let decode_row = session
                    .download_bf16(&decode.buffers[&OUT], width)?
                    .into_iter()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>();
                let verify_row = session.download_bf16(&verify.buffers[&OUT], 8 * width)?
                    [7 * width..]
                    .iter()
                    .map(|&bits| bf16_to_f32(bits))
                    .collect::<Vec<_>>();
                let reference = &decode.reference.values[..width];
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("verify_drift_attention"));
                receipt.insert("decode".into(), json!(decode_case.label()));
                receipt.insert("verify".into(), json!(verify_case.label()));
                receipt.insert(
                    "decode_partial".into(),
                    json!(decode.planned.partial_symbol),
                );
                receipt.insert(
                    "verify_partial".into(),
                    json!(verify.planned.partial_symbol),
                );
                receipt.insert(
                    "decode_split_capacity".into(),
                    json!(decode.planned.split_capacity),
                );
                receipt.insert(
                    "verify_split_capacity".into(),
                    json!(verify.planned.split_capacity),
                );
                receipt.insert(
                    "verify_versus_decode".into(),
                    stats::paths(&decode_row, &verify_row, true),
                );
                receipt.insert(
                    "decode_against_f64".into(),
                    stats::against(&decode_row, reference, true),
                );
                receipt.insert(
                    "verify_against_f64".into(),
                    stats::against(&verify_row, reference, true),
                );
                eprintln!(
                    "{}",
                    serde_json::to_string(&receipt).map_err(|e| e.to_string())?
                );
                session.receipt("verify_drift", receipt)?;
            }
        }
        Ok(())
    })
    .expect("attention verify drift");
}
