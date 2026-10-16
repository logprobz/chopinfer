//! Single-rank real-weight global attention of a long prefill chunk, as the
//! shared layer fixture plans it, against the f64 attention oracle.

use super::*;
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Argument, AttentionPhase, DType};
use kernel_core::oracle::{self, PagedAttention};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernels_sm120::attention::CacheLayout;

/// Oracle rows: both ends of the chunk, the rows around position 256, from
/// which a query tile's keys split into segments that the merge combines,
/// and rows spread over the rest.
fn sampled(rows: usize) -> Vec<usize> {
    let mut tokens: Vec<usize> = [0, 1, 255, 256, 257, 511, 512, rows - 1]
        .into_iter()
        .chain((1..8).map(|i| i * rows / 8 + 3))
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn buffer_id(argument: Option<&Argument>, what: &str) -> LabResult<u64> {
    match argument {
        Some(Argument::DeviceBuffer(id)) => Ok(*id),
        _ => Err(format!("global attention {what} is not a device buffer")),
    }
}

fn wide(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// Every row of the chunk's global attention output is written and matches
/// the oracle over the keys and queries the fixture's own RoPE stage wrote.
#[test]
fn prefill_global_attention_writes_every_row() {
    if std::env::var("CHOPINFER_PREFILL_GLOBAL_ATTENTION").as_deref() != Ok("1") {
        kernel_lab::skip(
            "global prefill attention: set CHOPINFER_PREFILL_GLOBAL_ATTENTION=1 and the checkpoint",
        )
        .unwrap();
        return;
    }
    let checkpoint = super::local_context::checkpoint();
    let ordinals = kernel_lab::selected_ordinals()
        .expect("GPU selection")
        .expect("one assigned GPU");
    assert_eq!(ordinals.len(), 1, "global prefill attention uses one GPU");
    let images = execution::images(Path::new(&checkpoint)).expect("TP4 layer images");
    kernel_lab::gpu_test(8 << 30, |session| {
        // Layer 5 is global; the capacity is the local context tests'.
        let rank = execution::Rank::build(session, &images, 1, 5, 1, true, 8192, false)?;
        // The layer's attention partial and merge, and the variant they
        // were planned from.
        let partial = rank
            .prefill
            .iter()
            .position(|s| s.name.starts_with("chop_attn_") && s.name.ends_with("_direct_v1"))
            .ok_or("missing attention partial")?;
        let entry = rank.prefill[partial]
            .name
            .strip_prefix("chop_attn_")
            .and_then(|name| name.strip_suffix("_direct_v1"))
            .ok_or("unexpected attention partial symbol")?;
        let merge = rank
            .prefill
            .iter()
            .position(|s| s.name == format!("chop_attn_{entry}_merge_v1"))
            .ok_or("missing attention merge")?;
        let query = kernels_sm120::attention::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|v| v.name == entry)
            .ok_or_else(|| format!("missing attention variant {entry}"))?
            .query;
        if query.layout != CacheLayout::Paged || query.phase != AttentionPhase::Prefill {
            return Err(format!("{entry} is not a paged prefill entry"));
        }
        let q_width = (query.q_heads * query.qk_dim) as usize;
        let k_width = (query.kv_heads * query.qk_dim) as usize;
        let v_width = (query.kv_heads * query.v_dim) as usize;
        let out_width = (query.q_heads * query.v_dim) as usize;
        let page = query.page_size as usize;
        let output = buffer_id(rank.prefill[partial].args.0.last(), "output")?;
        let workspace = buffer_id(rank.prefill[merge].args.0.first(), "workspace")?;
        let (output, workspace) = (&rank.buffers[&output], &rank.buffers[&workspace]);
        for rows in [1024, 2048] {
            let label = format!("layer 5 rank 1 {entry} rows {rows}");
            let seed = 0x6f18_0000 ^ (5 << 8);
            rank.set_inputs(session, &execution::seeded_hidden(rows, seed), 0, true)?;
            poison::poison(session, &[output, workspace])?;
            for index in 0..=merge {
                if !rank.prefill[index].name.starts_with("chop_comm_") {
                    execution::launch_local_prefill(session, &rank, index, rows, 188, false)?;
                }
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let written = poison::read(session, output)?;
            let coverage = poison::check(
                &format!("{label} global attention output"),
                &written,
                &WriteMap::of(DType::Bf16).prefix(rows * out_width)?,
                Fill::Pattern(&[POISON]),
            );
            let q = execution::intermediate(session, &rank, "rope_q", rows)?;
            let keys = wide(&execution::intermediate(session, &rank, "rope_k", rows)?);
            let values = wide(&execution::intermediate(session, &rank, "scaled_v", rows)?);
            if keys.len() != rows * k_width || values.len() != rows * v_width || rows % page != 0 {
                return Err(format!("{label}: keys and values do not fill whole pages"));
            }
            let pages: Vec<usize> = (0..rows / page).collect();
            let tokens = sampled(rows);
            let mut reference = oracle::OracleOutput {
                values: Vec::new(),
                sum_abs_products: Vec::new(),
                reduction_lengths: Vec::new(),
            };
            // Each sampled token is a one-row oracle item over the chunk's
            // pages.
            for chunk in tokens.chunks(query.max_batch_capacity as usize) {
                let q_rows: Vec<f64> = chunk
                    .iter()
                    .flat_map(|&t| wide(&q[t * q_width..(t + 1) * q_width]))
                    .collect();
                let part = oracle::attention(
                    &query,
                    &PagedAttention {
                        q: &q_rows,
                        keys: &keys,
                        values: &values,
                        page_tables: &vec![pages.clone(); chunk.len()],
                        query_lengths: &vec![1; chunk.len()],
                        query_starts: chunk,
                        key_lengths: &chunk.iter().map(|t| t + 1).collect::<Vec<_>>(),
                        sink_logits: None,
                    },
                )
                .map_err(|e| e.to_string())?;
                reference.values.extend(part.values);
                reference.sum_abs_products.extend(part.sum_abs_products);
                reference.reduction_lengths.extend(part.reduction_lengths);
            }
            let actual: Vec<f32> = tokens
                .iter()
                .flat_map(|&t| written[t * out_width * 2..(t + 1) * out_width * 2].chunks_exact(2))
                .map(|pair| bf16_to_f32(u16::from_ne_bytes([pair[0], pair[1]])))
                .collect();
            let mut receipt = BTreeMap::from([
                ("case".into(), json!(label)),
                ("planned_query_capacity".into(), json!(8192)),
                ("sampled_tokens".into(), json!(tokens)),
                ("all_rows_written".into(), json!(coverage.is_ok())),
            ]);
            let values = kernel_lab::check_output(&actual, &reference, DType::Bf16, &mut receipt);
            session.receipt("prefill_global_attention", receipt)?;
            coverage?;
            values.map_err(|e| format!("{label}: {e}"))?;
        }
        Ok(())
    })
    .expect("global prefill attention");
}
