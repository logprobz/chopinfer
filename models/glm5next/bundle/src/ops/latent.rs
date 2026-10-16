//! The latent attention's ops: the KV write, the absorbed head GEMMs
//! (query absorb and value up) and the attention over each row's key list,
//! for both latent KV formats.
//!
//! Each op is the model's query at the capacity its served entry compiles
//! (`Served`). The attention ops are the instance's output mode's: whole
//! rows without context-parallel groups, partials to merge across a group's
//! ranks with them. A prompt row runs prefill's op and a generated row
//! decode's or verify's, whatever the step's rows; a drafter pass on the
//! prefill path of at most `drafter_rows` rows runs verify's op
//! (`Lowering::attention_phase`).

use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::contract::{AttentionPhase, StaticQuery};
use kernel_core::latent_attention::{
    HeadGemmBuffers, HeadWeightLayout, LatentAttentionBuffers, LatentAttentionOutput,
    LatentAttentionQuery, LatentKvFormat, LatentKvWriteBuffers,
};
use model_glm5next_contract::contract::{DECODE_BATCH, PREFILL_TOKENS};

use super::{Asks, Plan, Served, exact};
use crate::buffers::{Buffer, widest_step_rows};
use crate::lower::DRAFT_SPLIT_ROWS;

/// The latent KV formats, in the order the ops hold them.
const FORMATS: [LatentKvFormat; 2] = [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16];

/// The latent ops of one KV format.
struct FormatOps {
    kv_write: Prepared<LatentKvWriteBuffers<View>>,
    /// Prefill's, decode's and verify's attention in the instance's output
    /// mode.
    attention: [Prepared<LatentAttentionBuffers<View>>; 3],
    /// The most rows of a prefill-path drafter pass that runs verify's
    /// attention: at most `DRAFT_SPLIT_ROWS`, whose split partials the
    /// scratch workspace holds. None in context-parallel groups.
    drafter_rows: u32,
}

/// The latent ops of an instance.
pub(crate) struct LatentOps {
    formats: [FormatOps; 2],
    /// The query absorb (`KByN`) and the value up (`NByK`).
    head_gemms: [Prepared<HeadGemmBuffers<View>>; 2],
}

/// The one served query `pick` keeps.
fn one<T>(
    served: &Served,
    what: &str,
    pick: impl FnMut(&StaticQuery) -> Option<T>,
) -> Result<T, Refused> {
    let mut found = served.find(pick);
    match found.len() {
        1 => Ok(found.remove(0)),
        count => Err(Refused::Rule(format!(
            "GLM 5.3 Flash's latent {what} has {count} served queries"
        ))),
    }
}

/// The index of `phase`'s attention.
fn phase_index(phase: AttentionPhase) -> usize {
    match phase {
        AttentionPhase::Prefill => 0,
        AttentionPhase::Decode => 1,
        AttentionPhase::Verify => 2,
    }
}

impl LatentOps {
    pub(super) fn prepare(
        backend: &dyn Backend,
        served: &Served,
        plan: Plan,
        asks: &Asks,
    ) -> Result<LatentOps, Refused> {
        let shape = plan.shape;
        let p = shape.profile;
        let output = if plan.context_parallel > 1 {
            LatentAttentionOutput::Partial
        } else {
            LatentAttentionOutput::Normalized
        };
        let attention_query = |kv, phase| {
            one(served, "attention", |query| match *query {
                StaticQuery::LatentAttention(
                    query @ LatentAttentionQuery::Attention {
                        heads,
                        latent,
                        kv: k,
                        phase: at,
                        output: o,
                        ..
                    },
                ) if heads == p.latent_heads
                    && latent == p.kv_rank
                    && k == kv
                    && at == phase
                    && o == output =>
                {
                    Some(query)
                }
                _ => None,
            })
        };
        let workspace = Buffer::Workspace.bytes(shape).map_err(Refused::Rule)?;
        // The KV write and the head GEMMs run in every phase.
        let every_phase = exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows()));
        let format = |kv| -> Result<FormatOps, Refused> {
            let kv_write = one(served, "KV write", |query| match *query {
                StaticQuery::LatentAttention(
                    query @ LatentAttentionQuery::KvWrite { latent, kv: k, .. },
                ) if latent == p.kv_rank && k == kv => Some(query),
                _ => None,
            })?;
            let verify = attention_query(kv, AttentionPhase::Verify)?;
            let drafter_rows = if output == LatentAttentionOutput::Normalized {
                drafter_rows(backend, &verify, workspace)?
            } else {
                0
            };
            let rows = [
                PREFILL_TOKENS,
                DECODE_BATCH,
                widest_step_rows().max(drafter_rows),
            ];
            let attention = |phase| {
                let query = attention_query(kv, phase)?;
                let what = format!("latent.attention {kv:?} {phase:?}");
                let rows = rows[phase_index(phase)];
                let need = exact::need(rows, exact::of_phase(phase, rows));
                asks.ask(what, need, |need| backend.latent_attention(&query, need))
            };
            Ok(FormatOps {
                kv_write: asks.ask(format!("latent.kv_write {kv:?}"), every_phase, |need| {
                    backend.latent_kv_write(&kv_write, need)
                })?,
                attention: [
                    attention(AttentionPhase::Prefill)?,
                    attention(AttentionPhase::Decode)?,
                    attention(AttentionPhase::Verify)?,
                ],
                drafter_rows,
            })
        };
        let head_gemm = |layout| {
            let query = one(served, "head GEMM", |query| match *query {
                StaticQuery::HeadBatchedGemm(query)
                    if query.heads == p.latent_heads && query.layout == layout =>
                {
                    Some(query)
                }
                _ => None,
            })?;
            let what = format!("latent.head_gemm {layout:?}");
            asks.ask(what, every_phase, |need| {
                backend.head_batched_gemm(&query, need)
            })
        };
        Ok(LatentOps {
            formats: [format(FORMATS[0])?, format(FORMATS[1])?],
            head_gemms: [
                head_gemm(HeadWeightLayout::KByN)?,
                head_gemm(HeadWeightLayout::NByK)?,
            ],
        })
    }

    fn format(&self, kv: LatentKvFormat) -> &FormatOps {
        &self.formats[usize::from(kv != FORMATS[0])]
    }

    /// The KV write into a `kv` cache.
    pub(crate) fn kv_write(&self, kv: LatentKvFormat) -> &Prepared<LatentKvWriteBuffers<View>> {
        &self.format(kv).kv_write
    }

    /// The attention of `phase` over a `kv` cache, in the instance's output
    /// mode.
    pub(crate) fn attention(
        &self,
        kv: LatentKvFormat,
        phase: AttentionPhase,
    ) -> &Prepared<LatentAttentionBuffers<View>> {
        &self.format(kv).attention[phase_index(phase)]
    }

    /// The most rows of a prefill-path drafter pass over a `kv` cache that
    /// runs verify's attention.
    pub(crate) fn drafter_rows(&self, kv: LatentKvFormat) -> u32 {
        self.format(kv).drafter_rows
    }

    /// The head-batched GEMM of `layout`.
    pub(crate) fn head_gemm(&self, layout: HeadWeightLayout) -> &Prepared<HeadGemmBuffers<View>> {
        &self.head_gemms[usize::from(layout == HeadWeightLayout::NByK)]
    }
}

/// The most rows, up to `DRAFT_SPLIT_ROWS` and the verify query's
/// capacity, whose split partials `workspace` bytes hold: an attention op
/// sizes its partials by its rows, one row's at a time.
fn drafter_rows(
    backend: &dyn Backend,
    verify: &LatentAttentionQuery,
    workspace: u64,
) -> Result<u32, Refused> {
    let LatentAttentionQuery::Attention { capacity, .. } = *verify else {
        unreachable!("an attention query")
    };
    let row = backend
        .latent_attention(verify, Need::rows(1))?
        .info()
        .bytes_of("workspace")
        .unwrap_or(0);
    let fit = if row == 0 {
        DRAFT_SPLIT_ROWS
    } else {
        u32::try_from(workspace / row).unwrap_or(u32::MAX)
    };
    Ok(fit.min(DRAFT_SPLIT_ROWS).min(capacity))
}
