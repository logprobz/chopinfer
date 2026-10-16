//! The Sinkhorn mixer's ops: the mix-collapse before each sublayer (the
//! backend runs the split or a tiled form by rows), the mix-collapse fused
//! with the sublayer's input norm that the decode and verify graphs run in
//! place of a main layer's mixer and norm, the expand of a sublayer's
//! output into the streams and the stream collapse before the final norm.
//!
//! Each op is the model's query (`Served`, the mixer of the served
//! entries) at the prefill capacity, which bounds every phase's rows.

use kernel_core::backend::{Backend, Prepared, Refused, View};
use kernel_core::contract::StaticQuery;
use kernel_core::sinkhorn_hyper_connection::{
    MixCollapseBuffers, MixCollapseNormBuffers, SinkhornHyperConnectionQuery as Query,
    StreamExpandBuffers, StreamHeadBuffers,
};
use model_glm5next_contract::contract::{PREFILL_TOKENS, Shape};

use super::{Asks, Served, exact};

/// The Sinkhorn ops of an instance.
pub(crate) struct SinkhornOps {
    pub(crate) mix: Prepared<MixCollapseBuffers<View>>,
    pub(crate) mix_norm: Prepared<MixCollapseNormBuffers<View>>,
    pub(crate) expand: Prepared<StreamExpandBuffers<View>>,
    pub(crate) head: Prepared<StreamHeadBuffers<View>>,
}

/// The one served query `pick` keeps, at the prefill capacity: entries of
/// one mixer differ only in their forms.
fn one(
    served: &Served,
    what: &str,
    pick: impl Fn(&Query) -> Option<Query>,
) -> Result<Query, Refused> {
    let mut found = served.find(|query| match query {
        StaticQuery::SinkhornHyperConnection(query) => pick(query),
        _ => None,
    });
    found.dedup();
    match found[..] {
        [query] => Ok(query),
        _ => Err(Refused::Rule(format!(
            "GLM 5.3 Flash's Sinkhorn {what} has {} served queries",
            found.len()
        ))),
    }
}

impl SinkhornOps {
    pub(super) fn prepare(
        backend: &dyn Backend,
        served: &Served,
        shape: Shape,
        asks: &Asks,
    ) -> Result<SinkhornOps, Refused> {
        let p = shape.profile;
        let capacity = PREFILL_TOKENS;
        let ours = |streams: u32, width: u32| streams == p.hc_streams && width == p.hidden;
        let mix = one(served, "mix-collapse", |query| match *query {
            Query::MixCollapse { mix, .. } if ours(mix.streams, mix.width) => {
                Some(Query::MixCollapse { mix, capacity })
            }
            _ => None,
        })?;
        let mix_norm = one(served, "mix-collapse with its norm", |query| match *query {
            Query::MixCollapseNorm {
                mix,
                rmsnorm_eps_bits,
                ..
            } if ours(mix.streams, mix.width) => Some(Query::MixCollapseNorm {
                mix,
                rmsnorm_eps_bits,
                capacity,
            }),
            _ => None,
        })?;
        let expand = one(served, "expand", |query| match *query {
            Query::Expand { streams, width, .. } if ours(streams, width) => Some(Query::Expand {
                streams,
                width,
                capacity,
            }),
            _ => None,
        })?;
        let head = one(served, "head", |query| match *query {
            Query::Head {
                streams,
                width,
                mode,
                ..
            } if ours(streams, width) => Some(Query::Head {
                streams,
                width,
                mode,
                capacity,
            }),
            _ => None,
        })?;
        // The mixer, expand and head run in every phase; the mix-collapse
        // with its norm only in decode and verify graphs.
        let need = exact::need(capacity, exact::every_phase(exact::verify_rows()));
        let generated = exact::need(capacity, exact::verify(exact::verify_rows()));
        Ok(SinkhornOps {
            mix: asks.ask("sinkhorn.mix", need, |need| {
                backend.sinkhorn_mix_collapse(&mix, need)
            })?,
            mix_norm: asks.ask("sinkhorn.mix_norm", generated, |need| {
                backend.sinkhorn_mix_collapse_norm(&mix_norm, need)
            })?,
            expand: asks.ask("sinkhorn.expand", need, |need| {
                backend.sinkhorn_expand(&expand, need)
            })?,
            head: asks.ask("sinkhorn.head", need, |need| {
                backend.sinkhorn_head(&head, need)
            })?,
        })
    }
}
