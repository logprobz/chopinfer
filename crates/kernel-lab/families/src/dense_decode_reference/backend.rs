//! Bind frozen symbols and plans using the shared checked buffer ABI.
use super::{mxfp8_fold_decode as split, mxfp8_fold_reuse as reuse, mxfp8_fold_ticket as ticket};
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery, Mxfp8FoldStreamBuffers};
use kernels_sm120::{
    backend::Sm120,
    gemm::{
        dense_decode::{Entry, Knobs},
        mxfp8_fold, mxfp8_fold_packed, mxfp8_fold_stream,
    },
};

#[derive(Clone)]
enum Source {
    Split(split::Entry),
    Ticket(ticket::Entry),
    Reuse(reuse::Entry, u32),
}
impl Source {
    fn entry(&self) -> Entry {
        let (q, paired, symbol, knobs) = match self {
            Self::Split(e) => (e.query, e.paired, e.symbol.clone(), Knobs::SPLIT),
            Self::Ticket(e) => (
                e.query,
                e.paired,
                e.symbol.clone(),
                Knobs {
                    weight_prefetch: e.prefetch_weights,
                    ..Knobs::ticket(e.query.n)
                },
            ),
            Self::Reuse(e, _) => (
                e.ticket.query,
                e.ticket.paired,
                e.symbol(),
                Knobs {
                    tile_n: e.tile_n,
                    row_groups: 2,
                    ..Knobs::ticket(e.ticket.query.n)
                },
            ),
        };
        Entry::new(q, &symbol, paired, knobs).expect("valid reference shape")
    }
    fn specs(&self) -> Vec<kernel_build::EntrySpec> {
        match self {
            Self::Split(e) => e.specs(),
            Self::Ticket(e) => e.specs(),
            Self::Reuse(e, _) => e.specs(),
        }
    }
    fn plans(&self, rows: u32) -> Result<Vec<kernel_core::contract::LaunchPlan>, Refusal> {
        Ok(match self {
            Self::Split(e) => e.plans(rows)?.to_vec(),
            Self::Ticket(e) => e.plans(rows)?.to_vec(),
            Self::Reuse(e, folds) => e.plans(rows, *folds)?.to_vec(),
        })
    }
    fn folds(&self) -> Option<u32> {
        if let Self::Reuse(_, folds) = self {
            Some(*folds)
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct ReferenceBackend {
    base: Sm120,
    sources: Vec<Source>,
    packed: bool,
}
impl From<Sm120> for ReferenceBackend {
    fn from(base: Sm120) -> Self {
        Self {
            base,
            sources: Vec::new(),
            packed: false,
        }
    }
}
impl ReferenceBackend {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_packed_fold_weights(mut self) -> Self {
        self.packed = true;
        self.base = self.base.with_packed_fold_weights();
        self
    }
    pub fn with_mxfp8_fold_ticket(mut self, entry: ticket::Entry) -> Self {
        self.sources.push(Source::Ticket(entry));
        self
    }
    pub fn with_mxfp8_fold_decode(mut self, entry: split::Entry) -> Self {
        self.sources.push(Source::Split(entry));
        self
    }
    pub fn with_mxfp8_fold_reuse(mut self, entry: reuse::Entry, folds: u32) -> Self {
        self.sources.push(Source::Reuse(entry, folds));
        self
    }
    pub fn with_mxfp8_fold(mut self, entry: mxfp8_fold::Entry) -> Self {
        self.base = self.base.with_mxfp8_fold(entry);
        self
    }
    pub fn with_mxfp8_prefill(mut self, entry: mxfp8_fold::Entry) -> Self {
        self.base = self.base.with_mxfp8_prefill(entry);
        self
    }
    pub fn with_mxfp8_prefill_split(mut self, entry: mxfp8_fold::Entry) -> Self {
        self.base = self.base.with_mxfp8_prefill_split(entry);
        self
    }
    pub fn with_mxfp8_fold_stream(mut self, entry: mxfp8_fold_stream::Entry) -> Self {
        self.base = self.base.with_mxfp8_fold_stream(entry);
        self
    }
    fn source(&self, q: &Mxfp8FoldGemmQuery, paired: bool) -> Option<&Source> {
        self.sources
            .iter()
            .filter(|s| {
                let e = s.entry();
                e.query == *q && e.paired == paired
            })
            .min_by_key(|s| match s {
                Source::Reuse(..) => 0,
                Source::Ticket(_) => 1,
                Source::Split(_) => 2,
            })
    }
    fn template(&self, source: &Source) -> Sm120 {
        let backend = Sm120::new().with_dense_decode(source.entry(), source.folds());
        if self.packed {
            backend.with_packed_fold_weights()
        } else {
            backend
        }
    }
    fn wrap<T: 'static>(&self, inner: Prepared<T>, source: &Source) -> Prepared<T> {
        let mut info = inner.info().clone();
        info.symbols = source
            .specs()
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                (
                    if self.packed && i == 1 {
                        mxfp8_fold_packed::symbol(&e.symbol)
                    } else {
                        e.symbol
                    },
                    0,
                )
            })
            .collect();
        Box::new(Op {
            inner,
            info,
            source: source.clone(),
        })
    }
}
impl Backend for ReferenceBackend {
    fn name(&self) -> &str {
        "sm120-lab-reference"
    }
    fn mxfp8_fold_gemm(
        &self,
        q: &Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        let Some(source) = self.source(q, false) else {
            return self.base.mxfp8_fold_gemm(q, need);
        };
        Ok(self.wrap(self.template(source).mxfp8_fold_gemm(q, need)?, source))
    }
    fn mxfp8_fold_gemm_pair(
        &self,
        q: &Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>, Refused> {
        let source = self.source(q, true).ok_or(Refused::Unsupported)?;
        Ok(self.wrap(self.template(source).mxfp8_fold_gemm_pair(q, need)?, source))
    }
    fn mxfp8_prefill_gemm(
        &self,
        q: &Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        self.base.mxfp8_prefill_gemm(q, need)
    }
    fn mxfp8_fold_gemm_stream(
        &self,
        q: &Mxfp8FoldGemmQuery,
        paired: bool,
        need: Need,
    ) -> Result<Prepared<Mxfp8FoldStreamBuffers<View>>, Refused> {
        self.base.mxfp8_fold_gemm_stream(q, paired, need)
    }
}
struct Op<T> {
    inner: Prepared<T>,
    info: OpInfo,
    source: Source,
}
impl<T> PreparedOp<T> for Op<T> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &T,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut calls = self.inner.launches(buffers, workspace, counts)?;
        let plans = self.source.plans(counts.rows)?;
        assert_eq!(calls.len(), plans.len());
        for ((call, (symbol, _)), plan) in calls.iter_mut().zip(&self.info.symbols).zip(plans) {
            assert_eq!(call.plan, plan, "reference launch geometry changed");
            call.symbol.clone_from(symbol);
        }
        Ok(calls)
    }
}
