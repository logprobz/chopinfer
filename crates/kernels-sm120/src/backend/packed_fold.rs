//! Packed fold weights (`Sm120::with_packed_fold_weights`): every MXFP8 fold
//! op the backend prepares reads E4M3 codes stored as
//! `mxfp8-fold-packed-8x128-v1`.
//!
//! Each registered entry's weight-reading launch takes its packed entry
//! (`gemm::mxfp8_fold_packed`), with the packed weight map for a TMA
//! prefill launch; launches that read no codes (admission, partial sums)
//! keep their symbols. The op's info names the packed symbols and declares
//! the packed layout for every weight role. Stream and direct folds already
//! read packed codes; they bind the planned view as their packed bank, so
//! their create-time prepare, which packs row-major codes, is refused. A
//! launch of an entry with no packed form is refused, never run on packed
//! codes.

use super::Sm120;
use crate::gemm::{mxfp8_fold, mxfp8_fold_packed as packed, mxfp8_prefill, mxfp8_prefill_split};
use kernel_build::EntrySpec;
use kernel_core::backend::{Counts, DeferredBf16Round, Launch, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::Refusal;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug)]
enum Conversion {
    Rename,
    /// A TMA prefill launch over the entry's row-major weight map, which
    /// takes the packed map and the named packed entry.
    Map(mxfp8_fold::Entry, String),
    /// An entry with no packed form.
    Refused(&'static str),
}

/// The packed form of the weight-reading symbols an op launches, from the
/// entries registered in a backend.
#[derive(Clone, Debug)]
pub(super) struct Conversions(Arc<BTreeMap<String, Conversion>>);

impl Conversions {
    /// Conversions of the registered entries among `symbols`; only those
    /// entries build their packed specs.
    pub(super) fn new(b: &Sm120, symbols: &BTreeSet<&str>) -> Self {
        let mut map = BTreeMap::new();
        let mut twin = |spec: &dyn Fn() -> Result<EntrySpec, Refusal>, row_major: String| {
            if !symbols.contains(row_major.as_str()) {
                return;
            }
            let conversion = match spec() {
                Ok(spec) if spec.symbol == packed::symbol(&row_major) => Conversion::Rename,
                Ok(_) => Conversion::Refused("a packed entry is not named for its twin"),
                Err(refusal) => Conversion::Refused(refusal.0),
            };
            map.insert(row_major, conversion);
        };
        for e in &b.mxfp8_folds {
            twin(&|| e.packed_spec(), e.symbol.clone());
        }
        for e in &b.mxfp8_fold_tickets {
            twin(&|| e.packed_spec(), e.symbols()[1].clone());
        }
        for e in &b.mxfp8_fold_decodes {
            twin(&|| e.packed_spec(), e.symbols()[1].clone());
        }
        for (e, _) in &b.mxfp8_fold_reuse {
            twin(&|| e.packed_spec(), e.symbol());
        }
        for (e, _) in &b.mxfp8_prefill_compacts {
            for tile in [16, 32, 64] {
                twin(&|| e.packed_spec(tile), e.symbol(tile));
            }
        }
        for (e, _) in &b.mxfp8_prefill_matrices {
            twin(&|| e.packed_spec(), e.symbol(e.tile_rows));
        }
        let mut map_entry = |e: &mxfp8_fold::Entry, row_major: String| {
            if !symbols.contains(row_major.as_str()) {
                return;
            }
            let conversion = match mxfp8_prefill::check_packed(e) {
                Ok(()) => Conversion::Map(e.clone(), packed::symbol(&row_major)),
                Err(refusal) => Conversion::Refused(refusal.0),
            };
            map.insert(row_major, conversion);
        };
        for e in &b.mxfp8_prefills {
            map_entry(e, mxfp8_prefill::symbol(e));
        }
        for (e, _) in &b.mxfp8_prefill_splits {
            map_entry(e, mxfp8_prefill::symbol(e));
            map_entry(e, mxfp8_prefill_split::symbol(e));
        }
        for (e, _) in &b.mxfp8_prefill_tma_scales {
            map_entry(&e.base, mxfp8_prefill::symbol(&e.base));
            map_entry(&e.base, e.symbol());
        }
        for (e, _) in &b.mxfp8_prefill_splits {
            let tail = mxfp8_prefill_split::tail_symbol(e);
            if !symbols.contains(tail.as_str()) {
                continue;
            }
            let conversion = match mxfp8_prefill_split::packed_tail_spec(e) {
                Ok(spec) if spec.symbol == packed::symbol(&tail) => Conversion::Rename,
                Ok(_) => Conversion::Refused("a packed entry is not named for its twin"),
                Err(refusal) => Conversion::Refused(refusal.0),
            };
            map.insert(tail, conversion);
        }
        for (e, _) in &b.mxfp8_prefill_tma_matrices {
            if symbols.contains(e.symbol().as_str()) {
                map.insert(
                    e.symbol(),
                    Conversion::Refused("the TMA matrix prefill has no packed entry"),
                );
            }
        }
        Self(Arc::new(map))
    }

    /// Conversions of the registered entries `info` names.
    pub(super) fn of(b: &Sm120, info: &OpInfo) -> Self {
        Self::new(b, &info.symbols.iter().map(|(s, _)| s.as_str()).collect())
    }

    fn symbol(&self, symbol: &str) -> Result<String, Refusal> {
        match self.0.get(symbol) {
            None => Ok(symbol.to_owned()),
            Some(Conversion::Rename) => Ok(packed::symbol(symbol)),
            Some(Conversion::Map(_, packed)) => Ok(packed.clone()),
            Some(Conversion::Refused(why)) => Err(Refusal(why)),
        }
    }

    fn launches(&self, mut launches: Vec<Launch>) -> Result<Vec<Launch>, Refusal> {
        for launch in &mut launches {
            match self.0.get(&launch.symbol) {
                None => {}
                Some(Conversion::Rename) => launch.symbol = packed::symbol(&launch.symbol),
                Some(Conversion::Map(entry, symbol)) => {
                    mxfp8_prefill::move_to_packed(entry, launch, symbol.clone())?
                }
                Some(Conversion::Refused(why)) => return Err(Refusal(why)),
            }
        }
        Ok(launches)
    }

    /// `info` with packed symbols and the packed layout on every weight
    /// role, declared or sized.
    fn info(&self, mut info: OpInfo) -> Result<OpInfo, Refusal> {
        for (symbol, _) in &mut info.symbols {
            *symbol = self.symbol(symbol)?;
        }
        for (role, layout) in &mut info.weight_layouts {
            if weight_role(role) {
                *layout = packed::layout();
            }
        }
        for (role, _) in &info.bytes {
            if weight_role(role) && info.layout_of(role).is_none() {
                info.weight_layouts.push((role.clone(), packed::layout()));
            }
        }
        if !info
            .weight_layouts
            .iter()
            .any(|(role, _)| weight_role(role))
        {
            info.weight_layouts
                .push(("weights".into(), packed::layout()));
        }
        Ok(info)
    }
}

/// A role binding a fold's weight codes: `weights`, or a bank's
/// (`first.weights`, `0.weights`).
fn weight_role(role: &str) -> bool {
    role == "weights" || role.ends_with(".weights")
}

/// `op`, a fold op a model prepares outside the backend, reading packed fold
/// weights: its launches of `readers` (row-major weight-reading symbols)
/// take their packed entries, as the backend's own fold ops do.
pub fn packed_op<B: 'static>(
    op: Prepared<B>,
    readers: impl IntoIterator<Item = String>,
) -> Result<Prepared<B>, Refusal> {
    let map = readers
        .into_iter()
        .map(|symbol| (symbol, Conversion::Rename))
        .collect();
    wrap(op, &Conversions(Arc::new(map)))
}

struct Op<B> {
    inner: Prepared<B>,
    info: OpInfo,
    conversions: Conversions,
}

/// `op` reading packed fold weights.
pub(super) fn wrap<B: 'static>(
    op: Prepared<B>,
    conversions: &Conversions,
) -> Result<Prepared<B>, Refusal> {
    let info = conversions.info(op.info().clone())?;
    Ok(Box::new(Op {
        inner: op,
        info,
        conversions: conversions.clone(),
    }))
}

impl<B> PreparedOp<B> for Op<B> {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        let mut original = launch.clone();
        if let Some(symbol) = original.symbol.strip_suffix("_packed_v1") {
            original.symbol = symbol.to_owned();
        }
        let mut tag = self.inner.route(counts, &original)?;
        tag.symbol = launch.symbol.clone();
        if original.symbol != launch.symbol {
            // Preserve the selected branch while translating its CUDA symbol.
            tag.entry = tag.entry.strip_suffix(&original.symbol).map_or_else(
                || format!("{}/packed", tag.entry),
                |prefix| format!("{prefix}{}", launch.symbol),
            );
        }
        Some(tag)
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn defer_bf16_round(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Option<DeferredBf16Round>, Refusal> {
        let Some(mut round) = self.inner.defer_bf16_round(buffers, workspace, counts)? else {
            return Ok(None);
        };
        round.launches = self.conversions.launches(round.launches)?;
        if round
            .launches
            .iter()
            .all(|l| self.info.symbols.iter().any(|(s, _)| *s == l.symbol))
        {
            Ok(Some(round))
        } else {
            Err(Refusal(
                "a fold launch names a symbol its op does not declare",
            ))
        }
    }

    fn initialize(&self, buffers: &B) -> Result<Vec<Launch>, Refusal> {
        if self.inner.initialize(buffers)?.is_empty() {
            Ok(Vec::new())
        } else {
            Err(Refusal(
                "packed fold weights are planned packed; zero the fold state instead of preparing a copy",
            ))
        }
    }

    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let launches = self
            .conversions
            .launches(self.inner.launches(buffers, workspace, counts)?)?;
        // A symbol the op's info does not name was never converted.
        if launches
            .iter()
            .all(|l| self.info.symbols.iter().any(|(s, _)| *s == l.symbol))
        {
            Ok(launches)
        } else {
            Err(Refusal(
                "a fold launch names a symbol its op does not declare",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{mxfp8_fold_stream, mxfp8_fold_ticket};
    use kernel_core::backend::{Backend, Need};
    use kernel_core::contract::{Argument, DType};
    use kernel_core::mxfp8_gemm::{
        Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery, Mxfp8FoldStreamBuffers,
    };
    use kernel_core::precision::execution::BlockScaledExecution;

    fn query(m_capacity: u32, n: u32, k: u32) -> Mxfp8FoldGemmQuery {
        Mxfp8FoldGemmQuery {
            m_capacity,
            n,
            k,
            groups: 1,
            activation_row_stride: k,
            activation_group_stride: k,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        }
    }

    fn view(handle: u64) -> View {
        View {
            handle,
            bytes: 1 << 30,
        }
    }

    fn buffers() -> Mxfp8FoldGemmBuffers<View> {
        Mxfp8FoldGemmBuffers {
            activations: view(1),
            activation_scales: view(2),
            weights: view(3),
            weight_scales: view(4),
            output: view(5),
            live_rows: view(6),
            status: view(7),
            status_row_offset: 0,
        }
    }

    #[test]
    fn packed_mode_renames_weight_readers_and_declares_the_packed_layout() {
        let q = query(8, 64, 576);
        let fold = mxfp8_fold::Entry::new(q, "chop_packed_mode_fold_v2").unwrap();
        let ticket =
            mxfp8_fold_ticket::Entry::new(query(16, 64, 576), "chop_packed_mode_ticket_v2", false)
                .unwrap();
        let backend = Sm120::new()
            .with_mxfp8_fold(fold.clone())
            .with_mxfp8_fold_ticket(ticket.clone());
        for (packed_mode, expected) in [
            (false, fold.symbol.clone()),
            (true, packed::symbol(&fold.symbol)),
        ] {
            let backend = if packed_mode {
                Sm120::new()
                    .with_mxfp8_fold(fold.clone())
                    .with_mxfp8_fold_ticket(ticket.clone())
                    .with_packed_fold_weights()
            } else {
                Sm120::new().with_mxfp8_fold(fold.clone())
            };
            let op = backend.mxfp8_fold_gemm(&q, Need::rows(8)).unwrap();
            assert_eq!(op.info().symbols[0].0, expected);
            let launches = op
                .launches(&buffers(), View::EMPTY, Counts::rows(8))
                .unwrap();
            assert_eq!(launches[0].symbol, expected);
            let layout = op.info().layout_of("weights").unwrap();
            assert_eq!(layout.id == packed::layout().id, packed_mode);
        }
        let op = backend
            .with_packed_fold_weights()
            .mxfp8_fold_gemm(&ticket.query, Need::rows(16))
            .unwrap();
        let symbols: Vec<_> = op.info().symbols.iter().map(|(s, _)| s.clone()).collect();
        let [first, second] = ticket.symbols();
        assert!(symbols.contains(&first) && symbols.contains(&packed::symbol(&second)));
        assert!(!symbols.contains(&second));
        let launches = op.launches(&buffers(), view(9), Counts::rows(16)).unwrap();
        assert!(launches.iter().all(|l| l.symbol != second));
        assert!(launches.iter().any(|l| l.symbol == packed::symbol(&second)));
    }

    #[test]
    fn packed_mode_moves_tma_prefill_to_the_packed_weight_map() {
        let q = query(8192, 256, 512);
        let entry = mxfp8_fold::Entry::new(q, "chop_packed_mode_prefill_v2").unwrap();
        let op = Sm120::new()
            .with_mxfp8_prefill(entry.clone())
            .with_packed_fold_weights()
            .mxfp8_prefill_gemm(&q, Need::rows(256))
            .unwrap();
        let compute = packed::symbol(&mxfp8_prefill::symbol(&entry));
        assert!(op.info().symbols.iter().any(|(s, _)| *s == compute));
        let launches = op.launches(&buffers(), view(9), Counts::rows(256)).unwrap();
        let launch = launches.iter().find(|l| l.symbol == compute).unwrap();
        let Argument::TensorMap(map) = &launch.arguments.0[1] else {
            panic!("no weight map")
        };
        assert_eq!(map.dims, [128, 8, 4, 32]);
        assert_eq!(map.base_buffer, 3);
        assert_eq!(
            op.info().layout_of("weights").unwrap().id,
            packed::layout().id
        );
    }

    #[test]
    fn packed_mode_refuses_stream_prepare_and_unported_shapes() {
        let q = query(1, 64, 576);
        let stream = mxfp8_fold_stream::Entry::new(q, "chop_packed_mode_stream_v1", false).unwrap();
        let op = Sm120::new()
            .with_mxfp8_fold_stream(stream.clone())
            .with_packed_fold_weights()
            .mxfp8_fold_gemm_stream(&q, false, Need::rows(1))
            .unwrap();
        // The planned weight view is the packed bank, bound under its own
        // handle as a model's bindings give every role.
        let b = Mxfp8FoldStreamBuffers {
            first: buffers(),
            second: None,
            packed_weights: [view(10), View::EMPTY],
            state: view(8),
        };
        assert!(op.initialize(&b).is_err());
        assert!(op.launches(&b, view(9), Counts::rows(1)).is_ok());
        // N = 60 has no packed form: its fold is refused rather than run.
        let odd = query(8, 60, 576);
        let fold = mxfp8_fold::Entry::new(odd, "chop_packed_mode_odd_v2").unwrap();
        assert!(
            Sm120::new()
                .with_mxfp8_fold(fold)
                .with_packed_fold_weights()
                .mxfp8_fold_gemm(&odd, Need::rows(8))
                .is_err()
        );
    }
}

#[cfg(test)]
mod route_identity_tests {
    use super::*;
    use kernel_core::backend::Need;
    use kernel_core::contract::{ArgumentList, LaunchPlan};
    use kernel_core::properties::Invariance;
    use kernel_core::route::{RouteClass, branch, tagged};

    struct Identity(OpInfo);
    impl PreparedOp<()> for Identity {
        fn info(&self) -> &OpInfo {
            &self.0
        }
        fn launches(&self, _: &(), _: View, _: Counts) -> Result<Vec<Launch>, Refusal> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn packed_translation_keeps_distinct_branches_of_one_symbol() {
        let launch = Launch {
            symbol: "shared_packed_v1".into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1; 3],
                block: [32, 1, 1],
                cluster: [1; 3],
                pdl: true,
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
            },
            arguments: ArgumentList(vec![]),
            collective: false,
            persistent: false,
        };
        let conversions = Conversions(Arc::new(BTreeMap::from([(
            "shared".into(),
            Conversion::Rename,
        )])));
        let mut tags = Vec::new();
        for (class, wrapper) in [
            (RouteClass::Fast, "standalone"),
            (RouteClass::Baseline, "fallback"),
        ] {
            let info = OpInfo::new(
                "identity",
                Invariance::any_rows(1),
                vec![("shared".into(), 0)],
            );
            Need::rows(1).check(&info).unwrap();
            let op = tagged(
                Box::new(Identity(info)),
                vec![branch("shared", class, "projection", wrapper, [1..=1])],
            );
            let op = wrap(op, &conversions).unwrap();
            let tag = op.route(Counts::rows(1), &launch).unwrap();
            assert_eq!(tag.entry, format!("projection/{wrapper}/shared_packed_v1"));
            assert_eq!(tag.symbol, launch.symbol);
            assert_eq!(tag.class, class);
            tags.push(tag);
        }
        assert_eq!(kernel_core::route::merge(tags).unwrap().len(), 2);
    }
}
