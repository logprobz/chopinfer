//! Latent attention through the model's latent entries (`latent::Entry`):
//! the KV write, the attention over key lists with the split merge of a
//! decode or verify entry, and the head-batched GEMMs of the absorbed
//! projections. Each prepared op launches its entry as `latent::Entry`
//! plans it for the step's rows, binding the entry's parameters by name.
//!
//! An attention op runs the forms the model registers for its entry in
//! the entry's place: the slice-widened kernel (`latent::slices`) of an
//! FP8 one-split entry, and the column merge (`latent::merge_columns`) of
//! a splitting entry's split merge. Each writes its entry's outputs bit
//! for bit from the same launch ABI, at every row count the entry plans,
//! so the op runs it at every row count and keeps the entry's
//! implementation name. Without a registered form the entry runs itself.
//!
//! Each op claims what its entry declares (`latent::Entry::invariance`:
//! row-local entries whose schedule fixes every row's partition, one split
//! for prefill and 64 for decode and verify at every bucket), met with the
//! declarations of the forms it runs in the entry's place, which declare
//! their entry's claims since they write its bits at every row count.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal, Validate};
use kernel_core::latent_attention::{
    HeadBatchedGemmQuery, HeadGemmBuffers, LatentAttentionBuffers, LatentAttentionOutput,
    LatentAttentionQuery, LatentKvCache, LatentKvWriteBuffers,
};

use super::select::{Candidate, choose, sole};
use super::{MAX_CAPACITY, check_plan};
use crate::latent::{Entry, Query, merge_columns, slices};

/// A model's latent entries and the forms that run in some of their places.
#[derive(Default)]
pub(super) struct Selection {
    pub entries: Vec<Entry>,
    /// Slice-widened forms of FP8 one-split attention entries.
    pub slices: Vec<slices::Entry>,
    /// Column merges of splitting attention entries' split merges.
    pub merges: Vec<merge_columns::Entry>,
}

/// The product of `factors` in bytes, refused rather than wrapped.
fn bytes(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1_u64, |product, &factor| product.checked_mul(factor))
        .ok_or(Refusal("latent buffer bytes overflow"))
}

/// Refuses a cache that does not hold `pages` whole pages of
/// `page_bytes` at its stride.
fn check_cache(cache: &LatentKvCache<View>, page_bytes: u64) -> Result<(), Refusal> {
    if cache.pages == 0 || cache.page_stride < page_bytes {
        return Err(Refusal(
            "a latent cache holds at least one page at a stride of at least one page",
        ));
    }
    let last = bytes(&[u64::from(cache.pages - 1), cache.page_stride])?;
    cache.pool.covers(
        last.checked_add(page_bytes)
            .ok_or(Refusal("latent buffer bytes overflow"))?,
    )
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// The arguments of `params` in parameter order, each bound by name.
fn arguments(params: &[EntryParam], bound: &[(&str, Argument)]) -> Result<ArgumentList, Refusal> {
    let arguments = params
        .iter()
        .map(|param| {
            let (name, pointer) = match param {
                EntryParam::DevicePointer { name, .. } => (name, true),
                EntryParam::U32 { name } | EntryParam::U64 { name } => (name, false),
                _ => return Err(Refusal("a latent entry takes an unsupported parameter")),
            };
            let argument = bound
                .iter()
                .find(|(role, _)| role == name)
                .map(|(_, argument)| argument.clone())
                .ok_or(Refusal("a latent entry parameter has no role"))?;
            let kind = matches!(
                (&argument, pointer, param),
                (Argument::DeviceBuffer(_), true, _)
                    | (Argument::U32(_), false, EntryParam::U32 { .. })
                    | (Argument::U64(_), false, EntryParam::U64 { .. })
            );
            if !kind {
                return Err(Refusal("a latent role binds the wrong kind"));
            }
            Ok(argument)
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let arguments = ArgumentList(arguments);
    arguments.validate()?;
    Ok(arguments)
}

/// The versioned implementation of an entry: its symbol and a digest of
/// its query and kernels (symbols, ABIs and bodies), which fix its
/// numerics.
fn implementation(entry: &Entry) -> Result<String, Refusal> {
    let declaration = format!("{:?}{:?}", entry.query, entry.entry_specs()?);
    let digest = kernel_build::digest(declaration.as_bytes());
    Ok(format!(
        "sm120.latent.v1[{}]#{}",
        entry.symbol(),
        &digest[..16]
    ))
}

/// What a latent entry launches: its symbol, query and kernel specs
/// (`implementation`) at its capacity.
fn launch(entry: &Entry) -> String {
    let Entry {
        profile: _, // filters catalogs; no launch reads it
        name: _,    // in the symbols and specs
        query,      // selects the op
    } = entry;
    format!("{} {query:?} {:?}", entry.symbol(), entry.entry_specs())
}

/// A kernel an op launches: the entry's main kernel or split merge, or the
/// form that runs in its place, planned at every row count.
struct Kernel {
    symbol: String,
    params: Vec<EntryParam>,
    plans: Vec<LaunchPlan>,
}

impl Kernel {
    /// `symbol` with `params`, planned by `plan` at every row count up to
    /// `capacity` and checked against SM120.
    fn planned(
        symbol: String,
        params: Vec<EntryParam>,
        capacity: u32,
        plan: impl Fn(u32) -> Result<LaunchPlan, Refusal>,
    ) -> Result<Self, Refusal> {
        let plans = (1..=capacity)
            .map(|rows| {
                let plan = plan(rows)?;
                check_plan(&plan)?;
                Ok(plan)
            })
            .collect::<Result<Vec<_>, Refusal>>()?;
        Ok(Self {
            symbol,
            params,
            plans,
        })
    }
}

/// The form of `entry` among `forms` (each with the entry it serves, its
/// symbol and what it launches), if the model registered one. Equal forms
/// are one; two that launch differently are a tie the op does not resolve.
fn form<'a, F>(
    entry: &Entry,
    forms: &'a [F],
    serves: impl Fn(&F) -> &Entry,
    describe: impl Fn(&F) -> (String, String),
    family: &str,
) -> Result<Option<&'a F>, Refused> {
    let key = launch(entry);
    let found = forms
        .iter()
        .filter(|form| launch(serves(form)) == key)
        .map(|form| {
            let (symbol, launches) = describe(form);
            (symbol, launches, form)
        });
    Ok(sole(
        found.map(Described),
        |described| described.0.0.clone(),
        family,
    )?
    .map(|described| described.0.2))
}

/// A form compared by what it launches.
struct Described<'a, F>((String, String, &'a F));

impl<F> PartialEq for Described<'_, F> {
    fn eq(&self, other: &Self) -> bool {
        self.0.1 == other.0.1
    }
}

/// One latent entry prepared for up to the needed rows: its main kernel's
/// plan, and its merge's, at every row count from one, checked against
/// SM120 at prepare; each the entry's or the form's that runs in its place.
struct LatentOp {
    entry: Entry,
    main: Kernel,
    merge: Option<Kernel>,
    info: OpInfo,
}

impl LatentOp {
    fn new(selection: &Selection, query: Query, need: Need) -> Result<Self, Refused> {
        let entries = &selection.entries;
        match query {
            Query::Latent(query) => query.validate()?,
            Query::HeadGemm(query) => query.validate()?,
        }
        crate::latent::check_limits(query)?;
        if need.capacity == 0 || need.capacity > MAX_CAPACITY {
            return Err(Refused::Rule(
                "a prepared op serves 1 to MAX_CAPACITY rows".into(),
            ));
        }
        let chosen = choose(
            "latent",
            entries
                .iter()
                .filter(|entry| entry.query == query)
                .map(|entry| {
                    Candidate::new(entry, entry.symbol(), launch(entry), entry.invariance())
                        .capacity(entry.capacity())
                }),
            need,
        )?;
        let (entry, mut guarantees) = (chosen.entry.clone(), chosen.guarantees);
        entry.contract()?;
        if need.capacity > entry.capacity() {
            return Err(Refused::Rule(
                "the latent capacity exceeds the SM120 entry's".into(),
            ));
        }
        let capacity = need.capacity;
        let slices = form(
            &entry,
            &selection.slices,
            |form| &form.attention,
            |form| (form.symbol(), format!("{:?}", form.entry_spec())),
            "slice-widened latent attention",
        )?;
        let main = match slices {
            Some(slices) => Kernel::planned(slices.symbol(), slices.params(), capacity, |rows| {
                slices.plan(rows)
            })?,
            None => Kernel::planned(entry.symbol(), entry.params(), capacity, |rows| {
                entry.plan(rows)
            })?,
        };
        if let Some(slices) = slices {
            guarantees = guarantees.meet(slices.invariance());
        }
        let main_shared = slices.map_or(entry.dynamic_shared_bytes(), |slices| {
            slices.dynamic_shared_bytes()
        });
        let columns = form(
            &entry,
            &selection.merges,
            |form| &form.attention,
            |form| (form.symbol(), format!("{:?}", form.entry_spec())),
            "latent column merge",
        )?;
        if let (Some(_), Some(columns)) = (entry.merge_symbol(), columns) {
            guarantees = guarantees.meet(columns.invariance());
        }
        let merge = match (entry.merge_symbol(), columns) {
            (None, _) => None,
            (Some(_), Some(columns)) => Some(Kernel::planned(
                columns.symbol(),
                columns.params(),
                capacity,
                |rows| columns.plan(rows),
            )?),
            (Some(symbol), None) => Some(Kernel::planned(
                symbol,
                entry.merge_params(),
                capacity,
                |rows| {
                    entry
                        .merge_plan(rows)?
                        .ok_or(Refusal("a latent merge has no plan"))
                },
            )?),
        };
        let mut symbols = vec![(main.symbol.clone(), main_shared)];
        symbols.extend(merge.as_ref().map(|merge| (merge.symbol.clone(), 0)));
        let mut info = OpInfo::new(implementation(&entry)?, guarantees, symbols);
        if matches!(query, Query::Latent(LatentAttentionQuery::Attention { .. })) {
            info.bytes
                .push(("workspace".into(), entry.workspace_bytes(capacity)));
        }
        need.check(&info)?;
        Ok(Self {
            entry,
            main,
            merge,
            info,
        })
    }

    /// The step's rows, refused outside the prepared capacity.
    fn rows(&self, counts: Counts) -> Result<u32, Refusal> {
        if counts.rows == 0 || counts.rows as usize > self.main.plans.len() {
            return Err(Refusal("latent rows outside the prepared capacity"));
        }
        Ok(counts.rows)
    }

    fn plan(&self, rows: u32) -> LaunchPlan {
        self.main.plans[rows as usize - 1]
    }

    /// The main kernel's launch, then the merge's, over `bound` roles.
    fn launches(&self, rows: u32, bound: &[(&str, Argument)]) -> Result<Vec<Launch>, Refusal> {
        let launch = |kernel: &Kernel| {
            Ok::<_, Refusal>(Launch {
                symbol: kernel.symbol.clone(),
                rows,
                plan: kernel.plans[rows as usize - 1],
                arguments: arguments(&kernel.params, bound)?,
                collective: false,
                persistent: false,
            })
        };
        let mut launches = vec![launch(&self.main)?];
        if let Some(merge) = &self.merge {
            launches.push(launch(merge)?);
        }
        Ok(launches)
    }
}

/// The latent KV write of `op` through the model's entry for it.
pub(super) fn kv_write(
    selection: &Selection,
    op: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentKvWriteBuffers<View>>, Refused> {
    let LatentAttentionQuery::KvWrite {
        latent,
        page_size,
        kv,
        ..
    } = *op
    else {
        return Err(Refused::Rule("expected a latent KV write".into()));
    };
    let op = LatentOp::new(selection, Query::Latent(*op), need)?;
    Ok(Box::new(KvWrite {
        op,
        latent: u64::from(latent),
        page_bytes: kv.page_bytes(latent, page_size),
    }))
}

struct KvWrite {
    op: LatentOp,
    latent: u64,
    page_bytes: u64,
}

impl PreparedOp<LatentKvWriteBuffers<View>> for KvWrite {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &LatentKvWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let row_count = u64::from(rows);
        buffers
            .latent
            .covers(bytes(&[row_count, self.latent, 2])?)?;
        buffers.row_batch.covers(bytes(&[row_count, 4])?)?;
        buffers.positions.covers(bytes(&[row_count, 4])?)?;
        buffers.live_rows.covers(4)?;
        check_cache(&buffers.cache, self.page_bytes)?;
        buffers.block_tables.check()?;
        self.op.launches(
            rows,
            &[
                ("latent", buffer(buffers.latent)),
                ("row_batch", buffer(buffers.row_batch)),
                ("positions", buffer(buffers.positions)),
                ("live_rows", buffer(buffers.live_rows)),
                ("block_tables", buffer(buffers.block_tables.tables)),
                ("pool", buffer(buffers.cache.pool)),
                ("max_pages", Argument::U32(buffers.block_tables.max_pages)),
                (
                    "page_stride_bytes",
                    Argument::U64(buffers.cache.page_stride),
                ),
            ],
        )
    }
}

/// The latent attention of `op` through the model's entry for it, with
/// the entry's split merge for decode and verify.
pub(super) fn attention(
    selection: &Selection,
    op: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentAttentionBuffers<View>>, Refused> {
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        softmax_scale_bits,
        page_size,
        kv,
        output,
        ..
    } = *op
    else {
        return Err(Refused::Rule("expected a latent attention".into()));
    };
    let op = LatentOp::new(selection, Query::Latent(*op), need)?;
    Ok(Box::new(Attention {
        splits: op.entry.splits(),
        op,
        heads: u64::from(heads),
        latent: u64::from(latent),
        max_keys,
        scale_bits: softmax_scale_bits,
        page_bytes: kv.page_bytes(latent, page_size),
        partial: output == LatentAttentionOutput::Partial,
    }))
}

struct Attention {
    op: LatentOp,
    heads: u64,
    latent: u64,
    max_keys: u32,
    scale_bits: u32,
    page_bytes: u64,
    splits: u32,
    partial: bool,
}

impl PreparedOp<LatentAttentionBuffers<View>> for Attention {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &LatentAttentionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let row_count = u64::from(rows);
        if buffers.budget > self.max_keys {
            return Err(Refusal("a latent key list budget exceeds the query's keys"));
        }
        let values = bytes(&[row_count, self.heads, self.latent])?;
        buffers.q.covers(bytes(&[values, 2])?)?;
        buffers.row_batch.covers(bytes(&[row_count, 4])?)?;
        buffers
            .key_lists
            .covers(bytes(&[row_count, u64::from(self.max_keys), 4])?)?;
        buffers.live_rows.covers(4)?;
        buffers
            .workspace
            .covers(self.op.plan(rows).workspace_bytes)?;
        check_cache(&buffers.cache, self.page_bytes)?;
        buffers.block_tables.check()?;
        let mut bound = vec![
            ("q", buffer(buffers.q)),
            ("pool", buffer(buffers.cache.pool)),
            ("block_tables", buffer(buffers.block_tables.tables)),
            ("row_batch", buffer(buffers.row_batch)),
            ("key_lists", buffer(buffers.key_lists)),
            ("live_rows", buffer(buffers.live_rows)),
            ("workspace", buffer(buffers.workspace)),
            ("max_pages", Argument::U32(buffers.block_tables.max_pages)),
            (
                "page_stride_bytes",
                Argument::U64(buffers.cache.page_stride),
            ),
            ("budget", Argument::U32(buffers.budget)),
            ("splits", Argument::U32(self.splits)),
            ("scale_bits", Argument::U32(self.scale_bits)),
        ];
        match (self.partial, buffers.output, buffers.partial) {
            (false, Some(output), None) => {
                output.covers(bytes(&[values, 2])?)?;
                bound.push(("out", buffer(output)));
            }
            (true, None, Some(partial)) => {
                let pairs = bytes(&[row_count, self.heads, 4])?;
                partial.max.covers(pairs)?;
                partial.den.covers(pairs)?;
                partial.num.covers(bytes(&[values, 4])?)?;
                bound.extend([
                    ("out_max", buffer(partial.max)),
                    ("out_den", buffer(partial.den)),
                    ("out_num", buffer(partial.num)),
                ]);
            }
            _ => {
                return Err(Refusal(
                    "a latent attention writes exactly its output mode's roles",
                ));
            }
        }
        self.op.launches(rows, &bound)
    }
}

/// The head-batched GEMM of `op` through the model's entry for it.
pub(super) fn head_gemm(
    selection: &Selection,
    op: &HeadBatchedGemmQuery,
    need: Need,
) -> Result<Prepared<HeadGemmBuffers<View>>, Refused> {
    Ok(Box::new(HeadGemm {
        op: LatentOp::new(selection, Query::HeadGemm(*op), need)?,
        query: *op,
    }))
}

struct HeadGemm {
    op: LatentOp,
    query: HeadBatchedGemmQuery,
}

impl PreparedOp<HeadGemmBuffers<View>> for HeadGemm {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &HeadGemmBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let q = self.query;
        let (heads, n, k) = (u64::from(q.heads), u64::from(q.n), u64::from(q.k));
        let matrix = bytes(&[n, k])?;
        if buffers.head_stride < matrix {
            return Err(Refusal(
                "a head-batched GEMM's head stride holds less than one head",
            ));
        }
        let weights = bytes(&[heads - 1, buffers.head_stride])?
            .checked_add(matrix)
            .ok_or(Refusal("latent buffer bytes overflow"))?;
        buffers.weight.covers(bytes(&[weights, 2])?)?;
        let output = if q.output == DType::F32 { 4 } else { 2 };
        let row_count = u64::from(rows);
        buffers.x.covers(bytes(&[row_count, heads, k, 2])?)?;
        buffers.y.covers(bytes(&[row_count, heads, n, output])?)?;
        buffers.live_rows.covers(4)?;
        self.op.launches(
            rows,
            &[
                ("x", buffer(buffers.x)),
                ("w", buffer(buffers.weight)),
                ("live_rows", buffer(buffers.live_rows)),
                ("y", buffer(buffers.y)),
                ("head_stride", Argument::U64(buffers.head_stride)),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::latent::{self, Geometry};
    use crate::{Profile, ProfileNames};
    use kernel_core::backend::Backend;
    use kernel_core::contract::AttentionPhase;
    use kernel_core::latent_attention::BlockTables;
    use kernel_core::latent_attention::{HeadWeightLayout, LatentKvFormat, LatentPartial};
    use kernel_core::properties::Invariance;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "latent-backend-test",
        tag: "latent_backend_test",
    });

    /// The served shape: 16 local heads over a 512-wide latent, 64-token
    /// pages, 2048 selected keys plus a 3-token pool tail, 256-wide halves.
    fn geometry() -> Geometry {
        Geometry {
            heads: 16,
            latent: 512,
            page_size: 64,
            max_keys: 2051,
            softmax_scale: 1.0 / 16.0,
            nope: 256,
            v_dim: 256,
        }
    }

    fn served() -> (Sm120, Vec<latent::Entry>) {
        let entries = latent::entries(PROFILE, geometry());
        let backend = entries
            .iter()
            .cloned()
            .fold(Sm120::new(), Sm120::with_latent);
        (backend, entries)
    }

    fn find(entries: &[latent::Entry], suffix: &str) -> latent::Entry {
        entries
            .iter()
            .find(|entry| entry.name.ends_with(suffix))
            .unwrap()
            .clone()
    }

    fn latent_query(entry: &latent::Entry) -> LatentAttentionQuery {
        let Query::Latent(query) = entry.query else {
            panic!("a latent entry")
        };
        query
    }

    const PAGES: u32 = 64;
    const MAX_PAGES: u32 = 130;
    const SEQUENCES: u32 = 32;

    /// A pool of [`PAGES`] pages at twice a page's bytes, as when other
    /// per-token data shares the page allocation.
    fn cache(kv: LatentKvFormat) -> LatentKvCache<View> {
        let stride = 2 * kv.page_bytes(512, 64);
        LatentKvCache {
            pool: view(20, u64::from(PAGES) * stride),
            pages: PAGES,
            page_stride: stride,
        }
    }

    fn tables() -> BlockTables<View> {
        BlockTables {
            tables: view(21, u64::from(SEQUENCES * MAX_PAGES) * 4),
            sequences: SEQUENCES,
            max_pages: MAX_PAGES,
        }
    }

    fn handles(handles: &[u64]) -> Vec<Argument> {
        handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect()
    }

    fn launch(symbol: String, rows: u32, plan: LaunchPlan, arguments: Vec<Argument>) -> Launch {
        Launch {
            symbol,
            rows,
            plan,
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }
    }

    /// The KV writes, prefill attention and head GEMMs declare one-row bits,
    /// any chunk split and row independence up to their capacity; decode and
    /// verify attention (and their merges) one-row bits and row
    /// independence. The forms declare their entry's claims, lab mutants
    /// nothing, and the ops claim what they launch.
    #[test]
    fn latent_entries_declare_their_schedules_claims() {
        let (backend, entries) = served();
        for entry in &entries {
            let expected = if entry.merge_symbol().is_some() {
                Invariance::one_row(256)
            } else {
                Invariance::any_rows(8192)
            };
            assert_eq!(entry.invariance(), expected, "{}", entry.name);
            let symbols = entry.declared_symbols();
            assert_eq!(symbols[0], (entry.symbol(), expected));
            assert_eq!(
                symbols.len(),
                1 + usize::from(entry.merge_symbol().is_some())
            );
            assert!(symbols.iter().all(|(_, claimed)| *claimed == expected));
        }
        for form in latent::slices::entries(&entries) {
            assert_eq!(form.invariance(), form.attention.invariance());
            let mutant = latent::slices::Entry {
                suffix: "_late_wait".into(),
                ..form.clone()
            };
            assert_eq!(mutant.invariance(), Invariance::NONE);
        }
        for form in latent::merge_columns::entries(&entries) {
            assert_eq!(form.invariance(), Invariance::one_row(256));
            let mutant = latent::merge_columns::Entry {
                suffix: "_descending".into(),
                ..form.clone()
            };
            assert_eq!(mutant.invariance(), Invariance::NONE);
        }
        let forms = latent::slices::entries(&entries)
            .into_iter()
            .fold(backend, Sm120::with_latent_slices);
        let forms = latent::merge_columns::entries(&entries)
            .into_iter()
            .fold(forms, Sm120::with_latent_merge_columns);
        for (suffix, rows, claimed) in [
            ("attn_fp8_prefill", 8192, Invariance::any_rows(8192)),
            ("attn_bf16_verify", 192, Invariance::one_row(256)),
            ("attn_fp8_decode_partial", 32, Invariance::one_row(256)),
        ] {
            let query = latent_query(&find(&entries, suffix));
            let prepared = forms
                .latent_attention(&query, Need::rows(rows).with_exact(claimed))
                .unwrap();
            assert_eq!(prepared.info().guarantees, claimed, "{suffix}");
        }
    }

    #[test]
    fn kv_writes_launch_what_the_builder_plans() {
        let (backend, entries) = served();
        for (suffix, kv) in [
            ("kv_write_bf16", LatentKvFormat::Bf16),
            ("kv_write_fp8", LatentKvFormat::Fp8E4M3Block128),
        ] {
            let entry = find(&entries, suffix);
            let prepared = backend
                .latent_kv_write(&latent_query(&entry), Need::rows(8192))
                .unwrap();
            assert!(
                prepared
                    .info()
                    .implementation
                    .starts_with(&format!("sm120.latent.v1[{}]#", entry.symbol()))
            );
            assert_eq!(prepared.info().guarantees, Invariance::any_rows(8192));
            assert_eq!(prepared.info().symbols, [(entry.symbol(), 0)]);
            let buffers = LatentKvWriteBuffers {
                latent: view(1, 8192 * 512 * 2),
                cache: cache(kv),
                block_tables: tables(),
                row_batch: view(2, 8192 * 4),
                positions: view(3, 8192 * 4),
                live_rows: view(4, 4),
            };
            let stride = cache(kv).page_stride;
            for rows in [1, 9, 64, 8192] {
                let mut arguments = handles(&[1, 2, 3, 4, 21, 20]);
                arguments.extend([Argument::U32(MAX_PAGES), Argument::U64(stride)]);
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [launch(
                        entry.symbol(),
                        rows,
                        entry.plan(rows).unwrap(),
                        arguments
                    )]
                );
            }
        }
    }

    /// The attention buffers of `rows` rows for an op of `output` mode.
    fn attention_buffers(
        kv: LatentKvFormat,
        output: LatentAttentionOutput,
        rows: u64,
        workspace: u64,
    ) -> LatentAttentionBuffers<View> {
        let values = rows * 16 * 512;
        let normalized = output == LatentAttentionOutput::Normalized;
        LatentAttentionBuffers {
            q: view(1, values * 2),
            cache: cache(kv),
            block_tables: tables(),
            row_batch: view(2, rows * 4),
            key_lists: view(3, rows * 2051 * 4),
            budget: 2050,
            output: normalized.then(|| view(4, values * 2)),
            partial: (!normalized).then(|| LatentPartial {
                max: view(5, rows * 16 * 4),
                den: view(6, rows * 16 * 4),
                num: view(7, values * 4),
            }),
            workspace: view(8, workspace),
            live_rows: view(9, 4),
        }
    }

    #[test]
    fn attention_launches_what_the_builder_plans() {
        let (backend, entries) = served();
        let scale = (1.0_f32 / 16.0).to_bits();
        let mut checked = 0;
        for entry in entries.iter().filter(|entry| entry.name.contains("_attn_")) {
            let query = latent_query(entry);
            let LatentAttentionQuery::Attention {
                kv, phase, output, ..
            } = query
            else {
                unreachable!()
            };
            let capacity = entry.capacity();
            let prepared = backend
                .latent_attention(&query, Need::rows(capacity))
                .unwrap();
            let info = prepared.info();
            assert_eq!(
                info.bytes_of("workspace"),
                Some(entry.workspace_bytes(capacity))
            );
            let mut symbols = vec![(entry.symbol(), entry.dynamic_shared_bytes())];
            symbols.extend(entry.merge_symbol().map(|merge| (merge, 0)));
            assert_eq!(info.symbols, symbols);
            let splits = if phase == AttentionPhase::Prefill {
                1
            } else {
                64
            };
            let outputs = if output == LatentAttentionOutput::Partial {
                handles(&[5, 6, 7])
            } else {
                handles(&[4])
            };
            let stride = cache(kv).page_stride;
            for rows in [1, 6, 32, capacity] {
                let workspace = entry.workspace_bytes(rows);
                let buffers = attention_buffers(kv, output, u64::from(rows), workspace);
                let mut main = handles(&[1, 20, 21, 2, 3, 9, 8]);
                if splits == 1 {
                    main.extend(outputs.clone());
                }
                main.extend([
                    Argument::U32(MAX_PAGES),
                    Argument::U64(stride),
                    Argument::U32(2050),
                    Argument::U32(splits),
                    Argument::U32(scale),
                ]);
                let mut expected = vec![launch(
                    entry.symbol(),
                    rows,
                    entry.plan(rows).unwrap(),
                    main,
                )];
                if let Some(merge) = entry.merge_symbol() {
                    let mut arguments = handles(&[8, 9]);
                    arguments.extend(outputs.clone());
                    arguments.push(Argument::U32(splits));
                    expected.push(launch(
                        merge,
                        rows,
                        entry.merge_plan(rows).unwrap().unwrap(),
                        arguments,
                    ));
                }
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    expected,
                    "{} at {rows} rows",
                    entry.name
                );
                // The workspace must hold the step's partials.
                if workspace > 0 {
                    let short = attention_buffers(kv, output, u64::from(rows), workspace - 1);
                    assert!(
                        prepared
                            .launches(&short, View::EMPTY, Counts::rows(rows))
                            .is_err()
                    );
                }
            }
            checked += 1;
        }
        assert_eq!(checked, 12);
    }

    /// The served entries with every form registered: the slice-widened
    /// FP8 one-split entries and the column merges of the splitting ones.
    fn with_forms() -> (Sm120, Vec<latent::Entry>) {
        let (backend, entries) = served();
        let backend = latent::slices::entries(&entries)
            .into_iter()
            .fold(backend, Sm120::with_latent_slices);
        let backend = latent::merge_columns::entries(&entries)
            .into_iter()
            .fold(backend, Sm120::with_latent_merge_columns);
        (backend, entries)
    }

    /// The kernels a builder launches for attention `entry` at `rows`
    /// rows: the slice-widened form of the same symbol's entry when there
    /// is one, else the entry; then the column merge of the same symbol's
    /// entry when there is one, else the entry's own merge.
    fn builder_kernels(
        entry: &latent::Entry,
        rows: u32,
    ) -> ((String, LaunchPlan), Option<(String, LaunchPlan)>) {
        let entries = [entry.clone()];
        let main = match latent::slices::entries(&entries)
            .into_iter()
            .find(|form| form.attention.symbol() == entry.symbol())
        {
            Some(form) => (form.symbol(), form.plan(rows).unwrap()),
            None => (entry.symbol(), entry.plan(rows).unwrap()),
        };
        let merge = match latent::merge_columns::entries(&entries)
            .into_iter()
            .find(|form| form.attention.symbol() == entry.symbol())
        {
            Some(form) => Some((form.symbol(), form.plan(rows).unwrap())),
            None => entry
                .merge_symbol()
                .map(|symbol| (symbol, entry.merge_plan(rows).unwrap().unwrap())),
        };
        (main, merge)
    }

    /// Every attention entry with its forms registered launches what the
    /// builder launches at every row count it is checked at: the FP8
    /// one-split entries (normalized and partial) their slice-widened
    /// form, every splitting entry its column merge, in both output modes.
    /// The op keeps the entry's implementation and workspace and lists the
    /// symbols it launches with the dynamic shared memory each takes.
    #[test]
    fn attention_runs_its_forms_as_the_builder_does() {
        let (backend, entries) = with_forms();
        let scale = (1.0_f32 / 16.0).to_bits();
        let (mut sliced, mut column_merged) = (0, 0);
        for entry in entries.iter().filter(|entry| entry.name.contains("_attn_")) {
            let query = latent_query(entry);
            let LatentAttentionQuery::Attention { kv, output, .. } = query else {
                unreachable!()
            };
            let capacity = entry.capacity();
            let prepared = backend
                .latent_attention(&query, Need::rows(capacity))
                .unwrap();
            let plain = served()
                .0
                .latent_attention(&query, Need::rows(capacity))
                .unwrap();
            let info = prepared.info();
            assert_eq!(info.implementation, plain.info().implementation);
            assert_eq!(info.bytes, plain.info().bytes);
            let ((main, _), merge) = builder_kernels(entry, 1);
            if main != entry.symbol() {
                sliced += 1;
                let form = latent::slices::Entry::of(entry).unwrap();
                assert_eq!(info.symbols[0], (main, form.dynamic_shared_bytes()));
            } else {
                assert_eq!(info.symbols[0], (main, entry.dynamic_shared_bytes()));
            }
            if merge.as_ref().map(|m| &m.0) != entry.merge_symbol().as_ref() {
                column_merged += 1;
            }
            assert_eq!(info.symbols.get(1).cloned(), merge.map(|m| (m.0, 0)));
            let splits = entry.splits();
            let outputs = if output == LatentAttentionOutput::Partial {
                handles(&[5, 6, 7])
            } else {
                handles(&[4])
            };
            let stride = cache(kv).page_stride;
            for rows in [1, 2, 6, 31, 32, 33, 64, capacity] {
                let workspace = entry.workspace_bytes(rows);
                let buffers = attention_buffers(kv, output, u64::from(rows), workspace);
                let mut arguments = handles(&[1, 20, 21, 2, 3, 9, 8]);
                if splits == 1 {
                    arguments.extend(outputs.clone());
                }
                arguments.extend([
                    Argument::U32(MAX_PAGES),
                    Argument::U64(stride),
                    Argument::U32(2050),
                    Argument::U32(splits),
                    Argument::U32(scale),
                ]);
                let ((symbol, plan), merge) = builder_kernels(entry, rows);
                let mut expected = vec![launch(symbol, rows, plan, arguments)];
                if let Some((symbol, plan)) = merge {
                    let mut arguments = handles(&[8, 9]);
                    arguments.extend(outputs.clone());
                    arguments.push(Argument::U32(splits));
                    expected.push(launch(symbol, rows, plan, arguments));
                }
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    expected,
                    "{} at {rows} rows",
                    entry.name
                );
            }
        }
        // FP8 prefill in both output modes; decode and verify over both
        // caches in both output modes.
        assert_eq!((sliced, column_merged), (2, 8));
    }

    /// A form serves the entry whose launches it was made from, never
    /// another entry of the op's query, in every registration order; equal
    /// forms are one, and two forms of one entry that launch differently
    /// are a tie the op refuses by name.
    #[test]
    fn forms_serve_their_own_entry_in_every_order() {
        let (_, entries) = served();
        let prefill = find(&entries, "attn_fp8_prefill");
        let decode = find(&entries, "attn_fp8_decode");
        let slices = latent::slices::Entry::of(&prefill).unwrap();
        let columns = latent::merge_columns::Entry::of(&decode).unwrap();
        // The forms of another profile's entry of the same queries.
        let other = |entry: &latent::Entry| latent::Entry {
            name: format!("{}_twin", entry.name),
            ..entry.clone()
        };
        let foreign_slices = latent::slices::Entry::of(&other(&prefill)).unwrap();
        let foreign_columns = latent::merge_columns::Entry::of(&other(&decode)).unwrap();
        #[derive(Clone, Debug)]
        enum Registered {
            Entry(latent::Entry),
            Slices(latent::slices::Entry),
            Columns(latent::merge_columns::Entry),
        }
        let register = |order: &[Registered]| {
            order.iter().cloned().fold(Sm120::new(), |b, r| match r {
                Registered::Entry(e) => b.with_latent(e),
                Registered::Slices(f) => b.with_latent_slices(f),
                Registered::Columns(f) => b.with_latent_merge_columns(f),
            })
        };
        let symbols = |backend: &Sm120, entry: &latent::Entry| {
            backend
                .latent_attention(&latent_query(entry), Need::rows(8))
                .map(|prepared| {
                    prepared
                        .info()
                        .symbols
                        .iter()
                        .map(|(symbol, _)| symbol.clone())
                        .collect::<Vec<_>>()
                })
        };
        let registered = [
            Registered::Entry(prefill.clone()),
            Registered::Entry(decode.clone()),
            Registered::Slices(foreign_slices.clone()),
            Registered::Columns(foreign_columns.clone()),
            Registered::Slices(slices.clone()),
            Registered::Columns(columns.clone()),
        ];
        for order in orders(&registered) {
            let backend = register(&order);
            assert_eq!(symbols(&backend, &prefill), Ok(vec![slices.symbol()]));
            assert_eq!(
                symbols(&backend, &decode),
                Ok(vec![decode.symbol(), columns.symbol()])
            );
        }
        // Only another entry's forms: the entries run themselves.
        let foreign = register(&registered[..4]);
        assert_eq!(symbols(&foreign, &prefill), Ok(vec![prefill.symbol()]));
        assert_eq!(
            symbols(&foreign, &decode),
            Ok(vec![decode.symbol(), decode.merge_symbol().unwrap()])
        );
        // The same form twice is one.
        let twice = register(&[
            registered[0].clone(),
            registered[4].clone(),
            registered[4].clone(),
        ]);
        assert_eq!(symbols(&twice, &prefill), Ok(vec![slices.symbol()]));
        // A lab mutant beside the served form: a tie, named in order.
        for (forms, entry) in [
            (
                vec![
                    Registered::Slices(slices.clone()),
                    Registered::Slices(latent::slices::Entry {
                        suffix: "_skip_first".into(),
                        ..slices.clone()
                    }),
                ],
                &prefill,
            ),
            (
                vec![
                    Registered::Columns(columns.clone()),
                    Registered::Columns(latent::merge_columns::Entry {
                        suffix: "_skip_last".into(),
                        ..columns.clone()
                    }),
                ],
                &decode,
            ),
        ] {
            for order in orders(&forms) {
                let mut all = vec![Registered::Entry(entry.clone())];
                all.extend(order);
                let refused =
                    rule(register(&all).latent_attention(&latent_query(entry), Need::rows(8)));
                assert!(refused.starts_with("two SM120"), "{refused}");
            }
        }
    }

    /// The facts a model's buffers rely on: a decode or verify op prepared
    /// for 32 rows sizes its workspace for 32 rows' 64 split partials, and
    /// a verify op prepared for the 64 rows of a prefill-path drafter pass
    /// launches the verify entry and its column merge at each of them,
    /// each reading the workspace of its own rows only.
    #[test]
    fn split_workspace_follows_the_prepared_and_launched_rows() {
        let (backend, entries) = with_forms();
        for suffix in ["attn_fp8_decode", "attn_bf16_verify", "attn_fp8_verify"] {
            let entry = find(&entries, suffix);
            let prepared = backend
                .latent_attention(&latent_query(&entry), Need::rows(32))
                .unwrap();
            assert_eq!(
                prepared.info().bytes_of("workspace"),
                Some(32 * 64 * 16 * 514 * 4)
            );
        }
        let verify = find(&entries, "attn_fp8_verify");
        let query = latent_query(&verify);
        let prepared = backend.latent_attention(&query, Need::rows(64)).unwrap();
        assert_eq!(
            prepared.info().bytes_of("workspace"),
            Some(verify.workspace_bytes(64))
        );
        let kv = LatentKvFormat::Fp8E4M3Block128;
        let normalized = LatentAttentionOutput::Normalized;
        for rows in 1..=64 {
            let need = verify.workspace_bytes(rows);
            let fits = attention_buffers(kv, normalized, u64::from(rows), need);
            let launches = prepared
                .launches(&fits, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let ((symbol, plan), merge) = builder_kernels(&verify, rows);
            assert_eq!(
                launches
                    .iter()
                    .map(|launch| (launch.symbol.clone(), launch.plan))
                    .collect::<Vec<_>>(),
                [(symbol, plan), merge.unwrap()]
            );
            // The main kernel's `splits` scalar: 64 at every row count.
            assert_eq!(launches[0].arguments.0[10], Argument::U32(64));
            let short = attention_buffers(kv, normalized, u64::from(rows), need - 1);
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(rows))
                    .is_err()
            );
        }
    }

    #[test]
    fn head_gemms_launch_what_the_builder_plans() {
        let (backend, entries) = served();
        let stride = (256 + 256) * 512;
        let up = 16 * stride * 2;
        for (suffix, layout, weight, x, y) in [
            ("absorb_key", HeadWeightLayout::KByN, up, 256, 512),
            (
                "absorb_value",
                HeadWeightLayout::NByK,
                up - 256 * 512 * 2,
                512,
                256,
            ),
        ] {
            let entry = find(&entries, suffix);
            let Query::HeadGemm(query) = entry.query else {
                unreachable!()
            };
            assert_eq!(query.layout, layout);
            let prepared = backend.head_batched_gemm(&query, Need::rows(8192)).unwrap();
            assert!(prepared.info().weight_layouts.is_empty());
            assert_eq!(
                prepared.info().symbols,
                [(entry.symbol(), entry.dynamic_shared_bytes())]
            );
            let buffers = HeadGemmBuffers {
                x: view(1, 8192 * 16 * x * 2),
                weight: view(2, weight),
                head_stride: stride,
                y: view(3, 8192 * 16 * y * 2),
                live_rows: view(4, 4),
            };
            for rows in [1, 5, 64, 8192] {
                let mut arguments = handles(&[1, 2, 4, 3]);
                arguments.push(Argument::U64(stride));
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [launch(
                        entry.symbol(),
                        rows,
                        entry.plan(rows).unwrap(),
                        arguments
                    )]
                );
            }
            // The last head's matrix must lie inside the weights.
            let needed = (15 * stride + 256 * 512) * 2;
            assert!(weight >= needed);
            let exact = HeadGemmBuffers {
                weight: view(2, needed),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&exact, View::EMPTY, Counts::rows(1))
                    .is_ok()
            );
            let short = HeadGemmBuffers {
                weight: view(2, needed - 2),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            let narrow = HeadGemmBuffers {
                head_stride: 256 * 512 - 1,
                ..buffers
            };
            assert!(
                prepared
                    .launches(&narrow, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
    }

    fn rule(refused: Result<impl Sized, Refused>) -> String {
        match refused {
            Err(Refused::Rule(rule)) => rule,
            Err(other) => panic!("expected a rule refusal, got {other:?}"),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    #[test]
    fn latent_ops_refuse_what_sm120_cannot_serve() {
        let (backend, entries) = served();
        let decode = latent_query(&find(&entries, "attn_fp8_decode"));
        let write = latent_query(&find(&entries, "kv_write_fp8"));
        let LatentAttentionQuery::Attention {
            latent,
            max_keys,
            softmax_scale_bits,
            page_size,
            kv,
            phase,
            output,
            capacity,
            ..
        } = decode
        else {
            unreachable!()
        };
        let with = |heads, latent, sink| LatentAttentionQuery::Attention {
            heads,
            latent,
            max_keys,
            softmax_scale_bits,
            page_size,
            kv,
            phase,
            output,
            sink,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity,
        };
        // The wrong contract op for the method.
        assert!(rule(backend.latent_kv_write(&decode, Need::rows(1))).contains("KV write"));
        assert!(rule(backend.latent_attention(&write, Need::rows(1))).contains("attention"));
        // An unsupported variant, refused by name.
        assert!(
            rule(backend.latent_attention(&with(16, latent, true), Need::rows(1))).contains("sink")
        );
        // SM120's head tile and shared memory, on entries the model lists.
        let entry = |query| latent::Entry {
            profile: PROFILE,
            name: "latent_backend_test_odd".into(),
            query: Query::Latent(query),
        };
        let odd = with(8, latent, false);
        let wide = with(16, 1024, false);
        let limited = Sm120::new()
            .with_latent(entry(odd))
            .with_latent(entry(wide));
        assert!(rule(limited.latent_attention(&odd, Need::rows(1))).contains("heads in 16s"));
        assert!(rule(limited.latent_attention(&wide, Need::rows(1))).contains("shared memory"));
        // SM120's limits refuse before any entry is looked up.
        let write_with = |latent, page_size| LatentAttentionQuery::KvWrite {
            latent,
            page_size,
            kv,
            capacity: 8192,
        };
        let empty = Sm120::new();
        for (op, limit) in [
            (with(144, latent, false), "at most 128 heads"),
            (with(8, latent, false), "heads in 16s"),
            (with(16, 1024, false), "shared memory"),
        ] {
            assert!(rule(empty.latent_attention(&op, Need::rows(1))).contains(limit));
        }
        assert!(rule(empty.latent_kv_write(&write_with(2048, 64), Need::rows(1))).contains("1024"));
        assert!(
            rule(empty.latent_kv_write(&write_with(512, 48), Need::rows(1)))
                .contains("power of two")
        );
        let Query::HeadGemm(gemm) = find(&entries, "absorb_key").query else {
            unreachable!()
        };
        assert!(
            rule(empty.head_batched_gemm(&HeadBatchedGemmQuery { k: 100, ..gemm }, Need::rows(1)))
                .contains("K in 128s")
        );
        // A query no entry serves.
        assert!(rule(Sm120::new().latent_attention(&decode, Need::rows(1))).contains("no SM120"));
        // Exactness the entries do not declare: a decode entry's chunk
        // split, a tie rule and rows past its capacity; its one-row bits
        // are granted.
        for need in [
            Invariance::any_rows(8),
            Invariance::one_row(8).with_ties(kernel_core::properties::Ties::LowerIndex),
            Invariance::one_row(capacity + 1),
        ] {
            assert!(
                matches!(
                    backend.latent_attention(&decode, Need::rows(8).with_exact(need)),
                    Err(Refused::Inexact { .. })
                ),
                "{need:?}"
            );
        }
        assert!(
            backend
                .latent_attention(&decode, Need::rows(8).with_exact(Invariance::one_row(8)))
                .is_ok()
        );
        // Capacities past the entry's or past what an op prepares tables for.
        assert!(
            rule(backend.latent_attention(&decode, Need::rows(capacity + 1))).contains("capacity")
        );
        assert!(
            rule(backend.latent_attention(&decode, Need::rows(u32::MAX))).contains("MAX_CAPACITY")
        );
        assert!(rule(backend.latent_kv_write(&write, Need::rows(0))).contains("MAX_CAPACITY"));
    }

    #[test]
    fn attention_launches_check_every_role() {
        let (backend, entries) = served();
        let kv = LatentKvFormat::Bf16;
        let verify = find(&entries, "attn_bf16_verify");
        let prepared = backend
            .latent_attention(&latent_query(&verify), Need::rows(32))
            .unwrap();
        let workspace = verify.workspace_bytes(32);
        let normalized = LatentAttentionOutput::Normalized;
        let buffers = attention_buffers(kv, normalized, 32, workspace);
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_ok()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        let partial = attention_buffers(kv, LatentAttentionOutput::Partial, 32, workspace);
        let broken = [
            LatentAttentionBuffers {
                q: view(1, 32 * 16 * 512 * 2 - 1),
                ..buffers
            },
            LatentAttentionBuffers {
                key_lists: view(3, 32 * 2051 * 4 - 4),
                ..buffers
            },
            LatentAttentionBuffers {
                budget: 2052,
                ..buffers
            },
            // A normalized op writes `output` only.
            LatentAttentionBuffers {
                output: None,
                ..buffers
            },
            LatentAttentionBuffers {
                partial: partial.partial,
                ..buffers
            },
            LatentAttentionBuffers {
                output: Some(view(4, 32 * 16 * 512 * 2 - 2)),
                ..buffers
            },
            // Whole pages at a stride of at least one page.
            LatentAttentionBuffers {
                cache: LatentKvCache {
                    page_stride: kv.page_bytes(512, 64) - 1,
                    ..buffers.cache
                },
                ..buffers
            },
            LatentAttentionBuffers {
                cache: LatentKvCache {
                    pages: PAGES + 1,
                    ..buffers.cache
                },
                ..buffers
            },
            LatentAttentionBuffers {
                cache: LatentKvCache {
                    pages: 0,
                    ..buffers.cache
                },
                ..buffers
            },
            LatentAttentionBuffers {
                block_tables: BlockTables {
                    sequences: SEQUENCES + 1,
                    ..buffers.block_tables
                },
                ..buffers
            },
        ];
        for (index, buffers) in broken.iter().enumerate() {
            assert!(
                prepared
                    .launches(buffers, View::EMPTY, Counts::rows(32))
                    .is_err(),
                "case {index}"
            );
        }
        // A partial op writes all three partial roles.
        let partial_op = backend
            .latent_attention(
                &latent_query(&find(&entries, "attn_bf16_verify_partial")),
                Need::rows(32),
            )
            .unwrap();
        assert!(
            partial_op
                .launches(&partial, View::EMPTY, Counts::rows(32))
                .is_ok()
        );
        let short = LatentAttentionBuffers {
            partial: partial.partial.map(|p| LatentPartial {
                num: view(7, 32 * 16 * 512 * 4 - 4),
                ..p
            }),
            ..partial
        };
        assert!(
            partial_op
                .launches(&short, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        assert!(
            partial_op
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
    }

    #[test]
    fn entries_prepare_alike_in_every_order() {
        let (_, entries) = served();
        let write = entries
            .iter()
            .find(|e| matches!(e.query, Query::Latent(LatentAttentionQuery::KvWrite { .. })))
            .unwrap()
            .clone();
        let Query::Latent(op) = write.query else {
            unreachable!()
        };
        let mut twin = write.clone();
        twin.name = "latent_backend_test_twin".into();
        assert_ne!(twin.symbol(), write.symbol());
        let prepare = |order: &[latent::Entry]| {
            order
                .iter()
                .cloned()
                .fold(Sm120::new(), Sm120::with_latent)
                .latent_kv_write(&op, Need::rows(4))
                .map(|prepared| prepared.info().clone())
        };
        let in_every_order = |registered: &[latent::Entry]| {
            let results = orders(registered)
                .iter()
                .map(|o| prepare(o))
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        let one = prepare(std::slice::from_ref(&write)).unwrap();
        assert_eq!(in_every_order(&[write.clone(), write.clone()]), Ok(one));
        let mut symbols = [write.symbol(), twin.symbol()];
        symbols.sort_unstable();
        assert!(matches!(
            in_every_order(&[write, twin]),
            Err(Refused::Rule(rule)) if rule.contains(&symbols.join(" and "))
        ));
    }

    /// The latent key holds what an entry launches, every field changed
    /// alone: the symbol and code its name gives; never its profile; the
    /// query selects the op.
    #[test]
    fn launch_keys_hold_what_latent_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let (_, entries) = served();
        let write = find(&entries, "kv_write_bf16");
        let op = latent_query(&write);
        let other = crate::Profile::synthetic(crate::ProfileNames {
            name: "latent-other",
            tag: "latent_other",
        });
        let fp8 = latent_query(&find(&entries, "kv_write_fp8"));
        let names = field_names!(latent::Entry {
            profile,
            name,
            query
        });
        let fields: Vec<Field<latent::Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "name",
                Class::Launch,
                Box::new(|e| e.name = "latent_backend_test_twin".into()),
            ),
            (
                "query",
                Class::Selects,
                Box::new(move |e| e.query = Query::Latent(fp8)),
            ),
        ];
        let buffers = LatentKvWriteBuffers {
            latent: view(1, 8192 * 512 * 2),
            cache: cache(LatentKvFormat::Bf16),
            block_tables: tables(),
            row_batch: view(2, 8192 * 4),
            positions: view(3, 8192 * 4),
            live_rows: view(4, 4),
        };
        keys_follow_launches(&write, names, fields, super::launch, |e| {
            let prepared = Sm120::new()
                .with_latent(e.clone())
                .latent_kv_write(&op, Need::rows(4))?;
            let launched = [1, 4].map(|rows| {
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?}", e.entry_specs()), launched))
        });
    }
}
