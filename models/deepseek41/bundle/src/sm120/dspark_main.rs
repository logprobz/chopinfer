//! The qualified main-projection ranges, using the existing compact template.
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers as Buffers, Mxfp8FoldGemmQuery};
use kernel_core::route::{RouteClass, RouteTag};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_prefill_compact::Entry;
use model_deepseek41_contract::tuning::{Family, table};
use std::ops::RangeInclusive;

const SITE: &str = "dense_prefill.dspark_main.prefill";
const FAMILY: &str = "dspark_main";

/// One admitted range of inner rows, every count of which uses one compact
/// tile, and its stages.
type Range = (RangeInclusive<u32>, u32);

fn recipe(shape: &crate::shape::Shape, query: Mxfp8FoldGemmQuery) -> Result<Vec<Range>, Refusal> {
    let Some(table) = table(shape.geometry, Family::DensePrefill) else {
        return Ok(Vec::new());
    };
    let key: std::collections::BTreeMap<String, u32> = [
        ("n".into(), query.n),
        ("k".into(), query.k),
        ("groups".into(), query.groups),
        ("ranks".into(), shape.ranks()),
    ]
    .into();
    let rows = table.rows().iter().filter(|row| {
        row.key.op == SITE
            && row.key.weights == "mxfp8"
            && row.key.shape == key
            && row.key.bucket == query.m_capacity
    });
    recipe_rows(rows)
}

/// The site's rows as ranges in row order. Each row is one nonempty range on
/// one tile; rows that overlap or come out of row order are refused.
fn recipe_rows<'a>(
    rows: impl Iterator<Item = &'a kernel_build::tuning::Row>,
) -> Result<Vec<Range>, Refusal> {
    let mut ranges: Vec<Range> = Vec::new();
    for row in rows {
        let [first, last] = row
            .key
            .rows
            .ok_or(Refusal("main projection range is absent"))?;
        if first == 0 || first > last {
            return Err(Refusal("main projection range is empty"));
        }
        if ranges
            .last()
            .is_some_and(|(previous, _)| first <= *previous.end())
        {
            return Err(Refusal(
                "main projection ranges overlap or are out of row order",
            ));
        }
        if row.schedule != "prefill_tiles"
            || row.knobs.get("matrix_fetch").and_then(|v| v.flag()) != Some(false)
            || row.knobs.get("tile_rows").and_then(|v| v.count()) != Some(Entry::tile(first))
            || Entry::tile(first) != Entry::tile(last)
        {
            return Err(Refusal("unsupported main compact recipe"));
        }
        let stages = row
            .knobs
            .get("stages")
            .and_then(|v| v.count())
            .ok_or(Refusal("main compact stages are absent"))?;
        ranges.push((first..=last, stages));
    }
    Ok(ranges)
}

fn compact_backend(
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
    ranges: &[Range],
) -> Result<Sm120, Refused> {
    let base = super::mxfp8_fold::prefill_entries(shape.geometry, shape.ranks())
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|entry| entry.query == query)
        .ok_or(Refusal("main compact query has no compiled prefill entry"))?;
    let mut backend = Sm120::new().with_mxfp8_prefill(base.clone());
    // The shared wrapper prepares each count once, with its own range's
    // stages.
    for (rows, stages) in ranges {
        let entry = Entry::new(base.clone(), *stages)?;
        for count in rows.clone() {
            backend = backend.with_mxfp8_prefill_compact(entry.clone(), count);
        }
    }
    Ok(backend.with_packed_fold_weights())
}

/// The selected-branch tag of each compact symbol, admitting the rows its
/// registrations admit. A symbol two ranges share admits both.
fn compact_tags(backend: &Sm120) -> Result<Vec<RouteTag>, Refused> {
    kernel_core::route::merge(
        backend
            .route_tags()
            .into_iter()
            .filter(|tag| tag.family == "mxfp8_prefill" && tag.class == RouteClass::Fast)
            .map(|tag| {
                kernel_core::route::branch(
                    &tag.symbol,
                    RouteClass::Fast,
                    FAMILY,
                    "compact",
                    tag.admitted,
                )
            }),
    )
    .map_err(Refused::Rule)
}

pub(crate) fn prepare(
    backend: &dyn Backend,
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    let ranges = recipe(shape, query)?;
    prepare_ranges(backend, shape, query, need, &ranges)
}

fn prepare_ranges(
    backend: &dyn Backend,
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
    need: Need,
    ranges: &[Range],
) -> Result<Prepared<Buffers<View>>, Refused> {
    let fallback = backend.mxfp8_fold_gemm(&query, need)?;
    if ranges.is_empty() {
        return Ok(fallback);
    }
    let mut info = fallback.info().clone();
    let backend = compact_backend(shape, query, ranges)?;
    let compact = backend.mxfp8_prefill_gemm(&query, need)?;
    if compact.info().bytes != info.bytes
        || compact.info().weight_layouts != info.weight_layouts
        || compact.info().workspace != info.workspace
        || compact.info().guarantees != info.guarantees
    {
        return Err(Refusal("main compact changes layout, workspace or guarantees").into());
    }
    let tags = compact_tags(&backend)?;
    info.symbols.extend(
        compact
            .info()
            .symbols
            .iter()
            .filter(|(symbol, _)| tags.iter().any(|tag| tag.symbol == *symbol))
            .cloned(),
    );
    info.symbols.sort();
    info.symbols.dedup();
    for (rows, stages) in ranges {
        info.implementation
            .push_str(&format!(";dspark-main-compact:{rows:?}:stages={stages}"));
    }
    need.check(&info)?;
    Ok(Box::new(Main {
        fallback,
        compact,
        ranges: ranges.iter().map(|(rows, _)| rows.clone()).collect(),
        tags,
        info,
    }))
}

struct Main {
    fallback: Prepared<Buffers<View>>,
    compact: Prepared<Buffers<View>>,
    ranges: Vec<RangeInclusive<u32>>,
    tags: Vec<RouteTag>,
    info: OpInfo,
}

impl Main {
    fn admits(&self, counts: Counts) -> bool {
        self.ranges.iter().any(|rows| rows.contains(&counts.rows))
    }
}

impl PreparedOp<Buffers<View>> for Main {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = if self.admits(counts) {
            &self.compact
        } else {
            &self.fallback
        };
        op.launches(buffers, workspace, counts)
    }

    fn route(&self, counts: Counts, launch: &Launch) -> Option<RouteTag> {
        if !self.admits(counts) {
            return self.fallback.route(counts, launch);
        }
        self.tags
            .iter()
            .find(|tag| tag.symbol == launch.symbol)
            .cloned()
    }
}

#[cfg(test)]
pub(crate) fn route_tags(shape: &crate::shape::Shape) -> Vec<RouteTag> {
    let mut tags = Vec::new();
    for base in super::mxfp8_fold::prefill_entries(shape.geometry, shape.ranks()).unwrap() {
        let ranges = recipe(shape, base.query).unwrap();
        if ranges.is_empty() {
            continue;
        }
        let backend = compact_backend(shape, base.query, &ranges).unwrap();
        tags.extend(compact_tags(&backend).unwrap());
    }
    kernel_core::route::merge(tags).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_build::tuning::{Knob, Row};
    use model_deepseek41_contract::geometry::FLASH;

    /// The served `[129, 511]` row.
    fn upper_row() -> Row {
        let table = table(&FLASH, Family::DensePrefill).unwrap();
        table
            .rows()
            .iter()
            .find(|row| row.key.op == SITE && row.key.rows == Some([129, 511]))
            .unwrap()
            .clone()
    }

    /// `row` admitting `rows` on their own tile at `stages`.
    fn with_rows(row: &Row, rows: [u32; 2], stages: u32) -> Row {
        let mut row = row.clone();
        row.key.rows = Some(rows);
        row.knobs
            .insert("tile_rows".into(), Knob::Count(Entry::tile(rows[0])));
        row.knobs.insert("stages".into(), Knob::Count(stages));
        row
    }

    #[test]
    fn overlapping_unordered_or_empty_main_ranges_are_refused_without_panicking() {
        let upper = upper_row();
        let lower = with_rows(&upper, [65, 128], 3);
        assert_eq!(recipe_rows([&upper].into_iter()).unwrap(), [(129..=511, 2)]);
        assert_eq!(
            recipe_rows([&lower, &upper].into_iter()).unwrap(),
            [(65..=128, 3), (129..=511, 2)]
        );
        assert!(recipe_rows(std::iter::empty()).unwrap().is_empty());
        let overlapping = with_rows(&upper, [256, 300], 2);
        let empty = with_rows(&upper, [300, 299], 2);
        let spanning = with_rows(&upper, [65, 129], 2);
        for rows in [
            vec![&upper, &upper],
            vec![&upper, &overlapping],
            vec![&upper, &lower],
            vec![&empty],
            vec![&spanning],
        ] {
            assert!(recipe_rows(rows.into_iter()).is_err());
        }
    }

    fn views(base: &kernels_sm120::gemm::mxfp8_fold::Entry, rows: u32) -> Buffers<View> {
        let b = base.bytes(rows);
        let v = |handle, bytes| View { handle, bytes };
        Buffers {
            activations: v(1, b.activations),
            activation_scales: v(2, b.activation_scales),
            weights: v(3, b.weights),
            weight_scales: v(4, b.weight_scales),
            output: v(5, b.output),
            live_rows: v(6, 4),
            status: v(7, b.status),
            status_row_offset: 0,
        }
    }

    fn main_entry(shape: &crate::shape::Shape) -> kernels_sm120::gemm::mxfp8_fold::Entry {
        super::super::mxfp8_fold::prefill_entries(&FLASH, shape.ranks())
            .unwrap()
            .into_iter()
            .find(|e| e.query.n == 5120 && e.query.k == 15360)
            .unwrap()
    }

    /// Two ranges that share the stage-2 admission symbol: it admits both,
    /// and each compute symbol admits only its own range.
    #[test]
    fn ranges_sharing_an_admission_symbol_admit_their_union() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let backend = crate::Kernels::new(&shape).unwrap();
        let base = main_entry(&shape);
        let q = base.query;
        let upper = upper_row();
        let lower = with_rows(&upper, [65, 128], 2);
        let ranges = recipe_rows([&lower, &upper].into_iter()).unwrap();
        let old = backend
            .backend
            .mxfp8_fold_gemm(&q, Need::rows(q.m_capacity))
            .unwrap();
        let new = prepare_ranges(
            backend.backend.as_ref(),
            &shape,
            q,
            Need::rows(q.m_capacity),
            &ranges,
        )
        .unwrap();
        assert!(new.info().implementation.ends_with(
            ";dspark-main-compact:65..=128:stages=2;dspark-main-compact:129..=511:stages=2"
        ));
        let buffers = views(&base, q.m_capacity);
        let mut admitted = std::collections::BTreeMap::new();
        for count in [64, 65, 66, 127, 128, 129, 130, 510, 511, 512] {
            let counts = Counts::rows(count);
            let a = old.launches(&buffers, View::EMPTY, counts).unwrap();
            let b = new.launches(&buffers, View::EMPTY, counts).unwrap();
            if !(65..=511).contains(&count) {
                assert_eq!(a, b, "rows={count}");
                continue;
            }
            assert_eq!(b.len(), 2, "rows={count}");
            let tile = if count <= 128 { 32 } else { 64 };
            assert!(b[0].symbol.ends_with("_compact_m64_n128_s2_v1_admit_v1"));
            assert!(
                b[1].symbol
                    .ends_with(&format!("_compact_m{tile}_n128_s2_v1_packed_v1"))
            );
            for call in &b {
                let tag = new.route(counts, call).unwrap();
                assert_eq!(tag.class, RouteClass::Fast);
                assert_eq!(tag.entry, format!("dspark_main/compact/{}", call.symbol));
                admitted.insert(call.symbol.clone(), tag.admitted);
                assert!(
                    new.info()
                        .symbols
                        .iter()
                        .any(|(symbol, _)| *symbol == call.symbol)
                );
            }
        }
        let admitted: Vec<_> = admitted
            .into_iter()
            .map(|(symbol, rows)| (symbol.rsplit("_compact_").next().unwrap().to_owned(), rows))
            .collect();
        assert_eq!(
            admitted,
            [
                ("m32_n128_s2_v1_packed_v1".to_owned(), vec![65..=128]),
                ("m64_n128_s2_v1_admit_v1".to_owned(), vec![65..=511]),
                ("m64_n128_s2_v1_packed_v1".to_owned(), vec![129..=511]),
            ]
        );
        let tags = compact_tags(&compact_backend(&shape, q, &ranges).unwrap()).unwrap();
        for (symbol, rows) in &admitted {
            assert!(
                tags.iter()
                    .any(|tag| tag.symbol.ends_with(&format!("_compact_{symbol}"))
                        && tag.admitted == *rows)
            );
        }
        assert_eq!(tags.len(), admitted.len());
    }

    #[test]
    fn contiguous_main_range_keeps_every_other_launch_and_layout() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            let backend = crate::Kernels::new(&shape).unwrap();
            let base = main_entry(&shape);
            let q = base.query;
            let old = backend
                .backend
                .mxfp8_fold_gemm(&q, Need::rows(q.m_capacity))
                .unwrap();
            let new = prepare(
                backend.backend.as_ref(),
                &shape,
                q,
                Need::rows(q.m_capacity),
            )
            .unwrap();
            assert_eq!(new.info().bytes, old.info().bytes);
            assert_eq!(new.info().workspace, old.info().workspace);
            assert_eq!(new.info().weight_layouts, old.info().weight_layouts);
            assert_eq!(new.info().guarantees, old.info().guarantees);
            let buffers = views(&base, q.m_capacity);
            for count in 1..=q.m_capacity {
                let counts = Counts::rows(count);
                let a = old.launches(&buffers, View::EMPTY, counts).unwrap();
                let b = new.launches(&buffers, View::EMPTY, counts).unwrap();
                if ranks == Ranks::TP4 && (129..=511).contains(&count) {
                    assert_eq!(a.len(), 1);
                    assert_eq!(b.len(), 2);
                    assert!(
                        b.iter()
                            .all(|call| call.symbol.contains("_compact_m64_n128_s2_v1"))
                    );
                    for call in b {
                        let tag = new.route(counts, &call).unwrap();
                        assert_eq!(tag.class, RouteClass::Fast);
                        assert_eq!(tag.admitted, [129..=511]);
                        assert!(
                            new.info()
                                .symbols
                                .iter()
                                .any(|(symbol, _)| *symbol == call.symbol)
                        );
                    }
                } else {
                    assert_eq!(a, b, "rank={ranks:?} rows={count}");
                    for call in &b {
                        assert_eq!(new.route(counts, call), old.route(counts, call));
                    }
                }
            }
        }
    }
}
