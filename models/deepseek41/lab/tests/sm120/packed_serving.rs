//! Every production fold symbol against its row-major twin at exact shapes.
#[path = "../../sm120/packed_serving.rs"]
mod family;
use family::Entry;
use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use kernel_harness::{
    LabResult, Session,
    packed_fold::{self, Image},
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::{FusedBuffers, Quantized};
use kernels_sm120::gemm::{mxfp8_fold_packed, mxfp8_prefill, mxfp8_prefill_split};
use std::collections::{BTreeMap, BTreeSet};

const CUBINS: &str = "CHOPINFER_DEEPSEEK41_PACKED_SERVING_CUBINS";
const GPU_BUDGET_SECONDS: u64 = 600;

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}
fn buffers(entry: &Entry, rows: u32) -> [Buffers<View>; 2] {
    let size = entry.base().bytes(rows);
    let first = Buffers {
        activations: view(1, size.activations),
        activation_scales: view(2, size.activation_scales),
        weights: view(3, size.weights),
        weight_scales: view(4, size.weight_scales),
        output: view(5, size.output),
        live_rows: view(6, 4),
        status: view(7, size.status + 12),
        status_row_offset: if matches!(entry, Entry::Wide(_)) {
            0
        } else {
            3
        },
    };
    [
        first,
        Buffers {
            weights: view(8, size.weights),
            weight_scales: view(9, size.weight_scales),
            output: view(10, size.output),
            ..first
        },
    ]
}
fn fused_buffers(
    entry: &kernels_sm120::gemm::mxfp8_fold_cluster_fused::Entry,
    fold: Buffers<View>,
    rows: u32,
) -> FusedBuffers {
    let columns =
        u64::from(rows) * u64::from(entry.cluster.base.query.n * entry.cluster.base.query.groups);
    FusedBuffers {
        fold,
        quantized: entry.quantize.then_some(Quantized {
            codes: view(12, columns),
            scales: view(13, columns / 32),
            flags: view(14, columns / 32),
        }),
        pending: (entry.pending != 0)
            .then_some(view(15, u64::from(rows) * u64::from(entry.pending))),
    }
}

fn launches(entry: &Entry, rows: u32) -> LabResult<([Vec<Launch>; 2], u64)> {
    let q = entry.base().query;
    let b = buffers(entry, rows);
    let need = Need::rows(rows);
    let counts = Counts::rows(rows);
    let issue = |op: Prepared<Buffers<View>>| -> LabResult<(Vec<Launch>, u64)> {
        let bytes = op.info().workspace;
        Ok((
            op.launches(&b[0], view(11, bytes), counts)
                .map_err(|e| e.to_string())?,
            bytes,
        ))
    };
    let result = match entry {
        Entry::Fold(e) => issue(
            Sm120::new()
                .with_mxfp8_fold(e.clone())
                .mxfp8_fold_gemm(&q, need)
                .map_err(|e| e.to_string())?,
        ),
        Entry::Ticket(e) => {
            let backend = Sm120::new().with_mxfp8_fold_ticket(e.clone());
            if e.paired {
                let op = backend
                    .mxfp8_fold_gemm_pair(&q, need)
                    .map_err(|e| e.to_string())?;
                let bytes = op.info().workspace;
                Ok((
                    op.launches(&b, view(11, bytes), counts)
                        .map_err(|e| e.to_string())?,
                    bytes,
                ))
            } else {
                issue(
                    backend
                        .mxfp8_fold_gemm(&q, need)
                        .map_err(|e| e.to_string())?,
                )
            }
        }
        Entry::Decode(e) => {
            let backend = Sm120::new().with_mxfp8_fold_decode(e.clone());
            if e.paired {
                let op = backend
                    .mxfp8_fold_gemm_pair(&q, need)
                    .map_err(|e| e.to_string())?;
                let bytes = op.info().workspace;
                Ok((
                    op.launches(&b, view(11, bytes), counts)
                        .map_err(|e| e.to_string())?,
                    bytes,
                ))
            } else {
                issue(
                    backend
                        .mxfp8_fold_gemm(&q, need)
                        .map_err(|e| e.to_string())?,
                )
            }
        }
        Entry::Reuse(e) => {
            let mut exact = e.clone();
            exact.ticket.query.m_capacity = rows;
            let folds = bundle_deepseek41::sm120::projection_reuse::choice(q, rows)
                .filter(|(tile, _)| *tile == e.tile_n)
                .map_or(4, |(_, folds)| folds);
            issue(
                Sm120::new()
                    .with_mxfp8_fold_reuse(exact.clone(), folds)
                    .mxfp8_fold_gemm(&exact.ticket.query, need)
                    .map_err(|e| e.to_string())?,
            )
        }
        Entry::Prefill(e) => issue(
            Sm120::new()
                .with_mxfp8_prefill(e.clone())
                .mxfp8_prefill_gemm(&q, need)
                .map_err(|e| e.to_string())?,
        ),
        Entry::Split(e) => issue(
            Sm120::new()
                .with_mxfp8_prefill_split(e.clone())
                .mxfp8_prefill_gemm(&q, need)
                .map_err(|e| e.to_string())?,
        ),
        Entry::Compact(e, _) => issue(e.prepare(rows).map_err(|e| e.to_string())?),
        Entry::Matrix(e) => issue(e.prepare(rows).map_err(|e| e.to_string())?),
        Entry::Scales(e) => issue(e.prepare(rows).map_err(|e| e.to_string())?),
        Entry::Wide(e) => issue(e.prepare(need).map_err(|e| e.to_string())?),
        Entry::Fused(e) => {
            let op = e.prepare(need).map_err(|e| e.to_string())?;
            let bytes = op.info().workspace;
            Ok((
                op.launches(&fused_buffers(e, b[0], rows), view(11, bytes), counts)
                    .map_err(|e| e.to_string())?,
                bytes,
            ))
        }
    }?;
    let (row_major, workspace) = result;
    let packed = match entry {
        Entry::Prefill(e) => {
            mxfp8_prefill::packed_launches(e, row_major.clone()).map_err(|e| e.to_string())?
        }
        Entry::Split(e) => {
            mxfp8_prefill_split::packed_launches(e, row_major.clone()).map_err(|e| e.to_string())?
        }
        Entry::Scales(e) => e
            .packed_launches(row_major.clone())
            .map_err(|e| e.to_string())?,
        _ => {
            let readers: BTreeSet<_> = entry
                .packed_specs()
                .into_iter()
                .map(|s| {
                    s.symbol
                        .strip_suffix(mxfp8_fold_packed::SUFFIX)
                        .unwrap()
                        .to_owned()
                })
                .collect();
            mxfp8_fold_packed::launches(row_major.clone(), |s| readers.contains(s))
        }
    };
    Ok(([row_major, packed], workspace))
}

fn images(entry: &Entry, rows: u32, workspace: u64, seed: u64) -> LabResult<Vec<Image>> {
    let [b, second] = buffers(entry, rows);
    let q = entry.base().query;
    let layout = Mxfp8FoldLayout {
        groups: q.groups as usize,
        n: q.n as usize,
        k: q.k as usize,
    };
    let mut status = vec![0; b.status.bytes as usize];
    let offset = b.status_row_offset as usize * 4;
    status[..offset].fill(0x5a);
    if rows > 1 {
        status[offset + 4] = 4;
    }
    let mut result = vec![
        Image::data(1, packed_fold::codes(b.activations.bytes as usize, seed)),
        Image::data(
            2,
            packed_fold::scales(b.activation_scales.bytes as usize, seed + 1),
        ),
        Image::codes(3, layout, seed + 2)?,
        Image::data(
            4,
            packed_fold::scales(b.weight_scales.bytes as usize, seed + 3),
        ),
        Image::data(5, vec![0xa5; b.output.bytes as usize]),
        Image::data(6, rows.to_le_bytes().to_vec()),
        Image::data(7, status),
        Image::data(11, vec![0xa5; workspace.max(4) as usize]),
    ];
    if entry.paired() {
        result.extend([
            Image::codes(8, layout, seed + 4)?,
            Image::data(
                9,
                packed_fold::scales(second.weight_scales.bytes as usize, seed + 5),
            ),
            Image::data(10, vec![0xa5; second.output.bytes as usize]),
        ]);
    }
    if let Entry::Fused(e) = entry {
        let f = fused_buffers(e, b, rows);
        for view in f
            .quantized
            .iter()
            .flat_map(|o| [o.codes, o.scales, o.flags])
        {
            result.push(Image::data(view.handle, vec![0xa5; view.bytes as usize]));
        }
        if let Some(view) = f.pending {
            let mut flags = vec![0; view.bytes as usize];
            if rows > 2 {
                flags[2 * e.pending as usize] = 1;
            }
            result.push(Image::data(view.handle, flags));
        }
    }
    Ok(result)
}

fn load(session: &Session) -> LabResult<runtime_cuda::Module> {
    let directory = std::env::var_os(CUBINS)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| format!("{CUBINS} is required"))?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    session.load_cubin(
        &std::fs::read(directory.join(format!("{}.cubin", family::CUBIN)))
            .map_err(|e| e.to_string())?,
    )
}

#[test]
fn packed_serving_catalog_preserves_specs_and_covers_every_symbol() {
    assert_eq!(family::FAMILY, "packed_serving");
    let cases = family::cases();
    let cubin = family::cubins().remove(0);
    let specs: BTreeMap<_, _> = cubin
        .entries
        .iter()
        .map(|s| (s.symbol.as_str(), s))
        .collect();
    let mut covered = BTreeSet::new();
    let mut topologies = BTreeSet::new();
    let mut families = BTreeMap::<_, usize>::new();
    let mut comparisons = 0;
    let mut upload_bytes = 0_u64;
    for case in &cases {
        topologies.extend(&case.topologies);
        *families.entry(case.entry.family()).or_default() += 1;
        for packed in case.entry.packed_specs() {
            let original = packed
                .symbol
                .strip_suffix(mxfp8_fold_packed::SUFFIX)
                .unwrap();
            let row = specs[original];
            let index = row.body.rfind(">(").expect("template call");
            let expected = format!("{},true{}", &row.body[..index], &row.body[index..]);
            assert_eq!(packed.body, expected, "{}", packed.symbol);
            assert_eq!(packed.params, row.params);
            assert_eq!(packed.max_threads, row.max_threads);
            assert_eq!(packed.min_blocks, row.min_blocks);
            assert_eq!(packed.allowed_local_bytes, row.allowed_local_bytes);
            assert!(specs.contains_key(packed.symbol.as_str()));
        }
        for rows in case.entry.rows() {
            let ([old, packed], workspace) = launches(&case.entry, rows).unwrap();
            assert_eq!(old.len(), packed.len());
            for (a, b) in old.iter().zip(&packed) {
                assert_eq!(a.plan, b.plan);
                assert!(specs.contains_key(a.symbol.as_str()), "{}", a.symbol);
                assert!(specs.contains_key(b.symbol.as_str()), "{}", b.symbol);
                if b.symbol.ends_with(mxfp8_fold_packed::SUFFIX) {
                    covered.insert(b.symbol.clone());
                }
            }
            let size = case.entry.base().bytes(rows);
            upload_bytes += 2
                * (size.activations
                    + size.activation_scales
                    + size.weights
                    + size.weight_scales
                    + size.output
                    + size.status
                    + workspace);
            comparisons += 1;
        }
    }
    let expected: BTreeSet<_> = cases
        .iter()
        .flat_map(|case| case.entry.packed_specs())
        .map(|s| s.symbol)
        .collect();
    assert_eq!(covered, expected);
    assert_eq!(topologies, BTreeSet::from([1, 2, 4]));
    assert_eq!(families.len(), 12);
    println!(
        "packed_serving: {} symbols, {comparisons} comparisons, {upload_bytes} nominal upload bytes, families={families:?}, budget={GPU_BUDGET_SECONDS}s",
        covered.len()
    );
}

#[test]
fn packed_serving_equals_row_major_on_gpu() {
    kernel_harness::gpu_test(4 << 30, |session| {
        let module = load(session)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        let start = std::time::Instant::now();
        for (index, case) in family::cases().iter().enumerate() {
            for rows in case.entry.rows() {
                if start.elapsed().as_secs() >= GPU_BUDGET_SECONDS {
                    return Err(
                        "packed serving parity exceeded its budget; coverage is incomplete".into(),
                    );
                }
                let ([old, packed], workspace) = launches(&case.entry, rows)?;
                let images = images(&case.entry, rows, workspace, index as u64 + 1)?;
                packed_fold::compare(session, &function, &old, &packed, &images).map_err(|e| {
                    format!(
                        "{} rows={rows} TP{:?}: {e}",
                        case.entry.key(),
                        case.topologies
                    )
                })?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn packed_serving_detects_a_moved_strip_per_family_on_gpu() {
    kernel_harness::gpu_test(4 << 30, |session| {
        let module = load(session)?;
        let function = |s: &str| module.function(s).map_err(|e| e.to_string());
        let mut families = BTreeSet::new();
        for case in family::cases() {
            if !families.insert(case.entry.family()) {
                continue;
            }
            let rows = case.entry.rows()[0];
            let ([old, packed], workspace) = launches(&case.entry, rows)?;
            let mut images = images(&case.entry, rows, workspace, 7)?;
            let strip = case.entry.base().query.k as usize * 8;
            let codes = images
                .iter_mut()
                .find(|image| image.handle == 3)
                .unwrap()
                .packed
                .as_mut()
                .unwrap();
            let (left, right) = codes.split_at_mut(strip);
            left.swap_with_slice(&mut right[..strip]);
            let error =
                packed_fold::compare(session, &function, &old, &packed, &images).unwrap_err();
            assert!(
                error.starts_with("view 5 byte"),
                "{}: {error}",
                case.entry.family()
            );
        }
        assert_eq!(families.len(), 12);
        Ok(())
    })
    .unwrap();
}
