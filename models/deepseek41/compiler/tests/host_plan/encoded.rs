//! The routed expert scale banks planned encoded from a host census
//! (`chop_bundle_weight_census_v1`): every encoded bank decodes to the raw
//! plan's bank, every other view keeps its bytes, and the plan refuses
//! counts measured for another census.
use super::*;
use kernel_core::precision::scale_offset;
use model_deepseek41_contract::weight_plan::{census, encoded_scale_layout};
use runtime_bundle_abi::weight_plan::WeightPlanRequestV1;
use runtime_host::direct::census_counts_v1;

/// Every rank's request carrying every rank's measured census.
fn measured_requests(
    checkpoint: &DirectCheckpointV1,
    ranks: u32,
) -> (
    Vec<WeightPlanRequestV1>,
    Vec<runtime_bundle_abi::weight_plan::WeightCensusV1>,
) {
    let implementation = implementation_v1([0; 32]);
    let mut requests = Vec::new();
    let mut documents = Vec::new();
    for rank in 0..ranks {
        let request =
            checkpoint.request(RANK_IMAGE_ROLE, ranks, rank, implementation, vec![], vec![]);
        let measured = census(&request)
            .unwrap()
            .expect("routed scale banks to measure");
        let bytes = measured.encode();
        documents.push((rank, measured, bytes));
        requests.push(request);
    }
    let (counts, unique, read) = census_counts_v1(checkpoint, &documents).unwrap();
    assert!(unique > 0 && read > 0);
    for request in &mut requests {
        request.census.clone_from(&counts);
    }
    (requests, documents.into_iter().map(|(_, c, _)| c).collect())
}

#[test]
fn encoded_scale_banks_decode_to_the_raw_banks() {
    let Some(geometry) = GEOMETRIES.iter().find(|g| g.id != "flash") else {
        eprintln!("no synthetic geometry is compiled: skipped");
        return;
    };
    let root = varied_checkpoint(geometry, "census");
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let (raw_image, raw_plans) = prepare(&root, ranks).unwrap();
        let checkpoint = Arc::new(DirectCheckpointV1::open(&root).unwrap());
        let (requests, censuses) = measured_requests(&checkpoint, ranks);
        let plans: Vec<_> = requests
            .iter()
            .map(|request| plan(request).unwrap().unwrap())
            .collect();
        let implementation = implementation_v1([0; 32]);
        let image = PlannedImageV1::new(
            checkpoint,
            plans.clone(),
            None,
            implementation,
            DirectLoadOptionsV1 {
                threads_per_rank: 2,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )
        .unwrap();
        let (mut raw_start, mut start) = (0, 0);
        let (mut raw_total, mut encoded_total) = (0, 0);
        for rank in 0..ranks as usize {
            let index = |plan: &WeightPlanV1| {
                bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                    .unwrap()
            };
            let (raw_index, encoded_index) = (index(&raw_plans[rank]), index(&plans[rank]));
            assert_eq!(
                raw_index.views.keys().collect::<Vec<_>>(),
                encoded_index.views.keys().collect::<Vec<_>>()
            );
            // Census items, in bank (view name) order and matrix order.
            let mut items = censuses[rank].items.iter();
            let mut banks = 0;
            for (name, view) in &encoded_index.views {
                let at = format!("tp{ranks} rank {rank} {name}");
                if image.mapped_view(rank as u32, name.as_bytes()).is_some() {
                    continue;
                }
                let raw_view = &raw_index.views[name];
                let mut raw = vec![0; raw_view.bytes as usize];
                raw_image
                    .read(raw_start + raw_view.offset, &mut raw)
                    .unwrap();
                let mut got = vec![0; view.bytes as usize];
                image.read(start + view.offset, &mut got).unwrap();
                let encodable = name.starts_with("layers.")
                    && name.contains(".ffn.experts.")
                    && name.ends_with(".scale");
                if !encodable {
                    same(&got, &raw, &at);
                    continue;
                }
                banks += 1;
                // The census measured every matrix of the bank, raw.
                let mut decoded = Vec::new();
                let mut matrices = Vec::new();
                while decoded.len() < raw.len() {
                    let item = items
                        .next()
                        .unwrap_or_else(|| panic!("{at}: census ends early"));
                    matrices.push((item.rows, item.groups));
                    decoded.extend(vec![0; (item.rows * item.groups) as usize]);
                }
                assert_eq!(decoded.len(), raw.len(), "{at}: census matrices");
                let width = (1..=4).find(|w| view.layout_id == encoded_scale_layout(*w));
                let Some(width) = width else {
                    // No width stored the bank smaller: it stays raw.
                    assert_eq!(view.layout_id, raw_view.layout_id, "{at}");
                    same(&got, &raw, &at);
                    continue;
                };
                assert_eq!(view.dtype, "U8", "{at}");
                let mut at_byte = 0;
                for (matrix, (rows, groups)) in matrices.iter().enumerate() {
                    let scales =
                        scale_offset::decode_bank_matrix(&got, matrix, *rows, *groups, width)
                            .unwrap();
                    decoded[at_byte..at_byte + scales.len()].copy_from_slice(&scales);
                    at_byte += scales.len();
                }
                same(&decoded, &raw, &at);
                raw_total += raw.len();
                encoded_total += got.len();
            }
            assert!(banks > 0, "tp{ranks} rank {rank}: no routed scale bank");
            assert!(
                items.next().is_none(),
                "tp{ranks} rank {rank}: census matrices left"
            );
            raw_start += raw_plans[rank].arena_bytes;
            start += plans[rank].arena_bytes;
        }
        eprintln!(
            "{} tp{ranks}: routed scale banks {raw_total} raw bytes, {encoded_total} encoded",
            geometry.id
        );
        assert!(encoded_total < raw_total);
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn plans_refuse_counts_for_another_census() {
    let Some(geometry) = GEOMETRIES.iter().find(|g| g.id != "flash") else {
        eprintln!("no synthetic geometry is compiled: skipped");
        return;
    };
    let root = sparse(geometry);
    let checkpoint = DirectCheckpointV1::open(&root).unwrap();
    let request = checkpoint.request(
        RANK_IMAGE_ROLE,
        1,
        0,
        implementation_v1([0; 32]),
        vec![],
        vec![],
    );
    let measured = census(&request).unwrap().unwrap();
    let mut stale = request.clone();
    stale.census = vec![runtime_bundle_abi::weight_plan::CensusRankV1 {
        rank: 0,
        document_sha256: [0; 32],
        counts: vec![[0; 3]; measured.items.len()],
    }];
    let error = plan(&stale).unwrap_err();
    assert!(error.contains("answer another census"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}
