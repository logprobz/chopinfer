//! The routed expert scale banks planned encoded from a host census
//! (`chop_bundle_weight_census_v1`): every encoded bank decodes to the raw
//! plan's bank, every other view keeps its bytes, and the plan refuses
//! counts measured for another census.
use super::*;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::precision::scale_offset;
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::moe;
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

fn image(checkpoint: Arc<DirectCheckpointV1>, plans: Vec<WeightPlanV1>) -> PlannedImageV1 {
    PlannedImageV1::new(
        checkpoint,
        plans,
        None,
        implementation_v1([0; 32]),
        DirectLoadOptionsV1 {
            threads_per_rank: 2,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )
    .unwrap()
}

fn encodable(name: &str) -> bool {
    name.starts_with("layers.") && name.contains(".ffn.experts.") && name.ends_with(".scale")
}

/// Checks every rank's encoded scale banks decode to the raw plan's banks
/// and, with `every_view`, that every other view keeps its bytes, and that
/// each bank's matrices are the shape the bundle expands at `geometry`
/// (`moe::scale_bank`). Returns each rank's raw and encoded routed scale
/// bytes.
fn check_banks(
    geometry: &Geometry,
    raw_image: &PlannedImageV1,
    raw_plans: &[WeightPlanV1],
    image: &PlannedImageV1,
    plans: &[WeightPlanV1],
    censuses: &[runtime_bundle_abi::weight_plan::WeightCensusV1],
    every_view: bool,
) -> Vec<(u64, u64)> {
    let ranks = plans.len();
    let (mut raw_start, mut start) = (0, 0);
    let mut totals = Vec::new();
    for rank in 0..ranks {
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
        let (mut raw_total, mut encoded_total, mut banks) = (0, 0, 0);
        for (name, view) in &encoded_index.views {
            let at = format!("tp{ranks} rank {rank} {name}");
            if image.mapped_view(rank as u32, name.as_bytes()).is_some()
                || !(every_view || encodable(name))
            {
                continue;
            }
            let raw_view = &raw_index.views[name];
            let mut raw = vec![0; raw_view.bytes as usize];
            raw_image
                .read(raw_start + raw_view.offset, &mut raw)
                .unwrap();
            let mut got = vec![0; view.bytes as usize];
            image.read(start + view.offset, &mut got).unwrap();
            if !encodable(name) {
                same(&got, &raw, &at);
                continue;
            }
            banks += 1;
            // The census measured every matrix of the bank, raw.
            let mut matrices = Vec::new();
            let mut covered = 0;
            while covered < raw.len() {
                let item = items
                    .next()
                    .unwrap_or_else(|| panic!("{at}: census ends early"));
                matrices.push((item.rows, item.groups));
                covered += (item.rows * item.groups) as usize;
            }
            assert_eq!(covered, raw.len(), "{at}: census matrices");
            let kind = if name.ends_with(".gate_up.scale") {
                GatedMlpProjection::GateUp
            } else {
                GatedMlpProjection::Down
            };
            let bank = moe::scale_bank(geometry, ranks as u32, moe::expert_sets(geometry)[0], kind)
                .unwrap();
            assert_eq!(
                matrices.len() as u64,
                u64::from(bank.experts * bank.projections),
                "{at}: matrices"
            );
            assert!(
                matrices
                    .iter()
                    .all(|&m| m == (u64::from(bank.rows), u64::from(bank.groups))),
                "{at}: {matrices:?} against {bank:?}"
            );
            assert_eq!(raw.len() as u64, bank.bytes(), "{at}");
            raw_total += raw.len() as u64;
            encoded_total += got.len() as u64;
            let Some(width) = (1..=4).find(|w| view.layout_id == encoded_scale_layout(*w)) else {
                // No width stored the bank smaller: it stays raw.
                assert_eq!(view.layout_id, raw_view.layout_id, "{at}");
                same(&got, &raw, &at);
                continue;
            };
            assert_eq!(view.dtype, "U8", "{at}");
            let mut decoded = Vec::with_capacity(raw.len());
            for (matrix, (rows, groups)) in matrices.iter().enumerate() {
                decoded.extend(
                    scale_offset::decode_bank_matrix(&got, matrix, *rows, *groups, width).unwrap(),
                );
            }
            same(&decoded, &raw, &at);
        }
        assert!(banks > 0, "tp{ranks} rank {rank}: no routed scale bank");
        assert!(
            items.next().is_none(),
            "tp{ranks} rank {rank}: census matrices left"
        );
        raw_start += raw_plans[rank].arena_bytes;
        start += plans[rank].arena_bytes;
        totals.push((raw_total, encoded_total));
    }
    totals
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
        let image = image(checkpoint, plans.clone());
        let totals = check_banks(
            geometry, &raw_image, &raw_plans, &image, &plans, &censuses, true,
        );
        let (raw, encoded) = totals
            .iter()
            .fold((0, 0), |(r, e), (raw, encoded)| (r + raw, e + encoded));
        eprintln!(
            "{} tp{ranks}: routed scale banks {raw} raw bytes, {encoded} encoded",
            geometry.id
        );
        assert!(encoded < raw);
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// Opt-in: every routed scale matrix of the official checkpoint at TP4,
/// planned encoded from its measured census and decoded bitwise against
/// the raw plan, reading only the scale planes. Reports each rank's bytes
/// and the census time.
#[test]
#[ignore = "reads every routed scale plane of the official checkpoint"]
fn official_routed_scale_banks_round_trip() {
    let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT") else {
        eprintln!("CHOPINFER_DEEPSEEK_CHECKPOINT is unset: skipped");
        return;
    };
    let ranks = 4;
    let checkpoint = Arc::new(DirectCheckpointV1::open(Path::new(&root)).unwrap());
    let started = std::time::Instant::now();
    let (requests, censuses) = measured_requests(&checkpoint, ranks);
    let census_seconds = started.elapsed().as_secs_f64();
    let implementation = implementation_v1([0; 32]);
    let raw_plans: Vec<_> = (0..ranks)
        .map(|rank| {
            plan(&checkpoint.request(RANK_IMAGE_ROLE, ranks, rank, implementation, vec![], vec![]))
                .unwrap()
                .unwrap()
        })
        .collect();
    let plans: Vec<_> = requests
        .iter()
        .map(|request| plan(request).unwrap().unwrap())
        .collect();
    let raw_image = image(checkpoint.clone(), raw_plans.clone());
    let encoded_image = image(checkpoint, plans.clone());
    let totals = check_banks(
        &FLASH,
        &raw_image,
        &raw_plans,
        &encoded_image,
        &plans,
        &censuses,
        false,
    );
    for (rank, (raw, encoded)) in totals.iter().enumerate() {
        eprintln!("tp4 rank {rank}: routed scale banks {raw} raw bytes, {encoded} encoded");
    }
    eprintln!("census of four ranks: {census_seconds:.2} s");
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
