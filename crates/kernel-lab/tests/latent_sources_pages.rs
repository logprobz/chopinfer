//! Native compressed source pages, boundaries, holes and address mutants.
use kernel_core::backend::{Backend, Need};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery};
use kernel_lab::families::latent_sources_pages as family;
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, cases};
use kernels_sm120::backend::Sm120;
use std::collections::BTreeMap;

fn backend() -> Sm120 {
    family::catalog()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_latent_sources)
}
fn check(report: &kit::Report, gpu: bool) {
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.ops.len(), family::REQUIRED);
    assert_eq!(report.prepared().len(), family::REQUIRED);
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required page entry refused")
        };
        for check in checks.iter().filter(|c| gpu || c.name == "declaration") {
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {check:?}",
                op.name
            );
        }
    }
}
#[test]
fn native_page_geometries_prepare_all_required_cases() {
    for nonzero in [false, true] {
        check(
            &kit::conform(
                &backend(),
                &LabModules,
                None,
                &cases::latent_sources::paged_family(&family::catalog(), nonzero),
            )
            .unwrap(),
            false,
        );
    }
}
#[test]
fn native_page_geometries_meet_f64_bounds_with_replay_and_pdl() {
    kernel_lab::gpu_test(1 << 30, |session| {
        for nonzero in [false, true] {
            let report = kit::conform(
                &backend(),
                &LabModules,
                Some(session),
                &cases::latent_sources::paged_family(&family::catalog(), nonzero),
            )?;
            check(&report, true);
            session.receipt(
                "native-latent-pages",
                BTreeMap::from([
                    ("nonzero".into(), serde_json::json!(nonzero)),
                    ("report".into(), report.json()),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("page-specific native source conformance");
}

#[test]
fn native_page_address_mutants_are_rejected_without_device_faults() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = family::catalog()
            .into_iter()
            .find(|e| {
                matches!(
                    e.query,
                    LatentAttentionQuery::Attention {
                        page_size: 32,
                        phase: AttentionPhase::Decode,
                        output: LatentAttentionOutput::Normalized,
                        ..
                    }
                )
            })
            .unwrap();
        let prepared = backend()
            .latent_attention_sources(&entry.query, Need::rows(4))
            .unwrap();
        let case = cases::latent_sources::paged_case(entry.query, 4, prepared)?;
        let mut images = case.images();
        let (width, page) = (512, 32);
        let stride = page * (width / 2 + width / 16) + 64;
        let mut pool = vec![0x38; images.bytes("compressed.cache.pool")?.len()];
        // One extra physical page backs the wrong scale-offset reads.
        // Every mutant address contains finite bytes, including guard bytes.
        for physical in 0..pool.len() / stride {
            pool[physical * stride..physical * stride + page * width / 2]
                .fill([0x22, 0x44, 0x66, 0x77][physical % 4]);
        }
        let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        images = images
            .with("compressed.cache.pool", pool)
            .with("q", vec![0; 4 * 4 * width * 2])
            .with("row_batch", i32s(&[0; 4]))
            .with("window.key_lists", i32s(&vec![-1; 4 * 128]))
            .with("sinks", vec![0; 4 * 4])
            .with(
                "cos_sin",
                [1.0_f32, 0.0]
                    .repeat(4 * 32)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            );
        let functions = LabModules.load(session)?;
        let correct = case.launch(session, functions.as_ref(), &images, 1)?;
        case.judge_outputs(&images, 1, &correct)?;
        let values = correct["output"]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        assert!(
            values[..4 * width]
                .iter()
                .all(|&v| v == f32_to_bf16(11.0 / 4.0))
        );
        let mut rejected = Vec::new();
        for mutant in family::mutants(&entry) {
            let source = entry.symbol();
            let wrong = |symbol: &str| {
                functions(if symbol == source {
                    &mutant.symbol
                } else {
                    symbol
                })
            };
            let actual = case.launch(session, &wrong, &images, 1)?;
            assert!(
                actual["output"][..4 * width * 2]
                    .chunks_exact(2)
                    .all(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])).is_finite())
            );
            let failure = case
                .judge_outputs(&images, 1, &actual)
                .expect_err("address mutant must fail the original f64 bound");
            rejected.push((mutant.symbol, failure));
        }
        assert_eq!(rejected.len(), 3);
        session.receipt(
            "native-latent-page-mutants",
            BTreeMap::from([("rejected".into(), serde_json::json!(rejected))]),
        )?;
        Ok(())
    })
    .expect("safe address mutants");
}

#[path = "latent_sources/address.rs"]
mod address;

fn repacked_case(
    entry: &kernels_sm120::latent::sources::Entry,
) -> kernel_lab::LabResult<(kit::OpCase<'static>, kernel_lab::conformance::Images)> {
    let LatentAttentionQuery::Attention { page_size, .. } = entry.query else {
        unreachable!()
    };
    let (page, width, items, rows) = (page_size as usize, 512, 3, 4);
    let prepared = backend()
        .latent_attention_sources(&entry.query, Need::rows(rows as u32))
        .unwrap();
    let case = cases::latent_sources::paged_case(entry.query, rows as u32, prepared)?;
    let mut images = case.images();
    let stride = page * (width / 2 + width / 16) + 64;
    let mut pool = vec![0xa5; images.bytes("compressed.cache.pool")?.len()];
    for item in 0..items {
        for logical_page in 0..4 {
            let base = (item * 4 + logical_page) * stride;
            for slot in 0..page {
                let position = logical_page * page + slot;
                for col in 0..width / 2 {
                    pool[base + slot * width / 2 + col] = ((position + col + item) % 16) as u8
                        | (((position * 3 + col + item) % 16) as u8) << 4;
                }
                for block in 0..width / 16 {
                    pool[base + page * width / 2 + slot * width / 16 + block] =
                        0x30 + ((position + block + item) % 8) as u8;
                }
            }
        }
    }
    let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let mut lists = vec![-1; rows * 128];
    for row in 0..rows {
        for (at, position) in [(0, 31), (65, 32), (66, 63), (127, 64)] {
            lists[row * 128 + at] = position;
        }
    }
    images = images
        .with("compressed.cache.pool", pool)
        .with(
            "compressed.block_tables.tables",
            i32s(&(0..12).collect::<Vec<_>>()),
        )
        .with("compressed.key_lists", i32s(&lists));
    assert!(images.u16s("q")?.iter().any(|&q| q != 0));
    Ok((case, images))
}

#[test]
fn native_repacking_preserves_decoded_rows() {
    use kernel_core::latent_attention::LatentKvFormat;
    use kernel_core::latent_attention_oracle::decode_native_page;
    let mut packed = Vec::new();
    for page in [32, 64] {
        let entry = family::catalog()
            .into_iter()
            .find(|e| {
                matches!(e.query,
            LatentAttentionQuery::Attention { page_size, .. } if page_size == page)
            })
            .unwrap();
        let (_, images) = repacked_case(&entry).unwrap();
        let page = page as usize;
        let stride = page * 288 + 64;
        let pool = images.bytes("compressed.cache.pool").unwrap();
        let mut values = Vec::new();
        for item in 0..3 {
            for position in [31, 32, 63, 64] {
                let base = (item * 4 + position / page) * stride;
                values.extend(
                    decode_native_page(
                        LatentKvFormat::Fp4E2M1E4M3Block16,
                        512,
                        page,
                        &pool[base..base + page * 288],
                        position % page,
                    )
                    .unwrap(),
                );
            }
        }
        packed.push(values);
    }
    assert_eq!(packed[0], packed[1]);
}

#[test]
fn native_repacked_page_geometries_are_bitwise_equal() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        let mut compared = 0;
        for phase in [AttentionPhase::Prefill, AttentionPhase::Decode, AttentionPhase::Verify] {
            for output in [LatentAttentionOutput::Normalized, LatentAttentionOutput::Partial] {
                let mut previous = None;
                for page in [32, 64] {
                    let entry = family::catalog().into_iter().find(|e| matches!(e.query,
                        LatentAttentionQuery::Attention { page_size, phase: p, output: o, .. } if page_size == page && p == phase && o == output)).unwrap();
                    let (case, images) = repacked_case(&entry)?;
                    let actual = case.launch(session, functions.as_ref(), &images, 4)?;
                    case.judge_outputs(&images, 4, &actual)?;
                    let names = if output == LatentAttentionOutput::Partial {
                        &["partial.max", "partial.den", "partial.num"][..]
                    } else { &["output"][..] };
                    let bits = names.iter().map(|name| actual[*name].clone()).collect::<Vec<_>>();
                    if let Some(old) = &previous { assert_eq!(&bits, old, "{phase:?}, {output:?}"); }
                    previous = Some(bits);
                }
                compared += 1;
            }
        }
        assert_eq!(compared, 6);
        session.receipt("native-latent-page-repacking", BTreeMap::from([("bitwise_pairs".into(), serde_json::json!(compared))]))?;
        Ok(())
    }).expect("same decoded operands and accumulation across page geometries");
}

#[test]
fn native_unadmitted_page_ids_are_holes() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        for page in [32, 64] {
            let entry = family::catalog().into_iter().find(|e| matches!(e.query,
                LatentAttentionQuery::Attention { page_size, phase: AttentionPhase::Decode, output: LatentAttentionOutput::Normalized, .. } if page_size == page)).unwrap();
            let (case, images) = repacked_case(&entry)?;
            let pages = images.bytes("compressed.cache.pool")?.len() / (page as usize * 288 + 64);
            let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
            for id in [pages as i32, i32::MAX] {
                let images = images.clone().with("window.key_lists", i32s(&vec![-1; 4 * 128]))
                    .with("compressed.block_tables.tables", i32s(&[id; 12]));
                let actual = case.launch(session, functions.as_ref(), &images, 4)?;
                case.judge_outputs(&images, 4, &actual)?;
                assert!(actual["output"].iter().all(|&byte| byte == 0));
            }
        }
        Ok(())
    }).expect("unadmitted page ids do not read storage");
}
