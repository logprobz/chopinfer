//! Full vocabulary shards through the serving F32 entries. The repeating
//! weight bank bounds CPU oracle work without sampling GPU output columns.
use bundle_deepseek41::sm120::{
    f32_head as registration, fold_cubin_for_symbol, register_f32_projections,
};
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, GemmQuery, Refusal, WeightFormat};
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights};
use kernel_core::precision::{OperationSpec, OperationUse, execution::Binding, f32_gemm};
use kernel_core::properties::Invariance;
use kernel_harness::conformance::{Expected, Images, Probes, Rows};
use kernel_harness::kit::cases::f32_projection::{self as cases, Witness};
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpCase, OpResult, Outcome, Required, Spec};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::{F32ProjectionOffers, Sm120};
use kernels_sm120::f32_projection::Entry;
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::serving::{DECODE_BATCH, SCORED_ROWS};
use std::cell::RefCell;
use std::collections::BTreeMap;

use super::{head_artifacts, head_probes as probes};

const BANK: usize = 17;
const LIVE: &[u32] = &[
    0,
    1,
    2,
    7,
    8,
    9,
    DECODE_BATCH - 1,
    DECODE_BATCH,
    DECODE_BATCH + 1,
    SCORED_ROWS - 1,
    SCORED_ROWS,
];

struct ModelModules;
impl Modules for ModelModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let loaded = RefCell::new(BTreeMap::new());
        Ok(Box::new(move |symbol| {
            let (name, bytes) = fold_cubin_for_symbol(symbol)
                .or_else(|| {
                    head_artifacts::cubins()
                        .iter()
                        .find(|(_, _, symbols)| symbols.iter().any(|s| s == symbol))
                        .map(|(name, bytes, _)| (name.as_str(), bytes.as_slice()))
                })
                .ok_or_else(|| format!("no head cubin for {symbol}"))?;
            let mut modules = loaded.borrow_mut();
            if !modules.contains_key(name) {
                modules.insert(name, session.load_cubin(bytes)?);
            }
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let (bytes, _) = bundle_deepseek41::sm120::dependency_probes();
        let module = session.load_cubin(bytes)?;
        let function = |name| module.function(name).map_err(|e| e.to_string());
        Ok(Some(Probes {
            live: LiveProducer::with_function(session, function(kernels_sm120::probes::LIVE)?)?,
            copy: Some(CopyProducer::with_function(function(
                kernels_sm120::probes::COPY,
            )?)),
        }))
    }
}

fn lab_backend(entry: Entry) -> Sm120 {
    let (_, bytes, symbols) = head_artifacts::cubins()
        .iter()
        .find(|(_, _, symbols)| symbols.iter().any(|s| s == entry.symbol()))
        .unwrap();
    Sm120::new().with_f32_projections(
        F32ProjectionOffers::new(
            vec![entry],
            symbols.iter().cloned().collect(),
            kernel_build::digest(bytes),
        )
        .unwrap(),
    )
}

fn heads() -> Vec<(Entry, Sm120)> {
    let mut entries = Vec::new();
    for g in [FLASH, probes::synthetic()] {
        let declaration = model_deepseek41_contract::projections::declaration(&g).unwrap();
        let roles = model_deepseek41_contract::projections::header_roles(&declaration);
        // Checkpoint binding uses these same header roles. The head is BF16,
        // unlike the block-scaled matrices elsewhere in the Flash checkpoint.
        assert_eq!(
            roles["head"].declared.as_ref().unwrap().weight,
            WeightFormat::Bf16
        );
        for &ranks in SUPPORTED_RANKS {
            let entry = registration::entry(&g, ranks).unwrap();
            assert_eq!(entry.query().weight, WeightFormat::Bf16);
            let backend = if g.base {
                assert!(fold_cubin_for_symbol(entry.symbol()).is_some());
                register_f32_projections(Sm120::new(), &g, ranks).unwrap()
            } else {
                lab_backend(entry.clone())
            };
            entries.push((entry, backend));
        }
    }
    assert_eq!(entries.len(), 6);
    entries
}

fn scratch<T>(mut unused: impl FnMut() -> T) -> ProjectionScratch<T> {
    ProjectionScratch {
        weight_domain: None,
        partial: unused(),
        semaphores: unused(),
        unused: unused(),
        temporary: None,
        gathered: None,
        codes: None,
        scales: None,
        activation_global: None,
        row_amax: None,
        output_codes: None,
        output_scales: None,
        quantizer_status: None,
        quantizer_status_row_offset: 0,
    }
}

fn weight_bank(k: usize) -> Vec<f32> {
    (0..BANK * k)
        .map(|i| {
            let (n, c) = (i / k, i % k);
            bf16_to_f32(f32_to_bf16(((n * 31 + c * 17) % 251) as f32 / 131.7 - 0.94))
        })
        .collect()
}

fn activations(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let value = ((r * 127 + c * 13) % 997) as f32 / 251.13 - 1.73;
            // Even rows model exact widening after the final norm; odd rows
            // retain F32 low bits to guard the projection's wider input contract.
            if r % 2 == 0 {
                bf16_to_f32(f32_to_bf16(value))
            } else {
                value
            }
        })
        .collect()
}

fn case(entry: &Entry, op: Prepared<ProjectionBuffers<View>>) -> LabResult<OpCase<'static>> {
    let q = entry.query();
    let k = q.k as usize;
    let bank = weight_bank(k);
    let bank_bytes = bank
        .iter()
        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(entry.weight_bytes() as usize);
    for n in 0..q.n as usize {
        let start = n % BANK * k * 2;
        weights.extend_from_slice(&bank_bytes[start..start + k * 2]);
    }
    let roles = ProjectionBuffers {
        activation: Spec::input(
            DType::F32,
            (Rows::Live, k),
            activations(SCORED_ROWS as usize, k)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        live_rows: Spec::live(),
        output: Spec::output(DType::F32, (Rows::Live, q.n as usize)),
        residual: None,
        weights: QuantizedWeights {
            codes: Spec::weight(weights),
            local_scales: None,
            global_scales: None,
        },
        activation_global: None,
        routing: None,
        // This entry has no quantizer or row-status argument.
        scratch: ProjectionScratch {
            weight_domain: op.info().bytes_of("scratch.weight_domain").map(|_| {
                Spec::weight(
                    vec![1_u32; q.n.div_ceil(16) as usize]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                )
            }),
            ..scratch(|| Spec::untouched(16))
        },
        row_offset: 0,
        shard_rows: 0,
    };
    let cached = RefCell::new(None::<(Vec<u8>, f32_gemm::Oracle)>);
    Ok(OpCase::new(op, SCORED_ROWS, roles)?.live_counts(LIVE.to_vec()).oracle(move |images: &Images, live| {
        let input = images.bytes("activation")?;
        if cached.borrow().as_ref().is_none_or(|(key, _)| key != input) {
            let a = images.f32s("activation")?;
            let oracle = f32_gemm::oracle(GemmQuery { n: BANK as u32, ..q }, a.len() / k, &a, &bank).map_err(|e| e.to_string())?;
            *cached.borrow_mut() = Some((input.to_vec(), oracle));
        }
        let oracle = cached.borrow().as_ref().unwrap().1.clone();
        Ok(vec![("output".into(), Expected::Judge(Box::new(move |bytes| {
            let count = live as usize * q.n as usize;
            if bytes.len() < count * 4 { return Err("head output extent differs".into()); }
            for (i, raw) in bytes.chunks_exact(4).take(count).enumerate() {
                let reference = i / q.n as usize * BANK + i % q.n as usize % BANK;
                let actual = f32::from_le_bytes(raw.try_into().unwrap()) as f64;
                let bound = f32_gemm::bound(oracle.k, oracle.sum_abs_products[reference]);
                if !actual.is_finite() || (actual - oracle.values[reference]).abs() > bound {
                    return Err(format!("output[{i}] differs from F64: actual={actual}, reference={}, bound={bound}", oracle.values[reference]));
                }
            }
            Ok(())
        })))])
    }))
}

fn required(entry: Entry) -> Required<'static> {
    let operation = cases::operation(&entry);
    let usage = entry.usage();
    Required::new(
        entry.symbol().to_owned(),
        Need::rows(SCORED_ROWS).with_exact(Invariance::any_rows(SCORED_ROWS)),
        move |backend, need| {
            backend
                .projection(&operation, usage, need)
                .map(|b| b.handle)
        },
        move |op| case(&entry, op),
    )
    .required()
}

#[test]
fn registered_heads_prepare_every_row_count() {
    for (entry, backend) in heads().into_iter().chain(
        probes::tails()
            .into_iter()
            .map(|e| (e.clone(), lab_backend(e))),
    ) {
        let q = entry.query();
        for rows in [1, 2, DECODE_BATCH, SCORED_ROWS] {
            let op = backend
                .projection(
                    &cases::operation(&entry),
                    entry.usage(),
                    Need::rows(rows).with_exact(Invariance::any_rows(rows)),
                )
                .unwrap()
                .handle;
            let view = |handle, bytes| View { handle, bytes };
            let domain_bytes = op.info().bytes_of("scratch.weight_domain");
            let selected = if domain_bytes.is_some() {
                entry.clone().with_split_decode()
            } else {
                entry.clone()
            };
            let roles = ProjectionBuffers {
                activation: view(1, u64::from(rows) * u64::from(q.k) * 4),
                live_rows: view(2, 4),
                output: view(3, u64::from(rows) * u64::from(q.n) * 4),
                residual: None,
                weights: QuantizedWeights {
                    codes: view(4, entry.weight_bytes()),
                    local_scales: None,
                    global_scales: None,
                },
                activation_global: None,
                routing: None,
                scratch: ProjectionScratch {
                    weight_domain: domain_bytes.map(|bytes| view(6, bytes)),
                    ..scratch(|| view(5, 16))
                },
                row_offset: 0,
                shard_rows: 0,
            };
            let launches = op
                .launches(&roles, view(7, op.info().workspace), Counts::rows(rows))
                .unwrap();
            if domain_bytes.is_some() && rows <= 256 {
                assert_eq!(launches.len(), 2);
                assert_eq!(launches[0].symbol, selected.stage_symbol("_prepare"));
                assert_eq!(launches[1].symbol, selected.stage_symbol(""));
                assert_eq!(launches[1].plan, selected.decode_plan(rows).unwrap());
            } else {
                assert_eq!(launches.len(), 1);
                assert_eq!(launches[0].symbol, selected.baseline_symbol());
                assert_eq!(
                    launches[0].plan.grid,
                    [q.n.div_ceil(16), rows.div_ceil(8), 1]
                );
            }
        }
        let report = kit::conform(&backend, &ModelModules, None, &[required(entry)]).unwrap();
        assert_eq!(report.prepared().len(), 1);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
            panic!("head refused CPU declaration checks")
        };
        for name in ["declaration", "need"] {
            assert!(
                checks
                    .iter()
                    .any(|c| c.name == name && matches!(c.outcome, Outcome::Passed(_)))
            );
        }
    }
}

#[test]
fn head_oracle_rejects_bf16_rounded_logits() {
    for (entry, _) in heads() {
        let q = entry.query();
        let a = activations(SCORED_ROWS as usize, q.k as usize);
        let oracle = f32_gemm::oracle(
            GemmQuery {
                n: BANK as u32,
                ..q
            },
            SCORED_ROWS as usize,
            &a,
            &weight_bank(q.k as usize),
        )
        .unwrap();
        for rows in [1, 2, DECODE_BATCH, SCORED_ROWS] {
            let count = rows as usize * BANK;
            let reference = f32_gemm::Oracle {
                k: oracle.k,
                values: oracle.values[..count].to_vec(),
                sum_abs_products: oracle.sum_abs_products[..count].to_vec(),
            };
            let f32 = reference
                .values
                .iter()
                .map(|&v| v as f32 as f64)
                .collect::<Vec<_>>();
            f32_gemm::check_numeric(&f32, &reference).unwrap();
            let rounded = f32
                .iter()
                .map(|&v| bf16_to_f32(f32_to_bf16(v as f32)) as f64)
                .collect::<Vec<_>>();
            assert!(
                f32_gemm::check_numeric(&rounded, &reference).is_err(),
                "{} rows={rows}",
                entry.symbol()
            );
        }
    }
}

#[test]
fn registered_heads_and_tails_pass_on_a_gpu() {
    kernel_harness::gpu_test(2 << 30, |session| {
        for (entry, backend) in heads().into_iter().chain(
            probes::tails()
                .into_iter()
                .map(|e| (e.clone(), lab_backend(e))),
        ) {
            println!(
                "head {} N={} K={} live={LIVE:?} weight_bytes={}",
                entry.symbol(),
                entry.query().n,
                entry.query().k,
                entry.weight_bytes()
            );
            let required = [required(entry.clone())];
            let report = kit::conform(&backend, &ModelModules, Some(session), &required)?;
            assert_eq!(report.prepared().len(), 1);
            assert!(report.failures().is_empty(), "{:?}", report.failures());
            assert!(
                report.not_exercised().is_empty(),
                "{:?}",
                report.not_exercised()
            );
            session.receipt(
                &format!("head-{}", entry.symbol().replace('_', "-")),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
        }
        Ok(())
    })
    .unwrap();
}

struct Mutant;
struct MutantOp {
    inner: Prepared<ProjectionBuffers<View>>,
    info: OpInfo,
}
fn mutant_symbol(symbol: &str) -> String {
    format!("{}_tf32_v1", symbol.strip_suffix("_v1").unwrap())
}
impl Backend for Mutant {
    fn name(&self) -> &str {
        "head-f32-device-control"
    }
    fn projection(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
        need: Need,
    ) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
        let binding = lab_backend(probes::control()).projection(op, usage, need)?;
        let mut info = binding.handle.info().clone();
        for (symbol, _) in &mut info.symbols {
            *symbol = mutant_symbol(symbol);
        }
        Ok(Binding {
            identity: binding.identity,
            handle: Box::new(MutantOp {
                inner: binding.handle,
                info,
            }),
        })
    }
}
impl PreparedOp<ProjectionBuffers<View>> for MutantOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = mutant_symbol(&launch.symbol);
        }
        Ok(launches)
    }
}

#[test]
fn shared_tf32_device_mutant_is_rejected() {
    kernel_harness::gpu_test(1 << 30, |session| {
        let entry = probes::control();
        let required = [cases::required(entry.clone(), Witness::LowComponent)];
        let positive = kit::conform(&lab_backend(entry), &ModelModules, Some(session), &required)?;
        assert_eq!(positive.prepared().len(), 1);
        assert!(positive.failures().is_empty(), "{:?}", positive.failures());
        assert!(positive.not_exercised().is_empty());
        let negative = kit::conform(&Mutant, &ModelModules, Some(session), &required)?;
        assert_eq!(negative.prepared().len(), 1);
        let OpResult::Prepared { checks, .. } = &negative.ops[0].result else { panic!("mutant refused") };
        assert!(checks.iter().any(|check| check.name == "conformance" && matches!(&check.outcome, Outcome::Failed(error) if error.contains("output") && !error.contains("CUDA") && !error.contains("unbound"))), "{checks:?}");
        session.receipt("head-tf32-control", BTreeMap::from([("positive".into(), positive.json()), ("negative".into(), negative.json())]))?;
        Ok(())
    }).unwrap();
}
