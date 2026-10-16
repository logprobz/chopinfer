//! Qualified HC candidate: retain the producer's BF16 down output and compute
//! its bottleneck once. Compare every output with the shipped producer/mix.
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::gemm::fused as gemm;
use model_qwen38_contract::sm120::{gemm as model_gemm, row};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "common/evidence.rs"]
#[allow(dead_code)]
mod evidence;

#[path = "common/accumulation.rs"]
mod accumulation;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase, WriteMap};

const WIDE: usize = 10240;
const RANK: usize = 320;
const WIDTH: usize = 2560;
const GUARD: usize = 16;
const SENTINEL: u16 = 0xffff;
const CALLS: usize = 32;
const X: u64 = 1;
const LIVE: u64 = 2;
const PARTIAL: u64 = 3;
const DOWN: u64 = 4;
const INJECT: u64 = 5;
const BOT: u64 = 6;
const OUT: u64 = 7;
const CANDIDATE: u64 = 4;
const DOWN_WEIGHT: u64 = 20;
const INJECT_WEIGHT: u64 = 21;
const UP_WEIGHT: u64 = 22;

fn values(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Include exact zeros, signed zero and subnormals among finite inputs.
            match i % 257 {
                0 => 0,
                1 => 0x8000,
                2 => 1,
                3 => 0x8001,
                _ => f32_to_bf16((((state >> 32) % 513) as i32 - 256) as f32 * scale),
            }
        })
        .collect()
}
/// Grid values times `scale`, or with `None` values over 13 binades whose
/// F32 sums round, so a different accumulation order changes output bits.
fn inputs(n: usize, seed: u64, scale: Option<f32>) -> Vec<u16> {
    match scale {
        Some(scale) => values(n, seed, scale),
        None => accumulation::wide_rows(n, 1, seed),
    }
}
fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn upload(s: &Session, v: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(v.len().max(4))?;
    b.copy_from_bytes(v, &s.stream).map_err(|e| e.to_string())?;
    Ok(b)
}
fn weight(rows: usize, k: usize, seed: u64, wide: bool) -> LabResult<Vec<u8>> {
    FragmentWeightLayout {
        rows,
        k,
        format: FragmentWeightFormat::Bf16,
    }
    .pack(&bytes(&inputs(
        rows * k,
        seed,
        (!wide).then_some(1.0 / 4096.0),
    )))
    .map(|p| p.bytes)
    .map_err(|e| e.to_string())
}
fn entry(rows: u32, inject: bool) -> gemm::Entry {
    let name = if inject { "in" } else { "down" };
    gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| e.symbol == format!("chop_gemm_qwen38_hc_{name}_c{rows}_v1"))
        .expect("shipped HC producer")
}
fn configure(f: Function, plan: LaunchPlan) -> LabResult<Function> {
    if plan.dynamic_shared_bytes >= 48 * 1024 {
        f.set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    Ok(f)
}

#[test]
fn hc_precompute_cpu_contract() {
    assert_eq!(
        gemm::hc_precompute_shared(kernel_lab::catalogs(), gemm::HC_PADDED_MIX),
        Some(29440)
    );
    assert_eq!(
        gemm::hc_precompute_shared(kernel_lab::catalogs(), gemm::HC_PRECOMPUTED_MIX),
        Some(29184)
    );
    for rows in [16, 32] {
        for inject in [false, true] {
            let e = entry(rows, inject);
            let spec = gemm::hc_precompute_spec(&e).unwrap();
            assert_eq!(
                spec.params.len(),
                e.arguments(
                    rows,
                    gemm::Buffers {
                        weights: [1; 4],
                        activation: 2,
                        outputs: [3; 4],
                        output_strides: [320; 4],
                        partial: 4,
                        semaphores: 5
                    }
                )
                .unwrap()
                .0
                .len()
                    + 1
            );
            assert_eq!(
                gemm::hc_precompute_shared(kernel_lab::catalogs(), &spec.symbol),
                Some(e.plan(rows).unwrap().dynamic_shared_bytes)
            );
            assert_eq!(e.query.k, WIDE as u32);
            assert_eq!(
                e.kind,
                gemm::Kind::ClusterDecode {
                    strips: 1,
                    warp_blocks: 2,
                    passes: rows / 16
                }
            );
            assert_eq!(e.segments, if inject { vec![320, 4] } else { vec![320] });
            for live in 1..=rows {
                let p = e.plan(live).unwrap();
                assert_eq!(p.block, [256, 1, 1]);
                assert_eq!(p.cluster, [5, 1, 1]);
                assert!(p.pdl);
            }
        }
        let p = gemm::hc_up_mix_plan(model_gemm::HC, rows).unwrap();
        assert_eq!(p.block, [128, 1, 1]);
        assert_eq!(p.grid, [160, 1, 1]);
        assert_eq!(p.dynamic_shared_bytes, 29184);
    }
}

struct Fixture {
    padded: bool,
    wide: bool,
    rows: usize,
    inject: bool,
    entry: gemm::Entry,
    functions: Vec<Function>,
    producer_plan: LaunchPlan,
    mix_plan: LaunchPlan,
    bottleneck: kernels_sm120::row::hyper_connection::Entry,
    b: BTreeMap<u64, DeviceBuffer>,
    banks: Vec<Vec<DeviceBuffer>>,
    call_buffers: Vec<BTreeMap<u64, DeviceBuffer>>,
    weight_bytes: usize,
}
impl Fixture {
    fn new(
        s: &Session,
        rows: usize,
        inject: bool,
        seed: u64,
        banks: usize,
        padded: bool,
        wide: bool,
    ) -> LabResult<Self> {
        let entry = entry(rows as u32, inject);
        let producer_plan = entry.plan(rows as u32).map_err(|e| e.to_string())?;
        let mix_plan =
            gemm::hc_up_mix_plan(model_gemm::HC, rows as u32).map_err(|e| e.to_string())?;
        let shipped = s.module("gemm_qwen38_own_sm120")?;
        let candidate = s.module(gemm::HC_PRECOMPUTE_CUBIN)?;
        let candidate_spec = gemm::hc_precompute_spec(&entry).map_err(|e| e.to_string())?;
        let bottleneck = row::hyper_connection_rows()
            .into_iter()
            .find(|e| e.symbol == "chop_row_qwen38_hc_bottleneck_v1")
            .unwrap();
        let row_module = s
            .context
            .load_module(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/row_qwen38.cubin"
            )))
            .map_err(|e| e.to_string())?;
        let functions = vec![
            configure(
                if padded {
                    candidate
                        .function(&candidate_spec.symbol)
                        .map_err(|e| e.to_string())?
                } else {
                    shipped.function(&entry.symbol).map_err(|e| e.to_string())?
                },
                producer_plan,
            )?,
            configure(
                (if padded { &candidate } else { &shipped })
                    .function(if padded {
                        gemm::HC_PRECOMPUTED_MIX
                    } else {
                        model_gemm::HC_UP_MIX
                    })
                    .map_err(|e| e.to_string())?,
                mix_plan,
            )?,
            configure(
                candidate
                    .function(&candidate_spec.symbol)
                    .map_err(|e| e.to_string())?,
                producer_plan,
            )?,
            configure(
                candidate
                    .function(if padded {
                        gemm::HC_PADDED_MIX
                    } else {
                        gemm::HC_PRECOMPUTED_MIX
                    })
                    .map_err(|e| e.to_string())?,
                mix_plan,
            )?,
            row_module
                .function(&bottleneck.symbol)
                .map_err(|e| e.to_string())?,
        ];
        let mut b = BTreeMap::new();
        b.insert(
            X,
            upload(
                s,
                &bytes(&inputs(rows * WIDE, seed, (!wide).then_some(1.0 / 128.0))),
            )?,
        );
        b.insert(LIVE, upload(s, &(rows as u32).to_ne_bytes())?);
        b.insert(
            PARTIAL,
            s.allocate(producer_plan.workspace_bytes.max(4) as usize)?,
        );
        for offset in [0, CANDIDATE] {
            for (id, width) in [(DOWN, RANK), (INJECT, 4), (BOT, RANK), (OUT, WIDTH)] {
                b.insert(
                    id + offset,
                    upload(s, &bytes(&vec![SENTINEL; rows * width + GUARD]))?,
                );
            }
        }
        let packed = [
            weight(RANK, WIDE, seed + 13, wide)?,
            weight(4, WIDE, seed + 17, wide)?,
            weight(WIDE, RANK, seed + 19, wide)?,
        ];
        let weight_bytes = packed.iter().map(Vec::len).sum::<usize>() * banks;
        let banks = (0..banks)
            .map(|_| {
                packed
                    .iter()
                    .map(|p| upload(s, p))
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        let call_buffers = (0..banks.len())
            .map(|bank| {
                let mut call = BTreeMap::new();
                if bank != 0 {
                    call.insert(
                        PARTIAL,
                        s.allocate(producer_plan.workspace_bytes.max(4) as usize)?,
                    );
                    for offset in [0, CANDIDATE] {
                        for (id, width) in [(DOWN, RANK), (INJECT, 4), (BOT, RANK), (OUT, WIDTH)] {
                            call.insert(
                                id + offset,
                                upload(s, &bytes(&vec![SENTINEL; rows * width + GUARD]))?,
                            );
                        }
                    }
                }
                Ok(call)
            })
            .collect::<LabResult<Vec<BTreeMap<u64, DeviceBuffer>>>>()?;
        Ok(Self {
            padded,
            wide,
            rows,
            inject,
            entry,
            functions,
            producer_plan,
            mix_plan,
            bottleneck,
            b,
            banks,
            call_buffers,
            weight_bytes,
        })
    }
    fn reset(&self, s: &Session, live: usize, seed: u64, scale: Option<f32>) -> LabResult<()> {
        self.b[&LIVE]
            .copy_from_bytes(&(live as u32).to_ne_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        self.b[&X]
            .copy_from_bytes(
                &bytes(&inputs(
                    self.rows * WIDE,
                    seed,
                    scale.filter(|_| !self.wide),
                )),
                &s.stream,
            )
            .map_err(|e| e.to_string())?;
        for bank in 0..self.banks.len() {
            for offset in [0, CANDIDATE] {
                for (id, width) in [(DOWN, RANK), (INJECT, 4), (BOT, RANK), (OUT, WIDTH)] {
                    self.buffer(bank, id + offset)
                        .copy_from_bytes(
                            &bytes(&vec![SENTINEL; self.rows * width + GUARD]),
                            &s.stream,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    fn buffer(&self, bank: usize, id: u64) -> &DeviceBuffer {
        self.call_buffers[bank].get(&id).unwrap_or(&self.b[&id])
    }

    fn checks<'a>(
        &'a self,
        s: &'a Session,
        live: usize,
        bank: usize,
        candidate: bool,
        mode: &str,
        oracle: bool,
    ) -> LabResult<replay::Checks<'a>> {
        let offset = if candidate { CANDIDATE } else { 0 };
        let mut checks = replay::Checks::new(s);
        if mode != "mix" {
            for (id, width) in [(DOWN, RANK), (INJECT, 4), (BOT, RANK)] {
                if id == BOT && !candidate && !self.padded && !oracle {
                    continue;
                }
                let written = if id == INJECT && !self.inject {
                    0
                } else {
                    live * width * 2
                };
                checks = checks.output(
                    &format!("producer {id}"),
                    self.buffer(bank, id + offset),
                    WriteMap::of(DType::Bf16).byte_prefix(written)?,
                );
            }
            let bytes = self.producer_plan.workspace_bytes as usize;
            if bytes != 0 {
                checks = checks.scratch_map(
                    "producer partials",
                    self.buffer(bank, PARTIAL),
                    0,
                    move |_| WriteMap::of(DType::F32).byte_prefix(bytes),
                );
            }
        }
        // OUT is write-only for both the chain and mix-only unit.
        let written = if mode == "in" { 0 } else { live * WIDTH * 2 };
        checks = checks.output(
            "mixed",
            self.buffer(bank, OUT + offset),
            WriteMap::of(DType::Bf16).byte_prefix(written)?,
        );
        Ok(checks)
    }

    fn bindings(&self, bank: usize) -> Bindings<'_> {
        let mut buffers = self
            .b
            .iter()
            .map(|(&id, b)| (id, b))
            .collect::<BTreeMap<_, _>>();
        buffers.extend(
            self.call_buffers[bank]
                .iter()
                .map(|(&id, buffer)| (id, buffer)),
        );
        for (i, weight) in self.banks[bank].iter().enumerate() {
            buffers.insert(DOWN_WEIGHT + i as u64, weight);
        }
        Bindings { buffers }
    }
    fn launch(
        &self,
        s: &Session,
        live: usize,
        bank: usize,
        candidate: bool,
        mode: &str,
        pdl: bool,
    ) -> LabResult<()> {
        let offset = if candidate { CANDIDATE } else { 0 };
        let index = if candidate { 2 } else { 0 };
        let bindings = self.bindings(bank);
        if mode != "mix" {
            let mut args = self
                .entry
                .arguments(
                    live as u32,
                    gemm::Buffers {
                        weights: [DOWN_WEIGHT, INJECT_WEIGHT, 0, 0],
                        activation: X,
                        outputs: [DOWN + offset, INJECT + offset, 0, 0],
                        output_strides: [RANK as u32, 4, 0, 0],
                        partial: PARTIAL,
                        ..Default::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            if candidate || self.padded {
                args.0.push(Argument::DeviceBuffer(BOT + offset));
            }
            let mut plan = self.producer_plan;
            plan.pdl = pdl;
            launch::launch(
                &s.context,
                &self.functions[index],
                &s.stream,
                plan,
                &args,
                &bindings,
            )?;
        }
        if mode != "in" {
            let args = gemm::hc_up_mix_arguments(
                UP_WEIGHT,
                if candidate || self.padded {
                    BOT + offset
                } else {
                    DOWN
                },
                X,
                OUT + offset,
                LIVE,
            )
            .map_err(|e| e.to_string())?;
            let mut plan = if candidate && self.padded {
                gemm::hc_padded_mix_plan(model_gemm::HC, self.rows as u32)
                    .map_err(|e| e.to_string())?
            } else {
                self.mix_plan
            };
            plan.pdl = pdl;
            launch::launch(
                &s.context,
                &self.functions[index + 1],
                &s.stream,
                plan,
                &args,
                &bindings,
            )?;
        }
        Ok(())
    }
    fn oracle(&self, s: &Session, live: usize) -> LabResult<()> {
        (0..self.banks.len()).try_for_each(|bank| self.oracle_bank(s, live, bank))
    }

    fn oracle_bank(&self, s: &Session, live: usize, bank: usize) -> LabResult<()> {
        let args: ArgumentList = self
            .bottleneck
            .pack_arguments(
                [DOWN, BOT, LIVE]
                    .into_iter()
                    .map(Argument::DeviceBuffer)
                    .collect(),
            )
            .map_err(|e| e.to_string())?;
        launch::launch(
            &s.context,
            &self.functions[4],
            &s.stream,
            self.bottleneck
                .plan(self.rows as u32, live as u32)
                .map_err(|e| e.to_string())?,
            &args,
            &self.bindings(bank),
        )
    }
    fn compare(&self, s: &Session, live: usize, mix: bool) -> LabResult<()> {
        s.stream.synchronize().map_err(|e| e.to_string())?;
        for bank in 0..self.banks.len() {
            for (id, width) in [(DOWN, RANK), (INJECT, 4), (BOT, RANK), (OUT, WIDTH)] {
                let count = self.rows * width + GUARD;
                let expected = s.download_bf16(self.buffer(bank, id), count)?;
                let actual = s.download_bf16(self.buffer(bank, id + CANDIDATE), count)?;
                if expected != actual {
                    let i = expected
                        .iter()
                        .zip(&actual)
                        .position(|(a, b)| a != b)
                        .unwrap();
                    return Err(format!(
                        "buffer {id}, live {live}, element {i}: candidate {:04x}, shipped {:04x}",
                        actual[i], expected[i]
                    ));
                }
                let written = if (id == INJECT && !self.inject) || (id == OUT && !mix) {
                    0
                } else {
                    live * width
                };
                if expected[written..]
                    .iter()
                    .chain(&actual[written..])
                    .any(|&v| v != SENTINEL)
                {
                    return Err(format!(
                        "buffer {id}, live {live}: dead row or guard changed"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[test]
fn hc_precompute_preserves_bits_and_graph_replay() {
    check_bits(false);
}

#[test]
fn hc_padded_mix_preserves_bits_and_graph_replay() {
    check_bits(true);
}

/// Outputs of the down projection's wide case (32 rows) whose bits differ
/// between 256-K runs (the producer's warp chains) and 128-K runs.
fn chain_sensitivity(s: &Session, seed: u64) -> LabResult<serde_json::Value> {
    let probe = s.module(accumulation::PROBE)?;
    let weight = inputs(RANK * WIDE, seed + 13, None);
    let x = inputs(32 * WIDE, seed, None);
    let problem = accumulation::Problem::upload(s, &weight, &x, RANK, 32, WIDE)?;
    let long = problem.run(s, &probe, accumulation::Scheme::Chain(Some(256)))?;
    let short = problem.run(s, &probe, accumulation::Scheme::Chain(Some(128)))?;
    let differ = long
        .iter()
        .zip(&short)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "HC down, wide inputs: {differ} of {} outputs differ between 256-K and 128-K runs",
        long.len()
    );
    Ok(json!({"outputs": long.len(), "differ": differ}))
}

fn check_bits(padded: bool) {
    gpu_test(2 << 30, |s| {
        evidence::exclusive(s)?;
        let mut cases = 0;
        let mut failures = replay::Failures::default();
        let mut sensitivity = None;
        for (seed, scale) in [
            (7, Some(1.0 / 4096.0)),
            (97, Some(1.0 / 128.0)),
            (1237, Some(1.0)),
            (4099, None),
        ] {
            if scale.is_none() {
                sensitivity = Some(chain_sensitivity(s, seed)?);
            }
            for rows in [16, 32] {
                for inject in [false, true] {
                    let result = (|| -> LabResult<()> {
                        let f = Fixture::new(s, rows, inject, seed, 1, padded, scale.is_none())?;
                        if padded {
                            let empty_result = (|| -> LabResult<()> {
                            // No producer runs for an empty launch. Both mix entries must
                            // leave every output and guard byte untouched.
                            f.reset(s, 0, seed, scale)?;
                            let mut reference = f.checks(s, 0, 0, false, "mix", false)?;
                            reference.phase(Phase::Eager, false, || {
                                f.launch(s, 0, 0, false, "mix", false)
                            })?;
                            let mut candidate = f.checks(s, 0, 0, true, "mix", false)?;
                            candidate.phase(Phase::Eager, false, || {
                                f.launch(s, 0, 0, true, "mix", false)
                            })?;
                            f.compare(s, 0, true)?;
                            let empty = s.capture(|_| f.launch(s, 0, 0, true, "mix", true))?;
                            candidate.phase(Phase::Graph, true, || {
                                empty.launch(&s.stream).map_err(|e| e.to_string())
                            })?;
                            f.compare(s, 0, true)?;
                            Ok(())
                            })();
                            failures.check(&format!("seed {seed} rows {rows} zero live padded mix"), empty_result);
                        }
                        for live in 1..=rows {
                            let live_result = (|| -> LabResult<()> {
                            f.reset(s, live, seed, scale)?;
                            let mut reference = f.checks(s, live, 0, false, "chain", true)?;
                            reference.phase(Phase::Eager, false, || {
                                f.launch(s, live, 0, false, "chain", false)?;
                                f.oracle(s, live)
                            })?;
                            let mut candidate = f.checks(s, live, 0, true, "chain", false)?;
                            candidate.phase(Phase::Eager, false, || {
                                f.launch(s, live, 0, true, "chain", false)
                            })?;
                            f.compare(s, live, true)?;
                            let graph = s.capture(|_| f.launch(s, live, 0, true, "chain", true))?;
                            for replay in 0..2 {
                                f.reset(s, live, seed + 100 + replay, scale)?;
                                let mut reference = f.checks(s, live, 0, false, "chain", true)?;
                                reference.phase(Phase::Eager, false, || {
                                    f.launch(s, live, 0, false, "chain", false)?;
                                    f.oracle(s, live)
                                })?;
                                // A changed input needs its own checked candidate eager image.
                                let mut candidate = f.checks(s, live, 0, true, "chain", false)?;
                                candidate.phase(Phase::Eager, false, || {
                                    f.launch(s, live, 0, true, "chain", false)
                                })?;
                                candidate.phase(Phase::Graph, true, || {
                                    graph.launch(&s.stream).map_err(|e| e.to_string())
                                })?;
                                candidate.phase(Phase::Pdl, true, || {
                                    f.launch(s, live, 0, true, "chain", true)
                                })?;
                                let mut fields = BTreeMap::from([
                                    ("live".into(), json!(live)),
                                    ("rows".into(), json!(rows)),
                                    ("padded".into(), json!(padded)),
                                    ("inject".into(), json!(inject)),
                                ]);
                                candidate.record(&mut fields);
                                evidence::receipt(s, "hc-precompute-replay", fields)?;
                                f.compare(s, live, true)?;
                            }
                            cases += 1;
                            Ok(())
                            })();
                            failures.check(&format!("seed {seed} rows {rows} live {live} inject {inject} padded {padded}"), live_result);
                        }
                        Ok(())
                    })();
                    failures.check(
                        &format!("seed {seed} rows {rows} inject {inject} padded {padded}"),
                        result,
                    );
                }
            }
        }
        failures.finish()?;
        evidence::receipt(
            s,
            if padded {
                "qwen38-hc-padded-mix-gate"
            } else {
                "qwen38-hc-precompute-gate"
            },
            BTreeMap::from([
                ("cases".into(), json!(cases)),
                (
                    "candidate".into(),
                    evidence::identity(gemm::HC_PRECOMPUTE_CUBIN)?,
                ),
                ("all_outputs_bitwise_equal".into(), json!(true)),
                ("wide_chain_256_vs_128".into(), json!(sensitivity)),
                ("dead_rows_untouched".into(), json!(true)),
                ("pdl_graph_replay_equal".into(), json!(true)),
                (
                    "reference".into(),
                    evidence::identity(if padded {
                        gemm::HC_PRECOMPUTE_CUBIN
                    } else {
                        "gemm_qwen38_own_sm120"
                    })?,
                ),
            ]),
        )
    })
    .unwrap();
}

#[test]
fn hc_precompute_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return;
    }
    let padded = std::env::var("CHOPINFER_HC_PADDED_MIX_BENCH").as_deref() == Ok("1");
    let rows: usize = std::env::var("CHOPINFER_HC_PRECOMPUTE_ROWS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([16, 32].contains(&rows));
    let inject = match std::env::var("CHOPINFER_HC_PRECOMPUTE_INJECT")
        .unwrap()
        .as_str()
    {
        "0" => false,
        "1" => true,
        _ => panic!("inject must be 0 or 1"),
    };
    let mode = std::env::var("CHOPINFER_HC_PRECOMPUTE_MODE").unwrap();
    assert!(["in", "mix", "chain"].contains(&mode.as_str()));
    gpu_test(2 << 30, |s| {
        evidence::exclusive(s)?;
        let f = Fixture::new(s, rows, inject, 97, CALLS, padded, false)?;
        f.reset(s, rows, 97, Some(1.0 / 128.0))?;
        // Populate both bottlenecks before mix-only timing and compare all
        // producer outputs outside the timed graph.
        for bank in 0..CALLS {
            for candidate in [false, true] {
                let mut checks = f.checks(s, rows, bank, candidate, "in", !candidate)?;
                checks.phase(Phase::Eager, false, || {
                    f.launch(s, rows, bank, candidate, "in", false)?;
                    if !candidate {
                        f.oracle_bank(s, rows, bank)?;
                    }
                    Ok(())
                })?;
            }
        }
        f.compare(s, rows, false)?;
        let graph = |candidate| {
            s.capture(|_| {
                (0..CALLS).try_for_each(|bank| f.launch(s, rows, bank, candidate, &mode, true))
            })
        };
        let reference = graph(false)?;
        let candidate = graph(true)?;
        let measure = |graph: &runtime_cuda::GraphExec| -> LabResult<f64> {
            let t = kernel_lab::time(s, 3, 11, |stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
            Ok(f64::from(t.median_ms) * 1000.0 / CALLS as f64)
        };
        let make_checks = |candidate| {
            (0..CALLS)
                .map(|bank| f.checks(s, rows, bank, candidate, &mode, !candidate && mode != "mix"))
                .collect::<LabResult<Vec<_>>>()
        };
        let run_all = |candidate| -> LabResult<()> {
            for bank in 0..CALLS {
                f.launch(s, rows, bank, candidate, &mode, true)?;
                if !candidate && mode != "mix" {
                    f.oracle_bank(s, rows, bank)?;
                }
            }
            Ok(())
        };
        let mut reference_checks = make_checks(false)?;
        let mut candidate_checks = make_checks(true)?;
        replay::phase_all(&mut reference_checks, Phase::Eager, false, || {
            run_all(false)
        })?;
        replay::phase_all(&mut candidate_checks, Phase::Eager, false, || run_all(true))?;
        let before = measure(&reference)?;
        let actual = measure(&candidate)?;
        let after = measure(&reference)?;
        replay::phase_all(&mut reference_checks, Phase::Graph, true, || {
            reference.launch(&s.stream).map_err(|e| e.to_string())?;
            if mode != "mix" {
                f.oracle(s, rows)?;
            }
            Ok(())
        })?;
        replay::phase_all(&mut candidate_checks, Phase::Graph, true, || {
            candidate.launch(&s.stream).map_err(|e| e.to_string())
        })?;
        f.compare(s, rows, mode != "in")?;
        evidence::receipt(
            s,
            if padded {
                "qwen38-hc-padded-mix-timing"
            } else {
                "qwen38-hc-precompute-timing"
            },
            BTreeMap::from([
                ("rows".into(), json!(rows)),
                ("inject".into(), json!(inject)),
                ("mode".into(), json!(mode)),
                ("calls_per_graph".into(), json!(CALLS)),
                ("weight_bank_bytes".into(), json!(f.weight_bytes)),
                ("reference_before_us".into(), json!(before)),
                ("candidate_us".into(), json!(actual)),
                ("reference_after_us".into(), json!(after)),
                ("bitwise_equal".into(), json!(true)),
                (
                    "candidate".into(),
                    evidence::identity(gemm::HC_PRECOMPUTE_CUBIN)?,
                ),
                (
                    "reference".into(),
                    evidence::identity(if padded {
                        gemm::HC_PRECOMPUTE_CUBIN
                    } else {
                        "gemm_qwen38_own_sm120"
                    })?,
                ),
            ]),
        )
    })
    .unwrap();
}
