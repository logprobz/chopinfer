//! Lab-only BF16 pair and frozen online-softmax accumulation measurements.
//! Real cases use a digest-bound manifest; missing inputs never become synthetic cases.
#[path = "common/accumulation.rs"]
mod accumulation;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::oracle::{NeumaierSum, OracleOutput};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Component, Path};

const PROBE: &str = "probe_non_dense_accumulation_sm120";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSpec {
    path: String,
    sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseSpec {
    id: String,
    family: String,
    origin: String,
    source_sha256: String,
    n: usize,
    m: usize,
    k: usize,
    reduction_k: usize,
    terms: usize,
    weight: FileSpec,
    activation: FileSpec,
    correction: Option<FileSpec>,
    denominator: Option<FileSpec>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    cases: Vec<CaseSpec>,
}
struct Input {
    spec: CaseSpec,
    weight: Vec<u16>,
    activation: Vec<f32>,
    correction: Vec<f32>,
    denominator: Vec<f32>,
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn bytes(root: &Path, spec: &FileSpec, count: usize, width: usize) -> LabResult<Vec<u8>> {
    let path = Path::new(&spec.path);
    if !path.components().all(|p| matches!(p, Component::Normal(_))) || !is_digest(&spec.sha256) {
        return Err("capture file must have a relative path and SHA256".into());
    }
    let expected = count.checked_mul(width).ok_or("capture size overflow")?;
    if expected > 512 << 20 {
        return Err("capture exceeds the bounded per-file allocation".into());
    }
    if std::fs::metadata(root.join(path))
        .map_err(|e| e.to_string())?
        .len()
        != expected as u64
    {
        return Err("capture size mismatch before allocation".into());
    }
    let data = std::fs::read(root.join(path)).map_err(|e| e.to_string())?;
    if data.len() != expected
        || format!("{:x}", Sha256::digest(&data)) != spec.sha256.to_lowercase()
    {
        return Err(format!("capture size/digest mismatch for {}", spec.path));
    }
    Ok(data)
}
fn f32_bytes(data: &[u8]) -> LabResult<Vec<f32>> {
    let values = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite capture".into());
    }
    Ok(values)
}
fn validate(spec: &CaseSpec) -> LabResult<()> {
    if !["encoder_pairs", "pv"].contains(&spec.family.as_str())
        || spec.n == 0
        || spec.n % 16 != 0
        || spec.m == 0
        || spec.m % 8 != 0
        || spec.k == 0
        || spec.k % 32 != 0
        || spec.reduction_k == 0
        || spec.reduction_k > spec.k
        || spec.n > 512
        || spec.m > 256
        || spec.k > 262_144
        || !is_digest(&spec.source_sha256)
        || spec.origin.is_empty()
        || spec.id.is_empty()
        || (spec.family == "pv"
            && (spec.m % 16 != 0
                || !(1..=3).contains(&spec.terms)
                || spec.correction.is_none()
                || spec.denominator.is_none()))
        || (spec.family == "encoder_pairs"
            && (spec.terms != 2
                || spec.reduction_k != spec.k
                || spec.correction.is_some()
                || spec.denominator.is_some()))
    {
        return Err("invalid accumulation case geometry/provenance".into());
    }
    Ok(())
}
fn load(root: &Path, spec: CaseSpec) -> LabResult<Input> {
    validate(&spec)?;
    let w = bytes(root, &spec.weight, spec.n * spec.k, 2)?;
    let weight = w
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("two bytes")))
        .collect::<Vec<_>>();
    if weight.iter().any(|&x| !bf16_to_f32(x).is_finite()) {
        return Err("nonfinite BF16 capture".into());
    }
    let activation = f32_bytes(&bytes(root, &spec.activation, spec.m * spec.k, 4)?)?;
    let correction = spec
        .correction
        .as_ref()
        .map(|f| bytes(root, f, spec.m * (spec.k / 32), 4).and_then(|d| f32_bytes(&d)))
        .transpose()?
        .unwrap_or_default();
    let denominator = spec
        .denominator
        .as_ref()
        .map(|f| bytes(root, f, spec.m, 4).and_then(|d| f32_bytes(&d)))
        .transpose()?
        .unwrap_or_default();
    if correction.iter().any(|&c| !(0.0..=1.0).contains(&c))
        || denominator.iter().any(|&d| d <= 0.0)
    {
        return Err("invalid online-softmax correction/denominator".into());
    }
    Ok(Input {
        spec,
        weight,
        activation,
        correction,
        denominator,
    })
}
fn split(x: f32) -> [f64; 2] {
    let hi = bf16_to_f32(f32_to_bf16(x));
    [f64::from(hi), f64::from(bf16_to_f32(f32_to_bf16(x - hi)))]
}

// Reference over decoded operands, with the exact frozen correction schedule.
// S sums absolute individual pair/residual products. Mathematical K is kept
// separate from the number of BF16 residual terms and padded instructions.
fn reference(input: &Input, original: bool) -> LabResult<OracleOutput> {
    let s = &input.spec;
    let pv = s.family == "pv";
    let mut operands = vec![[0.0_f64; 3]; s.m * s.k];
    let mut scales = vec![1.0_f64; s.m * (s.k / 32)];
    if pv {
        for row in 0..s.m {
            let mut scale = 1.0;
            for tile in (0..s.k / 32).rev() {
                scales[row * (s.k / 32) + tile] = scale;
                scale *= f64::from(input.correction[row * (s.k / 32) + tile]);
            }
        }
    }
    for (i, operand) in operands.iter_mut().enumerate() {
        let x = input.activation[i];
        if original {
            operand[0] = f64::from(x);
        } else if pv {
            let mut rest = x;
            for v in operand.iter_mut().take(s.terms) {
                let part = bf16_to_f32(f32_to_bf16(rest));
                *v = f64::from(part);
                rest -= part;
            }
        } else {
            operand[..2].copy_from_slice(&split(x));
        }
    }
    let mut values = vec![0.0; s.m * s.n];
    let mut sums = vec![0.0; s.m * s.n];
    for row in 0..s.m {
        for col in 0..s.n {
            let (mut sum, mut abs) = (NeumaierSum::default(), 0.0);
            for k in 0..s.k {
                let scale = if pv {
                    scales[row * (s.k / 32) + k / 32]
                } else {
                    1.0
                };
                let w = f64::from(bf16_to_f32(input.weight[col * s.k + k]));
                for &x in operands[row * s.k + k]
                    .iter()
                    .take(if original { 1 } else { s.terms })
                {
                    let product = w * x * scale;
                    sum.add(product);
                    abs += product.abs();
                }
            }
            values[row * s.n + col] = sum.value();
            sums[row * s.n + col] = abs;
        }
    }
    OracleOutput::new(values, sums, s.reduction_k).map_err(|e| e.to_string())
}
fn summary(actual: &[f32], oracle: &OracleOutput) -> LabResult<Value> {
    if actual.len() != oracle.values.len()
        || actual.is_empty()
        || actual.iter().any(|x| !x.is_finite())
        || oracle
            .values
            .iter()
            .chain(&oracle.sum_abs_products)
            .any(|x| !x.is_finite())
    {
        return Err("empty, nonfinite or mismatched measurement".into());
    }
    if actual
        .iter()
        .zip(&oracle.sum_abs_products)
        .any(|(&x, &s)| s == 0.0 && x != 0.0)
    {
        return Err("nonzero result for a zero-S reference".into());
    }
    let result = accumulation::summarize(actual, oracle, DType::F32);
    Ok(result.json())
}
fn fma(input: &Input, index: usize) -> f32 {
    let s = &input.spec;
    let row = index / s.n;
    let col = index % s.n;
    let mut sum = 0.0_f32;
    for k in 0..s.k {
        if s.family == "pv" && k % 32 == 0 {
            sum *= input.correction[row * (s.k / 32) + k / 32];
        }
        let w = bf16_to_f32(input.weight[col * s.k + k]);
        let x = input.activation[row * s.k + k];
        if s.family == "pv" {
            let mut rest = x;
            for _ in 0..s.terms {
                let p = bf16_to_f32(f32_to_bf16(rest));
                rest -= p;
                sum = w.mul_add(p, sum);
            }
        } else {
            for part in split(x) {
                sum = w.mul_add(part as f32, sum);
            }
        }
    }
    sum
}
fn run(session: &Session, input: &Input) -> LabResult<Value> {
    let began = std::time::Instant::now();
    let s = &input.spec;
    let oracle = reference(input, false)?;
    let original = reference(input, true)?;
    let module = session.module(PROBE)?;
    let weight = session.upload_bf16(&input.weight)?;
    let activation = session.upload_f32(&input.activation)?;
    let correction = if input.correction.is_empty() {
        vec![1.0]
    } else {
        input.correction.clone()
    };
    let correction = session.upload_f32(&correction)?;
    let output = session.allocate(s.n * s.m * 4)?;
    let pv = s.family == "pv";
    let modes: &[(&str, Option<u32>)] = if pv {
        &[
            ("persistent", Some(0)),
            ("tile_32", Some(1)),
            ("mma_8", Some(2)),
            ("key_128", Some(3)),
            ("fma", None),
        ]
    } else {
        &[
            ("persistent", Some(0)),
            ("combined_64", Some(1)),
            ("combined_128", Some(2)),
            ("separate_128", Some(3)),
            ("kahan_64", Some(4)),
            ("fma", None),
        ]
    };
    let mut rows = BTreeMap::new();
    for &(name, mode) in modes {
        let function = format!(
            "chop_probe_{}_accumulation_v1",
            if pv {
                if mode.is_none() { "pv_fma" } else { "pv" }
            } else if mode.is_none() {
                "encoder_fma"
            } else {
                "encoder"
            }
        );
        let mut args = vec![
            KernelArgument::Buffer(&weight),
            KernelArgument::Buffer(&activation),
        ];
        if pv {
            args.push(KernelArgument::Buffer(&correction));
        }
        args.extend([
            KernelArgument::Buffer(&output),
            KernelArgument::U32(s.n as u32),
            KernelArgument::U32(s.m as u32),
            KernelArgument::U32(s.k as u32),
        ]);
        if pv {
            args.push(KernelArgument::U32(s.terms as u32));
        }
        if let Some(mode) = mode {
            args.push(KernelArgument::U32(mode));
        }
        let (grid, threads) = if mode.is_none() {
            (((s.n * s.m).div_ceil(256) as u32, 1, 1), 256)
        } else {
            (
                (
                    s.n as u32 / if pv { 8 } else { 16 },
                    s.m as u32 / if pv { 16 } else { 8 },
                    1,
                ),
                32,
            )
        };
        module
            .function(&function)
            .map_err(|e| e.to_string())?
            .launch(
                &session.stream,
                launch::simple_config(grid, (threads, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
                &args,
            )
            .map_err(|e| e.to_string())?;
        let actual = session.download_f32(&output, s.n * s.m)?;
        if mode.is_none() {
            for index in accumulation::spread(actual.len(), 128) {
                if actual[index].to_bits() != fma(input, index).to_bits() {
                    return Err(format!("FMA emulator mismatch at {index}"));
                }
            }
        }
        if !pv
            && mode == Some(0)
            && s.n == 128
            && [
                384, 1152, 1280, 1536, 2048, 3072, 4096, 4352, 4608, 5120, 16384,
            ]
            .contains(&s.k)
        {
            let live = session.allocate(4)?;
            live.copy_from_bytes(&(s.m as u32).to_le_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            let served_args = [
                KernelArgument::Buffer(&weight),
                KernelArgument::Buffer(&activation),
                KernelArgument::Buffer(&output),
                KernelArgument::Buffer(&live),
                KernelArgument::U32(s.k as u32),
            ];
            // PairsGemmTraits: two stages of (128*64*4 + 128*128), plus alignment.
            let function = module
                .function("chop_probe_encoder_served_accumulation_v1")
                .map_err(|e| e.to_string())?;
            let shared = 2 * (128 * 64 * 4 + 128 * 128) + 1024;
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
            function
                .launch(
                    &session.stream,
                    launch::simple_config(
                        (1, 1, 1),
                        (384, 1, 1),
                        shared,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &served_args,
                )
                .map_err(|e| e.to_string())?;
            let served = session.download_f32(&output, s.n * s.m)?;
            if actual
                .iter()
                .zip(&served)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("pair probe differs from the unmodified served template".into());
            }
        }
        let mut row = json!({"decoded_operands": summary(&actual, &oracle)?});
        // The original reference includes representation error and is a
        // diagnostic, not the isolated accumulation acceptance threshold.
        row["original_operands"] = accumulation::summarize(&actual, &original, DType::F32).json();
        if pv {
            let normalized = actual
                .iter()
                .enumerate()
                .map(|(i, &x)| x / input.denominator[i / s.n])
                .collect::<Vec<_>>();
            let mut normalized_oracle = oracle.clone();
            for i in 0..normalized.len() {
                let d = f64::from(input.denominator[i / s.n]);
                normalized_oracle.values[i] /= d;
                normalized_oracle.sum_abs_products[i] /= d;
            }
            row["normalized_frozen_denominator"] = summary(&normalized, &normalized_oracle)?;
        }
        rows.insert(name, row);
    }
    Ok(
        json!({"id":s.id,"family":s.family,"origin":s.origin,"source_sha256":s.source_sha256,
        "shape":{"n":s.n,"m":s.m,"k":s.k,"reduction_k":s.reduction_k,"terms":s.terms},
        "input_digests":{"weight":s.weight.sha256,"activation":s.activation.sha256,
            "correction":s.correction.as_ref().map(|f|&f.sha256),"denominator":s.denominator.as_ref().map(|f|&f.sha256)},
        "small_tail_sample":s.n*s.m<100_000,"schemes":rows,
        "case_wall_seconds":began.elapsed().as_secs_f64(),
        "persistent_mma_calls":s.k/16*s.terms,
        "nonidentity_rescales":input.correction.iter().filter(|&&c|c!=1.0).count(),
        "pair_template_bitwise_check":!pv && s.n==128 && [384,1152,1280,1536,2048,3072,4096,4352,4608,5120,16384].contains(&s.k),
        "scope":"isolated accumulation; no full served-attention or epilogue qualification"}),
    )
}
fn violations(value: &Value) -> u64 {
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| {
                // Original operands include representation error and are diagnostic.
                if key == "original_operands" {
                    0
                } else if key == "violations" {
                    value.as_u64().unwrap_or(0)
                } else {
                    violations(value)
                }
            })
            .sum(),
        Value::Array(rows) => rows.iter().map(violations).sum(),
        _ => 0,
    }
}

fn real(family: &str) {
    gpu_test(2 << 30, |session| {
        let path = std::env::var_os("CHOPINFER_ACCUMULATION_MANIFEST")
            .ok_or("set CHOPINFER_ACCUMULATION_MANIFEST to matched captures")?;
        let path = Path::new(&path);
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if manifest.cases.len() > 64 || manifest.schema != "chopinfer.non-dense-accumulation.v1" {
            return Err("unknown capture schema".into());
        }
        let select = std::env::var("CHOPINFER_ACCUMULATION_CASE").ok();
        let mut rows = Vec::new();
        for spec in manifest
            .cases
            .into_iter()
            .filter(|s| s.family == family && select.as_ref().is_none_or(|x| x == &s.id))
        {
            let input = load(path.parent().ok_or("manifest parent")?, spec)?;
            rows.push(run(session, &input)?);
        }
        if rows.is_empty() {
            return Err("no matched real cases selected".into());
        }
        let receipt = session.receipt(
            "non_dense_accumulation",
            BTreeMap::from([
                ("cases".into(), json!(&rows)),
                ("synthetic".into(), json!(false)),
                (
                    "source_pin".into(),
                    json!(std::env::var("CHOPINFER_ACCUMULATION_SOURCE").ok()),
                ),
                (
                    "lab_lock_sha256".into(),
                    json!(format!(
                        "{:x}",
                        Sha256::digest(include_bytes!("../kernels.lock"))
                    )),
                ),
            ]),
        )?;
        println!("receipt {}", receipt.display());
        if rows.iter().map(violations).sum::<u64>() != 0 {
            return Err(format!(
                "accumulation contract failures recorded in {}",
                receipt.display()
            ));
        }
        Ok(())
    })
    .expect("real non-dense accumulation");
}
#[test]
fn encoder_pairs_real() {
    real("encoder_pairs");
}
#[test]
fn pv_real() {
    real("pv");
}

fn synthetic(family: &str, k: usize, class: &str) -> Input {
    let (n, m) = (if family == "pv" { 16 } else { 128 }, 16);
    let file = FileSpec {
        path: "synthetic".into(),
        sha256: "0".repeat(64),
    };
    let pv = family == "pv";
    let mut input = Input {
        spec: CaseSpec {
            id: format!("{family}_{class}_{k}"),
            family: family.into(),
            origin: format!("synthetic {class}"),
            source_sha256: "0".repeat(64),
            n,
            m,
            k,
            reduction_k: k,
            terms: if pv {
                if k >= 32768 { 1 } else { 2 }
            } else {
                2
            },
            weight: file.clone(),
            activation: file.clone(),
            correction: pv.then(|| file.clone()),
            denominator: pv.then_some(file),
        },
        weight: vec![0; n * k],
        activation: vec![0.; m * k],
        correction: vec![1.; m * (k / 32)],
        denominator: vec![1.; m],
    };
    for col in 0..n {
        for key in 0..k {
            let sign = if class == "cancellation" && key >= k / 2 {
                -1.
            } else {
                1.
            };
            let exponent = (key * 7 + col * 3) % 13;
            input.weight[col * k + key] =
                f32_to_bf16(sign * (1. + (col % 7) as f32 / 8.) * 2_f32.powi(-(exponent as i32)));
        }
    }
    for row in 0..m {
        for key in 0..k {
            input.activation[row * k + key] = if pv {
                (1. + row as f32 / 11.) * 2_f32.powi(-(((key * 3 + row) % 17) as i32))
            } else {
                (1. + (row * 17 + key % 19) as f32 / 131.) * 2_f32.powi(-(((key + row) % 9) as i32))
            };
        }
    }
    if pv && class == "rescale" {
        for row in 0..m {
            for tile in 0..k / 32 {
                input.correction[row * (k / 32) + tile] = if tile == 0 {
                    0.
                } else if tile % 37 == row % 7 {
                    0.5
                } else {
                    1.
                };
            }
        }
    }
    input
}
fn adversarial(family: &str) {
    gpu_test(2 << 30, |session| {
        let ks: Vec<usize> = if let Ok(k) = std::env::var("CHOPINFER_ACCUMULATION_K") {
            vec![k.parse().map_err(|_| "invalid K selector")?]
        } else if family == "pv" {
            vec![128, 4096, 32768, 262144]
        } else {
            vec![
                384, 1152, 1280, 1536, 2048, 3072, 4096, 4352, 4608, 5120, 16384,
            ]
        };
        let mut rows = Vec::new();
        for k in ks {
            for class in ["positive", "cancellation", "rescale"] {
                let input = synthetic(family, k, class);
                validate(&input.spec)?;
                rows.push(run(session, &input)?);
            }
        }
        if rows.is_empty() {
            return Err("empty adversarial selection".into());
        }
        let path = session.receipt(
            "non_dense_accumulation",
            BTreeMap::from([
                ("cases".into(), json!(&rows)),
                ("synthetic".into(), json!(true)),
                (
                    "source_pin".into(),
                    json!(std::env::var("CHOPINFER_ACCUMULATION_SOURCE").ok()),
                ),
                (
                    "lab_lock_sha256".into(),
                    json!(format!(
                        "{:x}",
                        Sha256::digest(include_bytes!("../kernels.lock"))
                    )),
                ),
            ]),
        )?;
        println!("receipt {}", path.display());
        if rows.iter().map(violations).sum::<u64>() != 0 {
            return Err(format!(
                "accumulation contract failures recorded in {}",
                path.display()
            ));
        }
        Ok(())
    })
    .expect("adversarial non-dense accumulation");
}
#[test]
fn encoder_pairs_adversarial() {
    adversarial("encoder_pairs");
}
#[test]
fn pv_adversarial() {
    adversarial("pv");
}

#[test]
fn pair_oracle_preserves_low_products() {
    let mut input = synthetic("encoder_pairs", 32, "positive");
    input.weight.fill(f32_to_bf16(1.));
    input.activation.fill(1. + 2_f32.powi(-10));
    let oracle = reference(&input, false).unwrap();
    assert_eq!(oracle.values[0], 32. * (1. + 2_f64.powi(-10)));
    assert_eq!(oracle.sum_abs_products[0], oracle.values[0]);
    assert_eq!(fma(&input, 0), oracle.values[0] as f32);
}
#[test]
fn pv_oracle_rescales_history_without_resetting_it() {
    let mut input = synthetic("pv", 64, "positive");
    input.weight.fill(f32_to_bf16(1.));
    input.activation.fill(1.);
    input.correction.fill(1.);
    for row in 0..input.spec.m {
        input.correction[row * 2] = 0.;
        input.correction[row * 2 + 1] = 0.5;
    }
    let oracle = reference(&input, false).unwrap();
    assert_eq!(oracle.values[0], 48.);
    assert_eq!(oracle.sum_abs_products[0], 48.);
    assert_eq!(fma(&input, 0), 48.);
    input.correction[1] = 0.;
    assert_eq!(reference(&input, false).unwrap().values[0], 32.);
}
#[test]
fn reject_zero_scale_nonfinite_and_bad_capture_geometry() {
    let mut input = synthetic("encoder_pairs", 32, "positive");
    input.spec.m = 7;
    assert!(validate(&input.spec).is_err());
    let oracle = OracleOutput::new(vec![0.], vec![0.], 32).unwrap();
    assert!(summary(&[1.], &oracle).is_err());
    assert!(summary(&[f32::NAN], &oracle).is_err());
    assert!(summary(&[0.], &oracle).is_ok());
}
