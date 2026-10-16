//! The F32 norm-weight forms every model catalog declares
//! (`ModelCatalog::norm_weights`), run from the cubins its bundle embeds:
//! grouped norm rows and the hyper-connection all-reduce's prefill, decode
//! and wide entries. The cubins come from the bundles' kernel-build reports
//! (`CHOPINFER_BUNDLE_REPORTS`, paths separated by `:`), each checked
//! against its lock and sources, so these cases cover the bytes a bundle
//! loads for an image with F32 norm weights.
//!
//! Each form computes `(1 + w)` in F32 from weights BF16 cannot represent,
//! values near -1 included, and must match the f64 oracle, also on a scaled
//! fixture whose bound narrowing the weights to BF16 exceeds (`SMALL`); give
//! the BF16 form's bits for BF16-representable weights; write only its live
//! rows; give a row the same bits at another row count and position, and an
//! all-reduce row the same bits through every entry that promises them; and
//! repeat its bits from captured graphs. The all-reduces run at the selected rank
//! count (`CHOPINFER_KERNEL_GPUS`, distinct devices) through peer transport,
//! on comm state shared with the BF16 forms as a bundle runs them.

#[path = "../norm_weights.rs"]
#[allow(dead_code)] // Only the row instances' names are used here.
mod lab_rows;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    ArgumentList, DType, LaunchPlan, SigmoidMeanHyperConnectionQuery, Validate,
};
use kernel_core::hyper_connection_oracle;
use kernel_core::oracle::OracleOutput;
use kernel_lab::launch::{self, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, Replay, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, check_output, gpu_test, gpu_test_multi};
use kernels_sm120::comm::hyper_connection as hc;
use kernels_sm120::comm::hyper_connection_decode as hcd;
use kernels_sm120::comm::norm_weights::{NormRowForms, NormTransportForms};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// Rows of every comm fixture: the prefill state's capacity and the most
/// prepared rows of a wide call.
const CAPACITY: usize = 256;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A reported cubin's file and entry symbols.
type Reported = BTreeMap<String, (std::path::PathBuf, Vec<String>)>;

/// Every reported bundle cubin by name, from lock-verified current reports.
fn reported() -> LabResult<&'static Reported> {
    static CUBINS: std::sync::OnceLock<Result<Reported, String>> = std::sync::OnceLock::new();
    CUBINS
        .get_or_init(|| {
            let reports = std::env::var_os("CHOPINFER_BUNDLE_REPORTS")
                .ok_or(
                    "this GPU test needs CHOPINFER_BUNDLE_REPORTS: the kernel-build-report.json of each \
                     bundle build holding the declared forms (in the bundle crate's build script \
                     output directory), separated by `:`; no in-repo script sets it, so a GPU run \
                     fails without it",
                )?;
            let mut cubins = BTreeMap::new();
            for path in std::env::split_paths(&reports) {
                let report = kernel_build::BuildReport::read(&path).map_err(err)?;
                report.verify_current().map_err(err)?;
                for cubin in report.cubins {
                    let symbols = cubin.declarations.iter().map(|d| d.symbol.clone());
                    let entry = (cubin.path, symbols.collect());
                    if cubins.insert(cubin.name.clone(), entry).is_some() {
                        return Err(format!("two reports build {}", cubin.name));
                    }
                }
            }
            Ok(cubins)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn embedded(name: &str) -> LabResult<Vec<u8>> {
    let (path, _) = reported()?
        .get(name)
        .ok_or_else(|| format!("no listed bundle report builds {name}"))?;
    std::fs::read(path).map_err(err)
}

/// With reports listed, each declared form's cubin is reported with its
/// entries: the CPU check of a GPU job's inputs.
#[test]
fn reported_bundles_build_every_declared_form() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none() {
        eprintln!("SKIP: set CHOPINFER_BUNDLE_REPORTS to check the reported cubins");
        return;
    }
    let reported = reported().unwrap();
    let has = |cubin: &str, symbol: &str| {
        reported
            .get(cubin)
            .is_some_and(|(_, symbols)| symbols.iter().any(|s| s == symbol))
    };
    let (rows, transports) = forms();
    for form in rows {
        for entry in [&form.f32, &form.bf16] {
            assert!(has(&entry.cubin, &entry.symbol), "{entry:?}");
        }
    }
    for form in transports {
        for f32 in [0, 1] {
            let prefill = form.prefill[f32];
            for symbol in [
                prefill.symbol,
                prefill.scatter_symbol,
                prefill.gather_symbol,
            ] {
                assert!(has(prefill.cubin, symbol), "{symbol}");
            }
            let decode = &form.decode[f32];
            for symbol in [
                hcd::symbol_v2(decode, form.ranks).unwrap(),
                hcd::symbol_v3(decode, form.ranks).unwrap(),
            ] {
                assert!(has(&form.decode_cubins[f32], symbol), "{symbol}");
            }
            let wide = hcd::wide_symbol(decode, form.ranks).unwrap();
            assert!(has(&form.wide_cubins[f32], wide), "{wide}");
        }
    }
}

/// Every catalog's declared row and transport forms.
fn forms() -> (Vec<NormRowForms>, Vec<NormTransportForms>) {
    let mut rows = Vec::new();
    let mut transports = Vec::new();
    for model in kernel_lab::catalogs().models() {
        let forms = (model.norm_weights)();
        rows.extend(forms.rows);
        transports.extend(forms.transports);
    }
    (rows, transports)
}

/// F32 norm weights BF16 cannot represent, including values near -1 whose
/// `1 + w` cancels.
fn weights(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| match i % 5 {
            0 => -0.9998,
            1 => -1.0002,
            2 => 0.10003,
            3 => -2.003,
            _ => (i % 71) as f32 / 91.0 + 3.0e-5,
        })
        .collect()
}

/// `weights` rounded to BF16: the BF16 form's weights, and the F32 weights
/// it represents exactly.
fn rounded(weights: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let codes = weights.iter().map(|&w| f32_to_bf16(w)).collect::<Vec<_>>();
    let values = codes.iter().map(|&c| bf16_to_f32(c)).collect();
    (codes, values)
}

/// The scale of the fixtures whose oracle bound rejects weights narrowed to
/// BF16. The bound's reduction term grows with each norm group's sum of
/// squares while the normalized output does not, so inputs scaled by a power
/// of two, exactly in BF16 and F32, shrink it below the error the narrowed
/// weights make (`scaled_fixtures_reject_bf16_narrowed_weights`).
const SMALL: f32 = 1.0 / 128.0;

fn scaled(values: &[u16]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| f32_to_bf16(bf16_to_f32(v) * SMALL))
        .collect()
}

/// The scaled row cases' input of `rows` rows of `wide` values.
fn small_rows(rows: usize, wide: usize) -> Vec<u16> {
    scaled(&codes(rows * wide, 3))
}

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}

/// Rows `ids` of `width`-wide deterministic BF16 input with full mantissas,
/// both signs and exponents -8 through 8.
fn rows_of(ids: &[usize], width: usize) -> Vec<u16> {
    ids.iter()
        .flat_map(|&row| {
            (0..width).map(move |c| {
                let h = (((row as u64 + 1) * 0x9e37_79b9) ^ ((c as u64 + 1) * 0x85eb_ca6b))
                    .wrapping_mul(0xc2b2_ae35);
                ((h >> 32) as u16 & 0x807f) | ((119 + (h % 17) as u16) << 7)
            })
        })
        .collect()
}

fn decoded(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(bf16_to_f32(v))).collect()
}

fn widened(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

/// The f64 oracle of a row form over `rows` rows of `input`.
fn row_oracle(form: &NormRowForms, input: &[u16], weight: &[f32]) -> LabResult<OracleOutput> {
    hyper_connection_oracle::grouped_norm(
        SigmoidMeanHyperConnectionQuery::GroupedNorm {
            streams: form.row.groups,
            width: form.row.group_width,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: 8,
        },
        &decoded(input),
        &widened(weight),
    )
    .map_err(err)
}

/// One all-reduce fixture's inputs: each rank's partials, the residual and
/// the inject logits, `CAPACITY` rows of each.
struct Inputs {
    partials: Vec<Vec<u16>>,
    x: Vec<u16>,
    logits: Vec<u16>,
}

impl Inputs {
    fn new(ranks: usize, width: usize, streams: usize) -> Self {
        Self {
            partials: (0..ranks)
                .map(|rank| codes(CAPACITY * width, rank + 4))
                .collect(),
            x: codes(CAPACITY * width * streams, 1),
            logits: codes(CAPACITY * streams, 2),
        }
    }

    /// The same logits over partials and residual scaled by `SMALL`.
    fn small(&self) -> Self {
        Self {
            partials: self.partials.iter().map(|p| scaled(p)).collect(),
            x: scaled(&self.x),
            logits: self.logits.clone(),
        }
    }

    /// The f64 oracle's residual and norm of the first `live` rows.
    fn oracle(
        &self,
        query: SigmoidMeanHyperConnectionQuery,
        weight: &[f32],
        live: usize,
    ) -> LabResult<hyper_connection_oracle::InjectNormOutput> {
        let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm { width, streams, .. } = query
        else {
            return Err("expected an all-reduce query".into());
        };
        let (width, streams) = (width as usize, streams as usize);
        hyper_connection_oracle::all_reduce_inject_norm(
            query,
            &self
                .partials
                .iter()
                .map(|p| decoded(&p[..live * width]))
                .collect::<Vec<_>>(),
            &decoded(&self.x[..live * width * streams]),
            &decoded(&self.logits[..live * streams]),
            &widened(weight),
        )
        .map_err(err)
    }
}

/// F32 grouped norm arithmetic on the CPU, as the F32 forms compute it:
/// `x * rsqrt(mean(x^2) + eps) * (1 + w)` with `(1 + w)` in F32 and one
/// BF16 rounding. Sequential sums stand in for the kernels' tree reductions.
fn modeled_norm(values: &[f32], weight: &[f32], width: usize) -> Vec<f32> {
    values
        .chunks_exact(width)
        .enumerate()
        .flat_map(|(group, x)| {
            let w = &weight[group * width % weight.len()..][..width];
            let square = x.iter().fold(0_f32, |sum, &v| sum + v * v);
            let inv = 1.0 / (square / width as f32 + 1e-6).sqrt();
            x.iter()
                .zip(w)
                .map(move |(&v, &w)| bf16_to_f32(f32_to_bf16(v * inv * (1.0 + w))))
        })
        .collect()
}

/// The F32 all-reduce epilogue's norm output on the CPU: the rank-ordered
/// F32 sum, injected without rounding, then `modeled_norm`.
fn modeled_inject_norm(
    inputs: &Inputs,
    width: usize,
    streams: usize,
    weight: &[f32],
    live: usize,
) -> Vec<f32> {
    let mut residual = Vec::with_capacity(live * streams * width);
    for row in 0..live {
        let y = (0..width)
            .map(|c| {
                inputs.partials.iter().fold(0_f32, |sum, partial| {
                    sum + bf16_to_f32(partial[row * width + c])
                })
            })
            .collect::<Vec<_>>();
        for stream in 0..streams {
            let z = bf16_to_f32(inputs.logits[row * streams + stream]) / streams as f32;
            let scale = 2.0 * (1.0 / (1.0 + (-z).exp()));
            let x = &inputs.x[(row * streams + stream) * width..][..width];
            residual.extend(x.iter().zip(&y).map(|(&x, &y)| bf16_to_f32(x) + scale * y));
        }
    }
    modeled_norm(&residual, weight, width)
}

/// The first `len` outputs of `reference`.
fn prefix(reference: &OracleOutput, len: usize) -> OracleOutput {
    OracleOutput {
        values: reference.values[..len].to_vec(),
        sum_abs_products: reference.sum_abs_products[..len].to_vec(),
        reduction_lengths: reference.reduction_lengths[..len].to_vec(),
    }
}

/// CPU negative control of the scaled fixtures, before any GPU runs them:
/// at the live rows of every serving call and of the row cases, F32
/// arithmetic over the F32 weights passes the oracle bound the GPU cases
/// apply, and the same arithmetic over the weights narrowed to BF16 fails
/// it.
#[test]
fn scaled_fixtures_reject_bf16_narrowed_weights() {
    let (rows, transports) = forms();
    assert!(!rows.is_empty() && !transports.is_empty());
    let gate = |actual: &[f32], reference: &OracleOutput| {
        check_output(actual, reference, DType::Bf16, &mut BTreeMap::new())
    };
    for form in &transports {
        let epilogue = form.prefill[1].shape.epilogue;
        let (width, streams) = (epilogue.width as usize, epilogue.streams as usize);
        let wide = width * streams;
        let w = weights(wide);
        let (_, narrowed) = rounded(&w);
        let inputs = Inputs::new(form.ranks as usize, width, streams).small();
        let lives = serving_calls(form)
            .unwrap()
            .into_iter()
            .map(|(_, live)| live)
            .collect::<BTreeSet<_>>();
        let most = *lives.last().unwrap();
        let reference = inputs.oracle(prefill_query(form), &w, most).unwrap();
        let f32 = modeled_inject_norm(&inputs, width, streams, &w, most);
        let bf16 = modeled_inject_norm(&inputs, width, streams, &narrowed, most);
        for live in lives {
            let reference = prefix(&reference.normed, live * wide);
            let len = live * wide;
            gate(&f32[..len], &reference)
                .unwrap_or_else(|e| panic!("{} ranks, {live} rows: F32 weights: {e}", form.ranks));
            assert!(
                gate(&bf16[..len], &reference).is_err(),
                "{} ranks, {live} rows: the bound admits BF16 weights",
                form.ranks
            );
        }
    }
    for form in &rows {
        let width = form.row.group_width as usize;
        let wide = width * form.row.groups as usize;
        let w = weights(wide);
        let (_, narrowed) = rounded(&w);
        let input = small_rows(5, wide);
        let reference = row_oracle(form, &input, &w).unwrap();
        let values = input.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
        gate(&modeled_norm(&values, &w, width), &reference)
            .unwrap_or_else(|e| panic!("{}: F32 weights: {e}", form.f32.symbol));
        assert!(
            gate(&modeled_norm(&values, &narrowed, width), &reference).is_err(),
            "{}: the bound admits BF16 weights",
            form.f32.symbol
        );
    }
}

fn bytes(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len())?;
    buffer.copy_from_bytes(data, &session.stream).map_err(err)?;
    Ok(buffer)
}

fn count(session: &Session, live: usize) -> LabResult<DeviceBuffer> {
    bytes(session, &(live as u32).to_le_bytes())
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The first `live` rows of `wide` values of a poisoned output, whose later
/// bytes must still hold poison.
fn live_rows(
    session: &Session,
    buffer: &DeviceBuffer,
    live: usize,
    wide: usize,
    label: &str,
) -> LabResult<Vec<u16>> {
    let data = poison::read(session, buffer)?;
    poison::check(
        label,
        &data,
        &WriteMap::of(DType::Bf16).prefix(live * wide)?,
        Fill::Pattern(&[POISON]),
    )?;
    Ok(data[..live * wide * 2]
        .chunks_exact(2)
        .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
        .collect())
}

/// The declared forms pair an F32 form with a BF16 form of the same shape.
#[test]
fn declared_norm_weight_forms_pair_like_forms() {
    let (rows, transports) = forms();
    assert!(!rows.is_empty() && !transports.is_empty());
    for form in &rows {
        form.row.validate().unwrap();
        assert_eq!(form.row.weight_dtype, DType::F32);
        assert_ne!(form.f32, form.bf16);
    }
    for form in &transports {
        assert!(hc::RANKS.contains(&form.ranks));
        let [bf16, f32] = form.prefill;
        assert_eq!(bf16.shape.epilogue.weight_pointer_type(), "const uint16_t");
        assert_eq!(f32.shape.epilogue.weight_pointer_type(), "const float");
        assert_eq!(
            (bf16.shape.epilogue.width, bf16.shape.epilogue.streams),
            (f32.shape.epilogue.width, f32.shape.epilogue.streams)
        );
        assert_eq!(form.decode[0].shape, bf16.shape);
        assert_eq!(form.decode[1].shape, f32.shape);
        assert_ne!(form.decode_cubins[0], form.decode_cubins[1]);
        assert_ne!(form.wide_cubins[0], form.wide_cubins[1]);
    }
}

/// One launch of a grouped norm row over `capacity` prepared rows.
#[allow(clippy::too_many_arguments)]
fn norm_row(
    session: &Session,
    form: &NormRowForms,
    module: &Module,
    symbol: &str,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    output: &DeviceBuffer,
    live: &DeviceBuffer,
    capacity: u32,
) -> LabResult<()> {
    let plan = form.row.plan(capacity, 0).map_err(err)?;
    module
        .function(symbol)
        .map_err(err)?
        .launch(
            &session.stream,
            LaunchConfig {
                grid: (plan.grid[0], plan.grid[1], plan.grid[2]),
                block: (plan.block[0], plan.block[1], plan.block[2]),
                shared_memory_bytes: 0,
                cluster: LaunchConfig::NO_CLUSTER,
                programmatic_stream_serialization: plan.pdl,
            },
            &[
                KernelArgument::Buffer(input),
                KernelArgument::Buffer(weight),
                KernelArgument::Buffer(output),
                KernelArgument::Buffer(live),
            ],
        )
        .map_err(err)
}

fn row_case(session: &Session, form: &NormRowForms) -> LabResult<()> {
    let wide = (form.row.group_width * form.row.groups) as usize;
    let f32_module = session.load_cubin(&embedded(&form.f32.cubin)?)?;
    let bf16_module = session.load_cubin(&embedded(&form.bf16.cubin)?)?;
    let w = weights(wide);
    let (bf16_codes, exact) = rounded(&w);
    if w.iter().zip(&exact).filter(|(a, b)| a != b).count() <= wide / 2 {
        return Err("the weights must mostly lie off the BF16 grid".into());
    }
    let weight = session.upload_f32(&w)?;
    let exact_weight = session.upload_f32(&exact)?;
    let bf16_weight = session.upload_bf16(&bf16_codes)?;
    // Five live rows of eight prepared, eagerly, relaunched and replayed.
    let ids = [0, 1, 2, 3, 4];
    let x = rows_of(&ids, wide);
    let input = session.upload_bf16(&x)?;
    let output = session.allocate(8 * wide * 2)?;
    let live = count(session, ids.len())?;
    let run = |module: &Module, symbol: &str, weight: &DeviceBuffer, output: &DeviceBuffer| {
        norm_row(
            session, form, module, symbol, &input, weight, output, &live, 8,
        )
    };
    let launch = || run(&f32_module, &form.f32.symbol, &weight, &output);
    let mut replay =
        Replay::new(session).output("norm", &output, WriteMap::of(DType::Bf16).prefix(x.len())?);
    replay.eager(launch)?;
    replay.relaunch(launch)?;
    let graph = session.capture(|_| launch())?;
    replay.graph(&graph)?;
    let actual = session.download_bf16(&output, x.len())?;
    let oracle = row_oracle(form, &x, &w)?;
    let mut receipt = BTreeMap::from([
        ("symbol".into(), json!(form.f32.symbol)),
        ("cubin".into(), json!(form.f32.cubin)),
        ("rows".into(), json!(ids)),
        ("capacity".into(), json!(8)),
    ]);
    check_output(
        &actual.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
        &oracle,
        DType::Bf16,
        &mut receipt,
    )?;
    replay.record(&mut receipt);
    // The scaled fixture, whose bound narrowing the weights to BF16 exceeds.
    let small = small_rows(ids.len(), wide);
    let small_input = session.upload_bf16(&small)?;
    let small_output = session.allocate(8 * wide * 2)?;
    norm_row(
        session,
        form,
        &f32_module,
        &form.f32.symbol,
        &small_input,
        &weight,
        &small_output,
        &live,
        8,
    )?;
    let mut scaled_receipt = BTreeMap::new();
    check_output(
        &session
            .download_bf16(&small_output, small.len())?
            .into_iter()
            .map(bf16_to_f32)
            .collect::<Vec<_>>(),
        &row_oracle(form, &small, &w)?,
        DType::Bf16,
        &mut scaled_receipt,
    )?;
    receipt.insert("scaled_oracle".into(), json!(scaled_receipt));
    // Another permutation and count of the rows gives each row the same
    // bits, as does one row alone.
    for (order, capacity) in [(vec![4, 2, 0, 3, 1], 5_usize), (vec![3], 1)] {
        let input = session.upload_bf16(&rows_of(&order, wide))?;
        let output = session.allocate(capacity * wide * 2)?;
        let live = count(session, order.len())?;
        norm_row(
            session,
            form,
            &f32_module,
            &form.f32.symbol,
            &input,
            &weight,
            &output,
            &live,
            capacity as u32,
        )?;
        let rows = session.download_bf16(&output, order.len() * wide)?;
        for (at, &id) in order.iter().enumerate() {
            if rows[at * wide..(at + 1) * wide] != actual[id * wide..(id + 1) * wide] {
                return Err(format!("row {id} differs at position {at} of {order:?}"));
            }
        }
    }
    // Representable weights give the BF16 row's bits.
    let parity = |module: &Module, symbol: &str, weight: &DeviceBuffer| {
        let output = session.allocate(8 * wide * 2)?;
        run(module, symbol, weight, &output)?;
        session.download_bf16(&output, x.len())
    };
    if parity(&f32_module, &form.f32.symbol, &exact_weight)?
        != parity(&bf16_module, &form.bf16.symbol, &bf16_weight)?
    {
        return Err(format!(
            "{} differs from {} for representable weights",
            form.f32.symbol, form.bf16.symbol
        ));
    }
    // The lab's instance of the same template, where it has one, gives the
    // bundle's bits.
    let shape = (form.row.group_width, form.row.groups);
    if lab_rows::SHAPES.contains(&shape)
        && (form.row.groups == 1)
            == (form.row.reduction == kernels_sm120::comm::epilogue::NormReduction::XorThenSerial)
    {
        let lab = session.module(lab_rows::CUBIN)?;
        let symbol = lab_rows::row_symbol(shape.0, shape.1, DType::F32);
        if parity(&lab, &symbol, &weight)? != actual {
            return Err(format!(
                "{} differs from the lab's {symbol}",
                form.f32.symbol
            ));
        }
        receipt.insert("lab_template_parity".into(), json!(symbol));
    }
    receipt.insert("bf16_parity".into(), json!(form.bf16.symbol));
    receipt.insert("permutation_invariance".into(), json!(true));
    session.receipt("norm_weight_forms", receipt)?;
    Ok(())
}

/// Every declared F32 grouped norm row.
#[test]
fn f32_norm_rows() {
    gpu_test(64 << 20, |session| {
        for form in forms().0 {
            row_case(session, &form).map_err(|e| format!("{}: {e}", form.f32.symbol))?;
        }
        Ok(())
    })
    .unwrap();
}

/// Which all-reduce a call runs, at how many prepared rows.
#[derive(Clone, Copy, Debug)]
enum Call {
    Prefill(u32),
    /// Decode v2, or v3 when the flag is set.
    Decode(u32, bool),
    Wide(u32),
}

/// The buffers of one call, by fixture id.
#[derive(Clone, Copy)]
struct Ids {
    input: u64,
    residual: u64,
    logits: u64,
    weight: u64,
    residual_out: u64,
    normed: u64,
    live: u64,
}

const MAIN: Ids = Ids {
    input: 100,
    residual: 101,
    logits: 102,
    weight: 103,
    residual_out: 104,
    normed: 105,
    live: 109,
};

/// One dtype's prefill, decode and wide modules.
struct Modules {
    prefill: CollectiveModules,
    decode: CollectiveModules,
    wide: CollectiveModules,
}

/// The prefill query of a form's fixtures.
fn prefill_query(form: &NormTransportForms) -> SigmoidMeanHyperConnectionQuery {
    let epilogue = form.prefill[1].shape.epilogue;
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: form.ranks,
        streams: epilogue.streams,
        width: epilogue.width,
        eps_bits: form.prefill[1].eps_bits,
        capacity: CAPACITY as u32,
    }
}

/// (symbol, plan, arguments) of each launch of `call` of `form` on `rank`.
fn launches(
    form: &NormTransportForms,
    query: SigmoidMeanHyperConnectionQuery,
    call: Call,
    f32: bool,
    ids: Ids,
    rank: usize,
) -> LabResult<Vec<(&'static str, LaunchPlan, ArgumentList)>> {
    let (rank, ranks) = (rank as u32, form.ranks);
    let slots = |first: u64| (first..first + u64::from(ranks)).collect::<Vec<_>>();
    let decode = &form.decode[f32 as usize];
    let decode_call = hcd::Call {
        input: ids.input,
        residual: ids.residual,
        logits: ids.logits,
        weight: ids.weight,
        residual_out: ids.residual_out,
        normed: ids.normed,
        inboxes: slots(21),
        flags: slots(31),
        epochs: 118,
        live_rows: ids.live,
        rows: 0,
        rank,
    };
    Ok(match call {
        Call::Prefill(rows) => hc::launches(
            &form.prefill[f32 as usize],
            query,
            rows,
            hc::Call {
                input: ids.input,
                residual: ids.residual,
                logits: ids.logits,
                weight: ids.weight,
                residual_out: ids.residual_out,
                normed: ids.normed,
                inboxes: slots(1),
                flags: slots(11),
                epoch: 108,
                live_rows: ids.live,
                capacity: CAPACITY as u32,
                rank,
            },
        )
        .map_err(err)?
        .into_iter()
        .map(|l| (l.symbol, l.plan, l.arguments))
        .collect(),
        Call::Decode(rows, v3) => vec![(
            if v3 {
                hcd::symbol_v3(decode, ranks)
            } else {
                hcd::symbol_v2(decode, ranks)
            }
            .map_err(err)?,
            hcd::plan_with(decode.shape, rows).map_err(err)?,
            hcd::arguments(hcd::Call {
                rows,
                ..decode_call
            })
            .map_err(err)?,
        )],
        Call::Wide(rows) => vec![(
            hcd::wide_symbol(decode, ranks).map_err(err)?,
            hcd::wide_plan_with(decode.shape, rows).map_err(err)?,
            hcd::wide_arguments(hcd::Call {
                rows,
                ..decode_call
            })
            .map_err(err)?,
        )],
    })
}

/// The calls a bundle serving `form`'s rank count launches, each with a live
/// count, some below the prepared rows:
/// - prefill at its live token count, the one-shot entry up to
///   `ONE_SHOT_ROWS` and the phased pair above, also at the power-of-two
///   graph buckets above the wide cap, which take the general entry;
/// - every graph bucket up to the decode entry's `MAX_ROWS` (graph buckets
///   are powers of two: `Items::bucket` and the verify and drafter buckets),
///   through v3 up to `LL_MAX_ROWS` rows and v2 above, as the bundle picks;
/// - every larger graph bucket that `graph_takes_wide`, through the wide
///   entry.
fn serving_calls(form: &NormTransportForms) -> LabResult<Vec<(Call, usize)>> {
    let one_shot = hc::ONE_SHOT_ROWS as usize;
    let mut calls = vec![
        (Call::Prefill(1), 1),
        (Call::Prefill(7), 7),
        (Call::Prefill(hc::ONE_SHOT_ROWS), one_shot - 3),
        (Call::Prefill(hc::ONE_SHOT_ROWS + 1), one_shot + 1),
        (Call::Prefill(100), 97),
    ];
    let buckets = bundle_runtime::geometry::Items::new(bundle_runtime::geometry::MAX_ITEMS)
        .ok_or("graph bucket items")?;
    for bucket in buckets.buckets() {
        let live = if bucket > 4 { bucket - 3 } else { bucket } as usize;
        if bucket <= hcd::MAX_ROWS {
            calls.push((Call::Decode(bucket, bucket <= hcd::LL_MAX_ROWS), live));
        } else if hcd::graph_takes_wide(form.ranks, bucket).map_err(err)? {
            calls.push((Call::Wide(bucket), live));
        } else if bucket as usize <= CAPACITY {
            calls.push((Call::Prefill(bucket), live));
        }
    }
    Ok(calls)
}

/// The one-row call each live decode row runs alone in the invariance case.
const ALONE: Ids = Ids {
    input: 130,
    residual: 131,
    logits: 132,
    weight: 103,
    residual_out: 124,
    normed: 125,
    live: 119,
};

/// The scaled fixture's call (`SMALL`): its partials and residual.
const SCALED: Ids = Ids {
    input: 140,
    residual: 141,
    ..MAIN
};

/// A call over rows of the main fixture gathered in another order.
const REORDERED: Ids = Ids {
    input: 150,
    residual: 151,
    logits: 152,
    ..MAIN
};

/// Each fixture row's F32 bits, residual then norm, from the first call
/// that computed it.
struct FirstBits {
    wide: usize,
    rows: RefCell<BTreeMap<usize, [Vec<u16>; 2]>>,
}

impl FirstBits {
    fn new(wide: usize) -> Self {
        Self {
            wide,
            rows: RefCell::default(),
        }
    }

    /// Record or compare the rows `order` names at their positions in
    /// `outputs`; returns how many it compared with earlier bits.
    fn compare(&self, what: &str, order: &[usize], outputs: &[Vec<u16>; 2]) -> LabResult<usize> {
        let wide = self.wide;
        let mut first = self.rows.borrow_mut();
        let mut compared = 0;
        for (at, &row) in order.iter().enumerate() {
            let bits = [0, 1].map(|index| outputs[index][at * wide..(at + 1) * wide].to_vec());
            match first.get(&row) {
                Some(earlier) if *earlier != bits => {
                    return Err(format!(
                        "{what}: row {row} at position {at} differs from its first call's bits"
                    ));
                }
                Some(_) => compared += 1,
                None => {
                    first.insert(row, bits);
                }
            }
        }
        Ok(compared)
    }
}

/// `FirstBits` records unseen rows, compares seen ones at any position and
/// names a row whose residual or norm bits changed.
#[test]
fn first_bits_compare_rows_at_any_position() {
    let rows = |ids: &[u16]| -> [Vec<u16>; 2] {
        [
            ids.iter().flat_map(|&id| [id, id + 1]).collect(),
            ids.iter().flat_map(|&id| [id + 2, id + 3]).collect(),
        ]
    };
    let first = FirstBits::new(2);
    assert_eq!(first.compare("a", &[0, 1, 2], &rows(&[0, 10, 20])), Ok(0));
    assert_eq!(first.compare("b", &[2, 0, 3], &rows(&[20, 0, 30])), Ok(2));
    let mut changed = rows(&[10, 30]);
    changed[1][2] ^= 1;
    assert_eq!(
        first.compare("c", &[1, 3], &changed),
        Err("c: row 3 at position 1 differs from its first call's bits".into())
    );
}

/// Greatest common divisor.
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Calls over rows the serving calls computed, each with the main
/// fixture's rows it gathers in order:
/// - every serving call again, over as many rows spread across every row
///   the serving calls cover, in a permuted order;
/// - the longest one-shot-or-more prefill split at the one-shot boundary:
///   the serving call of `ONE_SHOT_ROWS + 1` rows is its first chunk, and
///   its other rows run as a chunk of their own.
fn reordered_calls(form: &NormTransportForms) -> LabResult<Vec<(Call, Vec<usize>)>> {
    let calls = serving_calls(form)?;
    let covered = calls
        .iter()
        .map(|&(_, live)| live)
        .max()
        .ok_or("no serving calls")?;
    let stride = (37..).find(|&s| gcd(s, covered) == 1).ok_or("no stride")?;
    let mut reordered = calls
        .iter()
        .map(|&(call, live)| {
            let order = (0..live).map(|i| (i * stride + covered - 1) % covered);
            (call, order.collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let first = hc::ONE_SHOT_ROWS as usize + 1;
    let (_, long) = calls
        .iter()
        .copied()
        .filter(|(call, _)| matches!(call, Call::Prefill(_)))
        .max_by_key(|&(_, live)| live)
        .ok_or("no prefill call")?;
    let rest = (long - first).next_power_of_two() as u32;
    reordered.push((Call::Prefill(rest), (first..long).collect()));
    Ok(reordered)
}

/// Every launch of every serving call of every declared form, both dtypes
/// and every rank, plans a configuration the device accepts and arguments
/// that validate: the CPU check of the GPU cases' launch shapes.
#[test]
fn serving_launch_shapes_are_valid_on_cpu() {
    let (_, transports) = forms();
    assert!(!transports.is_empty());
    for form in &transports {
        let query = prefill_query(form);
        let mut calls = serving_calls(form).unwrap();
        let decode = calls
            .iter()
            .filter(|(call, _)| matches!(call, Call::Decode(..)))
            .count();
        assert_eq!(decode, 6, "decode buckets 1 to 32 at {} ranks", form.ranks);
        assert!(calls.iter().any(|(call, _)| matches!(call, Call::Wide(_))));
        let covered = calls.iter().map(|&(_, live)| live).max().unwrap();
        let mut entries = BTreeSet::new();
        for (call, order) in reordered_calls(form).unwrap() {
            let rows = order.iter().copied().collect::<BTreeSet<_>>();
            assert_eq!(rows.len(), order.len(), "{call:?} {order:?}");
            assert!(order.iter().all(|&row| row < covered), "{call:?} {order:?}");
            assert!(order.iter().enumerate().any(|(at, &row)| at != row));
            entries.insert(match call {
                Call::Prefill(rows) if rows <= hc::ONE_SHOT_ROWS => "one-shot",
                Call::Prefill(_) => "phased",
                Call::Decode(_, true) => "v3",
                Call::Decode(_, false) => "v2",
                Call::Wide(_) => "wide",
            });
            calls.push((call, order.len()));
        }
        // Rows move across every entry that promises their bits.
        assert_eq!(entries.len(), 5, "{entries:?}");
        calls.push((Call::Decode(1, true), 1));
        for (call, live) in calls {
            let rows = match call {
                Call::Prefill(rows) | Call::Decode(rows, _) | Call::Wide(rows) => rows as usize,
            };
            assert!(live >= 1 && live <= rows && rows <= CAPACITY, "{call:?}");
            for f32 in [false, true] {
                for rank in 0..form.ranks as usize {
                    let ids = if matches!(call, Call::Decode(1, true)) {
                        ALONE
                    } else {
                        MAIN
                    };
                    let planned = launches(form, query, call, f32, ids, rank)
                        .unwrap_or_else(|e| panic!("{call:?} at {} ranks: {e}", form.ranks));
                    assert!(!planned.is_empty());
                    for (symbol, plan, args) in planned {
                        launch::config(plan).unwrap_or_else(|e| panic!("{symbol} {call:?}: {e}"));
                        args.validate()
                            .unwrap_or_else(|e| panic!("{symbol} {call:?}: {e}"));
                    }
                }
            }
        }
    }
}

struct Lab<'a> {
    sessions: &'a [Session],
    fixtures: Vec<BTreeMap<u64, DeviceBuffer>>,
    form: &'a NormTransportForms,
    query: SigmoidMeanHyperConnectionQuery,
    /// BF16 then F32.
    modules: [Modules; 2],
    wide: usize,
}

impl Lab<'_> {
    /// (symbol, plan, arguments) of each launch of `call` on `rank`.
    fn launches(
        &self,
        call: Call,
        f32: bool,
        ids: Ids,
        rank: usize,
    ) -> LabResult<Vec<(&'static str, LaunchPlan, ArgumentList)>> {
        launches(self.form, self.query, call, f32, ids, rank)
    }

    fn bindings(&self, rank: usize) -> launch::Bindings<'_> {
        let mut buffers: BTreeMap<_, _> =
            self.fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
        for (peer, fixture) in self.fixtures.iter().enumerate() {
            let peer = peer as u64;
            buffers.insert(peer + 1, &fixture[&106]);
            buffers.insert(peer + 11, &fixture[&107]);
            buffers.insert(peer + 21, &fixture[&116]);
            buffers.insert(peer + 31, &fixture[&117]);
        }
        launch::Bindings { buffers }
    }

    /// One call on every rank, or only on `only`.
    fn run(
        &self,
        only: Option<usize>,
        call: Call,
        f32: bool,
        ids: Ids,
        pdl: bool,
    ) -> LabResult<()> {
        let modules = &self.modules[f32 as usize];
        let modules = match call {
            Call::Prefill(_) => &modules.prefill,
            Call::Decode(..) => &modules.decode,
            Call::Wide(_) => &modules.wide,
        };
        let first = self.launches(call, f32, ids, only.unwrap_or(0))?;
        for (entry, (symbol, plan, _)) in first.into_iter().enumerate() {
            launch::launch_collective(
                self.sessions,
                modules,
                symbol,
                LaunchPlan { pdl, ..plan },
                only,
                |rank| {
                    let (_, _, args) = self
                        .launches(call, f32, ids, rank)?
                        .into_iter()
                        .nth(entry)
                        .ok_or("missing entry")?;
                    Ok(launch::RankCall {
                        args,
                        bindings: self.bindings(rank),
                    })
                },
            )?;
        }
        Ok(())
    }

    fn poison(&self, ids: &[u64]) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (session, fixture) in self.sessions.iter().zip(&self.fixtures) {
            poison::poison(
                session,
                &ids.iter().map(|id| &fixture[id]).collect::<Vec<_>>(),
            )?;
        }
        Ok(())
    }

    fn set(&self, id: u64, data: impl Fn(usize) -> Vec<u8>) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            self.fixtures[rank][&id]
                .copy_from_bytes(&data(rank), &session.stream)
                .map_err(err)?;
        }
        poison::synchronize(self.sessions)
    }

    /// Every rank's live rows of outputs `ids`, whose later rows must hold
    /// poison.
    fn outputs(&self, ids: [u64; 2], live: usize, what: &str) -> LabResult<Vec<[Vec<u16>; 2]>> {
        poison::synchronize(self.sessions)?;
        (0..self.sessions.len())
            .map(|rank| {
                let rows = |id: u64| {
                    live_rows(
                        &self.sessions[rank],
                        &self.fixtures[rank][&id],
                        live,
                        self.wide,
                        &format!("{what}: rank {rank} output {id}"),
                    )
                };
                Ok([rows(ids[0])?, rows(ids[1])?])
            })
            .collect()
    }
}

fn transport_case(sessions: &[Session], form: &NormTransportForms) -> LabResult<()> {
    let ranks = form.ranks as usize;
    let epilogue = form.prefill[1].shape.epilogue;
    let (width, streams) = (epilogue.width as usize, epilogue.streams as usize);
    let wide = width * streams;
    let load = |name: &str| -> LabResult<CollectiveModules> {
        CollectiveModules::from_cubin(sessions, &embedded(name)?)
    };
    let modules = [0, 1].map(|f32| -> LabResult<Modules> {
        Ok(Modules {
            prefill: load(form.prefill[f32].cubin)?,
            decode: load(&form.decode_cubins[f32])?,
            wide: load(&form.wide_cubins[f32])?,
        })
    });
    let [bf16_modules, f32_modules] = modules;
    let query = prefill_query(form);
    let prefill_state = hc::buffers(&form.prefill[1], query).map_err(err)?;
    let decode_state = hcd::wide_buffers_with(form.decode[1].shape, form.ranks).map_err(err)?;
    let w = weights(wide);
    let (bf16_codes, exact) = rounded(&w);
    let inputs = Inputs::new(ranks, width, streams);
    let small = inputs.small();
    let Inputs {
        partials,
        x,
        logits,
    } = &inputs;
    let mut fixtures = Vec::new();
    for (rank, s) in sessions.iter().enumerate() {
        let mut b = BTreeMap::new();
        b.insert(100, s.upload_bf16(&partials[rank])?);
        b.insert(101, s.upload_bf16(x)?);
        b.insert(102, s.upload_bf16(logits)?);
        b.insert(103, s.upload_f32(&w)?);
        b.insert(113, s.upload_bf16(&bf16_codes)?);
        b.insert(123, s.upload_f32(&exact)?);
        for id in [104, 105, 114, 115, 124, 125] {
            b.insert(id, s.allocate(CAPACITY * wide * 2)?);
        }
        b.insert(140, s.upload_bf16(&small.partials[rank])?);
        b.insert(141, s.upload_bf16(&small.x)?);
        for (id, len) in [
            (130, width),
            (131, wide),
            (132, streams),
            (150, CAPACITY * width),
            (151, CAPACITY * wide),
            (152, CAPACITY * streams),
        ] {
            b.insert(id, s.allocate(len * 2)?);
        }
        b.insert(106, s.allocate(prefill_state.inbox_bytes as usize)?);
        b.insert(107, bytes(s, &vec![0; prefill_state.flags_bytes as usize])?);
        b.insert(108, bytes(s, &vec![0; prefill_state.epoch_bytes as usize])?);
        // Only the LL region must start zeroed; the rest holds garbage.
        let mut inbox = vec![0xff; decode_state.inbox_bytes as usize];
        inbox[..decode_state.zeroed_inbox_bytes as usize].fill(0);
        b.insert(116, bytes(s, &inbox)?);
        b.insert(117, bytes(s, &vec![0; decode_state.flags_bytes as usize])?);
        b.insert(118, bytes(s, &vec![0; decode_state.epoch_bytes as usize])?);
        b.insert(109, count(s, 0)?);
        b.insert(119, count(s, 1)?);
        fixtures.push(b);
    }
    let lab = Lab {
        sessions,
        fixtures,
        form,
        query,
        modules: [bf16_modules?, f32_modules?],
        wide,
    };
    let calls = serving_calls(form)?;
    // Every entry promises a row the same bits at any row count and
    // position, so every later call must repeat each row's first bits.
    let first_bits = FirstBits::new(wide);
    let f32_ids = |weight| Ids { weight, ..MAIN };
    let bf16_ids = Ids {
        weight: 113,
        residual_out: 114,
        normed: 115,
        ..MAIN
    };
    let case = |call: Call, live: usize| -> LabResult<()> {
        let what = format!("{call:?} live {live} at {ranks} ranks");
        lab.set(109, |_| (live as u32).to_le_bytes().to_vec())?;
        // F32 representable weights then the BF16 form, back to back on the
        // shared state: the same bits.
        lab.poison(&[104, 105, 114, 115])?;
        lab.run(None, call, true, f32_ids(123), false)?;
        lab.run(None, call, false, bf16_ids, false)?;
        let bf16 = lab.outputs([114, 115], live, &what)?;
        if lab.outputs([104, 105], live, &what)? != bf16 {
            return Err(format!("{what}: F32 form differs from the BF16 form"));
        }
        // Weights BF16 cannot represent: the BF16 form's residual, the same
        // bits on every rank, and the f64 oracle's residual and norm.
        lab.poison(&[104, 105])?;
        lab.run(None, call, true, f32_ids(103), false)?;
        let actual = lab.outputs([104, 105], live, &what)?;
        for (rank, outputs) in actual.iter().enumerate() {
            if outputs[0] != bf16[rank][0] || outputs != &actual[0] {
                return Err(format!("{what}: rank {rank} residual or rank agreement"));
            }
        }
        let reference = inputs.oracle(query, &w, live)?;
        let symbols = lab
            .launches(call, true, MAIN, 0)?
            .into_iter()
            .map(|(symbol, _, _)| symbol)
            .collect::<Vec<_>>();
        let mut receipt: BTreeMap<String, Value> = BTreeMap::from([
            ("symbols".into(), json!(symbols)),
            ("call".into(), json!(format!("{call:?}"))),
            ("ranks".into(), json!(ranks)),
            ("live".into(), json!(live)),
            ("bf16_parity".into(), json!(true)),
            ("rank_agreement".into(), json!(true)),
        ]);
        for (index, output) in [&reference.residual, &reference.normed]
            .into_iter()
            .enumerate()
        {
            check_output(
                &actual[0][index]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                output,
                DType::Bf16,
                &mut receipt,
            )?;
        }
        let order = (0..live).collect::<Vec<_>>();
        receipt.insert(
            "cross_call_rows".into(),
            json!(first_bits.compare(&what, &order, &actual[0])?),
        );
        // The scaled fixture, whose bound narrowing the weights to BF16
        // exceeds.
        lab.poison(&[104, 105])?;
        lab.run(None, call, true, SCALED, false)?;
        let scaled_outputs = lab.outputs([104, 105], live, &what)?;
        if scaled_outputs
            .iter()
            .any(|outputs| outputs != &scaled_outputs[0])
        {
            return Err(format!("{what}: scaled fixture rank agreement"));
        }
        let scaled_reference = small.oracle(query, &w, live)?;
        let mut scaled_receipt = BTreeMap::new();
        for (index, output) in [&scaled_reference.residual, &scaled_reference.normed]
            .into_iter()
            .enumerate()
        {
            check_output(
                &scaled_outputs[0][index]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                output,
                DType::Bf16,
                &mut scaled_receipt,
            )?;
        }
        receipt.insert("scaled_oracle".into(), json!(scaled_receipt));
        // Captured PDL graphs repeat the eager bits.
        {
            let lab = &lab;
            let replays = RankReplay::new(sessions, move |rank, _| {
                lab.run(Some(rank), call, true, f32_ids(103), true)
            })?;
            for _ in 0..2 {
                lab.poison(&[104, 105])?;
                replays.launch(sessions)?;
                if lab.outputs([104, 105], live, &what)? != actual {
                    return Err(format!("{what}: graph replay differs from eager"));
                }
            }
        }
        receipt.insert("graph_replays".into(), json!(2));
        // A decode or wide row gives a one-row call's bits: each live row
        // alone, moved to row 0, through the one-row LL entry.
        if let Call::Decode(rows, _) | Call::Wide(rows) = call
            && rows > 2
        {
            for row in 0..live {
                lab.set(130, |rank| {
                    bf16_bytes(&partials[rank][row * width..(row + 1) * width])
                })?;
                lab.set(131, |_| bf16_bytes(&x[row * wide..(row + 1) * wide]))?;
                lab.set(132, |_| {
                    bf16_bytes(&logits[row * streams..(row + 1) * streams])
                })?;
                lab.poison(&[124, 125])?;
                lab.run(None, Call::Decode(1, true), true, ALONE, false)?;
                for (rank, outputs) in lab.outputs([124, 125], 1, &what)?.iter().enumerate() {
                    for (index, output) in outputs.iter().enumerate() {
                        if output[..] != actual[rank][index][row * wide..(row + 1) * wide] {
                            return Err(format!("{what}: row {row} alone differs"));
                        }
                    }
                }
            }
            receipt.insert("one_row_invariance".into(), json!(live));
        }
        sessions[0].receipt("norm_weight_forms", receipt)?;
        Ok(())
    };
    for (call, live) in calls {
        case(call, live).map_err(|e| format!("{call:?} live {live}: {e}"))?;
    }
    // The serving calls' rows again at other positions, row counts and
    // protocols: each gives its first call's bits.
    let gather = |values: &[u16], len: usize, order: &[usize]| {
        bf16_bytes(
            &order
                .iter()
                .flat_map(|&row| values[row * len..(row + 1) * len].iter().copied())
                .collect::<Vec<_>>(),
        )
    };
    for (call, order) in reordered_calls(form)? {
        let live = order.len();
        let what = format!(
            "{call:?} over rows {:?}.. at {ranks} ranks",
            &order[..live.min(4)]
        );
        lab.set(150, |rank| gather(&partials[rank], width, &order))?;
        lab.set(151, |_| gather(x, wide, &order))?;
        lab.set(152, |_| gather(logits, streams, &order))?;
        lab.set(109, |_| (live as u32).to_le_bytes().to_vec())?;
        lab.poison(&[104, 105])?;
        lab.run(None, call, true, REORDERED, false)?;
        let outputs = lab.outputs([104, 105], live, &what)?;
        if outputs.iter().any(|rank| rank != &outputs[0]) {
            return Err(format!("{what}: rank agreement"));
        }
        if first_bits.compare(&what, &order, &outputs[0])? != live {
            return Err(format!("{what}: rows no serving call computed"));
        }
        sessions[0].receipt(
            "norm_weight_forms",
            BTreeMap::from([
                ("call".into(), json!(format!("{call:?}"))),
                ("ranks".into(), json!(ranks)),
                ("rows".into(), json!(order)),
                ("cross_call_rows".into(), json!(live)),
            ]),
        )?;
    }
    Ok(())
}

/// Every declared F32 all-reduce form at the selected rank count, through
/// peer transport.
#[test]
fn f32_norm_all_reduces() {
    gpu_test_multi(512 << 20, |sessions| {
        let forms = forms()
            .1
            .into_iter()
            .filter(|form| form.ranks as usize == sessions.len())
            .collect::<Vec<_>>();
        if forms.is_empty() {
            return Err(format!(
                "no declared F32 all-reduce form runs at {} ranks",
                sessions.len()
            ));
        }
        for form in &forms {
            transport_case(sessions, form)
                .map_err(|e| format!("F32 all-reduces at {} ranks: {e}", form.ranks))?;
        }
        Ok(())
    })
    .unwrap();
}
