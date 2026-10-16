//! Conventional grouped expert entries over expert ranges that omit some of
//! the step's routes, as an expert-sharded rank's ranges would: the unsplit
//! entry writes only the routes its experts own, and a split-K entry with
//! its reduction must write the same routes, eager and in a graph replay.

use super::*;
use kernel_lab::poison::{self, Fill, POISON, WriteMap};

/// Route r goes to expert r; experts from this one on are another rank's.
const OWNED_EXPERTS: usize = 4;

/// Runs `v` for one token over ranges holding only the owned routes, with
/// poisoned output and every split partial holding 1.0, then checks the
/// owned routes against the oracle and every other route for poison.
fn owned_routes(session: &Session, v: &Variant) -> LabResult<()> {
    let tokens = 1;
    let routes = tokens * TOP_K;
    let n = v.query.n as usize;
    let k = v.query.k as usize;
    let gate_up = v.projection == Projection::GateUp;
    let output_n = if gate_up { n / 2 } else { n };
    let experts = v.query.grouped.ok_or("missing grouped shape")?.experts as usize;
    let owned: Vec<usize> = (0..OWNED_EXPERTS).collect();
    let permutation =
        oracle::expert_permute(&owned, experts, v.knobs.tile_m as usize).map_err(err)?;
    let offsets: Vec<i32> = permutation
        .expert_offsets
        .iter()
        .map(|&x| x as i32)
        .collect();
    let sorted: Vec<i32> = permutation
        .forward
        .iter()
        .map(|x| x.map_or(-1, |r| r as i32))
        .collect();
    let offsets = upload_bytes(session, &i32_bytes(&offsets))?;
    let sorted = upload_bytes(session, &i32_bytes(&sorted))?;
    let padded = upload_bytes(session, &(permutation.forward.len() as i32).to_ne_bytes())?;
    let live = upload_bytes(session, &(tokens as i32).to_ne_bytes())?;
    let active = upload_bytes(session, &i32_bytes(&[OWNED_EXPERTS as i32, 0, 1, 2, 3]))?;
    let weights: Vec<f32> = (0..routes).map(|r| 0.07 + r as f32 * 0.0125).collect();
    let topk_weights = session.upload_f32(&weights)?;
    let input_rows = if gate_up { tokens } else { routes };
    let input: Vec<u16> = (0..input_rows * k)
        .map(|index| f32_to_bf16(sample(0xe190 ^ index as u64) * 0.1))
        .collect();
    let input_decoded: Vec<f64> = owned
        .iter()
        .flat_map(|&route| {
            let source = if gate_up { route / TOP_K } else { route };
            input[source * k..(source + 1) * k]
                .iter()
                .map(|&value| f64::from(bf16_to_f32(value)))
        })
        .collect();
    let bank = WeightBank::new(v, 0xf280)?;
    let mut decoded = bank.decoded.concat();
    a16_weight_operand(&mut decoded, v.query.weight, bank.f32_scales[0]);
    let gpu_weights = bank.upload(session)?;
    let activation = session.upload_bf16(&input)?;
    let unit_scale = upload_bytes(session, &[127])?;
    let unit_f32 = session.upload_f32(&[1.0])?;
    let output = session.allocate(routes * output_n * 2)?;
    let counts = RuntimeCounts {
        m: tokens as u32,
        batch: 0,
        context: 0,
        routed_rows: routes as u32,
        split_k: v.knobs.split_k_capacity,
    };
    let partials = session.allocate(v.plan(counts).map_err(err)?.workspace_bytes.max(4) as usize)?;
    poison::poison(session, &[&output])?;
    poison::fill_all(session, &partials, &1.0_f32.to_ne_bytes())?;
    let functions = load_moe(session, v, counts)?;
    let operands = MoeOperands {
        activation: &activation,
        activation_scales: &unit_scale,
        activation_f32_scales: &unit_f32,
        weights: &gpu_weights,
        output: &output,
        topk_weights: &topk_weights,
        offsets: &offsets,
        sorted: &sorted,
        padded: &padded,
        live: &live,
        active: &active,
        quantized_output: &unit_scale,
        quantized_scales: &unit_scale,
        partials: &partials,
    };
    launch_moe(session, v, &functions, counts, &operands, true)?;
    let ranges: Vec<ExpertRowRange> = owned
        .iter()
        .map(|&route| ExpertRowRange {
            expert: route & 1,
            start: route,
            end: route + 1,
        })
        .collect();
    let mut reference = oracle::gemm(
        &input_decoded,
        &decoded,
        GemmShape {
            m: owned.len(),
            n,
            k,
            experts: 2,
        },
        &ranges,
        GemmEpilogue {
            residual: None,
            swiglu: gate_up.then_some(SwigluColumns::ConcatenatedHalves),
        },
    )
    .map_err(err)?;
    if !gate_up {
        for &route in &owned {
            for col in 0..output_n {
                let at = route * output_n + col;
                reference.values[at] *= f64::from(weights[route]);
                reference.sum_abs_products[at] *= f64::from(weights[route]).abs();
            }
        }
    }
    let case = format!(
        "{} split_k {} owned routes {OWNED_EXPERTS} of {routes}",
        v.symbol, v.knobs.split_k_capacity
    );
    check_bf16(
        session,
        &output,
        output_n,
        &owned,
        &reference,
        &case,
        "owned_routes",
    )?;
    let written = poison::read(session, &output)?;
    poison::check(
        &format!("{case}: unowned routes"),
        &written,
        &WriteMap::of(DType::Bf16).prefix(owned.len() * output_n)?,
        Fill::Pattern(&[POISON]),
    )?;
    // A graph replay on the same fills writes the same bytes.
    let graph = session.capture(|_| launch_moe(session, v, &functions, counts, &operands, true))?;
    poison::poison(session, &[&output])?;
    poison::fill_all(session, &partials, &1.0_f32.to_ne_bytes())?;
    graph.launch(&session.stream).map_err(err)?;
    if poison::read(session, &output)? != written {
        return Err(format!("{case}: graph replay differs bitwise from eager"));
    }
    Ok(())
}

#[test]
fn grouped_moe_split_k_writes_only_owned_routes() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let catalog = moe::catalog(kernel_lab::catalogs());
        let mut failures = Vec::new();
        let cases: Vec<&Variant> = catalog
            .iter()
            .filter(|v| {
                v.profile == Profile::StdDenseMoe
                    && v.phase == Phase::Decode
                    && v.path == InputPath::A16Gather
                    && v.query.weight == WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
                    && v.query.epilogues.swiglu != Some(SwigluColumns::Interleaved)
            })
            .collect();
        // Unsplit entries of the same engine are the control.
        if !cases.iter().any(|v| v.reduction_symbol().is_some())
            || !cases.iter().any(|v| v.reduction_symbol().is_none())
        {
            return Err("standard FP8 decode MoE needs split and unsplit entries".into());
        }
        for v in cases {
            if let Err(error) = owned_routes(session, v) {
                failures.push(error);
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("split-K grouped MoE route writes");
}
