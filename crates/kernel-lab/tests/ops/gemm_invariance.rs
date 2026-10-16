//! The dense GEMM invariance adapter: `gemm::catalog(kernel_lab::catalogs())` entries through
//! `Case`, the Qwen3.8 skinny and short-K entries of
//! `gemm::fused::catalog(kernel_lab::catalogs())` on their prefill schedules, its cluster decode
//! buckets on their graph schedule, and the decode hyper-connection up mix
//! (`fused::HC_UP_MIX`). Rows are seeded BF16 values over 13 binades, so
//! any change of accumulation order changes output bits. Every entry
//! launches at its declared K splits, as the bundles do. A fragment prefill
//! chunk alternates the full grid with the 172 CTAs of a pipelined MiMo
//! half.
//!
//! Every compared launch starts with poisoned outputs and a poisoned split
//! workspace, and each output is checked afterwards (`written_rows`): the
//! live rows are written and finite, the rows past them, which every
//! entry's stores skip (`token < live_m`), still hold the poison. Equal
//! poison or the NaN of a sum over unwritten partials in both arms would
//! otherwise compare equal. The self-resetting semaphores are never
//! poisoned.

use super::*;
use crate::invariance::{Adapter, Element, Units, written};
use kernel_core::contract::{ArgumentList, LaunchPlan};
use kernels_sm120::gemm::fused;
use kernels_sm120::invariance::{Declared, OpKind};

pub(super) use kernel_lab::wide_values;

/// Rows past the live ones that a Qwen3.8 entry's outputs carry, which
/// must stay poisoned.
const PADDING_ROWS: usize = 8;

/// The write-check element of a BF16 or F32 GEMM output.
fn output_element(output: DType) -> Element {
    if output == DType::F32 {
        Element::F32
    } else {
        Element::Bf16
    }
}

/// Checks a GEMM output launched on poison: its first `live` rows of
/// `row_bytes` are written and finite, every later byte still holds the
/// poison. The map follows the kernels' stores, which skip tokens at or
/// past live M (`dense.cuh`, `fragment_decode*.cuh`, `fragment_router.cuh`,
/// `swiglu.cuh`, `cluster_decode.cuh`, `qwen38.cuh`, `qwen38_hc.cuh`) and,
/// at the strides the adapter binds, write every column of a live row.
pub(super) fn written_rows(
    label: &str,
    bytes: &[u8],
    live: usize,
    row_bytes: usize,
    element: Element,
) -> Result<(), String> {
    written(
        label,
        bytes,
        &element.write_map()?.byte_prefix(live * row_bytes)?,
        element,
    )
}

/// Replans `case` (prepared at the largest rows it runs) for `m` live rows
/// on `grid` CTAs, or its own plan's grid.
pub(super) fn resize(case: &mut Case, m: u32, grid: Option<u32>) -> LabResult<()> {
    let q = query(&case.variant);
    case.counts.m = m;
    case.counts.batch = m;
    case.plan = gemm::plan(&case.variant.key, case.counts).map_err(|e| e.to_string())?;
    if let Some(grid) = grid {
        case.plan.grid[0] = grid;
    }
    let maps =
        gemm::tensor_maps(&case.variant.key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
    case.args = gemm::arguments(
        &case.variant.key,
        case.counts,
        maps,
        GemmBuffers {
            output: OUTPUT,
            partial: PARTIAL,
            semaphores: SEMAPHORES,
            weight_scales: case
                .buffers
                .contains_key(&WEIGHT_SCALES)
                .then_some(WEIGHT_SCALES),
            activation_scales: None,
            residual: residual_binding(&q),
            weight_global_scale: None,
            activation_global_scale: None,
        },
    )
    .map_err(|e| e.to_string())?;
    let output_n = if q.epilogues.swiglu.is_some() {
        q.n / 2
    } else {
        q.n
    };
    case.output_len = (m * output_n) as usize;
    Ok(())
}

/// `case`'s output rows for activation rows `rows` (`m * k` values) on
/// `grid` CTAs, each row's output as bytes. The output (prepared at the
/// case's capacity) and the split partials start poisoned; the output's
/// `m` live rows must be written and finite and its rows past them
/// untouched. The partials are scratch: a reducing CTA that sums an
/// unwritten one writes NaN, which the finite check rejects.
pub(super) fn launch_rows(
    session: &Session,
    case: &mut Case,
    rows: &[f32],
    grid: Option<u32>,
) -> LabResult<Units> {
    let q = query(&case.variant);
    let k = q.k as usize;
    let m = rows.len() / k;
    if m > q.m_capacity as usize {
        return Err(format!("{m} rows past {}", q.m_capacity));
    }
    let mut padded = vec![0.0_f32; q.m_capacity as usize * k];
    padded[..rows.len()].copy_from_slice(rows);
    let data = activation(
        &padded,
        q.m_capacity as usize,
        k,
        ActivationFormat::Bf16,
        false,
    )?
    .data;
    case.buffers.insert(ACTIVATION, upload(session, &data)?);
    // An epilogue's residual reads the launch's own rows: a zero residual
    // adds exactly to every row, so a chunk's rows equal the whole run's.
    if residual_binding(&q).is_some() {
        let output_n = if q.epilogues.swiglu.is_some() {
            q.n / 2
        } else {
            q.n
        };
        let len = q.m_capacity as usize * output_n as usize;
        let (zeros, _) = residual_fixture(len, q.output, true);
        case.buffers.insert(RESIDUAL, upload(session, &zeros)?);
    }
    resize(case, m as u32, grid)?;
    let function = case.function(session)?;
    case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
    case.launch(session, &function, &session.stream)?;
    kernel_lab::launched(&case.variant.symbol);
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let bytes = poison::read(session, &case.buffers[&OUTPUT])?;
    let row = case.output_len / m * case.output_bytes();
    written_rows(
        &format!("{} output", case.variant.symbol),
        &bytes,
        m,
        row,
        output_element(q.output),
    )?;
    Ok(bytes[..m * row].chunks(row).map(<[u8]>::to_vec).collect())
}

/// The entry a one-row claim compares with: the same profile and query at
/// one row of capacity (for a Qwen3.8 cluster decode bucket, also its
/// segments), or the entry itself.
fn one_row_symbol(symbol: &str) -> LabResult<String> {
    if let Some(entry) = qwen_entry(symbol) {
        if !matches!(entry.kind, fused::Kind::ClusterDecode { .. }) {
            return Ok(symbol.to_owned());
        }
        let one = GemmQuery {
            m_capacity: 1,
            ..entry.query
        };
        let found = fused::catalog(kernel_lab::catalogs())
            .into_iter()
            .filter(|other| {
                other.profile == entry.profile
                    && other.query == one
                    && other.segments == entry.segments
                    && matches!(other.kind, fused::Kind::ClusterDecode { .. })
            })
            .map(|other| other.symbol)
            .collect::<Vec<_>>();
        return match <[String; 1]>::try_from(found) {
            Ok([one]) => Ok(one),
            Err(found) => Err(format!(
                "{symbol}: {} one-row cluster decode entries ({})",
                found.len(),
                found.join(", ")
            )),
        };
    }
    let catalog = gemm::catalog(kernel_lab::catalogs());
    let Some(item) = catalog.iter().find(|item| item.symbol == symbol) else {
        return Ok(symbol.to_owned());
    };
    let q = query(item);
    Ok(catalog
        .iter()
        .find(|other| {
            other.profile == item.profile
                && GemmQuery {
                    m_capacity: q.m_capacity,
                    ..query(other)
                } == q
                && query(other).m_capacity == 1
        })
        .map_or_else(|| symbol.to_owned(), |other| other.symbol.clone()))
}

fn qwen_entry(symbol: &str) -> Option<fused::Entry> {
    fused::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| e.symbol == symbol)
}

/// Prepared cases of the entry under test and its one-row entry: a prefill
/// head's weight is hundreds of megabytes, so each is seeded once. The
/// Qwen3.8 entries keep the last weights they uploaded, by K and segments,
/// which a cluster decode bucket shares with its one-row bucket.
#[derive(Default)]
pub(crate) struct Gemm {
    cases: Vec<(String, Case)>,
    qwen_weights: Option<QwenWeights>,
}

/// (K, segments) and each segment's uploaded weight.
type QwenWeights = ((u32, Vec<u32>), Vec<DeviceBuffer>);

impl Gemm {
    /// Each row's output of `rows` launched in one call.
    fn launch(
        &mut self,
        session: &Session,
        symbol: &str,
        rows: &[f32],
        grid: Option<u32>,
    ) -> LabResult<Units> {
        if let Some(hc) = fused::hc_geometry(kernel_lab::catalogs(), symbol) {
            return hc_up_mix_rows(session, hc, rows).map_err(|e| format!("{symbol}: {e}"));
        }
        if let Some(entry) = qwen_entry(symbol) {
            return self
                .qwen_rows(session, &entry, rows)
                .map_err(|e| format!("{symbol}: {e}"));
        }
        if !self.cases.iter().any(|(cached, _)| cached == symbol) {
            if self.cases.len() == 2 {
                self.cases.clear();
            }
            let capacity = gemm::catalog(kernel_lab::catalogs())
                .into_iter()
                .find(|item| item.symbol == symbol)
                .map(|item| query(&item).m_capacity)
                .ok_or_else(|| format!("{symbol}: not a GEMM entry"))?;
            self.cases.push((
                symbol.to_owned(),
                Case::prepare(session, symbol, capacity.min(1024), false)?,
            ));
        }
        let (_, case) = self
            .cases
            .iter_mut()
            .find(|(cached, _)| cached == symbol)
            .expect("a prepared case");
        launch_rows(session, case, rows, grid)
    }

    /// Values of one row: the activation's K, or for `HC_UP_MIX` a row's
    /// bottleneck inputs and then its four normalized streams.
    fn k(symbol: &str) -> LabResult<usize> {
        if let Some(hc) = fused::hc_geometry(kernel_lab::catalogs(), symbol) {
            return Ok((hc.rank + hc.width * hc.streams) as usize);
        }
        if let Some(entry) = qwen_entry(symbol) {
            return Ok(entry.query.k as usize);
        }
        gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == symbol)
            .map(|item| query(&item).k as usize)
            .ok_or_else(|| format!("{symbol}: not a GEMM entry"))
    }

    /// A Qwen3.8 entry over `rows`, each row's output as bytes. Cluster
    /// decode buckets run their graph schedule and write each segment (a
    /// SwiGLU segment's activated half) to its own output; a row's unit is
    /// its bytes of every output in segment order. Skinny and short-K
    /// entries run their prefill schedule into one output. The weights are
    /// seeded per segment; the activation, outputs and workspace are fresh.
    /// The outputs (with `PADDING_ROWS` rows past the live ones) and the
    /// workspace start poisoned, and each output's live rows must be
    /// written and finite and its padding rows untouched (`written_rows`).
    fn qwen_rows(
        &mut self,
        session: &Session,
        entry: &fused::Entry,
        rows: &[f32],
    ) -> LabResult<Units> {
        let q = entry.query;
        let k = q.k as usize;
        let m = (rows.len() / k) as u32;
        let decode = matches!(entry.kind, fused::Kind::ClusterDecode { .. });
        let columns = match (decode, q.epilogues.swiglu) {
            (false, _) => vec![q.n],
            (true, Some(_)) => vec![q.n / 2],
            (true, None) => entry.segments.clone(),
        };
        let element = output_element(q.output);
        let key = (q.k, entry.segments.clone());
        if self
            .qwen_weights
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            self.qwen_weights = None;
            let weight = seeded_values(q.n as usize * k, 0x0051_5745_4e38 + k as u64)
                .into_iter()
                .map(f32_to_bf16)
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            let mut uploaded = Vec::new();
            let mut first = 0;
            for &segment in &entry.segments {
                let segment = segment as usize;
                let packed = FragmentWeightLayout {
                    rows: segment,
                    k,
                    format: FragmentWeightFormat::Bf16,
                }
                .pack(&weight[first * k * 2..(first + segment) * k * 2])
                .map_err(|e| e.to_string())?
                .bytes;
                uploaded.push(upload(session, &packed)?);
                first += segment;
            }
            self.qwen_weights = Some((key, uploaded));
        }
        let plan = if decode {
            entry.plan(m)
        } else {
            entry.prefill_plan(m)
        }
        .map_err(|e| e.to_string())?;
        let activation = rows.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        let mut owned = BTreeMap::new();
        owned.insert(ACTIVATION, session.upload_bf16(&activation)?);
        let partial = session.allocate((plan.workspace_bytes as usize).max(4))?;
        poison::poison(session, &[&partial])?;
        owned.insert(PARTIAL, partial);
        owned.insert(
            SEMAPHORES,
            upload(
                session,
                &vec![0; (entry.semaphore_words() as usize).max(1) * 4],
            )?,
        );
        let mut b = fused::Buffers {
            activation: ACTIVATION,
            partial: PARTIAL,
            semaphores: SEMAPHORES,
            ..Default::default()
        };
        for (s, &width) in columns.iter().enumerate() {
            let rows = m as usize + PADDING_ROWS;
            let output = session.allocate(rows * width as usize * element.bytes())?;
            poison::poison(session, &[&output])?;
            b.outputs[s] = 200 + s as u64;
            b.output_strides[s] = width;
            owned.insert(b.outputs[s], output);
        }
        let mut buffers = owned
            .iter()
            .map(|(&id, b)| (id, b))
            .collect::<BTreeMap<_, _>>();
        let (_, weights) = self.qwen_weights.as_ref().expect("uploaded weights");
        for (s, weight) in weights.iter().enumerate() {
            b.weights[s] = 100 + s as u64;
            buffers.insert(b.weights[s], weight);
        }
        let args = if decode {
            entry.arguments(m, b)
        } else {
            entry.prefill_arguments(m, b)
        }
        .map_err(|e| e.to_string())?;
        launch_own(session, entry.profile, &entry.symbol, plan, &args, buffers)?;
        // Each output's row stride is the columns it receives: a cluster
        // decode segment's rows (a SwiGLU segment's activated half), or
        // short-K's and skinny's N, skinny's segments side by side
        // (`fused::Buffers`). So every element of a live row is written.
        let outputs = (0..columns.len())
            .map(|s| {
                let bytes = poison::read(session, &owned[&b.outputs[s]])?;
                written_rows(
                    &format!("output {s}"),
                    &bytes,
                    m as usize,
                    columns[s] as usize * element.bytes(),
                    element,
                )?;
                Ok(bytes)
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok((0..m as usize)
            .map(|t| {
                outputs
                    .iter()
                    .zip(&columns)
                    .flat_map(|(bytes, &width)| {
                        let row = width as usize * element.bytes();
                        bytes[t * row..(t + 1) * row].iter().copied()
                    })
                    .collect()
            })
            .collect())
    }
}

impl Adapter for Gemm {
    fn kind(&self) -> OpKind {
        OpKind::Gemm
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        let capacity = qwen_entry(&entry.symbol).map_or_else(
            || {
                gemm::catalog(kernel_lab::catalogs())
                    .into_iter()
                    .find(|item| item.symbol == entry.symbol)
                    .map_or(0, |item| query(&item).m_capacity)
            },
            |e| e.query.m_capacity,
        );
        capacity.min(1024)
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let session = &sessions[0];
        let k = Self::k(&entry.symbol)?;
        let all = wide_values(rows as usize * k, 0x4348_554e_4b53);
        let prefill = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == entry.symbol)
            .is_some_and(|item| gemm::prefill_segments(&item.key).is_ok());
        let mut out = Vec::new();
        let mut start = 0;
        for (i, &end) in ends.iter().enumerate() {
            let window = &all[start as usize * k..end as usize * k];
            let grid = (prefill && ends.len() > 1).then_some(if i % 2 == 0 { 188 } else { 172 });
            out.extend(self.launch(session, &entry.symbol, window, grid)?);
            start = end;
        }
        Ok(out)
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let k = Self::k(&entry.symbol)?;
        let rows = seeds
            .iter()
            .flat_map(|&seed| wide_values(k, seed))
            .collect::<Vec<_>>();
        self.launch(&sessions[0], &entry.symbol, &rows, None)
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let one = one_row_symbol(&entry.symbol)?;
        let k = Self::k(&entry.symbol)?;
        seeds
            .iter()
            .map(|&seed| {
                self.launch(&sessions[0], &one, &wide_values(k, seed), None)?
                    .pop()
                    .ok_or_else(|| format!("{one}: no row"))
            })
            .collect()
    }
}

/// `HC_UP_MIX` over `rows`, each a row's 320 bottleneck inputs and then its
/// four normalized streams, each row's BF16 mix as bytes. The up weight is
/// seeded for every launch; the live row count is a device scalar. The
/// output carries `PADDING_ROWS` rows past the live ones; it starts
/// poisoned, and its live rows must be written and finite and the rest
/// untouched (`run_hc_up_mix` stores tokens below the live count only).
fn hc_up_mix_rows(session: &Session, hc: fused::HcGeometry, rows: &[f32]) -> LabResult<Units> {
    const NORMED: u64 = 101;
    const LIVE: u64 = 102;
    let (rank, wide) = (hc.rank as usize, (hc.width * hc.streams) as usize);
    let width = hc.width as usize;
    let m = rows.len() / (rank + wide);
    let weight = seeded_values(wide * rank, 0x4843_5550)
        .into_iter()
        .map(f32_to_bf16)
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let packed = FragmentWeightLayout {
        rows: wide,
        k: rank,
        format: FragmentWeightFormat::Bf16,
    }
    .pack(&weight)
    .map_err(|e| e.to_string())?
    .bytes;
    let part = |range: std::ops::Range<usize>| {
        rows.chunks(rank + wide)
            .flat_map(|row| row[range.clone()].iter().map(|&v| f32_to_bf16(v)))
            .collect::<Vec<_>>()
    };
    let mut owned = BTreeMap::new();
    owned.insert(WEIGHT, upload(session, &packed)?);
    owned.insert(ACTIVATION, session.upload_bf16(&part(0..rank))?);
    owned.insert(NORMED, session.upload_bf16(&part(rank..rank + wide))?);
    owned.insert(LIVE, upload(session, &(m as u32).to_ne_bytes())?);
    let output = session.allocate((m + PADDING_ROWS) * width * 2)?;
    poison::poison(session, &[&output])?;
    owned.insert(OUTPUT, output);
    let plan = fused::hc_up_mix_plan(hc, m as u32).map_err(|e| e.to_string())?;
    let args = fused::hc_up_mix_arguments(WEIGHT, ACTIVATION, NORMED, OUTPUT, LIVE)
        .map_err(|e| e.to_string())?;
    let buffers = owned.iter().map(|(&id, b)| (id, b)).collect();
    launch_own(
        session,
        model_qwen38_contract::sm120::config::PROFILE,
        hc.up_mix_symbol,
        plan,
        &args,
        buffers,
    )?;
    let bytes = poison::read(session, &owned[&OUTPUT])?;
    written_rows("output", &bytes, m, width * 2, Element::Bf16)?;
    Ok(bytes[..m * width * 2]
        .chunks(width * 2)
        .map(<[u8]>::to_vec)
        .collect())
}

/// Launches `symbol` from the Qwen3.8 GEMM cubin of `profile` that holds
/// it, records the launch and waits for it.
fn launch_own(
    session: &Session,
    profile: Profile,
    symbol: &str,
    plan: LaunchPlan,
    args: &ArgumentList,
    buffers: BTreeMap<u64, &DeviceBuffer>,
) -> LabResult<()> {
    let cubin = fused::cubin_specs(kernel_lab::catalogs(), Some(profile))
        .into_iter()
        .find(|spec| spec.entries.iter().any(|e| e.symbol == symbol))
        .ok_or_else(|| format!("in no {} GEMM cubin", profile.tag()))?;
    let module = session.module(&cubin.name)?;
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    if plan.dynamic_shared_bytes >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    launch::launch(
        &session.context,
        &function,
        &session.stream,
        plan,
        args,
        &Bindings { buffers },
    )?;
    kernel_lab::launched(symbol);
    session.stream.synchronize().map_err(|e| e.to_string())
}

/// The output checks the adapter runs after every launch reject a live row
/// left poisoned in full or in part, a NaN from unwritten partials and a
/// write past the live rows, and accept live rows written in full (no GPU).
#[test]
fn gemm_write_checks_reject_unwritten_outputs() {
    for (element, one, nan) in [
        (
            Element::Bf16,
            0x3f80_u16.to_ne_bytes().to_vec(),
            0x7fc0_u16.to_ne_bytes().to_vec(),
        ),
        (
            Element::F32,
            1.0_f32.to_ne_bytes().to_vec(),
            f32::NAN.to_ne_bytes().to_vec(),
        ),
    ] {
        // Two live rows of three elements and one padding row.
        let row = 3 * element.bytes();
        let check = |bytes: &[u8]| written_rows("output", bytes, 2, row, element);
        let mut output = vec![POISON; 3 * row];
        output[..2 * row].copy_from_slice(&one.repeat(6));
        assert!(check(&output).is_ok(), "{element:?}");
        // A body that returned early: both arms would compare equal poison.
        assert!(check(&vec![POISON; 3 * row]).is_err(), "{element:?}");
        // A live element left poisoned.
        let mut hole = output.clone();
        hole[row + one.len()..row + 2 * one.len()].fill(POISON);
        assert!(check(&hole).is_err(), "{element:?}");
        // A live element summed from poisoned split partials.
        let mut not_finite = output.clone();
        not_finite[..one.len()].copy_from_slice(&nan);
        assert!(check(&not_finite).is_err(), "{element:?}");
        // A padding row written.
        let mut padding = output.clone();
        padding[2 * row..2 * row + one.len()].copy_from_slice(&one);
        assert!(check(&padding).is_err(), "{element:?}");
    }
}
