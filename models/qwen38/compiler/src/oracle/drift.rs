//! Batch drift between two bundle activation dumps of the same prompt
//! (`models/qwen38/bundle/src/dump.rs`), for example the prompt prefilled
//! alone and packed with other requests. Each dump step's `items.json` gives
//! the items' prefix and token IDs; the prompt's rows are gathered from its
//! prefill items wherever they sit in the packed rows. Taps go by their
//! declared names (`models/qwen38/taps.json`, also for older dumps). Per tap
//! it reports how many rows are bit-identical, the first differing row and
//! the relative distance; for `x.indexer.topk` the sparse rows whose key
//! sets differ and whether the ranks agree. With a reference runner's tap
//! directory (`tools/refrun/taps.py`, the same tap names), it also reports
//! each dump's distance from the reference on the positions the reference
//! stored.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kernel_core::select::{NanKeys, Ties, topk};
use serde::Serialize;

use super::selection::{BLOCK, TOP_BLOCKS};
use super::trace::Tensor;
use super::weights::bf16;
use crate::Result;

/// The ranks of a dump without a rank count in its capture.json: the older
/// schema was written at TP4.
const V1_RANKS: usize = 4;

/// One prefill item of the prompt: its step, rows in the step and prefix.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Piece {
    pub step: String,
    pub start: usize,
    pub rows: usize,
    pub prefix: usize,
    /// Rows of the whole step: the packing it ran in.
    pub step_rows: usize,
    pub items: usize,
    /// The dump's first kept position (`CHOPINFER_TAPS_FROM`), else 0.
    pub from: usize,
}

#[derive(serde::Deserialize)]
struct Item {
    start: usize,
    rows: usize,
    prefix: Option<i64>,
    tokens: Option<Vec<u32>>,
}

#[derive(serde::Deserialize)]
struct Items {
    program: String,
    rows: usize,
    items: Vec<Item>,
    #[serde(default)]
    dump_from: Option<i64>,
}

fn steps(root: &Path) -> Result<Vec<PathBuf>> {
    let mut steps = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join("items.json").exists())
        .collect::<Vec<_>>();
    steps.sort();
    Ok(steps)
}

/// The prefill pieces of occurrence `occurrence` (0 = first) of the prompt
/// `tokens` in a dump, in prompt order.
pub fn locate(root: &Path, tokens: &[u32], occurrence: usize) -> Result<Vec<Piece>> {
    let mut found = Vec::<Vec<Piece>>::new();
    for step in steps(root)? {
        let items: Items = serde_json::from_slice(&std::fs::read(step.join("items.json"))?)?;
        if !items.program.starts_with("Prefill") {
            continue;
        }
        let count = items.items.len();
        for item in items.items {
            let prefix = item.prefix.unwrap_or(0).max(0) as usize;
            let Some(ids) = item.tokens else { continue };
            if prefix + ids.len() > tokens.len() || tokens[prefix..prefix + ids.len()] != ids[..] {
                continue;
            }
            let piece = Piece {
                step: step
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_owned(),
                start: item.start,
                rows: item.rows,
                prefix,
                step_rows: items.rows,
                items: count,
                from: items.dump_from.unwrap_or(0).max(0) as usize,
            };
            match found.last_mut() {
                Some(run)
                    if prefix > 0 && run.last().is_some_and(|p| p.prefix + p.rows == prefix) =>
                {
                    run.push(piece)
                }
                _ if prefix == 0 => found.push(vec![piece]),
                _ => {}
            }
        }
    }
    let run = found
        .into_iter()
        .filter(|run| {
            run.last()
                .is_some_and(|p| p.prefix + p.rows == tokens.len())
        })
        .nth(occurrence)
        .ok_or_else(|| {
            format!(
                "{}: prompt occurrence {occurrence} not found",
                root.display()
            )
        })?;
    Ok(run)
}

#[derive(serde::Deserialize)]
struct IndexLine {
    name: String,
    rank: u32,
    cols: usize,
    dtype: String,
    #[serde(default)]
    first_row: usize,
    rows: usize,
    /// `encoder` for a tap of encoder output rows instead of positions.
    #[serde(default)]
    axis: Option<String>,
}

#[derive(Clone)]
struct TapFile {
    /// The name its files are stored under.
    stored: String,
    cols: usize,
    dtype: String,
    first_row: usize,
    rows: usize,
    encoder: bool,
}

/// Tap names and their layout from a step's `index.jsonl`; none when the
/// step dumped no rows.
fn taps(step: &Path) -> Result<BTreeMap<String, TapFile>> {
    let path = step.join("index.jsonl");
    let mut taps = BTreeMap::new();
    if !path.exists() {
        return Ok(taps);
    }
    let v1 = super::tap_names::is_v1(step)?;
    for line in std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let line: IndexLine = serde_json::from_str(line)?;
        if line.rank == 0
            && let Some(name) = super::tap_names::name(&line.name, v1)
        {
            taps.insert(
                name,
                TapFile {
                    stored: line.name,
                    cols: line.cols,
                    dtype: line.dtype,
                    first_row: line.first_row,
                    rows: line.rows,
                    encoder: line.axis.as_deref() == Some("encoder"),
                },
            );
        }
    }
    Ok(taps)
}

fn decode(raw: &[u8], dtype: &str) -> Result<Vec<f64>> {
    Ok(match dtype {
        "bf16" => raw
            .chunks_exact(2)
            .map(|b| bf16(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        "f32" => raw
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().expect("four bytes"))))
            .collect(),
        "f64" => raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
            .collect(),
        "u32" => raw
            .chunks_exact(4)
            .map(|b| f64::from(u32::from_le_bytes(b.try_into().expect("four bytes"))))
            .collect(),
        other => return Err(format!("unsupported dump dtype {other}").into()),
    })
}

/// The prompt's rows of one tap and rank at the dump's kept positions,
/// [rows, cols], and the position of the first row.
pub(super) fn gather(
    root: &Path,
    pieces: &[Piece],
    name: &str,
    rank: usize,
) -> Result<Option<(Tensor, usize)>> {
    let mut data = Vec::new();
    let mut cols = 0;
    let mut first_position = None;
    for piece in pieces {
        let kept = (0..piece.rows).find(|&i| piece.prefix + i >= piece.from);
        let Some(kept) = kept else { continue };
        let step = root.join(&piece.step);
        let Some(file) = taps(&step)?.get(name).cloned() else {
            return Ok(None);
        };
        cols = file.cols;
        let path = step.join(format!("{}.r{rank}.bin", file.stored));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        let width = bytes.len() / file.rows.max(1);
        let [start, end] = [piece.start + kept, piece.start + piece.rows];
        if start < file.first_row || end > file.first_row + file.rows {
            return Err(
                format!("{}: {name} does not hold the prompt's rows", step.display()).into(),
            );
        }
        let raw = &bytes[(start - file.first_row) * width..(end - file.first_row) * width];
        data.extend(decode(raw, &file.dtype)?);
        first_position.get_or_insert(piece.prefix + kept);
    }
    let Some(first_position) = first_position else {
        return Ok(None);
    };
    let rows = data.len() / cols.max(1);
    Ok(Some((
        Tensor {
            shape: vec![rows, cols],
            data,
        },
        first_position,
    )))
}

/// The ranks a dump step was written at: its capture.json `ranks`, else
/// `V1_RANKS`.
fn ranks(step: &Path) -> Result<usize> {
    let path = step.join("capture.json");
    if !path.exists() {
        return Ok(V1_RANKS);
    }
    let capture: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(capture["ranks"]
        .as_u64()
        .map_or(V1_RANKS, |ranks| ranks as usize))
}

/// A per-rank tap of every rank the dump was written at; `partial` taps are
/// summed over the ranks in rank order, as the all-reduce does.
pub(super) fn gather_ranks(
    root: &Path,
    pieces: &[Piece],
    name: &str,
) -> Result<Option<(Vec<Tensor>, usize)>> {
    let step = &pieces.first().ok_or("no pieces")?.step;
    let mut ranks = Vec::new();
    let mut first = 0;
    for rank in 0..self::ranks(&root.join(step))? {
        let Some((tensor, position)) = gather(root, pieces, name, rank)? else {
            return Ok(None);
        };
        first = position;
        ranks.push(tensor);
    }
    Ok(Some((ranks, first)))
}

fn rank_sum(ranks: &[Tensor]) -> Tensor {
    let mut sum = ranks[0].clone();
    for rank in &ranks[1..] {
        for (s, v) in sum.data.iter_mut().zip(&rank.data) {
            *s += v;
        }
    }
    sum
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        d += (x - y) * (x - y);
        n += y * y;
    }
    if n > 0.0 { (d / n).sqrt() } else { d.sqrt() }
}

/// Two dumps' values of one tap over the prompt's rows.
#[derive(Clone, Debug, Default, Serialize)]
pub struct TapDrift {
    pub name: String,
    pub rows: usize,
    pub rows_equal: usize,
    pub first_differing_row: Option<usize>,
    pub elements_differing: f64,
    pub max_abs: f64,
    pub rel_l2: f64,
    pub last_row_rel_l2: f64,
    /// `x.indexer.topk`: sparse rows (more than 512 complete blocks), those
    /// whose key sets differ, the mean blocks swapped in those rows, whether
    /// the last row differs, and rows where a dump's ranks disagree.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    /// Each dump's distance from the reference runner's tap of the same
    /// name, when the reference holds it and the dump's ranks reduce to it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceDistance>,
    /// `router.logits`: rows whose top-10 expert sets differ.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_flips: Option<usize>,
}

/// Relative L2 distances `||dump - reference|| / ||reference||` of dumps a
/// and b on the positions both they and the reference hold.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReferenceDistance {
    /// The positions compared.
    pub rows: usize,
    /// [a, b] over those rows.
    pub rel_l2: [f64; 2],
    /// [a, b] on the last prompt row, when the reference holds it.
    pub last_row_rel_l2: Option<[f64; 2]>,
}

#[derive(serde::Deserialize)]
struct RunnerItem {
    request: String,
    seq_len: usize,
    tokens: Vec<u32>,
    positions: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct RunnerItems {
    items: Vec<RunnerItem>,
}

/// One sequence's step in a reference runner's tap directory
/// (`tools/refrun/taps.py`, schema `chopinfer.taps.v1`): stored row i of
/// each tap holds position `positions[i]` of the sequence.
#[derive(Serialize)]
pub struct Reference {
    pub step: PathBuf,
    pub request: String,
    #[serde(skip)]
    positions: Vec<usize>,
    #[serde(skip)]
    taps: BTreeMap<String, TapFile>,
}

/// The step of a reference runner's tap directory whose sequence covers
/// the prompt `tokens`: at least as long, with the prompt's token at every
/// stored position inside it (a teacher-forced sequence is the prompt plus
/// its answer). Exactly one step must match.
pub fn reference(root: &Path, tokens: &[u32]) -> Result<Reference> {
    let mut found = Vec::new();
    for step in steps(root)? {
        if super::tap_names::is_v1(&step)? {
            return Err(format!("{}: not a chopinfer.taps.v1 step", step.display()).into());
        }
        let items: RunnerItems = serde_json::from_slice(&std::fs::read(step.join("items.json"))?)?;
        let [item] = &items.items[..] else {
            return Err(format!("{}: a runner step holds one sequence", step.display()).into());
        };
        if item.positions.len() != item.tokens.len() {
            return Err(
                format!("{}: positions and tokens differ in length", step.display()).into(),
            );
        }
        let inside = || {
            item.positions
                .iter()
                .zip(&item.tokens)
                .filter(|&(&p, _)| p < tokens.len())
        };
        if item.seq_len >= tokens.len()
            && inside().next().is_some()
            && inside().all(|(&p, &t)| tokens[p] == t)
        {
            found.push(Reference {
                request: item.request.clone(),
                positions: item.positions.clone(),
                taps: taps(&step)?,
                step,
            });
        }
    }
    match found.len() {
        1 => Ok(found.pop().expect("one step")),
        n => Err(format!(
            "{}: {n} reference sequences hold the prompt",
            root.display()
        )
        .into()),
    }
}

impl Reference {
    /// A position tap's stored rows by position, or none when the step
    /// holds no such tap or it is not a float tap.
    fn rows(&self, name: &str) -> Result<Option<(BTreeMap<usize, usize>, Tensor)>> {
        let Some(file) = self.taps.get(name) else {
            return Ok(None);
        };
        if file.encoder || !matches!(file.dtype.as_str(), "bf16" | "f32" | "f64") {
            return Ok(None);
        }
        let path = self.step.join(format!("{}.r0.bin", file.stored));
        let data = decode(&std::fs::read(&path)?, &file.dtype)?;
        let end = file.first_row + file.rows;
        if data.len() != file.rows * file.cols || end > self.positions.len() {
            return Err(format!("{}: does not hold its indexed rows", path.display()).into());
        }
        let positions = self.positions[file.first_row..end].iter().copied();
        let rows = positions.enumerate().map(|(row, p)| (p, row)).collect();
        let tensor = Tensor {
            shape: vec![file.rows, file.cols],
            data,
        };
        Ok(Some((rows, tensor)))
    }

    /// Dumps a and b's distance from the reference's tap `name`; both hold
    /// the prompt's positions `first..`, the last of which is the prompt's
    /// last row.
    fn distance(
        &self,
        name: &str,
        first: usize,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Option<ReferenceDistance>> {
        let Some((rows, reference)) = self.rows(name)? else {
            return Ok(None);
        };
        if reference.shape[1] != a.shape[1] {
            return Ok(None);
        }
        let last = first + a.shape[0] - 1;
        let shared = (0..a.shape[0])
            .filter_map(|i| rows.get(&(first + i)).map(|&r| (i, r)))
            .collect::<Vec<_>>();
        if shared.is_empty() {
            return Ok(None);
        }
        let expected = rows_at(&reference, shared.iter().map(|s| s.1));
        let distance = |t: &Tensor| rel_l2(&rows_at(t, shared.iter().map(|s| s.0)), &expected);
        let last_row = rows.get(&last).map(|&r| {
            let expected = rows_at(&reference, [r].into_iter());
            let at = |t: &Tensor| rel_l2(&rows_at(t, [last - first].into_iter()), &expected);
            [at(a), at(b)]
        });
        Ok(Some(ReferenceDistance {
            rows: shared.len(),
            rel_l2: [distance(a), distance(b)],
            last_row_rel_l2: last_row,
        }))
    }
}

/// Rows `at` of a [rows, cols] tensor, concatenated.
fn rows_at(t: &Tensor, at: impl Iterator<Item = usize>) -> Vec<f64> {
    let cols = t.shape[1];
    at.flat_map(|i| t.data[i * cols..(i + 1) * cols].iter().copied())
        .collect()
}

const ROUTED: usize = model_qwen38_contract::contract::TOP_K as usize;

/// The top-10 experts of one router row as a sorted set, ties to the lower
/// expert as the kernel breaks them. A dump's NaN logits rank by total order.
fn route(row: &[f64]) -> Vec<usize> {
    let mut order = topk(row, ROUTED, Ties::LowerIndex, NanKeys::TotalOrder, None)
        .expect("total-order keys without a mask");
    order.sort_unstable();
    order
}

fn route_flips(a: &Tensor, b: &Tensor) -> usize {
    let cols = a.shape[1];
    a.data
        .chunks_exact(cols)
        .zip(b.data.chunks_exact(cols))
        .filter(|(x, y)| x != y && route(x) != route(y))
        .count()
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Selection {
    pub sparse_rows: usize,
    pub rows_differing: usize,
    pub mean_blocks_swapped: f64,
    pub last_row_differs: bool,
    pub rank_disagreements: [usize; 2],
}

fn key_set(row: &[f64], position: usize) -> Option<Vec<u32>> {
    let complete = (position + 1) / BLOCK;
    if complete <= TOP_BLOCKS {
        return None;
    }
    let mut set = row.iter().map(|&b| b as u32).collect::<Vec<_>>();
    set.sort_unstable();
    Some(set)
}

fn selection(a: &[Tensor], b: &[Tensor], first_position: usize) -> Selection {
    let cols = a[0].shape[1];
    let rows = a[0].shape[0];
    let mut out = Selection::default();
    let mut swapped = 0usize;
    for row in 0..rows {
        let span = row * cols..(row + 1) * cols;
        let (Some(x), Some(y)) = (
            key_set(&a[0].data[span.clone()], first_position + row),
            key_set(&b[0].data[span.clone()], first_position + row),
        ) else {
            continue;
        };
        out.sparse_rows += 1;
        if x != y {
            out.rows_differing += 1;
            swapped += x
                .iter()
                .filter(|block| y.binary_search(block).is_err())
                .count();
            out.last_row_differs |= row == rows - 1;
        }
        for (index, ranks) in [a, b].iter().enumerate() {
            if ranks[1..]
                .iter()
                .any(|r| r.data[span.clone()] != ranks[0].data[span.clone()])
            {
                out.rank_disagreements[index] += 1;
            }
        }
    }
    out.mean_blocks_swapped = swapped as f64 / out.rows_differing.max(1) as f64;
    out
}

fn drift(name: &str, a: &Tensor, b: &Tensor) -> Result<TapDrift> {
    if a.shape != b.shape {
        return Err(format!("{name}: shapes {:?} and {:?} differ", a.shape, b.shape).into());
    }
    let [rows, cols] = [a.shape[0], a.shape[1]];
    let mut out = TapDrift {
        name: name.to_owned(),
        rows,
        ..TapDrift::default()
    };
    let mut differing = 0usize;
    for row in 0..rows {
        let (x, y) = (
            &a.data[row * cols..(row + 1) * cols],
            &b.data[row * cols..(row + 1) * cols],
        );
        let mut equal = true;
        for (&p, &q) in x.iter().zip(y) {
            if p.to_bits() != q.to_bits() {
                equal = false;
                differing += 1;
                out.max_abs = out.max_abs.max((p - q).abs());
            }
        }
        if equal {
            out.rows_equal += 1;
        } else if out.first_differing_row.is_none() {
            out.first_differing_row = Some(row);
        }
    }
    out.elements_differing = differing as f64 / (rows * cols).max(1) as f64;
    out.rel_l2 = rel_l2(&b.data, &a.data);
    if rows > 0 {
        let last = (rows - 1) * cols..rows * cols;
        out.last_row_rel_l2 = rel_l2(&b.data[last.clone()], &a.data[last]);
    }
    Ok(out)
}

/// Compares every tap present in both dumps over the prompt's kept rows,
/// and with `reference`, each dump with the reference's tap of the same
/// name: a densely compared `replicated` tap's rank 0 and a `rank_partial`
/// tap's rank sum (`models/qwen38/taps.json`); other taps are not compared
/// with it.
pub fn compare(
    a: (&Path, &[Piece]),
    b: (&Path, &[Piece]),
    reference: Option<&Reference>,
) -> Result<Vec<TapDrift>> {
    let last = a.1.last().ok_or("no pieces")?;
    let names = taps(&a.0.join(&last.step))?;
    let mut report = Vec::new();
    for (name, file) in &names {
        let per_rank =
            a.0.join(&last.step)
                .join(format!("{}.r1.bin", file.stored))
                .exists();
        let sharding = super::tap_names::dense_sharding(name);
        let (x, y, distance) = if per_rank {
            let (Some((x, first)), Some((y, other))) =
                (gather_ranks(a.0, a.1, name)?, gather_ranks(b.0, b.1, name)?)
            else {
                continue;
            };
            if first != other {
                return Err(format!("{name}: the dumps keep different positions").into());
            }
            if name.ends_with("x.indexer.topk") {
                let mut row = drift(name, &x[0], &y[0])?;
                row.selection = Some(selection(&x, &y, first));
                report.push(row);
                continue;
            }
            let (sum_x, sum_y) = (rank_sum(&x), rank_sum(&y));
            let distance = match (reference, sharding) {
                (Some(r), Some("replicated")) => r.distance(name, first, &x[0], &y[0])?,
                (Some(r), Some("rank_partial")) => r.distance(name, first, &sum_x, &sum_y)?,
                _ => None,
            };
            (sum_x, sum_y, distance)
        } else {
            let (Some((x, first)), Some((y, other))) =
                (gather(a.0, a.1, name, 0)?, gather(b.0, b.1, name, 0)?)
            else {
                continue;
            };
            if first != other {
                return Err(format!("{name}: the dumps keep different positions").into());
            }
            let distance = match (reference, sharding) {
                (Some(r), Some("replicated")) => r.distance(name, first, &x, &y)?,
                _ => None,
            };
            (x, y, distance)
        };
        let mut row = drift(name, &x, &y)?;
        if name.ends_with("router.logits") {
            row.route_flips = Some(route_flips(&x, &y));
        }
        row.reference = distance;
        report.push(row);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_step(
        root: &Path,
        step: &str,
        items: serde_json::Value,
        taps: &[(&str, Vec<u16>)],
        rows: usize,
    ) {
        let dir = root.join(step);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("items.json"), serde_json::to_vec(&items).unwrap()).unwrap();
        let mut index = String::new();
        for (name, values) in taps {
            let bytes = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(dir.join(format!("{name}.r0.bin")), bytes).unwrap();
            index += &serde_json::json!({"name": name, "rank": 0, "rows": rows, "cols": 2, "dtype": "bf16"}).to_string();
            index += "\n";
        }
        std::fs::write(dir.join("index.jsonl"), index).unwrap();
        if taps.iter().all(|(name, _)| name.starts_with('L')) {
            let capture = serde_json::json!({"schema": "chopinfer.taps.v1"});
            std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
        }
    }

    #[test]
    fn finds_the_prompt_in_packed_steps_and_compares_its_rows() {
        let root = std::env::temp_dir().join(format!("qwen38-drift-{}", std::process::id()));
        let (alone, packed) = (root.join("alone"), root.join("packed"));
        let prompt = [7, 8, 9];
        let one = 0x3f80; // 1.0
        let two = 0x4000; // 2.0
        // Alone: the prompt in two chunks, an older dump. Packed: another
        // item first, then the whole prompt with its last row different, in
        // the shared tap names.
        write_step(
            &alone,
            "step0000",
            serde_json::json!({"program": "Prefill", "rows": 2, "items": [
            {"start": 0, "rows": 2, "prefix": 0, "tokens": [7, 8]}]}),
            &[("layer00.out.x", vec![one; 4])],
            2,
        );
        write_step(
            &alone,
            "step0001",
            serde_json::json!({"program": "Prefill", "rows": 1, "items": [
            {"start": 0, "rows": 1, "prefix": 2, "tokens": [9]}]}),
            &[("layer00.out.x", vec![one; 2])],
            1,
        );
        write_step(
            &packed,
            "step0000",
            serde_json::json!({"program": "Prefill", "rows": 4, "items": [
            {"start": 0, "rows": 1, "prefix": 0, "tokens": [5]},
            {"start": 1, "rows": 3, "prefix": 0, "tokens": [7, 8, 9]}]}),
            &[("L00.out", vec![two, two, one, one, one, one, one, two])],
            4,
        );
        let a = locate(&alone, &prompt, 0).unwrap();
        let b = locate(&packed, &prompt, 0).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(
            (b.len(), b[0].start, b[0].step_rows, b[0].items),
            (1, 1, 4, 2)
        );
        assert!(locate(&packed, &prompt, 1).is_err());
        let report = compare((&alone, &a), (&packed, &b), None).unwrap();
        assert_eq!(report.len(), 1);
        let row = &report[0];
        assert_eq!(row.name, "L00.out");
        assert_eq!(
            (row.rows, row.rows_equal, row.first_differing_row),
            (3, 2, Some(2))
        );
        assert_eq!(row.max_abs, 1.0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn kept_positions_and_reference_positions_line_up() {
        let root = std::env::temp_dir().join(format!("qwen38-drift-from-{}", std::process::id()));
        let one = 0x3f80; // 1.0
        let two = 0x4000; // 2.0
        // Positions 1.. of the prompt [7, 8, 9] are kept: packed rows
        // [other, 7, 8, 9] dump rows 2..4.
        let dir = root.join("step0000");
        std::fs::create_dir_all(&dir).unwrap();
        let items = serde_json::json!({"program": "Prefill", "rows": 4, "dump_from": 1, "items": [
            {"start": 0, "rows": 1, "prefix": 0, "tokens": [5]},
            {"start": 1, "rows": 3, "prefix": 0, "tokens": [7, 8, 9]}]});
        std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
        let bytes = [one, one, two, two]
            .iter()
            .flat_map(|v: &u16| v.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(dir.join("layer00.out.x.r0.bin"), bytes).unwrap();
        std::fs::write(
            dir.join("index.jsonl"),
            serde_json::json!({"name": "layer00.out.x", "rank": 0, "first_row": 2, "rows": 2, "cols": 2, "dtype": "bf16"}).to_string(),
        )
        .unwrap();
        let pieces = locate(&root, &[7, 8, 9], 0).unwrap();
        // An older dump: its layer00.out.x is L00.out.
        let (tensor, first) = gather(&root, &pieces, "L00.out", 0).unwrap().unwrap();
        assert_eq!((first, tensor.data), (1, vec![1.0, 1.0, 2.0, 2.0]));
        // The reference ran the prompt and a forced answer [7, 8, 9, 4] and
        // stored positions 0, 2 and 3: of the dump's positions 1 and 2, only
        // 2, the prompt's last, is compared.
        let runner = root.join("runner");
        let out = ("L00.out", 2, vec![9.0, 9.0, 4.0, 4.0, 0.0, 0.0]);
        write_runner(&runner, "step0000", ([7, 8, 9, 4], [0, 2, 3]), &[out]);
        let reference = reference(&runner, &[7, 8, 9]).unwrap();
        assert_eq!(reference.request, "step0000");
        let report = compare((&root, &pieces), (&root, &pieces), Some(&reference)).unwrap();
        let expected = ReferenceDistance {
            rows: 1,
            rel_l2: [0.5; 2],
            last_row_rel_l2: Some([0.5; 2]),
        };
        assert_eq!(report[0].reference, Some(expected));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// One reference runner step (tools/refrun/taps.py) of a sequence with
    /// its stored positions, and F32 position taps (name, cols, values); it
    /// also holds an encoder tap and an integer tap, which are not compared.
    fn write_runner<const N: usize, const S: usize>(
        root: &Path,
        step: &str,
        (sequence, positions): ([u32; N], [usize; S]),
        taps: &[(&str, usize, Vec<f32>)],
    ) {
        let dir = root.join(step);
        std::fs::create_dir_all(&dir).unwrap();
        let capture = serde_json::json!({"schema": "chopinfer.taps.v1", "producer": "refrun"});
        std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
        let tokens = positions.map(|p| sequence[p]);
        let items = serde_json::json!({"program": "reference", "rows": S, "items": [
            {"request": step, "start": 0, "rows": S, "prefix": 0, "seq_len": N,
             "tokens": tokens.to_vec(), "positions": positions.to_vec()}]});
        std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
        let mut index = String::new();
        let extra = [
            ("enc.vision.out", 2, "f32", Some("encoder")),
            ("L00.router.ids", 1, "u32", None),
        ];
        for (name, cols, values) in taps {
            let bytes = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            std::fs::write(dir.join(format!("{name}.r0.bin")), bytes).unwrap();
            let line = serde_json::json!({"name": name, "rank": 0, "first_row": 0,
                "rows": values.len() / cols, "cols": cols, "dtype": "f32"});
            index += &format!("{line}\n");
        }
        for (name, cols, dtype, axis) in extra {
            std::fs::write(dir.join(format!("{name}.r0.bin")), vec![0; 4 * cols * S]).unwrap();
            let mut line = serde_json::json!({"name": name, "rank": 0, "first_row": 0,
                "rows": S, "cols": cols, "dtype": dtype});
            if let Some(axis) = axis {
                line["axis"] = axis.into();
            }
            index += &format!("{line}\n");
        }
        std::fs::write(dir.join("index.jsonl"), index).unwrap();
    }

    #[test]
    fn the_reference_is_the_one_runner_sequence_holding_the_prompt() {
        let root = std::env::temp_dir().join(format!("qwen38-drift-runner-{}", std::process::id()));
        let tap = || vec![("L00.out", 1, vec![0.0; 3])];
        // Another prompt, and the prompt's start only.
        write_runner(&root, "step0000", ([7, 8, 5, 4], [0, 2, 3]), &tap());
        write_runner(
            &root,
            "step0001",
            ([7, 8], [0, 1]),
            &[("L00.out", 1, vec![0.0; 2])],
        );
        write_runner(&root, "step0002", ([7, 8, 9], [0, 1, 2]), &tap());
        let found = reference(&root, &[7, 8, 9]).unwrap();
        assert_eq!(found.request, "step0002");
        assert!(found.rows("enc.vision.out").unwrap().is_none());
        assert!(found.rows("L00.router.ids").unwrap().is_none());
        assert!(reference(&root, &[6, 8, 9]).is_err());
        write_runner(&root, "step0003", ([7, 8, 9, 1], [0, 1, 2]), &tap());
        assert!(
            reference(&root, &[7, 8, 9]).is_err(),
            "two sequences hold it"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn reference_distances_reduce_ranks_by_their_sharding() {
        let root = std::env::temp_dir().join(format!("qwen38-drift-shards-{}", std::process::id()));
        let dir = root.join("step0000");
        std::fs::create_dir_all(&dir).unwrap();
        let items = serde_json::json!({"program": "Prefill", "rows": 1, "items": [
            {"start": 0, "rows": 1, "prefix": 0, "tokens": [7]}]});
        std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
        let capture = serde_json::json!({"schema": "chopinfer.taps.v1", "ranks": 2});
        std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
        // Rank partials 1 and 2; a replicated tap both ranks write; column
        // shards, whose rank sum is not the reference's value.
        let mut index = String::new();
        for (name, values) in [
            ("L00.mlp.out", [0x3f80_u16, 0x4000]),
            ("L03.x.indexer.qk", [0x3f80, 0x3f80]),
            ("L03.attn.core", [0x3f80, 0x4000]),
        ] {
            for (rank, bits) in values.iter().enumerate() {
                std::fs::write(dir.join(format!("{name}.r{rank}.bin")), bits.to_le_bytes())
                    .unwrap();
            }
            let line =
                serde_json::json!({"name": name, "rank": 0, "rows": 1, "cols": 1, "dtype": "bf16"});
            index += &format!("{line}\n");
        }
        std::fs::write(dir.join("index.jsonl"), index).unwrap();
        let runner = root.join("runner");
        let taps = [
            ("L00.mlp.out", 1, vec![3.0]),
            ("L03.x.indexer.qk", 1, vec![1.0]),
            ("L03.attn.core", 1, vec![3.0]),
        ];
        write_runner(&runner, "step0000", ([7], [0]), &taps);
        let pieces = locate(&root, &[7], 0).unwrap();
        let found = reference(&runner, &[7]).unwrap();
        let report = compare((&root, &pieces), (&root, &pieces), Some(&found)).unwrap();
        let distance = |name: &str| {
            let tap = report.iter().find(|t| t.name == name).unwrap();
            tap.reference.as_ref().map(|d| d.rel_l2)
        };
        assert_eq!(distance("L00.mlp.out"), Some([0.0; 2]));
        assert_eq!(distance("L03.x.indexer.qk"), Some([0.0; 2]));
        assert_eq!(distance("L03.attn.core"), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn per_rank_taps_take_the_dump_rank_count() {
        let root = std::env::temp_dir().join(format!("qwen38-drift-ranks-{}", std::process::id()));
        let dir = root.join("step0000");
        std::fs::create_dir_all(&dir).unwrap();
        let items = serde_json::json!({"program": "Prefill", "rows": 1, "items": [
            {"start": 0, "rows": 1, "prefix": 0, "tokens": [7]}]});
        std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
        let capture = serde_json::json!({"schema": "chopinfer.taps.v1", "ranks": 2});
        std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
        let line = serde_json::json!({"name": "L00.mlp.out", "rank": 0, "rows": 1, "cols": 1, "dtype": "bf16"});
        std::fs::write(dir.join("index.jsonl"), line.to_string()).unwrap();
        for (rank, bits) in [(0, 0x3f80_u16), (1, 0x4000)] {
            std::fs::write(
                dir.join(format!("L00.mlp.out.r{rank}.bin")),
                bits.to_le_bytes(),
            )
            .unwrap();
        }
        let pieces = locate(&root, &[7], 0).unwrap();
        let (ranks, _) = gather_ranks(&root, &pieces, "L00.mlp.out")
            .unwrap()
            .unwrap();
        assert_eq!(rank_sum(&ranks).data, vec![3.0]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn routes_compare_the_top_ten_as_sets() {
        let row = (0..12).map(f64::from).collect::<Vec<_>>();
        assert_eq!(route(&row), (2..12).collect::<Vec<_>>());
        let mut swapped = row.clone();
        swapped.swap(1, 2);
        let mut reordered = row.clone();
        reordered.swap(10, 11);
        let tensor = |data: Vec<f64>| Tensor {
            shape: vec![1, 12],
            data,
        };
        assert_eq!(route_flips(&tensor(row.clone()), &tensor(reordered)), 0);
        assert_eq!(route_flips(&tensor(row), &tensor(swapped)), 1);
    }

    #[test]
    fn key_sets_compare_as_sets_on_sparse_rows_only() {
        let dense = TOP_BLOCKS * BLOCK + BLOCK - 2;
        assert!(key_set(&[1.0, 0.0], dense).is_none());
        assert_eq!(key_set(&[3.0, 1.0], dense + 1), Some(vec![1, 3]));
    }
}
