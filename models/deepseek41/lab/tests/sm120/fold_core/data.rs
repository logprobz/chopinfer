use super::*;
pub struct Input {
    pub id: String,
    pub n: u32,
    pub k: u32,
    pub groups: u32,
    pub rows: u32,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub weights: Vec<u8>,
    pub weight_scales: Vec<u8>,
    pub expected: Option<Vec<u8>>,
}
pub fn empty(base: &mxfp8_fold::Entry, rows: u32) -> Input {
    let q = base.query;
    Input {
        id: "synthetic".into(),
        n: q.n,
        k: q.k,
        groups: q.groups,
        rows,
        codes: vec![],
        scales: vec![],
        weights: vec![],
        weight_scales: vec![],
        expected: None,
    }
}
fn pattern(n: u64, salt: u32, scale: bool) -> Vec<u8> {
    // A prime period longer than every tested row and column count avoids
    // repeating whole rows at the kernel tile boundaries.
    let seed = (0..65521_u32)
        .map(|i| {
            let v = i.wrapping_mul(747796405).wrapping_add(salt).rotate_left(7);
            if scale {
                119 + (v % 9) as u8
            } else {
                (8 + v % 96) as u8 | ((v >> 8) as u8 & 128)
            }
        })
        .collect::<Vec<_>>();
    let mut bytes = seed.repeat((n as usize).div_ceil(seed.len()));
    bytes.truncate(n as usize);
    bytes
}
pub fn synthetic(base: &mxfp8_fold::Entry, rows: u32) -> Input {
    synthetic_rank(base, rows, 0)
}
/// Distinct deterministic local operands for each logical TP4 rank.
pub fn synthetic_rank(base: &mxfp8_fold::Entry, rows: u32, rank: u32) -> Input {
    assert!(rank < 4);
    let size = base.bytes(rows);
    let mut input = empty(base, rows);
    let salt = rank * 7919;
    input.codes = pattern(size.activations, 173 + salt, false);
    input.scales = pattern(size.activation_scales, 33 + salt, true);
    input.weights = pattern(size.weights, 731 + salt, false);
    input.weight_scales = pattern(size.weight_scales, 381 + salt, true);
    input
}
/// Weights of a pair's second bank: the synthetic pattern with other salts.
pub fn second_bank(base: &mxfp8_fold::Entry) -> (Vec<u8>, Vec<u8>) {
    let size = base.bytes(1);
    (
        pattern(size.weights, 977, false),
        pattern(size.weight_scales, 419, true),
    )
}
pub fn manifest() -> LabResult<Value> {
    serde_json::from_slice(
        &std::fs::read(root().join("inputs/manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
pub(super) fn blob(meta: &Value) -> LabResult<Vec<u8>> {
    use std::{io::Read, os::unix::fs::OpenOptionsExt};
    let name = meta["file"].as_str().ok_or("missing input file")?;
    if name.contains('/') || name.contains('\\') || name == ".." {
        return Err("invalid blob name".into());
    }
    let size = meta["bytes"].as_u64().ok_or("missing input extent")?;
    if size > 512 << 20 {
        return Err("input too large".into());
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x20000)
        .open(root().join("inputs/blobs").join(name))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(size + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 != size || kernel_build::digest(&bytes) != meta["sha256"] {
        return Err("capture bytes changed".into());
    }
    Ok(bytes)
}
fn repeat_rows(source: &[u8], source_rows: usize, rows: usize) -> Vec<u8> {
    let width = source.len() / source_rows;
    (0..rows)
        .flat_map(|r| {
            source[(r % source_rows) * width..(r % source_rows + 1) * width]
                .iter()
                .copied()
        })
        .collect()
}
pub fn captured(base: &mxfp8_fold::Entry, decode: bool) -> LabResult<Vec<Input>> {
    let manifest = manifest()?;
    let q = base.query;
    let mut result = Vec::new();
    for c in manifest["captures"]
        .as_array()
        .ok_or("missing capture matrix")?
    {
        if (c["n"].as_u64(), c["k"].as_u64(), c["groups"].as_u64())
            != (Some(q.n as u64), Some(q.k as u64), Some(q.groups as u64))
        {
            continue;
        }
        let source_rows = c["rows"].as_u64().ok_or("capture rows")? as usize;
        let counts = if decode {
            if source_rows == 8 {
                vec![1, 8]
            } else {
                vec![32]
            }
        } else if c["phase"] == "verify-32" && c["layer"] == 0 {
            vec![64, 2496, 8192]
        } else {
            vec![]
        };
        if counts.is_empty() {
            continue;
        }
        let codes = blob(&c["codes"])?;
        let scales = blob(&c["scales"])?;
        let weights = blob(&c["weights"])?;
        let weight_scales = blob(&c["weight_scales"])?;
        let expected = blob(&c["output"])?;
        if codes.len() != source_rows * q.k as usize * q.groups as usize
            || scales.len() != codes.len() / 32
            || weights.len() != q.n as usize * q.k as usize * q.groups as usize
            || weight_scales.len() != weights.len() / 32
            || expected.len() != source_rows * q.n as usize * q.groups as usize * 4
        {
            return Err("capture shape differs".into());
        }
        for rows in counts {
            result.push(Input {
                id: format!(
                    "{}-{}",
                    c["id"].as_str().ok_or("capture id")?,
                    if rows > source_rows {
                        "repeated"
                    } else {
                        "prefix"
                    }
                ),
                n: q.n,
                k: q.k,
                groups: q.groups,
                rows: rows as u32,
                codes: repeat_rows(&codes, source_rows, rows),
                scales: repeat_rows(&scales, source_rows, rows),
                weights: weights.clone(),
                weight_scales: weight_scales.clone(),
                expected: Some(repeat_rows(&expected, source_rows, rows)),
            });
        }
    }
    Ok(result)
}
/// Extend authenticated Verify32 rows without changing their weights.
pub fn captured_wide(base: &mxfp8_fold::Entry, rows: u32) -> LabResult<Vec<Input>> {
    if !(33..=64).contains(&rows) {
        return Err("wide capture rows outside 33..64".into());
    }
    let mut inputs = captured(base, true)?
        .into_iter()
        .filter(|i| i.rows == 32)
        .collect::<Vec<_>>();
    // This corpus has WQ_a and the WO chain at layers0/39, but no WKV.
    let expected = if (base.query.n, base.query.k, base.query.groups) == (512, 5120, 1) {
        0
    } else {
        2
    };
    if inputs.len() != expected {
        return Err("wide captured layer coverage differs".into());
    }
    for input in &mut inputs {
        input.codes = repeat_rows(&input.codes, 32, rows as usize);
        input.scales = repeat_rows(&input.scales, 32, rows as usize);
        input.expected = input
            .expected
            .as_ref()
            .map(|v| repeat_rows(v, 32, rows as usize));
        input.rows = rows;
        input.id = format!("{}-wide{rows}", input.id);
    }
    Ok(inputs)
}

#[test]
fn row_replication_keeps_complete_rows() {
    assert_eq!(
        repeat_rows(&[1, 2, 3, 4], 2, 5),
        vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    );
}
#[test]
#[ignore = "requires CHOPINFER_FOLD_CORE_ROOT captured inputs"]
fn captured_inputs_match_shapes() {
    let mut count = 0;
    for base in family::bases() {
        let inputs = captured(&base, true).unwrap();
        count += inputs.len();
    }
    assert_eq!(count, 24);
}

#[test]
fn synthetic_rows_are_distinct_across_tile_boundaries() {
    for base in family::bases() {
        let stride = base.query.activation_row_stride as usize;
        let bytes = pattern((64 * stride) as u64, 173, false);
        let rows = bytes
            .chunks_exact(stride)
            .map(|r| &r[..32])
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(rows.len(), 64, "synthetic rows repeat for {:?}", base.query);
    }
}
