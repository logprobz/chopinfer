//! Sixteen-row boundaries of real captured rows. Each HC row reads only its
//! own streams, sum and coefficients and the site's weights, so a 16-row
//! case holds real rows of one site: both captured C8 decode forwards (8 + 8
//! rows), or the first 16 rows of each captured Verify-32 forward.
use super::{buffers::Input, corpus};
use kernel_harness::{LabResult, Session};
use std::collections::BTreeMap;
pub const ROWS: usize = 16;
const ROW_ROLES: [&str; 5] = ["streams", "sum", "post_in", "comb_in", "pre_in"];
const WEIGHTS: [&str; 4] = ["fn", "base", "scale", "norm_weight"];

pub struct Composite {
    pub id: String,
    /// `decode8x2` or `verify32`.
    pub kind: &'static str,
    /// (corpus case index, captured row) of each of the 16 rows.
    pub rows: Vec<(usize, usize)>,
}

fn phase(case: &corpus::Case) -> &str {
    case.id.split(':').next().unwrap_or("")
}

/// The authenticated complete corpus and its 16-row cases, site by site.
pub fn load() -> LabResult<(Vec<corpus::Case>, Vec<Composite>, String)> {
    let (cases, complete, sha) = corpus::load()?;
    if !complete {
        return Err("complete captured corpus required".into());
    }
    let mut sites: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        sites
            .entry(case.record["site"].to_string())
            .or_default()
            .push(index);
    }
    let mut composites = Vec::new();
    for (site, members) in &sites {
        let of = |name: &str, rows: usize| -> LabResult<Vec<usize>> {
            let found = members
                .iter()
                .copied()
                .filter(|&i| phase(&cases[i]) == name)
                .collect::<Vec<_>>();
            if found.len() != 2 || found.iter().any(|&i| cases[i].rows != rows) {
                return Err(format!("site {site}: {name} captures differ"));
            }
            Ok(found)
        };
        let decode = of("decode-8", 8)?;
        composites.push(Composite {
            id: format!("decode8x2:{site}"),
            kind: "decode8x2",
            rows: decode
                .iter()
                .flat_map(|&i| (0..8).map(move |r| (i, r)))
                .collect(),
        });
        for (n, &i) in of("verify-32", 32)?.iter().enumerate() {
            composites.push(Composite {
                id: format!("verify32:{n}:{site}"),
                kind: "verify32",
                rows: (0..ROWS).map(|r| (i, r)).collect(),
            });
        }
    }
    for c in &composites {
        check(&cases, c)?;
    }
    Ok((cases, composites, sha))
}

/// Every part of a case is one site: the same weights, epsilon and
/// widening policy.
fn check(cases: &[corpus::Case], c: &Composite) -> LabResult<()> {
    let first = &cases[c.rows[0].0];
    for &(i, _) in &c.rows {
        let other = &cases[i];
        if other.record["site"] != first.record["site"]
            || other.eps_bits()? != first.eps_bits()?
            || other.widening() != first.widening()
        {
            return Err(format!("{}: parts differ", c.id));
        }
        for role in WEIGHTS {
            if other.record["inputs"][role]["sha256"] != first.record["inputs"][role]["sha256"] {
                return Err(format!("{}: {role} differs between parts", c.id));
            }
        }
    }
    if c.rows.len() != ROWS {
        return Err("a case holds 16 rows".into());
    }
    Ok(())
}

pub fn widening(cases: &[corpus::Case], c: &Composite) -> bool {
    cases[c.rows[0].0].widening()
}

/// The case's rows in `map` order (indices into its 16 rows), live = all.
pub fn input(
    s: &Session,
    cases: &[corpus::Case],
    c: &Composite,
    map: &[usize],
) -> LabResult<Input> {
    if map.is_empty() || map.len() > ROWS || map.iter().any(|&r| r >= ROWS) {
        return Err("invalid replay row map".into());
    }
    let first = &cases[c.rows[0].0];
    let mut data = BTreeMap::new();
    for role in ROW_ROLES.into_iter().chain(WEIGHTS) {
        let bytes = if WEIGHTS.contains(&role) {
            first.operand("inputs", role)?
        } else {
            let mut operands = BTreeMap::new();
            let mut bytes = Vec::new();
            for &r in map {
                let (case, row) = c.rows[r];
                if let std::collections::btree_map::Entry::Vacant(e) = operands.entry(case) {
                    e.insert(cases[case].operand("inputs", role)?);
                }
                let operand = &operands[&case];
                let stride = operand.len() / cases[case].rows;
                bytes.extend_from_slice(&operand[row * stride..(row + 1) * stride]);
            }
            bytes
        };
        let b = s.allocate(bytes.len())?;
        b.copy_from_bytes(&bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        data.insert(role.into(), b);
    }
    let live = s.allocate(4)?;
    live.copy_from_bytes(&(map.len() as u32).to_ne_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(Input {
        data,
        live,
        rows: map.len(),
        eps: first.eps_bits()?,
    })
}

/// Captured outputs of row `r` of the case, by role, where the capture
/// recorded the role (the served outputs of that forward).
pub fn captured(
    cases: &[corpus::Case],
    c: &Composite,
) -> LabResult<BTreeMap<String, Vec<Vec<u8>>>> {
    let first = &cases[c.rows[0].0];
    let mut result = BTreeMap::new();
    let roles = first.record["outputs"]
        .as_object()
        .ok_or("missing captured outputs")?
        .iter()
        .filter(|(role, o)| o["file"].is_string() && role.as_str() != "partials")
        .map(|(role, _)| role.clone())
        .collect::<Vec<_>>();
    for role in roles {
        let mut operands = BTreeMap::new();
        let mut rows = Vec::new();
        for &(case, row) in &c.rows {
            if let std::collections::btree_map::Entry::Vacant(e) = operands.entry(case) {
                e.insert(cases[case].operand("outputs", &role)?);
            }
            let operand = &operands[&case];
            let stride = operand.len() / cases[case].rows;
            rows.push(operand[row * stride..(row + 1) * stride].to_vec());
        }
        result.insert(role, rows);
    }
    Ok(result)
}
