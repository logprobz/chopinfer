//! Sixteen- and 32-row boundaries of real captured rows. Each HC row reads
//! only its own streams, sum and coefficients and the site's weights, so a
//! case holds real rows of one site: both captured C8 decode forwards
//! (8 + 8 rows), the first 16 rows of each captured Verify-32 forward, or
//! all 32 rows of it.
use super::{buffers::Input, corpus};
use kernel_harness::{LabResult, Session};
use std::collections::BTreeMap;
const ROW_ROLES: [&str; 5] = ["streams", "sum", "post_in", "comb_in", "pre_in"];
const WEIGHTS: [&str; 4] = ["fn", "base", "scale", "norm_weight"];
/// Case kinds and their row counts.
pub const KINDS: [(&str, usize); 3] = [("decode8x2", 16), ("verify32h", 16), ("verify32", 32)];
/// Eligible HC boundaries per captured forward.
pub const SITES: usize = 77;
pub const WIDE_ROWS: [usize; 7] = [32, 33, 40, 48, 56, 63, 64];

pub struct Composite {
    pub id: String,
    pub kind: &'static str,
    /// Index of the site in the corpus's site order.
    pub site: usize,
    /// (corpus case index, captured row) of each row.
    pub rows: Vec<(usize, usize)>,
}
impl Composite {
    pub fn size(&self) -> usize {
        self.rows.len()
    }
}

fn phase(case: &corpus::Case) -> &str {
    case.id.split(':').next().unwrap_or("")
}

/// The authenticated complete corpus and its cases, site by site.
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
    if sites.len() != SITES {
        return Err("site count differs".into());
    }
    let mut composites = Vec::new();
    for (site_index, (site, members)) in sites.iter().enumerate() {
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
            site: site_index,
            rows: decode
                .iter()
                .flat_map(|&i| (0..8).map(move |r| (i, r)))
                .collect(),
        });
        for (n, &i) in of("verify-32", 32)?.iter().enumerate() {
            for (kind, size) in [("verify32h", 16), ("verify32", 32)] {
                composites.push(Composite {
                    id: format!("{kind}:{n}:{site}"),
                    kind,
                    site: site_index,
                    rows: (0..size).map(|r| (i, r)).collect(),
                });
            }
        }
    }
    for c in &composites {
        check(&cases, c)?;
    }
    composites.sort_by_key(|c| KINDS.iter().position(|(k, _)| *k == c.kind));
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
    if !(KINDS.contains(&(c.kind, c.size()))
        || c.kind == "verify_wide" && WIDE_ROWS.contains(&c.size()))
    {
        return Err(format!("{}: row count differs from its kind", c.id));
    }
    Ok(())
}

/// Concatenate the two authenticated Verify32 captures of each site. No
/// weight or row value is synthesized, and every output row remains pinned.
pub fn load_wide() -> LabResult<(Vec<corpus::Case>, Vec<Composite>, String)> {
    let (cases, original, sha) = load()?;
    let mut result = Vec::new();
    for site in 0..SITES {
        let pair: Vec<_> = original
            .iter()
            .filter(|c| c.site == site && c.kind == "verify32")
            .collect();
        if pair.len() != 2 {
            return Err("wide HC needs two captured forwards per site".into());
        }
        let all: Vec<_> = pair.iter().flat_map(|c| c.rows.iter().copied()).collect();
        for rows in WIDE_ROWS {
            let composite = Composite {
                id: format!("verify-wide-{rows}:{}", pair[0].id),
                kind: "verify_wide",
                site,
                rows: all[..rows].to_vec(),
            };
            check(&cases, &composite)?;
            result.push(composite);
        }
    }
    assert_eq!(result.len(), SITES * WIDE_ROWS.len());
    Ok((cases, result, sha))
}

pub fn widening(cases: &[corpus::Case], c: &Composite) -> bool {
    cases[c.rows[0].0].widening()
}

/// The case's rows in `map` order (indices into its rows), live = all.
pub fn input(
    s: &Session,
    cases: &[corpus::Case],
    c: &Composite,
    map: &[usize],
) -> LabResult<Input> {
    if map.is_empty() || map.len() > c.size() || map.iter().any(|&r| r >= c.size()) {
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

/// Captured outputs of each row of the case, by role, where the capture
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
        .filter(|(_, o)| o["file"].is_string())
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
