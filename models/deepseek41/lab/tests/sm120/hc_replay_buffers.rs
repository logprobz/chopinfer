//! Authenticated HC replay inputs and disjoint, guarded output allocations.
use super::corpus::{Case, H};
use kernel_harness::{LabResult, Session};
use runtime_cuda::{DeviceBuffer, Stream};
use std::collections::BTreeMap;
const GUARD: usize = 64;
const POISON: u8 = 0xa5;
pub const OUTPUTS: [(&str, usize); 12] = [
    ("rounded", H * 2),
    ("expanded", 4 * H * 2),
    ("collapsed", H * 2),
    ("pre", 16),
    ("post", 16),
    ("comb", 64),
    ("normed", H * 2),
    ("widened", H * 4),
    ("codes", H),
    ("scales", H / 32),
    ("status", 4),
    ("probe", 59 * 4),
];

pub struct Input {
    pub data: BTreeMap<String, DeviceBuffer>,
    pub live: DeviceBuffer,
    pub rows: usize,
    pub eps: u32,
}
impl Input {
    pub fn new(s: &Session, case: &Case, map: &[usize]) -> LabResult<Self> {
        if map.is_empty() || map.len() > 32 || map.iter().any(|&r| r >= case.rows) {
            return Err("invalid replay row map".into());
        }
        let mut data = BTreeMap::new();
        for role in [
            "streams",
            "sum",
            "post_in",
            "comb_in",
            "pre_in",
            "fn",
            "base",
            "scale",
            "norm_weight",
        ] {
            let bytes = case.operand("inputs", role)?;
            let bytes = if ["fn", "base", "scale", "norm_weight"].contains(&role) {
                bytes
            } else {
                let stride = bytes.len() / case.rows;
                map.iter()
                    .flat_map(|r| bytes[r * stride..(r + 1) * stride].iter().copied())
                    .collect()
            };
            let b = s.allocate(bytes.len())?;
            b.copy_from_bytes(&bytes, &s.stream)
                .map_err(|e| e.to_string())?;
            data.insert(role.into(), b);
        }
        let live = s.allocate(4)?;
        live.copy_from_bytes(&(map.len() as u32).to_ne_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            data,
            live,
            rows: map.len(),
            eps: case.eps_bits()?,
        })
    }
}
pub struct Output<const WORDS: usize> {
    pub data: BTreeMap<String, DeviceBuffer>,
    pub scratch: DeviceBuffer,
    pub rows: usize,
}
impl<const WORDS: usize> Output<WORDS> {
    pub fn new(s: &Session, rows: usize) -> LabResult<Self> {
        let mut data = BTreeMap::new();
        for (role, stride) in OUTPUTS {
            data.insert(role.into(), s.allocate(rows * stride + GUARD)?);
        }
        let scratch = s.allocate(rows * WORDS * 4 + GUARD)?;
        let out = Self {
            data,
            scratch,
            rows,
        };
        out.poison(&s.stream, 0)?;
        Ok(out)
    }
    pub fn poison(&self, stream: &Stream, prior: u32) -> LabResult<()> {
        for (role, b) in &self.data {
            let mut bytes = vec![POISON; b.byte_len()];
            if role == "status" {
                for row in 0..self.rows {
                    bytes[4 * row..4 * (row + 1)].copy_from_slice(&prior.to_ne_bytes());
                }
            }
            b.copy_from_bytes(&bytes, stream)
                .map_err(|e| e.to_string())?;
        }
        self.scratch
            .copy_from_bytes(&vec![POISON; self.scratch.byte_len()], stream)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn read(&self, stream: &Stream) -> LabResult<BTreeMap<String, Vec<u8>>> {
        let mut result = BTreeMap::new();
        for (role, b) in &self.data {
            let mut bytes = vec![0; b.byte_len()];
            b.copy_to_bytes(&mut bytes, stream)
                .map_err(|e| e.to_string())?;
            if bytes[bytes.len() - GUARD..].iter().any(|&x| x != POISON) {
                return Err(format!("{role} output guard changed"));
            }
            bytes.truncate(bytes.len() - GUARD);
            result.insert(role.clone(), bytes);
        }
        let mut scratch = vec![0; self.scratch.byte_len()];
        self.scratch
            .copy_to_bytes(&mut scratch, stream)
            .map_err(|e| e.to_string())?;
        if scratch[scratch.len() - GUARD..]
            .iter()
            .any(|&x| x != POISON)
        {
            return Err("scratch guard changed".into());
        }
        Ok(result)
    }
    pub fn assert_disjoint(&self, other: &Self) -> LabResult<()> {
        let regions = |o: &Self| -> LabResult<Vec<(u64, u64)>> {
            o.data
                .values()
                .chain([&o.scratch])
                .map(|b| {
                    Ok((
                        b.device_address_at(0, b.byte_len())
                            .map_err(|e| e.to_string())?,
                        b.byte_len() as u64,
                    ))
                })
                .collect()
        };
        let mut all = regions(self)?;
        all.extend(regions(other)?);
        for (i, &(a, n)) in all.iter().enumerate() {
            for &(b, m) in &all[i + 1..] {
                if a < b + m && b < a + n {
                    return Err("concurrent scratch or outputs overlap".into());
                }
            }
        }
        Ok(())
    }
}
