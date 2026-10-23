//! Exact Flash TP1/TP2/TP4 fold catalogs and their packed twins.
use bundle_deepseek41::sm120::{self, mxfp8_fold as catalog};
use kernel_build::{CubinSpec, EntrySpec};
use kernels_sm120::gemm::{
    mxfp8_fold, mxfp8_fold_cluster_fused, mxfp8_fold_decode, mxfp8_fold_local_wide,
    mxfp8_fold_reuse, mxfp8_fold_ticket, mxfp8_prefill, mxfp8_prefill_compact,
    mxfp8_prefill_matrix, mxfp8_prefill_split, mxfp8_prefill_tma_scales,
};
use model_deepseek41_contract::geometry::FLASH;
use std::collections::{BTreeMap, BTreeSet};

pub const FAMILY: &str = "packed_serving";
pub const CUBIN: &str = "chop_deepseek41_packed_serving_lab";

#[derive(Clone, Debug)]
pub enum Entry {
    Fold(mxfp8_fold::Entry),
    Ticket(mxfp8_fold_ticket::Entry),
    Decode(mxfp8_fold_decode::Entry),
    Reuse(mxfp8_fold_reuse::Entry),
    Prefill(mxfp8_fold::Entry),
    Split(mxfp8_fold::Entry),
    Compact(mxfp8_prefill_compact::Entry, u32),
    Matrix(mxfp8_prefill_matrix::Entry),
    Scales(mxfp8_prefill_tma_scales::Entry),
    Fused(mxfp8_fold_cluster_fused::Entry),
    Wide(mxfp8_fold_local_wide::Entry),
}
impl Entry {
    pub fn base(&self) -> mxfp8_fold::Entry {
        match self {
            Self::Fold(e) | Self::Prefill(e) | Self::Split(e) => e.clone(),
            Self::Ticket(e) => e.legacy().unwrap(),
            Self::Decode(e) => e.legacy().unwrap(),
            Self::Reuse(e) => e.ticket.legacy().unwrap(),
            Self::Compact(e, _) => e.base.clone(),
            Self::Matrix(e) => e.base.clone(),
            Self::Scales(e) => e.base.clone(),
            Self::Fused(e) => e.cluster.base.clone(),
            Self::Wide(e) => e.base.clone(),
        }
    }
    pub fn family(&self) -> &'static str {
        match self {
            Self::Fold(_) => "v2",
            Self::Ticket(e) if e.prefetch_weights => "prefetch",
            Self::Ticket(_) => "ticket",
            Self::Decode(_) => "decode",
            Self::Reuse(_) => "reuse",
            Self::Prefill(_) => "prefill",
            Self::Split(_) => "split_prefill",
            Self::Compact(..) => "compact",
            Self::Matrix(_) => "matrix",
            Self::Scales(_) => "tma_scales",
            Self::Fused(_) => "fused",
            Self::Wide(_) => "wide",
        }
    }
    pub fn paired(&self) -> bool {
        match self {
            Self::Ticket(e) => e.paired,
            Self::Decode(e) => e.paired,
            Self::Reuse(e) => e.ticket.paired,
            _ => false,
        }
    }
    pub fn packed_specs(&self) -> Vec<EntrySpec> {
        match self {
            Self::Fold(e) => vec![e.packed_spec().unwrap()],
            Self::Ticket(e) => vec![e.packed_spec().unwrap()],
            Self::Decode(e) => vec![e.packed_spec().unwrap()],
            Self::Reuse(e) => vec![e.packed_spec().unwrap()],
            Self::Prefill(e) => vec![mxfp8_prefill::packed_spec(e).unwrap()],
            Self::Split(e) => mxfp8_prefill_split::packed_specs(e).unwrap(),
            Self::Compact(e, tile) => vec![e.packed_spec(*tile).unwrap()],
            Self::Matrix(e) => vec![e.packed_spec().unwrap()],
            Self::Scales(e) => vec![e.packed_spec().unwrap()],
            Self::Fused(e) => vec![e.packed_spec().unwrap()],
            Self::Wide(e) => vec![e.packed_spec().unwrap()],
        }
    }
    pub fn specs(&self) -> Vec<EntrySpec> {
        match self {
            Self::Fold(e) => vec![e.spec()],
            Self::Ticket(e) => e.specs(),
            Self::Decode(e) => e.specs(),
            Self::Reuse(e) => e.specs(),
            Self::Prefill(e) => mxfp8_prefill::specs(e).unwrap(),
            Self::Split(e) => mxfp8_prefill_split::specs(e).unwrap(),
            Self::Compact(e, tile) => e
                .specs()
                .unwrap()
                .into_iter()
                .filter(|s| s.symbol == e.admission_symbol() || s.symbol == e.symbol(*tile))
                .collect(),
            Self::Matrix(e) => e
                .specs()
                .unwrap()
                .into_iter()
                .filter(|s| !s.symbol.ends_with("_omit_first_v1"))
                .collect(),
            Self::Scales(e) => e
                .specs()
                .unwrap()
                .into_iter()
                .filter(|s| !s.symbol.ends_with("_omit_first_v1"))
                .collect(),
            Self::Fused(e) => vec![e.spec()],
            Self::Wide(e) => vec![e.spec()],
        }
    }
    pub fn key(&self) -> String {
        self.packed_specs()[0].symbol.clone()
    }
    /// Decode covers a single row, a partial tile and capacity. Large
    /// prefill catalogs use one bucket per symbol, with full and tail
    /// paths in the split case. Exact-row variants prefer registered rows;
    /// retained unused variants use the smallest bucket for their tile.
    pub fn rows(&self) -> Vec<u32> {
        let base = self.base();
        let q = base.query;
        match self {
            Self::Fold(_) if q.m_capacity > 64 => vec![129],
            Self::Fold(_) | Self::Ticket(_) | Self::Decode(_) => {
                let mut rows = vec![1, q.m_capacity.min(7), q.m_capacity];
                rows.dedup();
                rows
            }
            Self::Reuse(e) => vec![
                (6..=32)
                    .find(|&r| {
                        sm120::projection_reuse::choice(q, r)
                            .is_some_and(|(tile, _)| tile == e.tile_n)
                    })
                    .unwrap_or(6),
            ],
            Self::Prefill(_) | Self::Split(_) => vec![129],
            Self::Compact(e, tile) => vec![
                catalog::compact::ROWS
                    .into_iter()
                    .find(|&rows| {
                        mxfp8_prefill_compact::Entry::tile(rows) == *tile
                            && catalog::compact::stages(&FLASH, q, rows) == Some(e.stages)
                    })
                    .unwrap_or(match tile {
                        16 => 1,
                        32 => 128,
                        _ => 512,
                    }),
            ],
            Self::Matrix(e) => vec![
                sm120::prefill_matrix::ROWS
                    .into_iter()
                    .find(|&r| sm120::prefill_matrix::choice(q, r) == Some((e.tile_rows, e.stages)))
                    .unwrap_or(192),
            ],
            Self::Scales(_) => vec![
                [4096, 8192]
                    .into_iter()
                    .find(|&r| sm120::prefill_tma_scales::choice(&FLASH, 4, q, r))
                    .expect("served staged-scale entry"),
            ],
            Self::Fused(e) => vec![e.cluster.tile_rows],
            Self::Wide(_) => vec![16, 32],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub entry: Entry,
    pub topologies: BTreeSet<u32>,
}

pub fn cases() -> Vec<Case> {
    let mut result = BTreeMap::<String, Case>::new();
    let mut add = |entry: Entry, rank| {
        result
            .entry(entry.key())
            .or_insert_with(|| Case {
                entry,
                topologies: BTreeSet::new(),
            })
            .topologies
            .insert(rank);
    };
    for ranks in [1, 2, 4] {
        for e in catalog::entries(&FLASH, ranks).unwrap() {
            add(Entry::Fold(e), ranks);
        }
        for e in catalog::ticket_entries(&FLASH, ranks).unwrap() {
            add(Entry::Ticket(e), ranks);
        }
        for e in catalog::prefetch_entries(&FLASH, ranks).unwrap() {
            add(Entry::Ticket(e), ranks);
        }
        for e in catalog::decode_entries(&FLASH, ranks).unwrap() {
            add(Entry::Decode(e), ranks);
        }
        let prefill = catalog::prefill_entries(&FLASH, ranks).unwrap();
        for e in &prefill {
            add(Entry::Prefill(e.clone()), ranks);
            add(Entry::Split(e.clone()), ranks);
        }
        for e in catalog::compact::catalog().unwrap() {
            if prefill.iter().any(|base| base.query == e.base.query) {
                for tile in [16, 32, 64] {
                    add(Entry::Compact(e.clone(), tile), ranks);
                }
            }
        }
    }
    for e in sm120::projection_reuse::entries().unwrap() {
        add(Entry::Reuse(e), 4);
    }
    for e in sm120::prefill_matrix::catalog().unwrap() {
        add(Entry::Matrix(e), 4);
    }
    for e in sm120::prefill_tma_scales::entries() {
        add(Entry::Scales(e), 4);
    }
    for e in sm120::fold_fused::entries() {
        add(Entry::Fused(e), 4);
    }
    add(Entry::Wide(sm120::fold_fused::wide(32).unwrap()), 4);
    result.into_values().collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let mut entries = BTreeMap::<String, EntrySpec>::new();
    for case in cases() {
        for spec in case
            .entry
            .specs()
            .into_iter()
            .chain(case.entry.packed_specs())
        {
            if let Some(previous) = entries.insert(spec.symbol.clone(), spec.clone()) {
                assert_eq!(previous.body, spec.body);
                assert_eq!(previous.params, spec.params);
                assert_eq!(previous.max_threads, spec.max_threads);
                assert_eq!(previous.min_blocks, spec.min_blocks);
            }
        }
    }
    let mut spec = mxfp8_fold::cubin_spec(CUBIN, &catalog::catalog(&FLASH).unwrap()).unwrap();
    spec.headers = [
        "mxfp8_fold",
        "mxfp8_fold_ticket",
        "mxfp8_fold_prefetch",
        "mxfp8_fold_decode",
        "mxfp8_fold_reuse",
        "mxfp8_prefill",
        "mxfp8_prefill_split",
        "mxfp8_prefill_compact",
        "mxfp8_prefill_matrix",
        "mxfp8_prefill_tma_scales",
        "mxfp8_fold_cluster_fused",
        "mxfp8_fold_local_wide",
    ]
    .map(|header| format!("#include \"gemm/{header}.cuh\""))
    .to_vec();
    spec.entries = entries.into_values().collect();
    vec![spec]
}
