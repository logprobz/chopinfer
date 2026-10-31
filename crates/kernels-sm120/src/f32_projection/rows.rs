//! Whole-K exact-F32 tiles for wide row counts. A CTA owns a feature by row
//! tile over every 128-K chain (`gemm/exact_f32_rows.cuh`), so no chain
//! partial leaves it and no reduction launch follows. Each output keeps the
//! original arithmetic: the prefill preparation's components, the same
//! eight-MMA hi, mid and lo chains merged in that order, chains added from
//! +0 in K order, and the original SIMT chains outside the tensor domain.
use super::{Entry, pointer};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, WeightFormat};

/// A CTA's outputs, each warp's share, the copy ring (`stages` planes of
/// `plane_k` K each) and the K slices: with one, a CTA adds its chains into
/// the output; with more, each chain's value goes to its own plane and a
/// reduction launch adds them in K order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tile {
    pub features: u32,
    pub rows: u32,
    pub warp_features: u32,
    pub warp_rows: u32,
    pub stages: u32,
    pub plane_k: u32,
    pub slices: u32,
}

impl Tile {
    pub fn warps(self) -> u32 {
        (self.features / self.warp_features) * (self.rows / self.warp_rows)
    }
    pub fn threads(self) -> u32 {
        self.warps() * 32
    }
    /// The ring: per plane, `plane_k` K of every weight line and of each
    /// of the three component lines of every row.
    pub fn shared_bytes(self) -> u32 {
        self.stages * (self.features + 3 * self.rows) * self.plane_k * 2
    }
    pub fn validate(self) -> Result<(), Refusal> {
        let fits = |part: u32, whole: u32| part != 0 && whole.is_multiple_of(part);
        let row_tiles = self.warp_rows / 8;
        if !fits(self.warp_features, self.features)
            || !fits(self.warp_rows, self.rows)
            || !self.warp_features.is_multiple_of(16)
            || !self.warp_rows.is_multiple_of(8)
            || !(row_tiles == 1 || row_tiles.is_multiple_of(2))
            || self.stages < 2
            || !matches!(self.plane_k, 32 | 64 | 128)
            || self.slices == 0
            || self.threads() > 1024
            || self.shared_bytes() > crate::backend::SHARED_MEMORY_BYTES
        {
            return Err(Refusal(
                "whole-K F32 tiles need 16-feature by 8-row warp tiles, two stages and bounded shared memory",
            ));
        }
        Ok(())
    }
    fn tag(self) -> String {
        let tag = format!(
            "f{}_r{}_w{}x{}_s{}_k{}",
            self.features, self.rows, self.warp_features, self.warp_rows, self.stages, self.plane_k
        );
        if self.slices > 1 {
            format!("{tag}_z{}", self.slices)
        } else {
            tag
        }
    }
}

/// The caller checks physical span alignment and disjointness before
/// binding these views. The workspace holds the preparation's row domain
/// and components (`prefill::Workspace`); it, the weights and the
/// activation must start on a 16-byte boundary, as the tiles and their
/// SIMT fallback copy them in 16-byte vectors.
#[derive(Clone, Copy)]
pub struct Buffers {
    pub activation: View,
    pub weight: View,
    pub workspace: View,
    pub live_rows: View,
    pub weight_domain: View,
    pub output: View,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    entry: Entry,
    tile: Tile,
    capacity: u32,
}

impl Candidate {
    /// `entry` names the projection (an uncached entry: its symbol is the
    /// stem); `capacity` bounds the planned rows.
    pub fn new(entry: Entry, tile: Tile, capacity: u32) -> Result<Self, Refusal> {
        let q = entry.query();
        tile.validate()?;
        if entry.uses_cached_weights()
            || entry.uses_fused_split()
            || q.weight != WeightFormat::Bf16
            || !q.n.is_multiple_of(16)
            || !q.k.is_multiple_of(128)
            || u64::from(q.n) * u64::from(q.k) >= 1 << 32
            || capacity == 0
            || capacity > q.m_capacity
            || u64::from(capacity) * 3 * u64::from(q.k) >= 1 << 32
            || !(q.k / 128).is_multiple_of(tile.slices)
        {
            return Err(Refusal(
                "whole-K F32 tiles need BF16 weights, 16-feature tiles, whole 128-K chains and bounded rows",
            ));
        }
        Ok(Self {
            entry,
            tile,
            capacity,
        })
    }
    pub fn entry(&self) -> &Entry {
        &self.entry
    }
    pub fn tile(&self) -> Tile {
        self.tile
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    fn stem(&self) -> &str {
        self.entry.symbol().trim_end_matches("_v1")
    }
    pub fn prepare_symbol(&self) -> String {
        format!("{}_rows_prepare_v1", self.stem())
    }
    pub fn symbol(&self) -> String {
        format!("{}_rows_{}_v1", self.stem(), self.tile.tag())
    }
    pub fn reduce_symbol(&self) -> String {
        format!("{}_rows_reduce_v1", self.stem())
    }
    /// The preparation, the compute and, with several slices, the reducer.
    pub fn symbols(&self) -> Vec<String> {
        let mut symbols = vec![self.prepare_symbol(), self.symbol()];
        if self.tile.slices > 1 {
            symbols.push(self.reduce_symbol());
        }
        symbols
    }
    /// The row domain words, padded to 16 bytes, then every row's three
    /// BF16 components, then with several slices every chain's plane.
    pub fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.check(rows)?;
        let q = self.entry.query();
        let chains = if self.tile.slices > 1 {
            u64::from(rows) * u64::from(q.n) * u64::from(q.k / 128) * 4
        } else {
            0
        };
        Ok(self.entry.prefill_workspace_bytes(rows) + chains)
    }
    fn check(&self, rows: u32) -> Result<(), Refusal> {
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("whole-K F32 rows exceed the declared capacity"));
        }
        Ok(())
    }
    pub fn plans(&self, rows: u32) -> Result<Vec<LaunchPlan>, Refusal> {
        self.check(rows)?;
        let q = self.entry.query();
        let prepare = self.entry.stage_plan([rows, 1, 1])?;
        let compute = LaunchPlan {
            grid: [
                rows.div_ceil(self.tile.rows),
                q.n.div_ceil(self.tile.features),
                self.tile.slices,
            ],
            block: [self.tile.threads(), 1, 1],
            dynamic_shared_bytes: self.tile.shared_bytes(),
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&compute)?;
        let mut plans = vec![prepare, compute];
        if self.tile.slices > 1 {
            let elements = rows
                .checked_mul(q.n)
                .ok_or(Refusal("whole-K F32 reduction grid overflows"))?;
            plans.push(self.entry.stage_plan([elements.div_ceil(128), 1, 1])?);
        }
        Ok(plans)
    }
    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.entry.query();
        let t = self.tile;
        let live = pointer("live_rows", "const uint32_t");
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let mut specs = vec![
            EntrySpec::new(
                self.prepare_symbol(),
                vec![
                    pointer("activation", "const float"),
                    live.clone(),
                    rows.clone(),
                    pointer("workspace", "void"),
                ],
                format!(
                    "chop::exact_f32::prefill::prepare<{}>(activation,live_rows,rows,workspace);",
                    q.k
                ),
                128,
                1,
            ),
            EntrySpec::new(
                self.symbol(),
                vec![
                    pointer("activation", "const float"),
                    pointer("weight", "const uint16_t"),
                    pointer("workspace", "void"),
                    pointer("output", "float"),
                    live,
                    rows,
                    pointer("weight_domain", "const uint32_t"),
                ],
                format!(
                    "chop::exact_f32::rows::run<{},{},{},{},{},{},{},{},{}>(activation,weight,workspace,output,live_rows,rows,weight_domain);",
                    q.n,
                    q.k,
                    t.features,
                    t.rows,
                    t.warp_features,
                    t.warp_rows,
                    t.stages,
                    t.plane_k,
                    t.slices
                ),
                t.threads(),
                1,
            ),
        ];
        if t.slices > 1 {
            specs.push(EntrySpec::new(
                self.reduce_symbol(),
                vec![
                    pointer("workspace", "void"),
                    pointer("output", "float"),
                    pointer("live_rows", "const uint32_t"),
                    EntryParam::U32 {
                        name: "rows".into(),
                    },
                ],
                format!(
                    "chop::exact_f32::reduce<{},{}>(chop::exact_f32::rows::partials<{}>(workspace,rows),output,live_rows,rows);",
                    q.n, q.k, q.k
                ),
                128,
                1,
            ));
        }
        specs
    }
    pub fn launches(&self, b: Buffers, rows: u32) -> Result<Vec<Launch>, Refusal> {
        let q = self.entry.query();
        b.activation.covers(u64::from(rows) * u64::from(q.k) * 4)?;
        b.weight.covers(self.entry.weight_bytes())?;
        b.workspace.covers(self.workspace_bytes(rows)?)?;
        b.live_rows.covers(4)?;
        b.weight_domain.covers(self.entry.weight_domain_bytes())?;
        b.output.covers(u64::from(rows) * u64::from(q.n) * 4)?;
        let views = [
            b.activation,
            b.weight,
            b.workspace,
            b.live_rows,
            b.weight_domain,
            b.output,
        ];
        for (i, a) in views.iter().enumerate() {
            if views[i + 1..].iter().any(|v| v.handle == a.handle) {
                return Err(Refusal("whole-K F32 views alias"));
            }
        }
        let ptr = |v: View| Argument::DeviceBuffer(v.handle);
        let plans = self.plans(rows)?;
        let symbols = self.symbols();
        let mut launches = vec![
            Launch {
                symbol: symbols[0].clone(),
                rows,
                plan: plans[0],
                arguments: ArgumentList(vec![
                    ptr(b.activation),
                    ptr(b.live_rows),
                    Argument::U32(rows),
                    ptr(b.workspace),
                ]),
                collective: false,
                persistent: false,
            },
            Launch {
                symbol: symbols[1].clone(),
                rows,
                plan: plans[1],
                arguments: ArgumentList(vec![
                    ptr(b.activation),
                    ptr(b.weight),
                    ptr(b.workspace),
                    ptr(b.output),
                    ptr(b.live_rows),
                    Argument::U32(rows),
                    ptr(b.weight_domain),
                ]),
                collective: false,
                persistent: false,
            },
        ];
        if self.tile.slices > 1 {
            launches.push(Launch {
                symbol: symbols[2].clone(),
                rows,
                plan: plans[2],
                arguments: ArgumentList(vec![
                    ptr(b.workspace),
                    ptr(b.output),
                    ptr(b.live_rows),
                    Argument::U32(rows),
                ]),
                collective: false,
                persistent: false,
            });
        }
        Ok(launches)
    }
}

/// One cubin of `candidates`, with one preparation per projection.
pub fn cubin_spec(name: &str, candidates: &[Candidate]) -> Result<CubinSpec, Refusal> {
    if candidates.is_empty() {
        return Err(Refusal("whole-K F32 tiles require entries"));
    }
    let mut entries: Vec<EntrySpec> = Vec::new();
    for spec in candidates.iter().flat_map(Candidate::specs) {
        match entries.iter().find(|e| e.symbol == spec.symbol) {
            Some(existing) if existing.body != spec.body => {
                return Err(Refusal("whole-K F32 symbols name different bodies"));
            }
            Some(_) => {}
            None => entries.push(spec),
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/exact_f32_rows.cuh\"".into()],
        defines: vec![],
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery};
    use kernel_core::precision::{ExecutionPhase, OperationUse};

    const TILE: Tile = Tile {
        features: 128,
        rows: 64,
        warp_features: 32,
        warp_rows: 32,
        stages: 2,
        plane_k: 64,
        slices: 1,
    };

    fn entry(n: u32, k: u32, weight: WeightFormat) -> Entry {
        Entry::new(
            GemmQuery {
                m_capacity: 192,
                n,
                k,
                weight,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Verify,
                capacity: 192,
                graph: false,
            },
            &format!("chop_test_rows_n{n}_k{k}"),
        )
        .unwrap()
    }

    fn buffers(c: &Candidate, rows: u32) -> Buffers {
        let q = c.entry.query();
        let v = |handle, bytes| View { handle, bytes };
        Buffers {
            activation: v(1, u64::from(rows) * u64::from(q.k) * 4),
            weight: v(2, c.entry.weight_bytes()),
            workspace: v(3, c.workspace_bytes(rows).unwrap()),
            live_rows: v(4, 4),
            weight_domain: v(5, c.entry.weight_domain_bytes()),
            output: v(6, u64::from(rows) * u64::from(q.n) * 4),
        }
    }

    #[test]
    fn plans_cover_every_row_with_whole_tiles_and_one_preparation() {
        let c = Candidate::new(entry(32320, 5120, WeightFormat::Bf16), TILE, 192).unwrap();
        assert_eq!(TILE.threads(), 256);
        assert_eq!(TILE.shared_bytes(), 2 * 320 * 128);
        for rows in [1, 63, 64, 65, 128, 191, 192] {
            let [prepare, compute]: [LaunchPlan; 2] = c.plans(rows).unwrap().try_into().unwrap();
            assert_eq!(prepare.grid, [rows, 1, 1]);
            assert_eq!(compute.grid, [rows.div_ceil(64), 253, 1]);
            assert_eq!(compute.block, [256, 1, 1]);
            assert!(compute.pdl && prepare.pdl);
            assert_eq!(
                c.workspace_bytes(rows).unwrap(),
                u64::from(rows).div_ceil(4) * 16 + u64::from(rows) * 5120 * 6
            );
            let b = buffers(&c, rows);
            let [first, second]: [Launch; 2] = c.launches(b, rows).unwrap().try_into().unwrap();
            assert_eq!(first.symbol, "chop_test_rows_n32320_k5120_rows_prepare_v1");
            assert_eq!(
                second.symbol,
                "chop_test_rows_n32320_k5120_rows_f128_r64_w32x32_s2_k64_v1"
            );
            assert_eq!(second.arguments.0[5], Argument::U32(rows));
            let mut short = b;
            short.workspace.bytes -= 1;
            assert!(c.launches(short, rows).is_err());
            let mut alias = b;
            alias.output.handle = alias.workspace.handle;
            assert!(c.launches(alias, rows).is_err());
        }
        for rows in [0, 193] {
            assert!(c.plans(rows).is_err());
        }
    }

    #[test]
    fn refuses_tiles_and_projections_outside_the_template() {
        let bf16 = entry(384, 5120, WeightFormat::Bf16);
        for tile in [
            Tile {
                warp_features: 24,
                ..TILE
            },
            Tile {
                warp_rows: 24,
                rows: 72,
                ..TILE
            },
            Tile { stages: 1, ..TILE },
            Tile { stages: 4, ..TILE },
            Tile {
                plane_k: 16,
                ..TILE
            },
            Tile {
                plane_k: 128,
                ..TILE
            },
            Tile {
                features: 2048,
                warp_features: 16,
                ..TILE
            },
        ] {
            assert!(Candidate::new(bf16.clone(), tile, 192).is_err(), "{tile:?}");
        }
        assert!(Candidate::new(entry(384, 5120, WeightFormat::F32), TILE, 192).is_err());
        assert!(Candidate::new(entry(384, 5056, WeightFormat::Bf16), TILE, 192).is_err());
        assert!(Candidate::new(bf16.clone().with_decode(), TILE, 192).is_err());
        assert!(Candidate::new(bf16.clone(), TILE, 193).is_err());
        assert!(Candidate::new(bf16, TILE, 192).is_ok());
    }

    #[test]
    fn sliced_tiles_plan_every_chain_plane_and_the_ordered_reducer() {
        let tile = Tile {
            features: 16,
            rows: 16,
            warp_features: 16,
            warp_rows: 8,
            stages: 4,
            plane_k: 64,
            slices: 8,
        };
        let c = Candidate::new(entry(384, 5120, WeightFormat::Bf16), tile, 192).unwrap();
        let rows = 129;
        assert_eq!(
            c.workspace_bytes(rows).unwrap(),
            u64::from(rows).div_ceil(4) * 16
                + u64::from(rows) * 5120 * 6
                + u64::from(rows) * 384 * 40 * 4
        );
        let plans = c.plans(rows).unwrap();
        assert_eq!(plans[1].grid, [9, 24, 8]);
        assert_eq!(plans[2].grid, [(129 * 384_u32).div_ceil(128), 1, 1]);
        let launches = c.launches(buffers(&c, rows), rows).unwrap();
        assert_eq!(
            launches
                .iter()
                .map(|l| l.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                "chop_test_rows_n384_k5120_rows_prepare_v1",
                "chop_test_rows_n384_k5120_rows_f16_r16_w16x8_s4_k64_z8_v1",
                "chop_test_rows_n384_k5120_rows_reduce_v1",
            ]
        );
        assert_eq!(launches[2].arguments.0[3], Argument::U32(rows));
        let specs = c.specs();
        assert!(
            specs[2]
                .body
                .starts_with("chop::exact_f32::reduce<384,5120>(")
        );
        for slices in [0, 3, 7, 16] {
            let tile = Tile { slices, ..tile };
            assert!(Candidate::new(entry(384, 5120, WeightFormat::Bf16), tile, 192).is_err());
        }
    }

    #[test]
    fn cubins_share_each_projections_preparation() {
        let bf16 = entry(384, 5120, WeightFormat::Bf16);
        let small = Tile {
            features: 32,
            rows: 32,
            warp_features: 16,
            warp_rows: 16,
            stages: 3,
            plane_k: 32,
            slices: 1,
        };
        let candidates = [
            Candidate::new(bf16.clone(), TILE, 192).unwrap(),
            Candidate::new(bf16, small, 192).unwrap(),
        ];
        let spec = cubin_spec("rows", &candidates).unwrap();
        assert_eq!(spec.entries.len(), 3);
        assert!(spec.entries[0].body.contains("prefill::prepare<5120>"));
        assert_eq!(
            spec.entries[2].body,
            "chop::exact_f32::rows::run<384,5120,32,32,16,16,3,32,1>(activation,weight,workspace,output,live_rows,rows,weight_domain);"
        );
        assert_eq!(spec.entries[2].max_threads, 128);
    }
}
