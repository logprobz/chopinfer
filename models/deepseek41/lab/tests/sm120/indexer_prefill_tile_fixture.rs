use super::*;

pub(super) struct Host {
    pub q: Vec<u8>,
    pub weights: Vec<u8>,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub tables: Vec<i32>,
    pub positions: Vec<i32>,
    pub ids: Vec<i32>,
    pub tags: Vec<u64>,
    pub retained_tags: Vec<u64>,
    pub visible: Vec<i32>,
    pub source_status: Vec<i32>,
    pub selected: Option<Vec<u8>>,
    pub expected_status: Option<Vec<u8>>,
}
impl Host {
    pub fn new(c: Case) -> LabResult<Self> {
        let ratio = FLASH.compress_ratio(c.layer);
        let width = c.context / ratio;
        if let Some(i) = c.capture {
            let manifest = read_json(&artifact().join("captures.json"))?;
            let dir = PathBuf::from(
                manifest["captures"][i]["path"]
                    .as_str()
                    .ok_or("capture path")?,
            );
            let meta = read_json(&dir.join("operands.json"))?;
            assert_eq!(meta["layer"], c.layer);
            assert_eq!(meta["context"], c.context);
            assert_eq!(meta["phase_rows"], 8192);
            let positions = values(&meta, "positions");
            assert_eq!(positions.len(), c.rows as usize);
            assert_eq!(positions[0], (c.context - 8192) as i32);
            let compact = c.layer >= 24;
            let tags = bytes(&dir.join("row_tags.u64"))?;
            let retained = if compact {
                bytes(&dir.join("candidate_tags.u64"))?
            } else {
                tags.clone()
            };
            return Ok(Self {
                q: bytes(&dir.join("query.bf16"))?,
                weights: bytes(&dir.join("weights.f32"))?,
                codes: bytes(&dir.join("keys.e2m1"))?,
                scales: bytes(&dir.join("scales.ue8m0"))?,
                tables: values(&meta, "page_table"),
                ids: if compact {
                    bytes(&dir.join("candidates.i32"))?
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                } else {
                    vec![-1; c.rows as usize * 2048]
                },
                tags: tags
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                retained_tags: retained
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                visible: if compact {
                    bytes(&dir.join("candidate_visible.i32"))?
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                } else {
                    positions.iter().map(|p| (p + 1) / ratio as i32).collect()
                },
                source_status: if compact {
                    bytes(&dir.join("candidate_status.u32"))?
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                } else {
                    vec![0; c.rows as usize]
                },
                positions,
                selected: Some(bytes(&dir.join("selected.i32"))?),
                expected_status: Some(bytes(&dir.join("status.u32"))?),
            });
        }
        let mut seed = 0x193ab217_u32 ^ c.context ^ c.layer;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut q = Vec::with_capacity(c.rows as usize * 8192);
        for row in 0..c.rows {
            for _ in 0..32 {
                for group in 0..4 {
                    for lane in 0..32 {
                        let r = random();
                        let exponent = if c.kind == 2 {
                            [0, 24, 0, -24][group]
                        } else {
                            group as i32 - 2
                        };
                        let value = if c.kind == 1 {
                            0.
                        } else {
                            let magnitude = if lane == 31 {
                                6.
                            } else {
                                [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(r & 7) as usize]
                            };
                            magnitude * if r & 8 == 0 { 1. } else { -1. } * 2_f32.powi(exponent)
                        };
                        let value = if c.kind == 3 && row == 0 && group == 0 && lane == 0 {
                            f32::NAN
                        } else if c.kind == 3 && row == 1 && group == 0 && lane == 0 {
                            // Finite BF16 outside the trained FP4 grid must
                            // retain the SIMT score, not become a refused row.
                            f32::from_bits(0x3f810000)
                        } else {
                            value
                        };
                        q.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                    }
                }
            }
        }
        let mut weights: Vec<f32> = (0..c.rows * 32)
            .map(|_| (random() % 65) as f32 / 32. - 1.)
            .collect();
        if c.kind == 4 {
            weights[0] = f32::NAN;
            weights[32] = f32::INFINITY;
            weights[64] = f32::MAX;
        }
        let codes = (0..width * 64).map(|_| random() as u8).collect();
        let mut scales: Vec<_> = (0..width * 4).map(|i| 125 + (i % 4) as u8).collect();
        if c.kind == 5 {
            scales[0] = 255;
            scales[4] = 252;
        }
        let mut positions: Vec<_> = (c.context - c.rows..c.context).map(|p| p as i32).collect();
        if c.kind == 6 {
            positions.fill(-1);
            positions[1] = 0;
            positions[2] = 1;
        }
        let mut tables: Vec<_> = (0..width / (64 / ratio)).map(|i| i as i32).collect();
        if c.kind == 7 {
            for i in (3..tables.len()).step_by(17) {
                tables[i] = -1;
            }
        }
        let visible: Vec<_> = positions
            .iter()
            .map(|p| if *p < 0 { 0 } else { (p + 1) / ratio as i32 })
            .collect();
        let mut ids = vec![-1; c.rows as usize * 2048];
        for (row, &n) in visible.iter().enumerate() {
            let blocks = (n as u32).div_ceil(8);
            let kept = blocks.min(2048);
            for i in 0..kept {
                ids[row * 2048 + i as usize] = if kept <= 1 {
                    blocks as i32 - 1
                } else {
                    (u64::from(i) * u64::from(blocks - 1) / u64::from(kept - 1)) as i32
                };
            }
        }
        let tags: Vec<_> = (0..c.rows)
            .map(|i| 0x345600000000_u64 + u64::from(i))
            .collect();
        let mut retained_tags = tags.clone();
        let mut source_status = vec![0; c.rows as usize];
        if c.kind == 8 {
            retained_tags[0] ^= 1;
        }
        if c.kind == 9 {
            ids[2047] = i32::MAX;
            ids[2048] = -2;
        }
        if c.kind == 10 {
            source_status[0] = 1;
        }
        Ok(Self {
            q,
            weights: weights.into_iter().flat_map(f32::to_le_bytes).collect(),
            codes,
            scales,
            tables,
            positions,
            ids,
            tags,
            retained_tags,
            visible,
            source_status,
            selected: None,
            expected_status: None,
        })
    }
}
pub(super) struct Input {
    pub dense: PositionSelectBuffers<View>,
    pub tags: View,
    pub visible: View,
    pub retained: CandidateBlockIds<View>,
}
impl Input {
    pub fn upload(m: &mut Memory<'_>, c: Case, h: &Host) -> LabResult<Self> {
        let ratio = FLASH.compress_ratio(c.layer);
        let page = 64 / ratio;
        let width = indexer::position_capacity(&FLASH, c.layer).unwrap();
        let mut tables = h.tables.clone();
        tables.resize((width / page) as usize, -1);
        let pages = h.codes.len() / (page as usize * 64);
        // Native serving embeds these planes in larger independent page records.
        const STRIDE: usize = 56960;
        let mut codes = vec![0xa5; pages * STRIDE];
        let mut scales = vec![0xa5; pages * STRIDE];
        for p in 0..pages {
            codes[p * STRIDE..p * STRIDE + page as usize * 64]
                .copy_from_slice(&h.codes[p * page as usize * 64..(p + 1) * page as usize * 64]);
            scales[p * STRIDE..p * STRIDE + page as usize * 4]
                .copy_from_slice(&h.scales[p * page as usize * 4..(p + 1) * page as usize * 4]);
        }
        let tags = m.input(
            &h.tags
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let visible = m.input(&ints(h.visible.iter().copied()))?;
        Ok(Self {
            dense: PositionSelectBuffers {
                q: m.input(&h.q)?,
                head_weights: m.input(&h.weights)?,
                keys: PositionKeyPages {
                    codes: m.input(&codes)?,
                    scales: m.input(&scales)?,
                    pages: pages as u32,
                    code_page_stride: STRIDE as u64,
                    scale_page_stride: STRIDE as u64,
                    page_tables: BlockTables {
                        tables: m.input(&ints(tables))?,
                        sequences: 1,
                        max_pages: width / page,
                    },
                },
                positions: m.input(&ints(h.positions.iter().copied()))?,
                row_batch: m.input(&ints(vec![0; c.rows as usize]))?,
                live_rows: View::EMPTY,
                candidate_mask: None,
                position_capacity: width,
                scores: View::EMPTY,
                status: View::EMPTY,
                output: View::EMPTY,
            },
            tags,
            visible,
            retained: CandidateBlockIds {
                ids: m.input(&ints(h.ids.iter().copied()))?,
                row_tags: m.input(
                    &h.retained_tags
                        .iter()
                        .flat_map(|t| t.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                visible_positions: visible,
                status: m.input(&ints(h.source_status.iter().copied()))?,
                position_capacity: width,
            },
        })
    }
}
