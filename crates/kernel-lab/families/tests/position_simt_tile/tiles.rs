use super::*;
use kernel_core::codec::{SmallFloat, bf16_to_f32};

fn tree(mut lanes: [f32; 32], wrong: bool) -> f32 {
    for level in 0..5 {
        let offset = if wrong { 1 << level } else { 16 >> level };
        for v in 0..32 {
            if (!wrong && v < offset) || (wrong && v % (2 * offset) == 0) {
                lanes[v] += lanes[v + offset];
            }
        }
    }
    0. + lanes[0]
}
fn register_score(x: &arithmetic::Operands, key: usize, bf16: bool, wrong: bool) -> f32 {
    let k: [f32; 128] = std::array::from_fn(|c| {
        if bf16 {
            let at = (key * 128 + c) * 2;
            bf16_to_f32(u16::from_le_bytes(x.codes[at..at + 2].try_into().unwrap()))
        } else {
            SmallFloat::E2M1.decode_f32((x.codes[key * 64 + c / 2] >> (4 * (c % 2))) & 15)
                * kernel_core::codec::decode_ue8m0_f32(x.scales[key * 4 + c / 32])
        }
    });
    let mut result = 0_f32;
    for h in 0..32 {
        let lanes = std::array::from_fn(|v| {
            let mut chain = 0_f32;
            for part in 0..4 {
                let c = part * 32 + v;
                chain = bf16_to_f32(x.queries[h * 128 + c]).mul_add(k[c], chain);
            }
            chain
        });
        let dot = tree(lanes, wrong);
        result = dot.max(0.).mul_add(x.weights[h] * (1. / 64.), result);
    }
    result
}
#[test]
fn cpu_register_tree_matches_lane_tree_and_rejects_adjacent_pairs() {
    for bf16 in [false, true] {
        let x = arithmetic::Operands::new(1, 128, bf16, arithmetic::SEED);
        let (native, _) = arithmetic::control_scores(&x, bf16);
        let mut rejected = 0;
        for (key, &bits) in native.iter().enumerate() {
            assert_eq!(register_score(&x, key, bf16, false).to_bits(), bits);
            rejected += usize::from(register_score(&x, key, bf16, true).to_bits() != bits);
        }
        assert!(rejected > 0);
        println!("register-tree control bf16={bf16}: {rejected}/128 adjacent-pair mismatches");
    }
}
#[test]
fn cpu_nonfinite_prefixes_remain_visible_before_relu() {
    let mut seed = 0x71592c75_u64;
    for case in 0..2048 {
        let mut old_invalid = false;
        let lanes = std::array::from_fn(|_| {
            let mut chain = 0_f32;
            for _ in 0..4 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let q = bf16_to_f32(seed as u16);
                let k = bf16_to_f32((seed >> 32) as u16);
                old_invalid |= !q.is_finite() || !k.is_finite();
                chain = q.mul_add(k, chain);
                old_invalid |= !chain.is_finite();
            }
            chain
        });
        let mut reduced = lanes;
        for offset in [16, 8, 4, 2, 1] {
            for lane in 0..offset {
                reduced[lane] += reduced[lane + offset];
                old_invalid |= !reduced[lane].is_finite();
            }
        }
        assert_eq!(old_invalid, !tree(lanes, false).is_finite(), "case {case}");
    }
}

#[test]
#[ignore = "requires the staged pair GPU job"]
fn gpu_mixed_sequence_tiles_and_tree_control() {
    let ran=kernel_harness::gpu_test(64<<20,|s| {
        let root=PathBuf::from(std::env::var_os("CHOPINFER_POSITION_LIVE_CUBINS").ok_or("prebuilt cubins required")?);
        let m=s.load_cubin(&std::fs::read(root.join(format!("{}.cubin",family::CUBIN))).map_err(|e|e.to_string())?)?;
        let bf16=std::env::var("CHOPINFER_POSITION_LIVE_STORAGE").map_err(|e|e.to_string())?=="bf16";
        let mut a=Arena{session:s,buffers:vec![]};
        let x=arithmetic::Operands::new(7,256,bf16,arithmetic::SEED);
        let b=PositionSelectBuffers {
            q:a.input(&x.query_bytes(false))?,head_weights:a.input(&x.weight_bytes(false))?,
            keys:PositionKeyPages{codes:a.input(&x.codes)?,scales:if bf16 {View::EMPTY} else {a.input(&x.scales)?},pages:4,
                code_page_stride:if bf16 {16384} else {4096},scale_page_stride:if bf16 {0} else {256},
                page_tables:BlockTables{tables:a.input(&i32s((0..4).flat_map(|b|(0..4).map(move |p|if (p+b)%5==0 {-1} else {(p+2*b)%4}))))?,sequences:4,max_pages:4}},
            positions:a.input(&i32s([-1,0,128,255,64,127,254]))?,row_batch:a.input(&i32s([0,1,0,2,1,3,3]))?,
            live_rows:a.input(&7_u32.to_le_bytes())?,candidate_mask:None,position_capacity:256,
            scores:a.output(7*256*4)?,status:a.output(7*4)?,output:a.output(7*512*4)?,
        };
        let storage=if bf16 {IndexerKeyFormat::Bf16} else {IndexerKeyFormat::Fp4};
        let mut seq=vec![];let mut scratch=vec![];
        for new in [false,true] {
            let op=backend(new).position_indexer_with_storage(&family::dense(1).query,storage,512,256,Need::rows(7)).map_err(|e|e.to_string())?;
            let w=a.output(op.info().workspace as usize)?;
            seq.push(op.launches(&b,w,Counts::rows(7)).map_err(|e|e.to_string())?);scratch.push(w);
        }
        let graphs=seq.iter().map(|calls|s.capture(|_|fire(s,&m,&a,calls))).collect::<LabResult<Vec<_>>>()?;
        for fault in [false,true] {
            let mut q=x.query_bytes(false);
            if fault {q[(2*4096+4095)*2..(2*4096+4096)*2].copy_from_slice(&0x7fc1_u16.to_le_bytes());}
            a.write(b.q,&q)?;
            for (replay,live) in [7_u32,3,0,7].into_iter().enumerate() {
                a.write(b.live_rows,&live.to_le_bytes())?;
                let mut reference=vec![];let mut raw=vec![];
                for arm in 0..2 {
                    a.poison(&[b.scores,b.status,b.output,scratch[arm]])?;
                    if replay==0 {
                        fire(s,&m,&a,&seq[arm][..1])?;
                        let score=a.read(b.scores)?;
                        if arm==0 {raw=score;} else {compare_bytes(&raw,&score,"raw mixed tile score")?;}
                        fire(s,&m,&a,&seq[arm][1..])?;
                    } else {graphs[arm].launch(&s.stream).map_err(|e|e.to_string())?;}
                    let actual=[b.scores,b.status,b.output].into_iter().map(|v|a.read(v)).collect::<LabResult<Vec<_>>>()?;
                    if arm==0 {reference=actual;} else {assert_eq!(reference,actual,"fault={fault} replay={replay}");}
                }
                if !fault && replay==0 {
                    let mut mutant=seq[1][0].clone();mutant.symbol=mutant.symbol.replace("_simt_tile_v1","_simt_tile_wrong_tree_v1");
                    a.poison(&[b.scores])?;fire(s,&m,&a,&[mutant])?;
                    assert_ne!(a.read(b.scores)?,raw,"adjacent-pair tree control escaped");
                }
            }
        }
        println!("TILE_MIXED_COMPLETE: four sequences, unequal visibility, missing pages, partial tile, poisoned inactive rows, isolated NaN, graph replay; wrong tree rejected");
        Ok(())
    }).unwrap();
    assert!(ran.is_some(), "mixed-sequence tile comparison skipped");
}
