//! Prepare deterministic rank-specific packed banks outside GPU job time.
use kernel_core::codec::f32_to_bf16;
use kernel_core::precision::{scale_offset, scale_tiles};
use kernel_core::transform::{QuadLaneLayout, RowTileLayout};
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const EXPERTS: usize = 384;
const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 576;

fn write(root: &Path, name: &str, bytes: &[u8], files: &mut Vec<Value>) -> Result<()> {
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))?;
    out.write_all(bytes)?;
    files.push(json!({"name":name,"bytes":bytes.len()}));
    Ok(())
}

fn random(bytes: usize, mut seed: u32) -> Vec<u8> {
    (0..bytes)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect()
}

fn scales(matrices: usize, rows: usize, groups: usize, rank: u32) -> Vec<u8> {
    (0..matrices)
        .flat_map(|matrix| {
            (0..rows * groups).map(move |at| {
                let hash = (at as u32)
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add((matrix as u32).wrapping_mul(0x85eb_ca6b))
                    .wrapping_add(rank * 17);
                let base = 121 + ((matrix as u32 + rank) % 4) as u8;
                base + if hash % 127 == 0 { 7 } else { (hash & 1) as u8 }
            })
        })
        .collect()
}

fn encoded(
    root: &Path,
    bank: &str,
    raw: &[u8],
    rows: usize,
    groups: usize,
    files: &mut Vec<Value>,
) -> Result<usize> {
    assert!(raw.len().is_multiple_of(rows * groups));
    let mut round_trips = 0;
    for width in [1, 2, 4] {
        let blocks = raw
            .chunks_exact(rows * groups)
            .map(|matrix| {
                let encoded = scale_tiles::encode(matrix, rows as u64, groups as u64, width)?;
                let decoded = scale_tiles::decode(&encoded, rows as u64, groups as u64, width)?;
                if decoded != matrix {
                    return Err("encoded scale fixture failed its round trip".into());
                }
                Ok(encoded)
            })
            .collect::<Result<Vec<_>>>()?;
        round_trips += blocks.len();
        let layout =
            scale_offset::bank_layout(&blocks.iter().map(|b| b.len() as u64).collect::<Vec<_>>())?;
        let mut bytes = layout.table;
        for (block, offset) in blocks.into_iter().zip(layout.offsets) {
            if bytes.len() as u64 != offset {
                return Err("scale matrix offset differs".into());
            }
            bytes.extend(block);
        }
        if bytes.len() as u64 != layout.bytes {
            return Err("scale bank extent differs".into());
        }
        write(root, &format!("{bank}-scale-w{width}.bin"), &bytes, files)?;
    }
    Ok(round_trips)
}

fn rank(root: &Path, rank: u32) -> Result<()> {
    let root = root.join(format!("rank{rank}"));
    std::fs::create_dir(&root)?;
    let mut files = Vec::new();
    let mut round_trips = 0;
    {
        let rows = EXPERTS * 2 * INTERMEDIATE;
        let mut codes = random(rows * HIDDEN / 2, 0x6e31_931d ^ (rank + 1));
        let mut raw = scales(EXPERTS * 2, INTERMEDIATE, HIDDEN / 32, rank);
        round_trips += encoded(&root, "gate", &raw, INTERMEDIATE, HIDDEN / 32, &mut files)?;
        let mut scratch = Vec::new();
        RowTileLayout {
            rows,
            row_bytes: HIDDEN / 2,
            tile_rows: 64,
            tile_bytes: 64,
        }
        .pack_in_place(&mut codes, &mut scratch)?;
        QuadLaneLayout { bytes: codes.len() }.pack_in_place(&mut codes)?;
        scratch.clear();
        RowTileLayout {
            rows,
            row_bytes: HIDDEN / 32,
            tile_rows: 64,
            tile_bytes: 4,
        }
        .pack_in_place(&mut raw, &mut scratch)?;
        write(&root, "gate-codes.bin", &codes, &mut files)?;
        write(&root, "gate-scale-raw.bin", &raw, &mut files)?;
    }
    {
        let codes = random(
            EXPERTS * HIDDEN * INTERMEDIATE / 2,
            0x419a_2fd3 ^ (rank + 1),
        );
        let raw = scales(EXPERTS, HIDDEN, INTERMEDIATE / 32, rank);
        round_trips += encoded(&root, "down", &raw, HIDDEN, INTERMEDIATE / 32, &mut files)?;
        write(&root, "down-codes.bin", &codes, &mut files)?;
        write(&root, "down-scale-raw.bin", &raw, &mut files)?;
    }
    for (name, n, k, seed) in [
        ("w1", INTERMEDIATE, HIDDEN, 0x14e1_2511),
        ("w3", INTERMEDIATE, HIDDEN, 0x16f2_5913),
        ("w2", HIDDEN, INTERMEDIATE, 0x14f2_1557),
    ] {
        let mut codes = random(n * k, seed ^ (rank + 1));
        for code in &mut codes {
            if *code & 127 == 127 {
                *code ^= 1;
            }
        }
        let packed = Mxfp8FoldLayout { groups: 1, n, k }.pack(&codes)?;
        let scales = (0..n * k / 32)
            .map(|at| 120 + ((at + rank as usize) % 3) as u8)
            .collect::<Vec<_>>();
        write(
            &root,
            &format!("shared-{name}-codes.bin"),
            &packed,
            &mut files,
        )?;
        write(
            &root,
            &format!("shared-{name}-scale.bin"),
            &scales,
            &mut files,
        )?;
    }
    let router = (0..EXPERTS)
        .flat_map(|e| {
            (0..HIDDEN).flat_map(move |k| {
                let value = if k < EXPERTS {
                    if k == e { 1. } else { 0. }
                } else {
                    ((k + rank as usize * 3) as i32 % 11 - 5) as f32 / 128.
                };
                f32_to_bf16(value).to_le_bytes()
            })
        })
        .collect::<Vec<_>>();
    write(&root, "router-bf16.bin", &router, &mut files)?;
    write(
        &root,
        "router-domain.bin",
        &vec![1_u32; EXPERTS / 16]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
        &mut files,
    )?;
    write(&root, "router-bias.bin", &vec![0; EXPERTS * 4], &mut files)?;
    let alternate = (0..EXPERTS)
        .flat_map(|e| ((e as i32 % 7 - 3) as f32 / 1024.).to_le_bytes())
        .collect::<Vec<_>>();
    write(&root, "router-bias-vl.bin", &alternate, &mut files)?;
    assert_eq!(files.len(), 20);
    assert_eq!(round_trips, 3456);
    std::fs::write(
        root.join("weights.json"),
        serde_json::to_vec(
            &json!({"schema":"chopinfer.moe-wide-weights.v1","rank":rank,"gate_layout":"row-tiles-64x128-lanes","files":files}),
        )?,
    )?;
    println!("MOE_WIDE_WEIGHTS rank={rank} files=20 encoded_round_trips={round_trips} PASS");
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("usage: prepare_moe_wide OUT")?);
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    std::fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    if root.starts_with(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()?,
    ) {
        return Err("fixtures must stay outside the repository".into());
    }
    for logical_rank in 0..4 {
        rank(&root, logical_rank)?;
    }
    println!("MOE_WIDE_WEIGHTS_COMPLETE ranks=4 files=80 PASS");
    Ok(())
}
