//! Opt-in positional safetensors tests. The checkpoint is never modified.

use super::*;
use crate::codec::SmallFloat;
use serde_json::Value;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

fn read_at(file: &File, offset: u64, length: usize) -> Vec<u8> {
    let mut bytes = vec![0; length];
    file.read_exact_at(&mut bytes, offset).unwrap();
    bytes
}

struct Checkpoint {
    root: std::path::PathBuf,
    index: Value,
}

impl Checkpoint {
    fn open(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            index: serde_json::from_slice(
                &std::fs::read(root.join("model.safetensors.index.json")).unwrap(),
            )
            .unwrap(),
        }
    }

    fn tensor(&self, suffix: &str, dtype: &str, shape: &[u64]) -> Vec<u8> {
        let owners: Vec<_> = self.index["weight_map"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.ends_with(suffix))
            .collect();
        assert_eq!(owners.len(), 1, "tensor suffix {suffix}");
        let filename = owners[0].1.as_str().unwrap();
        assert_eq!(Path::new(filename).components().count(), 1);
        assert!(matches!(
            Path::new(filename).components().next(),
            Some(std::path::Component::Normal(_))
        ));
        let file = File::open(self.root.join(filename)).unwrap();
        let length = u64::from_le_bytes(read_at(&file, 0, 8).try_into().unwrap());
        assert!(length < 16 * 1024 * 1024, "safetensors header too large");
        let header: Value = serde_json::from_slice(&read_at(&file, 8, length as usize)).unwrap();
        let data_start = 8 + length;
        let matches = header
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.ends_with(suffix))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "tensor suffix {suffix}");
        let (name, meta) = matches[0];
        eprintln!("checkpoint tensor: {name}");
        assert_eq!(meta["dtype"], dtype);
        assert_eq!(
            meta["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>(),
            shape
        );
        let offsets = meta["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        read_at(&file, data_start + start, (end - start) as usize)
    }
}

fn checkpoint_root() -> Option<String> {
    let root = std::env::var("CHOPINFER_MIMO_CHECKPOINT").ok()?;
    if root.is_empty() {
        return None;
    }
    Some(root)
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
}

#[test]
fn qkv_chunked_fp8_round_trips() {
    let Some(root) = checkpoint_root() else {
        eprintln!("SKIP: set CHOPINFER_MIMO_CHECKPOINT for QKV transform checks");
        return;
    };
    let cp = Checkpoint::open(Path::new(&root));
    for (layer, rows, scale_rows, rows_per_chunk) in [(5, 13568, 108, 3392), (1, 14848, 116, 3712)]
    {
        let prefix = format!("model.layers.{layer}.self_attn.qkv_proj");
        let weight = cp.tensor(&format!("{prefix}.weight"), "F8_E4M3", &[rows, 4096]);
        let scales = cp.tensor(
            &format!("{prefix}.weight_scale_inv"),
            "F32",
            &[scale_rows, 32],
        );
        // Each rank's weight shard is its contiguous chunk of rows.
        let weight_shards = weight
            .chunks_exact(rows as usize / 4 * 4096)
            .collect::<Vec<_>>();
        let layout = ChunkedScaleGrid {
            chunks: 4,
            rows_per_chunk,
            k_blocks: 32 * 4,
            block_rows: 128,
        };
        let scale_shards = layout.split_chunks(&scales).unwrap();
        assert_eq!(scale_shards.concat(), scales);
        let flat = layout.flatten(&scales).unwrap();
        for row in [
            0,
            rows_per_chunk - 1,
            rows_per_chunk,
            rows_per_chunk * 2,
            rows as usize - 1,
        ] {
            for col in [0, 127, 128, 4095] {
                let scale_index = layout.scale_row(row).unwrap() * 32 + col / 128;
                let direct_index = (row / rows_per_chunk * layout.scale_rows_per_chunk()
                    + (row % rows_per_chunk) / 128)
                    * 32
                    + col / 128;
                let direct = SmallFloat::E4M3.decode_f64(weight[row * 4096 + col])
                    * f64::from(f32_at(&scales, direct_index));
                let rank = row / rows_per_chunk;
                let local_row = row % rows_per_chunk;
                let sharded = SmallFloat::E4M3
                    .decode_f64(weight_shards[rank][local_row * 4096 + col])
                    * f64::from(f32_at(
                        &scale_shards[rank],
                        local_row / 128 * 32 + col / 128,
                    ));
                assert_eq!(
                    flat.weight_row_to_scale_row[row],
                    layout.scale_row(row).unwrap()
                );
                assert_eq!(scale_index, direct_index);
                assert_eq!(direct.to_bits(), sharded.to_bits());
            }
        }
    }
}
