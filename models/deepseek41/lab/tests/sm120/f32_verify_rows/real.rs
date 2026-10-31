//! The real-slice matrix: each rank's head slice, four routers, two
//! compressors and layer 8's index projections, read from the safetensors
//! of the checkpoint `CHOPINFER_DEEPSEEK_CHECKPOINT` names, with the
//! 16-feature tiles each slice's weights put outside the tensor domain.
use super::catalog::{self, Shape};
use kernel_core::contract::WeightFormat;
use kernel_harness::kit::cases::f32_projection as cases;
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One rank's weights of one projection: `rank` selects its quarter of the
/// tensor's rows (the vocabulary head), otherwise every rank holds them all.
pub struct Slice {
    pub stem: &'static str,
    pub shape: &'static str,
    pub tensor: &'static str,
    pub rank: Option<u64>,
    pub outside: &'static [usize],
}

const fn slice(
    stem: &'static str,
    shape: &'static str,
    tensor: &'static str,
    rank: Option<u64>,
    outside: &'static [usize],
) -> Slice {
    Slice {
        stem,
        shape,
        tensor,
        rank,
        outside,
    }
}

/// The checkpoint's five weights below 2^-30 in these projections: head
/// row 49423 (rank 1's local row 17103), router rows 355, 59 and 237 of
/// layers 19, 22 and 35, and row 161 of layer 8's compressor `wkv`; the
/// other slices are their clean controls.
pub const SLICES: [Slice; 12] = [
    slice("head-rank0", "head", "head.weight", Some(0), &[]),
    slice("head-rank1", "head", "head.weight", Some(1), &[1068]),
    slice("head-rank2", "head", "head.weight", Some(2), &[]),
    slice("head-rank3", "head", "head.weight", Some(3), &[]),
    slice(
        "router-layer0",
        "router",
        "layers.0.ffn.gate.weight",
        None,
        &[],
    ),
    slice(
        "router-layer19",
        "router",
        "layers.19.ffn.gate.weight",
        None,
        &[22],
    ),
    slice(
        "router-layer22",
        "router",
        "layers.22.ffn.gate.weight",
        None,
        &[3],
    ),
    slice(
        "router-layer35",
        "router",
        "layers.35.ffn.gate.weight",
        None,
        &[14],
    ),
    slice(
        "compressor-layer2-wkv",
        "compressor",
        "layers.2.attn.compressor.wkv.weight",
        None,
        &[],
    ),
    slice(
        "compressor-layer8-wkv",
        "compressor",
        "layers.8.attn.compressor.wkv.weight",
        None,
        &[10],
    ),
    slice(
        "index_weights-layer8",
        "index_weights",
        "layers.8.attn.indexer.weights_proj.weight",
        None,
        &[],
    ),
    slice(
        "index_key-layer8",
        "index_key",
        "layers.8.attn.indexer.wk.weight",
        None,
        &[],
    ),
];

pub fn find(stem: &str) -> Option<&'static Slice> {
    SLICES.iter().find(|s| s.stem == stem)
}

pub fn checkpoint() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT")
            .expect("set CHOPINFER_DEEPSEEK_CHECKPOINT to the checkpoint directory"),
    )
}

impl Slice {
    pub fn shape(&self) -> Shape {
        catalog::shape(self.shape).expect("known shape")
    }

    /// The slice's weights as its rank binds them, row-major N x K, widened
    /// from their BF16 codes; asserts its tiles outside the tensor domain.
    pub fn load(&self, root: &Path) -> Vec<f32> {
        let shape = self.shape();
        let (n, k) = (u64::from(shape.n), u64::from(shape.k));
        let index: Value = serde_json::from_slice(
            &std::fs::read(root.join("model.safetensors.index.json")).expect("checkpoint index"),
        )
        .expect("checkpoint index JSON");
        let file = index["weight_map"][self.tensor]
            .as_str()
            .unwrap_or_else(|| panic!("{} is not in the checkpoint index", self.tensor));
        assert!(
            Path::new(file).file_name() == Some(file.as_ref()),
            "{file}: shard outside the checkpoint directory"
        );
        let mut shard = std::fs::File::open(root.join(file)).expect("checkpoint shard");
        let mut length = [0; 8];
        shard.read_exact(&mut length).unwrap();
        let length = u64::from_le_bytes(length);
        let mut header = vec![0; length as usize];
        shard.read_exact(&mut header).unwrap();
        let header: Value = serde_json::from_slice(&header).expect("safetensors header");
        let info = &header[self.tensor];
        assert_eq!(info["dtype"], "BF16", "{}", self.tensor);
        let rows = n * if self.rank.is_some() { 4 } else { 1 };
        assert_eq!(
            info["shape"],
            serde_json::json!([rows, k]),
            "{}",
            self.tensor
        );
        let start = info["data_offsets"][0].as_u64().expect("data offsets");
        let first = self.rank.unwrap_or(0) * n;
        shard
            .seek(SeekFrom::Start(8 + length + start + first * k * 2))
            .unwrap();
        let mut bytes = vec![0; (n * k * 2) as usize];
        shard.read_exact(&mut bytes).unwrap();
        let b: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        let outside: Vec<usize> = cases::weight_domains(&b, shape.k as usize, WeightFormat::Bf16)
            .iter()
            .enumerate()
            .filter_map(|(tile, &tensor)| (tensor == 0).then_some(tile))
            .collect();
        assert_eq!(
            outside, self.outside,
            "{}: tiles outside the tensor domain",
            self.stem
        );
        b
    }
}

#[test]
fn cpu_real_slices_cover_every_rank_and_affected_projection() {
    let heads: Vec<_> = SLICES.iter().filter_map(|s| s.rank).collect();
    assert_eq!(heads, [0, 1, 2, 3]);
    let affected: Vec<_> = SLICES
        .iter()
        .filter(|s| !s.outside.is_empty())
        .map(|s| s.stem)
        .collect();
    assert_eq!(
        affected,
        [
            "head-rank1",
            "router-layer19",
            "router-layer22",
            "router-layer35",
            "compressor-layer8-wkv"
        ]
    );
    for shape in catalog::shapes() {
        assert!(SLICES.iter().any(|s| s.shape() == shape), "{}", shape.name);
        let tiles = shape.n as usize / 16;
        for s in SLICES.iter().filter(|s| s.shape() == shape) {
            assert!(s.outside.iter().all(|&t| t < tiles), "{}", s.stem);
        }
    }
}

#[test]
#[ignore = "requires the checkpoint CHOPINFER_DEEPSEEK_CHECKPOINT names"]
fn cpu_real_slices_match_the_checkpoint() {
    let root = checkpoint();
    for slice in &SLICES {
        assert_eq!(
            slice.load(&root).len(),
            (slice.shape().n * slice.shape().k) as usize
        );
    }
}
