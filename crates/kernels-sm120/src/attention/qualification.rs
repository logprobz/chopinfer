//! Finite attention instance coverage required by serving admission.
//!
//! kernel-lab checks every tuple against its compiled catalog, including both
//! KV formats and all phases. Additions require GPU oracle and invariance
//! evidence before release. Theta and epsilon are runtime operands; M-RoPE
//! sections, block size and kept counts are template arguments.

pub struct HeadShape {
    pub head_dim: u32,
    pub rows: &'static [u32],
    pub dense_groups: &'static [u32],
    pub sparse_groups: &'static [u32],
}

pub const HEAD_SHAPES: &[HeadShape] = &[
    HeadShape {
        head_dim: 128,
        rows: &[2, 4, 6, 8, 12, 16],
        dense_groups: &[2, 4, 6, 8],
        sparse_groups: &[2, 4, 6, 8, 12, 16],
    },
    HeadShape {
        head_dim: 256,
        rows: &[6, 8, 12, 16],
        dense_groups: &[6],
        sparse_groups: &[6, 8, 12, 16],
    },
];

pub const ROTARY_SECTIONS: &[(u32, [u32; 3])] = &[(32, [6, 5, 5]), (64, [11, 11, 10])];
pub const INDEX_HEADS: &[u32] = &[4, 8, 16];
pub const KEPT_BLOCKS: &[u32] = &[256, 512];
pub const BLOCK: u32 = 4;
pub const EXACT_VERIFY_ROWS: u32 = 4;
