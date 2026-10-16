//! Dense projection roles and aliases within one geometry.

use super::geometry::{GeometryShape, shape};
use kernel_core::contract::DType;
use kernels_sm120::Profile;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    LinearQkv,
    LinearZ,
    LinearOutput,
    AttentionQkv,
    AttentionOutput,
    IndexQk,
    Router,
    HcDown,
    PleKey,
    PleValue,
    MtpFc,
    Head,
}

pub type DenseShape = (&'static str, u32, u32, DType, u8);

impl Role {
    pub fn dimensions(self, shape: GeometryShape) -> (u32, u32, DType) {
        let g = shape.geometry;
        let h = g.hidden();
        let wide = h * g.streams();
        let (n, k) = match self {
            Self::LinearQkv => (shape.linear_qkv(), h),
            Self::LinearZ => (shape.linear_value(), h),
            Self::LinearOutput => (h, shape.linear_value()),
            Self::AttentionQkv => (shape.attn_row(), h),
            Self::AttentionOutput => (h, shape.attn_value()),
            Self::IndexQk => ((g.index_heads() + g.index_kv_heads()) * g.index_dim(), h),
            Self::Router => (g.experts(), h),
            Self::HcDown => (g.hc_rank(), wide),
            Self::PleKey => (wide, g.ple_embed_dim()),
            Self::PleValue => (h, g.ple_embed_dim()),
            Self::MtpFc => (h, h),
            Self::Head => (shape.vocab_shard(), h),
        };
        (
            n,
            k,
            if self == Self::Router {
                DType::F32
            } else {
                DType::Bf16
            },
        )
    }
}

// Flash's public names and K segments are retained when roles share a shape.
const ALIASES: &[(Role, &str)] = &[
    (Role::LinearQkv, "lin_qkv"),
    (Role::LinearZ, "lin_z"),
    (Role::LinearOutput, "out"),
    (Role::AttentionQkv, "attn_qkv"),
    (Role::IndexQk, "index_qk"),
    (Role::Router, "router"),
    (Role::HcDown, "hc_down"),
    (Role::PleKey, "ple_key"),
    (Role::Head, "head"),
    (Role::AttentionOutput, "attn_out"),
    (Role::PleValue, "ple_value"),
    (Role::MtpFc, "mtp_fc"),
];

pub fn shapes(profile: Profile) -> Vec<DenseShape> {
    shapes_for(shape(profile).expect("Qwen profile"))
}

pub fn shapes_for(s: GeometryShape) -> Vec<DenseShape> {
    let mut shapes: Vec<DenseShape> = Vec::new();
    for &(role, name) in ALIASES {
        if s.ranks != 4
            && matches!(
                role,
                Role::IndexQk
                    | Role::Router
                    | Role::HcDown
                    | Role::PleKey
                    | Role::PleValue
                    | Role::MtpFc
            )
        {
            continue;
        }
        let (n, k, output) = role.dimensions(s);
        if shapes
            .iter()
            .any(|&(_, pn, pk, po, _)| (pn, pk, po) == (n, k, output))
        {
            continue;
        }
        shapes.push((
            name,
            n,
            k,
            output,
            super::tuning::prefill(s.geometry, name, k),
        ));
    }
    shapes
}

pub fn alias(profile: Profile, role: Role) -> (Profile, &'static str) {
    let s = shape(profile).expect("Qwen profile");
    let (ranks, name) = alias_for(s, role);
    (
        super::geometry::owner(profile).unwrap().model.at(ranks),
        name,
    )
}

pub fn alias_for(s: GeometryShape, role: Role) -> (u32, &'static str) {
    let dimensions = role.dimensions(s);
    let ranks = if matches!(
        role,
        Role::PleValue | Role::MtpFc | Role::IndexQk | Role::Router | Role::HcDown | Role::PleKey
    ) {
        4
    } else {
        s.ranks
    };
    let (name, ..) = shapes_for(GeometryShape { ranks, ..s })
        .into_iter()
        .find(|&(_, n, k, output, _)| (n, k, output) == dimensions)
        .expect("dense role has an entry");
    (ranks, name)
}
