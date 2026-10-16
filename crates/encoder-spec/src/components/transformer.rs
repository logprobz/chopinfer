//! `transformer`: a stack of blocks built from slot components. Its weight
//! roles take `{i}`, the block index.

use serde::{Deserialize, Serialize};

use super::attention::{AttentionOverride, Mask, Sink, check_mask};
use super::{Attention, Boundary, Component, Context, Data, Mlp, Norm, Reorder, RotarySlot, Typed};
use crate::Sequences;
use crate::blocks::BlockSet;
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

/// An attention change for `blocks`. Overrides never overlap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Override {
    pub blocks: BlockSet,
    pub attention: AttentionOverride,
}

/// `reorder` applied before `blocks`' first norm and undone after their
/// final residual add.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReorderBlocks {
    pub blocks: BlockSet,
    pub reorder: Reorder,
}

/// Block `save_after`'s output, added to block `add_after`'s output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Carry {
    pub save_after: u32,
    pub add_after: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreNormParams {
    pub blocks: u32,
    pub width: u32,
    pub norm: Norm,
    pub attention: Attention,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotary: Option<RotarySlot>,
    pub mlp: Mlp,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<Override>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reorder: Vec<ReorderBlocks>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub carry: Vec<Carry>,
    /// The stage's own attention sequences: `groups` (or `item`) over rows
    /// a group op has joined into whole items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequences: Option<Sequences>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Transformer {
    /// Per block: `x += proj(attention(rotary(qkv(norm1(x)))))`, then
    /// `x += mlp(norm2(x))`.
    #[serde(rename = "transformer.pre-norm.v1")]
    PreNorm(Box<PreNormParams>),
}

impl Transformer {
    pub fn params(&self) -> &PreNormParams {
        match self {
            Transformer::PreNorm(params) => params,
        }
    }

    pub fn params_mut(&mut self) -> &mut PreNormParams {
        match self {
            Transformer::PreNorm(params) => params,
        }
    }

    /// The mask and sink of `block`'s attention.
    pub fn attention_at(&self, block: u32) -> (Mask, Sink) {
        let params = self.params();
        let base = params.attention.params();
        let mut attention = (base.mask, base.sink);
        for change in params.overrides.iter().filter(|o| o.blocks.contains(block)) {
            attention.0 = change.attention.mask.unwrap_or(attention.0);
            attention.1 = change.attention.sink.unwrap_or(attention.1);
        }
        attention
    }

    /// The blocks whose attention has a key-0 sink.
    pub fn sink_blocks(&self) -> Vec<u32> {
        (0..self.params().blocks)
            .filter(|&block| self.attention_at(block).1 == Sink::Key0)
            .collect()
    }
}

/// Refuses block sets that are not canonical among `blocks`, or overlap.
fn disjoint<'a>(
    what: &str,
    sets: impl Iterator<Item = &'a BlockSet>,
    blocks: u32,
) -> Result<(), String> {
    let mut seen = vec![false; blocks as usize];
    for set in sets {
        for block in set.canonical_members(blocks)? {
            if std::mem::replace(&mut seen[block as usize], true) {
                return Err(format!("{what} overlap at block {block}"));
            }
        }
    }
    Ok(())
}

impl Typed for Transformer {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Transformer {
    fn parallel(&self) -> ParallelClass {
        if self.params().reorder.is_empty() {
            ParallelClass::SequenceCoupled
        } else {
            ParallelClass::Permuting
        }
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        let params = self.params();
        let width = input.width()?;
        if params.blocks == 0 {
            return Err("`blocks` is zero".into());
        }
        if params.width != width {
            return Err(format!(
                "width {} does not match its input, {}",
                params.width, input.data
            ));
        }
        params.norm.check(context)?;
        params.attention.check(context)?;
        params.mlp.check(context)?;
        if let Some(slot) = &params.rotary {
            let rotary = slot.component()?;
            context.admit(&rotary.id(), rotary.policies())?;
            rotary.check()?;
            let dims = rotary.rotary_dims()?;
            let head_dim = params.attention.params().head_dim;
            if dims == 0 || dims > u64::from(head_dim) {
                return Err(format!(
                    "`{}` rotates {dims} dimensions of a {head_dim}-dimension head",
                    rotary.id()
                ));
            }
        }
        disjoint(
            "overrides",
            params.overrides.iter().map(|o| &o.blocks),
            params.blocks,
        )?;
        let base = params.attention.params();
        for change in &params.overrides {
            let (mask, sink) = (change.attention.mask, change.attention.sink);
            if mask.is_none_or(|mask| mask == base.mask)
                && sink.is_none_or(|sink| sink == base.sink)
            {
                return Err(format!(
                    "override for blocks `{}` changes nothing",
                    change.blocks.as_str()
                ));
            }
            if let Some(mask) = mask {
                check_mask(mask)?;
            }
        }
        disjoint(
            "reorders",
            params.reorder.iter().map(|r| &r.blocks),
            params.blocks,
        )?;
        for reorder in &params.reorder {
            reorder.reorder.check(context, input)?;
        }
        for carry in &params.carry {
            if carry.save_after >= carry.add_after || carry.add_after >= params.blocks {
                return Err(format!(
                    "carry from block {} to block {} is not forward within {} blocks",
                    carry.save_after, carry.add_after, params.blocks
                ));
            }
        }
        match &params.sequences {
            None => {}
            Some(Sequences::Groups { rows }) => {
                if *rows == 0 || input.segmented || input.multiple % u64::from(*rows) != 0 {
                    return Err(format!(
                        "groups of {rows} rows need joined items whose rows come in whole \
                         groups (a group op before); here they come in multiples of {}",
                        input.multiple
                    ));
                }
            }
            Some(Sequences::Item {}) if !input.segmented => {}
            Some(other) => {
                return Err(format!(
                    "stage-level sequences `{}` are not allowed here",
                    super::op(other)
                ));
            }
        }
        Ok(input.with(Data::Rows {
            width,
            element: context.rows_element(),
        }))
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        let params = self.params();
        let width = input.width()?;
        let wide = u64::from(width);
        let attention = params.attention.params();
        let qkv = params.attention.qkv_width()?;
        let mut roles = params.norm.slot_roles("norm1", width);
        roles.extend(params.norm.slot_roles("norm2", width));
        roles.push(Role::new("qkv.weight", vec![qkv, wide]));
        if attention.qkv_bias {
            roles.push(Role::new("qkv.bias", vec![qkv]));
        }
        if params.attention.qk_norm_eps()?.is_some() {
            let head = vec![u64::from(attention.head_dim)];
            roles.push(Role::new("q_norm.weight", head.clone()));
            roles.push(Role::new("k_norm.weight", head));
        }
        roles.push(Role::new(
            "proj.weight",
            vec![wide, params.attention.attention_width()?],
        ));
        if attention.proj_bias {
            roles.push(Role::new("proj.bias", vec![wide]));
        }
        let sinks = self.sink_blocks();
        if !sinks.is_empty() {
            let role =
                Role::new("sinks", vec![u64::from(attention.q_heads)]).element(super::Element::F32);
            roles.push(if sinks.len() == params.blocks as usize {
                role
            } else {
                role.blocks(sinks)
            });
        }
        roles.extend(params.mlp.roles(width)?);
        Ok(roles)
    }

    fn blocks(&self) -> Option<u32> {
        Some(self.params().blocks)
    }
}
