//! The weight mapping: each stage names the checkpoint tensors behind its
//! roles, through a closed set of exact transforms. Each role states the
//! shape and element the kernel reads; `Source::check` refuses what the
//! declaration alone contradicts and `Source::resolve` checks a source
//! against checkpoint metadata.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::blocks::BlockSet;
use crate::components::{Element, mul};
use crate::policy::Precision;

/// Role name to source, in role order.
pub type Weights = BTreeMap<String, Source>;

/// A source's shape and stored dtype after its transforms (`None` for
/// zeros, which take their siblings' dtype).
type Resolved = (Vec<u64>, Option<String>);

/// How a source tensor becomes the bytes a kernel reads. Every transform is
/// a reindex or an exact widening, except `narrow_bf16`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Transform {
    /// All axes after the first as one: `[out, c, t, h, w]` read as
    /// `[out, c*t*h*w]`. The same bytes.
    Flatten,
    /// A convolution weight stored `[out, channel, tap]`, rearranged to
    /// `[out, tap, channel]` as im2col columns take it.
    TapsFirst,
    /// Zero rows appended up to this many rows (a vector's elements).
    PadRows(u64),
    /// Zero columns appended up to this many columns.
    PadCols(u64),
    /// BF16 widened to F32.
    WidenF32,
    /// F32 rounded to BF16. Not exact: `bf16-legacy.v1` only.
    NarrowBf16,
    /// The columns of a `factor x factor` space-to-depth unfold permuted
    /// from channel-major (`c*f*f + ky*f + kx`) to position-major
    /// (`(ky*f + kx)*channels + c`).
    SpaceToDepthCols(u32),
}

/// One checkpoint tensor. Written as a plain name when it has no transform
/// and no block set.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    /// The name after the checkpoint prefix; `{i}` is the block index.
    pub tensor: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<Transform>,
    /// The blocks that have this tensor, when not every block does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks: Option<BlockSet>,
}

/// Sources stacked along their first axis, in order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Concat {
    pub concat_rows: Vec<Source>,
}

/// `count` tensors named by `concat_range` with `{j}` = 0..count, each
/// transformed, stacked along their first axis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcatRange {
    pub concat_range: String,
    pub count: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<Transform>,
}

/// A zero vector of this many elements: a bias, or part of one, that the
/// checkpoint does not hold.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Zeros {
    pub zeros: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source {
    Tensor(Tensor),
    Concat(Concat),
    ConcatRange(ConcatRange),
    Zeros(Zeros),
}

impl Serialize for Source {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Source::Tensor(tensor) if tensor.transforms.is_empty() && tensor.blocks.is_none() => {
                serializer.serialize_str(&tensor.tensor)
            }
            Source::Tensor(tensor) => tensor.serialize(serializer),
            Source::Concat(concat) => concat.serialize(serializer),
            Source::ConcatRange(range) => range.serialize(serializer),
            Source::Zeros(zeros) => zeros.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Source {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let key = match &value {
            Value::String(name) => {
                return Ok(Source::Tensor(Tensor {
                    tensor: name.clone(),
                    transforms: Vec::new(),
                    blocks: None,
                }));
            }
            Value::Object(object) => ["tensor", "concat_rows", "concat_range", "zeros"]
                .into_iter()
                .find(|key| object.contains_key(*key)),
            _ => None,
        };
        let parsed = match key {
            Some("tensor") => serde_json::from_value(value).map(Source::Tensor),
            Some("concat_rows") => serde_json::from_value(value).map(Source::Concat),
            Some("concat_range") => serde_json::from_value(value).map(Source::ConcatRange),
            Some(_) => serde_json::from_value(value).map(Source::Zeros),
            None => {
                return Err(D::Error::custom(
                    "a weight source is a tensor name or an object with `tensor`, \
                     `concat_rows`, `concat_range` or `zeros`",
                ));
            }
        };
        parsed.map_err(D::Error::custom)
    }
}

/// A weight a component reads: its shape and element after the source's
/// transforms, as the kernel takes it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Role {
    pub name: String,
    pub shape: Vec<u64>,
    /// The element the kernel reads. Every shared variant reads BF16
    /// weights unless its role states otherwise; binding refuses any other
    /// stored dtype that the transforms do not turn into it.
    pub element: Element,
    /// For a role that stacks a fixed sequence of tensors: each part's
    /// shape, in order. The kernel indexes the parts separately, so equal
    /// totals are not enough.
    pub parts: Option<Vec<Vec<u64>>>,
    /// The blocks that have the role, when not every block does.
    pub blocks: Option<Vec<u32>>,
}

impl Role {
    /// A BF16 role of `shape`.
    pub fn new(name: impl Into<String>, shape: Vec<u64>) -> Role {
        Role {
            name: name.into(),
            shape,
            element: Element::Bf16,
            parts: None,
            blocks: None,
        }
    }

    /// A BF16 role stacking `parts` along their first axis, in order.
    pub fn stacked(name: impl Into<String>, parts: Vec<Vec<u64>>) -> Result<Role, String> {
        let (first, rest) = parts
            .first()
            .and_then(|part| part.split_first())
            .ok_or("a stacked role has non-empty parts")?;
        let mut rows = *first;
        for part in &parts[1..] {
            if part.get(1..) != Some(rest) {
                return Err(format!(
                    "stacked parts differ past the first axis: {part:?}"
                ));
            }
            rows = rows.checked_add(part[0]).ok_or("stacked rows overflow")?;
        }
        let mut shape = vec![rows];
        shape.extend_from_slice(rest);
        let mut role = Role::new(name, shape);
        role.parts = Some(parts);
        Ok(role)
    }

    pub fn element(mut self, element: Element) -> Role {
        self.element = element;
        self
    }

    pub fn blocks(mut self, blocks: Vec<u32>) -> Role {
        self.blocks = Some(blocks);
        self
    }
}

/// A checkpoint tensor's metadata, for binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    /// The safetensors dtype, `BF16`, `F32`, ...
    pub dtype: String,
    pub shape: Vec<u64>,
    /// The file holding it, when known.
    pub file: Option<String>,
}

/// Where a source appears.
#[derive(Clone, Copy)]
pub(crate) struct Place {
    /// Inside a transformer stage: every tensor name has `{i}`.
    pub per_block: bool,
    pub precision: Precision,
}

fn check_name(name: &str, place: Place, range: bool) -> Result<(), String> {
    if name.is_empty() {
        return Err("a tensor name is empty".into());
    }
    if name.contains("{i}") != place.per_block {
        return Err(if place.per_block {
            format!("tensor `{name}` is inside a transformer stage but has no `{{i}}`")
        } else {
            format!("tensor `{name}` has `{{i}}` outside a transformer stage")
        });
    }
    if name.contains("{j}") != range {
        return Err(format!(
            "tensor `{name}`: `{{j}}` appears only in, and always in, a `concat_range`"
        ));
    }
    let rest = name.replace("{i}", "").replace("{j}", "");
    if rest.contains('{') || rest.contains('}') {
        return Err(format!("tensor `{name}` has an unknown placeholder"));
    }
    Ok(())
}

fn check_transforms(transforms: &[Transform], place: Place) -> Result<(), String> {
    let mut seen = HashSet::new();
    for transform in transforms {
        if !seen.insert(std::mem::discriminant(transform)) {
            return Err(format!("transform {transform:?} is repeated"));
        }
        match *transform {
            Transform::NarrowBf16 if place.precision != Precision::Bf16Legacy => {
                return Err(
                    "`narrow_bf16` rounds weights and is allowed only under bf16-legacy.v1".into(),
                );
            }
            Transform::PadRows(0) | Transform::PadCols(0) | Transform::SpaceToDepthCols(0..=1) => {
                return Err(format!("transform {transform:?} is degenerate"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn expand(prefix: &str, name: &str, blocks: &[u32], out: &mut BTreeSet<String>) {
    if name.contains("{i}") {
        for block in blocks {
            out.insert(format!(
                "{prefix}{}",
                name.replace("{i}", &block.to_string())
            ));
        }
    } else {
        out.insert(format!("{prefix}{name}"));
    }
}

fn element(dtype: &str) -> Option<Element> {
    match dtype {
        "F32" => Some(Element::F32),
        "BF16" => Some(Element::Bf16),
        _ => None,
    }
}

fn apply(shape: &mut Vec<u64>, dtype: &mut String, transform: Transform) -> Result<(), String> {
    match transform {
        Transform::Flatten => {
            if shape.len() < 2 {
                return Err(format!("cannot flatten shape {shape:?}"));
            }
            let mut rest = 1;
            for &dim in &shape[1..] {
                rest = mul(rest, dim)?;
            }
            *shape = vec![shape[0], rest];
        }
        Transform::TapsFirst => {
            let &[out, channels, taps] = shape.as_slice() else {
                return Err(format!(
                    "taps_first needs [out, channel, tap], not {shape:?}"
                ));
            };
            *shape = vec![out, taps, channels];
        }
        Transform::PadRows(rows) => match shape.first_mut() {
            Some(first) if *first <= rows => *first = rows,
            _ => return Err(format!("cannot pad shape {shape:?} to {rows} rows")),
        },
        Transform::PadCols(columns) => match shape.as_mut_slice() {
            [_, second] if *second <= columns => *second = columns,
            _ => return Err(format!("cannot pad shape {shape:?} to {columns} columns")),
        },
        Transform::WidenF32 | Transform::NarrowBf16 => {
            let (from, to) = if transform == Transform::WidenF32 {
                ("BF16", "F32")
            } else {
                ("F32", "BF16")
            };
            if dtype != from {
                return Err(format!("{transform:?} needs a {from} tensor, not {dtype}"));
            }
            *dtype = to.into();
        }
        Transform::SpaceToDepthCols(factor) => {
            let cell = mul(u64::from(factor), u64::from(factor))?;
            if shape.len() != 2 || shape[1] % cell != 0 {
                return Err(format!(
                    "shape {shape:?} has no {factor} x {factor} unfold columns"
                ));
            }
        }
    }
    Ok(())
}

impl Source {
    /// Refuses a source that cannot satisfy `role` whatever the checkpoint
    /// holds.
    pub(crate) fn check(&self, place: Place, role: &Role) -> Result<(), String> {
        let vector = role.shape.len() == 1;
        if let Some(parts) = &role.parts {
            let parts = parts.len() as u64;
            let written = match self {
                Source::Concat(concat) => concat.concat_rows.len() as u64,
                Source::ConcatRange(range) => u64::from(range.count),
                _ => 1,
            };
            if written != parts {
                return Err(format!(
                    "stacks {written} tensors where the role stacks {parts}"
                ));
            }
        }
        match self {
            Source::Tensor(tensor) => {
                check_name(&tensor.tensor, place, false)?;
                if tensor.blocks.is_some() && !place.per_block {
                    return Err(format!(
                        "tensor `{}` has a block set outside a transformer stage",
                        tensor.tensor
                    ));
                }
                check_transforms(&tensor.transforms, place)?;
                for transform in &tensor.transforms {
                    let (axis, target) = match *transform {
                        Transform::PadRows(rows) => (0, rows),
                        Transform::PadCols(columns) => (1, columns),
                        _ => continue,
                    };
                    if role.shape.get(axis) != Some(&target) {
                        return Err(format!(
                            "{transform:?} does not give the role's shape {:?}",
                            role.shape
                        ));
                    }
                }
                Ok(())
            }
            Source::Concat(concat) => {
                if concat.concat_rows.len() < 2 {
                    return Err("`concat_rows` stacks at least two sources".into());
                }
                let mut zeros = 0_u64;
                for part in &concat.concat_rows {
                    match part {
                        Source::Tensor(Tensor {
                            blocks: Some(_),
                            tensor,
                            ..
                        }) => {
                            return Err(format!(
                                "tensor `{tensor}` inside `concat_rows` has a block set"
                            ));
                        }
                        Source::Tensor(tensor) => {
                            check_name(&tensor.tensor, place, false)?;
                            check_transforms(&tensor.transforms, place)?;
                        }
                        Source::Zeros(part) if vector && part.zeros > 0 => {
                            zeros = zeros.saturating_add(part.zeros);
                        }
                        Source::Zeros(_) => {
                            return Err("zeros stack only into a non-empty vector".into());
                        }
                        Source::Concat(_) | Source::ConcatRange(_) => {
                            return Err("`concat_rows` parts are tensors or zeros".into());
                        }
                    }
                }
                if zeros >= role.shape[0] {
                    return Err(format!(
                        "{zeros} zeros leave no room for the other parts of {} elements",
                        role.shape[0]
                    ));
                }
                Ok(())
            }
            Source::ConcatRange(range) => {
                if range.count < 2 {
                    return Err("`concat_range` stacks at least two tensors".into());
                }
                check_name(&range.concat_range, place, true)?;
                check_transforms(&range.transforms, place)
            }
            Source::Zeros(zeros) if vector && role.shape == [zeros.zeros] => Ok(()),
            Source::Zeros(zeros) => Err(format!(
                "{} zeros cannot give the role's shape {:?}",
                zeros.zeros, role.shape
            )),
        }
    }

    /// The block set the source declares, if any.
    pub fn blocks(&self) -> Option<&BlockSet> {
        match self {
            Source::Tensor(tensor) => tensor.blocks.as_ref(),
            _ => None,
        }
    }

    /// Every checkpoint tensor name the source reads, with `{i}` taken over
    /// `blocks`.
    pub(crate) fn names(&self, prefix: &str, blocks: &[u32], out: &mut BTreeSet<String>) {
        match self {
            Source::Tensor(tensor) => {
                let own: Vec<u32> = blocks
                    .iter()
                    .copied()
                    .filter(|&block| tensor.blocks.as_ref().is_none_or(|set| set.contains(block)))
                    .collect();
                expand(prefix, &tensor.tensor, &own, out);
            }
            Source::Concat(concat) => {
                for part in &concat.concat_rows {
                    part.names(prefix, blocks, out);
                }
            }
            Source::ConcatRange(range) => {
                for j in 0..range.count {
                    expand(
                        prefix,
                        &range.concat_range.replace("{j}", &j.to_string()),
                        blocks,
                        out,
                    );
                }
            }
            Source::Zeros(_) => {}
        }
    }

    /// The shape and dtype the source gives for `block`, from checkpoint
    /// metadata. Zeros take their siblings' dtype (`None` alone).
    pub(crate) fn resolve(
        &self,
        prefix: &str,
        block: u32,
        lookup: &dyn Fn(&str) -> Option<TensorInfo>,
    ) -> Result<Resolved, String> {
        let one = |name: &str, transforms: &[Transform]| {
            let name = format!("{prefix}{}", name.replace("{i}", &block.to_string()));
            let info = lookup(&name).ok_or_else(|| format!("the checkpoint has no `{name}`"))?;
            let (mut shape, mut dtype) = (info.shape, info.dtype);
            for &transform in transforms {
                apply(&mut shape, &mut dtype, transform).map_err(|e| format!("`{name}`: {e}"))?;
            }
            Ok::<_, String>((shape, Some(dtype)))
        };
        let stack = |parts: Vec<Resolved>| {
            let mut rows = 0_u64;
            let mut tail: Option<Vec<u64>> = None;
            let mut dtype: Option<String> = None;
            for (shape, part_dtype) in parts {
                let (&first, rest) = shape.split_first().ok_or("a scalar cannot stack")?;
                if tail.get_or_insert_with(|| rest.to_vec()) != rest {
                    return Err(format!(
                        "stacked parts differ past the first axis: {rest:?}"
                    ));
                }
                if let Some(part_dtype) = part_dtype
                    && dtype.get_or_insert_with(|| part_dtype.clone()) != &part_dtype
                {
                    return Err(format!("stacked parts mix {dtype:?} and {part_dtype}"));
                }
                rows = rows.checked_add(first).ok_or("stacked rows overflow")?;
            }
            let mut shape = vec![rows];
            shape.extend(tail.unwrap_or_default());
            Ok((shape, dtype))
        };
        match self {
            Source::Tensor(tensor) => one(&tensor.tensor, &tensor.transforms),
            Source::Zeros(zeros) => Ok((vec![zeros.zeros], None)),
            Source::Concat(concat) => stack(
                concat
                    .concat_rows
                    .iter()
                    .map(|part| part.resolve(prefix, block, lookup))
                    .collect::<Result<_, _>>()?,
            ),
            Source::ConcatRange(range) => stack(
                (0..range.count)
                    .map(|j| {
                        one(
                            &range.concat_range.replace("{j}", &j.to_string()),
                            &range.transforms,
                        )
                    })
                    .collect::<Result<_, _>>()?,
            ),
        }
    }

    /// The shape and dtype of each stacked part, for a stacking source.
    fn resolve_parts(
        &self,
        prefix: &str,
        block: u32,
        lookup: &dyn Fn(&str) -> Option<TensorInfo>,
    ) -> Result<Vec<Resolved>, String> {
        match self {
            Source::Concat(concat) => concat
                .concat_rows
                .iter()
                .map(|part| part.resolve(prefix, block, lookup))
                .collect(),
            Source::ConcatRange(range) => (0..range.count)
                .map(|j| {
                    Source::Tensor(Tensor {
                        tensor: range.concat_range.replace("{j}", &j.to_string()),
                        transforms: range.transforms.clone(),
                        blocks: None,
                    })
                    .resolve(prefix, block, lookup)
                })
                .collect(),
            _ => Err("the role stacks parts; the source does not".into()),
        }
    }

    /// Checks the source against `role` with checkpoint metadata: the
    /// total shape, each stacked part's shape, and every part's element.
    pub(crate) fn bind(
        &self,
        role: &Role,
        prefix: &str,
        block: u32,
        lookup: &dyn Fn(&str) -> Option<TensorInfo>,
    ) -> Result<(), String> {
        let expected = crate::components::op(&role.element);
        let check_dtype = |dtype: &Option<String>| match dtype {
            Some(dtype) if element(dtype) != Some(role.element) => {
                Err(format!("gives {dtype}, the role reads {expected}"))
            }
            _ => Ok(()),
        };
        if let Some(parts) = &role.parts {
            let resolved = self.resolve_parts(prefix, block, lookup)?;
            for (index, ((shape, dtype), part)) in resolved.iter().zip(parts).enumerate() {
                if shape != part {
                    return Err(format!(
                        "part {index} gives shape {shape:?}, the role reads {part:?}"
                    ));
                }
                check_dtype(dtype).map_err(|error| format!("part {index} {error}"))?;
            }
        }
        let (shape, dtype) = self.resolve(prefix, block, lookup)?;
        if shape != role.shape {
            return Err(format!(
                "gives shape {shape:?}, the role reads {:?}",
                role.shape
            ));
        }
        check_dtype(&dtype)
    }
}
