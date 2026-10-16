//! Component ids: `[<model>:]<kind>.<name>.v<N>`.
//!
//! A shared variant has no namespace. A bespoke variant carries the model
//! that owns it (`<model>:quantize.rvq.v1`) and only that model's
//! declarations may use it.

/// The role a component plays. Each kind's variants are a closed set, one
/// module per kind under `components`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Linear,
    Conv1d,
    Position,
    Norm,
    Attention,
    Rotary,
    Mlp,
    Transformer,
    Reorder,
    Merge,
    Group,
    Cast,
    Projector,
    Pad,
    Quantize,
    Embed,
    Layout,
}

impl Kind {
    const NAMES: [(Kind, &'static str); 17] = [
        (Kind::Linear, "linear"),
        (Kind::Conv1d, "conv1d"),
        (Kind::Position, "position"),
        (Kind::Norm, "norm"),
        (Kind::Attention, "attention"),
        (Kind::Rotary, "rotary"),
        (Kind::Mlp, "mlp"),
        (Kind::Transformer, "transformer"),
        (Kind::Reorder, "reorder"),
        (Kind::Merge, "merge"),
        (Kind::Group, "group"),
        (Kind::Cast, "cast"),
        (Kind::Projector, "projector"),
        (Kind::Pad, "pad"),
        (Kind::Quantize, "quantize"),
        (Kind::Embed, "embed"),
        (Kind::Layout, "layout"),
    ];

    /// Whether a component of this kind can stand as a stage.
    pub fn is_stage(self) -> bool {
        !matches!(
            self,
            Kind::Attention | Kind::Rotary | Kind::Mlp | Kind::Reorder | Kind::Layout
        )
    }

    pub fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(kind, _)| *kind == self)
            .map(|(_, name)| *name)
            .unwrap_or_default()
    }

    fn parse(name: &str) -> Option<Kind> {
        Self::NAMES
            .iter()
            .find(|(_, known)| *known == name)
            .map(|(kind, _)| *kind)
    }
}

/// A parsed component id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComponentId<'a> {
    pub namespace: Option<&'a str>,
    pub kind: Kind,
    pub name: &'a str,
    pub version: u32,
}

fn word(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !text.starts_with('-')
        && !text.ends_with('-')
}

/// Parses `id`, or says which part of the grammar it breaks.
pub fn parse(id: &str) -> Result<ComponentId<'_>, String> {
    let malformed =
        |why: &str| format!("component id `{id}` {why}; expected `[<model>:]<kind>.<name>.v<N>`");
    let (namespace, rest) = match id.split_once(':') {
        Some((namespace, rest)) => {
            if !namespace
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                || namespace.is_empty()
            {
                return Err(malformed("has a malformed model namespace"));
            }
            (Some(namespace), rest)
        }
        None => (None, id),
    };
    let (kind, rest) = rest
        .split_once('.')
        .ok_or_else(|| malformed("has no kind"))?;
    let kind = Kind::parse(kind).ok_or_else(|| malformed("names an unknown kind"))?;
    let (name, version) = rest
        .rsplit_once('.')
        .ok_or_else(|| malformed("has no version"))?;
    if !name.split('.').all(word) {
        return Err(malformed("has a malformed name"));
    }
    let digits = version
        .strip_prefix('v')
        .ok_or_else(|| malformed("has no version"))?;
    if digits.is_empty() || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed("has a malformed version"));
    }
    let version = digits
        .parse()
        .map_err(|_| malformed("has a malformed version"))?;
    Ok(ComponentId {
        namespace,
        kind,
        name,
        version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_follow_the_grammar() {
        let id = parse("position.learned-grid.bilinear-align-corners.f32.v1").unwrap();
        assert_eq!(
            (id.namespace, id.kind, id.name, id.version),
            (
                None,
                Kind::Position,
                "learned-grid.bilinear-align-corners.f32",
                1
            )
        );
        let id = parse("model7:quantize.rvq.v12").unwrap();
        assert_eq!(
            (id.namespace, id.kind, id.version),
            (Some("model7"), Kind::Quantize, 12)
        );
        for bad in [
            "linear",
            "linear.pairs",
            "linear.pairs.v0",
            "linear.pairs.v01",
            "linear.pairs.1",
            "linear..v1",
            "linear.Pairs.v1",
            "kernel.pairs.v1",
            ":linear.pairs.v1",
            "Model:linear.pairs.v1",
            "linear.-pairs.v1",
        ] {
            assert!(parse(bad).is_err(), "accepted {bad}");
        }
    }
}
