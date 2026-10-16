use std::collections::BTreeMap;

pub fn validate(value: &str, kind: &str) -> Result<(), String> {
    if value.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && value
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
    {
        Ok(())
    } else {
        Err(format!(
            "invalid {kind} {value:?}: expected [a-z][a-z0-9_]*"
        ))
    }
}

pub fn insert(tags: &mut BTreeMap<String, String>, id: &str, tag: &str) -> Result<(), String> {
    validate(tag, "kernel tag")?;
    let reserved = |a: &str, b: &str| {
        b.strip_prefix(a).is_some_and(|suffix| {
            ["_tp", "_decode", "_engine", "_bundle", "_speculation"]
                .iter()
                .any(|reserved| suffix.starts_with(reserved))
        })
    };
    for (other, owner) in tags.iter() {
        if tag == other || reserved(other, tag) || reserved(tag, other) {
            return Err(format!(
                "geometry {id} kernel tag {tag} collides with geometry {owner} tag {other} or its derived names"
            ));
        }
    }
    tags.insert(tag.into(), id.into());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_refuse_invalid_symbols_and_derived_collisions() {
        for invalid in ["", "qwen-x", "1tag", "Tag"] {
            assert!(validate(invalid, "kernel tag").is_err());
        }
        for suffix in [
            "",
            "_tp2",
            "_decode_wide",
            "_engine",
            "_bundle",
            "_speculation_tp2",
        ] {
            let derived = format!("tag{suffix}");
            for (first, second) in [("tag", derived.as_str()), (derived.as_str(), "tag")] {
                let mut tags = BTreeMap::new();
                insert(&mut tags, "first", first).unwrap();
                let error = insert(&mut tags, "second", second).unwrap_err();
                assert!(error.contains("first") && error.contains("second"));
            }
        }
        let mut tags = BTreeMap::new();
        insert(&mut tags, "base", "tag").unwrap();
        insert(&mut tags, "small", "tag_small").unwrap();
    }
}
