//! Compact checkpoint shard templates. TP ownership and variable heads stay explicit.
use crate::contracts::{LookupShard, RowRange};
use crate::{Dim, Error, Geometry, TensorSource};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LookupShards {
    Explicit(Vec<LookupShard>),
    Template(LookupShardTemplate),
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupShardTemplate {
    pub count: Dim,
    pub source: TensorSource,
    pub rows: RowRange,
}
impl From<Vec<LookupShard>> for LookupShards {
    fn from(shards: Vec<LookupShard>) -> Self {
        Self::Explicit(shards)
    }
}
impl LookupShards {
    pub fn first_source(&self) -> Option<&TensorSource> {
        match self {
            Self::Explicit(shards) => shards.first().map(|s| &s.source),
            Self::Template(template) => Some(&template.source),
        }
    }
    pub fn explicit_mut(&mut self) -> Option<&mut Vec<LookupShard>> {
        match self {
            Self::Explicit(shards) => Some(shards),
            Self::Template(_) => None,
        }
    }
    /// Expand with a local `shard` dimension. Keep the compact wire object intact.
    pub fn resolve(&self, geometry: &Geometry) -> Result<Vec<LookupShard>, Error> {
        let template = match self {
            Self::Explicit(shards) => return Ok(shards.clone()),
            Self::Template(template) => template,
        };
        let count = template.count.resolve(geometry)?;
        if count > crate::validate::MAX_INSTANCES as u64 {
            return Err(Error("lookup shard count exceeds instance limit".into()));
        }
        crate::validate::pattern(&template.source.tensor, &["stem", "shard"])?;
        if !template.source.tensor.contains("{shard}") {
            return Err(Error("lookup shard template needs a shard selector".into()));
        }
        let mut shards = Vec::new();
        for index in 0..count {
            let mut scope = geometry.clone();
            scope.insert("shard".into(), index);
            let start = template.rows.start.resolve_offset(&scope)?;
            let length = template.rows.length.resolve_offset(&scope)?;
            if length == 0 || start.checked_add(length).is_none() {
                return Err(Error(
                    "lookup shard length must be positive and its range bounded".into(),
                ));
            }
            let mut source = template.source.clone();
            source.tensor = source.tensor.replace("{shard}", &index.to_string());
            shards.push(LookupShard {
                id: format!("shard.{index}"),
                source,
                rows: RowRange {
                    start: Dim::Literal(start),
                    length: Dim::Literal(length),
                },
            });
        }
        Ok(shards)
    }
}
