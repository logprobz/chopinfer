//! Model buffer identities bind declaration ports without duplicating source geometry.
use crate::contract::Shape;
use crate::precision::DenseRole;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::{
    MatrixGeometry,
    format::{KernelFamily, registry},
};
use projection_spec::derive::{Topology, rank_selection};
use projection_spec::{Declaration, Error, Geometry, Projection};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

/// One topology's expanded declaration, with the stem layers, rank-local
/// geometry and dense ports that lowering and admission read repeatedly.
pub struct Ports {
    pub declaration: Declaration,
    pub geometry: Geometry,
    pub shape: Shape,
    stems: BTreeMap<String, Vec<u64>>,
    /// Each stem's executed layers: its declared execution indices, else
    /// its materialized ones.
    executed: BTreeMap<String, Vec<u64>>,
    selections: BTreeMap<String, (MatrixGeometry, MatrixGeometry)>,
    /// The dense port of each declared projection, through the projection
    /// it publishes as.
    roles: BTreeMap<String, DenseRole>,
    /// Each executed layer's dense ports and their operation IDs: the main
    /// layers, then the drafter layers the declaration executes after them.
    layers: Vec<Vec<(DenseRole, String)>>,
    /// The declared projection of each port at each executed layer.
    projections: Vec<BTreeMap<DenseRole, String>>,
    bindings: OnceLock<Result<BTreeMap<String, SourceBinding>, Error>>,
    operations: OnceLock<BTreeMap<String, String>>,
    drafter_sources: OnceLock<Result<BTreeSet<String>, Error>>,
}

pub fn get(shape: Shape) -> Result<&'static Ports, Error> {
    type Cache = BTreeMap<(usize, u32), &'static Ports>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let key = (std::ptr::from_ref(shape.geometry) as usize, shape.ranks());
    let cache = CACHE.get_or_init(Mutex::default);
    let lock = || {
        cache
            .lock()
            .map_err(|_| Error("projection port cache".into()))
    };
    if let Some(ports) = lock()?.get(&key) {
        return Ok(ports);
    }
    let ports = Box::leak(Box::new(Ports::new(shape)?));
    Ok(*lock()?.entry(key).or_insert(ports))
}

impl Ports {
    fn new(shape: Shape) -> Result<Self, Error> {
        Self::with_declaration(shape, super::declaration_for(shape)?)
    }

    /// The ports of a parsed declaration other than the committed one, at
    /// one topology.
    pub fn with_declaration(shape: Shape, declaration: Declaration) -> Result<Self, Error> {
        let geometry = super::geometry_for(shape)?;
        let declaration = declaration.expanded_profiles(&geometry)?;
        let stems = declaration
            .layer_stems
            .iter()
            .map(|s| Ok((s.id.clone(), s.indices.resolve(&geometry)?)))
            .collect::<Result<_, Error>>()?;
        let executed = declaration
            .layer_stems
            .iter()
            .map(|s| Ok((s.id.clone(), s.executed(&geometry)?)))
            .collect::<Result<_, Error>>()?;
        let topology = Topology {
            ranks: shape.ranks(),
            rank: 0,
        };
        let selections = declaration
            .projections
            .iter()
            .filter_map(|p| {
                let selected = rank_selection(p, &geometry, topology, 0).ok()?;
                Some((p.id.clone(), (selected.logical, selected.physical)))
            })
            .collect();
        let mut ports = Self {
            declaration,
            geometry,
            shape,
            stems,
            executed,
            selections,
            roles: BTreeMap::new(),
            layers: vec![],
            projections: vec![],
            bindings: OnceLock::new(),
            operations: OnceLock::new(),
            drafter_sources: OnceLock::new(),
        };
        let output = |id| ports.geometry(id, false);
        let attention = output("full.output")? != output("linear.output")?;
        ports.roles = ports
            .declaration
            .projections
            .iter()
            .filter_map(|p| {
                let role = DenseRole::declared(ports.declaration.published(&p.id), attention)?;
                Some((p.id.clone(), role))
            })
            .collect();
        for layer in 0..u64::from(shape.layers() + shape.mtp_layers()) {
            let mut roles: Vec<(DenseRole, String)> = Vec::new();
            let mut projections = BTreeMap::new();
            for p in &ports.declaration.projections {
                let Some(&role) = ports.roles.get(&p.id) else {
                    continue;
                };
                let owned = p.stem().is_some_and(|stem| {
                    ports
                        .executed
                        .get(stem)
                        .is_some_and(|layers| layers.contains(&layer))
                });
                if owned && roles.iter().all(|(r, _)| *r != role) {
                    roles.push((role, ports.operation(&p.id, Some(layer))?));
                    projections.insert(role, p.id.clone());
                }
            }
            ports.layers.push(roles);
            ports.projections.push(projections);
        }
        Ok(ports)
    }

    /// The stem index a projection instance reads at an executed layer.
    fn stem_index(&self, id: &str, layer: Option<u64>) -> Result<Option<u64>, Error> {
        layer
            .map(|layer| self.declaration.source_layer(&self.geometry, id, layer))
            .transpose()
    }

    /// [`Self::stem_index`] for a projection's checkpoint source.
    pub fn source_layer(&self, id: &str, layer: u32) -> Result<u32, Error> {
        let index = self
            .declaration
            .source_layer(&self.geometry, id, u64::from(layer))?;
        u32::try_from(index).map_err(|e| Error(e.to_string()))
    }

    /// The projection whose published port and offers serve `id`.
    pub fn published<'a>(&'a self, id: &'a str) -> &'a str {
        self.declaration.published(id)
    }

    /// The declared projection executed at `layer` that publishes as
    /// `published`: the main one, or the drafter's at the MTP layer.
    pub fn executed_at(&self, layer: u32, published: &str) -> Result<&Projection, Error> {
        self.declaration
            .executed_at(&self.geometry, u64::from(layer), published)
    }

    /// The first executed layer of a stem.
    pub fn first_executed(&self, stem: &str) -> Result<Option<u64>, Error> {
        self.executed
            .get(stem)
            .map(|layers| layers.first().copied())
            .ok_or_else(|| Error(format!("undeclared layer stem {stem}")))
    }

    pub fn projection(&self, id: &str) -> Result<&Projection, Error> {
        self.declaration
            .projections
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| Error(format!("undeclared projection {id}")))
    }

    pub fn stem_layers(&self, id: &str) -> Result<&[u64], Error> {
        self.stems
            .get(id)
            .map(Vec::as_slice)
            .ok_or_else(|| Error(format!("undeclared layer stem {id}")))
    }

    pub fn source_name(
        &self,
        source: &projection_spec::TensorSource,
        layer: Option<u32>,
    ) -> Result<String, Error> {
        let mut name = source.tensor.clone();
        if let Some(id) = &source.stem {
            let stem = self
                .declaration
                .layer_stems
                .iter()
                .find(|s| s.id == *id)
                .ok_or_else(|| Error("missing source stem".into()))?;
            let layer = layer
                .map(u64::from)
                .or(self.stem_layers(id)?.first().copied())
                .ok_or_else(|| Error("missing source layer".into()))?;
            name = name.replace("{stem}", &stem.template.replace("{i}", &layer.to_string()));
        }
        Ok(name)
    }

    /// The checkpoint names of one projection instance's operands.
    pub fn names(&self, id: &str, layer: Option<u64>) -> Result<Vec<String>, Error> {
        self.declaration
            .instance_sources(&self.geometry, id, self.stem_index(id, layer)?)
    }

    /// The operation ID of one projection instance, named by its first
    /// operand; fused gate and up operands name the gate-up operation.
    pub fn operation(&self, id: &str, layer: Option<u64>) -> Result<String, Error> {
        let names = self.names(id, layer)?;
        let first = names
            .first()
            .ok_or_else(|| Error(format!("projection {id} names no operand")))?;
        Ok(first
            .trim_end_matches(".weight")
            .replace(".gate_proj", ".gate_up_proj"))
    }

    /// Rank-local geometry, logical or after rank padding.
    pub fn geometry(&self, id: &str, logical: bool) -> Result<MatrixGeometry, Error> {
        let (selected, physical) = self
            .selections
            .get(id)
            .ok_or_else(|| Error(format!("projection {id} has no rank selection")))?;
        Ok(if logical { *selected } else { *physical })
    }

    pub fn weight_geometry(&self, id: &str, weight: WeightFormat) -> Result<MatrixGeometry, Error> {
        let mut geometry = self.geometry(id, false)?;
        let padding = registry::by_format(weight)
            .map_err(|e| Error(e.to_string()))?
            .padding_tile(KernelFamily::Dense);
        geometry.rows = geometry.rows.next_multiple_of(padding.rows as u64);
        geometry.k = geometry.k.next_multiple_of(padding.k as u64);
        Ok(geometry)
    }

    pub fn role(&self, id: &str) -> Option<DenseRole> {
        self.roles.get(id).copied()
    }

    /// The dense ports of an executed layer in declaration order.
    pub fn layer(&self, layer: u32) -> &[(DenseRole, String)] {
        self.layers.get(layer as usize).map_or(&[], Vec::as_slice)
    }

    /// The declared projection of a dense port at an executed layer.
    pub fn layer_projection(&self, layer: u32, role: DenseRole) -> Option<&str> {
        self.projections
            .get(layer as usize)?
            .get(&role)
            .map(String::as_str)
    }

    /// The operation ID of a declared dense projection at an executed layer.
    pub fn layer_operation(&self, layer: u32, projection: &str) -> Option<&str> {
        let role = self.role(projection)?;
        self.layer(layer)
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, id)| id.as_str())
    }

    /// Every dense port of the topology, with the attention output last.
    pub fn roles(&self) -> Vec<DenseRole> {
        let mut result: Vec<_> = self.roles.values().copied().collect();
        result.sort_by_key(|r| {
            if *r == DenseRole::AttentionOutput {
                usize::MAX
            } else {
                *r as usize
            }
        });
        result.dedup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Ranks;
    use crate::geometry::{FLASH, Geometry as ModelGeometry, LayerKind};

    #[test]
    fn dimensions_and_sparse_selectors_follow_the_owning_geometry() {
        let mut changed = FLASH;
        changed.shape.layers = &[
            LayerKind::Full,
            LayerKind::Linear,
            LayerKind::Full,
            LayerKind::Linear,
        ];
        changed.shape.ple.layer_ids = &[2];
        changed.shape.linear.key_heads = 8;
        changed.shape.linear.value_heads = 24;
        changed.shape.attention.heads = 24;
        changed.shape.attention.head_dim = 128;
        changed.shape.indexer.heads = 8;
        let changed = Box::leak(Box::new(changed));
        for geometry in crate::geometry::GEOMETRIES
            .iter()
            .chain([changed as &ModelGeometry])
        {
            for topology in crate::contract::TOPOLOGIES {
                let shape = Shape::new(geometry, topology.ranks);
                let ports = get(shape).unwrap();
                for family in ["linear", "full", "ple"] {
                    let expected = match family {
                        "linear" => (0..shape.layers())
                            .filter(|&i| !shape.is_full(i))
                            .collect::<Vec<_>>(),
                        "full" => (0..shape.layers()).filter(|&i| shape.is_full(i)).collect(),
                        _ => geometry.ple_layer_ids().iter().map(|i| i - 1).collect(),
                    };
                    assert_eq!(
                        ports.stem_layers(family).unwrap(),
                        expected.iter().copied().map(u64::from).collect::<Vec<_>>()
                    );
                }
                let geometry = |id, logical| ports.geometry(id, logical).unwrap();
                assert_eq!(
                    geometry("linear.qkv", true).rows,
                    u64::from(shape.linear_qkv)
                );
                assert_eq!(
                    geometry("full.q", true).rows,
                    u64::from(2 * shape.q_heads * shape.attn_head())
                );
                assert_eq!(
                    geometry("full.output", false).k,
                    u64::from(shape.attn_value)
                );
                assert_eq!(
                    geometry("linear.output", false).k,
                    u64::from(shape.linear_value)
                );
                assert_eq!(
                    geometry("full.index", true).rows,
                    u64::from(shape.index_row())
                );
            }
        }
        let ports = get(Shape::new(changed, Ranks::TP4)).unwrap();
        assert_eq!(
            ports.names("full.q", Some(0)).unwrap(),
            ["model.language_model.layers.0.self_attn.q_proj.weight"]
        );
        assert_eq!(
            ports.layer(0)[0],
            (
                DenseRole::AttentionQ,
                "model.language_model.layers.0.self_attn.q_proj".into()
            )
        );
        assert!(ports.projection("unknown").is_err());
    }

    /// The MTP layer follows the main layers as a full-attention layer:
    /// its dense ports name the drafter layer's tensors.
    #[test]
    fn the_mtp_layer_has_full_attention_ports_with_drafter_names() {
        for topology in crate::contract::TOPOLOGIES {
            let shape = Shape::flash(topology.ranks);
            let ports = get(shape).unwrap();
            let layer = shape.mtp_layer();
            let full = (0..shape.layers()).find(|&l| shape.is_full(l)).unwrap();
            let roles = |layer| {
                ports
                    .layer(layer)
                    .iter()
                    .map(|(r, _)| *r)
                    .collect::<Vec<_>>()
            };
            assert_eq!(roles(layer), roles(full));
            for (role, id) in ports.layer(layer) {
                assert!(id.starts_with("mtp.layers.0."), "{id}");
                let projection = role.projection_id(shape, Some(layer));
                assert!(projection.starts_with("mtp."), "{projection}");
                assert_eq!(ports.layer_operation(layer, projection), Some(id.as_str()));
                assert_eq!(
                    ports.names(projection, Some(u64::from(layer))).unwrap()[0],
                    format!("{id}.weight").replace(".gate_up_proj", ".gate_proj")
                );
            }
            assert!(ports.layer(layer + 1).is_empty());
        }
    }
}

#[derive(Clone)]
pub struct SourceBinding {
    pub projection: String,
    pub operation: String,
    pub operand: usize,
    pub expert: Option<u32>,
    pub shard_rows: Option<std::ops::Range<u64>>,
}

impl Ports {
    /// Original checkpoint members retain their declared assembly ownership.
    pub fn bindings(&self) -> Result<&BTreeMap<String, SourceBinding>, projection_spec::Error> {
        self.bindings
            .get_or_init(|| self.build_bindings())
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The checkpoint tensors the declared drafters own.
    pub fn drafter_sources(&self) -> Result<&BTreeSet<String>, Error> {
        self.drafter_sources
            .get_or_init(|| self.declaration.drafter_sources(&self.geometry))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The declared projection of a prepared operation ID.
    pub fn operation_projection(
        &self,
        operation: &str,
    ) -> Result<Option<&str>, projection_spec::Error> {
        let bindings = self.bindings()?;
        let operations = self.operations.get_or_init(|| {
            bindings
                .values()
                .map(|b| (b.operation.clone(), b.projection.clone()))
                .collect()
        });
        Ok(operations.get(operation).map(String::as_str))
    }

    fn build_bindings(
        &self,
    ) -> Result<std::collections::BTreeMap<String, SourceBinding>, projection_spec::Error> {
        let mut result = std::collections::BTreeMap::new();
        let mut operations = BTreeMap::new();
        let narrow = |value: u64| u32::try_from(value).map_err(|e| Error(e.to_string()));
        for (name, position) in self.declaration.source_positions(&self.geometry)? {
            let projection_spec::bind::SourceOwner::Projection(index) = position.owner else {
                continue;
            };
            let projection = &self.declaration.projections[index];
            let (operation, shard_rows) = if let Some(shard) = position.shard {
                let shards = projection
                    .lookup
                    .as_ref()
                    .ok_or_else(|| Error("missing lookup shards".into()))?
                    .shards
                    .resolve(&self.geometry)?;
                let rows = &shards[shard as usize].rows;
                let start = rows.start.resolve_offset(&self.geometry)?;
                let operation = name
                    .split_once(".shard_")
                    .ok_or_else(|| Error("missing lookup shard identity".into()))?
                    .0
                    .into();
                (
                    operation,
                    Some(start..start + rows.length.resolve(&self.geometry)?),
                )
            } else {
                let key = (index, position.layer);
                let operation = match operations.get(&key) {
                    Some(operation) => String::clone(operation),
                    None => {
                        let operation = self.operation(&projection.id, position.layer)?;
                        operations.insert(key, operation.clone());
                        operation
                    }
                };
                (operation, None)
            };
            result.insert(
                name,
                SourceBinding {
                    projection: projection.id.clone(),
                    operation,
                    operand: position.operand.unwrap_or(0) as usize,
                    expert: position.expert.map(narrow).transpose()?,
                    shard_rows,
                },
            );
        }
        Ok(result)
    }
}
