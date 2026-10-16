//! Compare module-facing declarations without reading or staging weight bytes.
use crate::{BundleDraftV1, CompilerError, ComponentContentV1};
use runtime_bundle::{BundleGraphBucketV1, BundleOpaqueId};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

impl BundleDraftV1 {
    /// The ids of the components a deployment reads and the module never
    /// opens ([`runtime_bundle::deployment`]). They are not module-facing, so
    /// the declarations leave them out and the graph buckets the module
    /// declares gain them.
    fn deployment_components(&self) -> Vec<BundleOpaqueId> {
        self.components
            .iter()
            .filter(|c| runtime_bundle::deployment::is_deployment_role(c.role.as_str()))
            .map(|c| c.id.clone())
            .collect()
    }

    pub fn native_declarations(&self) -> Value {
        let deployment = self.deployment_components();
        let mut components: Vec<_> = self
            .components
            .iter()
            .filter(|c| !deployment.contains(&c.id))
            .map(|c| {
                let storage = match &c.content {
                    ComponentContentV1::Whole(_) => json!({"kind":"whole"}),
                    ComponentContentV1::Planned => json!({"kind":"planned"}),
                };
                let mut component =
                    json!({"id":c.id,"role":c.role,"format":c.format,"storage":storage});
                if !c.topologies.is_empty() {
                    component["topologies"] = json!(c.topologies);
                }
                component
            })
            .collect();
        components.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        let mut graphs: Vec<_> = self
            .graph_buckets
            .iter()
            .map(|bucket| {
                let components: Vec<_> = bucket
                    .components()
                    .iter()
                    .filter(|id| !deployment.contains(id))
                    .collect();
                json!({"id":bucket.id(),"components":components,"operations":bucket.operations()})
            })
            .collect();
        graphs.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        let mut declarations = json!({"id":self.id,"target":self.target,"resource_requirements":self.resource_requirements,
            "kv_layout":self.kv_layout,"kv_formats":self.kv_formats,"state_layout":self.state_layout,
            "recurrent_state_layouts":self.recurrent_state_layouts,"components":components,
            "graph_buckets":graphs,"batch_limits":self.batch_limits,"speculation":self.speculation,
            "operation_requirements":self.operation_requirements,"implementation_count":self.implementations.len(),
            "native_symbol_sequences":self.native_symbol_sequences,
            "native_entry":self.native_module.as_ref().map(|_| runtime_bundle::BUNDLE_NATIVE_ENTRY_SYMBOL_V1)});
        if let Some(options) = &self.model_options {
            declarations["model_options"] = json!(options);
        }
        // Declarations of one topology's resources stay as they were.
        if !self.topology_resources.is_empty() {
            declarations["topology_resources"] = json!(self.topology_resources);
        }
        declarations
    }

    /// Read static serving fields from the model-owned declaration, then compare
    /// the entire draft projection, including roles and graph dependencies.
    /// Model options must already match, including their absence.
    pub fn with_native_declarations(mut self, declaration: &Value) -> Result<Self, CompilerError> {
        fn field<T: DeserializeOwned>(v: &Value, name: &str) -> Result<T, CompilerError> {
            serde_json::from_value(
                v.get(name)
                    .ok_or_else(|| {
                        CompilerError::validation(format!("missing native declaration {name}"))
                    })?
                    .clone(),
            )
            .map_err(|e| {
                CompilerError::validation(format!("invalid native declaration {name}: {e}"))
            })
        }
        let model_options: Option<runtime_bundle::model_options::ModelOptionsV1> = declaration
            .get("model_options")
            .map(|_| field(declaration, "model_options"))
            .transpose()?;
        if self.model_options != model_options {
            return Err(CompilerError::validation(
                "draft model options differ from the native declaration",
            ));
        }
        self.id = field(declaration, "id")?;
        self.target = field(declaration, "target")?;
        self.resource_requirements = field(declaration, "resource_requirements")?;
        self.kv_layout = field(declaration, "kv_layout")?;
        self.kv_formats = field(declaration, "kv_formats")?;
        self.state_layout = field(declaration, "state_layout")?;
        self.recurrent_state_layouts = field(declaration, "recurrent_state_layouts")?;
        self.batch_limits = field(declaration, "batch_limits")?;
        self.speculation = field(declaration, "speculation")?;
        let declared: Vec<BundleGraphBucketV1> = field(declaration, "graph_buckets")?;
        let deployment = self.deployment_components();
        self.graph_buckets = declared
            .into_iter()
            .map(|bucket| {
                let mut components = bucket.components().to_vec();
                components.extend(deployment.iter().cloned());
                BundleGraphBucketV1::new(
                    bucket.id().clone(),
                    components,
                    bucket.operations().to_vec(),
                )
            })
            .collect::<Result<_, _>>()?;
        if declaration.get("topology_resources").is_some() {
            self.topology_resources = field(declaration, "topology_resources")?;
        }
        if self.native_declarations() != *declaration {
            return Err(CompilerError::validation(
                "draft differs from the model's native package declarations",
            ));
        }
        self.validate()?;
        Ok(self)
    }
}
