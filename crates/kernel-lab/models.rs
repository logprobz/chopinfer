// The build script and the lab share this model-owned inventory boundary.
use crate::lab_shapes;

pub const CATALOGS: &[&kernels_sm120::ModelCatalog] = &[
    &model_mimo26_contract::sm120::CATALOG,
    &model_qwen38_contract::sm120::CATALOG,
    &model_glm5next_contract::sm120::CATALOG,
    &lab_shapes::CATALOG,
];

pub use kernels_sm120::ProjectionInventory;

/// Every catalog's generated projection inventory, in catalog order.
pub fn projection_inventory() -> Result<ProjectionInventory, String> {
    let (mut offers, mut catalog, mut copies) = ProjectionInventory::default();
    for model in CATALOGS {
        let generated = (model.projections)()?;
        offers.extend(generated.0);
        catalog.extend(generated.1);
        copies.extend(generated.2);
    }
    Ok((offers, catalog, copies))
}
