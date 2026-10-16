//! The prepared native profiles the helper requires of the bundle module,
//! one per compiled geometry and served rank count, from the contract's
//! typed identity (`note::prepared_profile`). Deriving them here keeps the
//! admitted view table and its capability inventory out of every request;
//! the bundle's build derives the module's note from the same function.
use model_qwen38_contract::contract::{Ranks, Shape};
use model_qwen38_contract::{geometry::GEOMETRIES, note};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let profiles: Vec<_> = GEOMETRIES
        .iter()
        .flat_map(|geometry| {
            [Ranks::TP2, Ranks::TP4]
                .into_iter()
                .filter_map(move |ranks| Shape::of_geometry(geometry, ranks))
                .map(|shape| {
                    serde_json::json!({"geometry": shape.geometry.id,
                        "profile": note::prepared_profile(shape)})
                })
        })
        .collect();
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("output directory"));
    std::fs::write(
        out.join("prepared-profiles.json"),
        serde_json::to_vec(&profiles).expect("profile JSON"),
    )
    .expect("write prepared profiles");
}
