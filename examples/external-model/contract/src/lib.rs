//! Model-owned identities and synthetic arithmetic, independent of host crates.
#![forbid(unsafe_code)]
pub mod frontend;
#[cfg(feature = "ops")]
pub mod ops;
#[cfg(feature = "precision")]
pub mod precision;
#[cfg(feature = "precision")]
pub mod weight_plan;
pub const BUNDLE: &str = "example.stateful.tp1.v1";
pub const MODEL: &str = "example/stateful";
pub const TOPOLOGY: &str = "tp1";
pub const SEED_ROLE: &str = "chop.bundle-metadata.example.seed.v1";
pub const KV: &str = "example.stateful.kv.v1";
pub const STATE: &str = "example.stateful.state.v1";
pub const VOCAB: usize = 272;
pub const CONTEXT: u64 = 1024;
pub const ITEMS: u32 = 4;
pub const MARKERS: [&str; 16] = [
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<think>",
    "</think>",
    "<calls>",
    "</calls>",
    "<result>",
    "</result>",
    "<param>",
    "<name>",
    "<value>",
    "</param>",
];

pub fn fold(state: u64, token: u32) -> u64 {
    state.wrapping_mul(17).wrapping_add(u64::from(token))
}

pub fn logits(state: u64) -> Vec<u8> {
    let selected = b'a' as usize + (state % 3) as usize;
    (0..VOCAB)
        .flat_map(|i| if i == selected { 1.0_f32 } else { -1.0_f32 }.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn synthetic_row_is_bounded_and_readable() {
        let row = super::logits(super::fold(7, 3));
        assert_eq!(row.len(), super::VOCAB * 4);
        let values: Vec<_> = row
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values.iter().filter(|v| **v == 1.0).count(), 1);
        assert_eq!(values[b'c' as usize], 1.0);
    }
}

#[cfg(all(test, feature = "precision"))]
mod declaration_tests {
    use super::precision as model;
    use projection_spec::ProjectionKind;

    /// A declaration change is a reviewed contract change: update this pin
    /// with it, never to silence an unexpected difference.
    const DIGEST: &str = "c3b87b3181eb48a7c8af97960fee7e47883c362f5b5c004e958aadcd3d6f7b1f";

    #[test]
    fn the_declaration_is_valid_and_pinned() {
        let declaration =
            projection_spec::parse(model::DECLARATION, &model::geometry(), &model::registry())
                .unwrap();
        assert_eq!(declaration.digest(), DIGEST);
        let ids: Vec<_> = declaration.projections.iter().map(|p| &p.id).collect();
        assert_eq!(ids, ["projection", "switch", "lookup"]);
    }

    #[test]
    fn declared_shapes_follow_the_geometry() {
        let projections = &model::declaration().unwrap().projections;
        let projection = |id| projections.iter().find(|p| p.id == id).unwrap();
        let shape = |id| model::source_shape(projection(id)).unwrap();
        let (h, i) = (u64::from(model::HIDDEN), u64::from(model::INTERMEDIATE));
        assert_eq!(shape("projection"), [h, h]);
        assert_eq!(shape("switch"), [u64::from(model::EXPERTS), 2 * i, h]);
        assert_eq!(
            shape("lookup"),
            [model::VOCAB, u64::from(model::TABLE_WIDTH)]
        );
        assert!(matches!(
            projection("switch").kind,
            ProjectionKind::ExpertBank { .. }
        ));
        for p in projections {
            assert_eq!(model::operation_uses(&p.uses).unwrap(), model::uses());
        }
    }
}
