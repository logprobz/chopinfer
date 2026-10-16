//! The kit's case self-checks (`kit::cases::<family>::checks`) over the
//! lab's catalogs and lab shapes: each SM120 selection's cases map rows,
//! planes and bounds as the kit requires. CPU only.

use kernel_lab::kit::cases::{attention, latent, recurrent, rope_kv};

fn variants() -> Vec<kernels_sm120::attention::Variant> {
    attention::variants(kernel_lab::catalogs())
}

#[test]
fn attention_exact_verify_rows_map_to_their_decode_launches() {
    assert_eq!(
        attention::checks::exact_verify_rows_map_to_their_decode_launches(&variants()),
        2,
        "paged and ring exact rows"
    );
}

#[test]
fn attention_folds_cover_every_fold_op() {
    let cases = attention::checks::fold_cases(&variants());
    assert!(cases.len() >= 8, "{} fold ops", cases.len());
}

#[test]
fn attention_a_fold_bound_refuses_zero_outputs() {
    attention::checks::a_fold_bound_refuses_zero_outputs(&variants());
}

#[test]
fn attention_a_fold_key_may_round_past_a_binade() {
    attention::checks::a_fold_key_may_round_past_a_binade(&variants());
}

#[test]
fn attention_fold_plane_maps_follow_moved_items() {
    attention::checks::fold_plane_maps_follow_moved_items(&variants());
}

#[test]
fn attention_a_fold_attends_over_the_values_it_stores() {
    attention::checks::a_fold_attends_over_the_values_it_stores(&variants());
}

#[test]
fn latent_written_pool_elements_never_read_as_poison() {
    let entries = latent::sm120_entries(kernel_lab::latent_shapes::catalog());
    latent::checks::written_pool_elements_never_read_as_poison(&entries);
}

#[test]
fn recurrent_verify_rows_map_to_their_decode_steps() {
    let entries = recurrent::sm120_entries(kernel_lab::catalogs());
    assert_eq!(
        recurrent::checks::verify_rows_map_to_their_decode_steps(&entries),
        4,
        "conv and delta verify ops"
    );
}

#[test]
fn recurrent_a_verify_step_that_corrupts_its_slot_fails_the_mapping() {
    let entries = recurrent::sm120_entries(kernel_lab::catalogs());
    assert_eq!(
        recurrent::checks::a_verify_step_that_corrupts_its_slot_fails_the_mapping(&entries),
        4
    );
}

#[test]
fn recurrent_read_slots_are_rewritten_not_poisoned() {
    let entries = recurrent::sm120_entries(kernel_lab::catalogs());
    recurrent::checks::read_slots_are_rewritten_not_poisoned(&entries);
}

#[test]
fn rope_kv_plane_maps_follow_moved_rows() {
    let entries = rope_kv::sm120_entries(kernel_lab::catalogs());
    rope_kv::checks::plane_maps_follow_moved_rows(&entries);
}

#[test]
fn rope_kv_ring_planes_follow_the_ring_kv_write_oracle() {
    let entries = rope_kv::sm120_entries(kernel_lab::catalogs());
    rope_kv::checks::ring_planes_follow_the_ring_kv_write_oracle(&entries);
}

#[test]
fn rope_kv_every_computed_rotation_has_an_angle_case() {
    let entries = rope_kv::sm120_entries(kernel_lab::catalogs());
    rope_kv::checks::every_computed_rotation_has_an_angle_case(&entries);
}
