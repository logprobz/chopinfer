//! Print the capacity table of every compiled geometry and checkpoint
//! quantization layout at its served ranks.
use model_glm5next_contract::capacity::{DEVICE, Planner};
use model_glm5next_contract::geometry::GEOMETRIES;
use model_glm5next_contract::kernels::SERVED_RANKS;
use model_glm5next_contract::quantization::LAYOUTS;

fn main() -> Result<(), String> {
    for geometry in GEOMETRIES {
        for layout in LAYOUTS {
            for &ranks in SERVED_RANKS {
                println!(
                    "{}",
                    Planner::for_checkpoint(geometry, layout, ranks)?.table(DEVICE)
                );
            }
        }
    }
    Ok(())
}
