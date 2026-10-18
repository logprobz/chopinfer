//! Print the capacity table of every compiled geometry at its served ranks
//! for each checkpoint quantization layout named on the command line, or
//! for the representative layouts without one.
use model_glm5next_contract::capacity::{DEVICE, Planner};
use model_glm5next_contract::geometry::GEOMETRIES;
use model_glm5next_contract::kernels::SERVED_RANKS;
use model_glm5next_contract::quantization::{Layout, representative_layouts};

fn main() -> Result<(), String> {
    let named = std::env::args()
        .skip(1)
        .map(|id| Layout::named(&id.into()))
        .collect::<Result<Vec<_>, _>>()?;
    let layouts = if named.is_empty() {
        representative_layouts()
    } else {
        named
    };
    for geometry in GEOMETRIES {
        for layout in &layouts {
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
