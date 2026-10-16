//! Print, as a JSON array, every stage symbol of glm5next's projection
//! inventory (`sm120::CATALOG`): the `--symbols` selection of kernel-lab's
//! prepared projection jobs (`crates/kernel-lab/PROJECTION_JOBS.md`) that
//! runs glm5next's offers.
use model_glm5next_contract::sm120::CATALOG;
use std::collections::BTreeSet;

fn main() -> Result<(), String> {
    let (offers, _, copies) = (CATALOG.projections)()?;
    let mut symbols = BTreeSet::new();
    for offer in &offers {
        symbols.extend(offer.capability().map_err(|e| e.to_string())?.stages);
    }
    symbols.extend(copies.iter().map(|c| c.stage().entry().symbol.clone()));
    println!(
        "{}",
        serde_json::to_string_pretty(&symbols).map_err(|e| e.to_string())?
    );
    Ok(())
}
