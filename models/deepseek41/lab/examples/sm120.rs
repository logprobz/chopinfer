#[path = "sm120/build_exact_moe_prefill.rs"]
mod sm120;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    sm120::run()
}
