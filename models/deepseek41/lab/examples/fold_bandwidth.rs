#[path = "sm120/build_fold_bandwidth.rs"]
mod build;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    build::run()
}
