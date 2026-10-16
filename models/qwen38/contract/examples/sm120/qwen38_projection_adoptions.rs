fn main() {
    let entries = model_qwen38_contract::sm120::generated::adoptions().unwrap();
    println!("{}", serde_json::to_string_pretty(&entries).unwrap());
}
