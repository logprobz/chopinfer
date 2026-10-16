fn main() {
    let entries = model_mimo26_contract::capabilities::sm120::adoptions().unwrap();
    println!("{}", serde_json::to_string_pretty(&entries).unwrap());
}
