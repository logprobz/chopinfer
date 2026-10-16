fn main() {
    let records = model_mimo26_contract::capabilities::sm120::captured_schedules().unwrap();
    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}
