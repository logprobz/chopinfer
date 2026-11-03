//! Reduction-input taps on the CPU fake host: every rank's F32 partial of
//! a sublayer, read on its own stream before the sublayer's all-reduce.
use super::*;

/// Rows of the tapped prompt.
const ROWS: u64 = 5;

/// A prefill's attention and MoE partials of layer 0 are written for every
/// rank, each read on its rank's stream after the projection or routed sum
/// that completes it and before the all-reduce's first launch, and the
/// launches stay those of an untapped prefill.
#[test]
fn partial_taps_read_every_rank_before_the_all_reduce() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    if std::env::var_os(TAPS_CHILD).is_some() {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        let directory = std::env::var_os(runtime_bundle_sdk::taps::TAPS_ENV.dir).unwrap();
        let start = host.device.journal.borrow().len();
        instance
            .execute(&prompt(7, 0, 0, ROWS, true), &mut Views::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        let lines = host.device.journal.borrow()[start..].to_vec();
        std::fs::write(
            std::path::Path::new(&directory).join("journal.json"),
            serde_json::to_vec(&lines).unwrap(),
        )
        .unwrap();
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt(7, 0, 0, ROWS, true), &mut Views::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let untapped = launch_lines(&host, start);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let directory =
        std::env::temp_dir().join(format!("deepseek41-partial-taps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    in_child(
        "fake_host_tests::reduce_inputs::partial_taps_read_every_rank_before_the_all_reduce",
        &directory,
        &[
            (
                "CHOPINFER_TAPS_SITES",
                "L00.x.latent,L00.x.*.partial,L00.attn.*,L00.mlp.*",
            ),
            ("CHOPINFER_TAPS_LAYERS", "0"),
        ],
    );
    let step = directory.join("step0000");
    let index = std::fs::read_to_string(step.join("index.jsonl")).unwrap();
    let written: Vec<(String, u64)> = index
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .map(|entry| {
            (
                entry["name"].as_str().unwrap().to_owned(),
                entry["rank"].as_u64().unwrap(),
            )
        })
        .collect();
    let every = |name: &'static str| (0..4).map(move |rank| (name.to_owned(), rank));
    let expected: Vec<(String, u64)> = [("L00.attn.in".to_owned(), 0), ("L00.x.latent".into(), 0)]
        .into_iter()
        .chain(every("L00.x.attn.partial"))
        .chain([("L00.attn.out".into(), 0), ("L00.mlp.in".into(), 0)])
        .chain(every("L00.x.mlp.partial"))
        .chain([("L00.mlp.out".into(), 0)])
        .collect();
    assert_eq!(written, expected);
    let hidden = u64::from(shape.dims().hidden);
    for rank in 0..4 {
        for sublayer in ["attn", "mlp"] {
            let file = step.join(format!("L00.x.{sublayer}.partial.r{rank}.bin"));
            assert_eq!(
                std::fs::metadata(file).unwrap().len(),
                ROWS * hidden * 4,
                "{sublayer} rank {rank}"
            );
        }
    }
    let completion: serde_json::Value =
        serde_json::from_slice(&std::fs::read(step.join("completion.json")).unwrap()).unwrap();
    assert_eq!(completion["completion"], "full", "{completion}");
    let lines: Vec<String> =
        serde_json::from_slice(&std::fs::read(directory.join("journal.json")).unwrap()).unwrap();
    let tapped = {
        let mut streams = BTreeMap::<String, Vec<String>>::new();
        for line in lines.iter().filter_map(|line| line.strip_prefix("launch ")) {
            let stream = line.split(' ').next().unwrap_or_default().to_owned();
            streams
                .entry(stream)
                .or_default()
                .push(format!("launch {line}"));
        }
        streams
    };
    assert_eq!(tapped, untapped);
    // Each partial download, F32 rows of the hidden width, sits on its
    // rank's stream between the launch that completes the partial (the
    // attention's output projection, the MoE's routed sum) and the
    // all-reduce's first launch, which under the native Flash TP4
    // BF16-partial contract rounds the partial in place.
    let partial = format!("length: {}", ROWS * hidden * 4);
    let symbol = |line: &str| line.split(' ').nth(2).unwrap_or_default().to_owned();
    let mut seen = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("download ") else {
            continue;
        };
        if !line.contains(&partial) {
            continue;
        }
        let stream = rest.split(' ').next().unwrap();
        let on_stream = |line: &&String| {
            line.strip_prefix("launch ")
                .is_some_and(|rest| rest.split(' ').next() == Some(stream))
        };
        let before = symbol(lines[..at].iter().rev().find(on_stream).unwrap());
        let after = symbol(lines[at + 1..].iter().find(on_stream).unwrap());
        let producer = if seen.len() < 4 {
            "chop_deepseek41_dense_"
        } else {
            "chop_moe_deepseek41_sum_add_f32_"
        };
        assert!(before.starts_with(producer), "{before}");
        assert_eq!(after, "chop_comm_allreduce_phased_bf16_v1");
        seen.push(stream.to_owned());
    }
    // Every rank's stream once per sublayer, attention first.
    assert_eq!(seen.len(), 8);
    assert_eq!(seen[..4], seen[4..]);
    let mut streams = seen[..4].to_vec();
    streams.dedup();
    assert_eq!(streams.len(), 4);
    std::fs::remove_dir_all(&directory).unwrap();
}
