//! A local Markov recurrence uses each prior GPU winner as the next lookup
//! input. Full TP4 winner reduction remains a served-gate obligation.
use super::*;
use kernel_core::draft_noise::{DraftNoise, DraftNoiseBuffers, NoiseRecord};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{
    ConfidenceHead, ConfidenceHeadBuffers, MarkovStep, MarkovStepBuffers,
};

pub(super) fn compare(
    s: &Session,
    items: u32,
    hidden: &[u32],
    full: &[u32],
    packed: &[u32],
) -> LabResult<usize> {
    let k = FLASH.shape.hidden as usize;
    let n = (FLASH.shape.vocab / 4) as usize;
    let dim = FLASH.shape.dspark.markov_rank as usize;
    let own = crate::own_ops::Sm120Own::new(&FLASH);
    let markov = own
        .dspark_markov_step(
            &MarkovStep {
                rank: dim as u32,
                vocab_rows: n as u32,
                logits_stride: n as u32,
            },
            Need::rows(8),
        )
        .map_err(|e| e.to_string())?;
    let confidence = own
        .dspark_confidence(
            &ConfidenceHead {
                hidden: k as u32,
                markov_rank: dim as u32,
                hidden_stride: 5 * k as u32,
                score_stride: 5,
            },
            Need::rows(8),
        )
        .map_err(|e| e.to_string())?;
    let noise = Sm120::new()
        .with_draft_noise(sm120::dspark_noise::entry())
        .draft_noise(
            &DraftNoise {
                vocab_per_rank: n as u32,
                ranks: 4,
            },
            0,
            Need::rows(8),
        )
        .map_err(|e| e.to_string())?;
    let embedding = s.allocate(items as usize * dim * 2)?;
    let bank: Vec<_> = (0..17 * dim)
        .map(|i| f32_to_bf16(((i * 19) % 257) as f32 / 513. - 0.25))
        .collect();
    let mut w = Vec::with_capacity(n * dim * 2);
    for col in 0..n {
        w.extend(
            bank[col % 17 * dim..][..dim]
                .iter()
                .flat_map(|b| b.to_le_bytes()),
        );
    }
    let weight = upload(s, &w)?;
    let confidence_weight = upload(
        s,
        &(0..k + dim)
            .flat_map(|i| f32_to_bf16(((i * 7) % 17) as f32 / 31. - 0.25).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let base = s.allocate(items as usize * n * 4)?;
    let output = s.allocate((items as usize + 1) * n * 4)?;
    let x = s.allocate(items as usize * 5 * k * 4)?;
    let scores = s.allocate((items as usize * 5 + 1) * 4)?;
    let live = upload(s, &items.to_le_bytes())?;
    let width = upload(s, &(n as u32).to_le_bytes())?;
    let records = s.allocate(items as usize * 32)?;
    let bindings = Bindings {
        buffers: BTreeMap::from([
            (1, &embedding),
            (2, &weight),
            (3, &base),
            (4, &output),
            (5, &x),
            (6, &confidence_weight),
            (7, &scores),
            (8, &live),
            (9, &width),
            (10, &records),
        ]),
    };
    let view = |handle| View {
        handle,
        bytes: bindings.buffers[&handle].byte_len() as u64,
    };
    let mut core = markov
        .launches(
            &MarkovStepBuffers {
                embedding: view(1),
                weight: view(2),
                base: view(3),
                output: view(4),
                live_rows: view(8),
            },
            View::EMPTY,
            Counts::rows(items),
        )
        .map_err(|e| e.to_string())?;
    core.extend(
        confidence
            .launches(
                &ConfidenceHeadBuffers {
                    hidden: view(5),
                    markov: view(1),
                    weight: view(6),
                    scores: view(7),
                    live_rows: view(8),
                },
                View::EMPTY,
                Counts::rows(items),
            )
            .map_err(|e| e.to_string())?,
    );
    let noise_plans = (0..5)
        .map(|position| {
            noise
                .launches(
                    &DraftNoiseBuffers {
                        logits: view(4),
                        live_width: view(9),
                        noise: view(10),
                        live_rows: view(8),
                        position,
                    },
                    View::EMPTY,
                    Counts::rows(items),
                )
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let all: Vec<_> = core
        .iter()
        .chain(noise_plans.iter().flatten())
        .cloned()
        .collect();
    let modules = modules(s, &all)?;
    let graphs = noise_plans
        .iter()
        .map(|noise| {
            s.capture(|stream| {
                run(s, &modules, &core, &bindings, stream)?;
                run(s, &modules, noise, &bindings, stream)
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    for sampled in [false, true] {
        let records_bytes: Vec<_> = (0..items)
            .flat_map(|i| {
                NoiseRecord {
                    seed: 0x1728_0000 + u64::from(i),
                    generated: 0xffff_fffe + u64::from(i),
                    temperature: if sampled { 1. } else { 0. },
                }
                .encode()
            })
            .collect();
        records
            .copy_from_bytes(&records_bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        let mut outputs = Vec::new();
        for depth in [5, 3] {
            let mut previous = vec![7_u32; items as usize];
            let mut sequence = Vec::new();
            for position in 0..depth {
                let e: Vec<_> = previous
                    .iter()
                    .flat_map(|&id| {
                        (0..dim).flat_map(move |j| {
                            f32_to_bf16(((id as usize * 11 + j * 3) % 257) as f32 / 257. - 0.5)
                                .to_le_bytes()
                        })
                    })
                    .collect();
                embedding
                    .copy_from_bytes(&e, &s.stream)
                    .map_err(|e| e.to_string())?;
                let b: Vec<_> = (0..items as usize)
                    .flat_map(|i| {
                        let row = if depth == 5 {
                            5 * i + position
                        } else {
                            position * items as usize + i
                        };
                        let source = if depth == 5 { full } else { packed };
                        source[row * n..][..n].iter().copied()
                    })
                    .collect();
                base.copy_from_bytes(&words(b), &s.stream)
                    .map_err(|e| e.to_string())?;
                // Keep the original five-row hidden stride at each position.
                let mut h = vec![POISON; items as usize * 5 * k];
                for i in 0..items as usize {
                    h[i * 5 * k..][..k].copy_from_slice(&hidden[(i * 5 + position) * k..][..k]);
                }
                x.copy_from_bytes(&words(h), &s.stream)
                    .map_err(|e| e.to_string())?;
                output
                    .copy_from_bytes(
                        &words(std::iter::repeat_n(POISON, output.byte_len() / 4)),
                        &s.stream,
                    )
                    .map_err(|e| e.to_string())?;
                scores
                    .copy_from_bytes(&vec![0; scores.byte_len()], &s.stream)
                    .map_err(|e| e.to_string())?;
                graphs[position]
                    .launch(&s.stream)
                    .map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let logits = read(s, &output)?;
                let score = read(s, &scores)?;
                if logits[items as usize * n..].iter().any(|&v| v != POISON)
                    || score
                        .iter()
                        .enumerate()
                        .any(|(i, &v)| (i % 5 != 0 || i >= items as usize * 5) && v != 0)
                {
                    return Err("used-depth position wrote an unused slot".into());
                }
                for (i, id) in previous.iter_mut().enumerate() {
                    let row = &logits[i * n..][..n];
                    *id = row.iter().enumerate().fold(0, |best, (col, &value)| {
                        if f32::from_bits(value) > f32::from_bits(row[best]) {
                            col
                        } else {
                            best
                        }
                    }) as u32;
                }
                sequence.push((
                    logits[..items as usize * n].to_vec(),
                    previous.clone(),
                    (0..items as usize)
                        .map(|i| score[i * 5])
                        .collect::<Vec<_>>(),
                ));
            }
            outputs.push(sequence);
        }
        if outputs[0][..3] != outputs[1] {
            return Err(format!(
                "items={items} sampled={sampled}: used Markov logits, IDs or confidence differ"
            ));
        }
    }
    Ok(2)
}
