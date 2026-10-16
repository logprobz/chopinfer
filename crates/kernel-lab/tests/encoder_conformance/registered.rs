//! MiMo's own encoder row kernels (model-mimo26-contract `sm120::encoders::rows`),
//! run from the `mimo_encoder` cubin the bundle embeds, against their f64
//! oracles: the residual quantizer over F32 codebooks (every code equal to
//! the f64 nearest-code search, which no near tie of this data can flip) and
//! the 20-table embedding sum in F32 (the numerical contract's bound for
//! F32 output). Both also equal their contracts' oracles
//! (`own_ops::encoder`) bit for bit. Synthetic data.

use std::collections::BTreeMap;

use super::suite::Model;
use encoder_spec::components::StageOp;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::DType;
use kernel_core::multimodal_oracle as oracle;
use kernel_lab::launch::simple_config;
use kernel_lab::{LabResult, Session, check_output};
use model_mimo26_contract::{encoders::spec, own_ops::encoder as own, sm120::encoders::rows};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    buffer
        .copy_to_bytes(&mut host, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(host)
}

fn bytes<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&value| to(value)).collect()
}

/// F32 values in [-scale, scale] with all 24 bits populated.
fn seeded_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

/// The declaration's kernel, launched as the runtime plans it: one
/// CTA per row of its capacity, reading the live count on the device.
fn launch(
    session: &Session,
    module: &Module,
    kernel: &rows::RowKernel,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    module
        .function(&kernel.symbol)
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            simple_config(
                (kernel.capacity, 1, 1),
                (rows::THREADS, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?,
            arguments,
        )
        .map_err(|e| e.to_string())
}

fn rvq_f32(
    session: &Session,
    module: &Module,
    kernel: &rows::RowKernel,
    dim: usize,
    sizes: &[u32],
) -> LabResult<()> {
    let rows = 16;
    let mut offsets = vec![0_i32];
    for &size in sizes {
        offsets.push(offsets.last().unwrap() + size as i32);
    }
    // Codebooks shrink by stage, as a trained RVQ's residuals do.
    let codebooks = (0..*offsets.last().unwrap() as usize)
        .flat_map(|code| {
            let stage = offsets.iter().rposition(|&o| o as usize <= code).unwrap();
            seeded_f32(dim, 0x7710 + code as u64, 0.8_f32.powi(stage as i32))
        })
        .collect::<Vec<_>>();
    let input = seeded_f32(rows * dim, 0x7711, 1.5);
    let (input_bytes, book_bytes, offset_bytes) = (
        bytes(&input, f32::to_le_bytes),
        bytes(&codebooks, f32::to_le_bytes),
        bytes(&offsets, i32::to_le_bytes),
    );
    let input_buffer = upload(session, &input_bytes)?;
    let book_buffer = upload(session, &book_bytes)?;
    let offset_buffer = upload(session, &offset_bytes)?;
    let codes = upload(session, &vec![0xff; rows * 20 * 4])?;
    let live = upload(session, &(rows as u32).to_le_bytes())?;
    launch(
        session,
        module,
        kernel,
        &[
            KernelArgument::Buffer(&input_buffer),
            KernelArgument::Buffer(&book_buffer),
            KernelArgument::Buffer(&offset_buffer),
            KernelArgument::Buffer(&codes),
            KernelArgument::Buffer(&live),
        ],
    )?;
    let books = codebooks.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let tables = sizes
        .iter()
        .enumerate()
        .map(|(stage, &size)| &books[offsets[stage] as usize * dim..][..size as usize * dim])
        .collect::<Vec<_>>();
    let expected = oracle::rvq_nearest(
        &input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        rows,
        dim,
        &tables,
        &sizes.iter().map(|&x| x as usize).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let written = download(session, &codes, rows * 20 * 4)?;
    let actual = written
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if let Some(index) = actual.iter().zip(&expected.codes).position(|(a, b)| a != b) {
        return Err(format!(
            "RVQ code {index}: {} against {}",
            actual[index], expected.codes[index]
        ));
    }
    let op = own::Rvq {
        dim: dim as u32,
        stages: sizes.len() as u32,
        max_codes: sizes.iter().copied().max().unwrap_or(0),
        codebook_rows: sizes.iter().sum(),
    };
    let mut contract = vec![0xff; rows * 20 * 4];
    own::rvq(
        &op,
        &input_bytes,
        &book_bytes,
        &offset_bytes,
        &mut contract,
        rows as u32,
        rows as u32,
    )
    .map_err(|e| e.0.to_string())?;
    if written != contract {
        return Err("RVQ codes differ from the contract's oracle".into());
    }
    Ok(())
}

fn embedding_sum_f32(
    session: &Session,
    module: &Module,
    kernel: &rows::RowKernel,
    vocab: usize,
    width: usize,
) -> LabResult<()> {
    let (rows, tables) = (300, 20);
    let table = seeded_f32(tables * vocab * width, 0x7708, 0.5)
        .into_iter()
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    let codes = (0..rows * tables)
        .map(|i| ((i * 7919 + i / tables * 31) % vocab) as i32)
        .collect::<Vec<_>>();
    let (table_bytes, code_bytes) = (
        bytes(&table, u16::to_le_bytes),
        bytes(&codes, i32::to_le_bytes),
    );
    let table_buffer = upload(session, &table_bytes)?;
    let code_buffer = upload(session, &code_bytes)?;
    let output = upload(session, &vec![0xff; rows * width * 4])?;
    let live = upload(session, &(rows as u32).to_le_bytes())?;
    launch(
        session,
        module,
        kernel,
        &[
            KernelArgument::Buffer(&table_buffer),
            KernelArgument::Buffer(&code_buffer),
            KernelArgument::Buffer(&output),
            KernelArgument::Buffer(&live),
        ],
    )?;
    let decoded = table
        .iter()
        .map(|&x| f64::from(kernel_core::codec::bf16_to_f32(x)))
        .collect::<Vec<_>>();
    let refs = decoded.chunks_exact(vocab * width).collect::<Vec<_>>();
    let expected = oracle::multi_embedding_sum(
        &refs,
        &codes.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        vocab,
        width,
    )
    .map_err(|e| e.to_string())?;
    let written = download(session, &output, rows * width * 4)?;
    let op = own::EmbeddingSum {
        tables: tables as u32,
        vocab: vocab as u32,
        width: width as u32,
    };
    let mut contract = vec![0xff; rows * width * 4];
    own::embedding_sum(
        &op,
        &table_bytes,
        &code_bytes,
        &mut contract,
        rows as u32,
        rows as u32,
    )
    .map_err(|e| e.0.to_string())?;
    if written != contract {
        return Err("embedding sums differ from the contract's oracle".into());
    }
    let actual = written
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if actual.iter().any(|x| !x.is_finite()) {
        return Err("embedding sum: unwritten or non-finite output".into());
    }
    let mut receipt = BTreeMap::new();
    check_output(&actual, &expected, DType::F32, &mut receipt)
        .map_err(|e| format!("embedding sum: {e}; {receipt:?}"))
}

pub(crate) fn run(id: &str) {
    let mut count = 0;
    kernel_lab::gpu_test(1 << 30, |session| {
        let module = session.load_cubin(Model::Mimo.cubin(rows::CUBIN))?;
        for declaration in spec::declarations_flash().map_err(|e| e.0)? {
            for (stage, item) in declaration.stages.iter().enumerate() {
                if item.op.id() != id {
                    continue;
                }
                let StageOp::Registered(registered) = &item.op else {
                    return Err("expected registered stage".into());
                };
                let kernel = rows::stage_kernel(declaration, stage)
                    .map_err(|e| e.0)?
                    .ok_or("missing registered kernel")?;
                let value = registered.component.to_value();
                if id == super::suite::REGISTERED[0] {
                    let rvq: spec::Rvq20 =
                        serde_json::from_value(value).map_err(|e| e.to_string())?;
                    let dim = declaration.inputs().map_err(|e| e.0)?[stage].width()? as usize;
                    rvq_f32(session, &module, &kernel, dim, &rvq.params().codebooks)?;
                } else {
                    let sum: spec::Sum20 =
                        serde_json::from_value(value).map_err(|e| e.to_string())?;
                    embedding_sum_f32(
                        session,
                        &module,
                        &kernel,
                        sum.params().vocab as usize,
                        sum.params().width as usize,
                    )?;
                }
                count += 1;
            }
        }
        if count == 0 {
            return Err(format!("no declaration uses {id}"));
        }
        Ok(())
    })
    .expect(id);
}
