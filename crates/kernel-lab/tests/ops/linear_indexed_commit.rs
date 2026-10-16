//! Physical checkpoint ownership against the checkpoint-copy path.
use super::*;

fn decode_step(
    session: &Session,
    module: &Module,
    kind: &VerifyKind,
    op: &mut Operands,
    bank: &[Vec<u16>],
    row: usize,
    slot: i32,
) -> LabResult<Vec<u8>> {
    for (buffer, (name, stride, _)) in bank.iter().zip(&kind.rows) {
        op.bf16(session, name, &buffer[row * stride..(row + 1) * stride])?;
    }
    op.i32(session, "slot_ids", &[slot])?;
    op.u32(session, "live_sequences", 1)?;
    op.zero(session, "output", kind.output_width * 2)?;
    let kernel =
        Kernel::load(module, &symbol(&kind.entry.replace("verify", "decode")))?.with_bucket(1, 1);
    let initial = op.read(session, "state", op.buffers["state"].byte_len())?;
    let mut checks = Replay::new(session)
        .output(
            "output",
            &op.buffers["output"],
            WriteMap::of(DType::Bf16).prefix(kind.output_width)?,
        )
        .state(
            "state",
            &op.buffers["state"],
            initial,
            WriteMap::of(DType::F32).byte_ranges(Vec::new())?,
            std::iter::once(
                slot as usize * kind.entry_bytes..(slot as usize + 1) * kind.entry_bytes,
            )
            .collect(),
        );
    checks.eager(|| kernel.launch(session, &session.stream, op, 1, 1, true))?;
    checks.relaunch(|| kernel.launch(session, &session.stream, op, 1, 1, false))?;
    let graph = session.capture(|stream| kernel.launch(session, stream, op, 1, 1, true))?;
    checks.graph(&graph)?;
    checks.graph(&graph)?;
    op.read(session, "output", kind.output_width * 2)
}

fn compare(
    session: &Session,
    module: &Module,
    kind: &VerifyKind,
    op: &mut Operands,
    initial: &[u8],
    rows: usize,
    kept: usize,
) -> LabResult<()> {
    let mut rng = Rng::new(0x9130 + rows as u64 * 16 + kept as u64);
    let bank: Vec<_> = kind
        .rows
        .iter()
        .map(|&(_, stride, scale)| rng.bf16((rows + 1) * stride, scale))
        .collect();
    let indices: Vec<i32> = (1..=rows as i32).collect();
    let step = |slot, checkpoints: Vec<i32>| StepItem {
        slot,
        rows: (0..rows).collect(),
        checkpoints,
    };
    op.bytes(session, "state", initial)?;
    let output = verify_step(
        session,
        module,
        kind,
        op,
        &bank,
        &[step(0, indices.clone())],
        (rows + 1) as i32,
    )?;
    let after = op.read(session, "state", initial.len())?;
    // No commit, then repeat verification with poisoned write-only outputs.
    let repeat = verify_step(
        session,
        module,
        kind,
        op,
        &bank,
        &[step(0, indices.clone())],
        (rows + 1) as i32,
    )?;
    if output != repeat || op.read(session, "state", initial.len())? != after {
        return Err("repeat without commit changed output or checkpoint bits".into());
    }
    if after[..kind.entry_bytes] != initial[..kind.entry_bytes] {
        return Err("verify/cancel changed committed state".into());
    }
    let committed = if kept == 0 { 0 } else { indices[kept - 1] };
    let at = committed as usize * kind.entry_bytes;
    let captured = after[at..at + kind.entry_bytes].to_vec();
    let mut copied = after.clone();
    copied[..kind.entry_bytes].copy_from_slice(&captured);
    op.bytes(session, "state", &copied)?;
    let copy_output = decode_step(session, module, kind, op, &bank, rows, 0)?;
    let copy_state = op.read(session, "state", kind.entry_bytes)?;

    // Index commit returns the old committed entry to the verify workspace.
    let mut workspace = indices;
    if kept != 0 {
        workspace[kept - 1] = 0;
    }
    op.bytes(session, "state", &after)?;
    let index_output = decode_step(session, module, kind, op, &bank, rows, committed)?;
    let indexed = op.read(session, "state", initial.len())?;
    if copy_output != index_output || copy_state != indexed[at..at + kind.entry_bytes] {
        return Err("decode after index commit differs from checkpoint copy".into());
    }
    // A captured committed entry restored into another physical slot must
    // give exactly the continuation of the checkpoint-copy path.
    let restore_at = (rows + 1) * kind.entry_bytes;
    let mut restored = after.clone();
    restored[restore_at..restore_at + kind.entry_bytes].copy_from_slice(&captured);
    op.bytes(session, "state", &restored)?;
    let restored_output = decode_step(session, module, kind, op, &bank, rows, (rows + 1) as i32)?;
    let restored = op.read(session, "state", initial.len())?;
    if restored_output != copy_output
        || restored[restore_at..restore_at + kind.entry_bytes] != copy_state
    {
        return Err("captured committed entry restored with different continuation bits".into());
    }
    // Recycling the old source as a destination must preserve the newly
    // committed entry and all outputs of the next verification.
    op.bytes(session, "state", &indexed)?;
    let indexed_output = verify_step(
        session,
        module,
        kind,
        op,
        &bank,
        &[step(committed, workspace.clone())],
        (rows + 1) as i32,
    )?;
    let index_after = op.read(session, "state", initial.len())?;
    op.bytes(session, "state", &copied)?;
    decode_step(session, module, kind, op, &bank, rows, 0)?;
    let copy_verify = verify_step(
        session,
        module,
        kind,
        op,
        &bank,
        &[step(0, (1..=rows as i32).collect())],
        (rows + 1) as i32,
    )?;
    let copy_after = op.read(session, "state", initial.len())?;
    if indexed_output != copy_verify || index_after[at..at + kind.entry_bytes] != copy_state {
        return Err("recycled checkpoint pool changed committed state or output".into());
    }
    for (row, &entry) in workspace.iter().enumerate() {
        let indexed_at = entry as usize * kind.entry_bytes;
        let copied_at = (row + 1) * kind.entry_bytes;
        if index_after[indexed_at..indexed_at + kind.entry_bytes]
            != copy_after[copied_at..copied_at + kind.entry_bytes]
        {
            return Err("recycled checkpoint state differs bitwise".into());
        }
    }
    Ok(())
}

#[test]
fn commit_indices_match_checkpoint_copies_bitwise() {
    gpu_test(1 << 30, |session| {
        let module = module(session)?;
        let g = geometry();
        let mut report = Report::new("commit_indices_all_accepts");
        for rows in 1..=verify_row_limit() as usize {
            for kept in 0..=rows {
                let mut rng = Rng::new(0x8100 + rows as u64);
                let conv = ConvCase::new(&mut rng, rows + 1, rows + 2);
                let mut op = conv.operands(session, rows + 1)?;
                report.exact(
                    &format!("conv R{rows} accept{kept}"),
                    compare(
                        session,
                        &module,
                        &VerifyKind {
                            entry: "conv_verify",
                            rows: vec![("input", g.qkv_stride, 1.0)],
                            output_width: g.channels,
                            entry_bytes: HISTORY * g.channels * 2,
                        },
                        &mut op,
                        &bytes_u16(&conv.pool),
                        rows,
                        kept,
                    ),
                );
                let delta = DeltaCase::new(&mut rng, rows + 1, rows + 2);
                let mut op = delta.operands(session, rows + 1)?;
                report.exact(
                    &format!("delta R{rows} accept{kept}"),
                    compare(
                        session,
                        &module,
                        &VerifyKind {
                            entry: "delta_verify",
                            rows: vec![
                                ("qkv", g.qkv_stride, 1.0),
                                ("a", g.gate_stride, 1.5),
                                ("b", g.gate_stride, 1.5),
                            ],
                            output_width: g.norm_width,
                            entry_bytes: g.delta_state * 4,
                        },
                        &mut op,
                        &bytes_f32(&delta.pool),
                        rows,
                        kept,
                    ),
                );
            }
        }
        report.finish(session)
    })
    .expect("index commits preserve state and output bits for every accept count");
}
