//! Poisoned comparisons of dense fold sequences with the same launch ABI.
use kernel_core::{backend::Launch, contract::DType};
use kernel_harness::{LabResult, Session, launch, packed_fold::Image, poison};
use runtime_cuda::{Function, Module};
use std::collections::BTreeMap;

pub struct ReplayShape {
    pub dtype: DType,
    pub rows: u32,
    pub header: usize,
}

pub fn compare(
    session: &Session,
    module: &Module,
    producer: &poison::LiveProducer,
    calls: [&[Launch]; 2],
    images: &[Image],
    shape: ReplayShape,
) -> LabResult<()> {
    let ReplayShape {
        dtype,
        rows,
        header,
    } = shape;
    let buffers = images
        .iter()
        .map(|image| {
            let bytes = image.packed.as_ref().unwrap_or(&image.bytes);
            // The PDL rewrite producer copies whole u32 words. Preserve the
            // logical scale extent and guard its physical alignment padding.
            let allocated = if image.handle == 2 {
                bytes.len().next_multiple_of(4)
            } else {
                bytes.len().max(1)
            };
            let buffer = session.allocate(allocated)?;
            poison::write(session, &buffer, 0, bytes)?;
            if allocated > bytes.len() {
                poison::write(
                    session,
                    &buffer,
                    bytes.len(),
                    &vec![0x5a; allocated - bytes.len()],
                )?;
            }
            Ok(buffer)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let bindings = launch::Bindings {
        buffers: images
            .iter()
            .zip(&buffers)
            .map(|(i, b)| (i.handle, b))
            .collect(),
    };
    let buffer = |handle| {
        bindings
            .buffers
            .get(&handle)
            .copied()
            .ok_or_else(|| format!("missing twin view {handle}"))
    };
    let functions = calls
        .iter()
        .map(|calls| {
            calls
                .iter()
                .map(|call| module.function(&call.symbol).map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()
        })
        .collect::<LabResult<Vec<_>>>()?;
    let issue = |arm: usize| -> LabResult<()> {
        for (call, function) in calls[arm].iter().zip(&functions[arm]) {
            launch::launch(
                &session.context,
                function,
                &session.stream,
                call.plan,
                &call.arguments,
                &bindings,
            )?;
        }
        Ok(())
    };
    let live = images
        .iter()
        .find(|image| image.handle == 6)
        .ok_or("missing live count")?;
    let live = u32::from_le_bytes(
        live.bytes
            .as_slice()
            .try_into()
            .map_err(|_| "invalid live count")?,
    );
    if live > rows {
        return Err("live rows exceed the twin capacity".into());
    }
    let mut replay = poison::Replay::new(session);
    for handle in [5, 10] {
        if let Some(buffer) = bindings.buffers.get(&handle) {
            replay = replay.output(
                format!("output{handle}"),
                buffer,
                poison::WriteMap::of(dtype)
                    .byte_prefix(buffer.byte_len() / rows as usize * live as usize)?,
            );
        }
    }
    let status = images
        .iter()
        .find(|i| i.handle == 7)
        .ok_or("missing status image")?;
    replay = replay.state(
        "status",
        buffer(7)?,
        status.bytes.clone(),
        poison::WriteMap::of(DType::I32).none()?,
        std::iter::once(12..12 + live as usize * 4).collect(),
    );
    // Admission rewrites row flags. Only partials are poisoned after the
    // header: ticket counters keep their initialized/reset state.
    replay = replay.scratch_ranges(
        "workspace",
        buffer(11)?,
        vec![0..rows as usize * 4, header..buffer(11)?.byte_len()],
    );
    replay.eager(|| issue(0))?;
    let expected_workspace = poison::read(session, buffer(11)?)?;
    let check_workspace = || -> LabResult<()> {
        let actual = poison::read(session, buffer(11)?)?;
        if actual != expected_workspace {
            let byte = actual
                .iter()
                .zip(&expected_workspace)
                .position(|(a, b)| a != b)
                .expect("unequal workspace bytes");
            return Err(format!("workspace differs at byte {byte}"));
        }
        Ok(())
    };
    replay.relaunch(|| issue(0))?;
    check_workspace()?;
    replay.relaunch(|| issue(1))?;
    check_workspace()?;
    let graph = session.capture(|_| issue(1))?;
    replay.graph(&graph)?;
    check_workspace()?;
    let mut fresh = Vec::new();
    for handle in [1, 2] {
        let bytes = &images
            .iter()
            .find(|i| i.handle == handle)
            .ok_or("missing activation image")?
            .bytes;
        let allocated = buffer(handle)?.byte_len();
        let copy = session.allocate(allocated)?;
        poison::write(session, &copy, 0, bytes)?;
        if allocated > bytes.len() {
            poison::write(
                session,
                &copy,
                bytes.len(),
                &vec![0x5a; allocated - bytes.len()],
            )?;
        }
        fresh.push(copy);
    }
    let rewrites =
        poison::Rewrites::new(session, &[(&fresh[0], buffer(1)?), (&fresh[1], buffer(2)?)])?;
    replay.pdl(
        || producer.launch_rewriting(session, buffer(6)?, &rewrites),
        || issue(1),
        true,
    )?;
    check_workspace()?;
    for handle in [1, 2, 3, 4, 8, 9] {
        if let Some(image) = images.iter().find(|i| i.handle == handle) {
            let actual = poison::read(session, buffer(handle)?)?;
            let expected = image.packed.as_ref().unwrap_or(&image.bytes);
            if actual[..expected.len()] != *expected
                || actual[expected.len()..].iter().any(|&b| b != 0x5a)
            {
                return Err(format!(
                    "twin wrote input view {handle} or its alignment guard"
                ));
            }
        }
    }
    Ok(())
}

pub fn producer_spec() -> kernel_build::CubinSpec {
    use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let root = kernels_sm120::probes::cubin_spec()
        .sources_root
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    CubinSpec {
        name:"dense_decode_pdl_probe".into(),sources_root:root,include_dirs:vec![],
        headers:vec!["#include \"kernel-lab/cuda/probes/pdl_prefetch.cuh\"".into()],defines:vec![],
        entries:vec![EntrySpec::new("chop_dense_decode_producer_v1".into(),vec![
            ptr("eviction","const uint4"),ptr("checksum","uint32_t"),P::U64{name:"vectors".into()},
            ptr("operands","const chop::probe::PdlOperand"),P::U32{name:"operand_count".into()},ptr("live","uint32_t"),P::U32{name:"rows".into()},P::U64{name:"tail_ns".into()}
        ],"chop::probe::pdl_prefetch_producer(eviction,checksum,vectors,operands,operand_count,live,rows,tail_ns);".into(),128,1)],
    }
}

pub fn build(specs: &[kernel_build::CubinSpec], directory: &std::path::Path) -> LabResult<()> {
    use kernel_build::{BuildOptions, LockDir, Toolchain};
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let directory = directory.canonicalize().map_err(|e| e.to_string())?;
    let root = kernels_sm120::probes::cubin_spec()
        .sources_root
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if directory.starts_with(root) {
        return Err("twin artifacts must stay outside the repository".into());
    }
    let locks = directory.join("locks");
    let lock =
        LockDir::new(&locks).family("dense_decode_twins", specs.iter().map(|s| s.name.as_str()));
    let initialize = !locks.join("dense_decode_twins.lock").exists();
    let options = BuildOptions {
        lock_update: initialize,
        ..BuildOptions::default()
    };
    let built = kernel_build::build_in_families(
        &Toolchain::detect().map_err(|e| e.to_string())?,
        specs,
        &directory,
        &lock,
        options,
        None,
    )
    .map_err(|e| e.to_string())?;
    if built.len() != specs.len() || built.is_empty() {
        return Err("twin build omitted cubins".into());
    }
    let counts: BTreeMap<_, _> = specs
        .iter()
        .map(|spec| (spec.name.clone(), spec.entries.len()))
        .collect();
    println!("dense_decode built {counts:?}");
    Ok(())
}

pub fn load(session: &Session, directory: &std::path::Path, cubin: &str) -> LabResult<Module> {
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    session.load_cubin(
        &std::fs::read(directory.join(format!("{cubin}.cubin"))).map_err(|e| e.to_string())?,
    )
}

pub fn producer(module: &Module) -> LabResult<Function> {
    module
        .function("chop_dense_decode_producer_v1")
        .map_err(|e| e.to_string())
}
