//! Replay the exact served attention boundary from an authenticated operand capture.
use super::*;
use kernel_build::EntryParam;
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList};
use kernels_sm120::latent::sources_cached;

#[derive(Clone)]
pub(super) struct Captured {
    pub(super) label: String,
    pub(super) source: Entry,
    pub(super) rows: u32,
    pub(super) cached: bool,
    pub(super) scalars: Value,
    pub(super) data: BTreeMap<String, Vec<u8>>,
    pub(super) expected: [Vec<u8>; 2],
}
fn read_blob(root: &std::path::Path, value: &Value) -> LabResult<Vec<u8>> {
    let sha = value["sha256"].as_str().ok_or("blob digest absent")?;
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid blob digest".into());
    }
    let file = format!("blobs/{sha}.bin");
    if value["file"] != file {
        return Err("blob path does not match its digest".into());
    }
    let path = root.join(file);
    if !std::fs::symlink_metadata(&path)
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err("blob is not a regular file".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if value["bytes"].as_u64() != Some(bytes.len() as u64) || kernel_build::digest(&bytes) != sha {
        return Err("captured blob hash or length differs".into());
    }
    Ok(bytes)
}
pub(super) fn captures() -> LabResult<Vec<Captured>> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_ATTENTION_CAPTURE_ROOT")
            .ok_or("CHOPINFER_ATTENTION_CAPTURE_ROOT is required")?,
    );
    if !root.is_absolute() {
        return Err("capture root must be absolute".into());
    }
    let index: Value = serde_json::from_slice(
        &std::fs::read(root.join("corpus.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if index["schema"] != "chopinfer.verify-attention-corpus.v1" || index["complete"] != true {
        return Err("incomplete attention corpus".into());
    }
    let entries = candidate::entries(geometry(), 4)?;
    let mut all = Vec::new();
    let mut labels = BTreeSet::new();
    for frame in index["captures"].as_array().ok_or("capture index absent")? {
        let file = frame["file"].as_str().ok_or("frame file absent")?;
        let parts = file.split('/').collect::<Vec<_>>();
        if parts.len() != 2
            || parts[1] != "capture.json"
            || parts[0].is_empty()
            || !parts[0]
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err("invalid capture frame path".into());
        }
        let path = root.join(file);
        if !std::fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .is_file()
        {
            return Err("frame is not regular".into());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if frame["sha256"] != kernel_build::digest(&bytes) {
            return Err("frame digest differs".into());
        }
        let capture: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if capture["schema"] != "chopinfer.verify-attention-capture.v1"
            || capture["rank"] != 0
            || capture["ranks"] != 4
            || capture["phase"] != "verify"
        {
            return Err("wrong attention capture provenance".into());
        }
        let mut layers = BTreeSet::new();
        for record in capture["records"]
            .as_array()
            .ok_or("frame records absent")?
        {
            let layer = record["layer"].as_u64().ok_or("layer absent")?;
            if !layers.insert(layer) {
                return Err("duplicate captured layer".into());
            }
            let name = record["source_name"].as_str().ok_or("source name absent")?;
            let source = entries
                .iter()
                .find(|e| e.name == name)
                .ok_or("unrecognized capture source")?
                .clone();
            let rows = u32::try_from(record["rows"].as_u64().ok_or("rows absent")?)
                .map_err(|e| e.to_string())?;
            if !(2..=32).contains(&rows)
                || record["live_rows"] != rows
                || record["numerical_splits"] != 8
                || source.splits != 8
            {
                return Err("unqualified capture rows or splits".into());
            }
            let cached_entry = sources_cached::Entry {
                source: source.clone(),
                profile: false,
            };
            let cached = record["served_first_symbol"] == cached_entry.decode_symbol(false);
            if record["served_first_symbol"] != source.symbol() && !cached {
                return Err("unknown served source route".into());
            }
            if record["served_body_symbol"]
                != if cached {
                    cached_entry.symbol()
                } else {
                    source.symbol()
                }
                || record["served_merge_symbol"] != source.merge_symbol()
            {
                return Err("served body/merge mismatch".into());
            }
            let mut data = BTreeMap::new();
            for (name, value) in record["buffers"]
                .as_object()
                .ok_or("captured buffers absent")?
            {
                data.insert(name.clone(), read_blob(&root, value)?);
            }
            let expected = [
                read_blob(&root, &record["expected_workspace"])?,
                read_blob(&root, &record["expected_output"])?,
            ];
            if expected[0].len() != source.workspace_bytes(rows) as usize
                || expected[1].len() != rows as usize * 16 * 512 * 2
                || data["q"].len() != rows as usize * 16 * 512 * 2
                || data["row_batch"].len() != rows as usize * 4
                || data["live_rows"] != rows.to_le_bytes()
            {
                return Err("capture boundary extent mismatch".into());
            }
            data.insert("workspace".into(), vec![0xa5; expected[0].len()]);
            data.insert("out".into(), vec![0xa5; expected[1].len()]);
            let (decoded, valid) = cached_entry.cache_bytes(rows).map_err(|e| e.to_string())?;
            data.insert("decoded".into(), vec![0xa5; decoded as usize]);
            data.insert("cached_valid".into(), vec![0xa5; valid as usize]);
            let label = format!("{}-L{layer:02}", parts[0]);
            if !labels.insert(label.clone()) {
                return Err("duplicate capture label".into());
            }
            let host = Captured {
                label,
                source,
                rows,
                cached,
                scalars: record["scalars"].clone(),
                data,
                expected,
            };
            host.calls()?;
            all.push(host);
        }
        if layers != BTreeSet::from([0, 2, 20]) {
            return Err("capture lacks a required attention layer type".into());
        }
    }
    if all.is_empty() || index["records"].as_u64() != Some(all.len() as u64) {
        return Err("capture record count differs".into());
    }
    Ok(all)
}
impl Captured {
    pub(super) fn view(&self, name: &str) -> LabResult<View> {
        let index = self
            .data
            .keys()
            .position(|key| key == name)
            .ok_or_else(|| format!("missing buffer {name}"))?;
        Ok(View {
            handle: index as u64 + 1,
            bytes: self.data[name].len() as u64,
        })
    }
    fn arguments(&self, params: &[EntryParam]) -> LabResult<ArgumentList> {
        let mut args = Vec::new();
        for param in params {
            args.push(match param {
                EntryParam::DevicePointer { name, .. } => {
                    Argument::DeviceBuffer(self.view(name)?.handle)
                }
                EntryParam::U32 { name } => Argument::U32(
                    u32::try_from(self.scalars[name].as_u64().ok_or("scalar absent")?)
                        .map_err(|e| e.to_string())?,
                ),
                EntryParam::U64 { name } => {
                    Argument::U64(self.scalars[name].as_u64().ok_or("scalar absent")?)
                }
                _ => return Err("unsupported captured argument".into()),
            });
        }
        Ok(ArgumentList(args))
    }
    pub(super) fn calls(&self) -> LabResult<Vec<Launch>> {
        let body = Launch {
            symbol: self.source.symbol(),
            rows: self.rows,
            plan: self.source.plan(self.rows).map_err(|e| e.to_string())?,
            arguments: self.arguments(&self.source.params().map_err(|e| e.to_string())?)?,
            collective: false,
            persistent: false,
        };
        let mut calls = if self.cached {
            sources_cached::Entry {
                source: self.source.clone(),
                profile: false,
            }
            .launches(
                &body,
                self.view("decoded")?,
                self.view("cached_valid")?,
                None,
            )
            .map_err(|e| e.to_string())?
        } else {
            vec![body]
        };
        calls.push(Launch {
            symbol: self.source.merge_symbol(),
            rows: self.rows,
            plan: self
                .source
                .merge_plan(self.rows)
                .map_err(|e| e.to_string())?,
            arguments: self.arguments(&self.source.merge_params().map_err(|e| e.to_string())?)?,
            collective: false,
            persistent: false,
        });
        Ok(calls)
    }
}

#[test]
#[ignore = "requires CHOPINFER_ATTENTION_CAPTURE_ROOT naming authenticated real operands"]
fn frozen_captured_attention() {
    let inputs = captures().unwrap();
    for host in &inputs {
        assert_eq!(
            host.calls().unwrap().last().unwrap().plan.grid,
            [host.rows, 16, 1]
        );
    }
    println!("ATTENTION_CAPTURE_CPU_OK records={}", inputs.len());
}

#[test]
#[ignore = "requires an allocated GPU and a complete real attention operand capture"]
fn gpu_captured_attention_replay() {
    let inputs = captures().unwrap();
    let ran=kernel_harness::gpu_test_with(cubins().unwrap(),1<<30,|session| {
        let modules=Modules::load(session)?;let mut count=0;
        for host in &inputs {
            let mut device=BTreeMap::new();
            for (name,data) in &host.data {device.insert(name,session.allocate(data.len().max(16))?);}
            let bindings=kernel_harness::launch::Bindings{buffers:device.iter().map(|(name,data)|Ok((host.view(name)?.handle,data))).collect::<LabResult<_>>()?};
            let calls=host.calls()?;
            for poison in [0xa5,0x5a] {
                for (name,data) in &host.data {
                    let bytes=if matches!(name.as_str(),"workspace"|"out"|"decoded"|"cached_valid") {vec![poison;data.len()]} else {data.clone()};
                    device[name].copy_from_bytes(&bytes,&session.stream).map_err(|e|e.to_string())?;
                }
                for call in &calls {modules.launch(session,&session.stream,call,&bindings)?;}
                for (name,expected) in ["workspace","out"].into_iter().zip(&host.expected) {
                    let mut bytes=vec![0;expected.len()];
                    device[&name.to_owned()].copy_to_bytes(&mut bytes,&session.stream).map_err(|e|e.to_string())?;
                    if bytes!=*expected {return Err(format!("{} {name} capture replay differs at {:?}",host.label,bytes.iter().zip(expected).position(|(a,b)|a!=b)));}
                }
            }
            println!("ATTENTION_CAPTURE_REPLAY_OK {}",json!({"id":host.label,"rows":host.rows,"cached":host.cached,"poison_replays":2,
                "partial_sha256":kernel_build::digest(&host.expected[0]),"output_sha256":kernel_build::digest(&host.expected[1])}));
            count+=1;
        }
        assert_eq!(count,inputs.len());
        println!("ATTENTION_CAPTURE_REPLAY_COMPLETE records={count} poison_replays={}",count*2);
        Ok(())
    }).unwrap();
    assert!(ran.is_some());
}
