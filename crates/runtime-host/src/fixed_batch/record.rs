use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use runtime_engine::{ModelBatchItem, SampledRow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::diagnostic::{self, Description, Publication};
use crate::executor::SamplingDispatch;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleBits {
    pub request: u64,
    pub token: u32,
    pub logprob: u64,
    pub top: Vec<(u32, u64)>,
}
impl From<&SampledRow> for SampleBits {
    fn from(row: &SampledRow) -> Self {
        Self {
            request: row.request_id().get(),
            token: row.token_id(),
            logprob: row.logprob().to_bits(),
            top: row
                .top_logprobs()
                .iter()
                .map(|p| (p.token_id(), p.logprob().to_bits()))
                .collect(),
        }
    }
}

pub fn metadata(item: &ModelBatchItem, dispatch: SamplingDispatch) -> Value {
    let work = item.sampling().expect("every trace row samples");
    let p = &work.parameters;
    json!({"request":item.request_id().get(), "history":work.history.iter().copied().collect::<Vec<_>>(),
        "emitted":work.emitted_tokens, "seed":work.state.seed(), "generated":work.state.generated(),
        "stop_ids":work.stop_ids, "mask":work.constraint.as_ref().and_then(|c| c.mask()),
        "temperature":p.temperature().to_bits(), "top_k":p.top_k(), "top_p":p.top_p().to_bits(),
        "min_p":p.min_p().to_bits(), "minimum_new_tokens":p.minimum_new_tokens(),
        "presence":p.presence_penalty().to_bits(), "frequency":p.frequency_penalty().to_bits(),
        "repetition":p.repetition_penalty().to_bits(), "top_count":p.top_logprobs(),
        "bias":p.logit_bias().iter().map(|b| (b.token_id(),b.bias().to_bits())).collect::<Vec<_>>(),
        "dispatch":{"rows":dispatch.rows,"simple":dispatch.simple,"top":dispatch.top,"greedy":dispatch.greedy}})
}

pub struct CapturedRow {
    pub rank: u32,
    pub element_type: u32,
    pub width: u32,
    pub bytes: Vec<u8>,
    pub metadata: Value,
    pub sample: Option<SampleBits>,
}
#[derive(Default)]
pub struct Capture {
    pub rows: BTreeMap<usize, CapturedRow>,
}
impl Capture {
    pub fn insert(&mut self, index: usize, row: CapturedRow) -> Result<()> {
        if self.rows.insert(index, row).is_some() {
            return Err(format!("duplicate captured row {index}").into());
        }
        Ok(())
    }
    pub fn samples(&mut self, first: usize, rows: &[SampledRow]) -> Result<()> {
        for (offset, row) in rows.iter().enumerate() {
            let capture = self
                .rows
                .get_mut(&(first + offset))
                .ok_or("sample without raw row")?;
            if capture.sample.replace(row.into()).is_some() {
                return Err("duplicate sample".into());
            }
        }
        Ok(())
    }
    pub fn take(&mut self, count: usize) -> Result<Vec<CapturedRow>> {
        if self.rows.len() != count
            || self.rows.keys().copied().ne(0..count)
            || self.rows.values().any(|r| r.sample.is_none())
        {
            return Err("incomplete captured step".into());
        }
        Ok(std::mem::take(&mut self.rows).into_values().collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowEvidence {
    pub element_type: u32,
    pub width: u32,
    pub byte_count: usize,
    pub metadata: Value,
    pub sample: SampleBits,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    pub request: Vec<u8>,
    pub kept: Vec<(u64, usize)>,
    /// Decode token padding bucket; zero for a prefill-only step.
    pub bucket: usize,
    pub drafts: Vec<(u64, Vec<u32>)>,
    pub rows: Vec<RowEvidence>,
}
impl Frame {
    pub fn from_rows(
        request: Vec<u8>,
        kept: Vec<(u64, usize)>,
        bucket: usize,
        drafts: Vec<(u64, Vec<u32>)>,
        rows: &[CapturedRow],
    ) -> Self {
        Self {
            request,
            kept,
            bucket,
            drafts,
            rows: rows
                .iter()
                .map(|row| RowEvidence {
                    element_type: row.element_type,
                    width: row.width,
                    byte_count: row.bytes.len(),
                    metadata: row.metadata.clone(),
                    sample: row.sample.clone().expect("collected sample"),
                })
                .collect(),
        }
    }
}

pub fn create(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

pub struct Golden {
    output: PathBuf,
    publication: Option<Publication>,
    first: Option<Value>,
    metadata_bytes: u64,
    frames: BufWriter<File>,
    bytes: Option<BufWriter<File>>,
    reference: Option<(BufReader<File>, BufReader<File>)>,
    pub steps: usize,
    pub rows: usize,
    pub bytes_checked: u64,
}
impl Golden {
    pub fn open(output: &Path, reference: Option<&Path>, identity: &Value) -> Result<Self> {
        Self::open_inner(output, reference, identity, true)
    }

    pub fn open_diagnostic(
        output: &Path,
        reference: Option<&Path>,
        control: Option<&Path>,
        identity: &Value,
        description: Description,
    ) -> Result<Self> {
        diagnostic::external(output)?;
        description.validate()?;
        let mut publication = Publication {
            description,
            reference: None,
            control: None,
        };
        let mut status = "insufficient";
        let gate = (|| -> Result<()> {
            let mut reference_description = None;
            if let Some(reference) = reference {
                let old = diagnostic::read_terminal(reference)?;
                publication.reference = Some(diagnostic::binding(reference)?);
                let desc: Description = serde_json::from_value(old["description"].clone())?;
                status = "incompatible";
                diagnostic::compatible(&desc, &publication.description)?;
                status = "insufficient";
                reference_description = Some(desc);
            }
            if let Some(control) = control {
                let old = diagnostic::read_terminal(control)?;
                if old["status"] != "match" || old["reference"] != json!(publication.reference) {
                    return Err("repeat is not bound to this reference".into());
                }
                let desc: Description = serde_json::from_value(old["description"].clone())?;
                if !desc.declared_differences.is_empty() {
                    return Err("baseline repeat must have no declared differences".into());
                }
                diagnostic::compatible(
                    reference_description
                        .as_ref()
                        .ok_or("repeat requires a reference")?,
                    &desc,
                )
                .map_err(|error| format!("baseline repeat is incompatible: {error}"))?;
                publication.control = Some(diagnostic::binding(control)?);
            }
            Ok(())
        })();
        if let Err(error) = gate {
            std::fs::create_dir(output)?;
            publication.publish(output, status, (0, 0, 0), None, &error.to_string())?;
            return Err(error);
        }
        let mut result = Self::open_inner(output, reference, identity, false)?;
        result.publication = Some(publication);
        Ok(result)
    }

    fn open_inner(
        output: &Path,
        reference: Option<&Path>,
        identity: &Value,
        strict: bool,
    ) -> Result<Self> {
        std::fs::create_dir(output)?;
        let mut frames = BufWriter::new(create(&output.join("frames.jsonl"))?);
        serde_json::to_writer(&mut frames, identity)?;
        writeln!(frames)?;
        let reference = reference
            .map(|root| -> Result<_> {
                let receipt: Value =
                    serde_json::from_reader(File::open(root.join("receipt.json"))?)?;
                if receipt["passed"] != true {
                    return Err("reference has no successful receipt".into());
                }
                let mut records = BufReader::new(File::open(root.join("frames.jsonl"))?);
                let mut line = String::new();
                records.read_line(&mut line)?;
                if serde_json::from_str::<Value>(&line)? != *identity && strict {
                    return Err("reference identity differs".into());
                }
                Ok((records, BufReader::new(File::open(root.join("rows.bin"))?)))
            })
            .transpose()?;
        let bytes = if reference.is_none() {
            Some(BufWriter::new(create(&output.join("rows.bin"))?))
        } else {
            None
        };
        Ok(Self {
            output: output.to_owned(),
            publication: None,
            first: None,
            metadata_bytes: serde_json::to_vec(identity)?.len() as u64 + 1,
            frames,
            bytes,
            reference,
            steps: 0,
            rows: 0,
            bytes_checked: 0,
        })
    }

    pub fn step(&mut self, frame: &Frame, rows: &[CapturedRow]) -> Result<()> {
        let result = self.compare_step(frame, rows);
        if let Err(error) = &result
            && self.publication.is_some()
        {
            self.flush()?;
            if self.first.is_some() {
                let limits = &self.publication.as_ref().unwrap().description.limits;
                diagnostic::write_json(
                    &self.output.join("mismatch.json"),
                    &serde_json::to_value(frame)?,
                    limits.metadata_bytes,
                )?;
                let mut bytes = create(&self.output.join("mismatch.bin"))?;
                for row in rows {
                    bytes.write_all(&row.bytes)?;
                }
                bytes.sync_all()?;
            }
            self.publish(
                if self.first.is_some() {
                    "diverged"
                } else {
                    "insufficient"
                },
                &error.to_string(),
            )?;
        }
        result
    }

    fn compare_step(&mut self, frame: &Frame, rows: &[CapturedRow]) -> Result<()> {
        if frame.rows.len() != rows.len() {
            return Err("frame/capture row count differs".into());
        }
        let mut size = 0_u64;
        for (row, evidence) in rows.iter().zip(&frame.rows) {
            let expected = diagnostic::row_bytes(row.element_type, row.width)?;
            if row.bytes.len() as u64 != expected || evidence.byte_count as u64 != expected {
                return Err("incomplete captured logits bytes".into());
            }
            size = size.checked_add(expected).ok_or("capture size overflow")?;
        }
        let encoded = serde_json::to_vec(frame)?;
        let metadata_bytes = self.metadata_bytes + encoded.len() as u64 + 1;
        if let Some(p) = &self.publication {
            p.description.limits.check(
                size,
                self.bytes_checked
                    .checked_add(size)
                    .ok_or("capture size overflow")?,
            )?;
            if metadata_bytes > p.description.limits.metadata_bytes {
                return Err("frame metadata budget exceeded".into());
            }
        }
        if let Some((records, bytes)) = &mut self.reference {
            let mut line = String::new();
            if records.read_line(&mut line)? == 0 {
                return Err("reference ended before trace".into());
            }
            let expected: Frame = serde_json::from_str(&line)?;
            if expected.request != frame.request
                || expected.kept != frame.kept
                || expected.bucket != frame.bucket
                || expected.drafts != frame.drafts
            {
                self.first = Some(schedule_difference(&expected, frame, self.steps));
                return Err(
                    format!("step {} model input/draft/commit trace differs", self.steps).into(),
                );
            }
            if expected.rows.len() != rows.len() {
                self.first = Some(json!({"stage":"row-shape","step":self.steps,
                    "expected":expected.rows.len(),"observed":rows.len()}));
                return Err("reference row count differs".into());
            }
            for (index, ((actual, evidence), wanted)) in
                rows.iter().zip(&frame.rows).zip(&expected.rows).enumerate()
            {
                if evidence.element_type != wanted.element_type
                    || evidence.width != wanted.width
                    || evidence.byte_count != wanted.byte_count
                    || evidence.metadata != wanted.metadata
                {
                    let mut difference = row_context(frame, self.steps, index);
                    difference["stage"] = json!("sampling-input");
                    difference["expected"] = serde_json::to_value(wanted)?;
                    difference["observed"] = serde_json::to_value(evidence)?;
                    self.first = Some(difference);
                    return Err(format!(
                        "step {} row {index} sampling input/dispatch differs",
                        self.steps
                    )
                    .into());
                }
                let mut old = vec![0; wanted.byte_count];
                bytes.read_exact(&mut old)?;
                if let Some(offset) = old.iter().zip(&actual.bytes).position(|(a, b)| a != b) {
                    let mut difference = row_context(frame, self.steps, index);
                    let size = if evidence.element_type == 1 { 4 } else { 2 };
                    let start = offset / size * size;
                    difference["stage"] = json!("logits");
                    difference["byte_offset"] = json!(offset);
                    difference["element"] = json!(offset / size);
                    difference["expected"] = scalar(&old[start..start + size]);
                    difference["observed"] = scalar(&actual.bytes[start..start + size]);
                    self.first = Some(difference);
                    return Err(format!(
                        "step {} row {index} full logits differ at byte {offset}: {:02x} != {:02x}",
                        self.steps, old[offset], actual.bytes[offset]
                    )
                    .into());
                }
                if evidence.sample != wanted.sample {
                    let mut difference = row_context(frame, self.steps, index);
                    difference["stage"] = json!("sample");
                    difference["expected"] = serde_json::to_value(&wanted.sample)?;
                    difference["observed"] = serde_json::to_value(&evidence.sample)?;
                    self.first = Some(difference);
                    return Err(format!(
                        "step {} row {index} sampled record differs: {:?} != {:?}",
                        self.steps, wanted.sample, evidence.sample
                    )
                    .into());
                }
            }
        }
        if let Some(bytes) = &mut self.bytes {
            for row in rows {
                bytes.write_all(&row.bytes)?;
            }
        }
        self.frames.write_all(&encoded)?;
        writeln!(self.frames)?;
        self.steps += 1;
        self.rows += rows.len();
        self.bytes_checked += rows.iter().map(|r| r.bytes.len() as u64).sum::<u64>();
        self.metadata_bytes = metadata_bytes;
        Ok(())
    }
    pub fn finish(&mut self, expected_steps: usize) -> Result<()> {
        if self.steps != expected_steps || self.rows == 0 {
            return Err("trace did not complete".into());
        }
        if let Some((frames, bytes)) = &mut self.reference
            && (!frames.fill_buf()?.is_empty() || !bytes.fill_buf()?.is_empty())
        {
            return Err("reference contains trailing records or logits".into());
        }
        if let Some(p) = &self.publication
            && (expected_steps != p.description.expected_steps
                || self.rows != p.description.expected_rows)
        {
            return Err("capture coverage differs from description".into());
        }
        self.flush()
    }

    fn flush(&mut self) -> Result<()> {
        self.frames.flush()?;
        if let Some(bytes) = &mut self.bytes {
            bytes.flush()?;
        }
        Ok(())
    }

    pub fn publish(&mut self, status: &str, reason: &str) -> Result<()> {
        self.flush()?;
        if let Some(p) = &self.publication {
            p.publish(
                &self.output,
                status,
                (self.steps, self.rows, self.bytes_checked),
                self.first.as_ref(),
                reason,
            )?;
        }
        Ok(())
    }
}

fn scalar(bytes: &[u8]) -> Value {
    let (bits, value) = if bytes.len() == 4 {
        let bits = u32::from_le_bytes(bytes.try_into().unwrap());
        (format!("0x{bits:08x}"), f32::from_bits(bits))
    } else {
        let bits = u16::from_le_bytes(bytes.try_into().unwrap());
        (
            format!("0x{bits:04x}"),
            f32::from_bits(u32::from(bits) << 16),
        )
    };
    json!({"bits":bits,"value":value.is_finite().then_some(value),
        "classification":if value.is_nan() { "nan" } else if value.is_infinite() { "infinite" } else { "finite" }})
}

fn schedule(frame: &Frame) -> Value {
    json!({"request_sha256":runtime_weights::sha256_bytes(&frame.request),
        "kept":frame.kept,"drafts":frame.drafts,"calculated_bucket":frame.bucket})
}

fn schedule_difference(expected: &Frame, observed: &Frame, step: usize) -> Value {
    let mut value = json!({"stage":"schedule","step":step,
        "expected":schedule(expected),"observed":schedule(observed)});
    if let (Ok(a), Ok(b)) = (
        runtime_bundle_sdk::BatchCodecV1::decode(&expected.request),
        runtime_bundle_sdk::BatchCodecV1::decode(&observed.request),
    ) {
        let index = (0..a.items().len().max(b.items().len())).find(|&i| {
            a.items().get(i) != b.items().get(i)
                || a.appends().get(i) != b.appends().get(i)
                || a.state_slots().and_then(|s| s.get(i)) != b.state_slots().and_then(|s| s.get(i))
        });
        if let Some(index) = index {
            value["item"] = json!(index);
            value["request"] = json!(
                b.items()
                    .get(index)
                    .or_else(|| a.items().get(index))
                    .map(|i| i.request_id())
            );
            for (key, batch) in [("expected", &a), ("observed", &b)] {
                value[key]["batch_item"] = json!(format!("{:?}", batch.items().get(index)));
                value[key]["append"] = json!(format!("{:?}", batch.appends().get(index)));
                value[key]["state_slot"] = json!(batch.state_slots().and_then(|s| s.get(index)));
            }
        } else {
            value["request"] = json!(
                observed
                    .kept
                    .iter()
                    .zip(&expected.kept)
                    .find(|(a, b)| a != b)
                    .map(|(a, _)| a.0)
                    .or_else(|| observed
                        .drafts
                        .iter()
                        .zip(&expected.drafts)
                        .find(|(a, b)| a != b)
                        .map(|(a, _)| a.0))
            );
        }
    }
    value
}

fn row_context(frame: &Frame, step: usize, row: usize) -> Value {
    let evidence = &frame.rows[row];
    let request = evidence.sample.request;
    let mut value = json!({"step":step,"row":row,"request":request,
        "verify_row":frame.rows[..row].iter().filter(|r| r.sample.request == request).count(),
        "dtype":if evidence.element_type == 1 { "f32" } else { "bf16" },"shape":[evidence.width]});
    if let Ok(batch) = runtime_bundle_sdk::BatchCodecV1::decode(&frame.request)
        && let Some(index) = batch
            .items()
            .iter()
            .position(|item| item.request_id() == request)
    {
        value["item"] = json!(index);
        value["state_slot"] = json!(batch.state_slots().map(|s| s[index]));
        value["batch_item"] = json!(format!("{:?}", batch.items()[index]));
        value["append"] = json!(format!("{:?}", batch.appends().get(index)));
    }
    value
}
