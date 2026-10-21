//! Qwen3.8 Flash per-operation checks on bundle dumps: QSA selections
//! recomputed from a dump's own rows, batch drift between dumps and against
//! the GPU reference runner's taps, and prepared projections on captured
//! operands. Use `qwen38-oracle --help` for commands.

use model_qwen38::Result;
use model_qwen38::oracle::LAYERS;
use model_qwen38::oracle::prompts::{Prompt, load_prompts};
use model_qwen38::oracle::weights::Checkpoint;
use model_qwen38_contract::contract::CONTEXT_TOKENS;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage:
  qwen38-oracle qsa-capture --dump DIR --prompt NAME [--occurrence K] [--stats-every N]
      [--rope-table hf|f64-angle] [--export DIR [--export-every N]] --out FILE
  qwen38-oracle batch-drift --prompt NAME --a DUMP [--a-occurrence K] --b DUMP
      [--b-occurrence K] [--reference TAPDIR] --out FILE
  qwen38-oracle prepared-capture|measure-capture --a STEP --precision-component FILE --layers A-B [--tail N] --out FILE
Common: --checkpoint DIR (default $CHOPINFER_QWEN38_CHECKPOINT) and --prompts FILE
(default $CHOPINFER_DATA/qwen38-refs/prompts/prompts.json), or --completions FILE
(completion texts with their continuations). Layers are inclusive.
(--precision-component FILE supplies a prepared BF16 execution contract for mixed weights)";

struct Args {
    command: String,
    values: BTreeMap<String, Vec<String>>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let command = args.next().ok_or(USAGE)?;
        let mut values = BTreeMap::<String, Vec<String>>::new();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--checkpoint"
                | "--prompts"
                | "--completions"
                | "--prompt"
                | "--layers"
                | "--out"
                | "--stats-every"
                | "--export"
                | "--export-every"
                | "--dump"
                | "--occurrence"
                | "--rope-table"
                | "--tail"
                | "--reference"
                | "--a"
                | "--b"
                | "--a-occurrence"
                | "--b-occurrence"
                | "--precision-component" => {
                    let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
                    values.entry(arg).or_default().push(value);
                }
                _ => return Err(format!("unknown argument {arg}\n{USAGE}").into()),
            }
        }
        Ok(Self { command, values })
    }

    fn one(&self, name: &str) -> Option<&str> {
        self.values
            .get(name)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    fn path(&self, name: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            self.one(name)
                .ok_or_else(|| format!("{name} is required\n{USAGE}"))?,
        ))
    }

    fn checkpoint_root(&self) -> Result<PathBuf> {
        Ok(match self.one("--checkpoint") {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(
                std::env::var_os("CHOPINFER_QWEN38_CHECKPOINT")
                    .ok_or("set CHOPINFER_QWEN38_CHECKPOINT or --checkpoint")?,
            ),
        })
    }

    fn checkpoint(&self) -> Result<Checkpoint> {
        match self.one("--precision-component") {
            None => Checkpoint::open(&self.checkpoint_root()?),
            Some(path) => Checkpoint::open_prepared(
                &self.checkpoint_root()?,
                &model_compiler_sdk::quantization::identity::read_component(Path::new(path))?,
            ),
        }
    }

    fn prompts_path(&self) -> Result<PathBuf> {
        match self.one("--prompts") {
            Some(path) => Ok(PathBuf::from(path)),
            None => Ok(PathBuf::from(
                std::env::var_os("CHOPINFER_DATA").ok_or("set CHOPINFER_DATA or --prompts")?,
            )
            .join("qwen38-refs/prompts/prompts.json")),
        }
    }

    fn prompts(&self) -> Result<Vec<Prompt>> {
        let all = match self.one("--completions") {
            Some(path) => self.completion_prompts(Path::new(path))?,
            None => load_prompts(&self.prompts_path()?)?,
        };
        for prompt in &all {
            let tokens = prompt.token_ids.len();
            if tokens as u64 > CONTEXT_TOKENS {
                return Err(format!(
                    "prompt {} has {tokens} tokens, exceeding context limit {CONTEXT_TOKENS}",
                    prompt.name
                )
                .into());
            }
        }
        let Some(names) = self.one("--prompt") else {
            return Ok(all);
        };
        names
            .split(',')
            .map(|name| {
                all.iter()
                    .find(|p| p.name == name)
                    .cloned()
                    .ok_or_else(|| format!("no prompt {name}").into())
            })
            .collect()
    }

    /// `--completions FILE`: a JSON list of `{"name", "text", "continuation"}`.
    /// The text is tokenized as a completions request does and the
    /// continuation (token strings, one token each) is appended.
    fn completion_prompts(&self, path: &Path) -> Result<Vec<Prompt>> {
        #[derive(serde::Deserialize)]
        struct Completion {
            name: String,
            text: String,
            continuation: Vec<String>,
        }
        let tokenizer = model_qwen38::local_frontend::LocalFrontend::load(
            &self.checkpoint_root()?,
            &"0".repeat(64),
        )?
        .tokenizer()?;
        let completions: Vec<Completion> = serde_json::from_slice(&std::fs::read(path)?)?;
        completions
            .into_iter()
            .map(|c| {
                let mut token_ids = tokenizer.encode(&c.text, &[])?;
                for piece in &c.continuation {
                    match tokenizer
                        .encode(piece, model_qwen38::tokenizer::PROMPT_TOKENS)?
                        .as_slice()
                    {
                        [id] => token_ids.push(*id),
                        ids => return Err(format!("{piece:?} is {} tokens", ids.len()).into()),
                    }
                }
                Ok(Prompt {
                    name: c.name,
                    token_ids,
                })
            })
            .collect()
    }

    fn layers(&self, default: Range<usize>) -> Result<Range<usize>> {
        let Some(spec) = self.one("--layers") else {
            return Ok(default);
        };
        let (a, b) = spec.split_once('-').unwrap_or((spec, spec));
        let (a, b) = (a.parse::<usize>()?, b.parse::<usize>()?);
        if a > b || b >= LAYERS {
            return Err(format!("layers {spec} outside 0-{}", LAYERS - 1).into());
        }
        Ok(a..b + 1)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn log(line: String) {
    eprintln!("qwen38-oracle: {line}");
}

fn write_new(path: &Path, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create_new(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn record_precision(checkpoint: &Checkpoint, provenance: &mut serde_json::Value) {
    if let Some(precision) = checkpoint.precision_provenance() {
        provenance["precision"] = precision.clone();
    }
}

/// A bundle dump's QSA selections against the policies recomputed from its
/// own projection rows (oracle::selection `capture_selection`), and with
/// `--export` the operands for kernel-lab's operand check.
fn run_qsa_capture(args: &Args) -> Result<()> {
    use model_qwen38::oracle::selection::{EngineTable, Export, capture_selection, indexer_norms};
    let out = args.path("--out")?;
    if out.exists() {
        return Err(format!("{} exists", out.display()).into());
    }
    let prompts = args.prompts()?;
    let [prompt] = prompts.as_slice() else {
        return Err("qsa-capture takes one --prompt".into());
    };
    let dump = args.path("--dump")?;
    let occurrence = args.one("--occurrence").unwrap_or("0").parse()?;
    let every = args.one("--stats-every").unwrap_or("1").parse()?;
    let table = match args.one("--rope-table").unwrap_or("hf") {
        "hf" => EngineTable::Hf,
        "f64-angle" => EngineTable::F64Angle,
        other => return Err(format!("unknown --rope-table {other}: hf or f64-angle").into()),
    };
    let export = match args.one("--export") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            if dir.exists() {
                return Err(format!("{} exists", dir.display()).into());
            }
            Some(Export {
                dir,
                every: args.one("--export-every").unwrap_or("64").parse()?,
            })
        }
        None if args.one("--export-every").is_some() => {
            return Err("--export-every needs --export".into());
        }
        None => None,
    };
    let checkpoint = args.checkpoint()?;
    let layers = capture_selection(
        &|layer| indexer_norms(&checkpoint, layer),
        &dump,
        &prompt.token_ids,
        occurrence,
        every,
        table,
        export.as_ref(),
        log,
    )?;
    let mut value = serde_json::json!({
        "schema": "chopinfer.qwen38.qsa-capture.v1",
        "provenance": {
            "source": "qwen38-oracle qsa-capture (models/qwen38/compiler/src/oracle/selection.rs)",
            "checkpoint_revision": checkpoint.content_revision()?,
            "prompts_sha256": sha256_file(&args.prompts_path()?)?,
            "prompt": prompt.name,
            "dump": dump.display().to_string(),
            "occurrence": occurrence,
            "stats_every": every,
            "rope_table": table,
            "export": export.as_ref().map(|e| e.dir.display().to_string()),
            "export_every": export.as_ref().map(|e| e.every),
            "method": "per full-attention layer the dump holds, the device's prefill x.indexer.topk sets against the engine, kernel and hf-style selections recomputed from its own BF16 x.indexer.qk rows with the engine's cos/sin table (see selection.rs); the kernel emulation is also compared with engine; the export holds those projection rows, the engine operands emulated from them and the engine and kernel sets of the exported rows",
        },
        "layers": layers,
    });
    record_precision(&checkpoint, &mut value["provenance"]);
    write_new(&out, &value)?;
    log(format!("wrote {}", out.display()));
    Ok(())
}

/// Batch drift of one prompt between two bundle dumps (oracle::drift). The
/// prompt's token IDs come from the prompt set (`--prompt`); with
/// `--reference`, a reference runner's tap directory, each dump is also
/// compared with the reference's taps.
fn run_batch_drift(args: &Args) -> Result<()> {
    use model_qwen38::oracle::drift::{compare, locate, reference};
    let out = args.path("--out")?;
    if out.exists() {
        return Err(format!("{} exists", out.display()).into());
    }
    let prompts = args.prompts()?;
    let [prompt] = prompts.as_slice() else {
        return Err("batch-drift takes one --prompt".into());
    };
    let tokens = &prompt.token_ids;
    let occurrence = |name: &str| -> Result<usize> { Ok(args.one(name).unwrap_or("0").parse()?) };
    let (a, b) = (args.path("--a")?, args.path("--b")?);
    let pieces_a = locate(&a, tokens, occurrence("--a-occurrence")?)?;
    let pieces_b = locate(&b, tokens, occurrence("--b-occurrence")?)?;
    let reference = args
        .one("--reference")
        .map(|dir| reference(Path::new(dir), tokens))
        .transpose()?;
    let taps = compare((&a, &pieces_a), (&b, &pieces_b), reference.as_ref())?;
    for tap in &taps {
        let against = tap
            .reference
            .as_ref()
            .map(|d| {
                let last = d
                    .last_row_rel_l2
                    .map(|l| format!(" last a {:.2e} b {:.2e}", l[0], l[1]))
                    .unwrap_or_default();
                format!(
                    " ref ({} rows) a {:.2e} b {:.2e}{last}",
                    d.rows, d.rel_l2[0], d.rel_l2[1]
                )
            })
            .unwrap_or_default();
        let selection = tap
            .selection
            .as_ref()
            .map(|s| {
                format!(
                    " sets differ {}/{} (swapped {:.2}, last {}, rank disagreements {:?})",
                    s.rows_differing,
                    s.sparse_rows,
                    s.mean_blocks_swapped,
                    s.last_row_differs,
                    s.rank_disagreements
                )
            })
            .unwrap_or_default();
        let routes = tap
            .route_flips
            .map(|n| format!(" route flips {n}"))
            .unwrap_or_default();
        log(format!(
            "{:24} equal {:5}/{:5} first {:>5} elems {:.4} max {:.3e} rel {:.3e} last {:.3e}{selection}{routes}{against}",
            tap.name,
            tap.rows_equal,
            tap.rows,
            tap.first_differing_row
                .map(|r| r.to_string())
                .unwrap_or("-".into()),
            tap.elements_differing,
            tap.max_abs,
            tap.rel_l2,
            tap.last_row_rel_l2,
        ));
    }
    let value = serde_json::json!({
        "schema": "chopinfer.qwen38.batch-drift.v2",
        "definitions": "rows are the prompt's prefill rows gathered from each dump's items; rel is ||b - a|| / ||a|| over all rows, last over the last prompt row; reference is ||dump - ref|| / ||ref|| of [a, b] against the reference runner's tap of the same name on the positions the reference stored (rows), and on the last prompt row when it stored it, for densely compared replicated taps (rank 0) and rank partials (rank sum); x.indexer.topk compares key sets as sets on rows with more than 512 complete blocks",
        "prompt": prompt.name,
        "prompts_sha256": sha256_file(&args.prompts_path()?)?,
        "a": { "dump": a, "pieces": pieces_a },
        "b": { "dump": b, "pieces": pieces_b },
        "reference": reference,
        "taps": taps,
    });
    write_new(&out, &value)?;
    Ok(())
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.command.as_str() {
        "qsa-capture" => run_qsa_capture(&args),
        "batch-drift" => run_batch_drift(&args),
        "prepared-capture" | "measure-capture" => {
            let checkpoint = args.checkpoint()?;
            let value = model_qwen38::oracle::prepared_capture::compare(
                &checkpoint,
                &args.path("--a")?,
                args.layers(0..4)?,
                args.one("--tail").unwrap_or("2").parse()?,
                args.command == "prepared-capture",
            )?;
            write_new(&args.path("--out")?, &value)
        }
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38-oracle: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforce_model_context_limit_for_token_id_prompts() {
        let limit = CONTEXT_TOKENS as usize;
        for tokens in [limit, limit + 1] {
            let path = std::env::temp_dir()
                .join(format!("qwen38-token-context-{}.json", std::process::id()));
            let input = serde_json::json!({
                "teacher": [{"name": "captured", "token_ids": vec![42; tokens]}],
                "long": [],
            });
            std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
            let args = Args {
                command: "batch-drift".into(),
                values: BTreeMap::from([
                    ("--prompts".into(), vec![path.to_str().unwrap().into()]),
                    ("--prompt".into(), vec!["captured".into()]),
                ]),
            };
            let result = args.prompts();
            std::fs::remove_file(path).unwrap();
            if tokens == limit {
                assert_eq!(result.unwrap()[0].token_ids.len(), limit);
            } else {
                let error = result.expect_err("overlong prompt must fail").to_string();
                assert!(error.contains("context limit 262144"), "{error}");
                assert!(error.contains("262145 tokens"), "{error}");
            }
        }
    }
}
