//! The `encoder_catalog_stability` check: a model's encoder cubins, built
//! from its committed declarations, must match the records its kernels.lock
//! holds and the records the model adopted for those declarations
//! (`Adoption`). A change to a shared variant, engine or header that would
//! alter a model's cubins fails unless that model's declarations change with
//! it; refreshing the lock alone does not pass. CPU only: nvcc compiles into
//! a scratch directory, nothing runs on a GPU.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, CubinSpec, Toolchain, build};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

/// Builds `specs` (a model's generated encoder cubins) and checks the
/// records twice: against `lock` (the cubins the bundle builds) and against
/// the model's adoptions (`adoptions`, see `Adoption`) for `declarations`,
/// the digests of the declarations they were generated from. A refused
/// adoption's error shows this build's adoption, chained to the ledger, to
/// append when the declarations changed. `base` is the approved ledger
/// (`approved_base`), whose entries the current one must start with.
pub fn check(
    specs: &[CubinSpec],
    declarations: &[String],
    lock: &Path,
    adoptions: &Path,
    base: Option<&str>,
) -> Result<(), String> {
    check_records(specs, declarations, &read(lock)?, adoptions, base)
}

/// `check` against lock family `family` of the lock directory `directory`
/// (`kernel_build::LockDir`): its `<family>.lock` records, each with the
/// source closure its `<family>.provenance.json` holds.
pub fn check_family(
    specs: &[CubinSpec],
    declarations: &[String],
    directory: &Path,
    family: &str,
    adoptions: &Path,
    base: Option<&str>,
) -> Result<(), String> {
    let committed = read_family(directory, family)?;
    check_records(specs, declarations, &committed, adoptions, base)
}

fn check_records(
    specs: &[CubinSpec],
    declarations: &[String],
    committed: &[Value],
    adoptions: &Path,
    base: Option<&str>,
) -> Result<(), String> {
    let built = build_records(specs)?;
    compare(&built, committed)?;
    let text = std::fs::read_to_string(adoptions)
        .map_err(|error| format!("{}: {error}", adoptions.display()))?;
    check_adoption(&built, declarations, &text, base).map_err(|error| {
        let append = adoption(&built, declarations)
            .map(|mut adoption| {
                adoption.previous = serde_json::from_str::<Vec<Adoption>>(&text)
                    .ok()
                    .and_then(|ledger| ledger.last().map(digest))
                    .unwrap_or_else(|| GENESIS.to_owned());
                adoption
            })
            .and_then(|adoption| {
                serde_json::to_string_pretty(&adoption).map_err(|error| error.to_string())
            })
            .unwrap_or_default();
        format!(
            "{}: {error}; the adoption of this build is\n{append}",
            adoptions.display()
        )
    })
}

fn read(path: &Path) -> Result<Vec<Value>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// The records of lock family `family` in `directory`, each joined with
/// its source closure from the family's provenance file.
fn read_family(directory: &Path, family: &str) -> Result<Vec<Value>, String> {
    let mut records = read(&directory.join(format!("{family}.lock")))?;
    let provenance = read(&directory.join(format!("{family}.provenance.json")))?;
    for record in &mut records {
        let name = record["name"].clone();
        let closure = provenance
            .iter()
            .find(|entry| entry["name"] == name)
            .and_then(|entry| entry.get("source_closure_sha256"))
            .ok_or_else(|| format!("{family}.provenance.json has no source closure of {name}"))?
            .clone();
        record
            .as_object_mut()
            .ok_or_else(|| format!("{family}.lock record {name} is not an object"))?
            .insert("source_closure_sha256".into(), closure);
    }
    Ok(records)
}

/// The lock records of `specs`, compiled into a scratch directory.
fn build_records(specs: &[CubinSpec]) -> Result<Vec<Value>, String> {
    let toolchain = Toolchain::detect().map_err(|error| error.to_string())?;
    if toolchain.skipped {
        return Err("encoder catalog stability needs the CUDA compiler".into());
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let scratch =
        std::env::temp_dir().join(format!("encoder-stability-{}-{stamp}", std::process::id()));
    let built_lock = scratch.join("kernels.lock");
    let result = build(
        &toolchain,
        specs,
        &scratch.join("out"),
        &built_lock,
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .map_err(|error| error.to_string())
    .and_then(|_| read(&built_lock));
    std::fs::remove_dir_all(&scratch).map_err(|error| error.to_string())?;
    result
}

/// Every built record equals the committed record of its name.
fn compare(built: &[Value], committed: &[Value]) -> Result<(), String> {
    for record in built {
        let name = &record["name"];
        let locked = committed
            .iter()
            .find(|locked| locked["name"] == *name)
            .ok_or_else(|| format!("kernels.lock has no cubin {name}"))?;
        if locked != record {
            let fields = record
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(key, value)| locked.get(key.as_str()) != Some(value))
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>();
            return Err(format!(
                "cubin {name} differs from kernels.lock in {}",
                fields.join(", ")
            ));
        }
    }
    Ok(())
}

/// One adopted state of a model's generated encoder cubins: the digests of
/// the declarations they were generated from (in catalog order), the
/// compiler, and each cubin's source closure and bytes. A model keeps its
/// adoptions in `encoder-adoptions.json` beside its kernels.lock, append
/// only: its cubins change only when its declarations change (a declaration
/// adopting a new variant id, or another edit), and then an adoption is
/// appended for the new digests. A compiler change appends one for every
/// model. Each adoption names the digest of the one before it (`previous`,
/// `GENESIS` for the first; `digest`), so an edit to any but the last breaks
/// the chain, and `check_ledger` refuses a ledger that does not start with
/// the approved base's entries, which catches an edit to the last.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Adoption {
    pub previous: String,
    pub declarations: Vec<String>,
    pub nvcc_sha256: String,
    pub cubins: Vec<Adopted>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Adopted {
    pub name: String,
    pub source_closure_sha256: String,
    pub cubin_sha256: String,
}

/// The adoption of `built` lock records generated from `declarations`.
pub fn adoption(built: &[Value], declarations: &[String]) -> Result<Adoption, String> {
    let nvcc_sha256 = built
        .first()
        .and_then(|record| record["nvcc_sha256"].as_str())
        .ok_or("no built encoder cubin")?
        .to_owned();
    let cubins = built
        .iter()
        .map(|record| {
            serde_json::from_value(json!({
                "name": record["name"],
                "source_closure_sha256": record["source_closure_sha256"],
                "cubin_sha256": record["cubin_sha256"],
            }))
            .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    Ok(Adoption {
        previous: String::new(),
        declarations: declarations.to_vec(),
        nvcc_sha256,
        cubins,
    })
}

/// The `previous` of a ledger's first adoption.
pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// The hex SHA-256 an adoption's successor names as `previous`: of its own
/// `previous`, then one line per field (`tests/encoder-adoptions.py`
/// computes the same).
pub fn digest(adoption: &Adoption) -> String {
    let mut text = format!(
        "previous {}\ndeclarations {}\nnvcc {}\n",
        adoption.previous,
        adoption.declarations.join(" "),
        adoption.nvcc_sha256
    );
    for cubin in &adoption.cubins {
        text.push_str(&format!(
            "cubin {} {} {}\n",
            cubin.name, cubin.source_closure_sha256, cubin.cubin_sha256
        ));
    }
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Sets each adoption's `previous` to its predecessor's digest.
pub fn chain(ledger: &mut [Adoption]) {
    let mut previous = GENESIS.to_owned();
    for adoption in ledger {
        adoption.previous = previous;
        previous = digest(adoption);
    }
}

/// Parses an adoption ledger and checks that it is append only: written in
/// its canonical form (pretty JSON, two-space indent, final newline), each
/// adoption chained to the one before, and starting with `base`'s entries
/// when an approved base is given. A base written before adoptions were
/// chained (no `previous`) is compared without it.
pub fn check_ledger(text: &str, base: Option<&str>) -> Result<Vec<Adoption>, String> {
    let ledger: Vec<Adoption> =
        serde_json::from_str(text).map_err(|error| format!("encoder adoptions: {error}"))?;
    if serde_json::to_string_pretty(&ledger).map_err(|error| error.to_string())? + "\n" != text {
        return Err("encoder adoptions are not in their canonical form \
                    (pretty JSON, two-space indent, final newline)"
            .into());
    }
    let mut previous = GENESIS.to_owned();
    for (index, adoption) in ledger.iter().enumerate() {
        if adoption.previous != previous {
            return Err(format!(
                "encoder adoption {index} breaks the chain: it names {} but the adoptions \
                 before it digest to {previous}; adoptions are append only",
                adoption.previous
            ));
        }
        previous = digest(adoption);
    }
    if let Some(base) = base {
        let approved: Vec<Value> = serde_json::from_str(base)
            .map_err(|error| format!("approved encoder adoptions: {error}"))?;
        let current: Vec<Value> =
            serde_json::from_str(text).map_err(|error| format!("encoder adoptions: {error}"))?;
        for (index, approved) in approved.iter().enumerate() {
            let mut entry = current.get(index).cloned().ok_or_else(|| {
                format!("encoder adoption {index} of the approved base is missing; adoptions are append only")
            })?;
            if approved.get("previous").is_none()
                && let Some(entry) = entry.as_object_mut()
            {
                entry.remove("previous");
            }
            if entry != *approved {
                return Err(format!(
                    "encoder adoption {index} differs from the approved base; adoptions are \
                     append only: a change to a model's encoder cubins needs a declaration change \
                     and a new adoption"
                ));
            }
        }
    }
    Ok(ledger)
}

/// The approved ledger at `path`: its contents on `main`, or at the merge
/// base with `origin/main` when there is no `main`. `None` when the file is
/// new there; an error when git or both revisions are unavailable.
pub fn approved_base(path: &Path) -> Result<Option<String>, String> {
    let directory = path.parent().ok_or("adoptions path has no directory")?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("adoptions path has no file name")?;
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .current_dir(directory)
            .args(args)
            .output()
            .map_err(|error| format!("git: {error}"))
    };
    let revision = if git(&["rev-parse", "--verify", "--quiet", "main^{commit}"])?
        .status
        .success()
    {
        "main".to_owned()
    } else {
        let base = git(&["merge-base", "HEAD", "origin/main"])?;
        if !base.status.success() {
            return Err("no approved base: neither main nor origin/main exists".into());
        }
        String::from_utf8_lossy(&base.stdout).trim().to_owned()
    };
    let shown = git(&["show", &format!("{revision}:./{name}")])?;
    if !shown.status.success() {
        return Ok(None);
    }
    String::from_utf8(shown.stdout)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Checks `built` lock records against the adoption of `declarations` under
/// the records' compiler: it must exist, once, and hold exactly these
/// cubins, in a ledger `check_ledger` accepts against `base`.
pub fn check_adoption(
    built: &[Value],
    declarations: &[String],
    adoptions: &str,
    base: Option<&str>,
) -> Result<(), String> {
    let adoptions = check_ledger(adoptions, base)?;
    let current = adoption(built, declarations)?;
    let mut matching = adoptions.iter().filter(|adopted| {
        adopted.declarations == current.declarations && adopted.nvcc_sha256 == current.nvcc_sha256
    });
    match (matching.next(), matching.next()) {
        (Some(adopted), None) if adopted.cubins == current.cubins => Ok(()),
        (Some(_), None) => Err(format!(
            "the encoder cubins changed although declarations {declarations:?} are unchanged: \
             adopt the change through a declaration (a new variant id) or revert it"
        )),
        (None, _) => Err(format!(
            "no adopted encoder cubins for declarations {declarations:?} under this compiler: \
             a model's encoder cubins change only with its declarations; append an adoption \
             for their new digests"
        )),
        (Some(_), Some(_)) => Err(format!("declarations {declarations:?} are adopted twice")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str, closure: &str, cubin: &str) -> Value {
        json!({"name": name, "nvcc_sha256": "nvcc", "source_closure_sha256": closure,
               "cubin_sha256": cubin, "entries": []})
    }

    fn ledger(entries: &[(&[&str], &str, &str)]) -> Vec<Adoption> {
        let mut ledger = entries
            .iter()
            .map(|(declarations, closure, cubin)| Adoption {
                previous: String::new(),
                declarations: declarations.iter().map(|d| (*d).to_owned()).collect(),
                nvcc_sha256: "nvcc".into(),
                cubins: vec![Adopted {
                    name: "rows".into(),
                    source_closure_sha256: (*closure).into(),
                    cubin_sha256: (*cubin).into(),
                }],
            })
            .collect::<Vec<_>>();
        chain(&mut ledger);
        ledger
    }

    fn text(ledger: &[Adoption]) -> String {
        serde_json::to_string_pretty(ledger).unwrap() + "\n"
    }

    fn adoptions(entries: &[(&[&str], &str, &str)]) -> String {
        text(&ledger(entries))
    }

    #[test]
    fn a_lock_family_joins_its_provenance() {
        let directory = std::env::temp_dir().join(format!(
            "encoder-family-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let mut locked = record("rows", "c1", "k1");
        locked
            .as_object_mut()
            .unwrap()
            .remove("source_closure_sha256");
        let write = |name: &str, value: Value| {
            std::fs::write(directory.join(name), value.to_string()).unwrap();
        };
        write("encoder.lock", json!([locked]));
        write(
            "encoder.provenance.json",
            json!([{"name": "rows", "source_closure_sha256": "c1"}]),
        );
        let joined = read_family(&directory, "encoder").unwrap();
        assert_eq!(joined, vec![record("rows", "c1", "k1")]);
        write(
            "encoder.provenance.json",
            json!([{"name": "attention", "source_closure_sha256": "c1"}]),
        );
        let error = read_family(&directory, "encoder").unwrap_err();
        assert!(error.contains("no source closure"), "{error}");
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_lock_changes_only_with_its_declarations() {
        let declarations = ["d1".to_owned()];
        let adopted = adoptions(&[(&["d1"], "c1", "k1")]);
        assert!(
            check_adoption(&[record("rows", "c1", "k1")], &declarations, &adopted, None).is_ok()
        );
        // A refreshed lock under unchanged declarations: refused, even though
        // the lock and the build agree.
        let error = check_adoption(&[record("rows", "c2", "k2")], &declarations, &adopted, None)
            .unwrap_err();
        assert!(error.contains("unchanged"), "{error}");
        // A declaration change needs its own adoption, appended.
        let changed = ["d2".to_owned()];
        let error =
            check_adoption(&[record("rows", "c2", "k2")], &changed, &adopted, None).unwrap_err();
        assert!(error.contains("no adopted"), "{error}");
        let appended = adoptions(&[(&["d1"], "c1", "k1"), (&["d2"], "c2", "k2")]);
        assert!(
            check_adoption(
                &[record("rows", "c2", "k2")],
                &changed,
                &appended,
                Some(&adopted)
            )
            .is_ok()
        );
        // One adoption per declaration set and toolchain.
        let twice = adoptions(&[(&["d1"], "c1", "k1"), (&["d1"], "c2", "k2")]);
        assert!(
            check_adoption(&[record("rows", "c1", "k1")], &declarations, &twice, None).is_err()
        );
    }

    /// The confirm review's probe: arithmetic changed under unchanged
    /// declarations, the lock refreshed and the ledger's adoption rewritten
    /// to the new cubins. The rewritten ledger is a valid chain on its own;
    /// the approved base refuses it.
    #[test]
    fn a_rewritten_adoption_is_refused_against_the_approved_base() {
        let declarations = ["d1".to_owned()];
        let approved = adoptions(&[(&["d1"], "c1", "k1")]);
        let rewritten = adoptions(&[(&["d1"], "c2", "k2")]);
        let changed = [record("rows", "c2", "k2")];
        let error =
            check_adoption(&changed, &declarations, &rewritten, Some(&approved)).unwrap_err();
        assert!(error.contains("approved"), "{error}");
        // Appending after the approved entries passes.
        let appended = adoptions(&[(&["d1"], "c1", "k1"), (&["d2"], "c2", "k2")]);
        assert!(check_ledger(&appended, Some(&approved)).is_ok());
        assert!(check_ledger(&approved, Some(&approved)).is_ok());
        // Dropping an approved entry is refused.
        let dropped = adoptions(&[(&["d2"], "c2", "k2")]);
        assert!(check_ledger(&dropped, Some(&appended)).is_err());
    }

    #[test]
    fn the_digest_matches_the_ci_check() {
        // tests/test_encoder_adoptions.py pins the same value.
        let adoption = Adoption {
            previous: GENESIS.into(),
            declarations: vec!["d1".into(), "d2".into()],
            nvcc_sha256: "nvcc".into(),
            cubins: vec![Adopted {
                name: "rows".into(),
                source_closure_sha256: "c1".into(),
                cubin_sha256: "k1".into(),
            }],
        };
        assert_eq!(
            digest(&adoption),
            "476a1e5eb6fe93b6800edf0240350918279a5db218ac9e82ee5f7d0e9e12ca88"
        );
    }

    #[test]
    fn an_edit_before_the_last_adoption_breaks_the_chain() {
        let mut ledger = ledger(&[(&["d1"], "c1", "k1"), (&["d2"], "c2", "k2")]);
        assert!(check_ledger(&text(&ledger), None).is_ok());
        ledger[0].cubins[0].cubin_sha256 = "k3".into();
        let error = check_ledger(&text(&ledger), None).unwrap_err();
        assert!(error.contains("chain"), "{error}");
        // Not the canonical form: refused too.
        let compact = serde_json::to_string(&ledger).unwrap();
        assert!(check_ledger(&compact, None).is_err());
    }

    #[test]
    fn a_base_from_before_the_chain_is_compared_without_previous() {
        let chained = adoptions(&[(&["d1"], "c1", "k1")]);
        let mut unchained: Vec<Value> = serde_json::from_str(&chained).unwrap();
        unchained[0].as_object_mut().unwrap().remove("previous");
        let unchained = serde_json::to_string_pretty(&unchained).unwrap() + "\n";
        assert!(check_ledger(&chained, Some(&unchained)).is_ok());
        let other = adoptions(&[(&["d1"], "c9", "k9")]);
        assert!(check_ledger(&other, Some(&unchained)).is_err());
    }
}
