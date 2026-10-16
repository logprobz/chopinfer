//! Versioned local preparation protocol. Providers own model compatibility.
pub use runtime_bundle_sdk::native_note::encode_native_note;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;

pub const VERSION: u32 = 1;
/// The protocol version that adds direct loading: descriptions state their
/// capabilities, and [`Request::PrepareDirect`] exists. A helper answers a
/// version 1 request as version 1 does, so hosts from before it keep
/// working; hosts send version 2 only to discover and prepare direct loads.
pub const DIRECT_VERSION: u32 = 2;
pub const MAX_RECORD_BYTES: usize = 1024 * 1024;
pub const NOTE_SECTION: &str = ".note.chopinfer.model";
/// A helper that answers [`Request::PrepareDirect`] describes this capability.
pub const PREPARE_DIRECT_CAPABILITY: &str = "prepare-direct";
/// A helper that answers [`Request::DirectProfile`] describes this capability.
pub const DIRECT_PROFILE_CAPABILITY: &str = "direct-profile";

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProfile {
    pub profile: String,
    pub tensor_parallel_size: u32,
    pub weight_contract: String,
    pub package_contract: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImplementationContract {
    pub selection: String,
    pub dependencies: String,
    /// Digest of the publisher's bounded capability section. Its contents are
    /// interpreted by the compiler SDK and bundle, never by the serving engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeContracts {
    pub version: u32,
    /// Empty for a module that publishes only its implementation contract.
    pub profiles: Vec<NativeProfile>,
    /// Opaque publisher-selected implementation contract. Providers interpret
    /// it; shared serving code does not inspect operation precision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_contract: Option<ImplementationContract>,
}

impl NativeContracts {
    pub fn validate(&self) -> Result<()> {
        if self.version != VERSION
            || self.profiles.is_empty() && self.implementation_contract.is_none()
            || self.profiles.len() > 64
        {
            return Err("unsupported or empty native preparation contract".into());
        }
        if self.implementation_contract.as_ref().is_some_and(|value| {
            !valid_digest(&value.selection)
                || !valid_digest(&value.dependencies)
                || value
                    .capabilities
                    .as_ref()
                    .is_some_and(|v| !valid_digest(v))
        }) {
            return Err("invalid native implementation contract digest".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for p in &self.profiles {
            if p.profile.is_empty()
                || p.profile.len() > 128
                || !names.insert(&p.profile)
                || p.tensor_parallel_size == 0
                || p.tensor_parallel_size > 64
                || !valid_digest(&p.weight_contract)
                || !valid_digest(&p.package_contract)
            {
                return Err("invalid or duplicate native preparation profile".into());
            }
        }
        Ok(())
    }
}

pub fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// An input modality a direct package serves. Text is always served; each
/// other modality needs the encoder the checkpoint provides for it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

impl Modality {
    pub const ALL: [Self; 4] = [Self::Text, Self::Image, Self::Video, Self::Audio];

    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// The comma list of `modalities`, in their order.
    pub fn list(modalities: &BTreeSet<Self>) -> String {
        modalities
            .iter()
            .map(|modality| modality.name())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The modalities `--modalities` selects of those a checkpoint provides:
/// `auto`, the default, serves text and image; `all` serves every modality;
/// a list serves exactly the ones it names. On the wire `auto` is left out,
/// `all` is the string and a list is an array of names.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SelectionWire", into = "SelectionWire")]
pub enum ModalitySelection {
    #[default]
    Auto,
    All,
    Only(BTreeSet<Modality>),
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum SelectionWire {
    Keyword(String),
    List(BTreeSet<Modality>),
}

impl TryFrom<SelectionWire> for ModalitySelection {
    type Error = String;

    fn try_from(value: SelectionWire) -> std::result::Result<Self, String> {
        match value {
            SelectionWire::Keyword(keyword) if keyword == "auto" => Ok(Self::Auto),
            SelectionWire::Keyword(keyword) if keyword == "all" => Ok(Self::All),
            SelectionWire::Keyword(keyword) => {
                Err(format!("unknown modality selection {keyword:?}"))
            }
            SelectionWire::List(modalities) => Ok(Self::Only(modalities)),
        }
    }
}

impl From<ModalitySelection> for SelectionWire {
    fn from(value: ModalitySelection) -> Self {
        match value {
            ModalitySelection::Auto => Self::Keyword("auto".into()),
            ModalitySelection::All => Self::Keyword("all".into()),
            ModalitySelection::Only(modalities) => Self::List(modalities),
        }
    }
}

impl ModalitySelection {
    pub fn is_auto(&self) -> bool {
        *self == Self::Auto
    }

    /// The modalities a list names, or None for `auto` and `all`.
    pub fn listed(&self) -> Option<&BTreeSet<Modality>> {
        match self {
            Self::Only(modalities) => Some(modalities),
            Self::Auto | Self::All => None,
        }
    }

    /// The selection `--modalities` names: `auto`, `all`, or a comma list of
    /// `text`, `image`, `video` and `audio` that names text and each
    /// modality once.
    pub fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "auto" => return Ok(Self::Auto),
            "all" => return Ok(Self::All),
            _ => {}
        }
        let mut modalities = BTreeSet::new();
        for name in value.split(',') {
            let modality = Modality::ALL
                .into_iter()
                .find(|modality| modality.name() == name)
                .ok_or_else(|| {
                    format!(
                        "--modalities must be auto, all or a comma list of text, image, video \
                         and audio, not {value:?}"
                    )
                })?;
            if !modalities.insert(modality) {
                return Err(format!("--modalities names {name} twice"));
            }
        }
        if !modalities.contains(&Modality::Text) {
            return Err("--modalities must include text".into());
        }
        Ok(Self::Only(modalities))
    }
}

/// Select or emit a package without rank images for loading the checkpoint
/// directory `path` directly. The provider neither packs nor hashes tensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareDirectRequest {
    pub path: PathBuf,
    pub cache_root: PathBuf,
    pub tensor_parallel_size: u32,
    /// Opaque activation policy, interpreted by the provider rather than
    /// the serving executor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_policy: Option<String>,
    /// The modalities the package serves, which leaves out every encoder
    /// none of them needs. Naming one the checkpoint does not provide fails
    /// preparation.
    #[serde(default, skip_serializing_if = "ModalitySelection::is_auto")]
    pub modalities: ModalitySelection,
}

impl PrepareDirectRequest {
    /// The modalities the package serves, of those the checkpoint
    /// `provides` (text and those its encoders serve): text and image for
    /// `auto` (`runtime_bundle_sdk::modalities`), all of them for `all`,
    /// else the listed ones, each of which it must provide.
    pub fn served_modalities(
        &self,
        provides: &BTreeSet<Modality>,
    ) -> std::result::Result<BTreeSet<Modality>, String> {
        let requested = match &self.modalities {
            ModalitySelection::Auto => {
                return Ok(provides
                    .iter()
                    .copied()
                    .filter(|modality| {
                        runtime_bundle_sdk::modalities::AUTO.contains(&modality.name())
                    })
                    .collect());
            }
            ModalitySelection::All => return Ok(provides.clone()),
            ModalitySelection::Only(requested) => requested,
        };
        match requested.difference(provides).next() {
            Some(missing) => Err(format!(
                "--modalities names {}, which this checkpoint does not provide; it provides {}",
                missing.name(),
                Modality::list(provides)
            )),
            None => Ok(requested.clone()),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub seconds: f64,
    pub invocations: u64,
    pub bytes_read: u64,
    pub bytes_hashed: u64,
    pub status: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurements {
    pub total_seconds: f64,
    pub stages: BTreeMap<String, Stage>,
    pub decisions: Vec<String>,
}

/// The serving profile a family selects for a checkpoint and topology: the
/// entry `name` of the model's `chopinfer-serve-profiles-v2` file, which
/// serving package paths use for the same checkpoint and rank count. The host
/// interprets `profile` as it interprets `serve --profile`, as defaults that
/// explicit options override, without the entry's bundle, topology, device
/// and display-name fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectProfile {
    pub name: String,
    pub profile: serde_json::Value,
}

/// A package without rank images, selected for one checkpoint directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectPackage {
    pub bundle: PathBuf,
    pub cache_root: PathBuf,
    /// SHA-256 of the package's canonical manifest; the bundle is `p-<identity>`.
    pub package_identity: String,
    /// The direct load's checkpoint identity: the SHA-256 of every
    /// safetensors file's name, length and header. It authenticates no
    /// tensor payload.
    pub checkpoint_identity: String,
    /// SHA-256 of every other file the family may have read (configuration,
    /// index, tokenizer), measured before emission; see
    /// `runtime_weights::checkpoint::metadata_identity`.
    pub metadata_identity: String,
    pub bundle_id: String,
    pub tensor_parallel_size: u32,
    pub profile: DirectProfile,
    pub qualification: String,
    pub measurements: Measurements,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheEntry {
    pub identity: String,
    pub kind: String,
    pub state: String,
    pub logical_bytes: u64,
    pub allocated_bytes: u64,
    pub last_selected: u64,
    pub references: Vec<String>,
    pub protected: bool,
    pub selected_for_pruning: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheUsage {
    pub entries: Vec<CacheEntry>,
    pub logical_bytes: u64,
    pub unique_allocated_bytes: u64,
    pub reclaimable_bytes: u64,
    pub external_alias_bytes: u64,
    pub removed: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Request {
    Describe,
    Probe {
        path: PathBuf,
        tensor_parallel_size: u32,
    },
    PrepareDirect(PrepareDirectRequest),
    /// The serving profile a direct package for this checkpoint would carry,
    /// without emitting it, so the host can check the invocation's options
    /// against it before any package work.
    DirectProfile(PrepareDirectRequest),
    CacheList {
        cache_root: PathBuf,
    },
    CachePrune {
        cache_root: PathBuf,
        apply: bool,
        identities: Vec<String>,
    },
    Acknowledge {
        package_identity: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub version: u32,
    pub request: Request,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Response {
    Description {
        version: u32,
        provider: String,
        profiles: Vec<String>,
        /// Requests the helper answers beyond version 1, such as
        /// [`PREPARE_DIRECT_CAPABILITY`]; sent only in a version 2 reply,
        /// since version 1 hosts refuse unknown fields.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        capabilities: Vec<String>,
    },
    Probe {
        compatible: bool,
        reason: String,
    },
    PreparedDirect(Box<DirectPackage>),
    DirectProfileSelected(DirectProfile),
    Cache(CacheUsage),
    Error {
        message: String,
        measurements: Measurements,
    },
}

pub fn read_record<T: serde::de::DeserializeOwned>(input: &mut impl BufRead) -> Result<T> {
    let mut bytes = Vec::new();
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Err("truncated preparation record".into());
        }
        let end = available.iter().position(|&b| b == b'\n').map(|i| i + 1);
        let count = end.unwrap_or(available.len());
        if bytes.len() + count > MAX_RECORD_BYTES {
            return Err("preparation record exceeds limit".into());
        }
        bytes.extend_from_slice(&available[..count]);
        input.consume(count);
        if end.is_some() {
            return Ok(serde_json::from_value(crate::parse_unique_json(&bytes)?)?);
        }
    }
}

pub fn write_record(value: &impl Serialize, output: &mut impl Write) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() >= MAX_RECORD_BYTES {
        return Err("preparation record exceeds limit".into());
    }
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

pub fn read_request(input: &mut impl BufRead) -> Result<Request> {
    read_envelope(input).map(|envelope| envelope.request)
}

/// A request of protocol version [`VERSION`] or [`DIRECT_VERSION`], with its
/// version, which selects the form of the reply.
pub fn read_envelope(input: &mut impl BufRead) -> Result<Envelope> {
    let envelope: Envelope = read_record(input)?;
    if !(VERSION..=DIRECT_VERSION).contains(&envelope.version) {
        return Err("unsupported preparation protocol version".into());
    }
    if envelope.version < DIRECT_VERSION
        && matches!(
            envelope.request,
            Request::PrepareDirect(_) | Request::DirectProfile(_)
        )
    {
        return Err("direct preparation needs protocol version 2".into());
    }
    Ok(envelope)
}

/// An active package lease is held through the whole serving lifetime. A helper
/// keeps its lease until the host acquires this guard and acknowledges transfer.
#[derive(Debug)]
pub struct CacheLease {
    _files: Vec<std::fs::File>,
}

pub fn managed_key(value: &str) -> bool {
    ["p-", "w-", "s-"]
        .iter()
        .any(|prefix| value.strip_prefix(prefix).is_some_and(valid_digest))
}

pub fn cache_lock(root: &std::path::Path, name: &str) -> Result<std::fs::File> {
    if name != "gate" && !managed_key(name) && !name.strip_prefix("i-").is_some_and(valid_digest) {
        return Err("invalid managed cache lock key".into());
    }
    let directory = root.join("locks");
    if !std::fs::symlink_metadata(root)?.is_dir()
        || !std::fs::symlink_metadata(&directory)?.is_dir()
    {
        return Err("managed cache and locks must be real directories".into());
    }
    Ok(runtime_weights::open_regular_nofollow(
        &directory.join(format!("{name}.lock")),
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true),
    )?)
}

impl CacheLease {
    pub fn acquire(root: &std::path::Path, keys: &[String]) -> Result<Self> {
        let gate = cache_lock(root, "gate")?;
        gate.lock_shared()?;
        let mut keys = keys.to_vec();
        keys.sort();
        keys.dedup();
        let mut files = Vec::new();
        for key in keys {
            if !managed_key(&key) {
                return Err("invalid cache lease key".into());
            }
            let file = cache_lock(root, &key)?;
            file.lock_shared()?;
            files.push(file);
        }
        Ok(Self { _files: files })
    }

    /// Explicit bundle selection gets the same protection as `serve --model`.
    pub fn for_bundle(bundle: &std::path::Path) -> Result<Option<Self>> {
        let Some(root) = bundle.parent() else {
            return Ok(None);
        };
        let Some(name) = bundle.file_name().and_then(|s| s.to_str()) else {
            return Ok(None);
        };
        if !name.starts_with("p-") || !managed_key(name) {
            return Ok(None);
        }
        match std::fs::symlink_metadata(root.join("cache-version")) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
            Ok(_) => {
                let file = runtime_weights::open_nofollow(&root.join("cache-version"))?;
                let mut bytes = String::new();
                use std::io::Read;
                file.take(32).read_to_string(&mut bytes)?;
                if bytes != "chopinfer-preparation-cache-v1\n" {
                    return Err("unsupported preparation cache".into());
                }
                Ok(Some(Self::acquire(root, &[name.into()])?))
            }
        }
    }
}

/// Override selection belongs to the caller; these are the shared default locations.
pub fn default_cache_root() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("CHOPINFER_DATA") {
        return Ok(PathBuf::from(root).join("model-cache"));
    }
    if let Some(root) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(root).join("chopinfer/models"));
    }
    Ok(
        PathBuf::from(std::env::var_os("HOME").ok_or("set a model cache directory")?)
            .join(".cache/chopinfer/models"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_protocol_refuses_unknown_versions_and_truncated_records() {
        assert!(
            read_request(&mut &b"{\"version\":9,\"request\":{\"command\":\"describe\"}}\n"[..])
                .is_err()
        );
        assert!(
            read_request(&mut &b"{\"version\":1,\"request\":{\"command\":\"describe\"}}"[..])
                .is_err()
        );
        assert!(read_request(&mut vec![b' '; MAX_RECORD_BYTES + 1].as_slice()).is_err());
        // Version 2 adds direct preparation; version 1 keeps its requests.
        let envelope =
            read_envelope(&mut &b"{\"version\":2,\"request\":{\"command\":\"describe\"}}\n"[..])
                .unwrap();
        assert_eq!(envelope.version, DIRECT_VERSION);
        assert!(
            read_request(&mut &b"{\"version\":3,\"request\":{\"command\":\"describe\"}}\n"[..])
                .is_err()
        );
        let direct = |version: u32| {
            let mut bytes = Vec::new();
            write_record(
                &Envelope {
                    version,
                    request: Request::PrepareDirect(PrepareDirectRequest {
                        path: "m".into(),
                        cache_root: "c".into(),
                        tensor_parallel_size: 4,
                        activation_policy: None,
                        modalities: ModalitySelection::Auto,
                    }),
                },
                &mut bytes,
            )
            .unwrap();
            read_request(&mut bytes.as_slice())
        };
        assert!(direct(1).is_err());
        assert!(matches!(direct(2).unwrap(), Request::PrepareDirect(_)));
        let mut bytes = Vec::new();
        write_record(
            &Envelope {
                version: VERSION,
                request: Request::Describe,
            },
            &mut bytes,
        )
        .unwrap();
        assert!(matches!(
            read_request(&mut bytes.as_slice()).unwrap(),
            Request::Describe
        ));
    }

    #[test]
    fn modalities_are_auto_all_or_a_list_with_text_each_named_once() {
        use Modality::*;
        use ModalitySelection::{All, Auto, Only};
        assert_eq!(ModalitySelection::parse("auto"), Ok(Auto));
        assert_eq!(ModalitySelection::parse("all"), Ok(All));
        assert_eq!(
            ModalitySelection::parse("audio,text"),
            Ok(Only(BTreeSet::from([Text, Audio])))
        );
        assert_eq!(
            ModalitySelection::parse("text,image,video,audio")
                .map(|selection| Modality::list(selection.listed().unwrap())),
            Ok("text,image,video,audio".into())
        );
        for (value, refusal) in [
            ("image", "must include text"),
            ("text,text", "names text twice"),
            ("text,speech", "comma list"),
            ("", "comma list"),
            ("text,", "comma list"),
            ("Text", "comma list"),
            ("text,all", "comma list"),
        ] {
            let error = ModalitySelection::parse(value).unwrap_err();
            assert!(error.contains(refusal), "{value:?}: {error}");
        }
        let request = |modalities| PrepareDirectRequest {
            path: "m".into(),
            cache_root: "c".into(),
            tensor_parallel_size: 2,
            activation_policy: None,
            modalities,
        };
        let every = BTreeSet::from(Modality::ALL);
        // Auto serves text and image of what the checkpoint provides; all
        // serves everything it provides.
        assert_eq!(
            request(Auto).served_modalities(&every),
            Ok(BTreeSet::from([Text, Image]))
        );
        assert_eq!(request(All).served_modalities(&every), Ok(every.clone()));
        let text = BTreeSet::from([Text]);
        assert_eq!(request(Auto).served_modalities(&text), Ok(text.clone()));
        assert_eq!(request(All).served_modalities(&text), Ok(text.clone()));
        let provides = BTreeSet::from([Text, Image]);
        assert_eq!(
            request(Only(BTreeSet::from([Text]))).served_modalities(&provides),
            Ok(BTreeSet::from([Text]))
        );
        let refusal = request(Only(BTreeSet::from([Text, Audio])))
            .served_modalities(&provides)
            .unwrap_err();
        assert!(refusal.contains("names audio") && refusal.contains("provides text,image"));
        // Auto stays off the wire, so helpers from before the field accept it.
        let auto = serde_json::to_value(request(Auto)).unwrap();
        assert!(auto.get("modalities").is_none());
        let text = serde_json::to_value(request(Only(BTreeSet::from([Audio, Text])))).unwrap();
        assert_eq!(text["modalities"], serde_json::json!(["text", "audio"]));
        let all = serde_json::to_value(request(All)).unwrap();
        assert_eq!(all["modalities"], "all");
        for selection in [Auto, All, Only(BTreeSet::from([Text, Video]))] {
            let value = serde_json::to_value(request(selection.clone())).unwrap();
            let read: PrepareDirectRequest = serde_json::from_value(value).unwrap();
            assert_eq!(read.modalities, selection);
        }
        let unknown = serde_json::json!({"path": "m", "cache_root": "c",
            "tensor_parallel_size": 2, "modalities": "every"});
        assert!(serde_json::from_value::<PrepareDirectRequest>(unknown).is_err());
    }

    #[test]
    fn a_contract_only_note_needs_its_implementation_contract() {
        let digest = "a".repeat(64);
        let mut contracts = NativeContracts {
            version: VERSION,
            profiles: Vec::new(),
            implementation_contract: None,
        };
        assert!(contracts.validate().is_err());
        contracts.implementation_contract = Some(ImplementationContract {
            selection: digest.clone(),
            dependencies: digest,
            capabilities: None,
        });
        contracts.validate().unwrap();
        contracts
            .implementation_contract
            .as_mut()
            .unwrap()
            .selection = "a".into();
        assert!(contracts.validate().is_err());
    }
}
