//! Bounded ELF note inspection without loading executable code.
use crate::files::Result;
use crate::{FileIdentity, StagedContentV1};
use kernel_core::precision::capability::{
    CAPABILITY_SECTION, CapabilityDocument, CapabilityInventory, MAX_CAPABILITY_BYTES,
};
use runtime_bundle::preparation::{NOTE_SECTION, NativeContracts, NativeProfile};
use runtime_weights::source_file::SourceFile;
use std::path::{Path, PathBuf};

pub struct NativeModule {
    source: SourceFile,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub contracts: NativeContracts,
}

impl NativeModule {
    pub fn inspect(path: &Path) -> Result<Self> {
        let source = SourceFile::open(path)?;
        let bytes = source.metadata()?.len();
        let contracts = read_note(&source)?;
        let sha256 = runtime_weights::hash_range(&source, 0, bytes)?;
        source.verify()?;
        Ok(Self {
            source,
            path: path.into(),
            sha256,
            bytes,
            contracts,
        })
    }

    pub fn require(&self, expected: &NativeProfile) -> Result<()> {
        self.verify()?;
        let actual = self
            .contracts
            .profiles
            .iter()
            .find(|p| p.profile == expected.profile)
            .ok_or("native module does not support the requested preparation profile")?;
        if actual.weight_contract != expected.weight_contract
            || actual.tensor_parallel_size != expected.tensor_parallel_size
        {
            return Err("native weight-layout contract mismatch; install a matching family helper and module".into());
        }
        if actual.package_contract != expected.package_contract {
            return Err("native package-declaration contract mismatch; install a matching family helper (completed weights remain reusable)".into());
        }
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        Ok(self.source.verify()?)
    }

    pub fn require_implementation_contract(&self, expected: &str) -> Result<()> {
        self.verify()?;
        if !runtime_bundle::preparation::valid_digest(expected)
            || self
                .contracts
                .implementation_contract
                .as_ref()
                .map(|c| c.selection.as_str())
                != Some(expected)
        {
            return Err("native implementation contract mismatch; install a matching provider and prebuilt module (checkpoint preparation never compiles kernels)".into());
        }
        Ok(())
    }

    pub fn staged(&self) -> Result<StagedContentV1> {
        self.verify()?;
        let name = self
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid native module filename")?;
        Ok(StagedContentV1::authenticated_file(
            &self.path,
            FileIdentity::new(name, self.bytes, &self.sha256)?,
        ))
    }

    /// Read the supplied capability domain without loading executable code or
    /// importing kernel constructors. The native note authenticates its bytes.
    pub fn precision_inventory(&self) -> Result<CapabilityInventory> {
        self.verify()?;
        let contract = self
            .contracts
            .implementation_contract
            .as_ref()
            .ok_or("native module has no supplied implementation contract")?;
        let expected = contract.capabilities.as_ref()
            .ok_or("native module has no portable precision capabilities; install a matching prebuilt module")?;
        let bytes = read_section(
            &self.source,
            CAPABILITY_SECTION,
            1,
            MAX_CAPABILITY_BYTES as u64,
        )?;
        if runtime_weights::sha256_bytes(&bytes) != *expected {
            return Err("native precision capability digest mismatch".into());
        }
        let document: CapabilityDocument = serde_json::from_slice(&bytes)?;
        document.validate()?;
        let available = document.entries();
        let dependency = crate::preparation::cache::digest(&(
            "native-operation-bindings-v1",
            &contract.selection,
            &contract.dependencies,
            expected,
        ))?;
        self.verify()?;
        Ok(CapabilityInventory::new(document, available, dependency)?)
    }
}

fn range(file: &std::fs::File, offset: u64, count: u64, limit: u64) -> Result<Vec<u8>> {
    if count > limit
        || offset
            .checked_add(count)
            .is_none_or(|end| end > file.metadata().map_or(0, |m| m.len()))
    {
        return Err("native ELF range exceeds file or metadata limit".into());
    }
    let mut bytes = vec![0; count as usize];
    runtime_weights::read_exact_at(file, &mut bytes, offset)?;
    Ok(bytes)
}

fn u16le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("bounded ELF field"),
    )
}
fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("bounded ELF field"),
    )
}
fn u64le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounded ELF field"),
    )
}

pub fn read_note(file: &std::fs::File) -> Result<NativeContracts> {
    let note = read_section(file, NOTE_SECTION, 7, 64 * 1024)?;
    decode_note(&note)
}

fn read_section(file: &std::fs::File, name: &str, kind: u32, limit: u64) -> Result<Vec<u8>> {
    let header = range(file, 0, 64, 64)?;
    if &header[..6] != b"\x7fELF\x02\x01" || u16le(&header, 16) != 3 || u16le(&header, 18) != 62 {
        return Err("native preparation requires a little-endian x86-64 ELF shared module".into());
    }
    let count = u16le(&header, 60) as u64;
    let strings = u16le(&header, 62) as usize;
    if u16le(&header, 58) != 64 || count == 0 || count > 4096 || strings >= count as usize {
        return Err("unsupported native ELF section table".into());
    }
    let sections = range(file, u64le(&header, 40), count * 64, 4096 * 64)?;
    let strings = &sections[strings * 64..(strings + 1) * 64];
    let strings = range(file, u64le(strings, 24), u64le(strings, 32), 1024 * 1024)?;
    let mut found = None;
    for section in sections.chunks_exact(64) {
        let offset = u32le(section, 0) as usize;
        let tail = strings
            .get(offset..)
            .ok_or("invalid ELF section name offset")?;
        let end = tail
            .iter()
            .position(|&b| b == 0)
            .ok_or("unterminated ELF section name")?;
        if &tail[..end] != name.as_bytes() {
            continue;
        }
        if found.is_some() {
            return Err(format!("duplicate native section {name}").into());
        }
        if u32le(section, 4) != kind {
            return Err(format!("native section {name} has the wrong ELF section type").into());
        }
        found = Some(range(file, u64le(section, 24), u64le(section, 32), limit)?);
    }
    found.ok_or_else(|| {
        if name == NOTE_SECTION {
            "native module has no preparation contract note; install a matching prebuilt module"
                .into()
        } else {
            format!(
                "native module has no supplied section {name}; install a matching prebuilt module"
            )
            .into()
        }
    })
}

pub fn decode_note(note: &[u8]) -> Result<NativeContracts> {
    if note.len() < 24
        || u32le(note, 0) != 10
        || u32le(note, 8) != 1
        || &note[12..22] != b"CHOPINFER\0"
    {
        return Err("malformed native preparation note".into());
    }
    let length = u32le(note, 4) as usize;
    let end = 24_usize.checked_add(length).ok_or("native note overflow")?;
    if length > 64 * 1024 || end > note.len() || end.next_multiple_of(4) != note.len() {
        return Err("invalid native preparation note size".into());
    }
    let contracts: NativeContracts = serde_json::from_slice(&note[24..end])?;
    contracts.validate()?;
    Ok(contracts)
}

pub use runtime_bundle::preparation::encode_native_note;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_native_metadata_never_becomes_a_contract() {
        for bytes in [&b""[..], &b"CHOPINFER"[..], &[0; 24][..]] {
            assert!(decode_note(bytes).is_err());
        }
    }
}
