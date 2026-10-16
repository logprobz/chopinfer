//! Retained source descriptors bound to the metadata observed before parsing.
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Stamp {
    length: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

impl Stamp {
    fn read(metadata: &Metadata) -> io::Result<Self> {
        if !metadata.is_file() {
            return Err(io::Error::other("source is not a regular file"));
        }
        Ok(Self {
            length: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            identity: {
                use std::os::unix::fs::MetadataExt;
                (
                    metadata.dev(),
                    metadata.ino(),
                    metadata.ctime(),
                    metadata.ctime_nsec(),
                )
            },
        })
    }
}

/// A source remains valid only while both its retained inode and pathname agree.
/// Writers need not cooperate; a detected change aborts publication. As with any
/// metadata check, this is not an immutable snapshot against privileged writers.
#[derive(Clone, Debug)]
pub struct SourceFile {
    file: Arc<File>,
    path: PathBuf,
    stamp: Stamp,
}

impl SourceFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let stamp = Stamp::read(&fs::symlink_metadata(path)?)?;
        let source = Self {
            file: Arc::new(crate::open_nofollow(path)?),
            path: path.into(),
            stamp,
        };
        source.verify()?;
        Ok(source)
    }

    pub fn verify(&self) -> io::Result<()> {
        if Stamp::read(&self.file.metadata()?)? != self.stamp
            || Stamp::read(&fs::symlink_metadata(&self.path)?)? != self.stamp
        {
            return Err(io::Error::other(format!(
                "source changed: {}",
                self.path.display()
            )));
        }
        Ok(())
    }
}

impl std::ops::Deref for SourceFile {
    type Target = File;
    fn deref(&self) -> &File {
        &self.file
    }
}

impl Read for SourceFile {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        (&*self.file).read(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_header_cannot_be_replaced_before_payload_read() {
        let path = std::env::temp_dir().join(format!(
            "source-header-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let source_path = path.join("source");
        fs::write(&source_path, b"header:payload").unwrap();
        let mut source = SourceFile::open(&source_path).unwrap();
        let mut header = [0; 7];
        source.read_exact(&mut header).unwrap();
        assert_eq!(&header, b"header:");
        fs::write(path.join("replacement"), b"changed:bytes!").unwrap();
        fs::rename(path.join("replacement"), &source_path).unwrap();
        assert!(source.verify().is_err());
        fs::remove_dir_all(path).unwrap();
    }
}
