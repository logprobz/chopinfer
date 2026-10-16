//! Static names built from constant parts. Catalog fields hold
//! `&'static str` names; a model whose names come from constant data (a
//! geometry's tag, not a literal) builds them at compile time with
//! [`const_name!`] or [`Text`], or interns names it formats ([`intern`]).

/// The bytes of `parts` joined, `N` of them in all; [`const_name!`] sizes
/// `N`.
pub const fn join<const N: usize>(parts: &[&str]) -> [u8; N] {
    let mut out = [0; N];
    let mut at = 0;
    let mut part = 0;
    while part < parts.len() {
        let bytes = parts[part].as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            out[at] = bytes[index];
            at += 1;
            index += 1;
        }
        part += 1;
    }
    assert!(at == N);
    out
}

/// The `&'static str` of constant string parts, joined at compile time.
#[macro_export]
macro_rules! const_name {
    ($($part:expr),+ $(,)?) => {{
        const BYTES: [u8; 0 $(+ $part.len())+] =
            $crate::names::join(&[$($part),+]);
        match std::str::from_utf8(&BYTES) {
            Ok(value) => value,
            Err(_) => panic!("kernel name must be UTF-8"),
        }
    }};
}

/// Intern dynamic entry names once; repeated catalog queries reuse their storage.
pub fn intern(value: impl Into<String>) -> &'static str {
    use std::{
        collections::BTreeSet,
        sync::{Mutex, OnceLock},
    };
    static NAMES: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
    let value = value.into();
    let mut names = NAMES.get_or_init(Default::default).lock().unwrap();
    if let Some(name) = names.get(value.as_str()) {
        return name;
    }
    let name = Box::leak(value.into_boxed_str());
    names.insert(name);
    name
}

/// Static storage for profile names derived from a generated geometry list.
#[derive(Clone, Copy)]
pub struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}
impl<const N: usize> Text<N> {
    pub const fn new(parts: &[&str]) -> Self {
        let mut out = Self {
            bytes: [0; N],
            len: 0,
        };
        let mut part = 0;
        while part < parts.len() {
            let mut i = 0;
            while i < parts[part].len() {
                out.bytes[out.len] = parts[part].as_bytes()[i];
                out.len += 1;
                i += 1;
            }
            part += 1;
        }
        out
    }
    pub const fn as_str(&self) -> &str {
        match std::str::from_utf8(self.bytes.split_at(self.len).0) {
            Ok(value) => value,
            Err(_) => panic!("kernel profile name must be UTF-8"),
        }
    }
}
