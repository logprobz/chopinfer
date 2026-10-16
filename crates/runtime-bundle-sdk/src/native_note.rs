//! Reviewed embedding of immutable native model metadata.

/// Embed generated ELF model note bytes, optionally with a precision document.
/// Supply `include_bytes!` expressions at the call site so source analysis can
/// resolve the concrete generated files.
#[macro_export]
macro_rules! embed_note {
    ($note:expr) => {
        // SAFETY-REVIEWED: Immutable aligned data in a non-executable ELF section, with no exported ABI symbol.
        #[allow(unsafe_code)]
        const _: () = {
            #[repr(C, align(4))]
            struct Note<const N: usize>([u8; N]);
            #[used]
            // SAFETY-REVIEWED: The linker section contains only the generated note bytes.
            #[unsafe(link_section = ".note.chopinfer.model")]
            static NOTE: Note<{ ($note).len() }> = Note(*($note));
        };
    };
    ($note:expr, precision = $precision:expr) => {
        $crate::embed_note!($note);
        // SAFETY-REVIEWED: Immutable data in a non-executable ELF section, with no exported ABI symbol.
        #[allow(unsafe_code)]
        const _: () = {
            #[used]
            // SAFETY-REVIEWED: The linker section contains only the generated precision document.
            #[unsafe(link_section = ".chopinfer.precision")]
            static CAPABILITIES: [u8; ($precision).len()] = *($precision);
        };
    };
}

/// Encode the native model contract's standard ELF note without changing its
/// JSON bytes. The owner and description are padded to four-byte boundaries.
pub fn encode_native_note(description: &[u8]) -> Vec<u8> {
    let owner = b"CHOPINFER\0";
    let mut note = Vec::new();
    note.extend_from_slice(&(owner.len() as u32).to_le_bytes());
    note.extend_from_slice(
        &u32::try_from(description.len())
            .expect("bounded native note")
            .to_le_bytes(),
    );
    note.extend_from_slice(&1_u32.to_le_bytes());
    note.extend_from_slice(owner);
    note.resize(note.len().next_multiple_of(4), 0);
    note.extend_from_slice(description);
    note.resize(note.len().next_multiple_of(4), 0);
    note
}
