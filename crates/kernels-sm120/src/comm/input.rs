//! Comm transport inputs: the element type each rank reads and the ranks
//! exchange, or a push input policy that computes it (`cuda/comm/input/`).

/// A transport input. Entries take `policy` as their input template
/// argument and declare their input and inbox pointers as `cxx_type`;
/// `element_bytes` sizes the planners' inbox regions. A model whose
/// all-reduce input is another type adds a header and a value of this
/// type without editing the transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Input {
    /// The policy's header, relative to the CUDA source root.
    pub header: &'static str,
    /// The C++ input type or policy.
    pub policy: &'static str,
    /// The C++ element type of input and inbox pointers.
    pub cxx_type: &'static str,
    /// Bytes per element.
    pub element_bytes: u32,
}

impl Input {
    /// The cubin header line that declares the policy.
    pub fn include(&self) -> String {
        format!("#include \"{}\"", self.header)
    }
}

/// BF16 elements, read as they are. Every transport takes it.
pub const BF16: Input = Input {
    header: "comm/input/bf16.cuh",
    policy: "chop::comm::input::Bf16",
    cxx_type: "uint16_t",
    element_bytes: 2,
};

/// F32 elements, which the hyper-connection transports take; the push
/// transport exchanges BF16 rows and refuses them at compile time.
pub const F32: Input = Input {
    header: "comm/input/f32.cuh",
    policy: "chop::comm::input::F32",
    cxx_type: "float",
    element_bytes: 4,
};

/// Each row's eight preweighted routed rows as the push input of the
/// combine entries, summed in expert order and rounded to BF16
/// (`comm/input/routed_combine.cuh`).
pub const ROUTED_COMBINE_TOP8: Input = Input {
    header: "comm/input/routed_combine.cuh",
    policy: "chop::comm::input::RoutedCombine<8>",
    cxx_type: "uint16_t",
    element_bytes: 2,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn inputs_name_existing_headers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda");
        for input in [BF16, F32, ROUTED_COMBINE_TOP8] {
            assert!(root.join(input.header).is_file(), "{input:?}");
            assert_eq!(input.include(), format!("#include \"{}\"", input.header));
        }
    }
}
