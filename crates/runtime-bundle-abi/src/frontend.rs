//! The chat frontend a bundle exports: rendering a chat request into a
//! prompt, a request's tool-call constraint, and an incremental parser of the
//! request's output.
//!
//! A bundle that owns its chat format exports `chop_bundle_frontend_api_v1`
//! (capability [`FRONTEND_CAPABILITY_V1`]), which returns a static
//! [`BundleFrontendTableV1`]. The host opens one frontend over the package's
//! authenticated frontend component, renders requests and asks for their
//! constraints through it, and opens one parser per request, to which it
//! pushes the request's output text as it becomes stable. Every request,
//! answer and refusal crosses as a byte document; this module fixes how they
//! cross, and [`crate::frontend_document`] what they contain. A constraint
//! answer is a `runtime-grammar` wire document.
//!
//! A bundle implements [`FrontendV1`] and [`FrontendParserV1`] in safe Rust
//! and exports them with [`crate::export_frontend_v1!`], which catches every
//! panic. The host checks a loaded bundle's table with
//! [`crate::BundleLibraryV1::frontend_v1`] (or builds one in process with
//! [`FrontendExportV1::in_process`]) and calls it through
//! [`FrontendHandleV1`] and [`FrontendParserHandleV1`], which close what they
//! opened exactly once and tell a refusal document apart from a failed call.
//! The export and every handle hold the loaded module, so it stays mapped
//! until the last of them is gone.
//! `include/chop_bundle_v1.h` states the contract.

use std::ffi::c_void;
use std::fmt;
use std::marker::{PhantomData, PhantomPinned};

use crate::BundleStatusV1;
pub use crate::trampoline::{
    FrontendExportV1, FrontendHandleV1, FrontendParserHandleV1, frontend_table_v1,
};
pub use crate::{
    BundleFrontendApiFnV1, BundleFrontendCloseFnV1, BundleFrontendDocumentFnV1,
    BundleFrontendEmitFnV1, BundleFrontendOpenFnV1, BundleFrontendParserCloseFnV1,
    BundleFrontendParserOpenFnV1, BundleFrontendParserPushFnV1,
};

/// Capability a package states when its bundle exports the frontend.
pub const FRONTEND_CAPABILITY_V1: &str = "chop.frontend.v1";
/// The optional table export, NUL-terminated for `dlsym`.
pub const FRONTEND_API_SYMBOL_V1: &[u8] = b"chop_bundle_frontend_api_v1\0";
/// The layout of the table and of every sink.
pub const FRONTEND_LAYOUT_V1: u32 = 1;
/// Marks a request's last push.
pub const FRONTEND_PUSH_FLAG_V1_FINAL: u32 = 1;
/// Every push flag V1 defines; a push with another bit set is refused.
pub const FRONTEND_PUSH_FLAGS_V1_KNOWN: u32 = FRONTEND_PUSH_FLAG_V1_FINAL;

/// The longest refusal document the host takes from any entry; a longer one
/// is refused before it is copied, and the call fails as
/// [`FrontendCallErrorV1::Oversized`]. Each answer has the limit the host
/// passes to the call.
pub const FRONTEND_REFUSAL_BYTES_MAX_V1: usize = 1 << 20;
/// The longest answer the host takes from `open`.
pub const FRONTEND_OPENED_BYTES_MAX_V1: usize = 1 << 16;

/// An opened frontend, owned by the bundle.
#[repr(C)]
pub struct BundleFrontendV1 {
    _opaque: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// An opened parser, owned by the bundle.
#[repr(C)]
pub struct BundleFrontendParserV1 {
    _opaque: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Receives one call's answer (`emit`) or refusal (`emit_error`). The host
/// copies the document before either returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BundleFrontendSinkV1 {
    pub struct_size: u32,
    /// [`FRONTEND_LAYOUT_V1`]; the bundle refuses another before reading
    /// the request.
    pub layout: u32,
    pub context: *mut c_void,
    pub emit: Option<BundleFrontendEmitFnV1>,
    pub emit_error: Option<BundleFrontendEmitFnV1>,
}

/// The static table `chop_bundle_frontend_api_v1` returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BundleFrontendTableV1 {
    pub struct_size: u32,
    pub layout: u32,
    pub open: Option<BundleFrontendOpenFnV1>,
    pub close: Option<BundleFrontendCloseFnV1>,
    pub render: Option<BundleFrontendDocumentFnV1>,
    pub constraint: Option<BundleFrontendDocumentFnV1>,
    pub parser_open: Option<BundleFrontendParserOpenFnV1>,
    pub parser_push: Option<BundleFrontendParserPushFnV1>,
    pub parser_close: Option<BundleFrontendParserCloseFnV1>,
}

/// A bundle's frontend over byte documents. `Err` carries a refusal
/// document. Render and constraint may run concurrently.
pub trait FrontendV1: Sized + Send + Sync + 'static {
    type Parser: FrontendParserV1;

    /// Opens the frontend over the authenticated frontend component, with
    /// the open answer document.
    fn open(component: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>>;

    /// The rendered prompt document for a chat request document.
    fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>>;

    /// The request's constraint document, empty when it has none.
    fn constraint(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>>;

    /// A parser of one request's output.
    fn open_parser(&self, request: &[u8]) -> Result<Self::Parser, Vec<u8>>;
}

/// One request's incremental output parser. The export calls it from one
/// thread at a time and stops after the final push or any failure.
pub trait FrontendParserV1: Send + 'static {
    /// The event document for `text`, the output that follows what earlier
    /// pushes carried; `last` marks the request's final push.
    fn push(&mut self, text: &[u8], last: bool) -> Result<Vec<u8>, Vec<u8>>;
}

/// Why a frontend call returned no answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontendCallErrorV1 {
    /// The bundle refused the call with this refusal document.
    Refused(Vec<u8>),
    /// The bundle panicked; its export caught the panic.
    Panicked,
    /// The call failed with this status and no refusal document: the
    /// bundle refused an argument or the call's state.
    Status(BundleStatusV1),
    /// The bundle broke the call's contract, for example by answering
    /// twice or not at all, or with a malformed view.
    Contract(&'static str),
    /// The parser took its final push or a push failed, so it only closes
    /// now; the handle refused the push without calling the bundle.
    ParserFinished,
    /// The bundle emitted a `length`-byte document past the call's
    /// `maximum`; the host refused it without copying it.
    Oversized { length: u64, maximum: usize },
}

impl fmt::Display for FrontendCallErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(document) => write!(
                formatter,
                "the bundle frontend refused the call ({} byte refusal document)",
                document.len()
            ),
            Self::Panicked => formatter.write_str("the bundle frontend panicked"),
            Self::Status(status) => write!(
                formatter,
                "the bundle frontend call failed: {}",
                status.name().unwrap_or("unknown status")
            ),
            Self::Oversized { length, maximum } => write!(
                formatter,
                "the bundle frontend emitted a {length} byte document past its {maximum} byte limit"
            ),
            Self::ParserFinished => {
                formatter.write_str("the frontend parser took its final push or failed")
            }
            Self::Contract(rule) => {
                write!(formatter, "the bundle frontend broke its contract: {rule}")
            }
        }
    }
}

impl std::error::Error for FrontendCallErrorV1 {}
