//! A model's own chat format, as a bundle exports it over `chop.frontend.v1`.
//!
//! A model renders its prompt with a [`ChatFormatV1`]: Rust ported from its
//! chat template, written with the [`crate::toolkit`]. Its declaration's
//! `format.parser` section ([`FrontendParserDeclarationV1`]) says how its
//! completions read. [`ChatFormatExportV1`] serves the two over the
//! documents: a render through the format, a tool constraint and a parser
//! through the parser section and the shared parsers. A bundle exports it
//! with [`crate::export_chat_format_v1!`].

use std::sync::Arc;

use runtime_bundle_abi::frontend::{FrontendExportV1, FrontendV1};
use runtime_bundle_abi::frontend_document::OpenedV1;

use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1};
use crate::export::{
    DeclaredExportParserV1, constraint_document_v1, open_parser_document_v1, refusal_v1,
    render_document_v1,
};
use crate::format::{FormatRenderRequestV1, FormatRenderedV1};
use crate::request::FrontendErrorV1;
use crate::toolkit::reasoning_open_v1;

/// A model's chat format: the renderer a bundle exports with
/// [`crate::export_chat_format_v1!`].
///
/// Parsing is not the format's: the export reads every completion, and
/// builds every tool-choice grammar, from the declaration's `format.parser`
/// with the shared parsers.
pub trait ChatFormatV1: Sized + Send + Sync + 'static {
    /// Opens over the checked declaration: its host sections and its
    /// `format.parser`. A format refuses a declaration it cannot render, such
    /// as one that serves a media kind it writes no labels for; the message
    /// reaches the host as the open refusal.
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String>;

    /// What the render reads of a request besides the fields every request
    /// document carries: the export's open answer.
    fn opened(&self) -> OpenedV1;

    /// Renders one chat request into its prompt.
    ///
    /// `media` lists the request's media parts as the request carried them,
    /// in the order the segments place them. The export sets
    /// `reasoning_open` from the prompt with the parser section's reasoning
    /// markers ([`reasoning_open_v1`]), replacing the value a render
    /// returns, so the parser and constraint agree with the prompt the same
    /// way for every model. The shared parser reads no parser state, so a
    /// render leaves `parser_state` empty.
    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1>;
}

/// A [`ChatFormatV1`] and its declaration's parser section behind a
/// `chop.frontend.v1` export.
pub struct ChatFormatExportV1<F> {
    descriptor: Arc<FormatDescriptorV1>,
    format: F,
}

impl<F: ChatFormatV1> ChatFormatExportV1<F> {
    fn parser(&self) -> &FrontendParserDeclarationV1 {
        &self.descriptor.parser
    }
}

impl<F: ChatFormatV1> FrontendV1 for ChatFormatExportV1<F> {
    type Parser = DeclaredExportParserV1;

    fn open(component: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
        // The host checks the component file's size cap where it reads the
        // package; a host that opens over a descriptor it wrote back passes
        // JSON the cap was never meant for, so this reading leaves it out.
        let descriptor = FormatDescriptorV1::from_declaration_bytes(component)
            .map_err(|error| refusal_v1(error.to_string(), None))?;
        let format = F::open(&descriptor).map_err(|message| refusal_v1(message, None))?;
        let opened = format.opened().encode();
        Ok((
            Self {
                descriptor: Arc::new(descriptor),
                format,
            },
            opened,
        ))
    }

    fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        render_document_v1(request, |request| {
            let continue_final_message = request.continue_final_message;
            let mut rendered = self.format.render(request)?;
            if !rendered.parser_state.is_empty() {
                return Err(FrontendErrorV1::new(
                    "a chat format render leaves no parser state: the shared parser reads \
                     format.parser",
                    None,
                ));
            }
            let parser = self.parser();
            rendered.reasoning_open = reasoning_open_v1(
                rendered.mode,
                continue_final_message,
                &parser.reasoning_open,
                &parser.reasoning_close,
                &rendered.prompt,
            );
            Ok(rendered)
        })
    }

    fn constraint(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        constraint_document_v1(&self.descriptor, request)
    }

    fn open_parser(&self, request: &[u8]) -> Result<Self::Parser, Vec<u8>> {
        open_parser_document_v1(&self.descriptor, request)
    }
}

/// `F`'s export linked into this program, for in-process tests: an
/// in-process loopback of the ABI, with no module to load. Every call
/// crosses the same trampolines and documents a loaded package's export
/// does.
#[must_use]
pub fn chat_format_export_v1<F: ChatFormatV1>() -> FrontendExportV1 {
    FrontendExportV1::in_process::<ChatFormatExportV1<F>>()
}
