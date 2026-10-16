//! A bundle exports a model's own chat format with one line. It has its own
//! test binary because each export macro defines the same table symbol.

use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FormatRenderedV1, FrontendErrorV1,
    OpenedV1,
};

struct Format;

impl ChatFormatV1 for Format {
    fn open(_: &FormatDescriptorV1) -> Result<Self, String> {
        Ok(Self)
    }

    fn opened(&self) -> OpenedV1 {
        OpenedV1::default()
    }

    fn render(&self, _: FormatRenderRequestV1<'_>) -> Result<FormatRenderedV1, FrontendErrorV1> {
        Err(FrontendErrorV1::new("not rendered", None))
    }
}

frontend_format::export_chat_format_v1!(Format);

#[test]
fn the_macro_exports_the_frontend_table() {
    assert!(!chop_bundle_frontend_api_v1().is_null());
}
