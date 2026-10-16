"""Option identity admission, without exercising non-default model execution."""
import unittest
import common
import kinds


class ModelOptionsTest(unittest.TestCase):
    def test_legacy_and_declared_qualified_defaults(self):
        for serving in ({}, {"model_options": {"budget": 8}, "model_options_qualified": True}):
            server = {"health": {"serving": serving}}
            common.require_qualified_model_options(server)
            kinds.admit({"server": server})

    def test_nondefault_or_incomplete_options_refuse_receipts(self):
        for serving in (
                {"model_options": {"budget": 8}, "model_options_qualified": False},
                {"model_options": {"budget": 8}},
                {"model_options": {}, "model_options_qualified": True},
                {"model_options": None, "model_options_qualified": True},
                {"model_options": {"budget": 8}, "model_options_qualified": "true"}):
            server = {"health": {"serving": serving}}
            with self.assertRaisesRegex(ValueError, "model options"):
                common.require_qualified_model_options(server)
            with self.assertRaisesRegex(ValueError, "model options"):
                kinds.admit({"server": server})
