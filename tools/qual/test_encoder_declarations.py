"""Declaration changes must change qualification requirements too."""

import copy
import unittest

from encoder_declarations import for_model, parity_rules, unqualified


class EncoderDeclarationsTests(unittest.TestCase):
    def test_serving_taps_and_audio_grouping(self):
        mimo = for_model("mimo26")
        rules = parity_rules(mimo)
        self.assertEqual(set(rules["rows"]), {"enc.vision.out", "enc.audio.out"})
        self.assertEqual(rules["codes"], "enc.audio.x.codes")
        self.assertEqual(rules["every_row"], ("enc.audio.out:enc.audio.x.codes/4",))
        self.assertEqual(rules["direction_scale"], ("enc.vision.out",))
        changed = copy.deepcopy(mimo)
        audio = next(value for value in changed if value["tower"] == "audio")
        next(stage for stage in audio["stages"] if stage["op"] == "merge.consecutive.v1")["rows"] = 8
        self.assertEqual(parity_rules(changed)["every_row"], ("enc.audio.out:enc.audio.x.codes/8",))
        qwen = parity_rules(for_model("qwen38"))
        self.assertEqual(qwen["rows"], ("enc.vision.out",))
        self.assertIsNone(qwen["codes"])
        self.assertEqual(qwen["direction_scale"], ("enc.vision.out",))

    def test_only_encoders_of_wholly_unqualified_media_are_withdrawn(self):
        encoders = [{"name": "vision", "kinds": {"image", "video"}}, {"name": "audio", "kinds": {"audio"}}]
        reason = "no in-tree audio reference"
        withdrawn = unqualified(encoders, {"media": ["image", "video"], "unqualified_media": {"audio": reason}})
        self.assertEqual([(value["name"], why) for value, why in withdrawn], [("audio", reason)])
        # One qualified input kind keeps an encoder required.
        withdrawn = unqualified(encoders, {"media": ["image"], "unqualified_media": {"video": reason}})
        self.assertEqual(withdrawn, [])
        for capabilities in (None, {}, {"media": ["image", "audio", "video"]}):
            self.assertEqual(unqualified(encoders, capabilities), [])

    def test_unknown_models_and_paths_are_refused(self):
        for model in ("absent", "../mimo26", "..", "/mimo26", None):
            with self.assertRaises(ValueError):
                for_model(model)


if __name__ == "__main__":
    unittest.main()
