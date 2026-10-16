import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("schedule_guard", Path(__file__).with_name("projection-schedules.py"))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def text(records):
    return json.dumps(records, indent=2) + "\n"


class ScheduleHistory(unittest.TestCase):
    def test_superseding_schedule_preserves_the_approved_prefix(self):
        original = {"inputs": {"kind": "Dense"}, "schedule": {"kind": "Typed", "segments": 4}}
        tuned = dict(original, schedule={"kind": "Typed", "segments": 2})
        base = text([original])
        self.assertEqual(guard.check(text([original, tuned]), base), [])
        for invalid in [[], [tuned], [tuned, original], [original, original]]:
            self.assertTrue(guard.check(text(invalid), base))
        self.assertTrue(guard.check(json.dumps([original]), None))
        self.assertTrue(guard.check("{}", None))
        self.assertTrue(guard.check(text([{}]), None))


if __name__ == "__main__":
    unittest.main()
