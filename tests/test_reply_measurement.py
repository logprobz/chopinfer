"""Run the model-owned passive reply measurement contract in the CPU gate."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/deepseek41/tools'))
from test_reply_history_report import ReplyHistoryReportTest


if __name__ == '__main__':
    unittest.main()
