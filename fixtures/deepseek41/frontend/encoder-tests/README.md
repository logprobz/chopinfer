# The encoder's own test pairs

`test_input_N.json` and `test_output_N.txt` (N = 1-5) are copied unchanged
from the official checkpoint's `encoding/tests/` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`
at revision 1bae0e10), under the MIT licence in `LICENSE` beside them. The
vendor's `test_encoding.py` encodes each input with `encode_case(case,
thinking_mode="chat")` and compares the prompt with the output byte for byte;
`models/deepseek41/compiler/tests/frontend_goldens.rs` does the same with
the Rust port and checks each file's SHA-256 against the checkpoint's.
