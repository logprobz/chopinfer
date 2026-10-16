# Projection header fixtures

mopd.json records names, dtypes, stored shapes and precision rules from the active
MOPD headers, DFlash headers and retained audio-tokenizer encoder headers surveyed
on 2026-10-04. Family axes expand the original index selections. The fixture
contains no weights and does not derive header facts from projections.json.

The main index has 73081 tensors, DFlash has 63 and the retained tokenizer
encoder has 389. Decoder and tokenizer training-state tensors have a separate
owner scope and do not enter the text declaration. Existing encoder metadata
contracts validate every included external source before binding.

The four QKV chunks restart the 128-row inverse-scale grid; metadata binding
retains four validated views with their exact source scale ranges. The TP2
regroup description repeats scales into 64-row groups and bypasses TP4. No math
or regroup is executed by these parameter decoders. PT mask contents are outside
safetensors header evidence and are represented by a typed external stage role
with positions, prefix_len, token_batch and live_tokens control bindings.

Full-header tests use CHOPINFER_PROJECTION_HEADERS_MIMO26 and read confined main,
DFlash and tokenizer safetensors headers only. They print a skip reason when the
variable is unset. No GPU correctness or serving qualification is claimed.

Full-header checks compare every included tensor's name, dtype and shape, and
the complete quantization config, with this fixture. DFlash head norm sides
retain replicated 128-element BF16 weights and name their owning kernels;
shared affine arithmetic stays in those kernels. Registered stages and source
transforms remain metadata descriptions until their behavior is implemented.
