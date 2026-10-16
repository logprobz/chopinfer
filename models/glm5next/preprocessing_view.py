"""A checkpoint view the image preprocessing reference can load.

    python3 models/glm5next/preprocessing_view.py --checkpoint CHECKPOINT --output VIEW
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m tools.refrun preprocess images \\
        --checkpoint VIEW --output $CHOPINFER_DATA/glm5next/refrun/preprocessing/images

tools/refrun/preprocessing.py binds the checkpoint's preprocessor_config.json.
GLM 5.3 Flash ships none: its image processor configuration is the
`image_processor` member of processor_config.json, which AutoImageProcessor
reads when no preprocessor_config.json exists. VIEW links every checkpoint
file except the weights and adds preprocessor_config.json holding exactly
that member, so the processor loads the same configuration (checked here)
and the reference binds its bytes.
"""

import argparse
import json
from pathlib import Path


def config_bytes(checkpoint):
    """The checkpoint's image processor configuration as preprocessor_config.json bytes."""
    section = json.loads((Path(checkpoint) / "processor_config.json").read_text())["image_processor"]
    return (json.dumps(section, indent=2) + "\n").encode()


def write_view(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if (checkpoint / "preprocessor_config.json").exists():
        raise ValueError("the checkpoint has its own preprocessor_config.json; use it directly")
    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(checkpoint.iterdir()):
        if path.suffix != ".safetensors" and path.is_file():
            (output / path.name).symlink_to(path)
    (output / "preprocessor_config.json").write_bytes(config_bytes(checkpoint))


def check_view(checkpoint, output):
    from transformers import AutoImageProcessor
    loaded = [AutoImageProcessor.from_pretrained(str(path), local_files_only=True).to_dict()
              for path in (checkpoint, output)]
    if loaded[0] != loaded[1]:
        raise ValueError("the view loads another image processor configuration")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_view(args.checkpoint, args.output)
    check_view(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
