"""deepseek-recipe as the second authority of DeepSeek V4.1's frontend corpus.

`python -m tools.refrun.frontend_corpus generate --model deepseek41` runs this
script under `$CHOPINFER_DEEPSEEK_RECIPE_PYTHON`, an environment holding only
the package the `RECIPE` pin in frontend_cases.py names, installed from
requirements-recipe.txt beside this script (README.md, "Recipe oracle"). The
reference runner's environment, and with it every reference key, stays as it
is. The script reads `{"checkpoint", "cases", "rounds"}` as JSON on stdin and
writes `{"environment", "cases", "rounds"}` to stdout (`SubprocessOracle` in
tools/refrun/frontend_corpus.py):

- each case's request, sent as model DeepSeek-V4.1-Flash, converted by
  `ChatCompletionRequest.convert`, rendered by
  `DeepseekV41Encoding.render_conversation` and encoded with the checkpoint's
  tokenizer, or the refusal conversion raises, with its HTTP status and
  message; a conversion error with any status but a client error's 4xx
  stops the oracle, so generation fails instead of recording a refusal;
- each round trip's request converted the same way, and its completion's
  token IDs parsed by `StreamProcessor` with the conversion's parsing options
  and the checkpoint's tokenizer, then finished as a stop. A final
  end-of-sentence token is not pushed: the serving backend consumes the stop
  token, and pushed, it would read as answer text.

The recipe spells the image placeholder `<｜image｜>`; the checkpoint's
tokenizer and encoding.py spell it `<｜deepseek_image｜>`, so conversion and
rendering use that spelling. Conversion checks no special-token spelling in
client text, so the respelling changes only the placeholder it writes for an
image.
"""

import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

PACKAGE = "deepseek-recipe"
MODEL = "DeepSeek-V4.1-Flash"
IMAGE = "<｜deepseek_image｜>"


def installed_sha256(distribution):
    """The digest of the files a distribution installed, as its RECORD lists
    them: the sorted `path,hash` lines of every hashed file outside its
    .dist-info directory, whose files the installer partly writes itself.
    Each file must still have its recorded hash."""
    lines = []
    for file in distribution.files or ():
        if file.hash is None or file.parts[0].endswith(".dist-info"):
            continue
        digest = hashlib.new(file.hash.mode, file.read_binary()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != file.hash.value:
            raise SystemExit(f"{file} no longer has the hash its RECORD lists")
        lines.append(f"{file},{file.hash.mode}={file.hash.value}\n")
    if not lines:
        raise SystemExit(f"{distribution.metadata['Name']} lists no installed files")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def environment(recipe):
    distribution = importlib.metadata.distribution(PACKAGE)
    if recipe.__version__ != distribution.version:
        raise SystemExit(f"deepseek_recipe {recipe.__version__} is not the installed {PACKAGE} "
                         f"{distribution.version}")
    return {"package": PACKAGE, "version": distribution.version,
            "installed_sha256": installed_sha256(distribution), "python": platform.python_version()}


def convert(recipe, options, request):
    """The recipe's conversion of a request and None, or None and the
    refusal it raises: the HTTP status and the error body's message. A body
    the request type does not read raises ValueError, which the recipe's
    server answers with status 400 and the error's text (server-py/main.py
    `prepare_request`). A conversion error without a client error's 4xx
    status is the recipe failing, not refusing, and stops the oracle."""
    try:
        body = recipe.ChatCompletionRequest({**request, "model": MODEL})
    except ValueError as error:
        return None, {"status": 400, "message": str(error)}
    try:
        return body.convert(options), None
    except recipe.ConversionError as error:
        if not 400 <= error.status_code < 500:
            raise SystemExit(f"{PACKAGE} failed with status {error.status_code}: {error.body}") from error
        try:
            message = json.loads(error.body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = error.body
        return None, {"status": error.status_code, "message": message}


def render(recipe, options, encoding, request):
    converted, refusal = convert(recipe, options, request)
    if refusal is not None:
        return {"refusal": refusal}
    conversation = converted.conversation
    return {"rendered_text": encoding.render_conversation(conversation).prompt,
            "token_ids": encoding.encode(conversation)}


def completion_ids(token_ids, stop):
    """The completion's IDs as the backend hands them to the parser: less a
    final stop token."""
    return token_ids[:-len(stop)] if token_ids[-len(stop):] == stop else token_ids


def turn(choice):
    """A Chat Completions choice as a round trip compares it."""
    message = choice["message"]
    return {"reasoning": message.get("reasoning_content") or "", "content": message.get("content") or "",
            "tool_calls": [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                           for call in message.get("tool_calls") or []],
            "finish_reason": choice["finish_reason"]}


def parse(recipe, options, tokenizer, vector, stop):
    converted, refusal = convert(recipe, options, vector["request"])
    if refusal is not None:
        return {"refusal": refusal}
    generator = recipe.ChatCompletionRequest.chunk_generator(converted, "oracle", MODEL)
    processor = recipe.StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = recipe.ChatCompletionResponse("oracle", MODEL, 0, 0, 0)
    chunks = [recipe.InferenceChunk.ready(),
              *(recipe.InferenceChunk.tokens([token]) for token in completion_ids(vector["token_ids"], stop)),
              recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Stop)]
    try:
        for chunk in chunks:
            for output in processor.push(chunk):
                response.accumulate(output)
        for output in processor.finish():
            response.accumulate(output)
    except RuntimeError as error:
        return {"error": {"class": type(error).__name__, "message": str(error)}}
    finally:
        processor.close()
    return turn(json.loads(response.to_json())["choices"][0])


def main():
    import deepseek_recipe as recipe

    payload = json.loads(sys.stdin.buffer.read())
    tokenizer = recipe.Tokenizer.from_file(str(Path(payload["checkpoint"]) / "tokenizer.json"))
    special = recipe.SpecialTokens(image=IMAGE)
    options = recipe.ConversionOptions().with_special_tokens(special)
    encoding = recipe.DeepseekV41Encoding().with_special_tokens(special).with_tokenizer(tokenizer)
    stop = tokenizer.encode(recipe.SpecialToken.Eos.default_str())
    result = {"environment": environment(recipe),
              "cases": {case["id"]: render(recipe, options, encoding, case["request"]) for case in payload["cases"]},
              "rounds": {vector["id"]: parse(recipe, options, tokenizer, vector, stop)
                         for vector in payload["rounds"]}}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode())


if __name__ == "__main__":
    main()
