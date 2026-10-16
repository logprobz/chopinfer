"""Builds fixtures/qwen38/agentic-outcomes.json from the 2026-10-01 replay of
agentic sessions that never sent a final answer: the two recorded Qwen3.8 model outcomes as
token IDs, with no generated or parsed text.

    $CHOPINFER_REFRUN_PYTHON fixtures/qwen38/capture/agentic_outcomes.py

Needs CHOPINFER_DATA, CHOPINFER_QWEN38_CHECKPOINT and the `tokenizers`
package. Each turn keeps its prompt's last PROMPT_TAIL token IDs, every
generated token ID, the finish reason, the usage counts, and the SHA-256 of
what Chopinfer's parser streamed for it (`parsed_digest`), so a test can
rebuild the parsed stream from the raw tokens and prove it equal to the
recorded one. The bytes of each token ID come from the checkpoint's
tokenizer.json and are checked against its decode of every turn.
"""
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer

RUN = 'qwen-agentic-nofinal/runs/gpu-20261001T052115Z'
# (sweep, case, concurrency, workload index, outcome)
OUTCOMES = (('sweep4', 'coding-000050', 4, 49, 'ended inside thinking'),
            ('sweep3', 'coding-000035', 8, 34, 'malformed tool call'))
PROMPT_TAIL = 64
OUTPUT = Path(__file__).resolve().parents[1] / 'agentic-outcomes.json'


def parsed_digest(reasoning, content, calls):
    """SHA-256 of a turn's parsed reasoning, content and [name, arguments]
    calls; tools/qual/test_agentic_review.py computes the same."""
    value = dict(reasoning=reasoning, content=content, calls=calls)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def byte_decoder():
    """GPT-2 byte-level BPE: the byte behind each character of a token."""
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + \
        list(range(ord('®'), ord('ÿ') + 1))
    chars, extra = list(printable), 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, chars)}


def turn_ids(turn):
    """(prompt tail, generated IDs, finish reason, usage) of a recorded turn."""
    prompt, generated, finish, usage = None, [], None, None
    for block in turn['sse'].split('\n\n'):
        if not block.startswith('data: ') or block == 'data: [DONE]':
            continue
        event = json.loads(block[len('data: '):])
        if event.get('prompt_token_ids') is not None:
            assert prompt is None
            prompt = event['prompt_token_ids'][-PROMPT_TAIL:]
        for choice in event.get('choices', []):
            generated += choice.get('token_ids') or []
            finish = choice.get('finish_reason') or finish
        usage = event.get('usage') or usage
    assert finish == turn['finish_reason'] and usage['completion_tokens'] == len(generated)
    return prompt, generated, finish, usage


def main():
    data = Path(os.environ['CHOPINFER_DATA'])
    tokenizer_path = Path(os.environ['CHOPINFER_QWEN38_CHECKPOINT']) / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    added = {i: t.content for i, t in tokenizer.get_added_tokens_decoder().items()}
    bytes_of = byte_decoder()
    sessions, ids = [], set()
    for sweep, case, concurrency, index, outcome in OUTCOMES:
        path = data / RUN / sweep / 'sessions.jsonl'
        rows = [json.loads(line) for line in path.open()]
        row = next(r for r in rows if (r['case_id'], r['concurrency'], r['index'], r['phase'])
                   == (case, concurrency, index, 'measured'))
        turns = []
        for turn in row['turns']:
            prompt, generated, finish, usage = turn_ids(turn)
            answer = turn['answer']
            calls = [[c['function']['name'], json.loads(c['function']['arguments'])]
                     for c in answer.get('tool_calls') or []]
            turns.append(dict(prompt_tail=prompt, token_ids=generated, finish_reason=finish,
                              prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                              parsed_digest=parsed_digest(answer.get('reasoning') or '', answer.get('content') or '',
                                                          calls)))
            ids.update(prompt + generated)
        sessions.append(dict(sweep=sweep, case_id=case, concurrency=concurrency, index=index, phase='measured',
                             recorded_failure=row['failure'], outcome=outcome, turns=turns))
    token_bytes = {i: (added[i].encode() if i in added else bytes(bytes_of[c] for c in tokenizer.id_to_token(i)))
                   for i in sorted(ids)}
    for session in sessions:
        for turn in session['turns']:
            for group in (turn['prompt_tail'], turn['token_ids']):
                ours = b''.join(token_bytes[i] for i in group).decode('utf-8', 'replace')
                assert ours == tokenizer.decode(group, skip_special_tokens=False), (session['case_id'], group[:8])
    value = dict(provenance=dict(run=RUN, job='u8ba', report='qwen-agentic-nofinal/REPORT.md',
                                 tokenizer='Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13',
                                 tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                                 prompt_tail_ids=PROMPT_TAIL),
                 sessions=sessions, token_bytes={str(i): b.hex() for i, b in token_bytes.items()})
    OUTPUT.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    print(OUTPUT, OUTPUT.stat().st_size)


if __name__ == '__main__':
    main()
