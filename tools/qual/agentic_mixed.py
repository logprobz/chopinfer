#!/usr/bin/env python3
"""Self-contained Runyard agentic coding sweep, with Chopinfer receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import uuid
import threading
import time
from urllib.parse import urlsplit

import common
import declaration
import model_modules
from agentic_metrics import (INVALID_SESSION, RAW_CALLS_AGREE, RAW_MALFORMED_CALL, RAW_REASONING_ONLY, TAIL_SAMPLES,
                             burst_metrics, distribution, histogram_reports, level_verdict, metrics_delta,
                             prefill_evidence, ratio, summarize)
from agentic_server import observe_startup, mode_evidence, verify_arm
from agentic_workload import CORPUS, Workspace, decode_context, load_corpus, max_tool_result_bytes, task_for, valid_session

REVISION = 'chopinfer.agentic-mixed.v3'
MAX_REQUEST = 4 << 20
MAX_RESPONSE = 8 << 20
# Generated token IDs kept in memory per turn for re-parsing an invalid
# session's turns; a longer turn stays unclassified. The prompt's last IDs
# hold the generation prompt's assistant header.
RAW_TURN_IDS = 131072
PROMPT_TAIL_IDS = 64
# The served frontend's chat markup (`Markup`, or a model's own re-parser),
# set by `main` from the frontend declaration; the raw re-parse of a turn
# reads it.
MARKUP = None
TOOL_GRAMMARS = ('xml-function-parameters-v1', 'xml-key-value-arguments-v1')


class Markup:
    """The chat markup of a frontend declaration (`format.parser` and
    `vocabulary`) that the raw re-parse reads: the shared re-parser of the
    library tool grammars (TOOL_GRAMMARS). `format.parser` states the
    turn-opening boundary tokens (`turn_open`) and the assistant header.

    A turn opens with one of the turn-opening boundary tokens and ends
    with one of the declared stop tokens, as the served decoder ends it. The
    assistant header (the assistant prefix) opens the thinking block or
    leaves the model to open it, so generated text before the reasoning
    close is reasoning, and everything after it is the answer or tool calls,
    in the declared tool grammar.

    A format that declares no library tool grammar supplies its own
    re-parser: a class `Markup` in the model root's `agentic_markup.py`
    (`load_markup`). It has this interface, which is all the raw re-parse
    reads:
    - `Markup(frontend)`, from the parsed frontend declaration;
    - `grammar`, a name for the tool grammar it reads, recorded in the
      receipt;
    - `turn(opening, text)`, `last_class(reasoning, closed, after)`,
      `calls(after, tools)` and `same_value(raw, value)`, as below.
    """

    def __init__(self, frontend):
        markup = frontend['format']['parser']
        self.grammar = markup.get('tool_grammar')
        if self.grammar not in TOOL_GRAMMARS:
            raise ValueError(f'unsupported tool grammar {self.grammar!r}')
        vocabulary = frontend['vocabulary']
        boundary = {token['id']: token['text'] for token in vocabulary['boundary_tokens']}
        opens = set(markup['turn_open'])
        if not opens <= set(boundary.values()):
            raise ValueError(f'turn_open {sorted(opens - set(boundary.values()))} are not boundary tokens')
        self.assistant = markup['assistant_header']
        value_open = markup.get('parameter_value_open')
        self.turn_open = tuple(sorted(opens))
        self.reasoning_open, self.reasoning_close = markup['reasoning_open'], markup['reasoning_close']
        self.end = tuple(boundary[token] for token in vocabulary['stop_token_ids'])
        self.tool_open, self.tool_close = markup['tool_call_open'].strip(), markup['tool_call_close'].strip()
        self.parameter_open = markup['parameter_open'].strip()
        self.parameter_name_close = markup['parameter_name_close'].strip()
        self.parameter_close = markup['parameter_close'].strip()
        # The whitespace the declared markers write around a function
        # parameter's value, which the served parser removes once each.
        self.value_lead = markup['parameter_name_close'][len(markup['parameter_name_close'].rstrip()):]
        self.value_trail = markup['parameter_close'][:len(markup['parameter_close']) - len(
            markup['parameter_close'].lstrip())]
        if self.grammar == 'xml-function-parameters-v1':
            self.function_open = markup['function_open'].strip()
            name_close = markup['function_name_close'].strip()
            self.function_close = markup['function_close'].strip()
            self.call_markup = (self.tool_open, self.tool_close, self.function_open, self.function_close)
            # One tool region's function header, the region, and a parameter's
            # header.
            head = r'\s*' + re.escape(self.function_open) + name_pattern(name_close)
            self.function_head = re.compile(head)
            self.function_region = re.compile(head + r'(.*)' + re.escape(self.function_close) + r'\s*', re.S)
            self.parameter_head = re.compile(re.escape(self.parameter_open) + name_pattern(self.parameter_name_close))
        else:
            if value_open is None:
                raise ValueError('xml-key-value-arguments-v1 needs parameter_value_open')
            self.value_open = value_open.strip()
            self.call_markup = (self.tool_open, self.tool_close, self.parameter_open, self.parameter_name_close,
                                self.value_open, self.parameter_close)

    def turn(self, opening, text):
        """One turn's raw text, OPENING the decoded prompt tail and TEXT the
        decoded generation: (reasoning, closed, after), or a class when it is
        not one well-formed turn (`raw_turn`)."""
        return raw_turn(self, opening, text)

    def last_class(self, reasoning, closed, after):
        """The class of a last turn's parts (`last_turn_class`):
        RAW_REASONING_ONLY or RAW_MALFORMED_CALL for a model outcome, else
        any other class."""
        return last_turn_class(self, reasoning, closed, after)

    def calls(self, after, tools=()):
        """The content and calls of the text after a reasoning close:
        (content, [(name, [(parameter, raw value)])]), or None (`raw_calls`)."""
        return raw_calls(self, after, tools)

    def names_function(self, region):
        """Whether a tool region's text (inside its wrapper) opens with a
        function header."""
        if self.grammar == 'xml-function-parameters-v1':
            return self.function_open in region
        return bool(region.partition(self.parameter_open)[0].strip())

    def region(self, text, tools):
        """One tool region from TEXT, which follows its open: (name,
        [(parameter, raw value)], text after its close), or None unless it is
        exactly one well-formed call."""
        if self.grammar == 'xml-key-value-arguments-v1':
            return key_value_region(self, text, tools)
        # The served parser runs the arguments to the first function close
        # outside a JSON value's string literals (`function_close`), so the
        # region's close is the first tool close after it; a tool open or
        # close inside a value is content.
        head = self.function_head.match(text)
        if not head:
            return None
        name = head.group(1)

        def json_key(key):
            return json_valued(tools, name.strip(), key.strip())

        at = function_close(self, text[head.end():], json_key)
        end = text.find(self.tool_close, head.end() + at + len(self.function_close)) if at >= 0 else -1
        if end < 0:
            return None
        region, rest = text[:end], text[end + len(self.tool_close):]
        function = self.function_region.fullmatch(region)
        parameters = function and function_parameters(self, function.group(2), json_key)
        if parameters is None:
            return None
        return name, parameters, rest

    def same_value(self, raw, value):
        """Whether a raw parameter value is the parsed argument: a string
        (less the whitespace the declared markers write around it in the
        function-parameter grammar, `value_lead` and `value_trail`), else its
        JSON value."""
        if isinstance(value, str):
            if self.grammar == 'xml-function-parameters-v1':
                raw = raw.removeprefix(self.value_lead).removesuffix(self.value_trail)
            return raw == value
        try:
            return json.loads(raw.strip()) == value
        except ValueError:
            return False


def name_pattern(close):
    """A name up to CLOSE: one line without CLOSE, then CLOSE."""
    return r'((?:(?!' + re.escape(close) + r')[^\n])+)' + re.escape(close)


def function_parameters(markup, body, json_key):
    """The parameters of a function's arguments BODY (function-parameter
    grammar): [(parameter, raw value)], or None unless BODY is parameters
    separated only by whitespace. A value ends at its first close, except
    that a value whose key is JSON (`json_key`) ignores closes inside its
    string literals, as the served parser does.

    The walk reads BODY by offsets and copies only the names and values it
    returns, so many parameters cost the body once."""
    parameters, at = [], skip_space(body, 0)
    while at < len(body):
        head = markup.parameter_head.match(body, at)
        if not head:
            return None
        key = head.group(1)
        end = (outside_strings if json_key(key) else str.find)(body, markup.parameter_close, head.end())
        if end < 0:
            return None
        parameters.append((key, body[head.end():end]))
        at = skip_space(body, end + len(markup.parameter_close))
    return parameters


def skip_space(text, at):
    """The first index at or after AT in TEXT that is not whitespace."""
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def function_close(markup, arguments, json_key):
    """Where the served parser ends a function's ARGUMENTS, the text after
    its name (runtime-frontend's region scan): the first function close that
    the parameter walk (`walk_step`) of the text before it does not leave
    inside a string literal of a JSON value, or -1.

    The walk to each close resumes where every earlier step is final, before
    the last bytes a marker could still begin in, so many closes inside one
    string cost the arguments once."""
    reach = max(map(len, (markup.parameter_open, markup.parameter_name_close, markup.parameter_close)))
    settled = ('seek', 0, 0, False, False)
    at = arguments.find(markup.function_close)
    while at >= 0:
        while settled[1] < at - reach + 1:
            settled = walk_step(markup, arguments, at, settled, json_key)
        walk = settled
        while walk[1] < at:
            walk = walk_step(markup, arguments, at, walk, json_key)
        mode, _, _, in_string, _ = walk
        if not (mode == 'json' and in_string):
            return at
        at = arguments.find(markup.function_close, at + 1)
    return -1


def walk_step(markup, raw, end, walk, json_key):
    """One step of the served parser's walk of a function's arguments
    RAW[:END] (runtime-frontend's ParameterWalkV1). WALK is (mode, next
    character, where the name began, inside a string literal, after a
    backslash there), with mode 'seek' before a parameter open, 'name', and
    'raw' or 'json' in a value.

    A raw value ends at the first parameter open or close whatever it holds;
    a JSON value ends the same way outside its string literals."""
    mode, at, name, in_string, escaped = walk
    opens = raw.startswith(markup.parameter_open, at, end)
    closes = raw.startswith(markup.parameter_close, at, end)
    if mode == 'seek':
        if opens:
            at += len(markup.parameter_open)
            return 'name', at, at, False, False
    elif mode == 'name':
        if raw.startswith(markup.parameter_name_close, at, end):
            value = 'json' if json_key(raw[name:at]) else 'raw'
            return value, at + len(markup.parameter_name_close), 0, False, False
    elif mode == 'raw':
        # An open ends the value before a close at the same character.
        if opens or closes:
            return 'seek', at if opens else at + len(markup.parameter_close), 0, False, False
    elif in_string:
        character = raw[at]
        if escaped:
            escaped = False
        elif character == '\\':
            escaped = True
        elif character == '"':
            in_string = False
        return mode, at + 1, name, in_string, escaped
    elif raw[at] == '"':
        return mode, at + 1, name, True, False
    elif opens or closes:
        # The longer marker ends the value, an open at a tie.
        if closes and not (opens and len(markup.parameter_open) >= len(markup.parameter_close)):
            return 'seek', at + len(markup.parameter_close), 0, False, False
        return 'seek', at, 0, False, False
    return mode, at + 1, name, in_string, escaped


def key_value_region(markup, text, tools):
    """A key-value tool region (xml-key-value-arguments-v1) from TEXT, which
    follows its open, as `Markup.region` returns it: the function name up to
    the first key or the close, then key and value elements separated only by
    whitespace, then the close. A value ends at its first close, except that
    a value the tool's schema makes JSON (`json_valued`) ignores closes inside
    its string literals, as the served parser does.

    The walk reads TEXT by offsets and copies only the name, keys, values and
    rest it returns, so many parameters cost the region once."""
    ends = [at for at in (text.find(markup.parameter_open), text.find(markup.tool_close)) if at >= 0]
    if not ends:
        return None
    at = min(ends)
    name = text[slice(*strip_span(text, 0, at))]
    if not name or '\n' in name or markup.tool_open in name:
        return None
    parameters = []
    while not text.startswith(markup.tool_close, at):
        start = at + len(markup.parameter_open)
        close = text.find(markup.parameter_name_close, start)
        if close < 0 or text.find(markup.parameter_open, start, close) >= 0 or text.find(markup.tool_close, start, close) >= 0:
            return None
        key = text[slice(*strip_span(text, start, close))]
        at = skip_space(text, close + len(markup.parameter_name_close))
        if not text.startswith(markup.value_open, at):
            return None
        at += len(markup.value_open)
        end = (outside_strings if json_valued(tools, name, key) else str.find)(text, markup.parameter_close, at)
        if end < 0:
            return None
        parameters.append((key, text[at:end]))
        at = skip_space(text, end + len(markup.parameter_close))
        if not text.startswith((markup.parameter_open, markup.tool_close), at):
            return None
    return name, parameters, text[at + len(markup.tool_close):]


def strip_span(text, start, end):
    """The bounds of TEXT[START:END] without its leading and trailing
    whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def json_valued(tools, name, key):
    """Whether the served parser reads KEY of a call to NAME as JSON: the
    offered tool constrains its arguments (`strict` is not false and it
    declares parameters) and the key's schema, a declared property or else
    the additional-property schema, admits no string."""
    function = next((tool['function'] for tool in tools or () if tool['function']['name'] == name), None)
    if function is None or function.get('strict') is False or not isinstance(function.get('parameters'), dict):
        return False
    fields = function['parameters']
    if key in (fields.get('properties') or {}):
        schema = fields['properties'][key]
    elif 'patternProperties' in fields:
        return False
    else:
        schema = fields.get('additionalProperties')
    return isinstance(schema, dict) and not admits_string(schema)


# The type names the served coercion reads as a JSON type other than a
# string, aliases included (frontend-format chatml_decode.rs
# `json_type_name_v1`); any other name keeps its value as text.
NON_TEXT_TYPES = {
    'integer', 'int', 'int32', 'int64', 'uint', 'uint32', 'uint64', 'long', 'short', 'unsigned',
    'number', 'float', 'float32', 'float64', 'double', 'boolean', 'bool', 'object', 'dict',
    'array', 'arr', 'list', 'sequence', 'null',
}


def text_type(kind):
    """Whether the served coercion keeps a value of the type name KIND as
    raw text: a string type or alias, or a name it reads as no other type."""
    return isinstance(kind, str) and kind.strip().lower() not in NON_TEXT_TYPES


def admits_string(schema):
    """Whether a value of SCHEMA could be a string, so its text is raw."""
    if not isinstance(schema, dict):
        return True
    kind = schema.get('type')
    if isinstance(kind, str):
        return text_type(kind)
    if isinstance(kind, list):
        return any(text_type(name) for name in kind)
    for keyword in ('anyOf', 'oneOf'):
        if isinstance(schema.get(keyword), list):
            return any(admits_string(branch) for branch in schema[keyword])
    if isinstance(schema.get('enum'), list):
        return any(isinstance(value, str) for value in schema['enum'])
    if 'const' in schema:
        return isinstance(schema['const'], str)
    # With no type information the served parser keeps the text as a string.
    return not any(keyword in schema for keyword in ('properties', 'items', 'minimum', 'maximum'))


def outside_strings(text, marker, start=0):
    """The first MARKER in the JSON text that begins at START in TEXT,
    outside its string literals, or -1."""
    in_string = escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif text.startswith(marker, index):
            return index
    return -1


class HTTPStatusError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f'HTTP {status}')


class Transport:
    """No retries, redirects, or protocol fallback. Deadlines include stream drain."""
    def __init__(self, url, timeout=300, key=None):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('URL must be an HTTP(S) base without credentials, query or fragment')
        self.parsed, self.timeout, self.key = parsed, timeout, key
        self.url = url
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.active_sockets = set()
        self.observed = threading.local()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for sock in list(self.active_sockets):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def last_status(self):
        return getattr(self.observed, 'status', None)

    def chunks(self, path, body, deadline, cap):
        self.observed.status = None
        if self.cancelled.is_set():
            raise InterruptedError('cancelled')
        parsed = self.parsed
        connection_type = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('deadline')
        conn = connection_type(parsed.hostname, parsed.port, timeout=remaining)
        payload = None if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        if payload and len(payload) > MAX_REQUEST:
            raise ValueError('request byte ceiling')
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        sock = response = None
        try:
            conn.connect()
            sock = conn.sock
            with self.lock:
                self.active_sockets.add(sock)
            if self.cancelled.is_set():
                raise InterruptedError('cancelled')
            conn.request('GET' if body is None else 'POST', parsed.path.rstrip('/') + path, payload, headers)
            response = conn.getresponse()
            self.observed.status = response.status
            if response.status != 200:
                raise HTTPStatusError(response.status)
            size = 0
            while not response.isclosed():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('deadline')
                # HTTP/1.0 servers can detach conn.sock after their headers.
                response.fp.raw._sock.settimeout(remaining)
                chunk = response.read1(4096)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:
                    raise ValueError('response byte ceiling')
                yield time.monotonic(), chunk
        finally:
            with self.lock:
                self.active_sockets.discard(sock)
            if response is not None:
                response.close()
            conn.close()

    def get(self, path):
        return b''.join(chunk for _, chunk in self.chunks(path, None, time.monotonic() + self.timeout, MAX_RESPONSE)).decode()

    def scrape(self):
        start = time.monotonic()
        try:
            text = self.get('/metrics')
            return dict(started=start, ended=time.monotonic(), text=text, error=None)
        except Exception as error:
            return dict(started=start, ended=time.monotonic(), text=None, error=type(error).__name__)


def sse_events(chunks):
    """SSE framing across arbitrary socket chunks, UTF-8 splits and CRLF lines."""
    buffer = b''
    data = []
    for at, chunk in chunks:
        buffer += chunk
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            line = line.rstrip(b'\r')
            if not line:
                if data:
                    yield at, b'\n'.join(data).decode('utf-8')
                    data = []
            elif line.startswith(b'data:'):
                data.append(line[5:].lstrip(b' '))
    if buffer or data:
        raise ValueError('truncated SSE event')


def token_id_list(ids):
    if not isinstance(ids, list) or not all(type(i) is int and i >= 0 for i in ids):
        raise ValueError('invalid token IDs')
    return ids


class TokenIds:
    """The token IDs of one streamed turn, from `return_token_ids`: the
    prompt's last IDs and every generated ID.

    Only the generated count is recorded. The IDs stay in memory so that an
    invalid session's turns can be re-parsed from their raw text
    (`raw_class`), independently of the parsed text and tool deltas the
    server streamed.
    """

    def __init__(self):
        self.reported, self.count, self.ids, self.prompts, self.prompt_tail = False, 0, [], 0, []

    def event(self, event):
        if event.get('prompt_token_ids') is not None:
            self.prompts += 1
            self.prompt_tail = token_id_list(event['prompt_token_ids'])[-PROMPT_TAIL_IDS:]

    def chunk(self, choice):
        if 'token_ids' not in choice:
            return
        ids = token_id_list(choice['token_ids'] or [])
        self.reported = True
        self.count += len(ids)
        self.ids.extend(ids[:max(0, RAW_TURN_IDS - len(self.ids))])

    def complete(self):
        return self.prompts == 1 and bool(self.prompt_tail) and self.reported and len(self.ids) == self.count


def stream_request(client, body, deadline, stall_threshold_ms=100, burst_window_ms=2, evidence=None):
    """One streamed chat request: (row, assistant message). `evidence` (a
    TokenIds) receives the stream's prompt and generated token IDs."""
    evidence = TokenIds() if evidence is None else evidence
    started = time.monotonic()
    row = dict(started=started, ended=started, elapsed_seconds=0, ttft_ms=None,
               output_deltas=0, inter_delta_ms=[], inter_delta_summary_ms=distribution([]),
               decode_seconds=None, decode_tokens_per_second=None, prompt_tokens=None,
               cached_tokens=None, output_tokens=None, accepted_draft_tokens=None, draft_tokens=None,
               finish_reason=None, failure=None, timed_out=False, status=None, tool_names=[],
               token_ids=None, parsed_calls=None)
    message = dict(role='assistant', content='')
    calls, times, output_events = {}, [], []
    done = False
    try:
        events = sse_events(client.chunks('/v1/chat/completions', body, min(deadline, started + client.timeout), MAX_RESPONSE))
        for at, data in events:
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise ValueError('server stream error')
            evidence.event(event)
            usage = event.get('usage') or {}
            for field, key in [('prompt_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]:
                if key in usage:
                    row[field] = usage[key] if type(usage[key]) is int and usage[key] >= 0 else None
            details = usage.get('prompt_tokens_details') or {}
            if 'cached_tokens' in details:
                value = details['cached_tokens']
                row['cached_tokens'] = value if type(value) is int and value >= 0 else None
            spec = (event.get('metrics') or {}).get('speculative_decoding') or {}
            for field, key in [('accepted_draft_tokens', 'num_accepted_draft_tokens'), ('draft_tokens', 'num_draft_tokens')]:
                if key in spec:
                    value = spec[key]
                    row[field] = value if type(value) is int and value >= 0 else None
            generated = False
            kinds = set()
            for choice in event.get('choices', []):
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta') or {}
                if delta.get('refusal') or choice.get('finish_reason') == 'content_filter':
                    raise ValueError('request refused')
                evidence.chunk(choice)
                for field in ('content', 'reasoning_content', 'reasoning'):
                    if delta.get(field):
                        if not isinstance(delta[field], str):
                            raise ValueError('invalid generated delta')
                        message[field] = message.get(field, '') + delta[field]
                        generated = True
                        kinds.add('reasoning' if field.startswith('reasoning') else 'content')
                for fragment in delta.get('tool_calls') or []:
                    index = fragment.get('index')
                    if type(index) is not int or index < 0 or index > 1024:
                        raise ValueError('invalid tool index')
                    call = calls.setdefault(index, dict(id='', type='function', function=dict(name='', arguments='')))
                    if fragment.get('id'):
                        call['id'] += fragment['id']
                    function = fragment.get('function') or {}
                    for field in ('name', 'arguments'):
                        if function.get(field):
                            call['function'][field] += function[field]
                            if field == 'arguments':
                                generated = True
                                kinds.add(f'tool_args:{index}')
                if choice.get('finish_reason'):
                    row['finish_reason'] = choice['finish_reason']
            if generated:
                times.append(at)
                output_events.append(dict(at=at, kinds=sorted(kinds)))
        if not done or not row['finish_reason']:
            raise ValueError('stream missing completion marker or finish reason')
        if calls:
            message['tool_calls'] = [calls[i] for i in sorted(calls)]
        row.update(token_ids=evidence.count if evidence.reported else None, parsed_calls=len(calls))
    except Exception as error:
        # Never persist endpoint text, generated source or authorization values.
        row['failure'] = type(error).__name__
        row['timed_out'] = isinstance(error, TimeoutError)
        row['status'] = getattr(error, 'status', None)
    finally:
        if 'events' in locals():
            events.close()
        if hasattr(client, 'last_status'):
            row['status'] = client.last_status()
        elif times:
            row['status'] = row['status'] or 200
    row['ended'] = time.monotonic()
    row['elapsed_seconds'] = row['ended'] - started
    row['output_deltas'] = len(times)
    if times:
        row['ttft_ms'] = (times[0] - started) * 1000
    row['inter_delta_ms'] = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    row['inter_delta_summary_ms'] = distribution(row['inter_delta_ms'])
    row.update(burst_metrics(output_events, burst_window_ms, stall_threshold_ms))
    row['output_event_kinds'] = [event['kinds'] for event in output_events]
    row['tpot_ms'] = ((row['elapsed_seconds'] * 1000 - row['ttft_ms']) / (row['output_tokens'] - 1)
                      if row['ttft_ms'] is not None and (row['output_tokens'] or 0) > 1 and not row['failure'] else None)
    row['stall_threshold_ms'] = stall_threshold_ms
    if len(times) > 1 and times[-1] > times[0]:
        row['decode_seconds'] = times[-1] - times[0]
        if row['output_tokens'] is not None and not row['failure']:
            row['decode_tokens_per_second'] = row['output_tokens'] / row['decode_seconds']
    return row, message


def request_body(profile, model, messages, tools=None, output=None, temperature=None, seed=None):
    body = dict(model=model, messages=messages, stream=True,
                max_tokens=output or profile['max_output_tokens'],
                temperature=profile['temperature'] if temperature is None else temperature)
    if not profile['no_stream_usage']:
        body['stream_options'] = {'include_usage': True}
    if tools:
        body['tools'] = tools
    if seed is not None:
        body['seed'] = seed
    return body


def detokenize(client, model, ids):
    raw = b''.join(chunk for _, chunk in client.chunks('/detokenize', dict(model=model, tokens=ids),
                                                        time.monotonic() + client.timeout, MAX_RESPONSE))
    text = json.loads(raw)['prompt']
    if not isinstance(text, str):
        raise ValueError('invalid detokenized text')
    return text


def raw_turn(markup, opening, text):
    """One turn's raw text read with the template's markup: (reasoning,
    closed, after), or a class when it is not one well-formed turn.

    OPENING (the decoded prompt tail) must end in an assistant header that
    opens the thinking block, or leaves it to the generation to open. TEXT
    must end with exactly one end token and hold no turn markup, second end
    token or second reasoning open, and its reasoning no call markup.
    """
    at = max(opening.rfind(token) for token in markup.turn_open)
    if at < 0 or not opening.startswith(markup.assistant, at):
        return 'no assistant header'
    header = opening[at + len(markup.assistant):].strip()
    if header == '' and text.lstrip().startswith(markup.reasoning_open):
        text = text.lstrip()[len(markup.reasoning_open):]
    elif header != markup.reasoning_open:
        return 'thinking not open'
    end = next((m for m in markup.end if text.endswith(m)), None)
    if end is None:
        return 'no end token'
    body = text[:-len(end)]
    if any(m in body for m in markup.turn_open + (markup.reasoning_open,) + markup.end):
        return 'turn markup'
    reasoning, close, after = body.partition(markup.reasoning_close)
    if any(m in reasoning for m in markup.call_markup):
        return 'call markup'
    return reasoning, bool(close), after


def last_turn_class(markup, reasoning, closed, after):
    """RAW_REASONING_ONLY for nonempty reasoning that was never closed, and
    RAW_MALFORMED_CALL for reasoning closed and followed only by one tool
    region without a function header; else 'empty' or 'output after
    reasoning' (an answer or a well-formed call)."""
    if not closed:
        return RAW_REASONING_ONLY if reasoning.strip() else 'empty'
    region = after.strip()
    if (region.startswith(markup.tool_open) and region.endswith(markup.tool_close)
            and region.count(markup.tool_open) == 1 and region.count(markup.tool_close) == 1
            and not markup.names_function(region[len(markup.tool_open):-len(markup.tool_close)])
            and markup.reasoning_close not in region):
        return RAW_MALFORMED_CALL
    return 'output after reasoning'


def raw_calls(markup, after, tools=()):
    """The content and calls of the text after a reasoning close: (content,
    [(name, [(parameter, raw value)])]), or None unless it is content
    followed only by whitespace-separated well-formed tool regions in the
    declared grammar of the offered TOOLS."""
    content, opened, rest = after.partition(markup.tool_open)
    if any(m in content for m in markup.call_markup):
        return None
    calls = []
    while opened:
        region = markup.region(rest, tools)
        if region is None:
            return None
        name, parameters, rest = region
        calls.append((name, parameters))
        between, opened, rest = rest.partition(markup.tool_open)
        if between.strip():
            return None
    return content, calls


def calls_class(markup, reasoning, closed, after, answer, tools=()):
    """RAW_CALLS_AGREE when a tool-call turn's raw text agrees with the
    parsed ANSWER: the same reasoning and content (whitespace-trimmed), and
    one well-formed tool region per parsed call, in order, with its function
    name and exactly its arguments' names and values. Else the first
    disagreement."""
    if not closed:
        return 'no reasoning close'
    parsed = markup.calls(after, tools)
    if parsed is None:
        return 'malformed call region'
    content, calls = parsed
    if reasoning.strip() != (answer.get('reasoning') or answer.get('reasoning_content') or '').strip():
        return 'reasoning differs'
    if content.strip() != (answer.get('content') or '').strip():
        return 'content differs'
    streamed = answer.get('tool_calls') or []
    if len(calls) != len(streamed):
        return 'calls differ'
    for (name, parameters), call in zip(calls, streamed):
        try:
            arguments = json.loads(call['function']['arguments'])
        except (ValueError, RecursionError):
            return 'calls differ'
        if (name != call['function']['name'] or not isinstance(arguments, dict)
                or sorted(k for k, _ in parameters) != sorted(arguments)
                or not all(markup.same_value(v, arguments[k]) for k, v in parameters)):
            return 'calls differ'
    return RAW_CALLS_AGREE


def raw_class(client, model, evidence, answer=None, tools=()):
    """An independent parse of one turn's raw text, recorded as a class
    without the text: the last turn's shape (`last_turn_class`) when ANSWER
    is None, else whether a tool-call turn agrees with its parsed ANSWER
    (`calls_class`), both in the served frontend's markup (MARKUP).

    The prompt's last IDs and every generated ID are decoded through
    `/detokenize` (the tokenizer, not the chat parser) and read by
    `MARKUP.turn`. 'unavailable' means the IDs were incomplete or not decoded.
    """
    if not evidence.complete() or not evidence.ids:
        return 'unavailable'
    if MARKUP is None:
        raise ValueError('the raw re-parse needs the frontend declaration')
    try:
        opening, text = [detokenize(client, model, ids) for ids in (evidence.prompt_tail, evidence.ids)]
    except Exception:
        return 'unavailable'
    parts = MARKUP.turn(opening, text)
    if isinstance(parts, str):
        return parts
    return MARKUP.last_class(*parts) if answer is None else calls_class(MARKUP, *parts, answer, tools)


def request_seed(seed, concurrency, case, turn):
    raw = f'{seed}/C{concurrency}/{case}/turn-{turn}/sampling-v1'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & ((1 << 63) - 1)


def cache_salt(profile, concurrency, case):
    """The session's `cache_salt` under `--cache-salt session`, else None.
    It includes the run ID, so no other run's or session's cached prefix can
    absorb the session's prefill."""
    if profile.get('cache_salt', 'none') != 'session':
        return None
    raw = f"{profile.get('run_id', '')}/{profile['seed']}/C{concurrency}/{case}/cache-salt-v1".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def session(client, model, profile, manifest, files, concurrency, index, phase):
    task = task_for(manifest, files, profile['seed'], concurrency, index)
    salt = cache_salt(profile, concurrency, task['id'])
    workspace = Workspace(files, task['tool_result_bytes'])
    started = time.monotonic()
    deadline = started + profile['session_timeout_seconds']
    messages = [dict(role='system', content=manifest['setup']), dict(role='user', content=task['prompt'])]
    turns, failure, finished = [], None, False
    # Each turn's token IDs and parsed answer, for re-parsing an invalid session.
    parsed = []
    tool_seconds = 0.0
    runner_error = None
    try:
        for turn_index in range(profile['max_turns']):
            if getattr(client, 'cancelled', threading.Event()).is_set():
                failure = 'cancelled'
                break
            seed = request_seed(profile['seed'], concurrency, task['id'], turn_index)
            body = request_body(profile, model, messages, manifest['tools'], seed=seed)
            if not profile.get('no_token_evidence'):
                body['return_token_ids'] = True
            # The arms compare request identity without the run's salt.
            identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if salt:
                body['cache_salt'] = salt
            evidence = TokenIds()
            row, answer = stream_request(client, body, deadline, profile['stall_threshold_ms'], profile['burst_window_ms'],
                                         evidence)
            parsed.append((evidence, answer))
            row.update(seed=seed, turn_index=turn_index,
                       request_sha256=identity,
                       answer_sha256=hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest())
            turns.append(row)
            if row['failure'] or row['finish_reason'] == 'length':
                failure = row['failure'] or 'output token limit'
                row['failure'] = failure
                break
            calls = answer.get('tool_calls', [])
            messages.append(answer)
            if not calls:
                finished = bool(answer.get('content', '').strip())
                break
            tool_started = time.monotonic()
            for call in calls:
                if not call['id']:
                    workspace.tool_errors += 1
                    failure = 'missing tool call id'
                    break
                name = call['function']['name']
                row['tool_names'].append(name)
                result = workspace.call(name, call['function']['arguments'])
                messages.append(dict(role='tool', tool_call_id=call['id'], content=result))
            tool_seconds += time.monotonic() - tool_started
            if failure:
                break
    except Exception as error:
        runner_error = type(error).__name__
        failure = 'runner error: ' + runner_error
    if not finished and not failure and len(turns) == profile['max_turns']:
        failure = 'turn safety ceiling'
    valid = valid_session(workspace, finished, failure)
    if not valid and not failure:
        failure = INVALID_SESSION
        for index, (row, (evidence, answer)) in enumerate(zip(turns, parsed)):
            row['raw_class'] = raw_class(client, model, evidence, None if index == len(turns) - 1 else answer,
                                         manifest['tools'])
    ended = time.monotonic()
    return dict(index=index, case_id=task['id'], phase=phase, focus_path=task['focus_path'], task_kind=task['kind'],
                prompt_sha256=task['prompt_sha256'], tool_result_bytes=task['tool_result_bytes'], cache_salt=salt,
                started=started, ended=ended, elapsed_seconds=ended - started, turns=turns,
                tool_seconds=tool_seconds, model_seconds=sum(t['elapsed_seconds'] for t in turns),
                read_before_edit=bool(workspace.read_paths), patch_applied=workspace.patched,
                finished=finished, valid=valid, tool_errors=workspace.tool_errors,
                tool_results=workspace.results, truncated_tool_results=workspace.truncated,
                argument_errors=workspace.argument_errors, failure=failure, runner_error=runner_error,
                slot=index % concurrency, slot_sequence=index // concurrency)


def run_level(client, model, profile, manifest, files, concurrency, on_partial=None):
    outcomes = []
    lock = threading.Lock()
    stop = threading.Event()
    warmup_count = concurrency * profile['warmup_waves']
    started = time.monotonic()
    start, end = started, started + profile['seconds']
    before = boundary = {'text': None, 'error': 'window not started'}
    after = {'text': None, 'error': 'drain scrape not completed'}
    startup_before = startup_after = None
    cap_reached = False
    aborted = None

    def drive(index, phase):
        at = time.monotonic()
        try:
            result = session(client, model, profile, manifest, files, concurrency, index, phase)
        except Exception as error:
            # Protect the sweep even if task construction or result finalization fails.
            finished = time.monotonic()
            result = dict(index=index, case_id=f'coding-{index + 1:06d}', phase=phase, started=at,
                          ended=finished, elapsed_seconds=finished-at, turns=[], valid=False, finished=False,
                          read_before_edit=False, patch_applied=False, tool_errors=0,
                          runner_error=type(error).__name__, failure='runner error: ' + type(error).__name__)
        if phase != 'warmup' and result['ended'] > end:
            result['phase'] = 'drain'
        with lock:
            outcomes.append(result)
        return result

    def warmup(slot):
        for wave in range(profile['warmup_waves']):
            if stop.is_set():
                return
            drive(wave * concurrency + slot, 'warmup')

    def worker(slot):
        nonlocal cap_reached
        # Each slot gets the same case sequence regardless of other slots' speed.
        for wave in range(profile['max_sessions_per_c']):
            if stop.is_set() or time.monotonic() >= end:
                return
            drive(warmup_count + wave * concurrency + slot, 'measured')
        if time.monotonic() < end:
            cap_reached = True

    def report(after):
        with lock:
            saved = sorted(outcomes, key=lambda result: result['index'])
        measured_end = max(start + 1e-9, min(end, time.monotonic())) if aborted else end
        summary = summarize(saved, start, measured_end, concurrency, profile['stall_threshold_ms'])
        summary.update(aborted=aborted, target_window_end=end, burst_window_ms=profile['burst_window_ms'],
                       stall_threshold_ms=profile['stall_threshold_ms'], sessions=saved,
                       drain_seconds=max(0, max((r['ended'] for r in saved), default=measured_end) - measured_end))
        summary['insufficient'] = (aborted or ('session cap reached before measured interval ended' if cap_reached else
                                   'too few completed measured sessions' if summary['measured_completed'] < profile['min_samples'] else None))
        delta = metrics_delta(before, boundary)
        drain_delta = metrics_delta(boundary, after)
        summary['server_metrics'] = dict(scope='measured window; scrape timing uncertainty retained',
                                        before=before, after=boundary, **delta,
                                        histograms=histogram_reports(delta, profile['stall_threshold_ms']),
                                        sample_interval_seconds=max(0, boundary.get('ended', end) - before.get('ended', start)),
                                        boundary_lag_ms=max(0, boundary.get('started', end) - end) * 1000,
                                        boundary_scrape_seconds=max(0, boundary.get('ended', end) - boundary.get('started', end)))
        summary['prefill'] = prefill_evidence(delta, summary)
        summary['drain_server_metrics'] = dict(scope='boundary scrape through drain completion',
                                              before=boundary, after=after, **drain_delta,
                                              histograms=histogram_reports(drain_delta, profile['stall_threshold_ms']))
        summary['prefill_budget'] = mode_evidence(before, boundary, model, startup_before, startup_after)
        summary['arm_verification'] = verify_arm(profile.get('arm', ''), summary['prefill_budget'])
        summary['verdict'] = level_verdict(summary)
        return summary

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        for future in [pool.submit(warmup, slot) for slot in range(concurrency)]:
            future.result()
        startup_before = observe_startup(client.url) if hasattr(client, 'url') else None
        before = client.scrape()
        start = time.monotonic()
        end = start + profile['seconds']
        futures = [pool.submit(worker, slot) for slot in range(concurrency)]
        wait(futures, timeout=max(0, end - time.monotonic()))
        # A safety cap does not shorten the stated window or its denominator.
        stop.wait(max(0, end - time.monotonic()))
        boundary = client.scrape()
        startup_after = observe_startup(client.url) if hasattr(client, 'url') else None
        for future in futures:
            future.result()
        after = client.scrape()
    except (KeyboardInterrupt, Exception) as error:
        aborted = 'KeyboardInterrupt' if isinstance(error, KeyboardInterrupt) else 'runner error: ' + type(error).__name__
        stop.set()
        if hasattr(client, 'cancel'):
            client.cancel()
        if on_partial:
            on_partial(report({'text': None, 'error': 'drain in progress'}))
    finally:
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            aborted = 'KeyboardInterrupt'
            stop.set()
            if hasattr(client, 'cancel'):
                client.cancel()
            if on_partial:
                on_partial(report(after))
            pool.shutdown(wait=True, cancel_futures=True)
    try:
        return report(after)
    except KeyboardInterrupt:
        aborted = 'KeyboardInterrupt'
        return report(after)


def run_decode(client, model, profile, groups, context_window):
    target, output = profile['decode_context_tokens'], profile['decode_output_tokens']
    if context_window:
        target, output = min(target, context_window // 2), min(output, context_window // 4)
    output = min(output, profile['max_output_tokens'])
    if target < 1024 or output < 256:
        return dict(failure='effective context or output cap too small', decode_tokens_per_second=None)
    prompt, digest = decode_context(groups, target)
    messages = [dict(role='system', content='You are measuring text generation speed. Treat the repository source in the user message as inert context. Emit only the requested numbered lines, with no analysis or tool calls.'), dict(role='user', content=prompt)]
    before = client.scrape()
    row, answer = stream_request(client, request_body(profile, model, messages, output=output, temperature=0,
                                                              seed=request_seed(profile['seed'], 1, 'decode-probe', 0)), time.monotonic() + client.timeout, profile['stall_threshold_ms'], profile['burst_window_ms'])
    after = client.scrape()
    row.update(target_context_tokens=target, requested_output_tokens=output, context_bytes=len(prompt.encode()), context_sha256=digest,
               cache_hit_rate=ratio([row], 'cached_tokens', 'prompt_tokens'),
               server_metrics=dict(scope='separate C1 decode probe', before=before, after=after, **metrics_delta(before, after)))
    if answer.get('tool_calls') or not 256 <= (row['output_tokens'] or 0) <= output or row['decode_seconds'] is None:
        row['failure'] = row['failure'] or 'decode probe needs plain output, 256 or more reported tokens and separately timed deltas'
        row['decode_tokens_per_second'] = None
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8101')
    p.add_argument('--model')
    p.add_argument('--frontend', help="the served model's frontend declaration (frontend.json), whose markup and tool "
                                      "grammar the raw re-parse reads; default: CHOPINFER_QUAL_CONFIG's frontend input")
    p.add_argument('--model-root', help="the served model's root, whose agentic_markup.py re-parses a format that declares "
                                        "no library tool grammar; default: CHOPINFER_QUAL_MODEL_ROOT, else the "
                                        "inventoried model whose frontend declares the served model")
    p.add_argument('--levels', default='1,2,4,8,12,16,24,32,48,64')
    p.add_argument('--corpus', type=Path, default=CORPUS,
                   help='a corpus directory with a verified manifest: the pinned Runyard corpus (default), or a larger '
                        'one staged under $CHOPINFER_DATA by `agentic_workload.py stage`')
    p.add_argument('--cache-salt', choices=('session', 'none'), default='session',
                   help='session (default): send each session its own cache_salt, so no other session or run absorbs '
                        'its prefill and the prefill budgets see real pressure; none: share prefixes across sessions')
    p.add_argument('--seed', default='runyard-coding-v1')
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--warmup-waves', type=int, default=1)
    p.add_argument('--max-sessions-per-c', type=int, default=64)
    p.add_argument('--min-samples', type=int, default=2)
    p.add_argument('--max-turns', type=int, default=64)
    p.add_argument('--max-output-tokens', type=int, default=131072)
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--request-timeout-seconds', type=float, default=300)
    p.add_argument('--session-timeout-seconds', type=float, default=300)
    p.add_argument('--decode-context-tokens', type=int, default=32768)
    p.add_argument('--decode-output-tokens', type=int, default=2048)
    p.add_argument('--no-stream-usage', action='store_true')
    p.add_argument('--no-token-evidence', action='store_true',
                   help='do not request return_token_ids; invalid sessions then cannot count as model outcomes')
    p.add_argument('--no-decode', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='run against an owned loopback fake server, never --url')
    p.add_argument('--label', default='')
    p.add_argument('--arm', default='', help='comparison arm, e.g. tbt or fixed; excluded from profile digest')
    p.add_argument('--stall-threshold-ms', type=float, default=100)
    p.add_argument('--burst-window-ms', type=float, default=2)
    p.add_argument('--fake-prefill-budget', choices=('fixed', 'tbt'), default='fixed',
                   help='independent fake-server mode for validation, not inferred from --arm')
    return p


def validate(p, args):
    try:
        levels = [int(c) for c in args.levels.split(',')]
    except ValueError:
        p.error('levels must be comma-separated integers')
    if not levels or len(levels) > 64 or len(set(levels)) != len(levels) or min(levels) < 1 or max(levels) > 1024:
        p.error('levels must be unique integers from 1 to 1024')
    bounds = {'burst_window_ms': (.001, 100), 'stall_threshold_ms': (.001, 3600000), 'seconds': (0.001, 86400), 'warmup_waves': (0, 100), 'max_sessions_per_c': (2, 10000),
              'min_samples': (1, args.max_sessions_per_c), 'max_turns': (1, 64), 'max_output_tokens': (1, 131072),
              'temperature': (0, 2), 'request_timeout_seconds': (.001, 3600),
              'session_timeout_seconds': (args.request_timeout_seconds, 86400),
              'decode_context_tokens': (1024, 131072), 'decode_output_tokens': (256, 8192)}
    for name, (low, high) in bounds.items():
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            p.error(f'{name} must be in [{low}, {high}]')
    if not args.seed.strip() or len(args.seed.encode()) > 256:
        p.error('seed must contain 1 to 256 bytes')
    if sum(levels) * (args.warmup_waves + args.max_sessions_per_c) > 100000:
        p.error('profile exceeds 100000 sessions')
    return levels


class ReParserUnavailable(ValueError):
    """No raw re-parser for the served frontend: `main` prints it, since its
    message names the remedies and the receipt records only its type."""


def library_grammar(frontend):
    """The library tool grammar (TOOL_GRAMMARS) the declaration's format
    declares in `format.parser.tool_grammar`, or None."""
    markup = frontend.get('format') if isinstance(frontend, dict) else None
    markup = markup.get('parser') if isinstance(markup, dict) else None
    grammar = markup.get('tool_grammar') if isinstance(markup, dict) else None
    return grammar if grammar in TOOL_GRAMMARS else None


def inventoried(model):
    """{(frontend path, model directory)} of each inventoried model whose
    frontend declares MODEL, the served model ID. Models may share one
    frontend file."""
    root, found = Path(common.ROOT), set()
    for inventory in sorted(root.glob('models/*/model.toml')):
        declared = json.loads(declaration.path(inventory.parent.name, root).read_text())['inputs'].get('frontend')
        path = root / declared if isinstance(declared, str) else None
        if path and path.is_file() and model in [m.get('id') for m in json.loads(path.read_text()).get('models', [])]:
            found.add((path, inventory.parent))
    return found


def inventoried_frontend(model):
    """The one inventoried frontend file that declares MODEL (`inventoried`)."""
    paths = {path for path, _ in inventoried(model)}
    if len(paths) != 1:
        raise ReParserUnavailable(f'no single inventoried frontend declares model {model!r}: pass --frontend')
    return paths.pop()


def inventoried_root(model):
    """The model directory of the one inventoried model whose frontend
    declares MODEL (`inventoried`), or None."""
    roots = {directory for _, directory in inventoried(model)}
    return roots.pop() if len(roots) == 1 else None


def markup_for(frontend, args, model):
    """(re-parser, its SHA-256) of FRONTEND: the shared `Markup`, with no
    digest, when its format declares a library tool grammar, else the class
    `Markup` of
    `agentic_markup.py` (model_modules) at the model root: --model-root,
    else CHOPINFER_QUAL_MODEL_ROOT, else the directory of the one inventoried
    model whose frontend declares MODEL.
    The re-parse stays independent of the served parser, so neither reads
    frontend-format or asks the server to parse."""
    if library_grammar(frontend):
        return Markup(frontend), None
    root = model_modules.model_root(getattr(args, 'model_root', None)) or inventoried_root(model)
    try:
        loaded = model_modules.load('agentic_markup', root) if root is not None else None
    except ValueError as error:
        raise ReParserUnavailable(str(error)) from error
    if loaded is None:
        where = (f'the model root {root}' if root is not None
                 else 'no single inventoried model declares it (pass --model-root)')
        raise ReParserUnavailable(f"the frontend's format declares no library tool grammar ({', '.join(TOOL_GRAMMARS)}) "
                         f"and {where} holds no agentic_markup.py: declare a library grammar in "
                         "format.parser.tool_grammar, or supply the model's own re-parser as "
                         "agentic_markup.py in its model root (tools/qual/README.md)")
    module, digest = loaded
    return module.Markup(frontend), digest


def load_markup(args, model):
    """(re-parser, receipt record) of the served frontend declaration
    (`markup_for`): --frontend, else the qualification config's frontend
    input, else the inventoried model's whose frontend declares MODEL, the
    served model ID. The record holds the declaration's SHA-256, the
    re-parser's grammar and the SHA-256 of a model's own re-parser (None for
    the shared one). A dry run (whose fake server returns no token IDs) or
    --no-token-evidence re-parses no raw turn, so needs none."""
    if args.frontend:
        data = Path(args.frontend).read_bytes()
        frontend = json.loads(data)
    elif args.dry_run or args.no_token_evidence:
        return None, None
    elif os.environ.get('CHOPINFER_QUAL_CONFIG'):
        data, frontend = common.input_bytes('frontend'), common.frontend()
    else:
        data = inventoried_frontend(model).read_bytes()
        frontend = json.loads(data)
    markup, reparser_sha256 = markup_for(frontend, args, model)
    return markup, dict(sha256=hashlib.sha256(data).hexdigest(), tool_grammar=markup.grammar,
                        reparser_sha256=reparser_sha256)


def level_passed(level):
    return level_verdict(level)['passed']


def markdown(result):
    lines = ['# Agentic mixed workload', '', f"Revision: `{REVISION}`. Fake server: {result['dry_run']}.",
             f"Receipt: {result['receipt_kind']}. Aborted: {result.get('aborted') or 'no'}.", '',
             '| C | Mode | Valid/completed | Errors/timeouts | Output tok/s | Prefill tok/step | TTFT p50/p95 ms | Server step evidence | Client burst p50/p95/p99/max ms | Client stalls/1000 gaps |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def number(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    def quantiles(values, names, samples):
        # A star marks a tail with fewer samples than it needs (TAIL_SAMPLES).
        return '/'.join(number(values[k]) + ('*' if samples < TAIL_SAMPLES.get(k, 0) else '') for k in names)
    def percentiles(values, names, samples):
        return quantiles(values, names, samples) + f' (n={samples})'
    for level in result['levels']:
        gaps = level['step_gap_ms']
        histograms = level['server_metrics']['histograms']
        primary = [h for h in histograms if h['name'] == 'vllm:inter_token_latency_seconds']
        if not primary:
            primary = [h for h in histograms if h['name'].endswith('step_execution_seconds')]
        evidence = '; '.join(f"{h['name']} {h['labels']}: p95 upper {quantiles(h['quantile_upper_ms'], ['p95'], int(h['count'] or 0))} ms "
                             f"(n={int(h['count'] or 0)}), stalls/1000 [{number(h['stall_per_1000_lower'])}, "
                             f"{number(h['stall_per_1000_upper'])}]" for h in primary) or 'unavailable'
        prefill = level['prefill']['server_prefill_tokens_per_step']
        lines.append(f"| {level['concurrency']} | {level['prefill_budget']['mode'] or 'unknown'} | "
                     f"{level['valid_sessions']}/{level['measured_completed']} | {level['errors']}/{level['timeouts']} | "
                     f"{number(level['output_tokens_per_second']['value'])} | {number(prefill['value'])} (n={prefill['samples']}) | "
                     f"{percentiles(level['ttft_ms'], ('p50', 'p95'), level['ttft_ms']['samples'])} | {evidence} | "
                     f"{percentiles(gaps, ('p50', 'p95', 'p99', 'max'), gaps['samples'])} | "
                     f"{number(level['client_stalls']['per_1000_gaps'])} (n={level['client_stalls']['eligible_gaps']}) |")
        for warning in level['arm_verification'].get('warnings', []):
            lines.append('Arm activity: ' + warning)
        if not level['arm_verification']['passed']:
            lines.append(f"Arm verification failed: {level['arm_verification']['reason']}")
        verdict = level_verdict(level)
        if verdict['model_outcome_count']:
            lines.append(f"C{level['concurrency']} model outcomes: {verdict['model_outcomes']} "
                         f"(limit {verdict['model_outcome_limit']} of {verdict['measured_sessions']} measured sessions)")
        if verdict['problems']:
            lines.append(f"C{level['concurrency']} failed: " + '; '.join(verdict['problems']))
    if result.get('decode'):
        lines += ['', f"Separate C1 decode tok/s: {number(result['decode'].get('decode_tokens_per_second'))}."]
    lines += ['', 'n is the sample count of each figure: prefill steps, turns, histogram observations or client gaps. '
              'A star marks a percentile with fewer samples than it needs (20 for p95, 100 for p99).',
              'Server step histograms are primary stall evidence. Client bursts are a secondary proxy affected by transport and frontend holdback.',
              'Kind/tool transitions are marked as possible holdback; within-run gaps can still contain undetectable holdback or client jitter.',
              'Measured and drain counters are separate. Validity checks tool flow, not semantic correctness.', '']
    return '\n'.join(lines)


def main(argv=None):
    global MARKUP
    p = parser()
    args = p.parse_args(argv)
    levels = validate(p, args)
    common.startup()
    try:
        manifest, groups = load_corpus(args.corpus)
    except (OSError, ValueError, KeyError) as error:
        p.error(f'corpus: {error}')
    # The decode probe always reads the pinned Runyard source.
    decode_groups = load_corpus()[1]
    # The corpus is identified by its digest and commit, never its local path.
    profile = {k: v for k, v in vars(args).items()
               if k not in ('url', 'model', 'label', 'arm', 'dry_run', 'fake_prefill_budget', 'frontend', 'model_root',
                            'corpus')}
    profile['levels'] = levels
    profile.update(max_request_bytes=MAX_REQUEST, max_response_bytes=MAX_RESPONSE,
                   max_tool_result_bytes=max_tool_result_bytes(manifest),
                   case_assignment='slot-strided-v1', request_seed_derivation='sha256-case-turn-v1')
    result = dict(workload_schema=REVISION, source_revision=manifest['revision'], source_commit=manifest['source_commit'],
                  corpus_digest=manifest['corpus_digest'],
                  workload_digest=hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  profile=profile, label=args.label, arm=args.arm, run_id=uuid.uuid4().hex,
                  frontend=None, dry_run=args.dry_run, levels=[], decode=None, aborted=None, passed=False, level_receipts=[])
    identity = dict(url=None if args.dry_run else args.url, model=args.model, models=None, context_window=None)
    client = None

    def save(kind, saved_levels=None, aborted=None):
        snapshot = {**result, 'receipt_kind': kind, 'levels': list(result['levels'] if saved_levels is None else saved_levels),
                    'level_receipts': list(result['level_receipts'])}
        snapshot['profile_digest'] = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if aborted:
            snapshot['aborted'] = aborted
        snapshot['passed'] = (kind != 'partial' and not snapshot['aborted'] and bool(snapshot['levels'])
                              and all(level_passed(level) for level in snapshot['levels'])
                              and (not snapshot['decode'] or not snapshot['decode']['failure']))
        path = common.receipt('agentic-mixed', identity, snapshot, args.dry_run)
        with path.with_suffix('.md').open('x') as summary:
            summary.write(markdown(snapshot))
        return path, snapshot['passed']

    if args.dry_run:
        from agentic_fake import fake_server
        context = fake_server(groups['coding'], mode=args.fake_prefill_budget)
    else:
        context = nullcontext(args.url)
    try:
        with context as url:
            identity['url'] = url
            client = Transport(url, args.request_timeout_seconds, None if args.dry_run else os.environ.get('OPENAI_API_KEY'))
            listing = json.loads(client.get('/v1/models'))
            models = listing['data']
            model = args.model or models[0]['id']
            selected = [m for m in models if m['id'] == model]
            if len(selected) != 1:
                raise ValueError('selected model must occur exactly once in /v1/models')
            context_window = next((selected[0][k] for k in ('max_model_len', 'context_length')
                                   if type(selected[0].get(k)) is int and selected[0][k] > 0), None)
            identity.update(model=model, models=listing, context_window=context_window)
            MARKUP, frontend = load_markup(args, model)
            if MARKUP is not None:
                result['frontend'] = frontend
            if context_window:
                profile['max_output_tokens'] = min(profile['max_output_tokens'], max(1, context_window // 2))
            for c in levels:
                print(f'Running C{c}: warmup, {args.seconds:g}s measured interval, then drain', flush=True)
                def partial(level):
                    path, _ = save('partial', [level], level['aborted'])
                    result['level_receipts'].append(str(path))
                level = run_level(client, model, {**profile, 'arm': args.arm, 'run_id': result['run_id']}, manifest,
                                  groups['coding'], c, on_partial=partial)
                result['levels'].append(level)
                path, _ = save('level', [level], level.get('aborted'))
                result['level_receipts'].append(str(path))
                if level.get('aborted'):
                    result['aborted'] = level['aborted']
                    break
            if not args.no_decode and not result['aborted']:
                result['decode'] = run_decode(client, model, profile, decode_groups, context_window)
    except (KeyboardInterrupt, Exception) as error:
        result['aborted'] = 'KeyboardInterrupt' if isinstance(error, KeyboardInterrupt) else 'runner error: ' + type(error).__name__
        if isinstance(error, ReParserUnavailable):
            print(f'agentic_mixed: {error}', file=sys.stderr)
        if client:
            client.cancel()
    try:
        _, passed = save('partial' if result['aborted'] else 'sweep')
    except KeyboardInterrupt:
        result['aborted'] = 'KeyboardInterrupt'
        _, passed = save('partial')
    return 130 if result['aborted'] == 'KeyboardInterrupt' else 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
