"""Auxiliary device history: every recorded operation of a fixture's device
lifetime that lies outside its pinned bodies and its create range.

The history is compared line for line under the fixture's `history_policy`
and the same typed handle mapping as the pinned journals. The interleaving
of history, create and pinned operations is then checked for every retained
operation, so an operation cannot move across another owner's operation
unless a declared rewrite removes or merges it."""
import collections
import re
from engine import RAW, Unproven, require
from common import need

_SPAN = r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: \d+ \}, offset: \d+, length: \d+ \}'
# Every FakeDevice operation form. Numbered groups are typed identities.
FORMS = [
    ('launch', re.compile(r'launch (\d+) chop_\w+ (?:rows \d+ )?LaunchConfigV1 \{.*\} \[.*\]'), ('stream',)),
    ('launch', re.compile(r'launch r\d+ .+'), ()),
    ('upload', re.compile(rf'upload (\d+) {_SPAN} [0-9a-f]{{16}}'), ('stream',)),
    ('upload', re.compile(r'upload (?:encoder )?arena \d+'), ()),
    ('copy', re.compile(rf'copy (\d+) {_SPAN} {_SPAN}'), ('stream',)),
    # A copy between two ranks' buffers on `stream`; the contexts are host
    # contexts (rank slots), fixed and compared literally.
    ('copy', re.compile(rf'peer copy (\d+) \d+ {_SPAN} \d+ {_SPAN}'), ('stream',)),
    ('download', re.compile(rf'download (\d+) {_SPAN}'), ('stream',)),
    # A model's readback line (its download_line): the stream, the source and
    # the byte count read.
    ('download', re.compile(rf'readback (\d+) {_SPAN} \d+'), ('stream',)),
    ('zero', re.compile(rf'zero (\d+) {_SPAN}'), ('stream',)),
    ('synchronize', re.compile(r'synchronize (\d+)'), ('stream',)),
    ('capture', re.compile(r'capture (\d+)'), ('stream',)),
    ('captured', re.compile(r'captured (\d+) (\d+) \d+'), ('stream', 'issued')),
    ('replay', re.compile(r'replay (\d+) (\d+)'), ('stream', 'issued')),
    ('release', re.compile(r'release graph (\d+)'), ('issued',)),
    ('release', re.compile(r'release'), ()),
    ('record', re.compile(r'record (\d+) (\d+)'), ('stream', 'issued')),
    ('wait', re.compile(r'wait (\d+) (\d+)'), ('stream', 'issued')),
    ('event', re.compile(r'event (\d+)'), ('issued',)),
    ('stream', re.compile(r'stream (\d+)'), ('stream',)),
    # The returned handle, when a journal records it, maps as a buffer.
    ('allocate', re.compile(r'allocate \d+ \d+(?: -> raw: \d+)?'), ()),
    ('state', re.compile(r'state \d+'), ()),
    ('state', re.compile(r'serving kv \d+'), ()),
    ('resolve', re.compile(r'resolve (\d+) \S+'), ('function',)),
    ('configure', re.compile(r'configure (\d+) \d+'), ('function',)),
    ('module', re.compile(r'module (\d+) \S+'), ('function',)),
]

CLAIM = dict(
    pinned_journals='Every pinned journal body, line for line, after the declared journal policy and the fixture typed handle mapping.',
    auxiliary_create="The fixture's own create range, line for line, under create_policy and the same mapping.",
    auxiliary_history=('Every other operation the fixture device recorded from the start of its lifetime through its last '
                       'pinned checkpoint (setup and the history between pinned captures): launches, copies, uploads with '
                       'their payload FNV, downloads, zeros, allocations, releases, stream and event creation, event '
                       'record/wait, graph capture begin/end, replay and release. Compared line for line under '
                       'history_policy and the same mapping; unknown operation forms fail closed.'),
    ordering=('The relative order of every retained operation across owners (create, each pinned journal, history) '
              'is preserved; an operation inside a merged group or an expansion cannot be crossed.'),
    resources=('Issued IDs, allocation identity, rank, size, alignment and stream, module payloads and resolve '
               'ownership, under the declared additions, removals, resizes and module replacements.'),
)
NOT_COMPARED = [
    "Operations a device recorded after the fixture's last pinned checkpoint; each fixture reports their count.",
    'Upload contents beyond the recorded 64-bit FNV-1a of the journaled payload, and the contents of an arena upload '
    '(`upload arena N`), which FakeDevice journals by rank only.',
    'Which buffer a bare `release` frees: FakeDevice journals a buffer release without its handle, so only its place '
    'and count are compared.',
    'Host-side state that FakeDevice does not journal, kernel numerics, performance and source correctness.',
]


# The forms by their first word, so a line is matched only against its own.
_BY_WORD = collections.defaultdict(list)
for _form in FORMS:
    _word = _form[1].pattern.split(' ', 1)[0]
    assert _word.isalpha(), _form
    _BY_WORD[_word].append(_form)

# Journal bodies (create and pinned journals, never the history) also hold
# test annotations between operations. Each is an exact grammar with no
# device identity except a logits view's buffer handle. A label is a test's
# own name for a step, compared literally.
_LABEL = r'[A-Za-z0-9_.-]+(?: [A-Za-z0-9_.-]+)*'
_N = r'\d+'
ANNOTATIONS = [
    re.compile(rf'step {_LABEL}'),
    re.compile(rf'{_LABEL} view BundleLogitsViewV1 \{{ struct_size: {_N}, reserved: {_N}, '
               rf'context: BundleContextHandleV1 \{{ raw: {_N} \}}, rank: {_N}, element_type: {_N}, '
               rf'buffer: BundleDeviceBufferViewV1 \{{ struct_size: {_N}, reserved: {_N}, '
               rf'buffer: BundleDeviceBufferHandleV1 \{{ raw: {_N} \}}, offset: {_N}, length: {_N} \}}, '
               rf'row_stride_elements: {_N}, row_count: {_N}, live_width: {_N}, '
               rf'first_item_index: {_N}, reserved_tail: {_N} \}}'),
    re.compile(rf'{_LABEL} drafts OwnedDraftTokensV1 \{{ request_id: {_N}, tokens: \[(?:{_N}(?:, {_N})*)?\] \}}'),
]
BUFFER_HANDLE = re.compile(r'(BundleDeviceBufferHandleV1 \{ raw: )(\d+)( \})')


def form(line):
    """The (kind, match, roles) of `line`'s operation form, or None."""
    for kind, pattern, roles in _BY_WORD.get(line.split(' ', 1)[0], ()):
        m = pattern.fullmatch(line)
        if m:
            return kind, m, roles
    return None


def classify(line):
    found = form(line)
    if found is None:
        raise Unproven(f'unrecognized auxiliary operation form: {line[:200]}')
    return found


def stream_of(line):
    """The stream an operation runs on, or None: an operation without a
    stream (an encoder launch `launch rN`, an arena upload, a release), an
    annotation, or a line of no known form."""
    found = form(line)
    if found is None or 'stream' not in found[2]:
        return None
    return int(found[1][found[2].index('stream') + 1])


class Mapper:
    """The fixture's typed mapping, applied to every identity of a line's
    operation form and to every buffer handle.

    A stream with no recorded creation (a host stream, such as the one an
    encoder launches on) stays fixed, and may never equal the image of a
    created stream: fixed and mapped streams stay distinct. History lines are
    `closed`: every line must have a known operation form, and a fixed stream
    may not share its number with any issued resource. Create and pinned
    journal bodies are not closed in two exact ways: a line of no operation
    form must be one of ANNOTATIONS (only a view's buffer handle is mapped),
    and a fixed stream may share its number with an issued resource that is
    not a stream (an encoder's resolve 1 and host stream 1). Any other line
    is refused. Every other identity is mapped the same way in all owners."""

    def __init__(self, buffers, functions, streams, ids, closed=True):
        self.buffers, self.functions, self.streams, self.ids = buffers, functions, streams, ids
        self.closed = closed
        self.issued = set(ids) | set(ids.values())
        self.stream_images = set(streams.values())

    def identity(self, role, value):
        if role == 'function':
            require(value in self.functions, f'function ID {value} has no mapping')
            return self.functions[value]
        if role == 'stream':
            # Only a recorded stream creation maps a stream. An unrecorded
            # stream (for example a test's synthetic stream) stays fixed and
            # may not share a number with any issued resource.
            if value in self.streams:
                return self.streams[value]
            require(value not in self.stream_images,
                    f'stream {value} has no recorded creation and is the image of a created stream')
            require(not self.closed or value not in self.issued,
                    f'stream {value} has no recorded creation and shares its number with an issued resource')
            return value
        require(value in self.ids, f'{role} ID {value} was not issued by this device')
        return self.ids[value]

    def __call__(self, line):
        found = form(line)
        if found is None:
            require(not self.closed, f'unrecognized auxiliary operation form: {line[:200]}')
            require(any(a.fullmatch(line) for a in ANNOTATIONS), f'unrecognized journal line form: {line[:200]}')
            return BUFFER_HANDLE.sub(lambda r: f'{r[1]}{self.buffers.get(int(r[2]), int(r[2]))}{r[3]}', line)
        kind, m, roles = found
        for group in range(len(roles), 0, -1):
            new = self.identity(roles[group - 1], int(m[group]))
            line = line[:m.start(group)] + str(new) + line[m.end(group):]
        return RAW.sub(lambda r: f'raw: {self.buffers.get(int(r[1]), int(r[1]))}', line)


def owners(notes, bindings, create):
    """Each note's owner: the pinned journal whose positions hold it, the
    fixture create range, or the history."""
    owner = [None] * len(notes)
    for name, binding in bindings.items():
        for i in binding['positions']:
            need(0 <= i < len(notes), f'{name}: trace witness position outside the fixture prefix')
            require(owner[i] is None, f'pinned journals {owner[i]} and {name} share a recorded operation')
            owner[i] = name
    for i in range(create['start_note'], create['end_note']):
        if owner[i] is None:
            owner[i] = 'create'
    return [o or 'history' for o in owner]


def ranks(out, head_owner, label, positions, kept, trace, base_positions):
    """Record, for each head note of `label`, the lowest and highest base note
    index its line derives from. Declared additions have no source."""
    for k, source in enumerate(trace):
        t = positions[kept[k]]
        if source and head_owner[t] == label:
            out[t] = (min(base_positions[j] for j in source), max(base_positions[j] for j in source))


def check_order(head_owner, head_notes, rank):
    """Cross-owner order: every head operation must follow, in the base, all
    earlier head operations of every other owner."""
    highest = {}
    for t, label in enumerate(head_owner):
        r = rank.get(t)
        if r is None:
            continue
        lo, hi = r
        others = [v for k, v in highest.items() if k != label]
        if others and lo <= max(others):
            culprit = max((v, k) for k, v in highest.items() if k != label)[1]
            raise Unproven(f'operation order changed: head note {t + 1} ({label}: {head_notes[t][:160]}) '
                           f'now follows a later {culprit} operation')
        highest[label] = max(highest.get(label, -1), hi)


CAPTURED = re.compile(r'captured (\d+) (\d+) (\d+)')


def check_graph_nodes(notes, side):
    """Every captured graph's node count must equal the launches the device
    recorded between its capture begin and end. Node counts are then derived
    from the compared launches, never from a declaration alone."""
    open_ = {}
    for i, line in enumerate(notes):
        if line.startswith('capture '):
            stream = line.split()[1]
            require(stream not in open_, f'{side}: note {i + 1}: nested capture on stream {stream}')
            open_[stream] = i
        elif (m := CAPTURED.fullmatch(line)):
            require(m[1] in open_, f'{side}: note {i + 1}: graph capture ends without its begin')
            start = open_.pop(m[1])
            launches = sum(x.startswith('launch ') for x in notes[start + 1:i])
            require(launches == int(m[3]), f'{side}: note {i + 1}: graph has {m[3]} nodes but its capture recorded {launches} launches')


def graph_node_changes(expected, observed, policy, counts):
    """Accept node-count-only differences of captured graphs, exactly as many
    as `graph_nodes` declares. Both sides' counts were checked against the
    launches inside each capture, and those launches are compared and kept in
    order, so the declaration only states how many graphs changed size."""
    rules = policy.get('graph_nodes', [])
    need(isinstance(rules, list) and len(rules) <= 1 and all(isinstance(r, dict) and set(r) == {'count'} and
         type(r['count']) is int and r['count'] > 0 for r in rules), 'graph_nodes needs one exact positive count')
    changed = 0
    for i in range(min(len(expected), len(observed))):
        if expected[i] == observed[i]:
            continue
        e, o = CAPTURED.fullmatch(expected[i]), CAPTURED.fullmatch(observed[i])
        if rules and e and o and e[1] == o[1] and e[2] == o[2]:
            expected[i] = observed[i]
            changed += 1
    if rules:
        require(changed == rules[0]['count'], f'{changed} graph node-count changes, expected {rules[0]["count"]}')
        counts['graph_nodes'] += changed
    return expected


# The shared runtime's graph source, after and before the crate rename. A
# spec may name either; each revision is read at the path it has.
PIECE_SOURCES = ('crates/bundle-runtime/src/graphs.rs', 'crates/bundle-runtime-sm120/src/graphs.rs')
PIECE_SIGNATURE = 'pub fn pieces ( launches : usize ) - > Vec < Range < usize > > {'
PIECE_BODY = ('let bounds = [ 0 , launches / {a} , launches / {b} , launches ] ; bounds . windows ( 2 ) . map ( | pair | '
              'pair [ 0 ] . . pair [ 1 ] ) . filter ( | range | ! range . is_empty ( ) ) . collect ( ) }')
PIECE_USE = 'for range in pieces ( launches ) {'


RAW_STRING = re.compile(r'b?r(#*)"')


def strip_comments(text):
    """Rust executable source: comments removed (nested block comments,
    line comments) and every string, raw string, byte string and character
    literal replaced by the opaque token "LIT", so text inside a literal can
    never match code."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        word = i > 0 and (text[i - 1].isalnum() or text[i - 1] == '_')
        if depth:
            if text.startswith('/*', i):
                depth += 1
                i += 2
            elif text.startswith('*/', i):
                depth -= 1
                i += 2
            else:
                i += 1
        elif text.startswith('/*', i):
            depth = 1
            i += 2
            out.append(' ')
        elif text.startswith('//', i):
            while i < n and text[i] != '\n':
                i += 1
        elif not word and (m := RAW_STRING.match(text, i)):
            close = '"' + m[1]
            j = text.find(close, m.end())
            need(j >= 0, 'unterminated raw string in the piece policy source')
            out.append(' "LIT" ')
            i = j + len(close)
        elif c == '"' or (c == 'b' and not word and text.startswith('b"', i)):
            j = i + (2 if c == 'b' else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            need(j < n, 'unterminated string in the piece policy source')
            out.append(' "LIT" ')
            i = j + 1
        elif c == "'" and i + 2 < n and (text[i + 2] == "'" or (text[i + 1] == '\\' and (k := text.find("'", i + 2)) > 0)):
            j = i + 3 if text[i + 2] == "'" else k + 1
            out.append(' "LIT" ')
            i = j
        else:
            out.append(c)
            i += 1
    need(depth == 0, 'unterminated block comment in the piece policy source')
    return ''.join(out)


def tokens(text):
    return re.findall(r'\w+|[^\s\w]', text)


ASSIGNMENTS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^='}


def mutates_launches(words):
    """Any binding, mutable binding or assignment of `launches` other than
    the two recognized parameters."""
    for k, word in enumerate(words):
        if word != 'launches':
            continue
        if k and words[k - 1] in ('mut', 'let', 'ref'):
            return True
        op = ''
        for w in words[k + 1:k + 4]:
            if len(w) == 1 and w in '=+-*/%<>&|^!':
                op += w
            else:
                break
        if op in ASSIGNMENTS:
            return True
    return False


def piece_divisors(source):
    """The divisors of the recognized `pieces` implementation, or Invalid.
    In the executable source (comments removed, literals opaque) `pieces`
    must be defined exactly once, never aliased, with exactly the
    historical body ([0, L/a, L/b, L], empty ranges dropped, in order); it
    must be called exactly once as `for range in pieces(launches)` inside a
    function taking `launches: usize`; and `launches` must never be bound,
    borrowed mutably or assigned anywhere else."""
    words = tokens(strip_comments(source))
    text = ' '.join(words)
    need(text.count(PIECE_SIGNATURE) == 1 and words.count('pieces') >= 1 and
         sum(1 for k in range(1, len(words)) if words[k] == 'pieces' and words[k - 1] == 'fn') == 1,
         'the piece policy source must define pieces(launches: usize) exactly once')
    need(not any(words[k] == 'pieces' and words[k - 1] in ('as', 'use', '!', 'macro_rules') for k in range(1, len(words))),
         'pieces is aliased or generated in the piece policy source')
    body = text.split(PIECE_SIGNATURE, 1)[1]
    m = re.match(re.escape(PIECE_BODY).replace(r'\{a\}', r'(\d+)').replace(r'\{b\}', r'(\d+)'), body.strip())
    need(m is not None, 'pieces has an unsupported body; only the recognized split is accepted')
    need(text.count(PIECE_USE) == 1, 'the capture loop must call pieces(launches) exactly once')
    before = text.split(PIECE_USE, 1)[0]
    caller = before[before.rfind(' fn '):] if ' fn ' in before else ''
    need(' launches : usize ' in caller.split('{', 1)[0] + ' ',
         'the capture loop is not inside a function taking launches: usize')
    need(not mutates_launches(words), 'launches is rebound or mutated in the piece policy source')
    return [int(m[1]), int(m[2])]


def authenticate_piece_policy(repo, base, head, declaration):
    """The graph piece split policy, read from both revisions' source with
    git show: each program of L launches is captured as pieces at
    [0, L / a, L / b, L], empty pieces dropped. The declared divisors must
    equal both revisions' recognized implementation."""
    from common import git
    need(isinstance(declaration, dict) and set(declaration) == {'source', 'divisors'} and
         declaration['source'] in PIECE_SOURCES and isinstance(declaration['divisors'], list) and
         len(declaration['divisors']) == 2 and all(type(d) is int and d > 0 for d in declaration['divisors']),
         f'graph_piece_policy needs a source of {PIECE_SOURCES} and two positive divisors')
    for revision in (base, head):
        found = piece_divisors(git(repo, 'show', f'{revision}:{piece_source(repo, revision)}').decode())
        need(found == declaration['divisors'], f'{revision}: graph piece bounds differ from the declared policy')
    return dict(declaration)


def piece_source(repo, revision):
    """The one graph source path of PIECE_SOURCES that `revision` has."""
    from common import git
    present = [path for path in PIECE_SOURCES
               if git(repo, 'ls-tree', '--name-only', revision, '--', path).decode().strip() == path]
    need(len(present) == 1, f'{revision}: expected exactly one graph source among {PIECE_SOURCES}')
    return present[0]


def piece_sizes(launches, divisors):
    bounds = [0, launches // divisors[0], launches // divisors[1], launches]
    return [b - a for a, b in zip(bounds, bounds[1:]) if b > a]


def capture_runs(notes):
    """Consecutive graph captures on one stream with nothing between them:
    lists of (capture note, captured note, nodes)."""
    runs, begin, last = [], {}, None
    for i, line in enumerate(notes):
        if line.startswith('capture '):
            stream = line.split()[1]
            if not (runs and last == i - 1 and runs[-1][0] == stream):
                runs.append((stream, []))
            begin[stream] = i
        elif (m := CAPTURED.fullmatch(line)) and m[1] in begin:
            if runs and runs[-1][0] == m[1]:
                runs[-1][1].append((begin.pop(m[1]), i, int(m[3])))
            last = i
    return [blocks for _, blocks in runs]


def programs(notes, divisors):
    """Split each base capture run into programs whose pieces follow the
    policy. Only a run with exactly one such split yields programs; any
    other run keeps every boundary under the strict order check."""
    out = []
    for blocks in capture_runs(notes):
        sizes = [n for _, _, n in blocks]
        found = []

        def walk(i, acc):
            if len(found) > 1:
                return
            if i == len(sizes):
                found.append(acc)
                return
            for k in (1, 2, 3):
                if i + k <= len(sizes) and piece_sizes(sum(sizes[i:i + k]), divisors) == sizes[i:i + k]:
                    walk(i + k, acc + [blocks[i:i + k]])
        walk(0, [])
        if len(found) == 1:
            out.extend(program for program in found[0] if len(program) > 1)
    return out


def launches_only(notes, program, stream):
    """A program's capture holds only launches on its capture stream, apart
    from its own inner capture markers."""
    inner = {i for begin, end, _ in program for i in (begin, end)}
    for t in range(program[0][0] + 1, program[-1][1]):
        if t in inner:
            continue
        if not (notes[t].startswith('launch ') and stream_of(notes[t]) == int(stream)):
            return t
    return None


def whole_replays(notes, graphs):
    """Every replay of a program's graphs runs all of them, in order, on one
    stream, with no other operation on that stream between its pieces.
    Returns the first offending note index, or None."""
    position = {g: k for k, g in enumerate(graphs)}
    pending, started = {}, {}
    for t, line in enumerate(notes):
        on = stream_of(line)
        if line.startswith('replay '):
            stream, graph = line.split()[1:3]
            if graph in position:
                expect = pending.get(stream, 0)
                if position[graph] != expect:
                    return t
                if not expect:
                    started[stream] = t
                pending[stream] = expect + 1 if expect + 1 < len(graphs) else 0
                if not pending[stream]:
                    del pending[stream]
                continue
        if on is not None and str(on) in pending:
            return t
        if line.startswith('release graph ') and line.split()[2] in position and pending:
            return t
    return min(started[stream] for stream in pending) if pending else None


def regrouped_pieces(notes_a, notes_b, policy, divisors, correspond, rank, counts):
    """Graph pieces under the authenticated split policy (`graph_pieces`).

    For every base program with several pieces, its markers map to the head
    through the compared history. The head's piece sizes must be exactly the
    policy applied to the head program's launch total, its inner boundaries
    adjacent capture markers on the same stream. A program whose sizes
    change is relaxed only if, on both sides, its captures hold nothing but
    launches on the capture stream and every replay of its graphs runs the
    whole program in order with nothing else on that stream between pieces;
    otherwise the fixture is refused. Its inner markers then leave the order
    check, their place recomputed instead; the outer markers stay ordered, so
    the program holds the same launches. `count` is the exact number of
    programs whose sizes change."""
    rules = policy.get('graph_pieces', [])
    need(isinstance(rules, list) and len(rules) <= 1 and all(isinstance(r, dict) and set(r) == {'count'} and
         type(r['count']) is int and r['count'] > 0 for r in rules), 'graph_pieces needs one exact positive count')
    if not rules:
        return
    need(divisors is not None, 'graph_pieces requires the spec-level graph_piece_policy')
    changed = 0
    for program in programs(notes_a, divisors):
        markers = [correspond.get(i) for begin, end, _ in program for i in (begin, end)]
        require(None not in markers, 'a graph piece marker has no compared head counterpart')
        sizes = []
        for begin, end in zip(markers[0::2], markers[1::2]):
            m = CAPTURED.fullmatch(notes_b[end])
            require(m is not None and notes_b[begin] == f'capture {m[1]}', 'head graph piece markers differ')
            sizes.append(int(m[3]))
        for end, begin in zip(markers[1:-1:2], markers[2::2]):
            require(begin == end + 1, f'head note {end + 2}: an operation sits between two pieces of one graph program')
        require(piece_sizes(sum(sizes), divisors) == sizes,
                f'head note {markers[0] + 1}: graph pieces {sizes} do not follow the split policy for {sum(sizes)} launches')
        if sizes == [n for _, _, n in program]:
            continue
        head_program = [(b, e, n) for b, e, n in zip(markers[0::2], markers[1::2], sizes)]
        for side, notes, prog in (('base', notes_a, program), ('head', notes_b, head_program)):
            stream = CAPTURED.fullmatch(notes[prog[0][1]])[1]
            t = launches_only(notes, prog, stream)
            require(t is None, f'{side} note {t + 1 if t is not None else 0}: a re-split graph program captures a non-launch operation')
            graphs = [CAPTURED.fullmatch(notes[e])[2] for _, e, _ in prog]
            t = whole_replays(notes, graphs)
            require(t is None, f'{side} note {t + 1 if t is not None else 0}: a re-split graph program is not replayed whole and in order')
        changed += 1
        for t in markers[1:-1]:
            rank.pop(t, None)
    require(changed == rules[0]['count'], f'{changed} graph programs re-split, expected {rules[0]["count"]}')
    counts['graph_pieces'] += changed


def kinds(lines):
    return dict(collections.Counter(classify(line)[0] for line in lines))
