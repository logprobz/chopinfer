"""Which template lines one chat-template render runs.

A Jinja template compiles to a Python module whose `debug_info` marks the
generated code line where each template line's code starts. Tracing the lines
the module's code runs during one render, and mapping each back, gives the
template lines that render covered. An unmarked code line (a branch's `pass`,
constant output, a loop's epilogue, a macro's `return`) belongs to the
nearest marked simple statement before it at its own indentation, or else to
the block it sits in. A compound statement before it at the same
indentation (an `if` chain, a loop) is a finished sibling whose branches may
not have run, so it is passed over. A line therefore counts as run only when
its own code, or code its block runs unconditionally, ran. The tracer is keyed
on a code filename (and optionally on the module namespace), so another
reference, such as a Python encoder, can be
traced the same way with its own line map; only the Jinja path exists here.

The template is compiled by transformers' chat-template compiler, the one
`apply_chat_template` uses and caches, so a render through
`apply_chat_template(..., chat_template=source)` runs this very module.
Nothing here imports transformers until that compiler is asked for.
"""

from contextlib import contextmanager
import re
import sys

# The first line of a compound statement in generated code.
COMPOUND = re.compile(r"(if|elif|else|for|while|try|except|finally|with|def|async def)\b")


class LineMap:
    """Code line to source line. `mapping` holds every code line that maps;
    `marks` the source lines that have code of their own."""

    def __init__(self, mapping, marks):
        self._mapping = dict(mapping)
        self._marks = set(marks)

    def source_line(self, code_line):
        return self._mapping.get(code_line)

    def source_lines(self):
        """Every source line with code."""
        return set(self._marks)


def structural_line_map(code, marks):
    """The LineMap of generated Python `code` whose marked lines (`marks`,
    code line to source line) start each source line's code."""
    lines = code.split("\n")

    def indent(text):
        return len(text) - len(text.lstrip(" "))

    def opens_block(text):
        return COMPOUND.match(text.strip()) is not None and text.rstrip().endswith(":")

    def resolve(number):
        if number in marks:
            return marks[number]
        depth = indent(lines[number - 1])
        for previous in range(number - 1, 0, -1):
            text = lines[previous - 1]
            if not text.strip():
                continue
            if indent(text) < depth:
                # The block this line sits in.
                return resolve(previous)
            if indent(text) == depth and previous in marks and not opens_block(text):
                return marks[previous]
        return None

    mapping = {}
    for number in range(1, len(lines) + 1):
        if lines[number - 1].strip():
            source = resolve(number)
            if source is not None:
                mapping[number] = source
    return LineMap(mapping, marks.values())


class LineTracer:
    """Records the source lines run by code compiled from `filename`, and,
    when `namespace` is given, only by functions whose globals are it."""

    def __init__(self, filename, line_map, namespace=None):
        self.filename = filename
        self.line_map = line_map
        self.namespace = namespace

    def _local(self, hit):
        def trace(frame, event, arg):
            if event == "line":
                line = self.line_map.source_line(frame.f_lineno)
                if line is not None:
                    hit.add(line)
            return trace
        return trace

    @contextmanager
    def tracing(self):
        """Yields the set of source lines run inside the block. Only the
        calling thread is traced; any tracer already set is restored."""
        hit = set()
        local = self._local(hit)

        def call(frame, event, arg):
            if frame.f_code.co_filename != self.filename:
                return None
            if self.namespace is not None and frame.f_globals is not self.namespace:
                return None
            return local

        previous = sys.gettrace()
        sys.settrace(call)
        try:
            yield hit
        finally:
            sys.settrace(previous)


def transformers_compiler():
    """transformers' cached chat-template compiler."""
    from transformers.utils.chat_template_utils import _compile_jinja_template
    return _compile_jinja_template


class TemplateCoverage:
    """One compiled chat template and the lines its renders run.

    `compile` turns the source into a `jinja2.Template`; it defaults to
    transformers' compiler, and a test passes a plain jinja2 one."""

    def __init__(self, source, compile=None):
        self.source = source
        self.template = (compile or transformers_compiler())(source)
        root = self.template.root_render_func
        # The same environment compiles the same code again, as source.
        code = self.template.environment.compile(source, raw=True)
        if f"debug_info = {self.template._debug_info!r}" not in code:
            raise ValueError("the template's generated code does not match its compiled module")
        # debug_info holds (template_line, code_line) marks.
        self.line_map = structural_line_map(code, {code: line for line, code in self.template.debug_info})
        self.tracer = LineTracer(root.__code__.co_filename, self.line_map, root.__globals__)

    @property
    def lines_with_code(self):
        return self.line_map.source_lines()

    def tracing(self):
        """A block whose renders of this template record their lines."""
        return self.tracer.tracing()

    def render(self, **context):
        """Renders the compiled template directly: (text, covered lines)."""
        with self.tracing() as hit:
            text = self.template.render(**context)
        return text, set(hit)

    def line_text(self, line):
        return self.source.split("\n")[line - 1]
