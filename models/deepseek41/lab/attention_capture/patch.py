"""Add a diagnostic attention observer to a separate exact source export."""
from pathlib import Path
import argparse
import hashlib
import json


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError('capture patch anchor is absent or ambiguous: ' + before[:80])
    return text.replace(before, after)


def apply(source):
    path = source / 'models/deepseek41/bundle/src/instance/execute.rs'
    old = path.read_text()
    text = replace(old, 'pub(crate) mod skip;\n', 'pub(crate) mod skip;\n#[cfg(feature = "diagnostic-taps")]\nmod attention_capture;\n')
    text = replace(text, '        let (phase, arena, layout, mut graph) = self.phase_of(&inputs)?;', '''        let (phase, arena, layout, mut graph) = self.phase_of(&inputs)?;
        #[cfg(feature = "diagnostic-taps")]
        if attention_capture::enabled() {
            graph = None;
        }''')
    text = replace(text, '        let mut collectives = std::mem::take(&mut run.collectives);', '''        let mut collectives = std::mem::take(&mut run.collectives);
        #[cfg(feature = "diagnostic-taps")]
        if attention_capture::selected(run) {
            return attention_capture::launch(self, run, &streams, &programs, &collectives);
        }''')
    hook = Path(__file__).with_name('hook.rs').read_bytes()
    path.write_text(text)
    (path.parent / 'execute/attention_capture.rs').write_bytes(hook)
    sm120_path = source / 'models/deepseek41/bundle/src/sm120/mod.rs'
    sm120_old = sm120_path.read_text()
    sm120_new = sm120_old + '\n#[cfg(feature = "diagnostic-taps")]\nmod attention_capture;\n'
    sm120_path.write_text(sm120_new)
    helper = Path(__file__).with_name('sm120.rs').read_bytes()
    (sm120_path.parent / 'attention_capture.rs').write_bytes(helper)
    result = dict(changes=[dict(file=str(p.relative_to(source)), before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                               after_sha256=hashlib.sha256(after.encode()).hexdigest())
                           for p,before,after in [(path,old,text),(sm120_path,sm120_old,sm120_new)]],
                  observer_sha256=hashlib.sha256(hook).hexdigest(), sm120_sha256=hashlib.sha256(helper).hexdigest())
    (source / 'attention-capture-patch.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    print(json.dumps(apply(parser.parse_args().source),indent=2))
