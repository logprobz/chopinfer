"""Offline comparison of model-owned oracle attachments."""
import math
import re
from pathlib import Path
import shlex
import struct

from tools.model_sdk import diagnostic as wire
from tools.qual import model_modules


def in_tree_root(model):
    """The directory of MODEL in this tree, `models/<model>`."""
    return Path(__file__).resolve().parents[2] / "models" / model


def model_adapter(model):
    """Load the diagnostic contract `diagnostic.py` (model_modules) of MODEL
    from its model root: CHOPINFER_QUAL_MODEL_ROOT for a model outside this
    tree, else the owning in-tree model's `models/<model>`."""
    if not isinstance(model, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", model):
        raise ValueError("invalid diagnostic model")
    root = model_modules.model_root()
    if root is None:
        root = in_tree_root(model)
        if not root.is_dir():
            root = None
    loaded = model_modules.load("diagnostic", root) if root is not None else None
    if loaded is None:
        raise ValueError("no model-owned diagnostic adapter for " + model)
    return loaded[0]


def row_key(row):
    # Physical indices can differ; every other recorded identity field must match.
    return wire.canonical({k:v for k,v in row.items() if k != 'row'})


def row_map(requested, observed):
    keys = [row_key(row) for row in observed]
    if len(set(keys)) != len(keys):
        raise ValueError('ambiguous captured row correspondence')
    lookup = {key:i for i,key in enumerate(keys)}
    mapped = []
    for row in requested:
        if row_key(row) not in lookup:
            raise wire.Incompatible('input token/position/request/state row differs')
        mapped.append(lookup[row_key(row)])
    if len(set(mapped)) != len(mapped):
        raise ValueError('row mapping is not injective')
    return mapped


def checkpoint(attachment):
    identity = attachment['identity']
    cp = identity.get('checkpoint')
    if not isinstance(cp, dict) or not cp.get('source_lock_sha256'):
        raise ValueError('missing authenticated checkpoint identity; recapture with the verified producer')
    digest = cp['source_lock_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid checkpoint content digest')
    prepared = identity.get('prepared_weights')
    if not isinstance(prepared, dict) or not prepared.get('kind'):
        raise ValueError('missing prepared-weight observation (or explicit absence)')
    if prepared['kind'] != 'none':
        if prepared.get('source_lock_sha256') != digest:
            raise wire.Incompatible('prepared weights are not linked to this checkpoint')
        if prepared['kind'] == 'planned':
            # Weights a direct load planned from this checkpoint: each rank
            # names the plan identity the host serves for it, which covers
            # headers and plan only, and the content digest of the bytes the
            # run read from it.
            images = prepared.get('images')
            if not prepared.get('checkpoint_identity') or not images \
                    or not all(image.get('direct_identity') for image in images):
                raise ValueError('planned weights lack their checkpoint or plan identities')
            if not all(_hex64((image.get('read') or {}).get('content_sha256')) for image in images):
                raise ValueError('planned weights lack the content digest of the bytes read')
        elif not prepared.get('images') or not prepared.get('verified_source_ranges'):
            raise ValueError('prepared images lack verified source/layout binding')
    producer = attachment['description'].get('producer', {})
    if not all(producer.get(k) for k in ('implementation', 'implementation_sha256', 'source_sha256', 'execution')):
        raise ValueError('missing observed producer implementation/source/execution')
    return cp


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def selected_tensor(root, tensor, mapping, observed_rows, wanted_shape, attachment):
    if tensor['shape'][0] != len(observed_rows) or tensor['shape'][1:] != wanted_shape[1:]:
        raise wire.Incompatible('full tensor shape differs; no declared model layout transform')
    if wanted_shape[0] != len(mapping):
        raise wire.Incompatible('requested tensor rows differ from declared correspondence')
    data = wire.raw_tensor(root, tensor, attachment["artifacts"][tensor["file"]])
    width = math.prod(tensor['shape'][1:])
    row_bytes = width * wire.WIDTH[tensor['dtype']]
    selected = b''.join(data[index*row_bytes:(index+1)*row_bytes] for index in mapping)
    return selected, wire.values(selected, tensor['dtype']), width


def nonfinite(raw, values, dtype):
    bad = [i for i,v in enumerate(values) if not math.isfinite(v)]
    if not bad:
        return None
    i = bad[0]
    width = wire.WIDTH[dtype]
    return dict(count=len(bad), first_index=i, dtype=dtype,
                bits=raw[i*width:(i+1)*width][::-1].hex(), value=None,
                value_tag='nan' if math.isnan(values[i]) else 'positive-infinity' if values[i] > 0 else 'negative-infinity')


def compare(reference, candidate, requested=None, floor_reference=None, *, reference_guard=None):
    report = dict(format=wire.FORMAT, mode='oracle', status='insufficient', evidence='scoped-oracle',
                  scope=requested, first=None, chained=[], op_isolated=[], missing=[], coverage={'gated':0,'complete':False})
    try:
        reference, candidate = wire.external(reference), wire.external(candidate)
        report['reproduction']=['python3','tools/dev/diagnostics.py','compare','--mode','oracle',
                                '--reference',str(reference),'--candidate',str(candidate)]
        if floor_reference is not None:
            report['reproduction'] += ['--floor-reference', str(floor_reference)]
        report['command'] = shlex.join(report['reproduction'])
        if reference_guard:
            reference_guard(reference)
        a, b = wire.load(reference), wire.load(candidate)
        sa, sb = a['description']['scope'], b['description']['scope']
        scope = requested or sb
        report['scope'] = scope
        required = wire.validate_scope(scope)
        report.update(scope=scope, limitations=list(dict.fromkeys(a['description'].get('limitations',[]) + b['description'].get('limitations',[]))))
        if checkpoint(a) != checkpoint(b):
            raise wire.Incompatible('checkpoint identities differ; pinned expectations cannot transfer to other weights')
        if not sa['model'] == sb['model'] == scope['model']:
            raise wire.Incompatible('model identity differs')
        model = scope['model']
        adapter = model_adapter(model)
        adapter.validate(a, b, scope)
        adapter.validate_input_payload(reference, a)
        adapter.validate_input_payload(candidate, b)
        for capture in (a,b):
            parent=capture['description'].get('parent')
            if parent:
                from tools.dev.diagnostics import load as load_parent
                root=wire.external(parent['root'])
                if wire.identity(root/'diagnostic.json')['sha256']!=parent['sha256']:
                    raise ValueError('parent capture seal changed')
                mode=parent.get('mode','fixed-schedule')
                if mode not in ('fixed-schedule','journal'): raise ValueError('unsupported parent kind')
                loaded=load_parent(root,mode)
                if mode=='fixed-schedule' and parent['step']>=loaded['description']['expected_steps']:
                    raise ValueError('parent step is outside declared schedule')
                if mode=='journal' and not 0<=parent['event']<loaded['events']:
                    raise ValueError('parent event is outside recorded journal')
        for original in (sa,sb,scope):
            if {k:v for k,v in original['inputs'].items() if k not in ('rows','token_ids')} != {
                    k:v for k,v in scope['inputs'].items() if k not in ('rows','token_ids')}:
                raise wire.Incompatible('input kind/history/initial state differs')
        ma = row_map(scope['inputs']['rows'], sa['inputs']['rows'])
        mb = row_map(scope['inputs']['rows'], sb['inputs']['rows'])
        ta, tb = wire.named(a['tensors'],'reference probe'), wire.named(b['tensors'],'candidate probe')
        floor = None
        if floor_reference is not None:
            floor_reference = wire.external(floor_reference)
            if reference_guard:
                reference_guard(floor_reference)
            floor = wire.load(floor_reference)
            if checkpoint(floor) != checkpoint(a):
                raise wire.Incompatible('floor reference checkpoint differs')
            sf = floor['description']['scope']
            if {k:v for k,v in sf['inputs'].items() if k not in ('rows','token_ids')}!={k:v for k,v in scope['inputs'].items() if k not in ('rows','token_ids')}:
                raise wire.Incompatible('floor reference input kind/history differs')
            if sf['model'] != scope['model'] or sf['inputs']['state'] != scope['inputs']['state'] or sf.get('policy_sha256') != scope.get('policy_sha256'):
                raise wire.Incompatible('floor reference model/state/policy differs')
            mf = row_map(scope['inputs']['rows'],sf['inputs']['rows'])
            tf = wire.named(floor['tensors'],'floor probe')
        report['attachments']={'reference':{'root':str(reference),**wire.identity(reference/'diagnostic.json')},
                               'candidate':{'root':str(candidate),**wire.identity(candidate/'diagnostic.json')}}
        if floor is not None:
            report['attachments']['floor_reference']={'root':str(floor_reference),**wire.identity(floor_reference/'diagnostic.json')}
        report['reused'] = {'reference':wire.identity(reference/'diagnostic.json'),
                            'candidate':wire.identity(candidate/'diagnostic.json'), 'reference_rows':ma,'candidate_rows':mb}
        report['identity'] = {'reference':a['identity'], 'candidate':b['identity']}
        report['producer_commands']={arm:value['description']['reproduction'] for arm,value in [('reference',a),('candidate',b)]}
        orders=[]
        for original in (sa,sb):
            declared=list(wire.named(original['probes'],'predeclared probe'))
            selected=[name for name in required if name in declared]
            if selected!=[name for name in declared if name in required]:
                raise wire.Incompatible('requested probe order differs from predeclared execution order')
            orders.append({name:index for index,name in enumerate(declared)})
        outcomes = report['op_isolated' if scope['evidence'] == 'op-isolated' else 'chained']
        for ordinal,(name,expected) in enumerate(required.items()):
            if name not in ta or name not in tb or any(name not in order for order in orders):
                report['missing'].append(name)
                continue
            if scope['evidence'] == 'op-isolated':
                dependencies = adapter.projection_dependencies(name)
                if expected.get('dependencies') != dependencies:
                    raise ValueError('missing predeclared operation dependencies for ' + name)
                for dependency in dependencies:
                    if dependency in ta and dependency in tb:
                        shape = [len(ma), *ta[dependency]['shape'][1:]]
                        if reference_guard:
                            reference_guard(reference)
                        ra, va, _ = selected_tensor(reference,ta[dependency],ma,sa['inputs']['rows'],shape,a)
                        rb, vb, _ = selected_tensor(candidate,tb[dependency],mb,sb['inputs']['rows'],shape,b)
                        if nonfinite(ra,va,ta[dependency]['dtype']) or nonfinite(rb,vb,tb[dependency]['dtype']):
                            raise ValueError('nonfinite substituted input ' + dependency)
                        if [struct.pack('<d',v) for v in va] != [struct.pack('<d',v) for v in vb]:
                            raise wire.Incompatible('decoded operation input differs: ' + dependency)
                    else:
                        da = a['identity'].get('decoded_operands',{}).get(dependency)
                        db = b['identity'].get('decoded_operands',{}).get(dependency)
                        if not da or not db:
                            raise ValueError('missing decoded operand ' + dependency)
                        if da != db:
                            raise wire.Incompatible('decoded operation weight/scale differs: ' + dependency)
            if reference_guard:
                reference_guard(reference)
            raw_a, va, width = selected_tensor(reference,ta[name],ma,sa['inputs']['rows'],expected['shape'],a)
            raw_b, vb, _ = selected_tensor(candidate,tb[name],mb,sb['inputs']['rows'],expected['shape'],b)
            adapter.validate_dtypes(scope['contract'],name,ta[name]['dtype'],tb[name]['dtype'])
            if nonfinite(raw_a,va,ta[name]['dtype']):
                report['missing'].append(name + ': nonfinite reference')
                continue
            vf = None
            if floor is not None:
                if name not in tf:
                    report['missing'].append(name + ': floor reference')
                    continue
                if reference_guard:
                    reference_guard(floor_reference)
                rf, vf, _ = selected_tensor(floor_reference,tf[name],mf,sf['inputs']['rows'],expected['shape'],floor)
                adapter.validate_dtypes(scope['contract'],name,ta[name]['dtype'],tf[name]['dtype'])
                if nonfinite(rf,vf,tf[name]['dtype']):
                    report['missing'].append(name + ': nonfinite floor reference')
                    continue
            bounds = None
            if scope['contract'] == getattr(adapter,'NUMERIC',None):
                bound_name = name + '.sum_abs_products'
                if bound_name not in ta:
                    report['missing'].append(bound_name)
                    continue
                if reference_guard:
                    reference_guard(reference)
                raw_s, sums, _ = selected_tensor(reference,ta[bound_name],ma,sa['inputs']['rows'],expected['shape'],a)
                if ta[bound_name]['dtype'] != 'F64' or nonfinite(raw_s,sums,'F64') or any(v<0 for v in sums):
                    raise ValueError('invalid numeric bound operands: '+bound_name)
                bounds=adapter.numeric_bounds(va,sums)
                if any(not math.isfinite(bound) for bound in bounds):
                    raise ValueError('numeric bounds exceed finite range')
            bad = nonfinite(raw_b,vb,tb[name]['dtype'])
            if bad:
                result = dict(passed=False, metric='nonfinite-output', nonfinite=bad,
                              worst_index=bad['first_index'], bound=None)
            else:
                extra = {'bounds':bounds} if bounds is not None else {}
                result = adapter.evaluate(scope['contract'],name,va,vb,width,floor_reference=vf,**extra)
            boundary=adapter.boundary(name)
            if 'rank' in expected and boundary.get('rank')!=expected['rank']:
                raise wire.Incompatible('probe rank differs from model boundary')
            result.update(boundary)
            result['layout']=tb[name].get('layout','model-declared-row-major')
            result.update(name=name, ordinal=ordinal, reference_ordinal=orders[0][name], candidate_ordinal=orders[1][name], shape=expected['shape'],
                          reference_dtype=ta[name]['dtype'],candidate_dtype=tb[name]['dtype'], contract=scope['contract'])
            index = result.get('worst_index')
            if index is not None:
                row = index // width
                result.update(row=scope['inputs']['rows'][row], reference_row=ma[row], candidate_row=mb[row],
                              expected=va[index], observed=vb[index] if math.isfinite(vb[index]) else None)
            outcomes.append(result)
        report['coverage'].update(selected=len(required), compared=len(outcomes), gated=len(outcomes),
                                  missing=report['missing'], omitted=[n for n in tb if n not in required],
                                  complete=not report['missing'] and len(outcomes) == len(required))
        failures = [row for row in outcomes if not row['passed']]
        if report['coverage']['complete']:
            report.update(status='diverged' if failures else 'match', first=failures[0] if failures else None)
        else:
            report['observed_failures'] = failures
            report['reason'] = 'missing required evidence; first divergence cannot be established'
        report['capture_required'] = report['missing']
    except wire.Incompatible as error:
        report.update(status='incompatible', reason=str(error))
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        report.update(status='insufficient', reason=str(error), action='Obtain a fresh scoped producer capture with the named missing identity, input or probe; reuse other verified captures.')
    return report


def verify_report(value):
    """Recompute from sealed artifacts; a report's asserted result is not evidence."""
    arms=value['attachments']
    for arm in arms.values():
        actual=wire.identity(wire.external(arm['root'])/'diagnostic.json')
        if any(actual[k]!=arm[k] for k in ('bytes','sha256')):
            raise ValueError('attachment identity changed since comparison')
    checked=compare(arms['reference']['root'],arms['candidate']['root'],value['scope'],
                    arms.get('floor_reference',{}).get('root'))
    if checked['status']!=value['status'] or checked['coverage']!=value['coverage']:
        raise ValueError('stored comparison result or coverage differs from verified artifacts')
    return checked


def reuse(candidate,scope):
    """Preflight existing data without running a model or asserting numerical agreement."""
    result={'format':wire.FORMAT,'mode':'oracle','status':'insufficient','comparison_status':'not_run',
            'scope':scope,'missing':[],'reusable':[]}
    try:
        root=wire.external(candidate)
        value=wire.load(root)
        checkpoint(value)
        original=value['description']['scope']
        required=wire.validate_scope(scope)
        if original['model']!=scope['model'] or original.get('policy_sha256')!=scope.get('policy_sha256'):
            raise wire.Incompatible('requested model/policy differs')
        adapter = model_adapter(scope['model'])
        adapter.validate(value, value, scope)
        adapter.validate_input_payload(root, value)
        if scope.get('checkpoint') and scope['checkpoint']!=value['identity']['checkpoint']:
            raise wire.Incompatible('requested checkpoint differs')
        if {k:v for k,v in scope['inputs'].items() if k not in ('rows','token_ids')}!={k:v for k,v in original['inputs'].items() if k not in ('rows','token_ids')}:
            raise wire.Incompatible('requested input/state differs')
        mapping=row_map(scope['inputs']['rows'],original['inputs']['rows'])
        tensors=wire.named(value['tensors'],'observed probe')
        declared=wire.named(original['probes'],'predeclared probe')
        for name,probe in required.items():
            if name not in declared or name not in tensors:
                result['missing'].append(name);continue
            selected_tensor(root,tensors[name],mapping,original['inputs']['rows'],probe['shape'],value)
            if scope['evidence']=='op-isolated':
                dependencies=adapter.projection_dependencies(name)
                if probe.get('dependencies')!=dependencies: raise ValueError('missing operation dependency declaration')
                for dependency in dependencies:
                    if dependency not in tensors and dependency not in value['identity'].get('decoded_operands',{}):
                        result['missing'].append(dependency)
            result['reusable'].append(name)
        result.update(status='insufficient' if result['missing'] else 'reusable',
            source={'root':str(root),**wire.identity(root/'diagnostic.json')},rows=mapping,
            source_scope=original,producer_reproduction=value['description']['reproduction'],
            action='Capture only missing arms/probes and their dependencies. Reconstructing later state may still execute earlier layers or tokens.')
    except wire.Incompatible as error: result.update(status='incompatible',reason=str(error))
    except (ValueError,OSError,KeyError,TypeError,IndexError) as error: result.update(reason=str(error))
    return result
