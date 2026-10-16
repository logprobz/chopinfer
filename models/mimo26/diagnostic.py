"""MiMo-owned layer cosine and matched-operand QKV contracts."""
import json
import math
from pathlib import Path
from tools.model_sdk import diagnostic as wire

COSINE = 'mimo26.layer-cosine.v1'
NUMERIC = 'gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1'
POLICY = wire.sha256((Path(__file__).parent/'diagnostic-contracts.json').read_bytes())
PINS = json.loads((Path(__file__).parent/'checkpoint-files.json').read_text())
SOURCE_LOCKS = {wire.sha256(wire.canonical(value)) for value in PINS.values()}


def validate(a,b,scope):
    for item in (a,b):
        value=item['description']['scope']
        if value.get('policy_sha256') != POLICY or value.get('topology') != {'ranks':4,'fold':1}:
            raise wire.Incompatible('MiMo policy/topology is unsupported')
        if item['identity']['checkpoint']['source_lock_sha256'] not in SOURCE_LOCKS:
            raise wire.Incompatible('MiMo checkpoint does not match a registered source lock')
    if scope.get('policy_sha256') != POLICY or scope['contract'] not in (COSINE,NUMERIC) or scope.get('topology')!={'ranks':4,'fold':1}:
        raise wire.Incompatible('MiMo numerical contract is not applicable')
    if scope['inputs'].get('kind') != 'hidden-bf16':
        raise wire.Incompatible('MiMo layer evidence requires exact hidden inputs')


def projection_dependencies(name):
    stem,rank=name.rsplit('.r',1)
    if not stem.endswith('.qkv') or rank not in ('0','1','2','3'):
        raise ValueError('no isolated operation descriptor for '+name)
    return [stem.removesuffix('.qkv')+'.input_norm.r'+rank,stem+'.weight.r'+rank]


def validate_dtypes(contract,name,reference,candidate):
    if reference != 'F64' or candidate not in ('BF16','F64'):
        raise wire.Incompatible('MiMo comparison requires F64 oracle and BF16 or F64 candidate')
    if contract == NUMERIC and candidate != 'BF16':
        raise wire.Incompatible('QKV numeric contract requires BF16 output rounding')


def evaluate(contract,name,reference,candidate,width,*,floor_reference=None,bounds=None):
    if contract == COSINE:
        scores=[]
        for first in range(0,len(reference),width):
            a,b=candidate[first:first+width],reference[first:first+width]
            scale=max(map(abs,a+b))
            if scale==0: scores.append(1.0); continue
            a,b=[v/scale for v in a],[v/scale for v in b]
            aa,bb=math.fsum(v*v for v in a),math.fsum(v*v for v in b)
            scores.append(math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else 0.0)
        row=min(range(len(scores)),key=scores.__getitem__)
        return dict(passed=scores[row]>=.9999,metric='per-token-cosine',metric_value=scores[row],bound=.9999,worst_index=row*width)
    if contract != NUMERIC or bounds is None or len(bounds)!=len(reference) or any(not math.isfinite(b) or b<0 for b in bounds):
        raise ValueError('missing existing per-element QKV bounds')
    errors=[abs(x-y) for x,y in zip(reference,candidate)]
    index=max(range(len(errors)),key=lambda i:errors[i]-bounds[i])
    return dict(passed=all(e<=b for e,b in zip(errors,bounds)),metric='absolute-error',metric_value=errors[index],
                bound=bounds[index],worst_index=index)


def numeric_bounds(values,sums):
    return [(2.0**-134 if abs(v)<2.0**-126 else 2.0**(math.floor(math.log2(abs(v)))-8)) + 2.0**-22*4096*s for v,s in zip(values,sums)]


def validate_input_payload(root,capture):
    data=capture['description']['scope']['inputs']
    tensor=wire.named(capture['tensors'],'input probe').get('input.hidden')
    if not tensor or tensor['dtype']!='BF16' or tensor['shape']!=[len(data['rows']),4096]:
        raise ValueError('missing exact BF16 hidden-input payload')
    if wire.sha256(wire.raw_tensor(root,tensor,capture['artifacts'][tensor['file']]))!=data['hidden_sha256']:
        raise wire.Incompatible('hidden-input payload differs from declared input identity')


def boundary(name):
    prefix,operation=name.split('.',1)
    operation,rank=operation.rsplit('.r',1)
    if not prefix.startswith('layer') or not prefix[5:].isdigit() or rank not in ('0','1','2','3'):
        raise ValueError('unsupported MiMo boundary '+name)
    return {'layer':int(prefix[5:]),'operation':operation,'rank':int(rank)}
