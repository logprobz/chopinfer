"""Lab-only E5 source derivatives, cubin reports and counter-free decisions.

Generated cubins never enter the kernel catalog or serving package. Replacement
anchors are checked so a changed production source cannot silently change a probe.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess

HC_HEADER = Path('models/qwen38/bundle/cuda/gemm/qwen38_hc.cuh')

HC = ('control', 'no_weight_loads', 'no_bottleneck', 'no_mma', 'trace')
DELTA = ('control', 'stream_rows', 'no_state_loads', 'no_checkpoints', 'trace')


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('diagnostic source anchor changed: ' + old[:80])
    return source.replace(old, new)


MARK = r'''
namespace e5 {
template<bool Enabled>
__device__ __forceinline__ void mark(uint64_t* stamps, int stage, int stride) {
  if constexpr (Enabled) {
    if (threadIdx.x == 0) {
      uint64_t t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
      const uint64_t cta = (uint64_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      stamps[cta * stride + stage] = t;
    }
  }
}
__device__ __forceinline__ uint32_t opaque() {
  uint32_t x;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(x));
  return 0x3d803d80u + (x & 7u);
}
}
'''


def hc_source(root):
    source = (root / HC_HEADER).read_text()
    source = source.replace('#pragma once\n', '')
    source = replace(source, '#include "qwen38.cuh"', '#include "gemm/qwen38.cuh"')
    source = replace(source, 'template<int Tokens, bool Precomputed = false, bool Padded = false>\n__device__ void run_hc_up_mix(',
                     'template<int Tokens, int Variant, bool Trace, bool Precomputed = false, bool Padded = false>\n__device__ void run_hc_up_mix(')
    source = replace(source, 'const uint32_t* live_rows) {',
                     'const uint32_t* live_rows, uint64_t* stamps) {\n  e5::mark<Trace>(stamps, 0, 7);')
    source = replace(source, 'for (int t = 0; t < T::k_tiles; ++t) a[t] = load_weight_fragment(base + t * 512);',
        '''for (int t = 0; t < T::k_tiles; ++t) {
    if constexpr (Variant == 1) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) a[t].v[j] = e5::opaque();
    } else a[t] = load_weight_fragment(base + t * 512);
  }
  e5::mark<Trace>(stamps, 1, 7);''')
    source = replace(source, '  ptx::griddep_wait();',
                     '  ptx::griddep_wait();\n  e5::mark<Trace>(stamps, 2, 7);')
    source = replace(source, 'half[h] = row::to_bf(chop::sigmoid::times(v, v));',
        '''if constexpr (Variant == 2) half[h] = static_cast<uint16_t>(words[w] >> (16 * h));
          else half[h] = row::to_bf(chop::sigmoid::times(v, v));''')
    source = replace(source, '  const uint32_t b_row =',
                     '  e5::mark<Trace>(stamps, 3, 7);\n  const uint32_t b_row =')
    source = replace(source, '  ptx::D4 acc[Tokens / 8], block[Tokens / 8];',
                     '  uint32_t diagnostic_sink = 0;\n  ptx::D4 acc[Tokens / 8], block[Tokens / 8];')
    source = replace(source, '''      ptx::mma_bf16(block[pair * 2], a[t], first);
      ptx::mma_bf16(block[pair * 2 + 1], a[t], second);''', '''      if constexpr (Variant == 3) {
        // Observable checksum keeps both weight and activation loads after ptxas.
        diagnostic_sink ^= a[t].v[0] ^ a[t].v[1] ^ a[t].v[2] ^ a[t].v[3]
                         ^ first.v[0] ^ first.v[1] ^ second.v[0] ^ second.v[1];
      } else {
        ptx::mma_bf16(block[pair * 2], a[t], first);
        ptx::mma_bf16(block[pair * 2 + 1], a[t], second);
      }''')
    source = replace(source, '  // terms[stream]', '  if constexpr (Variant == 3) acc[0].v[0] = __uint_as_float(0x3f000000u | (diagnostic_sink & 0x007fffffu));\n  e5::mark<Trace>(stamps, 4, 7);\n  // terms[stream]')
    source = replace(source, '  for (uint32_t index = threadIdx.x;',
                     '  e5::mark<Trace>(stamps, 5, 7);\n  for (uint32_t index = threadIdx.x;')
    source = replace(source, '\n}\n\n} // namespace chop::gemm::qwen38',
                     '\n  e5::mark<Trace>(stamps, 6, 7);\n}\n\n} // namespace chop::gemm::qwen38')
    for number, name in enumerate(HC):
        source += f'''
extern "C" __global__ void __launch_bounds__(128, 1) e5_hc_{name}(
 const uint8_t* weight, const uint16_t* down, const uint16_t* normed,
 uint16_t* output, const uint32_t* live_rows, uint64_t* stamps) {{
 chop::gemm::qwen38::run_hc_up_mix<32,{0 if name == 'trace' else number},{str(name == 'trace').lower()}>(
 weight,down,normed,output,live_rows,stamps);
}}
'''
    return source


def delta_source(root):
    source = (root / 'crates/kernels-sm120/cuda/linear/delta.cuh').read_text().replace('#pragma once\n', '')
    source = replace(source, '#include "common.cuh"', '#include "linear/common.cuh"')
    source = replace(source, 'template<int KeyHeads, int ValueHeads, int Dim, int MaxRows>',
                     'template<int KeyHeads, int ValueHeads, int Dim, int MaxRows, int Variant, bool Trace>')
    source = replace(source, 'const uint32_t* live_sequences) {',
                     'const uint32_t* live_sequences, uint64_t* stamps) {\n  e5::mark<Trace>(stamps, 0, 21);')
    source = replace(source, '  begin();', '  begin();\n  e5::mark<Trace>(stamps, 1, 21);')
    source = replace(source, '    s[i] = *reinterpret_cast<const float4*>(source + i * Dim);', '''    if constexpr (Variant == 2) {
      const float x = __uint_as_float(e5::opaque());
      s[i] = make_float4(x, x, x, x);
    } else s[i] = *reinterpret_cast<const float4*>(source + i * Dim);''')
    source = replace(source, '  RecurrentRow in[Rows];',
                     '  e5::mark<Trace>(stamps, 2, 21);\n  RecurrentRow in[Variant == 1 ? 1 : Rows];')
    source = replace(source, '    if (r < rows)\n      load_recurrent_row',
                     '    if (Variant != 1 && r < rows)\n      load_recurrent_row')
    source = replace(source, '  const float neg_a =', '  e5::mark<Trace>(stamps, 3, 21);\n  const float neg_a =')
    source = replace(source, '''  #pragma unroll
  for (int r = 0; r < Rows; ++r) {
    if (r >= rows) break;
    const float out = recurrent_step<Dim>(s, in[r], neg_a, bias, lane);''', '''  #pragma unroll
  for (int r = 0; r < Rows; ++r) {
    if (r >= rows) break;
    if constexpr (Variant == 1)
      load_recurrent_row<KeyHeads, ValueHeads, Dim>(qkv, qkv_stride, a, b, gate_stride,
          uint64_t(first + r), key_head, head, v0, lane, in[0]);
    const float out = recurrent_step<Dim>(s, in[Variant == 1 ? 0 : r], neg_a, bias, lane);
    e5::mark<Trace>(stamps, 4 + 2 * r, 21);''')
    source = replace(source, '''      #pragma unroll
      for (int i = 0; i < kRowsPerWarp; ++i)
        *reinterpret_cast<float4*>(checkpoint + i * Dim) = s[i];''', '''      if constexpr (Variant == 3) {
        // The output observes all state updates even without checkpoint stores.
        uint32_t checksum = 0;
        #pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i)
          checksum ^= __float_as_uint(s[i].x) ^ __float_as_uint(s[i].y)
                    ^ __float_as_uint(s[i].z) ^ __float_as_uint(s[i].w);
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
          checksum ^= __shfl_xor_sync(0xffffffffu, checksum, offset);
        if (lane < kRowsPerWarp)
          output[(uint64_t(first + r) * ValueHeads + head) * Dim + v0 + lane] =
              to_bf(out + __uint_as_float(0x35800000u | (checksum & 0x007fffffu)));
      } else {
        #pragma unroll
        for (int i = 0; i < kRowsPerWarp; ++i)
          *reinterpret_cast<float4*>(checkpoint + i * Dim) = s[i];
      }
      e5::mark<Trace>(stamps, 5 + 2 * r, 21);''')
    source = replace(source, '\n}\n\n} // namespace chop::linear',
                     '\n  e5::mark<Trace>(stamps, 20, 21);\n}\n\n} // namespace chop::linear')
    for number, name in enumerate(DELTA):
        source += f'''
extern "C" __global__ void __launch_bounds__(128, 1) e5_delta_{name}(
 const uint16_t* qkv, uint32_t qkv_stride, const uint16_t* a, const uint16_t* b,
 uint32_t gate_stride, const float* a_log, const float* dt_bias, float* state,
 uint16_t* output, const int32_t* cu_rows, const int32_t* slot_ids,
 const int32_t* checkpoint_ids, const uint32_t* live_sequences, uint64_t* stamps) {{
 chop::linear::gated_delta_recurrent<4,12,128,8,{0 if name == 'trace' else number},{str(name == 'trace').lower()}>(
 qkv,qkv_stride,a,b,gate_stride,a_log,dt_bias,state,output,cu_rows,slot_ids,checkpoint_ids,live_sequences,stamps);
}}
'''
    return source


def resource_report(text):
    records = {}
    for symbol, body in re.findall(r'Function (\w+):\s*([^\n]+)', text):
        values = {key: int(value) for key, value in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', body)}
        if set(values) != {'REG', 'STACK', 'SHARED', 'LOCAL'}:
            raise ValueError('incomplete cubin resources: ' + symbol)
        records[symbol] = values
    required = {'e5_hc_' + x for x in HC} | {'e5_delta_' + x for x in DELTA}
    if set(records) != required:
        raise ValueError('incomplete diagnostic cubin symbols')
    return records


def build(root, base, env, call):
    # Hash the transitive project CUDA inputs conservatively, including helpers.
    inputs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for tree in ('crates/kernels-sm120/cuda', 'models/qwen38/bundle/cuda')
              for p in sorted((root / tree).rglob('*.cuh'))}
    # These lock defines are emitted by the contract's config::hc_defines().
    records = json.loads((root / HC_HEADER.parents[2] / 'kernels.lock').read_text())
    model = HC_HEADER.stem.removesuffix('_hc')
    hc_defines = dict(next(record['defines'] for record in records
                           if record['name'] == f'gemm_{model}_own_sm120'))
    source = '#include <stdint.h>\n#include <cuda.h>\n' + MARK + hc_source(root) + delta_source(root)
    identity = hashlib.sha256((source + json.dumps(inputs, sort_keys=True)
                               + json.dumps(hc_defines, sort_keys=True)).encode()).hexdigest()
    folder = base / 'target/e5-diagnostics' / identity[:16]
    folder.mkdir(parents=True, exist_ok=True)
    unit = folder / 'diagnostics.cu'
    unit.write_text(source)
    cubin = folder / 'diagnostics.cubin'
    command = ['nice', '-n', '19', 'nvcc', '-arch=sm_120f', '-cubin', '-std=c++20', '-O3',
               '-Xptxas=-warn-spills,-warn-lmem-usage', '-I', str(root / 'crates/kernels-sm120/cuda'),
               '-I', str(root / 'models/qwen38/bundle/cuda'),
               *[f'-D{name}={value}' for name, value in hc_defines.items()],
               str(unit), '-o', str(cubin)]
    call(command, env, folder / 'build.log')
    raw = subprocess.check_output(['cuobjdump', '--dump-resource-usage', str(cubin)], env=env, text=True)
    (folder / 'resources.txt').write_text(raw)
    sass = subprocess.check_output(['cuobjdump', '--dump-sass', str(cubin)], env=env, text=True)
    (folder / 'sass.txt').write_text(sass)
    instructions = {}
    for section in sass.split('Function : ')[1:]:
        symbol = section.splitlines()[0].strip()
        instructions[symbol] = {op: len(re.findall(r'\b' + op + r'(?:\.|\b)', section))
                                for op in ('HMMA', 'LDG', 'STG', 'MUFU')}
    checks = (
        instructions['e5_hc_control']['HMMA'] > 0 == instructions['e5_hc_no_mma']['HMMA'],
        instructions['e5_hc_no_mma']['LDG'] >= instructions['e5_hc_control']['LDG'],
        instructions['e5_hc_no_weight_loads']['LDG'] < instructions['e5_hc_control']['LDG'],
        instructions['e5_hc_no_bottleneck']['MUFU'] < instructions['e5_hc_control']['MUFU'],
        instructions['e5_delta_no_state_loads']['LDG'] < instructions['e5_delta_control']['LDG'],
        instructions['e5_delta_no_checkpoints']['STG'] < instructions['e5_delta_control']['STG'],
    )
    if not all(checks):
        raise ValueError('compiled ablation did not retain/remove the intended instruction families: ' + str(instructions))
    result = {'path': str(cubin), 'instructions': instructions, 'sha256': hashlib.sha256(cubin.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'inputs': inputs,
              'source_closure_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'sass_sha256': hashlib.sha256((folder / 'sass.txt').read_bytes()).hexdigest(),
              'resources': resource_report(raw), 'command': command,
              'nvcc_version': subprocess.check_output(['nvcc', '--version'], env=env, text=True),
              'diagnostic_only': True}
    evidence = base / 'e5-diagnostics' / identity[:16]
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ('diagnostics.cu', 'build.log', 'resources.txt', 'sass.txt'):
        (evidence / name).write_bytes((folder / name).read_bytes())
    result['evidence_path'] = str(evidence)
    (evidence / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify_resources(prepared):
    """Check the exact file and its CPU report before any driver access."""
    path = Path(prepared['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != prepared['sha256']:
        raise ValueError('diagnostic cubin changed')
    raw = subprocess.check_output(['cuobjdump', '--dump-resource-usage', str(path)],
                                  text=True, timeout=20)
    if resource_report(raw) != prepared['resources']:
        raise ValueError('diagnostic cubin CPU resource report changed')


def loaded_resources(symbol, resource, registers, shared, reserved):
    # CUDA exposes driver-reserved bytes separately from per-function static
    # shared memory. cuobjdump includes the reserved section in these cubins.
    same_shared = shared == resource['SHARED']
    excludes_reserved = reserved > 0 and shared + reserved == resource['SHARED']
    if registers != resource['REG'] or not (same_shared or excludes_reserved):
        raise ValueError(f'loaded cubin disagrees with CPU resource report: {symbol}: '
                         f'cubin={resource}, loaded_REG={registers}, '
                         f'loaded_SHARED={shared}, driver_reserved_SHARED={reserved}')
    return dict(resource, REG=registers, SHARED=shared,
                cubin_shared_bytes=resource['SHARED'], driver_reserved_shared_bytes=reserved,
                shared_accounting='identical' if same_shared else 'excludes driver-reserved bytes')


def occupancy(prepared):
    """CUDA's calculator on the assigned visible device, without counters or launches."""
    driver = C.CDLL('libcuda.so.1')
    def call(name, types, *args):
        function = getattr(driver, name)
        function.argtypes = types
        function.restype = C.c_int
        result = function(*args)
        if result:
            raise RuntimeError(f'{name} returned CUDA error {result}')
    I, P = C.c_int, C.c_void_p
    device, context, module = I(), P(), P()
    call('cuInit', [C.c_uint], 0)
    call('cuDeviceGet', [C.POINTER(I), I], C.byref(device), 0)
    call('cuDevicePrimaryCtxRetain', [C.POINTER(P), I], C.byref(context), device)
    try:
        call('cuCtxSetCurrent', [P], context)
        call('cuModuleLoad', [C.POINTER(P), C.c_char_p], C.byref(module), prepared['path'].encode())
        max_threads, sms, reserved = I(), I(), I()
        for value, attribute in ((max_threads, 39), (sms, 16), (reserved, 111)):
            call('cuDeviceGetAttribute', [C.POINTER(I), I, I], C.byref(value), attribute, device)
        records = {}
        for symbol, resource in prepared['resources'].items():
            function, blocks, registers, shared = P(), I(), I(), I()
            call('cuModuleGetFunction', [C.POINTER(P), P, C.c_char_p], C.byref(function), module, symbol.encode())
            for value, attribute in ((registers, 4), (shared, 1)):
                call('cuFuncGetAttribute', [C.POINTER(I), I, P], C.byref(value), attribute, function)
            loaded = loaded_resources(symbol, resource, registers.value, shared.value, reserved.value)
            dynamic = 32 * (320 * 2 + 16) + 4 * 16 * 32 * 4 if symbol.startswith('e5_hc_') else 0
            call('cuOccupancyMaxActiveBlocksPerMultiprocessor', [C.POINTER(I), P, I, C.c_size_t],
                 C.byref(blocks), function, 128, dynamic)
            records[symbol] = dict(loaded, dynamic_shared_bytes=dynamic,
                max_active_blocks_per_sm=blocks.value, resident_warp_limit=blocks.value * 4,
                theoretical_occupancy=blocks.value * 128 / max_threads.value)
        return {'sm_count': sms.value, 'max_threads_per_sm': max_threads.value,
                'kind': 'CUDA occupancy calculator, not measured occupancy', 'kernels': records}
    finally:
        if module.value:
            call('cuModuleUnload', [P], module)
        call('cuDevicePrimaryCtxRelease_v2', [I], device)


def decide(root, prepared, baseline):
    """Rank hypotheses from paired controls; ablations never qualify a kernel."""
    expected = {(str(n), mode) for n in (16, 32) for mode in ('mix', 'chain')}
    expected |= {(shape, 'delta') for shape in ('16x1', '32x1', '16x2')}
    groups = {}
    for path in (root / 'receipts/qwen38-e5-diagnostic').glob('*.json'):
        record = json.loads(path.read_text())
        family = record.get('family')
        key = (str(record.get('rows')), record.get('mode')) if family == 'hc' else (record.get('shape'), 'delta')
        variants = HC if family == 'hc' else DELTA
        variant = record.get('variant')
        elapsed = record.get('median_us_per_call', 0)
        if (family not in ('hc', 'delta') or key not in expected
                or variant not in (*variants, 'control_after') or not record.get('diagnostic_only')
                or record.get('measurement_label') != 'exclusive-device'
                or not isinstance(elapsed, (int, float)) or not 0 < elapsed < float('inf')):
            raise ValueError('invalid diagnostic receipt: ' + str(path))
        group = groups.setdefault(key, {})
        if variant in group:
            raise ValueError('duplicate diagnostic case: ' + str(path))
        if variant in ('control', 'control_after', 'trace') and record.get('matches_shipped_bits') is not True:
            raise ValueError('diagnostic control is not exact: ' + str(path))
        if variant == 'trace' and not record.get('trace', {}).get('cta_stamps_ns'):
            raise ValueError('missing globaltimer trace: ' + str(path))
        group[variant] = record
    if set(groups) != expected:
        raise ValueError('missing HC or delta diagnostic shape')
    occupancy_report = json.loads((root / 'occupancy.json').read_text())
    sms = occupancy_report.get('sm_count', 0)
    max_threads = occupancy_report.get('max_threads_per_sm', 0)
    occupancy_kernels = occupancy_report.get('kernels', {})
    if (sms <= 0 or max_threads <= 0 or set(occupancy_kernels) != set(prepared['resources'])
            or any(v.get('max_active_blocks_per_sm', 0) <= 0 for v in occupancy_kernels.values())):
        raise ValueError('missing or incomplete occupancy report')
    report = []
    hypotheses = {'no_weight_loads': 'HC weight staging or register lifetime',
                  'no_bottleneck': 'Compute the HC bottleneck once in its producer',
                  'no_mma': 'HC MMA tile geometry',
                  'stream_rows': 'Shorten delta operand lifetimes while keeping eight-row support',
                  'no_state_loads': 'Delta committed-state read traffic',
                  'no_checkpoints': 'Delta checkpoint-store scheduling; preserve every checkpoint'}
    for key, group in sorted(groups.items()):
        family = 'delta' if key[1] == 'delta' else 'hc'
        variants = DELTA if family == 'delta' else HC
        if set(group) != set(variants) | {'control_after'}:
            raise ValueError('incomplete diagnostic variants: ' + str(key))
        before = group['control']['median_us_per_call']
        after = group['control_after']['median_us_per_call']
        control = (before + after) / 2
        drift = abs(after - before) / min(before, after)
        shipped = baseline[key]
        clone_gap = abs(control - shipped) / shipped
        ranking = []
        for variant in variants[1:-1]:
            value = group[variant]
            gain = 1 - value['median_us_per_call'] / control
            ranking.append({'variant': variant, 'hypothesis': hypotheses[variant],
                            'gain_fraction': gain, 'saving_us_per_call': control - value['median_us_per_call'],
                            'matches_shipped_bits': value.get('matches_shipped_bits'),
                            'registers': prepared['resources'][f'e5_{family}_{variant}']['REG']})
        ranking.sort(key=lambda r: r['gain_fraction'], reverse=True)
        # Only the streamed-operand delta variant is a possible exact candidate.
        eligible = [r for r in ranking if r['variant'] != 'stream_rows' or r['matches_shipped_bits'] is True]
        threshold = max(0.05, 2 * drift)
        recommended = None
        if drift > 0.10 or clone_gap > 0.15:
            decision = 'Repeat the measurement: control drift or clone overhead is too large'
        elif eligible and eligible[0]['gain_fraction'] > threshold:
            decision = eligible[0]['hypothesis']
            recommended = eligible[0]
        else:
            decision = 'Keep the current kernel; no measured variant justifies a change'
        occupancy_control = occupancy_kernels[f'e5_{family}_control']
        grid = 160 if family == 'hc' else 8 * 12 * int(key[0].split('x')[0])
        grid_limit = min(grid, sms * occupancy_control['max_active_blocks_per_sm']) * 128 / (sms * max_threads)
        report.append({'shape': key[0], 'mode': key[1], 'decision': decision,
                       'recommended_variant': recommended,
                       'control_resources_and_occupancy': occupancy_control,
                       'single_grid_occupancy_ceiling': grid_limit,
                       'variant_occupancy': {v: occupancy_kernels[f'e5_{family}_{v}'] for v in variants},
                       'control_us': control, 'shipped_us': shipped, 'control_drift_fraction': drift,
                       'clone_gap_fraction': clone_gap, 'gain_threshold_fraction': threshold,
                       'ranked_sensitivities': ranking,
                       'trace_overhead_fraction': group['trace']['median_us_per_call'] / control - 1,
                       'trace_intervals': group['trace']['trace']['intervals']})
    target = []
    for case in report:
        count = 96 if (case['shape'], case['mode']) == ('32', 'chain') else 36 if case['shape'] == '16x2' else 0
        best = case['recommended_variant']
        if count and best:
            target.append({'hypothesis': best['hypothesis'], 'shape': case['shape'], 'mode': case['mode'],
                           'ablation_sensitivity_ms_per_target_step': count * best['saving_us_per_call'] / 1000,
                           'note': 'Prioritization only; not a predicted serving speedup'})
    target.sort(key=lambda x: x['ablation_sensitivity_ms_per_target_step'], reverse=True)
    return {'cases': report, 'next_experiment': target[0] if target else 'No change justified by stable controls',
            'diagnostic_receipts': 42, 'serving_change_qualified': False,
            'interpretation': 'Ablations change register allocation and dependencies; gains are sensitivities, not additive phase costs. Globaltimer observes thread 0 control-flow boundaries. Occupancy is theoretical.'}


if __name__ == '__main__':
    import argparse
    from datetime import datetime, timezone
    import os
    parser = argparse.ArgumentParser(description='Internal E5 occupancy child, requires the same explicit GPU assignment')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--go', action='store_true')
    parser.add_argument('--device', choices=('0', '1', '2', '3'), required=True)
    parser.add_argument('--end-time', required=True)
    args = parser.parse_args()
    end = datetime.fromisoformat(args.end_time.replace('Z', '+00:00'))
    if not args.go or end.tzinfo is None or end.utcoffset().total_seconds() != 0 or end <= datetime.now(timezone.utc):
        parser.error('occupancy requires --go and an unexpired UTC end time')
    record = json.loads(Path(args.manifest).read_text())['diagnostics']
    verify_resources(record)
    uuid = subprocess.check_output(['nvidia-smi', '--id=' + args.device,
        '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid):
        raise ValueError('assigned physical GPU has no UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    print(json.dumps(occupancy(record), indent=2))
