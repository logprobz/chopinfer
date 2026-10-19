#!/usr/bin/env python3
"""Unprivileged CUDA resource queries. Run only inside the allocated pair job."""
import ctypes as C
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P, I = C.c_void_p, C.c_int


def report_function(request, attrs, blocks, max_threads_per_sm):
    """Compare like attributes; driver occupancy owns runtime residency."""
    dynamic = request['dynamic_shared_bytes']
    assert blocks > 0 and attrs['max_dynamic_shared_bytes'] >= dynamic, (attrs, request)
    assert attrs['registers'] == request['registers'], (attrs, request)
    assert attrs['local_bytes'] == request['local_bytes'] == request['stack_bytes'] == 0, (attrs, request)
    assert request['max_threads'] <= attrs['max_threads'], (attrs, request)
    assert attrs['static_shared_bytes'] >= 0 and request['shared_static_bytes'] >= 0, (attrs, request)
    assert 0 < blocks * request['max_threads'] <= max_threads_per_sm, (blocks, request)
    # A compiled shared-section extent need not equal the driver's static
    # allocation. An aligned extern-shared entry reported 1024 and 0 bytes,
    # respectively. Keep both; derive residency from the driver query.
    return dict(request, driver=attrs, active_ctas_per_sm=blocks,
                occupancy_source='cuOccupancyMaxActiveBlocksPerMultiprocessor',
                shared_memory_accounting=dict(
                    compiler_section_bytes=request['shared_static_bytes'],
                    driver_static_bytes=attrs['static_shared_bytes'],
                    requested_dynamic_bytes=dynamic,
                    driver_static_plus_requested_dynamic_bytes=attrs['static_shared_bytes'] + dynamic),
                theoretical_warp_occupancy_percent=100 * blocks * request['max_threads'] / max_threads_per_sm)


def collect(root=ROOT, driver=None):
    cuda = driver if driver is not None else C.CDLL('libcuda.so.1')
    context = {}
    def trace(**fields):
        print(json.dumps(dict(context, **fields)), file=sys.stderr, flush=True)
    def api(name, types):
        fn = getattr(cuda, name)
        fn.argtypes = types
        fn.restype = I
        def checked(*args):
            trace(call=name, phase='begin')
            code = fn(*args)
            trace(call=name, phase='end', code=code)
            if code: raise RuntimeError(f'{name} returned {code}; context={context}')
        return checked
    init = api('cuInit', [C.c_uint])
    count = api('cuDeviceGetCount', [C.POINTER(I)])
    get_device = api('cuDeviceGet', [C.POINTER(I), I])
    retain = api('cuDevicePrimaryCtxRetain', [C.POINTER(P), I])
    release = api('cuDevicePrimaryCtxRelease_v2', [I])
    set_current = api('cuCtxSetCurrent', [P])
    load = api('cuModuleLoad', [C.POINTER(P), C.c_char_p])
    unload = api('cuModuleUnload', [P])
    get_function = api('cuModuleGetFunction', [C.POINTER(P), P, C.c_char_p])
    set_attribute = api('cuFuncSetAttribute', [P, I, I])
    get_attribute = api('cuFuncGetAttribute', [C.POINTER(I), I, P])
    get_device_attribute = api('cuDeviceGetAttribute', [C.POINTER(I), I, I])
    occupancy = api('cuOccupancyMaxActiveBlocksPerMultiprocessor', [C.POINTER(I), P, I, C.c_size_t])
    requests = json.loads((root / 'resource-requests.json').read_text())
    trace(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device_order=os.environ.get('CUDA_DEVICE_ORDER'), root=str(root))
    init(0)
    n = I()
    count(C.byref(n))
    trace(visible_device_count=n.value)
    assert n.value == 2, f'expected two visible devices, got {n.value}'
    result = []
    for rank in range(n.value):
        context.clear()
        context['rank'] = rank
        device, ctx, modules = I(), P(), {}
        get_device(C.byref(device), rank)
        retain(C.byref(ctx), device)
        try:
            set_current(ctx)
            device_attrs = {}
            for name, attr in [('multiprocessors',16),('l2_bytes',38),('max_threads_per_sm',39)]:
                context['attribute'] = name
                value = I()
                get_device_attribute(C.byref(value), attr, device)
                device_attrs[name] = value.value
                trace(value=value.value)
            context.pop('attribute', None)
            functions = []
            for request in requests:
                module = request['module']
                context.update(module=module, symbol=request['symbol'])
                if module not in modules:
                    handle = P()
                    load(C.byref(handle), str(root / 'snapshot/cubins' / (module + '.cubin')).encode())
                    modules[module] = handle
                fn = P()
                get_function(C.byref(fn), modules[module], request['symbol'].encode())
                dynamic = request['dynamic_shared_bytes']
                trace(requested_dynamic_shared_bytes=dynamic)
                set_attribute(fn, 8, dynamic)
                attrs = {}
                for name, attr in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers',4),('max_dynamic_shared_bytes',8)]:
                    context['attribute'] = name
                    value = I()
                    get_attribute(C.byref(value), attr, fn)
                    attrs[name] = value.value
                    trace(value=value.value)
                context.pop('attribute', None)
                blocks = I()
                occupancy(C.byref(blocks), fn, request['max_threads'], dynamic)
                trace(active_ctas_per_sm=blocks.value, driver=attrs, compiled=request)
                functions.append(report_function(request, attrs, blocks.value, device_attrs['max_threads_per_sm']))
            result.append(dict(rank=rank, device=device_attrs, functions=functions))
        finally:
            failed = sys.exc_info()[0] is not None
            cleanup_errors = []
            for cleanup, argument in [(unload, module) for module in modules.values()] + [(set_current, None), (release, device)]:
                try:
                    cleanup(argument)
                except Exception as error:
                    cleanup_errors.append(error)
                    traceback.print_exc(file=sys.stderr)
            if cleanup_errors and not failed:
                raise cleanup_errors[0]
    return {'scope':'static function attributes and theoretical occupancy; no hardware counters', 'ranks':result}

if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
