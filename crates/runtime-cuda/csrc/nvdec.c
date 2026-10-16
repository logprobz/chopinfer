#define _GNU_SOURCE
#include "nvdec.h"

#include <cuda.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "dynlink_nvcuvid.h"

struct runtime_cuda_nvdec {
    void* handle;
    tcuvidGetDecoderCaps* get_decoder_caps;
    tcuvidCreateDecoder* create_decoder;
    tcuvidDestroyDecoder* destroy_decoder;
    tcuvidDecodePicture* decode_picture;
    tcuvidMapVideoFrame64* map_frame;
    tcuvidUnmapVideoFrame64* unmap_frame;
    tcuvidCreateVideoParser* create_parser;
    tcuvidParseVideoData* parse_data;
    tcuvidDestroyVideoParser* destroy_parser;
    tcuvidCtxLockCreate* create_lock;
    tcuvidCtxLockDestroy* destroy_lock;
};

/* The decoder's own CUDA context, never one the engine serves on. Work the
 * engine synchronizes on its contexts, cuCtxSynchronize included, never waits
 * on NVDEC, and NVDEC's waits never wait on the engine. */
struct runtime_cuda_nvdec_session {
    runtime_cuda_nvdec* library;
    CUcontext context;
    CUvideoctxlock lock;
    CUstream stream;
};

/* Decode surfaces beyond the stream's declared minimum, for display delay. */
#define EXTRA_DECODE_SURFACES 4

typedef struct {
    runtime_cuda_nvdec* library;
    CUvideoctxlock lock;
    CUstream stream;
    CUvideodecoder decoder;
    uint32_t coded_width;
    uint32_t coded_height;
    uint32_t width;
    uint32_t height;
    uint32_t max_width;
    uint32_t max_height;
    uint8_t* host;
    void* user;
    runtime_cuda_nvdec_select select;
    runtime_cuda_nvdec_frame frame;
    int status;
    unsigned int driver_status;
} decode_state;

static int fail(decode_state* state, int status, CUresult driver) {
    if (state->status == RUNTIME_CUDA_NVDEC_OK) {
        state->status = status;
        state->driver_status = (unsigned int)driver;
    }
    return 0;
}

static int resolve(void* handle, const char* name, void** slot) {
    *slot = dlsym(handle, name);
    return *slot != NULL;
}

static void runtime_cuda_nvdec_close(runtime_cuda_nvdec* library) {
    if (library == NULL) return;
    dlclose(library->handle);
    free(library);
}

static int runtime_cuda_nvdec_open(runtime_cuda_nvdec** library) {
    if (library == NULL) return RUNTIME_CUDA_NVDEC_INVALID;
    *library = NULL;
    void* handle = dlopen("libnvcuvid.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) return RUNTIME_CUDA_NVDEC_LIBRARY_ABSENT;
    runtime_cuda_nvdec* loaded = calloc(1, sizeof *loaded);
    if (loaded == NULL) {
        dlclose(handle);
        return RUNTIME_CUDA_NVDEC_INVALID;
    }
    loaded->handle = handle;
    int complete =
        resolve(handle, "cuvidGetDecoderCaps", (void**)&loaded->get_decoder_caps) &&
        resolve(handle, "cuvidCreateDecoder", (void**)&loaded->create_decoder) &&
        resolve(handle, "cuvidDestroyDecoder", (void**)&loaded->destroy_decoder) &&
        resolve(handle, "cuvidDecodePicture", (void**)&loaded->decode_picture) &&
        resolve(handle, "cuvidMapVideoFrame64", (void**)&loaded->map_frame) &&
        resolve(handle, "cuvidUnmapVideoFrame64", (void**)&loaded->unmap_frame) &&
        resolve(handle, "cuvidCreateVideoParser", (void**)&loaded->create_parser) &&
        resolve(handle, "cuvidParseVideoData", (void**)&loaded->parse_data) &&
        resolve(handle, "cuvidDestroyVideoParser", (void**)&loaded->destroy_parser) &&
        resolve(handle, "cuvidCtxLockCreate", (void**)&loaded->create_lock) &&
        resolve(handle, "cuvidCtxLockDestroy", (void**)&loaded->destroy_lock);
    if (!complete) {
        runtime_cuda_nvdec_close(loaded);
        return RUNTIME_CUDA_NVDEC_SYMBOL_MISSING;
    }
    *library = loaded;
    return RUNTIME_CUDA_NVDEC_OK;
}

void runtime_cuda_nvdec_session_close(runtime_cuda_nvdec_session* session) {
    if (session == NULL) return;
    if (session->context != NULL) {
        cuCtxSetCurrent(session->context);
        if (session->stream != NULL) cuStreamDestroy(session->stream);
        if (session->lock != NULL) session->library->destroy_lock(session->lock);
        cuCtxSetCurrent(NULL);
        cuCtxDestroy(session->context);
    }
    runtime_cuda_nvdec_close(session->library);
    free(session);
}

int runtime_cuda_nvdec_session_open(int device_ordinal, runtime_cuda_nvdec_session** session,
                                    unsigned int* driver_status) {
    if (session == NULL || driver_status == NULL) return RUNTIME_CUDA_NVDEC_INVALID;
    *session = NULL;
    *driver_status = CUDA_SUCCESS;
    runtime_cuda_nvdec_session* opened = calloc(1, sizeof *opened);
    if (opened == NULL) return RUNTIME_CUDA_NVDEC_INVALID;
    int status = runtime_cuda_nvdec_open(&opened->library);
    if (status != RUNTIME_CUDA_NVDEC_OK) {
        free(opened);
        return status;
    }
    CUdevice device = 0;
    /* Blocking sync: a wait for NVDEC sleeps rather than spinning a core. */
    CUresult result = cuInit(0);
    if (result == CUDA_SUCCESS) result = cuDeviceGet(&device, device_ordinal);
    if (result == CUDA_SUCCESS) {
        result = cuCtxCreate(&opened->context, NULL, CU_CTX_SCHED_BLOCKING_SYNC, device);
        if (result != CUDA_SUCCESS) opened->context = NULL;
    }
    if (result == CUDA_SUCCESS) {
        result = opened->library->create_lock(&opened->lock, opened->context);
        if (result != CUDA_SUCCESS) opened->lock = NULL;
    }
    if (result == CUDA_SUCCESS) {
        result = cuStreamCreate(&opened->stream, CU_STREAM_NON_BLOCKING);
        if (result != CUDA_SUCCESS) opened->stream = NULL;
    }
    if (result != CUDA_SUCCESS) {
        *driver_status = (unsigned int)result;
        runtime_cuda_nvdec_session_close(opened);
        return RUNTIME_CUDA_NVDEC_DRIVER;
    }
    *session = opened;
    return RUNTIME_CUDA_NVDEC_OK;
}

int runtime_cuda_nvdec_h264_capacity(runtime_cuda_nvdec_session* session, uint32_t* max_width,
                                     uint32_t* max_height, unsigned int* driver_status) {
    if (session == NULL || max_width == NULL || max_height == NULL || driver_status == NULL) {
        return RUNTIME_CUDA_NVDEC_INVALID;
    }
    CUresult current = cuCtxSetCurrent(session->context);
    *driver_status = (unsigned int)current;
    if (current != CUDA_SUCCESS) return RUNTIME_CUDA_NVDEC_DRIVER;
    runtime_cuda_nvdec* library = session->library;
    CUVIDDECODECAPS caps;
    memset(&caps, 0, sizeof caps);
    caps.eCodecType = cudaVideoCodec_H264;
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = 0;
    CUresult result = library->get_decoder_caps(&caps);
    *driver_status = (unsigned int)result;
    if (result != CUDA_SUCCESS) return RUNTIME_CUDA_NVDEC_DRIVER;
    if (!caps.bIsSupported) return RUNTIME_CUDA_NVDEC_UNSUPPORTED;
    *max_width = caps.nMaxWidth;
    *max_height = caps.nMaxHeight;
    return RUNTIME_CUDA_NVDEC_OK;
}

static int CUDAAPI on_sequence(void* user, CUVIDEOFORMAT* format) {
    decode_state* state = user;
    if (state->status != RUNTIME_CUDA_NVDEC_OK) return 0;
    int left = format->display_area.left, top = format->display_area.top;
    int right = format->display_area.right, bottom = format->display_area.bottom;
    if (format->codec != cudaVideoCodec_H264 ||
        format->chroma_format != cudaVideoChromaFormat_420 ||
        format->bit_depth_luma_minus8 != 0 || format->bit_depth_chroma_minus8 != 0 ||
        left < 0 || top < 0 || right <= left || bottom <= top) {
        return fail(state, RUNTIME_CUDA_NVDEC_UNSUPPORTED, CUDA_SUCCESS);
    }
    uint32_t width = (uint32_t)(right - left), height = (uint32_t)(bottom - top);
    if (width % 2 != 0 || height % 2 != 0 || width > state->max_width ||
        height > state->max_height) {
        return fail(state, RUNTIME_CUDA_NVDEC_UNSUPPORTED, CUDA_SUCCESS);
    }
    unsigned int surfaces = format->min_num_decode_surfaces + EXTRA_DECODE_SURFACES;
    if (state->decoder != NULL) {
        /* A repeated sequence header is fine; a new geometry is not. */
        if (format->coded_width != state->coded_width ||
            format->coded_height != state->coded_height || width != state->width ||
            height != state->height) {
            return fail(state, RUNTIME_CUDA_NVDEC_UNSUPPORTED, CUDA_SUCCESS);
        }
        return (int)surfaces;
    }
    CUVIDDECODECREATEINFO info;
    memset(&info, 0, sizeof info);
    info.ulWidth = format->coded_width;
    info.ulHeight = format->coded_height;
    info.ulNumDecodeSurfaces = surfaces;
    info.CodecType = format->codec;
    info.ChromaFormat = format->chroma_format;
    info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    info.bitDepthMinus8 = 0;
    info.ulMaxWidth = format->coded_width;
    info.ulMaxHeight = format->coded_height;
    info.display_area.left = (short)left;
    info.display_area.top = (short)top;
    info.display_area.right = (short)right;
    info.display_area.bottom = (short)bottom;
    info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    info.ulTargetWidth = width;
    info.ulTargetHeight = height;
    info.ulNumOutputSurfaces = 2;
    info.vidLock = state->lock;
    CUresult result = state->library->create_decoder(&state->decoder, &info);
    if (result != CUDA_SUCCESS) {
        state->decoder = NULL;
        return fail(state, RUNTIME_CUDA_NVDEC_DRIVER, result);
    }
    state->host = malloc((size_t)width * height * 3 / 2);
    if (state->host == NULL) return fail(state, RUNTIME_CUDA_NVDEC_INVALID, CUDA_SUCCESS);
    state->coded_width = format->coded_width;
    state->coded_height = format->coded_height;
    state->width = width;
    state->height = height;
    return (int)surfaces;
}

static int CUDAAPI on_decode(void* user, CUVIDPICPARAMS* picture) {
    decode_state* state = user;
    if (state->status != RUNTIME_CUDA_NVDEC_OK || state->decoder == NULL) return 0;
    CUresult result = state->library->decode_picture(state->decoder, picture);
    if (result != CUDA_SUCCESS) return fail(state, RUNTIME_CUDA_NVDEC_DRIVER, result);
    return 1;
}

static int CUDAAPI on_display(void* user, CUVIDPARSERDISPINFO* display) {
    decode_state* state = user;
    if (display == NULL) return 1;
    if (state->status != RUNTIME_CUDA_NVDEC_OK || state->decoder == NULL) return 0;
    if (!state->select(state->user, display->timestamp)) return 1;
    CUVIDPROCPARAMS params;
    memset(&params, 0, sizeof params);
    params.progressive_frame = display->progressive_frame;
    params.top_field_first = display->top_field_first;
    params.output_stream = state->stream;
    unsigned long long frame = 0;
    unsigned int pitch = 0;
    CUresult result = state->library->map_frame(state->decoder, display->picture_index, &frame,
                                                &pitch, &params);
    if (result != CUDA_SUCCESS) return fail(state, RUNTIME_CUDA_NVDEC_DRIVER, result);
    CUDA_MEMCPY2D copy;
    memset(&copy, 0, sizeof copy);
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = (CUdeviceptr)frame;
    copy.srcPitch = pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = state->host;
    copy.dstPitch = state->width;
    copy.WidthInBytes = state->width;
    /* NV12 keeps its chroma plane directly below the luma plane in the same
     * pitch, so one copy of 3/2 the rows carries both planes. */
    copy.Height = (size_t)state->height * 3 / 2;
    CUresult copied = cuMemcpy2DAsync(&copy, state->stream);
    if (copied == CUDA_SUCCESS) copied = cuStreamSynchronize(state->stream);
    CUresult unmapped = state->library->unmap_frame(state->decoder, frame);
    if (copied != CUDA_SUCCESS) return fail(state, RUNTIME_CUDA_NVDEC_DRIVER, copied);
    if (unmapped != CUDA_SUCCESS) return fail(state, RUNTIME_CUDA_NVDEC_DRIVER, unmapped);
    if (state->frame(state->user, display->timestamp, state->width, state->height,
                     state->host) != 0) {
        return fail(state, RUNTIME_CUDA_NVDEC_SINK, CUDA_SUCCESS);
    }
    return 1;
}

int runtime_cuda_nvdec_decode_h264(runtime_cuda_nvdec_session* session, const uint8_t* stream,
                                   const uint64_t* unit_ends, const int64_t* timestamps,
                                   uint64_t units, uint32_t max_width, uint32_t max_height,
                                   void* user, runtime_cuda_nvdec_select select,
                                   runtime_cuda_nvdec_frame frame,
                                   unsigned int* driver_status) {
    if (session == NULL || driver_status == NULL || select == NULL || frame == NULL ||
        (units > 0 && (stream == NULL || unit_ends == NULL || timestamps == NULL))) {
        return RUNTIME_CUDA_NVDEC_INVALID;
    }
    CUresult current = cuCtxSetCurrent(session->context);
    *driver_status = (unsigned int)current;
    if (current != CUDA_SUCCESS) return RUNTIME_CUDA_NVDEC_DRIVER;
    runtime_cuda_nvdec* library = session->library;
    decode_state state;
    memset(&state, 0, sizeof state);
    state.library = library;
    state.lock = session->lock;
    state.stream = session->stream;
    state.max_width = max_width;
    state.max_height = max_height;
    state.user = user;
    state.select = select;
    state.frame = frame;

    CUVIDPARSERPARAMS parser_params;
    memset(&parser_params, 0, sizeof parser_params);
    parser_params.CodecType = cudaVideoCodec_H264;
    parser_params.ulMaxNumDecodeSurfaces = 1;
    parser_params.ulMaxDisplayDelay = 1;
    parser_params.pUserData = &state;
    parser_params.pfnSequenceCallback = on_sequence;
    parser_params.pfnDecodePicture = on_decode;
    parser_params.pfnDisplayPicture = on_display;
    CUvideoparser parser = NULL;
    CUresult result = library->create_parser(&parser, &parser_params);
    if (result != CUDA_SUCCESS) {
        *driver_status = (unsigned int)result;
        return RUNTIME_CUDA_NVDEC_DRIVER;
    }

    uint64_t start = 0;
    for (uint64_t unit = 0; unit < units && state.status == RUNTIME_CUDA_NVDEC_OK; ++unit) {
        uint64_t end = unit_ends[unit];
        if (end < start) {
            fail(&state, RUNTIME_CUDA_NVDEC_INVALID, CUDA_SUCCESS);
            break;
        }
        CUVIDSOURCEDATAPACKET packet;
        memset(&packet, 0, sizeof packet);
        packet.flags = CUVID_PKT_TIMESTAMP;
        packet.payload_size = (unsigned long)(end - start);
        packet.payload = stream + start;
        packet.timestamp = timestamps[unit];
        result = library->parse_data(parser, &packet);
        if (result != CUDA_SUCCESS) fail(&state, RUNTIME_CUDA_NVDEC_STREAM, result);
        start = end;
    }
    if (state.status == RUNTIME_CUDA_NVDEC_OK) {
        CUVIDSOURCEDATAPACKET packet;
        memset(&packet, 0, sizeof packet);
        packet.flags = CUVID_PKT_ENDOFSTREAM;
        result = library->parse_data(parser, &packet);
        if (result != CUDA_SUCCESS) fail(&state, RUNTIME_CUDA_NVDEC_STREAM, result);
    }
    library->destroy_parser(parser);
    if (state.decoder != NULL) library->destroy_decoder(state.decoder);
    free(state.host);
    if (state.status == RUNTIME_CUDA_NVDEC_OK && state.width == 0) {
        state.status = RUNTIME_CUDA_NVDEC_STREAM;
    }
    *driver_status = state.driver_status;
    return state.status;
}
