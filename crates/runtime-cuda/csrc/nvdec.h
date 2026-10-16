#ifndef RUNTIME_CUDA_NVDEC_H
#define RUNTIME_CUDA_NVDEC_H

#include <stdint.h>

/* Results. RUNTIME_CUDA_NVDEC_DRIVER carries a CUresult in *driver_status. */
#define RUNTIME_CUDA_NVDEC_OK 0
#define RUNTIME_CUDA_NVDEC_LIBRARY_ABSENT 1
#define RUNTIME_CUDA_NVDEC_SYMBOL_MISSING 2
#define RUNTIME_CUDA_NVDEC_DRIVER 3
#define RUNTIME_CUDA_NVDEC_UNSUPPORTED 4
#define RUNTIME_CUDA_NVDEC_STREAM 5
#define RUNTIME_CUDA_NVDEC_SINK 6
#define RUNTIME_CUDA_NVDEC_INVALID 7

typedef struct runtime_cuda_nvdec runtime_cuda_nvdec;

/* libnvcuvid plus a CUDA context, context lock and stream of its own on one
 * device. Use a session from one thread at a time. */
typedef struct runtime_cuda_nvdec_session runtime_cuda_nvdec_session;

/* Returns nonzero when the displayed frame with this timestamp is wanted. */
typedef int (*runtime_cuda_nvdec_select)(void* user, int64_t timestamp);

/* Receives one wanted frame as tightly packed NV12 in host memory, valid only
 * during the call. Returns zero to continue and nonzero to stop decoding. */
typedef int (*runtime_cuda_nvdec_frame)(void* user, int64_t timestamp, uint32_t width,
                                        uint32_t height, const uint8_t* nv12);

/* Loads libnvcuvid.so.1, resolves every entry point this helper uses, and
 * creates the session's context on device_ordinal, current on the calling
 * thread afterwards. A missing library is reported before the driver is
 * touched. */
int runtime_cuda_nvdec_session_open(int device_ordinal, runtime_cuda_nvdec_session** session,
                                    unsigned int* driver_status);

void runtime_cuda_nvdec_session_close(runtime_cuda_nvdec_session* session);

/* Reports the largest 8-bit 4:2:0 H.264 frame the session's device decodes. */
int runtime_cuda_nvdec_h264_capacity(runtime_cuda_nvdec_session* session, uint32_t* max_width,
                                     uint32_t* max_height, unsigned int* driver_status);

/* Decodes an Annex B H.264 stream split into access units on the session's
 * context. Unit i spans [unit_ends[i - 1], unit_ends[i]) of stream and carries
 * timestamps[i], which comes back with the frame it produces. Frames arrive in
 * display order. A sequence that is not 8-bit 4:2:0, has odd or zero display
 * dimensions, or is larger than max_width x max_height is refused. */
int runtime_cuda_nvdec_decode_h264(runtime_cuda_nvdec_session* session, const uint8_t* stream,
                                   const uint64_t* unit_ends, const int64_t* timestamps,
                                   uint64_t units, uint32_t max_width, uint32_t max_height,
                                   void* user, runtime_cuda_nvdec_select select,
                                   runtime_cuda_nvdec_frame frame,
                                   unsigned int* driver_status);

#endif
