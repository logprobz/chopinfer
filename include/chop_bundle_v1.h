#ifndef CHOP_BUNDLE_V1_H
#define CHOP_BUNDLE_V1_H
/* chop.model-options.v1 extends chopbundle-config-v1 only for packages
 * requiring that capability. Before the optional tap producer, append the
 * ASCII tag chop.model-options.v1, a little-endian u32 JSON byte length,
 * and an object of all resolved typed values, ordered by option name.
 * Bundles validate names, types and compiled limits at create. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DRAFT: this V1 header is not stable.
 *
 * Both independently compiled bundle fixtures now run through one unchanged
 * host binary, and that alone does not freeze the ABI. This header becomes
 * stable only once this repository carries both of:
 *
 *   fixtures/bundle-abi-v1/linux-unchanged-host.json
 *   fixtures/bundle-abi-v1/sm120-smoke.json
 *
 * the first being an authentic Linux unchanged-host result and the second an
 * SM120 smoke receipt. Neither exists, so this stays a draft. The
 * runtime-conformance abi_v1_stability test holds this comment, the decision
 * record, and those two files to one answer.
 */

#define CHOP_BUNDLE_ABI_VERSION_V1 UINT32_C(1)

/*
 * struct_size and abi_version, and why one is a range and the other is exact.
 *
 * Every record below opens with struct_size, which is the byte length of the
 * record the caller actually built. A reader refuses a record shorter than the
 * layout it knows, because the fields it would read were never written. It
 * accepts one at least that long and reads exactly its own V1 layout: the V1
 * fields are a fixed prefix, so a later revision of this ABI may append to the
 * tail, and a reader of this version never reads past what it declares here.
 *
 * That forward tail is deliberate, and it is deliberately inert. Nothing past
 * the V1 prefix is read, so a longer record is not a way to ask this version
 * for behaviour it does not have; it only says the caller's own record was
 * longer. Where a record is read through a pointer, which is the two tables and
 * the custom-launch record a bundle entry is handed, that tail is really there
 * and is left unread. Where a record is passed by value, the call itself
 * carries the V1 layout, so struct_size is the caller's claim about the record
 * it filled in rather than a longer thing crossing the call.
 *
 * That range rule holds only where the record's own start is what a reader was
 * handed. Every record here is in exactly one of three classes:
 *
 * 1. Exact-size array elements. Reached by advancing from the element before
 *    them, and no callback that takes an array carries an element stride:
 *
 *      chop_bundle_kernel_argument_v1   launch_cuda_symbol,
 *                                          execute_collective
 *      chop_bundle_custom_operand_v1    run_custom_operation
 *
 *    A longer record has a longer stride, which would put the reader's second
 *    element inside the caller's first; a shorter one runs the array off its own
 *    end. The first element's declared length is read on its own, before
 *    anything advances by a record, so an array strided differently is refused
 *    rather than walked.
 *
 * 2. Exact-size by-value nested records. Reached at a fixed offset inside a
 *    parent, whose every following field sits at an offset computed from this
 *    record's length:
 *
 *      chop_bundle_device_buffer_view_v1   compiled_image_view.buffer,
 *                                             tensor_map_2d_v1.buffer,
 *                                             kernel_argument_v1.buffer,
 *                                             custom_operand_v1.buffer,
 *                                             device_pointer_array_v1.slots[]
 *      chop_bundle_physical_kv_layout_v1   physical_kv_request_v1.layout
 *
 *    A longer nested record moves every parent field after it, and the parent's
 *    own range rule would accept that: the parent is longer too. Each is
 *    therefore checked at its exact length, and checked before any later parent
 *    field is read.
 *
 * 3. Tail-compatible records. Everything else, reached through its own pointer,
 *    its own table prefix, or a by-value call. These keep the range rule: a
 *    longer one is accepted and its tail left unread.
 *
 * Classes 1 and 2 admit no tail in either direction and V1 cannot grow them: a
 * revision that needs a longer one is changing where every reader of an array or
 * a container looks, which is a new ABI version rather than an addition to this
 * one. The lists above are complete. Every other pointer this ABI pairs with a
 * count is an array of a fixed-width scalar with no declared length to disagree
 * about, which is the byte views over uint8_t, the logits row's binary32 bits,
 * and the lent launch's operand_addresses and operand_lengths over uint64_t;
 * and the only other by-value members are chop_bundle_byte_view_v1 and the
 * opaque handles, which carry no struct_size because this version fixes their
 * layout outright.
 *
 * A class 3 record passed or returned by value, which is
 * chop_bundle_create_result_v1, chop_bundle_logits_row_v1,
 * chop_bundle_execute_result_sink_v1, and the several services that take a
 * chop_bundle_device_buffer_view_v1 directly, is not made exact by that
 * position. The function signature already fixes what crosses the call, and
 * struct_size there is the caller's claim about the record it filled in. A
 * revision that changed one of those layouts would change the calling
 * convention of the entry that carries it, which is a new ABI version for the
 * same reason. (The device-buffer view is exact anyway, but for its class 2
 * nesting rather than for its by-value parameter positions.)
 *
 * abi_version is exact. It appears in chop_bundle_host_services_v1,
 * chop_bundle_api_header_v1, and chop_bundle_api_v1, and V1 accepts
 * exactly CHOP_BUNDLE_ABI_VERSION_V1. A different version may renumber or
 * repurpose the prefix, which is the one thing that would make reading a prefix
 * unsafe, so it is rejected before an entry point is called no matter what
 * struct_size says.
 */

/*
 * The two rules one instance is used under, for its whole life.
 *
 * One instance belongs to one thread. Whichever thread called create owns that
 * instance: execute, commit, cancel, and destroy are called from it and from no
 * other, and every host service the bundle calls while it is inside one of those
 * is called from that same thread. A bundle may use threads of its own, but no
 * such thread may enter this ABI in either direction. V1 offers no lock, and the
 * host's own service state behind user_data is reference counted without atomics,
 * so a second thread is not slower here, it is unsound. Two instances of one
 * bundle are two owners and may run on two threads.
 *
 * At most one transaction is pending per instance. A successful execute opens
 * one, and commit or cancel closes it; there is no identifier for a second,
 * because there is never a second to name. Until it is closed, another execute
 * and destroy are both CHOP_BUNDLE_STATUS_V1_INVALID_STATE, and while none is
 * open, commit and cancel are the same. An execute that reports failure opened
 * nothing, so nothing is left to close; an execute that reports success opened
 * one even if it emitted no row, and the caller closes it.
 */

typedef uint32_t chop_bundle_status_v1;

#define CHOP_BUNDLE_STATUS_V1_SUCCESS UINT32_C(0)
#define CHOP_BUNDLE_STATUS_V1_INVALID_ARGUMENT UINT32_C(1)
#define CHOP_BUNDLE_STATUS_V1_UNSUPPORTED UINT32_C(2)
#define CHOP_BUNDLE_STATUS_V1_NOT_FOUND UINT32_C(3)
#define CHOP_BUNDLE_STATUS_V1_RESOURCE_EXHAUSTED UINT32_C(4)
#define CHOP_BUNDLE_STATUS_V1_STALE_HANDLE UINT32_C(5)
#define CHOP_BUNDLE_STATUS_V1_INVALID_STATE UINT32_C(6)
#define CHOP_BUNDLE_STATUS_V1_HOST_ERROR UINT32_C(7)
#define CHOP_BUNDLE_STATUS_V1_BUNDLE_ERROR UINT32_C(8)
#define CHOP_BUNDLE_STATUS_V1_ABI_MISMATCH UINT32_C(9)
#define CHOP_BUNDLE_STATUS_V1_PANIC UINT32_C(10)
#define CHOP_BUNDLE_STATUS_V1_INTERNAL_ERROR UINT32_C(11)

#define CHOP_BUNDLE_AUTHENTICATED_COMPONENT_ROLE_BYTES_MAX_V1 UINT64_C(4096)
#define CHOP_BUNDLE_AUTHENTICATED_COMPONENT_BYTES_MAX_V1 UINT64_C(1048576)

typedef struct chop_bundle_byte_view_v1 {
  const uint8_t *data;
  uint64_t length;
} chop_bundle_byte_view_v1;

typedef struct chop_bundle_mut_byte_view_v1 {
  uint8_t *data;
  uint64_t length;
} chop_bundle_mut_byte_view_v1;

/*
 * The caller keeps every nonempty byte view valid only until the receiving
 * function or callback returns. Fields named reserved are zero in V1.
 */

/*
 * Each opaque handle packs a host-owned slot and generation. Both values must
 * be nonzero.
 */
typedef struct chop_bundle_handle_v1 {
  uint64_t raw;
} chop_bundle_handle_v1;

typedef struct chop_bundle_context_handle_v1 {
  uint64_t raw;
} chop_bundle_context_handle_v1;

typedef struct chop_bundle_memory_pool_handle_v1 {
  uint64_t raw;
} chop_bundle_memory_pool_handle_v1;

typedef struct chop_bundle_device_buffer_handle_v1 {
  uint64_t raw;
} chop_bundle_device_buffer_handle_v1;

typedef struct chop_bundle_rank_arena_handle_v1 {
  uint64_t raw;
} chop_bundle_rank_arena_handle_v1;

typedef struct chop_bundle_compiled_image_handle_v1 {
  uint64_t raw;
} chop_bundle_compiled_image_handle_v1;

typedef struct chop_bundle_authenticated_component_handle_v1 {
  uint64_t raw;
} chop_bundle_authenticated_component_handle_v1;

typedef struct chop_bundle_peer_access_handle_v1 {
  uint64_t raw;
} chop_bundle_peer_access_handle_v1;

typedef struct chop_bundle_stream_handle_v1 {
  uint64_t raw;
} chop_bundle_stream_handle_v1;

typedef struct chop_bundle_graph_handle_v1 {
  uint64_t raw;
} chop_bundle_graph_handle_v1;

typedef struct chop_bundle_event_handle_v1 {
  uint64_t raw;
} chop_bundle_event_handle_v1;

typedef struct chop_bundle_cuda_module_handle_v1 {
  uint64_t raw;
} chop_bundle_cuda_module_handle_v1;

typedef struct chop_bundle_cuda_symbol_handle_v1 {
  uint64_t raw;
} chop_bundle_cuda_symbol_handle_v1;

typedef struct chop_bundle_physical_kv_handle_v1 {
  uint64_t raw;
} chop_bundle_physical_kv_handle_v1;

typedef struct chop_bundle_kv_request_handle_v1 {
  uint64_t raw;
} chop_bundle_kv_request_handle_v1;

typedef struct chop_bundle_kv_transaction_handle_v1 {
  uint64_t raw;
} chop_bundle_kv_transaction_handle_v1;

typedef struct chop_bundle_state_handle_v1 {
  uint64_t raw;
} chop_bundle_state_handle_v1;

typedef struct chop_bundle_collective_handle_v1 {
  uint64_t raw;
} chop_bundle_collective_handle_v1;

/* One bounded range of a host-issued allocation.

   struct_size must be sizeof(chop_bundle_device_buffer_view_v1) exactly.
   This is a class 2 record: it is embedded by value in five others, and each of
   them computes the offset of every field after it from this length, so a tail
   here would move the parent's own fields rather than only its own. The length
   rule at the top of this header states the three classes. */
typedef struct chop_bundle_device_buffer_view_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  chop_bundle_device_buffer_handle_v1 buffer;
  uint64_t offset;
  uint64_t length;
} chop_bundle_device_buffer_view_v1;

typedef struct chop_bundle_compiled_image_view_v1 {
  uint32_t struct_size;
  uint32_t rank;
  chop_bundle_device_buffer_view_v1 buffer;
  uint64_t alignment;
} chop_bundle_compiled_image_view_v1;

typedef struct chop_bundle_authenticated_component_metadata_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t role_length;
  uint64_t byte_length;
  uint8_t sha256[64];
} chop_bundle_authenticated_component_metadata_v1;

/* The blocks one thread-block cluster may hold.

   Eight is the portable maximum: a cluster of that size launches on every
   architecture that forms clusters at all, and a larger one is an opt-in this
   version does not carry. A launch asking for more is refused rather than
   reduced to one that would have run. */
#define CHOP_BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS UINT32_C(8)

/* One launch shape, with the cluster it is grouped into.

   cluster_x, cluster_y, and cluster_z are blocks per cluster, and each is at
   least one. All three at one is the default and asks for no cluster at all:
   that is the ordinary launch, and it is what every kernel written without
   cluster synchronization or distributed shared memory takes.

   Any other shape is exact. Its blocks, multiplied out, stay inside
   CHOP_BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS, and each grid extent divides by
   the cluster extent beside it, because a partial cluster is not a launch CUDA
   forms. A host that cannot launch the cluster asked for refuses the launch. It
   never drops the cluster and launches the blocks anyway: a kernel that reads
   another block's shared memory or waits on a cluster barrier does something
   else entirely without one, and the flattened launch would be a different
   program reported as this one. */
typedef struct chop_bundle_launch_config_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  uint32_t shared_memory_bytes;
  uint32_t cluster_x;
  uint32_t cluster_y;
  uint32_t cluster_z;
} chop_bundle_launch_config_v1;

/* Launch config flags are bits, not C enum values. A host that cannot apply a
   flag refuses the launch, and any bit outside the known mask is refused. */
#define CHOP_BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION UINT32_C(1)
/* Every block of the grid resident at once, or the launch is refused; with
   the launch's cluster when it has one. Takes no programmatic stream
   serialization. */
#define CHOP_BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE UINT32_C(2)
#define CHOP_BUNDLE_LAUNCH_FLAGS_V1_KNOWN \
  (CHOP_BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION | \
   CHOP_BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE)

/* Tensor-map element types are closed scalar codes. FP8 values are encoded as
   CU_TENSOR_MAP_DATA_TYPE_UINT8, with FP8 interpretation owned by the kernel.
   The packed types use CUDA's 16-element storage groups. */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_U8 UINT32_C(1)       /* 1 byte  */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_I32 UINT32_C(2)      /* 4 bytes */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16 UINT32_C(3)     /* 2 bytes */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_F16 UINT32_C(4)      /* 2 bytes */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_F32 UINT32_C(5)      /* 4 bytes */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3 UINT32_C(6) /* 1 byte  */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2 UINT32_C(7) /* 1 byte  */
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_U16 UINT32_C(8)
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B UINT32_C(9)
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B UINT32_C(10)
#define CHOP_BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B UINT32_C(11)

/* Tensor-map swizzles are a closed set of scalar codes. */
#define CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE UINT32_C(0)
#define CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128 UINT32_C(1)
#define CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32 UINT32_C(2)
#define CHOP_BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64 UINT32_C(3)
#define CHOP_BUNDLE_TENSOR_MAP_INTERLEAVE_V1_NONE UINT32_C(0)
#define CHOP_BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_16 UINT32_C(1)
#define CHOP_BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32 UINT32_C(2)
#define CHOP_BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_NONE UINT32_C(0)
#define CHOP_BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_64 UINT32_C(1)
#define CHOP_BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_128 UINT32_C(2)
#define CHOP_BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_256 UINT32_C(3)
#define CHOP_BUNDLE_TENSOR_MAP_OOB_FILL_V1_NONE UINT32_C(0)
#define CHOP_BUNDLE_TENSOR_MAP_OOB_FILL_V1_NAN_REQUEST_ZERO_FMA UINT32_C(1)

/* The largest box extent the tiled encoding admits per dimension, and the
   alignment it requires of a map's global base address and its row stride.
   A bundle builds a descriptor that satisfies both, or the host refuses it. */
#define CHOP_BUNDLE_TENSOR_MAP_MAX_BOX_V1 UINT32_C(256)
#define CHOP_BUNDLE_TENSOR_MAP_ADDRESS_ALIGNMENT_V1 UINT64_C(16)

/* One tiled tensor map over a host-issued device-buffer span. Rank is 1 to 5.
   Only the first rank dimensions, rank-1 strides, and rank box and element
   strides are used; all unused array entries and reserved must be zero.
   Dimensions are 1..2^32, strides are aligned to 16 bytes and below 2^40,
   box dimensions are 1..256, and element strides are 1..8. The final row's
   last byte must fit inside buffer, with checked arithmetic. The resolved
   address is aligned to 16 bytes, or 32 bytes for 32B interleave and packed
   16B types. Interleave requires rank at least 3; 32B interleave requires
   32B swizzle and 32B strides. Packed 16B types require 32B strides and
   global_dims[0] divisible by 128; packed 8B U4 requires an even inner dim.
   16U6 requires no interleave. For no interleave, inner box bytes must be a
   multiple of 16, and swizzled inner box bytes must fit the swizzle span.
   Packed 16B types require box_dims[0] of 128. NaN OOB fill requires F16,
   BF16, or F32. The host validates these rules before calling the driver. */
typedef struct chop_bundle_tensor_map_v1 {
  uint32_t struct_size;
  uint32_t rank;
  uint32_t element_type;
  uint32_t swizzle;
  uint32_t interleave;
  uint32_t l2_promotion;
  uint32_t oob_fill;
  uint32_t reserved;
  chop_bundle_device_buffer_view_v1 buffer;
  uint64_t global_dims[5];
  uint64_t global_strides_bytes[4];
  uint32_t box_dims[5];
  uint32_t element_strides[5];
} chop_bundle_tensor_map_v1;

/* The slots one by-value device-pointer array carries.

   Eight is the bound, not a shape: a fixed kernel ABI that takes a small
   by-value table of device addresses declares how many of them it uses, and
   the rest are null. Nothing here knows what a slot means. */
#define CHOP_BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS UINT32_C(8)

/* A bounded by-value array of device addresses the host resolves.

   Some fixed kernel ABIs take a small struct of device pointers by value. A
   bundle cannot build one: it holds spans, not addresses. So it names the spans
   in slot order and the host writes the addresses it has for them into the
   argument, in that order.

   `slot_count` is how many slots the argument carries, from one through
   `CHOP_BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS`. A slot whose view names
   the null handle with a zero offset and length is the null pointer, which is
   how such an ABI's unused slots are expressed. Every other slot is resolved
   and bounds checked exactly as a device-buffer argument is.

   Opaque bytes are deliberately not this. A bundle that wanted to pass
   addresses as bytes would have to know them, and it never does. */
typedef struct chop_bundle_device_pointer_array_v1 {
  uint32_t struct_size;
  uint32_t slot_count;
  chop_bundle_device_buffer_view_v1
      slots[CHOP_BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS];
} chop_bundle_device_pointer_array_v1;

/* Kernel argument kind values are scalar codes, not C enum values.
   `U32`, `I32`, and `U64` are launched at exactly the width and signedness they
   name. Opaque bytes stay opaque: this version defines no launch encoding for
   them, and a host asked to launch one reports the argument unsupported rather
   than inventing a width for it. */
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER UINT32_C(1)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_U64 UINT32_C(2)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES UINT32_C(3)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_U32 UINT32_C(4)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER UINT32_C(5)
/* Draft ABI value 6 now names the rank-1-to-5 tensor map. */
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP UINT32_C(6)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_I32 UINT32_C(7)
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY UINT32_C(8)

/* The bytes one kernel argument may occupy.

   An argument list is an array of these, passed by value, and a launch may
   carry tens of them. The two large payloads are therefore named rather than
   carried, which is what keeps this record small enough that the array is a
   cheap thing to build and copy. The number is a ceiling on the V1 layout, and
   the size gate below is what holds it under the ceiling. It is not room this
   record may grow into: the length rule below admits no tail at all. */
#define CHOP_BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES UINT32_C(96)

/* One kernel argument: a kind, and exactly the payload that kind selects.

   struct_size must be sizeof(chop_bundle_kernel_argument_v1) exactly. This
   is the one record whose length is not a range, because it is read as an array
   element and no callback carries a stride; the length rule at the top of this
   header states why, and why V1 cannot grow this record.

   `kind` is the discriminant and it is load bearing. Exactly one payload
   belongs to each kind, and the fields no kind selects carry nothing:

     DEVICE_BUFFER          buffer
     U64 / U32 / I32        the scalar of that width
     NULL_DEVICE_POINTER    nothing
     OPAQUE_BYTES           opaque
     TENSOR_MAP             tensor_map, nonnull
     DEVICE_POINTER_ARRAY   pointer_array, nonnull

   `tensor_map` and `pointer_array` point at records the caller keeps valid for
   the length of the receiving call and no longer, exactly as a byte view is.
   The record a kind selects must be nonnull, and each pointer-bearing field a
   kind does not select must be absent: a null pointer, and the empty view for
   `opaque`. A record that carries a payload its kind does not name says two
   things at once, so it is refused rather than read as whichever one the reader
   happened to look at first. `buffer` is a handle and two offsets rather than a
   pointer, so it is read only for DEVICE_BUFFER and decides nothing elsewhere.

   Nothing here is dereferenced before the kind and the pointers have been
   checked, and the payload records are themselves validated after they are
   read, exactly as they were when this record carried them by value. The two
   payload records are read through a pointer rather than as array elements, so
   each keeps the ordinary length rule and may carry a tail. */
typedef struct chop_bundle_kernel_argument_v1 {
  uint32_t struct_size;
  uint32_t kind;
  chop_bundle_device_buffer_view_v1 buffer;
  uint64_t scalar_u64;
  uint32_t scalar_u32;
  int32_t scalar_i32;
  chop_bundle_byte_view_v1 opaque;
  const chop_bundle_tensor_map_v1 *tensor_map;
  const chop_bundle_device_pointer_array_v1 *pointer_array;
} chop_bundle_kernel_argument_v1;

/* The size gate, checked wherever this header is compiled. A record that
   outgrew the ceiling gives this typedef a negative length, which is a
   diagnostic in every C and C++ dialect this header is read by. */
typedef char chop_bundle_kernel_argument_v1_size_gate[
    (sizeof(chop_bundle_kernel_argument_v1) <=
     CHOP_BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES)
        ? 1
        : -1];

/* One model-neutral physical KV page layout.

   struct_size must be sizeof(chop_bundle_physical_kv_layout_v1) exactly.
   This is a class 2 record: it is embedded by value in the physical KV request,
   whose page_capacity and request_capacity sit after it, so a tail here would
   move both. The length rule at the top of this header states the three
   classes. */
typedef struct chop_bundle_physical_kv_layout_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t rank_count;
  uint64_t page_bytes;
  uint64_t page_alignment;
  uint64_t page_tokens;
  uint64_t state_lifetime;
  chop_bundle_byte_view_v1 layout_id;
} chop_bundle_physical_kv_layout_v1;

typedef struct chop_bundle_physical_kv_request_v1 {
  uint32_t struct_size;
  uint32_t flags;
  chop_bundle_physical_kv_layout_v1 layout;
  uint64_t page_capacity;
  uint64_t request_capacity;
} chop_bundle_physical_kv_request_v1;

typedef struct chop_bundle_state_request_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t byte_length;
  uint64_t alignment;
  chop_bundle_byte_view_v1 state_id;
} chop_bundle_state_request_v1;

typedef struct chop_bundle_collective_request_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t rank_count;
  chop_bundle_byte_view_v1 semantic_contract;
} chop_bundle_collective_request_v1;

/*
 * One device-buffer span a bundle asks the host to lend a custom operation.
 *
 * struct_size must be sizeof(chop_bundle_custom_operand_v1) exactly. This is
 * the second of the two records whose length is not a range, because it is read
 * as an element of the array run_custom_operation takes and that callback
 * carries no stride; the length rule at the top of this header states why, and
 * why V1 cannot grow this record.
 *
 * `flags` is a bit mask, not an enum. A writable operand may share no byte with
 * any other operand of the same call; the host refuses the whole call rather
 * than invoking anything when one does.
 */
#define CHOP_BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE UINT32_C(1)
#define CHOP_BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN \
  CHOP_BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE

typedef struct chop_bundle_custom_operand_v1 {
  uint32_t struct_size;
  uint32_t flags;
  chop_bundle_device_buffer_view_v1 buffer;
} chop_bundle_custom_operand_v1;

/*
 * Which memory the lent addresses and the lent stream belong to.
 *
 * A host with a driver lends device addresses and its own driver stream. A host
 * without one lends host addresses and a null stream. An operation that needs
 * one of them checks this field and refuses the other, so a callback compiled
 * against a driver never runs on host memory.
 */
#define CHOP_BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE UINT32_C(1)
#define CHOP_BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST UINT32_C(2)

/*
 * The raw values one host lends a custom operation for exactly one call.
 *
 * `operand_addresses` and `operand_lengths` each hold `operand_count` entries,
 * in the order the operands were passed, and both are null exactly when the
 * count is zero. `stream` is the host's own stream, already made current with
 * its context.
 *
 * Nothing here may be retained. The record, both arrays, the stream, and every
 * address stop being the operation's the moment its callback returns. The host
 * waited on that stream before the callback, and waits again afterwards when
 * the operation said it may enqueue, so no work of the operation outlives what
 * it was lent, and the host alone releases, synchronizes, and orders every
 * resource behind these values.
 */
typedef struct chop_bundle_custom_launch_v1 {
  uint32_t struct_size;
  uint32_t address_space;
  uint64_t operand_count;
  void *stream;
  const uint64_t *operand_addresses;
  const uint64_t *operand_lengths;
} chop_bundle_custom_launch_v1;

/*
 * What one custom operation may leave behind on the stream it is lent.
 *
 * The flags are a bit mask, not an enum, and this version defines exactly one.
 * A call without it is synchronous: the operation must complete every piece of
 * work it started before its callback returns, and the host waits for nothing
 * afterwards. A call with MAY_ENQUEUE is asynchronous: the operation may leave
 * work ordered on the lent stream, and the host waits for that work before it
 * returns and before anything the operation was lent can be released.
 *
 * The host waits on the stream before every callback, whatever the flags, so an
 * operation is never handed a stream with work it did not order still running,
 * and a wait the host cannot complete invokes nothing at all.
 */
#define CHOP_BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE UINT32_C(1)
#define CHOP_BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN \
  CHOP_BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE

/*
 * The one operation code this version reserves.
 *
 * An entry that contained a fault of its own reports it with this value rather
 * than letting it cross the boundary. A bundle numbers its own codes with
 * anything else.
 */
#define CHOP_BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT INT32_MIN

/*
 * The bundle-side entry the host invokes with those lent values.
 *
 * Zero is success. Any other value is the operation's own code, which the host
 * carries back unchanged beside CHOP_BUNDLE_STATUS_V1_BUNDLE_ERROR. The
 * callback must not let an exception or a panic cross this boundary.
 */
typedef int32_t (*chop_bundle_custom_operation_fn_v1)(
    void *operation_user_data,
    const chop_bundle_custom_launch_v1 *launch);

/*
 * Runs one bundle-supplied custom operation under host-established context.
 *
 * `operands` is null exactly when `operand_count` is zero, and both it and
 * `operation_id` remain valid for the callback duration. `operation` is
 * nonnull. `out_operation_status` is nonnull and receives the value the
 * callback returned, and only when the callback ran. A host that reports
 * CHOP_BUNDLE_STATUS_V1_BUNDLE_ERROR writes the operation's own nonzero code
 * there; zero is not a code a failure can carry.
 *
 * `flags` is the completion policy above. A bit this version does not define is
 * refused.
 *
 * The host resolves every handle, bounds checks every span against the
 * allocation it names, checks that the context, the stream, and every operand
 * belong to one another, refuses aliased writable operands, and waits for the
 * stream. A refusal at any of those, the wait included, invokes nothing. Only
 * after all of it does the host make the context current and lend the raw
 * values above.
 */
typedef chop_bundle_status_v1 (*chop_bundle_run_custom_operation_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    chop_bundle_stream_handle_v1 stream,
    uint32_t flags,
    chop_bundle_byte_view_v1 operation_id,
    const chop_bundle_custom_operand_v1 *operands,
    uint64_t operand_count,
    chop_bundle_custom_operation_fn_v1 operation,
    void *operation_user_data,
    int32_t *out_operation_status);

typedef chop_bundle_status_v1 (*chop_bundle_log_fn_v1)(
    void *user_data,
    chop_bundle_byte_view_v1 message);

typedef chop_bundle_status_v1 (*chop_bundle_select_context_fn_v1)(
    void *user_data,
    uint32_t rank,
    int32_t device_ordinal,
    chop_bundle_context_handle_v1 *out_context);
typedef chop_bundle_status_v1 (*chop_bundle_release_context_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context);
/*
 * Selects the one pool a context has.
 *
 * A context has one pool, so selecting it twice returns the same handle rather
 * than a second name for one object, and each selection is one user: the pool
 * goes back to the driver when the last selection releases it. Selecting a
 * context is the same rule, one handle per rank.
 *
 * That is what makes a pool handle and a context handle exact names for the
 * driver objects underneath, which is what a peer lease is keyed by, and it
 * holds because the host validates its topology first: two ranks may not name
 * one device ordinal, so no two ranks share a primary context.
 */
typedef chop_bundle_status_v1 (*chop_bundle_select_memory_pool_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    chop_bundle_memory_pool_handle_v1 *out_pool);
/*
 * Takes or gives back one lease over a pool's access for one peer context.
 *
 * enabled != 0 acquires a lease, enabled == 0 gives one back. The pair is a
 * lease, not a switch: the host counts the leases one instance holds over each
 * directed edge, asks the device only on the first acquire, and enables the
 * grant only if it was not already there. The last release disables it only if
 * this host enabled it, so a grant another user of the device made stays.
 *
 * A release with no lease behind it is CHOP_BUNDLE_STATUS_V1_INVALID_STATE.
 * A refused enable or disable leaves the lease exactly as it was and is
 * retryable. A pool a lease was granted through cannot be released while that
 * lease stands.
 */
typedef chop_bundle_status_v1 (*chop_bundle_set_memory_pool_peer_access_fn_v1)(
    void *user_data,
    chop_bundle_memory_pool_handle_v1 pool,
    chop_bundle_context_handle_v1 peer_context,
    uint32_t enabled);
typedef chop_bundle_status_v1 (*chop_bundle_release_memory_pool_fn_v1)(
    void *user_data,
    chop_bundle_memory_pool_handle_v1 pool);
typedef chop_bundle_status_v1 (*chop_bundle_allocate_device_buffer_fn_v1)(
    void *user_data,
    chop_bundle_memory_pool_handle_v1 pool,
    chop_bundle_stream_handle_v1 stream,
    uint64_t byte_length,
    uint64_t alignment,
    chop_bundle_device_buffer_handle_v1 *out_buffer);
typedef chop_bundle_status_v1 (*chop_bundle_release_device_buffer_fn_v1)(
    void *user_data,
    chop_bundle_device_buffer_handle_v1 buffer);

typedef chop_bundle_status_v1 (*chop_bundle_open_compiled_image_fn_v1)(
    void *user_data,
    chop_bundle_byte_view_v1 component_role,
    chop_bundle_compiled_image_handle_v1 *out_image);
typedef chop_bundle_status_v1 (*chop_bundle_upload_rank_arena_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    chop_bundle_memory_pool_handle_v1 pool,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_compiled_image_handle_v1 image,
    uint32_t rank,
    chop_bundle_rank_arena_handle_v1 *out_arena);
typedef struct chop_bundle_rank_arena_upload_v1 {
  chop_bundle_context_handle_v1 context;
  chop_bundle_memory_pool_handle_v1 pool;
  chop_bundle_stream_handle_v1 stream;
  chop_bundle_compiled_image_handle_v1 image;
  uint32_t rank;
  uint32_t reserved;
} chop_bundle_rank_arena_upload_v1;
typedef chop_bundle_status_v1 (*chop_bundle_upload_rank_arenas_fn_v1)(
    void *user_data,
    const chop_bundle_rank_arena_upload_v1 *uploads,
    uint64_t count,
    chop_bundle_rank_arena_handle_v1 *out_arenas);
typedef chop_bundle_status_v1 (*chop_bundle_release_rank_arena_fn_v1)(
    void *user_data,
    chop_bundle_rank_arena_handle_v1 arena);
typedef chop_bundle_status_v1 (*chop_bundle_resolve_compiled_image_view_fn_v1)(
    void *user_data,
    chop_bundle_compiled_image_handle_v1 image,
    chop_bundle_rank_arena_handle_v1 arena,
    uint32_t rank,
    chop_bundle_byte_view_v1 view_id,
    chop_bundle_compiled_image_view_v1 *out_view);
typedef chop_bundle_status_v1 (*chop_bundle_release_compiled_image_fn_v1)(
    void *user_data,
    chop_bundle_compiled_image_handle_v1 image);

/*
 * Opens only bytes authenticated by the package verifier. Inspect returns the
 * exact package role, size, and lowercase hexadecimal SHA-256. Read requires
 * exact-sized destinations and copies the exact role and payload. The handle
 * owns the stable snapshot until release and rejects stale or foreign values.
 */
typedef chop_bundle_status_v1 (*chop_bundle_open_authenticated_component_fn_v1)(
    void *user_data,
    chop_bundle_byte_view_v1 component_role,
    chop_bundle_authenticated_component_handle_v1 *out_component);
typedef chop_bundle_status_v1 (*chop_bundle_inspect_authenticated_component_fn_v1)(
    void *user_data,
    chop_bundle_authenticated_component_handle_v1 component,
    chop_bundle_authenticated_component_metadata_v1 *out_metadata);
typedef chop_bundle_status_v1 (*chop_bundle_read_authenticated_component_fn_v1)(
    void *user_data,
    chop_bundle_authenticated_component_handle_v1 component,
    chop_bundle_mut_byte_view_v1 role_destination,
    chop_bundle_mut_byte_view_v1 bytes_destination);
typedef chop_bundle_status_v1 (*chop_bundle_release_authenticated_component_fn_v1)(
    void *user_data,
    chop_bundle_authenticated_component_handle_v1 component);

/*
 * Takes one lease over one directed context peer edge.
 *
 * Each handle is one lease. The host counts the leases one instance holds over
 * each edge, enables the edge on the first one only if it was not enabled
 * already, and remembers which of those happened. Releasing a handle gives one
 * lease back; the last one disables the edge only if this host enabled it, so
 * an edge another user of the devices made stays.
 *
 * A stale handle is CHOP_BUNDLE_STATUS_V1_STALE_HANDLE, and a release out of
 * order is CHOP_BUNDLE_STATUS_V1_INVALID_STATE. A refused enable or disable
 * leaves the lease exactly as it was and is retryable. Neither context of a
 * live lease can be released while it stands.
 */
typedef chop_bundle_status_v1 (*chop_bundle_acquire_peer_access_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 owner_context,
    chop_bundle_context_handle_v1 peer_context,
    chop_bundle_peer_access_handle_v1 *out_peer_access);
typedef chop_bundle_status_v1 (*chop_bundle_release_peer_access_fn_v1)(
    void *user_data,
    chop_bundle_peer_access_handle_v1 peer_access);

typedef chop_bundle_status_v1 (*chop_bundle_create_stream_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    uint32_t stream_kind,
    chop_bundle_stream_handle_v1 *out_stream);
typedef chop_bundle_status_v1 (*chop_bundle_select_stage_stream_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    uint32_t rank,
    chop_bundle_byte_view_v1 stage_id,
    uint32_t stream_kind,
    chop_bundle_stream_handle_v1 *out_stream);
typedef chop_bundle_status_v1 (*chop_bundle_synchronize_stream_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream);
typedef chop_bundle_status_v1 (*chop_bundle_release_stream_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream);

typedef chop_bundle_status_v1 (*chop_bundle_create_event_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    chop_bundle_event_handle_v1 *out_event);
typedef chop_bundle_status_v1 (*chop_bundle_record_event_fn_v1)(
    void *user_data,
    chop_bundle_event_handle_v1 event,
    chop_bundle_stream_handle_v1 stream);
typedef chop_bundle_status_v1 (*chop_bundle_wait_event_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_event_handle_v1 event);
typedef chop_bundle_status_v1 (*chop_bundle_synchronize_event_fn_v1)(
    void *user_data,
    chop_bundle_event_handle_v1 event);
typedef chop_bundle_status_v1 (*chop_bundle_release_event_fn_v1)(
    void *user_data,
    chop_bundle_event_handle_v1 event);

typedef chop_bundle_status_v1 (*chop_bundle_copy_host_to_device_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_device_buffer_view_v1 destination,
    chop_bundle_byte_view_v1 source);
typedef chop_bundle_status_v1 (*chop_bundle_copy_device_to_host_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_mut_byte_view_v1 destination,
    chop_bundle_device_buffer_view_v1 source);
typedef chop_bundle_status_v1 (*chop_bundle_copy_device_to_device_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_device_buffer_view_v1 destination,
    chop_bundle_device_buffer_view_v1 source);
typedef chop_bundle_status_v1 (*chop_bundle_copy_peer_to_peer_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_context_handle_v1 destination_context,
    chop_bundle_device_buffer_view_v1 destination,
    chop_bundle_context_handle_v1 source_context,
    chop_bundle_device_buffer_view_v1 source);
typedef chop_bundle_status_v1 (*chop_bundle_zero_device_buffer_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_device_buffer_view_v1 destination);
/*
 * Writes a bounded table of resolved device addresses into a host-owned span.
 *
 * This is the one thing a bundle cannot express as a copy: the bytes it wants
 * written are addresses only the host knows. So the bundle names the
 * destination and the spans whose addresses go into it, in slot order, and the
 * host resolves every one of them, requires the destination to hold exactly
 * `slot_count` eight-byte slots, and writes them little-endian in that order. A
 * slot naming the null handle writes a null pointer.
 *
 * The transfer completes before this returns, exactly as a host transfer does,
 * so the table is readable by anything ordered on `stream` afterwards.
 *
 * `table` is nonnull and remains valid for the callback duration.
 */
typedef chop_bundle_status_v1 (*chop_bundle_write_device_pointer_table_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_device_buffer_view_v1 destination,
    const chop_bundle_device_pointer_array_v1 *table);

typedef chop_bundle_status_v1 (*chop_bundle_load_cuda_module_fn_v1)(
    void *user_data,
    chop_bundle_context_handle_v1 context,
    chop_bundle_byte_view_v1 module_bytes,
    chop_bundle_cuda_module_handle_v1 *out_module);
typedef chop_bundle_status_v1 (*chop_bundle_resolve_cuda_symbol_fn_v1)(
    void *user_data,
    chop_bundle_cuda_module_handle_v1 module,
    chop_bundle_byte_view_v1 symbol_name,
    chop_bundle_cuda_symbol_handle_v1 *out_symbol);
typedef chop_bundle_status_v1 (*chop_bundle_configure_cuda_symbol_fn_v1)(
    void *user_data,
    chop_bundle_cuda_symbol_handle_v1 symbol,
    uint32_t dynamic_shared_memory_bytes);
/*
 * `config` is nonnull. `arguments` is null exactly when `argument_count` is
 * zero. Both records remain valid for the callback duration.
 */
typedef chop_bundle_status_v1 (*chop_bundle_launch_cuda_symbol_fn_v1)(
    void *user_data,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_cuda_symbol_handle_v1 symbol,
    const chop_bundle_launch_config_v1 *config,
    const chop_bundle_kernel_argument_v1 *arguments,
    uint64_t argument_count);
typedef chop_bundle_status_v1 (*chop_bundle_release_cuda_symbol_fn_v1)(
    void *user_data,
    chop_bundle_cuda_symbol_handle_v1 symbol);
typedef chop_bundle_status_v1 (*chop_bundle_release_cuda_module_fn_v1)(
    void *user_data,
    chop_bundle_cuda_module_handle_v1 module);

/* `request` is nonnull and remains valid for the callback duration. */
typedef chop_bundle_status_v1 (*chop_bundle_allocate_physical_kv_fn_v1)(
    void *user_data,
    const chop_bundle_physical_kv_request_v1 *request,
    chop_bundle_physical_kv_handle_v1 *out_kv);
typedef chop_bundle_status_v1 (*chop_bundle_create_kv_request_fn_v1)(
    void *user_data,
    chop_bundle_physical_kv_handle_v1 kv,
    uint64_t quota_tokens,
    chop_bundle_kv_request_handle_v1 *out_request);
typedef chop_bundle_status_v1 (*chop_bundle_prepare_kv_fn_v1)(
    void *user_data,
    chop_bundle_kv_request_handle_v1 request,
    chop_bundle_byte_view_v1 operation,
    chop_bundle_kv_transaction_handle_v1 *out_transaction);
typedef chop_bundle_status_v1 (*chop_bundle_commit_kv_fn_v1)(
    void *user_data,
    chop_bundle_kv_transaction_handle_v1 transaction);
typedef chop_bundle_status_v1 (*chop_bundle_cancel_kv_fn_v1)(
    void *user_data,
    chop_bundle_kv_transaction_handle_v1 transaction);
typedef chop_bundle_status_v1 (*chop_bundle_resolve_kv_view_fn_v1)(
    void *user_data,
    chop_bundle_kv_request_handle_v1 request,
    uint32_t rank,
    chop_bundle_byte_view_v1 view_id,
    chop_bundle_device_buffer_view_v1 *out_view);
typedef chop_bundle_status_v1 (*chop_bundle_release_kv_request_fn_v1)(
    void *user_data,
    chop_bundle_kv_request_handle_v1 request);
typedef chop_bundle_status_v1 (*chop_bundle_release_physical_kv_fn_v1)(
    void *user_data,
    chop_bundle_physical_kv_handle_v1 kv);

/* Serving KV is one process-lifetime pool per instance, selected by the
   manifest's layout_id. A serving bundle calls acquire_serving_kv near the
   end of create, after weights and workspaces are resident, and before graph
   capture. The host sizes every rank from the tightest rank's remaining free
   memory and returns the common page capacity. Each page ID in an execute
   batch indexes page_bytes at that ID in every rank view. Pool addresses stay
   fixed through graph capture and execute. The host applies prepared partial
   tail copies before execute and commits or rolls back logical pages after
   sampling. Bundle commit/cancel still finalize bundle-owned state. These
   optional tail services require a struct_size check before a bundle reads
   them; no bundle releases the serving pool. */
typedef chop_bundle_status_v1 (*chop_bundle_acquire_serving_kv_fn_v1)(
    void *user_data,
    chop_bundle_byte_view_v1 layout_id,
    uint64_t *out_page_capacity);
typedef chop_bundle_status_v1 (*chop_bundle_resolve_serving_kv_pool_fn_v1)(
    void *user_data,
    uint32_t rank,
    chop_bundle_device_buffer_view_v1 *out_view);

/* `request` is nonnull and remains valid for the callback duration. */
typedef chop_bundle_status_v1 (*chop_bundle_allocate_state_fn_v1)(
    void *user_data,
    const chop_bundle_state_request_v1 *request,
    chop_bundle_state_handle_v1 *out_state);
typedef chop_bundle_status_v1 (*chop_bundle_resolve_state_view_fn_v1)(
    void *user_data,
    chop_bundle_state_handle_v1 state,
    uint32_t rank,
    chop_bundle_device_buffer_view_v1 *out_view);
typedef chop_bundle_status_v1 (*chop_bundle_release_state_fn_v1)(
    void *user_data,
    chop_bundle_state_handle_v1 state);

/* `request` is nonnull and remains valid for the callback duration. */
typedef chop_bundle_status_v1 (*chop_bundle_create_collective_fn_v1)(
    void *user_data,
    const chop_bundle_collective_request_v1 *request,
    chop_bundle_collective_handle_v1 *out_collective);
typedef chop_bundle_status_v1 (*chop_bundle_execute_collective_fn_v1)(
    void *user_data,
    chop_bundle_collective_handle_v1 collective,
    chop_bundle_stream_handle_v1 stream,
    chop_bundle_byte_view_v1 operation,
    const chop_bundle_kernel_argument_v1 *arguments,
    uint64_t argument_count);
typedef chop_bundle_status_v1 (*chop_bundle_release_collective_fn_v1)(
    void *user_data,
    chop_bundle_collective_handle_v1 collective);

/* Capture uses CUDA's thread-local mode. Begin and end run on the same host
   thread. A second stream in the same context joins the capture through
   record_event and wait_event. End returns an instantiated graph handle.
   While a context captures, services that synchronize, allocate, or copy host
   memory return INVALID_STATE. Graph handles are context-owned and must be
   released before their context. */
typedef chop_bundle_status_v1 (*chop_bundle_begin_graph_capture_fn_v1)(
    void *user_data, chop_bundle_stream_handle_v1 stream);
typedef chop_bundle_status_v1 (*chop_bundle_end_graph_capture_fn_v1)(
    void *user_data, chop_bundle_stream_handle_v1 stream,
    chop_bundle_graph_handle_v1 *out_graph);
typedef chop_bundle_status_v1 (*chop_bundle_launch_graph_fn_v1)(
    void *user_data, chop_bundle_stream_handle_v1 stream,
    chop_bundle_graph_handle_v1 graph);
typedef chop_bundle_status_v1 (*chop_bundle_release_graph_fn_v1)(
    void *user_data, chop_bundle_graph_handle_v1 graph);

/* Query during create. A zero draft limit means speculation is disabled. */
typedef struct chop_bundle_speculation_v1 {
  uint32_t struct_size;
  uint32_t maximum_draft_tokens;
  uint32_t maximum_verify_tokens;
} chop_bundle_speculation_v1;
typedef chop_bundle_status_v1 (*chop_bundle_query_speculation_fn_v1)(
    void *user_data, chop_bundle_speculation_v1 *out);

/* Query during create. Zero means the host reports no such limit. */
typedef struct chop_bundle_batch_limits_v1 {
  uint32_t struct_size;
  uint32_t maximum_scored_rows;
} chop_bundle_batch_limits_v1;
typedef chop_bundle_status_v1 (*chop_bundle_query_batch_limits_fn_v1)(
    void *user_data, chop_bundle_batch_limits_v1 *out);

/* Query during create for one resolved symbol and launch shape. A zero field
   is not reported. */
#define CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT 1u
#define CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED 2u
/* The device accepts cooperative launches: a cooperative launch whose
   clusters fit max_active_clusters runs every block at once. */
#define CHOP_BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH 4u
typedef struct chop_bundle_cluster_residency_v1 {
  uint32_t struct_size;
  uint32_t multiprocessor_count;
  uint32_t usable_multiprocessors;
  uint32_t max_active_clusters;
  uint32_t flags;
} chop_bundle_cluster_residency_v1;
typedef chop_bundle_status_v1 (*chop_bundle_query_cluster_residency_fn_v1)(
    void *user_data, chop_bundle_cuda_symbol_handle_v1 symbol,
    const chop_bundle_launch_config_v1 *config,
    chop_bundle_cluster_residency_v1 *out);

/*
 * Host-mapped gather (CHOP_HOST_MAPPED_GATHER_CAPABILITY_V1). Optional tail: a
 * bundle reads gather_mapped_rows_v1 only when struct_size covers it.
 *
 * gather_mapped_rows_v1 copies rows of one view the image's plan placed
 * "mapped" into the destination, ordered on the stream like
 * copy_host_to_device: row i of the destination is row rows[i] of the view
 * view_id of image for rank, each row_bytes long, so the destination is
 * row_count * row_bytes bytes. The host reads the rows on the CPU into
 * page-locked memory it owns before the call returns, and keeps that memory
 * until the copy completes; rows may be reused at once. rows is null exactly
 * when row_count is zero, which is refused. A view the image does not map for
 * this rank is NOT_FOUND; a row length that does not divide the view, a row
 * past its end or a destination of another length is INVALID_ARGUMENT; a
 * stream being captured is refused like other host copies.
 */
typedef struct chop_bundle_mapped_row_gather_v1 {
  uint32_t struct_size;
  uint32_t rank;
  chop_bundle_compiled_image_handle_v1 image;
  chop_bundle_byte_view_v1 view_id;
  uint64_t row_bytes;
  const uint32_t *rows;
  uint64_t row_count;
} chop_bundle_mapped_row_gather_v1;
typedef chop_bundle_status_v1 (*chop_bundle_gather_mapped_rows_fn_v1)(
    void *user_data, chop_bundle_stream_handle_v1 stream,
    chop_bundle_device_buffer_view_v1 destination,
    const chop_bundle_mapped_row_gather_v1 *gather);

/* Optional prepared gather: bounded pinned ID/output staging, one pending
 * enqueue per handle. wait orders the first consumer; complete is nonblocking
 * and must succeed before results are published. Release fences pending work.
 * All five callbacks must be present under chop.async-mapped-gather.v1. */
typedef struct chop_bundle_mapped_gather_handle_v1 {
  uint64_t raw;
} chop_bundle_mapped_gather_handle_v1;
typedef struct chop_bundle_mapped_gather_plane_v1 {
  uint32_t struct_size;
  uint32_t rank;
  chop_bundle_compiled_image_handle_v1 image;
  chop_bundle_byte_view_v1 view_id;
  uint64_t row_bytes;
  uint64_t index_offset;
  chop_bundle_stream_handle_v1 stream;
  chop_bundle_device_buffer_view_v1 destination;
} chop_bundle_mapped_gather_plane_v1;
typedef struct chop_bundle_mapped_gather_completion_v1 {
  uint32_t struct_size;
  uint32_t ready;
  uint64_t host_nanoseconds;
  uint64_t exposed_nanoseconds;
} chop_bundle_mapped_gather_completion_v1;
typedef chop_bundle_status_v1 (*chop_bundle_create_mapped_gather_fn_v1)(
    void *, chop_bundle_stream_handle_v1, chop_bundle_device_buffer_view_v1,
    const chop_bundle_mapped_gather_plane_v1 *, uint64_t,
    chop_bundle_mapped_gather_handle_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_enqueue_mapped_gather_fn_v1)(
    void *, chop_bundle_mapped_gather_handle_v1, uint64_t);
typedef chop_bundle_status_v1 (*chop_bundle_wait_mapped_gather_fn_v1)(
    void *, chop_bundle_mapped_gather_handle_v1, chop_bundle_stream_handle_v1);
typedef chop_bundle_status_v1 (*chop_bundle_complete_mapped_gather_fn_v1)(
    void *, chop_bundle_mapped_gather_handle_v1, chop_bundle_mapped_gather_completion_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_release_mapped_gather_fn_v1)(
    void *, chop_bundle_mapped_gather_handle_v1);

/* Optional CPU-only row-set registration. Array elements have exact size.
 * Registration never allocates device or pinned staging; caching is optional. */
typedef struct chop_bundle_mapped_row_plane_v1 {
  uint32_t struct_size;
  uint32_t rank;
  chop_bundle_compiled_image_handle_v1 image;
  chop_bundle_byte_view_v1 view_id;
  uint64_t row_bytes;
} chop_bundle_mapped_row_plane_v1;
typedef chop_bundle_status_v1 (*chop_bundle_register_mapped_row_set_fn_v1)(
    void *, const chop_bundle_mapped_row_plane_v1 *, uint64_t);

typedef struct chop_bundle_host_services_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void *user_data;
  chop_bundle_log_fn_v1 log;
  chop_bundle_select_context_fn_v1 select_context;
  chop_bundle_release_context_fn_v1 release_context;
  chop_bundle_select_memory_pool_fn_v1 select_memory_pool;
  chop_bundle_set_memory_pool_peer_access_fn_v1 set_memory_pool_peer_access;
  chop_bundle_release_memory_pool_fn_v1 release_memory_pool;
  chop_bundle_allocate_device_buffer_fn_v1 allocate_device_buffer;
  chop_bundle_release_device_buffer_fn_v1 release_device_buffer;
  chop_bundle_open_compiled_image_fn_v1 open_compiled_image;
  chop_bundle_upload_rank_arena_fn_v1 upload_rank_arena;
  chop_bundle_release_rank_arena_fn_v1 release_rank_arena;
  chop_bundle_resolve_compiled_image_view_fn_v1 resolve_compiled_image_view;
  chop_bundle_release_compiled_image_fn_v1 release_compiled_image;
  chop_bundle_open_authenticated_component_fn_v1 open_authenticated_component;
  chop_bundle_inspect_authenticated_component_fn_v1 inspect_authenticated_component;
  chop_bundle_read_authenticated_component_fn_v1 read_authenticated_component;
  chop_bundle_release_authenticated_component_fn_v1 release_authenticated_component;
  chop_bundle_acquire_peer_access_fn_v1 acquire_peer_access;
  chop_bundle_release_peer_access_fn_v1 release_peer_access;
  chop_bundle_create_stream_fn_v1 create_stream;
  chop_bundle_select_stage_stream_fn_v1 select_stage_stream;
  chop_bundle_synchronize_stream_fn_v1 synchronize_stream;
  chop_bundle_release_stream_fn_v1 release_stream;
  chop_bundle_create_event_fn_v1 create_event;
  chop_bundle_record_event_fn_v1 record_event;
  chop_bundle_wait_event_fn_v1 wait_event;
  chop_bundle_synchronize_event_fn_v1 synchronize_event;
  chop_bundle_release_event_fn_v1 release_event;
  chop_bundle_copy_host_to_device_fn_v1 copy_host_to_device;
  chop_bundle_copy_device_to_host_fn_v1 copy_device_to_host;
  chop_bundle_copy_device_to_device_fn_v1 copy_device_to_device;
  chop_bundle_copy_peer_to_peer_fn_v1 copy_peer_to_peer;
  chop_bundle_zero_device_buffer_fn_v1 zero_device_buffer;
  chop_bundle_write_device_pointer_table_fn_v1 write_device_pointer_table;
  chop_bundle_load_cuda_module_fn_v1 load_cuda_module;
  chop_bundle_resolve_cuda_symbol_fn_v1 resolve_cuda_symbol;
  chop_bundle_configure_cuda_symbol_fn_v1 configure_cuda_symbol;
  chop_bundle_launch_cuda_symbol_fn_v1 launch_cuda_symbol;
  chop_bundle_release_cuda_symbol_fn_v1 release_cuda_symbol;
  chop_bundle_release_cuda_module_fn_v1 release_cuda_module;
  chop_bundle_allocate_physical_kv_fn_v1 allocate_physical_kv;
  chop_bundle_create_kv_request_fn_v1 create_kv_request;
  chop_bundle_prepare_kv_fn_v1 prepare_kv;
  chop_bundle_commit_kv_fn_v1 commit_kv;
  chop_bundle_cancel_kv_fn_v1 cancel_kv;
  chop_bundle_resolve_kv_view_fn_v1 resolve_kv_view;
  chop_bundle_release_kv_request_fn_v1 release_kv_request;
  chop_bundle_release_physical_kv_fn_v1 release_physical_kv;
  chop_bundle_allocate_state_fn_v1 allocate_state;
  chop_bundle_resolve_state_view_fn_v1 resolve_state_view;
  chop_bundle_release_state_fn_v1 release_state;
  chop_bundle_create_collective_fn_v1 create_collective;
  chop_bundle_execute_collective_fn_v1 execute_collective;
  chop_bundle_release_collective_fn_v1 release_collective;
  chop_bundle_run_custom_operation_fn_v1 run_custom_operation;
  chop_bundle_begin_graph_capture_fn_v1 begin_graph_capture;
  chop_bundle_end_graph_capture_fn_v1 end_graph_capture;
  chop_bundle_launch_graph_fn_v1 launch_graph;
  chop_bundle_release_graph_fn_v1 release_graph;
  /* Optional tail. Check struct_size before reading either field. */
  chop_bundle_acquire_serving_kv_fn_v1 acquire_serving_kv;
  chop_bundle_resolve_serving_kv_pool_fn_v1 resolve_serving_kv_pool;
  /* Optional batch upload tail. Check struct_size before reading this field. */
  chop_bundle_upload_rank_arenas_fn_v1 upload_rank_arenas;
  /* Optional speculation tail. An absent callback means disabled. */
  chop_bundle_query_speculation_fn_v1 query_speculation_v1;
  /* Optional batch limits tail. An absent callback reports no limits. */
  chop_bundle_query_batch_limits_fn_v1 query_batch_limits_v1;
  /* Optional cluster residency tail. An absent callback reports nothing. */
  chop_bundle_query_cluster_residency_fn_v1 query_cluster_residency_v1;
  /* Optional host-mapped gather tail. Check struct_size before reading this. */
  chop_bundle_gather_mapped_rows_fn_v1 gather_mapped_rows_v1;
  chop_bundle_create_mapped_gather_fn_v1 create_mapped_gather_v1;
  chop_bundle_enqueue_mapped_gather_fn_v1 enqueue_mapped_gather_v1;
  chop_bundle_wait_mapped_gather_fn_v1 wait_mapped_gather_v1;
  chop_bundle_complete_mapped_gather_fn_v1 complete_mapped_gather_v1;
  chop_bundle_release_mapped_gather_fn_v1 release_mapped_gather_v1;
  chop_bundle_register_mapped_row_set_fn_v1 register_mapped_row_set_v1;
} chop_bundle_host_services_v1;

typedef struct chop_bundle_create_result_v1 {
  uint32_t struct_size;
  chop_bundle_status_v1 status;
  chop_bundle_handle_v1 handle;
} chop_bundle_create_result_v1;

/*
 * One logits row from a pending model transaction. `logit_bits` contains
 * little-endian IEEE 754 binary32 bit patterns. Its length is exactly four
 * times `logit_count`.
 */
typedef struct chop_bundle_logits_row_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t request_id;
  uint64_t logit_count;
  chop_bundle_byte_view_v1 logit_bits;
} chop_bundle_logits_row_v1;

typedef chop_bundle_status_v1 (*chop_bundle_emit_logits_row_fn_v1)(
    void *user_data,
    chop_bundle_logits_row_v1 row);

/* One ordered device logits view. The buffer belongs to `context` and `rank`.
 * Rows occupy `row_stride_elements` elements each. `live_width` excludes any
 * padded vocabulary tail. Normally `first_item_index` indexes the execute
 * call's ordered items that expect logits. With `chop.prompt-rows.v2` or
 * `chop.speculative.v1`, it indexes flattened output slots: one per prompt
 * token for a flagged item,
 * one per verify row for a verify item, otherwise one per item that expects
 * logits. The view remains valid through
 * the result callback and the host's queued sampling copy. */
#define CHOP_BUNDLE_LOGITS_ELEMENT_V1_F32 UINT32_C(1)
#define CHOP_BUNDLE_LOGITS_ELEMENT_V1_BF16 UINT32_C(2)
typedef struct chop_bundle_logits_view_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  chop_bundle_context_handle_v1 context;
  uint32_t rank;
  uint32_t element_type;
  chop_bundle_device_buffer_view_v1 buffer;
  uint64_t row_stride_elements;
  uint32_t row_count;
  uint32_t live_width;
  uint32_t first_item_index;
  uint32_t reserved_tail;
} chop_bundle_logits_view_v1;

typedef chop_bundle_status_v1 (*chop_bundle_emit_logits_view_fn_v1)(
    void *user_data,
    chop_bundle_logits_view_v1 view);

/* Device acceptance records are little-endian. They do not enable async
 * execution by themselves. Request/slot membership, completion dependencies,
 * model status and every sampled-row record must be checked before commit. */
#define CHOP_BUNDLE_ACCEPTANCE_INPUT_BYTES_V1 UINT32_C(40)
#define CHOP_BUNDLE_ACCEPTANCE_RESULT_BYTES_V1 UINT32_C(32)
#define CHOP_BUNDLE_ACCEPTANCE_NO_DRAFT_V1 UINT32_MAX
#define CHOP_BUNDLE_ACCEPTANCE_NO_FAILURE_V1 UINT32_MAX
#define CHOP_BUNDLE_ACCEPTANCE_TERMINAL_V1 UINT32_C(1)
#define CHOP_BUNDLE_ACCEPTANCE_REJECTED_V1 UINT32_C(2)
typedef struct chop_bundle_acceptance_input_v1 {
  uint64_t cycle_epoch;
  uint64_t remaining_tokens;
  uint32_t first_sample_row;
  uint32_t draft_item_index;
  uint32_t draft_tokens;
  uint32_t stop_offset;
  uint32_t stop_count;
  uint32_t reserved;
} chop_bundle_acceptance_input_v1;
/* status 0 is success and requires failure_row=UINT32_MAX. Status 2 is
 * an empty distribution at failure_row; its counts cannot authorize commit.
 * Every other nonzero status is fatal. Stop IDs are termination IDs, not
 * the sampler mask list. Accepted drafts exclude the bonus sample. */
typedef struct chop_bundle_acceptance_result_v1 {
  uint64_t cycle_epoch;
  uint32_t accepted_drafts;
  uint32_t emitted_rows;
  uint32_t flags;
  uint32_t status;
  uint32_t failure_row;
  uint32_t reserved;
} chop_bundle_acceptance_result_v1;

/* Optional device-result records. Spans and ready events remain borrowed
 * until all consumers and final collection finish. The host checks handle
 * generations, context ownership and capacity before submitting work. */
#define CHOP_DEVICE_RESULTS_HOST_VALUES_V1 UINT32_C(0)
#define CHOP_DEVICE_RESULTS_DEVICE_FEEDBACK_V1 UINT32_C(1)
#define CHOP_DEVICE_SAMPLE_RECORD_BYTES_V1 UINT32_C(256)
typedef struct chop_bundle_ready_device_view_v1 {
  chop_bundle_context_handle_v1 context;
  uint32_t rank;
  uint32_t reserved;
  chop_bundle_device_buffer_view_v1 buffer;
  chop_bundle_event_handle_v1 ready;
} chop_bundle_ready_device_view_v1;
typedef struct chop_bundle_device_drafts_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t cycle_epoch;
  uint64_t request_id;
  uint64_t slot_generation;
  uint32_t requested_tokens;
  uint32_t computed_tokens;
  uint64_t token_stride_bytes;
  chop_bundle_ready_device_view_v1 proposals;
  chop_bundle_device_buffer_view_v1 confidence;
} chop_bundle_device_drafts_v1;
typedef struct chop_bundle_ready_logits_view_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t cycle_epoch;
  chop_bundle_logits_view_v1 logits;
  chop_bundle_event_handle_v1 ready;
} chop_bundle_ready_logits_view_v1;
typedef struct chop_bundle_device_sampling_request_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  chop_bundle_ready_logits_view_v1 logits;
  chop_bundle_device_buffer_view_v1 output;
  chop_bundle_event_handle_v1 done;
} chop_bundle_device_sampling_request_v1;
typedef struct chop_bundle_device_completion_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t cycle_epoch;
  uint32_t item_count;
  uint32_t sample_rows;
  uint64_t acceptance_offset;
  uint64_t samples_offset;
  chop_bundle_ready_device_view_v1 packet;
} chop_bundle_device_completion_v1;
/* Metadata arrays are immutable through the enclosing execute. Items are
 * 56-byte little-endian records: request ID, slot generation, acceptance
 * input. Stop IDs are packed U32 termination IDs. Copy before retaining. */
typedef struct chop_bundle_device_cycle_view_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t cycle_epoch;
  chop_bundle_byte_view_v1 items;
  chop_bundle_byte_view_v1 termination_stop_ids;
} chop_bundle_device_cycle_view_v1;

/* Draft tokens are little-endian u32 values. Emit before logits for this item. */
typedef struct chop_bundle_draft_tokens_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t request_id;
  uint32_t token_count;
  chop_bundle_byte_view_v1 token_bytes;
} chop_bundle_draft_tokens_v1;
typedef chop_bundle_status_v1 (*chop_bundle_emit_draft_tokens_fn_v1)(
    void *user_data, chop_bundle_draft_tokens_v1 drafts);

/*
 * A bounded, host-owned result channel for one execute call. The bundle calls
 * `emit_logits_row` synchronously before execute returns. The host copies row
 * bytes before the callback returns. The bundle retains neither this sink nor
 * any row pointer. The bundle rejects output over these limits with
 * CHOP_BUNDLE_STATUS_V1_RESOURCE_EXHAUSTED.
 */
typedef struct chop_bundle_execute_result_sink_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  void *user_data;
  uint64_t maximum_rows;
  uint64_t maximum_logits_per_row;
  uint64_t maximum_total_logits;
  chop_bundle_emit_logits_row_fn_v1 emit_logits_row;
  /* Optional tail. Check struct_size before reading this field. */
  chop_bundle_emit_logits_view_fn_v1 emit_logits_view;
  /* Optional speculation tail. Check struct_size before reading this field. */
  chop_bundle_emit_draft_tokens_fn_v1 emit_draft_tokens;
} chop_bundle_execute_result_sink_v1;

/* Separate pointer-based sink, used only by the optional export below.
 * The legacy execute's by-value sink and signature are unchanged. Every
 * callback is required when this extension is used. Short tables are refused. */
typedef chop_bundle_status_v1 (*chop_bundle_describe_device_cycle_fn_v1)(void *, chop_bundle_device_cycle_view_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_declare_input_dependency_fn_v1)(void *, uint32_t);
typedef chop_bundle_status_v1 (*chop_bundle_emit_device_drafts_fn_v1)(void *, const chop_bundle_device_drafts_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_sample_device_logits_fn_v1)(void *, const chop_bundle_device_sampling_request_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_emit_ready_logits_fn_v1)(void *, const chop_bundle_ready_logits_view_v1 *);
typedef chop_bundle_status_v1 (*chop_bundle_emit_device_completion_fn_v1)(void *, const chop_bundle_device_completion_v1 *);
typedef struct chop_bundle_device_result_sink_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  chop_bundle_execute_result_sink_v1 base;
  chop_bundle_describe_device_cycle_fn_v1 describe_device_cycle_v1;
  chop_bundle_declare_input_dependency_fn_v1 declare_input_dependency_v1;
  chop_bundle_emit_device_drafts_fn_v1 emit_device_drafts_v1;
  chop_bundle_sample_device_logits_fn_v1 sample_device_logits_v1;
  chop_bundle_emit_ready_logits_fn_v1 emit_ready_logits_view_v1;
  chop_bundle_emit_device_completion_fn_v1 emit_device_completion_v1;
} chop_bundle_device_result_sink_v1;
typedef chop_bundle_status_v1 (*chop_bundle_execute_device_results_fn_v1)(
    chop_bundle_handle_v1, chop_bundle_byte_view_v1, const chop_bundle_device_result_sink_v1 *);
typedef struct chop_bundle_device_results_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  chop_bundle_execute_device_results_fn_v1 execute_device_results_v1;
} chop_bundle_device_results_api_v1;
/* Optional export under chop.device-results.v1. Loading it does not enable
 * execution: the host must explicitly admit the package and select this path. */
const chop_bundle_device_results_api_v1 *chop_bundle_get_device_results_api_v1(void);


/*
 * Once the host calls create, it keeps that service table, every callback in it,
 * and its user_data resident until the process exits. This holds whatever create
 * returns, because a bundle may keep the table it was handed and then report a
 * failure, and the host cannot see which copies a bundle kept. A bundle may
 * therefore hold the table for as long as it runs. This bounded retention is
 * intentional: V1 loads bundles at process start and never unloads them. Destroy
 * releases the instance and its device resources, not the table.
 *
 * If commit, cancel, or destroy returns an error, the instance is poisoned. The
 * host makes no further lifecycle calls and also keeps the bundle module
 * resident until process exit.
 */
typedef chop_bundle_create_result_v1 (*chop_bundle_create_fn_v1)(
    const chop_bundle_host_services_v1 *host_services,
    chop_bundle_byte_view_v1 configuration);
typedef chop_bundle_status_v1 (*chop_bundle_execute_fn_v1)(
    chop_bundle_handle_v1 handle,
    chop_bundle_byte_view_v1 request,
    chop_bundle_execute_result_sink_v1 result_sink);
typedef chop_bundle_status_v1 (*chop_bundle_commit_fn_v1)(
    chop_bundle_handle_v1 handle);
typedef chop_bundle_status_v1 (*chop_bundle_cancel_fn_v1)(
    chop_bundle_handle_v1 handle);
typedef chop_bundle_status_v1 (*chop_bundle_destroy_fn_v1)(
    chop_bundle_handle_v1 handle);

/* The loader reads this fixed prefix before it copies the full API table. */
typedef struct chop_bundle_api_header_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
} chop_bundle_api_header_v1;

typedef struct chop_bundle_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  chop_bundle_create_fn_v1 create;
  chop_bundle_execute_fn_v1 execute;
  chop_bundle_commit_fn_v1 commit;
  chop_bundle_cancel_fn_v1 cancel;
  chop_bundle_destroy_fn_v1 destroy;
} chop_bundle_api_v1;

/* Optional draft V1 extension. The entry still returns &table.base.
 * Read the callback only when base.struct_size covers its entire field.
 * A null callback or an old table means retirement is unsupported (a no-op).
 * A required-capability host must validate callback presence before GPU setup.
 * Retirement is synchronous, after commit/cancel and before the next execute.
 * Call only while Ready. Failure poisons the instance, forbidding more callbacks.
 */
#define CHOP_REQUEST_RETIREMENT_CAPABILITY_V1 "chop.request-retirement.v1"
typedef chop_bundle_status_v1 (*chop_bundle_retire_request_fn_v1)(
    chop_bundle_handle_v1 instance_handle, uint64_t request_id);

typedef struct chop_bundle_api_with_request_retirement_v1 {
  chop_bundle_api_v1 base;
  chop_bundle_retire_request_fn_v1 retire_request_v1;
} chop_bundle_api_with_request_retirement_v1;

#define CHOP_SPECULATIVE_CAPABILITY_V1 "chop.speculative.v1"

/*
 * Host-resident compiled-image views. A manifest view marked
 * "placement": "host" is not uploaded to the device: the host reads its bytes
 * into page-locked, device-mapped host memory, and resolve_compiled_image_view
 * answers that view with a buffer the device reads over PCIe. The rank arena
 * holds the rest of the rank image: each host span, rounded up to the image
 * alignment, is cut out and the bytes after it move down by that length, so
 * the whole-arena view is shorter than the rank image by the rounded host
 * bytes. Such an image aligns no view above the image alignment and leaves
 * every rank at least one device byte. A package declaring host views must
 * require this capability.
 */
#define CHOP_HOST_RESIDENT_VIEWS_CAPABILITY_V1 "chop.host-resident-views.v1"

/*
 * Host-mapped weight-plan views. A planned view with placement 2 ("mapped"), or
 * 3 ("mapped, prewarmed"), is
 * one checkpoint tensor, or one contiguous run of its leading-axis rows, stored
 * unchanged by exactly one plain-copy record. The host maps those bytes
 * read-only from the checkpoint file it opened without following links and
 * checked against its header, once per process for every rank naming the same
 * bytes, and cuts the span out of the rank arena as it does a host-resident
 * view. Neither the device nor a page-locked budget holds them, and no bundle
 * sees their address: gather_mapped_rows_v1 copies chosen rows to a device
 * buffer. A prewarmed view's bytes are read into the page cache at startup,
 * before the host reports ready. The checkpoint must not change while it is
 * served: a file truncated
 * while mapped makes the host's reads of the lost pages fault. A package
 * declaring mapped views must require this capability.
 */
#define CHOP_HOST_MAPPED_GATHER_CAPABILITY_V1 "chop.host-mapped-gather.v1"

/* Array elements have exact size because this callback carries no stride. */
typedef struct chop_bundle_kept_rows_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t request_id;
  uint32_t kept_rows; /* 1 through the item's verify-row count */
  uint32_t reserved_tail;
} chop_bundle_kept_rows_v1;

typedef chop_bundle_status_v1 (*chop_bundle_commit_accepted_fn_v1)(
    chop_bundle_handle_v1 handle,
    const chop_bundle_kept_rows_v1 *records,
    uint32_t record_count);

/* A speculative bundle exports the base pointer with this table's full size.
 * commit_accepted_v1 closes the pending transaction. It receives one record
 * per verify item, in item order; all other items commit whole. Calling the
 * old commit on a transaction with verify items returns INVALID_STATE. */
typedef struct chop_bundle_api_with_speculation_v1 {
  chop_bundle_api_with_request_retirement_v1 base;
  chop_bundle_commit_accepted_fn_v1 commit_accepted_v1;
} chop_bundle_api_with_speculation_v1;

/* Prefix caching. A bundle that requires this capability
 * exports the base pointer with this table's full size; commit_accepted_v1
 * may be null. All three callbacks must be present. They run only while the
 * instance is ready, and a failure changes nothing and does not poison it,
 * except that a capture or restore returns INTERNAL_ERROR when it cannot
 * establish that its device work on `pages` finished. The host then poisons
 * the instance and never reuses those pages.
 *
 * describe_prefix_state_v1 reports how many serving-pool pages one snapshot
 * of a request's bundle-owned state needs (zero is allowed).
 *
 * capture_prefix_state_v1 writes the request's state after its committed
 * `tokens` into `pages`. The host calls it right after a prefill commit and
 * before the request retires; token_ids is empty. The pages belong to no
 * sequence until the host evicts the anchor.
 *
 * restore_prefix_state_v1 creates the state of a request the bundle has not
 * seen, as if it had committed exactly `token_ids` (`tokens` IDs), from pages
 * a capture wrote. The host calls it before the request's first execute.
 *
 * Both finish their device work before returning. */
#define CHOP_PREFIX_CACHE_CAPABILITY_V1 "chop.prefix-cache.v1"

/* Host-memory KV offload uses the prefix-state callbacks, with no new ABI
 * tail. Requires chop.prefix-cache.v1. Capture may follow any committed
 * prefill, decode or verify boundary, with no open transaction. It either
 * captures that exact boundary or refuses without changing request state.
 * Restore at B followed by a decode or verify item at B reproduces the
 * uncaptured request's rows (subject to the bundle's numerical contract).
 *
 * On successful capture, all device writes to that request's serving KV
 * pages by preceding steps and commits, and to the supplied snapshot pages
 * by capture, are complete on every rank. The host may then copy those
 * opaque pages on its own streams. On failure, submitted snapshot work must
 * still be drained; unconfirmed completion returns INTERNAL_ERROR, so the
 * host must not reuse pages. Restore finishes before snapshot pages return
 * to the pool. No capability changes the page layout or ownership rules. */
#define CHOP_KV_OFFLOAD_CAPABILITY_V1 "chop.kv-offload.v1"

typedef struct chop_bundle_prefix_state_info_v1 {
  uint32_t struct_size;
  chop_bundle_status_v1 status;
  uint64_t state_pages;
} chop_bundle_prefix_state_info_v1;

typedef struct chop_bundle_prefix_state_v1 {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t request_id;
  uint64_t tokens;
  chop_bundle_byte_view_v1 token_ids; /* little-endian u32; empty for capture */
  chop_bundle_byte_view_v1 pages;     /* little-endian u64 page IDs */
} chop_bundle_prefix_state_v1;

typedef chop_bundle_prefix_state_info_v1 (*chop_bundle_describe_prefix_state_fn_v1)(
    chop_bundle_handle_v1 handle);
typedef chop_bundle_status_v1 (*chop_bundle_prefix_state_fn_v1)(
    chop_bundle_handle_v1 handle, chop_bundle_prefix_state_v1 state);

typedef struct chop_bundle_api_with_prefix_cache_v1 {
  chop_bundle_api_with_speculation_v1 base;
  chop_bundle_describe_prefix_state_fn_v1 describe_prefix_state_v1;
  chop_bundle_prefix_state_fn_v1 capture_prefix_state_v1;
  chop_bundle_prefix_state_fn_v1 restore_prefix_state_v1;
} chop_bundle_api_with_prefix_cache_v1;

/* Restore into the request's state slot. A bundle with request state
 * (chop.recurrent-state.v1) keeps it in the slot the host assigned at
 * admission, but restore_prefix_state_v1 names only the request, and the
 * snapshot pages are free again when it returns. A bundle that requires this
 * capability exports the base pointer with this table's full size, and the
 * host then restores through restore_prefix_state_in_slot_v1 alone: the same
 * call, plus `state_slot`, the slot every batch of the request carries. The
 * bundle writes the snapshot into that slot before returning. The rules of
 * restore_prefix_state_v1 apply unchanged. */
#define CHOP_PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1 "chop.prefix-cache.state-slot.v1"

typedef chop_bundle_status_v1 (*chop_bundle_restore_prefix_state_in_slot_fn_v1)(
    chop_bundle_handle_v1 handle, chop_bundle_prefix_state_v1 state, uint64_t state_slot);

typedef struct chop_bundle_api_with_prefix_state_slot_v1 {
  chop_bundle_api_with_prefix_cache_v1 base;
  chop_bundle_restore_prefix_state_in_slot_fn_v1 restore_prefix_state_in_slot_v1;
} chop_bundle_api_with_prefix_state_slot_v1;

typedef const chop_bundle_api_v1 *(*chop_bundle_get_api_v1_fn)(void);

/* The one native bundle entry point. */
const chop_bundle_api_v1 *chop_bundle_get_api_v1(void);

/*
 * Direct weight loading (optional exports, capability chop.weight-plan.v1).
 *
 * A bundle that loads a checkpoint without compiled rank images exports
 * chop_bundle_weight_plan_v1. The host calls it before create, once per
 * compiled image of the selected topology and rank, with an encoded request
 * and a sink. The bundle answers through the sink with that image's and
 * rank's encoded plan (emit, exactly once), an empty document (emit with
 * length 0) to decline the image, or refusal text (emit_error, at most once,
 * then a failure status). The host copies either before the callback returns.
 * For a planned image the host executes the plan's records with no model
 * code, registers the image under its role, and serves the plan's components
 * as authenticated components of the same roles; a direct load reads no
 * compiled image, so it refuses a package with an image the bundle declines.
 * A planned rank's identity (what the chop.compiled-image-identity.v1
 * component reports as its direct_identity, in place of an image sha256) is
 * the SHA-256 of "CHOPWPI2", the 32-byte checkpoint identity, the request's
 * 32-byte shared_transforms digest, one byte (1 when a record's transform is
 * bundle-owned, else 0), the request's 32-byte bundle_module digest only when
 * that byte is 1, and the rank's plan document with an empty component list.
 * It covers the checkpoint's headers and the code that produces the bytes,
 * not tensor payloads.
 *
 * Both documents are little-endian: an 8-byte magic, a u32 layout (3), then
 * the fields below. Readers refuse any other layout before reading a field;
 * layout 1 predates the request's image_role, layout 2 its implementation.
 * The host states the same layout in the sink, which the bundle checks
 * before decoding the request. "str" and "bytes" are a u64 length and that many bytes
 * (names are UTF-8, 1 to 4096 bytes); "list" is a u32 count and the entries.
 * Named lists are sorted by name without duplicates.
 *
 *   request "CHOPWREQ": bytes image_role (UTF-8; empty asks for the
 *     bundle's primary image, for a package without rank images), u32 ranks,
 *     u32 rank, 32-byte checkpoint identity (SHA-256 over each safetensors
 *     file's name, length and header, in name order), 32-byte
 *     shared_transforms (SHA-256 of every source on the host's byte path),
 *     32-byte bundle_module (SHA-256 of the bundle's native module, zero
 *     without one); list files {str name, u64 byte_length,
 *     u64 data_offset, bytes header} (the checkpoint directory's safetensors
 *     files and those of its immediate subdirectories, named "dir/file");
 *     list documents {str name, bytes} (the JSON files and other small
 *     files in the same places, not covered by the identity); list
 *     components {str role, bytes} (the package's chop.bundle-metadata.
 *     components); list host_views {str id}; then, only when the host
 *     measured a census (below), list census {u32 rank, 32-byte SHA-256 of
 *     that rank's census document as the bundle emitted it, list counts
 *     {u64, u64, u64}} sorted by rank without duplicates. A request without
 *     a census ends after host_views.
 *
 *   plan "CHOPWPLN": u32 ranks, u32 rank, str image_role, u64 alignment,
 *     u64 arena_bytes; list views {str id, u64 alignment, u32 placement
 *     (0 device, 1 host, 2 mapped, 3 mapped and prewarmed), u64 offset,
 *     u64 byte_length}; list
 *     components {str role, bytes}; list records {u64 arena_offset, u64 byte_length,
 *     u32 transform, bytes params, list sources {u32 file, str tensor},
 *     u32 axis (UINT32_MAX for the whole tensor), list ranges {u64 start,
 *     u64 end}, list stored_shape {u64}}. Records are in increasing arena
 *     order without overlap; bytes no record covers are zero.
 *
 * Transform identifiers are kernel-core's closed set (weight_transform):
 * 1 copy (no parameters, or u32 payload 1: quantization scales, each checked
 * finite with a clear sign bit), 2 BF16-to-F32 widening, 3 MMA fragments
 * (u32 format: 1 BF16 16x16, 2 FP8 E4M3 16x32), 4 scale tiles (u32 tile
 * rows, u32 K blocks per tile), 5 a prepared precision operand section (JSON
 * parameters), 6 MXFP4 MoE fragments of one expert matrix (u32 section:
 * 1 codes, 2 scales), 7 F32 to BF16 narrowing (nearest, ties to even),
 * 8 inline bytes (the parameters, 1 byte to 64 KiB, with no sources),
 * 9 swap the last two axes (no parameters), 10 packed E4M3 folds in layout
 * mxfp8-fold-packed-8x128-v1 (no parameters, stored N divisible by 8 and K
 * divisible by 64, leading dimensions are groups). The
 * host refuses an E4M3 NaN code in the gathering transforms (1 to 4) when
 * the source is F8_E4M3 or the fragments are FP8, and in a prepared operand's
 * FP8 codes (5), packed fold codes (10), and an MXFP4 MoE scale above 252
 * (6). Swapped axes, inline
 * and bundle transforms store their bytes unchecked. Identifiers from
 * 0x80000000 belong to the bundle: the host calls chop_bundle_weight_transform_v1
 * with the record's whole source tensors and a part of its output. A bundle
 * uses one only for a layout the closed set cannot express.
 */
#define CHOP_WEIGHT_PLAN_CAPABILITY_V1 "chop.weight-plan.v1"
#define CHOP_WEIGHT_PLAN_LAYOUT_V1 3u

typedef chop_bundle_status_v1 (*chop_bundle_weight_plan_emit_fn_v1)(
    void *context, chop_bundle_byte_view_v1 bytes);

typedef struct chop_bundle_weight_plan_sink_v1 {
  uint32_t struct_size; /* exact */
  uint32_t layout;      /* CHOP_WEIGHT_PLAN_LAYOUT_V1 */
  void *context;
  chop_bundle_weight_plan_emit_fn_v1 emit;
  chop_bundle_weight_plan_emit_fn_v1 emit_error;
} chop_bundle_weight_plan_sink_v1;

typedef chop_bundle_status_v1 (*chop_bundle_weight_plan_fn_v1)(
    chop_bundle_byte_view_v1 request, chop_bundle_weight_plan_sink_v1 sink);

chop_bundle_status_v1 chop_bundle_weight_plan_v1(
    chop_bundle_byte_view_v1 request, chop_bundle_weight_plan_sink_v1 sink);

/* One bundle transform call: fill output with record bytes
 * [output_offset, output_offset + output.length). The call may run on any
 * host thread, concurrently with other calls. */
typedef struct chop_bundle_weight_transform_call_v1 {
  uint32_t struct_size; /* exact */
  uint32_t transform;
  chop_bundle_byte_view_v1 params;
  const chop_bundle_byte_view_v1 *sources;
  uint64_t source_count;
  uint64_t output_offset;
  chop_bundle_mut_byte_view_v1 output;
} chop_bundle_weight_transform_call_v1;

typedef chop_bundle_status_v1 (*chop_bundle_weight_transform_fn_v1)(
    const chop_bundle_weight_transform_call_v1 *call);

chop_bundle_status_v1 chop_bundle_weight_transform_v1(
    const chop_bundle_weight_transform_call_v1 *call);

/*
 * Weight census (optional export under chop.weight-plan.v1).
 *
 * A bundle whose plan sizes records from tensor payloads (compressed scale
 * planes, for example) exports chop_bundle_weight_census_v1, with the plan
 * export's signature. Before planning an image, the host calls it once per
 * rank with that rank's request (without a census) and the same sink
 * contract; the bundle answers with a census document, or an empty document
 * to measure nothing. The host produces each item's record on the CPU as it
 * would a plan record's, stores it nowhere, and measures it. It then asks for
 * every rank's plan with every rank's counts in the request's census list,
 * each with the SHA-256 of the census document it answers; a bundle refuses
 * counts whose digest is not its own census's. A host that skips the census
 * sends requests without one, and the bundle plans without it.
 *
 *   census "CHOPWCEN" (layout 3): u32 ranks, u32 rank; list items {u32 kind,
 *     u64 rows, u64 groups, then a plan record's fields after its
 *     arena_offset: u64 byte_length, u32 transform, bytes params, list
 *     sources {u32 file, str tensor}, u32 axis, list ranges {u64 start,
 *     u64 end}, list stored_shape {u64}}. byte_length is rows * groups.
 *
 * Kind 1 (scale-row-offset-v1) reads the record's bytes as a [rows, groups]
 * matrix of 1-byte scales and counts, for offset widths 1, 2 and 4 in that
 * order, the scales outside each row's best window (kernel-core's
 * precision::scale_offset::census): the request's counts entry for the item.
 */
#define CHOP_WEIGHT_CENSUS_SCALE_ROW_OFFSET_V1 1u

chop_bundle_status_v1 chop_bundle_weight_census_v1(
    chop_bundle_byte_view_v1 request, chop_bundle_weight_plan_sink_v1 sink);

/*
 * Chat frontend (capability chop.frontend.v1).
 *
 * A served bundle owns its chat format and exports chop_bundle_frontend_api_v1;
 * serve refuses a package that does not require chop.frontend.v1 or a module
 * without the export. The loader resolves the symbol as optional and calls it
 * only when the host needs the chat frontend, so a module loaded only for its
 * model program need not export it. The export returns a static table. The
 * host keeps HTTP, stop handling, UTF-8, limits and grammar masks; the bundle
 * renders a chat request into prompt segments, answers a request's tool-call
 * constraint, and parses the request's output text incrementally. Every
 * request, answer and refusal is a byte document. This section fixes how
 * documents cross the boundary, not what they contain; a constraint answer is
 * a runtime-grammar wire document.
 *
 * The table's struct_size is exactly sizeof(chop_bundle_frontend_table_v1)
 * and its layout is CHOP_FRONTEND_LAYOUT_V1; the host refuses another size
 * or layout, and a table with a null entry.
 *
 *   open reads the authenticated frontend component's bytes, which the host
 *     lends for the call, answers with its open document through the sink's
 *     emit, and then writes an opaque bundle-owned frontend to *frontend. It
 *     may refuse through emit_error. When the host refuses the answer, open
 *     returns that status and writes no frontend.
 *   close releases a frontend. The host calls it exactly once per opened
 *     frontend, after closing every parser opened on it. While a parser
 *     opened on it is open, close returns CHOP_BUNDLE_STATUS_V1_INVALID_STATE
 *     and the frontend stays open. Closing a frontend twice, or using a
 *     frontend or parser after its close succeeded, is undefined behavior,
 *     as with free(): the bundle need not detect it.
 *   render answers a chat request document with the rendered prompt
 *     document through the sink's emit.
 *   constraint answers a request document with its constraint document
 *     through emit; an empty document means no constraint.
 *   parser_open starts parsing one request's output and writes an opaque
 *     bundle-owned parser to *parser. It may refuse through emit_error.
 *   parser_push hands the parser a push document: how many bytes of earlier
 *     pushes to keep, and the bytes that follow them. The host withholds a
 *     possible client stop and an incomplete character the next token could
 *     complete. It answers with an event document through emit, whose
 *     reasoning, content and tool-call ranges are raw-byte offsets into the
 *     whole accumulated completion. CHOP_FRONTEND_PUSH_FLAG_V1_FINAL marks
 *     the last push of the request, which may carry no new bytes. Other flag
 *     bits are refused. After a final push, a refusal or a failure, the
 *     parser takes no further push; the host only closes it.
 *   parser_close releases a parser. The host calls it exactly once per
 *     opened parser; a second close is undefined behavior.
 *
 * Sinks follow the weight plan's: struct_size is exact, layout is
 * CHOP_FRONTEND_LAYOUT_V1 (the bundle refuses another through emit_error
 * before reading the request), and both callbacks are set. A successful
 * open, render, constraint or push calls emit exactly once; parser_open
 * answers through its output pointer and never calls emit. A refusal calls
 * emit_error at most once with a refusal document and returns
 * CHOP_BUNDLE_STATUS_V1_BUNDLE_ERROR. The host copies either document
 * before the callback returns. It sets a byte limit for each call's answer,
 * CHOP_FRONTEND_OPENED_BYTES_MAX_V1 for open's,
 * and takes a refusal of at most CHOP_FRONTEND_REFUSAL_BYTES_MAX_V1 bytes;
 * a longer document is refused unread with
 * CHOP_BUNDLE_STATUS_V1_INVALID_ARGUMENT. A sink call the host refuses (a
 * second document of either kind, a malformed view, or a document past its
 * limit) fails the whole call
 * whatever the entry does next or returns, and the host refuses every later
 * sink call of that call.
 *
 * A frontend or parser pointer stays valid until its close; a failed open
 * or parser_open writes none. Entries run on the CPU and use no host
 * service. render, constraint and parser_open may run concurrently on one
 * frontend from any thread, each with the others and with itself, and with
 * pushes and closes of parsers already open on it; only the frontend's own
 * close excludes every other call on it. A parser is used by one thread at a
 * time, though not always the same one, and different parsers run
 * concurrently. The export catches a panic in any entry, including the
 * closes, and returns CHOP_BUNDLE_STATUS_V1_PANIC; the host fails that call
 * as the bundle's error and does not retry a close. Null or misaligned
 * pointers and malformed views are CHOP_BUNDLE_STATUS_V1_INVALID_ARGUMENT.
 */
#define CHOP_FRONTEND_CAPABILITY_V1 "chop.frontend.v1"
#define CHOP_FRONTEND_LAYOUT_V1 1u
#define CHOP_FRONTEND_PUSH_FLAG_V1_FINAL UINT32_C(0x1)
#define CHOP_FRONTEND_PUSH_FLAGS_V1_KNOWN UINT32_C(0x1)
#define CHOP_FRONTEND_REFUSAL_BYTES_MAX_V1 UINT64_C(1048576)
#define CHOP_FRONTEND_OPENED_BYTES_MAX_V1 UINT64_C(65536)

typedef struct chop_bundle_frontend_v1 chop_bundle_frontend_v1;
typedef struct chop_bundle_frontend_parser_v1 chop_bundle_frontend_parser_v1;

typedef chop_bundle_status_v1 (*chop_bundle_frontend_emit_fn_v1)(
    void *context, chop_bundle_byte_view_v1 document);

typedef struct chop_bundle_frontend_sink_v1 {
  uint32_t struct_size; /* exact */
  uint32_t layout;      /* CHOP_FRONTEND_LAYOUT_V1 */
  void *context;
  chop_bundle_frontend_emit_fn_v1 emit;
  chop_bundle_frontend_emit_fn_v1 emit_error;
} chop_bundle_frontend_sink_v1;

typedef chop_bundle_status_v1 (*chop_bundle_frontend_open_fn_v1)(
    chop_bundle_byte_view_v1 component, chop_bundle_frontend_sink_v1 sink,
    chop_bundle_frontend_v1 **frontend);
typedef chop_bundle_status_v1 (*chop_bundle_frontend_close_fn_v1)(
    chop_bundle_frontend_v1 *frontend);
typedef chop_bundle_status_v1 (*chop_bundle_frontend_document_fn_v1)(
    const chop_bundle_frontend_v1 *frontend, chop_bundle_byte_view_v1 request,
    chop_bundle_frontend_sink_v1 sink);
typedef chop_bundle_status_v1 (*chop_bundle_frontend_parser_open_fn_v1)(
    const chop_bundle_frontend_v1 *frontend, chop_bundle_byte_view_v1 request,
    chop_bundle_frontend_sink_v1 sink, chop_bundle_frontend_parser_v1 **parser);
typedef chop_bundle_status_v1 (*chop_bundle_frontend_parser_push_fn_v1)(
    chop_bundle_frontend_parser_v1 *parser, chop_bundle_byte_view_v1 text,
    uint32_t flags, chop_bundle_frontend_sink_v1 sink);
typedef chop_bundle_status_v1 (*chop_bundle_frontend_parser_close_fn_v1)(
    chop_bundle_frontend_parser_v1 *parser);

typedef struct chop_bundle_frontend_table_v1 {
  uint32_t struct_size; /* exact */
  uint32_t layout;      /* CHOP_FRONTEND_LAYOUT_V1 */
  chop_bundle_frontend_open_fn_v1 open;
  chop_bundle_frontend_close_fn_v1 close;
  chop_bundle_frontend_document_fn_v1 render;
  chop_bundle_frontend_document_fn_v1 constraint;
  chop_bundle_frontend_parser_open_fn_v1 parser_open;
  chop_bundle_frontend_parser_push_fn_v1 parser_push;
  chop_bundle_frontend_parser_close_fn_v1 parser_close;
} chop_bundle_frontend_table_v1;

typedef const chop_bundle_frontend_table_v1 *(*chop_bundle_frontend_api_fn_v1)(void);

const chop_bundle_frontend_table_v1 *chop_bundle_frontend_api_v1(void);

/* Optional prefill contract. Existing entry tables and batch records keep
 * their bytes. An older host ignores this export and calls legacy execute,
 * which must retain full-row behavior. Zero chunk_tokens disables the policy.
 * An enabled host schedules eligible unscored text prefills to the next
 * absolute chunk multiple or prompt end and permits prefill captures there.
 * Each such end retains complete state; decode/verify boundaries stay complete.
 * Scored and media requests may use legacy complete-row execution.
 *
 * execute_prefill_v1 receives a separate little-endian envelope:
 * 16 bytes "chop-prefill-v1\0", U32 original batch byte count, U32 boundary
 * count, the unchanged V1-V5 batch bytes, then 32 bytes per boundary:
 * U64 request_id, U64 prompt_tokens, U64 chunk_end, U32 chunk_tokens,
 * U32 reserved (zero). Boundary IDs are unique prefill items; chunk_end must
 * equal the append end and min(next absolute chunk multiple, prompt_tokens).
 * Missing boundaries select full-row execution. Both callbacks use the
 * ordinary instance lifecycle; policy queries leave request state unchanged. */
typedef struct chop_bundle_prefill_policy_v1 {
  uint32_t struct_size;
  chop_bundle_status_v1 status;
  uint32_t chunk_tokens;
  uint32_t reserved;
} chop_bundle_prefill_policy_v1;

typedef chop_bundle_prefill_policy_v1 (*chop_bundle_describe_prefill_fn_v1)(
    chop_bundle_handle_v1 handle);

typedef struct chop_bundle_prefill_table_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  chop_bundle_describe_prefill_fn_v1 describe_prefill_v1;
  chop_bundle_execute_fn_v1 execute_prefill_v1;
} chop_bundle_prefill_table_v1;

const chop_bundle_prefill_table_v1 *chop_bundle_prefill_api_v1(void);

#ifdef __cplusplus
}
#endif

#endif
