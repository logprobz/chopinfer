# NVDEC header provenance

`dynlink_cuviddec.h` and `dynlink_nvcuvid.h` are copied byte for byte from
FFmpeg's `nv-codec-headers`, the dynamic-loading edition of the NVIDIA Video
Codec SDK decode headers. Each file carries NVIDIA's MIT license notice.

- source: `https://github.com/FFmpeg/nv-codec-headers`
- tag: `n13.0.19.1`
- commit: `88fee5c37318c991a8762d423530f91681e32e3a`
- source paths: `include/ffnvcodec/dynlink_cuviddec.h`,
  `include/ffnvcodec/dynlink_nvcuvid.h`
- `dynlink_cuviddec.h` SHA-256:
  `1244165ea5521d4be865995eb1c49b8910f755d0ca35004e1c4010ef7f612143`
- `dynlink_nvcuvid.h` SHA-256:
  `2141dade75f3178520092dce2c02df9b47c4b19036b56bb9316bd83a07bf16f1`

The headers declare function pointer types (`tcuvid*`) rather than linked
functions, so nothing links `libnvcuvid`. `csrc/nvdec.c` includes them after
the pinned CUDA header, which supplies the CUDA types they use, and resolves
every entry point with `dlsym` from a run-time `dlopen` of `libnvcuvid.so.1`.

The normal Linux build runs `verify-headers.sh`, which checks both hashes
before compiling the helper. To update, copy both files from a newer tag,
update the tag, commit and hashes here and in the script, and rebuild.
