#include "../common/ptx_matrix.cuh"

extern "C" __global__ void common_matrix(unsigned* out) {
  __shared__ alignas(128) unsigned tile[256];
  unsigned a[1], b[2], c[4], p[4];
  unsigned sum = 0;
  chop::ptx::ldmatrix_b16<1>(a, tile);
  sum += a[0];
  chop::ptx::ldmatrix_b16<1, true>(a, tile);
  sum += a[0];
  chop::ptx::ldmatrix_b16<2>(b, tile);
  sum += b[0] + b[1];
  chop::ptx::ldmatrix_b16<2, true>(b, tile);
  sum += b[0] + b[1];
  chop::ptx::ldmatrix_b16<4>(c, tile);
  sum += c[0] + c[1] + c[2] + c[3];
  chop::ptx::ldmatrix_b16<4, true>(c, tile);
  sum += c[0] + c[1] + c[2] + c[3];
  chop::ptx::ldmatrix_b8x16<4, chop::ptx::PackedTile::b4x16_p64>(p, tile);
  sum += p[0] + p[1] + p[2] + p[3];
  chop::ptx::ldmatrix_b8x16<4, chop::ptx::PackedTile::b6x16_p32>(p, tile);
  sum += p[0] + p[1] + p[2] + p[3];
  chop::ptx::ldmatrix_b8x16_trans<chop::ptx::PackedTile::b4x16_p64>(p, tile);
  sum += p[0] + p[1] + p[2] + p[3];
  chop::ptx::ldmatrix_b8x16_trans<chop::ptx::PackedTile::b6x16_p32>(p, tile);
  sum += p[0] + p[1] + p[2] + p[3];
  chop::ptx::stmatrix<1>(tile, a);
  chop::ptx::stmatrix<2>(tile, b);
  chop::ptx::stmatrix<4>(tile, c);
  chop::ptx::stmatrix<1, true>(tile, a);
  chop::ptx::stmatrix<2, true>(tile, b);
  chop::ptx::stmatrix<4, true>(tile, p);
  chop::ptx::stmatrix_trans_b16<1>(tile, a);
  chop::ptx::stmatrix_trans_b16<2>(tile, b);
  chop::ptx::stmatrix_trans_b16<4>(tile, c);
  out[threadIdx.x] = sum;
}
