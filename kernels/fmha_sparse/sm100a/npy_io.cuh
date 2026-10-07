// npy_io.cuh -- minimal .npy v1.0/v2.0 reader for the unified VSA benchmark.
//
// Reads flat C-order arrays written by block_sparse_bf16_gen_inputs.py (uint16
// bf16-bits for Q/K/V, int32 for the top-k index). No dtype/shape validation
// beyond an element-count check by the caller -- we own both ends of the format.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Load the entire data payload of a .npy file as a flat vector<T>. Exits on
// error. Verifies the byte count is a whole multiple of sizeof(T).
template <typename T>
static std::vector<T> npy_load_vec(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "npy_load: cannot open %s\n", path.c_str()); exit(1); }
  unsigned char magic[8];
  if (fread(magic, 1, 8, f) != 8 || magic[0] != 0x93 ||
      magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' ||
      magic[4] != 'P' || magic[5] != 'Y') {
    fprintf(stderr, "npy_load: bad magic in %s\n", path.c_str()); exit(1);
  }
  const int major = magic[6];
  uint32_t hlen = 0;
  if (major >= 2) { if (fread(&hlen, 4, 1, f) != 1) { exit(1); } }
  else            { uint16_t h16 = 0; if (fread(&h16, 2, 1, f) != 1) { exit(1); } hlen = h16; }
  if (fseek(f, (long)hlen, SEEK_CUR) != 0) { exit(1); }   // skip ASCII dict header
  const long data_off = ftell(f);
  fseek(f, 0, SEEK_END);
  const long fsz = ftell(f);
  fseek(f, data_off, SEEK_SET);
  const long bytes = fsz - data_off;
  if (bytes < 0 || (size_t)bytes % sizeof(T) != 0) {
    fprintf(stderr, "npy_load: size mismatch in %s (%ld bytes)\n", path.c_str(), bytes); exit(1);
  }
  std::vector<T> out((size_t)bytes / sizeof(T));
  if (fread(out.data(), sizeof(T), out.size(), f) != out.size()) {
    fprintf(stderr, "npy_load: short read in %s\n", path.c_str()); exit(1);
  }
  fclose(f);
  return out;
}
