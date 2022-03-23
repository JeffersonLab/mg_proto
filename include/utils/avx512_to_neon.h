#ifndef __AVX512_TO_NEON_H_
#define __AVX512_TO_NEON_H_
#include <arm_neon.h>

#pragma push_macro("FORCE_INLINE")
#define FORCE_INLINE static inline __attribute__((always_inline))

#define _MM_SHUFFLE(fp3, fp2, fp1, fp0)                                        \
  (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

typedef float32x4_t __m128;

typedef struct {
  float32x4_t vf32[4];
} __m512;

FORCE_INLINE __m512 _mm512_load_ps(void const *mem_addr) {
  __m512 res;

  return res;
}

FORCE_INLINE void _mm512_store_ps(void *mem_addr, __m512 a) {}

FORCE_INLINE __m512 _mm512_set1_ps(float a) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512
_mm512_set_ps(float e15, float e14, float e13, float e12, float e11, float e10,
              float e9, float e8, float e7, float e6,
              float e5, float e4, float e3, float e2, float e1, float e0) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512 _mm512_setzero_ps() {}

FORCE_INLINE __m512 _mm512_shuffle_ps(__m512 a, __m512 b, const int imm8) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512 _mm512_mul_ps(__m512 a, __m512 b) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512 _mm512_fmaddsub_ps(__m512 a, __m512 b, __m512 c) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512 _mm512_fmsubadd_ps(__m512 a, __m512 b, __m512 c) {
  __m512 res;

  return res;
}

FORCE_INLINE __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
  __m512 res;

  return res;
}

#endif
