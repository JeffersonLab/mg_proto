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

  res.vf32[0] = vld1q_f32((const float*)mem_addr);
  res.vf32[1] = vld1q_f32((const float*)mem_addr + 4);
  res.vf32[2] = vld1q_f32((const float*)mem_addr + 8);
  res.vf32[3] = vld1q_f32((const float*)mem_addr + 12);

  return res;
}

FORCE_INLINE void _mm512_store_ps(void *mem_addr, __m512 a) {
    vst1q_f32((float32_t*)mem_addr, a.vf32[0]);
    vst1q_f32((float32_t*)mem_addr + 4, a.vf32[1]);
    vst1q_f32((float32_t*)mem_addr + 8, a.vf32[2]);
    vst1q_f32((float32_t*)mem_addr + 12, a.vf32[3]);
}

FORCE_INLINE __m512 _mm512_set1_ps(float a) {
  __m512 res;

  res.vf32[0] = vdupq_n_f32(a);
  res.vf32[1] = vdupq_n_f32(a);
  res.vf32[2] = vdupq_n_f32(a);
  res.vf32[3] = vdupq_n_f32(a);

  return res;
}

FORCE_INLINE __m512
_mm512_set_ps(float e15, float e14, float e13, float e12, float e11, float e10,
              float e9, float e8, float e7, float e6,
              float e5, float e4, float e3, float e2, float e1, float e0) {
  __m512 res;
  
  float temp0[4] = {e0, e1, e2, e3};
  res.vf32[0] = vld1q_f32(temp0);

  float temp1[4] = {e4, e5, e6, e7};
  res.vf32[1] = vld1q_f32(temp1);

  float temp2[4] = {e8, e9, e10, e11};
  res.vf32[2] = vld1q_f32(temp2);

  float temp3[4] = {e12, e13, e14, e15};
  res.vf32[4] = vld1q_f32(temp3);

  return res;
}

FORCE_INLINE __m512 _mm512_setzero_ps() {
	__m512 res;

	res.vf32[0] = vdupq_n_f32(0.0);
	res.vf32[1] = vdupq_n_f32(0.0);
	res.vf32[2] = vdupq_n_f32(0.0);
	res.vf32[3] = vdupq_n_f32(0.0);

	return res;
}

/* 
* the imm8 should be immediate value, therefore use macro definition here,
* there maybe other nicer way to define the function
* reference: https://github.com/simd-everywhere/simde/blob/master/simde/x86/sse.h
*/
#define _mm512_shuffle_ps(a, b, imm8)                                          \
  (({                                                                          \
    __m512 res;                                                                \
    res.vf32[0] =                                                          \
        vmovq_n_f32(vgetq_lane_f32(a.vf32[0], (imm8) & (0x3)));            \
    res.vf32[0] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(a.vf32[0], ((imm8) >> 2) & 0x3),     \
                       res.vf32[0], 1);                                    \
    res.vf32[0] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[0], ((imm8) >> 4) & 0x3),     \
                       res.vf32[0], 2);                                    \
    res.vf32[0] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[0], ((imm8) >> 6) & 0x3),     \
                       res.vf32[0], 3);                                    \
                                                                               \
    res.vf32[1] =                                                          \
        vmovq_n_f32(vgetq_lane_f32(a.vf32[1], (imm8) & (0x3)));            \
    res.vf32[1] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(a.vf32[1], ((imm8) >> 2) & 0x3),     \
                       res.vf32[1], 1);                                    \
    res.vf32[1] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[1], ((imm8) >> 4) & 0x3),     \
                       res.vf32[1], 2);                                    \
    res.vf32[1] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[1], ((imm8) >> 6) & 0x3),     \
                       res.vf32[1], 3);                                    \
                                                                               \
    res.vf32[2] =                                                          \
        vmovq_n_f32(vgetq_lane_f32(a.vf32[2], (imm8) & (0x3)));            \
    res.vf32[2] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(a.vf32[2], ((imm8) >> 2) & 0x3),     \
                       res.vf32[2], 1);                                    \
    res.vf32[2] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[2], ((imm8) >> 4) & 0x3),     \
                       res.vf32[2], 2);                                    \
    res.vf32[2] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[2], ((imm8) >> 6) & 0x3),     \
                       res.vf32[2], 3);                                    \
                                                                               \
    res.vf32[3] =                                                          \
        vmovq_n_f32(vgetq_lane_f32(a.vf32[3], (imm8) & (0x3)));            \
    res.vf32[3] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(a.vf32[3], ((imm8) >> 2) & 0x3),     \
                       res.vf32[3], 1);                                    \
    res.vf32[3] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[3], ((imm8) >> 4) & 0x3),     \
                       res.vf32[3], 2);                                    \
    res.vf32[3] =                                                          \
        vsetq_lane_f32(vgetq_lane_f32(b.vf32[3], ((imm8) >> 6) & 0x3),     \
                       res.vf32[3], 3);                                    \
    res;                                                                       \
  }))

FORCE_INLINE __m512 _mm512_mul_ps(__m512 a, __m512 b) {
  __m512 res;

  res.vf32[0] = vmulq_f32(a.vf32[0], b.vf32[0]);
  res.vf32[1] = vmulq_f32(a.vf32[1], b.vf32[1]);
  res.vf32[2] = vmulq_f32(a.vf32[2], b.vf32[2]);
  res.vf32[3] = vmulq_f32(a.vf32[3], b.vf32[3]);

  return res;
}

FORCE_INLINE __m512 _mm512_addsub_ps(__m512 a, __m512 b)
{
	__m512 res;
	__m128 temp1, temp2;
	temp1 = vsubq_f32(a.vf32[0], b.vf32[0]);
	temp2 = vaddq_f32(a.vf32[0], b.vf32[0]);
	res.vf32[0] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vsubq_f32(a.vf32[1], b.vf32[1]);
	temp2 = vaddq_f32(a.vf32[1], b.vf32[1]);
	res.vf32[1] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vsubq_f32(a.vf32[2], b.vf32[2]);
	temp2 = vaddq_f32(a.vf32[2], b.vf32[2]);
	res.vf32[2] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vsubq_f32(a.vf32[3], b.vf32[3]);
	temp2 = vaddq_f32(a.vf32[3], b.vf32[3]);
	res.vf32[3] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);

	return res;
}

FORCE_INLINE __m512 _mm512_fmaddsub_ps(__m512 a, __m512 b, __m512 c) {
  return _mm512_addsub_ps(_mm512_mul_ps(a, b), c);
}

FORCE_INLINE __m512 _mm512_subadd_ps(__m512 a, __m512 b)
{
	__m512 res;
	__m128 temp1, temp2;
	temp1 = vaddq_f32(a.vf32[0], b.vf32[0]);
	temp2 = vsubq_f32(a.vf32[0], b.vf32[0]);
	res.vf32[0] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vaddq_f32(a.vf32[1], b.vf32[1]);
	temp2 = vsubq_f32(a.vf32[1], b.vf32[1]);
	res.vf32[1] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vaddq_f32(a.vf32[2], b.vf32[2]);
	temp2 = vsubq_f32(a.vf32[2], b.vf32[2]);
	res.vf32[2] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);
	
	temp1 = vaddq_f32(a.vf32[3], b.vf32[3]);
	temp2 = vsubq_f32(a.vf32[3], b.vf32[3]);
	res.vf32[3] = vtrn2q_f32(vreinterpretq_f32_s32(vrev64q_s32(vreinterpretq_s32_f32(temp1))), temp2);

	return res;
}

FORCE_INLINE __m512 _mm512_fmsubadd_ps(__m512 a, __m512 b, __m512 c) {
  return _mm512_subadd_ps(_mm512_mul_ps(a, b), c);
}

FORCE_INLINE __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
  __m512 res;

  res.vf32[0] = vfmaq_f32(c.vf32[0], a.vf32[0], b.vf32[0]);
  res.vf32[1] = vfmaq_f32(c.vf32[1], a.vf32[1], b.vf32[1]);
  res.vf32[2] = vfmaq_f32(c.vf32[2], a.vf32[2], b.vf32[2]);
  res.vf32[3] = vfmaq_f32(c.vf32[3], a.vf32[3], b.vf32[3]);
  return res;
}

#endif
