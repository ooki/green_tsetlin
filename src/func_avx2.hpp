#ifndef _FUNC_AVX2_H_
#define _FUNC_AVX2_H_

#include <immintrin.h> // intrics

namespace green_tsetlin
{

    // to be used for counting active ta's
    inline int _avx2_mask_count(const __m256i in)
    {
        __m256i _zeros = _mm256_set1_epi8(0);
        __m256i _on = _mm256_cmpgt_epi8(in, _zeros);
        int bits = _mm256_movemask_epi8(_on);
        return __builtin_popcount(bits); // TODO: wrap into save version (or force std::popcount() c++20)
    }


    

}; // namespace green_tsetlin










#endif // #ifndef _FUNC_AVX2_H_