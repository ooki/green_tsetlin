#ifndef _RANDOM_GENERATOR_HPP_
#define _RANDOM_GENERATOR_HPP_



#include <cstdint>
#include <random>

#ifdef USE_AVX2
    #include <immintrin.h> // intrics
#endif 

// #ifdef USE_NEON
//     #include <arm_neon.h>
// #endif 


namespace green_tsetlin
{
    class Wyhash64
    {
        public:

            Wyhash64()
            {
            }

            void seed(unsigned int start_seed)
            {
                seed_internal(start_seed);
                for(int i = 0; i < 32; i++)  // help it mix         
                    next_u();
            }

            double next_u()
            {
                wyhash64_x += 0x60bee2bee120fc15;
                __uint128_t tmp;
                tmp = (__uint128_t) wyhash64_x * 0xa3b195354a39b70d;
                uint64_t m1 = (tmp >> 64) ^ tmp;
                tmp = (__uint128_t)m1 * 0x1b03738712fad5c9;
                uint64_t m2 = ((tmp >> 64) ^ tmp);

                double r = 0x1.0p-64 * m2;
                return r;
            }
            


            uint64_t wyhash64_x;

        private:
            void seed_internal(unsigned int start_seed)
            {
                std::mt19937_64 seed_rng(start_seed);
                std::uniform_int_distribution<uint64_t> random_uint32(15, 4294967295);
                wyhash64_x = (random_uint32(seed_rng) << 32) | random_uint32(seed_rng);
            }
    };
    

#ifdef USE_AVX2    
    class XorShift128plus4G
    {
        public:
            XorShift128plus4G()
            {
            }

            void seed(unsigned int start_seed)
            {
                seed_internal(start_seed);
                for(int i = 0; i < 32; i++)  // help it mix         
                    next();
            }

            __m256i next()
            {
                __m256i s1 = part1;
                const __m256i s0 = part2;

                part1 = part2;
                s1 = _mm256_xor_si256(part2, _mm256_slli_epi64(part2, 23));
                part2 = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(s1, s0),_mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));

                return _mm256_add_epi64(part2, s0);
            }
            
            __m256i part1;
            __m256i part2;


        private:
            void seed_internal(unsigned int seed) // hack to get slightly better seeds that the standard '42' type seed.
            {
                const size_t num_bytes_per_part = 32;
                const size_t num_bytes = num_bytes_per_part * 2;

                std::default_random_engine seed_rng(seed);
                std::uniform_int_distribution random_byte(5, 251);

                uint8_t tmp_seed[num_bytes];
                for(size_t i = 0; i < num_bytes; ++i)
                {
                    tmp_seed[i] = random_byte(seed_rng);
                }

                part1 = _mm256_loadu_si256((__m256i const*)&tmp_seed[0]);
                part2 = _mm256_loadu_si256((__m256i const*)&tmp_seed[num_bytes_per_part]);

            }

    };
#endif // #ifdef USE_AVX2

}; // namespace green_tsetlin






#endif // #ifndef _RANDOM_GENERATOR_HPP_

