#ifndef _RANDOM_GENERATOR_HPP_
#define _RANDOM_GENERATOR_HPP_



#include <cstdint>
#include <random>

// #ifdef USE_AVX2
//     #include <immintrin.h> // intrics
// #endif 

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
    
}; // namespace green_tsetlin






#endif // #ifndef _RANDOM_GENERATOR_HPP_

