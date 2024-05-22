#ifndef __BITWISE_HPP__
#define __BITWISE_HPP__

#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

#include <gt_common.hpp>
#include <random_generator.hpp>


namespace green_tsetlin
{

    template<int _vector_size, int _bits_per_state>
    class BitwiseState
    {
        public:
            constexpr const int const_vector_size = _vector_size;
            constexpr const int const_bits_per_state = _bits_per_state;

            int num_clauses = 0;
            int num_classes = 0;
            int num_class_weights_mem = 0;            
            WeightInt* class_votes = nullptr;
            
            int num_literals = 0;
            int num_literals_mem = 0;
            int num_reminder = 0;


            uint64_t* clauses = nullptr;


            int8_t gtcmp_for_s = 0;

            // rng's
            std::default_random_engine rng;
            Wyhash64                   fast_rng;

            #ifdef USE_AVX2
            XorShift128plus4G           avx2_rng;
            #endif 

            #ifdef USE_NEON
            Xoshiro128Plus  rng_neon;
            #endif 


        inline void set_s(double s_param)
        {
            s = s_param;
            s_inv = 1.0 / s;
            s_min1_inv = (s - 1.0) / s;

            double p = 1 / s;
            int32_t tmp = ((int32_t)(p * 255)) - 127;                
            tmp += 1; // size we use < to compare and not <=

            gtcmp_for_s = (int8_t)std::clamp(tmp, -127, 126);
        }
        inline double get_s() const { return s; }

        inline WeightInt* get_class_votes() const
        {
            return class_votes;
        }

        inline void set_clause_weights(WeightInt* src, int clause_offset)
        {
            //const size_t weight_mem = num_clauses * num_classes * sizeof(WeightInt);
            //memcpy(clause_weights, src, weight_mem);

            for(int i = 0; i < num_clauses; i++)
            {                    
                const int src_o = (i+clause_offset) * num_classes;
                const int dst_o = i * num_class_weights_mem;

                memcpy(&clause_weights[dst_o],
                        &src[src_o], num_classes * sizeof(WeightInt));
            }
        }

        inline void get_clause_weights(WeightInt* dst, int clause_offset)
        {
            for(int i = 0; i < num_clauses; i++)
            {
                const int src_o = i * num_class_weights_mem;
                const int dst_o = (i+clause_offset) * num_classes;

                memcpy(&dst[dst_o],
                        &clause_weights[src_o], num_classes * sizeof(WeightInt));
            }
        }


    };


};




#endif // __BITWISE_HPP__