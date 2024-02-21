#ifndef _ALIGNED_TSETLIN_STATE_HPP_
#define _ALIGNED_TSETLIN_STATE_HPP_

#include <stdlib.h>
#include <cmath>
#include <algorithm>



#include <gt_common.hpp>


namespace green_tsetlin
{
    class AlignedTsetlinState
    {
        public:
            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;
            int num_class_weights_mem = 0;

            int num_literals = 0;
            int num_literals_mem = 0;
            int num_reminder = 0;

            int8_t* clauses = nullptr;
            ClauseOutputUint* clause_outputs = nullptr;
            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;

            uint32_t* literal_counts = nullptr;
            uint32_t literal_budget = 0xFFFF;

            int8_t* reminder_mask = nullptr;
            int8_t gtcmp_for_s = 0;

            std::default_random_engine rng;


            inline void set_s(double s_param)
            {
                s = s_param;
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

            inline void get_clause_state(int8_t* dst, int clause_offset)
            {
                for(int i = 0; i < num_clauses; i++)
                {
                    const int src_o = i * num_literals_mem * 2;
                    const int dst_o = (i+clause_offset) * num_literals * 2;

                    memcpy(&dst[dst_o],
                           &clauses[src_o], num_literals);

                    memcpy(&dst[dst_o + num_literals],
                           &clauses[src_o + num_literals_mem], num_literals);
                }
            }

            inline void set_clause_state(int8_t* src, int clause_offset)
            {

                for(int i = 0; i < num_clauses; i++)
                {                    
                    const int src_o = (i+clause_offset) * (num_literals * 2);
                    const int dst_o = i * (num_literals_mem * 2);

                    memcpy(&clauses[dst_o],
                           &src[src_o], num_literals);

                    memcpy(&clauses[dst_o + num_literals_mem],
                           &src[src_o + num_literals], num_literals);
                }
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
}; // namespace green_tsetlin




#endif // #define _ALIGNED_TSETLIN_STATE_HPP_