#ifndef _SPARSE_TSETLIN_STATE_HPP_
#define _SPARSE_TSETLIN_STATE_HPP_

#include <vector>
#include <map>
#include <gt_common.hpp>

namespace green_tsetlin
{
    typedef typename std::vector<uint32_t> SparseLiterals;
    typedef typename std::vector<uint32_t> SparseClause;
    typedef typename std::vector<int8_t> SparseClauseStates;




    class SparseTsetlinState
    {
        public:
            double s = -42.0;
            double s_inv = 1.0 / s;
            double s_min1_inv = (s - 1.0) / s;

            int num_clauses = 0;
            int num_classes = 0;        
            int num_class_weights_mem = 0;

            int lower_ta_threshold = 0;

            int num_literals = 0;
            size_t active_literals_size = 0;
            SparseLiterals al_replace_index;

            // want this as percentage or just as number of states in clauses?
            size_t clause_size = 0;

            uint32_t* literal_counts = nullptr;
            uint32_t literal_budget = 0xFFFF;

            // support for convolution (leave it at 1 for now)
            int num_patches_per_example = 1;

            std::default_random_engine rng;
            Wyhash64                   fast_rng; // can only generate a random [0,1] float 

            std::vector<SparseClause> clauses;
            std::vector<SparseClauseStates> clause_states;
            
            std::vector<SparseLiterals> active_literals;

            ClauseOutputUint* clause_outputs = nullptr;
            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;

            
            inline void set_s(double s_param)
            {
                s = s_param;
                s_inv = 1.0 / s;
                s_min1_inv = (s - 1.0) / s;
            }
            inline double get_s() const { return s; }

            inline int get_number_of_patches_per_example() const { return num_patches_per_example; }
            inline void set_number_of_patches_per_example(int a_num_patches_per_example) { num_patches_per_example = a_num_patches_per_example; }

            inline WeightInt* get_class_votes() const
            {
                
                return class_votes;
            }

            inline void get_clause_state(int8_t* dst, int clause_offset)
            {

            }

            inline void set_clause_state(int8_t* src, int clause_offset)
            {

            }

            inline void set_clause_weights(WeightInt* src, int clause_offset)
            {
                for(int i = 0; i < num_clauses; i++)
                {                    
                    const int src_o = (i+clause_offset) * num_class_weights_mem;
                    const int dst_o = i * num_classes;

                    memcpy(&clause_weights[dst_o],
                           &src[src_o], num_classes * sizeof(WeightInt));
                }
            }

            inline void get_clause_weights(WeightInt* dst, int clause_offset)
            {
                for(int i = 0; i < num_clauses; i++)
                {
                    const int src_o = i * num_classes;
                    const int dst_o = (i+clause_offset) * num_class_weights_mem;

                    memcpy(&dst[dst_o],
                           &clause_weights[src_o], num_classes * sizeof(WeightInt));
                }
            }
    };  
    
}; // namespace green_tsetlin





#endif // #ifndef _SPARSE_TSETLIN_STATE_HPP_