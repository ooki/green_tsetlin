#ifndef _SPARSE_TSETLIN_STATE_HPP_
#define _SPARSE_TSETLIN_STATE_HPP_

#include <vector>
#include <gt_common.hpp>

namespace green_tsetlin
{
    typedef typename std::vector<uint32_t> SparseLiterals;

    class SparseTsetlinState
    {
        public:
            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;            
            int num_literals = 0;

            uint32_t literal_budget = 0xFFFF;

            // support for convolution (leave it at 1 for now)
            int num_patches_per_example = 1;

            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;
            
            std::default_random_engine rng;
            Wyhash64                   fast_rng; // can only generate a random [0,1] float 

            

            
            inline void set_s(double s_param)
            {
                s = s_param;
            }
            inline double get_s() const { return s; }

            inline int get_number_of_patches_per_example() const { return num_patches_per_example; }
            inline void set_number_of_patches_per_example(int a_num_patches_per_example) { num_patches_per_example = a_num_patches_per_example; }

            inline WeightInt* get_class_votes() const
            {
                return class_votes;
            }

            inline void get_clause_state()
            {
            }

            inline void set_clause_state()
            {
            }

            inline void set_clause_weights()
            {
            }

            inline void get_clause_weights()
            {
            }
    };  

    
}; // namespace green_tsetlin





#endif // #ifndef _SPARSE_TSETLIN_STATE_HPP_