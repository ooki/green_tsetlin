#ifndef __FUNC_BITWISE_AVX2_HPP__
#define __FUNC_BITWISE_AVX2_HPP__

#include <iostream>

#include <bitwise_tsetlin_state.hpp>


namespace green_tsetlin
{
    template <typename _State, bool pad_class_weights,  bool do_literal_budget>
    class InitializeBitwiseAVX2
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {            
                if(seed == 0)
                    return false;

                state.avx2_rng.seed(seed);
                state.fast_rng.seed(seed);
                state.rng.seed(seed);                


                // allocate mem
                constexpr int bytes_per_vector = state.const_vector_size / 8;
                const int num_aligned_clause_output =  ((state.num_clauses / bytes_per_vector) + 1) * bytes_per_vector;
                static_assert(sizeof(ClauseOutputUint) == 1, "ClauseOutputUint must be 1 byte for now.");

                state.clause_outputs = reinterpret_cast<ClauseOutputUint*>(aligned_alloc(32, num_aligned_clause_output * sizeof(ClauseOutputUint)));
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * num_aligned_clause_output);


                state.num_class_weights_mem = state.num_classes;
                size_t class_weights_total_mem = state.num_clauses * state.num_class_weights_mem * sizeof(WeightInt);
                state.clause_weights = reinterpret_cast<WeightInt*>(aligned_alloc(32, class_weights_total_mem));
                memset(state.clause_weights , 0, class_weights_total_mem);

                state.class_votes =  reinterpret_cast<WeightInt*>(aligned_alloc(32, state.num_class_weights_mem * sizeof(WeightInt)));
                memset(state.class_votes , 0, state.num_class_weights_mem * sizeof(WeightInt));
                
                // clause mem
                // we expected the examples to be in expanded format (that is, we don't negate anything)
                // this is to avoid large portion of the data to be padding unless they perfectly align with vector size.
                const int n_vectors_per_clause = ( (state.n_literals * 2) / state.const_vector_size) + 1;
                static_assert( (state.const_vector_size % sizeof(uint64_t)) == 0, "A vector must fit in a uint64.");
                

                const int clause_mem_in_bytes = state.n_clauses * n_vectors_per_clause * state.const_bits_per_state * bytes_per_vector;                
                state.clauses = reinterpret_cast<uint64_t*>(safe_aligned_alloc(32, clause_mem_in_bytes));
                std::cout << "bitwise clause allocated with " << clause_mem_in_bytes << " bytes." << std::endl;


                std::cout << "todo: init clause weights and clauses." << std::endl;
            }



}; //namespace green_tsetlin





#endif // #ifndef __FUNC_BITWISE_AVX2_HPP__