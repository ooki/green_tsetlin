#ifndef _FUNC_CONV_AVX2_H_
#define _FUNC_CONV_AVX2_H_

#include <immintrin.h> // intrics

#include <func_avx2.hpp>

namespace green_tsetlin
{

    template <typename _State, bool pad_class_weights,  bool do_literal_budget>
    class InitializeConvAVX2
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {
                {
                    InitializeAVX2<_State, pad_class_weights, do_literal_budget> f;
                    if(!f(state, seed))
                        return false;
                }

                state.active_patches = new uint32_t[state.num_clauses];
                if(state.num_patches_per_example > 0)                
                    state.literal_counts_per_patch = new uint32_t[state.num_patches_per_example];     
                else
                    return false;    

                return true;       
            }
    };


    template <typename _State, bool do_literal_budget>
    class CleanupConvAVX2
    {
        public:                        
            void operator()(_State& state)
            {
                if(state.active_patches != nullptr)
                {
                    delete[] state.active_patches;
                    state.active_patches = nullptr;
                }

                if(state.literal_counts_per_patch != nullptr)
                {
                    delete[] state.literal_counts_per_patch;
                    state.literal_counts_per_patch = nullptr;
                }

                {
                    CleanupAVX2<_State, do_literal_budget> f;
                    f(state);
                }                
            }
    };


    template <typename _State, bool do_literal_budget>
    class SetClauseOutputConvAVX2
    {
        public: 
            void operator()(_State& state, uint8_t* literals)
            {
                int8_t* clauses = (int8_t*)__builtin_assume_aligned(state.clauses, 32);
                __m256i _reminder_mask = _mm256_load_si256((__m256i const*)state.reminder_mask);
                __m256i _zeros = _mm256_set1_epi8(0);
                const int n_chunks = (state.num_literals_mem / state.literals_per_vector) - 1;


                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    state.clause_outputs[clause_k] = 0;
                    state.active_patches_storage.clear();

                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int patch_k = 0; patch_k < state.num_patches_per_example; ++patch_k)
                    {
                        uint32_t literal_count = 0;                        
                        bool patch_output = true;

                        const int literal_path_offset = patch_k * state.num_literals;
                        const uint8_t* curr_literal = &literals[literal_path_offset];
                       
                        for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                        {
                            __m256i _literals = _mm256_load_si256((__m256i const*)&curr_literal[chunk_i * 32]);                        
                            __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[chunk_i * 32]);
                            __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);
                            
                            if(do_literal_budget)
                                literal_count += __builtin_popcount(_mm256_movemask_epi8(~_not_active));

                            __m256i clause_imply_literal = _mm256_or_si256(_not_active, _literals);                        
                            __m256i _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                goto endpatch;
                            }

                            __m256i _neg_literals = _mm256_cmpeq_epi8(_zeros, _literals);                        
                            _clauses = _mm256_load_si256((__m256i const*)&clause_row[state.num_literals_mem + (chunk_i * 32)]);

                            _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);
                            
                            if(do_literal_budget)
                                literal_count += __builtin_popcount(_mm256_movemask_epi8(~_not_active));

                            clause_imply_literal = _mm256_or_si256(_not_active, _neg_literals);
                            
                            _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                goto endpatch;
                            }
                        }


                        if(state.reminder_mask != nullptr)
                        {
                            __m256i _literals = _mm256_load_si256((__m256i const*)&curr_literal[n_chunks * 32]);                        
                            __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[n_chunks * 32]);
                            __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                            __m256i _active_masked =  _mm256_and_si256(~_not_active, _reminder_mask);
                            if(do_literal_budget)
                                literal_count += __builtin_popcount(_mm256_movemask_epi8(_active_masked));
                                
                            __m256i _clause_imply_literal = _mm256_or_si256(~_active_masked, _literals);                                                                                                                                    
                            __m256i _is_false = _mm256_cmpeq_epi8(_clause_imply_literal, _zeros);
                            //_is_false = _mm256_and_si256(_is_false, _reminder_mask);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                continue;
                            }

                            __m256i _neg_literals = _mm256_cmpeq_epi8(_zeros, _literals);                        
                            _clauses = _mm256_load_si256((__m256i const*)&clause_row[state.num_literals_mem + (n_chunks * 32)]);

                            _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);
                            _active_masked =  _mm256_and_si256(~_not_active, _reminder_mask);
                            if(do_literal_budget)
                                literal_count += __builtin_popcount(_mm256_movemask_epi8(_active_masked));

                            //_clause_imply_literal = _mm256_or_si256(_not_active, _neg_literals);
                            _clause_imply_literal = _mm256_or_si256(~_active_masked, _neg_literals);                                                                                                                                    

                            
                            _is_false = _mm256_cmpeq_epi8(_clause_imply_literal, _zeros);
                            //_is_false = _mm256_and_si256(_is_false, _reminder_mask);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;   
                                continue;                         
                            }
                        }

                        endpatch:;
                                                  

                        if(patch_output)
                        {
                            state.clause_outputs[clause_k] = 1;
                            state.active_patches_storage.push_back(patch_k);
                            if(do_literal_budget)                                                            
                                state.literal_counts_per_patch[patch_k] = literal_count;                            
                        }
                    }    

                    if(state.active_patches_storage.size())
                    {
                        std::uniform_int_distribution<uint32_t> patch_picker(0, state.active_patches_storage.size()-1);
                        uint32_t random_active_patch = patch_picker(state.rng);

                        state.active_patches[clause_k] = state.active_patches_storage[random_active_patch];
                        if(do_literal_budget)
                            state.literal_counts[clause_k] = state.literal_counts_per_patch[random_active_patch];
                    }   
                }                             
            }
    };


    template <typename _State>
    class EvalClauseOutputConvAVX2
    {
        public: 
            void operator()(_State& state, uint8_t* literals)
            {
                int8_t* clauses = (int8_t*)__builtin_assume_aligned(state.clauses, 32);
                __m256i _reminder_mask = _mm256_load_si256((__m256i const*)state.reminder_mask);
                __m256i _zeros = _mm256_set1_epi8(0);
                const int n_chunks = (state.num_literals_mem / state.literals_per_vector) - 1;


                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    state.clause_outputs[clause_k] = 0;
                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int patch_k = 0; patch_k < state.num_patches_per_example; ++patch_k)
                    {                      
                        bool patch_output = true;

                        const int literal_path_offset = patch_k * state.num_literals;
                        const uint8_t* curr_literal = &literals[literal_path_offset];
                       
                        int active_literals = 0;
                        for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                        {
                            __m256i _literals = _mm256_load_si256((__m256i const*)&curr_literal[chunk_i * 32]);                        
                            __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[chunk_i * 32]);
                            __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                            active_literals |= _mm256_movemask_epi8(~_not_active);

                            __m256i clause_imply_literal = _mm256_or_si256(_not_active, _literals);                        
                            __m256i _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                goto endclause;
                            }

                            __m256i _neg_literals = _mm256_cmpeq_epi8(_zeros, _literals);                        
                            _clauses = _mm256_load_si256((__m256i const*)&clause_row[state.num_literals_mem + (chunk_i * 32)]);
                            _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                            active_literals |= _mm256_movemask_epi8(~_not_active);

                            clause_imply_literal = _mm256_or_si256(_not_active, _neg_literals);                    
                            _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                goto endclause;
                            }
                        }
                        
                        if(state.reminder_mask != nullptr)
                        {
                            __m256i _literals = _mm256_load_si256((__m256i const*)&curr_literal[n_chunks * 32]);                        
                            __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[n_chunks * 32]);
                            __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                            active_literals |= _mm256_movemask_epi8(_mm256_and_si256(~_not_active, _reminder_mask));

                            __m256i _clause_imply_literal = _mm256_or_si256(_not_active, _literals);                                                                                                                                    
                            __m256i _is_false = _mm256_cmpeq_epi8(_clause_imply_literal, _zeros);
                            _is_false = _mm256_and_si256(_is_false, _reminder_mask);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;
                                goto endclause;
                            }

                            __m256i _neg_literals = _mm256_cmpeq_epi8(_zeros, _literals);                        
                            _clauses = _mm256_load_si256((__m256i const*)&clause_row[state.num_literals_mem + (n_chunks * 32)]);
                            _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                            active_literals |= _mm256_movemask_epi8(_mm256_and_si256(~_not_active, _reminder_mask));

                            _clause_imply_literal = _mm256_or_si256(_not_active, _neg_literals);

                            _is_false = _mm256_cmpeq_epi8(_clause_imply_literal, _zeros);
                            _is_false = _mm256_and_si256(_is_false, _reminder_mask);
                            if(_mm256_testz_si256(_is_false, _is_false) == 0)
                            {
                                patch_output = false;                                
                                goto endclause;
                            }
                        }

                        if(active_literals == 0)
                            patch_output = false;      

                        if(patch_output)
                        {
                            state.clause_outputs[clause_k] = 1;    
                            break;                        
                        }

                        endclause:;                
                    }
                    
                }                                            
            }
    };


    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateConvAVX2
    {
        public:
            void operator()(_State& state, uint8_t* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                std::uniform_real_distribution<double> u(0.0,1.0);

                const int n_features = state.num_literals_mem * 2;
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    int8_t* clause_row = &state.clauses[clause_k * n_features];
                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_class_weights_mem];

                    uint8_t* patch_literals = &literals[state.active_patches[clause_k] * state.num_literals];

                    if(do_literal_budget)
                    {
                        if(state.literal_counts[clause_k] > state.literal_budget)
                            state.clause_outputs[clause_k] = 0;
                    }

                    if(state.fast_rng.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_weights + positive_class, 1, patch_literals, state.clause_outputs[clause_k]);                                            
                    }
      
                    if(state.fast_rng.next_u() < prob_negative)
                    {
                        _ClauseUpdate update_clause;
                        update_clause(state, clause_row, clause_weights + negative_class, -1, patch_literals, state.clause_outputs[clause_k]);
                    }
                }
            }
    };

}; // namespace green_tsetlin


#endif // #ifndef _FUNC_CONV_AVX2_H_