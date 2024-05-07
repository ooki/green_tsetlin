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


    template <typename _State, bool pad_class_weights,  bool do_literal_budget>
    class InitializeAVX2
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {            
                if(seed == 0)
                    return false;

                state.avx2_rng.seed(seed);
                state.fast_rng.seed(seed);
                state.rng.seed(seed);

                state.reminder_mask = reinterpret_cast<int8_t*>(aligned_alloc(32, state.literals_per_vector));
                for(int i = 0; i < state.literals_per_vector; ++i)
                        state.reminder_mask[i] = 0xFF;
                        
                if( (state.num_literals % state.literals_per_vector) == 0)
                {
                    state.num_literals_mem = state.num_literals;
                    state.num_reminder = 0;                    
                }
                else                
                {
                    state.num_literals_mem = ((state.num_literals / state.literals_per_vector) + 1) * state.literals_per_vector;                    
                    state.num_reminder = state.num_literals % state.literals_per_vector;
                    
                    for(int i = state.num_reminder; i < state.literals_per_vector; ++i)
                        state.reminder_mask[i] = 0x00;
                }
                
                #ifdef PRINT_DEBUG
                    std::cout << "[DEBUG]" << "allocated " << state.num_literals_mem << " <- from " << state.num_literals <<" literals. #vec:" << state.literals_per_vector << std::endl;
                #endif 

                int clause_mem = state.num_clauses * state.num_literals_mem * 2;
                state.clauses = reinterpret_cast<int8_t*>(safe_aligned_alloc(32, clause_mem));

                if(do_literal_budget)
                    state.literal_counts = new uint32_t[state.num_clauses];

                
                // align so that we can loop over with vectors of 32ints
                
                int num_aligned_clause_output =  ((state.num_clauses / state.outputs_per_vector) + 1) * state.outputs_per_vector;
                state.clause_outputs = reinterpret_cast<ClauseOutputUint*>(aligned_alloc(32, num_aligned_clause_output * sizeof(ClauseOutputUint)));
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * num_aligned_clause_output);
            
                
                
                if(pad_class_weights)
                {
                    state.num_class_weights_mem = ((state.num_classes / state.outputs_per_vector) + 1) * state.outputs_per_vector;                
                    
                    size_t class_weights_total_mem = state.num_clauses * state.num_class_weights_mem * sizeof(WeightInt);
                    state.clause_weights = reinterpret_cast<WeightInt*>(aligned_alloc(32, class_weights_total_mem));
                    memset(state.clause_weights , 0, class_weights_total_mem);
                }
                else
                {
                    state.num_class_weights_mem = state.num_classes;
                    size_t class_weights_total_mem = state.num_clauses * state.num_class_weights_mem * sizeof(WeightInt);
                    state.clause_weights = reinterpret_cast<WeightInt*>(aligned_alloc(32, class_weights_total_mem));
                    memset(state.clause_weights , 0, class_weights_total_mem);   
                }

                state.class_votes =  reinterpret_cast<WeightInt*>(aligned_alloc(32, state.num_class_weights_mem * sizeof(WeightInt)));
                memset(state.class_votes , 0, state.num_class_weights_mem * sizeof(WeightInt));
         
                init_clauses(state);
                init_clause_weights(state);

                return true;
            }

        
        private:
            void init_clauses(_State& state)
            {
                std::uniform_int_distribution<int8_t> dist(-1, 0);

                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    int8_t* clause_row = &state.clauses[clause_k * (state.num_literals_mem * 2)];
                                        
                    for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                    {
                        if(state.fast_rng.next_u() > 0.5)
                            clause_row[literal_k] = 0; // pos
                        else
                            clause_row[literal_k] = -1; 

                        if(state.fast_rng.next_u() > 0.5) // negated
                            clause_row[literal_k + state.num_literals_mem] = 0;
                        else
                            clause_row[literal_k + state.num_literals_mem] = -1;
                    }

                    // set filler states to -109 and -111 (pos and neg)
                    for(int filler_lit_k = state.num_literals; filler_lit_k < state.num_literals_mem; ++filler_lit_k)
                    {
                        clause_row[filler_lit_k] = -109;
                        clause_row[filler_lit_k + state.num_literals_mem] = -111;
                    }
                }      
            }

            void init_clause_weights(_State& state)
            {
                std::bernoulli_distribution dist(0.5);
                for(int clause_k = 0; clause_k < state.num_clauses; clause_k++)
                {
                    for(int class_i = 0; class_i < state.num_classes; class_i++)
                    {
                        if(state.fast_rng.next_u() > 0.5)
                            state.clause_weights[(clause_k * state.num_class_weights_mem) + class_i] = 1;
                        else
                            state.clause_weights[(clause_k * state.num_class_weights_mem) + class_i] = -1;
                    }
                }
            }        
    };


    template <typename _State, bool do_literal_budget>
    class CleanupAVX2
    {
        public:                        
            void operator()(_State& state)
            {
                safe_aligned_free(state.clauses);
                state.clauses = nullptr;     

                safe_aligned_free(state.class_votes);
                state.class_votes = nullptr;

                safe_aligned_free(state.clause_weights);
                state.clause_weights = nullptr;

                safe_aligned_free(state.reminder_mask);
                state.reminder_mask = nullptr;     

                if(do_literal_budget)
                {
                    delete[] state.literal_counts;
                    state.literal_counts = nullptr;
                }
            }
    };


    template <typename _State, bool do_literal_budget>
    class SetClauseOutputAVX2
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
                    uint32_t literal_count = 0;

                    state.clause_outputs[clause_k] = 1;

                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                    {
                        __m256i _literals = _mm256_load_si256((__m256i const*)&literals[chunk_i * 32]);                        
                        __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[chunk_i * 32]);
                        __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);
                        
                        if(do_literal_budget)
                            literal_count += __builtin_popcount(_mm256_movemask_epi8(~_not_active));

                        __m256i clause_imply_literal = _mm256_or_si256(_not_active, _literals);                        
                        __m256i _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                        if(_mm256_testz_si256(_is_false, _is_false) == 0)
                        {
                            state.clause_outputs[clause_k] = 0;
                            goto endclause;
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
                            state.clause_outputs[clause_k] = 0;
                            goto endclause;
                        }
                    }

                    
                    if(state.reminder_mask != nullptr)
                    {
                        __m256i _literals = _mm256_load_si256((__m256i const*)&literals[n_chunks * 32]);                        
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
                            state.clause_outputs[clause_k] = 0;
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
                            state.clause_outputs[clause_k] = 0;                            
                        }
                    }
                    
                    endclause:
                        if(do_literal_budget)
                            state.literal_counts[clause_k] = literal_count;
                }
            }
    };


    template <typename _State>
    class EvalClauseOutputAVX2
    {
        public:
            void operator()(_State& state, uint8_t* literals)
            {
                int8_t* clauses = (int8_t*)__builtin_assume_aligned(state.clauses, 32);

                __m256i _reminder_mask = _mm256_load_si256((__m256i const*)state.reminder_mask);
                __m256i _zeros = _mm256_set1_epi8(0);

                const int n_chunks = (state.num_literals_mem / 32) - 1;

                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    state.clause_outputs[clause_k] = 1;

                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    int active_literals = 0;
                    for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                    {
                        __m256i _literals = _mm256_load_si256((__m256i const*)&literals[chunk_i * 32]);                        
                        __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[chunk_i * 32]);
                        __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                        active_literals |= _mm256_movemask_epi8(~_not_active);

                        __m256i clause_imply_literal = _mm256_or_si256(_not_active, _literals);                        
                        __m256i _is_false = _mm256_cmpeq_epi8(clause_imply_literal, _zeros);
                        if(_mm256_testz_si256(_is_false, _is_false) == 0)
                        {
                            state.clause_outputs[clause_k] = 0;
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
                            state.clause_outputs[clause_k] = 0;
                            goto endclause;
                        }
                    }
                    
                    if(state.reminder_mask != nullptr)
                    {
                        __m256i _literals = _mm256_load_si256((__m256i const*)&literals[n_chunks * 32]);                        
                        __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[n_chunks * 32]);
                        __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);

                        active_literals |= _mm256_movemask_epi8(_mm256_and_si256(~_not_active, _reminder_mask));

                        __m256i _clause_imply_literal = _mm256_or_si256(_not_active, _literals);                                                                                                                                    
                        __m256i _is_false = _mm256_cmpeq_epi8(_clause_imply_literal, _zeros);
                        _is_false = _mm256_and_si256(_is_false, _reminder_mask);
                        if(_mm256_testz_si256(_is_false, _is_false) == 0)
                        {
                            state.clause_outputs[clause_k] = 0;
                            continue;
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
                            state.clause_outputs[clause_k] = 0;                            
                            continue;
                        }
                    }

                    if(active_literals == 0)
                        state.clause_outputs[clause_k] = 0;    

                    endclause:;                
                    
                }
            }
    };

    template <typename _State>
    class CountVotesAVX2
    {
        public:
            // TODO: rewrite to avx2
            void operator()(_State& state)
            {                
                //int32_t* votes = (int32_t*)__builtin_assume_aligned(state.class_votes, 32);
                //const int32_t* cw = (int32_t*)__builtin_assume_aligned(state.clause_weights, 32);

                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);                
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    if(state.clause_outputs[clause_k] == 1)
                    {
                        for(int class_k = 0; class_k < state.num_classes; ++class_k )
                        {                                                                            
                            WeightInt to_add = state.clause_weights[(clause_k * state.num_class_weights_mem) + class_k];
                            state.class_votes[class_k] += to_add;
                        }
                    }
                }
            }
    };

    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateAVX2
    {
        public:
            void operator()(_State& state, uint8_t* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {

                const int n_features = state.num_literals_mem * 2;
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    int8_t* clause_row = &state.clauses[clause_k * n_features];
                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_class_weights_mem];


                    if(do_literal_budget)
                    {
                        if(state.literal_counts[clause_k] > state.literal_budget)
                            state.clause_outputs[clause_k] = 0;
                    }

                    if( state.fast_rng.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_weights + positive_class, 1, literals, state.clause_outputs[clause_k]);                    
                    }
      
                    if( state.fast_rng.next_u() < prob_negative)
                    {
                        _ClauseUpdate update_clause;
                        update_clause(state, clause_row, clause_weights + negative_class, -1, literals, state.clause_outputs[clause_k]);
                    }
                }
            }
    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateAVX2
    {
        public:
            void operator()(_State& state, int8_t* clause_row, WeightInt* clause_weight, int target, uint8_t* literals, ClauseOutputUint clause_output)
            {
                
                WeightInt sign = (*clause_weight) >= 0 ? +1 : -1;                
                
                if( (target * sign) > 0)
                {
                    if(clause_output == 1)
                    {
                        (*clause_weight) += sign;

                        _T1aFeedback t1a;
                        t1a(state, clause_row, literals);                    
                    }
                    else
                    {
                        _T1bFeedback t1b;
                        t1b(state, clause_row);
                    }
                }

                else if((target * sign) < 0 && clause_output == 1)
                {
                    (*clause_weight) -= sign;

                    _T2Feedback t2;
                    t2(state, clause_row, literals);
                }
            }
    };

    template <typename _State, bool use_boost_true_positive>
    class Type1aFeedbackAVX2
    {
        public:
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {

                 int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);
                 const uint8_t* literals = (uint8_t*)__builtin_assume_aligned(literals_in, 32);

                __m256i _minus_one = _mm256_set1_epi8(-1);                
                __m256i _cmp_s = _mm256_set1_epi8(state.gtcmp_for_s);

                const int n_chunks = (state.num_literals_mem / 32);                            
                
                __m256i _ones = _mm256_set1_epi8(1);

                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {                                                
                    __m256i _literals = _mm256_load_si256((__m256i const*)&literals[chunk_i * state.literals_per_vector]);
                    __m256i _clause = _mm256_load_si256((__m256i const*)&clause[chunk_i * state.literals_per_vector]);                    
                    __m256i _literal_on = _mm256_cmpeq_epi8(_literals, _ones);

                    __m256i _rand = state.avx2_rng.next();
                    __m256i _update = _mm256_cmpgt_epi8(_cmp_s, _rand);

                    __m256i _subtract = _mm256_and_si256(~_literal_on, _update);
                    _subtract = _mm256_and_si256(_minus_one, _subtract);
                    
                    __m256i _add;
                    if(use_boost_true_positive)
                    {
                        _add = _mm256_and_si256(_literal_on, _ones);
                    }
                    else
                    {
                        _add = _mm256_and_si256(_literal_on, ~_update);
                        _add = _mm256_and_si256(_ones, _add); // -> boost
                    }                    
                    
                    
                    _clause = _mm256_adds_epi8(_clause, _mm256_or_si256(_subtract, _add));
                    _mm256_store_si256((__m256i*)&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // --- negated ----
                    _clause = _mm256_load_si256((__m256i const*)&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _rand = state.avx2_rng.next();
                    _update = _mm256_cmpgt_epi8(_cmp_s, _rand);

                    _subtract = _mm256_and_si256(_literal_on, _update);
                    _subtract = _mm256_and_si256(_minus_one, _subtract);

                    // boost
                    _add = _mm256_and_si256(~_literal_on, ~_update);
                    _add = _mm256_and_si256(_ones, _add);

                                            
                    _clause = _mm256_adds_epi8(_clause, _mm256_or_si256(_subtract, _add));
                    _mm256_store_si256((__m256i*)&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);
                }
            }
    };

    template <typename _State>
    class Type1bFeedbackAVX2
    {
        public:
            void operator()(_State& state, int8_t* clause_row)
            {

                int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);

  

                __m256i _minus_one = _mm256_set1_epi8(-1);                
                __m256i _cmp_s = _mm256_set1_epi8(state.gtcmp_for_s);

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);

                                         
                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {
                    //__m256i _clause = _mm256_load_si256((__m256i const*)&clause[chunk_i * state.literals_per_vector]);
                    __m256i _clause = _mm256_load_si256((__m256i const*)&clause[chunk_i * state.literals_per_vector]);
                    __m256i _rand = state.avx2_rng.next();

                    __m256i _update = _mm256_cmpgt_epi8(_cmp_s, _rand);
                    _update = _mm256_and_si256(_minus_one, _update);

                    _clause = _mm256_adds_epi8(_clause, _update);
                    _mm256_store_si256((__m256i*)&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // -- negated
                    _clause = _mm256_load_si256((__m256i const*)&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _rand = state.avx2_rng.next(); // TODO:  check if doing both and store is faster ( rand0 = next(); rand1 = next() )

                    _update = _mm256_cmpgt_epi8(_cmp_s, _rand);
                    _update = _mm256_and_si256(_minus_one, _update);
                    _clause = _mm256_adds_epi8(_clause, _update);

                    _mm256_store_si256((__m256i*)&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);
                }
            }
    };


    template <typename _State>
    class Type2FeedbackAVX2
    {
        public:
            // Assume that clause_output == 1
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals)
            {                
                
                int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);
                __m256i _zeros = _mm256_set1_epi8(0);
                __m256i _ones = _mm256_set1_epi8(1);

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);
                //__m256i _reminder_mask = _mm256_load_si256((__m256i const*)state.reminder_mask);

                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {
                    __m256i _literals = _mm256_load_si256((__m256i const*)&literals[chunk_i * state.literals_per_vector]);             
                    __m256i _clause = _mm256_load_si256((__m256i const*)&clause[chunk_i * state.literals_per_vector]);

                    __m256i _lit_mask = _mm256_cmpeq_epi8(_literals, _zeros);                        
                    __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clause);

                    __m256i _add_mask = _mm256_and_si256(_lit_mask, _not_active);
                    __m256i _inc = _mm256_and_si256(_add_mask, _ones);

                    _clause = _mm256_adds_epi8(_clause, _inc);
                    _mm256_store_si256((__m256i*)&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // -- negated --

                    _clause = _mm256_load_si256((__m256i const*)&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _not_active = _mm256_cmpgt_epi8(_zeros, _clause);

                    _add_mask = _mm256_and_si256(~_lit_mask, _not_active);
                    _inc = _mm256_and_si256(_add_mask, _ones);
                    _clause = _mm256_adds_epi8(_clause, _inc);
                    _mm256_store_si256((__m256i*)&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);
                }        

                /*
                if(state.reminder_mask != nullptr)
                {
                    __m256i _literals = _mm256_load_si256((__m256i const*)&literals[n_chunks * 32]);             
                    __m256i _clause = _mm256_load_si256((__m256i const*)&clause[n_chunks * 32]);

                    __m256i _lit_mask = _mm256_cmpeq_epi8(_literals, _zeros);                        
                    __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clause);

                    __m256i _add_mask = _mm256_and_si256(_lit_mask, _not_active);
                    __m256i _inc = _mm256_and_si256(_add_mask, _ones);

                    _clause = _mm256_adds_epi8(_clause, _inc);
                    _mm256_store_si256((__m256i*)&clause_row[n_chunks * 32], _clause);

                    _clause = _mm256_load_si256((__m256i const*)&clause[state.num_literals_mem + (n_chunks * 32)]);
                    _not_active = _mm256_cmpgt_epi8(_zeros, _clause);

                    _add_mask = _mm256_and_si256(~_lit_mask, _not_active);
                    _inc = _mm256_and_si256(_add_mask, _ones);
                    _clause = _mm256_adds_epi8(_clause, _inc);
                    _mm256_store_si256((__m256i*)&clause_row[state.num_literals_mem + (n_chunks * 32)], _clause);
                }
                */                
            }
    };

}; // namespace green_tsetlin










#endif // #ifndef _FUNC_AVX2_H_