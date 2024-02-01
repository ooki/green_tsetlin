#ifndef _FUNC_AVX2_HPP_
#define _FUNC_AVX2_HPP_

#include <stdlib.h>
#include <cmath>
#include <algorithm>

#include <immintrin.h> // intrics

#include <random_generator.hpp>
#include <gt_common.hpp>



namespace green_tsetlin
{
    void int8_print(const int8_t* c, int n)
    {
        for(int i = 0; i < n; ++i)
            printf(" %d ", c[i]);
        printf("\n");
    }


    void _mm256_print_epi8(__m256i vec)
    {
        int8_t temp[32];
        _mm256_storeu_si256((__m256i*)&temp[0], vec);
        int8_print(temp, 32);
    }

    // to be used for counting active ta's
    inline int _avx2_mask_count(const __m256i in)
    {
        __m256i _zeros = _mm256_set1_epi8(0);
        __m256i _on = _mm256_cmpgt_epi8(in, _zeros);
        int bits = _mm256_movemask_epi8(_on);
        return __builtin_popcount(bits); // TODO: wrap into save version (or force std::popcount() c++20)
    }

    class CoaleasedTsetlinStateAligned32
    {
        public:
            constexpr const static int literals_per_vector = 32;
            constexpr const static int outputs_per_vector = 256 / sizeof(WeightInt);
            constexpr const static uint32_t high_number_if_no_positive_literal_is_present = 65000;

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
            XorShift128plus4G rng_avx2;
            Wyhash64 rng_nv;


            inline void set_s(double s_param)
            {
                s = s_param;
                double p = 1 / s;
                int32_t tmp = ((int32_t)(p * 255)) - 127;                
                tmp += 1; // size we use < to compare and not <=

                gtcmp_for_s = (int8_t)std::clamp(tmp, -127, 126);
            }

            inline double get_s() const { return s; }

            inline int8_t get_ta_state(int clause_k, int ta_i, bool ta_polarity)
            {
                if(ta_polarity)
                    return clauses[(clause_k * num_literals_mem * 2) + ta_i];
                else
                    return clauses[(clause_k * num_literals_mem * 2) + num_literals_mem + ta_i];
            }

            inline void set_ta_state(int clause_k, int ta_i, bool ta_polarity, int8_t new_state)
            {
                if(ta_polarity)
                    clauses[(clause_k * num_literals_mem * 2) + ta_i] = new_state;
                else
                {
                    std::cout << "setting TA state: " << ta_i << " index:" << (clause_k * num_literals_mem * 2) + num_literals_mem + ta_i << " to: " << (int)new_state << std::endl;
                    clauses[(clause_k * num_literals_mem * 2) + num_literals_mem + ta_i] = new_state;                    
                }
            }

            inline WeightInt get_clause_weight(int clause_index, int target_class)
            {
                return clause_weights[(clause_index * num_class_weights_mem) + target_class];
            }

            inline void set_clause_weight(int clause_index, int target_class, WeightInt new_weight)
            {
                clause_weights[(clause_index * num_class_weights_mem) + target_class] = new_weight;
            }

            inline WeightInt* get_class_votes() const
            {
                return class_votes;
            }

            inline std::vector<int8_t> get_copy_clauses() const
            {                
                std::size_t n_total_literals = num_clauses * (num_literals_mem*2);                
                std::vector<int8_t>  states(clauses, clauses + n_total_literals);
                return states;
            }

            inline std::vector<uint32_t> get_copy_literal_counts() const 
            {          
                std::vector<uint32_t>  counts(literal_counts, literal_counts + num_clauses);
                return counts;
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

    template <typename _State, bool pad_class_weights,  bool do_literal_budget>
    class InitializeAligned32
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {            
                state.rng_avx2.seed(seed);
                state.rng_nv.seed(seed);
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
                state.clauses = reinterpret_cast<int8_t*>(aligned_alloc(32, clause_mem));

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
                        clause_row[literal_k] = dist(state.rng); // pos
                        clause_row[literal_k + state.num_literals_mem] = dist(state.rng); // neg
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
                        if(dist(state.rng))
                            state.clause_weights[(clause_k * state.num_class_weights_mem) + class_i] = 1;
                        else
                            state.clause_weights[(clause_k * state.num_class_weights_mem) + class_i] = -1;
                    }
                }
                /*
                const int num_weights_total = state.num_clauses * state.num_classes;
                for(int k = 0; k < num_weights_total; ++k)
                {
                    if(dist(state.rng))
                        state.clause_weights[k] = 1;
                    else
                        state.clause_weights[k] = -1;
                }
                */
            }        
    };

    template <typename _State, bool do_literal_budget>
    class CleanupAligned32
    {
        public:                        
            void operator()(_State& state)
            {
                free(state.clauses);
                state.clauses = nullptr;     

                free(state.class_votes);
                state.class_votes = nullptr;

                free(state.clause_weights);
                state.clause_weights = nullptr;

                free(state.reminder_mask);
                state.reminder_mask = nullptr;     

                if(do_literal_budget)
                {
                    delete[] state.literal_counts;
                    state.literal_counts = nullptr;
                }
            }
    };





    template <typename _State, bool do_literal_budget, bool force_at_least_one_positive_literal>
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
                    uint32_t pos_literal_count = 0;
                    uint32_t neg_literal_count = 0;

                    state.clause_outputs[clause_k] = 1;

                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                    {
                        __m256i _literals = _mm256_load_si256((__m256i const*)&literals[chunk_i * 32]);                        
                        __m256i _clauses = _mm256_load_si256((__m256i const*)&clause_row[chunk_i * 32]);
                        __m256i _not_active = _mm256_cmpgt_epi8(_zeros, _clauses);
                        
                        if(do_literal_budget)
                            pos_literal_count += __builtin_popcount(_mm256_movemask_epi8(~_not_active));

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
                            neg_literal_count += __builtin_popcount(_mm256_movemask_epi8(~_not_active));

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
                            pos_literal_count += __builtin_popcount(_mm256_movemask_epi8(_active_masked));
                            
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
                            neg_literal_count += __builtin_popcount(_mm256_movemask_epi8(_active_masked));

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
                        {
                            if(force_at_least_one_positive_literal)
                            {                                

                                if(neg_literal_count == 0 || pos_literal_count > 0)
                                {
                                    state.literal_counts[clause_k] = pos_literal_count + neg_literal_count;
                                }
                                else
                                {
                                    state.literal_counts[clause_k] = state.high_number_if_no_positive_literal_is_present;
                                }                                    
                            }
                            else
                                state.literal_counts[clause_k] = pos_literal_count + neg_literal_count;
                        }
                        // else
                        //     state.literal_counts[clause_k] = 0;
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
    class EmptyCountVotesAVX2
    {
        public:
            void operator()(_State& state) {}
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

    #include <unistd.h>

    template <typename _State>
    class CountVotesVectorAVX2
    {
        public:
            void operator()(_State& state)
            {   
                std::cout << " --------------- Not Impl CountVotesVectorAVX2 --------" << std::endl;

                const int num_chunks = state.num_class_weights_mem / state.outputs_per_vector;                

                int32_t* votes = (int32_t*)__builtin_assume_aligned(state.class_votes, 32);
                __m256i _zero = _mm256_set1_epi32(0);                
                for(int chunk_i = 0; chunk_i < num_chunks; chunk_i++)
                    _mm256_store_si256((__m256i*)&votes[chunk_i * state.outputs_per_vector], _zero);
                

                const int32_t* cw = (int32_t*)__builtin_assume_aligned(state.clause_weights, 32);
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    if(state.clause_outputs[clause_k] == 1)
                    {
                        const int clause_weight_i = clause_k * state.num_class_weights_mem;
                        for(int chunk_i = 0; chunk_i < num_chunks; chunk_i++)
                        {
                            __m256i _votes = _mm256_load_si256((__m256i const*)&votes[chunk_i * state.outputs_per_vector]);
                            __m256i _weights = _mm256_load_si256((__m256i const*)&cw[clause_weight_i + (chunk_i * state.outputs_per_vector)]);

                            _votes = _mm256_add_epi32(_votes, _weights);
                            _mm256_store_si256((__m256i*)&votes[chunk_i * state.outputs_per_vector], _votes);
                        }
                    }
                }   

                //usleep(200 * 1000);
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

                    if( state.rng_nv.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_weights + positive_class, 1, literals, state.clause_outputs[clause_k]);                    
                    }
      
                    if( state.rng_nv.next_u() < prob_negative)
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
                        t1b(state, clause_row, literals);      
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

    template <typename _State>
    class Type1aFeedbackAVX2
    {
        public:
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {
                constexpr bool use_boost_true_positive = false;

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

                    __m256i _rand = state.rng_avx2.next();
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
                    _rand = state.rng_avx2.next();
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
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {
                 int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);

                __m256i _minus_one = _mm256_set1_epi8(-1);                
                __m256i _cmp_s = _mm256_set1_epi8(state.gtcmp_for_s);

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);
                                
                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {
                    __m256i _clause = _mm256_load_si256((__m256i const*)&clause[chunk_i * state.literals_per_vector]);
                    __m256i _rand = state.rng_avx2.next();

                    __m256i _update = _mm256_cmpgt_epi8(_cmp_s, _rand);
                    _update = _mm256_and_si256(_minus_one, _update);

                    _clause = _mm256_adds_epi8(_clause, _update);
                    _mm256_store_si256((__m256i*)&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // -- negated
                    _clause = _mm256_load_si256((__m256i const*)&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _rand = state.rng_avx2.next(); // TODO:  check if doing both and store is faster ( rand0 = next(); rand1 = next() )

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

#endif // #ifndef _FUNC_AVX2_HPP_