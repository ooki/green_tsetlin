#ifndef _FUNC_NEON_HPP_
#define _FUNC_NEON_HPP_

#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <string>

#include <arm_neon.h>
#include <random_generator.hpp>

#include <gt_common.hpp>

#define PRINT_DEBUG

namespace green_tsetlin
{
    void int8_print(const int8_t* c, int n)
    {
        for(int i = 0; i < n; ++i)
            printf(" %d ", c[i]);
        printf("\n");
    }

    void neon_print(std::string name, int8x16_t v)
    {
        int8_t buffer[16];
        vst1q_s8(buffer, v);
        std::cout << "vector '" << name << "':" << std::endl;
        int8_print(buffer, 16);
    }

    // TODO: check for faster ways of doing this - a horizontal add seems slow
    inline uint32_t count_on_masks_neon(const uint8x16_t& v)
    {
        uint8x16_t msb_shifted = vshrq_n_u8(v, 7);
        return static_cast<uint32_t>(vaddvq_u8(msb_shifted));
    }

    void test_neon()
    {
        std::cout << "start neon test" << std::endl;
        int8x16_t  v;

        int8_t clauses[16];
        for(int i = 0; i < 16; i++)
            clauses[i] = 42;

        
            //clauses[i] = (i < 8) ? -1 - i : 0 + i;
        clauses[8] = 0;

        int8_t literals[16];
        for(int i = 0; i < 16; i++)
            literals[i] = ((i % 2) == 0) ? 1 : 0;
        

        // int8_t numbers[16];
        // for(int i = 0; i < 16; i++)
        //     numbers[i] = -8 + i;
        // // v = vld1q_s8(clauses);
        // neon_print("v", v);
        
        // int8x16_t _zeros = vmovq_n_s8(0);
        // int8x16_t _is_false = vcgeq_s8(v, _zeros);

        // int8_t buffer[16];
        // vst1q_s8(buffer, _is_false);
        // int8_print(buffer, 16);

        // if( vmaxvq_s8(_is_false) == 0)
        //     std::cout << "vmaxvq_s8() == 0" << std::endl;
        // else
        //     std::cout << "vmaxvq_s8() == 1" << std::endl;

        //int8x16_t _minus_ones = vmovq_n_s8(-1);
        int8x16_t _zeros = vmovq_n_s8(0);
        int8x16_t _literals = vld1q_s8(literals);                        
        int8x16_t _clauses = vld1q_s8(clauses);
        // int8x16_t _numbers = vld1q_s8(numbers);
        //neon_print("_minus_ones", _minus_ones);
        neon_print("_zeros", _zeros);
        neon_print("literals", _literals);
        neon_print("clauses", _clauses);


        //int8x16_t neg_literals2 = vceqq_s8(_zeros, _literals);                        
        //neon_print("neg_literals2", neg_literals2);
        //_literals = neg_literals2;
        // //neon_print("_numbers", _numbers);
        // int8x16_t _gt_zeros = vcgeq_s8(_numbers, _zeros);
        // //neon_print("_gt_zeros", _gt_zeros);


        // int8x16_t _gt_zeros_flip = vcgeq_s8(_zeros, _numbers);
        // //neon_print("_gt_zeros_flip", _gt_zeros_flip);

        int8x16_t _not_active = vcgtq_s8(_zeros, _clauses);
        neon_print("_not_active", _not_active);


        uint8_t count = count_on_masks_neon(~_not_active);
        std::cout << "Count: " << (int)count << std::endl;


                // int8x16_t _minus_one = vdupq_n_s8(-1);                
                // int8x16_t _cmp_s = vdupq_n_s8(state.gtcmp_for_s);

                                

                // const int n_chunks = (state.num_literals_mem / state.literals_per_vector);                                            
                // int8x16_t _ones = vdupq_n_s8(1);



                    //         int8x16_t _literals = vld1q_s8(&literals[chunk_i * state.literals_per_vector]);
                    // int8x16_t _clause = vld1q_s8(&clause[chunk_i * state.literals_per_vector]);
                    // int8x16_t _literal_on = vceqq_s8(_literals, _ones);

                    
                    // int8x16_t _rand = state.rng_neon.next();
                    // int8x16_t _update = vcgtq_s8(_cmp_s, _rand);

                    // int8x16_t _subtract = vandq_s8(~_literal_on, _update);
                    // _subtract = vandq_s8(_minus_one, _subtract);

                    // int8x16_t _add;
                    // if(use_boost_true_positive)
                    // {
                    //     _add = vandq_s8(_literal_on, _ones);
                    // }
                    // else
                    // {
                    //     _add = vandq_s8(_literal_on, ~_update);
                    //     _add = vandq_s8(_ones, _add); // -> boost
                    // }

                    // _clause = vqaddq_s8(_clause, vorrq_s8(_subtract, _add));
                    // vst1q_s8(&clause_row[chunk_i * state.literals_per_vector], _clause);


        // int8x16_t _clause_imply_literal = vorrq_s8(_not_active, _literals);                        
        // neon_print("clause_imply_literal", _clause_imply_literal);


        // std::cout << " max s8: " << (int)vmaxvq_s8(_clause_imply_literal) << std::endl;
        // if(vmaxvq_s8(_clause_imply_literal) == 1)
        // {
        //     std::cout << "clause output == 0"<< std::endl;
        // }
        // else
        // {
        //     std::cout << "clause is true" << std::endl;
        // }

        // int8x16_t _neg_literals = vnegq_s8(_literals);
        // neon_print("neg literals", _neg_literals);


        // int8x16_t _signed_or = vqaddq_s8(_neg_literals, _not_active);
        // std::cout << " max s8 (neg): " << (int)vmaxvq_s8(_signed_or) << std::endl;
        // if(vmaxvq_s8(_signed_or) == 1)
        // {
        //     std::cout << "neg clause output == 0"<< std::endl;
        // }
        // else
        // {
        //     std::cout << "neg clause is true" << std::endl;
        // }

        // int8x16_t _is_false = vcgeq_s8(clause_imply_literal, _zeros);

        // int8x16_t out = vaddq_s8(v, a);
        
        // int8_t results[16];
        // vst1q_s8(results, out);

        // int8_print(results, 16);

        // std::cout << "- neon  generate random !! "<< std::endl;

        // XorShift128Plus2G   rng(16);

        // int8x16_t o = rng.next();
        // vst1q_s8(results, o);

        // int8_print(results, 16);

        // std::cout << "- neon  generate random !! DONE "<< std::endl;
    }

    inline uint64_t neon_movemask(uint8x16_t mask)
    {
        const uint16x8_t equal_mask = vreinterpretq_u16_u8(mask);
        const uint8x8_t res = vshrn_n_u16(equal_mask, 4);
        const uint64_t matches = vget_lane_u64(vreinterpret_u64_u8(res), 0);
        return matches;
    }

    
    

    void* neon_alloc(int align_to, int mem)
    {
        int new_mem = mem;
        int r = mem % align_to;

        if(mem < align_to)
            new_mem = align_to;
            
        else if(r > 0)
            new_mem += (align_to - r);

        void* p = std::aligned_alloc(align_to, new_mem);
        if(p == nullptr)
        {
            std::string msg = "NEON failed to allocate mem: " + std::to_string(new_mem) + " aligned to: " + std::to_string(align_to) + " req:" + std::to_string(mem);
            throw std::runtime_error(msg);
        }
        return p;
    }


    //
    // TODO: to increase efficency we should unroll the 16 into 2x16 that are interleaved
    // 
    // load 16 first -> a
    // load 16 next  -> b
    //  cmp a mask
    //  cmp b mask
    // Since each vector operation has some cycles delay (we dont want to stall the pipeline)
    //

    class CoaleasedTsetlinStateAlignedNeon32
    {
        public:
            constexpr static const int literals_per_vector = 16;
            constexpr static const int outputs_per_vector = 8;
            constexpr static const int align_to = 32;


            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;
            int num_literals = 0;
            int num_literals_mem = 0;
            int num_reminder = 0;
            int num_class_weights_mem = 0;

            std::default_random_engine rng;
            int8_t* clauses = nullptr;
            ClauseOutputUint* clause_outputs = nullptr;
            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;

            uint32_t* literal_counts = nullptr;
            uint32_t literal_budget = 0xFFFF;

            int8_t* reminder_mask = nullptr;
            int8_t gtcmp_for_s = 0;

            //XorShift128Plus2G rng_neon;
            Xoshiro128Plus rng_neon;
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
                    clauses[(clause_k * num_literals_mem * 2) + num_literals_mem + ta_i] = new_state;
            }

            inline WeightInt get_clause_weight(int clause_index, int target_class)
            {
                return clause_weights[(clause_index * num_classes) + target_class];
            }

            inline void set_clause_weight(int clause_index, int target_class, WeightInt new_weight)
            {
                clause_weights[(clause_index * num_classes) + target_class] = new_weight;
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



    // TODO: impl padded class weights (for neon support on vote counts)
    template <typename _State,  bool do_literal_budget>
    class InitializeAlignedNeon32
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {            
                state.rng_neon.seed(seed);
                state.rng_nv.seed(seed);
                
                state.reminder_mask = reinterpret_cast<int8_t*>(neon_alloc(state.align_to, state.literals_per_vector));
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

                int clause_mem = state.num_clauses * state.num_literals_mem * 2;

                state.clauses = reinterpret_cast<int8_t*>(neon_alloc(state.align_to, clause_mem));
                if(do_literal_budget)
                    state.literal_counts = new uint32_t[state.num_clauses];
                
                // align so that we can loop over with vectors of 32ints
                int num_aligned_clause_output =  ((state.num_clauses / state.outputs_per_vector) + 1) * state.outputs_per_vector;
                state.clause_outputs = reinterpret_cast<ClauseOutputUint*>(neon_alloc(32, num_aligned_clause_output * sizeof(ClauseOutputUint)));
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * num_aligned_clause_output);

                //state.clause_weights = new int32_t[state.num_clauses * state.num_classes];
                int weights_mem = state.num_clauses * state.num_classes * sizeof(WeightInt);
                state.num_class_weights_mem = state.num_classes;

                state.clause_weights = reinterpret_cast<WeightInt*>(neon_alloc(32, weights_mem));

                //state.class_votes = new int32_t[state.num_classes];
                state.class_votes =  reinterpret_cast<WeightInt*>(neon_alloc(16, state.num_classes * sizeof(WeightInt)));

                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);
                state.rng.seed(seed);

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
                const int num_weights_total = state.num_clauses * state.num_classes;
                for(int k = 0; k < num_weights_total; ++k)
                {
                    if(dist(state.rng))
                        state.clause_weights[k] = 1;
                    else
                        state.clause_weights[k] = -1;

                }
            }        
    };

    template <typename _State,  bool do_literal_budget>
    class CleanupAlignedNeon32
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


    


    template <typename _State, bool do_literal_budget>
    class SetClauseOutputNeon
    {
        public:
            void operator()(_State& state, uint8_t* literals)
            {
                // TODO: rewrite to use neon_movemask instead of max?

                // https://gcc.gnu.org/onlinedocs/gcc-4.9.4/gcc/ARM-NEON-Intrinsics.html

                int8_t* clauses = (int8_t*)__builtin_assume_aligned(state.clauses, 32);
                int8x16_t _zeros = vmovq_n_s8(0);
                const int n_chunks = (state.num_literals_mem / state.literals_per_vector) - 1;

                int8x16_t _reminder_mask;

                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    // std::cout << "---------- CLAUSE: " << clause_k << "---------------" << std::endl;
                    state.clause_outputs[clause_k] = 1;
                    state.literal_counts[clause_k] = 0;
                    
                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                    {
                        // std::cout << "chunk_i:" << chunk_i << std::endl;
                        int8x16_t _literals = vld1q_s8((const int8_t*)&literals[chunk_i * state.literals_per_vector]);                        
                        int8x16_t _clauses = vld1q_s8(&clause_row[chunk_i * state.literals_per_vector]);

                        int8x16_t _not_active = vcgtq_s8(_zeros, _clauses);

                        if(do_literal_budget)
                            state.literal_counts[clause_k] += count_on_masks_neon(~_not_active);

                        int8x16_t _clause_imply_literal = vorrq_s8(_not_active, _literals);    
                        int8x16_t _is_false = vceqq_s8(_clause_imply_literal, _zeros);


                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            // std::cout << "  --- POS : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                            goto endclause; 
                        }


                        _literals = vceqq_s8(_zeros, _literals);
                        _clauses = vld1q_s8(&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                        _not_active = vcgtq_s8(_zeros, _clauses);

                        if(do_literal_budget)
                            state.literal_counts[clause_k] += count_on_masks_neon(~_not_active);

                        _clause_imply_literal = vorrq_s8(_not_active, _literals);
                        _is_false = vceqq_s8(_clause_imply_literal, _zeros);

                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            // std::cout << "  --- NEG : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                            goto endclause; 
                        }

 
                    }
                    
                    if(state.reminder_mask != nullptr)
                    {
                        _reminder_mask = vld1q_s8(state.reminder_mask);

                        // std::cout << "reminder:" << n_chunks << std::endl;
                        int8x16_t _literals = vld1q_s8((const int8_t*)&literals[n_chunks * state.literals_per_vector]);                        
                        int8x16_t _clauses = vld1q_s8(&clause_row[n_chunks * state.literals_per_vector]);
                        int8x16_t _not_active = vcgtq_s8(_zeros, _clauses);

                        int8x16_t _active_masked =  vandq_s8(~_not_active, _reminder_mask);
                        if(do_literal_budget)
                            state.literal_counts[clause_k] += count_on_masks_neon(_active_masked);

                        int8x16_t _clause_imply_literal = vorrq_s8(~_active_masked, _literals);
                        int8x16_t _is_false = vceqq_s8(_clause_imply_literal, _zeros);

                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            state.clause_outputs[clause_k] = 0;
                            continue; 
                        }




                        // neon_print("_literals", _literals);
                        // neon_print("_clauses", _clauses);
                        // neon_print("_not_active", _not_active);
                        // neon_print("_clause_imply_literal", _clause_imply_literal);
                        // neon_print("_is_false", _is_false);
                        // std::cout << "max(_impl) = " << (int)vmaxvq_u8(_is_false) <<  std::endl;
                        

                        // std::cout << "-------------------" << std::endl;

                        _literals = vceqq_s8(_zeros, _literals);
                        _clauses = vld1q_s8(&clause_row[state.num_literals_mem + (n_chunks * state.literals_per_vector)]);
                        _not_active = vcgtq_s8(_zeros, _clauses);
                        
                        _active_masked =  vandq_s8(~_not_active, _reminder_mask);
                        if(do_literal_budget)
                            state.literal_counts[clause_k] += count_on_masks_neon(_active_masked);


                        _clause_imply_literal = vorrq_s8(~_active_masked, _literals);
                        _is_false = vceqq_s8(_clause_imply_literal, _zeros);
                        // neon_print("NOT _literals", _literals);
                        // neon_print("_clauses", _clauses);
                        // neon_print("_not_active", _not_active);
                        // neon_print("_clause_imply_literal", _clause_imply_literal);
                        // neon_print("_is_false", _is_false);
                        // std::cout << "max(_impl) = " << (int)vmaxvq_u8(_is_false) <<  std::endl;

                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            //std::cout << "  --- NEG : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                        }
                    }

                    endclause: ;
                    //std::cout << " out(" << clause_k << ") = " << state.clause_outputs[clause_k] << std::endl;
                    
                }
            }
    };
    

    template <typename _State>
    class EvalClauseOutputNeon
    {
        public:
            void operator()(_State& state, uint8_t* literals)
            {
                int8_t* clauses = (int8_t*)__builtin_assume_aligned(state.clauses, 32);
                int8x16_t _zeros = vmovq_n_s8(0);
                const int n_chunks = (state.num_literals_mem / state.literals_per_vector) - 1;

                int8x16_t _reminder_mask;
                uint64_t active_mask = 0;
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    // std::cout << "---------- CLAUSE: " << clause_k << "---------------" << std::endl;
                    state.clause_outputs[clause_k] = 1;
                    const int8_t* clause_row =  &clauses[clause_k * (state.num_literals_mem * 2)];

                    for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                    {
                        // std::cout << "chunk_i:" << chunk_i << std::endl;
                        int8x16_t _literals = vld1q_s8((const int8_t*)&literals[chunk_i * state.literals_per_vector]);                        
                        int8x16_t _clauses = vld1q_s8(&clause_row[chunk_i * state.literals_per_vector]);

                        int8x16_t _not_active = vcgtq_s8(_zeros, _clauses);

                        active_mask |= neon_movemask(~_not_active);

                        int8x16_t _clause_imply_literal = vorrq_s8(_not_active, _literals);    
                        int8x16_t _is_false = vceqq_s8(_clause_imply_literal, _zeros);


                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            // std::cout << "  --- POS : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                            goto endclause; 
                        }


                        _literals = vceqq_s8(_zeros, _literals);
                        _clauses = vld1q_s8(&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                        _not_active = vcgtq_s8(_zeros, _clauses);
                        active_mask |= neon_movemask(~_not_active);

                        _clause_imply_literal = vorrq_s8(_not_active, _literals);
                        _is_false = vceqq_s8(_clause_imply_literal, _zeros);

                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            // std::cout << "  --- NEG : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                            goto endclause; 
                        }

 
                    }
                    
                    if(state.reminder_mask != nullptr)
                    {
                        _reminder_mask = vld1q_s8(state.reminder_mask);

                        // std::cout << "reminder:" << n_chunks << std::endl;
                        int8x16_t _literals = vld1q_s8((const int8_t*)&literals[n_chunks * state.literals_per_vector]);                        
                        int8x16_t _clauses = vld1q_s8(&clause_row[n_chunks * state.literals_per_vector]);
                        int8x16_t _not_active = vcgtq_s8(_zeros, _clauses);

                        int8x16_t _clause_imply_literal = vorrq_s8(_not_active, _literals);

                        active_mask |= neon_movemask(vandq_s8(~_not_active, _reminder_mask));
                        int8x16_t _is_false = vceqq_s8(_clause_imply_literal, _zeros);
                        _is_false = vandq_s8(_is_false, _reminder_mask);


                        // neon_print("_literals", _literals);
                        // neon_print("_clauses", _clauses);
                        // neon_print("_not_active", _not_active);
                        // neon_print("_clause_imply_literal", _clause_imply_literal);
                        // neon_print("_is_false", _is_false);
                        // std::cout << "max(_impl) = " << (int)vmaxvq_u8(_is_false) <<  std::endl;
                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            // std::cout << "  --- POS : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                            continue;   
                        }
                        

                        // std::cout << "-------------------" << std::endl;

                        _literals = vceqq_s8(_zeros, _literals);
                        _clauses = vld1q_s8(&clause_row[state.num_literals_mem + (n_chunks * state.literals_per_vector)]);
                        _not_active = vcgtq_s8(_zeros, _clauses);

                        _clause_imply_literal = vorrq_s8(_not_active, _literals);
                        active_mask |= neon_movemask(vandq_s8(~_not_active, _reminder_mask));
                        
                        _is_false = vceqq_s8(_clause_imply_literal, _zeros);
                        _is_false = vandq_s8(_is_false, _reminder_mask);

                        // neon_print("NOT _literals", _literals);
                        // neon_print("_clauses", _clauses);
                        // neon_print("_not_active", _not_active);
                        // neon_print("_clause_imply_literal", _clause_imply_literal);
                        // neon_print("_is_false", _is_false);
                        // std::cout << "max(_impl) = " << (int)vmaxvq_u8(_is_false) <<  std::endl;

                        if(vmaxvq_u8(_is_false) > 0)
                        {
                            //std::cout << "  --- NEG : CONT ---" << std::endl;
                            state.clause_outputs[clause_k] = 0;
                        }
                    }

                    if(active_mask == 0)
                        state.clause_outputs[clause_k] = 0;


                    endclause: ;
                    //std::cout << " out(" << clause_k << ") = " << state.clause_outputs[clause_k] << std::endl;
                    
                }
            }
    };


    template <typename _State>
    class CountVotesNeon
    {
        public:
            // TODO: rewrite to simd
            void operator()(_State& state)
            {                
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);                
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    if(state.clause_outputs[clause_k] == 1)
                    {
                        for(int class_k = 0; class_k < state.num_classes; ++class_k )
                        {                                                                            
                            WeightInt to_add = state.clause_weights[(clause_k * state.num_classes) + class_k];
                            state.class_votes[class_k] += to_add;
                        }
                    }
                }   
            }
    };

    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateNeon
    {
        public:
            void operator()(_State& state, uint8_t* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {

                const int n_features = state.num_literals_mem * 2;
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    int8_t* clause_row = &state.clauses[clause_k * n_features];
                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_classes];

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
    class ClauseUpdateNeon
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
    class Type1aFeedbackNeon
    {
        public:
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {
                //std::cout << "==== NEON T1a - start ========" << std::endl;
                constexpr bool use_boost_true_positive = false;

                int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);
                const int8_t* literals = (int8_t*)__builtin_assume_aligned(literals_in, 32);

                int8x16_t _minus_one = vdupq_n_s8(-1);                
                int8x16_t _cmp_s = vdupq_n_s8(state.gtcmp_for_s);

                                

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);                                            
                int8x16_t _ones = vdupq_n_s8(1);

                //std::cout << "n_chunks: " << n_chunks << std::endl;

                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {                                                
                    int8x16_t _literals = vld1q_s8(&literals[chunk_i * state.literals_per_vector]);
                    int8x16_t _clause = vld1q_s8(&clause[chunk_i * state.literals_per_vector]);
                    int8x16_t _literal_on = vceqq_s8(_literals, _ones);

                    
                    // neon_print("literals", _literals);
                    // neon_print("_literal_on", _literal_on);
                    // neon_print("clauses", _clause);

                    
                    int8x16_t _rand = state.rng_neon.next();
                    // int8x16_t _minus_one = vdupq_n_s8(-1);   // hard set the update
                    int8x16_t _update = vcgtq_s8(_cmp_s, _rand);

                    // neon_print("_rand", _rand);
                    // neon_print("_cmp_s", _cmp_s);
                    // neon_print("_update", _update);

                    int8x16_t _subtract = vandq_s8(~_literal_on, _update);
                    // neon_print("_subtract", _subtract);

                    //_subtract = vandq_s8(_minus_one, _subtract);
                    //neon_print("_subtract", _subtract);

                    int8x16_t _add;
                    if(use_boost_true_positive)
                    {
                        _add = vandq_s8(_literal_on, _ones);
                    }
                    else
                    {
                        _add = vandq_s8(_literal_on, ~_update);
                        // neon_print("_add", _add);
                        _add = vandq_s8(_ones, _add); // -> boost
                        // neon_print("_add", _add);
                    }

                    _clause = vqaddq_s8(_clause, vorrq_s8(_subtract, _add));
                    //neon_print("_clause O-P", _clause);
                    vst1q_s8(&clause_row[chunk_i * state.literals_per_vector], _clause);


                    // --- negated ----
                    _clause = vld1q_s8(&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _rand = state.rng_neon.next();
                    _update = vcgtq_s8(_cmp_s, _rand);
                    _subtract = vandq_s8(_literal_on, _update);
                    

                    // boost - only use boost true positive on 1 literals (not negated)
                    _add = vandq_s8(~_literal_on, ~_update);
                    //neon_print("_add", _add);
                    _add = vandq_s8(_ones, _add);
                    //neon_print("_add", _add);
                    
                    _clause = vqaddq_s8(_clause, vorrq_s8(_subtract, _add));
                    //neon_print("_clause O-N", _clause);
                    vst1q_s8(&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);


                }
            }
    };

    template <typename _State>
    class Type1bFeedbackNeon
    {
        public:
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {
                 int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);

                int8x16_t _minus_one = vdupq_n_s8(-1);                
                int8x16_t _cmp_s = vdupq_n_s8(state.gtcmp_for_s);

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);
                                
                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {
                    int8x16_t _clause = vld1q_s8(&clause[chunk_i * state.literals_per_vector]);
                    int8x16_t _rand = state.rng_neon.next();

                    int8x16_t _update = vcgtq_s8(_cmp_s, _rand);
                    _update = vandq_s8(_minus_one, _update);

                    _clause = vqaddq_s8(_clause, _update);
                    vst1q_s8(&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // -- negated
                    _clause = vld1q_s8(&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _rand = state.rng_neon.next();

                    _update = vcgtq_s8(_cmp_s, _rand);
                    _update = vandq_s8(_minus_one, _update);
                    _clause = vqaddq_s8(_clause, _update);

                    vst1q_s8(&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);
                }
            }
    };


    template <typename _State>
    class Type2FeedbackNeon
    {
        public:
            // Assume that clause_output == 1
            void operator()(_State& state, int8_t* clause_row, const uint8_t* literals_in)
            {                
                int8_t* clause = (int8_t*)__builtin_assume_aligned(clause_row, 32);
                const int8_t* literals = (int8_t*)__builtin_assume_aligned(literals_in, 32);

                
                //__m256i
                int8x16_t _zeros = vdupq_n_s8(0);    
                int8x16_t _ones = vdupq_n_s8(1);    
                

                const int n_chunks = (state.num_literals_mem / state.literals_per_vector);

                for(int chunk_i = 0; chunk_i < n_chunks; ++chunk_i)
                {
                    int8x16_t _literals = vld1q_s8(&literals[chunk_i * state.literals_per_vector]);
                    int8x16_t _clause = vld1q_s8(&clause[chunk_i * state.literals_per_vector]);


                    int8x16_t _lit_mask = vceqq_s8(_literals, _zeros);                      
                    int8x16_t _not_active = vcgtq_s8(_zeros, _clause);                          

                    int8x16_t _add_mask = vandq_s8(_lit_mask, _not_active);
                    int8x16_t _inc = vandq_s8(_add_mask, _ones);

                    _clause = vqaddq_s8(_clause, _inc);
                    vst1q_s8(&clause_row[chunk_i * state.literals_per_vector], _clause);

                    // -- negated --

                    _clause = vld1q_s8(&clause[state.num_literals_mem + (chunk_i * state.literals_per_vector)]);
                    _not_active = vcgtq_s8(_zeros, _clause);

                    _add_mask = vandq_s8(~_lit_mask, _not_active);
                    _inc = vandq_s8(_add_mask, _ones);
                    _clause = vqaddq_s8(_clause, _inc);
                    vst1q_s8(&clause_row[state.num_literals_mem + (chunk_i * state.literals_per_vector)], _clause);
                }  
            }
    };
    
}; // namespace green_tsetlin

#endif // #ifndef _FUNC_AVX2_HPP_