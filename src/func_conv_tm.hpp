#ifndef _FUNC_CONV_TM_HPP_
#define _FUNC_CONV_TM_HPP_

#include <random>
#include <vector>

#include <gt_common.hpp>

#include <func_tm.hpp>

namespace green_tsetlin
{

    template <typename _State, bool do_literal_budget>
    class InitializeConvTM
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {
                {
                    InitializeTM<_State, do_literal_budget> f;
                    f(state, seed);
                }

                state.m_active_patches = new uint32_t[m_num_clauses];
            }
    };

    template <typename _State, bool do_literal_budget>
    class CleanupConvTM
    {
        public:                        
            void operator()(_State& state)
            {
                if(state.m_active_patches != nullptr)
                {
                    delete[] state.m_active_patches;
                    state.m_active_patches = nullptr;
                }

                {
                    CleanupTM<_State, do_literal_budget> f;
                    f(state);
                }                
            }
    };

    template <typename _State, bool do_literal_budget>
    class SetClauseOutputConvTM
    {
        public: 
            void operator()(_State& state, uint8_t* literals)
            {

                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    uint32_t pos_literal_count = 0;
                    uint32_t neg_literal_count = 0;

                    state.clause_outputs[clause_k] = 1;
                        
                    int8_t* pl_pos = &state.clauses[clause_k * (state.num_literals*2)];
                    int8_t* pl_neg = &state.clauses[(clause_k * (state.num_literals*2)) + state.num_literals];

                    for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                    {                        

                        if(*pl_pos >= 0)
                        {                                                        
                            if(literals[literal_k] == 0)
                            {
                                state.clause_outputs[clause_k] = 0;
                                break;
                            }

                            if(do_literal_budget)
                               pos_literal_count++;
                        }
                        pl_pos++;

                        if(*pl_neg >= 0) 
                        {                            
                            if(literals[literal_k] == 1)
                            {
                                state.clause_outputs[clause_k] = 0;
                                break;
                            }

                            if(do_literal_budget)
                                neg_literal_count++;
                        }
                        pl_neg++;                                                                     
                    }
                        
                    if(do_literal_budget)
                        state.literal_counts[clause_k] = pos_literal_count + neg_literal_count;
                    
                }
            }
    };








}; // namespace green_tsetlin

#endif // #ifndef _FUNC_CONV_TM_HPP_
