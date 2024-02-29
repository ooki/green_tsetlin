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
    class CleanupConvTM
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
                    state.clause_outputs[clause_k] = 0;
                    state.active_patches_storage.clear();

                    for(int patch_k = 0; patch_k < state.num_patches_per_example; ++patch_k)
                    {
                        uint32_t literal_count = 0;                        
                        bool patch_output = true;

                        int8_t* pl_pos = &state.clauses[clause_k * (state.num_literals*2)];
                        int8_t* pl_neg = &state.clauses[(clause_k * (state.num_literals*2)) + state.num_literals];

                        const int literal_path_offset = patch_k * state.num_literals;
                        const uint8_t* curr_literal = &literals[literal_path_offset];

                        for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                        {                        

                            if(*pl_pos >= 0)
                            {                                                        
                                if(curr_literal[literal_k] == 0)
                                {
                                    patch_output = false;
                                    break;
                                }

                                if(do_literal_budget)
                                    literal_count++;
                            }
                            pl_pos++;

                            if(*pl_neg >= 0) 
                            {                            
                                if(curr_literal[literal_k] == 1)
                                {
                                    patch_output = false;
                                    break;
                                }

                                if(do_literal_budget)
                                    literal_count++;
                            }
                            pl_neg++;                                                                     
                        }

                        if(patch_output)
                        {
                            state.clause_outputs[clause_k] = 1;
                            state.active_patches_storage.push_back(patch_k);
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
    class EvalClauseOutputConvTM
    {
        public:
            void operator()(_State& state, uint8_t* literals)
            {
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {             
                    state.clause_outputs[clause_k] = 0;

                    for(int patch_k = 0; patch_k < state.num_patches_per_example; ++patch_k)
                    {
                        int8_t* pl_pos = &state.clauses[clause_k * (state.num_literals*2)];
                        int8_t* pl_neg = &state.clauses[(clause_k * (state.num_literals*2)) + state.num_literals];

                        const int literal_path_offset = patch_k * state.num_literals;
                        const uint8_t* curr_literal = &literals[literal_path_offset];
                        
                        bool patch_output = true;
                        bool is_empty_clause = true;
                        for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                        {                        
                            if(*pl_pos >= 0)
                            {
                                is_empty_clause = false;
                                if(curr_literal[literal_k] == 0)
                                {
                                    patch_output = false;
                                    break;
                                }
                            }
                            pl_pos++;

                            if(*pl_neg >= 0)
                            {
                                is_empty_clause = false;
                                if(curr_literal[literal_k] == 1) 
                                {
                                    patch_output = false;
                                    break;
                                }
                            }
                            pl_neg++;                                                                     
                        }

                        if(is_empty_clause)
                            patch_output = false;

                        if(patch_output)
                        {
                            state.clause_outputs[clause_k] = 1;
                            goto end_of_clause;
                        }   
                    }

                    end_of_clause:;                                
                }
            }
    };




    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateConvTM
    {
        public:
            void operator()(_State& state, uint8_t* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                std::uniform_real_distribution<double> u(0.0,1.0);

                const int n_features = state.num_literals * 2;
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    int8_t* clause_row = &state.clauses[clause_k * n_features];
                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_classes];

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

#endif // #ifndef _FUNC_CONV_TM_HPP_
