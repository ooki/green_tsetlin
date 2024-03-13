#ifndef _FUNC_SPARSE_HPP_
#define _FUNC_SPARSE_HPP_

#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>

#include <random_generator.hpp>
#include <gt_common.hpp>
#include <sparse_tsetlin_state.hpp>

namespace  green_tsetlin
{

    


    template <typename _State, bool do_literal_budget>
    class InitializeSparseTM
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {
                if(seed == 0)
                    return false;
                    
                
                if(do_literal_budget)
                {
                    if(state.literal_budget < 1)
                        return false;
                }
                    
                state.rng.seed(seed);
                state.fast_rng.seed(seed);

                // placeholder, needs to be user defined
                state.lower_ta_threshold = -20;

                state.clauses.resize(2*state.num_clauses);
                state.clause_states.resize(2*state.num_clauses);
                for (int i = 0; i < state.num_clauses; ++i)
                {   
                    // num_literals is placeholder. Need change to load factor
                    state.clauses[i].reserve(state.num_literals);
                    state.clause_states[i].reserve(state.num_literals);
                }


                state.active_literals.resize(state.num_classes);
                for (int i = 0; i < state.num_classes; ++i)
                {
                    // num_literals is placeholder. Need change to user defined AL size
                    state.active_literals[i].reserve(state.num_literals);
                }


                state.clause_outputs = new ClauseOutputUint[state.num_clauses];
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * state.num_clauses);

                state.class_votes = new WeightInt[state.num_classes];
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);

                state.clause_weights = new WeightInt[state.num_clauses * state.num_classes];
                state.num_class_weights_mem = state.num_classes;

                return true;
            } 
        private:
            void init_clause_weights(_State& state)
            {
                std::bernoulli_distribution dist(0.5);
                const int num_weights_total = state.num_clauses * state.num_classes;
                for(int k = 0; k < num_weights_total; ++k)
                {
                    if(state.fast_rng.next_u() > 0.5)
                        state.clause_weights[k] = 1;
                    else
                        state.clause_weights[k] = -1;
                }
            }   
    }; 

    template <typename _State, bool do_literal_budget>
    class CleanupSparseTM
    {
        public:                        
            void operator()(_State& state)
            {
            }
    };


    template <typename _State, bool do_literal_budget>
    class SetClauseOutputSparseTM
    {
        public: 
            void operator()(_State& state, SparseLiterals* literals)
            {

                for (int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    // uint32t pos_literal_count = 0;
                    // uint32t neg_literal_count = 0;
                    SparseClause pos_clause = state.clauses[clause_k];
                    SparseClause neg_clause = state.clauses[clause_k + state.num_clauses];


                    SparseClauseStates pos_clause_states = state.clause_states[clause_k];
                    SparseClauseStates neg_clause_states = state.clause_states[clause_k + state.num_clauses];

                    state.clause_outputs[clause_k] = 1;

                    if ((state.clauses[clause_k].size() == 0) && (state.clauses[clause_k + state.num_clauses].size() == 0))
                    {
                        continue;
                    }

                    
                    for (int ta_k = 0; ta_k < pos_clause.size(); ++ta_k)
                    {
                        bool ta_found = false;
                        if (pos_clause_states[ta_k] <= 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }
                        for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                        {
                            if (pos_clause[ta_k] == literals->at(lit_k))
                            {
                                ta_found = true;
                                break;
                            }

                            else if (literals->at(lit_k) < pos_clause[ta_k])
                            {
                                continue;
                            }

                            else if (literals->at(lit_k) > pos_clause[ta_k])
                            {
                                state.clause_outputs[clause_k] = 0;
                                goto endclause;

                            }
                            
                        }

                        if ((!ta_found) && (pos_clause[ta_k] == pos_clause.back()))
                        {
                            state.clause_outputs[clause_k] = 0;
                            goto endclause;
                        }

                    }
                    for (int ta_k = 0; ta_k < neg_clause.size(); ++ta_k)
                    {

                        if (neg_clause_states[ta_k] <= 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }

                        for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                        {
                            if (neg_clause[ta_k] == literals->at(lit_k))
                            {
                                state.clause_outputs[clause_k] = 0;
                                goto endclause;
                                // break;
                            }

                            else if (literals->at(lit_k) < neg_clause[ta_k])
                            {
                                continue;
                            }

                            else if (literals->at(lit_k) > neg_clause[ta_k])
                            {
                                break;
                            }
                            
                        }

                    }
                
                    endclause:
                        if (do_literal_budget)
                            ;

                
                }
                //  print clause outputs
                
                
            }
    };


    template <typename _State>
    class EvalClauseOutputSparseTM
    {
        public:
            void operator()(_State& state, SparseLiterals* literals)
            {
                
            }
    };

    template <typename _State>
    class CountVotesSparseTM
    {
        public:
            void operator()(_State& state)
            {   
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);                
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    if(state.clause_outputs[clause_k] == 1)
                    {
                        for(int class_k = 0; class_k < state.num_classes; ++class_k )
                        {                                                                            
                            int32_t to_add = state.clause_weights[(clause_k * state.num_classes) + class_k];
                            state.class_votes[class_k] += to_add;
                        }
                    }
                }
            }
    };

    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateSparseTM
    {
        public:
            void operator()(_State& state, SparseLiterals* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                for (int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {
                    // WeightInt* clause_weights = state.clause_weights + clause_k * state.num_classes;

                    //  pass clause_row or only clause_k? prev. sparse imp only take clause_k. Can we use func_tm imp using &state.clauses[clause_k * n_features]? 
                    SparseClause* clause_row = &state.clauses[clause_k];
                    SparseClauseStates* clause_states = &state.clause_states[clause_k];

                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_classes];
                    

                    if (state.fast_rng.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_states, clause_weights + positive_class, 1, literals, state.clause_outputs[clause_k]);
                    }
                    else if (state.fast_rng.next_u() < prob_negative)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_states, clause_weights + negative_class, -1, literals, state.clause_outputs[clause_k]);

                    }
                }

            }
    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states, WeightInt* clause_weight, int target, SparseLiterals* literals, ClauseOutputUint clause_output)
            {
                int32_t sign = (*clause_weight) >= 0 ? +1 : -1;

                if ( (target * sign) > 0)
                {
                    if (clause_output == 1)
                    {
                        (*clause_weight) += sign;

                        _T1aFeedback t1a;
                        t1a(state, clause_row, clause_states, literals);
                        prune_clause(state, clause_row, clause_states);
                    }
                    else
                    {
                        _T1bFeedback t1b;
                        t1b(state, clause_row, clause_states);
                        prune_clause(state, clause_row, clause_states);
                    }
                }
                else if ((target * sign) < 0 && clause_output == 1)
                {
                    (*clause_weight) -= sign;

                    _T2Feedback t2;
                    t2(state, clause_row, clause_states, literals);
                }

            }
        private:
            void prune_clause(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
                // Function to remove automata from clauses, if state is below threshold

            }
            void sort_clauses_and_states(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
                // Function to sort clauses and states
            }

    };

    template <typename _State, bool boost_true_positive>
    class Type1aFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states, SparseLiterals* literals)
            {
            }
    };

    template <typename _State>
    class Type1bFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
            }
    };


    template <typename _State>
    class Type2FeedbackSparseTM
    {
        public:
            // Assume that clause_output == 1 (check ClauseUpdate)
            // // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states, SparseLiterals* literals)
            {
            }
    };


    class InputBlock;
    template <typename _State, typename _TrainSetClauseOutput>
    ClauseOutputUint* test_train_set_clause_output(SparseLiterals* literals)
    {
        
        _State state;
        std::vector<SparseClause> clauses = {{1}, {}, {0}, {0, 1}};
        std::vector<SparseClauseStates> clause_states = {{-2}, {}, {100}, {100, 100}};

        state.clauses = clauses;
        state.clause_states = clause_states;
        state.num_clauses = 2;
        state.clause_outputs = new ClauseOutputUint[2];

        _TrainSetClauseOutput train_set_clause_output;
        train_set_clause_output(state, literals);
        for (int i = 0; i < state.num_clauses; ++i)
        {
            std::cout << "clause output: " << i << " = " << (int)state.clause_outputs[i] << std::endl;
        }        

        return state.clause_outputs;
    }

}; // namespace  green_tsetlin




#endif // #ifndef _FUNC_SPARSE_HPP_