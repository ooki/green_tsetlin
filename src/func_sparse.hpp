#ifndef _FUNC_SPARSE_HPP_
#define _FUNC_SPARSE_HPP_

#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>

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

                state.clauses.resize(2*state.num_clauses);
                state.clause_states.resize(2*state.num_clauses);
                for (int i = 0; i < 2*state.num_clauses; ++i)
                {   
                    state.clauses[i].reserve(state.clause_size);
                    state.clause_states[i].reserve(state.clause_size);
                }


                state.active_literals.resize(state.num_classes);
                for (int i = 0; i < state.num_classes; ++i)
                {
                    state.active_literals[i].reserve(state.active_literals_size);
                }


                state.al_replace_index.resize(state.num_classes);
                for (int i = 0; i < state.num_classes; ++i)
                {
                    state.al_replace_index[i] = 0;
                }


                if (do_literal_budget)
                    state.literal_counts = new uint32_t[state.num_clauses];
                

                state.clause_outputs = new ClauseOutputUint[state.num_clauses];
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * state.num_clauses);

                state.class_votes = new WeightInt[state.num_classes];
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);

                state.clause_weights = new WeightInt[state.num_clauses * state.num_classes];
                state.num_class_weights_mem = state.num_classes;
                init_clause_weights(state);

                return true;
            } 
        private:
            void init_clause_weights(_State& state)
            {
                
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
                state.clauses.clear();
                state.clauses.shrink_to_fit();

                state.clause_states.clear();
                state.clause_states.shrink_to_fit();

                delete[] state.class_votes;
                state.class_votes = nullptr;

                delete[] state.clause_weights;
                state.clause_weights = nullptr;

                state.al_replace_index.clear();
                state.al_replace_index.shrink_to_fit();

                if(do_literal_budget)
                {
                    delete[] state.literal_counts;
                    state.literal_counts = nullptr;
                }

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
                    
                    uint32_t pos_literal_count = 0;
                    uint32_t neg_literal_count = 0;
                   

                    SparseClause pos_clause = state.clauses[clause_k];
                    SparseClause neg_clause = state.clauses[clause_k + state.num_clauses];


                    SparseClauseStates pos_clause_states = state.clause_states[clause_k];
                    SparseClauseStates neg_clause_states = state.clause_states[clause_k + state.num_clauses];

                    // sort literals

                    state.clause_outputs[clause_k] = 1;

                    if ((state.clauses[clause_k].size() == 0) && (state.clauses[clause_k + state.num_clauses].size() == 0))
                    {
                        if (do_literal_budget)
                            state.literal_counts[clause_k] = 0;
                        continue;
                    }

                    
                    for (size_t ta_k = 0; ta_k < pos_clause.size(); ++ta_k)
                    {
                        bool ta_found = false;
                        if (pos_clause_states[ta_k] < 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }


                        for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                        {
                            if (literals->at(lit_k) > pos_clause[ta_k])
                            {
                                state.clause_outputs[clause_k] = 0;
                                goto endclause;
                                // break;

                            }

                            else if (literals->at(lit_k) < pos_clause[ta_k])
                            {

                                continue;
                            }

                            else if (pos_clause[ta_k] == literals->at(lit_k))
                            {
                                
                                ta_found = true;
                                break;
                            }
                            
                        }

                        if ((!ta_found) && (pos_clause[ta_k] == pos_clause.back()))
                        {
                            state.clause_outputs[clause_k] = 0;
                            goto endclause;
                        }

                    }
                    for (size_t ta_k = 0; ta_k < neg_clause.size(); ++ta_k)
                    {

                        if (neg_clause_states[ta_k] < 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }


                        for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
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
                        {
                            for (size_t ta_k = 0; ta_k < pos_clause.size(); ++ta_k)
                            {
                                if (pos_clause_states[ta_k] >= 0)
                                {
                                    pos_literal_count++;
                                }
                            }
                            for (size_t ta_k = 0; ta_k < neg_clause.size(); ++ta_k)
                            {
                                if (neg_clause_states[ta_k] >= 0)
                                {
                                    neg_literal_count++;
                                }
                            }
                            // std::cout << "pos_literal_count: " << pos_literal_count << std::endl;
                            // std::cout << "neg_literal_count: " << neg_literal_count << std::endl;
                            state.literal_counts[clause_k] = pos_literal_count + neg_literal_count;
                        };
                
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
                for (int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {

                    SparseClause pos_clause = state.clauses[clause_k];
                    SparseClause neg_clause = state.clauses[clause_k + state.num_clauses];


                    SparseClauseStates pos_clause_states = state.clause_states[clause_k];
                    SparseClauseStates neg_clause_states = state.clause_states[clause_k + state.num_clauses];

                    state.clause_outputs[clause_k] = 1;



                    if ((pos_clause.size() == 0) || (neg_clause.size() == 0))
                    {
                        state.clause_outputs[clause_k] = 0;
                        continue;
                    }

                    
                    for (size_t ta_k = 0; ta_k < pos_clause.size(); ++ta_k)
                    {
                        bool ta_found = false;
                        if (pos_clause_states[ta_k] < 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }
                        for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
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
                    for (size_t ta_k = 0; ta_k < neg_clause.size(); ++ta_k)
                    {

                        if (neg_clause_states[ta_k] < 0)
                        {
                            //  only evaluate when ta state is > 0
                            continue;
                        }

                        for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
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
                
                    endclause:;

                
                }
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
                    
                    
                    SparseClause* pos_clause_row = &state.clauses[clause_k];
                    SparseClause* neg_clause_row = &state.clauses[clause_k + state.num_clauses];
                    SparseClauseStates* pos_clause_states = &state.clause_states[clause_k];
                    SparseClauseStates* neg_clause_states = &state.clause_states[clause_k + state.num_clauses];
                    
                    SparseLiterals* active_literals_positive_class = &state.active_literals[positive_class];
                    // SparseLiterals* neg_active_literals_positive_class = &state.active_literals[positive_class + state.num_classes];

                    SparseLiterals* active_literals_negative_class = &state.active_literals[negative_class];
                    // SparseLiterals* neg_active_literals_negative_class = &state.active_literals[negative_class + state.num_classes];


                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_classes];



                    if (do_literal_budget)
                    {
                        if(state.literal_counts[clause_k] > state.literal_budget)
                        {
                            state.clause_outputs[clause_k] = 0;
                        }
                    }

                    if (state.fast_rng.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, active_literals_positive_class, clause_weights + positive_class, 1, positive_class, literals, state.clause_outputs[clause_k]);
                    }

                    if (state.fast_rng.next_u() < prob_negative)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, active_literals_negative_class, clause_weights + negative_class, -1, negative_class, literals, state.clause_outputs[clause_k]);

                    }
                }

            }
    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* active_literals, WeightInt* clause_weight, int target, int class_k, SparseLiterals* literals, ClauseOutputUint clause_output)
            {
                int32_t sign = (*clause_weight) >= 0 ? +1 : -1;

                if ( (target * sign) > 0)
                {
                    if (clause_output == 1)
                    {
                        (*clause_weight) += sign;

                        _T1aFeedback t1a;
                        t1a(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, active_literals, literals, class_k);
                        prune_clause(state, pos_clause_row, pos_clause_states);
                        prune_clause(state, neg_clause_row, neg_clause_states);
                    }
                    else if (clause_output == 0)
                    {
                        _T1bFeedback t1b;
                        t1b(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states);
                        prune_clause(state, pos_clause_row, pos_clause_states);
                        prune_clause(state, neg_clause_row, neg_clause_states);
                    }
                }
                else if ((target * sign) < 0 && clause_output == 1)
                {
                    (*clause_weight) -= sign;

                    _T2Feedback t2;
                    t2(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, active_literals, literals);
                    sort_clauses_and_states(state, pos_clause_row, pos_clause_states);
                    sort_clauses_and_states(state, neg_clause_row, neg_clause_states);
                }

            }
        private:
            void prune_clause(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
                // Function to remove automata from clauses if state is below threshold. clause_row and states remains sorted

                for (size_t ta_k = 0; ta_k < clause_row->size(); ++ta_k)
                {
                    if (clause_states->at(ta_k) < state.lower_ta_threshold)
                    {   
                        clause_row->erase(clause_row->begin() + ta_k);
                        clause_states->erase(clause_states->begin() + ta_k);
                    }
                }



            }

            void sort_clauses_and_states(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
                // Function to sort clauses and states, when new elements have been added from t2 feedback
                
                // if 2 elements, just check if they are in order, if not, swap
                if (clause_row->size() == 1)
                    return;
                
                
                if (clause_row->size() == 2)
                {
                    if (clause_row->at(0) > clause_row->at(1))
                    {
                        std::swap(clause_row->at(0), clause_row->at(1));
                        std::swap(clause_states->at(0), clause_states->at(1));
                    }
                    return;
                }

                bool swapped = false;
                for (size_t i = 0; i < clause_row->size(); i++)
                {
                    for (size_t j = 0; j < clause_row->size() - i - 1; j++)
                    {
                        if (clause_row->at(j) > clause_row->at(j + 1))
                        {
                            std::swap(clause_row->at(j), clause_row->at(j + 1));
                            std::swap(clause_states->at(j), clause_states->at(j + 1));
                            swapped = true;
                        }
                    }
                    if (!swapped)
                        break;
                }

            
            }

    };

    template <typename _State, typename _UpdateAL, bool boost_true_positive>
    class Type1aFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* active_literals, SparseLiterals* literals, int class_k)
            {
                // const double s_inv = (1.0 / state.s);
                // const double s_min1_inv = (state.s - 1.0) / state.s;

                const int8_t lower_state = -127;
                const int8_t upper_state = 127;


                _UpdateAL update_al;

                // loop literals
                for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                {
                    const uint32_t literal = literals->at(lit_k);
                    int8_t literal_state_pos = 1;
                    int8_t literal_state_neg = 1;

                    // loop pos clauses
                    for (size_t ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                    {
                        if (pos_clause_row->at(ta_k) == literal)
                        {
                            if ((pos_clause_states->at(ta_k) < upper_state))
                            {
                                if (boost_true_positive)
                                {
                                    pos_clause_states->at(ta_k) += 1;
                                }
                                else
                                {
                                    if (state.fast_rng.next_u() <= state.s_min1_inv)
                                    {
                                        pos_clause_states->at(ta_k) += 1;
                                    }
                                }
                            }
                            literal_state_pos = 0;
                            break; // updated pos_ta_state for current lit, go next lit
                        }
                    }
                    // no ta for literal found in clause, add to AL
                    if (literal_state_pos == 1) //  && state.fast_rng.next_u() <= s_inv
                    {
                        update_al(state, active_literals, literal, class_k);
                    }

                    // loop neg clauses
                    for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                    {
                        if (neg_clause_row->at(ta_k) == literal)
                        {
                            if ((neg_clause_states->at(ta_k) > lower_state) && (state.fast_rng.next_u() <= state.s_inv))
                            {
                                neg_clause_states->at(ta_k) -= 1;
                            }
                            literal_state_neg = 0;
                            break; // updated neg_ta_state for current lit, go next lit
                        }
                    }
                    // no ta for literal found in clause, add to AL
                    // if (literal_state_neg == 1 && state.fast_rng.next_u() <= s_inv)
                    // {
                    //     update_al(state, neg_active_literals, literal, class_k + state.num_classes);
                    // }
                }
            

                // loop pos clauses
                for (size_t ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {

                    for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (pos_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_pos;
                        }
                    }
                    if ((pos_clause_states->at(ta_k) > lower_state) && (state.fast_rng.next_u() <= state.s_inv))
                        pos_clause_states->at(ta_k) -= 1;

                    endloop_pos:;
                }

                // loop neg clauses
                for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {
                    
                    for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (neg_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_neg;
                        }
                    }
                    if (neg_clause_states->at(ta_k) < upper_state)
                    {
                        if (boost_true_positive)
                        {
                            neg_clause_states->at(ta_k) += 1;
                        }
                        else
                        {
                            if (state.fast_rng.next_u() <= state.s_min1_inv)
                                neg_clause_states->at(ta_k) += 1;
                        }
                    }

                    endloop_neg:;
                }

            }
    };


    template <typename _State>
    class Type1bFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states)
            {
                // const double s_inv = (1.0 / state.s);
                const int8_t lower_state = -127;


                // loop pos clauses
                for (size_t ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {
                    if (state.fast_rng.next_u() <= state.s_inv)
                    {
                        if (pos_clause_states->at(ta_k) > lower_state)
                            pos_clause_states->at(ta_k) -= 1;
                    }
                }

                // loop neg clauses
                for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {
                    if (state.fast_rng.next_u() <= state.s_inv)
                    {
                        if (neg_clause_states->at(ta_k) > lower_state)
                            neg_clause_states->at(ta_k) -= 1;
                    }
                }
            }
    };


    template <typename _State>
    class Type2FeedbackSparseTM
    {
        public:
            // Assume that clause_output == 1 (check ClauseUpdate)
            // // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* active_literals, SparseLiterals* literals)
            {

                // loop pos clauses, check if ta is not in literals, if so, increment ta state if its above threshold
                for (size_t ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {
                
                    // loop literals, if we find ta in literals, skip increment
                    for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (pos_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_pos;
                        }
                    }
                    if (pos_clause_states->at(ta_k) < 0)
                        pos_clause_states->at(ta_k) += 1;
                    
                    

                    endloop_pos:;
                }

                for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {

                    for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (neg_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            if (neg_clause_states->at(ta_k) < 0)
                                neg_clause_states->at(ta_k) += 1;
                            goto endloop_neg;
                        }
                    }

                    endloop_neg:;
                }


                for (size_t pos_lit_k = 0; pos_lit_k < active_literals->size(); ++pos_lit_k)
                {
                    for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (active_literals->at(pos_lit_k) == literals->at(lit_k))
                        {
                            // goto endloop_pos_al;
                            for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                            {
                                if (active_literals->at(pos_lit_k) == neg_clause_row->at(ta_k))
                                {
                                    goto endloop_pos_al;
                                }
                            }

                            if (neg_clause_row->size() < state.clause_size)
                                {    
                                neg_clause_row->push_back(active_literals->at(pos_lit_k));
                                neg_clause_states->push_back(state.lower_ta_threshold + 5);
                                }

                            goto endloop_pos_al;

                        }
                    }
                    for (size_t ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                    {
                        if (active_literals->at(pos_lit_k) == pos_clause_row->at(ta_k))
                        {
                            goto endloop_pos_al;
                        }
                    }

                    if (pos_clause_row->size() < state.clause_size)
                    {
                        pos_clause_row->push_back(active_literals->at(pos_lit_k));
                        pos_clause_states->push_back(state.lower_ta_threshold + 5);
                    }

                    endloop_pos_al:;
                }

                // for (size_t neg_lit_k = 0; neg_lit_k < neg_active_literals->size(); ++neg_lit_k)
                // {
                //     for (size_t ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                //     {
                //         if (neg_active_literals->at(neg_lit_k) == neg_clause_row->at(ta_k))
                //         {
                //             goto endloop_neg_al;
                //         }
                //     }
                //     for (size_t lit_k = 0; lit_k < literals->size(); ++lit_k)
                //     {
                //         if (neg_active_literals->at(neg_lit_k) == literals->at(lit_k))
                //         {   
                            // if (neg_clause_row->size() < state.clause_size)
                            // {    
                            //     neg_clause_row->push_back(neg_active_literals->at(neg_lit_k));
                            //     neg_clause_states->push_back(state.lower_ta_threshold + 5);
                            // }
                            // goto endloop_neg_al;
                //         }
                //     }



                //     endloop_neg_al:;
                // }

            }
    };

    template <typename _State, bool dynamic_AL>
    class UpdateAL
    {
        public:
            void operator()(_State& state, SparseLiterals* active_literals_class_k, uint32_t literal, int class_k) // active_literals_class_k might need to be pointer
            {
                // Function to update active literals

                if (active_literals_class_k->size() < state.active_literals_size)
                {
                    // check if literal is already in active_literals
                    for (size_t i = 0; i < active_literals_class_k->size(); ++i)
                    {
                        if (active_literals_class_k->at(i) == literal)
                        {
                            return;
                        }
                    }
                    active_literals_class_k->push_back(literal);
                }
                else
                {
                    // if (dynamic_AL)
                    //     active_literals_class_k->at(state.fast_rng.next_u() * state.active_literals_size) = literal;

                    if (dynamic_AL)
                    {
                        for (size_t i = 0; i < active_literals_class_k->size(); ++i)
                        {
                            if (active_literals_class_k->at(i) == literal)
                            {
                                return;
                            }
                        }
                        active_literals_class_k->at(state.al_replace_index[class_k]) = literal;
                        state.al_replace_index[class_k] ++;
                        
                        if(state.al_replace_index[class_k] == state.active_literals_size)
                            state.al_replace_index[class_k] = 0;
                    }

                }
                // dynamic al stuff, think need diff datatype for active_literals to do this effectively
                // else
                // {
                //     if (dynamic_AL)
                // }
                
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

    class InputBlock;
    template <typename _State, typename _CBImpl, typename _T2Feedback>
    void test_Type2FeedbackSparse(_CBImpl* cb, InputBlock* ib, int n_clauses, int example, int class_num, bool do_AL)
    {
        
        if(n_clauses < 1)
            n_clauses = 1;

        ib->prepare_example(example);

        _State& state = cb->get_state();
        cb->pull_example();
        auto lits = cb->get_current_literals();

        if (do_AL)
        {
            
            state.active_literals[0] = {0, 1};
            state.active_literals[1] = {0, 1};
        
        }

        for (int i = 0; i < n_clauses; i++)
        {
            _T2Feedback t2;
            t2(state, &state.clauses[i], &state.clauses[i + state.num_clauses], &state.clause_states[i], &state.clause_states[i + state.num_clauses], &state.active_literals[class_num], &state.active_literals[class_num + 1], lits);
        }

    }


    class InputBlock;
    template <typename _State, typename _CBImpl, typename _T1aFeedback>
    void test_Type1aFeedbackSparse(_CBImpl* cb, InputBlock* ib, int n_clauses, int example, int class_num)
    {
        if(n_clauses < 1)
            n_clauses = 1;


        ib->prepare_example(0);


        SparseTsetlinState &state = cb->get_state();
        cb->pull_example();
        auto lits = cb->get_current_literals();


        for (int i = 0; i < n_clauses; i++)
        {
            _T1aFeedback t1a;
            t1a(state, &state.clauses[i], &state.clauses[i + state.num_clauses], &state.clause_states[i], &state.clause_states[i + state.num_clauses], &state.active_literals[class_num], &state.active_literals[class_num + 1], lits, 0); // remember that test now needs to accunt for new al indexing. 
        }


    }


    // class InputBlock;
    // template <typename _State, typename _T1bFeedback>
    // void test_Type1bFeedbackSparseNV(_CBImpl* cb, InputBlock* ib)
    // {
    //     if(n_clauses < 1)
    //         n_clauses = 1;


    //     ib->prepare_example(0);


    //     SparseTsetlinState &state = cb->get_state();
    //     cb->pull_example();
    //     auto lits = cb->get_current_literals();


    //     for (int i = 0; i < n_clauses; i++)
    //     {
    //         _T1bFeedback t1b;
    //         t1b(state, i, lits);
    //     }


    // }

}; // namespace  green_tsetlin




#endif // #ifndef _FUNC_SPARSE_HPP_