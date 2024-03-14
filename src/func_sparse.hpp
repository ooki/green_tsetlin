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

// using namespace std;
// struct Tuple;
// struct RefTuple;
// #define TUPLE_COMMON_FUNC(C, D, E)            \
//     C##::C## (Tuple& t) ##D                        \
//     C##::C## (RefTuple& t) ##D                    \
//     void C##::operator = (Tuple& t) ##E        \
//     void C##::operator = (RefTuple& t) ##E    
// #define ASSIGN_1    : i(t.i), j(t.j) {}
// #define ASSIGN_2    { i = t.i; j = t.j; }
// #define SORT_CRITERIA \
//     return (j < t.j) || (j == t.j && (i < t.i));
// struct Tuple {
//     int i, j;
//     TUPLE_COMMON_FUNC(Tuple, ; , ;)
// };
// struct RefTuple {
//     int &i, &j;
//     RefTuple(int &x, int &y): i(x), j(y) {}
//     TUPLE_COMMON_FUNC(RefTuple, ; , ;)
// };
// TUPLE_COMMON_FUNC(Tuple, ASSIGN_1, ASSIGN_2, {SORT_CRITERIA})
// TUPLE_COMMON_FUNC(RefTuple, ASSIGN_1, ASSIGN_2, {SORT_CRITERIA})

// void swap(RefTuple& t1, RefTuple& t2) {
//     t1.i ^= t2.i; t2.i ^= t1.i; t1.i ^= t2.i;
//     t1.j ^= t2.j; t2.j ^= t1.j; t1.j ^= t2.j;
// }

// class IterTuple : public iterator<random_access_iterator_tag, Tuple> {
//     int *i, *j, idx;
// public:
//     IterTuple(int* x, int*y, int l) : i(x), j(y), idx(l) {}
//     IterTuple(const IterTuple& e) : i(e.i), j(e.j), idx(e.idx) {}
//     RefTuple operator*() { return RefTuple(i[idx], j[idx]);  }
//     IterTuple& operator ++ () { idx++; return *this; }
//     IterTuple& operator -- () { idx--; return *this; }
//     IterTuple operator ++ (int) { IterTuple tmp(*this); idx++; return tmp; }
//     IterTuple operator -- (int) { IterTuple tmp(*this); idx--; return tmp; }
//     int operator - (IterTuple& rhs) { return idx - rhs.idx;    }
//     IterTuple operator + (int n) { IterTuple tmp(*this); tmp.idx += n; return tmp; }
//     IterTuple operator - (int n) { IterTuple tmp(*this); tmp.idx -= n; return tmp; }
//     bool operator==(const IterTuple& rhs) {        return idx == rhs.idx;    }
//     bool operator!=(const IterTuple& rhs) {     return idx != rhs.idx;  }
//     bool operator<(IterTuple& rhs) {     return idx < rhs.idx;   }
// };



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

                // num_literals is placeholder. Need change active_literals_size to be user defined
                state.active_literals_size = state.num_literals;
                state.active_literals.resize(state.num_classes);
                for (int i = 0; i < state.num_classes; ++i)
                {
                    state.active_literals[i].reserve(state.active_literals_size);
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
                    
                    SparseLiterals* pos_active_literals = &state.active_literals[positive_class];
                    SparseLiterals* neg_active_literals = &state.active_literals[negative_class];

                    WeightInt* clause_weights = &state.clause_weights[clause_k * state.num_classes];
                    

                    if (state.fast_rng.next_u() < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, pos_active_literals, neg_active_literals, clause_weights + positive_class, 1, literals, state.clause_outputs[clause_k]);
                    }
                    else if (state.fast_rng.next_u() < prob_negative)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, pos_active_literals, neg_active_literals, clause_weights + negative_class, -1, literals, state.clause_outputs[clause_k]);

                    }
                }

            }
    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* pos_active_literals, SparseLiterals* neg_active_literals, WeightInt* clause_weight, int target, SparseLiterals* literals, ClauseOutputUint clause_output)
            {
                int32_t sign = (*clause_weight) >= 0 ? +1 : -1;

                if ( (target * sign) > 0)
                {
                    if (clause_output == 1)
                    {
                        (*clause_weight) += sign;

                        _T1aFeedback t1a;
                        t1a(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, pos_active_literals, neg_active_literals, literals);
                        prune_clause(state, pos_clause_row, pos_clause_states);
                        prune_clause(state, neg_clause_row, neg_clause_states);
                    }
                    else
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
                    t2(state, pos_clause_row, neg_clause_row, pos_clause_states, neg_clause_states, pos_active_literals, neg_active_literals, literals);
                    sort_clauses_and_states(state, pos_clause_row, pos_clause_states);
                    sort_clauses_and_states(state, neg_clause_row, neg_clause_states);
                }

            }
        private:
            void prune_clause(_State& state, SparseClause* clause_row, SparseClauseStates* clause_states)
            {
                // Function to remove automata from clauses if state is below threshold. clause_row and states remains sorted

                for (int ta_k = 0; ta_k < clause_row->size(); ++ta_k)
                {
                    if (clause_states->at(ta_k) <= state.lower_ta_threshold)
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
                for (int i = 0; i < clause_row->size(); i++)
                {
                    for (int j = 0; j < clause_row->size() - i - 1; j++)
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
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* pos_active_literals, SparseLiterals* neg_active_literals, SparseLiterals* literals)
            {
                const double s_inv = 1.0 / state.s;
                const double s_min1_inv = (state.s - 1.0) / state.s;

                const int8_t lower_state = -127;
                const int8_t upper_state = 127;

                _UpdateAL update_al;

                // loop literals
                for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                {
                    const int8_t literal = literals->at(lit_k);
                    int8_t literal_state_pos = 1;
                    int8_t literal_state_neg = 1;

                    // loop pos clauses
                    for (int ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
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
                                    if (state.fast_rng.next_u() <= s_min1_inv)
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
                    if (literal_state_pos == 1)
                    {
                        update_al(state, pos_active_literals, literal);
                    }

                    // loop neg clauses
                    for (int ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                    {
                        if (neg_clause_row->at(ta_k) == literal)
                        {
                            if ((neg_clause_states->at(ta_k) > lower_state) && (state.fast_rng.next_u() <= s_inv))
                            {
                                neg_clause_states->at(ta_k) -= 1;
                            }
                            literal_state_neg = 0;
                            break; // updated neg_ta_state for current lit, go next lit
                        }
                    }
                    // no ta for literal found in clause, add to AL
                    if (literal_state_neg == 1)
                    {
                        update_al(state, neg_active_literals, literal);
                    }
                }


                // loop pos clauses
                for (int ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {

                    if (pos_clause_states->at(ta_k) <= 0)
                    {
                        continue;
                    }

                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (pos_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_pos;
                        }
                    }
                    pos_clause_states->at(ta_k) -= 1;

                    endloop_pos:;
                }

                // loop neg clauses
                for (int ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {
                    
                    // check if this is right
                    if (neg_clause_states->at(ta_k) <= 0)
                    {
                        continue;
                    }

                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (neg_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_neg;
                        }
                    }
                    neg_clause_states->at(ta_k) += 1;

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
                const double s_inv = 1.0 / state.s;
                const int8_t lower_state = -127;


                // loop pos clauses
                for (int ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {
                    if (state.fast_rng.next_u() <= s_inv)
                    {
                        if (pos_clause_states->at(ta_k) > lower_state)
                            pos_clause_states->at(ta_k) -= 1;
                    }
                }

                // loop neg clauses
                for (int ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {
                    if (state.fast_rng.next_u() <= s_inv)
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
            void operator()(_State& state, SparseClause* pos_clause_row, SparseClause* neg_clause_row, SparseClauseStates* pos_clause_states, SparseClauseStates* neg_clause_states, SparseLiterals* pos_active_literals, SparseLiterals* neg_active_literals, SparseLiterals* literals)
            {
                // loop pos clauses, check if ta is not in literals, if so, increment ta state if its above threshold
                for (int ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                {
                    // if ((pos_clause_states->at(ta_k) <= state.lower_ta_threshold))
                    // {
                    //     continue;
                    // }
                
                    // loop literals, if we find ta in literals, skip increment
                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (pos_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            goto endloop_pos;
                        }
                    }
                    pos_clause_states->at(ta_k) += 1;

                    endloop_pos:;
                }

                for (int ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                {
                    // if ((neg_clause_states->at(ta_k) <= state.lower_ta_threshold))
                    // {
                    //     continue;
                    // }
                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (neg_clause_row->at(ta_k) == literals->at(lit_k))
                        {
                            neg_clause_states->at(ta_k) += 1;
                            goto endloop_neg;
                        }
                    }

                    endloop_neg:;
                }

                for (int lit_k = 0; lit_k < pos_active_literals->size(); ++lit_k)
                {
                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (pos_active_literals->at(lit_k) == literals->at(lit_k))
                        {
                            goto endloop_pos_al;
                        }
                    }
                    for (int ta_k = 0; ta_k < pos_clause_row->size(); ++ta_k)
                    {
                        if (pos_active_literals->at(lit_k) == pos_clause_row->at(ta_k))
                        {
                            goto endloop_pos_al;
                        }
                    }

                    pos_clause_row->push_back(pos_active_literals->at(lit_k));
                    pos_clause_states->push_back(state.lower_ta_threshold + 5);

                    endloop_pos_al:;
                }

                for (int lit_k = 0; lit_k < neg_active_literals->size(); ++lit_k)
                {
                    for (int lit_k = 0; lit_k < literals->size(); ++lit_k)
                    {
                        if (neg_active_literals->at(lit_k) == literals->at(lit_k))
                        {
                            goto endloop_neg_al;
                        }
                    }
                    for (int ta_k = 0; ta_k < neg_clause_row->size(); ++ta_k)
                    {
                        if (neg_active_literals->at(lit_k) == neg_clause_row->at(ta_k))
                        {
                            goto endloop_neg_al;
                        }
                    }

                    neg_clause_row->push_back(neg_active_literals->at(lit_k));
                    neg_clause_states->push_back(state.lower_ta_threshold + 5);

                    endloop_neg_al:;
                }

            }
    };

    template <typename _State>
    class UpdateAL
    {
        public:
            void operator()(_State& state, SparseLiterals* active_literals_class_k, int8_t literal) // active_literals_class_k might need to be pointer
            {
                // Function to update active literals

                if (active_literals_class_k->size() < state.active_literals_size)
                {
                    // check if literal is already in active_literals
                    for (int i = 0; i < active_literals_class_k->size(); ++i)
                    {
                        if (active_literals_class_k->at(i) == literal)
                        {
                            return;
                        }
                    }
                    active_literals_class_k->push_back(literal);
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

}; // namespace  green_tsetlin




#endif // #ifndef _FUNC_SPARSE_HPP_