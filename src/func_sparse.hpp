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

                // need set size for each, now there is num_clauses empty vectors
                state.sparse_clauses.reserve(state.num_clauses);

                state.clause_outputs = new ClauseOutputUint[state.num_clauses];
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * state.num_clauses);

                state.class_votes = new WeightInt[state.num_classes];
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);

                state.clause_weights = new WeightInt[state.num_clauses * state.num_classes];

                // init clauses and clause weigths like in func_tm?

                return true;
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
            }
    };

    template <typename _State, typename _ClauseUpdate, bool do_literal_budget>
    class TrainUpdateSparseTM
    {
        public:
            void operator()(_State& state, SparseLiterals* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
            }
    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, int8_t* clause_row, WeightInt* clause_weight, int target, SparseLiterals* literals, ClauseOutputUint clause_output)
            {
            }
    };

    template <typename _State, bool boost_true_positive>
    class Type1aFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, int8_t* clause_row, SparseLiterals* literals)
            {
            }
    };

    template <typename _State>
    class Type1bFeedbackSparseTM
    {
        public:
            // TODO: clause_row needs a sparse type
            void operator()(_State& state, int8_t* clause_row)
            {
            }
    };


    template <typename _State>
    class Type2FeedbackSparseTM
    {
        public:
            // Assume that clause_output == 1 (check ClauseUpdate)
            // // TODO: clause_row needs a sparse type
            void operator()(_State& state, int8_t* clause_row, SparseLiterals* literals)
            {
            }
    };

}; // namespace  green_tsetlin




#endif // #ifndef _FUNC_SPARSE_HPP_