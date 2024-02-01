#ifndef _FUNC_SPARSE_HPP_
#define _FUNC_SPARSE_HPP_

#include <circular_buffer.hpp>
#include <gt_common.hpp>

namespace green_tsetlin
{
    // TODO: set these types correctly 
    // either": unordered_set / unordered_map / vector / set / map    
    typedef typename std::unordered_set<uint32_t> SparseLiterals;  // used in SparseInput

    typedef typename std::unordered_map<uint32_t, int8_t> SparseClause;

    

    // a clause index is type: uint32_t

    class CoaleasedTsetlinStateSparseNV
    {
        public:
            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;
            int num_literals = 0;

            int literal_buffer_size = 0;
            int literal_budget = 0;
            
            //std::vector<SparseClause> sparse_clauses;
            CircularBuffer<uint32_t>  active_literals;

            std::default_random_engine rng;
            int8_t* clauses = nullptr;
            int8_t* clause_outputs = nullptr;
            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;

            inline void set_s(double s_param) { s = s_param; }
            inline double get_s() const { return s; }


            inline int8_t get_ta_state(int clause_k, int ta_i, bool ta_polarity)
            {
                return 0;
            }

            inline void set_ta_state(int clause_k, int ta_i, bool ta_polarity, int8_t new_state)
            {
            }

            inline WeightInt get_clause_weight(int clause_index, int target_class)
            {
                return 0;
            }

            inline void set_clause_weight(int clause_index, int target_class, int32_t new_weight)
            {
            }

            inline WeightInt* get_class_votes() const
            {
                // just return the class votes
                return class_votes;
            }

            // ---
            inline void get_clause_state(int8_t* dst, int clause_offset)
            {
            }

            inline void set_clause_state(int8_t* src, int clause_offset)
            {
            }

            inline void set_clause_weights(WeightInt* src, int clause_offset)
            {
            }

            inline void get_clause_weights(WeightInt* dst, int clause_offset)
            {
            }

    };

    template <typename _State>
    class InitializeSparseNV
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {
                state.rng.seed(seed);

                init_clauses(state);
                init_clause_weights(state);

                return true;
            }

        
        private:
            void init_clauses(_State& state)
            {
            }

            void init_clause_weights(_State& state)
            {
            }        
    };

    template <typename _State>
    class CleanupSparseNV
    {
        public:                        
            void operator()(_State& state)
            {
            }
    };



    template <typename _State>
    class SetClauseOutputSparseNV
    {
        public: 
            void operator()(_State& state, const SparseLiterals* literals)
            {
            }
    };
    
    template <typename _State>
    class EvalClauseOutputSparseNV
    {
        public:
            void operator()(_State& state, const SparseLiterals* literals)
            {
            }
    };

    template <typename _State, typename _SparseClauseUpdate>
    class TrainUpdateSparseNV
    {
        public:
            void operator()(_State& state, const SparseLiterals* literals, int positive_class, double prob_positive, int negative_class, double prob_negative)
            {      
            }
    };

    template <typename _State, typename _T1Feedback, typename _T2Feedback>
    class ClauseUpdateSparseNV
    {
        public:
            void operator()(_State& state, int clause_k, int32_t* clause_weight, int target, const SparseLiterals* literals, int8_t clause_output)
            {
            }
    };


    template <typename _State>
    class Type1FeedbackSparseNV
    {
        public:
            void operator()(_State& state, int clause_k, int8_t clause_output, const SparseLiterals* literals)
            {
            }
    };


    template <typename _State>
    class Type2FeedbackSparseNV
    {
        public:
            // Assume that clause_output == 1
            void operator()(_State& state, int clause_k, const SparseLiterals* literals)
            {               
            }
    };
}; // namespace green_tsetlin




#endif // #ifndef _FUNC_SPARSE_HPP_