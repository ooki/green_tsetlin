#ifndef _FUNC_UPDATE_HPP_
#define _FUNC_UPDATE_HPP_



#include <random>
#include <vector>

#include <gt_common.hpp>


namespace green_tsetlin
{
    class CoaleasedTsetlinStateNV
    {
        public:
            constexpr const static uint32_t high_number_if_no_positive_literal_is_present = 65000;


            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;
            int num_literals = 0;
            int num_literals_mem = 0; 

            std::default_random_engine rng;
            int8_t* clauses = nullptr;
            ClauseOutputUint* clause_outputs = nullptr;
            WeightInt* class_votes = nullptr;
            WeightInt* clause_weights = nullptr;

            uint32_t* literal_counts = nullptr;
            uint32_t literal_budget = 0xFFFF;

            inline void set_s(double s_param) { s = s_param; }
            inline double get_s() const { return s; }

            inline int8_t get_ta_state(int clause_k, int ta_i, bool ta_polarity)
            {
                if(ta_polarity)
                    return clauses[(clause_k * num_literals * 2) + ta_i];
                else
                    return clauses[(clause_k * num_literals * 2) + num_literals + ta_i];
            }

            inline void set_ta_state(int clause_k, int ta_i, bool ta_polarity, int8_t new_state)
            {
                if(ta_polarity)
                    clauses[(clause_k * num_literals * 2) + ta_i] = new_state;
                else
                    clauses[(clause_k * num_literals * 2) + num_literals + ta_i] = new_state;
            }

            inline WeightInt get_clause_weight(int clause_index, int target_class)
            {
                //std::cout << "get_clause_weight(NV):" << clause_index << ", " << target_class << std::endl;
                //int index = (clause_index * num_classes) + target_class;
                //std::cout << "@:" << index << std::endl;
                //clause_index << ", " << target_class << " num_classes:" << num_classes << std::endl;
                return clause_weights[(clause_index * num_classes) + target_class];
            }

            inline void set_clause_weight(int clause_index, int target_class, int32_t new_weight)
            {
                clause_weights[(clause_index * num_classes) + target_class] = new_weight;
            }

            inline WeightInt* get_class_votes() const
            {
                return class_votes;
            }

            inline std::vector<int8_t> get_copy_clauses() const
            {                
                std::size_t n_total_literals = num_clauses * (num_literals*2);                
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
                const size_t state_size = num_clauses * (num_literals * 2);
                const size_t dst_offset = clause_offset * (num_literals * 2);

                memcpy(&dst[dst_offset], clauses, state_size);
            }

            inline void set_clause_state(int8_t* src, int clause_offset)
            {
                const size_t state_size = num_clauses * num_literals * 2;
                const size_t src_offset = clause_offset * (num_literals * 2);
                memcpy(clauses, &src[src_offset], state_size);
            }

            inline void get_clause_weights(WeightInt* dst, int clause_offset) 
            {
                const size_t weight_mem = num_clauses * num_classes * sizeof(WeightInt);
                const size_t dst_offset = (clause_offset * num_classes);
                memcpy(&dst[dst_offset], clause_weights, weight_mem);
            }

            inline void set_clause_weights(WeightInt* src, int clause_offset)
            {
                const size_t weight_mem = num_clauses * num_classes * sizeof(WeightInt);
                const size_t src_offset = (clause_offset * num_classes);

                memcpy(clause_weights, &src[src_offset], weight_mem);
            }
    };

    template <typename _State, bool do_literal_budget>
    class InitializeNV
    {
        public:                        
            bool operator()(_State& state, unsigned int seed)
            {
                int clause_mem = state.num_clauses * state.num_literals * 2;
                state.clauses = new int8_t[clause_mem];

                state.num_literals_mem = state.num_literals;

                state.clause_outputs = new ClauseOutputUint[state.num_clauses];                
                memset(state.clause_outputs, 0, sizeof(ClauseOutputUint) * state.num_clauses);
                
                state.clause_weights = new WeightInt[state.num_clauses * state.num_classes];

                state.class_votes = new WeightInt[state.num_classes];
                memset(state.class_votes, 0, sizeof(WeightInt) * state.num_classes);

                if(do_literal_budget)
                    state.literal_counts = new uint32_t[state.num_clauses];

                state.rng.seed(seed);

                init_clauses(state);
                init_clause_weights(state);

                return true;
            }

        
        private:
            void init_clauses(_State& state)
            {
                std::uniform_int_distribution<int8_t> dist(-1, 0);

                const int num_literals_total = state.num_clauses * (state.num_literals*2);                
                for(int literal_k = 0; literal_k < num_literals_total; ++literal_k)
                {
                    state.clauses[literal_k] = dist(state.rng);
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

    template <typename _State, bool do_literal_budget>
    class CleanupNV
    {
        public:                        
            void operator()(_State& state)
            {
                delete[] state.clauses;
                state.clauses = nullptr;     

                delete[] state.class_votes;
                state.class_votes = nullptr;

                delete[] state.clause_weights;
                state.clause_weights = nullptr;

                if(do_literal_budget)
                {
                    delete[] state.literal_counts;
                    state.literal_counts = nullptr;
                }
            }
    };


    

    template <typename _State, bool do_literal_budget, bool force_at_least_one_positive_literal>
    class SetClauseOutputNV
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
                }
            }
    };

    template <typename _State>
    class EvalClauseOutputNV
    {
        public:
            void operator()(_State& state, uint8_t* literals)
            {
                for(int clause_k = 0; clause_k < state.num_clauses; ++clause_k)
                {                    
                    state.clause_outputs[clause_k] = 1;
                    bool is_empty_clause = true;

                    int8_t* pl_pos = &state.clauses[clause_k * (state.num_literals*2)];
                    int8_t* pl_neg = &state.clauses[(clause_k * (state.num_literals*2)) + state.num_literals];

                    for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                    {                        
                        if(*pl_pos >= 0) 
                        {
                            is_empty_clause = false;
                            if(literals[literal_k] == 0)
                            {
                                state.clause_outputs[clause_k] = 0;
                                break;
                            }
                        }
                        pl_pos++;

                        if(*pl_neg >= 0)
                        {
                            is_empty_clause = false;
                            if(literals[literal_k] == 1) 
                            {
                                state.clause_outputs[clause_k] = 0;
                                break;
                            }
                        }
                        pl_neg++;                                                                     
                    }

                    if(is_empty_clause)
                        state.clause_outputs[clause_k] = 0;
                }
            }
    };

    template <typename _State>
    class EmptyCountVotesNV
    {
        public:
            void operator()(_State& state) {}
    };

    template <typename _State>
    class CountVotesNV
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
    class TrainUpdateNV
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

                    if(do_literal_budget)
                    {
                        if(state.literal_counts[clause_k] > state.literal_budget)
                            state.clause_outputs[clause_k] = 0;
                    }

                    if( u(state.rng) < prob_positive)
                    {
                        _ClauseUpdate clause_update;
                        clause_update(state, clause_row, clause_weights + positive_class, 1, literals, state.clause_outputs[clause_k]);                    
                    }
      
                    if( u(state.rng) < prob_negative)
                    {
                        _ClauseUpdate update_clause;
                        update_clause(state, clause_row, clause_weights + negative_class, -1, literals, state.clause_outputs[clause_k]);
                    }
                }
            }

    };

    template <typename _State, typename _T1aFeedback, typename _T1bFeedback, typename _T2Feedback>
    class ClauseUpdateNV
    {
        public:
            void operator()(_State& state, int8_t* clause_row, WeightInt* clause_weight, int target, uint8_t* literals, ClauseOutputUint clause_output)
            {
                int32_t sign = (*clause_weight) >= 0 ? +1 : -1;                
                
                if( (target * sign) > 0)
                {
                    if(clause_output == 1)
                    {
                        (*clause_weight) += sign;

                        _T1aFeedback    t1a;
                        t1a(state, clause_row, literals);
                    }
                    else
                    {
                        _T1bFeedback    t1b;
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
    class Type1aFeedbackNV
    {
        public:
            void operator()(_State& state, int8_t* clause_row, uint8_t* literals)
            {
                constexpr bool use_boost_true_positive = false;

                std::uniform_real_distribution<double> u(0.0,1.0);
                
                const double s_inv = (1.0 / state.s);
                const double s_min1_inv = (state.s - 1.0) / state.s;

                const int8_t lower_state = -127;
                const int8_t upper_state =  127;
                
                int8_t* neg_clause_row = &clause_row[state.num_literals]; 
                               
                for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                {
                    if(literals[literal_k] == 1)
                    {
                        
                        // POS 2
                        if(use_boost_true_positive)
                        {
                            if(clause_row[literal_k] < upper_state)
                                clause_row[literal_k] += 1;
                        }
                        else
                        {
                            if( u(state.rng) <= s_min1_inv)
                            {
                                if(clause_row[literal_k] < upper_state)
                                    clause_row[literal_k] += 1;
                            }
                        }
                        

                        // NEG 3
                        if( u(state.rng) <= s_inv )
                        {
                            if(neg_clause_row[literal_k] > lower_state)
                                neg_clause_row[literal_k] -= 1;
                        }
                    }
                    else
                    {
                        // POS 3
                        if( u(state.rng) <= s_inv )
                        {
                            if(clause_row[literal_k] > lower_state)
                                clause_row[literal_k] -= 1;
                        }

                        // NEG 2
                        if( u(state.rng) <= s_min1_inv)
                        {
                            if(neg_clause_row[literal_k] < upper_state)
                                neg_clause_row[literal_k] += 1;
                        }
                    }
                }
            }
    };

     template <typename _State>
    class Type1bFeedbackNV
    {
        public:
            void operator()(_State& state, int8_t* clause_row, uint8_t* literals)
            {
                std::uniform_real_distribution<double> u(0.0,1.0);
                
                const double s_inv = (1.0 / state.s);
                const int8_t lower_state = -127;
                
                int8_t* neg_clause_row = &clause_row[state.num_literals]; 
                for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                {
                    // POS 1
                    if(u(state.rng) <= s_inv)
                    {
                        if(clause_row[literal_k] > lower_state)
                            clause_row[literal_k] -= 1;
                        
                    }

                    // NEG 1
                    if(u(state.rng) <= s_inv)
                    {
                        if(neg_clause_row[literal_k] > lower_state)
                            neg_clause_row[literal_k] -= 1;
                        
                    }
                }
                
            }
    };


    template <typename _State>
    class Type2FeedbackNV
    {
        public:
            // Assume that clause_output == 1
            void operator()(_State& state, int8_t* clause_row, uint8_t* literals)
            {
                int8_t* neg_clause_row = &clause_row[state.num_literals];                
                for(int literal_k = 0; literal_k < state.num_literals; ++literal_k)
                {
                    if(literals[literal_k] == 0)
                    {
                        if(clause_row[literal_k] < 0)
                            clause_row[literal_k] += 1;
                        
                    }
                    else
                    {                    
                        if(neg_clause_row[literal_k] < 0)                        
                            neg_clause_row[literal_k] += 1;                        
                    }
                }
            }
    };

    
}; // namespace name





#endif // #ifndef _FUNC_UPDATE_HPP_