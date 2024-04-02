#ifndef __CLAUSE_BLOCK_H_
#define __CLAUSE_BLOCK_H_

#include <stdint.h>
#include <vector>
#include <iostream>

namespace green_tsetlin
{
    class FeedbackBlock;    
    class InputBlock;
    class ClauseBlock // TODO: once finished move into the NV class.
    {
        public:
            explicit ClauseBlock()
            {
                m_is_init = false;
            }
            virtual ~ClauseBlock() {}

            virtual int get_number_of_literals() const { return -42; }
            virtual int get_number_of_clauses() const { return -42; }
            virtual int get_number_of_classes() const { return -42; }

            virtual int get_number_of_patches_per_example() const { return -42; }
            virtual void set_number_of_patches_per_example(int num_patches_per_example) {}

            void set_trainable(bool is_trainable) { m_is_trainable = is_trainable; }
            bool is_trainable() const { return m_is_trainable; }

            bool is_init() const { return m_is_init; }

            virtual double  get_s() const { return -42.0; }
            virtual void    set_s(double s) {}
            
            virtual uint32_t get_literal_budget() { return 0; }
            virtual void     set_literal_budget(uint32_t budget) {}

            virtual bool initialize(unsigned int seed = 42)
            {
                m_is_init = false;
                return false;
            }

            virtual void cleanup()
            {       
            }

            void set_feedback(FeedbackBlock* feedback_block)
            {
                m_feedback_block = feedback_block;
            }

            FeedbackBlock* get_feedback()
            {                            
                return m_feedback_block;
            }

            virtual InputBlock* get_input_block() { return nullptr; }
            virtual void pull_example() {}
            
            // virtual void train_set_clause_output_and_set_votes() {}
            // virtual void eval_set_clause_output_and_set_votes() {}

            virtual void train_example()
            {
                pull_example();
                train_set_clause_output();
                set_votes();
            }

            virtual void eval_example()
            {
                pull_example();
                eval_set_clause_output();
                set_votes();
            }
            
            virtual void train_set_clause_output() {}
            virtual void eval_set_clause_output() {}
            virtual void set_votes() {}
            
            
            virtual void train_update(int positive_class, double prob_positive, int negative_class, double prob_negative) {}

        protected:
            bool m_is_trainable = true;
            bool m_is_init = false;
            FeedbackBlock* m_feedback_block = nullptr;
    };

    
    template <typename _State,
              typename _Initializer,
              typename _Cleanup,
              typename _TrainSetClauseOutput,
              typename _EvalClauseOutput,
              typename _CountVotes,
              typename _TrainUpdate,
              typename _InputBlock>
    class ClauseBlockT : public ClauseBlock
    {
        public:
            typedef _State StateType;

            explicit ClauseBlockT(int num_literals, int num_clauses, int num_classes)
            {            
                m_state.num_literals = num_literals;
                m_state.num_clauses = num_clauses;
                m_state.num_classes = num_classes;                    
            }
            
            virtual int get_number_of_literals() const { return m_state.num_literals; }
            virtual int get_number_of_clauses() const { return m_state.num_clauses; }
            virtual int get_number_of_classes() const { return m_state.num_classes; }

            virtual int get_number_of_patches_per_example() const { return m_state.get_number_of_patches_per_example(); }
            virtual void set_number_of_patches_per_example(int num_patches_per_example){ m_state.set_number_of_patches_per_example(num_patches_per_example); }

            virtual double get_s() const { return m_state.get_s(); }
            virtual void set_s(double s) { m_state.set_s(s); }

            virtual uint32_t get_literal_budget() { return m_state.literal_budget; }
            virtual void    set_literal_budget(uint32_t budget) { m_state.literal_budget = budget; }

            virtual void get_clause_state_npy(pybind11::array_t<int8_t> out_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;

                int8_t* p = static_cast<int8_t*>(buffer_info.ptr); 
                m_state.get_clause_state(p, clause_offset);
            }

            virtual void set_clause_state_npy(pybind11::array_t<int8_t> in_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = in_array.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;

                int8_t* p = static_cast<int8_t*>(buffer_info.ptr);                
                m_state.set_clause_state(p, clause_offset);  
            }

            virtual void get_clause_weights_npy(pybind11::array_t<WeightInt> out_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                WeightInt* p = static_cast<WeightInt*>(buffer_info.ptr);                
                m_state.get_clause_weights(p, clause_offset);
            }

            virtual void set_clause_weights_npy(pybind11::array_t<WeightInt> in_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = in_array.request();                            
                WeightInt* p = static_cast<WeightInt*>(buffer_info.ptr);                
                m_state.set_clause_weights(p, clause_offset);
            }

            virtual bool initialize(unsigned int seed = 42)
            {
                _Initializer f;
                m_is_init = f(m_state, seed);
                return m_is_init;
            }

            virtual void cleanup()
            {
                _Cleanup f;
                f(m_state);
                m_is_init = false;
            }

            void set_input_block(_InputBlock* ib)
            {
                m_input_block = ib;
            }

            virtual void pull_example()
            {
                m_literals = m_input_block->pull_current_example();
            }

            uint8_t* get_current_literals()
            {
                return m_literals;
            }

            virtual void set_votes()
            {
                _CountVotes vote_counter;
                vote_counter(m_state);

                m_feedback_block->register_votes(m_state.get_class_votes());
            }

            virtual void train_set_clause_output()
            {
                _TrainSetClauseOutput set_clause_output;
                set_clause_output(m_state, m_literals);
                
            }            

            virtual void eval_set_clause_output()
            {
                _EvalClauseOutput eval_clause_output;
                eval_clause_output(m_state, m_literals);
            }

            virtual void train_update(int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                _TrainUpdate f;
                f(m_state, m_literals, positive_class, prob_positive, negative_class, prob_negative);
            }

            void get_clause_output_npy(pybind11::array_t<uint8_t> out_array)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                uint8_t* out = static_cast<uint8_t*>(buffer_info.ptr);                
                for(int i = 0; i < m_state.num_clauses; ++i)
                    out[i] = m_state.clause_outputs[i];
            }
            
            virtual InputBlock* get_input_block() { return m_input_block; }
            _State& get_state()
            {
                return m_state;
            }

        protected:
            _InputBlock* m_input_block;
            typename _InputBlock::example_type* m_literals = nullptr;

            _State  m_state;
         
    };
   

    template <typename _State,
              typename _Initializer,
              typename _Cleanup,
              typename _TrainSetClauseOutput,
              typename _EvalClauseOutput,
              typename _CountVotes,
              typename _TrainUpdate,
              typename _InputBlock>
    class ClauseBlockSparseT : public ClauseBlock
    {
        public:
            typedef _State StateType;

            explicit ClauseBlockSparseT(int num_literals, int num_clauses, int num_classes)
            {            
                m_state.num_literals = num_literals;
                m_state.num_clauses = num_clauses;
                m_state.num_classes = num_classes; 
                
                // default to clause_size being num_literals
                m_state.clause_size = num_literals;      
                m_state.active_literals_size = num_literals; 
                m_state.lower_ta_threshold = -20;            
            }
            
            virtual int get_number_of_literals() const { return m_state.num_literals; }
            virtual int get_number_of_clauses() const { return m_state.num_clauses; }
            virtual int get_number_of_classes() const { return m_state.num_classes; }

            virtual int get_number_of_patches_per_example() const { return m_state.get_number_of_patches_per_example(); }
            virtual void set_number_of_patches_per_example(int num_patches_per_example){ m_state.set_number_of_patches_per_example(num_patches_per_example); }

            virtual double get_s() const { return m_state.get_s(); }
            virtual void set_s(double s) { m_state.set_s(s); }

            virtual uint32_t get_literal_budget() { return m_state.literal_budget; }
            virtual void    set_literal_budget(uint32_t budget) { m_state.literal_budget = budget; }

            
            virtual int get_lower_ta_threshold() { return m_state.lower_ta_threshold; }
            virtual void set_lower_ta_threshold(int lower_ta_threshold) { m_state.lower_ta_threshold = lower_ta_threshold; }

            virtual int get_active_literals_size() { return (int)m_state.active_literals_size; }
            virtual void set_active_literals_size(int active_literals_size) { m_state.active_literals_size = (size_t)active_literals_size; }

            virtual int get_clause_size() { return (int)m_state.clause_size; }
            virtual void set_clause_size(int clause_size) { m_state.clause_size = (size_t)clause_size; }


            virtual pybind11::list get_clause_state_sparse_npy() // pybind11::array_t<int8_t> out_array, int clause_offset
            {
                // create py::list containing np.array of clause states, data=states, indices = column_indices, indptr = row_offsets
                
                std::vector<int8_t> data;
                std::vector<uint32_t> indices;
                std::vector<uint32_t> indptr;

                int offset = 0;

                for (int i = 0; i < m_state.num_clauses*2; ++i)
                {
                    indptr.push_back(offset);
                    for (size_t ta_k = 0; ta_k < m_state.clauses[i].size(); ++ta_k)
                    {
                        data.push_back(m_state.clause_states[i][ta_k]);
                        indices.push_back(m_state.clauses[i][ta_k]);
                    }
                    offset += m_state.clauses[i].size();
                }
                
                indptr.push_back(offset);

                pybind11::list out;
                pybind11::array_t<int8_t> data_array = pybind11::cast(data);
                pybind11::array_t<uint32_t> indices_array = pybind11::cast(indices);
                pybind11::array_t<uint32_t> indptr_array = pybind11::cast(indptr);
                out.append(data_array);
                out.append(indices_array);
                out.append(indptr_array);

                return out;

            }


            virtual void set_clause_state_sparse_npy(pybind11::array data, pybind11::array indices, pybind11::array indptr)
            {
                // fill clauses and clause_states with data, indices and indptr

                pybind11::buffer_info states_info = data.request();
                int8_t* data_p = static_cast<int8_t*>(states_info.ptr);

                pybind11::buffer_info indices_info = indices.request();
                uint32_t* indices_p = static_cast<uint32_t*>(indices_info.ptr);

                pybind11::buffer_info indptr_info = indptr.request();
                std::vector<long int> shape_indptr = indptr_info.shape;
                int _num_clauses = shape_indptr[0] - 1;
                uint32_t* indptr_p = static_cast<uint32_t*>(indptr_info.ptr);


                std::vector<std::vector<int8_t>> clause_states;
                std::vector<std::vector<uint32_t>> clauses;
                clauses.resize(_num_clauses);
                clause_states.resize(_num_clauses);


                for (int i = 0; i < _num_clauses; ++i)
                {
                                    
                    for (uint32_t j = indptr_p[i]; j < indptr_p[i+1]; ++j)
                    {
                        clause_states[i].push_back(data_p[j]);
                        clauses[i].push_back(indices_p[j]);
                    }
                    
                }
                m_state.clauses = clauses;
                m_state.clause_states = clause_states;

            }


            virtual pybind11::list get_active_literals_npy()
            {
                pybind11::list out;
                
                for (int i = 0; i < m_state.num_classes; ++i)
                {
                    std::vector<uint32_t> temp_active_literals = m_state.active_literals[i];
                    pybind11::array_t<uint32_t> to_add = pybind11::cast(temp_active_literals);
                    out.append(to_add);
                }
                
                return out;
            }

            virtual void set_active_literals_npy(pybind11::array_t<uint32_t> in_array)
            {
                pybind11::buffer_info buffer_info = in_array.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;

                uint32_t* p = static_cast<uint32_t*>(buffer_info.ptr);                
                for (int i = 0; i < m_state.num_classes; ++i)
                {
                    m_state.active_literals[i].clear();
                    m_state.active_literals[i] = std::vector<uint32_t>(p, p + shape[1]);
                    p += shape[1];
                }
            }

            virtual void get_clause_weights_npy(pybind11::array_t<WeightInt> out_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                WeightInt* p = static_cast<WeightInt*>(buffer_info.ptr);                
                m_state.get_clause_weights(p, clause_offset);
            }

            virtual void set_clause_weights_npy(pybind11::array_t<WeightInt> in_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = in_array.request();                            
                WeightInt* p = static_cast<WeightInt*>(buffer_info.ptr);                
                m_state.set_clause_weights(p, clause_offset);
            }

            virtual bool initialize(unsigned int seed = 42)
            {
                _Initializer f;
                m_is_init = f(m_state, seed);
                return m_is_init;
            }

            virtual void cleanup()
            {
                _Cleanup f;
                f(m_state);
                m_is_init = false;
            }

            void set_input_block(_InputBlock* ib)
            {
                m_input_block = ib;
            }

            virtual void pull_example()
            {
                m_literals = m_input_block->pull_current_example();
            }

            typename _InputBlock::example_type* get_current_literals()
            {
                return m_literals;
            }

            virtual void set_votes()
            {
                _CountVotes vote_counter;
                vote_counter(m_state);

                m_feedback_block->register_votes(m_state.get_class_votes());
            }

            virtual void train_set_clause_output()
            {
                _TrainSetClauseOutput set_clause_output;
                set_clause_output(m_state, m_literals);
            }            

            virtual void eval_set_clause_output()
            {
                _EvalClauseOutput eval_clause_output;
                eval_clause_output(m_state, m_literals);
            }

            virtual void train_update(int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                _TrainUpdate f;
                f(m_state, m_literals, positive_class, prob_positive, negative_class, prob_negative);
            }

            void get_clause_output_npy(pybind11::array_t<uint8_t> out_array)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                uint8_t* out = static_cast<uint8_t*>(buffer_info.ptr);                
                for(int i = 0; i < m_state.num_clauses; ++i)
                    out[i] = m_state.clause_outputs[i];
            }
            
            virtual InputBlock* get_input_block() { return m_input_block; }
            _State& get_state()
            {
                return m_state;
            }

        protected:
            _InputBlock* m_input_block;
            typename _InputBlock::example_type* m_literals = nullptr;

            _State  m_state;
         
    };


}; // namespace green_tsetlin




#endif // #ifndef __CLAUSE_BLOCK_H_