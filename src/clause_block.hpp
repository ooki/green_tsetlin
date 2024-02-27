#ifndef __CLAUSE_BLOCK_H_
#define __CLAUSE_BLOCK_H_

#include <stdint.h>
#include <vector>

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
   


}; // namespace green_tsetlin




#endif // #ifndef __CLAUSE_BLOCK_H_