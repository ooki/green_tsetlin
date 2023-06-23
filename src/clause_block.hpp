#ifndef __CLAUSE_BLOCK_H_
#define __CLAUSE_BLOCK_H_


#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <mutex>
#include <random>
#include <vector>
#include <fstream>

//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>

namespace green_tsetlin
{
    class FeedbackBlock;    
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

            bool is_init() const { return m_is_init; }

            virtual double  get_s() const { return -42.0; }
            virtual void    set_s(double s) {}
            
            virtual uint32_t get_literal_budget() { return 0; }
            virtual void     set_literal_budget(uint32_t budget) {}

            virtual void write(std::ostream& buffer) {}

            virtual bool initialize(unsigned int seed = 42)
            {
                m_is_init = false;
                return false;
            }

            virtual void cleanup()
            {
                m_is_init = false;                
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
            // tm
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

            virtual void get_clause_state_npy(pybind11::array out_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;

                int8_t* p = static_cast<int8_t*>(buffer_info.ptr);                
                m_state.get_clause_state(p, clause_offset);
            }

            virtual void set_clause_state_npy(pybind11::array in_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = in_array.request();                            
                std::vector<ssize_t> shape = buffer_info.shape;

                int8_t* p = static_cast<int8_t*>(buffer_info.ptr);                
                m_state.set_clause_state(p, clause_offset);  
            }

            virtual void get_clause_weights_npy(pybind11::array out_array, int clause_offset)
            {
                pybind11::buffer_info buffer_info = out_array.request();                            
                WeightInt* p = static_cast<WeightInt*>(buffer_info.ptr);                
                m_state.get_clause_weights(p, clause_offset);
            }

            virtual void set_clause_weights_npy(pybind11::array in_array, int clause_offset)
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

            int8_t get_ta_state(int clause_k, int ta_i, bool ta_polarity)
            {
                return m_state.get_ta_state(clause_k, ta_i, ta_polarity);
            }

            void set_ta_state(int clause_k, int ta_i, bool ta_polarity, int8_t new_state)
            {
                m_state.set_ta_state(clause_k, ta_i, ta_polarity, new_state);
            }

            int16_t get_clause_weight(int clause_index, int target_class)
            {
                //std::cout << "get_clause_weight(cbT):" << clause_index << ", " << target_class << std::endl;
                return m_state.get_clause_weight(clause_index, target_class);
            }

            void set_clause_weight(int clause_index, int target_class, int16_t new_weigth)
            {
                m_state.set_clause_weight(clause_index, target_class, new_weigth);
            }



            // TODO: make functor or state member
            virtual std::vector<int8_t> get_copy_clause_outputs() const
            {
                std::vector<int8_t> out(m_state.num_clauses);
                for(int i = 0; i < m_state.num_clauses; ++i)
                    out[i] = m_state.clause_outputs[i];

                return out;
            }

            virtual std::vector<int8_t> get_copy_clause_states() const
            {
                return m_state.get_copy_clauses();
            }

            std::vector<uint32_t> get_copy_literal_counts()
            {
                return m_state.get_copy_literal_counts();
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
    class SparseClauseBlock : public ClauseBlock
    {
        public:
            explicit SparseClauseBlock(int num_literals, int num_clauses, int num_classes, int literal_budget, int literal_buffer_size)
            {
                m_state.num_literals = num_literals;
                m_state.num_clauses = num_clauses;
                m_state.num_classes = num_classes;        
                m_state.literal_budget = literal_budget;
                m_state.literal_buffer_size = literal_buffer_size;
            }

            virtual double get_s() const { return m_state.get_s(); }
            virtual void set_s(double s) { m_state.set_s(s); }

            void set_input_block(_InputBlock* ib)
            {
                m_input_block = ib;
            }

            virtual void pull_example()
            {
                m_literals = m_input_block->pull_current_example();
            }
            
            virtual bool initialize(unsigned int seed = 42)
            {
                _Initializer f;
                m_is_init = f(m_state, seed);
                return m_is_init;
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

            virtual std::vector<int8_t> get_copy_clause_outputs() const
            {
                std::vector<int8_t> out(m_state.num_clauses);
                for(int i = 0; i < m_state.num_clauses; ++i)
                    out[i] = m_state.clause_outputs[i];

                return out;
            }

            int8_t get_ta_state(int clause_k, int ta_i, bool ta_polarity)
            {
                return m_state.get_ta_state(clause_k, ta_i, ta_polarity);
            }

            void set_ta_state(int clause_k, int ta_i, bool ta_polarity, int8_t new_state)
            {
                m_state.set_ta_state(clause_k, ta_i, ta_polarity, new_state);
            }


            int32_t get_clause_weight(int clause_index, int target_class)
            {
                //std::cout << "get_clause_weight(cbT):" << clause_index << ", " << target_class << std::endl;
                return m_state.get_clause_weight(clause_index, target_class);
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
    

    /*
    class ConvolutionalClauseBlock: public ClauseBlock
    {
        public:
            // TODO: maybe reorder the CCB constructor so new args are at the end
            explicit ConvolutionalClauseBlock(int num_literals_per_patch, int num_clauses, int num_classes, int num_patches_per_example)
                : ClauseBlock(num_literals_per_patch, num_clauses, num_classes)
            {
                m_num_patches_per_example = num_patches_per_example;
            }

            

            virtual bool initialize()
            {
                if(ClauseBlock::initialize() == false)
                    return false;

                if(m_active_patches == nullptr)
                    m_active_patches = new uint32_t[m_num_clauses];   

                return true;             
            }

            virtual void cleanup()
            {
                if(m_active_patches != nullptr)
                {
                    delete[] m_active_patches;
                    m_active_patches = nullptr;
                }

                ClauseBlock::cleanup();
            }

            virtual void train_set_clause_output_and_set_votes()
            {       
                //std::uniform_int_distribution<int8_t> start_patch_dist(0, m_num_patches_per_example-1);
                
                //std::cout << "#1" << std::endl;
                for(int clause_k = 0; clause_k < m_num_clauses; ++clause_k)
                {
                    m_active_patches_storage.clear();
                    m_clause_outputs[clause_k] = 0;
                    m_active_patches[clause_k] = 9999999;
                    
                    const int clause_k_offset = clause_k * (m_num_literals*2);

                    //std::cout << "#2 clause_k_offset:" << clause_k_offset << std::endl;
                                        
                    //int patch_k = start_patch_dist(m_rng);
                    //for(int patch_counter = 0; patch_counter < m_num_patches_per_example; ++patch_counter)
                    
                    for(int patch_k = 0; patch_k < m_num_patches_per_example; ++patch_k)
                    {        
                        //std::cout << "#3 patch_k:" << patch_k << std::endl;

                        bool patch_output = true;
                        int8_t* curr_ta_pos = &m_clauses[clause_k_offset];
                        int8_t* curr_ta_neg = curr_ta_pos + m_num_literals;

                        const int literal_path_offset = patch_k * m_num_literals;
                        const uint8_t* curr_literal = &m_literals[literal_path_offset];

                        //std::cout << "#4 literal_path_offset:" << literal_path_offset << std::endl;
                        
                        for(int literal_k = 0; literal_k < m_num_literals; ++literal_k)
                        {                        
                            if(*curr_ta_pos >= 0 && *curr_literal == 0)
                            {                                
                                patch_output = false;
                                break;
                            }
                            curr_ta_pos++;

                            if(*curr_ta_neg >= 0 && *curr_literal == 1) 
                            {                                
                                patch_output = false;
                                break;
                            }
                            
                            curr_ta_neg++;            
                            curr_literal++;                                                         
                        }

                        // clause active on patch
                        //std::cout << "#5 patch_output:" << patch_output << std::endl;
                        if(patch_output)
                        {              
                            m_clause_outputs[clause_k] = 1;              
                            m_active_patches_storage.push_back(patch_k);
                        }
                                   
                            //std::cout << "clause: " << clause_k << " set active patch: " << patch_k << std::endl;
                            //m_active_patches[clause_k] = patch_k;
                            //break;
                        
                        
                        // wrap around
                        //patch_k = (patch_k + 1) % m_num_patches_per_example;
                    }
                    
                    if(m_active_patches_storage.size())
                    {
                        //std::cout << "#6 m_active_patches_storage.size():" << m_active_patches_storage.size() << std::endl;
                        std::uniform_int_distribution<uint32_t> patch_picker(0, m_active_patches_storage.size()-1);

                        // auto auto_bk = m_active_patches[clause_k];
                        uint32_t random_active_patch = patch_picker(m_rng); 
                        m_active_patches[clause_k] = m_active_patches_storage[random_active_patch];

                        //std::cout << "#7 m_active_patches[clause_k]:" << m_active_patches[clause_k] << std::endl;
                    }

                    // if(m_active_patches_storage.size() > 1 && m_active_patches_storage.size() < 10)
                    // {
                    //     std::cout << "m_active_patches_storage: (old: " << auto_bk << " -> " << m_active_patches[clause_k] << " :" << std::endl;
                    //     std::cout << ">";
                    //     for (auto pk: m_active_patches_storage)
                    //         std::cout << pk << ' ';
                    //     std::cout << std::endl;
                    // }
                    

                    
                }
                
                count_votes();

                //std::cout << "\t\t <- cb output and votes gathered: next is reporting to voter" << std::endl;
                //m_feedback_block->register_votes(m_class_votes);
                //std::cout << "\t\t <- cb output and votes gathered: next is reporting to voter : DONE" << std::endl;



                // const int num_weights_total = m_num_clauses * m_num_classes;
                // for(int k = 0; k < num_weights_total; ++k)
                //     std::cout << "CW(" << k << ") = " << m_clause_weights[k] << std::endl;

            }


            virtual void train_update(int positive_class, double prob_positive, int negative_class, double prob_negative)
            {
                std::uniform_real_distribution<double> u(0.0,1.0);
                const int n_features = m_num_literals * 2;
                
                for(int clause_k = 0; clause_k < m_num_clauses; ++clause_k)
                {
                    // patch is only used for T-1a og T-2 feedback, for T-1b its is not needed and 
                    // we can pass an arbitary clause.                                    
                    uint8_t* patch = &m_literals[m_active_patches[clause_k] * m_num_literals];
                    
                    int8_t* clause_row = m_clauses + (clause_k * n_features);
                    int32_t* clause_weights = m_clause_weights + (clause_k * m_num_classes);

                    if( u(m_rng) < prob_positive)
                    {
                        //std::cout << "[" << clause_k << "] update - positive" << std::endl;
                        update_clause(clause_row, clause_weights + positive_class, 1, patch, m_clause_outputs[clause_k]);                    
                    }
      
                    if( u(m_rng) < prob_negative)
                    {
                        //std::cout << "[" << clause_k << "] update - negative" << std::endl;
                        update_clause(clause_row, clause_weights + negative_class, -1, patch, m_clause_outputs[clause_k]);
                    }
                }
                
            }


        protected:
            std::vector<uint32_t> m_active_patches_storage;
            int m_num_patches_per_example;
            uint32_t* m_active_patches = nullptr;
    };
    */
    

    


}; // namespace green_tsetlin


#endif // #ifndef __CLAUSE_BLOCK_H_


