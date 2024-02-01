#ifndef __EXECUTOR_HPP_
#define __EXECUTOR_HPP_


#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <mutex>
#include <random>
#include <vector>

#include <BS_thread_pool.hpp>


// #include <input_block.hpp>
// #include <clause_block.hpp>
// #include <feedback_block.hpp>


namespace green_tsetlin
{
    class InputBlock;
    class ClauseBlock;
    class FeedbackBlock;

    class DummyThreadPool
    {
        public:
            DummyThreadPool() {}
            DummyThreadPool(int i) {}

            void push_task(...) {}
            void wait_for_tasks() {}


    };
    
    template <bool enable_multithread, typename _ThreadPool>
    class Executor    
    {
        public:
            Executor(std::vector<InputBlock*> input_blocks, std::vector<ClauseBlock*> clause_blocks, FeedbackBlock* feedback_block, int seed)
                : Executor(input_blocks, clause_blocks, feedback_block, 0, seed) {}

            Executor(std::vector<InputBlock*> input_blocks, std::vector<ClauseBlock*> clause_blocks, FeedbackBlock* feedback_block, int num_threads, int seed)
                : m_pool(num_threads)
            {
                if(input_blocks.size() < 1)
                    throw std::runtime_error("Cannot create executor with 0 input blocks.");

                if(clause_blocks.size() < 1)
                    throw std::runtime_error("Cannot create executor with 0 clause blocks.");

                if(feedback_block == nullptr)
                    throw std::runtime_error("Cannot create executor with a null feedback block.");

                m_inputs = input_blocks;
                m_clause_blocks = clause_blocks;
                m_feedback_block = feedback_block;
                m_label_input = nullptr;
                m_num_threads = num_threads;

                for(auto ib : input_blocks)
                {
                    if(ib->is_label_block())
                    {
                        if(m_label_input != nullptr)                   
                            throw std::runtime_error("Cannot have more than one LabelBlock in an Executor()");

                        m_label_input = ib;                                            
                    }
                }

                if(m_label_input == nullptr)                   
                    throw std::runtime_error("No InputBlock is marked as a LabelBlock in an Executor()");

                int n_labels_per_example = m_label_input->get_num_labels_per_example();
                if(m_label_input->is_multi_label())
                {
                    // std::cout << "exec:" << "n_labels_per_example: " << n_labels_per_example << " " << "#classes:" << feedback_block->get_number_of_classes() << std::endl;
                    if(n_labels_per_example != feedback_block->get_number_of_classes())
                        throw std::runtime_error("InputBlock is multilabel but Input labels does match feedback labels() in Executor()");
                    
                    for(auto cb : clause_blocks)
                    {
                        if(cb->get_number_of_classes() != n_labels_per_example*2)
                            throw std::runtime_error("InputBlock is multilabel but a CB has wrong number of classes (should be 2x labels) - from Executor()");                        
                    }
                        
                }


                for( auto cb : clause_blocks )
                {
                    if(!cb->is_init())                   
                        throw std::runtime_error("All ClauseBlocks must be init() before constructing an Executor()");
                }

                m_rng.seed(seed);                        
            }            

            double train_epoch()
            {                
                m_feedback_block->reset_train_predict_counter();                
                int n_examples = get_number_of_examples_ready();
                train_slice(0, n_examples);

                return m_feedback_block->get_train_accuracy();
            }

            void train_slice(int start_index, int end_index)
            {                                
                int n_examples = get_number_of_examples_ready();
                //bool m_label_input->is_multi_label();
                
                if(start_index == 0)
                {                                       
                    //std::cout << "n_examples:" << n_examples << std::endl;
                    m_index_set.resize(n_examples);
                    std::iota (std::begin(m_index_set), std::end(m_index_set), 0); // fill 0 -> n_examples
                    std::shuffle(std::begin(m_index_set), std::end(m_index_set), m_rng);
                }

                
                for(int i = start_index; i < end_index; ++i)
                {
                    
                    m_feedback_block->reset();

                    int example_index = m_index_set[i];
                    for(auto ib : m_inputs)
                    {                
                        ib->prepare_example(example_index);
                    }

                    for(auto cb : m_clause_blocks)
                    {             
                        if(enable_multithread)
                        {
                            m_pool.push_task(&ClauseBlock::train_example, cb);
                        }
                        else
                        {
                            cb->train_example();
                        }
                                       
                    }

                    if(enable_multithread)
                        m_pool.wait_for_tasks();


                    m_feedback_block->process(m_label_input->pull_current_label());

                    uint32_t positive_class = m_feedback_block->get_positive_class();
                    uint32_t negative_class = m_feedback_block->get_negative_class();
                    double pup = m_feedback_block->get_positive_update_probability();                    
                    double nup = m_feedback_block->get_negative_update_probability();
                                                                                
                    for(auto cb : m_clause_blocks)
                    {      
                        if(enable_multithread)
                            m_pool.push_task(&ClauseBlock::train_update, cb, positive_class, pup, negative_class, nup);
                        else                        
                            cb->train_update(positive_class, pup, negative_class, nup);
                    }

                    if(enable_multithread)
                        m_pool.wait_for_tasks();
                }
            }


           std::vector<int> eval_predict()
           {
                int n_examples = get_number_of_examples_ready();
                
                std::vector<int> output;
                output.resize(n_examples);

                for(int i = 0; i < n_examples; ++i)
                {
                    m_feedback_block->reset();
                    for(auto ib : m_inputs)   
                        ib->prepare_example(i);

                    for(auto cb : m_clause_blocks)                    
                        cb->eval_example();                    

                    output[i] = m_feedback_block->predict();
                }
                
                return output;
            }

            std::vector<std::vector<int>> eval_predict_multi()
            {
                int n_examples = get_number_of_examples_ready();                
                
                std::vector<std::vector<int>> output;
                output.resize(n_examples);

                for(int i = 0; i < n_examples; ++i)
                {
                    m_feedback_block->reset();
                    for(auto ib : m_inputs)   
                        ib->prepare_example(i);

                    for(auto cb : m_clause_blocks)
                        cb->eval_example();
                    
                    //output[i] = m_feedback_block->predict();
                    output[i] = m_feedback_block->predict_multi();
                }
                return output;
            }

            int get_number_of_examples_ready() const
            {
                return m_label_input->get_number_of_examples();
            }

        private:        
            std::vector<InputBlock*>   m_inputs;
            InputBlock*                m_label_input;
            std::vector<ClauseBlock*>  m_clause_blocks;
            FeedbackBlock* m_feedback_block;
            std::vector<int> m_index_set;

            // TODO: fix random
            std::default_random_engine m_rng;

            int m_num_threads;
            
            //BS::thread_pool m_pool;
            _ThreadPool m_pool;

    };

    int get_recommended_number_of_threads()
    {
        return std::thread::hardware_concurrency() - 1;
    }

}; // namespace green_tsetlin


#endif // #ifndef __EXECUTOR_HPP_