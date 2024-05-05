#ifndef __EXECUTOR_HPP_
#define __EXECUTOR_HPP_



#include <BS_thread_pool.hpp>


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
            Executor(InputBlock* input_block, std::vector<ClauseBlock*> clause_blocks, FeedbackBlock* feedback_block, int num_threads, int seed)
                : m_pool(num_threads)
            {
                if(!enable_multithread)
                {
                    if(num_threads != 1)
                        throw std::runtime_error("Can only create a Single Thread Executor with num_threads set to 1.");                        
                }

                if(input_block == nullptr)
                    throw std::runtime_error("Cannot create executor with nullptr input block.");

                if(clause_blocks.size() < 1)
                    throw std::runtime_error("Cannot create executor with 0 clause blocks.");

                if(feedback_block == nullptr)
                    throw std::runtime_error("Cannot create executor with a null feedback block.");

                m_input_block = input_block;
                m_clause_blocks = clause_blocks;
                m_feedback_block = feedback_block;
                m_num_threads = num_threads;

                int n_labels_per_example = m_input_block->get_num_labels_per_example();
                if(m_input_block->is_multi_label())
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

                    if(cb->get_input_block() == nullptr)
                        throw std::runtime_error("All ClauseBlocks must be have a InputBlock before constructing an Executor()");

                    if(cb->get_feedback() == nullptr)
                        throw std::runtime_error("All ClauseBlocks must be have a FeedbackBlock before constructing an Executor()");

                    if(cb->is_trainable())
                        m_trainable_clause_blocks.push_back(cb);
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

                    m_input_block->prepare_example(example_index);

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

                    m_feedback_block->process(m_input_block->pull_current_label());

                    uint32_t positive_class = m_feedback_block->get_positive_class();
                    uint32_t negative_class = m_feedback_block->get_negative_class();
                    double pup = m_feedback_block->get_positive_update_probability();                    
                    double nup = m_feedback_block->get_negative_update_probability();
                    
                    for(auto cb : m_trainable_clause_blocks)
                    {   
                        if(!cb->is_trainable())
                            continue;

                        if(enable_multithread)
                            m_pool.push_task(&ClauseBlock::train_update, cb, positive_class, pup, negative_class, nup);
                        else                        
                            cb->train_update(positive_class, pup, negative_class, nup);
                    }

                    if(enable_multithread)
                        m_pool.wait_for_tasks();
                }
            }


           bool eval_predict(pybind11::array_t<uint32_t> out_array)
           {
                pybind11::buffer_info buffer_info = out_array.request();
                std::vector<ssize_t> shape2 = buffer_info.shape;

                int n_examples = get_number_of_examples_ready();
                if(shape2[0] != n_examples)
                    return false;

                uint32_t* output = static_cast<uint32_t*>(buffer_info.ptr);
                
                //std::vector<int> output;
                //output.resize(n_examples);

                for(int i = 0; i < n_examples; ++i)
                {
                    m_feedback_block->reset();  
                    m_input_block->prepare_example(i);

                    for(auto cb : m_clause_blocks)
                    {   
                        if(enable_multithread)
                        {
                            m_pool.push_task(&ClauseBlock::eval_example, cb);
                        }
                        else
                        {
                            cb->eval_example();
                        }                     
                    }

                    if(enable_multithread)
                        m_pool.wait_for_tasks();

                    output[i] = m_feedback_block->predict();
                }         

                return true;   
            }

            
            bool eval_predict_multi(pybind11::array_t<uint32_t> out_array)
            {
                std::cout << "eval_predict_multi IS BROKEN!!!!!!!!!!" << std::endl;
                exit(1);

                pybind11::buffer_info buffer_info = out_array.request();      
                std::vector<ssize_t> shape2 = buffer_info.shape;                      
                

                int n_examples = get_number_of_examples_ready();                                      
                if(shape2[0] != n_examples)
                    return false;        

                int n_classes = m_feedback_block->get_number_of_classes();
                if(shape2[1] != n_classes)
                    return false;

                uint32_t* output = static_cast<uint32_t*>(buffer_info.ptr);        

                for(int i = 0; i < n_examples; ++i)
                {
                    m_feedback_block->reset();
                    m_input_block->prepare_example(i);

                    for(auto cb : m_clause_blocks)
                        cb->eval_example();
                    
                    uint32_t* example_ptr = &output[n_examples]; // fix this index to be multi label
                    m_feedback_block->predict_multi(example_ptr);                    
                }

                return true;
            }

            int get_number_of_examples_ready() const
            {
                return m_input_block->get_number_of_examples();
            }

        private:        
            InputBlock*                m_input_block;
            std::vector<ClauseBlock*>  m_clause_blocks;
            std::vector<ClauseBlock*>  m_trainable_clause_blocks;
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

