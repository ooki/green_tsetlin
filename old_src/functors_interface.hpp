
#ifndef _FUNCTORS_PERFORMANCE_TESTS_HPP_
#define _FUNCTORS_PERFORMANCE_TESTS_HPP_



#include <random>
#include <vector>
#include <chrono>


namespace green_tsetlin
{
   

    int flush_cache_with_rand_data(int flush_size, int value)
    {    
        int *c = new int[flush_size];

        int total = 1;
        if(value & 1)
            total += 1;
        
        for(int i = 0; i < flush_size; ++i)
        {
            int r = rand() % 256;
            c[i] = r;
        }

        for(int i = 0; i < flush_size;++i)
        {
            total += c[i] + i + value;
        }

        delete[] c;
        return total;
    }

    
    class ClauseBlock;
    int64_t time_train_set_clause_output_and_set_votes(ClauseBlock* cb)
    {
        if(cb->get_input_block() == nullptr)
            throw std::runtime_error("time_train_set_clause_output_and_set_votes - ClauseBlock has no input_block connected.");

        if(cb->get_feedback() == nullptr)
            throw std::runtime_error("time_train_set_clause_output_and_set_votes - ClauseBlock has no feedback connected.");
            
        cb->pull_example(); // setup

        auto t0 = std::chrono::steady_clock::now();
        cb->train_set_clause_output();
        cb->set_votes();
        auto t1 = std::chrono::steady_clock::now();

        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }

    class ClauseBlock;
    int64_t time_eval_set_clause_output_and_set_votes(ClauseBlock* cb)
    {
        if(cb->get_input_block() == nullptr)
            throw std::runtime_error("time_train_set_clause_output_and_set_votes - ClauseBlock has no input_block connected.");

        if(cb->get_feedback() == nullptr)
            throw std::runtime_error("time_train_set_clause_output_and_set_votes - ClauseBlock has no feedback connected.");
            
        cb->pull_example(); // setup

        auto t0 = std::chrono::steady_clock::now();        
        cb->eval_set_clause_output();
        cb->set_votes();
        auto t1 = std::chrono::steady_clock::now();

        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }

    
    int64_t time_count_votes_on_already_set_clause_outputs(ClauseBlock* cb)
    {
        if(cb->get_feedback() == nullptr)
            throw std::runtime_error("time_train_set_clause_output_and_set_votes - ClauseBlock has no feedback connected.");

        auto t0 = std::chrono::steady_clock::now();              
        cb->set_votes();
        auto t1 = std::chrono::steady_clock::now();
        
        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }


    class InputBlock;
    template <typename _CBImpl, typename _T1aFeedback>
    int64_t time_Type1aFeedback(_CBImpl* cb, InputBlock* ib, int n_clauses)
    {            
        if(n_clauses < 1)
            n_clauses = 1;

        ib->prepare_example(0);

        //clause_row 
        cb->pull_example(); // setup

        auto lits = cb->get_current_literals();
        typename _CBImpl::StateType& state = cb->get_state();

        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < n_clauses; ++i)
        {
            int8_t* clause_row = &state.clauses[i * (state.num_literals_mem * 2)];
            _T1aFeedback t1a;  
            t1a(state, clause_row, lits);
        }

        auto t1 = std::chrono::steady_clock::now();

        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }

    class InputBlock;
    template <typename _CBImpl, typename _T1bFeedback>
    int64_t time_Type1bFeedback(_CBImpl* cb, InputBlock* ib, int n_clauses)
    {            
        if(n_clauses < 1)
            n_clauses = 1;

        ib->prepare_example(0);

        //clause_row 
        cb->pull_example(); // setup

        auto lits = cb->get_current_literals();
        typename _CBImpl::StateType& state = cb->get_state();

        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < n_clauses; ++i)
        {
            int8_t* clause_row = &state.clauses[i * (state.num_literals_mem * 2)];
            _T1bFeedback t1b;  
            t1b(state, clause_row, lits);
        }

        auto t1 = std::chrono::steady_clock::now();

        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }



    class InputBlock;
    template <typename _CBImpl, typename _T2Feedback>
    int64_t time_Type2Feedback(_CBImpl* cb, InputBlock* ib, int n_clauses)
    {            
        if(n_clauses < 1)
            n_clauses = 1;

        ib->prepare_example(0);

        //clause_row 
        cb->pull_example(); // setup

        auto lits = cb->get_current_literals();
        typename _CBImpl::StateType& state = cb->get_state();
        
        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < n_clauses; ++i)
        {
            int8_t* clause_row = &state.clauses[i * (state.num_literals_mem * 2)];
            _T2Feedback t2;
            t2(state, clause_row, lits);
        }

        auto t1 = std::chrono::steady_clock::now();

        int64_t elapsed = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();    
        return elapsed;
    }

};






#endif // #ifndef _FUNCTORS_PERFORMANCE_TESTS_HPP_










