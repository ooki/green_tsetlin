#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)



// #include <executor.hpp>
#include <input_block.hpp>
#include <feedback_block.hpp>
#include <clause_block.hpp>
#include <aligned_tsetlin_state.hpp>
#include <func_tm.hpp>
#include <executor.hpp>

namespace py = pybind11;
namespace gt = green_tsetlin;


//-------------------- Input Blocks ---------------------
typedef typename gt::DenseInputBlock<uint8_t>   DenseInputBlock8u;

//-------------------- Executors ---------------------
typedef gt::Executor<false, gt::DummyThreadPool> SingleThreadExecutor;
typedef gt::Executor<true, BS::thread_pool> MultiThreadExecutor;


//-------------------- Vanilla TM ---------------------

typedef typename gt::AlignedTsetlinState VanillaTsetlinState;

typedef typename gt::ClauseUpdateTM<VanillaTsetlinState,
                                    gt::Type1aFeedbackTM<VanillaTsetlinState>,
                                    gt::Type1bFeedbackTM<VanillaTsetlinState>,
                                    gt::Type2FeedbackTM<VanillaTsetlinState>>
                                ClauseUpdateTMImpl;


typedef typename gt::TrainUpdateTM<VanillaTsetlinState,
                                   ClauseUpdateTMImpl,
                                   true > // do_literal_budget = true
                                TrainUpdateTMImpl;


typedef typename gt::ClauseBlockT<
                                    VanillaTsetlinState,
                                    gt::InitializeTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::CleanupTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::SetClauseOutputTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::EvalClauseOutputTM<VanillaTsetlinState>,
                                    gt::CountVotesTM<VanillaTsetlinState>,
                                    TrainUpdateTMImpl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockTMImpl;


template<typename _T>
void define_clause_block(py::module& m, const char* name)
{
    py::class_<_T, gt::ClauseBlock>(m, name)
        .def(py::init<int, int, int>())
        .def("set_input_block", &_T::set_input_block)
        
        .def("get_clause_output", &_T::get_clause_output_npy)

        .def("set_clause_state", &_T::set_clause_state_npy)
        .def("get_clause_state", &_T::get_clause_state_npy)
        .def("set_clause_weights", &_T::set_clause_weights_npy)
        .def("get_clause_weights", &_T::get_clause_weights_npy);
}

PYBIND11_MODULE(green_tsetlin_core, m) {

    m.doc() = R"pbdoc(
        green_tsetlin_core
        -----------------------

        .. currentmodule:: green_tsetlin_core

        .. autosummary::
           :toctree: _generate

           A impl. of the Tsetlin Machine
    )pbdoc";

    // hw info
    // m.def("get_recommended_number_of_threads", &gt::get_recommended_number_of_threads);
    
    py::class_<gt::InputBlock>(m, "InputBlock")
        .def("prepare_example", &gt::InputBlock::prepare_example)
        .def("is_multi_label", &gt::InputBlock::is_multi_label)        
        .def("get_num_labels_per_example", &gt::InputBlock::get_num_labels_per_example)        
        .def("get_number_of_examples", &gt::InputBlock::get_number_of_examples)
    ;

    py::class_<DenseInputBlock8u, gt::InputBlock>(m, "DenseInputBlock")
        .def(py::init<int>())
        .def("set_data", &DenseInputBlock8u::set_data)
    ;
    
    py::class_<gt::FeedbackBlock>(m, "FeedbackBlock")
        .def(py::init<int, double, int>())
        .def("reset_votes", &gt::FeedbackBlock::reset)
        .def("register_votes", &gt::FeedbackBlock::register_votes_npy)
        .def("get_votes", &gt::FeedbackBlock::get_votes_npy)
        .def("get_number_of_classes", &gt::FeedbackBlock::get_number_of_classes)        
    ;

    py::class_<gt::FeedbackBlockMultiLabel, gt::FeedbackBlock>(m, "FeedbackBlockMultiLabel")
        .def(py::init<int, double, int>())
    ;

    py::class_<SingleThreadExecutor>(m, "SingleThreadExecutor")
        .def(py::init<gt::InputBlock*, std::vector<gt::ClauseBlock*>, gt::FeedbackBlock*, int, int>())        
        .def("get_number_of_examples_ready", &SingleThreadExecutor::get_number_of_examples_ready)        
        .def("train_epoch", &SingleThreadExecutor::train_epoch)        
        .def("train_slice", &SingleThreadExecutor::train_slice)
        .def("eval_predict", &SingleThreadExecutor::eval_predict)
        .def("eval_predict_multi", &SingleThreadExecutor::eval_predict_multi)    
    ;

    py::class_<MultiThreadExecutor>(m, "MultiThreadExecutor")
        .def(py::init<gt::InputBlock*, std::vector<gt::ClauseBlock*>, gt::FeedbackBlock*, int, int>())        
        .def("get_number_of_examples_ready", &MultiThreadExecutor::get_number_of_examples_ready)        
        .def("train_epoch", &MultiThreadExecutor::train_epoch)        
        .def("train_slice", &MultiThreadExecutor::train_slice)
        .def("eval_predict", &MultiThreadExecutor::eval_predict)
        .def("eval_predict_multi", &MultiThreadExecutor::eval_predict_multi)
    ;


    py::class_<gt::ClauseBlock>(m, "ClauseBlock")
        .def("get_number_of_literals", &gt::ClauseBlock::get_number_of_literals)
        .def("get_number_of_clauses", &gt::ClauseBlock::get_number_of_clauses)
        .def("get_number_of_classes", &gt::ClauseBlock::get_number_of_classes)
        .def("get_s", &gt::ClauseBlock::get_s)
        .def("set_s", &gt::ClauseBlock::set_s)

        .def("get_literal_budget", &gt::ClauseBlock::get_literal_budget)
        .def("set_literal_budget", &gt::ClauseBlock::set_literal_budget)
        
        .def("initialize", &gt::ClauseBlock::initialize, py::arg("seed") = 42)
        .def("is_initialized", &gt::ClauseBlock::is_init)           
        .def("cleanup", &gt::ClauseBlock::cleanup)           

        .def("set_feedback", &gt::ClauseBlock::set_feedback);
    ;

    

    // Clause Block Impl's
    define_clause_block<ClauseBlockTMImpl>(m, "ClauseBlockTM"); // Vanilla TM


    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
