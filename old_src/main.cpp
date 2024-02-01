#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <feedback_block.hpp>
#include <input_block.hpp>
#include <clause_block.hpp>
#include <functors_interface.hpp>
#include <executor.hpp>
#include <func_nv.hpp>


#include <inference.hpp>

#ifdef USE_AVX2
    #include <func_avx2.hpp>
#endif

#ifdef USE_NEON
    #include <func_neon.hpp>
#endif 

#include <func_sparse.hpp>
    





namespace py = pybind11;
namespace gt = green_tsetlin;

//-------------------- Inference ---------------------

typedef typename gt::Inference<uint8_t, true>  Inference8; // calculate features and literals importance
typedef typename gt::Inference<uint8_t, false>   Inference8NoLiteralsImportance; // do not calculate literals importance

//-------------------- Input ---------------------


typedef typename gt::DenseInputBlock<uint8_t>   DenseInputBlock8u;
typedef typename gt::SparseInputBlock<gt::SparseLiterals> SparseInputBlock32u;


//-------------------- Executors ---------------------

typedef gt::Executor<false, gt::DummyThreadPool> SingleThreadExecutor;
typedef gt::Executor<true, BS::thread_pool> MultiThreadExecutor;


//-------------------- non vectorized ---------------------

typedef typename gt::ClauseUpdateNV<gt::CoaleasedTsetlinStateNV,
                                    gt::Type1aFeedbackNV<gt::CoaleasedTsetlinStateNV>,
                                    gt::Type1bFeedbackNV<gt::CoaleasedTsetlinStateNV>,
                                    gt::Type2FeedbackNV<gt::CoaleasedTsetlinStateNV>>
                            ClauseUpdateNVImpl;

typedef typename gt::TrainUpdateNV<gt::CoaleasedTsetlinStateNV,
                                   ClauseUpdateNVImpl,
                                   true > // do_literal_budget = true
                                TrainUpdateNVImpl;

typedef typename gt::ClauseBlockT<
                                    gt::CoaleasedTsetlinStateNV,
                                    gt::InitializeNV<gt::CoaleasedTsetlinStateNV, true>, // do_literal_budget = true
                                    gt::CleanupNV<gt::CoaleasedTsetlinStateNV, true>, // do_literal_budget = true
                                    gt::SetClauseOutputNV<gt::CoaleasedTsetlinStateNV, true, false>, // do_literal_budget = true, force_at_least_one_positive_literal=false
                                    gt::EvalClauseOutputNV<gt::CoaleasedTsetlinStateNV>,
                                    gt::CountVotesNV<gt::CoaleasedTsetlinStateNV>,
                                    TrainUpdateNVImpl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockNVImpl;


typedef typename gt::ClauseBlockT<
                                    gt::CoaleasedTsetlinStateNV,
                                    gt::InitializeNV<gt::CoaleasedTsetlinStateNV, true>, // do_literal_budget = true
                                    gt::CleanupNV<gt::CoaleasedTsetlinStateNV, true>, // do_literal_budget = true
                                    gt::SetClauseOutputNV<gt::CoaleasedTsetlinStateNV, true, true>, // do_literal_budget = true, force_at_least_one_positive_literal=true
                                    gt::EvalClauseOutputNV<gt::CoaleasedTsetlinStateNV>,
                                    gt::CountVotesNV<gt::CoaleasedTsetlinStateNV>,
                                    TrainUpdateNVImpl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockNVImplPU;


//-------------------- sparse impl ---------------------

typedef typename gt::ClauseUpdateSparseNV<gt::CoaleasedTsetlinStateSparseNV,
                                            gt::Type1FeedbackSparseNV<gt::CoaleasedTsetlinStateSparseNV>,
                                            gt::Type2FeedbackSparseNV<gt::CoaleasedTsetlinStateSparseNV>>
                                        ClauseUpdateSparseNVImpl;


typedef typename gt::TrainUpdateSparseNV<gt::CoaleasedTsetlinStateSparseNV,
                                            ClauseUpdateSparseNVImpl>
                                        TrainUpdateSparseNVImpl;                            


typedef typename gt::SparseClauseBlock<
                                    gt::CoaleasedTsetlinStateSparseNV,
                                    gt::InitializeSparseNV<gt::CoaleasedTsetlinStateSparseNV>,
                                    gt::CleanupSparseNV<gt::CoaleasedTsetlinStateSparseNV>,
                                    gt::SetClauseOutputSparseNV<gt::CoaleasedTsetlinStateSparseNV>,
                                    gt::EvalClauseOutputSparseNV<gt::CoaleasedTsetlinStateSparseNV>,
                                    gt::CountVotesNV<gt::CoaleasedTsetlinStateSparseNV>,
                                    TrainUpdateSparseNVImpl,
                                    SparseInputBlock32u
                                >
                                ClauseBlockSparseNVImpl;


//-------------------- avx2 aligned ---------------------
#ifdef USE_AVX2
#pragma message "AVX2 Support enabled."

typedef typename gt::ClauseUpdateAVX2<gt::CoaleasedTsetlinStateAligned32,
                                    gt::Type1aFeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>,
                                    gt::Type1bFeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>,
                                    gt::Type2FeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>>
                            ClauseUpdateAVX2Impl;

typedef typename gt::TrainUpdateAVX2<gt::CoaleasedTsetlinStateAligned32,
                                     ClauseUpdateAVX2Impl,
                                     true> // do_literal_budget = true
                                TrainUpdateAVX2Impl;                            

typedef typename gt::ClauseBlockT<
                                    gt::CoaleasedTsetlinStateAligned32,
                                    gt::InitializeAligned32<gt::CoaleasedTsetlinStateAligned32, true, true>, // pad_class=true, do_literal_budget = true
                                    gt::CleanupAligned32<gt::CoaleasedTsetlinStateAligned32, true>, // do_literal_budget = true
                                    gt::SetClauseOutputAVX2<gt::CoaleasedTsetlinStateAligned32, true, false>,  // do_literal_budget = true, force_at_least_one_positive_literal=False
                                    gt::EvalClauseOutputAVX2<gt::CoaleasedTsetlinStateAligned32>,
                                    gt::CountVotesAVX2<gt::CoaleasedTsetlinStateAligned32>,    // CountVotesAVX2  CountVotesVectorAVX2                                
                                    TrainUpdateAVX2Impl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockAVX2Impl;


typedef typename gt::ClauseBlockT<
                                    gt::CoaleasedTsetlinStateAligned32,
                                    gt::InitializeAligned32<gt::CoaleasedTsetlinStateAligned32, true, true>, // pad_class=true, do_literal_budget = true
                                    gt::CleanupAligned32<gt::CoaleasedTsetlinStateAligned32, true>, // do_literal_budget = true
                                    gt::SetClauseOutputAVX2<gt::CoaleasedTsetlinStateAligned32, true, true>,  // do_literal_budget = true, force_at_least_one_positive_literal=True
                                    gt::EvalClauseOutputAVX2<gt::CoaleasedTsetlinStateAligned32>,
                                    gt::CountVotesAVX2<gt::CoaleasedTsetlinStateAligned32>,    // CountVotesAVX2  CountVotesVectorAVX2                                
                                    TrainUpdateAVX2Impl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockAVX2ImplPB;                                

#endif

#ifdef USE_NEON
#pragma message "NEON Support enabled."

typedef typename gt::ClauseUpdateNeon<gt::CoaleasedTsetlinStateAlignedNeon32,
                                    gt::Type1aFeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>,
                                    gt::Type1bFeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>,
                                    gt::Type2FeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>
                                    >
                            ClauseUpdateNeonImpl;


typedef typename gt::TrainUpdateNeon<gt::CoaleasedTsetlinStateAlignedNeon32,
                                   ClauseUpdateNeonImpl,
                                   true> // do_literal_budget = true
                                TrainUpdateNeonImpl;                            

typedef typename gt::ClauseBlockT<
                                    gt::CoaleasedTsetlinStateAlignedNeon32,
                                    gt::InitializeAlignedNeon32<gt::CoaleasedTsetlinStateAlignedNeon32, true>, // do_literal_budget=true
                                    gt::CleanupAlignedNeon32<gt::CoaleasedTsetlinStateAlignedNeon32, true>, // do_literal_budget=true
                                    gt::SetClauseOutputNeon<gt::CoaleasedTsetlinStateAlignedNeon32, true>, // do_literal_budget=true
                                    gt::EvalClauseOutputNeon<gt::CoaleasedTsetlinStateAlignedNeon32>,
                                    gt::CountVotesNeon<gt::CoaleasedTsetlinStateAlignedNeon32>,                                    
                                    TrainUpdateNeonImpl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockNeonImpl;

#endif 


template<typename _T>
void define_clause_block(py::module& m, const char* name)
{
    py::class_<_T, gt::ClauseBlock>(m, name)
        .def(py::init<int, int, int>())
        .def("set_input_block", &_T::set_input_block)
        .def("set_clause_weight", &_T::set_clause_weight)
        .def("get_clause_weight", &_T::get_clause_weight) 
        .def("set_ta_state", &_T::set_ta_state)
        .def("get_ta_state", &_T::get_ta_state)
        .def("get_copy_clause_outputs", &_T::get_copy_clause_outputs)
        .def("get_copy_literal_counts", &_T::get_copy_literal_counts)
        .def("get_copy_clause_states", &_T::get_copy_clause_states)
        .def("set_clause_state", &_T::set_clause_state_npy)
        .def("get_clause_state", &_T::get_clause_state_npy)
        .def("set_clause_weights", &_T::set_clause_weights_npy)
        .def("get_clause_weights", &_T::get_clause_weights_npy);
}



PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MODULE(green_tsetlin_core, m) {

    py::bind_vector<std::vector<int>>(m, "StlVectorInt", "A stl vector of ints");
    py::bind_vector<std::vector<double>>(m, "StlVectorDouble", "A stl vector of doubles");

    m.doc() = R"pbdoc(
        green_tsetlin_core
        -----------------------

        .. currentmodule:: green_tsetlin_core

        .. autosummary::
           :toctree: _generate

           A impl. of the Tsetlin Machine
    )pbdoc";

    // m.def("tsetlin_im2col", &gt::tsetlin_im2col, R"pbdoc(
    //     Perform im2col over a NWHC mem block.        
    // )pbdoc");
    

    // hw info
    m.def("get_recommended_number_of_threads", &gt::get_recommended_number_of_threads);

    // inference for hpp
    m.def("time_train_set_clause_output_and_set_votes", &gt::time_train_set_clause_output_and_set_votes);
    m.def("time_eval_set_clause_output_and_set_votes", &gt::time_eval_set_clause_output_and_set_votes);
    m.def("flush_cache_with_rand_data", &gt::flush_cache_with_rand_data);

#ifdef USE_AVX2
    m.def("time_type1a_feedback_avx2", &gt::time_Type1aFeedback<ClauseBlockAVX2Impl, gt::Type1aFeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>>);  
    m.def("time_type1b_feedback_avx2", &gt::time_Type1bFeedback<ClauseBlockAVX2Impl, gt::Type1bFeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>>);
    m.def("time_type2_feedback_avx2", &gt::time_Type2Feedback<ClauseBlockAVX2Impl, gt::Type2FeedbackAVX2<gt::CoaleasedTsetlinStateAligned32>>);
    //m.def("time_vote_count_avx2", &gt::time_count_votes_on_already_set_clause_outputs<gt::CoaleasedTsetlinStateAligned32, ClauseBlockAVX2Impl, gt::CountVotesVectorAVX2<gt::CoaleasedTsetlinStateAligned32>>);
#endif 

#ifdef USE_NEON
    m.def("time_type1a_feedback_neon", &gt::time_Type1aFeedback<ClauseBlockNeonImpl, gt::Type1aFeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>>);
    m.def("time_type1b_feedback_neon", &gt::time_Type1bFeedback<ClauseBlockNeonImpl, gt::Type1bFeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>>);
    m.def("time_type2_feedback_neon", &gt::time_Type2Feedback<ClauseBlockNeonImpl, gt::Type2FeedbackNeon<gt::CoaleasedTsetlinStateAlignedNeon32>>);
    m.def("test_neon", &gt::test_neon);
#endif 

    //CoaleasedTsetlinStateNV 
    m.def("time_type1a_feedback_nv", &gt::time_Type1aFeedback<ClauseBlockNVImpl, gt::Type1aFeedbackNV<gt::CoaleasedTsetlinStateNV>>);
    m.def("time_type1b_feedback_nv", &gt::time_Type1bFeedback<ClauseBlockNVImpl, gt::Type1bFeedbackNV<gt::CoaleasedTsetlinStateNV>>);
    
    m.def("time_type2_feedback_nv", &gt::time_Type2Feedback<ClauseBlockNVImpl, gt::Type2FeedbackNV<gt::CoaleasedTsetlinStateNV>>);
    m.def("time_vote_count", &gt::time_count_votes_on_already_set_clause_outputs);
    

    py::class_<gt::FeedbackBlock>(m, "FeedbackBlock")
        .def(py::init<int, double>())
        .def(py::init<int, double, int>());

    py::class_<gt::FeedbackBlockMultiLabel, gt::FeedbackBlock>(m, "FeedbackBlockMultiLabel")
        .def(py::init<int, double, int>())
        .def("predict_multi", &gt::FeedbackBlockMultiLabel::predict_multi)
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
        .def("cleanup", &gt::ClauseBlock::cleanup)
                
        .def("set_feedback", &gt::ClauseBlock::set_feedback);

    // // NV
    // py::class_<ClauseBlockNVImpl, gt::ClauseBlock>(m, "ClauseBlockNV")
    //     .def(py::init<int, int, int>())
    //     .def("set_input_block", &ClauseBlockNVImpl::set_input_block)
    //     .def("set_clause_weight", &ClauseBlockNVImpl::set_clause_weight)
    //     .def("get_clause_weight", &ClauseBlockNVImpl::get_clause_weight)
    //     .def("set_ta_state", &ClauseBlockNVImpl::set_ta_state)
    //     .def("get_ta_state", &ClauseBlockNVImpl::get_ta_state)
    //     .def("get_copy_clause_outputs", &ClauseBlockNVImpl::get_copy_clause_outputs)
    //     .def("get_copy_literal_counts", &ClauseBlockNVImpl::get_copy_literal_counts)
    //     .def("get_copy_clause_states", &ClauseBlockNVImpl::get_copy_clause_states)
    //     .def("set_clause_state", &ClauseBlockNVImpl::set_clause_state_npy)
    //     .def("get_clause_state", &ClauseBlockNVImpl::get_clause_state_npy)
    //     .def("set_clause_weights", &ClauseBlockNVImpl::set_clause_weights_npy)
    //     .def("get_clause_weights", &ClauseBlockNVImpl::get_clause_weights_npy)
    //     ;

    define_clause_block<ClauseBlockNVImpl>(m, "ClauseBlockNV");
    define_clause_block<ClauseBlockNVImplPU>(m, "ClauseBlockNVPU");




#ifdef USE_AVX2




    // py::class_<ClauseBlockAVX2Impl, gt::ClauseBlock>(m, "ClauseBlockAVX2")
    //     .def(py::init<int, int, int>())
    //     .def("set_input_block", &ClauseBlockAVX2Impl::set_input_block)
    //     .def("set_clause_weight", &ClauseBlockAVX2Impl::set_clause_weight)
    //     .def("get_clause_weight", &ClauseBlockAVX2Impl::get_clause_weight) 
    //     .def("set_ta_state", &ClauseBlockAVX2Impl::set_ta_state)
    //     .def("get_ta_state", &ClauseBlockAVX2Impl::get_ta_state)
    //     .def("get_copy_clause_outputs", &ClauseBlockAVX2Impl::get_copy_clause_outputs)
    //     .def("get_copy_literal_counts", &ClauseBlockAVX2Impl::get_copy_literal_counts)
    //     .def("get_copy_clause_states", &ClauseBlockAVX2Impl::get_copy_clause_states)
    //     .def("set_clause_state", &ClauseBlockAVX2Impl::set_clause_state_npy)
    //     .def("get_clause_state", &ClauseBlockAVX2Impl::get_clause_state_npy)
    //     .def("set_clause_weights", &ClauseBlockAVX2Impl::set_clause_weights_npy)
    //     .def("get_clause_weights", &ClauseBlockAVX2Impl::get_clause_weights_npy)
    //     ;
    define_clause_block<ClauseBlockAVX2Impl>(m, "ClauseBlockAVX2");
    define_clause_block<ClauseBlockAVX2ImplPB>(m, "ClauseBlockAVX2PB");
        

#endif 


#ifdef USE_NEON
    // py::class_<ClauseBlockNeonImpl, gt::ClauseBlock>(m, "ClauseBlockNeon")
    //     .def(py::init<int, int, int>())
    //     .def("set_input_block", &ClauseBlockNeonImpl::set_input_block)
    //     .def("set_clause_weight", &ClauseBlockNeonImpl::set_clause_weight)
    //     .def("get_clause_weight", &ClauseBlockNeonImpl::get_clause_weight)
    //     .def("set_ta_state", &ClauseBlockNeonImpl::set_ta_state)
    //     .def("get_ta_state", &ClauseBlockNeonImpl::get_ta_state)
    //     .def("get_copy_clause_outputs", &ClauseBlockNeonImpl::get_copy_clause_outputs)
    //     .def("get_copy_literal_counts", &ClauseBlockNeonImpl::get_copy_literal_counts)
    //     .def("get_copy_clause_states", &ClauseBlockNeonImpl::get_copy_clause_states)
    //     .def("set_clause_state", &ClauseBlockNeonImpl::set_clause_state_npy)
    //     .def("get_clause_state", &ClauseBlockNeonImpl::get_clause_state_npy)
    //     .def("set_clause_weights", &ClauseBlockNeonImpl::set_clause_weights_npy)
    //     .def("get_clause_weights", &ClauseBlockNeonImpl::get_clause_weights_npy)
    ;

    define_clause_block<ClauseBlockNeonImpl>(m, "ClauseBlockNeon");
#endif 

    /*
    py::class_<ClauseBlockSparseNVImpl, gt::ClauseBlock>(m, "SparseClauseBlockNV")
        .def(py::init<int, int, int, int, int>())
        .def("set_input_block", &ClauseBlockSparseNVImpl::set_input_block)
        .def("get_copy_clause_outputs", &ClauseBlockSparseNVImpl::get_copy_clause_outputs)
        .def("set_ta_state", &ClauseBlockSparseNVImpl::set_ta_state)
        .def("get_ta_state", &ClauseBlockSparseNVImpl::get_ta_state)
    ;
    */

    /*
    py::class_<gt::ConvolutionalClauseBlock, gt::ClauseBlock>(m, "ConvolutionalClauseBlock")
        .def(py::init<int, int, int, int>())
        .def("initialize", &gt::ConvolutionalClauseBlock::initialize)
        .def("cleanup", &gt::ConvolutionalClauseBlock::cleanup);
    */

    py::class_<gt::InputBlock>(m, "InputBlock")
        .def("prepare_example", &gt::InputBlock::prepare_example)
        .def("is_label_block", &gt::InputBlock::is_label_block)
        .def("is_multi_label", &gt::InputBlock::is_multi_label)        
        .def("get_num_labels_per_example", &gt::InputBlock::get_num_labels_per_example)        
        .def("get_number_of_examples", &gt::InputBlock::get_number_of_examples)
    ;

    
    py::class_<DenseInputBlock8u, gt::InputBlock>(m, "DenseInputBlock")
        .def(py::init<int>())
        .def("set_data", &DenseInputBlock8u::set_data)
    ;

    py::class_<SparseInputBlock32u, gt::InputBlock>(m, "SparseInputBlock")
        .def(py::init<int>())
        .def("set_data", &SparseInputBlock32u::set_data)
    ;

     py::class_<SingleThreadExecutor>(m, "SingleThreadExecutor")
        .def(py::init<std::vector<gt::InputBlock*>, std::vector<gt::ClauseBlock*>, gt::FeedbackBlock*, int>())        
        .def("get_number_of_examples_ready", &SingleThreadExecutor::get_number_of_examples_ready)        
        .def("train_epoch", &SingleThreadExecutor::train_epoch)        
        .def("train_slice", &SingleThreadExecutor::train_slice)
        .def("eval_predict", &SingleThreadExecutor::eval_predict)
        .def("eval_predict_multi", &SingleThreadExecutor::eval_predict_multi)    
    ;

    py::class_<MultiThreadExecutor>(m, "MultiThreadExecutor")
        .def(py::init<std::vector<gt::InputBlock*>, std::vector<gt::ClauseBlock*>, gt::FeedbackBlock*, int, int>())        
        .def("get_number_of_examples_ready", &MultiThreadExecutor::get_number_of_examples_ready)        
        .def("train_epoch", &MultiThreadExecutor::train_epoch)        
        .def("train_slice", &MultiThreadExecutor::train_slice)
        .def("eval_predict", &MultiThreadExecutor::eval_predict)
        .def("eval_predict_multi", &MultiThreadExecutor::eval_predict_multi)
    ;

    py::class_<Inference8>(m, "Inference")
        .def(py::init<int, int, int, int>())
        .def("set_rules_and_features", &Inference8::set_rules_and_features)
        .def("predict", &Inference8::predict_npy)
        .def("predict_multi", &Inference8::predict_multi_npy)
        .def("calc_local_importance", &Inference8::calc_local_importance_npy)
        .def("calculate_global_importance", &Inference8::calculate_global_importance)
        .def("get_active_clauses", &Inference8::get_active_clauses_npy)        
        .def("set_empty_class_output", &Inference8::set_empty_class_output)
        .def("get_empty_class_output", &Inference8::get_empty_class_output)
        .def("get_votes", &Inference8::get_votes_npy)
        .def("get_rule_by_literals", &Inference8::get_rule_by_literals_npy)
        
        .def("get_cached_literal_importance", &Inference8::get_cached_literal_importance_npy)                
    ;

    py::class_<Inference8NoLiteralsImportance>(m, "InferenceNoLiteralsImportance")
        .def(py::init<int, int, int, int>())
        .def("set_rules_and_features", &Inference8NoLiteralsImportance::set_rules_and_features)
        .def("predict", &Inference8NoLiteralsImportance::predict_npy)
        .def("predict_multi", &Inference8NoLiteralsImportance::predict_multi_npy)
        .def("calc_local_importance", &Inference8NoLiteralsImportance::calc_local_importance_npy)
        .def("calculate_global_importance", &Inference8NoLiteralsImportance::calculate_global_importance)
        .def("get_active_clauses", &Inference8NoLiteralsImportance::get_active_clauses_npy)        
        .def("set_empty_class_output", &Inference8NoLiteralsImportance::set_empty_class_output)
        .def("get_empty_class_output", &Inference8NoLiteralsImportance::get_empty_class_output)
        .def("get_votes", &Inference8NoLiteralsImportance::get_votes_npy)
        .def("get_rule_by_literals", &Inference8NoLiteralsImportance::get_rule_by_literals_npy)
    ;

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
