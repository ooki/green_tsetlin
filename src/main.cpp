#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);

// PYBIND11_MAKE_OPAQUE(green_tsetlin::InferenceRule);
// PYBIND11_MAKE_OPAQUE(green_tsetlin::RuleVector);
// PYBIND11_MAKE_OPAQUE(green_tsetlin::RuleWeights);



bool has_avx2()
{
    #ifdef USE_AVX2
        return true;
    #else
        return false;
    #endif
}

bool has_neon()
{
    return false;
}

#include <random_generator.hpp>
#include <input_block.hpp>
#include <feedback_block.hpp>
#include <clause_block.hpp>
#include <aligned_tsetlin_state.hpp>
#include <func_tm.hpp>
#include <func_conv_tm.hpp>
#include <executor.hpp>
#include <inference.hpp>
#include <func_sparse.hpp> 
#include <sparse_tsetlin_state.hpp>

namespace py = pybind11;
namespace gt = green_tsetlin;


//-------------------- Input Blocks ---------------------
typedef typename gt::DenseInputBlock<uint8_t>   DenseInputBlock8u;
typedef typename gt::SparseInputBlock<gt::SparseLiterals>   SparseInputBlock32u;

//-------------------- Executors ---------------------
typedef gt::Executor<false, gt::DummyThreadPool> SingleThreadExecutor;
typedef gt::Executor<true, BS::thread_pool> MultiThreadExecutor;


//-------------------- Vanilla TM ---------------------

typedef typename gt::AlignedTsetlinState<-1,-1> VanillaTsetlinState;

typedef typename gt::ClauseUpdateTM<VanillaTsetlinState,
                                    gt::Type1aFeedbackTM<VanillaTsetlinState, false>, // boost_true_positive = false
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

//-------------------- Vanilla Conv TM ---------------------


typedef typename gt::TrainUpdateConvTM<VanillaTsetlinState,
                                   ClauseUpdateTMImpl,
                                   true > // do_literal_budget = true
                                TrainUpdateConvTMImpl;

typedef typename gt::ClauseBlockT<
                                    VanillaTsetlinState,
                                    gt::InitializeConvTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::CleanupConvTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::SetClauseOutputConvTM<VanillaTsetlinState, true>, // do_literal_budget = true
                                    gt::EvalClauseOutputConvTM<VanillaTsetlinState>,
                                    gt::CountVotesTM<VanillaTsetlinState>,
                                    TrainUpdateConvTMImpl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockConvTMImpl;


//-------------------- Sparse TM --------------------- tentative

typedef typename gt::SparseTsetlinState SparseTsetlinState;

typedef typename gt::ClauseUpdateSparseTM<SparseTsetlinState,
                                    gt::Type1aFeedbackSparseTM<SparseTsetlinState, gt::UpdateAL<SparseTsetlinState>, false>, // boost_true_positive = false
                                    gt::Type1bFeedbackSparseTM<SparseTsetlinState>,
                                    gt::Type2FeedbackSparseTM<SparseTsetlinState>>
                                ClauseUpdateSparseTMImpl;


typedef typename gt::TrainUpdateSparseTM<SparseTsetlinState,
                                   ClauseUpdateSparseTMImpl,
                                   true > // do_literal_budget = true
                                TrainUpdateSparseTMImpl;

typedef typename gt::ClauseBlockSparseT<
                                    SparseTsetlinState,
                                    gt::InitializeSparseTM<SparseTsetlinState, true>,       // do_literal_budget = true
                                    gt::CleanupSparseTM<SparseTsetlinState, true>,          // do_literal_budget = true
                                    gt::SetClauseOutputSparseTM<SparseTsetlinState, true>,  // do_literal_budget = true
                                    gt::EvalClauseOutputSparseTM<SparseTsetlinState>,
                                    gt::CountVotesSparseTM<SparseTsetlinState>,
                                    TrainUpdateSparseTMImpl,
                                    SparseInputBlock32u
                                    >
                                ClauseBlockSparseImpl;
 

//-------------------- AVX 2 TM ---------------------

#ifdef USE_AVX2
#include <func_avx2.hpp>
#include <func_conv_avx2.hpp>

typedef typename gt::AlignedTsetlinState<32, 256 / sizeof(gt::WeightInt)> TsetlinStateAVX2;

typedef typename gt::ClauseUpdateAVX2<TsetlinStateAVX2,
                                    gt::Type1aFeedbackAVX2<TsetlinStateAVX2>,
                                    gt::Type1bFeedbackAVX2<TsetlinStateAVX2>,
                                    gt::Type2FeedbackAVX2<TsetlinStateAVX2>>
                                ClauseUpdateAVX2Impl;

typedef typename gt::TrainUpdateAVX2<TsetlinStateAVX2,
                                   ClauseUpdateAVX2Impl,
                                   true > // do_literal_budget = true
                                TrainUpdateAVX2Impl;


typedef typename gt::ClauseBlockT<
                                    TsetlinStateAVX2,
                                    gt::InitializeAVX2<TsetlinStateAVX2, true, true>, // pad_class (weights) = true, do_literal_budget = true
                                    gt::CleanupAVX2<TsetlinStateAVX2, true>, // do_literal_budget = true
                                    gt::SetClauseOutputAVX2<TsetlinStateAVX2, true>, // do_literal_budget = true
                                    gt::EvalClauseOutputAVX2<TsetlinStateAVX2>,
                                    gt::CountVotesAVX2<TsetlinStateAVX2>,
                                    TrainUpdateAVX2Impl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockAVX2Impl;


typedef typename gt::TrainUpdateConvAVX2<TsetlinStateAVX2,
                                   ClauseUpdateAVX2Impl,
                                   true > // do_literal_budget = true
                                TrainUpdateConvAVX2Impl;

typedef typename gt::ClauseBlockT<
                                    TsetlinStateAVX2,
                                    gt::InitializeConvAVX2<TsetlinStateAVX2, true, true>, // pad_class (weights) = true, do_literal_budget = true
                                    gt::CleanupConvAVX2<TsetlinStateAVX2, true>, // do_literal_budget = true
                                    gt::SetClauseOutputConvAVX2<TsetlinStateAVX2, true>, // do_literal_budget = true
                                    gt::EvalClauseOutputConvAVX2<TsetlinStateAVX2>,
                                    gt::CountVotesAVX2<TsetlinStateAVX2>,
                                    TrainUpdateConvAVX2Impl,
                                    DenseInputBlock8u
                                >
                                ClauseBlockConvAVX2Impl;


#endif // USE_AVX2

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


template<typename _T>
void define_clause_block_sparse(py::module& m, const char* name)
{
    py::class_<_T, gt::ClauseBlock>(m, name)
        .def(py::init<int, int, int>())
        .def("set_input_block", &_T::set_input_block)
        
        .def("get_clause_output", &_T::get_clause_output_npy)

        .def("set_clause_state_sparse", &_T::set_clause_state_sparse_npy)
        .def("get_clause_state_sparse", &_T::get_clause_state_sparse_npy)
        .def("set_clause_weights", &_T::set_clause_weights_npy)
        .def("get_clause_weights", &_T::get_clause_weights_npy)

        
        .def("get_lower_ta_threshold", &_T::get_lower_ta_threshold)
        .def("set_lower_ta_threshold", &_T::set_lower_ta_threshold)

        .def("get_active_literals_size", &_T::get_active_literals_size)
        .def("set_active_literals_size", &_T::set_active_literals_size)

        .def("get_clause_size", &_T::get_clause_size)
        .def("set_clause_size", &_T::set_clause_size);
}


template<typename _T>
void define_inference_module(py::module& m, const char* name)
{
    py::class_<_T>(m, name)
        .def(py::init<int, int, int>())
        .def("set_rules", &_T::set_rules)
        .def("set_features", &_T::set_features)

        .def("set_empty_class_output", &_T::set_empty_class_output)
        .def("get_empty_class_output", &_T::get_empty_class_output)

        .def("predict", &_T::predict_npy)

        .def("get_votes", &_T::get_votes_npy)
        .def("get_active_clauses", &_T::get_active_clauses_npy)
        .def("calculate_importance", &_T::calculate_importance_npy);
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

    // py::bind_vector<gt::InferenceRule>(m, "InferenceRule");
    // py::bind_vector<gt::RuleVector>(m, "RuleVector");
    // py::bind_vector<gt::RuleWeights>(m, "RuleWeights");
    // py::bind_vector<std::vector<std::vector<int>>>(m, "VectorVectorInt");
    // py::bind_vector<std::vector<double>>(m, "VectorDouble");


    // hw info
    // m.def("get_recommended_number_of_threads", &gt::get_recommended_number_of_threads);

    m.def("has_avx2", has_avx2);
    m.def("has_neon", has_neon);
    // m.def("test_train_set_clause_output_sparse", &gt::test_train_set_clause_output<SparseTsetlinState, gt::SetClauseOutputSparseTM<SparseTsetlinState, true>>);
    m.def("test_type2_feedback", &gt::test_Type2FeedbackSparse<SparseTsetlinState, ClauseBlockSparseImpl, gt::Type2FeedbackSparseTM<SparseTsetlinState>>);
    m.def("test_type1a_feedback", &gt::test_Type1aFeedbackSparse<SparseTsetlinState, ClauseBlockSparseImpl, gt::Type1aFeedbackSparseTM<SparseTsetlinState, gt::UpdateAL<SparseTsetlinState>, false>>);


    
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

    py::class_<SparseInputBlock32u, gt::InputBlock>(m, "SparseInputBlock")
        .def(py::init<int>())
        .def("set_data", &SparseInputBlock32u::set_data)
    ;

    m.def("im2col", &gt::tsetlin_im2col);
    
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

    py::class_<gt::FeedbackBlockUniform, gt::FeedbackBlock>(m, "FeedbackBlockUniform")
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

        .def("get_number_of_patches_per_example", &gt::ClauseBlock::get_number_of_patches_per_example)
        .def("set_number_of_patches_per_example", &gt::ClauseBlock::set_number_of_patches_per_example)

        .def("set_trainable", &gt::ClauseBlock::set_trainable)
        .def("is_trainable", &gt::ClauseBlock::is_trainable)
        
        .def("get_literal_budget", &gt::ClauseBlock::get_literal_budget)
        .def("set_literal_budget", &gt::ClauseBlock::set_literal_budget)
        
        .def("initialize", &gt::ClauseBlock::initialize, py::arg("seed") = 42)
        .def("is_initialized", &gt::ClauseBlock::is_init)        
        .def("cleanup", &gt::ClauseBlock::cleanup)       
        
        .def("set_feedback", &gt::ClauseBlock::set_feedback)
    ;

    // py::class_<gt::ClauseBlock>(m, "ClauseBlockSparse")
    //     .def("get_number_of_literals", &gt::ClauseBlock::get_number_of_literals)
    //     .def("get_number_of_clauses", &gt::ClauseBlock::get_number_of_clauses)
    //     .def("get_number_of_classes", &gt::ClauseBlock::get_number_of_classes)

    //     .def("get_s", &gt::ClauseBlock::get_s)
    //     .def("set_s", &gt::ClauseBlock::set_s)

    //     .def("get_number_of_patches_per_example", &gt::ClauseBlock::get_number_of_patches_per_example)
    //     .def("set_number_of_patches_per_example", &gt::ClauseBlock::set_number_of_patches_per_example)

    //     .def("set_trainable", &gt::ClauseBlock::set_trainable)
    //     .def("is_trainable", &gt::ClauseBlock::is_trainable)
        
    //     .def("get_literal_budget", &gt::ClauseBlock::get_literal_budget)
    //     .def("set_literal_budget", &gt::ClauseBlock::set_literal_budget)
        
    //     .def("initialize", &gt::ClauseBlock::initialize, py::arg("seed") = 42)
    //     .def("is_initialized", &gt::ClauseBlock::is_init)        
    //     .def("cleanup", &gt::ClauseBlock::cleanup)       
        
    //     .def("set_feedback", &gt::ClauseBlock::set_feedback)

    // ;


    

    // Clause Block Impl's
    define_clause_block<ClauseBlockTMImpl>(m, "ClauseBlockTM"); // Vanilla TM
    define_clause_block<ClauseBlockConvTMImpl>(m, "ClauseBlockConvTM"); // Vanilla TM with Convolutional TM
    
    define_clause_block<ClauseBlockAVX2Impl>(m, "ClauseBlockAVX2"); // AVX2 TM
    define_clause_block<ClauseBlockConvAVX2Impl>(m, "ClauseBlockConvAVX2"); // AVX2 Conv TM

    // Sparse TM tentative
    define_clause_block_sparse<ClauseBlockSparseImpl>(m, "ClauseBlockSparse"); // Sparse TM


    typedef typename gt::Inference<uint8_t, false, false, false>    Inference8u_Ff_Lf_Wf;
    typedef typename gt::Inference<uint8_t, false, true, false>     Inference8u_Ff_Lt_Wf;
    typedef typename gt::Inference<uint8_t, false, true, true>      Inference8u_Ff_Lt_Wt;
    typedef typename gt::Inference<uint8_t, true, false, false>     Inference8u_Ft_Lf_Wf;
    typedef typename gt::Inference<uint8_t, true, false, true>      Inference8u_Ft_Lf_Wt;

    define_inference_module<Inference8u_Ff_Lf_Wf>(m, "Inference8u_Ff_Lf_Wf"); // Feature Importance : false , Literal Importance : false, Exclude Negative Clauses : false
    // Literal 
    define_inference_module<Inference8u_Ff_Lt_Wf>(m, "Inference8u_Ff_Lt_Wf"); 
    define_inference_module<Inference8u_Ff_Lt_Wt>(m, "Inference8u_Ff_Lt_Wt"); 

    // Feature
    define_inference_module<Inference8u_Ft_Lf_Wf>(m, "Inference8u_Ft_Lf_Wf"); 
    define_inference_module<Inference8u_Ft_Lf_Wt>(m, "Inference8u_Ft_Lf_Wt"); 



    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
