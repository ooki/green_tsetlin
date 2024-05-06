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



#ifdef USE_AVX2
#include <cpuid.h>
#endif 


bool has_avx2()
{
    #ifdef USE_AVX2
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if(__get_cpuid(0, &eax, &ebx, &ecx, &edx))
        return (bit_AVX2 & edx) != 0;

    #endif 
    
    return false;
}

bool has_neon()
{
    #ifdef USE_NEON
        #if defined(__ARM_NEON__) 
            return true;
        #else
            return false;
        #endif
    #endif

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
typedef typename gt::SparseInputDenseOutputBlock<uint8_t>   SparseInputDenseOutputBlocku8;
typedef typename gt::SparseInputBlock<gt::SparseLiterals>   SparseInputBlock32u;

//-------------------- Executors ---------------------
typedef gt::Executor<false, gt::DummyThreadPool> SingleThreadExecutor;
typedef gt::Executor<true, BS::thread_pool> MultiThreadExecutor;


//-------------------- Vanilla TM ---------------------

typedef typename gt::AlignedTsetlinState<-1,-1> VanillaTsetlinState;



template<bool lit_budget, bool btp>
using ClauseBlockTMImpl = gt::ClauseBlockT<
                                    VanillaTsetlinState,
                                    gt::InitializeTM<VanillaTsetlinState, lit_budget>, // do_literal_budget = true
                                    gt::CleanupTM<VanillaTsetlinState, lit_budget>, // do_literal_budget = true
                                    gt::SetClauseOutputTM<VanillaTsetlinState, lit_budget>, // do_literal_budget = true
                                    gt::EvalClauseOutputTM<VanillaTsetlinState>,
                                    gt::CountVotesTM<VanillaTsetlinState>,
                                    gt::TrainUpdateTM<VanillaTsetlinState,
                                                        gt::ClauseUpdateTM<VanillaTsetlinState,
                                                                            gt::Type1aFeedbackTM<VanillaTsetlinState, btp>, // boost_true_positive = false
                                                                            gt::Type1bFeedbackTM<VanillaTsetlinState>,
                                                                            gt::Type2FeedbackTM<VanillaTsetlinState>
                                                                            >,
                                                        lit_budget // do_literal_budget = true
                                                        >,
                                    DenseInputBlock8u
                                >;


//-------------------- Vanilla Conv TM ---------------------


typedef typename gt::ClauseUpdateTM<VanillaTsetlinState,
                                    gt::Type1aFeedbackTM<VanillaTsetlinState, false>, // boost_true_positive = false
                                    gt::Type1bFeedbackTM<VanillaTsetlinState>,
                                    gt::Type2FeedbackTM<VanillaTsetlinState>>
                                ClauseUpdateConvTMImpl;


typedef typename gt::TrainUpdateConvTM<VanillaTsetlinState,
                                   ClauseUpdateConvTMImpl,
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


template<bool lit_budget, bool dynamic_AL, bool btp>
using ClauseBlockSparseImpl = gt::ClauseBlockSparseT<
                                    SparseTsetlinState,
                                    gt::InitializeSparseTM<SparseTsetlinState, lit_budget>,       // do_literal_budget = true
                                    gt::CleanupSparseTM<SparseTsetlinState, lit_budget>,          // do_literal_budget = true
                                    gt::SetClauseOutputSparseTM<SparseTsetlinState, lit_budget>,  // do_literal_budget = true
                                    gt::EvalClauseOutputSparseTM<SparseTsetlinState>,
                                    gt::CountVotesSparseTM<SparseTsetlinState>,
                                    gt::TrainUpdateSparseTM<SparseTsetlinState,
                                                            gt::ClauseUpdateSparseTM<SparseTsetlinState,
                                                                                        gt::Type1aFeedbackSparseTM<SparseTsetlinState, gt::UpdateAL<SparseTsetlinState, dynamic_AL>, btp>, // dynamic_AL = true, boost_true_positive = false
                                                                                        gt::Type1bFeedbackSparseTM<SparseTsetlinState>,
                                                                                        gt::Type2FeedbackSparseTM<SparseTsetlinState>
                                                                                    >,
                                                            lit_budget
                                                            >,
                                    SparseInputBlock32u
                                    >;



//-------------------- AVX 2 TM ---------------------

#ifdef USE_AVX2
#include <func_avx2.hpp>
#include <func_conv_avx2.hpp>

typedef typename gt::AlignedTsetlinState<32, 256 / sizeof(gt::WeightInt)> TsetlinStateAVX2;


template<bool lit_budget, bool btp>
using ClauseBlockAVX2Impl = gt::ClauseBlockT<
                                    TsetlinStateAVX2,
                                    gt::InitializeAVX2<TsetlinStateAVX2, false, lit_budget>, // pad_class (weights) = false
                                    gt::CleanupAVX2<TsetlinStateAVX2, lit_budget>, 
                                    gt::SetClauseOutputAVX2<TsetlinStateAVX2, lit_budget>, 
                                    gt::EvalClauseOutputAVX2<TsetlinStateAVX2>,
                                    gt::CountVotesAVX2<TsetlinStateAVX2>,
                                    gt::TrainUpdateAVX2<TsetlinStateAVX2,
                                                        gt::ClauseUpdateAVX2<TsetlinStateAVX2,
                                                                            gt::Type1aFeedbackAVX2<TsetlinStateAVX2, btp>,
                                                                            gt::Type1bFeedbackAVX2<TsetlinStateAVX2>,
                                                                            gt::Type2FeedbackAVX2<TsetlinStateAVX2>
                                                                            >,
                                                        lit_budget 
                                                        >,
                                    DenseInputBlock8u
                                >;

typedef typename gt::ClauseUpdateAVX2<TsetlinStateAVX2,
                                    gt::Type1aFeedbackAVX2<TsetlinStateAVX2, false>,
                                    gt::Type1bFeedbackAVX2<TsetlinStateAVX2>,
                                    gt::Type2FeedbackAVX2<TsetlinStateAVX2>>
                                ClauseUpdateConvAVX2Impl;

typedef typename gt::TrainUpdateConvAVX2<TsetlinStateAVX2,
                                   ClauseUpdateConvAVX2Impl,
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


#ifdef USE_NEON

#include <func_neon.hpp>
typedef typename gt::AlignedTsetlinState<16, 128 / sizeof(gt::WeightInt)> TsetlinStateNeon;

template<bool lit_budget, bool btp>
using ClauseBlockNeonImpl = gt::ClauseBlockT<
                                    TsetlinStateNeon,
                                    gt::InitializeAlignedNeon<TsetlinStateNeon, lit_budget>, // pad_class (weights) = false
                                    gt::CleanupAlignedNeon<TsetlinStateNeon, lit_budget>, 
                                    gt::SetClauseOutputNeon<TsetlinStateNeon, lit_budget>, 
                                    gt::EvalClauseOutputNeon<TsetlinStateNeon>,
                                    gt::CountVotesNeon<TsetlinStateNeon>,
                                    gt::TrainUpdateNeon<TsetlinStateNeon,
                                                        gt::ClauseUpdateNeon<TsetlinStateNeon,
                                                                            gt::Type1aFeedbackNeon<TsetlinStateNeon, btp>,
                                                                            gt::Type1bFeedbackNeon<TsetlinStateNeon>,
                                                                            gt::Type2FeedbackNeon<TsetlinStateNeon>
                                                                            >,
                                                        lit_budget 
                                                        >,
                                    DenseInputBlock8u
                                >;

#endif // USE_NEON


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
        .def("set_active_literals", &_T::set_active_literals_npy)
        .def("get_active_literals", &_T::get_active_literals_npy)


        
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
        .def("calculate_explanations", &_T::calculate_explanations)
        .def("get_literal_importance", &_T::get_literal_importance_npy)
        .def("get_feature_importance", &_T::get_feature_importance_npy);
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
    // m.def("test_type2_feedback", &gt::test_Type2FeedbackSparse<SparseTsetlinState, ClauseBlockSparseImpl, gt::Type2FeedbackSparseTM<SparseTsetlinState>>);
    // m.def("test_type1a_feedback", &gt::test_Type1aFeedbackSparse<SparseTsetlinState, ClauseBlockSparseImpl, gt::Type1aFeedbackSparseTM<SparseTsetlinState, gt::UpdateAL<SparseTsetlinState, true>, false>>);


    
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

    py::class_<SparseInputDenseOutputBlocku8, DenseInputBlock8u>(m, "SparseInputDenseOutputBlock")
        .def(py::init<int>())
        .def("set_data", &SparseInputDenseOutputBlocku8::set_data_sparse)
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


    

    // Dense TM non-vectorized implementations
    define_clause_block<ClauseBlockTMImpl<true, true>>(m, "ClauseBlockTM_Lt_Bt"); // Vanilla TM, (L)lit_budget = true, (B)btp = true
    define_clause_block<ClauseBlockTMImpl<true, false>>(m, "ClauseBlockTM_Lt_Bf"); // Vanilla TM, (L)lit_budget = true, (B)btp = false
    define_clause_block<ClauseBlockTMImpl<false, true>>(m, "ClauseBlockTM_Lf_Bt"); // Vanilla TM, (L)lit_budget = false, (B)btp = true
    define_clause_block<ClauseBlockTMImpl<false, false>>(m, "ClauseBlockTM_Lf_Bf"); // Vanilla TM, (L)lit_budget = false, (B)btp = false
    

    // Conv TM non-vectorized implementations
    define_clause_block<ClauseBlockConvTMImpl>(m, "ClauseBlockConvTM"); // Vanilla Convolutional TM
    
#ifdef USE_AVX2
    // AVX2 TM implementations
    define_clause_block<ClauseBlockAVX2Impl<true, true>>(m, "ClauseBlockAVX2_Lt_Bt"); // AVX2 TM, (L)lit_budget = true, (B)btp = true
    define_clause_block<ClauseBlockAVX2Impl<true, false>>(m, "ClauseBlockAVX2_Lt_Bf"); // AVX2 TM, (L)lit_budget = true, (B)btp = false
    define_clause_block<ClauseBlockAVX2Impl<false, true>>(m, "ClauseBlockAVX2_Lf_Bt"); // AVX2 TM, (L)lit_budget = false, (B)btp = true
    define_clause_block<ClauseBlockAVX2Impl<false, false>>(m, "ClauseBlockAVX2_Lf_Bf"); // AVX2 TM, (L)lit_budget = false, (B)btp = false
    
    // AVX2 Conv TM implementations
    define_clause_block<ClauseBlockConvAVX2Impl>(m, "ClauseBlockConvAVX2"); // AVX2 Conv TM
#endif // USE_AVX2

#ifdef USE_NEON

    // NEON TM implementations
    define_clause_block<ClauseBlockNeonImpl<true, true>>(m, "ClauseBlockNeon_Lt_Bt"); // NEON TM, (L)lit_budget = true, (B)btp = true
    define_clause_block<ClauseBlockNeonImpl<true, false>>(m, "ClauseBlockNeon_Lt_Bf"); // NEON TM, (L)lit_budget = true, (B)btp = false
    define_clause_block<ClauseBlockNeonImpl<false, true>>(m, "ClauseBlockNeon_Lf_Bt"); // NEON TM, (L)lit_budget = false, (B)btp = true
    define_clause_block<ClauseBlockNeonImpl<false, false>>(m, "ClauseBlockNeon_Lf_Bf"); // NEON TM, (L)lit_budget = false, (B)btp = false

#endif // USE_NEON

    // Sparse TM implementations
    define_clause_block_sparse<ClauseBlockSparseImpl<true, true, true>>(m, "ClauseBlockSparse_Lt_Dt_Bt"); // Sparse TM, (L)lit_budget = true, (D)dynamic_AL = true, (B)btp = true
    define_clause_block_sparse<ClauseBlockSparseImpl<true, true, false>>(m, "ClauseBlockSparse_Lt_Dt_Bf"); // Sparse TM, (L)lit_budget = true, (D)dynamic_AL = true, (B)btp = false
    define_clause_block_sparse<ClauseBlockSparseImpl<true, false, true>>(m, "ClauseBlockSparse_Lt_Df_Bt"); // Sparse TM, (L)lit_budget = true, (D)dynamic_AL = false, (B)btp = true
    define_clause_block_sparse<ClauseBlockSparseImpl<true, false, false>>(m, "ClauseBlockSparse_Lt_Df_Bf"); // Sparse TM, (L)lit_budget = true, (D)dynamic_AL = false, (B)btp = false
    define_clause_block_sparse<ClauseBlockSparseImpl<false, true, true>>(m, "ClauseBlockSparse_Lf_Dt_Bt"); // Sparse TM, (L)lit_budget = false, (D)dynamic_AL = true, (B)btp = true
    define_clause_block_sparse<ClauseBlockSparseImpl<false, true, false>>(m, "ClauseBlockSparse_Lf_Dt_Bf"); // Sparse TM, (L)lit_budget = false, (D)dynamic_AL = true, (B)btp = false
    define_clause_block_sparse<ClauseBlockSparseImpl<false, false, true>>(m, "ClauseBlockSparse_Lf_Df_Bt"); // Sparse TM, (L)lit_budget = false, (D)dynamic_AL = false, (B)btp = true
    define_clause_block_sparse<ClauseBlockSparseImpl<false, false, false>>(m, "ClauseBlockSparse_Lf_Df_Bf"); // Sparse TM, (L)lit_budget = false, (D)dynamic_AL = false, (B)btp = false


    typedef typename gt::Inference<uint8_t, false, false, false>    Inference8u_Ff_Lf_Wf;
    typedef typename gt::Inference<uint8_t, false, true, false>     Inference8u_Ff_Lt_Wf;
    typedef typename gt::Inference<uint8_t, false, true, true>      Inference8u_Ff_Lt_Wt;
    typedef typename gt::Inference<uint8_t, true, false, false>     Inference8u_Ft_Lf_Wf;
    typedef typename gt::Inference<uint8_t, true, false, true>      Inference8u_Ft_Lf_Wt;
    typedef typename gt::Inference<uint8_t, true, true, false>      Inference8u_Ft_Lt_Wf;
    typedef typename gt::Inference<uint8_t, true, true, true>       Inference8u_Ft_Lt_Wt;

    //
    define_inference_module<Inference8u_Ff_Lf_Wf>(m, "Inference8u_Ff_Lf_Wf"); // Feature Importance : false , Literal Importance : false, Exclude Negative Clauses : false
    // Literal 
    define_inference_module<Inference8u_Ff_Lt_Wf>(m, "Inference8u_Ff_Lt_Wf"); 
    define_inference_module<Inference8u_Ff_Lt_Wt>(m, "Inference8u_Ff_Lt_Wt"); 

    // Feature
    define_inference_module<Inference8u_Ft_Lf_Wf>(m, "Inference8u_Ft_Lf_Wf"); 
    define_inference_module<Inference8u_Ft_Lf_Wt>(m, "Inference8u_Ft_Lf_Wt");

    // both
    define_inference_module<Inference8u_Ft_Lt_Wf>(m, "Inference8u_Ft_Lt_Wf"); 
    define_inference_module<Inference8u_Ft_Lt_Wt>(m, "Inference8u_Ft_Lt_Wt");



    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
