import warnings

import green_tsetlin.py_gtc as py_gtc

impl = {
    "cb" : py_gtc.ClauseBlock,
    "conv_cb" : None,
    "sparse_cb" : py_gtc.SparseInputBlock, # for now own ib, will see if can be same as denseIb.
    "feedback" : py_gtc.FeedbackBlock,
    "feedback_uniform" : None,
    "feedback_multi" : py_gtc.FeedbackBlockMultiLabel,    
    "single_executor": py_gtc.SingleThreadExecutor,
    "thread_executor" : py_gtc.MultiThreadExecutor,
    "dense_input" : py_gtc.DenseInputBlock,
    "sparse_input" : py_gtc.SparseInputBlock,
    "sparse_input_dense_output": py_gtc.SparseInpuDenseOutputBlock
}


try: 
    import green_tsetlin_core as gtc
    
    impl["cb"] = gtc.ClauseBlockTM_Lt_Bt
    impl["conv_cb"] = gtc.ClauseBlockConvTM,
    impl["sparse_cb"] = gtc.ClauseBlockSparse_Lt_Dt_Bf # fallback
    impl["feedback"] = gtc.FeedbackBlock
    impl["feedback_uniform"] = gtc.FeedbackBlockUniform
    impl["feedback_multi"] = gtc.FeedbackBlockMultiLabel
    impl["single_executor"] = gtc.SingleThreadExecutor
    impl["thread_executor"] =  gtc.MultiThreadExecutor
    impl["dense_input"] = gtc.DenseInputBlock
    impl["sparse_input"] = gtc.SparseInputBlock
    impl["sparse_input_dense_output"] = gtc.SparseInputDenseOutputBlock

    impl["cb_Lt_Bt"] = gtc.ClauseBlockTM_Lt_Bt # (L)lit_budget = true, (B)boost_true_positives = true
    impl["cb_Lt_Bf"] = gtc.ClauseBlockTM_Lt_Bf
    impl["cb_Lf_Bt"] = gtc.ClauseBlockTM_Lf_Bt
    impl["cb_Lf_Bf"] = gtc.ClauseBlockTM_Lf_Bf

    impl["sparse_cb_Lt_Dt_Bt"] = gtc.ClauseBlockSparse_Lt_Dt_Bt # (L)lit_budget = true, (D)dynamic_AL = true, (B)boost_true_positives = true
    impl["sparse_cb_Lt_Dt_Bf"] = gtc.ClauseBlockSparse_Lt_Dt_Bf
    impl["sparse_cb_Lt_Df_Bt"] = gtc.ClauseBlockSparse_Lt_Df_Bt
    impl["sparse_cb_Lt_Df_Bf"] = gtc.ClauseBlockSparse_Lt_Df_Bf
    impl["sparse_cb_Lf_Dt_Bt"] = gtc.ClauseBlockSparse_Lf_Dt_Bt
    impl["sparse_cb_Lf_Dt_Bf"] = gtc.ClauseBlockSparse_Lf_Dt_Bf
    impl["sparse_cb_Lf_Df_Bt"] = gtc.ClauseBlockSparse_Lf_Df_Bt
    impl["sparse_cb_Lf_Df_Bf"] = gtc.ClauseBlockSparse_Lf_Df_Bf

    impl["Inference8u_Ff_Lf_Wf"] = gtc.Inference8u_Ff_Lf_Wf    
    impl["Inference8u_Ff_Lt_Wf"] = gtc.Inference8u_Ff_Lt_Wf
    impl["Inference8u_Ff_Lt_Wt"] = gtc.Inference8u_Ff_Lt_Wt    
    impl["Inference8u_Ft_Lf_Wf"] = gtc.Inference8u_Ft_Lf_Wf
    impl["Inference8u_Ft_Lf_Wt"] = gtc.Inference8u_Ft_Lf_Wt    
    impl["Inference8u_Ft_Lt_Wf"] = gtc.Inference8u_Ft_Lt_Wf
    impl["Inference8u_Ft_Lt_Wt"] = gtc.Inference8u_Ft_Lt_Wt
    
    if gtc.has_avx2():
        impl["cb"] = gtc.ClauseBlockAVX2_Lt_Bt
        impl["conv_cb"] = gtc.ClauseBlockConvAVX2
    
        impl["cb_Lt_Bt"] = gtc.ClauseBlockAVX2_Lt_Bt # (L)lit_budget = true, (B)boost_true_positives = true
        impl["cb_Lt_Bf"] = gtc.ClauseBlockAVX2_Lt_Bf
        impl["cb_Lf_Bt"] = gtc.ClauseBlockAVX2_Lf_Bt
        impl["cb_Lf_Bf"] = gtc.ClauseBlockAVX2_Lf_Bf

    if gtc.has_neon():
        impl["cb_Lt_Bt"] = gtc.ClauseBlockNeon_Lt_Bt # (L)lit_budget = true, (B)boost_true_positives = true
        impl["cb_Lt_Bf"] = gtc.ClauseBlockNeon_Lt_Bf
        impl["cb_Lf_Bt"] = gtc.ClauseBlockNeon_Lf_Bt
        impl["cb_Lf_Bf"] = gtc.ClauseBlockNeon_Lf_Bf


except ImportError:
    warnings.warn("Cannot load c++ backend (green_tsetlin_core) fallback to pure python.")
    





