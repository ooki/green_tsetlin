import warnings

import py_gtc

impl = {
    "cb" : py_gtc.ClauseBlock,
    "conv_cb" : None,
    "sparse_cb" : None,
    "feedback" : py_gtc.FeedbackBlock,
    "feedback_multi" : py_gtc.FeedbackBlockMultiLabel,
    "single_executor": py_gtc.SingleThreadExecutor,
    "thread_executor" : py_gtc.MultiThreadExecutor,
    "dense_input" : py_gtc.DenseInputBlock,
    "sparse_input" : None
}


try: 
    import green_tsetlin_core as gtc
    
    impl["cb"] = gtc.ClauseBlockTM
    impl["conv_cb"] = gtc.ClauseBlockConvTM,
    # impl["sparse_cb"] = None # fallback
    impl["feedback"] = gtc.FeedbackBlock
    impl["feedback_multi"] = gtc.FeedbackBlockMultiLabel
    impl["single_executor"] = gtc.SingleThreadExecutor
    impl["thread_executor"] =  gtc.MultiThreadExecutor
    impl["dense_input"] = gtc.DenseInputBlock
    # impl["sparse_input"] = None # fallback
    
    
    if gtc.has_avx2():
        impl["cb"] = gtc.ClauseBlockAVX2
        impl["conv_cb"] = gtc.ClauseBlockConvAVX2
        
    if gtc.has_neon():
        impl["cb"] = gtc.ClauseBlockNeon
        impl["conv_cb"] = gtc.ClauseBlockConvNeon


except ImportError:
    warnings.warn("Cannot load c++ backend (green_tsetlin_core) fallback to pure python.")
    





