
import numpy as np
import pytest 
import green_tsetlin_core as gtc 


def test_votes_register_rest_get():
    
    n_classes = 7
    threshold = 30
    seed = 41
    fb = gtc.FeedbackBlock(n_classes, threshold, seed)
    
    votes = np.arange(n_classes).astype(np.int16)
    votes[0] = -42
    
    fb.register_votes(votes)
    fb.register_votes(votes)
    fb.register_votes(votes)
        
    out = fb.get_votes()        
    assert np.array_equal(out, votes*3)
    
    fb.reset_votes()
    out = fb.get_votes()        
    assert (out==0).all()
    
    
def test_number_of_classes():
    
    n_classes = 7
    threshold = 30
    seed = 41
    fb = gtc.FeedbackBlock(n_classes, threshold, seed)    
    fb_multi = gtc.FeedbackBlockMultiLabel(n_classes, threshold, seed)
    
    assert fb.get_number_of_classes() == n_classes
    assert fb_multi.get_number_of_classes() == n_classes
    
def test_multi_gives_2x_num_classes_for_votes():
    n_classes = 7
    threshold = 30
    seed = 41
    fb = gtc.FeedbackBlock(n_classes, threshold, seed)    
    fb_multi = gtc.FeedbackBlockMultiLabel(n_classes, threshold, seed)
    
    n_fb_votes = fb.get_votes().shape[0]
    n_fb_multi_votes = fb_multi.get_votes().shape[0]
    
    assert n_fb_votes*2 == n_fb_multi_votes
    
    
if __name__ == "__main__":
    # test_votes_register_rest_get()
    # test_number_of_classes()
    test_multi_gives_2x_num_classes_for_votes()
    
    print("<done>")