
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
    
    
    
if __name__ == "__main__":
    test_votes_register_rest_get()
    
    print("<done>")