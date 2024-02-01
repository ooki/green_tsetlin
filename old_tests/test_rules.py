
import pytest
import numpy as np
import pickle

import green_tsetlin as gt


def get_simple_state():
    n_literals = 2
    n_clauses = 3
    n_classes = 2

    c = np.zeros((n_clauses, n_literals*2), dtype=np.int8)
    w = np.ones((n_clauses, n_classes), dtype=np.int16)

    return {"w": w, "c": c}


def get_xor_state():
    c = np.array([  [4, -128, -128, -127, -97, -128, -117, -126, -79, -122, -128, 3, -124, -82, -122, -107, -54, -113, -122, -128],
                    [-19, -61, -68, -2, -47, -124, -9, -70, -2, -18, -67, 2, -2, -56, -22, -19, -34, -13, -4, -10],
                    [-8, -55, -10, -41, -39, -1, -14, -15, -8, -88, -35, -48, -9, -61, -8, 4, -21, -23, -14, -28],
                    [-124, -127, -123, -93, -105, -124, -118, -101, -115, -92, 5, 10, -119, -121, -113, -126, -115, -127, -123, -128],
                    [-37, -41, 2, -2, -51, -88, -9, -21, -53, -21, -64, -17, -1, -45, -86, -13, -80, -4, -44, -1],
                    [2, 2, -128, -122, -123, -110, -109, -103, -121, -120, -125, -123, -119, -118, -108, -124, -109, -100, -107, -62],
                    [-23, -75, -4, -69, -93, -9, -61, -47, 0, -17, -29, -27, -78, -33, -2, -53, -39, -11, 0, -28],
                    [-116, 1, -114, -124, -121, -111, -122, -110, -128, -115, 2, -126, -124, -126, -126, -121, -103, -124, -117, -124],
                    [-36, -29, -29, -33, -34, -15, -34, -126, -48, -46, -40, -2, -7, -3, -30, -57, -17, 1, 0, -8],
                    [-99, -9, -21, -44, 0, -24, -10, -24, -44, -23, 0, -29, -6, -58, 0, -88, -2, -58, -76, -2],
                    [2, -19, -46, -92, -109, -54, -48, -70, -35, -17, 0, -90, -13, -17, -28, -71, -2, -43, -33, -85],
                    [-1, -121, -59, 0, -19, -18, -28, -112, -17, -29, -21, -5, -1, -85, -25, -21, -32, -5, -83, -45],
                    [-19, -21, -39, -40, -6, -24, 4, -44, -7, -11, -16, -128, -46, -13, -17, -19, 0, -8, -37, -26],
                    [-85, -16, -2, -28, -70, -15, -38, -59, -9, -57, -69, -25, -95, -54, 1, -21, -33, -47, 0, -35],
                    [-50, -22, -20, 1, -35, -37, -34, -3, -58, -3, -70, -19, -15, -48, -15, -14, -32, -37, -19, 0],
                    [1, -2, -26, -34, -34, -94, -50, -64, -49, -67, -53, -61, -50, -8, -10, 0, -11, -28, -116, -10]], dtype=np.int8)
    
    w = np.array([  [-11, 12],
                    [0, 0],
                    [0, -1],
                    [14, -13],
                    [-6, 0],
                    [11, -11],
                    [-1, 2],
                    [-8, 10],
                    [-3, 0],
                    [0, -1],
                    [0, -2],
                    [-2, 1],
                    [0, -2],
                    [0, 3],
                    [1, -2],
                    [3, 2]], dtype=np.int16)
    
    return {"w": w, "c": c}
    

def test_multi_label_output():
    state = get_xor_state()

    w = state["w"]    
    state["w"] = np.concatenate([w, -w, -w, w], axis=1)
    n_weights = state["w"].shape[1]
    n_classes = n_weights // 2

    
    rp = gt.RulePredictor(multi_label=True)
    rp.create_from_state(state)

    x_y0 = np.array([0, 0], dtype=np.uint8)
    out = rp.predict(x_y0)

    assert n_classes == out.shape[0]
    assert rp.n_classes == n_classes


def test_multi_label_output_pickle_support():
    state = get_xor_state()

    w = state["w"]    
    state["w"] = np.concatenate([w, -w, -w, w], axis=1)
    n_weights = state["w"].shape[1]
    n_classes = n_weights // 2

    rp = gt.RulePredictor(multi_label=True)
    rp.create_from_state(state)

    byte_data = pickle.dumps(rp, protocol=pickle.HIGHEST_PROTOCOL)
    del rp

    rp2 = pickle.loads(byte_data)
    x_y0 = np.array([0, 0], dtype=np.uint8)
    out = rp2.predict(x_y0)

    assert rp2.n_classes == n_classes 
    assert n_classes == out.shape[0]

    byte_data = pickle.dumps(rp2, protocol=pickle.HIGHEST_PROTOCOL)
    del rp2

    rp3 = pickle.loads(byte_data)
    x_y0 = np.array([0, 0], dtype=np.uint8)
    out = rp3.predict(x_y0)

    assert rp3.n_classes == n_classes 
    assert n_classes == out.shape[0]




def test_empty_output_default_get_set():

    state = get_simple_state()
    rp = gt.RulePredictor()
    rp.create_from_state(state)

    assert rp.empty_class_output > -1

def test_empty_output_sets():

    state = get_simple_state()
    rp = gt.RulePredictor(empty_class_output=42)
    rp.create_from_state(state)

    o = rp.predict( np.array([1,1], dtype=np.uint8))
    assert rp.get_active_rules().size == 0
    assert o == 42



def test_pickle_support():
    state = get_xor_state()
    rp = gt.RulePredictor()    
    rp.create_from_state(state)

    byte_data = pickle.dumps(rp, protocol=pickle.HIGHEST_PROTOCOL)
    del rp

    rp2 = pickle.loads(byte_data)

    assert rp2.n_classes == 2
    assert hasattr(rp2, "_inference")
    assert rp2._inference



def test_global_importance_xor():

    state = get_xor_state()
    rp = gt.RulePredictor()    
    rp.create_from_state(state)
    
    i0 = rp.get_global_importance(0, True)
    i0_0 = i0[0] / (i0.sum() - i0[1])
    i0_1 = i0[1] / (i0.sum() - i0[0])
    assert i0_0 > 0.45
    assert i0_1 > 0.45
    #print(i0, i0_0, i0_1)
    #assert False
    

    i1 = rp.get_global_importance(1, True)
    i1_0 = i1[0] / (i1.sum() - i1[1])
    i1_1 = i1[1] / (i1.sum() - i1[0])
    assert i1_0 > 0.45
    assert i1_1 > 0.45

    assert i0.sum() > 0.99
    assert i1.sum() > 0.99




def test_local_importance_xor():
    state = get_xor_state()
    n_literals = state["c"].shape[1] // 2

    rp = gt.RulePredictor(support_literal_importance=True)
    rp.create_from_state(state)

    x_y0 = np.array([0, 0], dtype=np.uint8)
    x_y1 = np.array([1, 0], dtype=np.uint8)

    e0, e1 = rp.explain(x_y0, [0, 1])
    e0_0 = e0[0] / (e0.sum() - e0[1])
    e0_1 = e0[1] / (e0.sum() - e0[0])

    assert e0_0 > 0.50
    assert e0_1 > 0.50        
    assert e0.sum() > 0.99
    assert e1.sum() > 0.99 or e1.sum() == 0


    e0, e1 = rp.explain(x_y1, [0, 1])
    e1_0 = e1[0] / (e1.sum() - e1[1])
    e1_1 = e1[1] / (e1.sum() - e1[0])
    assert e1_0 > 0.50
    assert e1_1 > 0.50    
    assert e0.sum() > 0.99 or e0.sum() == 0
    assert e1.sum() > 0.99


    # test literal importance (should be the same as feature importance since we use literal level features)
    (e0_f, e0_l), (e1_f, e1_l) = rp.explain(x_y0, [0, 1], normalize=False, literal_importance=True)

    # make sure we test that literals are mapped correctly (we double up the pos and neg literals)
    o = np.array(e0_l)[:n_literals] + np.array(e0_l)[n_literals:]
    assert np.allclose(o, np.array(e0_f))



    rp_no_lit = gt.RulePredictor(support_literal_importance=False)
    rp_no_lit.create_from_state(state)
    with pytest.raises(ValueError): # should raise attribute error as 
        (e0_f, e0_l), (e1_f, e1_l) = rp_no_lit.explain(x_y0, [0, 1], normalize=False, literal_importance=True)



def test_pickeled_rules_still_solves_xor():

    state = get_xor_state()
    rp = gt.RulePredictor()    
    rp.create_from_state(state)

    byte_data = pickle.dumps(rp, protocol=pickle.HIGHEST_PROTOCOL)
    del rp

    rp2 = pickle.loads(byte_data)
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals= rp2.n_literals)    

    preds = []
    for xk, yk in zip(ex, ey):
        y_pred = rp2.predict(xk)
        preds.append(y_pred==yk)

    acc = np.mean(preds)
    assert acc > 0.99


if __name__ == "__main__":
    # test_empty_output_default_get_set()
    # test_empty_output_sets()
    
    # test_global_importance_xor()        
    # test_local_importance_xor()
    # test_multi_label_output()
    # test_pickle_support()
    test_multi_label_output_pickle_support()
    
    print("<done tests:", __file__, ">")
