
import numpy as np
import green_tsetlin.dataset_generator as ds_g


def test_xor_no_noise():
    #def xor_dataset(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_dummy_features: int = 2,
    #            seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    
    n_literals = 6
    xt, yt, xe, ye = ds_g.xor_dataset(noise=None, n_train=200, n_test=100, n_literals=n_literals, seed=42)

    assert xt.shape == (200, n_literals)
    assert xe.shape == (100, n_literals)
    assert yt.shape == (200,)
    assert ye.shape == (100,)

    no_noise_yt = np.logical_xor(xt[:, 0], xt[:, 1])
    assert np.array_equal(no_noise_yt, yt)

    no_noise_ye = np.logical_xor(xe[:, 0], xe[:, 1])
    assert (no_noise_ye==ye).all()

def test_xor_throws_if_too_few_literals():

    n_literals = 1
    did_throw = False
    try:
        ds_g.xor_dataset(noise=0.1, n_train=50, n_test=30, n_literals=n_literals, seed=42)

    except ValueError:
        did_throw = True 

    assert did_throw is True

def test_xor_noise_20():
    #def xor_dataset(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_dummy_features: int = 2,
    #            seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    
    n_literals = 4
    noise = 0.20
    xt, yt, xe, ye = ds_g.xor_dataset(noise=noise, n_train=500, n_test=300, n_literals=n_literals, seed=42)

    assert xt.shape == (500, n_literals)
    assert xe.shape == (300, n_literals)
    assert yt.shape == (500,)
    assert ye.shape == (300,)

    pred_train = np.logical_xor(xt[:, 0], xt[:, 1])
    pred_acc = (pred_train == yt).mean()
    assert np.abs(pred_acc - (1.0 - noise)) < 0.02

    
    pred_test = np.logical_xor(xe[:, 0], xe[:, 1])
    assert np.array_equal(pred_test, ye)

def test_xor_noise_40():
    #def xor_dataset(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_dummy_features: int = 2,
    #            seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    
    n_literals = 4
    noise = 0.40
    xt, yt, xe, ye = ds_g.xor_dataset(noise=noise, n_train=1000, n_test=300, n_literals=n_literals, seed=42)

    assert xt.shape == (1000, n_literals)
    assert xe.shape == (300, n_literals)
    assert yt.shape == (1000,)
    assert ye.shape == (300,)

    pred_train = np.logical_xor(xt[:, 0], xt[:, 1])
    pred_acc = (pred_train == yt).mean()    
    assert np.abs(pred_acc - (1.0 - noise)) < 0.03
    
    pred_test = np.logical_xor(xe[:, 0], xe[:, 1])
    assert np.array_equal(pred_test, ye)



def test_multi_label_xor():
    n_literals = 10
    n_train = 111
    n_test = 69
    n_classes = 5
    noise = 0.40

    xt, yt, xe, ye = ds_g.multi_label_xor(noise=noise, n_train=n_train, n_test=n_test, n_literals=n_literals, n_classes=n_classes, seed=42)
    
    assert xt.shape == (n_train, n_literals)
    assert xe.shape == (n_test, n_literals)
    
    assert yt.shape == (n_train, n_classes)
    assert ye.shape == (n_test, n_classes)
    
    assert np.isin(yt, np.array([0, 1], dtype=np.uint32)).all()
    assert np.isin(ye, np.array([0, 1], dtype=np.uint32)).all()
        
    for k in range(0, n_classes):
        class_y = np.logical_xor(xe[:, 0], xe[:, k+1])
        assert np.array_equal(class_y, ye[:, k])

        
        
    


if __name__ == "__main__":
    test_xor_no_noise()
    test_xor_noise_20()
    test_xor_noise_40()
    test_xor_throws_if_too_few_literals()
    test_multi_label_xor()
    print("<tests: ", __file__, "- ok>")




