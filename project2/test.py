import pytest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import activations

def toy_gd():
    # just an intuition test for gradient descent
    a = 10
    b = 3
    x = 0.1
    res = a*(x-10)**2 + b # y = a(x-10)**2 + b:  min. for x = 10, y = b = 3
    # dres/dx = a
    lr = lambda k: 1/(k+1) # learning rate decay

    for i in tqdm(range(100)):
        x = x - lr(i)*2*a*(x-10)
        y = a*(x-10)**2 + b
    print('Final x', x, 'Final y', y)

def test_relu():
    # Test Relu activation
    relu = activations.Relu()
    # random test array
    test = np.random.uniform(-1, 1, (10, 40))
    relued = relu(test) # apply relu in two different ways
    jerry_rigged = np.where(test < 0, 0, test)
    assert np.allclose(relued, jerry_rigged), 'Relu wrong'


if __name__ == '__main__':
    #toy_gd()
    test_relu()
