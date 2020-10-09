import pytest
import numpy as np

import regression as reg

def test_ols():
    # test OLS class
    a = 2  # f = ax + b
    b = 1
    x = np.linspace(0, 1, 10)
    y = a*x + b
    xx = np.stack((x, x*0), axis = -1) # [10, 2]
    linear = reg.Linear(1) # 1st order polynomial
    linear.fit(xx, y)
    print(linear.beta_) # expected result [2, 1, 0]

def test_design_matrix_algo():
    all_inds = ['0', '1', '2', '3', '4', '5']
    p = 3 # should find unique combos of indices (powers) up to x^3, y^3 (30, 03)
    for i in range(p+1):
        inds = [str(j) for j in range(i+1)]
        d = []
        for y in range(i+1):
            for x in range(i+1 - y):
                d.append(inds[x]+inds[y])
        print(d)

if __name__ == '__main__':
    print('Regression coeffs. for 2x + 1')
    test_ols()
    print('\nDesign matrix indices for p = 3')
    test_design_matrix_algo()
    # was supposed to implement more tests using pytest,
    # but didn't have time bc. of midterms :(
