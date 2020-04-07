#!/usr/bin/env python

import unittest

from unittest.mock import MagicMock

import nbconvert

import numpy as np



with open("assignment13.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)

with open("assignment13.py", "w") as f:
    f.write(python_file)

from assignment13 import LinearSystem

try:
    from assignment12 import Matrix
except:
    pass

try:
    from assignment13 import Matrix
except:
    pass

class TestSolution(unittest.TestCase):
    
    def test_is_derived_from_Matrix(self):
        
        self.assertTrue(issubclass(LinearSystem, Matrix))
        
    def test_row_swap_called(self):
        
        A = np.array([[1, 2], [4, 3]])
        ls = LinearSystem(A)
        ls.row_swap = MagicMock()
        ls.row_echelon()
        assert ls.row_swap.called
        
    def test_row_combine_called(self):
        
        A = np.array([[1, 2], [4, 3]])
        ls = LinearSystem(A)
        ls.row_combine = MagicMock()
        ls.row_echelon()
        assert ls.row_combine.called
        
    def test_row_echelon(self):
        
        A = np.array([[1, 3, 4],[5, 4, 2],[1, 7, 9]])
        ls = LinearSystem(A)
        ls.row_echelon() 
        np.testing.assert_array_almost_equal(ls.mat, 
                                             np.array([[ 5., 4., 2.],
                                                       [0., 6.2, 8.6],
                                                       [ 0., 0., 0.5483871]]),                                                      decimal=6)
        
    def test_back_substitute(self):
        
        A = np.array([[1, 3, 5],[5, 2, 2],[1, 7, 1]])
        b = np.array([1, 3, 4])
        ls = LinearSystem(A, b)
        ls.row_echelon()
        np.testing.assert_array_almost_equal(ls.back_substitute(),
                                             np.linalg.solve(A, b),
                                             decimal=6)
        
    def test_gauss_solve(self):
        
        A = np.array([[1, 3, 5],[5, 2, 2],[1, 7, 1]])
        b = np.array([1, 3, 4])
        ls = LinearSystem(A, b)
        np.testing.assert_array_almost_equal(ls.gauss_solve(),
                                             np.linalg.solve(A, b),
                                             decimal=6)
        
    def test_reduced_row_echelon(self):
        
        A = np.array([[1, 3, 5],[5, 2, 2],[1, 7, 1]])
        b = np.array([1, 3, 4])
        ls = LinearSystem(A, b)
        ls.reduced_row_echelon()
        np.testing.assert_array_almost_equal(ls.mat[:,-1],
                                             np.linalg.solve(A, b),
                                             decimal=6)
        
    def test_inverse(self):
        
        A = np.array([[1, 2, 5],[5, 22, 17],[11, 7, 1]])
        ls = LinearSystem(A)
        np.testing.assert_array_almost_equal(ls.inverse(),
                                             np.linalg.inv(A),
                                             decimal=6)
        
    
if __name__ == '__main__':
    unittest.main()
