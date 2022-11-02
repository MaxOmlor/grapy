import unittest
from grapy import np_extensions as ne
from grapy import grapy as gp
import numpy as np

class TestNpExtensions(unittest.TestCase):
    pass


class TestGrapy(unittest.TestCase):
    double_tris = gp.from_edges([[1,2], [2,3], [3,1], [3,4], [4,5], [5,6], [6,4], [6,1]])

    def test_len(self):
        self.assertEqual(len(TestGrapy.double_tris), 6)

    def test_get_edges_1v(self):
        result = TestGrapy.double_tris.get_edges(1)
        self.assertTrue(np.array_equal(result, np.array([[1,2],[3,1],[6,1]])))
    def test_get_edges_vs(self):
        result = TestGrapy.double_tris.get_edges([1,2])
        self.assertTrue(np.array_equal(result, np.array([[1,2],[2,3],[3,1],[6,1]])))
    def test_get_edges_from_to_vs(self):
        result = TestGrapy.double_tris.get_edges([1, 2, 3], [4, 5, 6])
        self.assertTrue(np.array_equal(result, np.array([[3,4],[6,1]])))

    def test_neighb_1v(self):
        result = TestGrapy.double_tris.neighb(1)
        self.assertTrue(np.array_equal(result, np.array([2,3,6])))
    def test_neighb_vs(self):
        result = TestGrapy.double_tris.neighb([1,3])
        self.assertTrue(np.array_equal(result, np.array([1,2,3,4,6])))
