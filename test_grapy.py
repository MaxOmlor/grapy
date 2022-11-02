import unittest
from grapy import np_extensions as ne
from grapy import grapy as gp
import numpy as np

class TestNpExtensions(unittest.TestCase):
    pass


class TestGrapy(unittest.TestCase):
    double_tris = gp.from_edges([[1,2], [2,3], [3,1], [3,4], [4,5], [5,6], [6,4], [6,1]])

    def test_from_edges(self):
        edges = [[1,2], [2,3]]
        g = gp.from_edges(edges)
        self.assertArrayEqual(g.verts, np.array([1,2,3]))
        self.assertArrayEqual(g.edges, np.array(edges))

    def test_len(self):
        self.assertEqual(len(TestGrapy.double_tris), 6)

    def test_get_edges_1v(self):
        result = TestGrapy.double_tris.get_edges(1)
        self.assertArrayEqual(result, np.array([[1,2],[3,1],[6,1]]))
    def test_get_edges_vs(self):
        result = TestGrapy.double_tris.get_edges([1,2])
        self.assertArrayEqual(result, np.array([[1,2],[2,3],[3,1],[6,1]]))
    def test_get_edges_from_to_vs(self):
        result = TestGrapy.double_tris.get_edges([1, 2, 3], [4, 5, 6])
        self.assertArrayEqual(result, np.array([[3,4],[6,1]]))

    def test_neighb_1v(self):
        result = TestGrapy.double_tris.neighb(1)
        self.assertArrayEqual(result, np.array([2,3,6]))
    def test_neighb_vs(self):
        result = TestGrapy.double_tris.neighb([1,3])
        self.assertArrayEqual(result, np.array([1,2,3,4,6]))

    def test_contains_1v_true(self):
        self.assertTrue(1 in TestGrapy.double_tris)
    def test_contains_1v_false(self):
        self.assertFalse(7 in TestGrapy.double_tris)
    def test_contains_vs_true(self):
        self.assertTrue([1,2] in TestGrapy.double_tris)
        self.assertTrue([2,1] in TestGrapy.double_tris)
    def test_contains_vs_false(self):
        self.assertFalse([1,5,7] in TestGrapy.double_tris)
    def test_contains_1e_true(self):
        self.assertTrue([1,2] in TestGrapy.double_tris)
        self.assertTrue([2,1] in TestGrapy.double_tris)
    def test_contains_1e_false(self):
        self.assertFalse([2,5] in TestGrapy.double_tris)
    def test_contains_es_true(self):
        self.assertTrue([[1,2],[2,3]] in TestGrapy.double_tris)
        self.assertTrue([[1,2],[3,2]] in TestGrapy.double_tris)
    def test_contains_es_false(self):
        self.assertFalse([[1,2],[2,5]] in TestGrapy.double_tris)
    def test_contains_g_true(self):
        g = gp.from_edges([[1,2],[2,3]])
        self.assertTrue(g in TestGrapy.double_tris)
    def test_contains_g_false(self):
        g = gp.from_edges([[1,2],[2,5]])
        self.assertFalse(g in TestGrapy.double_tris)

    def test_get_item(self):
        result = TestGrapy.double_tris[[1,2,3,5]]
        expected = gp.graph([1,2,3,5], [[1,2], [2,3], [3,1]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)


    
    def assertArrayEqual(self, first, second):
        self.assertTrue(np.array_equal(first, second))