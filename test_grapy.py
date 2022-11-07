import unittest
from grapy import numpy_extensions as ne
from grapy import grapy as gp
import numpy as np

class TestNpExtensions(unittest.TestCase):
    pass


class TestGrapy(unittest.TestCase):
    double_tris = gp.from_edges([[1,2], [2,3], [3,1], [3,4], [4,5], [5,6], [6,4], [6,1]])

    def test_from_edges(self):
        edges = [[1,2], [2,3], [3,2]]
        expected_edges = [[1,2], [2,3]]
        g = gp.from_edges(edges)
        self.assertArrayEqual(g.verts, np.array([1,2,3]))
        self.assertArrayEqual(g.edges, np.array(expected_edges))

    def test_len(self):
        self.assertEqual(len(TestGrapy.double_tris), 6)

    def test_get_edges_1v(self):
        result = TestGrapy.double_tris.get_edges(1)
        self.assertArrayEqual(result, np.array([[1,2],[1,3],[1,6]]))
    def test_get_edges_vs(self):
        result = TestGrapy.double_tris.get_edges([1,2])
        self.assertArrayEqual(result, np.array([[1,2],[1,3],[1,6],[2,3]]))
    def test_get_edges_from_to_vs(self):
        result = TestGrapy.double_tris.get_edges([1, 2, 3], [4, 5, 6])
        self.assertArrayEqual(result, np.array([[1,6],[3,4]]))

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
        self.assertTrue([2,5] in TestGrapy.double_tris)
    def test_contains_vs_false(self):
        self.assertFalse([1,5,7] in TestGrapy.double_tris)
    #def test_contains_1e_true(self):
    #    self.assertTrue([1,2] in TestGrapy.double_tris)
    #    self.assertTrue([2,1] in TestGrapy.double_tris)
    #def test_contains_1e_false(self):
    #    self.assertFalse([2,5] in TestGrapy.double_tris)
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

    def test_sub_1v(self):
        g = gp.from_edges([[1,2], [2,3]])
        result = g-3
        expected = gp.from_edges([[1,2]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)
    def test_sub_vs(self):
        g = gp.from_edges([[1,2], [2,3], [3,4], [4,5], [5,6]])
        result = g-[3,4]
        expected = gp.from_edges([[1,2], [5,6]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)
    def test_sub_1e(self):
        g = gp.from_edges([[1,2], [2,3], [3,1]])
        result1 = g-[[1,3]]
        result2 = g-[[3,1]]
        expected = gp.from_edges([[1,2], [2,3]])
        self.assertArrayEqual(result1.verts, expected.verts)
        self.assertArrayEqual(result1.edges, expected.edges)
        self.assertArrayEqual(result2.verts, expected.verts)
        self.assertArrayEqual(result2.edges, expected.edges)
    def test_sub_es(self):
        g = gp.from_edges([[1,2], [2,3], [3,4], [4,5]])
        result1 = g-[[2,3], [3,4]]
        result2 = g-[[2,3], [4,3]]
        expected = gp.graph([1,2,3,4,5], [[1,2], [4,5]])
        self.assertArrayEqual(result1.verts, expected.verts)
        self.assertArrayEqual(result1.edges, expected.edges)
        self.assertArrayEqual(result2.verts, expected.verts)
        self.assertArrayEqual(result2.edges, expected.edges)
    def test_sub_g(self):
        g1 = gp.from_edges([[4,5], [5,6], [6,4]])
        g2 = gp.from_edges([[5,4], [6,5], [4,6]])
        result1 = TestGrapy.double_tris - g1
        result2 = TestGrapy.double_tris - g2
        expected = gp.from_edges([[1,2], [2,3], [3,1]])
        self.assertArrayEqual(result1.verts, expected.verts)
        self.assertArrayEqual(result1.edges, expected.edges)
        self.assertArrayEqual(result2.verts, expected.verts)
        self.assertArrayEqual(result2.edges, expected.edges)
    
    def test_add_1v(self):
        g = gp.from_edges([[1,2]])
        result = g + 3
        expected = gp.graph([1,2,3], [[1,2]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)
    def test_add_vs(self):
        g = gp.from_edges([[1,2]])
        result = g + [3,4]
        expected = gp.graph([1,2,3,4], [[1,2]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)
    def test_add_es(self):
        g = gp.from_edges([[1,2], [3,4]])
        result = g + [[2,3], [4,1]]
        expected = gp.graph([1,2,3,4], [[1,2], [2,3], [3,4], [4,1]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)
    def test_add_g(self):
        g1 = gp.from_edges([[1,2], [3,4]])
        g2 = gp.from_edges([[1,2], [2,3], [4,1]])
        result = g1 + g2
        expected = gp.graph([1,2,3,4], [[1,2], [2,3], [3,4], [4,1]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)

    def test_deg_1v(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertEqual(g.deg(2), 2)
        self.assertEqual(g.deg(4), 0)
    def test_deg_vs(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        expected = np.array([2,0])
        self.assertArrayEqual(g.deg([2,4]), expected)
    def test_deg(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        expected = np.array([1,2,1,0])
        self.assertArrayEqual(g.deg(), expected)

    def test_mindeg_0(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertArrayEqual(g.mindeg(), 0)
    def test_mindeg_1(self):
        g = gp.graph([1,2,3], [[1,2],[2,3]])
        self.assertArrayEqual(g.mindeg(), 1)
    def test_maxdeg_0(self): pass
    def test_maxdeg_2(self): pass
    def test_argmindeg(self): pass
    def test_argmaxdeg(self): pass


    def test_perimeter(self): pass # umfang -> länge des längsten kreises in graph
    def test_waistsize(self): pass # taillenweite -> länge eines kuerzesten kreises in graph
    def test_diameter(self): pass # durchmesser -> größter abstand zweier ecken in graph

    def test_center(self): pass # zentrale ecke -> größter abstand von anderen ecken möglichst klein
    def test_radius(self): pass # größter abstand von center zu ecke

    def test_is_connected(self): pass
    def test_components(self): pass

    def test_is_forest(self): pass
    def test_is_tree(self): pass

    def assertArrayEqual(self, first, second):
        result = np.array_equal(first, second)
        msg = '' if result else f'arrays are different\n{first}\n!=\n{second}'
        self.assertTrue(result, msg)