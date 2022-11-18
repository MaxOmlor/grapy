import unittest
from grapy import numpy_extensions as ne
from grapy import grapy as gp
import numpy as np

class TestCaseNp(unittest.TestCase):
    def assertArrayEqual(self, first, second):
        result = np.array_equal(first, second)
        msg = '' if result else f'arrays are different\n{first}\n!=\n{second}'
        self.assertTrue(result, msg)

class TestNumpyExtensions(TestCaseNp):
    def test_contains_1d(self):
        a = [0,1,2]
        b = [0,3,1,4]
        self.assertArrayEqual(ne.contains(a,b), [True, False, True, False])
    def test_contains_2d_2(self):
        a = [[0,1],[1,2]]
        b = [[0,1],[0,2],[1,2]]
        self.assertArrayEqual(ne.contains(a,b), [True, False, True])
    def test_contains_2d_3(self):
        a = [[0,1,2],[1,2,3]]
        b = [[0,1,2],[0,2,4],[1,2,3]]
        self.assertArrayEqual(ne.contains(a,b), [True, False, True])
    def test_contains_3d_2(self):
        a = np.arange(8).reshape((2,2,2))
        b = [[[0,1],[2,3]], [[4,5],[6,0]]]
        self.assertArrayEqual(ne.contains(a,b), [True, False])

    def test_argcontains(self): pass

    def test_replace_single_vals(self):
        a = [1,2,3,2,1]
        expected = [0,2,3,2,0]
        self.assertArrayEqual(ne.replace(a, 1, 0), expected)
    def test_replace_multi_vals(self):
        a = [1,2,3,2,1]
        expected = [-1,-2,3,-2,-1]
        self.assertArrayEqual(ne.replace(a, [1,2], [-1,-2]), expected)
    def test_replace_multi_dims(self):
        a = [[1,2],[1,3],[2,4]]
        expected = [[0, 1],[0, 2],[1, 3]]
        self.assertArrayEqual(ne.replace(a, [1,2,3,4], [0,1,2,3]), expected)



class TestGrapy(TestCaseNp):
    double_tris = gp.from_edges([[1,2], [2,3], [3,1], [3,4], [4,5], [5,6], [6,4], [6,1]])

    def test_from_edges_empty(self):
        g = gp.from_edges([])
        self.assertArrayEqual(g.verts, [])
        self.assertArrayEqual(g.edges, [])
    def test_from_edges_es(self):
        edges = [[1,2], [2,3], [3,2]]
        expected_edges = [[1,2], [2,3]]
        g = gp.from_edges(edges)
        self.assertArrayEqual(g.verts, [1,2,3])
        self.assertArrayEqual(g.edges, expected_edges)
    def test_from_adjacency_mtx_empty(self):
        g = gp.from_adjacency_mtx([])
        self.assertArrayEqual(g.verts, [])
        self.assertArrayEqual(g.edges, [])
    def test_from_adjacency_mtx(self):
        g = gp.from_adjacency_mtx([
            [0,1,0],
            [1,0,1],
            [0,1,0],
        ])
        self.assertArrayEqual(g.verts, [0,1,2])
        self.assertArrayEqual(g.edges, [[0,1],[1,2]])
    def test_from_adjacency_mtx_bool(self):
        g = gp.from_adjacency_mtx([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        self.assertArrayEqual(g.verts, [0,1,2])
        self.assertArrayEqual(g.edges, [[0,1],[1,2]])


    def test_len(self):
        self.assertEqual(len(TestGrapy.double_tris), 6)

    def test_get_edges_1v_graph(self): pass
    def test_get_edges_1v(self):
        result = TestGrapy.double_tris.get_edges(1)
        self.assertArrayEqual(result, [[1,2],[1,3],[1,6]])
    def test_get_edges_vs(self):
        result = TestGrapy.double_tris.get_edges([1,2])
        self.assertArrayEqual(result, [[1,2],[1,3],[1,6],[2,3]])
    def test_get_edges_from_to_vs(self):
        result = TestGrapy.double_tris.get_edges([1, 2, 3], [4, 5, 6])
        self.assertArrayEqual(result, [[1,6],[3,4]])

    def test_neighb_1v(self):
        result = TestGrapy.double_tris.neighb(1)
        self.assertArrayEqual(result, [2,3,6])
    def test_neighb_vs(self):
        result = TestGrapy.double_tris.neighb([1,3])
        self.assertArrayEqual(result, [2,4,6])

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
    def test_contains_empty_g(self): pass

    def test_get_item(self):
        result = TestGrapy.double_tris[[1,2,3,5]]
        expected = gp.graph([1,2,3,5], [[1,2], [2,3], [3,1]])
        self.assertArrayEqual(result.verts, expected.verts)
        self.assertArrayEqual(result.edges, expected.edges)

    def test_sub_empty_g(self): pass
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
    
    def test_add_empty_g(self): pass
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

    def test_deg_0(self):
        g = gp.graph([1], [])
        self.assertArrayEqual(g.deg(1), 0)
        g = gp.graph([], [])
        self.assertArrayEqual(g.deg(), np.array([]))
    def test_deg_1v(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertEqual(g.deg(2), 2)
        self.assertEqual(g.deg(4), 0)
    def test_deg_vs(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertArrayEqual(g.deg([2,4]), [2,0])
    def test_deg(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertArrayEqual(g.deg(), [1,2,1,0])

    def test_mindeg_0(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertArrayEqual(g.mindeg(), 0)
    def test_mindeg_1(self):
        g = gp.from_edges([[1,2],[2,3]])
        self.assertArrayEqual(g.mindeg(), 1)
    def test_maxdeg_0(self):
        g = gp.graph([1,2], [])
        self.assertArrayEqual(g.maxdeg(), 0)
    def test_maxdeg_2(self):
        g = gp.from_edges([[1,2],[2,3]])
        self.assertArrayEqual(g.maxdeg(), 2)
    def test_argmindeg_0(self):
        g = gp.graph([1,2,3,4], [[1,2],[2,3]])
        self.assertArrayEqual(g.argmindeg(), 4)
    def test_argmindeg_0_vs(self):
        g = gp.graph([1,2,3,4,5], [[1,2],[2,3]])
        self.assertArrayEqual(g.argmindeg(), [4,5])
    def test_argmindeg_1(self):
        g = gp.from_edges([[1,2],[2,3],[3,1],[3,4]])
        self.assertArrayEqual(g.argmindeg(), 4)
    def test_argmindeg_1_vs(self):
        g = gp.from_edges([[1,2],[2,3]])
        self.assertArrayEqual(g.argmindeg(), [1,3])
    def test_argmaxdeg_0(self):
        g = gp.graph([1], [])
        self.assertArrayEqual(g.argmaxdeg(), 1)
    def test_argmaxdeg_0_vs(self):
        g = gp.graph([1,2], [])
        self.assertArrayEqual(g.argmaxdeg(), [1,2])
    def test_argmaxdeg_2(self):
        g = gp.from_edges([[1,2],[2,3]])
        self.assertArrayEqual(g.argmaxdeg(), 2)
    def test_argmaxdeg_2_vs(self):
        g = gp.from_edges([[1,2],[2,3],[3,1]])
        self.assertArrayEqual(g.argmaxdeg(), [1,2,3])

    def test_adjacency_mtx_empty(self):
        g = gp.graph([], [])
        self.assertArrayEqual(g.adjacency_mtx(), [])
    def test_adjacency_mtx(self):
        g = gp.from_edges([[1,2],[2,3]])
        expected = [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
        self.assertArrayEqual(g.adjacency_mtx(), expected)
    def test_adjacency_tree(self):
        g = gp.from_edges([[1,2], [2,3], [1,4]])
        expected = [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ]
        self.assertArrayEqual(g.adjacency_mtx(), expected)
    

    def test_cycles(self): pass
    def test_perimeter(self): pass # umfang -> länge des längsten kreises in graph
    def test_waistsize(self): pass # taillenweite -> länge eines kuerzesten kreises in graph
    def test_diameter(self): pass # durchmesser -> größter abstand zweier ecken in graph

    def test_center(self): pass # zentrale ecke -> größter abstand von anderen ecken möglichst klein
    def test_radius(self): pass # größter abstand von center zu ecke

    def test_is_connected(self): pass
    def test_components(self): pass

    def test_is_forest(self): pass
    def test_is_tree(self): pass