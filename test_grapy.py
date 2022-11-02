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