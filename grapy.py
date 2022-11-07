from __future__ import annotations
import numpy as np
import scipy as sp



class numpy_extensions():
    '''
    class dict(dict):
        def __init__(self, iterable=None):
            if iterable:
                super().__init__(iterable)
            else:
                super().__init__()
        def __getitem__(self, __key):
            if isinstance(__key, Iterable) and type(__key) is not np.ndarray:
                __key = np.array(__key)
            if type(__key) is np.ndarray:
                return np.frompyfunc(lambda k: super().__getitem__(k), 1, 1)(__key)
            return super().__getitem__(__key)
        def __setitem__(self, __key, __value) -> None:
            for k, v in zip(__key, __value):
                super().__setitem__(k, v)
    '''

    @classmethod
    def contains_vec(cls, a: np.ndarray, b: np.ndarray, axis=1) -> np.ndarray:
        if type(a) is not np.ndarray:
            a = np.array(a)
        if type(b) is not np.ndarray:
            b = np.array(b)
            
        comparsion = np.expand_dims(b, 1) == a
        for i in range(len(a.shape), axis, -1):
            comparsion = comparsion.all(axis=i)
        return comparsion.any(axis=axis)

    @classmethod
    def replace(cls, a: np.ndarray, values: any, replacements: any) -> np.ndarray:
        if len(a) == 0:
            return np.copy(a)
        if type(replacements) is not np.ndarray:
            replacements = np.array(replacements)

        d = dict(zip(values, replacements))
        result = np.frompyfunc(lambda x: d[x], 1, 1)(a)
        return result.astype(replacements.dtype)
    
    

class grapy():
    class graph():
        def __init__(self, verts: np.ndarray, edges: np.ndarray) -> None:
            if type(verts) is not np.ndarray:
                verts = np.array(verts)
            if type(edges) is not np.ndarray:
                edges = np.array(edges)

            if edges.size > 0 and (len(edges.shape) != 2 or edges.shape[1] != 2):
                raise ValueError(f'edges.shape must be (n,2) not {edges.shape}')

            self.verts = verts
            self.edges = grapy.edges.rm_dupes(edges) if edges.size > 0 else edges

            # predicates
            # - start and end of every edges must be in verts

        def __str__(self) -> str:
            return f'({self.verts}\n{self.edges})'

        def __repr__(self) -> str:
            return f'ggraph(\nedges={repr(self.verts)},\nverts={repr(self.edges)})'

        def __len__(self) -> int:
            return len(self.verts)

        def get_edges(self, verts: int|np.ndarray, verts2: None|int|np.ndarray=None) -> np.ndarray:
            return grapy.get_egdes(self, verts, verts2)

        def neighb(self, verts: int|np.ndarray) -> np.ndarray:
            return grapy.neighb(self, verts)

        def __contains__(self, other: int|np.ndarray|grapy.graph) -> bool:
            return grapy.contains(self, other)
        def contains_verts(self, verts: int|np.ndarray) -> bool:
            return grapy.contains_verts(self, verts)
        def contains_edges(self, edges: int|np.ndarray) -> bool:
            return grapy.contains_edges(self, edges)

        def __getitem__(self, verts) -> grapy.graph:
            if not np.all(numpy_extensions.contains_vec(self.verts, verts)):
                raise ValueError(f'verts {verts} not in {self.verts}')
            
            return grapy.graph(verts, self.get_edges(verts, verts))
        
        def __sub__(self, other) -> grapy.graph:
            return grapy.sub(self, other)
        def sub_verts(self, verts) -> grapy.graph:
            return grapy.sub_verts(self, verts)
        def sub_edges(self, edges) -> grapy.graph:
            return grapy.sub_edges(self, edges)

        def __add__(self, other: int|np.ndarray|grapy.graph) -> grapy.graph:
            return grapy.add(self, other)
        def add_verts(self, verts: int|np.ndarray) -> grapy.graph:
            return grapy.add_verts(self, verts)
        def add_edges(self, edges: np.ndarray) -> grapy.graph:
            return grapy.add_edges(self, edges)

        def deg(self, verts: int|np.ndarray=None) -> grapy.graph:
            return grapy.deg(self, verts)
        
        def mindeg(self) -> grapy.graph:
            return grapy.mindeg(self)
        def maxdeg(self) -> grapy.graph:
            return grapy.maxdeg(self)
        def argmindeg(self) -> grapy.graph:
            return grapy.argmindeg(self)
        def argmaxdeg(self) -> grapy.graph:
            return grapy.argmaxdeg(self)

        def adjacency_mtx(self) -> np.ndarray:
            return grapy.adjacency_mtx(self)

    class edges():
        @classmethod
        def contains_edges(cls, edges1, edges2) -> np.ndarray:
            if edges2.shape == (2,):
                edges2 = edges2[np.newaxis]

            edges2_fliped = np.flip(edges2, axis=1)
            edges1_expanded = np.expand_dims(edges1, 1)

            comparsion1 = edges1_expanded == edges2
            comparsion1 = comparsion1.all(axis=2)
            comparsion2 = edges1_expanded == edges2_fliped
            comparsion2 = comparsion2.all(axis=2)
            comparsion = comparsion1 | comparsion2
            return comparsion.any(axis=0)

        @classmethod
        def setdiff(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dims = np.maximum(a.max(0), b.max(0)) + 1
            return a[~np.in1d(np.ravel_multi_index(a.T,dims), np.ravel_multi_index(b.T,dims))]

        @classmethod
        def rm_dupes(cls, edges):
            return np.unique(np.sort(edges, axis=1), axis=0)

    @classmethod
    def from_edges(cls, edges: np.ndarray) -> graph:
        verts = np.unique(edges)
        return cls.graph(verts, edges)
    @classmethod
    def from_adjacency_mtx(cls, adjacency_mtx: np.ndarray) -> graph:
        if type(adjacency_mtx) is not np.ndarray:
            adjacency_mtx = np.array(adjacency_mtx)
        if len(adjacency_mtx.shape) == 2 and adjacency_mtx.shape[0] == adjacency_mtx.shape[0]:
            return cls.from_edges(np.argwhere(adjacency_mtx))
        if adjacency_mtx.shape == (0,):
            return cls.graph([], [])
        raise ValueError(f'adjacency mtx must be of shape (n,n) or (0,) not {adjacency_mtx.shape}')

    @classmethod
    def get_egdes(cls, g: graph, verts: int|np.ndarray, verts2: None|int|np.ndarray=None) -> np.ndarray:
        if type(verts) is int:
            verts = np.array([verts])
        if type(verts2) is int:
            verts = np.array([verts2])

        mask = (numpy_extensions.contains_vec(verts, g.edges[:,0])
            | numpy_extensions.contains_vec(verts, g.edges[:, 1])
            if verts2 is None else
            (numpy_extensions.contains_vec(verts, g.edges[:,0])
            & numpy_extensions.contains_vec(verts2, g.edges[:,1]))
            | (numpy_extensions.contains_vec(verts, g.edges[:, 1]))
            & numpy_extensions.contains_vec(verts2, g.edges[:,0]))
        
        return g.edges[mask]
        
    @classmethod
    def neighb(cls, g: graph, verts: int|np.ndarray) -> np.ndarray:
        if type(verts) is int:
            verts = np.array([verts])

        mask1 = numpy_extensions.contains_vec(verts, g.edges[:,0])
        mask2 = numpy_extensions.contains_vec(verts, g.edges[:,1])
        neighb1 = g.edges[mask1][:,1]
        neighb2 = g.edges[mask2][:,0]

        return np.setdiff1d(np.union1d(neighb1, neighb2), verts)

    @classmethod
    def contains(cls, g: graph, other: int|np.ndarray|graph) -> bool|np.ndarray:
        # one vert
        if type(other) is int:
            return other in g.verts
        # ggraph
        if type(other) is grapy.graph:
            return (numpy_extensions.contains_vec(g.verts, other.verts).all()
                and numpy_extensions.contains_vec(g.edges, other.edges).all())

        if type(other) is not np.ndarray:
            other = np.array(other)
        # one edge
        #if type(other) is np.ndarray and other.shape == (2,):
        #    return cls.contains_edges(g, other)
        
        # multiple verts
        if len(other.shape) == 1:
            return numpy_extensions.contains_vec(g.verts, other).all()
        # multiple edges
        if len(other.shape) == 2:
            return cls.contains_edges(g, other)
    @classmethod
    def contains_verts(cls, g: graph, verts: int|np.ndarray) -> bool|np.ndarray:
        if type(verts) is int:
            return verts in g.verts
        return numpy_extensions.contains_vec(g.verts, verts).all()
    @classmethod
    def contains_edges(cls, g: graph, edges: np.ndarray) -> bool|np.ndarray:
        return cls.edges.contains_edges(g.edges, edges).all()

    @classmethod
    def sub(cls, g: graph, other: int|np.ndarray|graph) -> graph:
        if type(other) is int:
            return cls.sub_verts(g, other)

        if type(other) is grapy.graph:
            return cls.sub_verts(g, other.verts)

        if type(other) is not np.ndarray:
            other = np.array(other)

        if len(other.shape) == 1:
            return cls.sub_verts(g, other)
        if len(other.shape) == 2:
            return cls.sub_edges(g, other)
    @classmethod
    def sub_verts(cls, g: graph, verts: int|np.ndarray) -> graph:
        if type(verts) is int:
            verts = np.array([verts])
        
        remaining_verts = np.setdiff1d(g.verts, verts)
        return g[remaining_verts]
    @classmethod
    def sub_edges(cls, g: graph, edges: np.ndarray) -> graph:
        if edges.shape == (2,):
            edges = edges[np.newaxis]
        
        remaining_edges = cls.edges.setdiff(g.edges, edges)
        remaining_edges = cls.edges.setdiff(remaining_edges, np.flip(edges, axis=1))
        return cls.graph(np.copy(g.verts), remaining_edges)

    @classmethod
    def add(cls, g: graph, other: int|np.ndarray|graph) -> graph:
        if type(other) is int:
            return cls.add_verts(g, other)
        
        if type(other) is grapy.graph:
            new_g = cls.add_verts(g, other.verts)
            return cls.add_edges(new_g, other.edges)

        if type(other) is not np.ndarray:
            other = np.array(other)

        if len(other.shape) == 1:
            return cls.add_verts(g, other)
        if len(other.shape) == 2:
            return cls.add_edges(g, other)
    @classmethod
    def add_verts(cls, g: graph, verts: int|np.ndarray) -> graph:
        if type(verts) is int:
            verts = np.array([verts])

        result_verts = np.union1d(g.verts, verts)
        return cls.graph(result_verts, np.copy(g.edges))
    @classmethod
    def add_edges(cls, g: graph, edges: np.ndarray) -> graph:
        if edges.shape == (2,):
            edges = edges[np.newaxis]
        if edges.flatten() not in g:
            raise ValueError('all starts and ends of edges must be in g.verts')
        edges = grapy.edges.rm_dupes(edges)

        united_edges = np.unique(np.append(g.edges, edges, axis=0), axis=0)
        return cls.graph(g.verts.copy(), united_edges)

    @classmethod
    def deg(cls, g: graph, verts: int|np.ndarray=None) -> graph:
        if type(verts) is int:
            return np.sum(g.edges == verts)
        if verts is None:
            verts = g.verts

        edges = g.edges[np.newaxis] if len(g.edges.shape) == 1 else g.edges
        comparsion_mtx = np.repeat(edges.flatten()[:,np.newaxis], len(verts), axis=1)
        comparsion_mtx = comparsion_mtx == verts
        return np.sum(comparsion_mtx, axis=0)

    @classmethod
    def mindeg(cls, g: graph) -> graph:
        return np.min(cls.deg(g))
    @classmethod
    def maxdeg(cls, g: graph) -> graph:
        return np.max(cls.deg(g))
    @classmethod
    def argmindeg(cls, g: graph) -> graph:
        degs = g.deg()
        verts = g.verts[degs == np.min(degs)]
        return verts[0] if len(verts) == 1 else verts
    @classmethod
    def argmaxdeg(cls, g: graph) -> graph:
        degs = g.deg()
        verts = g.verts[degs == np.max(degs)]
        return verts[0] if len(verts) == 1 else verts

    @classmethod
    def adjacency_mtx(cls, g: graph) -> np.ndarray:
        if len(g.verts) == 0:
            return np.array([])

        shape = (len(g.verts), len(g.verts))
        if not np.any(g.edges):
            return np.full(shape, False)

        ids = numpy_extensions.replace(g.edges, g.verts, np.arange(len(g.verts)))
        ids = np.append(ids, np.flip(ids, axis=1), axis=0)
        values = (np.full(ids.shape[0], True), (ids[:,0], ids[:,1]))
        return sp.sparse.coo_matrix(values, shape=shape, dtype=bool).toarray()
