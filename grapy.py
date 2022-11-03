from __future__ import annotations
import numpy as np

class np_extensions():
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
    def setdiff2d(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dims = np.maximum(a.max(0), b.max(0)) + 1
        return a[~np.in1d(np.ravel_multi_index(a.T,dims), np.ravel_multi_index(b.T,dims))]

class grapy():
    class graph():
        def __init__(self, verts: np.ndarray, edges: np.ndarray) -> None:
            if type(verts) is not np.ndarray:
                verts = np.array(verts)
            if type(edges) is not np.ndarray:
                edges = np.array(edges)
                
            self.verts = verts
            self.edges = edges

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
            if not np.all(np_extensions.contains_vec(self.verts, verts)):
                raise ValueError(f'verts {verts} not in {self.verts}')
            
            return grapy.graph(verts, self.get_edges(verts, verts))
        
        def __sub__(self, other) -> grapy.graph:
            return grapy.sub(self, other)
        def sub_verts(self, verts) -> grapy.graph:
            return grapy.sub_verts(self, verts)
        def sub_edges(self, edges) -> grapy.graph:
            return grapy.sub_edges(self, edges)

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
    def from_edges(cls, edges: np.ndarray) -> graph:
        verts = np.unique(edges)
        return cls.graph(verts, edges)

    @classmethod
    def get_egdes(cls, g: graph, verts: int|np.ndarray, verts2: None|int|np.ndarray=None) -> np.ndarray:
        if type(verts) is int:
            verts = np.array([verts])
        if type(verts2) is int:
            verts = np.array([verts2])

        mask = (np_extensions.contains_vec(verts, g.edges[:,0])
            | np_extensions.contains_vec(verts, g.edges[:, 1])
            if verts2 is None else
            (np_extensions.contains_vec(verts, g.edges[:,0])
            & np_extensions.contains_vec(verts2, g.edges[:,1]))
            | (np_extensions.contains_vec(verts, g.edges[:, 1]))
            & np_extensions.contains_vec(verts2, g.edges[:,0]))
        
        return g.edges[mask]
        
    @classmethod
    def neighb(cls, g: graph, verts: int|np.ndarray) -> np.ndarray:
        if type(verts) is int:
            verts = np.array([verts])

        mask1 = np_extensions.contains_vec(verts, g.edges[:,0])
        mask2 = np_extensions.contains_vec(verts, g.edges[:,1])
        neighb1 = g.edges[mask1][:,1]
        neighb2 = g.edges[mask2][:,0]

        return np.union1d(neighb1, neighb2)

    @classmethod
    def contains(cls, g: graph, other: int|np.ndarray|graph) -> bool|np.ndarray:
        # one vert
        if type(other) is int:
            return other in g.verts
        # ggraph
        if type(other) is grapy.graph:
            return (np_extensions.contains_vec(g.verts, other.verts).all()
                and np_extensions.contains_vec(g.edges, other.edges).all())

        if type(other) is not np.ndarray:
            other = np.array(other)
        # one edge
        #if type(other) is np.ndarray and other.shape == (2,):
        #    return cls.contains_edges(g, other)
        
        # multiple verts
        if len(other.shape) == 1:
            return np_extensions.contains_vec(g.verts, other).all()
        # multiple edges
        if len(other.shape) == 2:
            return cls.contains_edges(g, other)

    @classmethod
    def contains_verts(cls, g: graph, verts: int|np.ndarray) -> bool|np.ndarray:
        if type(verts) is int:
            return verts in g.verts
        return np_extensions.contains_vec(g.verts, verts).all()

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
        
        remaining_edges = np_extensions.setdiff2d(g.edges, edges)
        remaining_edges = np_extensions.setdiff2d(remaining_edges, np.flip(edges, axis=1))
        return cls.graph(np.copy(g.verts), remaining_edges)