import numpy as np
from __future__ import annotations

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

        def __len__(self):
            return len(self.verts)

        def get_edges(self, verts: int|np.ndarray, verts2: None|int|np.ndarray=None) -> np.ndarray:
            return grapy.get_egdes(self, verts, verts2)

        def neighb(self, verts: int|np.ndarray) -> np.ndarray:
            return grapy.neighb(self, verts)

        def __contains__(self, other: int|np.ndarray|grapy.graph):
            return grapy.contains(self, other)

        def contains_verts(self, verts: int|np.ndarray):
            return grapy.contains_verts(self, verts)

        def contains_edges(self, edges: int|np.ndarray):
            return grapy.contains_edges(self, edges)

        def __getitem__(self, verts):
            if not np.all(np_extensions.contains_vec(self.verts, verts)):
                raise ValueError(f'verts {verts} not in {self.verts}')
            
            return grapy.graph(verts, self.get_edges(verts, verts))

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

        #return np.append(neighb1, neighb2)
        return np.unique([neighb1, neighb2])

    @classmethod
    def contains(cls, g: graph, other: int|np.ndarray|graph) -> bool|np.ndarray:
        # one vert
        if type(other) is int:
            return other in g.verts
        # ggraph
        if type(other) is grapy.graph:
            return (np_extensions.contains_vec(g.verts, other.verts).all()
                and np_extensions.contains_vec(g.edges, other.edges).all())

        if type(other) is list:
            other = np.array(other)
        # one edge
        if type(other) is np.ndarray and other.shape == (2,):
            return other in g.edges
        # multiple verts
        if type(other) is np.ndarray and len(other.shape) == 1:
            return np_extensions.contains_vec(g.verts, other).all()
        # multiple edges
        if type(other) is np.ndarray and len(other.shape) == 2:
            return np_extensions.contains_vec(g.edges, other).all()

    @classmethod
    def contains_verts(cls, g: graph, verts: int|np.ndarray) -> bool|np.ndarray:
        if type(verts) is int:
            return verts in g.verts
        return np_extensions.contains_vec(g.verts, verts).all()

    @classmethod
    def contains_edges(cls, g: graph, edges: np.ndarray) -> bool|np.ndarray:
        if edges.shape == (2,):
            return edges in g.edges
        return np_extensions.contains_vec(g.edges, edges).all()