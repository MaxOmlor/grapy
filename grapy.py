from __future__ import annotations
import numpy as np
#import scipy as sp
from collections.abc import Iterable
from sklearn.cluster import KMeans

class numpy_extensions():
    @classmethod
    def extend_nd(cls, a, dim):
        if len(a) >= dim:
            return a

        a_extended = np.zeros(dim)
        a_extended[:len(a)] = a

        return a_extended
    @classmethod
    def brodcast_extend_nd(cls, a, dim):
        return np.apply_along_axis(numpy_extensions.extend_nd, axis=1, arr=a, dim=dim)

    @classmethod
    def contains(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(np.shape(a)) == 1 and len(np.shape(a)) == 1:
            return np.in1d(b, a)

        dims = (np.maximum(np.max(a),np.max(b))+1,)*(np.sum(np.shape(a)[1:]))
        hash_a = cls.flatten_multi_index(a, dims)
        hash_b = cls.flatten_multi_index(b, dims)
        return np.in1d(hash_b, hash_a)


    @classmethod
    def replace(cls, a: np.ndarray, values: any, replacements: any) -> np.ndarray:
        '''
        # Notes

        Replaces given values in a by given replacements.

        # Examples

        >>> ne.replace([1,2,3,2,1], 1, 0)
        array([0, 2, 3, 2, 0])

        multiple values
        >>> ne.replace([1,2,3,2,1], [1,2], [-1,-2])
        array([-1, -2,  3, -2, -1])

        multiple dimensions
        >>> a = np.array([[1,2], [1,3], [2,4]])
        >>> a
        array([[1, 2],
                [1, 3],
                [2, 4]])
        >>> ne.replace(a, [1,2,3,4], [0,1,2,3])
        array([[0, 1],
                [0, 2],
                [1, 3]])
        '''
        if len(a) == 0:
            return np.copy(a)
        if not isinstance(values, Iterable) and not isinstance(replacements, Iterable):
            result = np.copy(a)
            result[np.equal(a, values)] = replacements
            return result

        if isinstance(values, Iterable) and not isinstance(replacements, Iterable):
            replacements = np.full(values.shape, replacements)
        if type(replacements) is not np.ndarray:
            replacements = np.array(replacements)

        result = np.copy(a)
        shape = result.shape
        result = result.flatten()

        d = dict(zip(values, replacements))
        def r(x): return d[x]

        mask = np.in1d(result, values)
        result[mask] = np.frompyfunc(r, 1, 1)(result[mask])
        result = result.reshape(shape)
        return result.astype(replacements.dtype)

    @classmethod
    def setitem_multi_index(cls, a: np.ndarray, ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        flat_a = a.flatten()
        flat_ids = np.ravel_multi_index(np.transpose(ids), a.shape)

        flat_a[flat_ids] = values

        return flat_a.reshape(a.shape)
    @classmethod
    def getitem_multi_index(cls, a: np.ndarray, ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        flat_a = a.flatten()
        flat_ids = np.ravel_multi_index(np.transpose(ids), a.shape)

        return flat_a[flat_ids]


    @classmethod
    def get_basis(cls, dims: tuple[int,int,int]) -> np.ndarray:
        return np.array([np.prod(dims[i+1:]) for i in np.arange(len(dims))])
    @classmethod
    def flatten_multi_index(cls, ids: np.ndarray, dims, dtype: np.dtype=None, axis: int=1) -> np.ndarray:
        '''
        # Notes

        Encodes a vector of n dimensions to scalar.
        There for the vector must be inside the hypercube given by dims parameter.
        An unique id is assigned to every position in this hypercube.
        The resulting scalar of a given vector is determined by the id of the cell of the hypercube the vector is pointing to.

        this concept is extended for tensors insted of vectors,
        by reshapeing the tensor and the dims to a 1d tensor.

        This implementation of the method only workes for vectors of integer values.

        # Examples

        >>> dims = (2,2)
        >>> vec_for_every_pos = [[0,0],[0,1],[1,0],[1,1]]
        >>> flatten_multi_index(vec_for_every_pos, dims)
        array([0., 1., 2., 3.])

        possible for n dims
        >>> ne.flatten_multi_index([[0,1,2],[1,2,3]], (4,4,4))
        array([ 6., 27.])

        encode a tensor
        >>> a = np.arange(8).reshape((2,2,2))
        >>> dims = (np.max(a),) *np.shape(a)[1] *np.shape(a)[2]
        >>> ne.flatten_multi_index(a, dims)
        array([  66, 1666])
        '''
        if dtype and type(ids) is np.ndarray:
            ids = ids.astype(dtype)
        elif dtype:
            ids = np.array(ids).astype(dtype)

        flatten_ids = cls.flatten(ids, axis=axis)
        flatten_dims = cls.flatten(dims)

        basis = cls.get_basis(flatten_dims)
        ids_transformed = np.multiply(flatten_ids, basis)
        return np.sum(ids_transformed, axis=1)


    @classmethod
    def one_hot(cls, values: np.ndarray, class_count: int) -> np.ndarray:
        return np.eye(class_count)[np.reshape(values,-1)]

    @classmethod
    def flatten(cls, a: np.ndarray, axis: int=0) -> np.ndarray:
        shape = np.shape(a)
        new_shape = shape[:axis] + (np.prod(shape[axis:]),)
        return np.reshape(a, new_shape)

    @classmethod
    def getitem_nd(cls, a: np.ndarray, id_mtx: np.ndarray) -> np.ndarray:
        '''
        # Notes

        Makes it possible to generate view from multidimensional id-array.
        Therefor every id in id_mtx gets substituted by its related item in a.
        In the simplest case view_nd is equevalent to a[id].

        # Examples

        equevalent to a[id]
        >>> ne.view_nd([1,2,3], [2,0,1])
        array([3, 1, 2])
        
        multi dim id_mtx
        >>> id_mtx = np.array([[0,1,2], [1,2,0], [2,0,1]])
        >>> id_mtx
        array([[0, 1, 2],
                [1, 2, 0],
                [2, 0, 1]])
        >>> ne.view_nd([1,2,3], id_mtx)
        array([[1, 2, 3],
                [2, 3, 1],
                [3, 1, 2]])

        multi dim a and id_mtx
        >>> a = np.array([[1,2],[3,4],[5,6]])
        >>> a
        array([[1, 2],
                [3, 4],
                [5, 6]])
        >>> ne.view_nd(a, id_mtx)
        array([[[1, 2],
                [3, 4],
                [5, 6]],
               [[3, 4],
                [5, 6],
                [1, 2]],
               [[5, 6],
                [1, 2],
                [3, 4]]])
        '''
        if type(a) is not np.ndarray:
            a = np.array(a)
        if type(id_mtx) is not np.ndarray:
            id_mtx = np.array(id_mtx)

        ids_flatten = np.ravel(id_mtx)
        shape = (*id_mtx.shape, *(a.shape[1:])) if len(a.shape) > 1 else id_mtx.shape
        return a[ids_flatten].reshape(shape)

    @classmethod
    def setitem_nd(cls, a: np.ndarray, id_mtx: np.ndarray, values: np.ndarray) -> np.ndarray:
        if type(id_mtx) is not np.ndarray:
            id_mtx = np.array(id_mtx)
        if type(values) is not np.ndarray:
            values = np.array(values)

        ids_flatten = np.ravel(id_mtx)
        result = np.copy(a)
        shape = (np.prod(values.shape[:len(id_mtx.shape)]), *values.shape[len(id_mtx.shape):])
        result[ids_flatten] = np.reshape(values, shape)
        return result
    
    @classmethod
    def spectral_clustering(cls, laplacian_matrix: np.ndarray, k: int) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

        sorted_ids = np.argsort(eigenvalues)
        embeddings = eigenvectors[sorted_ids][1:].T

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
        return cluster_labels
    
    @classmethod
    def relabel(cls, a: np.ndarray) -> np.ndarray:
        """
        assigns ints to every value in a, such that same values in a get the same int starting by 0 for the first value in values.
        """
        _, unique_ids = np.unique(a, return_index=True)
        unique_ids_sorted = np.sort(unique_ids)

        values = a[unique_ids_sorted]
        replacements = np.arange(len(unique_ids_sorted))

        result = np.copy(a)
        shape = result.shape
        result = result.flatten()

        d = dict(zip(values, replacements))
        def r(x): return d[x]

        mask = np.in1d(result, values)
        result[mask] = np.frompyfunc(r, 1, 1)(result[mask])
        result = result.reshape(shape)
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
            if not np.all(numpy_extensions.contains(self.verts, verts)):
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

        mask = (numpy_extensions.contains(verts, g.edges[:,0])
            | numpy_extensions.contains(verts, g.edges[:, 1])
            if verts2 is None else
            (numpy_extensions.contains(verts, g.edges[:,0])
            & numpy_extensions.contains(verts2, g.edges[:,1]))
            | (numpy_extensions.contains(verts, g.edges[:, 1]))
            & numpy_extensions.contains(verts2, g.edges[:,0]))
        
        return g.edges[mask]
        
    @classmethod
    def neighb(cls, g: graph, verts: int|np.ndarray) -> np.ndarray:
        if type(verts) is int:
            verts = np.array([verts])

        mask1 = numpy_extensions.contains(verts, g.edges[:,0])
        mask2 = numpy_extensions.contains(verts, g.edges[:,1])
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
            return (numpy_extensions.contains(g.verts, other.verts).all()
                and numpy_extensions.contains(g.edges, other.edges).all())

        if type(other) is not np.ndarray:
            other = np.array(other)
        # one edge
        #if type(other) is np.ndarray and other.shape == (2,):
        #    return cls.contains_edges(g, other)
        
        # multiple verts
        if len(other.shape) == 1:
            return numpy_extensions.contains(g.verts, other).all()
        # multiple edges
        if len(other.shape) == 2:
            return cls.contains_edges(g, other)
    @classmethod
    def contains_verts(cls, g: graph, verts: int|np.ndarray) -> bool|np.ndarray:
        if type(verts) is int:
            return verts in g.verts
        return numpy_extensions.contains(g.verts, verts).all()
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
    def adjacency_mtx(cls, g: graph, directed: bool=False) -> np.ndarray:
        n = len(g.verts)
        if n == 0:
            return np.array([])

        shape = (n, n)
        if not np.any(g.edges):
            return np.full(shape, False)

        ids = numpy_extensions.replace(g.edges, g.verts, np.arange(n))
        
        ids = np.transpose(ids)
        if not directed:
            ids = np.append(ids, np.flip(ids, axis=0), axis=1)
        
        flat_ids = np.ravel_multi_index(ids, shape)
        flat_mtx = np.full(shape[0]*shape[1], False)

        flat_mtx[flat_ids] = True

        return flat_mtx.reshape(shape)