# grapy: A Graph Processing Library

`grapy` is a Python library designed for manipulating and analyzing graphs using NumPy. It provides a collection of tools to work with graphs represented as sets of vertices and edges, offering functionalities to perform operations such as addition and subtraction of vertices or edges, adjacency matrix generation, and more.

This project was created as a little test- and playgorund for my studie in the univerity module 'Graph Theorie'.

## Features

- **Graph Operations:** Add and remove vertices and edges to/from the graph.
- **Querying:** Check for the presence of vertices or edges, get neighbors of a vertex, and more.
- **Graph Metrics:** Calculate degree, minimum and maximum degree, and their corresponding vertices.
- **Adjacency Matrix:** Generate the adjacency matrix for a graph, with support for directed graphs.

## Classes and Methods

### `numpy_extensions`

A collection of utility functions extending NumPy's functionality to support graph operations.

- **`contains`**: Check if elements of one array are contained in another array.
- **`replace`**: Replace elements in an array with specified values.
- **`setitem_multi_index` & `getitem_multi_index`**: Set or get items from a multi-dimensional array based on multi-dimensional indices.
- **`flatten_multi_index`**: Convert multi-dimensional indices to a flat index.
- **`one_hot`**: Generate a one-hot encoding of integers.
- **`flatten`**: Flatten an array from a specified axis.
- **`getitem_nd` & `setitem_nd`**: Advanced getting and setting of items for multi-dimensional arrays.

### `grapy.graph`

The primary class representing a graph.

- **Initialization**: Create a graph with vertices and edges.
- **`add_verts`, `add_edges`**: Add vertices or edges to the graph.
- **`sub_verts`, `sub_edges`**: Remove vertices or edges from the graph.
- **`deg`, `mindeg`, `maxdeg`, `argmindeg`, `argmaxdeg`**: Compute degrees and related metrics.
- **`adjacency_mtx`**: Generate the adjacency matrix for the graph.

### `grapy.edges`

Utility class for edge manipulation.

- **`contains_edges`**: Check if edges are contained within a set of edges.
- **`setdiff`**: Compute the set difference between two sets of edges.
- **`rm_dupes`**: Remove duplicate edges.

## Usage Example

```python
from grapy import grapy as gp
import numpy as np

# Create a graph from edges
edges = np.array([[0, 1], [1, 2], [2, 0]])
graph = gp.grapy.from_edges(edges)

# Add a vertex and an edge
graph = graph.add_verts(np.array([3]))
graph = graph.add_edges(np.array([[2, 3]]))

# Print the adjacency matrix
print(graph.adjacency_mtx())
```

## Testing

`grapy` comes with a suite of unit tests to ensure functionality. To run the tests, execute the test scripts using a Python test runner such as `unittest`.
