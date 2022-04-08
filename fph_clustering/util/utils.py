import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp

def networkx_from_torch_sparse(adjacency: torch.sparse_coo_tensor):
    values = adjacency.values()
    rows = adjacency.indices()[0]
    cols = adjacency.indices()[1]
    A = sp.coo_matrix((values, (rows, cols)), shape=adjacency.shape)
    return nx.from_scipy_sparse_matrix(A)

# Converts a linkage matrix (i.e. dendrogram) in a tree structure.
def dendrogram_to_tree(dendrogram, n_leaves):
    tree, d = {}, {u: float('inf') for u in range(n_leaves)}
    for t in range(n_leaves - 1):
        u = dendrogram[t, 0]
        v = dendrogram[t, 1]
        dist = dendrogram[t, 2]
        if dist == d[u] and dist == d[v]:
            tree[n_leaves + t] = tree.pop(u)
            tree[n_leaves + t] += tree.pop(v)
        elif dist == d[u]:
            tree[n_leaves + t] = [v]
            tree[n_leaves + t] += tree.pop(u)
        elif dist == d[v]:
            tree[n_leaves + t] = [u]
            tree[n_leaves + t] += tree.pop(v)
        else:
            tree[n_leaves + t] = [u, v]
        d[n_leaves + t] = dist
    return tree


# Select a tree from the learned parent probabilities by choosing for each node its most likely parent.
def best_tree(A, B):
    n_leaves, n_internal = A.shape
    tree = {}
    parents = {}

    tree = {k: [] for k in range(n_leaves, n_leaves + n_internal)}

    # Sample tree.
    for leaf in range(A.shape[0]):
        parent = np.argmax(A[leaf, :]) + A.shape[0]
        tree[parent].append(leaf)
        parents[leaf] = parent

    for internal in range(B.shape[0] - 1):
        parent = np.argmax(B[internal, :]) + A.shape[0]
        tree[parent].append(internal + A.shape[0])
        parents[internal + A.shape[0]] = parent

    # Prune tree.
    updated = True
    while updated:
        updated = False
        for k, v in tree.items():
            # Do not prune out the root node.
            if k == n_leaves + n_internal - 1:
                continue

            # Prune out node if it has none or only one child.
            if len(v) <= 1:
                assert (k in parents)
                parent = parents[k]
                if len(v) == 1:
                    # Add child to parent.
                    tree[parent].append(v[0])
                    parents[v[0]] = parent

                # Remove node from children of parent.
                tree[parent] = [n for n in tree[parent] if n != k]
                # Remove node from tree.
                del parents[k]
                del tree[k]
                updated = True
                break

    # Special handling of the root node.
    if len(tree[n_leaves + n_internal - 1]) == 1:
        child = tree[n_leaves + n_internal - 1][0]
        tree[n_leaves + n_internal - 1] = tree[child]
        del tree[child]
        del parents[child]
        for child in tree[n_leaves + n_internal - 1]:
            parents[child] = n_leaves + n_internal - 1
    return tree

def tree_to_A_B(T, num_nodes, num_internal_nodes, dtype=torch.float32):
    T_tmp = {}
    label_mapping = {internal_label: num_nodes + new_internal_label for new_internal_label, internal_label in
                     enumerate(sorted(list(T.keys())))}
    for internal_label, new_internal_label in label_mapping.items():
        T_tmp[new_internal_label] = []
        for child in T[internal_label]:
            if child < num_nodes:
                T_tmp[new_internal_label].append(child)
            else:
                T_tmp[new_internal_label].append(label_mapping[child])

    if isinstance(dtype, np.dtype):
        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype == np.int32:
            dtype = torch.int32
        elif dtype == np.int64:
            dtype = torch.int64
        else:
            raise NotImplementedError(f'unknown dtype {dtype}')
    # A and B
    A_tree = torch.zeros((num_nodes, num_internal_nodes), dtype=dtype)
    B_tree = torch.zeros((num_internal_nodes, num_internal_nodes), dtype=dtype)
    for internal_label, children in T_tmp.items():
        j = int(internal_label - num_nodes)
        for child in children:
            if child < num_nodes:
                i = int(child)
                A_tree[i, j] = 1
            else:
                i = int(child - num_nodes)
                B_tree[i, j] = 1
    return A_tree, B_tree

def tree_to_dendrogram(tree, n_leaves, height_map = None):
    tree = [(k, list(v)) for k, v in tree.items()]

    dendrogram = []    
    dendro_map = {i: i for i in range(n_leaves)}
    link_map = {} # Needed for plotting.
    
    mod = True
    
    while len(tree) > 0 and mod:
        mod = False
        for idx, (k, v) in enumerate(tree):
            if height_map != None:
                height = height_map[k - n_leaves] + 1
            else:
                height = k - n_leaves + 1

            # Check if we can add node to dendrogram.
            if not all (n in dendro_map for n in v):
                continue
 
            mod = True
            # This creates the parent node.
            link_map[len(dendrogram)] = k - n_leaves

            dendrogram.append((dendro_map[v[0]], dendro_map[v[1]], height, height))
            
            for n in v[2::]:
                n1 = n_leaves - 1 + len(dendrogram)
                n2 = dendro_map[n]
                link_map[len(dendrogram)] = k - n_leaves
                dendrogram.append((n1, n2, height, height))
            
            dendro_map[k] = n_leaves - 1 + len(dendrogram)
            tree.pop(idx)
            break
            
    if len(tree) > 0:
        raise Exception('Invalid tree!')

    return np.array(dendrogram, float), link_map
