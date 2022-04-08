import torch
from fph_clustering.models.utils import chunker
from fph_clustering.models.utils import preprocess_graph
from torch.utils.checkpoint import checkpoint

def compute_dasgupta(A, B, adj=None, preprocessed_adj=None):
    """
    Compute the soft Dasgupta cost.
    Parameters
    ----------
    A: row-stochastic torch.Tensor, shape [n_nodes, n_internal]
        The parent probabilities of the internal nodes to the leaves (graph nodes).
    B: row-stochastic, nil-potent, upper-triangular torch.Tensor, shape [n_internal, n_internal]
        The parent probabilities of the internal nodes to each other.
    adj: torch.sparse_coo_tensor, shape [n_nodes, n_nodes], optioonal
        If preprocessed_adj is not provided, we perform the preprocessing on the fly here using adj. Will not be used when preprocessed_adj is not None.
    preprocessed_adj: fph_clustering.models.utils.PreprocessedGraph, Optional
        The graph on which to compute the Soft Dasgupta cost.
    Returns
    -------
    A scalar, the soft Dasgupta cost.
    """
    device = B.device
    dtype = B.dtype
    if preprocessed_adj is None:
        assert adj is not None
        preprocessed_adj = preprocess_graph(adj)
    
    edge_weights = preprocessed_adj.p_uv.to(device).to(dtype)
    edge_ixs = preprocessed_adj.edge_ixs.to(device)

    B_anc = compute_B_anc(B)
    B_square_inv = compute_B_square_inv(B_anc)

    m_sum = A.to(device) @ (B_anc + torch.eye(B_anc.shape[0], device=device, dtype=dtype))
    soft_cardinalities = m_sum.sum(0)

    if len(edge_ixs) > 1e5:
        size = 64000
        if device == 'cpu':
            size = int(1e6)
        edge_chunks = chunker(torch.arange(len(edge_ixs)), size=size)
        fun = lambda m_sum, edge_ixs, B_square_inv, edge_weights: (m_sum[edge_ixs[:,0]].mul(m_sum[edge_ixs[:,1]]) @ B_square_inv).mul(m_sum.sum(0)[None, :]).sum(-1).mul(edge_weights.to(device)).sum(0, keepdims=True)
        dasgupta = torch.cat([checkpoint(fun, m_sum, edge_ixs[chunk], B_square_inv, edge_weights[chunk]) for chunk in edge_chunks]).sum()
    else:
        p_anc_x_uv = m_sum[edge_ixs[:, 0]].mul(m_sum[edge_ixs[:, 1]])
        lca_probs = p_anc_x_uv @ B_square_inv
        dasgupta = lca_probs.mul(soft_cardinalities[None, :]).sum(-1).mul(edge_weights.to(device)).sum()

    return dasgupta


def compute_B_anc(B: torch.tensor):
    device = B.device
    dtype = B.dtype
    eye = torch.eye(B.shape[0], device=device, dtype=dtype)
    return torch.inverse(eye - B) - eye


def compute_B_square_inv(B_anc: torch.tensor):
    device = B_anc.device
    dtype = B_anc.dtype
    eye = torch.eye(B_anc.shape[0], device=device, dtype=dtype)

    B_diag_square = torch.triu(B_anc, diagonal=1) ** 2
    return torch.inverse((eye + B_diag_square))


def compute_p(A, B, edge_ixs, p_uv, B_anc=None, B_square_inv=None,):
    """
    Computes p(x), the probability distribution over the internal nodes.

    Parameters
    ----------
    A: row-stochastic torch.Tensor, shape [n_nodes, n_internal]
        The parent probabilities of the internal nodes to the leaves (graph nodes).
    B: row-stochastic, nil-potent, upper-triangular torch.Tensor, shape [n_internal, n_internal]
        The parent probabilities of the internal nodes to each other.
    edge_ixs: torch.Tensor, shape [n_edges, 2]
        The edge indices.
    p_uv: torch.Tensor, shape [n_edges, ]
        The edge probabilities p(u,v).
    B_anc: (optional) torch.Tensor, shape [n_internal, n_internal]
        In order to prevent redundant computation with q(x), we can pass B_anc directly here.
    B_square_inv: (optional) torch.Tensor, shape [n_internal, n_internal]
        In order to prevent redundant computation with q(x), we can pass B_square_inv directly here.

    Returns
    -------
    p_x: stochastic torch.Tensor, shape [n_internal,]
        The probability distribution p(x) over the internal nodes.
    """

    device = B.device
    dtype = B.dtype
    eye = torch.eye(B.shape[0], device=device, dtype=dtype)

    # eq. 8 in the paper
    if B_anc is None:
        B_anc = compute_B_anc(B)
    m_sum = A.to(device) @ (B_anc + eye) 

    if B_square_inv is None:
        B_square_inv = compute_B_square_inv(B_anc)

    if len(edge_ixs) > 1e5:
        edge_chunks = chunker(torch.arange(len(edge_ixs)), size=64000)
        fun = lambda m_sum, edge_ixs, B_square_inv, p_uv: (m_sum[edge_ixs[:,0]].mul(m_sum[edge_ixs[:,1]]) @ B_square_inv).mul(p_uv[:,None]).sum(0, keepdims=True)
        p_x = torch.cat([checkpoint(fun, m_sum, edge_ixs[chunk], B_square_inv, p_uv[chunk])
                         for chunk in edge_chunks]).sum(0)
    else:
        # p(x_i | u) * p(x_i | v)
        # [n_edges, n_internal]
        p_anc_x_uv = m_sum[edge_ixs[:,0 ]].mul(m_sum[edge_ixs[:, 1]])
        lca_probs = p_anc_x_uv @ B_square_inv
        # multiply by p(u,v) to get the expectation
        p_x = lca_probs.mul(p_uv[:, None]).sum(0)

    if p_x.min() < 0:
        p_x = p_x - p_x.min()
        p_x = p_x / p_x.sum()
        assert p_x.min() >= 0
    return p_x


def compute_q(A, B, P_u, B_anc=None, B_square_inv=None,
              same_leaf_correction=False):
    """
    Computes the probability distribution over the internal nodes under the null model.

    Parameters
    ----------
    A: row-stochastic torch.Tensor, shape [n_nodes, n_internal]
        The parent probabilities of the internal nodes to the leaves (graph nodes).
    B: row-stochastic, nil-potent, upper-triangular torch.Tensor, shape [n_internal, n_internal]
        The parent probabilities of the internal nodes to each other.
    P_u: stochastic torch.Tensor, shape [n_nodes,]
        The probability distribution over the nodes in the graph.
    B_anc: (optional) torch.Tensor, shape [n_internal, n_internal]
        In order to prevent redundant computation with q(x), we can pass B_anc directly here.
    B_square_inv: (optional) torch.Tensor, shape [n_internal, n_internal]
        In order to prevent redundant computation with q(x), we can pass B_square_inv directly here.
    same_leaf_correction: bool, default: False
        Whether to correct q_x for the random walks starting at the identical leaf. 
    Returns
    -------
    q_x: stochastic torch.Tensor, shape [n_internal,]
        The probability distribution q(x) over the internal nodes under the null model.
    """
    device = B.device
    dtype = B.dtype
    assert len(P_u) == 2
    p_u_out, p_u_in = P_u
    del P_u

    eye = torch.eye(B.shape[0], device=device, dtype=dtype)

    # eq. 8 in the paper
    if B_anc is None:
        B_anc = compute_B_anc(B)

    P_anc = A.to(device) @ (B_anc + eye)
    m_sum_out = p_u_out @ P_anc   # eq. 19 in the paper
    m_sum_in = p_u_in @ P_anc   # eq. 19 in the paper

    # p(x_i | u) * p(x_i | v)
    # [n_edges, n_internal]
    p_anc_x_uv = m_sum_out.mul(m_sum_in)
    if B_square_inv is None:
        B_square_inv = compute_B_square_inv(B_anc)

    lca_probs = p_anc_x_uv @ B_square_inv
    
    q_x = lca_probs
    if same_leaf_correction:
        # remove influence of same-leaf random walks
        corr = -(P_anc * P_anc) @ (B_square_inv)
        q_x_correction = (p_u_out * p_u_in) @ corr
        q_x = q_x + q_x_correction
        # add influence of modified same-leaf random walks where the LCA probabilities
        # are equal to M_1[u], i.e. the two random walks meet after the first transition.
        add_corr = (p_u_out * p_u_in) @ A
        q_x = q_x + add_corr

    if q_x.min() < 0:
        q_x = q_x - q_x.min()
        q_x = q_x / q_x.sum()
    return q_x

def compute_TSD(A: torch.tensor, B: torch.tensor, adj=None, preprocessed_adj=None, same_leaf_correction=False):
    device = B.device
    dtype = A.dtype
    
    if preprocessed_adj is None:
        assert adj is not None
        preprocessed_adj = preprocess_graph(adj)
    
    p_uv_drop = preprocessed_adj.p_uv.to(device).to(dtype)
    p_u_out_drop = preprocessed_adj.p_u_out.to(device).to(dtype)
    p_u_in_drop = preprocessed_adj.p_u_in.to(device).to(dtype)
    edges_sampled = preprocessed_adj.edge_ixs.to(device)

    B_anc = compute_B_anc(B)
    B_square_inv = compute_B_square_inv(B_anc)

    p_x = compute_p(A, B, edges_sampled, p_uv_drop,
                    B_anc=B_anc, B_square_inv=B_square_inv)
    q_x = compute_q(A, B, (p_u_out_drop, p_u_in_drop),
                    B_anc=B_anc, B_square_inv=B_square_inv,
                    same_leaf_correction=same_leaf_correction) + 1e-7
    if q_x.min() < 0:
        q_x = q_x - q_x.min() + 1e-7
    q_x = q_x / q_x.sum()

    if (p_x <= 0).any():
        p_x = p_x - p_x.min() + 1e-7
        p_x = p_x/p_x.sum()

    TSD = p_x.mul((p_x / q_x.clamp_min(1e-7)).log()).sum()
    return TSD