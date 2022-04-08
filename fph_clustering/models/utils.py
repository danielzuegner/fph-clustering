from torch import optim, nn
import torch
from torch_geometric.utils import subgraph
from enum import Enum
from torch_sparse import SparseTensor
from torch_sparse import coalesce
import numpy as np
from typing import List, Mapping
from fph_clustering.optimizers.pgd import PGD
from collections import namedtuple

PreprocessedGraph = namedtuple("PreprocessedGraph", field_names=["node_ids", "p_uv", "p_u_out", "p_u_in", "edge_ixs", "adj", "mutual_information"])

def configure_optimizers(model):
    if 'opt_params' in model.optimizer_params:
        if isinstance(model.optimizer_params['opt_params'], List):
            opt_params = []
            named_params = dict(model.named_parameters())
            for l in model.optimizer_params['opt_params']:
                l = {**l}
                params = l.pop('params')
                model_params = []
                for p in params:
                    model_params.append(named_params[p])
                opt_params.append({**l, **{'params': model_params}})
        else:
            opt_params = {**model.optimizer_params['opt_params']}
    else:
        opt_params = {}

    scheduler = model.optimizer_params.get('scheduler', None)
    if 'scheduler_params' in model.optimizer_params:
        scheduler_params = {**model.optimizer_params['scheduler_params']}
    else:
        scheduler_params = {}

    optimizer = 'Adam'
    if 'optimizer_type' in model.optimizer_params:
        optimizer = model.optimizer_params['optimizer_type']

    if optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(),
                               **opt_params) if isinstance(opt_params, Mapping) else optim.Adam(opt_params)
    elif optimizer.lower() == "pgd":
        optimizer = PGD(model.parameters(), **opt_params) if isinstance(opt_params, Mapping) else PGD(opt_params)
    else:
        raise NotImplementedError(f"Unknown optimizer: {optimizer}")

    if scheduler is None or scheduler == "None":
        return [optimizer], []

    if scheduler == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                     **scheduler_params)
    elif scheduler == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                      **scheduler_params)
    elif scheduler == "CyclicLR":
        if 'amplitude_gamma' in scheduler_params:
            scheduler_params = dict(**scheduler_params)
            scale_val = float(scheduler_params['amplitude_gamma'])
            scheduler_params['scale_fn'] = lambda x: (scale_val ** (x - 1))
            del scheduler_params['amplitude_gamma']
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                   **scheduler_params)
    elif scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                        gamma=scheduler_params['gamma'])
        lr_scheduler.min_lr = scheduler_params['min_lr']
    else:
        raise NotImplementedError("unknown scheduler")
    return [optimizer], [lr_scheduler]


def chunker(seq, size):
    """
    Chunk a list into chunks of size `size`.
    From
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

    Parameters
    ----------
    seq: input list
    size: size of chunks

    Returns
    -------
    The list of lists of size `size`
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def torch_coo_eliminate_zeros(x: torch.sparse_coo_tensor):
    # from https://github.com/pytorch/pytorch/issues/31742
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, x.shape)


def preprocess_graph(adjacency, deg=None, subset=None):
    device = adjacency.device
    dtype = adjacency.dtype
    adjacency = adjacency.coalesce()
    row, col = adjacency.indices().cpu()
    edge_weight = adjacency.values().cpu()
    edge_ixs = torch.stack([row, col], ).t().cpu()
    num_nodes = adjacency.shape[0]

    if deg is None:
        deg_out = (adjacency @ torch.ones([num_nodes, 1], device=device, dtype=dtype)).squeeze(1)
        deg_in = (adjacency.t() @ torch.ones([num_nodes, 1], device=device, dtype=dtype)).squeeze(1)
    else:
        assert len(deg) == 2
        deg_out, deg_in = deg

    if subset is not None:
        edge_ixs, edge_weight = subgraph(subset, edge_ixs.t(), edge_weight, num_nodes=num_nodes,
                                         relabel_nodes=True)
        edge_ixs = edge_ixs.t().to(device)
        edge_weight = edge_weight.to(device)

        adjacency = torch.sparse_coo_tensor((edge_ixs.t()), values=torch.ones(edge_ixs.shape[0],
                                                                        device=device,
                                                                        dtype=dtype),
                                      size=[subset.shape[0], subset.shape[0]]).coalesce()
        deg_out = (adjacency @ torch.ones([subset.shape[0], 1], device=device, dtype=dtype)).squeeze(1)
        deg_in = (adjacency.t() @ torch.ones([subset.shape[0], 1], device=device, dtype=dtype)).squeeze(1)
    else:
        subset = torch.arange(num_nodes)

    p_uv_drop = (edge_weight / edge_weight.sum()).to(dtype)
    p_u_out_drop = deg_out / deg_out.sum()
    p_u_in_drop = deg_in / deg_in.sum()

    return PreprocessedGraph(node_ids=subset, p_uv=p_uv_drop.to(device), p_u_out=p_u_out_drop.to(device), 
                             p_u_in=p_u_in_drop.to(device), edge_ixs=edge_ixs.to(device), adj=adjacency.to(device),
                             mutual_information=p_uv_drop @ torch.log(p_uv_drop / (p_u_in_drop[edge_ixs[:, 0]] * p_u_out_drop[edge_ixs[:, 1]])))


class NodeDropoutTypes(Enum):
    UniformNodeDropout = 1
    NHopNodeDropout = 2

    def get(self, adjacency: torch.sparse_coo_tensor, node_dropout_params: dict):
        if self.value == NodeDropoutTypes.UniformNodeDropout.value:
            return UniformNodeDropout(adjacency, **node_dropout_params)
        elif self.value == NodeDropoutTypes.NHopNodeDropout.value:
            return NHopNodeDropout(adjacency, **node_dropout_params)
        else:
            raise ValueError('Unknown node dropout type.')


class NodeDropout(nn.Module):
    def __init__(self, adjacency: torch.sparse_coo_tensor,  cap=None):
        super(NodeDropout, self).__init__()
        self.register_buffer("adjacency", adjacency)
        self.no_dropout = False
        self.num_nodes = self.adjacency.shape[0]
        device = adjacency.device
        dtype = adjacency.dtype
        self.cap = cap
        self.register_buffer("deg_out", (adjacency @ torch.ones(adjacency.shape[0], 1,
                                                                device=device, dtype=dtype)).squeeze())
        self.register_buffer("deg_in", (adjacency.t() @ torch.ones(adjacency.shape[0], 1,
                                                                   device=device, dtype=dtype)).squeeze())
        self.deg = (self.deg_out, self.deg_in)


    def forward(self):
        raise NotImplementedError()


class UniformNodeDropout(NodeDropout):

    def __init__(self, adjacency: torch.sparse_coo_tensor, p_node_drop, cap=None):
        super(UniformNodeDropout, self).__init__(adjacency=adjacency, cap=cap)
        self.p_node_drop = p_node_drop
        self.dropout = nn.Dropout(self.p_node_drop)

    def forward(self):
        if self.no_dropout or not self.training:
            return preprocess_graph(self.adjacency, deg=self.deg, subset=None)
        dtype = self.adjacency.dtype
        device = self.adjacency.device
        if self.p_node_drop == 0:
            return preprocess_graph(self.adjacency, deg=self.deg, subset=None)

        else:
            node_idx_drop = self.dropout(torch.ones([self.num_nodes, ], device=device, dtype=dtype))
            remain_nodes = node_idx_drop.nonzero()[:, 0].cpu()
            if self.cap is not None and remain_nodes.shape[0] > self.cap:
                rdm_subset = np.random.permutation(np.arange(self.cap))
                remain_nodes = remain_nodes[rdm_subset]
            return preprocess_graph(self.adjacency, deg=self.deg, subset=remain_nodes)


class NHopNodeDropout(NodeDropout):

    def __init__(self, adjacency: torch.sparse_coo_tensor, hops=1, n_samples=1000, cap=None):
        super(NHopNodeDropout, self).__init__(adjacency=adjacency, cap=cap)
        self.hops = hops
        self.n_samples = n_samples
        adj = SparseTensor.from_torch_sparse_coo_tensor(adjacency)
        adj_res = adj.clone()
        for hop in range(self.hops-1):
            adj_res = adj @ adj_res
        self.adj_hops = adj_res

    def forward(self):
        if self.no_dropout or not self.training:
            return preprocess_graph(self.adjacency, deg=self.deg, subset=None)
        random_nodes = torch.randint(0, self.num_nodes, size=[self.n_samples]).unique()
        sel = self.adj_hops[random_nodes]
        other_nodes = sel.storage.col()
        all_remaining = torch.cat([random_nodes, other_nodes]).unique()
        if self.cap is not None and all_remaining.shape[0] > self.cap:
            rdm_subset = np.random.permutation(np.arange(self.cap))
            all_remaining = all_remaining[rdm_subset]
        return preprocess_graph(self.adjacency, deg=self.deg, subset=all_remaining)

