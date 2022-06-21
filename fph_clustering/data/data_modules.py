import torch
import pytorch_lightning as pl
# from gust import train_val_test_split_adjacency
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from collections import namedtuple
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from fph_clustering.models.utils import preprocess_graph, torch_coo_eliminate_zeros, NodeDropoutTypes
import pickle
import numpy as np
import gzip
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_undirected


GraphBatch = namedtuple("GraphBatch", field_names=["adjacency"])


class FPHDataset(Dataset):
    def __init__(self, adjacency: torch.sparse_coo_tensor, node_dropout=None, make_undirected=True, select_lcc=True, remove_selfloops=False, make_unweighted=True, dtype=torch.float32, **kwargs) -> None:
        super().__init__()
        self.preprocessed_cache = None
        adjacency = adjacency.coalesce()
        num_nodes = adjacency.shape[0]

        if remove_selfloops:
            adjacency = adjacency - torch.sparse_coo_tensor(torch.arange(num_nodes).tile([2, 1]), torch.ones(num_nodes), size=adjacency.shape) * adjacency
            adjacency = torch_coo_eliminate_zeros(adjacency).coalesce()

        edge_index = adjacency.indices()
        values = adjacency.values().to(dtype)

        if make_unweighted:
            values = torch.ones_like(values)

        if make_undirected:
            edge_index, values = to_undirected(edge_index, values, adjacency.shape[0])
        
        if select_lcc:
            A_sp = sp.csr_matrix((values, edge_index), shape=adjacency.shape)
            n_components, labels = connected_components(csgraph=A_sp, directed=False, return_labels=True)
            unq_labels, counts = np.unique(labels, return_counts=True)
            asort = np.argsort(counts)[::-1]
            lcc_nodes = (labels == unq_labels[asort[0]]).nonzero()[0]
            A_sp = A_sp[np.ix_(lcc_nodes, lcc_nodes)]
            r, c = A_sp.nonzero()
            edge_index = torch.tensor(np.array((r, c)))
            values = torch.tensor(A_sp[r, c].A1, dtype=dtype)
            num_nodes = A_sp.shape[0]
        self.adjacency = torch.sparse_coo_tensor(edge_index, values, size=[num_nodes, num_nodes], dtype=dtype).coalesce()
        self.node_dropout = None
        if node_dropout is not None:
            self.node_dropout = NodeDropoutTypes[node_dropout['type']].get(self.adjacency, node_dropout['params'])
            
    
    @classmethod
    def from_ogb_dataset(self, dataset_name: str, node_dropout=None, make_undirected=True, make_unweighted=True, remove_selfloops=False, select_lcc=True, prefix=".", dtype=torch.float32):
        if dataset_name.startswith("ogbl"):
            dataset = PygLinkPropPredDataset(name=dataset_name, root=prefix)
        elif dataset_name.startswith('ogbn'):
            dataset = PygNodePropPredDataset(name=dataset_name, root=prefix)
        else:
            raise NotImplementedError(f'Unknown OGB prefix {dataset_name[:4]}')
        graph = dataset[0]
        num_edges = graph.edge_index.shape[1]
        num_nodes = graph.x.shape[0]
        if graph.edge_attr is None:
                values = torch.ones(num_edges)
        else:
            raise NotImplementedError("edge-attributed OGB graphs not implemented")
        values = values.to(dtype)
        adjacency = torch.sparse_coo_tensor(graph.edge_index, values, (num_nodes, num_nodes))
        return FPHDataset(adjacency, make_undirected=make_undirected, select_lcc=select_lcc, remove_selfloops=remove_selfloops, make_unweighted=make_unweighted, node_dropout=node_dropout)
    
    @classmethod
    def from_pickle(self, path: str, node_dropout=None, make_undirected=True, make_unweighted=True, remove_selfloops=False, select_lcc=True, dtype=torch.float32):
        if path.endswith('.gzip'):
            with gzip.open(path, 'rb') as f:
                loader = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                loader = pickle.load(f)
        assert 'adjacency' in loader
        adjacency = loader['adjacency']

        if sp.isspmatrix(adjacency):
            r, c = adjacency.nonzero()
            adjacency = torch.sparse_coo_tensor(torch.tensor(np.array((r, c))), torch.tensor(adjacency[r, c].A1, dtype=dtype), adjacency.shape)
        return FPHDataset(adjacency, node_dropout=node_dropout, make_undirected=make_undirected, make_unweighted=make_unweighted, remove_selfloops=remove_selfloops, select_lcc=select_lcc)

    def __len__(self):
        return 1   # only one sample (which is the whole graph)

    def __getitem__(self, item):
        return GraphBatch(adjacency=self.adjacency,)

    def collate_fn_train(self, batch: GraphBatch):
        if self.node_dropout is not None:
            preprocessed = self.node_dropout()
        elif self.preprocessed_cache is not None:
            preprocessed = self.preprocessed_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_cache = preprocessed
        return preprocessed

    def collate_fn_val(self, batch: GraphBatch):
        # no node dropout
        if self.preprocessed_cache is not None:
            preprocessed = self.preprocessed_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_cache = preprocessed
        return preprocessed

    def collate_fn_test(self, batch: GraphBatch):
        # no node dropout
        if self.preprocessed_cache is not None:
            preprocessed = self.preprocessed_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_cache = preprocessed
        return preprocessed


class FPHDataModule(pl.LightningDataModule):
    """
    Wrapper class around a FPHDataset to work with Pytorch Lightning.

    **Note**: All data loaders simply return the whole graph.
    """

    def __init__(self, dataset: FPHDataset):
        super().__init__()
        self.dataset = dataset
        self.num_nodes = self.dataset.adjacency.shape[0]

    @classmethod
    def from_adjacency_sparse_tensor(self, adjacency: torch.sparse_coo_tensor, node_dropout=None, make_undirected=True, 
                 make_unweighted=True, remove_selfloops=False, select_lcc=True,
                 dtype=torch.float32, 
                 **kwargs):
        dataset = FPHDataset(adjacency, node_dropout=node_dropout, make_undirected=make_undirected, make_unweighted=make_unweighted,
                                  remove_selfloops=remove_selfloops, select_lcc=select_lcc,
                                  dtype=dtype)
        return FPHDataModule(dataset)

    @classmethod
    def from_pickle(self, path: str, node_dropout=None, make_undirected=True, 
                 make_unweighted=True, remove_selfloops=False, select_lcc=True,
                 dtype=torch.float32, 
                 **kwargs):
        dataset = FPHDataset.from_pickle(path, node_dropout=node_dropout, make_undirected=make_undirected, make_unweighted=make_unweighted,
                                         remove_selfloops=remove_selfloops, select_lcc=select_lcc,
                                         dtype=dtype)
        return FPHDataModule(dataset)
    
    @classmethod
    def from_ogb_dataset(self, dataset_name: str, node_dropout=None, make_undirected=True, 
                         make_unweighted=True, remove_selfloops=False, select_lcc=True,
                         dtype=torch.float32, prefix='.',
                         **kwargs):
        dataset = FPHDataset.from_ogb_dataset(dataset_name, node_dropout=node_dropout, make_undirected=make_undirected, make_unweighted=make_unweighted,
                                              remove_selfloops=remove_selfloops, select_lcc=select_lcc,
                                              dtype=dtype, prefix=prefix)
        return FPHDataModule(dataset)
    

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, collate_fn=self.dataset.collate_fn_train)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, collate_fn=self.dataset.collate_fn_val)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, collate_fn=self.dataset.collate_fn_test)
