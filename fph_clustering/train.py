from sacred import Experiment
from fph_clustering.data.data_modules import FPHDataModule
from fph_clustering.models.direct_constrained_parameterization import FPHConstrainedDirectParameterization
from fph_clustering.util.constants import ModelTypes
import pytorch_lightning as pl
import os
import torch
import fph_clustering.util.utils as util
from fph_clustering.algorithms.agglomerative_linkage import agglomerative_linkage
from fph_clustering.util.utils import networkx_from_torch_sparse
from fph_clustering.algorithms.hierarchy_compression import compress_hierarchy_dasgupta, compress_hierarchy_tsd
from sknetwork.hierarchy.metrics import tree_sampling_divergence
import scipy.sparse as sp
import pickle
import os

ex = Experiment(base_dir='../', interactive=False)


class Harness(object):

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    def init_all(self):
        self._load_data()
        self._init_model()

    @ex.capture(prefix='load_data')
    def _load_data(self, dataset_name, data_root_path, dtype="float32", dataset_params=None):
        dataset_params = {} if dataset_params is None else dataset_params
        self.data_root_path = data_root_path
        if dtype == 'float64':
            self.dtype = torch.float64
        elif dtype == 'float32':
            self.dtype = torch.float32
        else:
            raise ValueError(f'Unknown dtype {dtype}.')

        self.dataset_params = dataset_params
        self.dataset_name = dataset_name.lower().strip()

        full_path = f'{data_root_path}/{self.dataset_name}.pkl.gzip'        
        if os.path.exists(full_path):
            self.data_module = FPHDataModule.from_pickle(full_path, **dataset_params)
        elif self.dataset_name.startswith('ogb'):
            dataset_params = {**dataset_params, **{'prefix': data_root_path}}
            self.data_module = FPHDataModule.from_ogb_dataset(self.dataset_name, **dataset_params)
        else:
            raise ValueError(f'Dataset {self.dataset_name} not found at {data_root_path}.')


    @ex.capture(prefix='model')
    def _init_model(self, model_type, model_params=None, seed=None, tree_init=None, store_best_hierarchy=None,):

        adjacency = self.data_module.dataset.adjacency
        self.internal_nodes = model_params['internal_nodes']

        self.tree_init = tree_init
        init_from = None
        if tree_init == 'avg':
            graph = networkx_from_torch_sparse(adjacency)
            den = agglomerative_linkage(graph, affinity='unitary', linkage='average', check=True)
            if model_params['loss'] == 'TSD':
                compressed = compress_hierarchy_tsd(graph, den, self.internal_nodes)
            else:
                compressed = compress_hierarchy_dasgupta(graph, den, self.internal_nodes)
            init_from = util.tree_to_A_B(compressed, adjacency.shape[0], self.internal_nodes)

        self.model_type = ModelTypes[model_type]
        if model_params is None:
            model_params = {}
        if seed is not None:
            torch.manual_seed(seed)

        if self.model_type == ModelTypes.FPHConstrainedDirectParameterization:
            self.model = FPHConstrainedDirectParameterization(num_nodes=self.data_module.num_nodes,
                                                              initialize_from=init_from,
                                                              store_best_hierarchy=store_best_hierarchy,
                                                              dtype=self.dtype,
                                                              **model_params)
        else:
            raise NotImplementedError()


    @ex.capture(prefix="training")
    def train(self, max_epochs=None, use_gpu=True, val_every=1,
              trainer_params=None):
        if trainer_params is None:
            trainer_params = {}

        trainer = pl.Trainer(gpus=1 if use_gpu and torch.cuda.is_available() else 0,
                             max_epochs=max_epochs,
                             checkpoint_callback=False,
                             check_val_every_n_epoch=int(val_every),
                             progress_bar_refresh_rate=1, 
                             num_sanity_val_steps=1,
                             **trainer_params,
                             )
        trainer.fit(self.model, self.data_module,)
        self.model.cpu()
        self.model.eval()
        self.model.store_on_cpu_process_on_gpu = False
        A = B = None

        if self.model.best_A is None:
            A, B = self.model.compute_A_B()
            A = A.detach().cpu().numpy()
            B = B.detach().cpu().numpy()
            T = util.best_tree(A, B)
            A, B = util.tree_to_A_B(T, A.shape[0], A.shape[1])
            A = torch.tensor(A).to(self.model.A_u.device)
            B = torch.tensor(B).to(self.model.B_u.device)
        
        else:
            A = self.model.best_A.detach()
            B = self.model.best_B.detach()
            T = self.model.best_tree

        graph = next(iter(self.data_module.test_dataloader()))
        with torch.no_grad():
            res_tsd = self.model.compute_TSD(graph, A=A, B=B)
            res_das = self.model.compute_dasgupta(graph, A=A, B=B)
        
        # In the paper we report the sknetwork TSD results.
        # However, due to a bug in sknetwork 0.24.0, the results
        # are slightly different than our own TSD metric.
        # See: https://github.com/sknetwork-team/scikit-network/issues/504.
        skn_TSD = None
        if self.model.num_nodes < 1e6:
            # sknetwork's TSD implementation is inefficient and takes very long for large datasets.
            adj = self.data_module.dataset.adjacency
            A_sp = sp.csr_matrix((adj.values().cpu(), adj.indices().cpu()), shape=adj.shape)
            den, _ = util.tree_to_dendrogram(T, A.shape[0])
            skn_TSD_raw = tree_sampling_divergence(A_sp, den, normalized=False)
            skn_TSD = (100 * skn_TSD_raw / graph.mutual_information).item()

        trainer.log_dir
        import gzip
        fp = f'{trainer.log_dir}/A_B.gzip'
        with gzip.open(fp, 'wb') as f:
            pickle.dump({'A': A, 'B': B, 'T': T}, f)
        print(f'Learned hierarchy stored at {fp}.')

        TSD = res_tsd.metric.cpu().detach().item()
        dasgupta = res_das.loss.detach().cpu().item()

        results = {}
        results['TSD'] = TSD
        results['dasgupta'] = dasgupta
        results['skn_TSD'] = skn_TSD
        results['A_B_path'] = fp

        return results


@ex.command(unobserved=True)
def get_harness(init_all=False):
    harness = Harness(init_all=init_all)
    return harness


@ex.automain
def train(harness=None):
    if harness is None:
        harness = Harness()
    return harness.train()
