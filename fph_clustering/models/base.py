import numpy as np
import pytorch_lightning as pl
import torch
import numpy as np
from collections import namedtuple
from typing import Dict

from fph_clustering.algorithms.shared import (compute_B_anc, compute_B_square_inv, compute_p, compute_q,
                                                compute_dasgupta, compute_TSD)
from fph_clustering.models.utils import PreprocessedGraph, configure_optimizers
from fph_clustering.util.utils import tree_to_A_B
from fph_clustering.util.constants import Losses
import fph_clustering.util.utils as util


Result = namedtuple("Result", field_names=["A", "B", "loss", 'tree', 'metric'])

class FPHModel(pl.LightningModule):

    def __init__(self, num_nodes, optimizer_params=None,
                 loss=None, dtype=torch.float32,
                 store_on_cpu_process_on_gpu=False, store_best_hierarchy=True,
                 same_leaf_correction=False):
        super(FPHModel, self).__init__()
        if isinstance(dtype, str):
            if dtype == "float32":
                dtype = torch.float32
            elif dtype == "float64":
                dtype = torch.float64
            elif dtype == "double":
                dtype = torch.float64
            elif dtype == "float16":
                dtype = torch.float16
            else:
                raise NotImplementedError("unknown dtype")

        self.num_nodes = num_nodes
        self.use_dtype = dtype
        torch.set_default_dtype(dtype)
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.losses = []
        self.same_leaf_correction = same_leaf_correction
        self.lowest_loss = float('inf')
        self.store_best_hierarchy = store_best_hierarchy

        self.store_on_cpu_process_on_gpu = store_on_cpu_process_on_gpu

        if loss is None:
            loss = "TSD"
        self.loss = Losses[loss]

        self.best_A = None
        self.best_B = None
        self.best_tree = None

    def dtype(self):
        return self.use_dtype

    def compute_A_B(self, **kwargs):
        raise NotImplementedError()

    def forward(self, **kwargs):
        return self.compute_TSD(**kwargs)

    def compute_TSD(self, graph: PreprocessedGraph, A: torch.tensor = None, B: torch.tensor = None) -> Result:
        """
        Compute the TSD metric of the input graph according to the model at hand.

        Parameters
        ----------
        graph: PreprocessedGraph
            The input graph.
        A_B_kwargs: dict
            Keyword arguments passed to self.compute_A_B.
        Returns
        -------
        Result: Result. Named tuple containing
            * TSD: torch.float
                The resulting TSD score
            * p_x: torch.Tensor, shape [N,], dtype torch.float
                The p(x) distribution over the internal nodes.
            * q_x: torch.Tensor, shape [N,], dtype torch.float
                The q(x) distribution over the internal nodes.
            * A: torch.Tensor, shape [V', N], where V' is the number of nodes after dropout. dtype torch.float
                The row-stochastic matrix containing the parent probabilities of the internal nodes
                w.r.t. the graph nodes.
            * B: torch.Tensor, shape [N, N], dtype torch.float
                The matrix containing the parent probabilities between the internal nodes.
                Properties:
                    - row-stochastic
                    - upper-triangular
                    - zero diagonal
        """

        node_ids = graph.node_ids
        if len(node_ids) == self.num_nodes:
            node_ids = None
        
        T = None      
        if A is None or B is None:
            A, B = self.compute_A_B(input_nodes=node_ids)    
            device = A.device
            if not self.training:
                T = util.best_tree(A.detach().cpu().numpy(), B.detach().cpu().numpy())
                A, B = tree_to_A_B(T, A.shape[0], B.shape[0])
                A = A.to(device)
                B = B.to(device)

        TSD = compute_TSD(A=A, B=B, preprocessed_adj=graph)
        return Result(A=A, B=B,
                      loss=-TSD, tree=T, metric=100 * (TSD / graph.mutual_information))

    def compute_dasgupta(self, graph: PreprocessedGraph,
                        A: torch.tensor = None, B: torch.tensor = None) -> Result:

        node_ids = graph.node_ids
        if len(node_ids) == self.num_nodes:
            node_ids = None
        T = None
        if A is None or B is None:
            A, B = self.compute_A_B(input_nodes=node_ids)    
            device = A.device
            if not self.training:
                T = util.best_tree(A.detach().cpu().numpy(), B.detach().cpu().numpy())
                A, B = tree_to_A_B(T, A.shape[0], B.shape[0])
                A = A.to(device)
                B = B.to(device)

        dasgupta_loss = compute_dasgupta(A, B, adj=None, preprocessed_adj=graph )
        return Result(A=A, B=B, loss=dasgupta_loss, tree=T, metric=dasgupta_loss)

    def compute_loss(self, batch: PreprocessedGraph, batch_idx):
        if self.loss == Losses.TSD:
            res = self.compute_TSD(batch)
            loss = res.loss
            metric = res.metric

        elif self.loss == Losses.DASGUPTA:
            res = self.compute_dasgupta(batch)
            loss = res.loss
            metric = res.metric
        else:
            raise NotImplementedError("unknown loss.")
        return loss, metric

    def training_step(self, batch: PreprocessedGraph, batch_idx):
        loss, metric = self.compute_loss(batch=batch, batch_idx=batch_idx)
        self.log(f"train_{self.loss.name}", metric.detach(), prog_bar=True, on_epoch=True)
        return dict(loss=loss)

    def validation_step(self, batch: PreprocessedGraph, batch_idx):
        store_bef = self.store_on_cpu_process_on_gpu
        if store_bef:
            self.store_on_cpu_process_on_gpu = False
        if self.loss == Losses.DASGUPTA:
            res = self.compute_dasgupta(batch,)
            loss = res.loss
            metric = loss
        elif self.loss == Losses.TSD:
            res = self.compute_TSD(batch)
            loss = res.loss
            metric = res.metric
        self.log(f"val_{self.loss.name}", metric.detach(), prog_bar=True, on_epoch=True)
        self.log(f"best_val_loss", self.lowest_loss, prog_bar=True, on_epoch=True)

        self.losses.append(loss.detach().cpu())

        if self.store_best_hierarchy and loss < self.lowest_loss:
            self.lowest_loss = loss.detach()
            self.best_A = res.A.clone()
            self.best_B = res.B.clone()
            self.best_tree = res.tree

        self.store_on_cpu_process_on_gpu = store_bef
        return dict(loss=loss)

    def validation_end(self):
        pass

    def configure_optimizers(self):
        return configure_optimizers(self)

    def link_prediction(self, graph: PreprocessedGraph, eval_ixs=None, cpu=False,
                        same_leaf_correction=False):
        self.eval()
        if cpu:
            device_before = next(self.parameters()).device
            store_on_cpu_before = self.store_on_cpu_process_on_gpu
            self.cpu()
            self.store_on_cpu_process_on_gpu = False
        
        device = next(self.parameters()).device

        if self.best_A is None:
            A, B = self.compute_A_B(input_nodes=None)
        else:
            A = self.best_A.to(device)
            B = self.best_B.to(device)

        p_u_out = graph.p_u_out
        p_u_in = graph.p_u_in
        p_uv = graph.p_uv
        edges_sampled = graph.edge_ixs.to(device)

        B_anc = compute_B_anc(B)
        B_square_inv = compute_B_square_inv(B_anc)

        p_x = compute_p(A, B, edges_sampled, p_uv,
                        B_anc=B_anc, B_square_inv=B_square_inv)
        q_x = compute_q(A, B, (p_u_out, p_u_in),
                        B_anc=B_anc, B_square_inv=B_square_inv,
                        same_leaf_correction=same_leaf_correction) + 1e-7
        sigma = p_x / q_x

        internal_nodes = B.shape[0]
        p_anc = A @ (B_anc + torch.eye(internal_nodes, dtype=B_anc.dtype, device=device))

        if eval_ixs is not None:
            if isinstance(eval_ixs, dict):
                true_head = eval_ixs['head']
                true_tail = eval_ixs['tail']
                false_tail = eval_ixs['tail_neg']
                B_sq_sigma = B_square_inv @ sigma
                edge_scores = ((p_anc[true_head].mul(p_anc[true_tail]) @ (B_sq_sigma)).mul(p_u_out[true_head]).mul(p_u_in[true_tail]))
                non_edge_tail = []

                for ix in range(false_tail.shape[0]):
                    r = (p_anc[true_head[ix].repeat(false_tail.shape[-1])].mul(p_anc[false_tail[ix]]) @ (
                        B_sq_sigma))
                    scores = (
                        r.mul(p_u_out[true_head[ix].repeat(false_tail.shape[-1])]).mul(p_u_in[false_tail[ix]]))
                    non_edge_tail.append(scores.numpy())
                non_edge_tail = torch.tensor(np.row_stack(non_edge_tail))
                lpred = dict(
                    edge_scores=edge_scores,
                    non_edge_tail=non_edge_tail,
                    non_edge_head=None
                )
            else:
                edge_ixs, non_edge_ixs = eval_ixs
                true_head, true_tail = edge_ixs
                false_head, false_tail = non_edge_ixs
                B_sq_sigma = B_square_inv @ sigma

                edge_scores = ((p_anc[true_head].mul(p_anc[true_tail]) @ (B_sq_sigma)).mul(p_u_out[true_head]).mul(
                    p_u_in[true_tail]))

                non_edge_scores = ((p_anc[false_head].mul(p_anc[false_tail]) @ (B_sq_sigma)).mul(p_u_out[false_head]).mul(
                    p_u_in[false_tail]))

                lpred = dict(edge_scores=edge_scores.detach(), non_edge_scores=non_edge_scores.detach())

        else:
            p_anc_tensor = (p_anc[:, None, :].mul(p_anc[None, :, :]))
            lpred = (p_u_out[:, None] @ p_u_in[None, :]).mul(p_anc_tensor @ (B_square_inv @ sigma))

        if cpu:
            if "cuda" in str(device_before):
                self.cuda()
            self.store_on_cpu_process_on_gpu = store_on_cpu_before

        return lpred
