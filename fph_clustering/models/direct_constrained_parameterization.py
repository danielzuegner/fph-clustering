import torch
from torch import nn
from fph_clustering.models.base import FPHModel

class FPHConstrainedDirectParameterization(FPHModel):
    def __init__(self, internal_nodes, num_nodes, optimizer_params, loss=None,
                 dtype=torch.float32,
                 store_on_cpu_process_on_gpu=False, initialize_from=None, 
                 store_best_hierarchy=True, **kwargs,
                 ):
        super(FPHConstrainedDirectParameterization, self).__init__(num_nodes=num_nodes,
                                                                   optimizer_params=optimizer_params,
                                                                   dtype=dtype,
                                                                   loss=loss,
                                                                   store_on_cpu_process_on_gpu=store_on_cpu_process_on_gpu,
                                                                   store_best_hierarchy=store_best_hierarchy,
                                                                   **kwargs)
        self.internal_nodes = internal_nodes

        if initialize_from is None:
            self.A_u = nn.Embedding(self.num_nodes, embedding_dim=self.internal_nodes)
            self.A_u.weight.data = self.A_u.weight.data.to(self.dtype()).softmax(-1)

            self.B_u = -1e30 * torch.ones((self.internal_nodes, self.internal_nodes), dtype=self.dtype())
            triu_ixs = torch.triu_indices(*self.B_u.shape, offset=1)
            self.B_u[tuple(triu_ixs)] = 0
            B_rand = torch.rand(self.internal_nodes, self.internal_nodes, dtype=self.dtype())
            B_rand = torch.triu(B_rand, diagonal=1)
            self.B_u[tuple(triu_ixs)] = B_rand[tuple(triu_ixs)]
            self.B_u = self.B_u.softmax(-1)
            self.B_u = self.B_u.mul(torch.triu(torch.ones_like(self.B_u), diagonal=1))

            self.B_u = nn.Parameter(self.B_u, requires_grad=True)
        elif type(initialize_from) == tuple:
            A, B = initialize_from
            self.A_u = nn.Embedding(self.num_nodes, embedding_dim=self.internal_nodes)
            self.A_u.weight.data = A.to(self.dtype())

            self.B_u = nn.Parameter(B.to(self.dtype()), requires_grad=True)
        elif type(initialize_from) == str and initialize_from == 'avg':
            pass
        else:
            raise NotImplementedError('Unknown initialization.')


    def compute_A_B(self, input_nodes: torch.Tensor = None,):
        """
        Assemble the A and B matrices.

        Parameters
        ----------
        input_nodes: torch.Tensor, shape [V']

        Returns
        -------
        A: torch.Tensor, shape [V', N], dtype: torch.float
            The row-stochastic matrix containing the parent probabilities of the internal nodes
            w.r.t. the graph nodes.
        B: torch.Tensor, shape [N, N], dtype torch.float
            The matrix containing the parent probabilities between the internal nodes.
            Properties:
                - row-stochastic
                - upper-triangular
                - zero diagonal

        """
        device = self.B_u.device if not self.store_on_cpu_process_on_gpu else "cuda"

        if input_nodes is not None:
            A_u = self.A_u(input_nodes)
        else:
            A_u = self.A_u.weight

        A = A_u.to(device)
        B = self.B_u.to(device)
        B = torch.cat([B[:-1], torch.zeros_like(B[-1])[None, :]], dim=0)
        B = B.mul(torch.triu(torch.ones_like(B), diagonal=1))
        return A, B
