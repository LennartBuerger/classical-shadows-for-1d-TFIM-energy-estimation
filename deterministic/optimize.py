import time

import numpy as np
import torch as pt

from typing import Tuple

from openfermion.ops import QubitOperator
from openfermion.linalg import get_sparse_operator

from .batch_hamiltonian import BatchHamiltonian
from .rbm_wave_function import RBMWaveFunction
from .constants import DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import DEFAULT_MAX_BOND_DIM, DEFAULT_CUTOFF


def optimize_batch(*,
                   visible_num: int = None,
                   hidden_num: int = None,
                   hamiltonian: BatchHamiltonian = None,
                   max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                   cutoff: float = DEFAULT_CUTOFF,
                   svd_backend: str = DEFAULT_SVD_BACKEND,
                   backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                   electron_num: int = None) -> Tuple[pt.Tensor, RBMWaveFunction]:
    assert visible_num is not None
    assert hamiltonian is not None
    device = hamiltonian.device
    wf = RBMWaveFunction(visible_num=visible_num,
                         hidden_num=hidden_num,
                         device=device,
                         max_bond_dim=max_bond_dim,
                         cutoff=cutoff,
                         svd_backend=svd_backend,
                         backprop_backend=backprop_backend,
                         electron_num=electron_num)
    iter_num = 500
    optimizer = pt.optim.Adam(wf.parameters())
    loss = None
    for iter_idx in range(iter_num):
        start_time = time.time()
        loss = hamiltonian.energy(wf=wf)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # exact_diag_time = time.time()
        # hamiltonian_matrix = get_sparse_operator(hamiltonian.of_hamiltonian).todense()
        # state_vector = np.matrix(np.expand_dims(wf.to_state_vector().detach().cpu().numpy(), axis=-1))
        # true_energy = np.asarray(state_vector.H @ hamiltonian_matrix @ state_vector / (state_vector.H @ state_vector))
        # exact_diag_time = time.time() - exact_diag_time
        # print(f'Exact diagonalisation time: {exact_diag_time}')

        # print(f'Iteration #{iter_idx}/{iter_num}, energy = {loss.detach().cpu().numpy()}, true_energy = {true_energy}, elapsed time = {time.time() - start_time}\n')
        print(f'Iteration #{iter_idx}/{iter_num}, energy = {loss.detach().cpu().numpy()}, elapsed time = {time.time() - start_time}\n')

    return loss, wf
