import numpy as np
import torch as pt

from openfermion.ops import QubitOperator


from .constants import BASE_COMPLEX_TYPE
from .rbm_wave_function import RBMWaveFunction


class QubitHamiltonian:
    def __init__(self,
                 visible_num: int = None,
                 of_hamiltonian: QubitOperator = None,
                 phys_space_idx: pt.Tensor = None,
                 dtype=BASE_COMPLEX_TYPE,
                 device=None):
        assert visible_num is not None
        self.visible_num = visible_num

        self.dtype = dtype
        assert device is not None
        self.device = device

        self.of_hamiltonian = of_hamiltonian
        self.terms_num = len(of_hamiltonian.terms)

        self.weights, self.coupling_masks, self.sign_masks = self.parse_of_hamiltonian()
        self.unq_coupling_masks, self.unq_coupling_map = pt.unique(self.coupling_masks,
                                                                   return_inverse=True)

        self.phys_space_idx = phys_space_idx
        self.coupled_indices, self.coupling_weights = self.calc_coupled_indices_and_weights(self.phys_space_idx)

    def parse_of_hamiltonian(self, of_hamiltonian: QubitOperator = None):
        weights, coupling_masks, sign_masks = [], [], []
        for qubit_ops, weight in of_hamiltonian.terms.items():
            weights.append(weight + 0j)
            coupling_mask = 0
            sign_mask = 0
            for qubit_op in qubit_ops:
                if qubit_op[1] == 'X' or qubit_op[1] == 'Y':
                    coupling_mask = coupling_mask | (2 ** (self.visible_num - qubit_op[0] - 1))
                if qubit_op[1] == 'Y' or qubit_op[1] == 'Z':
                    sign_mask = sign_mask | (2 ** (self.visible_num - qubit_op[0] - 1))
                    if qubit_op[1] == 'Y':
                        weights[-1] *= 1j

            coupling_masks.append(coupling_mask)
            sign_masks.append(sign_mask)

            enum_coupling_masks = sorted(enumerate(coupling_masks), key=lambda tup: tup[1])
            weights = [weights[ecm[0]] for ecm in enum_coupling_masks]
            coupling_masks = [coupling_masks[ecm[0]] for ecm in enum_coupling_masks]
            sign_masks = [sign_masks[ecm[0]] for ecm in enum_coupling_masks]

            return (pt.tensor(weights).type(self.dtype).to(self.device),
                    pt.tensor(coupling_masks).to(self.device),
                    pt.tensor(sign_masks).to(self.device))

    def popcount(self, indices):
        return None

    def calc_coupled_signs_matrix(self, indices):
        return 1 - 2 * pt.bitwise_and(1, )

    def calc_coupled_indices_and_weights(self, phys_space_idx: pt.Tensor = None):
        return None
