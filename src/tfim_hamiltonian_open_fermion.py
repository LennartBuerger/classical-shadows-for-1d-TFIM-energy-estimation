from abc import ABC
import abstract_hamiltonian
import torch as pt
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import constants

from abstract_quantum_state import AbstractQuantumState


class TfimHamiltonianOpenFermion(abstract_hamiltonian.AbstractHamiltonian, ABC):
    ENERGY_METHODS = ('BF',)
    BOUNDARY_CONDITIONS = ('open', 'periodic',)

    # we pass the number of qubits N
    def __init__(self, qubit_num, h, j, boundary_cond):
        super(TfimHamiltonianOpenFermion, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.h: float = h
        self.J: float = j
        self.boundary_cond = boundary_cond

    def energy(self, method: str, psi: pt.tensor) -> pt.float:
        assert method in TfimHamiltonianOpenFermion.ENERGY_METHODS
        # brute force option
        if method == 'BF':
            return self.energy_bf(psi)

    def energy_bf(self, psi: pt.tensor):
        return pt.conj(psi) @ pt.tensor(self.to_matrix().dot(psi), dtype=self.dtype)

    def diagonalize(self, nr_eigs: int, return_eig_vecs: bool) -> (pt.float, pt.tensor):
        if return_eig_vecs:
            return scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
        else:
            eigenvalues, _ = scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
            return eigenvalues

    def ground_state_energy(self) -> pt.double:
        return self.diagonalize(1, False)

    def energy_eigenvalues_theo(self, k_val) -> pt.double:
        return 2 * np.abs(self.J) * np.sqrt((np.cos(k_val) - np.abs(self.h / self.J))**2 + np.sin(k_val)**2)

    def ground_state_energy_theo(self) -> pt.double:
        n_vals = pt.linspace(1, self.qubit_num / 2, int(self.qubit_num / 2))
        k_vals = (2 * n_vals - pt.ones(int(self.qubit_num / 2))) * pt.pi / self.qubit_num
        energy_eigenvalues = 2 * self.J * pt.sqrt((pt.cos(k_vals) - self.h / self.J) ** 2 + pt.sin(k_vals) ** 2)
        return pt.sum(energy_eigenvalues)

    # for h/j < 1 the energy gap is given by E_gap = E_0 - E_2 and for h/j >= 1 by E_0 - E_1
    def energygap(self) -> pt.float:
        if self.h / self.J > 1:
            eigenvalues = self.diagonalize(2, False)
            energy_gap = eigenvalues[1] - eigenvalues[0]
        else:
            eigenvalues = self.diagonalize(3, False)
            energy_gap = eigenvalues[2] - eigenvalues[0]
        return np.abs(energy_gap)

    # for infintely long chain
    def theoretical_energygap(self) -> pt.float:
        if self.h / self.J >= 1:
            energy_gap = 2 * np.abs(self.J) * (np.abs(self.h / self.J) - 1)
        else:
            energy_gap = 2 * np.abs(self.J) * (1 - np.abs(self.h / self.J))
        return energy_gap

    # for chain of finite size
    def theo_energygap_finite_size(self) -> pt.float:
        if self.boundary_cond == 'open':
            energy_gap = 2 * np.abs(self.J) * np.sqrt((np.cos(np.pi / self.qubit_num) - self.h / self.J) ** 2
                                                      + np.sin(np.pi / self.qubit_num) ** 2)
        if self.boundary_cond == 'periodic':
            if self.h <= 1:
                n_vals = pt.linspace(1, self.qubit_num / 2, int(self.qubit_num / 2))
                k_abc = (2 * n_vals - pt.ones(int(self.qubit_num / 2))) * pt.pi / self.qubit_num
                energy_eigenvalues = 2 * self.J * pt.sqrt((pt.cos(k_abc) - self.h / self.J) ** 2 + pt.sin(k_abc) ** 2)
                energy_gap = 2 * pt.min(energy_eigenvalues)
            else:
                energy_gap = 2 * np.abs(self.J) * np.sqrt((np.cos(1 * np.pi / self.qubit_num) - self.h / self.J) ** 2
                                                          + np.sin(1 * np.pi / self.qubit_num) ** 2)

        return energy_gap

    # calculate theoretical prediction of the energy gap

    def ground_state_wavevector(self) -> pt.tensor:
        eigenvalues, eigenvector = self.diagonalize(1, True)
        return pt.tensor(eigenvector[:, 0])

    def to_matrix(self):
        assert self.boundary_cond in TfimHamiltonianOpenFermion.BOUNDARY_CONDITIONS
        ham = QubitOperator(None)
        if self.boundary_cond == 'open':
            for i in range(0, self.qubit_num):
                x_string = 'X' + str(i)
                if i <= self.qubit_num - 2:
                    z_string = 'Z' + str(i) + ' Z' + str(i + 1)
                else:
                    z_string = None
                ham = ham + QubitOperator(z_string, coefficient=-1 * self.J) + QubitOperator(x_string,
                                                                                             coefficient=self.h)
            return opf_lin.get_sparse_operator(ham, n_qubits=self.qubit_num)
        if self.boundary_cond == 'periodic':
            for i in range(0, self.qubit_num):
                x_string = 'X' + str(i)
                if i <= self.qubit_num - 2:
                    z_string = 'Z' + str(i) + ' Z' + str(i + 1)
                elif i <= self.qubit_num - 1:
                    z_string = 'Z' + str(i) + ' Z' + str(0)
                else:
                    z_string = None
                ham = ham + QubitOperator(z_string, coefficient=-1 * self.J) + QubitOperator(x_string,
                                                                                             coefficient=self.h)
            return opf_lin.get_sparse_operator(ham, n_qubits=self.qubit_num)


def main():
    qubit_num: int = 12

    #print(TfimHamiltonianOpenFermion(qubit_num, 2, 1, 'periodic').ground_state_energy())
    #print(TfimHamiltonianOpenFermion(qubit_num, 2, 1, 'open').ground_state_energy_theo())


if __name__ == '__main__':
    main()
