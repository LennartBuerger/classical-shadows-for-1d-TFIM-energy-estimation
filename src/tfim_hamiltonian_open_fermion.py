import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from abc import ABC
from src.abstract_hamiltonian import AbstractHamiltonian
import torch as pt
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
from src import constants
from display_data.data_acquisition_shadow import derandomized_classical_shadow, randomized_classical_shadow
from display_data.prediction_shadow import estimate_exp
from src.bf_quantum_state import BFQuantumState


class TfimHamiltonianOpenFermion(AbstractHamiltonian, ABC):
    ENERGY_METHODS = ('BF', 'BF_shadow')
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

    def observables_for_energy_estimation(self):
        assert self.boundary_cond in TfimHamiltonianOpenFermion.BOUNDARY_CONDITIONS
        if self.boundary_cond == 'periodic':
            observables = []
            for i in range(0, self.qubit_num):
                x_arr = [['X', i]]
                if i <= self.qubit_num - 2:
                    z_arr = [['Z', i], ['Z', i + 1]]
                elif i <= self.qubit_num - 1:
                    z_arr = [['Z', i], ['Z', 0]]
                else:
                    z_arr = None
                observables.append(x_arr)
                observables.append(z_arr)

        if self.boundary_cond == 'open':
            observables = []
            for i in range(0, self.qubit_num):
                x_arr = [['X', i]]
                if i <= self.qubit_num - 2:
                    z_arr = [['Z', i], ['Z', i + 1]]
                else:
                    z_arr = None
                observables.append(x_arr)
                observables.append(z_arr)
        return observables

# we either have to pass psi or measurement, when no measurement=None the method needs psi to do the measurement
    def energy_shadow(self, psi: pt.tensor, num_of_measurements: int,
                      measurement_method: str, measurement):
        observables = self.observables_for_energy_estimation()
        if measurement is None:
            measurement = BFQuantumState(self.qubit_num,
                                         psi).measurement_shadow(num_of_measurements,
                                                                 measurement_method, observables)
        # now we have our measurement outcome and our observables stored in the correct format
        energy = 0
        for i in range(0, len(observables)):
            sum_product, cnt_match = estimate_exp(measurement, observables[i])
            if sum_product == 0 and cnt_match == 0:
                expectation_val = 0
            elif cnt_match == 0 and sum_product != 0:
                print('cnt_match is zero (problemo)!')
            else:
                expectation_val = sum_product / cnt_match
            if i % 2 == 0:
                energy = energy + self.h * expectation_val
            else:
                energy = energy + self.J * expectation_val
        return energy

    def diagonalize(self, nr_eigs: int, return_eig_vecs: bool) -> (pt.float, pt.tensor):
        if return_eig_vecs:
            return scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
        else:
            eigenvalues, _ = scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
            return eigenvalues

    def ground_state_energy(self) -> pt.double:
        return self.diagonalize(1, False)

    def energy_eigenvalues_theo(self, k_val) -> pt.double:
        return 2 * np.abs(self.J) * np.sqrt((np.cos(k_val) - np.abs(self.h / self.J)) ** 2 + np.sin(k_val) ** 2)

    def ground_state_energy_theo(self) -> pt.double:
        n_vals = pt.linspace(1, self.qubit_num / 2, int(self.qubit_num / 2))
        k_vals = (2 * n_vals - pt.ones(int(self.qubit_num / 2))) * pt.pi / self.qubit_num
        energy_eigenvalues = 2 * self.J * pt.sqrt((pt.cos(k_vals) - self.h / self.J) ** 2 + pt.sin(k_vals) ** 2)
        return pt.sum(energy_eigenvalues)

    # for h/j < 1 the energy gap is given by E_gap = E_0 - E_2 and for h/j >= 1 by E_0 - E_1
    def energygap(self, energy_eigenvalues) -> pt.float:
        if energy_eigenvalues is None:
            if self.h / self.J > 1:
                eigenvalues = self.diagonalize(2, False)
                energy_gap = eigenvalues[1] - eigenvalues[0]
            else:
                eigenvalues = self.diagonalize(3, False)
                energy_gap = eigenvalues[2] - eigenvalues[0]
            return np.abs(energy_gap)
        else:
            if self.h / self.J > 1:
                energy_gap = energy_eigenvalues[1] - energy_eigenvalues[0]
            else:
                energy_gap = energy_eigenvalues[2] - energy_eigenvalues[0]
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
        if self.boundary_cond == 'periodic':
            # first excited state in the ABC sector
            n_abc = pt.linspace(1, int(self.qubit_num / 2), int(self.qubit_num / 2))
            k_abc = (2 * n_abc - 1) * pt.pi / self.qubit_num
            energy_gap_abc = 2 * pt.min(
                2 * self.J * pt.sqrt((pt.cos(k_abc) - self.h / self.J) ** 2 + pt.sin(k_abc) ** 2))

            # ground state energy in the PBC sector
            n_pbc = pt.linspace(1, int(self.qubit_num / 2) - 1, int(self.qubit_num / 2) - 1)
            k_pbc = (2 * n_pbc) * pt.pi / self.qubit_num
            ground_energy_pbc = -2 * self.J - pt.sum(
                2 * self.J * pt.sqrt((pt.cos(k_pbc) - self.h / self.J) ** 2 + pt.sin(k_pbc) ** 2))
            energy_gap_pbc = pt.abs(self.ground_state_energy_theo()) - pt.abs(ground_energy_pbc)

            if self.h / self.J > 1:
                return pt.min(pt.tensor([energy_gap_abc, energy_gap_pbc]))
            else:
                # there are only two energygaps in pt.max, thus using pt.max the second lowest energygap is choosen
                return pt.max(pt.tensor([energy_gap_abc, energy_gap_pbc]))

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
                ham = ham + QubitOperator(z_string, coefficient=self.J) + QubitOperator(x_string,
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
                ham = ham + QubitOperator(z_string, coefficient=self.J) + QubitOperator(x_string,
                                                                                        coefficient=self.h)
            return opf_lin.get_sparse_operator(ham, n_qubits=self.qubit_num)


def main():
    qubit_num: int = 12

    # print(TfimHamiltonianOpenFermion(qubit_num, 2, 1, 'periodic').ground_state_energy())
    # print(TfimHamiltonianOpenFermion(qubit_num, 2, 1, 'periodic').ground_state_energy_theo())

    print(TfimHamiltonianOpenFermion(4, 0, 1, 'periodic').ground_state_wavevector())



if __name__ == '__main__':
    main()
