import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from abc import ABC
from src.abstract_hamiltonian import AbstractHamiltonian
import torch as pt
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import scipy.sparse.linalg
import numpy as np
from src import constants
from display_data.prediction_shadow import estimate_exp
from src.bf_quantum_state import BFQuantumState



class HeisenbergHamiltonainAntiferro(AbstractHamiltonian, ABC):
    ENERGY_METHODS = ('BF', 'BF_shadow')
    BOUNDARY_CONDITIONS = ('open', 'periodic',)

    # we pass the number of qubits N
    def __init__(self, qubit_num, h, j, boundary_cond):
        super(HeisenbergHamiltonainAntiferro, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.h: float = h
        self.J: float = j
        self.boundary_cond = boundary_cond

    def energy(self, method: str, psi: pt.tensor) -> pt.float:
        assert method in HeisenbergHamiltonainAntiferro.ENERGY_METHODS
        # brute force option
        if method == 'BF':
            return self.energy_bf(psi)

    def energy_bf(self, psi: pt.tensor):
        return pt.conj(psi) @ pt.tensor(self.to_matrix().dot(psi), dtype=self.dtype)

    def observables_for_energy_estimation(self):
        assert self.boundary_cond in HeisenbergHamiltonainAntiferro.BOUNDARY_CONDITIONS
        if self.boundary_cond == 'periodic':
            observables = []
            for i in range(0, self.qubit_num):
                x_arr_b_field = [['X', i]]
                if i <= self.qubit_num - 2:
                    z_arr = [['Z', i], ['Z', i + 1]]
                    y_arr = [['Y', i], ['Y', i + 1]]
                    x_arr = [['X', i], ['X', i + 1]]
                elif i <= self.qubit_num - 1:
                    z_arr = [['Z', i], ['Z', 0]]
                    y_arr = [['Y', i], ['Y', 0]]
                    x_arr = [['X', i], ['X', 0]]
                else:
                    z_arr = None
                    y_arr = None
                    x_arr = None
                observables.append(x_arr_b_field)
                observables.append(z_arr)
                observables.append(y_arr)
                observables.append(x_arr)
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
        if self.boundary_cond == 'periodic':
            for observable in observables:
                sum_product, cnt_match = estimate_exp(measurement, observable)
                if sum_product == 0 and cnt_match == 0:
                    expectation_val = 0
                elif cnt_match == 0 and sum_product != 0:
                    print('cnt_match is zero (problemo)!')
                else:
                    expectation_val = sum_product / cnt_match
                if observable[0][0] == 'Z' or observable[0][0] == 'Y':
                    energy = energy + self.J * expectation_val
                elif len(observable) == 1:
                    energy = energy + self.h * expectation_val
                elif observable[0][0] == 'X':
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

    def ground_state_wavevector(self) -> pt.tensor:
        eigenvalues, eigenvector = self.diagonalize(1, True)
        return pt.tensor(eigenvector[:, 0])

    def to_matrix(self):
        assert self.boundary_cond in HeisenbergHamiltonainAntiferro.BOUNDARY_CONDITIONS
        ham = QubitOperator(None)
        if self.boundary_cond == 'open':
            for i in range(0, self.qubit_num):
                x_string_b_field = 'X' + str(i)
                if i <= self.qubit_num - 2:
                    z_string = 'Z' + str(i) + ' Z' + str(i + 1)
                    y_string = 'Y' + str(i) + ' Y' + str(i + 1)
                    x_string = 'X' + str(i) + ' X' + str(i + 1)
                else:
                    z_string = None
                    y_string = None
                    x_string = None
                ham = ham + QubitOperator(z_string, coefficient=self.J) + QubitOperator(y_string, coefficient=self.J) \
                      + QubitOperator(x_string, coefficient=self.J) + QubitOperator(x_string_b_field, coefficient=self.h)
            return opf_lin.get_sparse_operator(ham, n_qubits=self.qubit_num)
        if self.boundary_cond == 'periodic':
            for i in range(0, self.qubit_num):
                x_string_b_field = 'X' + str(i)
                if i <= self.qubit_num - 2:
                    z_string = 'Z' + str(i) + ' Z' + str(i + 1)
                    y_string = 'Y' + str(i) + ' Y' + str(i + 1)
                    x_string = 'X' + str(i) + ' X' + str(i + 1)
                elif i <= self.qubit_num - 1:
                    z_string = 'Z' + str(i) + ' Z' + str(0)
                    y_string = 'Y' + str(i) + ' Y' + str(0)
                    x_string = 'X' + str(i) + ' X' + str(0)
                else:
                    z_string = None
                    y_string = None
                    x_string = None
                ham = ham + QubitOperator(z_string, coefficient=self.J) + QubitOperator(y_string, coefficient=self.J) \
                      + QubitOperator(x_string, coefficient=self.J) \
                      + QubitOperator(x_string_b_field, coefficient=self.h)
            return opf_lin.get_sparse_operator(ham, n_qubits=self.qubit_num)


def main():
    pass


if __name__ == '__main__':
    main()
