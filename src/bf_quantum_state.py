import torch as pt
from abstract_quantum_state import AbstractQuantumState
import constants
import numpy as np
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import kron


class BFQuantumState(AbstractQuantumState):
    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, qubit_num, psi):
        super(BFQuantumState, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        if psi is None:
            self.psi = pt.rand(2 ** self.qubit_num, dtype=self.dtype)
            self.psi = self.psi / pt.linalg.norm(self.psi)
        else:
            self.psi = psi

    # measuring amplitude with respect to some basis vector

    def amplitude(self, basis_idx: pt.Tensor) -> pt.tensor:
        return self.psi[basis_idx]

    # probability of measuring our quantum state in a certain basis vector state

    def prob(self, basis_idx: pt.Tensor) -> float:
        return self.psi[basis_idx] * pt.conj(self.psi[basis_idx])

    def norm(self) -> float:
        return pt.linalg.norm(self.psi)

    # measures state in computational basis, B is number of samples we take

    def measure(self, batch_size: int) -> (pt.tensor, pt.tensor):
        # to perform the measurement we first generate a multinomial probability distribution from our known state
        # from which we sample afterwards
        distribution: float = self.psi * pt.conj(self.psi)
        sampled_indices = pt.multinomial(distribution.real, batch_size, replacement=True)
        existing_indices, counts = pt.unique(sampled_indices, return_counts=True)
        prob_index = counts / batch_size
        return existing_indices, prob_index

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state
    # we use sparse matrices to do the rotation since this way it can be done efficiently for more than 20 qubits
    # e.g. a rotation in the X-basis would be given by H \tensor H |Psi> = I \tensor H * H \tensor I |Psi>
    # which can be done really efficient using sparse matrices
    def rotate_pauli(self, pauli_string: dict):
        psi_rot = csr_matrix(self.psi)
        counter = 0
        for i in pauli_string:
            matrix_rot = csr_matrix([1])
            for j in range(0, self.qubit_num):
                if j == i:
                    matrix_rot = kron(matrix_rot, constants.PAULI_ROT_SPARSE[pauli_string[i]])
                else:
                    matrix_rot = kron(matrix_rot, constants.PAULI_ROT_SPARSE['I'])
                matrix_rot.eliminate_zeros()
            if counter == 0:
                psi_rot = matrix_rot.dot(psi_rot.transpose())
                counter = counter + 1
            else:
                psi_rot = matrix_rot.dot(psi_rot)
        psi_rot = pt.flatten(pt.tensor(psi_rot.toarray()))
        return BFQuantumState(self.qubit_num, psi_rot)

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int) -> dict:
        return self.rotate_pauli(pauli_string).measure(batch_size)

    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        matrix_rot = pt.tensor([1], dtype=self.dtype)
        for i in range(self.qubit_num):
            if i in clifford_string:
                matrix_rot = pt.kron(matrix_rot, constants.CLIFFORD[clifford_string[i]])
            else:
                matrix_rot = pt.kron(matrix_rot, constants.PAULI['Id'])
        psi_clifford = matrix_rot @ self.psi
        return BFQuantumState(self.qubit_num, psi_clifford)

    def entan_spectrum(self, partition_idx: int = None):
        if partition_idx is None:
            partition_idx = self.qubit_num // 2
        else:
            assert (0 <= partition_idx) and (partition_idx < self.qubit_num)
        a_qubit_num = partition_idx
        b_qubit_num = self.qubit_num - a_qubit_num
        psi_matrix = self.psi.reshape((2 ** a_qubit_num, 2 ** b_qubit_num))
        schmidt_coeffs = pt.linalg.svdvals(psi_matrix)

        return schmidt_coeffs ** 2

    # entanglement entropy of arbitrary state, to get the entropy of the ground state we have to pass the ground
    # state to our class
    def entanglement_entropy(self, partition_idx: int = None):
        # procedure for computing the entropy: get Schmidt decomposition of Psi, then entropy is determined by
        # schmidt coefficients
        entan_spectrum = self.entan_spectrum(partition_idx=partition_idx)
        return -pt.sum(entan_spectrum * pt.log(entan_spectrum))

    # two point correlation of ground state for different distances and different ratios h/j
    def two_point_correlation(self, dist: int, basis: str) -> float:
        if basis == 'Z':
            z_string = 'Z0' + ' Z' + str(dist)
            corr_operator = QubitOperator(z_string)
        if basis == 'XYZ':
            z_string = 'Z0' + ' Z' + str(dist)
            y_string = 'Y0' + ' Y' + str(dist)
            x_string = 'X0' + ' X' + str(dist)
            corr_operator = QubitOperator(z_string) + QubitOperator(y_string) + QubitOperator(x_string)
        if basis == 'X':
            x_string = 'X0' + ' X' + str(dist)
            corr_operator = QubitOperator(x_string)
        if basis == 'Y':
            x_string = 'Y0' + ' Y' + str(dist)
            corr_operator = QubitOperator(x_string)
        corr_operator_sparse = opf_lin.get_sparse_operator(corr_operator, n_qubits=self.qubit_num)
        correlation = pt.conj(self.psi) @ pt.tensor(corr_operator_sparse.dot(self.psi))
        return correlation.real

    def correlation_length(self, basis) -> int:
        correlation_length = None
        dist = np.arange(0, self.qubit_num, 1)
        for i in range(0, np.size(dist)):
            correlation = np.abs(self.two_point_correlation(dist[i], basis))
            if correlation <= np.exp(-1):
                correlation_length = dist[i]
                return correlation_length
                break
        return correlation_length








# down here comes testing rubbish which can be removed later
def main():
    # specify number of qubits
    nq = 12
    pauli_str = {0: 'X', 1: 'X', 2: 'Y', 3: 'X'}
    psi = pt.rand(2 ** 4, dtype=constants.DEFAULT_COMPLEX_TYPE)
    print(BFQuantumState(4, psi).rotate_pauli_sparse(pauli_str))
    print(BFQuantumState(4, psi).rotate_pauli(pauli_str))



if __name__ == '__main__':
    main()
