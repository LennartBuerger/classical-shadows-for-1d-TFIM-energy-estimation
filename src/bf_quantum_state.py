import torch as pt
from abstract_quantum_state import AbstractQuantumState
import constants
import numpy as np
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
from tfim_hamiltonian_open_fermion import TfimHamiltonianOpenFermion
import matplotlib.pyplot as plt


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
        print(sampled_indices)
        return existing_indices, prob_index

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state

    def rotate_pauli(self, pauli_string: dict):
        matrix_rot = pt.tensor([1], dtype=self.dtype)
        for i in range(0, self.qubit_num):
            if i in pauli_string:
                matrix_rot = pt.kron(matrix_rot, constants.PAULI_ROT[pauli_string[i]])
            else:
                matrix_rot = pt.kron(matrix_rot, constants.PAULI['I'])
        psi_rot = matrix_rot @ self.psi
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

    # entanglement entropy of arbitrary state, to get the entropy of the ground state we have to pass the ground state to our class
    def entanglement_entropy(self, partition_idx: int = None):
        # procedure for computing the entropy: get Schmidt decomposition of Psi, then entropy is determined by schmidt coefficients
        # first write Psi into matrix which is tensor product of state in system A and in system
        entan_spectrum = self.entan_spectrum(partition_idx=partition_idx)
        return -pt.sum(entan_spectrum * pt.log(entan_spectrum))

    # two point correlation of ground state for different distances and different ratios h/j
    def two_point_correlation(self, dist: int) -> float:
        z_string = 'Z0' + ' Z' + str(dist)
        corr_operator = QubitOperator(z_string)
        corr_operator_sparse = opf_lin.get_sparse_operator(corr_operator, n_qubits=self.qubit_num)
        correlation: float = np.abs(float(pt.conj(self.psi) @ pt.tensor(corr_operator_sparse.dot(self.psi))))
        return correlation

    def correlation_length(self) -> int:
        correlation_length = None
        dist = np.arange(0, self.qubit_num, 1)
        for i in range(0, np.size(dist)):
            correlation = self.two_point_correlation(dist[i])
            if correlation <= np.exp(-1):
                correlation_length = dist[i]
                return correlation_length
                break
        return correlation_length








# down here comes testing rubbish which can be removed later
def main():
    # specify number of qubits
    nq = 12

    #print(BFQuantumState(4, None).measure(10))
    ground_state = TfimHamiltonianOpenFermion(nq, 5,
                                              1, 'open').ground_state_wavevector()
    correlation_length = BFQuantumState(nq, ground_state).correlation_length()
    print(correlation_length)

if __name__ == '__main__':
    main()
