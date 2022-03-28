import torch as pt
from abstract_quantum_state import AbstractQuantumState
import constants
import numpy as np
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from data_acquisition_shadow import derandomized_classical_shadow, randomized_classical_shadow
from prediction_shadow import estimate_exp


class BFQuantumState(AbstractQuantumState):
    perm_zero_dict = {}
    perm_one_dict = {}
    phase_minus_one_dict = {}
    phase_i_dict = {}

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, qubit_num, psi):
        super(BFQuantumState, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.psi = psi
        if self.qubit_num not in BFQuantumState.perm_zero_dict:
            self.init_cached_perms_and_phases()

    def init_cached_perms_and_phases(self):
        perm_zero = []
        perm_one = []
        phase_minus_one = []
        phase_i = []
        indices = pt.arange(0, 2 ** self.qubit_num, 1)
        for qubit_idx in range(0, self.qubit_num):
            perm_zero.append(pt.bitwise_and(indices,
                                            pt.bitwise_not(pt.tensor([2 ** (self.qubit_num - 1 - qubit_idx)]))))
            perm_one.append(pt.bitwise_or(indices, 2 ** (self.qubit_num - 1 - qubit_idx)))
            phase_minus_one.append((-1) ** (pt.bitwise_and(indices,
                                                           2 ** (self.qubit_num - 1 - qubit_idx))
                                            >> (self.qubit_num - 1 - qubit_idx)))
            phase_i.append((-1j) ** (pt.bitwise_and(indices,
                                                    2 ** (self.qubit_num - 1 - qubit_idx))
                                     >> (self.qubit_num - 1 - qubit_idx)))
        BFQuantumState.perm_zero_dict[self.qubit_num] = perm_zero
        BFQuantumState.perm_one_dict[self.qubit_num] = perm_one
        BFQuantumState.phase_i_dict[self.qubit_num] = phase_i
        BFQuantumState.phase_minus_one_dict[self.qubit_num] = phase_minus_one

    # measuring amplitude with respect to some basis vector

    def amplitude(self, basis_idx: pt.Tensor) -> pt.tensor:
        return self.psi[basis_idx]

    # probability of measuring our quantum state in a certain basis vector state

    def prob(self, basis_idx: pt.Tensor) -> float:
        return self.psi[basis_idx] * pt.conj(self.psi[basis_idx])

    def norm(self) -> float:
        return pt.linalg.norm(self.psi)

    # measures state in computational basis, B is number of samples we take
    # does NOT measure each qubit in the computational basis
    # gives an index of the state our wavevector collapsed into

    def measure(self, batch_size: int) -> (pt.tensor, pt.tensor):
        # to perform the measurement we first generate a multinomial probability distribution from our known state
        # from which we sample afterwards
        distribution: float = self.psi * pt.conj(self.psi)
        sampled_indices = pt.multinomial(distribution.real, batch_size, replacement=True)
        existing_indices, counts = pt.unique(sampled_indices, return_counts=True)
        prob_index = counts / batch_size
        return existing_indices, prob_index

    def measurement_shadow(self, num_of_measurements: int, measurement_method: str,
                           observables):
        # feed observables to derandomized classical shadow,
        # the generation of a measurement procedure becomes very slow when num_of_measurements becomes big
        # (~1000). In the case of the ising model it is alright to proceed by using batches because we only
        # measure two different Pauli strings. We have to remove this batched procedure though when
        # dealing with more complicated systems where we have to measure more than 100 different Pauli Strings
        # for now we form batches of 100 measurements
        if measurement_method == 'derandomized':
            batch_size = 100
            measurement_procedure = []
            # the derandomization procedure makes two measurements per measurement_per_observable which is the input
            # --> we divide by two to obtain the same number of measurements for randomized and derandomized
            for i in range(0, int(num_of_measurements / batch_size)):
                measurement_procedure_batch = derandomized_classical_shadow(observables,
                                                                            int(batch_size / 2), self.qubit_num)
                for j in range(0, batch_size):
                    measurement_procedure.append(measurement_procedure_batch[j])
        if measurement_method == 'randomized':
            measurement_procedure = randomized_classical_shadow(num_of_measurements, self.qubit_num)
        # convert the array measurement_procedure to array of dicts to have the right format for the measurement
        measurement_procedure_dict = []
        for i in range(0, len(measurement_procedure)):
            pauli_dict = {}
            for j in range(0, len(measurement_procedure[i])):
                pauli_dict[j] = measurement_procedure[i][j]
            measurement_procedure_dict.append(pauli_dict)
        # now we apply the measurements to our state psi
        measurement_index = []
        for i in range(0, len(measurement_procedure_dict)):
            measurement_index.append(int(self.measure_pauli(measurement_procedure_dict[i], 1)[0]))
        # returns one array with measurement basis, one array with the index of the measured state in the shape
        # e.g. [10, 52, 92, 0, 7, 17, 29, 13]
        return measurement_procedure, measurement_index

    def rotate_pauli(self, pauli_string: dict):

        def apply_x_rot(psi, qubit_num, qubit_idx):
            return 1 / pt.sqrt(pt.tensor([2])) * (psi[BFQuantumState.perm_zero_dict[qubit_num][qubit_idx]] +
                                                  BFQuantumState.phase_minus_one_dict[qubit_num][qubit_idx]
                                                  * psi[BFQuantumState.perm_one_dict[qubit_num][qubit_idx]])

        def apply_y_rot(psi, qubit_num, qubit_idx):
            return 1 / pt.sqrt(pt.tensor([2])) * \
                   ((BFQuantumState.phase_i_dict[qubit_num][qubit_idx]
                     * psi)[BFQuantumState.perm_zero_dict[qubit_num][qubit_idx]] +
                    BFQuantumState.phase_minus_one_dict[qubit_num][qubit_idx] *
                    (BFQuantumState.phase_i_dict[qubit_num][qubit_idx]
                     * psi)[BFQuantumState.perm_one_dict[qubit_num][qubit_idx]])

        psi_rot = self.psi
        for i in pauli_string:
            if pauli_string[i] == 'X':
                psi_rot = apply_x_rot(psi_rot, self.qubit_num, i)

            if pauli_string[i] == 'Y':
                psi_rot = apply_y_rot(psi_rot, self.qubit_num, i)

        return BFQuantumState(self.qubit_num, psi_rot)

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int) -> (pt.tensor, pt.tensor):
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

    def two_point_correlation_shadow(self, num_of_measurements, measurement,
                                     measurement_method: str, dist: int, basis: str):
        if measurement_method == 'randomized':
            if basis == 'Z':
                observables = [[['Z', 0], ['Z', dist]]]

        if measurement_method == 'derandomized':
            if basis == 'Z':
                observables = [[['Z', 0], ['Z', dist]]]
                if measurement is None:
                    measurement = self.measurement_shadow(num_of_measurements, 'derandomized', observables)

        sum_product, cnt_match = estimate_exp(measurement, observables[0])
        if sum_product == 0 and cnt_match == 0:
            expectation_val = 0
        elif cnt_match == 0 and sum_product != 0:
            print('cnt_match is zero (problemo)!')
        else:
            expectation_val = sum_product / cnt_match
        return expectation_val

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

    def local_observable(self, obs) -> float:
        # converting the observable with shape e.g. [[Z, 0], [Z,1]] to right shape 'Z0 Z1'
        observable: str = ''
        for obis in obs:
            observable = observable + obis[0] + str(obis[1]) + ' '
        operator_sparse = opf_lin.get_sparse_operator(QubitOperator(observable), n_qubits=self.qubit_num)
        exp_val = pt.conj(self.psi) @ pt.tensor(operator_sparse.dot(self.psi))
        return exp_val

    def local_observable_shadow(self, obs, measurements):
        sum_product, cnt_match = estimate_exp(measurements, obs)
        if sum_product == 0 and cnt_match == 0:
            expectation_val = 0
        elif cnt_match == 0 and sum_product != 0:
            print('cnt_match is zero (problemo)!')
        else:
            expectation_val = sum_product / cnt_match
        return expectation_val

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

    # BFQuantumState(12, None).measurement_shadow(1, 'derandomized', [[['Z', 0], ['Z', 3]]])
    # print(BFQuantumState(2, 1 / pt.sqrt(pt.tensor([2])) * pt.tensor([0, 1, 1, 0], dtype=constants.DEFAULT_COMPLEX_TYPE)).measure_pauli({0: 'Z', 1: 'Z'}, 1))
    psi = pt.rand(2 ** 6, dtype=constants.DEFAULT_COMPLEX_TYPE)
    psi = psi / pt.sqrt((pt.dot(pt.conj(psi), psi)))
    z_arr = [['Z', 0], ['Z', 4]]
    print(BFQuantumState(6, psi).local_observable(z_arr))
    print(BFQuantumState(6, psi).two_point_correlation(4, 'Z'))



if __name__ == '__main__':
    main()
