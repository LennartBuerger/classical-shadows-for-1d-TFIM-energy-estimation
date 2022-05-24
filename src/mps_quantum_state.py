import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as pt
from src.abstract_quantum_state import AbstractQuantumState
from src import constants
import numpy as np
sys.path.append(Path('../deterministic'))
from deterministic.mps import MPS
from display_data.data_acquisition_shadow import derandomized_classical_shadow, randomized_classical_shadow
from display_data.prediction_shadow import estimate_exp


class MPSQuantumState(AbstractQuantumState):

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, qubit_num, tensor_list):
        super(MPSQuantumState, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.tensor_list = tensor_list

    # measuring amplitude with respect to some basis vector

    def amplitude(self, basis_idx: pt.Tensor) -> pt.tensor:
        pass

    # probability of measuring our quantum state in a certain basis vector state

    def prob(self, basis_idx: pt.Tensor) -> float:
        pass

    def norm(self):
        result = pt.einsum('abc,abj->cj', self.tensor_list[0], self.tensor_list[0].conj())
        for idx in range(1, self.qubit_num - 1):
            result = pt.einsum('cj,cab->jab', result, self.tensor_list[idx])
            result = pt.einsum('jab,jad->bd', result, self.tensor_list[idx].conj())
        result = pt.einsum('ab,acj->bcj', result, self.tensor_list[self.qubit_num - 1])
        result = pt.einsum('bcj,bci->ji', result, self.tensor_list[self.qubit_num - 1].conj())
        return pt.sqrt(result)

    def canonicalise_left_to_index(self, idx, phys_dim):
        # from the left
        for index in range(0, idx):
            bond_dim_left = self.tensor_list[index][:, 0, 0].size()[0]
            bond_dim_right = self.tensor_list[index][0, 0, :].size()[0]
            Qm, R = pt.linalg.qr(self.tensor_list[index].reshape(bond_dim_left * phys_dim, bond_dim_right))
            self.tensor_list[index] = pt.reshape(Qm, (bond_dim_left, phys_dim, Qm.size()[1]))
            self.tensor_list[index + 1] = pt.einsum('ab,bcd->acd', R, self.tensor_list[index + 1])

    # measures state in computational basis
    def measure(self, num_samples: int):
        bits_sampled = pt.zeros((self.qubit_num, num_samples))
        probabilities_for_bits = pt.ones((self.qubit_num, num_samples))
        self.canonicalise_left_to_index(self.qubit_num - 1, 2)
        # we only need to do this step if the MPS is not normalised
        part_func = pt.einsum('ijk,ijl->kl', self.tensor_list[self.qubit_num - 1], self.tensor_list[self.qubit_num - 1].conj())[
            0, 0]

        for index in range(self.qubit_num):
            idx = self.qubit_num - 1 - index
            # because the probabilities for all samples is different, we cannot draw them all at once
            # but have to draw them one by one by looping
            for k in range(num_samples):
                # contract the network
                if idx == self.qubit_num - 1:
                    result = pt.einsum('ijl,iml->jm', self.tensor_list[idx], self.tensor_list[idx].conj())
                else:
                    result = pt.einsum('fh,jh->fj', self.tensor_list[self.qubit_num - 1][:,
                                                    int(bits_sampled[self.qubit_num - 1, k].item()), :],
                                       self.tensor_list[self.qubit_num - 1][:,
                                       int(bits_sampled[self.qubit_num - 1, k].item()), :].conj())
                    for counter in range(self.qubit_num - 1 - idx - 1):
                        index = self.qubit_num - 1 - counter - 1
                        result = pt.einsum('fj,df->dj', result,
                                           self.tensor_list[index][:, int(bits_sampled[index, k].item()), :])
                        result = pt.einsum('dj,lj->dl', result,
                                           self.tensor_list[index][:, int(bits_sampled[index, k].item()),
                                           :].conj())
                    result = pt.einsum('rs,acr->acs', result, self.tensor_list[idx])
                    result = pt.einsum('acs,ams->cm', result, self.tensor_list[idx].conj())
                # contraction done
                prob_for_previous_bits = pt.prod(probabilities_for_bits[:, k])
                probs = [abs(result[0, 0]) / part_func / prob_for_previous_bits,
                         abs(result[1, 1]) / part_func / prob_for_previous_bits]
                bits_sampled[idx, k] = pt.multinomial(pt.tensor([probs[0].real.item(), probs[1].real.item()]), 1,
                                                      replacement=True)[0].item()
                probabilities_for_bits[idx, k] = probs[int(bits_sampled[idx, k].item())]
        return bits_sampled, probabilities_for_bits

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state

    def rotate_pauli(self, pauli_string: dict):
        rot_tensors = []
        for idx in range(len(self.tensor_list)):
            rot_tensors.append(pt.einsum('ab,cbd->cad', constants.PAULI_ROT[pauli_string[idx]], self.tensor_list[idx]))
        return MPSQuantumState(self.qubit_num, rot_tensors)

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int):
        return self.rotate_pauli(pauli_string).measure(batch_size)

    def measurement_shadow(self, number_of_measurements, num_measurements_per_rot):
        measurement_outcomes = []
        probabilities = []
        measurement_procedure = randomized_classical_shadow(number_of_measurements, self.qubit_num)
        for i in range(number_of_measurements):
            mps_rotated = self.rotate_pauli(measurement_procedure[i])
            measurement_bits, probs = mps_rotated.measure(num_measurements_per_rot)
            probabilities.append(pt.prod(probs, dim=0))
            # convert binary torch tensor to index
            measurement_outcome = pt.zeros(num_measurements_per_rot, dtype=pt.int)
            for k in range(0, self.qubit_num):
                measurement_outcome[:] = measurement_outcome + measurement_bits[k] * 2 ** (self.qubit_num - 1 - k)
            measurement_outcomes.append(measurement_outcome)
        return measurement_outcomes, measurement_procedure, probabilities

    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        pass

    def entanglement_entropy(self):
        pass

    def two_point_correlation(self, dist: int, basis: str) -> float:
        pass

