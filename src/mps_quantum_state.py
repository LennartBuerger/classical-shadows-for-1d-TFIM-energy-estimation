import sys
from pathlib import Path

import torch as pt

from src.abstract_quantum_state import AbstractQuantumState
from src import constants

sys.path.append(Path('../deterministic'))
from deterministic.mps import MPS
from display_data.data_acquisition_shadow import randomized_classical_shadow


class MPSQuantumState(AbstractQuantumState):

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, qubit_num, mps: MPS = None):
        super(MPSQuantumState, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.mps = mps

    # measuring amplitude with respect to some basis vector
    def amplitude(self, basis_idx: pt.Tensor) -> pt.Tensor:
        return self.mps.amplitude(basis_idx)

    # probability of measuring our quantum state in a certain basis vector state
    def prob(self, basis_idx: pt.Tensor) -> pt.Tensor:
        amps = self.amplitude(basis_idx)
        return (pt.conj(amps) * amps).real

    def norm(self):
        return self.mps.norm()

    # measures state in computational basis
    def measure(self, batch_size: int):
        sampled_visible = pt.zeros((self.qubit_num, batch_size))
        probabilities_for_bits = pt.ones((self.qubit_num, batch_size))
        self.mps.canonicalise(self.qubit_num - 1)
        # we only need to do this step if the MPS is not normalised
        part_func = self.mps.norm()

        for idx in range(self.qubit_num):
            rev_idx = self.qubit_num - 1 - idx
            # because the probabilities for all samples is different, we cannot draw them all at once
            # but have to draw them one by one by looping
            for k in range(batch_size):
                # contract the network
                if rev_idx == self.qubit_num - 1:
                    result = pt.einsum('ijl,iml->jm', self.tensor_list[rev_idx], self.tensor_list[rev_idx].conj())
                else:
                    result = pt.einsum('fh,jh->fj', self.tensor_list[self.qubit_num - 1][:,
                                                    int(sampled_visible[self.qubit_num - 1, k].item()), :],
                                       self.tensor_list[self.qubit_num - 1][:,
                                       int(sampled_visible[self.qubit_num - 1, k].item()), :].conj())
                    for counter in range(self.qubit_num - 1 - rev_idx - 1):
                        idx = self.qubit_num - 1 - counter - 1
                        result = pt.einsum('fj,df->dj', result,
                                           self.tensor_list[idx][:, int(sampled_visible[idx, k].item()), :])
                        result = pt.einsum('dj,lj->dl', result,
                                           self.tensor_list[idx][:, int(sampled_visible[idx, k].item()), :].conj())
                    result = pt.einsum('rs,acr->acs', result, self.tensor_list[rev_idx])
                    result = pt.einsum('acs,ams->cm', result, self.tensor_list[rev_idx].conj())
                # contraction done
                prob_for_previous_bits = pt.prod(probabilities_for_bits[:, k])
                probs = [pt.abs(result[0, 0]) / part_func / prob_for_previous_bits,
                         pt.abs(result[1, 1]) / part_func / prob_for_previous_bits]
                sampled_visible[rev_idx, k] = pt.multinomial(pt.tensor([probs[0].real.item(), probs[1].real.item()]), 1,
                                                             replacement=True)[0].item()
                probabilities_for_bits[rev_idx, k] = probs[int(sampled_visible[rev_idx, k].item())]
        return sampled_visible, probabilities_for_bits

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state

    def rotate_pauli(self, pauli_string: dict):
        rot_tensors = []
        for idx in range(self.qubit_num):
            rot_tensors.append(pt.einsum('ab,cbd->cad', constants.PAULI_ROT[pauli_string[idx]], self.mps.tensors[idx]))
        return MPSQuantumState(self.qubit_num, MPS.from_tensor_list(rot_tensors))

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int):
        return self.rotate_pauli(pauli_string).measure(batch_size)

    def measurement_shadow(self, meas_num, meas_per_basis):
        meas_results = []
        probs = []
        meas_bases = randomized_classical_shadow(meas_num, self.qubit_num)
        for i in range(meas_num):
            mps_rotated = self.rotate_pauli(meas_bases[i])
            meas_visible, probs = mps_rotated.measure(meas_per_basis)
            probs.append(pt.prod(probs, dim=0))
            meas_results.append(self.mps.visible_to_idx(meas_visible))

        return meas_results, meas_bases, probs

    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        pass

    def entanglement_entropy(self):
        pass

    def two_point_correlation(self, dist: int, basis: str) -> float:
        pass


