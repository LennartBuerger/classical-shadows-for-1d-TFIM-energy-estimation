import sys
from pathlib import Path

import torch as pt

from src.abstract_quantum_state import AbstractQuantumState
from src import constants

from src.mps import MPS
from display_data.data_acquisition_shadow import randomized_classical_shadow


class MPSQuantumState(AbstractQuantumState):

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, qubit_num, mps: MPS = None):
        super(MPSQuantumState, self).__init__(qubit_num)
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.mps = mps

    # measuring amplitude with respect to some basis vector
    def amplitude(self, basis_idx: pt.Tensor) -> pt.Tensor:
        # different index to binary conventions are used, which is why we have to flip first
        mask = 2 ** pt.arange(self.qubit_num).to(basis_idx.device, basis_idx.dtype)
        bin = basis_idx.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        bin_flip = pt.flip(bin, dims=[1])
        powers = 2 ** pt.arange(0, self.qubit_num, dtype=pt.int)
        bin_flip = bin_flip.to(pt.int)
        flipped_idx = pt.einsum('ba,a->b', bin_flip, powers)
        return self.mps.amplitude(flipped_idx)

    # probability of measuring our quantum state in a certain basis vector state
    def prob(self, basis_idx: pt.Tensor) -> pt.Tensor:
        amps = self.amplitude(basis_idx)
        return (pt.conj(amps) * amps).real

    def norm(self):
        return self.mps.norm()

    # measures state in computational basis
    def measure(self, batch_size: int = 1):
        vis_sampled = pt.zeros(self.qubit_num)
        probs_vis = pt.ones(self.qubit_num)
        # self.mps.canonicalise(self.qubit_num - 1) we already call this in the measurement shadow function
        part_func = self.mps.norm().real
        for idx_rev in range(self.qubit_num - 1, -1, -1):
            if idx_rev == self.qubit_num - 1:
                result = self.mps.tensors[idx_rev]
            prob_result = pt.einsum('iaj,ibj->ab', result, self.mps.tensors[idx_rev].conj())
            probs_prev_vis = pt.prod(probs_vis, dim=0)
            probs = [(pt.abs(prob_result[0, 0]) / part_func) / probs_prev_vis,
                     (pt.abs(prob_result[1, 1]) / part_func) / probs_prev_vis]
            vis_sampled[idx_rev] = pt.multinomial(pt.tensor([probs[0].item(), probs[1].item()]), 1, replacement=True)[
                0].item()
            probs_vis[idx_rev] = probs[int(vis_sampled[idx_rev].item())]
            if idx_rev == 0:
                continue
            # Left -> Right
            result = pt.einsum('ij,kj->ik', result[:, int(vis_sampled[idx_rev].item()), :],
                               self.mps.tensors[idx_rev][:, int(vis_sampled[idx_rev].item()), :].conj())
            # Top -> Bottom
            result = pt.einsum('ik,jai->jak', result, self.mps.tensors[idx_rev - 1])
        measurement_idx = 0
        for k in range(0, self.qubit_num):
            measurement_idx = measurement_idx + int(vis_sampled[k].item()) * (2 ** (self.qubit_num - 1 - k))
        return measurement_idx, pt.prod(probs_vis)

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
            mps_rotated.mps.canonicalise(self.qubit_num - 1)
            meas_res_basis = pt.zeros(meas_per_basis, dtype=pt.int)
            prob_basis = pt.zeros(meas_per_basis)
            for j in range(meas_per_basis):
                meas_res_basis[j], prob_basis[j] = mps_rotated.measure()
            meas_results.append(meas_res_basis)
            probs.append(prob_basis)
        return meas_results, meas_bases, probs

    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        pass

    def entanglement_entropy(self):
        pass

    def two_point_correlation(self, dist: int, basis: str) -> float:
        pass
