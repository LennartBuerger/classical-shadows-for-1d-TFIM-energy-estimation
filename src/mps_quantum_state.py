import sys
from pathlib import Path

import torch as pt
import math

from src.abstract_quantum_state import AbstractQuantumState
from src import constants

from src.mps import MPS
from display_data.data_acquisition_shadow import randomized_classical_shadow, derandomized_classical_shadow


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
        if flipped_idx.size()[0] == 1:
            flipped_idx = pt.cat(
                (flipped_idx, pt.tensor([0])))  # code fails when only one index is passed to amplitude function
            amplitudes = self.mps.amplitude(flipped_idx)
            return pt.tensor([amplitudes[0].item()])
        else:
            return self.mps.amplitude(flipped_idx)

    # probability of measuring our quantum state in a certain basis vector state
    def prob(self, basis_idx: pt.Tensor) -> pt.Tensor:
        amps = self.amplitude(basis_idx)
        return (pt.conj(amps) * amps).real

    def norm(self):
        return self.mps.norm()

    # measures state in computational basis
    # we have to pass a normalized and canonicalised mps
    def measure_old(self, batch_size: int = 1):
        vis_sampled = pt.zeros(self.qubit_num)
        probs_vis = pt.ones(self.qubit_num)
        # self.mps.canonicalise(self.qubit_num - 1) we already call this in the measurement shadow function
        # part_func = self.mps.norm().real # we pass a normalised mps
        for idx_rev in range(self.qubit_num - 1, -1, -1):
            if idx_rev == self.qubit_num - 1:
                result = self.mps.tensors[idx_rev]
            prob_result = pt.einsum('iaj,ibj->ab', result, self.mps.tensors[idx_rev].conj())
            probs_prev_vis = pt.prod(probs_vis, dim=0)
            probs = [pt.abs(prob_result[0, 0]) / probs_prev_vis,
                     pt.abs(prob_result[1, 1]) / probs_prev_vis]
            vis_sampled[idx_rev] = pt.multinomial(pt.tensor([probs[0].item(), probs[1].item()]), 1, replacement=True)[
                0].item()
            probs_vis[idx_rev] = probs[int(vis_sampled[idx_rev].item())]
            if idx_rev == 0:
                continue
            # Top -> Bottom
            result = pt.einsum('ij,kj->ik', result[:, int(vis_sampled[idx_rev].item()), :],
                               self.mps.tensors[idx_rev][:, int(vis_sampled[idx_rev].item()), :].conj())
            # Left -> Right
            result = pt.einsum('ik,jai->jak', result, self.mps.tensors[idx_rev - 1])
        measurement_idx = 0
        for k in range(0, self.qubit_num):
            measurement_idx = measurement_idx + int(vis_sampled[k].item()) * (2 ** (self.qubit_num - 1 - k))
        return measurement_idx, pt.prod(probs_vis)

    def measure(self, batch_size, canonicalise=False):
        if canonicalise:
            self.mps.canonicalise(self.qubit_num - 1)
        num_samples_tensor = pt.tensor([batch_size])
        sampled_visibles = [pt.tensor([])]
        probs_sampled = pt.tensor([1])
        contraction_results = []
        for idx_rev in range(self.qubit_num - 1, -1, -1):
            contr_res_intermed = []
            for hist in range(len(sampled_visibles)):
                if idx_rev == self.qubit_num - 1:
                    result = self.mps.tensors[idx_rev]
                else:
                    result = contraction_results[hist]
                    # Top -> Bottom
                    result = pt.einsum('ij,kj->ik', result, self.mps.tensors[idx_rev + 1][:, int(
                        sampled_visibles[hist][(self.qubit_num - 1) - (idx_rev + 1)]), :].conj())
                    # Left -> Right
                    result = pt.einsum('ik,jai->jak', result, self.mps.tensors[idx_rev])
                contr_res_intermed.append(result)
                prob_result = pt.einsum('iaj,ibj->ab', result, self.mps.tensors[idx_rev].conj())
                if hist == 0:
                    result_zero = pt.abs(pt.tensor([prob_result[0, 0].item()]))
                    result_one = pt.abs(pt.tensor([prob_result[1, 1].item()]))
                else:
                    result_zero = pt.cat((result_zero, pt.abs(pt.tensor([prob_result[0, 0].item()]))))
                    result_one = pt.cat((result_one, pt.abs(pt.tensor([prob_result[1, 1].item()]))))

            probs_in_zero = result_zero / probs_sampled
            try:
                distrib = pt.distributions.binomial.Binomial(num_samples_tensor, probs_in_zero)
            except ValueError:
                # sometimes the probabilities become a tiny bit bigger than one due to rounding errors
                for j in range(int(probs_in_zero.size()[0])):
                    if probs_in_zero[j] >= 1:
                        probs_in_zero[j] = 1
                distrib = pt.distributions.binomial.Binomial(num_samples_tensor, probs_in_zero)
            num_zeros = distrib.sample()
            num_ones = num_samples_tensor - num_zeros
            num_samples_tensor = pt.cat((num_zeros, num_ones))
            sampled_visible_update = []
            contraction_results = []
            for j in range(0, len(sampled_visibles)):
                if num_samples_tensor[j] > 0:
                    sampled_visible_update.append(pt.cat((sampled_visibles[j], pt.tensor([0]))))
                    contraction_results.append(contr_res_intermed[j][:, 0, :])
            for j in range(0, len(sampled_visibles)):
                if num_samples_tensor[j + len(sampled_visibles)] > 0:
                    sampled_visible_update.append(pt.cat((sampled_visibles[j], pt.tensor([1]))))
                    contraction_results.append(contr_res_intermed[j][:, 1, :])
            sampled_visibles = sampled_visible_update
            probs_sampled = pt.cat((result_zero, result_one))
            non_zeros = pt.nonzero(num_samples_tensor, as_tuple=True)
            num_samples_tensor = num_samples_tensor[non_zeros]
            probs_sampled = probs_sampled[non_zeros]

        expon = pt.tensor([2]) ** pt.arange(0, self.qubit_num, 1)
        sampled_indices = []
        for sampled_visible in sampled_visibles:
            sampled_indices.append(int(pt.sum(sampled_visible * expon, dim=0).item()))
        return sampled_indices, probs_sampled

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state
    def rotate_pauli(self, pauli_string: dict):
        rot_tensors = []
        for idx in range(self.qubit_num):
            rot_tensors.append(pt.einsum('ab,cbd->cad', constants.PAULI_ROT[pauli_string[idx]], self.mps.tensors[idx]))
        return MPSQuantumState(self.qubit_num, MPS.from_tensor_list(rot_tensors))

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int):
        return self.rotate_pauli(pauli_string).measure(batch_size)

    def measurement_shadow(self, meas_num: int, meas_per_basis: int, meas_method: str, observables):
        meas_results = []
        probs = []
        if meas_method == 'derandomized':
            batch_size = 1  # for simple hamiltonians we can simply measure each observable once, but for more
            # complicated
            meas_bases = []
            measurement_procedure_batched = derandomized_classical_shadow(observables, batch_size, self.qubit_num)
            num_pauli_strings_in_one_batch = len(measurement_procedure_batched)
            number_of_measurements_for_one_pauli_string = math.ceil(meas_num / num_pauli_strings_in_one_batch)
            for i in range(0, number_of_measurements_for_one_pauli_string):
                for j in range(0, num_pauli_strings_in_one_batch):
                    meas_bases.append(measurement_procedure_batched[j])
        if meas_method == 'randomized':
            meas_bases = randomized_classical_shadow(meas_num, self.qubit_num)
        for i in range(len(meas_bases)):
            mps_rotated = self.rotate_pauli(meas_bases[i])
            mps_rotated.mps.canonicalise(self.qubit_num - 1)
            mps_rotated.mps.normalise()
            meas_res_basis, prob_basis = mps_rotated.measure(meas_per_basis)
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
