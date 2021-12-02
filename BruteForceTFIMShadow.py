import torch as pt
import AbstractQuantumState
import math


class BFQuantumState(AbstractQuantumState.AbstractQuantumStates):

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, nqb, psi):
        self.nqb = nqb
        if psi == None:
            Psi_real = pt.rand(nq ** 2, dtype= pt.cfloat)
            Psi_img = pt.rand(nq ** 2, dtype = pt.cfloat)
            self.psi_not_normed = Psi_real + Psi_img * 1j
            self.psi = self.psi_not_normed / pt.sqrt(self.psi_not_normed @ pt.conj(self.psi_not_normed))
        else:
            self.psi = psi

    # measuring amplitude with respect to some basis vector

    def amplitude(self, basis_idx: pt.Tensor) -> pt.tensor:
        amplitude_val = pt.empty(basis_idx.size(dim=0), dtype=pt.cfloat)
        for i in range(0, basis_idx.size(dim=0)):
            amplitude_val[i] = self.psi[basis_idx[i]]
        return amplitude_val

    # probability of measuring our quantum state in a certain basis vector state

    def prob(self, basis_idx: pt.Tensor) -> float:
        prob_val = pt.empty(basis_idx.size(dim=0))
        for i in range(0, basis_idx.size(dim=0)):
            prob_val[i] = self.psi[basis_idx[i]].abs() ** 2
        return prob_val

    @property
    def norm(self) -> float:
        norm_val: float = pt.sqrt(self.psi @ pt.conj(self.psi))
        return norm_val

    # measures state in computational basis, B is number of samples we take

    def measure(self, batch_size: int) -> dict:
        # to perform the measurement we first generate a multinomial probability distribution from our known state
        # from which we sample afterwards
        distribution = pt.empty(self.nqb ** 2, dtype=float)
        distribution: float = self.psi * pt.conj(self.psi)
        sampled_basisvec = pt.multinomial(distribution.real, batch_size, replacement=True)
        sample_probs = dict([])
        for i in range(0, batch_size):
            if int(sampled_basisvec[i]) in sample_probs:
                sample_probs[int(sampled_basisvec[i])] = sample_probs[int(sampled_basisvec[i])] + 1
            else:
                sample_probs[int(sampled_basisvec[i])] = 1
        sample_probs_norm = {key: sample_probs[key] / batch_size for key in sample_probs.keys()}
        return sample_probs_norm

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state

    def rotate_pauli(self, pauli_string: dict):
        x_rot = 1 / math.sqrt(2) * pt.tensor([[1, 1], [1, -1]], dtype=pt.cfloat)
        identity = pt.tensor([[1, 0], [0, 1]], dtype=pt.cfloat)
        z_rot = identity
        y_rot = 1 / math.sqrt(2) * pt.tensor([[1, 1j], [1, -1j]], dtype=pt.cfloat)
        matrix_rot = pt.tensor([1], dtype=pt.cfloat)
        for i in range(1, self.nqb + 1):
            if i in pauli_string:
                matrix_append = identity
                if pauli_string[i] == 'X':
                    matrix_append = x_rot
                if pauli_string[i] == 'Y':
                    matrix_append = y_rot
                if pauli_string[i] == 'Z':
                    matrix_append == z_rot
                matrix_rot = pt.kron(matrix_rot, matrix_append)
            else:
                matrix_rot = pt.kron(matrix_rot, identity)
        psi_rot = matrix_rot @ self.psi
        return psi_rot



    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict, batch_size: int) -> dict:
        psi_rot = self.rotate_pauli(pauli_string)
        sample_probs_norm = self.measure(batch_size)
        return sample_probs_norm


    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        x_pauli = pt.tensor([[0, 1], [1, 0]], dtype=pt.cfloat)
        y_pauli = pt.tensor([[0, 1j], [-1j, 0]], dtype=pt.cfloat)
        z_pauli = pt.tensor([[1, 0], [0, -1]], dtype=pt.cfloat)
        identity = pt.tensor([[1, 0], [0, 1]], dtype=pt.cfloat)
        hadamard = 1/ math.sqrt(2) * pt.tensor([[1, 1], [1, -1]], dtype=pt.cfloat)
        s_gate = pt.tensor([[1, 0], [0, 1j]], dtype=pt.cfloat)

        matrix_rot = pt.tensor([1], dtype=pt.cfloat)

        for i in range(1, self.nqb + 1):
            if i in clifford_string:
                matrix_append = identity
                if clifford_string[i] == 'X':
                    matrix_append = x_pauli
                if clifford_string[i] == 'Y':
                    matrix_append = y_pauli
                if clifford_string[i] == 'Z':
                    matrix_append = z_pauli
                if clifford_string[i] == 'H':
                    matrix_append = hadamard
                if clifford_string[i] == 'S':
                    matrix_append = s_gate
                matrix_rot = pt.kron(matrix_rot, matrix_append)
            else:
                matrix_rot = pt.kron(matrix_rot, identity)
        psi_clifford = matrix_rot @ self.psi
        return psi_clifford


# down here comes testing rubbish which can be removed later

# specify number of qubits
nq = 2

# first generate a random quantum state Psi which we normalize
Psi_real = pt.rand(nq ** 2)
Psi_img = pt.rand(nq ** 2)
psi = Psi_real + Psi_img * 1j             #1/math.sqrt(2) * pt.tensor([1, 1j], dtype=pt.cfloat)          #
norm = BFQuantumState(nq, psi).norm
psi = psi / norm
#basisbla = pt.tensor([0, 1, 2, 3])
print(psi)
#print(BFQuantumState(nq, psi).prob(basisbla))
#print(BFQuantumState(nq, psi).amplitude(basisbla))
print(BFQuantumState(nq,None).norm)
print(BFQuantumState(nq, None).apply_clifford({1: 'X', 2: 'H'}))
print(BFQuantumState(nq,psi).measure_pauli({1: 'X', 2: 'Z'}, 10))
