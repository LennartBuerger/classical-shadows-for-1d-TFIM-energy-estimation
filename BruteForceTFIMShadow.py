import torch as pt
import AbstractQuantumState


class BFQuantumState(AbstractQuantumState.AbstractQuantumStates):

    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, nqb, psi):
        self.nqb = nqb
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
            prob_val[i] = self.psi[basis_idx[i]].abs()**2
        return prob_val

    @property
    def norm(self) -> float:
        norm_val: float = pt.sqrt(self.psi @ pt.conj(self.psi))
        return norm_val

    # measures state in computational basis, B is number of samples we take

    def measure(self, batch_size: int) -> dict:
        # to perform the measurement we first generate a multinomial probability distribution from our known state
        # from which we sample afterwards
        distribution = pt.empty(self.nqb**2, dtype=float)
        distribution: float = self.psi * pt.conj(self.psi)
        sampled_basisvec = pt.multinomial(distribution.real, batch_size, replacement=True)
        return sampled_basisvec

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state

    def rotate_pauli(self, pauli_string: dict):
        pass

    # here we rotate first and then do a measurement in the computational basis
    def measure_pauli(self, pauli_string: dict) -> dict:
        pass

    # apply a string of single qubit clifford gates
    def apply_clifford(self, clifford_string: dict):
        pass


# specify number of qubits
nq = 2

# first generate a random quantum state Psi which we normalize
Psi_real = pt.rand(nq ** 2)
Psi_img = pt.rand(nq ** 2)
psi = Psi_real + Psi_img * 1j
norm = BFQuantumState(nq, psi).norm
psi = psi / norm
basisbla = pt.tensor([0,1,2,3])
print(psi)
print(BFQuantumState(nq,psi).prob(basisbla))
print(BFQuantumState(nq,psi).amplitude(basisbla))
print(BFQuantumState(nq,psi).measure(10))

