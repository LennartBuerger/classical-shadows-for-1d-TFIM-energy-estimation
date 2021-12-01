from abc import ABC, abstractmethod
import torch as pt


class AbstractQuantumStates(ABC):
    @abstractmethod
    # we pass the number of qubits N and our Quantum State Psi
    def __init__(self, N, Psi):
        self.N = N
        self.Psi = Psi

    # measuring amplitude with respect to some basis vector
    @abstractmethod
    def amplitude(self, basisvec: pt.Tensor) -> float:
        pass

    # probability of measuring our quantum state in a certain basis vector state
    @abstractmethod
    def probability(self, basisvec: pt.Tensor) -> float:
        pass

    @abstractmethod
    def norm(self) -> float:
        pass

    # measures state in computational basis, B is number of samples we take
    @abstractmethod
    def measure(self, Batch_size: int) -> dict:
        pass

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state
    @abstractmethod
    def rotate_pauli(self, pauli_string: dict):
        pass

    # here we rotate first and then do a measurement in the computational basis
    @abstractmethod
    def measure_pauli(self, pauli_string: dict) -> dict:
        pass

    # apply a string of single qubit clifford gates
    @abstractmethod
    def apply_clifford(self, clifford_string: dict):
        pass
