from abc import ABC, abstractmethod
import torch as pt


class AbstractQuantumState(ABC):
    # we pass the number of qubits N
    def __init__(self, qubit_num):
        self.qubit_num = qubit_num

    # measuring amplitude with respect to some basis vector
    @abstractmethod
    def amplitude(self, basis_idx: pt.Tensor) -> pt.tensor:
        pass

    # probability of measuring our quantum state in a certain basis vector state
    @abstractmethod
    def prob(self, basis_idx: pt.Tensor) -> float:
        pass

    @abstractmethod
    def norm(self) -> float:
        pass

    # measures state in computational basis, B is number of samples we take
    @abstractmethod
    def measure(self, batch_size: int) -> dict:
        pass

    # takes a pauli string and rotates to the basis given by this string, returns a new instance of our quantum state
    @abstractmethod
    def rotate_pauli(self, pauli_string: dict):
        pass

    # here we rotate first and then do a measurement in the computational basis
    @abstractmethod
    def measure_pauli(self, pauli_string: dict, batch_size: int) -> dict:
        pass

    # apply a string of single qubit clifford gates
    @abstractmethod
    def apply_clifford(self, clifford_string: dict):
        pass

    @abstractmethod
    def entanglement_entropy(self):
        pass

    @abstractmethod
    def two_point_correlation(self, dist: int, basis: str) -> float:
        pass
