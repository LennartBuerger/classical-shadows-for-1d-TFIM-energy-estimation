from abc import ABC, abstractmethod
import torch as pt


class AbstractHamiltonian(ABC):
    # we pass the number of qubits N
    def __init__(self, qubit_num):
        self.qubit_num = qubit_num

    @abstractmethod
    def energy_estimation(self, method: str) -> pt.float:
        pass

    @abstractmethod
    def diagonalize(self) -> (pt.float, pt.tensor):
        pass
