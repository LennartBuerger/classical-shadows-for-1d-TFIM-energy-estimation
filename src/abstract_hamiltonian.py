from abc import ABC, abstractmethod
import torch as pt


class AbstractHamiltonian(ABC):
    # we pass the number of qubits N
    def __init__(self, qubit_num):
        self.qubit_num = qubit_num

    @abstractmethod
    def energy(self, method: str, psi: pt.tensor) -> pt.float:
        pass

    @abstractmethod
    def diagonalize(self, nr_eig_vals: int, nr_eig_vecs: int) -> (pt.float, pt.tensor):
        pass
