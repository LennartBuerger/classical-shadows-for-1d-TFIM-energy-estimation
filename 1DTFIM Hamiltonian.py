import torch as pt
import AbstractHamiltonian


class TFIMHamiltonian1d(AbstractHamiltonian.AbstractHamiltonian):
    # we pass the number of qubits N
    def __init__(self, qubit_num, psi, h_val, j_val):
        super(TFIMHamiltonian1d, self).__init__(qubit_num)
        self.h_val = h_val
        self.J_val = j_val
        self.psi = psi

    def energy_estimation(self, method: str) -> pt.float:
        # brute force option
        if method == 'B':
            hamiltonian = self.to_matrix()
            energy = pt.conj(self.psi) @ hamiltonian @ self.psi
            return energy

    def diagonalize(self) -> (pt.float, pt.tensor):
        hamiltonian = self.to_matrix()
        eigenvalues, eigenvectors = pt.linalg.eig(hamiltonian)
        return eigenvalues, eigenvectors

    def to_matrix(self) -> pt.tensor:
        hamil = pt.zeros((2 ** self.qubit_num, 2 ** self.qubit_num), dtype=pt.cfloat)
        x_pauli = pt.tensor([[0, 1], [1, 0]], dtype=pt.cfloat)
        z_pauli = pt.tensor([[1, 0], [0, -1]], dtype=pt.cfloat)
        identity = pt.tensor([[1, 0], [0, 1]], dtype=pt.cfloat)
        for i in range(0, self.qubit_num):
            hamil_iz = pt.tensor([1], dtype=pt.cfloat)
            hamil_ix = pt.tensor([1], dtype=pt.cfloat)
            for j in range(0, self.qubit_num):
                if j == i or j == i + 1:
                    hamil_iz = pt.kron(hamil_iz, z_pauli)
                else:
                    hamil_iz = pt.kron(hamil_iz, identity)
                if j == i:
                    hamil_ix = pt.kron(hamil_ix, x_pauli)
                else:
                    hamil_ix = pt.kron(hamil_ix, identity)
                if i == self.qubit_num - 1:
                    hamil_iz = pt.zeros((2 ** self.qubit_num, 2 ** self.qubit_num), dtype=pt.cfloat)
            hamil = hamil + self.J_val * hamil_iz + self.h_val * hamil_ix
        return hamil


def main():
    print(TFIMHamiltonian1d(2, None, 1, 1).to_matrix())
    print(TFIMHamiltonian1d(2, pt.tensor([-3.7175e-01 + 0.j, 6.0150e-01 + 0.j, 6.0150e-01 + 0.j, -3.7175e-01 + 0.j]), 1,
                            1).energy_estimation('B'))
    print(TFIMHamiltonian1d(2, None, 1, 1).diagonalize())


if __name__ == '__main__':
    main()
