from abc import ABC
import AbstractHamiltonian
import torch as pt
from openfermion.ops import QubitOperator
import openfermion.linalg as opf_lin
import scipy.sparse.linalg
import cirq.linalg
import numpy as np
import matplotlib.pyplot as plt


class tfim_hamiltonian_openfermion(AbstractHamiltonian.AbstractHamiltonian, ABC):
    # we pass the number of qubits N
    def __init__(self, qubit_num, psi, h_val, j_val):
        super(tfim_hamiltonian_openfermion, self).__init__(qubit_num)
        self.h_val: float = h_val
        self.J_val: float = j_val
        self.psi = psi

    def energy_estimation(self, method: str) -> pt.float:
        # brute force option
        if method == 'B':
            hamiltonian = self.ham_to_matrix()
            energy = pt.conj(self.psi) @ pt.tensor(hamiltonian.dot(self.psi), dtype=pt.cfloat)
            return energy

    def diagonalize(self) -> (pt.float, pt.tensor):
        hamiltonian = self.ham_to_matrix()
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamiltonian, which='SA', k=2)
        return eigenvalues, eigenvectors

    # energy gap between E_0 and E_1
    def energygap(self) -> pt.float:
        eigenvalues, eigenvectors = self.diagonalize()
        energy_gap = eigenvalues[1] - eigenvalues[0]
        return energy_gap

    def ground_state_wave(self) -> pt.tensor:
        eigenvalues, eigenvectors = self.diagonalize()
        eigenvector_ground = pt.tensor(eigenvectors[:, 0])
        return eigenvector_ground

    def entanglement_entropy(self):
        ground_state = self.ground_state_wave()
        # procedure for computing the entropy: get Schmidt decomposition of Psi, then entropy is determined by schmidt coefficients
        # first write Psi into matrix which is tensor product of state in system A and in system B
        m_psi = pt.empty((int(np.sqrt(2**self.qubit_num)), int(np.sqrt(2**self.qubit_num))), dtype=pt.cfloat)
        for i in range(0, int(np.sqrt(2**self.qubit_num))):
            for j in range(0, int(np.sqrt(2**self.qubit_num))):
                m_psi[i, j] = ground_state[i * int(np.sqrt(2**self.qubit_num)) + j]
        singular_values = pt.linalg.svdvals(m_psi)
        entropy: float = float( - pt.sum(singular_values**2*pt.log(singular_values**2)))
        return entropy

    # two point correlation of ground state for different distances and different ratios h/J
    def two_point_correlation(self, dist: int) -> float:
        ground_state = self.ground_state_wave()
        z_string = 'Z0' + ' Z' + str(dist)
        corr_operator = QubitOperator(z_string)
        corr_operator_sparse = opf_lin.get_sparse_operator(corr_operator, n_qubits=self.qubit_num)
        correlation: float = np.abs(float(pt.conj(ground_state) @ pt.tensor(corr_operator_sparse.dot(ground_state))))
        return correlation

    def ham_to_matrix(self):
        hamil = QubitOperator(None)
        for i in range(0, self.qubit_num):
            x_string = 'X' + str(i)
            if i <= self.qubit_num - 2:
                z_string = 'Z' + str(i) + ' Z' + str(i + 1)
            else:
                z_string = None
            hamil = hamil + QubitOperator(z_string, coefficient=self.J_val) + QubitOperator(x_string,
                                                                                            coefficient=self.h_val)
        hamil_sparse = opf_lin.get_sparse_operator(hamil, n_qubits=self.qubit_num)
        return hamil_sparse


# calculate energy gap E_1 - E_0  for different ratios h/J and different qubit numbers
def calculate_energy_gap(qubit_numbers: np.array):
    ratios_h_j = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], dtype=float)
    energy_gaps = pt.zeros(13)
    for j in range(0, np.size(qubit_numbers)):
        for i in range(0, 13):
            energy_gaps[i] = tfim_hamiltonian_openfermion(qubit_numbers[j], None, ratios_h_j[i] / 2, 2).energygap()
        save_direc: str = 'data\energygap\e_gap_qubit_num_' + str(qubit_numbers[j])
        np.savetxt(save_direc, energy_gaps)


def plot_energy_gap(qubit_numbers: np.array):
    ratios_h_j = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], dtype=float)
    for i in range(0, np.size(qubit_numbers)):
        save_direc: str = 'data\energygap\e_gap_qubit_num_' + str(qubit_numbers[i])
        energy_gaps = np.loadtxt(save_direc)
        plt.scatter(ratios_h_j[:-5], energy_gaps[:-5], label='qubit number ' + str(qubit_numbers[i]))
        plt.xscale('log')
        plt.title('energy gaps for different ratios h/J')
        plt.xlabel('ratios h/J (J=1)')
        plt.ylabel('energy gap in units of J')
        plt.legend()
    plot_save_direc = 'plots\shadow_project_plots\energy gaps for different ratios up to ratio 2 for qubits ' + str(
        qubit_numbers)
    plt.savefig(plot_save_direc)


# calculate two point correlation for different distances and ratios h/J

def calc_two_point_correlation(ratios: np.array, qubit_num: int):
    distance = np.arange(0, qubit_num, 1)
    correlations = np.zeros(qubit_num)
    for j in range(0, np.size(ratios)):
        for i in range(0, qubit_num):
            correlations[i] = tfim_hamiltonian_openfermion(qubit_num, None, ratios[j], 1).two_point_correlation(
                distance[i])
        save_direc = 'data\Twopointcorrelation\correlation_for_ratio ' + str(ratios[j]).replace('.', ',') + ' and qubit number ' + str(
            qubit_num)
        np.savetxt(save_direc, correlations)

def plot_two_point_correlation(ratios: np.array, qubit_num: int):
    for i in range(0, np.size(ratios)):
        save_direc = 'data\Twopointcorrelation\correlation_for_ratio ' + str(ratios[i]).replace('.', ',') + ' and qubit number ' + str(qubit_num)
        correlations = np.loadtxt(save_direc)
        distance = np.arange(0, qubit_num, 1)
        plt.scatter(distance, correlations, label='ratio h/J = ' + str(ratios[i]))
        plt.title('two point correlation for different distances and ratios')
        plt.xlabel('distance, j-th neighbour')
        plt.ylabel('two_point_correlation')
        plt.legend()
    plot_save_direc = 'plots\Two_point_correlation_brute_force\correlation_for_ratio ' + str(ratios).replace('.', ',') + ' and qubit number ' + str(qubit_num)
    plt.savefig(plot_save_direc)


def main():
    qubit_num = 16

    psi_not_normed = pt.rand(2 ** qubit_num, dtype=pt.cfloat)
    psi = psi_not_normed / pt.sqrt(psi_not_normed @ pt.conj(psi_not_normed))

    ratios = np.array([0.2, 0.5, 1, 2, 5, 10])
    #calc_two_point_correlation(ratios, qubit_num)
    #plot_two_point_correlation(ratios, qubit_num)

    print(tfim_hamiltonian_openfermion(qubit_num, None
                                 , 1, 1).entanglement_entropy())

    qubit_numbers = np.array([18])
    # calculate_energy_gap(qubit_numbers)
    # plot_energy_gap(qubit_numbers)


if __name__ == '__main__':
    main()
