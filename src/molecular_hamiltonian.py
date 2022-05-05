import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as pt
import scipy.sparse.linalg
from src import constants
from display_data.prediction_shadow import estimate_exp
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator


class Molecular_Hamiltonian():
    ENERGY_METHODS = ('BF', 'BF_shadow')
    MOLECULES = ('LiH', 'NH3', 'H2O', 'H2')

    # we pass the molecule name
    def __init__(self, mol_name):
        self.dtype = constants.DEFAULT_COMPLEX_TYPE
        self.mol_name = mol_name

    def energy(self, method: str, psi: pt.tensor) -> pt.float:
        assert method in Molecular_Hamiltonian.ENERGY_METHODS
        # brute force option
        if method == 'BF':
            return self.energy_bf(psi)

    def energy_bf(self, psi: pt.tensor):
        return pt.conj(psi) @ pt.tensor(self.to_matrix().dot(psi), dtype=self.dtype)

    def diagonalize(self, nr_eigs: int, return_eig_vecs: bool) -> (pt.float, pt.tensor):
        if return_eig_vecs:
            return scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
        else:
            eigenvalues, _ = scipy.sparse.linalg.eigsh(self.to_matrix(), which='SA', k=nr_eigs)
            return eigenvalues

    def ground_state_energy(self) -> pt.double:
        return self.diagonalize(1, False)

    def ground_state_wavevector(self) -> pt.tensor:
        eigenvalues, eigenvector = self.diagonalize(1, True)
        return pt.tensor(eigenvector[:, 0])

    def to_matrix(self):
        assert self.mol_name in Molecular_Hamiltonian.MOLECULES
        mol_filename = os.path.join(constants.DEFAULT_MOLECULE_ROOT, f'{self.mol_name}.hdf5')
        mol = MolecularData(filename=mol_filename)
        of_fermion_operator = get_fermion_operator(mol.get_molecular_hamiltonian())
        symb_ham = jordan_wigner(of_fermion_operator)
        return get_sparse_operator(symb_ham)

    def qubit_number_after_jordan_wigner(self):
        assert self.mol_name in Molecular_Hamiltonian.MOLECULES
        mol_filename = os.path.join(constants.DEFAULT_MOLECULE_ROOT, f'{self.mol_name}.hdf5')
        mol = MolecularData(filename=mol_filename)
        return 2 * mol.n_orbitals

    # this function does not return identity matrices in contrast to the function
    # observables_and_coefficients_for_energy_estimation
    def observables_for_energy_estimation(self):
        assert self.mol_name in Molecular_Hamiltonian.MOLECULES
        mol_filename = os.path.join(constants.DEFAULT_MOLECULE_ROOT, f'{self.mol_name}.hdf5')
        mol = MolecularData(filename=mol_filename)
        of_fermion_operator = get_fermion_operator(mol.get_molecular_hamiltonian())
        symb_ham = jordan_wigner(of_fermion_operator)
        symb_ham_str = str(symb_ham)
        observable_strings = []
        for j in range(0, 10 ** 5):
            try:
                rubbish, symb_ham_str = symb_ham_str.split('[', 1)
                observable_string, symb_ham_str = symb_ham_str.split(']', 1)
                observable_strings.append(observable_string)
            except:
                break
        observable_arrays = []
        for observable_string in observable_strings:
            observable_array_right_format = []
            observable_array_wrong_format = observable_string.split(' ')
            for observable in observable_array_wrong_format:
                try:
                    basis, qubit_idx = observable
                    observable_right_format = [basis, int(qubit_idx)]
                    observable_array_right_format.append(observable_right_format)
                except:
                    break
            if observable_array_right_format:
                observable_arrays.append(observable_array_right_format)
        return observable_arrays

    def observables_and_coefficients_for_energy_estimation(self):
        assert self.mol_name in Molecular_Hamiltonian.MOLECULES
        mol_filename = os.path.join(constants.DEFAULT_MOLECULE_ROOT, f'{self.mol_name}.hdf5')
        mol = MolecularData(filename=mol_filename)
        of_fermion_operator = get_fermion_operator(mol.get_molecular_hamiltonian())
        symb_ham = jordan_wigner(of_fermion_operator)
        symb_ham_str = 'rubbish' + str(symb_ham)
        observable_strings = []
        coefficients = []
        for j in range(0, 10 ** 5):
            try:
                rubbish, symb_ham_str = symb_ham_str.split('(', 1)
                coefficient, symb_ham_str = symb_ham_str.split(')', 1)
                rubbish, symb_ham_str = symb_ham_str.split('[', 1)
                observable_string, symb_ham_str = symb_ham_str.split(']', 1)
                observable_strings.append(observable_string)
                coefficients.append(complex(coefficient))
            except:
                break
        observable_arrays = []
        for observable_string in observable_strings:
            observable_array_right_format = []
            observable_array_wrong_format = observable_string.split(' ')
            for observable in observable_array_wrong_format:
                try:
                    basis, qubit_idx = observable
                    observable_right_format = [basis, int(qubit_idx)]
                    observable_array_right_format.append(observable_right_format)
                except:
                    break
            observable_arrays.append(observable_array_right_format)
        return observable_arrays, coefficients

    def energy_shadow(self, measurement):
        observables, coefficients = self.observables_and_coefficients_for_energy_estimation()
        energy = 0
        for i in range(0, len(observables)):
            if not observables[i]:
                energy = energy + coefficients[i]
                continue
            else:
                sum_product, cnt_match = estimate_exp(measurement, observables[i])
            if cnt_match == 0:
                print('The observable ' + str(observables[i]) + ' has not been measured once.')
            if sum_product == 0 and cnt_match == 0:
                expectation_val = 0
            elif cnt_match == 0 and sum_product != 0:
                print('cnt_match is zero (problemo)!')
            else:
                expectation_val = sum_product / cnt_match
            energy = energy + coefficients[i] * expectation_val
        return energy


def main():
    molecule = 'H2'
    print(Molecular_Hamiltonian(molecule).diagonalize(1, True))


if __name__ == '__main__':
    main()
