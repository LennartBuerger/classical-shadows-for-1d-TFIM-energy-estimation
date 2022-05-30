import torch as pt
import numpy as np
from src.mps import MPS
from src.tfim_hamiltonian_open_fermion import TfimHamiltonianOpenFermion
from src.mps_quantum_state import MPSQuantumState

# this function converts the stored measurement outcomes to the correct shape needed for the shadow prediction
def conversion_to_prediction_shadow_dict_shape(measurement_procedure, measurement_index, qubit_num):
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(qubit_num))
    strs = to_str_func(measurement_index)
    dirac_rep = np.zeros(list(measurement_index.shape) + [qubit_num], dtype=np.int8)
    for bit_ix in range(0, qubit_num):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        dirac_rep[..., bit_ix] = fetch_bit_func(strs).astype("int8")
    measurement_array = np.where(dirac_rep == 1, -1, dirac_rep)
    measurement_array = np.where(dirac_rep == 0, 1, measurement_array)
    measurement = np.dstack((measurement_procedure, np.array(measurement_array, dtype=int)))
    return measurement

qubit_num = 6
ratio_h_j = 0.1
bond_dim = 50
tensor_liste_rand = [pt.rand([1, 2, bond_dim], dtype=pt.cdouble)]
for idx in range(qubit_num - 2):
    tensor_liste_rand.append(pt.rand([bond_dim, 2, bond_dim], dtype=pt.cdouble))
tensor_liste_rand.append(pt.rand([bond_dim, 2, 1], dtype=pt.cdouble))
mps = MPS.from_tensor_list(tensor_liste_rand).normalise()
param_vec = mps.to_param_vec()
# we only needed this param vec to determine the size of the tensor
random_param_vec = pt.rand(param_vec.shape[0], dtype=pt.cdouble)

random_param_vec_brute = random_param_vec.detach()
random_param_vec_brute.requires_grad_(True)
mps.from_param_vec(param_vec=random_param_vec_brute)
mps.normalise()
psi = mps.to_state_vector()
hamiltonian = pt.from_numpy(TfimHamiltonianOpenFermion(qubit_num, ratio_h_j, 1, 'open').to_matrix().todense())
energy_bf = psi.conj() @ hamiltonian @ psi / (psi.conj() @ psi)
print(f'BF E = {energy_bf}')
energy_bf.backward()

step_num = 5
lr = 0.01
random_param_vec_shadow = random_param_vec.detach()
random_param_vec_shadow.requires_grad_(True)
opt = pt.optim.Adam([random_param_vec_shadow], lr=lr)
num_of_measurements = 50
num_of_measurements_per_rot = 100  # this has to be high enough, if there is only one unique measurement outcome aleksei's amplitude method fails

opt.zero_grad()
mps.from_param_vec(param_vec=random_param_vec_shadow)
mps.normalise()
with pt.no_grad():
    meas_outcomes, meas_procedure, probs = MPSQuantumState(qubit_num, mps).measurement_shadow(
        num_of_measurements, num_of_measurements_per_rot)
energies = pt.zeros(len(meas_outcomes))
for n in range(len(meas_outcomes)):
    unique_meas_outcomes, index_perm = np.unique(meas_outcomes[n].numpy(), return_index=True)
    probs_direct = MPSQuantumState(qubit_num, mps).rotate_pauli(meas_procedure[n]).prob(
        pt.tensor(unique_meas_outcomes))
    measurements = conversion_to_prediction_shadow_dict_shape([meas_procedure[n]] * len(unique_meas_outcomes),
                                                              unique_meas_outcomes, qubit_num)
    energies[n] = TfimHamiltonianOpenFermion(qubit_num, ratio_h_j, 1, 'open').energy_shadow_mps_modified(
        measurements, probs_direct)
energy_shadow = pt.mean(energies)
energy_shadow.backward()

print('the gradients: ')
print(random_param_vec_brute.grad)
print(random_param_vec_shadow.grad)


