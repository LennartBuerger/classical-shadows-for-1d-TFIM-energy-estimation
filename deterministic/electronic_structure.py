import numpy as np
import torch as pt

import os
import time

from nnqs.deterministic_torch.batch_hamiltonian import BatchHamiltonian
from nnqs.deterministic_torch.rbm_wave_function import RBMWaveFunction

from nnqs.deterministic_torch.preparation import create_parser, create_logger
from nnqs.deterministic_torch.preparation import calc_molecule_dir, create_molecule
from nnqs.deterministic_torch.preparation import create_of_qubit_operator

from openfermion.linalg import get_sparse_operator

CHEMICAL_ACCURACY = 1.6e-3


def main():
    parser = create_parser()
    args = parser.parse_args([])
    args.molecule_root = '../../../RBM/molecules'
    args.molecule_name = 'LiH'
    args.molecule_type = 'carleo'
    args.internuclear_distance = 1.26
    logger = create_logger(args)

    molecule_dir = calc_molecule_dir(args)
    molecule = create_molecule(args, logger)
    of_hamiltonian, visible_num = create_of_qubit_operator(molecule)

    electron_num = molecule.n_electrons
    spin = (molecule.multiplicity - 1) // 2
    CHEMICAL_ACCURACY = 1.6e-3
    print(f'Hartree-Fock energy: {molecule.hf_energy}')
    print(f'FCI energy: {molecule.fci_energy}')
    print(f'FCI energy up to chemical accuracy: {molecule.fci_energy + CHEMICAL_ACCURACY}')

    best_ever_energy = None
    best_ever_energy_file = os.path.join(molecule_dir, 'best_ever_energy.npy')
    if os.path.exists(best_ever_energy_file):
        best_ever_energy = np.load(best_ever_energy_file)
    else:
        best_ever_energy = np.inf
    print(f'Best ever energy: {best_ever_energy}')

    visible_num = visible_num
    hidden_num = visible_num
    device = pt.device('cpu')
    hamiltonian = BatchHamiltonian(visible_num=visible_num,
                                   of_hamiltonian=of_hamiltonian,
                                   device=device)
    wf = RBMWaveFunction(visible_num=visible_num,
                         hidden_num=hidden_num,
                         device=device,
                         electron_num=electron_num,
                         max_bond_dim=32)

    hamiltonian_pt = pt.from_numpy(get_sparse_operator(of_hamiltonian).todense())

    full_space_idx = pt.arange(2 ** visible_num)
    full_space = wf.idx_to_visible(full_space_idx)

    electron_num_mask = full_space.sum(dim=1) == electron_num
    spin_mask = (full_space[:, ::2].sum(dim=1) - full_space[:, 1::2].sum(dim=1)) == 0
    phys_mask = electron_num_mask  # & spin_mask
    phys_space_idx = full_space_idx[phys_mask]
    phys_space = wf.idx_to_visible(phys_space_idx)
    phys_hamiltonian_pt = hamiltonian_pt[phys_mask, :][:, phys_mask]

    new_x_pt = 0.01 * pt.randn((wf.param_num,), dtype=pt.cdouble)
    new_x_pt.requires_grad = True

    iter_num = 100
    start_time = time.time()
    for iter_idx in range(iter_num):
        print(f'Iteration #{iter_idx}: {hamiltonian.energy(wf=wf)}')
        # print(f'Iteration #{iter_idx}: {wf.param_vec_overlap(new_x_pt, new_x_pt)}')
    print(f'Average elapsed time: {(time.time() - start_time) / iter_num}')
    print(BatchHamiltonian.TIMING_DICT)
    exit(-1)

    best_run_energy = np.inf
    print(f'Best run energy: {best_run_energy}')

    iter_num = 500
    optimizer = pt.optim.SGD([new_x_pt], lr=0.1)
    loss = None

    for iter_idx in range(iter_num):
        start_time = time.time()
        wf.from_param_vec(new_x_pt)
        # loss = bf_energy_func_pt(new_x_pt)
        loss = hamiltonian.energy(wf=wf)

        # Comparing against the best values
        np_energy_value = loss.detach().numpy().real
        if np_energy_value < best_run_energy:
            best_run_energy = np_energy_value
        if np_energy_value < best_ever_energy:
            best_ever_energy = np_energy_value
            np.save(best_ever_energy_file, best_ever_energy)
            pt.save(wf.from_param_vec(new_x_pt), os.path.join(molecule_dir, 'best_ever_model'))

        print(f'Iteration #{iter_idx}/{iter_num}, '
              f'<E> = {np_energy_value}, '
              f'BR <E> = {best_run_energy}, '
              f'BE <E> = {best_ever_energy}')

        optimizer.zero_grad()
        loss.backward()

        # sr_matrix = wf.sr_matrix().detach()
        sr_matrix = wf.full_rbm_sr_matrix(full_space)
        sr_matrix += 1e-2 * pt.eye(*sr_matrix.shape,
                                   dtype=sr_matrix.dtype,
                                   device=sr_matrix.device)
        sr_matrix_inv = pt.linalg.pinv(sr_matrix)
        new_x_pt.grad = sr_matrix_inv @ new_x_pt.grad
        new_x_pt.grad = pt.div(new_x_pt.grad, pt.linalg.norm(new_x_pt.grad))

        #     sr_grad = wf.sr_grad(grad)
        #     sr_grad = pt.div(sr_grad, pt.linalg.norm(sr_grad))

        optimizer.step()

        print(f'Elapsed time = {time.time() - start_time}\n')

    # iter_num = 5000
    # # optimizer = pt.optim.Adam(wf.parameters())
    # optimizer = pt.optim.SGD(wf.parameters(), lr=0.1)
    # loss = None
    #
    # energy_time = 0.0
    # energy_grad_time = 0.0
    # sr_time = 0.0
    #
    # prev_energy_conv_time = 0.0
    # prev_energy_denom_time = 0.0
    # prev_energy_num_time = 0.0
    # prev_energy_time = 0.0
    # prev_energy_grad_time = 0.0
    # prev_sr_ket_conv_time = 0.0
    # prev_sr_bra_conv_time = 0.0
    # prev_sr_overlap_time = 0.0
    # prev_sr_grad_time = 0.0
    # prev_sr_grad_grad_time = 0.0
    # prev_sr_time = 0.0

    # for iter_idx in range(iter_num):
    #     iter_start_time = time.time()
    #     start_time = time.time()
    #     loss = hamiltonian.energy(wf=wf)
    #     energy_time += time.time() - start_time
    #
    #     optimizer.zero_grad()
    #     start_time = time.time()
    #     loss.backward()
    #     energy_grad_time += time.time() - start_time
    #
    #     grad = pt.cat((wf.vb.grad, wf.hb.grad, pt.reshape(wf.wm.grad, (-1,))))
    #
    #     #     sr_matrix = wf.sr_matrix().detach()
    #     #     sr_matrix += 1e-5 * pt.eye(*sr_matrix.shape,
    #     #                                dtype=sr_matrix.dtype,
    #     #                                device=sr_matrix.device)
    #     #     sr_matrix_inv = pt.linalg.pinv(sr_matrix)
    #     #     sr_grad = sr_matrix_inv @ sr_grad
    #     grad = pt.div(grad, pt.norm(grad))
    #     start_time = time.time()
    #     sr_grad = wf.sr_grad(grad)
    #     sr_time += time.time() - start_time
    #     # sr_grad = grad
    #     sr_grad = pt.div(sr_grad, pt.linalg.norm(sr_grad))
    #
    #     wf.vb.grad = sr_grad[:visible_num]
    #     wf.hb.grad = sr_grad[visible_num:visible_num + hidden_num]
    #     wf.wm.grad = pt.reshape(sr_grad[visible_num + hidden_num:], (hidden_num, visible_num))
    #     optimizer.step()
    #
    #     print(f'Iteration #{iter_idx}/{iter_num}, '
    #           f'energy = {loss.detach().cpu().numpy()}, '
    #           f'elapsed time = {time.time() - iter_start_time}\n')
    #
    #     print(f'Energy to MPS time: {RBMWaveFunction.ENERGY_CONV_TIME - prev_energy_conv_time}')
    #     print(f'Energy denominator time: {RBMWaveFunction.ENERGY_DENOM_TIME - prev_energy_denom_time}')
    #     print(f'Energy numerator time: {RBMWaveFunction.ENERGY_NUM_TIME - prev_energy_num_time}')
    #     print(f'Total energy time: {energy_time - prev_energy_time}')
    #
    #     print(f'\nEnergy grad time: {energy_grad_time - prev_energy_grad_time}\n')
    #
    #     print(f'SR ket to MPS time: {RBMWaveFunction.SR_KET_CONV_TIME - prev_sr_ket_conv_time}')
    #     print(f'SR bra to MPS time: {RBMWaveFunction.SR_BRA_CONV_TIME - prev_sr_bra_conv_time}')
    #     print(f'SR overlap time: {RBMWaveFunction.SR_OVERLAP_TIME - prev_sr_overlap_time}')
    #     print(f'SR grad time: {RBMWaveFunction.SR_GRAD_TIME - prev_sr_grad_time}')
    #     print(f'SR grad grad time: {RBMWaveFunction.SR_GRAD_GRAD_TIME - prev_sr_grad_grad_time}')
    #     print(f'Total SR time: {sr_time - prev_sr_time}')
    #
    #     prev_energy_conv_time = RBMWaveFunction.ENERGY_CONV_TIME
    #     prev_energy_denom_time = RBMWaveFunction.ENERGY_DENOM_TIME
    #     prev_energy_num_time = RBMWaveFunction.ENERGY_NUM_TIME
    #     prev_energy_time = energy_time
    #
    #     prev_energy_grad_time = energy_grad_time
    #     prev_sr_ket_conv_time = RBMWaveFunction.SR_KET_CONV_TIME
    #     prev_sr_bra_conv_time = RBMWaveFunction.SR_BRA_CONV_TIME
    #     prev_sr_overlap_time = RBMWaveFunction.SR_OVERLAP_TIME
    #     prev_sr_grad_time = RBMWaveFunction.SR_GRAD_TIME
    #     prev_sr_grad_grad_time = RBMWaveFunction.SR_GRAD_GRAD_TIME
    #     prev_sr_time = sr_time
    #
    # print(f'\nEnergy to MPS time: {RBMWaveFunction.ENERGY_CONV_TIME}')
    # print(f'Energy denominator time: {RBMWaveFunction.ENERGY_DENOM_TIME}')
    # print(f'Energy numerator time: {RBMWaveFunction.ENERGY_NUM_TIME}')
    # print(f'Total energy time: {energy_time}')
    #
    # print(f'\nEnergy grad time: {energy_grad_time}\n')
    #
    # print(f'SR ket to MPS time: {RBMWaveFunction.SR_KET_CONV_TIME}')
    # print(f'SR bra to MPS time: {RBMWaveFunction.SR_BRA_CONV_TIME}')
    # print(f'SR overlap time: {RBMWaveFunction.SR_OVERLAP_TIME}')
    # print(f'SR grad time: {RBMWaveFunction.SR_GRAD_TIME}')
    # print(f'SR grad grad time: {RBMWaveFunction.SR_GRAD_GRAD_TIME}')
    # print(f'Total SR time: {sr_time}')

if __name__ == '__main__':
    main()