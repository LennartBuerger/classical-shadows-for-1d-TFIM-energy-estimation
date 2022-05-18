import os
import numpy as np
import torch as pt

from typing import List, Dict

from ..mps import MPS
from .constants import BEST_ENERGIES_DIR, BEST_MODELS_DIR
from ..constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return dir_name


def load_best_energy(*,
                     root_dir: str = None,
                     identifier: str = None):
    energy_filename = os.path.join(root_dir,
                                   BEST_ENERGIES_DIR,
                                   f'{identifier}.npy')
    if os.path.exists(energy_filename):
        return np.load(energy_filename)
    else:
        return np.inf


def update_best_energy(*,
                       descr: str = None,
                       root_dir: str = None,
                       identifier: str = None,
                       iter_idx: int = None,
                       cur_energy: float = None,
                       cur_model: np.ndarray = None,
                       experiment_name: str = None,
                       logger=None):
    best_energy = load_best_energy(root_dir=root_dir, identifier=identifier)
    if cur_energy < best_energy:
        assert cur_energy is not None
        energy_filename = os.path.join(root_dir,
                                       BEST_ENERGIES_DIR,
                                       f'{identifier}.npy')
        np.save(energy_filename, cur_energy)

        assert cur_model is not None
        model_filename = os.path.join(root_dir,
                                      BEST_MODELS_DIR,
                                      f'{identifier}.npy')
        np.save(model_filename, cur_model)

        with open(os.path.join(root_dir, BEST_ENERGIES_DIR, f'{identifier}_info.txt'), 'w') as f:
            f.write(f'Experiment: {experiment_name}; iter_idx: {iter_idx}')

        logger.info(f'New best {descr} energy found!')

    return best_energy


def load_ham_mpses(molecule_dir: str = None,
                   ham_mps_num: int = 1,
                   work_dtype=None,
                   visible_num: int = None,
                   verbose: bool = False):
    ham_mpses = []
    ham_mpses_dir = os.path.join(molecule_dir,
                                 f'{ham_mps_num}_ham_mpses_{work_dtype}')
    if os.path.exists(ham_mpses_dir):
        print(f'Hamiltonians exist on the disk, we load them')
        for ham_idx in range(ham_mps_num):
            tensor_list = []
            for idx in range(visible_num):
                tensor_list.append(pt.from_numpy(np.load(os.path.join(ham_mpses_dir,
                                                                      f'ham_mps_{ham_idx}_tensor_{idx}.npy'))))
            ham_mpses.append(MPS.from_tensor_list(tensor_list))
            if verbose:
                print(f'MPS #{ham_idx}: {ham_mpses[-1]}')
    else:
        print(f'Hamiltonians do not exist on the disk, you have to create them manually!')

    return ham_mpses


def load_param_vecs(param_vec_dir: str = None,
                    max_bond_dims: List[int] = None,
                    param_vec_num: int = None,
                    prefix: str = None,
                    param_vec_lens: Dict[int, int] = None,
                    init_scale: float = 0.001,
                    dtype=None,
                    verbose: bool = False):
    param_vecs = {
        max_bond_dim: [] for max_bond_dim in max_bond_dims
    }
    for max_bond_dim in max_bond_dims:
        if verbose:
            print(f'Max bond dim = {max_bond_dim}')
        for param_vec_idx in range(param_vec_num):
            param_vec_filename = os.path.join(param_vec_dir,
                                              f'{prefix}_{max_bond_dim}_{dtype}_{param_vec_idx}.npy')
            if not os.path.exists(param_vec_filename):
                if verbose:
                    print(f'Creating param vec #{param_vec_idx}')
                param_vec = init_scale * pt.randn((param_vec_lens[max_bond_dim], ), dtype=dtype)
                param_vecs[max_bond_dim].append(param_vec)
                np.save(param_vec_filename,
                        param_vecs[max_bond_dim][param_vec_idx].detach().numpy())
            else:
                if verbose:
                    print(f'Loading param vec #{param_vec_idx}')
                param_vecs[max_bond_dim].append(pt.from_numpy(np.load(param_vec_filename)))
        if verbose:
            print()

    return param_vecs


def init_param_vec(param_vecs: Dict[int, Dict[int, pt.Tensor]],
                   max_bond_dim: int = None,
                   param_vec_idx: int = None,
                   dtype=None,
                   device=None):
    param_vec = pt.from_numpy(np.copy(param_vecs[max_bond_dim][param_vec_idx].detach().numpy()))
    param_vec = param_vec.to(device)
    # if dtype == BASE_COMPLEX_TYPE:
    #     param_vec = pt.view_as_real(param_vec)
    param_vec.requires_grad = True

    return param_vec
