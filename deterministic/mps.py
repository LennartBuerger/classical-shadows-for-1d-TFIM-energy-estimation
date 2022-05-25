from __future__ import annotations

import numpy as np
import torch as pt

from typing import Tuple, Union, List, Callable

from .constants import BASE_INT_TYPE, BASE_COMPLEX_TYPE
from .constants import DEFAULT_INIT_SCALE

from .svd import pt_H, trunc_svd, robust_qr
from .constants import DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import DEFAULT_MAX_BOND_DIM, DEFAULT_CUTOFF

from .symmetries import _e_num_symmetry, _spin_symmetry


def non_batched(f):
    def wrapped_f(self, *args, **kwargs):
        if self.is_batched:
            raise NotImplementedError(f'The function {f} is not yet adjusted for '
                                      f'batched execution')
        else:
            return f(self, *args, **kwargs)

    return wrapped_f


class MPS:
    TIMING_DICT = {}

    def __init__(self,
                 *,
                 name: str = None,
                 visible_num: int,
                 phys_dims: Union[int, list] = None,
                 bond_dims: Union[int, list] = None,
                 dtype=BASE_COMPLEX_TYPE,
                 idx_dtype=BASE_INT_TYPE,
                 device=None,
                 tensors: list = None,
                 init_scale: float = DEFAULT_INIT_SCALE,
                 max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                 cutoff: float = DEFAULT_CUTOFF,
                 svd_backend: str = DEFAULT_SVD_BACKEND,
                 backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                 given_orth_idx: int = None,
                 new_orth_idx: int = None,):
        self.name = name

        self.visible_num = visible_num

        self.dtype = dtype
        self.idx_dtype = idx_dtype
        assert device is not None
        self.device = device
        
        if isinstance(phys_dims, int):
            self.phys_dims = [phys_dims] * visible_num
        elif isinstance(phys_dims, list):
            assert len(phys_dims) == visible_num
            self.phys_dims = [phys_dim for phys_dim in phys_dims]
        else:
            raise TypeError(f'phys_dims should be either int, or list. '
                            f'In fact they are: {type(bond_dims)}.')

        if isinstance(bond_dims, int):
            self.bond_dims = [bond_dims] * (visible_num - 1)
        elif isinstance(bond_dims, list):
            if visible_num > 0:
                assert len(bond_dims) == (visible_num - 1)
            self.bond_dims = [bond_dim for bond_dim in bond_dims]
        else:
            raise TypeError(f'bond_dims should be either int, or list. '
                            f'In fact they are: {type(bond_dims)}.')

        self.ext_bond_dims = [1] + [bond_dim for bond_dim in self.bond_dims] + [1]

        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff
        self.svd_backend = svd_backend
        self.backprop_backend = backprop_backend

        self.init_scale = pt.tensor(init_scale,
                                    dtype=self.dtype).to(self.device)
        self.is_batched = False
        if tensors is None:
            # Initialise tensors
            self.device = device
            self.tensors = [self.init_scale * pt.randn((self.ext_bond_dims[idx],
                                                        self.phys_dims[idx],
                                                        self.ext_bond_dims[idx + 1]),
                                                       device=self.device,
                                                       dtype=self.dtype)
                            for idx in range(self.visible_num)]
        else:
            assert np.all([tensor.dtype == self.dtype for tensor in tensors])
            assert np.all([tensor.device == self.device for tensor in tensors])
            assert isinstance(tensors, list)
            if self.visible_num > 0:
                assert (len(tensors) == self.visible_num)

                # Check consistency of all physical and bond dimensions
                input_phys_dims = [tensor.shape[-2] for tensor in tensors]
                assert np.all(np.equal(input_phys_dims, self.phys_dims))

                input_bond_dims = [tensor.shape[-1] for tensor in tensors[:-1]]
                assert np.all(np.equal(input_bond_dims, self.bond_dims))

                input_bond_dims = [tensor.shape[-3] for tensor in tensors[1:]]
                assert np.all(np.equal(input_bond_dims, self.bond_dims))

                batch_dims = [tensor.shape[:-3] for tensor in tensors]
                assert np.all(np.equal(batch_dims, batch_dims[0]))
                if len(batch_dims[0]) > 0:
                    self.is_batched = True

            # Check left and right caps are caps
            assert tensors[0].shape[-3] == 1
            assert tensors[-1].shape[-1] == 1

            self.tensors = tensors

        self.orth_idx = given_orth_idx
        if new_orth_idx is not None:
            if self.visible_num > 0:
                self.canonicalise(new_orth_idx)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.phys_dims)

    @property
    def hidden_shape(self) -> Tuple[int, ...]:
        return tuple(self.ext_bond_dims)

    def __str__(self) -> str:

        return (f'MPS {self.name}:\n'
                f'\tvisible_num = {self.visible_num}\n'
                f'\tphys_dims = {self.phys_dims}\n'
                f'\tbond_dims = {self.bond_dims}\n'
                f'\text_bond_dims = {self.ext_bond_dims}\n'
                f'\torth_idx = {self.orth_idx}\n')

    def conj(self) -> MPS:
        conj_tensors = [pt.conj(tensor) for tensor in self.tensors]
        return MPS(name=f'{self.name}_H',
                   visible_num=self.visible_num,
                   phys_dims=self.phys_dims,
                   bond_dims=self.bond_dims,
                   dtype=self.dtype,
                   idx_dtype=self.idx_dtype,
                   device=self.device,
                   tensors=conj_tensors,
                   max_bond_dim=self.max_bond_dim,
                   cutoff=self.cutoff,
                   svd_backend=self.svd_backend,
                   backprop_backend=self.backprop_backend,
                   given_orth_idx=self.orth_idx)

    def to(self, device):
        for idx in range(self.visible_num):
            self.tensors[idx] = self.tensors[idx].to(device)
        self.init_scale = self.init_scale.to(device)
        self.device = device

        return self

    @property
    def param_num(self):
        theor = np.asarray([self.ext_bond_dims[:-1],
                            self.phys_dims,
                            self.ext_bond_dims[1:]]).prod(axis=0).sum()
        exp = np.asarray([tensor.numel() for tensor in self.tensors]).sum()
        assert exp == theor

        return theor

    def to_param_vec(self):
        with pt.no_grad():
            param_vec = pt.reshape(self.tensors[0], (-1, ))
            for idx in range(1, self.visible_num):
                param_vec = pt.cat((param_vec, pt.reshape(self.tensors[idx], (-1, ))), dim=0)

            assert param_vec.numel() == self.param_num

            return param_vec

    def param_vec_to_tensor_list(self, param_vec: pt.Tensor = None):
        section_sizes = np.asarray([self.ext_bond_dims[:-1],
                                    self.phys_dims,
                                    self.ext_bond_dims[1:]]).prod(axis=0)
        tensors = pt.split(param_vec, tuple(section_sizes))
        tensors = [pt.reshape(tensors[idx], (self.ext_bond_dims[idx],
                                             self.phys_dims[idx],
                                             self.ext_bond_dims[idx + 1])) for idx in range(self.visible_num)]

        return tensors

    def from_param_vec(self, param_vec: pt.Tensor = None):
        assert param_vec.dtype == self.dtype
        assert param_vec.shape == (self.param_num, )
        self.tensors = None
        self.tensors = self.param_vec_to_tensor_list(param_vec)

    @non_batched
    def idx_to_visible(self, idx: pt.Tensor) -> pt.Tensor:
        if not pt.is_tensor(idx):
            idx = pt.tensor(idx, dtype=self.idx_dtype, device=self.device)
        mul = int(pt.log2(pt.tensor(self.phys_dims[0])))
        shifts = mul * pt.arange(0,
                                 self.visible_num, dtype=self.idx_dtype, device=self.device)
        visible = pt.squeeze(pt.remainder((idx.reshape((-1, 1)) >> shifts), self.phys_dims[0]))

        return visible

    @non_batched
    def visible_to_idx(self, visible: pt.Tensor) -> pt.Tensor:
        if visible.dtype != self.idx_dtype:
            visible = visible.type(self.idx_dtype)
        powers = self.phys_dims[0] ** pt.arange(0, self.visible_num, dtype=self.idx_dtype, device=self.device)

        return pt.tensordot(visible.to(self.device), powers, dims=1)

    @non_batched
    def visible_to_amplitude(self, visible: pt.Tensor) -> pt.Tensor:
        """
        Batched calculation of MPS amplitudes corresponding to the ndarray
        of input visible states with shape (batch_index, visible_num).

        :param visible:
        :return:
        """
        assert visible.shape[-1] == self.visible_num
        visible = visible.to(self.device)
        rolling_tensor = pt.index_select(self.tensors[0], 1, visible[:, 0])
        for idx in range(1, self.visible_num):
            rolling_tensor = pt.einsum('...iaj,jak->...iak',
                                       rolling_tensor,
                                       pt.index_select(self.tensors[idx], 1, visible[:, idx]))

        return pt.squeeze(rolling_tensor)

    @non_batched
    def amplitude(self, idx: pt.Tensor) -> pt.Tensor:

        return self.visible_to_amplitude(self.idx_to_visible(idx))

    @non_batched
    def part_func(self) -> pt.Tensor:
        result = pt.sum(self.tensors[0], dim=1)
        for idx in range(1, self.visible_num):
            result = pt.matmul(result, pt.sum(self.tensors[idx], dim=1))

        return pt.sum(result)

    @staticmethod
    def overlap(*,
                bra: MPS = None,
                ket: MPS = None) -> pt.Tensor:
        assert bra.visible_num == ket.visible_num
        assert bra.dtype == ket.dtype
        assert bra.idx_dtype == ket.idx_dtype
        assert bra.device == ket.device

        result = pt.squeeze(pt.squeeze(pt.tensordot(ket.tensors[0],
                                                    bra.tensors[0],
                                                    dims=[[1], [1]]),
                                       dim=2),
                            dim=0)
        for idx in range(1, bra.visible_num):
            # Left -> right
            result = pt.tensordot(result,
                                  bra.tensors[idx],
                                  dims=[[-1], [0]])

            # Up -> bottom
            result = pt.tensordot(ket.tensors[idx],
                                  result,
                                  dims=[[0, 1], [0, 1]])

        return pt.squeeze(result)

    @staticmethod
    def batched_overlap(*,
                bra: MPS = None,
                ket: MPS = None) -> pt.Tensor:
        assert bra.visible_num == ket.visible_num
        assert bra.dtype == ket.dtype
        assert bra.idx_dtype == ket.idx_dtype
        assert bra.device == ket.device

        result = pt.einsum('...aj,...ak->...jk',
                           pt.squeeze(ket.tensors[0]),
                           pt.squeeze(bra.tensors[0]))
        for idx in range(1, bra.visible_num):
            # Left -> right
            result = pt.einsum('...ij,...jak->...iak',
                               result,
                               bra.tensors[idx])


            # Up -> bottom
            result = pt.einsum('...iaj,...iak->...jk',
                               ket.tensors[idx],
                               result)

        return pt.squeeze(result)

    @non_batched
    def norm(self) -> pt.Tensor:

        return pt.sqrt(self.overlap(bra=self.conj(), ket=self))

    @non_batched
    def norm_bf(self) -> pt.Tensor:
        return self.norm()

    @non_batched
    def normalise(self) -> MPS:
        norm = self.norm()
        scaling = pt.pow(norm, 1.0 / self.visible_num)
        # print(scaling)
        # print(f'Hei!')
        for idx in range(self.visible_num):
            self.tensors[idx] = self.tensors[idx] / scaling

        return self

    @non_batched
    def to_tensor(self) -> pt.Tensor:
        result = self.tensors[0]
        for idx in range(1, self.visible_num):
            result = pt.tensordot(result, self.tensors[idx], dims=[[-1], [0]])

        return pt.squeeze(result)

    @non_batched
    def to_state_vector(self) -> pt.Tensor:
        result = self.tensors[0]
        for idx in range(1, self.visible_num):
            result = pt.einsum('iaj,jbk->iabk',
                               result,
                               self.tensors[idx])
            result = pt.permute(result, (0, 2, 1, 3))
            shape = result.shape
            result = pt.reshape(result, (shape[0], -1, shape[-1]))

        return pt.squeeze(result)

    @non_batched
    def to_state_vector_old(self) -> pt.Tensor:
        return self.amplitude(pt.arange(self.phys_dims[0] ** self.visible_num))

    @staticmethod
    def from_state_vector(visible_num: int,
                          state_vector: pt.Tensor,
                          phys_dim: int = 2):
        assert len(state_vector.shape) == 1
        assert state_vector.shape[0] == phys_dim ** visible_num

        tensor_list = []

        state_vector = pt.reshape(state_vector, (1, -1))
        for idx in range(visible_num - 1):
            shape = state_vector.shape
            state_vector = pt.reshape(state_vector, (shape[0], phys_dim, -1))
            q, r = pt.linalg.qr(state_vector.reshape(shape[0] * phys_dim, -1))
            tensor_list.append(pt.reshape(q, (-1, phys_dim, q.shape[1])))
            state_vector = r

        tensor_list.append(pt.reshape(state_vector, (state_vector.shape[0], state_vector.shape[1], -1)))

        return MPS.from_tensor_list([pt.permute(tensor, (2, 1, 0)) for tensor in tensor_list][::-1])

    def set_bond_dim(self, bond_idx, val):
        """
        A function to simultaneously update an entry in the list of bond
        dimensions (`self._bond_dims`) and in the extended list of bond
        dimensions (`self._ext_bond_dims`)

        :param bond_idx:
        :param val:
        :return:
        """
        self.bond_dims[bond_idx] = val
        self.ext_bond_dims[bond_idx + 1] = val

    @non_batched
    def canonicalise(self, new_orth_idx: int = -1):
        assert (0 <= new_orth_idx) and (new_orth_idx < self.visible_num)

        if self.orth_idx is None:
            forward_start_idx = 0
            backward_start_idx = self.visible_num - 1
        else:
            forward_start_idx = self.orth_idx
            backward_start_idx = self.orth_idx

        for idx in range(forward_start_idx, new_orth_idx):
            matrix = pt.reshape(self.tensors[idx],
                                (self.ext_bond_dims[idx] * self.phys_dims[idx],
                                 self.ext_bond_dims[idx + 1]))
            matrix = matrix.to(pt.device('cpu'))
            q, r = pt.linalg.qr(matrix)
            q = q.to(self.device)
            r = r.to(self.device)

            self.set_bond_dim(idx, q.shape[1])
            self.tensors[idx] = pt.reshape(q, (self.ext_bond_dims[idx],
                                               self.phys_dims[idx],
                                               self.ext_bond_dims[idx + 1]))
            self.tensors[idx + 1] = pt.tensordot(r,
                                                 self.tensors[idx + 1],
                                                 dims=[[1], [0]])

        for idx in range(backward_start_idx, new_orth_idx, -1):
            matrix = pt.t(pt.reshape(self.tensors[idx],
                                     (self.ext_bond_dims[idx],
                                      self.ext_bond_dims[idx + 1] * self.phys_dims[idx])))
            matrix = matrix.to(pt.device('cpu'))
            q_t, r_t = pt.linalg.qr(matrix)
            q = pt.t(q_t).to(self.device)
            r = pt.t(r_t).to(self.device)

            self.set_bond_dim(idx - 1, q.shape[0])
            self.tensors[idx] = pt.reshape(q, (self.ext_bond_dims[idx],
                                               self.phys_dims[idx],
                                               self.ext_bond_dims[idx + 1]))
            self.tensors[idx - 1] = pt.tensordot(self.tensors[idx - 1],
                                                 r,
                                                 dims=[[-1], [0]])

        self.orth_idx = new_orth_idx

    def cut_bond_dims(self):
        for bond_idx, bond_dim in enumerate(self.bond_dims):
            if (self.max_bond_dim is not None) and (bond_dim > self.max_bond_dim):
                ldx, rdx = bond_idx, bond_idx + 1
                self.canonicalise(ldx)

                bond_tensor = pt.einsum('iaj,jbk->iabk',
                                        self.tensors[ldx],
                                        self.tensors[rdx])
                # Calculate external bond dimensions of left and right matrices
                left_dim = self.ext_bond_dims[ldx] * self.phys_dims[ldx]
                right_dim = self.ext_bond_dims[rdx + 1] * self.phys_dims[rdx]

                bond_tensor = pt.reshape(bond_tensor, (left_dim, right_dim))
                u, s, v = trunc_svd(matrix=bond_tensor,
                                    max_bond_dim=self.max_bond_dim,
                                    cutoff=self.cutoff,
                                    svd_backend=self.svd_backend,
                                    backprop_backend=self.backprop_backend)
                self.set_bond_dim(ldx, u.shape[1])
                ltensor = pt.matmul(u, pt.diag(s))
                rtensor = pt_H(v)

                self.tensors[ldx] = pt.reshape(ltensor, (self.ext_bond_dims[ldx],
                                                         self.phys_dims[ldx],
                                                         self.ext_bond_dims[ldx + 1]))
                self.tensors[rdx] = pt.reshape(rtensor, (self.ext_bond_dims[rdx],
                                                         self.phys_dims[rdx],
                                                         self.ext_bond_dims[rdx + 1]))

    def trim_bond_dims(self):
        for bond_idx, bond_dim in enumerate(self.bond_dims):
                ldx, rdx = bond_idx, bond_idx + 1
                self.canonicalise(ldx)

                bond_tensor = pt.einsum('iaj,jbk->iabk',
                                        self.tensors[ldx],
                                        self.tensors[rdx])
                # Calculate external bond dimensions of left and right matrices
                left_dim = self.ext_bond_dims[ldx] * self.phys_dims[ldx]
                right_dim = self.ext_bond_dims[rdx + 1] * self.phys_dims[rdx]

                bond_tensor = pt.reshape(bond_tensor, (left_dim, right_dim))
                u, s, v = trunc_svd(matrix=bond_tensor,
                                    max_bond_dim=None,
                                    cutoff=self.cutoff,
                                    svd_backend=self.svd_backend,
                                    backprop_backend=self.backprop_backend)
                self.set_bond_dim(ldx, u.shape[1])
                ltensor = pt.matmul(u, pt.diag(s))
                rtensor = pt_H(v)

                self.tensors[ldx] = pt.reshape(ltensor, (self.ext_bond_dims[ldx],
                                                         self.phys_dims[ldx],
                                                         self.ext_bond_dims[ldx + 1]))
                self.tensors[rdx] = pt.reshape(rtensor, (self.ext_bond_dims[rdx],
                                                         self.phys_dims[rdx],
                                                         self.ext_bond_dims[rdx + 1]))

    @staticmethod
    def tensor_diag(rank: int = -1,
                    diag: pt.Tensor = None):
        assert len(diag.shape) == 1
        size = diag.shape[0]
        result = pt.zeros(size ** rank,
                          dtype=diag.dtype,
                          device=diag.device)
        step = (size ** rank - 1) // (size - 1)
        indices = step * pt.arange(0, size, device=diag.device)
        return result.scatter_(0, indices, diag).reshape([size] * rank)

    @staticmethod
    def diag_to_mps(*,
                    name: str = None,
                    diag: pt.Tensor = None,
                    visible_num: int = -1,
                    max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                    cutoff: float = DEFAULT_CUTOFF,
                    svd_backend: str = DEFAULT_SVD_BACKEND,
                    backprop_backend: str = DEFAULT_BACKPROP_BACKEND) -> MPS:
        assert len(diag.shape) == 1
        tensors = list()
        if visible_num == 1:
            tensors.append(pt.unsqueeze(pt.unsqueeze(diag, dim=0), dim=-1))
        else:
            tensors.append(pt.unsqueeze(MPS.tensor_diag(2, diag),
                                        dim=0))
            for idx in range(1, visible_num - 1):
                tensors.append(MPS.tensor_diag(3, pt.ones(diag.shape[0], dtype=diag.dtype, device=diag.device)))
            tensors.append(pt.unsqueeze(MPS.tensor_diag(2, pt.ones(diag.shape[0], dtype=diag.dtype, device=diag.device)),
                                        dim=-1))

        return MPS(name=name,
                   visible_num=visible_num,
                   phys_dims=diag.shape[0],
                   bond_dims=diag.shape[0],
                   dtype=diag.dtype,
                   device=diag.device,
                   tensors=tensors,
                   max_bond_dim=max_bond_dim,
                   cutoff=cutoff,
                   svd_backend=svd_backend,
                   backprop_backend=backprop_backend,
                   given_orth_idx=0)

    @staticmethod
    def from_tensor_list(tensor_list,
                         *,
                         max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                         cutoff: float = DEFAULT_CUTOFF):
        shapes = [tensor.shape for tensor in tensor_list]
        return MPS(visible_num=len(shapes),
                   phys_dims=[shape[-2] for shape in shapes],
                   bond_dims=[shape[-1] for shape in shapes[:-1]],
                   max_bond_dim=max_bond_dim,
                   cutoff=cutoff,
                   dtype=tensor_list[0].dtype,
                   device=tensor_list[0].device,
                   tensors=tensor_list)

    @non_batched
    def canonicalise_to_centre(self, qr_lambda: Callable = None):
        centre_idx = self.visible_num // 2
        if qr_lambda is None:
            def ordinary_qr(x):
                x = x.to(pt.device('cpu'))
                q_loc, r_loc = pt.linalg.qr(x)

                return q_loc, r_loc

            qr_lambda = ordinary_qr
        for idx in range(0, centre_idx):
            matrix = pt.reshape(self.tensors[idx],
                                (self.ext_bond_dims[idx] * self.phys_dims[idx],
                                 self.ext_bond_dims[idx + 1]))

            q, r = qr_lambda(matrix)
            q = q.to(self.device)
            r = r.to(self.device)

            self.set_bond_dim(idx, q.shape[1])
            self.tensors[idx] = pt.reshape(q, (self.ext_bond_dims[idx],
                                               self.phys_dims[idx],
                                               self.ext_bond_dims[idx + 1]))
            self.tensors[idx + 1] = pt.tensordot(r,
                                                 self.tensors[idx + 1],
                                                 dims=[[1], [0]])

        for idx in range(self.visible_num - 1, centre_idx, -1):
            matrix = pt.t(pt.reshape(self.tensors[idx],
                                     (self.ext_bond_dims[idx],
                                      self.ext_bond_dims[idx + 1] * self.phys_dims[idx])))
            q_t, r_t = qr_lambda(matrix)
            q = pt.t(q_t).to(self.device)
            r = pt.t(r_t).to(self.device)

            self.set_bond_dim(idx - 1, q.shape[0])
            self.tensors[idx] = pt.reshape(q, (self.ext_bond_dims[idx],
                                               self.phys_dims[idx],
                                               self.ext_bond_dims[idx + 1]))
            self.tensors[idx - 1] = pt.tensordot(self.tensors[idx - 1],
                                                 r,
                                                 dims=[[-1], [0]])

        self.orth_idx = centre_idx

    # ==========================================================================================
    # ====================================== SYMMETRIES ========================================
    # ==========================================================================================
    def apply_symmetry_mpo(self,
                           *,
                           sym_mpo=None):
        if sym_mpo is not None:
            assert len(sym_mpo) == self.visible_num
            for idx in range(self.visible_num):
                assert sym_mpo[idx].shape[1] == self.tensors[idx].shape[1]
                assert sym_mpo[idx].shape[-1] == self.tensors[idx].shape[1]

                new_shape = (self.tensors[idx].shape[0] * sym_mpo[idx].shape[0],
                             self.tensors[idx].shape[1],
                             self.tensors[idx].shape[-1] * sym_mpo[idx].shape[2])
                self.tensors[idx] = pt.einsum('iaj,kbla->ikbjl',
                                              self.tensors[idx],
                                              sym_mpo[idx])
                self.tensors[idx] = pt.reshape(self.tensors[idx], new_shape)

            for idx in range(self.visible_num - 1):
                self.set_bond_dim(idx, self.tensors[idx].shape[-1])

        return self

    def apply_symmetry_mpo_and_cano(self,
                                    *,
                                    sym_mpo=None,
                                    qr_lambda: Callable = None):
        if sym_mpo is not None:
            centre_idx = self.visible_num // 2
            if qr_lambda is None:
                def ordinary_qr(x):
                    x = x.to(pt.device('cpu'))
                    q_loc, r_loc = pt.linalg.qr(x)

                    return q_loc, r_loc

                qr_lambda = ordinary_qr

            assert len(sym_mpo) == self.visible_num
            forward_tensor = pt.ones((1, 1, 1), dtype=self.dtype, device=self.device)
            for idx in range(centre_idx):
                assert sym_mpo[idx].shape[1] == self.tensors[idx].shape[1]
                assert sym_mpo[idx].shape[-1] == self.tensors[idx].shape[1]

                matrix_shape = (forward_tensor.shape[0] * sym_mpo[idx].shape[1],
                                self.tensors[idx].shape[2] * sym_mpo[idx].shape[2])

                matrix = pt.einsum('ijk,jal,kbma->iblm',
                                   forward_tensor,
                                   self.tensors[idx],
                                   sym_mpo[idx])
                matrix = pt.reshape(matrix, matrix_shape)
                q, r = qr_lambda(matrix)
                q = q.to(self.device)
                r = r.to(self.device)
                self.set_bond_dim(idx, q.shape[1])

                self.tensors[idx] = pt.reshape(q, (forward_tensor.shape[0],
                                                   sym_mpo[idx].shape[1],
                                                   q.shape[1]))
                forward_tensor = pt.reshape(r, (q.shape[1],
                                                self.tensors[idx + 1].shape[0],
                                                sym_mpo[idx + 1].shape[0]))

            backward_tensor = pt.ones((1, 1, 1), dtype=self.dtype, device=self.device)
            for idx in range(self.visible_num - 1, centre_idx - 1, -1):
                assert sym_mpo[idx].shape[1] == self.tensors[idx].shape[1]
                assert sym_mpo[idx].shape[-1] == self.tensors[idx].shape[1]

                matrix_shape = (self.tensors[idx].shape[0] * sym_mpo[idx].shape[0],
                                backward_tensor.shape[-1] * sym_mpo[idx].shape[1])
                matrix = pt.einsum('iaj,kbla,jlm->ikbm',
                                   self.tensors[idx],
                                   sym_mpo[idx],
                                   backward_tensor)
                matrix = pt.t(pt.reshape(matrix, matrix_shape))
                q_t, r_t = qr_lambda(matrix)
                q = pt.t(q_t).to(self.device)
                r = pt.t(r_t).to(self.device)
                self.set_bond_dim(idx - 1, q.shape[0])
                self.tensors[idx] = pt.reshape(q, (q.shape[0],
                                                   sym_mpo[idx].shape[1],
                                                   backward_tensor.shape[-1]))
                backward_tensor = pt.reshape(r, (-1,
                                                 sym_mpo[idx - 1].shape[2],
                                                 q.shape[0]))
            norm_tensor = pt.einsum('ijk,jkl->il',
                                    forward_tensor,
                                    backward_tensor)
            self.tensors[centre_idx] = pt.tensordot(norm_tensor,
                                                    self.tensors[centre_idx],
                                                    dims=[[-1], [0]])
            self.orth_idx = centre_idx
            self.set_bond_dim(centre_idx - 1, self.tensors[centre_idx].shape[0])

        return self

    @staticmethod
    def hf_mps(visible_num: int = None,
               electron_num: int = None,
               dtype=None,
               device=None):
        left_cap, site_tensor, right_cap = _e_num_symmetry(electron_num)
        non_hf_tensor = pt.ones((1, 2, 1), )
        non_hf_tensor[0, 1, 0] = 0

        tensors = []
        for _ in range(visible_num - electron_num):
            tensors.append(non_hf_tensor.type(dtype).to(device))
        tensors.append(left_cap.type(dtype).to(device))
        for _ in range(electron_num - 2):
            tensors.append(site_tensor.type(dtype).to(device))
        tensors.append(right_cap.type(dtype).to(device))

        hf_mps = MPS.from_tensor_list(tensors)
        hf_mps.trim_bond_dims()

        return hf_mps

    @staticmethod
    def anti_hf_mps(visible_num: int = None,
                    electron_num: int = None,
                    dtype=None,
                    device=None):
        left_cap, site_tensor, right_cap = _e_num_symmetry(visible_num)

        neg_left_cap = pt.zeros_like(left_cap)
        neg_left_cap[:, 0, :] = left_cap[:, 1, :]
        neg_left_cap[:, 1, :] = left_cap[:, 0, :]

        neg_site_tensor = pt.zeros_like(site_tensor)
        neg_site_tensor[:, 0, :] = site_tensor[:, 1, :]
        neg_site_tensor[:, 1, :] = site_tensor[:, 0, :]

        right_cap = pt.ones_like(right_cap) - right_cap

        tensors = []
        tensors.append(neg_left_cap.type(dtype).to(device))
        for _ in range(1, visible_num - electron_num):
            tensors.append(neg_site_tensor.type(dtype).to(device))
        for _ in range(visible_num - electron_num, visible_num - 1):
            tensors.append(site_tensor.type(dtype).to(device))
        tensors.append(right_cap.type(dtype).to(device))

        anti_hf_mps = MPS.from_tensor_list(tensors)
        anti_hf_mps.trim_bond_dims()

        return anti_hf_mps

    @staticmethod
    def alpha_mps(visible_num: int = None,
                  electron_num: int = None,
                  spin: int = 0,
                  dtype=None,
                  device=None):
        left_cap, site_tensor, right_cap = _e_num_symmetry(electron_num=(electron_num + 2 * spin) // 2,
                                                           dtype=dtype,
                                                           device=device)
        ones_diag = pt.ones(electron_num // 2 + 1,
                            dtype=dtype,
                            device=device)
        unity_tensor = pt.stack([pt.diag(ones_diag),
                                 pt.diag(ones_diag)],
                                dim=0)
        unity_tensor = unity_tensor.transpose(0, 1)

        alpha_list = list()
        alpha_list.append(left_cap)
        for idx in range(1, visible_num - 1):
            if (idx % 2) == 1:
                alpha_list.append(unity_tensor)
            else:
                alpha_list.append(site_tensor)
        a = pt.stack([right_cap[:, 0, :], right_cap[:, 0, :]],
                     dim=1)
        alpha_list.append(a)

        alpha_mps = MPS.from_tensor_list(alpha_list)
        alpha_mps.trim_bond_dims()

        return alpha_mps

    @staticmethod
    def beta_mps(visible_num: int = None,
                  electron_num: int = None,
                  spin: int = 0,
                  dtype=None,
                  device=None):
        left_cap, site_tensor, right_cap = _e_num_symmetry(electron_num=(electron_num - 2 * spin) // 2,
                                                           dtype=dtype,
                                                           device=device)
        ones_diag = pt.ones(electron_num // 2 + 1,
                            dtype=dtype,
                            device=device)
        unity_tensor = pt.stack([pt.diag(ones_diag),
                                 pt.diag(ones_diag)],
                                dim=0)
        unity_tensor = unity_tensor.transpose(0, 1)

        beta_list = list()
        b = pt.stack([left_cap[:, 0, :], left_cap[:, 0, :]],
                     dim=1)
        beta_list.append(b)
        for idx in range(1, visible_num - 1):
            if (idx % 2) == 1:
                beta_list.append(site_tensor)
            else:
                beta_list.append(unity_tensor)
        beta_list.append(right_cap)

        beta_mps = MPS.from_tensor_list(beta_list)
        beta_mps.trim_bond_dims()

        return beta_mps

    @staticmethod
    def e_num_mps(*,
                  visible_num: int = None,
                  electron_num: int = None,
                  dtype=None,
                  device=None):
        if electron_num is None:
            return None
        else:
            left_cap, site_tensor, right_cap = _e_num_symmetry(electron_num=electron_num,
                                                               dtype=dtype,
                                                               device=device)
            mps_tensors = [left_cap]
            for idx in range(1, visible_num - 1):
                mps_tensors.append(site_tensor)
            mps_tensors.append(right_cap)

            mps = MPS.from_tensor_list(mps_tensors)
            mps.canonicalise_to_centre()

            return mps

    @staticmethod
    def spin_mps(*,
                 visible_num: int = None,
                 spin: int = None,
                 dtype=None,
                 device=None):
        if spin is None:
            return None
        else:
            left_cap, odd_site_tensor, even_site_tensor, right_cap = _spin_symmetry(visible_num=visible_num,
                                                                                    spin=spin,
                                                                                    dtype=dtype,
                                                                                    device=device)
            mps_tensors = [left_cap]
            for idx in range(1, visible_num - 1):
                if (idx % 2) != 0:
                    mps_tensors.append(odd_site_tensor)
                else:
                    mps_tensors.append(even_site_tensor)
            mps_tensors.append(right_cap)

            mps = MPS.from_tensor_list(mps_tensors)
            mps.canonicalise_to_centre()

            return mps

    @staticmethod
    def sym_mps_to_mpo(mps: MPS = None):
        if mps is None:
            return None
        else:
            mpo = []
            for tensor in mps.tensors:
                assert tensor.shape[1] == 2
                zeros = pt.zeros((*tensor.shape, 2), dtype=tensor.dtype, device=mps.device)
                zeros[:, 0, :, 0] = tensor[:, 0, :]
                zeros[:, 1, :, 1] = tensor[:, 1, :]
                mpo.append(zeros)

            return mpo

    @staticmethod
    def mpo_mps_to_mpo(mps,
                       phys_dim: int = None):
        assert pt.allclose(pt.tensor(mps.phys_dims), pt.tensor(phys_dim ** 2))
        mpo = []
        for idx in range(mps.visible_num):
            shape = mps.tensors[idx].shape
            mpo.append(pt.reshape(mps.tensors[idx],
                                  (shape[0],
                                   phys_dim,
                                   phys_dim,
                                   shape[-1])))
            mpo[idx] = pt.permute(mpo[idx], (0, 1, 3, 2))

        return mpo

    def clone(self):
        return MPS.from_tensor_list([pt.clone(tensor)
                                     for tensor in self.tensors],
                                    max_bond_dim=self.max_bond_dim,
                                    cutoff=self.cutoff)
