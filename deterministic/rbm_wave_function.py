from __future__ import annotations

import time

import numpy as np
import torch as pt
from torch import nn
from torch.autograd.functional import jacobian

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, lgmres

from typing import NewType, List, Tuple, Callable

from .logging import Logging

from .constants import BASE_COMPLEX_TYPE, BASE_INT_TYPE
from .constants import DEFAULT_INIT_SCALE

from .svd import pt_H, robust_qr
from .constants import DEFAULT_MAX_BOND_DIM, DEFAULT_CUTOFF
from .constants import DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import DEFAULT_SR_REGULISER

from .mps import MPS

PEPS = NewType('PEPS', List[List[pt.Tensor]])


class RBMWaveFunction(Logging, nn.Module):
    ENERGY_CONV_TIME = 0.0
    ENERGY_DENOM_TIME = 0.0
    ENERGY_NUM_TIME = 0.0

    SR_KET_CONV_TIME = 0.0
    SR_BRA_CONV_TIME = 0.0
    SR_OVERLAP_TIME = 0.0
    SR_GRAD_TIME = 0.0
    SR_GRAD_GRAD_TIME = 0.0

    def __init__(self,
                 *,
                 name: str = None,
                 visible_num: int = -1,
                 hidden_num: int = -1,
                 dtype=BASE_COMPLEX_TYPE,
                 idx_dtype=BASE_INT_TYPE,
                 init_scale=DEFAULT_INIT_SCALE,
                 device=None,
                 is_param_vec_complex: bool = False,
                 grad_mask: pt.Tensor = None,
                 vb: pt.Tensor = None,
                 hb: pt.Tensor = None,
                 wm: pt.Tensor = None,
                 max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                 cutoff: float = DEFAULT_CUTOFF,
                 svd_backend: str = DEFAULT_SVD_BACKEND,
                 backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                 electron_num: int = None,
                 spin: int = None,
                 anti_hf: bool = False,
                 sr_reguliser: float = DEFAULT_SR_REGULISER):
        super(RBMWaveFunction, self).__init__()
        self.name_ = name

        self.visible_num = visible_num
        self.hidden_num = hidden_num
        self.param_num = self.visible_num + self.hidden_num + self.hidden_num * self.visible_num
        self.balanced_num = min(self.visible_num, self.hidden_num)

        self.dtype = dtype
        self.idx_dtype = idx_dtype
        assert device is not None
        self.device = device
        self.init_scale = pt.tensor(init_scale,
                                    dtype=self.dtype,
                                    device=self.device)

        self.is_param_vec_complex = is_param_vec_complex
        if grad_mask is None:
            if self.is_param_vec_complex:
                self.grad_mask = pt.ones((self.param_num, ), dtype=pt.bool)
            else:
                self.grad_mask = pt.ones((2 * self.param_num, ), dtype=pt.bool)
        else:
            self.grad_mask = grad_mask

        if vb is None:
            self.vb = nn.Parameter(self.init_scale * pt.randn((visible_num,),
                                                              dtype=self.dtype,
                                                              device=self.device))
        else:
            assert vb.shape == (visible_num,)
            assert vb.dtype == self.dtype
            if vb.requires_grad:
                self.vb = vb
            else:
                self.vb = nn.Parameter(vb)
        if hb is None:
            self.hb = nn.Parameter(self.init_scale * pt.randn((hidden_num,),
                                                              dtype=self.dtype,
                                                              device=self.device))
        else:
            assert hb.shape == (hidden_num,)
            assert hb.dtype == self.dtype
            if hb.requires_grad:
                self.hb = hb
            else:
                self.hb = nn.Parameter(hb)
        if wm is None:
            self.wm = nn.Parameter(self.init_scale * pt.randn((hidden_num, visible_num),
                                                              dtype=self.dtype,
                                                              device=self.device))
        else:
            assert wm.shape == (hidden_num, visible_num)
            assert wm.dtype == self.dtype
            if wm.requires_grad:
                self.wm = wm
            else:
                self.wm = nn.Parameter(wm)

        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff
        self.svd_backend = svd_backend
        self.backprop_backend = backprop_backend

        # Constants required for conversion to custom TensorNetwork
        self.empty_diag = pt.tensor([0.0, 1.0], dtype=self.dtype, device=self.device)
        self.empty_square = pt.tensor([[0.0, 0.0], [0.0, 1.0]],
                                      dtype=self.dtype,
                                      device=self.device)
        # Symmetries:
        self.electron_num = electron_num
        self.spin = spin
        self.anti_hf = anti_hf

        # SR wrapped functions:
        self.sr_reguliser = sr_reguliser

        # SR caches
        self.mean_log_grad_cache = None
        self.overlap_cache = None

    def __str__(self):
        return (f'RBMWaveFunction {self.name_}:\n'
                f'\tvisible_num: {self.visible_num}\n'
                f'\thidden_num: {self.hidden_num}\n'
                f'\tdtype: {self.dtype}\n'
                f'\tidx_dtype: {self.idx_dtype}\n'
                f'\tvb: {self.vb}\n'
                f'\thb: {self.hb}\n'
                f'\twm: {self.wm}\n')

    # ==========================================================================================
    # ====================== FUNCTIONS RELATED TO MPS-PEPS TRANSFORMATION ======================
    # ==========================================================================================
    def calc_bias_diag(self,
                       *,
                       bias_vector=None,
                       pos=None):
        result = pt.mul(self.empty_diag, bias_vector[pos])

        return pt.exp(result)

    def calc_weight_tensor(self,
                           *,
                           pos_0=None,
                           pos_1=None):
        result = pt.mul(self.empty_square, self.wm[pos_0, pos_1])

        return pt.exp(result)

    def to_mps_peps(self) -> PEPS:
        visible_tensors = []
        for idx in range(self.visible_num):
            visible_tensors.append([])
            cur_diag = self.calc_bias_diag(bias_vector=self.vb, pos=idx)
            cur_mps = MPS.diag_to_mps(name=f'v_{idx}',
                                      diag=cur_diag,
                                      visible_num=self.hidden_num + 1,
                                      max_bond_dim=self.max_bond_dim,
                                      cutoff=self.cutoff,
                                      svd_backend=self.svd_backend,
                                      backprop_backend=self.backprop_backend)
            for jdx in range(self.hidden_num):
                weight_tensor = self.calc_weight_tensor(pos_0=jdx, pos_1=idx)
                cur_mps.tensors[jdx] = pt.transpose(pt.tensordot(cur_mps.tensors[jdx],
                                                                 weight_tensor,
                                                                 dims=[[1], [0]]),
                                                    1, 2)
                visible_tensors[idx].append(cur_mps.tensors[jdx])
            visible_tensors[idx].append(cur_mps.tensors[-1])

        hidden_tensors = []
        for jdx in range(self.hidden_num):
            hidden_tensors.append([])
            cur_diag = self.calc_bias_diag(bias_vector=self.hb, pos=jdx)
            cur_mps = MPS.diag_to_mps(name=f'h_{jdx}',
                                      diag=cur_diag,
                                      visible_num=self.visible_num,
                                      max_bond_dim=self.max_bond_dim,
                                      cutoff=self.cutoff,
                                      svd_backend=self.svd_backend,
                                      backprop_backend=self.backprop_backend)
            for idx in range(self.visible_num):
                hidden_tensors[jdx].append(cur_mps.tensors[idx])

        peps = []
        for jdx in range(self.hidden_num):
            peps.append([])
            for idx in range(self.visible_num):
                peps[jdx].append(pt.transpose(pt.tensordot(visible_tensors[idx][jdx],
                                                           hidden_tensors[jdx][idx],
                                                           dims=[[1], [1]]),
                                              1, 2))
        for idx in range(self.visible_num):
            peps[-1][idx] = pt.transpose(pt.tensordot(peps[-1][idx],
                                                      pt.squeeze(visible_tensors[idx][-1]),
                                                      dims=[[2], [1]]),
                                         2, 3)

        return peps

    def mps_peps_to_mps(self,
                        *,
                        peps: PEPS = None):
        edge_tensors = []
        for idx in range(self.visible_num):
            edge_tensors.append(pt.squeeze(peps[0][idx], dim=0))
        mps = MPS(visible_num=self.visible_num,
                  phys_dims=[2] * self.visible_num,
                  bond_dims=[2] * (self.visible_num - 1),
                  device=self.device,
                  max_bond_dim=self.max_bond_dim,
                  cutoff=self.cutoff,
                  tensors=edge_tensors,
                  dtype=self.dtype)

        for jdx in range(1, self.hidden_num):
            for idx in range(self.visible_num):
                sum_tensor = pt.einsum('min,ikjl->mkjnl',
                                       mps.tensors[idx],
                                       peps[jdx][idx])
                shape = sum_tensor.shape
                sum_tensor = pt.reshape(sum_tensor,
                                        (shape[0] * shape[1],
                                         shape[2],
                                         shape[3] * shape[4]))
                assert mps.ext_bond_dims[idx] == sum_tensor.shape[0]
                mps.phys_dims[idx] = sum_tensor.shape[1]
                if idx < self.visible_num - 1:
                    mps.set_bond_dim(idx, sum_tensor.shape[2])
                else:
                    assert sum_tensor.shape[2] == 1
                mps.tensors[idx] = sum_tensor

            mps.canonicalise_to_centre()
            #print(f'Her')
            mps.cut_bond_dims()

        return mps

    def to_mps(self):
        return self.mps_peps_to_mps(peps=self.to_mps_peps())
    # ==========================================================================================
    # =================== FUNCTIONS RELATED TO AUTOGRAD CALCULATIONS ===========================
    # ==========================================================================================
    def to_param_vec(self):
        with pt.no_grad():
            if self.is_param_vec_complex:
                return pt.cat((self.vb, self.hb, pt.reshape(self.wm, (-1, ))))
            else:
                return pt.reshape(pt.view_as_real(pt.cat((self.vb,
                                                          self.hb,
                                                          pt.reshape(self.wm, (-1, ))))),
                                  (-1, ))

    def from_param_vec(self, param_vec: pt.Tensor = None):
        if self.is_param_vec_complex:
            assert param_vec.shape == (self.param_num, )
            del self.vb
            del self.hb
            del self.wm

            self.vb = param_vec[:self.visible_num]
            self.hb = param_vec[self.visible_num:self.visible_num + self.hidden_num]
            self.wm = pt.reshape(param_vec[self.visible_num + self.hidden_num:],
                                 (self.hidden_num, self.visible_num))
        else:
            assert param_vec.shape == (2 * self.param_num, )
            del self.vb
            del self.hb
            del self.wm

            cur_param_vec = pt.view_as_complex(pt.reshape(param_vec, (-1, 2)))
            self.vb = cur_param_vec[:self.visible_num]
            self.hb = cur_param_vec[self.visible_num:self.visible_num + self.hidden_num]
            self.wm = pt.reshape(cur_param_vec[self.visible_num + self.hidden_num:],
                                 (self.hidden_num, self.visible_num))

    @staticmethod
    def from_param_vec_static(*,
                              name: str = None,
                              visible_num: int = None,
                              hidden_num: int = None,
                              device=None,
                              param_vec: pt.Tensor = None,
                              is_param_vec_complex: bool = False,
                              grad_mask: pt.Tensor = None,
                              max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                              cutoff: float = DEFAULT_CUTOFF,
                              svd_backend: str = DEFAULT_SVD_BACKEND,
                              backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                              electron_num: int = None,
                              spin: int = None,
                              anti_hf: bool = False):
        param_num = visible_num + hidden_num + hidden_num * visible_num

        if is_param_vec_complex:
            assert param_vec.shape == (param_num, )
            cur_param_vec = param_vec
        else:
            assert param_vec.shape == (2 * param_num,)
            cur_param_vec = pt.view_as_complex(pt.reshape(param_vec, (-1, 2)))

        vb = cur_param_vec[:visible_num]
        hb = cur_param_vec[visible_num:(visible_num + hidden_num)]
        wm = pt.reshape(cur_param_vec[(visible_num + hidden_num):], (hidden_num, visible_num))

        return RBMWaveFunction(name=name,
                               visible_num=visible_num,
                               hidden_num=hidden_num,
                               dtype=cur_param_vec.dtype,
                               device=device if device is not None else param_vec.device,
                               is_param_vec_complex=is_param_vec_complex,
                               grad_mask=grad_mask,
                               vb=vb,
                               hb=hb,
                               wm=wm,
                               max_bond_dim=max_bond_dim,
                               cutoff=cutoff,
                               svd_backend=svd_backend,
                               backprop_backend=backprop_backend,
                               electron_num=electron_num,
                               spin=spin,
                               anti_hf=anti_hf)

    @staticmethod
    def masked_parts_to_param_vec(grad_part: pt.Tensor = None,  
                                  non_grad_part: pt.Tensor = None,
                                  grad_mask: pt.Tensor = None):
        assert grad_part.device == non_grad_part.device
        param_num = grad_part.shape[0] + non_grad_part.shape[0]
        indices = pt.arange(param_num).to(grad_part.device)
        param_vec = pt.zeros((param_num,), dtype=grad_part.dtype, device=grad_part.device)
        if grad_part.shape[0]:
            param_vec = pt.scatter(param_vec, 0, indices[grad_mask], grad_part)
        if non_grad_part.shape[0]:
            param_vec = pt.scatter(param_vec, 0, indices[pt.logical_not(grad_mask)], non_grad_part)

        return param_vec

    @staticmethod
    def param_vec_to_masked_parts(param_vec: pt.Tensor = None,
                                  grad_mask: pt.Tensor = None):

        return param_vec[grad_mask], param_vec[pt.logical_not(grad_mask)].detach()

    @staticmethod
    def param_vec_overlap_core(*,
                               grad_bra_param_vec: pt.Tensor = None,
                               grad_ket_param_vec: pt.Tensor = None,
                               non_grad_bra_param_vec: pt.Tensor = None,
                               non_grad_ket_param_vec: pt.Tensor = None,
                               grad_mask: pt.Tensor = None,
                               visible_num: int = None,
                               hidden_num: int = None,
                               is_param_vec_complex: bool = False,
                               device=None,
                               max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                               cutoff: float = DEFAULT_CUTOFF,
                               svd_backend: str = DEFAULT_SVD_BACKEND,
                               backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                               electron_num: int = None,
                               spin: int = None,
                               anti_hf: bool = False,
                               canonicalise: bool = True,
                               wf: RBMWaveFunction = None):
        bra_param_vec = RBMWaveFunction.masked_parts_to_param_vec(grad_bra_param_vec,
                                                                  non_grad_bra_param_vec,
                                                                  grad_mask)
        ket_param_vec = RBMWaveFunction.masked_parts_to_param_vec(grad_ket_param_vec,
                                                                  non_grad_ket_param_vec,
                                                                  grad_mask)
        bra = RBMWaveFunction.from_param_vec_static(visible_num=visible_num,
                                                    hidden_num=hidden_num,
                                                    is_param_vec_complex=is_param_vec_complex,
                                                    grad_mask=grad_mask,
                                                    device=device,
                                                    param_vec=bra_param_vec,
                                                    max_bond_dim=max_bond_dim,
                                                    cutoff=cutoff,
                                                    svd_backend=svd_backend,
                                                    backprop_backend=backprop_backend,
                                                    electron_num=electron_num,
                                                    spin=spin,
                                                    anti_hf=anti_hf)
        ket = RBMWaveFunction.from_param_vec_static(visible_num=visible_num,
                                                    hidden_num=hidden_num,
                                                    is_param_vec_complex=is_param_vec_complex,
                                                    grad_mask=grad_mask,
                                                    device=device,
                                                    param_vec=ket_param_vec,
                                                    max_bond_dim=max_bond_dim,
                                                    cutoff=cutoff,
                                                    svd_backend=svd_backend,
                                                    backprop_backend=backprop_backend,
                                                    electron_num=electron_num,
                                                    spin=spin,
                                                    anti_hf=anti_hf)
        bra_mps = bra.to_mps()
        bra_mps = bra_mps.conj()

        ket_mps = ket.to_mps()
        sym_mpo = MPS.sym_mps_to_mpo(ket.sym_mps)
        ket_mps.apply_symmetry_mpo(sym_mpo=sym_mpo)
        print(f'Ket after the symmetry: {ket_mps}')
        if canonicalise:
            print(f'We do canonicalise with robust QR')
            ket_mps.canonicalise_to_centre(qr_lambda=lambda x: robust_qr(matrix=x,
                                                                         max_bond_dim=None,
                                                                         svd_backend=ket.svd_backend,
                                                                         backprop_backend=ket.backprop_backend))
            print(f'Ket after the canonicalisation: {ket_mps}')

        else:
            print(f'We do not canonicalise')

        wf.overlap_cache = MPS.batched_overlap(bra=bra_mps, ket=ket_mps)
        if is_param_vec_complex is False:
            wf.overlap_cache = pt.abs(wf.overlap_cache)

        return wf.overlap_cache

    def param_vec_overlap(self,
                          grad_bra_param_vec: pt.Tensor = None,
                          grad_ket_param_vec: pt.Tensor = None,
                          non_grad_bra_param_vec: pt.Tensor = None,
                          non_grad_ket_param_vec: pt.Tensor = None,
                          canonicalise: bool = True) -> pt.Tensor:
        return RBMWaveFunction.param_vec_overlap_core(grad_bra_param_vec=grad_bra_param_vec,
                                                      grad_ket_param_vec=grad_ket_param_vec,
                                                      non_grad_bra_param_vec=non_grad_bra_param_vec,
                                                      non_grad_ket_param_vec=non_grad_ket_param_vec,
                                                      grad_mask=self.grad_mask,
                                                      visible_num=self.visible_num,
                                                      hidden_num=self.hidden_num,
                                                      is_param_vec_complex=self.is_param_vec_complex,
                                                      device=self.device,
                                                      max_bond_dim=self.max_bond_dim,
                                                      cutoff=self.cutoff,
                                                      svd_backend=self.svd_backend,
                                                      backprop_backend=self.backprop_backend,
                                                      electron_num=self.electron_num,
                                                      spin=self.spin,
                                                      anti_hf=self.anti_hf,
                                                      canonicalise=canonicalise,
                                                      wf=self)

    @staticmethod
    def mean_log_grad(*,
                      grad_param_vec: pt.Tensor = None,
                      non_grad_param_vec: pt.Tensor = None,
                      param_vec_overlap: Callable = None,
                      canonicalise: bool = False,
                      wf: RBMWaveFunction = None):
        grad_bra_param_vec, non_grad_bra_param_vec = grad_param_vec, non_grad_param_vec
        grad_ket_param_vec, non_grad_ket_param_vec = grad_param_vec, non_grad_param_vec
        wf.mean_log_grad_cache = pt.conj(jacobian(lambda x: param_vec_overlap(grad_bra_param_vec,
                                                                              x,
                                                                              non_grad_bra_param_vec,
                                                                              non_grad_ket_param_vec,
                                                                              canonicalise=canonicalise),
                                                  grad_ket_param_vec.detach(),
                                                  create_graph=True,
                                                  vectorize=True))

        return wf.mean_log_grad_cache

    def sr_matrix(self,
                  canonicalise: bool = False):
        param_vec = self.to_param_vec()
        param_vec = param_vec.detach()
        grad_param_vec, non_grad_param_vec = RBMWaveFunction.param_vec_to_masked_parts(param_vec, self.grad_mask)
        mean_of_prods = jacobian(lambda x: RBMWaveFunction.mean_log_grad(grad_param_vec=x,
                                                                         non_grad_param_vec=non_grad_param_vec,
                                                                         param_vec_overlap=self.param_vec_overlap,
                                                                         canonicalise=canonicalise,
                                                                         wf=self),
                                 grad_param_vec,
                                 vectorize=True)
        denom = self.overlap_cache
        mean_log_grad = self.mean_log_grad_cache
        return (mean_of_prods - pt.einsum('i,j->ij', mean_log_grad, pt.conj(mean_log_grad)) / denom) / denom

    def sr_grad(self, grad: pt.Tensor = None):
        assert grad.shape == (self.param_num,)
        x_pt = self.to_param_vec()
        x_pt = pt.from_numpy(x_pt.detach().numpy())
        x_pt_copy = pt.from_numpy(x_pt.numpy())
        x_pt.requires_grad = True
        x_pt_copy.requires_grad = True

        overlap = self.param_vec_overlap(x_pt, x_pt_copy)
        start_time = time.time()
        overlap.backward(create_graph=True)
        RBMWaveFunction.SR_GRAD_TIME += time.time() - start_time
        mean_log_grad = x_pt_copy.grad.detach() / overlap.detach()

        def matvec(p):
            clone_x_pt_copy_grad = pt.clone(x_pt_copy.grad)

            jvp = pt.dot(p, x_pt_copy.grad) / overlap.detach()
            x_pt.grad.data.zero_()
            with pt.no_grad():
                start_time = time.time()
                jvp.backward(retain_graph=True)
                RBMWaveFunction.SR_GRAD_GRAD_TIME += time.time() - start_time
            x_pt_copy.grad = clone_x_pt_copy_grad

            return (pt.conj(x_pt.grad) - pt.conj(mean_log_grad) * jvp.detach()).detach()

        cg_iter_num = 0
        linear_operator = LinearOperator((self.param_num, self.param_num),
                                         matvec=lambda v: matvec(pt.from_numpy(v)).detach().numpy() + self.sr_reguliser * v,
                                         dtype=np.complex128)

        def count_inc(x):
            nonlocal cg_iter_num
            print(f'CG iteration #{cg_iter_num + 1}')
            cg_iter_num += 1

        result = pt.from_numpy(cg(linear_operator, grad.detach().numpy(),
                                  callback=count_inc)[0])
        x_pt.grad = x_pt.grad.detach()
        x_pt_copy.grad = x_pt_copy.grad.detach()
        print(f'Number of CG iterations: {cg_iter_num}')
        return result

    # ==========================================================================================
    # ================ FUNCTIONS RELATED TO RBM MARGINALISED OVER HIDDEN NODES =================
    # ==========================================================================================
    # noinspection DuplicatedCode
    def idx_to_visible(self, idx: pt.Tensor) -> pt.Tensor:
        if not pt.is_tensor(idx):
            idx = pt.tensor(idx, dtype=self.idx_dtype, device=self.device)
        shifts = pt.arange(0, self.visible_num, dtype=self.idx_dtype, device=self.device)
        visible = pt.squeeze(pt.remainder((idx.reshape((-1, 1)) >> shifts), 2))

        return visible

    # noinspection DuplicatedCode
    def visible_to_idx(self, visible: pt.Tensor) -> pt.Tensor:
        if not pt.is_tensor(visible):
            visible = pt.tensor(visible, dtype=self.idx_dtype, device=self.device)
        if visible.dtype != self.idx_dtype:
            visible = visible.type(self.idx_dtype)
        if visible.device != self.device:
            visible = visible.to(self.device)

        two_powers = 2 ** pt.arange(self.visible_num,
                                    dtype=self.idx_dtype,
                                    device=self.device)

        return pt.tensordot(visible, two_powers, dims=1)

    # noinspection DuplicatedCode
    def visible_to_amplitude(self, visible: pt.Tensor) -> pt.Tensor:
        assert visible.shape[-1] == self.visible_num
        if len(visible.shape) == 1:
            visible = pt.unsqueeze(visible, dim=0)
        if visible.dtype != self.dtype:
            visible = visible.type(self.dtype)
        if visible.device != self.device:
            visible = visible.to(self.device)

        return pt.mul(pt.exp(pt.mv(visible, self.vb)),
                      pt.exp(pt.sum(pt.log(pt.add(1.0, pt.exp(pt.add(pt.matmul(visible,
                                                                               self.wm.t()),
                                                                     self.hb)))),
                                    dim=-1)))

    def amplitude(self, idx):
        return self.visible_to_amplitude(self.idx_to_visible(idx))

    def to_state_vector(self):
        return self.visible_to_amplitude(self.idx_to_visible(pt.arange(2 ** self.visible_num,
                                                                       device=self.device)))

    def part_func_bf(self):
        return pt.sum(self.to_state_vector())

    def norm_bf(self):
        return pt.linalg.norm(self.to_state_vector())

    # ==========================================================================================
    # ================================= FULL RBM BRUTE FORCE SR  ===============================
    # ==========================================================================================
    @staticmethod
    def sigmoid(x):
        return pt.div(pt.exp(x), pt.add(1, pt.exp(x)))

    def visible_to_hidden(self, visible: pt.Tensor) -> pt.Tensor:
        if visible.dtype != self.dtype:
            visible = visible.type(self.dtype)

        return self.sigmoid(pt.add(pt.tensordot(visible,
                                                self.wm,
                                                dims=[[1], [1]]),
                                   self.hb))

    def vb_log_grad(self, visible):

        return visible.type(self.dtype)

    def hb_log_grad(self, visible):

        return self.visible_to_hidden(visible)

    def wm_log_grad(self, visible):
        hidden = self.visible_to_hidden(visible)

        return pt.einsum('sj,si->sji', hidden, visible.type(self.dtype))

    def log_grad(self, visible):
        vb_log_grad = self.vb_log_grad(visible)
        hb_log_grad = self.hb_log_grad(visible)
        wm_log_grad = self.wm_log_grad(visible).reshape((visible.shape[0], -1))

        log_grad = pt.cat([vb_log_grad, hb_log_grad, wm_log_grad], dim=1)

        return log_grad

    def full_rbm_sr_matrix(self, visible):
        amplitude = self.visible_to_amplitude(visible)
        probs = pt.mul(pt_H(amplitude), amplitude)
        probs = pt.div(probs, pt.sum(probs))
        log_grad = self.log_grad(visible)

        mean_log_grad = pt.einsum('si,s', log_grad, probs)
        prod_of_means = pt.einsum('i,j->ij',
                                  mean_log_grad,
                                  pt_H(mean_log_grad))
        mean_of_prods = pt.einsum('si,sj,s->ij',
                                  log_grad,
                                  pt.conj(log_grad),
                                  probs)

        return mean_of_prods - prod_of_means

    # ==========================================================================================
    # ============================ ONCE DIFFERENTIATED BRUTE FORCE SR  =========================
    # ==========================================================================================
    @staticmethod
    def mps_amplitude_core(*,
                           visible_idx: pt.Tensor = None,
                           param_vec: pt.Tensor = None,
                           visible_num: int = None,
                           hidden_num: int = None,
                           device=None,
                           max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
                           cutoff: float = DEFAULT_CUTOFF,
                           svd_backend: str = DEFAULT_SVD_BACKEND,
                           backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
                           electron_num: int = None,
                           spin: int = None):
        wf = RBMWaveFunction.from_param_vec_static(visible_num=visible_num,
                                                   hidden_num=hidden_num,
                                                   device=device,
                                                   param_vec=param_vec,
                                                   max_bond_dim=max_bond_dim,
                                                   cutoff=cutoff,
                                                   svd_backend=svd_backend,
                                                   backprop_backend=backprop_backend,
                                                   electron_num=electron_num,
                                                   spin=spin)

        mps = wf.to_mps()
        return mps.amplitude(visible_idx)

    def mps_amplitude(self, *, visible_idx: pt.Tensor, param_vec: pt.Tensor) -> pt.Tensor:
        return RBMWaveFunction.mps_amplitude_core(visible_idx=visible_idx,
                                                  param_vec=param_vec,
                                                  visible_num=self.visible_num,
                                                  hidden_num=self.hidden_num,
                                                  device=self.device,
                                                  max_bond_dim=self.max_bond_dim,
                                                  cutoff=self.cutoff,
                                                  svd_backend=self.svd_backend,
                                                  backprop_backend=self.backprop_backend,
                                                  electron_num=self.electron_num,
                                                  spin=self.spin)

    def bf_sr_matrix(self, visible_idx):
        param_vec = self.to_param_vec()
        param_vec = param_vec.detach()
        grad = jacobian(lambda x: self.mps_amplitude(visible_idx=visible_idx, param_vec=x),
                        param_vec,
                        vectorize=True)
        grad = pt.conj(grad)

        amps = self.mps_amplitude(visible_idx=visible_idx, param_vec=param_vec)
        probs = pt.mul(pt.conj(amps), amps)
        probs = pt.div(probs, pt.sum(probs))
        log_grad = grad / pt.unsqueeze(amps, dim=-1)

        mean_log_grad = pt.einsum('si,s', log_grad, probs)
        prod_of_means = pt.einsum('i,j->ij',
                                  mean_log_grad,
                                  pt_H(mean_log_grad))
        mean_of_prods = pt.einsum('si,sj,s->ij',
                                  log_grad,
                                  pt.conj(log_grad),
                                  probs)

        return mean_of_prods - prod_of_means
