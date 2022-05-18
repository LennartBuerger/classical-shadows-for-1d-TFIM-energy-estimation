import logging

import torch as pt

import time

from typing import Tuple

from .constants import BACKENDS
from .constants import DEFAULT_SVD_BACKEND, DEFAULT_BACKPROP_BACKEND
from .constants import DEFAULT_MAX_BOND_DIM, DEFAULT_CUTOFF, DEFAULT_LORENTZIAN


def pt_H(x: pt.Tensor) -> pt.Tensor:
    return pt.conj(pt.t(x))


def lorentz_inverse(x, lorentzian: float = DEFAULT_LORENTZIAN):
    if lorentzian is None:
        return pt.div(1, x)
    else:
        return pt.div(x, pt.mul(x, x) + DEFAULT_LORENTZIAN)


def inverse(x):
    return pt.div(1.0, x)


def pt_svd_grad(u: pt.Tensor,
                s: pt.Tensor,
                v: pt.Tensor,
                du: pt.Tensor,
                ds: pt.Tensor,
                dv: pt.Tensor,
                lorentzian: float = DEFAULT_LORENTZIAN) -> pt.Tensor:
    row_s = pt.unsqueeze(s, dim=1)
    singular_sub = pt.sub(pt.t(row_s), row_s)
    singular_add = pt.add(pt.t(row_s), row_s)

    excl_diag = pt.diag(pt.ones(s.shape, dtype=pt.bool).to(s.device))
    singular_sub = pt.where(excl_diag,
                            pt.ones_like(singular_sub).to(singular_sub.device),
                            singular_sub)
    singular_add = pt.where(excl_diag,
                            pt.ones_like(singular_add).to(singular_add.device),
                            singular_add)

    f_sub = pt.sub(lorentz_inverse(singular_sub, lorentzian),
                   lorentz_inverse(singular_add, lorentzian))

    singular_add = pt.where(excl_diag,
                            -pt.ones_like(singular_add).to(singular_add.device),
                            singular_add)

    f_add = pt.add(lorentz_inverse(singular_sub, lorentzian),
                   lorentz_inverse(singular_add, lorentzian))

    term_1 = 0.5 * (u @ (pt.mul(f_add, pt_H(u) @ du - pt_H(du) @ u) + pt.mul(f_sub, pt_H(v) @ dv - pt_H(dv) @ v)) @ pt_H(v))
    term_2 = u @ pt.diag(ds) @ pt_H(v)
    term_3 = (pt.eye(u.shape[0], dtype=u.dtype, device=u.device) - u @ pt_H(u)) @ du @ pt.diag(lorentz_inverse(s, lorentzian)) @ pt_H(v)
    term_4 = u @ pt.diag(lorentz_inverse(s, lorentzian)) @ pt_H(dv) @ (pt.eye(v.shape[0], dtype=v.dtype, device=v.device) - v @ pt_H(v))

    da = pt.add(term_1, pt.add(term_2, pt.add(term_3, term_4)))

    return da


class TruncSVD(pt.autograd.Function):
    BACKEND = DEFAULT_SVD_BACKEND
    BACKPROP_BACKEND = DEFAULT_BACKPROP_BACKEND

    MAX_BOND_DIM = None
    CUTOFF = DEFAULT_CUTOFF
    LORENTZIAN = None

    call_string = None

    @staticmethod
    def forward(ctx, a: pt.Tensor = None) -> Tuple[pt.Tensor, ...]:
        assert len(a.shape) == 2
        assert TruncSVD.BACKEND in BACKENDS

        if TruncSVD.BACKEND == 'TORCH':
            device = a.device
            u, s, v_h = pt.linalg.svd(a.to(pt.device('cpu')), full_matrices=False)
            #u, s, v_h = pt.linalg.svd(a, full_matrices=False)
            a = a.to(device)
            u = u.to(device)
            s = s.to(device)
            v_h = v_h.to(device)
        else:
            raise ValueError(f'Wrong backend {TruncSVD.BACKEND} for the SVD. Choose one of {BACKENDS}')
        s = s.type(a.dtype)
        v = pt_H(v_h)

        cutoff_dim = pt.masked_select(s, pt.greater(pt.abs(s) / pt.abs(pt.norm(s)), TruncSVD.CUTOFF)).shape[0]
        # cutoff_dim = pt.masked_select(s, pt.greater(pt.abs(s), TruncSVD.CUTOFF)).shape[0]

        new_bond_dim = min(TruncSVD.MAX_BOND_DIM, cutoff_dim) if TruncSVD.MAX_BOND_DIM is not None else cutoff_dim
        if new_bond_dim == 0:
            logger = logging.getLogger(f'nnqs.MPS')
            logger.warning(f'Zero new_bond_dim encountered during truncated_svd')
            new_bond_dim = 1

        s = s[:new_bond_dim]
        u = u[:, :new_bond_dim]
        v = v[:, :new_bond_dim]

        ctx.save_for_backward(u, s, v)
        ctx.call_string = TruncSVD.call_string
        ctx.lorentzian = TruncSVD.LORENTZIAN

        return u, s, v

    @staticmethod
    def backward(ctx, *grad_outputs) -> pt.Tensor:
        u, s, v = ctx.saved_tensors
        TruncSVD.LORENTZIAN = ctx.lorentzian
        du, ds, dv = grad_outputs
        if TruncSVD.BACKPROP_BACKEND == 'TORCH':
            # pt.cuda.synchronize()
            # loc_time = time.time()
            da = pt_svd_grad(u, s, v, du, ds, dv, lorentzian=TruncSVD.LORENTZIAN)
            # print(f'SVD backprop time = {time.time() - loc_time}')
            # print(f'shapes = {[t.shape for t in (da, du, ds, dv, u, s, v)]}')
            # pt.cuda.synchronize()
        else:
            raise ValueError(f'Wrong backend {TruncSVD.BACKPROP_BACKEND} for the SVD backprop. Choose one of {BACKENDS}')

        return da


def svd(*,
        matrix: pt.Tensor = None,
        svd_backend: str = DEFAULT_SVD_BACKEND,
        backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
        lorentzian: float = None,
        string: str = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    TruncSVD.BACKEND = svd_backend
    TruncSVD.BACKPROP_BACKEND = backprop_backend
    TruncSVD.MAX_BOND_DIM = None
    TruncSVD.LORENTZIAN = lorentzian
    TruncSVD.call_string = string
    result = TruncSVD.apply(matrix)
    TruncSVD.MAX_BOND_DIM = None
    TruncSVD.LORENTZIAN = None

    return result


def trunc_svd(*,
              matrix: pt.Tensor = None,
              max_bond_dim: int = None,
              cutoff: float = DEFAULT_CUTOFF,
              svd_backend: str = DEFAULT_SVD_BACKEND,
              backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
              lorentzian: float = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    TruncSVD.BACKEND = svd_backend
    TruncSVD.BACKPROP_BACKEND = backprop_backend
    TruncSVD.MAX_BOND_DIM = max_bond_dim
    TruncSVD.CUTOFF = cutoff
    TruncSVD.LORENTZIAN = lorentzian
    u, s, v = TruncSVD.apply(matrix)
    TruncSVD.MAX_BOND_DIM = None
    TruncSVD.CUTOFF = DEFAULT_CUTOFF
    TruncSVD.LORENTZIAN = None

    return u, s, v


def robust_qr(*,
              matrix: pt.Tensor,
              max_bond_dim: int = None,
              svd_backend: str = DEFAULT_SVD_BACKEND,
              backprop_backend: str = DEFAULT_BACKPROP_BACKEND,
              lorentzian: float = DEFAULT_LORENTZIAN):
    TruncSVD.BACKEND = svd_backend
    TruncSVD.BACKPROP_BACKEND = backprop_backend
    u, s, v = trunc_svd(matrix=matrix,
                        max_bond_dim=max_bond_dim,
                        svd_backend=svd_backend,
                        backprop_backend=backprop_backend,
                        lorentzian=lorentzian)

    return u, pt.diag(s) @ pt_H(v)
