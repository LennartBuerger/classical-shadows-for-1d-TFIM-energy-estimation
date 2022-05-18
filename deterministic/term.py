import numpy as np
import torch as pt

from typing import List, Tuple

from .constants import BASE_COMPLEX_TYPE


class Term:
    def __init__(self,
                 *,
                 dtype=BASE_COMPLEX_TYPE,
                 device=None,
                 weight: float = None,
                 visible_num: int = None,
                 qubit_ops: List[Tuple[int, str]] = None):
        self.dtype = dtype
        assert device is not None
        self.device = device

        assert weight is not None
        self.weight = pt.tensor(weight, dtype=dtype)

        assert visible_num is not None
        self.visible_num = visible_num
        vb_add = np.zeros((visible_num,), dtype=np.complex128)
        vb_mul = np.ones((visible_num,), dtype=np.complex128)

        wm_mul = np.ones((visible_num,), dtype=np.complex128)

        renorm = 1.0

        str_repr = ['I'] * visible_num
        self.qubit_ops = qubit_ops
        for qubit_op in self.qubit_ops:
            qubit_idx = self.visible_num - qubit_op[0] - 1
            pauli = qubit_op[1]

            if pauli == 'Z':
                vb_add[qubit_idx] -= 0.5j * np.pi
                renorm *= np.exp(0.5j * np.pi)
                str_repr[qubit_idx] = 'Z'
            elif pauli == 'X':
                vb_mul[qubit_idx] *= -1.0
                wm_mul[qubit_idx] *= -1.0
                str_repr[qubit_idx] = 'X'
            elif pauli == 'Y':
                
                vb_add[qubit_idx] += 0.5j * np.pi
                vb_mul[qubit_idx] *= -1.0
                wm_mul[qubit_idx] *= -1.0
                str_repr[qubit_idx] = 'Y'
            else:
                raise ValueError(f'Wrong Pauli operation: {pauli}')

        self.vb_add = pt.tensor(vb_add,
                                dtype=self.dtype,
                                device=self.device)
        self.vb_mul = pt.tensor(vb_mul,
                                dtype=self.dtype,
                                device=self.device)
        self.wm_mul = pt.tensor(wm_mul,
                                dtype=self.dtype,
                                device=self.device)
        self.renorm = pt.tensor(renorm,
                                dtype=self.dtype,
                                device=self.device)
        self.str_repr = ''.join(str_repr)

    def __str__(self):
        return (f'Term {self.qubit_ops}:\n'
                f'\tweight: {self.weight}\n'
                f'\tvisible_num: {self.visible_num}\n'
                f'\tdtype: {self.dtype}\n'
                f'\tvb_add: {self.vb_add}\n'
                f'\tvb_mul: {self.vb_mul}\n'
                f'\twm_mul: {self.wm_mul}\n'
                f'\trenorm: {self.renorm}\n'
                f'\tstr_repr: {self.str_repr}\n')

    def full_space_action(self):
        perm = pt.arange(2 ** self.visible_num)
        indices = pt.arange(2 ** self.visible_num)
        phase = pt.ones((2 ** self.visible_num, ),
                        dtype=self.dtype,
                        device=self.device)
        for qubit_op in self.qubit_ops:
            qubit_idx = self.visible_num - qubit_op[0] - 1
            pauli = qubit_op[1]

            if pauli == 'X':
                perm = pt.bitwise_xor(perm,
                                      pt.tensor(2 ** qubit_idx))
            elif pauli == 'Y':
                perm = pt.bitwise_xor(perm,
                                      pt.tensor(2 ** qubit_idx))
                cur_phase = pt.bitwise_and(indices, pt.tensor(2 ** qubit_idx)) >> qubit_idx
                phase = pt.mul(phase, 1j * pt.tensor(1 - 2 * cur_phase).type(self.dtype))
            elif pauli == 'Z':
                cur_phase = pt.bitwise_and(indices, pt.tensor(2 ** qubit_idx)) >> qubit_idx
                phase = pt.mul(phase, pt.tensor(1 - 2 * cur_phase).type(self.dtype))

        return perm, phase
