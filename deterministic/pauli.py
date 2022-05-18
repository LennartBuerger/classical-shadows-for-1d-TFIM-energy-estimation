import torch as pt

from .constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE

PAULI_CHARS = ('I', 'X', 'Y', 'Z', '0')


REAL_PAULI_MATRICES = {
    'I': pt.tensor([[1, 0], [0, 1]], dtype=BASE_REAL_TYPE),
    'X': pt.tensor([[0, 1], [1, 0]], dtype=BASE_REAL_TYPE),
    'Y': pt.tensor([[0, -1], [1, 0]], dtype=BASE_REAL_TYPE),
    'Z': pt.tensor([[1, 0], [0, -1]], dtype=BASE_REAL_TYPE)
}


COMPLEX_PAULI_MATRICES = {
    'I': pt.tensor([[1, 0], [0, 1]], dtype=BASE_COMPLEX_TYPE),
    'X': pt.tensor([[0, 1], [1, 0]], dtype=BASE_COMPLEX_TYPE),
    'Y': pt.tensor([[0, -1j], [1j, 0]], dtype=BASE_COMPLEX_TYPE),
    'Z': pt.tensor([[1, 0], [0, -1]], dtype=BASE_COMPLEX_TYPE)
}
