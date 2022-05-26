import torch as pt
import numpy as np
from scipy.sparse import csr_matrix

DEFAULT_COMPLEX_TYPE = pt.cdouble

PAULI: dict = {'X': pt.tensor([[0, 1], [1, 0]], dtype=DEFAULT_COMPLEX_TYPE),
               'Y': pt.tensor([[0, -1j], [1j, 0]], dtype=DEFAULT_COMPLEX_TYPE),
               'Z': pt.tensor([[1, 0], [0, -1]], dtype=DEFAULT_COMPLEX_TYPE),
               'I': pt.tensor([[1, 0], [0, 1]], dtype=DEFAULT_COMPLEX_TYPE)
               }
# rotate to pauli basis
PAULI_ROT: dict = {'X': 1 / pt.sqrt(pt.tensor([2])) * pt.tensor([[1, 1], [1, -1]], dtype=DEFAULT_COMPLEX_TYPE),
                   'Y': 1 / pt.sqrt(pt.tensor([2])) * pt.tensor([[1, -1j], [1, 1j]], dtype=DEFAULT_COMPLEX_TYPE),
                   'Z': pt.tensor([[1, 0], [0, 1]], dtype=DEFAULT_COMPLEX_TYPE)}

PAULI_ROT_SPARSE: dict = {'X': 1 / np.sqrt(2) * csr_matrix([[1, 1], [1, -1]]),
                          'Y': 1 / np.sqrt(2) * csr_matrix([[1, -1j], [1, 1j]]),
                          'Z': csr_matrix([[1, 0], [0, 1]]),
                          'I': csr_matrix([[1, 0], [0, 1]])}

CLIFFORD: dict = {'X': pt.tensor([[0, 1], [1, 0]], dtype=DEFAULT_COMPLEX_TYPE),
                  'Y': pt.tensor([[0, -1j], [1j, 0]], dtype=DEFAULT_COMPLEX_TYPE),
                  'Z': pt.tensor([[1, 0], [0, -1]], dtype=DEFAULT_COMPLEX_TYPE),
                  'I': pt.tensor([[1, 0], [0, 1]], dtype=DEFAULT_COMPLEX_TYPE),
                  'H': 1 / pt.sqrt(pt.tensor([2])) * pt.tensor([[1, 1], [1, -1]], dtype=DEFAULT_COMPLEX_TYPE),
                  'S': pt.tensor([[1, 0], [0, 1j]], dtype=DEFAULT_COMPLEX_TYPE)}

DEFAULT_MOLECULE_ROOT = './../molecules'
CHEMICAL_ACCURACY = 1.6e-3
BASE_INT_TYPE = pt.int32
BASE_REAL_TYPE = pt.double
BASE_COMPLEX_TYPE = pt.cdouble

DEFAULT_INIT_SCALE = 0.001

BACKENDS = ('TORCH', 'SCIPY')

DEFAULT_SVD_BACKEND = 'TORCH'
DEFAULT_BACKPROP_BACKEND = 'TORCH'

DEFAULT_MAX_BOND_DIM = 100
DEFAULT_CUTOFF = 1e-15
DEFAULT_LORENTZIAN = 1e-12

DEFAULT_SR_REGULISER = 1e-2
