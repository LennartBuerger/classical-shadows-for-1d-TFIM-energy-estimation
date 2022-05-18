import time

import numpy as np
import torch as pt


from typing import Tuple, List

from openfermion.ops import QubitOperator

from .logging import Logging
from .timed import timed

from .constants import BASE_COMPLEX_TYPE
from .rbm_wave_function import RBMWaveFunction
from .mps import MPS


class BatchHamiltonian(Logging):
    PAULI_OP_TO_IDX = {
        'X': 1,
        'Y': 2,
        'Z': 3,
    }
    TIMING_DICT = {}

    def __init__(self,
                 *,
                 visible_num: int = None,
                 of_hamiltonian: QubitOperator = None,
                 batch_terms: List[pt.Tensor] = None,
                 weights: pt.Tensor = None,
                 dtype=BASE_COMPLEX_TYPE,
                 device=None):
        super(BatchHamiltonian, self).__init__()
        assert visible_num is not None
        self.visible_num = visible_num

        self.dtype = dtype
        assert device is not None
        self.device = device

        self.pauli_stack = pt.tensor([[[1, 0],
                                       [0, 1]],
                                      [[0, 1],
                                       [1, 0]],
                                      [[0, -1j],
                                       [1j, 0]],
                                      [[1, 0],
                                       [0, -1]]],
                                     dtype=self.dtype,
                                     device=self.device)

        if of_hamiltonian is not None:
            assert batch_terms is None
            assert weights is None
            self.term_num = len(of_hamiltonian.terms)
            self.batch_terms, self.weights = self.parse_of_hamiltonian(of_hamiltonian=of_hamiltonian)
            self.of_hamiltonian = of_hamiltonian
        else:
            assert batch_terms is not None
            assert weights is not None
            self.term_num = weights.shape[0]
            self.batch_terms = [batch_term.type(self.dtype).to(self.device) for batch_term in batch_terms]
            self.weights = weights.type(self.dtype).to(self.device)
            self.of_hamiltonian = None

    def parse_of_hamiltonian(self, *, of_hamiltonian: QubitOperator = None) -> Tuple[list, pt.Tensor]:
        term_masks = np.zeros((self.visible_num, self.term_num), dtype=int)
        weights = []
        for term_idx, (pauli_ops, weight) in enumerate(of_hamiltonian.terms.items()):
            weights.append(weight)
            for visible_idx, pauli_op in pauli_ops:
                term_masks[self.visible_num - visible_idx - 1][term_idx] = BatchHamiltonian.PAULI_OP_TO_IDX[pauli_op]
        weights = pt.tensor(weights, dtype=self.dtype, device=self.device)
        batch_terms = []
        for visible_idx in range(self.visible_num):
            batch_terms.append(pt.index_select(self.pauli_stack,
                                               0,
                                               pt.from_numpy(term_masks[visible_idx, :]).to(self.device)))

        return batch_terms, weights

    @timed(timing_dict=TIMING_DICT)
    def wf_to_mps(self, *, wf: RBMWaveFunction = None) -> MPS:
        return wf.to_mps()

    @timed(timing_dict=TIMING_DICT)
    def denominator(self, *, bra: MPS = None, ket: MPS = None):
        return MPS.overlap(bra=bra, ket=ket)

    # @timed(timing_dict=TIMING_DICT)
    # def numerator(self, *, bra: MPS = None, ket: MPS = None):
    #     assert bra.device == self.device
    #     assert ket.device == self.device
    #     pauli_ket_tensors = [pt.einsum('iak,sba->sibk',
    #                                    ket.tensors[visible_idx],
    #                                    self.batch_terms[visible_idx])
    #                          for visible_idx in range(self.visible_num)]
    #     numerator = pt.squeeze(pt.squeeze(pt.tensordot(pauli_ket_tensors[0],
    #                                                    bra.tensors[0],
    #                                                    dims=[[2], [1]]),
    #                                       dim=3),
    #                            dim=1)
    #     for visible_idx in range(1, self.visible_num):
    #         # Left -> right
    #         numerator = pt.einsum('sij,jak->siak',
    #                               numerator,
    #                               bra.tensors[visible_idx])
    #         # Up -> bottom
    #         numerator = pt.einsum('siak,siaj->sjk',
    #                               numerator,
    #                               pauli_ket_tensors[visible_idx])
    #
    #     numerator = pt.squeeze(numerator) * self.weights
    #
    #     return numerator

    @timed(timing_dict=TIMING_DICT)
    def numerator(self, *, bra: MPS = None, ket: MPS = None):
        assert bra.device == self.device
        assert ket.device == self.device
        numerator = pt.einsum('iak,sba->sibk',
                              ket.tensors[0],
                              self.batch_terms[0])
        numerator = pt.einsum('sbj,bk->sjk',
                              pt.squeeze(numerator, dim=1),
                              pt.squeeze(bra.tensors[0], dim=0))
        for visible_idx in range(1, self.visible_num):
            # Left -> right
            numerator = pt.einsum('sij,iak,sba->skjb',
                                  numerator,
                                  ket.tensors[visible_idx],
                                  self.batch_terms[visible_idx])
            # Up -> bottom
            numerator = pt.einsum('sijb,jbk->sik',
                                  numerator,
                                  bra.tensors[visible_idx])

        numerator = pt.squeeze(numerator) * self.weights

        return numerator

    @timed(timing_dict=TIMING_DICT)
    def energy(self,
               *,
               wf: RBMWaveFunction = None) -> pt.Tensor:
        #assert wf.device == self.device
        self.logger.debug(f'Staring calculation on energy')
        ket = self.wf_to_mps(wf=wf)
        self.logger.debug(f'RBMWaveFunction was converted to an MPS ket')

        ket.apply_symmetry_mpo(sym_mpo=MPS.sym_mps_to_mpo(wf.sym_mps))
        self.logger.debug(f'Symmetries were applied to the MPS ket')

        ket = ket.to(self.device)
        bra = ket.conj()
        self.logger.debug(f'An MPS bra was obtained from the MPS ket')

        denominator = self.denominator(bra=bra, ket=ket)
        self.logger.debug(f'Denominator was calculated')

        numerator = self.numerator(bra=bra, ket=ket)
        self.logger.debug(f'Numerator was calculated')

        return pt.div(pt.sum(numerator), denominator)
