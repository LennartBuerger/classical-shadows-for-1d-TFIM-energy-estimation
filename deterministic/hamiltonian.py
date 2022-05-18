import time

import torch as pt

from openfermion.ops import QubitOperator

from .constants import BASE_COMPLEX_TYPE
from .term import Term
from .rbm_wave_function import RBMWaveFunction


class Hamiltonian:
    def __init__(self,
                 *,
                 visible_num: int = None,
                 dtype=BASE_COMPLEX_TYPE,
                 device=None,
                 of_qubit_operator: QubitOperator):
        assert visible_num is not None
        self.visible_num = visible_num

        self.dtype = dtype
        assert device is not None
        self.device = device

        self.of_qubit_operator = of_qubit_operator
        self.terms = self.parse_of_qubit_operator(of_qubit_operator)

    @property
    def term_num(self):
        return len(self.of_qubit_operator.terms)

    def parse_of_qubit_operator(self, of_qubit_operator: QubitOperator = None):
        assert of_qubit_operator is not None
        terms = []
        for qubit_ops, weight in of_qubit_operator.terms.items():
            terms.append(Term(dtype=self.dtype,
                              device=self.device,
                              weight=weight,
                              visible_num=self.visible_num,
                              qubit_ops=qubit_ops))
        return terms

    def energy(self,
               *,
               wf: RBMWaveFunction = None,
               method: str = 'bf',
               **kwargs) -> pt.Tensor:
        assert wf is not None
        assert wf.device == self.device
        numerator = 0.0
        denominator = wf.overlap(bra=wf,
                                 ket=wf,
                                 method=method,
                                 **kwargs)
        term_idx = 1
        for term in self.terms:
            start_time = time.time()
            term_wf = wf.apply_term(term)
            numerator = pt.add(numerator, term.weight * term.renorm * wf.overlap(bra=wf,
                                                                                 ket=term_wf,
                                                                                 method=method,
                                                                                 **kwargs))
            print(f'Calculating overlap for term #{term_idx}/{len(self.terms)} took {time.time() - start_time}')
            term_idx += 1

        return pt.mul(numerator, denominator)
