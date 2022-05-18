from __future__ import annotations

import torch as pt

from openfermion.ops import QubitOperator

from .constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE
from .pauli import PAULI_CHARS, REAL_PAULI_MATRICES
from .mps import MPS


# Class representing a single node in the TreeHamiltonian.
# Every Node contains a Pauli char associated with it, the value of weight of the corresponding term,
# and a set of parent and child nodes.
class Node:
    node_count = 0

    def __init__(self,
                 char: str = None,
                 weight: complex = None,
                 parent: Node = None,
                 child: Node = None):
        self.id = Node.node_count
        Node.node_count += 1

        assert char in PAULI_CHARS
        self.char = char
        self.weight = weight

        self.parents = set([])
        if parent is not None:
            self.assign_parent(parent)
        self.children = set([])
        if child is not None:
            self.assign_child(child)

        self.init_list_pos = None

    def __str__(self):
        return f'Node #{self.id}: {self.char}\n' \
               f'Parents = {[parent.id for parent in self.parents]}\n' \
               f'Children = {[child.id for child in self.children]}'

    def assign_parent(self, parent: Node = None):
        assert parent is not None

        assert parent not in self.parents
        self.parents = self.parents.union({parent})

        assert self not in parent.children
        parent.children = parent.children.union({self})

    def assign_child(self, child: Node = None):
        assert child is not None

        assert child not in self.children
        self.children = self.children.union({child})

        assert self not in child.parents
        child.parents = child.parents.union({self})

    @staticmethod
    def equ_parents(node_1: Node = None, node_2: Node = None):
        return (node_1.char == node_2.char) and (node_1.parents == node_2.parents)

    @staticmethod
    def equ_children(node_1: Node = None, node_2: Node = None):
        return (node_1.char == node_2.char) and (node_1.children == node_2.children)

    def merge_top_bottom(self, node: Node = None):
        assert Node.equ_parents(self, node)
        for parent in self.parents:
            parent.children.remove(node)
        for child in node.children:
            child.parents.remove(node)
            child.assign_parent(self)

    def merge_bottom_top(self, node: Node = None):
        assert Node.equ_children(self, node)
        for child in self.children:
            child.parents.remove(node)
        for parent in node.parents:
            parent.children.remove(node)
            parent.assign_child(self)

    def get_children_pauli_strings(self):
        if self.char == '0':
            return ['']
        else:
            child_strings = []
            for child in self.children:
                child_strings += child.get_children_pauli_strings()

            for string_idx in range(len(child_strings)):
                child_strings[string_idx] = self.char + child_strings[string_idx]

            return child_strings

    def get_parent_pauli_strings(self):
        if self.char == '0':
            return ['']
        else:
            parent_strings = []
            for parent in self.parents:
                parent_strings += parent.get_parent_pauli_strings()

            for string_idx in range(len(parent_strings)):
                parent_strings[string_idx] = parent_strings[string_idx] + self.char

            return parent_strings

    def get_pauli_strings(self):
        parent_strings = self.get_parent_pauli_strings()
        children_strings = self.get_children_pauli_strings()

        pauli_strings = []
        for parent_string in parent_strings:
            for child_string in children_strings:
                pauli_strings.append(parent_string + child_string[1:])

        return pauli_strings

    @staticmethod
    def pauli_strings_to_terms(pauli_strings):
        terms = []
        for pauli_string in pauli_strings:
            term = ()
            for qubit_idx, pauli_char in enumerate(pauli_string[::-1]):
                if pauli_char != 'I':
                    term += (tuple((qubit_idx, pauli_char)),)
            terms.append(term)

        return terms


# This class allows to compress a QubitHamiltonian by representing it as a symbolic tree.
class TreeHamiltonian:
    def __init__(self,
                 qubit_num: int,
                 of_hamiltonian: QubitOperator = None,
                 dtype=BASE_COMPLEX_TYPE):
        self.qubit_num = qubit_num
        self.dtype = dtype

        self.weights = None
        self.term_num = 0

        self.tree = [[] for _ in range(self.qubit_num + 2)]
        self.top_root = Node(char='0')
        self.bot_root = Node(char='0')
        self.tree[0].append(self.top_root)
        self.tree[-1].append(self.bot_root)

        print(f'Parsing of OF QubitOperator started...')
        self.parse_of_hamiltonian(of_hamiltonian)
        print(f'Finished!')
        for term_idx in range(self.term_num):
            self.tree[self.qubit_num][term_idx].assign_child(self.bot_root)
        self.weights_map = None

    def parse_of_hamiltonian(self, of_hamiltonian: QubitOperator = None):
        weights = []
        for qubit_ops, weight in of_hamiltonian.terms.items():
            pauli_y_num = 0
            weights.append(weight + 0j)
            qubit_ops_dict = {self.qubit_num - tup[0] - 1: tup[1] for tup in qubit_ops}
            for qubit_idx in range(self.qubit_num):
                char = qubit_ops_dict[qubit_idx] if qubit_idx in qubit_ops_dict else 'I'
                if char == 'Y':
                    pauli_y_num += 1
                self.tree[qubit_idx + 1].append(Node(char=char,
                                                     weight=weight,
                                                     parent=self.tree[qubit_idx][-1]))
            assert (pauli_y_num % 2) == 0
            weights[-1] *= (-1) ** (pauli_y_num // 2)
            for qubit_idx in range(self.qubit_num):
                self.tree[qubit_idx + 1][-1].weight = weights[-1]
            self.term_num += 1
        self.weights = pt.tensor(weights).type(self.dtype)

    def compress(self, verbose: bool = False):
        for layer_idx in range(1, self.qubit_num // 2 + 1):
            if verbose:
                print(f'Layer #{layer_idx}')
            self.merge_top_bottom(layer_idx)
        for layer_idx in range(self.qubit_num, self.qubit_num // 2 + 1, -1):
            if verbose:
                print(f'Layer #{layer_idx}')
            self.merge_bottom_top(layer_idx)
        self.merge_middle_bottom_top()
        if verbose:
            for layer in self.tree:
                print(len(layer))

        return self

    def merge_top_bottom(self, layer_idx: int = None):
        assert (1 <= layer_idx) and (layer_idx <= self.qubit_num)

        layer = self.tree[layer_idx]
        for to_idx in range(len(layer)):
            merged_nodes = []
            for from_idx in range(to_idx + 1, len(layer)):
                if layer[from_idx] not in merged_nodes:
                    if Node.equ_parents(layer[to_idx], layer[from_idx]):
                        layer[to_idx].merge_top_bottom(layer[from_idx])
                        merged_nodes.append(layer[from_idx])
            for node in merged_nodes:
                layer.remove(node)

    def merge_bottom_top(self, layer_idx: int = None):
        assert (1 <= layer_idx) and (layer_idx <= self.qubit_num)
        layer = self.tree[layer_idx]
        for to_idx in range(len(layer)):
            merged_nodes = []
            for from_idx in range(to_idx + 1, len(layer)):
                if layer[from_idx] not in merged_nodes:
                    if Node.equ_children(layer[to_idx], layer[from_idx]):
                        layer[to_idx].merge_bottom_top(layer[from_idx])
                        merged_nodes.append(layer[from_idx])
            for node in merged_nodes:
                layer.remove(node)

    def merge_middle_bottom_top(self):
        layer_idx = (self.qubit_num // 2 + 1)
        layer = self.tree[layer_idx]

        weights_map = pt.zeros((len(self.tree[layer_idx - 1]), len(self.tree[layer_idx])), dtype=pt.cdouble)
        for parent_idx in range(len(self.tree[layer_idx - 1])):
            for child_idx in range(len(self.tree[layer_idx])):
                if layer[child_idx] in self.tree[layer_idx - 1][parent_idx].children:
                    weights_map[parent_idx, child_idx] = self.tree[layer_idx][child_idx].weight

        for idx in range(len(layer)):
            layer[idx].init_list_pos = idx
        merge_to = pt.zeros(len(layer)) - 1

        # Merger loops
        for to_idx in range(len(layer)):
            if to_idx < len(layer):
                merge_to[layer[to_idx].init_list_pos] = layer[to_idx].init_list_pos
            merged_nodes = []
            for from_idx in range(to_idx + 1, len(layer)):
                if layer[from_idx] not in merged_nodes:
                    if Node.equ_children(layer[to_idx], layer[from_idx]):
                        layer[to_idx].merge_bottom_top(layer[from_idx])
                        merged_nodes.append(layer[from_idx])

                        # Memorizing the merge
                        merge_to[layer[from_idx].init_list_pos] = layer[to_idx].init_list_pos

            for node in merged_nodes:
                layer.remove(node)

        unique_to = pt.unique(merge_to, sorted=True)

        sum_matrix = pt.eq(pt.unsqueeze(merge_to, dim=-1), pt.unsqueeze(unique_to, dim=0))
        self.weights_map = weights_map @ sum_matrix.to(weights_map.dtype)

    def calc_mps_tensor(self, qubit_idx: int = None):
        result = pt.zeros((len(self.tree[qubit_idx]),
                           len(self.tree[qubit_idx + 1]),
                           2,
                           2),
                          dtype=self.dtype)
        for parent_idx, parent in enumerate(self.tree[qubit_idx]):
            for child_idx, child in enumerate(self.tree[qubit_idx + 1]):
                if child in parent.children:
                    matrix = REAL_PAULI_MATRICES[child.char].type(self.dtype)
                    if qubit_idx == (self.qubit_num // 2):
                        matrix = matrix * self.weights_map[parent_idx, child_idx]
                    result[parent_idx, child_idx, :, :] = matrix
        if qubit_idx == (self.qubit_num - 1):
            result = pt.sum(result, dim=1, keepdim=True)

        shape = result.shape
        result = pt.permute(pt.reshape(result,
                                       (shape[0], shape[1], shape[2] * shape[3])),
                            (0, 2, 1))

        return result

    def to_mps(self):
        return MPS.from_tensor_list([self.calc_mps_tensor(idx) for idx in range(self.qubit_num)])
