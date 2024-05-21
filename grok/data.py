import itertools
import math
import os
import sys
import random
import primefac

import torch
from torch import Tensor, LongTensor
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from tqdm import tqdm

from sympy.combinatorics.permutations import Permutation
from mod import Mod

import blobfile as bf
import ipdb
st = ipdb.set_trace

VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "pfactor" : "prime_factors",
    "2x" : "2x",
    "x**3" : "x**3",
    "2x+1" : "2x+1",
    "x+11" : "x+11",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
    'interleaved_halves': 'interleaved_halves',
    'reverse_pool': 'reverse_pool',
    'k_shift': 'k_shift',
    'random_swaps': 'random_swaps',
    'idx_add': 'idx_add',
    "caesarcipher": "caesarcipher",
    "permutev1": "permutev1",
    "permutev2": "permutev2",
    "permutev3": "permutev3",
    "strdeletev1": "strdeletev1",
    "strdeletev2": "strdeletev2",
    "caesarcipher_permutev1": "caesarcipher_permutev1"
}

EOS_TOKEN = "<|endoftext|>" # in line with gpt2
SOS_TOKEN = "<|startoftext|>" # in line with gpt2
PAD_TOKEN = "<|pad|>"
DOT = "."
E_TOKEN = 'e'
MINUS_TOKEN = "-"
PLUS_TOKEN = "+"
EQ_TOKEN = "="
MODULUS = 97
NUMS = list(range(MODULUS))
MODULUS_BIJECTIONS = 10
NUMS_BIJECTIONS = list(range(MODULUS_BIJECTIONS))
SPACE = " "
# BIJECTIVE_OPERATORS = ["pfactor", "2x", "x**3", "2x+1", "x+11"]

DEFAULT_DATA_DIR = "data"



def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)


def create_data_files(data_dir: str = DEFAULT_DATA_DIR):
    ArithmeticTokenizer.create_token_file(data_dir)
    ArithmeticDataset.create_dataset_files(data_dir)


class ArithmeticTokenizerDigits:
    """Stores the list of token text to token id mappings and converts between them"""

    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR, max_length=50, max_digits=4, use_regression=False) -> None:
        self.token_file = bf.join(data_dir, self.token_file)

        self.itos = self.get_tokens()

        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

        self.max_length = max_length
        self.max_digits = max_digits
        self.use_regression = use_regression

    def _encode(self, s: str) -> Tensor:
        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        output = np.ones(self.max_length)*self.stoi[PAD_TOKEN]
        ctr = 0
        s = s.strip()
        for t in s.split(" "):
            if t.isdigit() or is_float(t):
                for c in t:
                    output[ctr] = self.stoi[c]
                    ctr += 1
            else:
                output[ctr] = self.stoi[t]
                ctr += 1
            output[ctr] = self.stoi[SPACE]
            ctr += 1
        st()
        return LongTensor(output)

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            if self.use_regression:
                return (
                    torch.stack([torch.stack([self._encode(s[0]),self._encode(s[1])]) for s in obj]),
                    torch.stack([torch.stack([torch.tensor(s[2], dtype=torch.float32),torch.tensor(s[3], dtype=torch.float32)]) for s in obj]) # forward and backward targets in numbers
                )

            return torch.stack([torch.stack([self._encode(s[0]),self._encode(s[1])]) for s in obj])
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    def get_tokens(cls):
        tokens = (
            [SOS_TOKEN, EOS_TOKEN, EQ_TOKEN, SPACE, DOT, E_TOKEN, MINUS_TOKEN, PLUS_TOKEN, PAD_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS_BIJECTIONS))
            + ['']
        )
        return tokens



class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)

        self.itos = self.get_tokens()

        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        # st()
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([torch.stack([self._encode(s[0]),self._encode(s[1])]) for s in obj])
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS))
            + list(map(render, itertools.permutations(range(5))))  # s5
        )
        return tokens


class ArithmeticDataset:
    """A Dataset of arithmetic equations"""

    @classmethod
    def splits(
        cls,
        train_pct: float,
        operator: str,
        operand_length: Optional[int] = None,
        data_dir: str = DEFAULT_DATA_DIR,
        max_context_len: int = 50,
        hparams = {}
    ):
        """
        Creates training and validation datasets

        :param train_pct: percentage of total equations used for training data
        :param operator: The arithmetic operator for this dataset e.g. '+', '-', '*', '/', 'sort'
        :param operand_length: for list based datasets the length of the lists
        :returns: (train_dataset, validation_dataset)
        """

        assert (0 < train_pct) and (train_pct < 100)
        ds_name = cls.get_dsname(operator, operand_length)
        eqs = cls.make_data(operator, operand_length, hparams=hparams)

        train_rows, _ = cls.calc_split_len(train_pct, len(eqs))
        train_ds = cls(ds_name, eqs[:train_rows], train=True, data_dir=data_dir, operator=operator, max_context_len=max_context_len, hparams=hparams)
        val_ds = cls(ds_name, eqs[train_rows:], train=False, data_dir=data_dir, operator=operator, max_context_len=max_context_len, hparams=hparams)
        return train_ds, val_ds


    @classmethod
    def calc_split_len(cls, train_pct, ds_len):
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    # def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir, max_context_len, hparams) -> None:
    def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir, operator, max_context_len, hparams) -> None:

        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        self.hparams = hparams
        self.max_context_len = max_context_len
        if operator == '2x':
            self.tokenizer = ArithmeticTokenizerDigits(data_dir, max_length=50, max_digits=4)
        elif operator == '2x+1':
            self.tokenizer = ArithmeticTokenizerDigits(data_dir, max_length=50, max_digits=4)
        elif operator == 'x+11':
            self.tokenizer = ArithmeticTokenizerDigits(data_dir, max_length=50, max_digits=4)
        elif operator == 'x**3':
            self.tokenizer = ArithmeticTokenizerDigits(data_dir, max_length=100, max_digits=12)
        elif operator == 'pfactor':
            self.tokenizer = ArithmeticTokenizerDigits(data_dir, max_length=100, max_digits=4)
        else:
            # binary case regression wont work
            self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        # st()
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

    # @classmethod
    # def _render(cls, operand):
    #    return render(operand, join_str=" ")
    #
    # @classmethod
    # def _render_eq(parts):
    #    return " ".join(map(render, parts))

    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None, hparams=None) -> List[str]:
        if operator == "s5":
            operands = operands or list(range(5))
            elems = map(np.array, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif operator in ["s5conj", "s5aba"]:
            operands = operands or list(range(5))
            elems = map(Permutation, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif "_mod_" in operator:
            modulo = int(operator.split("_mod_")[-1])
            elems = [Mod(i, modulo) for i in range(modulo)]
            tuples = itertools.product(elems, repeat=2)
        else:
            operands = operands or NUMS
            tuples = itertools.product(operands, repeat=2)

        # if operator == "s5":
        #     print("elems", list(elems))
        #     print("tuples", list(tuples))
        eqs = []
        # st()
        for a, b in tuples:
            if operator == "/":
                if b == 0:
                    continue
                else:
                    c = a
                    a = (b * c) % MODULUS
            elif operator == "s5":
                # st()
                c = [a[b[i]] for i in range(len(b))]
            elif operator == "s5conj":
                c = a * b * (a.__invert__())
            elif operator == "s5aba":
                c = a * b * a
            elif operator == "+*":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a * b) % MODULUS
            elif operator == "+-":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a - b) % MODULUS
            elif "_mod_" in operator:
                expression = operator.split("_mod_")[0]
                function = eval(f"lambda x, y: ({expression})")
                c = function(a, b)
            else:
                c = eval(f"({a} {operator} {b}) % {MODULUS}")
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            invert_eq = " ".join(map(render, [c, "=", a, operator, b ]))
            eqs.append([eq,invert_eq])


        # if operator == "s5":
        #     print("eqs", eqs)
        return eqs

    # @staticmethod
    # def _render_unop_example(operator, lhs, rhs):
    #    return " ".join([operator, render(lhs), "=", render(rhs)])

    @staticmethod
    def _make_unary_operation_data(operator: str, operands: Tensor, hparams) -> List[str]:
        """
        :param operator: The unary operator to apply to each operand e.g. '+'
        :param operands: A tensor of operands
        :returns: list of equations"""
        # operands = list(range(4))
        # st()

        if operator == "2x":
            operands= torch.tensor(list(range(10000))).unsqueeze(-1)
            rhs = [[(i.item())*2] for i in operands]
            rhs_list = rhs

        elif operator == "2x+1":
            operands= torch.tensor(list(range(10000-1))).unsqueeze(-1)
            rhs = [[(i.item())*2+1] for i in operands]
            rhs_list = rhs

        elif operator == "x**3":
            operands= torch.tensor(list(range(10000))).unsqueeze(-1)
            rhs = [[(i.item())**3] for i in operands]
            rhs_list = rhs

        elif operator == "x+11":
            operands= torch.tensor(list(range(10000))).unsqueeze(-1)
            rhs = [[(i.item())+11] for i in operands]
            rhs_list = rhs

        elif operator == "pfactor":
            operands= torch.tensor(list(range(10000))).unsqueeze(-1)
            rhs = [list(primefac.primefac(i.item())) for i in operands]
            rhs_list = rhs
        else:
            # list operations
            list_len = 5 if not hparams else hparams.get("data_list_len", 5)
            elems = map(np.array, itertools.permutations(list(range(list_len))))
            operands = torch.stack([torch.from_numpy(i) for i in elems])
            if operator == "sort":
                # sort each list
                rhs = torch.sort(operands, dim=1).values

            elif operator == "reverse":
                rhs = torch.flip(operands, dims=(1,))
            elif operator == "copy":
                rhs = operands
            elif operator == 'interleaved_halves':
                even_indices = torch.arange(0, list_len, 2)  # indices for even index elements: 0, 2, 4, ...
                odd_indices = torch.arange(1, list_len, 2)   # indices for odd index elements: 1, 3, 5, ...
                interleaved_indices = torch.cat((even_indices, odd_indices))  # concatenate indices
                rhs = operands[:, interleaved_indices]
            elif operator == 'reverse_pool':
                k = hparams.get("k", 3)
                # each sets of k elements are reversed
                rhs = torch.cat([torch.flip(operands[:, i:i+k], dims=(1,)) for i in range(0, list_len, k)], dim=1)
            elif operator == 'k_shift':
                # shift each list by k to the left
                k = hparams.get("k", 3)
                rhs = torch.cat([torch.roll(operands, shifts=-k, dims=1)], dim=1)
            elif operator == 'random_swaps':
                # make a unique random mapping for each index to another index making sure that it's cyclic
                generator = torch.Generator().manual_seed(hparams.get("seed", 42))
                rp = torch.randperm(list_len, generator=generator)
                mp = {}
                for i, j in enumerate(rp):
                    mp[i] = j # since we are using 1 based indexing for the list
                print(f'Random mapping: {mp}'.center(100, '-') )
                rhs = torch.tensor([[mp[i.item()] for i in row] for row in operands])
            elif operator == 'idx_add':
                # add the index to each element
                rhs = operands + torch.arange(list_len)
            elif operator == 'caesarcipher_permutev1': # since this is first it won't go into the permute case
                elems = map(np.array, itertools.permutations(list(range(5))))
                operands = [torch.from_numpy(i) for i in elems]
                rhs = []
                for i in operands:
                    rhs.append([(i[1] + 1) % 5, (i[0] + 1) % 5, (i[2] + 1) % 5, (i[3] + 1) % 5, (i[4] + 1) % 5])
                operands = torch.stack(operands)
                rhs = torch.tensor(rhs)
            elif operator == 'caesarcipher':
                rhs = (operands + 1) % list_len
            elif 'permute' in operator:
                elems = map(np.array, itertools.permutations(list(range(5))))
                operands = [torch.from_numpy(i) for i in elems]
                rhs = []
                for i in operands:
                    if operator == 'permutev1':
                        rhs.append([i[1], i[0], i[2], i[3], i[4]])
                    elif operator == 'permutev2':
                        rhs.append([i[1], i[0], i[3], i[4], i[2]])
                    elif operator == 'permutev3':
                        rhs.append([i[4], i[0], i[1], i[2], i[3]])
                    else:
                        raise NotImplementedError
                operands = torch.stack(operands)
                rhs = torch.tensor(rhs)
            elif 'strdelete' in operator:
                elems = map(np.array, itertools.permutations(list(range(5))))
                operands = [torch.from_numpy(i) for i in elems]
                rhs = []
                for i in operands:
                    if operator == 'strdeletev1':
                        rhs.append([0, i[1], i[2], i[3], i[4]])
                    elif operator == 'strdeletev2':
                        rhs.append([0, 0, i[2], i[3], i[4]])
                    else:
                        raise NotImplementedError
                operands = torch.stack(operands)
                rhs = torch.tensor(rhs)
            else:
                raise NotImplementedError
            rhs_list = rhs.tolist()
        num_examples = operands.shape[0]

        def func(L, R):
            L = map(str, L)
            R = map(str, R)
            return f"{operator} {' '.join(L)} = {' '.join(R)}"


        # st()
        if num_examples < 1000000000:
            eqs = [
                (func(L, R),func(R,L))
                for L, R in tqdm(
                    zip(operands.tolist(), rhs_list), total=num_examples
                )
            ]
        else:
            with ProcessPoolExecutor() as executor:
                eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))
        return eqs

    # @staticmethod
    # def _make_s5_data(abstract=False) -> List[str]:
    #    elems = itertools.permutations([0, 1, 2, 3, 4])
    #    pairs = itertools.product(elems, repeat=2)
    #    eqs = []
    #    for a, b in pairs:
    #        a = np.array(a)
    #        b = np.array(b)
    #        c = b[a]
    #        eq = " ".join(map(render, (a, "s5", b, "=", c)))
    #        eq = cls._render_eq([a, , b, "=", c])
    #        eqs.append(eq)
    #
    #    return eqs

    @classmethod
    def get_dsname(cls, operator, operand_length) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    def make_data(cls, operator, operands=None, shuffle=True, seed=0, hparams=None) -> List[str]:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator in VALID_OPERATORS


        if operator not in ["sort", "reverse", "copy","pfactor","2x","x**3","2x+1", "interleaved_halves", "reverse_pool", "k_shift", "random_swaps", "idx_add","caesarcipher_permutev1","caesarcipher","permutev1","permutev2","permutev3","strdeletev1","strdeletev2","pfactor","2x","x**3","2x+1","x+11"]:
            data = cls._make_binary_operation_data(operator, hparams=hparams)
        else:
            # st()
            data = cls._make_unary_operation_data(operator, operands, hparams=hparams)
        # st()
        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(data)

        if noise_level > 0:
            random_answer_eqns = rng.choice(data, size=noise_level)
            random_answers = [
                random_eq.split(" = ")[1] for random_eq in random_answer_eqns
            ]
            for i in range(noise_level):
                data[i] = data[i].split(" = ")[0] + " = " + random_answers[i]
        # st()
        data = [[EOS_TOKEN + " " + eq[0] + " " + EOS_TOKEN,EOS_TOKEN + " " + eq[1] + " " + EOS_TOKEN] for eq in data]
        # st()
        return data

    # @classmethod
    # def create_data_file(
    #    cls, operator, operand_length=None, shuffle=True, data_dir=DEFAULT_DATA_DIR
    # ):
    #    if VALID_OPERATORS[operator]["binary_eval"]:
    #        cls.write_dataset(
    #            cls.make_binary_operation_data(operator), paths["ds_file"]
    #        )
    #
    #    pass

    # @classmethod
    # def write_dataset(eqs: List[str], ds_file: str):
    #    print(f"-> writing {ds_file}", flush=True)
    #    with open(ds_file, "w") as fh:
    #        fh.writelines([EOS_TOKEN + " " + eq + " " + EOS_TOKEN + "\n" for eq in eqs])

    @classmethod
    def _make_lists(cls, sizes=[2, 3], nums=NUMS):
        lists: dict = {}
        for size in sizes:
            lists[size] = torch.tensor(
                list(itertools.permutations(nums, r=size)),
                dtype=torch.int,
            )
        return lists


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float = 0,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(
            len(dataset), batchsize_hint=batchsize_hint
        )
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int = 0) -> int:
        """
        Calculates which batch size to use

        :param ds_size: the number of equations in the dataset
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :returns: the actual batchsize to use
        """

        if batchsize_hint == -1:
            return ds_size
        elif batchsize_hint == 0:
            return min(512, math.ceil(ds_size / 2.0))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return math.ceil(ds_size * batchsize_hint)
        elif batchsize_hint > 1:
            return min(batchsize_hint, ds_size)
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        # st()
        text = self.dataset.data[indices,:, :-1]
        target = self.dataset.data[indices,:, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        # st()
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
