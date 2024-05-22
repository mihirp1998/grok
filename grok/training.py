import argparse
import copy
import wandb
import matplotlib.pyplot as plt
# import data
import time
import json
import pandas as pd
import logging
import math
import os
import sys
import pickle
from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import ipdb
st = ipdb.set_trace
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import grok.metrics as metrics
from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    EQ_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import Transformer
from grok.measure import get_sharpness

DEFAULT_LOG_DIR = "logs"


class TrainableTransformer(LightningModule):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()
        # self.hparams = hparams  # type: ignore
        # st()
        hparams_dict = hparams

        self.val_accuracy = []

        for key in hparams.keys():
            self.hparams[key]=hparams_dict[key]

        # st()

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.prepare_data()

        # ip_out_map = self.ip_out_map

        # now print the degree of bijectivity by calculating how many inputs map to the each output
        # output_counts = {}
        # when uncommenting, uncomment one line in data.py as well (line 403)
        # for k,v in ip_out_map.items():
            # output_counts[v] = output_counts.get(v, 0) + 1

        # get mean of counts to get the degree of bijectivity
        # mean_count = np.mean(list(output_counts.values()))
        # print(f'Mean count for {self.hparams.math_operator} = {mean_count} out of {len(ip_out_map)}')
        # degree_of_bijectivity = 1 - max_count/len(ip_out_map)
        # print(f"Degree of bijectivity for {self.hparams.math_operator} = {degree_of_bijectivity}".center(80, '-'))
        # time.sleep(15)
        # sys.exit(0)
        # st()



        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            len(self.train_dataset.tokenizer),
            hparams.embed_style,
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
            operator=self.hparams.math_operator,
        )

        self.margin = torch.Tensor([0])
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0


        if self.hparams.multi_task:
            self.train_iterator_2  = self.train_dataloader_2()

        self.trainer_step_val_dict = {}

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments needed by instances of this
        class. This is intended to be called when parsing command line
        arguments.

        :param parser: an argparse.ArgumentParser created by the caller
        :returns: the argument parser with the command line arguments added
                  for this class.
        """
        parser.add_argument(
            "--batchsize",
            type=float,
            # default=0.25,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
        parser.add_argument("--max_context_len", type=int, default=50)
        parser.add_argument("--math_operator", type=str, default="+")

        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=5)
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)

        parser.add_argument(
            "--debug", dest="debug", action="store_true"
        )
        parser.add_argument(
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )

        return parser

    def prepare_data(self) -> None:
        """
        Used by pytorch_lighting

        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        (self.train_dataset, self.val_dataset, self.ip_out_map) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
            max_context_len=self.hparams.max_context_len,
            hparams=self.hparams,
        )
        if self.hparams.multi_task:
            (self.train_dataset_2, self.val_dataset_2, self.ip_out_map_2) = ArithmeticDataset.splits(
                train_pct=self.hparams.train_data_pct,  # type: ignore
                operator=self.hparams.math_operator_2,  # type: ignore
                operand_length=self.hparams.operand_length,  # type: ignore
                data_dir=self.hparams.datadir,  # type: ignore
                max_context_len=self.hparams.max_context_len,
                hparams=self.hparams,
            )


    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset,
            device,
            batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        # st()
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator


    def train_dataloader_2(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset_2,
            device,
            batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        # st()
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)

        return iterator

    def val_train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        # st()
        # self.train_batchsize = iterator.batchsize
        # self.batches_per_epoch = len(iterator)

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset, device, batchsize_hint=-1  # type: ignore
        )
        return iterator

    def _scheduler_lr(self, step: int) -> float:
        """
        Used by pytorch_lighting

        :returns: the learning_rate for this training step
        """
        max_lr = self.hparams.max_lr  # type: ignore
        min_lr = self.hparams.max_lr / 10  # type: ignore
        warmup_steps = self.hparams.warmup_steps  # type: ignore
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
                # lr = max_lr - ((effective_step / max_effective_step) * (max_lr - min_lr))
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        """
        Used by pytorch_lighting

        :returns: optimizers and schedulers.
        """
        optimizer = CustomAdamW(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        # optimizer = SAM(
        #     self.parameters(),
        #     base_optimizer=CustomAdamW,
        #     rho=0.05,
        #     betas=(0.9, 0.98),
        #     eps=1e-8,
        #     lr=1,
        #     weight_decay=self.hparams.weight_decay,
        #     noise_factor=self.hparams.noise_factor,
        # )
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """

        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = row_accuracy.float() * 100  # shape: batchsize
        return accuracy

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        cc_dict: Dict = None,
        train: bool = True,
        cc: bool = False,
        reduction: str = "mean",
        grads: bool = False,
        inverse_mapping: bool = False,
        reverse_mode: bool = False,
    ) -> Tuple[Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
                  Margin for this batch
        """
        # st()
        if cc:
            x = batch["text"]  # shape = batchsize * context_len
            y = batch["target"]  # shape = batchsize * context_len
            # y_hat_cc = batch["y_hat_rhs"]
            # st()
        else:
            if inverse_mapping:
                # st()
                x = batch["text"][:,1]  # shape = batchsize * context_len
                y = batch["target"][:,1]  # shape = batchsize * context_len
                # st()
            else:
                x = batch["text"][:,0]  # shape = batchsize * context_len
                y = batch["target"][:,0]  # shape = batchsize * context_len
            cc_dict = None
        if reverse_mode:
            y_hat, attentions, values = self.transformer.reverse(x=x, save_activations=self.hparams.save_activations, inverse_mapping=inverse_mapping)
        else:
            y_hat, attentions, values = self(
                x=x, save_activations=self.hparams.save_activations, inverse_mapping=inverse_mapping, cc=cc, cc_dict=cc_dict  # type: ignore
            )  # shape = batchsize * context_len * vocab_size

        y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len
        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        x_lhs = x[..., : eq_position + 1]
        # st()

        if train:
            coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
        else:
            coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
        # st()
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        with torch.no_grad():
            acc = self._accuracy(y_hat_rhs, y_rhs)
            if reduction == "mean":
                acc = acc.mean()

        """
        device = self.transformer.embedding.weight.device
        self.margin = self.margin.to(device)

        output = y_hat_rhs.clone()  # batchsize, vocabsize, rhs tokens
        output_m = output.clone()  # batchsize, vocabsize, rhs tokens
        target = y_rhs.clone()  # batchsize, rhs tokens

        for i in range(output.size(0)):  # batch
            for j in range(output.size(2)):  # rhs tokens
                output_m[i, target[i, j], j] = output_m[i, :, j].min()

        for i in range(output.size(2)):  # rhs tokens
            output_compressed = output[:, target[:, i], i].squeeze().diag()
            output_m_compressed = (
                output_m[:, output_m.max(dim=1).indices[:, i], i].squeeze().diag()
            )
            self.margin = torch.cat(
                (
                    self.margin,
                    (output_compressed - output_m_compressed),
                ),
                0,
            )
        """
        grad_vec = None
        if grads:
            loss.backward()
            for p in self.parameters():
                p.grad.data.div_(batch["text"].shape[0])
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
            return loss, grad_vec
        return loss, acc, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values


    def _save_inputs(self, outputs: Dict, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        :param ds: a string ('train' or 'val') naming which dataset
                   these inputs are from.
        :param train: True is this is a training batch, false otherwise
        """
        logdir = self.hparams.logdir + "/inputs/" + ds  # type: ignore
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = torch.cat([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        # num_batches = len(partial_activations)
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    # # print(f"head_attn = {head_attn}")
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: Dict, ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        """

        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:  # type: ignore
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:  # type: ignore
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:  # type: ignore
            logdir = self.hparams.logdir + "/outputs/" + ds  # type: ignore
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward training pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0

        start = time.time()
        losses = []
        forward_loss, accuracy, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=True,
        )
        # print(batch['text'].shape, y_hat_rhs.shape, y_rhs.shape, x_lhs.shape)

        forward_loss = forward_loss * self.hparams.f_coef
        losses.append(forward_loss)
        # recreate batch
        # batch_copy = copy.deepcopy(batch)
        # data
        # st()
        if self.hparams.cyclic_consistency:
            eq_token_index = torch.tensor(self.train_dataset.tokenizer.stoi["="]).repeat(batch['text'].shape[0])[:,None].to(x_lhs.device)
            eos_token = torch.tensor(self.train_dataset.tokenizer.stoi["<|eos|>"]).repeat(batch['text'].shape[0])[:,None].to(x_lhs.device)
            batch_cc = copy.deepcopy(batch)
            cc_dict = {}
            y_hat_rhs_pred = y_hat_rhs.argmax(1)

            data_tmp = torch.cat([eos_token, y_rhs[:,:-1], eq_token_index, x_lhs[:,1:], eos_token], dim=1)
            # assert (batch['text'][0][1] == data_tmp[:,:-1][0]).all()

            data = torch.cat([eos_token, y_rhs[:,:-1], eq_token_index, x_lhs[:,1:], eos_token], dim=1)

            batch_cc['text']  = data[:,:-1]
            batch_cc['target']  = data[:,1:]

            cc_dict['y_hat_rhs'] = y_hat_rhs
            cc_dict['eos_token'] = eos_token
            cc_dict['y_rhs'] = y_rhs[:,:-1]
            cc_dict['eq_token_index'] = eq_token_index
            cc_dict['x_lhs'] = x_lhs[:,1:]
            # x_lhs
            # st()


            inv_loss_cc, inv_accuracy_cc, inv_coeff_cc, inv_x_lhs_cc, inv_y_hat_rhs_cc, inv_y_rhs_cc, inv_attentions_cc, inv_values_cc = self._step(
                batch=batch_cc, batch_idx=batch_idx, train=True, inverse_mapping=True, cc=True, cc_dict= cc_dict
            )

            inv_loss_cc = inv_loss_cc * self.hparams.cc_coef
            # st()
            losses.append(inv_loss_cc)
            # st()
        else:
            pass

        if self.hparams['do_tta']:
            batch_val = next(self.val_dataloader())
            # st()
            if self.hparams['val_batchify']:
                index_rand = torch.randint(0,batch_val['text'].shape[0], (512,))
                batch_val['text'] = batch_val['text'][index_rand]
                batch_val['target'] = batch_val['target'][index_rand]
                # st()

            #     batch_val = next(self.val_train_dataloader())
            #     st()
            # else:
            #     batch_val = next(self.val_dataloader())
            # st()

            loss, accuracy, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
                batch=batch_val, batch_idx=batch_idx, train=False
            )

            eq_token_index = torch.tensor(self.train_dataset.tokenizer.stoi["="]).repeat(batch_val['text'].shape[0])[:,None].to(batch_val['text'].device)
            eos_token = torch.tensor(self.train_dataset.tokenizer.stoi["<|eos|>"]).repeat(batch_val['text'].shape[0])[:,None].to(batch_val['text'].device)
            batch_val_cc = copy.deepcopy(batch_val)
            cc_dict = {}
            # y_hat_rhs_pred = y_hat_rhs.argmax(1)

            data_tmp = torch.cat([eos_token, y_rhs[:,:-1], eq_token_index, x_lhs[:,1:], eos_token], dim=1)
            # st()
            # assert (batch_val['text'][0][1] == data_tmp[:,:-1][0]).all()
            # st()
            if self.hparams.math_operator not in ["sort", "reverse", "copy","pfactor","2x","x**3","2x+1", "interleaved_halves", "reverse_pool", "k_shift", "random_swaps", "idx_add","caesarcipher_permutev1","caesarcipher","permutev1","permutev2","permutev3","strdeletev1","strdeletev2","pfactor","2x","x**3","2x+1","x+11"]:
                data =  torch.cat([eos_token, y_rhs[:,:-1], eq_token_index, x_lhs[:,1:], eos_token], dim=1)
            else:
                data = torch.cat([eos_token, x_lhs[:,1].unsqueeze(1), y_rhs[:,:-1], eq_token_index, x_lhs[:,2:], eos_token], dim=1)
            # st()
            batch_val_cc['text']  = data[:,:-1]
            batch_val_cc['target']  = data[:,1:]

            cc_dict['y_hat_rhs'] = y_hat_rhs
            cc_dict['eos_token'] = eos_token
            cc_dict['y_rhs'] = y_rhs[:,:-1]
            cc_dict['eq_token_index'] = eq_token_index
            cc_dict['x_lhs'] = x_lhs[:,1:]

            # print(batch_val_cc['text'].shape, y_hat_rhs.shape, y_rhs.shape, x_lhs.shape)

            inv_loss_cc_tta, inv_accuracy_cc_tta, inv_coeff_cc_tta, inv_x_lhs_cc_tta, inv_y_hat_rhs_cc_tta, inv_y_rhs_cc_tta, inv_attentions_cc_tta, inv_values_cc_tta = self._step(
                batch=batch_val_cc, batch_idx=batch_idx, train=False, inverse_mapping=True, cc=True, cc_dict= cc_dict
            )
            # st()
            inv_loss_cc_tta = inv_loss_cc_tta * self.hparams.tta_coef
            losses.append(inv_loss_cc_tta)
            # st()

        if self.hparams.multi_task:
            # st()
            batch_2 = next(self.train_iterator_2)
            batch_2['text'] = batch_2['text'].to(self.device)
            batch_2['target'] = batch_2['target'].to(self.device)
            loss_2, accuracy_2, coeff_2, x_lhs_2, y_hat_rhs_2, y_rhs_2, attentions_2, values_2 = self._step(
                batch=batch_2, batch_idx=batch_idx, train=True
            )
            loss_2 = loss_2 * self.hparams.multi_coef
            losses.append(loss_2)
        if self.hparams.forward_forward_mode:
            inv_loss, inv_accuracy, inv_coeff, inv_x_lhs, inv_y_hat_rhs, inv_y_rhs, inv_attentions, inv_values = self._step(
                batch=batch, batch_idx=batch_idx, train=True, inverse_mapping=True
            )
            inv_loss = inv_loss * self.hparams.inv_coef
            losses.append(inv_loss)

        elif self.hparams.reverse_mode:
            inv_loss, inv_accuracy, inv_coeff, inv_x_lhs, inv_y_hat_rhs, inv_y_rhs, inv_attentions, inv_values = self._step(
                batch=batch, batch_idx=batch_idx, train=True, inverse_mapping=True, reverse_mode=True
            )
            inv_loss = inv_loss * self.hparams.inv_coef
            losses.append(inv_loss)

        total_loss = torch.mean(torch.stack(losses))

        self.fwd_time_in_epoch += time.time() - start

        # schedulers = self.trainer.lr_schedulers[0]
        # if self.current_epoch != self.next_train_epoch_to_log:
        #     return {"loss": loss}
        lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]

        output = {
            "loss": total_loss,
            'forward_loss': forward_loss,
            "partial_train_loss": coeff * total_loss,
            "partial_train_accuracy": coeff * accuracy,
            "learning_rate": torch.tensor([lr]),
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }

        if self.hparams.cyclic_consistency:
            output["inv_cc_loss"] = inv_loss_cc
            output["inv_cc_accuracy"] = inv_accuracy_cc

        if self.hparams['do_tta']:
            output["inv_loss_cc_tta"] = inv_loss_cc_tta
            output["inv_accuracy_cc_tta"] = inv_accuracy_cc_tta

        if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
            # outputs['inv_loss'] = inv_loss
            output['inv_partial_train_loss'] = inv_loss
            output['inv_partial_train_accuracy'] = inv_accuracy

        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs
        # st()
        self.training_step_outputs.append(output)

        if self.trainer.global_step % 1000 == 0:
            pth = '/home/mprabhud/sp/grok/model.pth'
            torch.save(self.transformer.state_dict(), pth)
            print(f'Saved model at {pth} at global step {self.trainer.global_step}')

        return output

    def on_train_epoch_end(self):
        """
        Used by pytorch_lightning
        Accumulates results of all forward training passes in this epoch

        :param outputs: a list of dicts from self.training_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        # st()
        epoch_is_to_be_logged = self.current_epoch == self.next_train_epoch_to_log
        epoch_is_to_be_logged = True

        outputs = self.training_step_outputs

        # st()

        if epoch_is_to_be_logged and len(outputs) > 0:
            self.next_train_epoch_to_log = max(
                int(1.01 * self.next_train_epoch_to_log),
                self.next_train_epoch_to_log + 1,
            )
            with torch.no_grad():
                try:
                    loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
                except Exception as e:
                    print("!" * 80)
                    print(outputs)
                    raise e
                perplexity = torch.exp(loss)
                accuracy = torch.stack(
                    [x["partial_train_accuracy"] for x in outputs]
                ).sum()

                forward_loss = torch.stack([x["forward_loss"] for x in outputs]).mean()

                if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
                    inv_loss = torch.stack([x["inv_partial_train_loss"] for x in outputs]).mean()
                    inv_perplexity = torch.exp(inv_loss)
                    inv_accuracy = torch.stack(
                        [x["inv_partial_train_accuracy"] for x in outputs]
                    ).mean()

                if self.hparams.do_tta:
                    inv_loss_cc_tta = torch.stack([x["inv_loss_cc_tta"] for x in outputs]).mean()
                    inv_accuracy_cc_tta = torch.stack(
                        [x["inv_accuracy_cc_tta"] for x in outputs]
                    ).mean()

                if self.hparams.cyclic_consistency:
                    inv_loss_cc = torch.stack([x["inv_cc_loss"] for x in outputs]).mean()
                    inv_accuracy_cc = torch.stack(
                        [x["inv_cc_accuracy"] for x in outputs]
                    ).mean()


            # avg_lr = torch.stack([x["learning_rate"] for x in outputs]).mean()
            # max_lr = torch.stack([x["learning_rate"] for x in outputs]).max()
            # last_lr = outputs[-1]["learning_rate"]
            # st()
            first_lr = outputs[0]["learning_rate"]

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                self._save_activations(outputs, ds="train")
            # st()
            logs = {
                "train_loss": loss,
                'forward_loss': forward_loss,
                "train_accuracy": accuracy,
                "train_perplexity": perplexity,
                "learning_rate": first_lr,
                "len_train_ds": len(self.train_dataset),
                "len_val_ds": len(self.val_dataset),
                "batches_per_epoch": self.batches_per_epoch,
                "time_per_epoch": time.time() - self.training_epoch_start_time,
                "fwd_time_in_epoch": self.fwd_time_in_epoch,
            }
            # st()

            if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
                logs['inv_loss'] = inv_loss
                # logs['inv_perplexity'] = torch.exp(inv_loss)
                # pass
                logs['inv_accuracy'] = inv_accuracy
                # st()

            if self.hparams.do_tta:
                logs['inv_loss_cc_tta'] = inv_loss_cc_tta
                logs['inv_accuracy_cc_tta'] = inv_accuracy_cc_tta


            if self.hparams.cyclic_consistency:
                logs['inv_loss_cc'] = inv_loss_cc
                logs['inv_accuracy_cc'] = inv_accuracy_cc

            # st()
            for k, v in logs.items():
                # self.log(k, v)
                self.logger.log_metrics({k: v}, step=self.trainer.global_step)

            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """
        # st()
        if self.next_epoch_to_eval < self.current_epoch:
            self.next_epoch_to_eval = self.current_epoch
        if self.current_epoch != self.next_epoch_to_eval:
            return {}


        with torch.no_grad():
            # st()
            loss, accuracy, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=False
            )
            # st()
            if self.hparams.forward_forward_mode:
                inv_loss, inv_accuracy, inv_coeff, inv_x_lhs, inv_y_hat_rhs, inv_y_rhs, inv_attentions, inv_values = self._step(
                    batch=batch, batch_idx=batch_idx, train=False, inverse_mapping=True
                )
            elif self.hparams.reverse_mode:
                inv_loss, inv_accuracy, inv_coeff, inv_x_lhs, inv_y_hat_rhs, inv_y_rhs, inv_attentions, inv_values = self._step(
                    batch=batch, batch_idx=batch_idx, train=False, inverse_mapping=True, reverse_mode=True
                )
            # st()



        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        self.val_accuracy.append(coeff * accuracy)
        partial_dict = {
            'val_loss': coeff * loss,
            'val_accuracy': coeff * accuracy,
        }


        if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
            output["inv_partial_val_loss"]  = inv_coeff * inv_loss
            output["inv_partial_val_accuracy"]  = inv_coeff * inv_accuracy

        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # st()
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        # st()
        outputs = self.validation_step_outputs
        validation_is_real = len(outputs) != 0

        if validation_is_real:
            self.next_epoch_to_eval = max(
                int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
            )

            # st()
            assert len(outputs) ==1

            loss = torch.stack([x["partial_val_loss"] for x in outputs]).mean()
            perplexity = torch.exp(loss)
            accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).mean()
            # st()
            max_val_accuracy = max(self.val_accuracy)

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                self._save_activations(outputs, ds="val")
            # st()
            logs = {
                "val_loss": loss,
                "val_accuracy": accuracy,
                'max_val_accuracy':max_val_accuracy,
                "val_perplexity": perplexity,
            }
            to_store = logs.copy()

            # make all values float
            for k, v in to_store.items():
                if isinstance(v, torch.Tensor):
                    to_store[k] = v.item()
            self.trainer_step_val_dict[self.trainer.global_step] = to_store

            if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
                inv_loss = torch.stack([x["inv_partial_val_loss"] for x in outputs]).mean()
                inv_perplexity = torch.exp(inv_loss)
                inv_accuracy = torch.stack([x["inv_partial_val_accuracy"] for x in outputs]).mean()
                logs['inv_val_loss'] = inv_loss
                logs['inv_val_perplexity'] = inv_perplexity
                logs['inv_val_accuracy'] = inv_accuracy

                # st()


            for name, param in self.named_parameters():
                # n parameters
                n_params = param.numel()
                # get the l2 norm of the parameter
                logs["paramnorm_" + name] = torch.norm(
                    param, 2
                ).detach().cpu().numpy() / np.sqrt(n_params)
            # st()
            # train accuracy
            device = self.transformer.embedding.weight.device
            train_data = self.train_dataset.data.to(device)
            training_data = {"text": train_data[:,:, :-1], "target": train_data[:,:, 1:]}
            # st()
            with torch.no_grad():
                # st()
                tr_loss, tr_acc, *_ = self._step(training_data, 0)
                logs["full_train_loss"] = tr_loss
                logs["full_train_acc"] = tr_acc

                if self.hparams.forward_forward_mode:
                    inv_tr_loss, inv_tr_acc, *_ = self._step(training_data, 0, inverse_mapping=True)
                    logs["inv_full_train_loss"] = inv_tr_loss
                    logs["inv_full_train_acc"] = inv_tr_acc
                elif self.hparams.reverse_mode:
                    inv_tr_loss, inv_tr_acc, *_ = self._step(training_data, 0, inverse_mapping=True, reverse_mode=True)
                    logs["inv_full_train_loss"] = inv_tr_loss
                    logs["inv_full_train_acc"] = inv_tr_acc

            if self.hparams.plot_pca_last_layer:
                if self.hparams.math_operator not in ['2x', '2x+1', 'x+11', 'x**3', 'pfactor']:
                    START_IDX = 39 # of 0
                    END_IDX =  136 # of 97
                    NUM_COMPONENTS = 20
                    C=97
                else:
                    START_IDX = 46 # of 0
                    END_IDX = 56 # of 10
                    NUM_COMPONENTS = 10
                    C=10

                def get_circles(layer):
                    fig,ax=plt.subplots(1,4,figsize=(20/3*4,4/3*4))
                    aa=[0,1,2,3,4,5,6,7] # put the desired dimensions here
                    for uu in range(0,8,2):
                        # ok, now let's manipulate the embedding weights
                        we=layer #
                        # now use scikit PCA to reduce the dimensionality of the embedding
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=NUM_COMPONENTS)
                        we2=pca.fit_transform(we.detach().cpu().numpy())
                        X=aa[uu]
                        Y=aa[uu+1]
                        ax1=ax[uu//2]
                        box = ax1.get_position()
                        box.y0-=0.09
                        box.y1-=0.09
                        ax1.set_position(box)
                        ax[uu//2].set_title(f'Circle from Principal Component {X+1}, {Y+1}',fontsize=14,y=1.03)
                        ax[uu//2].scatter(we2[:C,X],we2[:C,Y],c='r',s=20)
                        for i in range(C):
                            ax[uu//2].annotate(str(i), (we2[i,X],we2[i,Y]))
                    return fig

                def get_2d_pca(layer):
                    fig,ax=plt.subplots(NUM_COMPONENTS,NUM_COMPONENTS,figsize=(40,15))
                    # model.load_state_dict(torch.load(model_file,map_location=DEVICE))
                    # we=model.embed.W_E.T
                    we = layer
                    # now use scikit PCA to reduce the dimensionality of the embedding
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=NUM_COMPONENTS)
                    we2=pca.fit_transform(we.detach().cpu().numpy())
                    for i in range(NUM_COMPONENTS):
                        for j in range(NUM_COMPONENTS):
                            ax[i,j].scatter(we2[:,i],we2[:,j],s=5)
                    return fig

                def get_feature_viz(layer):
                    we=layer
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=NUM_COMPONENTS)
                    we2=pca.fit_transform(we.detach().cpu().numpy())
                    # make a line plot of each 97 components with each PCA having its own subplot

                    # subplot
                    fig, axs = plt.subplots(3, 4, figsize=(24, 12))

                    # plt.figure(figsize=(30,6))
                    for ix in range(min(NUM_COMPONENTS, 12)): # dont ask
                        #  visualize the PCA components with shape[0] on the x-axis
                        vs=we2[:,ix] # shape (97, 1)
                        # make a line plot of each 97 components

                        # plot in the subplot
                        axs[ix//4, ix%4].plot(vs)
                        axs[ix//4, ix%4].set_title(f"PCA {ix+1}")

                    return fig

                embed_viz = get_circles(self.transformer.embedding.weight[START_IDX:END_IDX, :])
                last_layer_viz = get_circles(self.transformer.linear.weight[START_IDX:END_IDX, :])

                embed_2d_viz = get_2d_pca(self.transformer.embedding.weight[START_IDX:END_IDX, :])
                last_layer_2d_viz = get_2d_pca(self.transformer.linear.weight[START_IDX:END_IDX, :])

                embed_feature_viz = get_feature_viz(self.transformer.embedding.weight[START_IDX:END_IDX, :])
                last_layer_feature_viz = get_feature_viz(self.transformer.linear.weight[START_IDX:END_IDX, :])


                captions = ['PCA of Embedding Weights', 'PCA of Last Layer Weights', '2D PCA of Embedding Weights', '2D PCA of Last Layer Weights', 'Feature Viz of Embedding Weights', 'Feature Viz of Last Layer Weights']

                # log figure to wandb
                self.logger.log_image(key='PCA', images=[embed_viz, last_layer_viz, embed_2d_viz, last_layer_2d_viz, embed_feature_viz, last_layer_feature_viz], step=self.trainer.global_step, caption=captions)

            for k, v in logs.items():
                # self.log(k, v)
                self.logger.log_metrics({k: v}, step=self.trainer.global_step)





            # if max_val_accuracy > 99.5:
            #     # st()
            #     exit()
        # save when self.trainer.global_step is a multiple of 1000

        self.validation_step_outputs.clear()
        # save a checkpoint if the epoch is a power of 2
        # if (
        #     self.current_epoch > 0
        #     and int(2 ** (int(np.log(self.current_epoch) / np.log(2))))
        #     == self.current_epoch
        # ):
        #     self.trainer.save_checkpoint(
        #         os.path.join(
        #             self.hparams.checkpoint_path,
        #             "epoch_" + str(self.current_epoch) + ".ckpt",
        #         )
        #     )

        if validation_is_real:
            return logs

    def test_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """

        loss, accuracy, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=False, reduction="none"
        )
        output = {
            "partial_test_loss": coeff * loss,
            "partial_test_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def test_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        loss = torch.cat([x["partial_test_loss"] for x in outputs], dim=0)  # .sum()
        # loss = list([x["partial_test_loss"] for x in outputs])  # .sum()
        perplexity = torch.exp(loss)
        accuracy = torch.cat([x["partial_test_accuracy"] for x in outputs], dim=0)

        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_perplexity": perplexity,
        }

        return {"test_loss": loss, "log": logs}

    def forward(self, *args, **kwargs) -> Any:
        """Passes all arguments directly to Tranformer.forward()"""
        return self.transformer(*args, **kwargs)


def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """
    # st()
    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path
    # st()
    # Create the model
    model = TrainableTransformer(hparams).float()

    # torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    hparams_dict = dict(hparams)

    # import wandb
    # wandb.init(config=hparams_dict)
    # st()
    group_name = hparams_dict['group']

    if group_name == 'none':
        group_name = None

    # st()
    if hparams_dict['debug']:
        logger = WandbLogger(project=hparams['project_name'],  config=hparams_dict, mode='disabled')
    else:
        logger = WandbLogger(project=hparams['project_name'], group=group_name, config=hparams_dict)
    # st()

    # checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     monitor="save_ckpt",
    #     mode="max",
    #     save_top_k=len(hparams.ckpt_epochs),
    #     verbose=False,
    # )

    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        # "flush_logs_every_n_steps": 1000,
    }
    # if torch.cuda.is_available() and hparams.gpu >= 0:
    #     trainer_args["gpus"] = [hparams.gpu]

    trainer = Trainer(**trainer_args)

    trainer.fit(model=model)  # type: ignore
    """
    margin = np.percentile(model.margin.detach().cpu().numpy(), 5)
    device = transformer.embedding.weight.device
    measures, bounds = metrics.calculate(
        transformer,
        transformer_init.to(device),
        device,
        dataset_size,
        margin,
        input_dim=hparams.d_model,
    )

    measures_file = os.path.join(logger.log_dir, "measures.json")
    bounds_file = os.path.join(logger.log_dir, "bounds.json")
    with open(measures_file, "w") as fh:
        json.dump(measures, fh)
    with open(bounds_file, "w") as fh:
        json.dump(bounds, fh)
    """
    val_df = pd.DataFrame(model.trainer_step_val_dict).transpose()
    # make 0th col name steps
    val_df.index.name = "steps"
    csv_folder = os.path.join(hparams.logdir, hparams.group)
    os.makedirs(csv_folder, exist_ok=True)
    csv_folder = os.path.join(csv_folder, f'op_{hparams.math_operator}')
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, f'val_{hparams.mode}_t{hparams.train_data_pct}.csv')
    val_df.to_csv(csv_path)
    return hparams.logdir


def compute_sharpness(hparams: Namespace, ckpts) -> None:
    """
    This is the compute_sharpness method. This loads a series of checkpoints in
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    model = TrainableTransformer(hparams).float()

    # torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    logger = CSVLogger(hparams.logdir)


    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        # "flush_logs_every_n_steps": 1000,
    }
    # if torch.cuda.is_available() and hparams.gpu >= 0:
    #     trainer_args["gpus"] = [hparams.gpu]

    trainer = Trainer(**trainer_args)

    for ckpt in ckpts:
        print(f"Loading checkpoint {ckpt}")
        # model = torch.load(ckpt)
        # model.load_state_dict(torch.load(ckpt))

        checkpoint = torch.load(ckpt)
        # print(dir(checkpoint), type(checkpoint), "Ckpt")
        # for k, v in checkpoint.items():
        #     print(k)
        # print(checkpoint["hyper_parameters"])

        hps = checkpoint["hyper_parameters"]
        hps = argparse.Namespace(**hps)
        model = TrainableTransformer(hps).float()
        model.load_state_dict(checkpoint["state_dict"])

        phi = get_sharpness(model.train_dataloader(), model)
        results = {}
        results[ckpt] = phi
        pickle.dump(results, open(f"results/results_SD-{i}.pkl", "wb"))


def add_args(parser=None) -> Namespace:
    """
    Parses the command line arguments

    :returns: an argparse.Namespace with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    # parser.add_argument("--checkpoint_period", type=int, default=1)
    parser = TrainableTransformer.add_model_specific_args(parser)
    return parser


class CustomAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not weight_decay_form in ["to_zero", "to_init", "jiggle", "honest"]:
            raise ValueError(
                f"Invalid weight decay form: {weight_decay_form}, should be one of ['to_zero', 'to_init', 'jiggle']"
            )
        # if not 0.0 <= weight_decay:
        #     raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            noise_factor=noise_factor,
            weight_decay_form=weight_decay_form,
        )
        super(CustomAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "honest":
                        grad = grad + group["weight_decay"] * p.detach()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_(
                            (state["init"] - p) * (group["lr"] * group["weight_decay"])
                        )
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(
                            torch.exp(
                                torch.randn(1).cuda()
                                * (group["lr"] * group["weight_decay"])
                            )
                        )
                    elif group["weight_decay_form"] == "honest":
                        pass
                    else:
                        raise ValueError(
                            f"Invalid weight decay form: {group['weight_decay_form']}"
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                step_size = group["lr"] / bias_correction1

                upd = exp_avg / denom
                # add uniform gaussian noise to the update
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                # if group['noise_factor'] > 0:
                #     upd *= torch.exp(torch.randn_like(upd) * group['noise_factor'])
                p.add_(-step_size * upd)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        grad_norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        print("grad norms is ", grad_norms, "!" * 1000)
        norm = torch.norm(
            torch.stack(grad_norms),
            p=2,
        )
        return norm