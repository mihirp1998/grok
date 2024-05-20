#!/usr/bin/env python

import argparse
import copy
# import data
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
        hparams_dict = hparams

        self.f_val_accuracy = []

        for key in hparams.keys():
            self.hparams[key]=hparams_dict[key]

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.prepare_data()

        if hparams.forward_train or hparams.tta_train:
            self.forward_transformer = Transformer(
                hparams.n_layers,
                hparams.n_heads,
                hparams.d_model,
                hparams.dropout,
                hparams.max_context_len,
                len(self.train_dataset.tokenizer),
                hparams.embed_style,
                hparams.non_linearity,
                weight_noise=self.hparams.weight_noise,
            )
        
        if hparams.inverse_train or hparams.tta_train:
            self.inverse_transformer = Transformer(
                hparams.n_layers,
                hparams.n_heads,
                hparams.d_model,
                hparams.dropout,
                hparams.max_context_len,
                len(self.train_dataset.tokenizer),
                hparams.embed_style,
                hparams.non_linearity,
                weight_noise=self.hparams.weight_noise,
            )
        
        
        self.margin = torch.Tensor([0])
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0

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
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
        )

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        device = self.forward_transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset,
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
        device = self.forward_transformer.embedding.weight.device
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
        tta_dict: Dict = None,
        tta: bool = False,
        train: bool = True,
        reduction: str = "mean",
        grads: bool = False,
        inverse_mapping: bool = False,
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
        if tta:
            x = batch["text"]  # shape = batchsize * context_len
            y = batch["target"]  # shape = batchsize * context_len            
        else:
            if inverse_mapping:
                x = batch["text"][:,1]  # shape = batchsize * context_len
                y = batch["target"][:,1]  # shape = batchsize * context_len
            else:
                x = batch["text"][:,0]  # shape = batchsize * context_len
                y = batch["target"][:,0]  # shape = batchsize * context_len
                
            tta_dict = None
        
        if inverse_mapping:
            y_hat, attentions, values = self.inverse_transformer(
                x=x, save_activations=self.hparams.save_activations, inverse_mapping=inverse_mapping, tta=tta, tta_dict=tta_dict  # type: ignore
            )  # shape = batchsize * context_len * vocab_size
                    
        else:
            y_hat, attentions, values = self.forward_transformer(
                x=x, save_activations=self.hparams.save_activations, inverse_mapping=inverse_mapping  # type: ignore
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
        zero_tensor = torch.tensor(0.0, device=self.forward_transformer.embedding.weight.device)
        # st()

        forward_loss = zero_tensor
        inverse_loss = zero_tensor
        tta_loss = zero_tensor
        
        coeff = zero_tensor
        forward_accuracy = zero_tensor
        
        inverse_coeff = zero_tensor
        inverse_accuracy = zero_tensor
        
        f_coeff_tta = zero_tensor
        f_accuracy_tta = zero_tensor
        
        inv_coeff_tta = zero_tensor
        inv_accuracy_tta = zero_tensor
        
        
        
        if self.hparams.forward_train:
            forward_loss, forward_accuracy, coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=True, 
            )
            losses.append(forward_loss)
            
        if self.hparams.inverse_train:
            inverse_loss, inverse_accuracy, inverse_coeff, x_lhs, y_hat_rhs, y_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=True, inverse_mapping=True
            )
            losses.append(inverse_loss)        
        
        # if forward_accuracy

        if self.hparams.tta_train:
            batch_val = next(self.val_dataloader())
            
            if self.hparams.val_batchify:
                index_rand = torch.randint(0,batch_val['text'].shape[0], (512,))
                batch_val['text'] = batch_val['text'][index_rand]
                batch_val['target'] = batch_val['target'][index_rand]
            
            
            _, f_accuracy_tta, f_coeff_tta, x_lhs_tta, y_hat_rhs_tta, y_rhs_tta, attentions_tta, values_tta = self._step(
                batch=batch_val, batch_idx=batch_idx, train=False
            )
            

            eq_token_index = torch.tensor(self.train_dataset.tokenizer.stoi["="]).repeat(batch_val['text'].shape[0])[:,None].to(batch_val['text'].device)
            eos_token = torch.tensor(self.train_dataset.tokenizer.stoi["<|eos|>"]).repeat(batch_val['text'].shape[0])[:,None].to(batch_val['text'].device)            
            batch_val_cc = copy.deepcopy(batch_val)
            
            
            data_tmp = torch.cat([eos_token, y_rhs_tta[:,:-1], eq_token_index, x_lhs_tta[:,1:], eos_token], dim=1)        
            assert (batch_val['text'][0][1] == data_tmp[:,:-1][0]).all()
            
            data = torch.cat([eos_token, y_rhs_tta[:,:-1], eq_token_index, x_lhs_tta[:,1:], eos_token], dim=1)        

            batch_val_cc['text']  = data[:,:-1]
            batch_val_cc['target']  = data[:,1:]
            
            tta_dict = {}
            
            tta_dict['y_hat_rhs'] = y_hat_rhs_tta
            tta_dict['eos_token'] = eos_token
            tta_dict['y_rhs'] = y_rhs_tta[:,:-1]
            tta_dict['eq_token_index'] = eq_token_index
            tta_dict['x_lhs'] = x_lhs_tta[:,1:]

            tta_loss, inv_accuracy_tta, inv_coeff_tta, inv_x_lhs_tta, inv_y_hat_rhs_tta, inv_y_rhs_tta, inv_attentions_tta, inv_values_tta = self._step(
                batch=batch_val_cc, batch_idx=batch_idx, train=False, inverse_mapping=True, tta=True, tta_dict= tta_dict
            )
            tta_loss = tta_loss * self.hparams.tta_coef
            losses.append(tta_loss)
            # st()/

        # st()
        total_loss = torch.mean(torch.stack(losses))

        self.fwd_time_in_epoch += time.time() - start

        lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        
        
        output = {
            "loss": total_loss,
            'forward_train_accuracy': coeff * forward_accuracy,
            'inverse_train_accuracy': inverse_coeff * inverse_accuracy,
            'forward_train_accuracy_tta': f_coeff_tta * f_accuracy_tta,
            'inverse_train_accuracy_tta': inv_coeff_tta * inv_accuracy_tta,   
            'forward_train_loss': coeff * forward_loss,
            'inverse_train_loss': inverse_coeff * inverse_loss,
            'tta_train_loss': inv_coeff_tta* tta_loss,
            # "partial_train_loss": coeff * total_loss,
            # "partial_train_accuracy": coeff * accuracy,            
            # "learning_rate": torch.tensor([lr]),
            # "y_hat_rhs": y_hat_rhs,
            # "partial_attentions": attentions,
            # "partial_values": values,
        }
        
        # if self.current_epoch == 0:
        #     output["x_lhs"] = x_lhs

        self.training_step_outputs.append(output)

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
        outputs = self.training_step_outputs
        
        if len(outputs) > 0:
            with torch.no_grad():
                forward_train_accuracy = torch.stack([x["forward_train_accuracy"] for x in outputs]).sum()
                inverse_train_accuracy = torch.stack([x["inverse_train_accuracy"] for x in outputs]).sum()
                
                if self.hparams.val_batchify:
                    forward_train_accuracy_tta = torch.stack([x["forward_train_accuracy_tta"] for x in outputs]).sum()
                    inverse_train_accuracy_tta = torch.stack([x["inverse_train_accuracy_tta"] for x in outputs]).sum()
                else:
                    forward_train_accuracy_tta = torch.stack([x["forward_train_accuracy_tta"] for x in outputs]).mean()
                    inverse_train_accuracy_tta = torch.stack([x["inverse_train_accuracy_tta"] for x in outputs]).mean()                    
                    # if self.hparams.tta_train:
                    #     st()                    

                
                forward_train_loss = torch.stack([x["forward_train_loss"] for x in outputs]).sum()
                inverse_train_loss = torch.stack([x["inverse_train_loss"] for x in outputs]).sum()                
                tta_train_loss = torch.stack([x["tta_train_loss"] for x in outputs]).sum()                
            
            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                self._save_activations(outputs, ds="train")

            logs = {
                "forward_train_accuracy": forward_train_accuracy,
                'inverse_train_accuracy': inverse_train_accuracy,
                "forward_train_accuracy_tta": forward_train_accuracy_tta,
                "inverse_train_accuracy_tta": inverse_train_accuracy_tta,
                'forward_train_loss': forward_train_loss,
                "inverse_train_loss": inverse_train_loss,
                "tta_train_loss": tta_train_loss,                
            }

            # st()
            
            for k, v in logs.items():
                self.log(k, v)
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
            f_loss, f_accuracy, f_coeff, f_x_lhs, f_y_hat_rhs, f_y_rhs, f_attentions, f_values = self._step(
                batch=batch, batch_idx=batch_idx, train=False
            )

            inv_loss, inv_accuracy, inv_coeff, inv_x_lhs, inv_y_hat_rhs, inv_y_rhs, inv_attentions, inv_values = self._step(
                batch=batch, batch_idx=batch_idx, train=False, inverse_mapping=True
            )


        output = {
            "f_partial_val_loss": f_coeff * f_loss,
            "f_partial_val_accuracy": f_coeff * f_accuracy,
        }


        output["inv_partial_val_loss"]  = inv_coeff * inv_loss
        output["inv_partial_val_accuracy"]  = inv_coeff * inv_accuracy
        
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
        
        outputs = self.validation_step_outputs
        validation_is_real = len(outputs) != 0

        if validation_is_real:
            self.next_epoch_to_eval = max(
                int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
            )
            
            # st()
            assert len(outputs) ==1

            f_loss = torch.stack([x["f_partial_val_loss"] for x in outputs]).sum()
            f_perplexity = torch.exp(f_loss)
            f_accuracy = torch.stack([x["f_partial_val_accuracy"] for x in outputs]).sum()
            
            
            self.f_val_accuracy.append(f_accuracy)

            f_max_val_accuracy = max(self.f_val_accuracy)
            
            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                self._save_activations(outputs, ds="val")
            
            logs = {
                "f_val_loss": f_loss,
                "f_val_accuracy": f_accuracy,
                'f_max_val_accuracy':f_max_val_accuracy,
            }
            to_store = logs.copy()

            # make all values float
            for k, v in to_store.items():
                if isinstance(v, torch.Tensor):
                    to_store[k] = v.item()
            self.trainer_step_val_dict[self.trainer.global_step] = to_store

            # if self.hparams.forward_forward_mode or self.hparams.reverse_mode:
            inv_loss = torch.stack([x["inv_partial_val_loss"] for x in outputs]).sum()
            inv_accuracy = torch.stack([x["inv_partial_val_accuracy"] for x in outputs]).sum()
            logs['inv_val_loss'] = inv_loss
            logs['inv_val_accuracy'] = inv_accuracy


            # for name, param in self.named_parameters():
            #     # n parameters
            #     n_params = param.numel()
            #     # get the l2 norm of the parameter
            #     logs["paramnorm_" + name] = torch.norm(
            #         param, 2
            #     ).detach().cpu().numpy() / np.sqrt(n_params)

            # train accuracy
            device = self.forward_transformer.embedding.weight.device
            train_data = self.train_dataset.data.to(device)
            training_data = {"text": train_data[:,:, :-1], "target": train_data[:,:, 1:]}
            # st()
            # st()
            with torch.no_grad():
                # st()
                # if self.hparams.forward_train:
                tr_loss, tr_acc, *_ = self._step(training_data, 0)
                logs["f_train_loss"] = tr_loss
                logs["f_train_acc"] = tr_acc

                if self.hparams.steps_to_tta > self.trainer.global_step  and tr_acc == 100 and not self.hparams.tta_train:
                    self.hparams.tta_train = True
                    self.hparams.inverse_train = False
                    self.hparams.forward_train = False
                    # st()
                    
                
                # if self.hparams.inverse_train:
                inv_tr_loss, inv_tr_acc, *_ = self._step(training_data, 0, inverse_mapping=True)
                logs["inv_train_loss"] = inv_tr_loss
                logs["inv_train_acc"] = inv_tr_acc


            # st()

            for k, v in logs.items():
                self.log(k, v)

            # if max_val_accuracy > 99.5:
            #     # st()
            #     exit()
        
        self.validation_step_outputs.clear()

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
    csv_path = os.path.join(csv_folder, f'val_{hparams.mode}.csv')
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
