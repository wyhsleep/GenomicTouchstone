# import inspect
from typing import List

import torch.nn as nn
import torch
from einops import rearrange

import src.models.nn.utils as U
import src.tasks.metrics as M
import torchmetrics as tm
from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection


class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        try:
            metrics = metrics[0]
        except:
            pass
        self.metric_names = to_list(metrics)

        # print('metrics: ', self.metric_names)
        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

       # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False
                )
            elif name in ['MultilabelAUROC', 'MultilabelAveragePrecision']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_labels=self.dataset.d_output
                )
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k
                )
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        id, x, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x, w = encoder(x, **z)
        x, state = model(x, **w, state=_state)

        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * self.c





class MultiClass(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_metrics = {}
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec] = M.output_metric_fns[name](spec_idx)
            elif name in ['precision_species', 'recall_species']:
                self.continual_metrics[name] = M.output_metric_fns[name](num_classes=len(self.dataset.species))

    def metrics(self, x, y, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name.endswith('_per_class'):
                    for spec_idx, spec in enumerate(self.dataset.species):
                        self.continual_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].to(
                            x.device)
                        self.continual_metrics[name + '_' + spec].update(x, y)
                        output_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].compute()
                elif name in ['precision_species', 'recall_species']:
                    self.continual_metrics[name] = self.continual_metrics[name].to(x.device)
                    metrics = self.continual_metrics[name](x, y)
                    for spec_idx, spec in enumerate(self.dataset.species):
                        output_metrics[name[:-7] + spec] = metrics[spec_idx]
                else:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        return {**output_metrics, **loss_metrics}

    def _reset_torchmetrics(self, prefix=None):
        super()._reset_torchmetrics(prefix)
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec].reset()
        

class Regression(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def metrics(self, x, y, len_batch=None, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name in ['mse', 'mae', 'r2', 'spearman', 'mse_masked', 'r2_structure']:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        # print('loss_metrics: ', loss_metrics)
        return {**output_metrics, **loss_metrics}

class VariationBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        id, x1, x2, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x1, w = encoder(x1, **z)
        x2, w = encoder(x2, **z)
        x1, state = model(x1, **w, state=_state)
        x2, state = model(x2, **w, state=_state)
        x = x2 - x1
        
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w

   




class VariationMultiClass(MultiClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        id, x1, x2, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x1, w = encoder(x1, **z)
        x2, w = encoder(x2, **z)
        x1, state = model(x1, **w, state=_state)
        x2, state = model(x2, **w, state=_state)
        x = x2 - x1
        
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w



class PairedBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        id, x1, x2, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x1, w = encoder(x1, **z)
        x2, w = encoder(x2, **z)
        x1, state = model(x1, **w, state=_state)
        x2, state = model(x2, **w, state=_state)
        x = x2 + x1
        # x = torch.cat([x1, x2], dim=1)
        
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w
    

class PairedMultiClass(MultiClass):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        id, x1, x2, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x1, w = encoder(x1, **z)
        x2, w = encoder(x2, **z)
        x1, state = model(x1, **w, state=_state)
        x2, state = model(x2, **w, state=_state)
        x = x2 + x1
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w
    

registry = {
    'base': BaseTask,
    'multiclass': MultiClass,
    'regression': Regression,
    'variation_base': VariationBaseTask,
    'variation_multiclass': VariationMultiClass,
    'paired_base': PairedBaseTask,
    'paired_multiclass': PairedMultiClass,
    
}
