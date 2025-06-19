"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import copy
from typing import Any, List, Union

import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.datasets.base_dataset import GeneralDataset, VariationDataset, PairedDataset
logger = src.utils.train.get_logger(__name__)




#TODO: refactor the init function
class Base(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "base"  # this name is how the dataset config finds the right dataloader

    def __init__(self, dest_path = None, dataset_name = None,
                 tokenizer_name=None, tokenizer_path = None, max_length=1024, d_output=2,
                 rc_aug=False, conjoin_train=False, conjoin_test=False, use_padding=True,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, padding_side="left",
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, shuffle=False,
                 num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 return_mask=None, 
                 *args, **kwargs):
        self.dest_path = dest_path
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer_path = tokenizer_path
        self.d_output = d_output
        self.use_padding = use_padding
        
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.padding_side = padding_side,
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.return_mask = return_mask

        # handle if file paths are None (default paths)

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        # To be instantiated in `setup`
        self.tokenizer = None
        self.vocab_size = 0

    def _init_tokenizer(self):
        """
        Initialize the tokenizer.
        """
        if self.tokenizer_name in ['hyena', 'char']:
            # for hyena
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
                padding_side=self.padding_side,
            )
            
        elif self.tokenizer_name in ["genalm", "nt", "ntv2", "grover", "caduceus", "gpn", "omnigenome", 'mistraldna', 'generator'] or 'dnabert' in self.tokenizer_name:
            logger.info(f"**Using {self.tokenizer_name} tokenizer from hugging face**")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        elif self.tokenizer_name in ['rnafm', 'rnabert', 'rnaernie', 'splicebert']:
            from multimolecule import RnaTokenizer
            logger.info(f"**Using {self.tokenizer_name} tokenizer from multimolecule**")
            self.tokenizer = RnaTokenizer.from_pretrained(f'multimolecule/{self.tokenizer_name}', trust_remote_code=True)
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")
    
    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(
                self.dataset_train,
                **distributed_sampler_kwargs
            ) if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    "epoch": self.fast_forward_epochs,
                    "counter": self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler, **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # kwargs["shuffle"] = True
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval, **kwargs)

    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
            # TD [2022-08-07] ["epoch_loop.batch_progress"]["total"]["completed"] is 1 iteration
            # behind, so we're using the optimizer"s progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
                "completed"]
        # At this point the train loader hasn't been constructed yet


class Genearal(Base):
    _name_ = "generaldataset"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, benchmark_name, train_val_split_seed, padding_side="left", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_name = benchmark_name
        self.padding_side = padding_side
        self.train_val_split_seed = train_val_split_seed
    def setup(self, stage=None):
        self._init_tokenizer()
        
        self.dataset_train, self.dataset_test = [
            GeneralDataset(
                split=split,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                use_padding=self.use_padding,
                benchmark_name=self.benchmark_name,
                rc_aug=self.rc_aug,
                dataset_name=self.dataset_name,
                dest_path=self.dest_path,
                return_mask=self.return_mask,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split in ["train", "test"]
        ]
        self.dataset_val = GeneralDataset(
            split="val",
            tokenizer=self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            use_padding=self.use_padding,
            benchmark_name=self.benchmark_name,
            rc_aug=self.rc_aug,
            dataset_name=self.dataset_name,
            dest_path=self.dest_path,
            return_mask=self.return_mask,
            conjoin_train=self.conjoin_train,
            conjoin_test=self.conjoin_test,
            return_augs=False
        )


class Variation(Base):
    _name_ = "variationdataset"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, benchmark_name, train_val_split_seed, padding_side="left", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_name = benchmark_name
        self.padding_side = padding_side
        self.train_val_split_seed = train_val_split_seed
    def setup(self, stage=None):
        self._init_tokenizer()
        
        self.dataset_train, self.dataset_test = [
            VariationDataset(
                split=split,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                use_padding=self.use_padding,
                benchmark_name=self.benchmark_name,
                rc_aug=self.rc_aug,
                dataset_name=self.dataset_name,
                dest_path=self.dest_path,
                return_mask=self.return_mask,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split in ["train", "test"]
        ]
        if 'ours' in self.benchmark_name:
            self.dataset_val = VariationDataset(
                split="val",
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                use_padding=self.use_padding,
                benchmark_name=self.benchmark_name,
                rc_aug=self.rc_aug,
                dataset_name=self.dataset_name,
                dest_path=self.dest_path,
                return_mask=self.return_mask,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
        else:
            raise ValueError(f"Unknown benchmark name: {self.benchmark_name}")

class Paired(Base):
    _name_ = "paireddataset"
    l_output = 0  # need to set this for decoder to work correctly


    def __init__(self, benchmark_name, train_val_split_seed, padding_side="left", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_name = benchmark_name
        self.padding_side = padding_side
        self.train_val_split_seed = train_val_split_seed
    
    def setup(self, stage=None):
        self._init_tokenizer()
        
        self.dataset_train, self.dataset_test = [
            PairedDataset(
                split=split,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                use_padding=self.use_padding,
                benchmark_name=self.benchmark_name,
                rc_aug=self.rc_aug,
                dataset_name=self.dataset_name,
                dest_path=self.dest_path,
                return_mask=self.return_mask,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split in ["train", "test"]
        ]
        if 'ours' in self.benchmark_name:
            self.dataset_val = PairedDataset(
                split="val",
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                use_padding=self.use_padding,
                benchmark_name=self.benchmark_name,
                rc_aug=self.rc_aug,
                dataset_name=self.dataset_name,
                dest_path=self.dest_path,
                return_mask=self.return_mask,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
        else:
            raise ValueError(f"Unknown benchmark name: {self.benchmark_name}")