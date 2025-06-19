"""
Base Dataset.
"""

import torch
import os 

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement

import pandas as pd 
import numpy as np

class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    using all csv files like
    ID  Sequence    Label
    1   ATGCTAGCTAG 0
    """

    def __init__(
            self,
            benchmark_name:str,
            dataset_name: str,
            dest_path: str,

            split: str,
            max_length: int,
            d_output: int = 2,  # default binary classification
            tokenizer: Optional[object] = None,
            tokenizer_name: Optional[str] = None,
            use_padding: Optional[bool] = None,
            add_eos: bool = False,
            return_mask: bool = True,

            rc_aug: bool = False, # only used for caduceus
            conjoin_train: bool = False, # only used for caduceus
            conjoin_test: bool = False, # only used for caduceus
            return_augs: bool = False, # only used for caduceus
    ):

        self.benchmark_name = benchmark_name
        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.return_mask = return_mask        
        self.split = split

        # the following is only used for caduceus
        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

        print(f"Testing {self.benchmark_name} dataset {self.dataset_name} with {self.tokenizer_name} tokenizer")
        
        base_path = os.path.join(dest_path, dataset_name)

    @abstractmethod
    def __len__(self):
        '''
        Modify this based on the dataset
        '''
        pass

    @staticmethod
    def group_by_kmer(seq: str, kmer: int) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), kmer)).upper()

    def mapping_from_rna_to_dna(self, seq: str) -> str:
        return seq.replace("U", "T")
    
    def mapping_from_dna_to_rna(self, seq: str) -> str:
        return seq.replace("T", "U")

    

    def _tokenize(self, x) -> Tuple[torch.LongTensor, Union[torch.BoolTensor, List[int]]]:
        '''
        Tokenize the input sequence based on different pretrained model .
        '''
        # make sure the sequence is upper case
        x = x.upper()
        text = x

        if self.tokenizer_name == 'caduceus' and (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            # print("Using RC Augmentation for Caduceus")
            x = string_reverse_complement(x)

        if 'dnabert' in self.tokenizer_name and self.tokenizer_name not in  ['dnabert2','dnaberts'] :
            kmer = self.tokenizer_name.split('_')[-1]
            x = self.group_by_kmer(x, int(kmer))
            seq = self.tokenizer(
                x,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            attention_mask = seq["attention_mask"]
        
        # elif self.tokenizer_name == 'evo':
        #     # seq_ids = torch.tensor(
        #     #     self.tokenizer.tokenize(x),
        #     #     dtype=torch.int,
        #     # )
        #     # x: (B, L)
        #     # padding manually for evo
        #     x = torch.tensor(
        #         self.tokenizer.tokenize(x),
        #         dtype=torch.int,
        #     ).unsqueeze(0)

        #     # pad_id from vortex/model/tokenizer.py
        #     pad_id = 1
            
        #     if len(x) < self.max_length:
        #         seq_ids = x + [pad_id] * (self.max_length - len(x))
        #         attention_mask = torch.cat((torch.ones(1, len(x))), torch.zeros(1, self.max_length - len(x)))
        #     else:
        #         seq_ids = x[:self.max_length]
        #         attention_mask = torch.ones(1, self.max_length)


        elif self.tokenizer_name in ['genalm', 'dnabert2', 'dnaberts', 'nt', 'ntv2', 'protbert', 'grover', 'omnibiote', 'caduceus', 'hyena', 'DNAHLM', 'gpn', 'char', 'llamadna', 'mistraldna', 'evo', 'generator', 'evo2']:
            seq = self.tokenizer(
                x,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )

            seq_ids = seq["input_ids"]
            if self.tokenizer_name == 'caduceus':
                # caduceus' tokenizer does not have attention_mask
                # process attention mask manually
                pad_token_id = self.tokenizer.pad_token_id
                attention_mask = [1 if token_id != pad_token_id else 0 for token_id in seq_ids]
            else:
                attention_mask = seq["attention_mask"]

        elif self.tokenizer_name in ['rnafm', 'rnabert', 'rnaernie', 'splicebert', 'omnigenome']:
            # for RNA models, we need to convert T to U
            # print('for RNA models, we need to convert T to U')
            x = self.mapping_from_dna_to_rna(x)
            seq = self.tokenizer(
                x,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            attention_mask = seq["attention_mask"]
        else:
            raise ValueError(f"Tokenizer {self.tokenizer_name} not supported")

        
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq_ids.append(self.tokenizer.sep_token_id)

        if self.tokenizer_name == 'caduceus' and (self.conjoin_train or (self.conjoin_test and self.split != "train")):
            x_rc = string_reverse_complement(x)
            seq_rc = self.tokenizer(
                x_rc,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True
            )
            seq_rc_ids = seq_rc["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)
            pad_token_id = self.tokenizer.pad_token_id
            # print(seq_ids)
            # attention_mask = [1 if token_id != pad_token_id else 0 for token_id in seq_ids]
        else:
            # convert to tensor
            seq_ids = seq_ids if isinstance(seq_ids, torch.Tensor) else torch.tensor(seq_ids)
            seq_ids = seq_ids.long()


        
        return seq_ids, attention_mask
    
    @abstractmethod
    def __getitem__(self, idx):
        '''
        Modify this based on the dataset
        '''
        pass






        

class GeneralDataset(BaseDataset):
    """
    using all csv files like
    ID  Sequence    Label
    1   ATGCTAGCTAG 0
    """
    def __init__(self,*args, **kwargs):

        super().__init__(*args, **kwargs)
        print(self.conjoin_test)
        print(self.conjoin_train)
        print(f'using {self.dataset_name} from {self.benchmark_name}')
        base_path = os.path.join(self.dest_path, self.dataset_name)
        # For NT tasks, we use data from InstaDeepAI/nucleotide_transformer_downstream_tasks
        
        if os.path.exists(self.dest_path):
            print(f"Already downloaded {self.dataset_name}")
        else:
            raise FileNotFoundError(f"Dataset not found at {base_path}")
        try:
            data = pd.read_csv(os.path.join(base_path, f"{self.split}.csv"))
        except:
            print(f'using tsv for {self.dataset_name} {self.split}')
            data = pd.read_csv(os.path.join(base_path, f"{self.split}.tsv"), sep='\t')
        
        self.all_seqs = data['sequence'].values
        self.all_labels = data['label'].values
        self.all_ids = data['ID'].values

        # length = len(self.all_seqs)
        # print(f"Loaded {length} sequences from {self.dataset_name} for {self.split} split")
        if self.tokenizer_name == 'caduceus' and (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            print("Using RC Augmentation for Caduceus")

    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]
        id = self.all_ids[idx]
        
        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)

        # if self.benchmark_name in ['RGB', 'RnaBench', 'BEACON']:
        # mapping from rna to dna
        x = self.mapping_from_rna_to_dna(x)
        
        seq_ids, attention_mask = self._tokenize(x)

        if self.benchmark_name in ['ours_multilabel_classification']:
            if type(y) == str:
                target = torch.LongTensor(list(map(int, y.strip('[]').split(','))))
            else:
                target = torch.LongTensor([y])
            # print(id)
        elif self.dataset_name == "ProgrammableRNASwitches":
            if type(y) == str:
                target = torch.FloatTensor(list(map(float, y.strip('[]').split(','))))
            else:
                target = torch.FloatTensor([y])
        elif self.benchmark_name in ['ours_regression']:
            if 'bulk_rna_expression' in self.dataset_name:
                target = torch.FloatTensor(list(map(float, y.strip('[]').split(','))))
            else:
                target = torch.FloatTensor([y])
        else:
            target = torch.LongTensor([y])

        if self.return_mask:
            return id, seq_ids, target, {"mask": torch.BoolTensor(attention_mask)}
        else:
            return id, seq_ids, target


class VariationDataset(BaseDataset):
    """
    using all csv files like
    ID  sequence variant_sequence    Label
    1   ATGCTAGCTAG 0
    """
    def __init__(self,*args, **kwargs):

        super().__init__(*args, **kwargs)
        print(self.conjoin_test)
        print(self.conjoin_train)
        print(f'using {self.dataset_name} from {self.benchmark_name}')
        base_path = os.path.join(self.dest_path, self.dataset_name)
        
        if os.path.exists(self.dest_path):
            print(f"Already downloaded {self.dataset_name}")
        else:
            raise FileNotFoundError(f"Dataset not found at {base_path}")

        data = pd.read_csv(os.path.join(base_path, f"{self.split}.csv"))
        
        self.all_seqs = data['sequence'].values
        self.all_var_seqs = data['variant_sequence'].values
        self.all_labels = data['label'].values
        self.all_ids = data['ID'].values


        if self.tokenizer_name == 'caduceus' and (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            print("Using RC Augmentation for Caduceus")

    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        x_var = self.all_var_seqs[idx]
        y = self.all_labels[idx]
        id = self.all_ids[idx]
        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)
            x_var = string_reverse_complement(x_var)

        
        seq_ids, attention_mask = self._tokenize(x)
        seq_var_ids, attention_mask_var = self._tokenize(x_var)

        # if self.benchmark_name in ['true_ds_protein', 'beacon_regression']:
        #     target = torch.FloatTensor([y])
        # else:
        target = torch.LongTensor([y])

        if self.return_mask:
            return id, seq_ids, seq_var_ids, target, {"mask": torch.BoolTensor(attention_mask)}
        else:
            return id, seq_ids, seq_var_ids, target
        

class PairedDataset(BaseDataset):
    """
    using all csv files like
    ID  sequence1 sequence2    Label
    1   ATGCTAGCTAG 0
    """
    def __init__(self,*args, **kwargs):

        super().__init__(*args, **kwargs)
        print(self.conjoin_test)
        print(self.conjoin_train)
        print(f'using {self.dataset_name} from {self.benchmark_name}')
        base_path = os.path.join(self.dest_path, self.dataset_name)
        
        if os.path.exists(self.dest_path):
            print(f"Already downloaded {self.dataset_name}")
        else:
            raise FileNotFoundError(f"Dataset not found at {base_path}")

        data = pd.read_csv(os.path.join(base_path, f"{self.split}.csv"))
        

        if 'Enhancer_Promoter_Interaction' in self.dataset_name:
            self.all_seq1 = data['enhancer'].values
            self.all_seq2 = data['promoter'].values
        
        else:
            self.all_seq1 = data['sequence1'].values
            self.all_seq2 = data['sequence2'].values
        
        self.all_labels = data['label'].values
        self.all_ids = data['ID'].values


        if self.tokenizer_name == 'caduceus' and (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            print("Using RC Augmentation for Caduceus")

    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        x_1 = self.all_seq1[idx]
        x_2 = self.all_seq2[idx]
        y = self.all_labels[idx]
        id = self.all_ids[idx]
        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x_1 = string_reverse_complement(x_1)
            x_2 = string_reverse_complement(x_2)

        
        seq_ids_1, attention_mask_1 = self._tokenize(x_1)
        seq_ids_2, attention_mask_2 = self._tokenize(x_2)

        # if self.benchmark_name in ['true_ds_protein', 'beacon_regression']:
        #     target = torch.FloatTensor([y])
        # else:
        target = torch.LongTensor([y])

        if self.return_mask:
            return id, seq_ids_1, seq_ids_2, target, {"mask": torch.BoolTensor(attention_mask_1)}
        else:
            return id, seq_ids_1, seq_ids_2, target


