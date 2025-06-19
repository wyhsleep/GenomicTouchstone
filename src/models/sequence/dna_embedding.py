"""DNA Embedding Model.

Backbones from LM pre-training models, used for downstream tasks.
"""

from functools import partial
from omegaconf import OmegaConf
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, GPT2LMHeadModel, AutoConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
import torch
import sys
# require this to import the model from OmniBioTE dircetly torch.load

import torch.nn as nn
try:
    from flash_attn.utils.generation import GenerationMixin
except ImportError:
    GenerationMixin = None
# from mamba_ssm.models.config_mamba import MambaConfig

# from mamba_ssm.models.mixer_seq_simple import MixerModel
# from mamba_ssm.models.mixer_seq_simple import _init_weights as _init_weights_mamba

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None


# from src.models.caduceus.configuration_caduceus import CaduceusConfig
# from src.models.caduceus.modeling_caduceus import Caduceus
from src.models.sequence.long_conv_lm import LMBackbone
from src.models.sequence.long_conv_lm import _init_weights
from src.models.sequence.lora_utils import replace_conv1d_with_lora, freeze_non_lora_params, lora_target_modules_mapping, count_params

class DNAEmbeddingModel(nn.Module, GenerationMixin):
    """DNA Embedding Model.

    Same as ConvLMHeadModel (in long_conv_lm.py), except no decoder head, we just pass back the hidden states for
    downstream tasks.
    """

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 checkpoint_mlp=False,
                 checkpoint_mixer=False,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, sequence_parallel=True,
                 device=None, dtype=None, return_hidden_state=False, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model  # for decoder
        self.process_group = process_group
        self.return_hidden_state = return_hidden_state
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs, **kwargs
        )

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None, **kargs):  # state for the repo interface
        """DNA Embedding Model forward pass."""
        hidden_states = self.backbone(input_ids, position_ids=position_ids,
                                      inference_params=inference_params)
        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        return hidden_states, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


# class DNAEmbeddingModelMamba(DNAEmbeddingModel):
#     """Custom DNA Embedding Model that is compatible with open-source Mamba repo."""

#     def __init__(
#             self,
#             config: MambaConfig,
#             initializer_cfg=None,
#             conjoin_train=False,
#             conjoin_test=False,
#             device=None,
#             dtype=None,
#     ):
#         super(DNAEmbeddingModel, self).__init__()  # nn.Module.__init__()
#         self.config = config
#         d_model = config.d_model
#         self.d_model = d_model  # for decoder
#         n_layer = config.n_layer
#         vocab_size = config.vocab_size
#         ssm_cfg = config.ssm_cfg
#         rms_norm = config.rms_norm
#         residual_in_fp32 = config.residual_in_fp32
#         fused_add_norm = config.fused_add_norm
#         pad_vocab_size_multiple = config.pad_vocab_size_multiple
#         factory_kwargs = {"device": device, "dtype": dtype}

#         if vocab_size % pad_vocab_size_multiple != 0:
#             vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
#         self.backbone = MixerModel(
#             d_model=d_model,
#             n_layer=n_layer,
#             vocab_size=vocab_size,
#             ssm_cfg=ssm_cfg,
#             rms_norm=rms_norm,
#             initializer_cfg=initializer_cfg,
#             fused_add_norm=fused_add_norm,
#             residual_in_fp32=residual_in_fp32,
#             **factory_kwargs,
#         )
#         # Initialize weights and apply final processing
#         self.apply(
#             partial(
#                 _init_weights_mamba,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )

#         self.conjoin_train = conjoin_train
#         self.conjoin_test = conjoin_test

#     def forward(self, input_ids, position_ids=None, inference_params=None, state=None, **kargs):  # state for the repo interface
#         """Mamba backbone-specific forward pass that does not use `position_ids`."""
#         hidden_states = self.backbone(input_ids, inference_params=inference_params)
#         # we only need the last hidden state for embeddings (decoder head will predict classification task)
#         return hidden_states, None



class DNAEmbeddingModelCustom(nn.Module, GenerationMixin):
    """DNA Embedding Model for Custom model from Hugging Face .

    Same as ConvLMHeadModel (in long_conv_lm.py), except no decoder head, we just pass back the hidden states for
    downstream tasks.
    """

    def __init__(self, 
                config,
                device=None,
                dtype=None,
                **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.config = config
        self.d_model = config.d_model  # for decoder
        self.pretrained_model_path = config.pretrained_model_path
        self.model_name = config.model_name
        self.lora = config.lora if hasattr(config, 'lora') else False
        self.random_init = config.random_init if hasattr(config, 'random_init') else False
        # Load pretrained model from Hugging Face
        if self.model_name == 'caduceus':
            ### https://github.com/kuleshov-group/caduceus/blob/main/src/models/sequence/dna_embedding.py
            self.conjoin_test = getattr(config, 'conjoin_test', False)
            self.conjoin_train = getattr(config, 'conjoin_train', False)
            print(f"Conjoin train: {self.conjoin_train}, Conjoin test: {self.conjoin_test}")
            # caduceus from hugging face https://huggingface.co/kuleshov-group
            if self.random_init:
                print(f"random init for caduceus")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModelForMaskedLM.from_config(auto_config)
            else:
                print(f"load caduceus from {self.pretrained_model_path}")
                self.backbone = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'dnabert':
            # DNABERT from hugging face https://huggingface.co/zhihan1996
            if self.random_init:
                print(f"random init for dnabert")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load dnabert from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
            # Manually reinitialize all parameters using Kaiming initialization
        elif self.model_name == 'dnabert2':
            # DNABERT2 from hugging face https://huggingface.co/zhihan1996
            # there is a bug for triton version 
            if self.random_init:
                print(f"random init for dnabert2")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load dnabert2 from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'dnaberts':
            # DNABERT2 from hugging face https://huggingface.co/zhihan1996
            # there is a bug for triton version 
            if self.random_init:
                print(f"random init for dnaberts")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load dnaberts from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        
        elif self.model_name == 'genalm':
            # Gena LM from hugging face https://huggingface.co/AIRI-Institute
            if self.random_init:
                print(f"random init for genalm")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load genalm from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'grover':
            if self.random_init:
                print(f"random init for grover")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load grover from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)

        elif self.model_name == 'NT':
            if self.random_init:
                print(f"random init for NT")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load NT from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'NTv2':
            if self.random_init:
                print(f"random init for NTv2")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModelForMaskedLM.from_config(auto_config)
            else:
                print(f"load NTv2 from {self.pretrained_model_path}")
                self.backbone = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'gpn':
            import src.models.GPN.model
            if self.random_init:
                print(f"random init for GPN")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load GPN from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'rnafm':
            from multimolecule import RnaFmModel
            try:
                self.backbone = RnaFmModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
            except Exception as e:
                print(f"Error loading RnaFmModel: {e} since {self.pretrained_model_path} is not support import from local path")
                self.backbone = RnaFmModel.from_pretrained('multimolecule/rnafm', trust_remote_code=True)
        elif self.model_name == 'rnaernie':
            from multimolecule import RnaErnieModel
            try:
                self.backbone = RnaErnieModel.from_pretrained('multimolecule/rnaernie', trust_remote_code=True)
            except Exception as e:
                print(f"Error loading RnaErnieModel: {e} since {self.pretrained_model_path} is not support import from local path")
                self.backbone = RnaErnieModel.from_pretrained('multimolecule/rnaernie', trust_remote_code=True)
        elif self.model_name == 'splicebert':
            from multimolecule import SpliceBertModel
            try:
                self.backbone = SpliceBertModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
            except Exception as e:
                print(f"Error loading SpliceBertModel: {e} since {self.pretrained_model_path} is not support import from local path")
                self.backbone = SpliceBertModel.from_pretrained('multimolecule/splicebert', trust_remote_code=True)
        elif self.model_name == 'mistraldna':
            if self.random_init:
                print(f"random init for mistraldna")
                auto_config = AutoConfig.from_pretrained(self.pretrained_model_path)
                self.backbone = AutoModel.from_config(auto_config)
            else:
                print(f"load mistraldna from {self.pretrained_model_path}")
                self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        elif self.model_name == 'generator':
            self.backbone = AutoModel.from_pretrained(self.pretrained_model_path, trust_remote_code=True)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented yet.")

          
    def forward(self, input_ids, mask=None, state=None, **kargs):  # state for the repo interface
        """DNA Embedding Model forward pass."""
        if self.model_name == 'caduceus':
            # caduceus from hugging face https://huggingface.co/kuleshov-group
            ### code from https://github.com/kuleshov-group/caduceus/blob/main/src/models/sequence/dna_embedding.py
            if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
                hidden_states = self.backbone(input_ids, output_hidden_states=True)['hidden_states'][-1]
                num_chan = hidden_states.shape[-1]
                return torch.stack(
                    [hidden_states[..., :num_chan // 2], torch.flip(hidden_states[..., num_chan // 2:], dims=[1, 2])],
                    dim=-1
                ), None
            if self.conjoin_train or (self.conjoin_test and not self.training):  # For conjoining / post-hoc conjoining
                assert input_ids.ndim == 3, "Input must be 3D tensor, where channels corresponds to forward and rc strands"
                hidden_states = self.backbone(input_ids[..., 0], output_hidden_states=True)['hidden_states'][-1]
                hidden_states_rc =self.backbone(input_ids[..., 1], output_hidden_states=True)['hidden_states'][-1]
                # Stack along channel dimension (dim=-1)
                return torch.stack([hidden_states, hidden_states_rc], dim=-1), None
            hidden_states = self.backbone(input_ids, output_hidden_states=True)['hidden_states'][-1]
        elif self.model_name == 'dnabert':
            # DNABERT from hugging face https://huggingface.co/zhihan1996
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'dnabert2':
            # DNABERT2 from hugging face https://huggingface.co/zhihan1996
            # there is a bug for triton version 
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'dnaberts':
            # DNABERT2 from hugging face https://huggingface.co/zhihan1996
            # there is a bug for triton version 
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
            # print(hidden_states.shape)
        elif self.model_name == 'genalm':
            # Gena LM from hugging face https://huggingface.co/AIRI-Institute
            # checked
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True)['hidden_states'][-1]
        elif self.model_name == 'grover':
            # checked
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'NT':
            #checked
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'NTv2':
            # checked
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)['hidden_states'][-1]
        elif self.model_name == 'gpn':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'rnafm':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'rnaernie':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'splicebert':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)[0]
        elif self.model_name == 'mistraldna':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)['hidden_states'][-1]
        elif self.model_name == 'generator':
            hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask.int(), output_hidden_states=True,)['hidden_states'][-1]
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented yet.")

        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        # print('hidden_states:', hidden_states.shape)
        return hidden_states, None


    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True):
    """

    Modifies state dict loading with custom function.  This is necessary because the head of
    a lm outputs logits for vocab, but we just need the embeddings for downstream tasks.

    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes, then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # consumes prefix from pretrained model, if necessary
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )

    model_new_params_dict = model.state_dict()
    updated_model_state_dict = {}

    # loop through scratch model keys (pretrained may have extra stuff)
    for key in sorted(model_new_params_dict.keys()):

        loaded_params = state_dict.get(key, None)
        if loaded_params is None:
            # This should never happen, it should be there!
            print("Missing key in pretrained model!", key)
            raise Exception

        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif "decoder" in key:
            print("found decoder key / parameter, load from scratch", key)
            used_params = model_new_params_dict[key]
        else:
            print('key: shape MATCH, loading', key)  # load matched weights
            used_params = loaded_params

        # we need to pass back a state dict with the '.model' prefix!!!!!
        key_with_prefix = 'model.' + key
        updated_model_state_dict[key_with_prefix] = used_params

    if freeze_backbone:
        print("freezing model backbone params!")
        # note, decoder not included in backbone
        for name, param in model.named_parameters():
            param.requires_grad = False

    # we have updated the new model state dict with pretrained now
    return updated_model_state_dict

