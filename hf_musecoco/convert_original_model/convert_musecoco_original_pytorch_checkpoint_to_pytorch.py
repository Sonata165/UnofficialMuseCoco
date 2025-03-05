# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert MUSECOCO checkpoint from original Fairseq checkpoint."""

import argparse
import os
import json
from pathlib import Path

import fairseq
import torch
from packaging import version
from torch import nn

from transformers import MuseCocoTokenizer, MuseCocoLMHeadModel
from transformers.models.musecoco.utils import read_json, jpath, save_json, create_dir_if_not_exist

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
    MuseCocoConfig,
)
from transformers.utils import logging

FAIRSEQ_MODELS = [
    "bart.large",
    "bart.large.mnli",
    "bart.large.cnn",
    "bart_xsum/model.pt",
]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "s-9 o-0 t-26 i-52 p-77 d-24 v-20 p-62 d-24 v-20 o-12 t-26 i-52 p-64 d-12 v-20 o-36 t-26 i-52 p-69 d-12 v-20 p-65 d-12 v-20 b-1"

large_rename_keys = [
    (
        "decoder.output_projection.weight",
        "lm_head.weight",
    ),
]

def main():
    pass
    convert_1b_checkpoint()

def procedures():
    convert_200m_checkpoint()

def convert_200m_checkpoint():
    '''
    Convert the 200M model to hugging face format
    '''
    convert_musecoco_checkpoint(
        fairseq_ckpt_fp='/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt',
        pytorch_dump_folder_path='/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_hf_pt',
        dict_fp='/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/dicts/large/dict.txt',
        args_fp='/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_args.json',
        hf_checkpoint_name='musecoco-large', # Not used for now
    )

def convert_1b_checkpoint():
    '''
    Convert the 200M model to hugging face format
    '''
    fairseq_res_dir = '/data2/longshen/Checkpoints/musecoco/fairseq'
    fs_ckpt_fp = jpath(fairseq_res_dir, 'model_1b.pt')
    dict_fp = jpath(fairseq_res_dir, 'dict_1b/dict.txt')
    args_fp = jpath(fairseq_res_dir, 'task_args_1b.json')

    huggingface_res_dir = '/data2/longshen/Checkpoints/musecoco/transformers/1b'
    tokenizer_dir = jpath(huggingface_res_dir, 'tokenizer')
    # hf_model_dir = jpath(huggingface_res_dir, 'model')

    convert_musecoco_checkpoint(
        fairseq_ckpt_fp=fs_ckpt_fp,
        pytorch_dump_folder_path=huggingface_res_dir,
        hf_tokenizer_dir=tokenizer_dir,
        dict_fp=dict_fp,
        args_fp=args_fp,
        hf_checkpoint_name='musecoco-1b', # Not used for now
    )


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        # "_float_tensor",
        # "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_musecoco_checkpoint(
    fairseq_ckpt_fp,
    pytorch_dump_folder_path,
    hf_tokenizer_dir,
    dict_fp=None,
    args_fp=None,
    hf_checkpoint_name=None,
):
    """
    Copy/paste/tweak model's weights to our MUSECOCO structure.
    """
    if not os.path.exists(fairseq_ckpt_fp):
        raise Exception("Fairseq checkpoint path not exist.")
    else:
        from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint
        model_fs, ori_dict = load_fairseq_checkpoint(fairseq_ckpt_fp, dict_fp, args_fp)
        # musecoco = load_xsum_checkpoint(fairseq_ckpt_fp)
    print("Fairseq model class: ", type(model_fs))
    # musecoco.model.upgrade_state_dict(musecoco.model.state_dict())

    # Initialize huggingface musecoco config
    if hf_checkpoint_name is None:
        hf_checkpoint_name = fairseq_ckpt_fp.replace(".", "-")
    config = MuseCocoConfig(
        vocab_size=1253,
        n_positions=8192,
        n_embd=2048,
        scale_emb=True,
        ffn_dim=8192,
        layernorm_emb=False,
        norm_before=True,
        cross_self_attention=False,
        n_layer=24,
        n_head=24,
        n_inner=None,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.0,
        decoder_layerdrop=0.0,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=2,
        eos_token_id=2,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    # config = BartConfig.from_pretrained(hf_checkpoint_name)

    # Initialize a musecoco huggingface tokenizer
    tk = MuseCocoTokenizer.from_pretrained(hf_tokenizer_dir)

    # Ensure the tokenizer match with each other
    tokens_hf = tk(SAMPLE_TEXT, return_tensors="pt")['input_ids']
    
    # Obtain statedict from fairseq model, rename content if necessary
    state_dict = model_fs.state_dict()
    remove_ignore_keys_(state_dict)
    # state_dict["model.wtd.weight"] = state_dict[
    #     "decoder.embed_tokens.weight"
    # ]
    for src, dest in large_rename_keys:
        rename_key(state_dict, src, dest)
    temp_dir = jpath(pytorch_dump_folder_path, 'temp')
    create_dir_if_not_exist(temp_dir)
    state_dict_keys_fp = jpath(temp_dir, 'state_dict.json')
    save_json(list(state_dict), state_dict_keys_fp)

    # Initialize Hugging Face model, load state dict
    model_hf = MuseCocoLMHeadModel(config).eval()
    model_hf.load_state_dict(state_dict, strict=True) # BUG: embed_tokens.weight has different values in the model after loading
    # for key in state_dict.keys():
    #     try:
    #         model_hf.load_state_dict({key: state_dict[key]}, strict=False)
    #         print(f"Loaded {key} successfully.")
    #     except Exception as e:
    #         print(f"Failed to load {key}: {str(e)}")
    #     print(model_hf.decoder.embed_tokens.weight[:10])
    #     print(model_hf.lm_head.weight[:10])

    # Make a single forward step
    model_fs.eval()
    model_hf.eval()
    output_fs = model_fs(tokens_hf, sep_pos=None)[0]
    output_hf = model_hf(tokens_hf).logits  # logits

    # Check results
    if output_fs.shape != output_hf.shape:
        raise ValueError(
            f"`fairseq_output` shape and `new_model_output` shape are different: {output_fs.shape=}, {output_hf.shape}"
        )
    if (output_fs != output_hf).any().item():
        c = (output_hf - output_fs).abs().max()
        print(c) # (7.6294e-06)
        if c >= 1e-3:
            raise ValueError(
                "Some values in `fairseq_output` are different from `new_model_outputs`"
            )
        else:
            print('This is a successful porting, although minor differences were found')

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)

    model_save_dir = jpath(pytorch_dump_folder_path, 'model')
    model_hf.save_pretrained(model_save_dir)


@torch.no_grad()
def initialize_hf_model(
    fairseq_ckpt_fp="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt",
    pytorch_dump_folder_path="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_hf.pt",
    dict_fp="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/dicts/large/dict.txt",
    args_fp="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_args.json",
    hf_checkpoint_name=None,
):
    """
    Initialize a hugging face model of musecoco, from
    - Default configs
    - The original checkpoint, dict
    """
    if not os.path.exists(fairseq_ckpt_fp):
        raise Exception("Fairseq checkpoint path not exist.")
    else:
        from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint
        model_fs, ori_dict = load_fairseq_checkpoint(fairseq_ckpt_fp, dict_fp, args_fp)

    # Initialize huggingface musecoco config
    if hf_checkpoint_name is None:
        hf_checkpoint_name = fairseq_ckpt_fp.replace(".", "-")
    config = MuseCocoConfig()

    # Initialize a musecoco huggingface tokenizer
    tk = MuseCocoTokenizer.from_pretrained('/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/tokenizers/ori_large')

    # Obtain statedict from fairseq model, rename content if necessary
    state_dict = model_fs.state_dict()
    remove_ignore_keys_(state_dict)
    for src, dest in large_rename_keys:
        rename_key(state_dict, src, dest)
    save_json(list(state_dict), jpath('/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/convert_original_model', 'state_dict.json'))

    # Initialize Hugging Face model, load state dict
    model_hf = MuseCocoLMHeadModel(config).eval()
    model_hf.load_state_dict(state_dict)

    return model_hf, tk


if __name__ == "__main__":
    main()
