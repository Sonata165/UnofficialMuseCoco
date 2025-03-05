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

from transformers import MuseCocoTokenizer


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

mnli_rename_keys = [
    (
        "model.classification_heads.mnli.dense.weight",
        "classification_head.dense.weight",
    ),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    (
        "model.classification_heads.mnli.out_proj.weight",
        "classification_head.out_proj.weight",
    ),
    (
        "model.classification_heads.mnli.out_proj.bias",
        "classification_head.out_proj.bias",
    ),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
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
    config = MuseCocoConfig()
    # config = BartConfig.from_pretrained(hf_checkpoint_name)

    # Initialize a musecoco huggingface tokenizer
    tk = MuseCocoTokenizer.from_pretrained('/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/tokenizers/ori_large')

    # Ensure the tokenizer match with each other
    tokens_hf = tk(SAMPLE_TEXT, return_tensors="pt")['input_ids']
    
    # if fairseq_ckpt_fp == "bart.large.mnli":
    state_dict = model_fs.state_dict()
    remove_ignore_keys_(state_dict)
    state_dict["model.shared.weight"] = state_dict[
        "model.decoder.embed_tokens.weight"
    ]
    for src, dest in mnli_rename_keys:
        rename_key(state_dict, src, dest)
    model_hf = BartForSequenceClassification(config).eval()
    model_hf.load_state_dict(state_dict)
    output_fs = model_fs.predict("mnli", tokens_hf, return_logits=True)
    output_hf = model_hf(tokens_hf)[0]  # logits

    # else:  # no classification heads to worry about
    #     state_dict = model_fs.model.state_dict()
    #     remove_ignore_keys_(state_dict)
    #     state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    #     fairseq_output = model_fs.extract_features(tokens_fs)
    #     if hf_checkpoint_name == "facebook/bart-large":
    #         model = BartModel(config).eval()
    #         model.load_state_dict(state_dict)
    #         new_model_outputs = model(tokens_fs).model[0]
    #     else:
    #         model = BartForConditionalGeneration(
    #             config
    #         ).eval()  # an existing summarization ckpt
    #         model.model.load_state_dict(state_dict)
    #         if hasattr(model, "lm_head"):
    #             model.lm_head = make_linear_from_emb(model.model.shared)
    #         new_model_outputs = model.model(tokens_fs)[0]

    # Check results
    if output_fs.shape != output_hf.shape:
        raise ValueError(
            f"`fairseq_output` shape and `new_model_output` shape are different: {output_fs.shape=}, {output_hf.shape}"
        )
    if (output_fs != output_hf).any().item():
        raise ValueError(
            "Some values in `fairseq_output` are different from `new_model_outputs`"
        )
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model_hf.save_pretrained(pytorch_dump_folder_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem.",
        default="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        help="Path to the output PyTorch model.",
        default="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_hf.pt",
        required=False,
    )
    parser.add_argument(
        "--dict_fp",
        type=str,
        help="Path to the dictionary file of the original fairseq model.",
        default="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/dicts/large/dict.txt",
        required=False,
    )
    parser.add_argument(
        "--args_fp",
        type=str,
        help="Path to the args file of the original fairseq model.",
        default="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_args.json",
        required=False,
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        help="Which huggingface architecture to use, 'musecoco-large' or 'musecoco-1b'",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    convert_musecoco_checkpoint(
        args.fairseq_path,
        args.pytorch_dump_folder_path,
        args.dict_fp,
        args.args_fp,
        hf_checkpoint_name=args.hf_config,
    )
