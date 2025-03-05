import os
import sys
import torch

from transformers.models.musecoco.utils import read_json, jpath, save_json
from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint_from_arg_dict

def main():
    fairseq_ckpt_fp = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt'
    dict_fp = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/dicts/large/dict.txt'
    args_fp = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_args.json'

    # Modify the args
    args = read_json(args_fp)
    args['model_overrides'] = {'attention_type': 'CausalLinearAttention'}
    # args['model_overrides'] = {'attention_type': 'CausalLinearAttentionPy'}
    # args['attention_type'] = 'CausalLinearAttention'
    model_fs, ori_dict = load_fairseq_checkpoint_from_arg_dict(fairseq_ckpt_fp, dict_fp, args)
    model_fs.eval() # cpp: -5.1184, py: -5.1184
    inp = torch.tensor([[1,2,3]])
    out = model_fs(inp, sep_pos=None)

    a = 1

if __name__ == '__main__':
    main()