import os
import sys

import torch
from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint
        

def main():
    model, ori_dict = load_fairseq_checkpoint()
    model.eval()
    inp = torch.tensor([[1,2,3,4,5]])
    out = model(inp, sep_pos=None)
    a = 1

if __name__ == '__main__':
    main()