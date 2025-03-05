import os
import sys

import torch
from transformers.models.musecoco.convert_original_model.convert_musecoco_original_pytorch_checkpoint_to_pytorch import initialize_hf_model

def main():
    model, tk = initialize_hf_model()
    model.eval()
    inp = torch.tensor([[1,2,3,4,5]])
    out = model(inp)
    a = 1

if __name__ == '__main__':
    main()