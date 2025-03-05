import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import torch
from torch.utils.data import Dataset, DataLoader
from utils_midi import remi_utils

class RemiDatasetWithFeatureExtraction(Dataset):
    '''
    The dataset class that read 
    It is a language pair dataset, but the source sequence (conditions) is generated from the target sequence
    '''
    def __init__(self, data_fp, split, with_hist):
        # Read the remi data (one sample one line, one split one file)
        with open(data_fp) as f:
            data = f.readlines()
        data = [l.strip() for l in data] # a list of strings
        self.data = data

        self.split = split
        self.with_hist = with_hist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the conditions from the remi
        remi_str = self.data[index]
        remi_seq = remi_str.split(' ')
        if self.with_hist:
            condition_seq, tgt_remi_seq = remi_utils.obtain_input_tokens_from_remi_seg_for_sss_with_hist(remi_seq)
        else:
            condition_seq, tgt_remi_seq = remi_utils.obtain_input_tokens_from_remi_seg_for_sss_no_hist(remi_seq)

        # TODO: Do the modification if needed, to condition and target sequence

        # Concatenate the sample
        condition_str = ' '.join(condition_seq)
        tgt_remi_str = ' '.join(tgt_remi_seq)

        if self.split != 'test':
            tot_seq = condition_str + ' <sep> ' + tgt_remi_str
        else: # Do not provide target when doing generation for test set
            tot_seq = condition_str + ' <sep>'
        return tot_seq
    