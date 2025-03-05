import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # only use when debugging

from transformers import MuseCocoTokenizer, MuseCocoConfig
from lightning_dataset import *
from lightning_model import LitMuseCoco, LitMuseCocoInstPred, LitMuseCocoInstPredLoRA, LitMuseCocoChordPred, LitMuseCocoChordPredLoRA, LitMuseCocoLoRA
from utils import jpath, read_yaml

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def main():
    if not len(sys.argv) == 2: # Debug
        # os.environ["CUDA_VISIBLE_DEVICES"] = ''
        config_fp = '/home/longshen/work/MuseCoco/musecoco/src_hf/hparams/arrangement/20m_baseline.yaml'
        config = read_yaml(config_fp)
        config['num_workers'] = 0
        config['fast_dev_run'] = 10
        config['train_with'] = 'valid'
    else:
        config_fp = sys.argv[1]
        config = read_yaml(config_fp)
    
    # Init the model
    model_fp = config['pt_ckpt']
    # model = MuseCocoLMHeadModel.from_pretrained(model_fp)
    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    lit_model_class = config['lit_model_class'] if 'lit_model_class' in config else 'LitMuseCoco'
    print(lit_model_class)
    lit_model_class = eval(lit_model_class)
    lit_model = lit_model_class(
        model_fp,
        tokenizer=tk, 
        hparams=config
    )

    # Setup data
    train_loader = get_dataloader(config, config['train_with'] if 'train_with' in config else 'train')
    valid_loader = get_dataloader(config, 'valid')

    # Train the model
    out_dir = jpath(config['result_root'], config['out_dir'])
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        mode="min",
        filename='{epoch:02d}-{valid_loss:.2f}',
        save_top_k=1,
    )
    earlystop_callback = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        mode='min',
    )
    trainer = L.Trainer(
        max_epochs=config['n_epoch'],
        default_root_dir=out_dir, # output and log dir
        callbacks=[checkpoint_callback, earlystop_callback],
        fast_dev_run=config['fast_dev_run'] if 'fast_dev_run' in config else False,
        val_check_interval=config['val_check_interval'] if 'val_check_interval' in config else 1.0, # validate every 0.5 epoch
        # accelerator="cpu",
        # devices=1,
        # precision='bf16',
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader,
    )
    ''' time overhead: 
    Generation model
        fp32: 1.40 it/s 
            + tf32: 1.87 it/s
        fp16-mixed: 2.17 it/s, but nan
        bf16-mixed: 2.17 it/s (use this)
            + tf32: 2.15 it/s
    '''

def get_dataloader(config, split):
    bs = config['bs']
    data_root = config['data_root']
    data_fn = '{}.txt'.format(split)
    data_fp = jpath(data_root, data_fn)

    dataset_class_name = config['dataset_class'] if 'dataset_class' in config else 'ArrangerDataset'
    dataset_class = eval(dataset_class_name)

    dataset = dataset_class(data_fp=data_fp, split=split, config=config)
    if dataset_class_name in ['InstPredDataset', 'ChordPredDataset']:
        dataloader = utils.data.DataLoader(
            dataset=dataset, 
            batch_size=bs,
            num_workers=config['num_workers'] if 'num_workers' in config else 4,
            collate_fn=lambda x: x,
        )
    else:
        dataloader = utils.data.DataLoader(
            dataset=dataset, 
            batch_size=bs,
            num_workers=config['num_workers'] if 'num_workers' in config else 4,
        )
    return dataloader


if __name__ == '__main__':
    main()