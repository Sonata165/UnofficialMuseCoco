import os
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import lightning as L
from torch import optim
import transformers
from transformers import MuseCocoLMHeadModel, MuseCocoConfig
from hf_musecoco.modeling_musecoco import MuseCocoPhraseClsModelSingleHead, MuseCocoPhraseClsModelDoubleHead, MuseCocoPhraseClsModel8Head
from src_hf.utils import jpath, read_json
from utils_midi import remi_utils
from m2m.evaluate import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from utils import jpath, read_yaml, get_latest_checkpoint
from peft import get_peft_model, LoraConfig, TaskType


def load_lit_model(config):
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    pt_ckpt = config['pt_ckpt']
    l_model = LitMuseCoco.load_from_checkpoint(ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=None, infer=True) # TODO: change to model_fp
    return l_model


# define the LightningModule
class LitMuseCoco(L.LightningModule):
    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        if infer is False:
            config_fp = jpath(pt_ckpt, 'config.json')
            config = read_json(config_fp)
            config = MuseCocoConfig.from_pretrained(config_fp)
            self.model = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)
        else:
            config_fp = jpath(pt_ckpt, 'config.json')
            config = MuseCocoConfig.from_pretrained(config_fp)
            self.model = MuseCocoLMHeadModel(config=config)

        

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Tokenize the batch
        tgt_seqs = self.tk(
            batch, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Right shift to get the input sequence
        inp_seqs = torch.zeros_like(tgt_seqs, device=tgt_seqs.device)
        inp_seqs[:, 0] = self.tk.bos_token_id
        inp_seqs[:, 1:] = tgt_seqs[:, :-1]

        # Set the part before <sep> in target to -100, to get accurate target sequence loss
        sep_id = self.tk.convert_tokens_to_ids('<sep>')
        sep_pos = find_token(tgt_seqs, sep_id)
        for i in range(tgt_seqs.shape[0]):
            tgt_seqs[i, :sep_pos[i]+1] = -100

        out = self.model(inp_seqs).logits.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(out, tgt_seqs)

        sch = self.lr_schedulers()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Tokenize the batch
        tgt_seqs = self.tk(
            batch, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Right shift to get the input sequence
        inp_seqs = torch.zeros_like(tgt_seqs, device=tgt_seqs.device)
        inp_seqs[:, 0] = self.tk.bos_token_id
        inp_seqs[:, 1:] = tgt_seqs[:, :-1]

        # Set the part before <sep> in target to -100, to get accurate target sequence loss
        tgt_seq_for_loss = tgt_seqs.clone()
        sep_id = self.tk.convert_tokens_to_ids('<sep>')
        sep_pos = find_token(tgt_seqs, sep_id)
        for i in range(tgt_seqs.shape[0]):
            tgt_seq_for_loss[i, :sep_pos[i]+1] = -100

        out = self.model(inp_seqs).logits.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(out, tgt_seq_for_loss)

        # Generation in validation
        metric = Metric()
        samples_to_gen = self.config['n_gen_val_per_batch'] if 'n_gen_val_per_batch' in self.config else 1
        cnt = 0
        if batch_idx % 5 == 0:
            for _ in range(samples_to_gen):
                generate_kwargs = {
                    # 'min_length': 500,
                    'max_length': 800,
                    'use_cache': False, # There is unsolved bugs in KV cache
                    'do_sample': False, # User searching method
                    'num_beams': 5,
                    'bad_words_ids': [[self.tk.pad_token_id]],
                }
                inp_seq_gen = inp_seqs[cnt:cnt+1, :]
                
                # Remove pad tokens in the input in the end of sequence
                sep_idx = (inp_seq_gen == self.tk.convert_tokens_to_ids('<sep>')).nonzero(as_tuple=True)[1][0]
                inp_seq_gen = inp_seq_gen[:, :sep_idx+1] # <PITCH> <INS> <HIST> <sep>

                # ''' Random instrument infer '''
                # # Replace the instrument tokens with a random set of instruments
                # new_insts = remi_utils.random_get_insts_list(remove_drum=self.config.get('remove_drum', False))
                # if self.config.get('texture_control') is True:
                #     t = []
                #     for inst in new_insts:
                #         t.append(inst)
                #         t.append('txt-0')
                # inp_seq = self.tk.convert_ids_to_tokens(inp_seq_gen[0])
                # inst_start_idx = inp_seq.index('INS') + 1
                # inst_end_idx = inp_seq.index('HIST') if 'HIST' in inp_seq else len(inp_seq)-1
                # new_inp_seq = inp_seq[:inst_start_idx] + new_insts + inp_seq[inst_end_idx:]
                # inp_seq_gen = torch.tensor([self.tk.convert_tokens_to_ids(new_inp_seq)], device=tgt_seqs.device)

                out = self.model.generate(
                    inp_seq_gen,
                    pad_token_id=self.tk.pad_token_id,
                    **generate_kwargs
                )

                # Convert to text
                out_str = self.tk.decode(out[0], skip_special_tokens=True).split('<sep>')[1]
                tgt_str = self.tk.decode(tgt_seqs[cnt, :], skip_special_tokens=True).split('<sep>')[1]
                out_seq = out_str.split(' ')
                tgt_seq = tgt_str.split(' ')

                # Calculate scores
                # inst_iou = metric.calculate_inst_iou_from_condition(out_seq=out_seq, condition=new_inp_seq)
                inst_iou = metric.calculate_inst_iou_from_condition(out_seq=out_seq, condition=self.tk.decode(inp_seq_gen[0]))
                metric.update('inst_iou', inst_iou)

                melody_sor = metric.calculate_melody_sor(out_seq, tgt_seq)
                metric.update('melody_sor', melody_sor)

                chroma_iou = metric.calculate_chroma_iou(out_seq, tgt_seq)
                metric.update('chroma_iou', chroma_iou)

                cnt += 1
        
        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = tgt_seqs.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Linear scheduler
        # max_steps = self.num_training_steps()
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config['warmup_steps'],
        #     num_training_steps=max_steps,
        # )
        # ret = {"optimizer": optimizer, "lr_scheduler": scheduler},

        # Annealing
        scheduler = ReduceLROnPlateauPatch(
            optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            verbose=True
        )

        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])




class LitMuseCocoLoRA(L.LightningModule):
    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        if infer is False:
            config_fp = jpath(pt_ckpt, 'config.json')
            config = read_json(config_fp)
            config = MuseCocoConfig.from_pretrained(config_fp)
            self.model = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)
        else:
            config_fp = jpath(pt_ckpt, 'config.json')
            config = MuseCocoConfig.from_pretrained(config_fp)
            self.model = MuseCocoLMHeadModel(config=config)

        peft_config = LoraConfig(
            # task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=['v_proj'], # try 3 settings: v, qv, qkv
        )

        # Initialize musecoco
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()
        self.model = model
        
        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Tokenize the batch
        tgt_seqs = self.tk(
            batch, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Right shift to get the input sequence
        inp_seqs = torch.zeros_like(tgt_seqs, device=tgt_seqs.device)
        inp_seqs[:, 0] = self.tk.bos_token_id
        inp_seqs[:, 1:] = tgt_seqs[:, :-1]

        out = self.model(inp_seqs).logits.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(out, tgt_seqs, ignore_index=self.tk.pad_token_id)

        sch = self.lr_schedulers()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Tokenize the batch
        tgt_seqs = self.tk(
            batch, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Right shift to get the input sequence
        inp_seqs = torch.zeros_like(tgt_seqs, device=tgt_seqs.device)
        inp_seqs[:, 0] = self.tk.bos_token_id
        inp_seqs[:, 1:] = tgt_seqs[:, :-1]

        out = self.model(inp_seqs).logits.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(out, tgt_seqs, ignore_index=self.tk.pad_token_id)

        # # Generation in validation
        # metric = Metric()
        # samples_to_gen = self.config['n_gen_val_per_batch'] if 'n_gen_val_per_batch' in self.config else 1
        # cnt = 0
        # for _ in range(samples_to_gen):
        #     generate_kwargs = {
        #         # 'min_length': 500,
        #         'max_length': 800,
        #         'use_cache': False, # There is unsolved bugs in KV cache
        #         'do_sample': False, # User searching method
        #         'num_beams': 5,
        #         'bad_words_ids': [[self.tk.pad_token_id]],
        #     }
        #     inp_seq_gen = inp_seqs[cnt:cnt+1, :]
            
        #     # Remove pad tokens in the input in the end of sequence
        #     sep_idx = (inp_seq_gen == self.tk.convert_tokens_to_ids('<sep>')).nonzero(as_tuple=True)[1][0]
        #     inp_seq_gen = inp_seq_gen[:, :sep_idx+1] # <PITCH> <INS> <HIST> <sep>

        #     # Replace the instrument tokens with a random set of instruments
        #     new_insts = remi_utils.random_get_insts_list(remove_drum=self.config.get('remove_drum', False))
        #     if self.config.get('texture_control') is True:
        #         t = []
        #         for inst in new_insts:
        #             t.append(inst)
        #             t.append('txt-0')
        #     inp_seq = self.tk.convert_ids_to_tokens(inp_seq_gen[0])
        #     inst_start_idx = inp_seq.index('INS') + 1
        #     inst_end_idx = inp_seq.index('HIST') if 'HIST' in inp_seq else len(inp_seq)-1
        #     new_inp_seq = inp_seq[:inst_start_idx] + new_insts + inp_seq[inst_end_idx:]
        #     inp_seq_gen = torch.tensor([self.tk.convert_tokens_to_ids(new_inp_seq)], device=tgt_seqs.device)
        #     # new_inp_str = ' '.join(new_inp_seq)
        #     # inp_seq_gen = self.tk([new_inp_str], return_tensors='pt')['input_ids'].cuda()

        #     out = self.model.generate(
        #         inp_seq_gen,
        #         pad_token_id=self.tk.pad_token_id,
        #         **generate_kwargs
        #     )

        #     # Convert to text
        #     out_str = self.tk.decode(out[0], skip_special_tokens=True).split('<sep>')[1]
        #     tgt_str = self.tk.decode(tgt_seqs[cnt, :], skip_special_tokens=True).split('<sep>')[1]
        #     out_seq = out_str.split(' ')
        #     tgt_seq = tgt_str.split(' ')

        #     # Calculate scores
        #     inst_iou = metric.calculate_inst_iou_from_condition(out_seq=out_seq, condition=new_inp_seq)
        #     metric.update('inst_iou', inst_iou)

        #     melody_sor = metric.calculate_melody_sor(out_seq, tgt_seq)
        #     metric.update('melody_sor', melody_sor)

        #     chroma_iou = metric.calculate_chroma_iou(out_seq, tgt_seq)
        #     metric.update('chroma_iou', chroma_iou)

        #     cnt += 1
        
        # scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = tgt_seqs.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        # for k, v in scores.items():
        #     self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # Linear scheduler
        max_steps = self.num_training_steps()
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=max_steps,
        )

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)


class LitMuseCocoInstPred(L.LightningModule):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        n_labels = 35 # 35 types of instruments
        self.model = MuseCocoPhraseClsModelSingleHead(pt_ckpt, n_labels=n_labels, random_init=hparams['random_init'])
        # model = MuseCocoPhraseClsModelSingleHead(pt_ckpt, n_labels=n_labels, random_init=hparams['random_init'])
        # model = torch.compile(model)

        # self.model = model

        if 'freeze_musecoco' in hparams and hparams['freeze_musecoco'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels, ignore_index=self.tk.pad_token_id)

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels, ignore_index=self.tk.pad_token_id)

        ''' Calculate metrics '''
        metric = Metric()
        pred = torch.argmax(out, dim=1)

        # Compute the accuracy
        acc = (pred == labels).float().mean().item()
        metric.update('acc', acc)

        # Compute the top-3 accuracy
        top3 = torch.topk(out, 3, dim=1).indices
        correct = top3.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc = correct.mean().item()
        metric.update('top3_acc', top3_acc)

        # Compute the top-5 accuracy
        top5 = torch.topk(out, 5, dim=1).indices
        correct = top5.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-5 predictions
        top5_acc = correct.mean().item()
        metric.update('top5_acc', top5_acc)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Different learning rate for different parts of the model
        # optimizer = optim.AdamW([
        #     {'params': self.model.cls_head.parameters(), 'lr': float(self.config['lr']), 'weight_decay':self.config['weight_decay']},
        #     {'params': self.model.decoder.parameters(), 'lr': float(self.config['lr_musecoco']), 'weight_decay':self.config['weight_decay']},
        # ])

        # # Linear scheduler
        # max_steps = self.num_training_steps()
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config['warmup_steps'],
        #     num_training_steps=max_steps,
        # )

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        # Annealing
        scheduler = ReduceLROnPlateauPatch(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }

        # ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)


class LitMuseCocoChordPred(L.LightningModule):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        n_labels = 35 # 35 types of instruments
        # self.model = MuseCocoPhraseClsModelDoubleHead(pt_ckpt, random_init=hparams['random_init'])
        self.model = MuseCocoPhraseClsModel8Head(pt_ckpt, random_init=hparams['random_init'])

        if 'freeze_musecoco' in hparams and hparams['freeze_musecoco'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        root_labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        type_labels = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        root_out, type_out = self.model(batch_tokenized) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1)
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, root_labels)
        loss_type = nn.functional.cross_entropy(type_out, type_labels)
        loss = (loss_root + loss_type) / 2

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels_root = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        labels_type = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        root_out, type_out = self.model(batch_tokenized) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1) # [bs, n_labels, n_pos]
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, labels_root)
        loss_type = nn.functional.cross_entropy(type_out, labels_type)
        loss = (loss_root + loss_type) / 2

        ''' Calculate metrics '''
        # Flatten the out and tgt
        root_out = root_out.permute(0, 2, 1).reshape(-1, root_out.shape[1])
        type_out = type_out.permute(0, 2, 1).reshape(-1, type_out.shape[1])
        labels_root = labels_root.reshape(-1)
        labels_type = labels_type.reshape(-1)
        
        metric = Metric()
        pred_root = torch.argmax(root_out, dim=1)
        pred_type = torch.argmax(type_out, dim=1)

        # Compute the accuracy
        acc_root = (pred_root == labels_root).float().mean().item()
        acc_type = (pred_type == labels_type).float().mean().item()
        metric.update('acc_root', acc_root)
        metric.update('acc_type', acc_type)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Different learning rate for different parts of the model
        # optimizer = optim.AdamW([
        #     {'params': self.model.cls_head.parameters(), 'lr': float(self.config['lr']), 'weight_decay':self.config['weight_decay']},
        #     {'params': self.model.decoder.parameters(), 'lr': float(self.config['lr_musecoco']), 'weight_decay':self.config['weight_decay']},
        # ])

        # # Linear scheduler
        # max_steps = self.num_training_steps()
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config['warmup_steps'],
        #     num_training_steps=max_steps,
        # )

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        # Annealing
        scheduler = ReduceLROnPlateauPatch(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.config['lr_anneal_patience'],
            verbose=True
        )

        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }

        # ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)

class LitMuseCocoChordPredLoRA(LitMuseCocoChordPred):
    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__(pt_ckpt, tokenizer, hparams)

        peft_config = LoraConfig(
            inference_mode=False, 
            r=4, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=['v_proj'], # try 3 settings: v, qv, qkv
        )

        # Initialize musecoco
        n_labels = 35 # 35 types of instruments
        model = MuseCocoPhraseClsModelSingleHead(pt_ckpt, n_labels=n_labels)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        self.model = model

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        root_labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        type_labels = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        root_out = self.model(batch_tokenized) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, root_labels)
        loss = loss_root

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels_root = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        labels_type = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        root_out, type_out = self.model(batch_tokenized) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1) # [bs, n_labels, n_pos]
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, labels_root)
        loss_type = nn.functional.cross_entropy(type_out, labels_type)
        loss = (loss_root + loss_type) / 2

        ''' Calculate metrics '''
        # Flatten the out and tgt
        root_out = root_out.permute(0, 2, 1).reshape(-1, root_out.shape[1])
        type_out = type_out.permute(0, 2, 1).reshape(-1, type_out.shape[1])
        labels_root = labels_root.reshape(-1)
        labels_type = labels_type.reshape(-1)
        
        metric = Metric()
        pred_root = torch.argmax(root_out, dim=1)
        pred_type = torch.argmax(type_out, dim=1)

        # Compute the accuracy
        acc_root = (pred_root == labels_root).float().mean().item()
        acc_type = (pred_type == labels_type).float().mean().item()
        metric.update('acc_root', acc_root)
        metric.update('acc_type', acc_type)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    
class LitMuseCocoInstPredLoRA(L.LightningModule):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()

        peft_config = LoraConfig(
            # task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=4, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=['v_proj'], # try 3 settings: v, qv, qkv
        )

        # Initialize musecoco
        n_labels = 35 # 35 types of instruments
        model = MuseCocoPhraseClsModelSingleHead(pt_ckpt, n_labels=n_labels)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        self.model = model

        # if hparams['freeze_musecoco'] is True:
        #     # Set self.model.decoder require_grad to False
        #     for param in self.model.decoder.parameters():
        #         param.requires_grad = False
        #     for param in self.model.decoder.layers[-1].parameters():
        #         param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels, ignore_index=self.tk.pad_token_id)

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels, ignore_index=self.tk.pad_token_id)

        ''' Calculate metrics '''
        metric = Metric()
        pred = torch.argmax(out, dim=1)

        # Compute the accuracy
        acc = (pred == labels).float().mean().item()
        metric.update('acc', acc)

        # Compute the top-3 accuracy
        top3 = torch.topk(out, 3, dim=1).indices
        correct = top3.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc = correct.mean().item()
        metric.update('top3_acc', top3_acc)

        # Compute the top-5 accuracy
        top5 = torch.topk(out, 5, dim=1).indices
        correct = top5.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-5 predictions
        top5_acc = correct.mean().item()
        metric.update('top5_acc', top5_acc)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Different learning rate for different parts of the model
        # optimizer = optim.AdamW([
        #     {'params': self.model.cls_head.parameters(), 'lr': float(self.config['lr']), 'weight_decay':self.config['weight_decay']},
        #     {'params': self.model.decoder.parameters(), 'lr': float(self.config['lr_musecoco']), 'weight_decay':self.config['weight_decay']},
        # ])

        # # Linear scheduler
        # max_steps = self.num_training_steps()
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config['warmup_steps'],
        #     num_training_steps=max_steps,
        # )

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        # Annealing
        scheduler = ReduceLROnPlateauPatch(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }

        # ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)

class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

def find_token(input_ids, target_id=1):
    '''
    Find a specific token in the input_ids tensor.
    If the token is not found, return the last token's position.

    input_ids: [bs, seq_len], a integer tensor
    '''
    bs, seq_len = input_ids.shape

    # Initialize the positions with the last index
    s_positions = torch.full((bs,), seq_len-1, dtype=torch.long)

    # Iterate over each sequence in the batch
    for i in range(bs):
        # Find the index of the first occurrence of the target_id
        target_indices = (input_ids[i] == target_id).nonzero(as_tuple=True)[0]
        if len(target_indices) > 0:
            s_positions[i] = target_indices[0]
    
    return s_positions