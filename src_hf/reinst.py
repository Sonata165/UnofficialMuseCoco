'''
Do full-song reinstrumentation with HF model
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import torch
from tqdm import tqdm
from transformers import MuseCocoTokenizer
from utils_common.utils import get_latest_checkpoint
from src_hf.lightning_model import *
from torch import utils
from utils_midi.utils_midi import RemiTokenizer, RemiUtil
from utils_midi import remi_utils
from utils import jpath, read_yaml, create_dir_if_not_exist
from src_hf.lightning_dataset import *

torch.backends.cuda.matmul.allow_tf32 = True

def main():
    if len(sys.argv) == 2:
        config_fp = sys.argv[1]
        assert os.path.exists(config_fp), 'Config file not found'
        config = read_yaml(config_fp)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        config_fp = './hparams/arrangement/reorder_vc_rphist.yaml'
        config = read_yaml(config_fp)
    
    # Prepare paths
    midi_fp = config['midi_fp']
    exp_name = 'reinst_{}'.format(config['reinst_exp_name'])

    # Load the model and tokenizer
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    pt_ckpt = config['pt_ckpt']
    model_cls = eval(config['lit_model_class'])
    l_model = model_cls.load_from_checkpoint(ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=None, infer=True) # TODO: change to model_fp
    model = l_model.model
    model.eval() # class: MuseCocoLMHeadModel
    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    # Generate remi seqs, save to song_seq_remi.txt
    remi_tk = RemiTokenizer()
    midi_dir_name = os.path.dirname(midi_fp)
    remi_fp = jpath(midi_dir_name, 'song_remi.txt')
    song_remi_seq, pitch_shift, is_major = remi_tk.midi_to_remi(
            midi_fp, return_pitch_shift=True, return_key=True, normalize_pitch=True
        )
    RemiUtil.save_remi_seq(song_remi_seq, remi_fp)
    # detokenized_fp = jpath(midi_dir_name, "detokenized.mid")
    # remi_tk.remi_to_midi(song_remi_seq, detokenized_fp)
    seg_remi_seqs = remi_utils.song_remi_split_to_segments_2bar_hop1(song_remi_seq)
    seg_remi_seqs_fp = jpath(midi_dir_name, 'seg_remi.txt')
    t = [' '.join(i) + '\n' for i in seg_remi_seqs]
    with open(seg_remi_seqs_fp, 'w') as f:
        f.writelines(t)

    # Construct dataset, 
    bs = 1
    split = 'test'
    dataset_class = config['dataset_class'] if 'dataset_class' in config else 'ArrangerDataset'
    dataset_class = eval(dataset_class)
    test_dataset = dataset_class(
        data_fp=seg_remi_seqs_fp, 
        split=split, 
        config=config,
    )
    test_loader = utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=bs,
    )

    t = tk.pad_token

    # Config inference
    generate_kwargs = {
        # 'min_length': 500,
        'max_length': 800,
        'use_cache': False, # There is unsolved bugs in KV cache
        'bad_words_ids': [[tk.pad_token_id], [tk.convert_tokens_to_ids('INS')], [tk.convert_tokens_to_ids('<sep>')]],

        'no_repeat_ngram_size': 6,
        # 'num_beams': 5,
        'do_sample': True, # User searching method
        'top_k': 50,
        'top_p': 1.0,
        # 'temperature': 0.6,
    }

    if 'replace_inst' in config and config['replace_inst'] is not False:
        print('Inst setting: ', config['replace_inst'])

    # Iterate over dataset
    # NOTE: use the previous bar out as hist of next bar
    inputs = []
    test_out = []
    hist_remi = None
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

            # Get the batch, replace the hist
            if config['hist_autoregressive'] and config['with_hist']:
                if hist_remi != None:
                    if 'replace_inst' in config and config['replace_inst'] is not False:
                        inst_spec = config['replace_inst']
                    else:
                        condition_seq = batch[0].strip().split(' ')[:-1]
                        inst_start_idx = condition_seq.index('INS') + 1
                        inst_end_idx = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
                        inst_spec = condition_seq[inst_start_idx:inst_end_idx]
                    tgt_insts = remi_utils.from_remi_get_insts(inst_spec)
                    hist_seq = remi_utils.from_remi_hist_refine_history(
                        hist_remi, 
                        tgt_insts=tgt_insts,
                        hist_type=config['with_hist'], 
                        voice_control=config['voice_control'],
                        reorder_tgt=config['reorder_tgt'],
                        hist_is_reordered=config['reorder_tgt']
                    )
                    new_seq = remi_utils.in_remi_replace_hist(batch[0].strip().split(' '), hist_seq)
                    batch = [' '.join(new_seq)]

            # Replace the instrument
            if 'replace_inst' in config and config['replace_inst'] is not False:
                inst_spec = config['replace_inst']
                condition_seq = batch[0].strip().split(' ')[:-1] # Remove <sep> when replacing inst
                new_seq = remi_utils.in_condition_seq_replace_inst(condition_seq, inst_spec) + ['<sep>']  # Add <sep> back
                batch = [' '.join(new_seq)]

            inputs.append(batch[0])

            # Tokenize the batch
            batch_tokenized = tk(
                batch, 
                return_tensors="pt",
                padding=True,
                add_special_tokens=False, # Don't add eos token
            )['input_ids'].cuda()

            # Prepare bos token (</s>, id=2)
            bos = torch.full(
                size=[batch_tokenized.shape[0],1], 
                fill_value=tk.bos_token_id, 
                dtype=torch.long,
                device=batch_tokenized.device,
            )
            inp = torch.cat([bos, batch_tokenized], dim=1)

            # Feed to the model
            out = model.generate(
                inp,
                pad_token_id=tk.pad_token_id,
                **generate_kwargs
            )
            out_str = tk.batch_decode(out)[0] # because we do bs=1 here

            # Truncate only useful part, i.e., between <sep> and </s>
            out_seq = out_str.split(' ')

            # if config['reorder_tgt']:
            #     out_seq.insert(-1, 'b-1')
            # out_seq.insert(-1, 'b-1') # for tgt reorganized model

            sep_pos = out_seq.index('<sep>')
            eos_pos = out_seq[1:].index('</s>')+1 # Ignore eos when searching
            out_seq = out_seq[sep_pos+1:eos_pos]
            
            hist_remi = out_seq
            test_out.extend(out_seq)

    # Save result
    save_dir = jpath(out_dir, 'lightning_logs', latest_version_dir)
    token_out_dir = jpath(save_dir, 'txt_in_out')
    create_dir_if_not_exist(token_out_dir)

    # Save input
    input_save_fp = jpath(token_out_dir, 'song_in_{}.txt'.format(exp_name))
    with open(input_save_fp, 'w') as f:
        f.writelines([i+'\n' for i in inputs])

    # Save midi
    out_remi_fp = jpath(token_out_dir, 'song_out_{}.txt'.format(exp_name))
    out_midi_fp = jpath(save_dir, 'song_out_{}.mid'.format(exp_name))
    RemiUtil.save_remi_seq(test_out, out_remi_fp)
    remi_tk.remi_to_midi(test_out, out_midi_fp)


if __name__ == '__main__':
    main()