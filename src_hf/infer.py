'''
Do the generation with the test set
Comput the performance metrics
'''
import os
import re
import sys
import torch
from tqdm import tqdm
from torch import utils
from utils import jpath, read_yaml, get_latest_checkpoint
from lightning_model import LitMuseCoco
from lightning_dataset import ArrangerDataset
from lightning.pytorch import seed_everything
from transformers import MuseCocoTokenizer
import mlconfig

torch.backends.cuda.matmul.allow_tf32 = True

seed_everything(42, workers=True)


def main():
    if len(sys.argv) == 2:
        config_fp = sys.argv[1]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        config_fp = './hparams/arrangement/ar_hist_reorder.yaml'

    config = mlconfig.load(config_fp)  

    do_infer = True
    do_eval = True

    infer_exp_name = config['infer_exp_name'] if 'infer_exp_name' in config else 'recon'
    if infer_exp_name == 'rand_inst':
        rand_inst_infer = True
    else:
        rand_inst_infer = False

    infer_with = config['infer_inp_fn'] if 'infer_inp_fn' in config else 'test.txt'

    # Load the model and tokenizer
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    pt_ckpt = config['pt_ckpt']
    l_model = LitMuseCoco.load_from_checkpoint(ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=None, infer=True) # TODO: change to model_fp
    model = l_model.model
    model.eval() # class: MuseCocoLMHeadModel
    
    tk_fp = config['tokenizer_fp']
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    # Prepare the test set
    bs = config['bs_test']
    data_root = config['data_root']
    data_fn = infer_with
    data_fp = jpath(data_root, data_fn)
    test_dataset = ArrangerDataset(
        data_fp=data_fp, 
        split='test', 
        config=config,
        rand_inst_infer=rand_inst_infer,
    )
    test_loader = utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=bs,
    )

    # Config inference
    generate_kwargs = {
        # 'min_length': 500,
        'max_length': 800,
        'use_cache': False, # There is unsolved bugs in KV cache
        'do_sample': False, # User searching method
        'num_beams': 5,
        'bad_words_ids': [[tk.pad_token_id]],
        # 'repetition_penalty': 1.5, # Not penalize repetition for now
        # 'early_stopping': True, # Not using heuristics to stop beam search for now
        # 'top_k': 1000, # Only works for sampling algorithm
        # 'no_repeat_ngram_size': 10,
    }

    # Iterate over test set, do the generation, save result
    # History is provided by teacher forcing
    test_out = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

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
            out_str = tk.batch_decode(out)
            test_out.extend(out_str)

    
    # Save the output file
    out_fn = config['infer_out_fn']
    out_fp = jpath(out_dir, 'lightning_logs', latest_version_dir, out_fn)
    test_out = [line + '\n' for line in test_out]
    with open(out_fp, 'w') as f:
        f.writelines(test_out)



if __name__ == '__main__':
    main()