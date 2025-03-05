'''
Convert the original tokenizer to Hugging Face MuseCoco tokenizer
Or, more specifically, create a huggingface tokenizer for musecoco that replicate the behavior of 
MuseCoco dict in fairseq.

Tokenization: converting remi tokens to integers
Detokenization: converting integers back to remi tokens
'''
import torch

from transformers.models.musecoco.utils import read_json, jpath
from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint

from transformers import MuseCocoTokenizer
from transformers.models.musecoco.utils import read_json, jpath, save_json, create_dir_if_not_exist

SAMPLE_TEXT = "s-9 o-0 t-26 i-52 p-77 d-24 v-20 p-62 d-24 v-20 o-12 t-26 i-52 p-64 d-12 v-20 o-36 t-26 i-52 p-69 d-12 v-20 p-65 d-12 v-20 b-1"

def main():
    procedures_200m()
    # procedures_1b()
    

def procedures_200m():
    '''
    Convert the 200M model's vocab file to corresponding Hugging Face tokenizer
    '''
    fairseq_resource_root = '/data2/longshen/Checkpoints/musecoco/fairseq'
    fairseq_ckpt_fp = jpath(fairseq_resource_root, 'model_200m.pt')
    # dict_fp = '/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/convert_original_model/large/3-dict-3241.txt'
    dict_fp = '/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/dicts/dict_large_ft/dict.txt'
    args_fp = jpath(fairseq_resource_root, 'task_args_200m.json')
    pytorch_dump_folder_path = '/data2/longshen/Checkpoints/musecoco/temp'
    create_dir_if_not_exist(pytorch_dump_folder_path)

    dump_vocab_file_from_fairseq(
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
        pytorch_dump_folder_path=pytorch_dump_folder_path
    )

    vocab_fp = jpath(pytorch_dump_folder_path, "vocab.json")
    tk_save_dir = '/data2/longshen/Checkpoints/musecoco/transformers/200m/tokenizer_ft'
    
    validate_tokenizer(
        vocab_fp=vocab_fp, 
        tk_save_dir=tk_save_dir, 
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )

    validate_tokenizer_loading_local(
        tk_dir=tk_save_dir,
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )



def procedures_1b():
    '''
    Convert the 1B model's vocab file to corresponding Hugging Face tokenizer
    '''
    fairseq_resource_root = '/data2/longshen/Checkpoints/musecoco/fairseq'
    fairseq_ckpt_fp = jpath(fairseq_resource_root, 'model_1b.pt')
    dict_fp = jpath(fairseq_resource_root, 'dict_1b/dict.txt')
    args_fp = jpath(fairseq_resource_root, 'task_args_1b.json')

    huggingface_resource_root = '/data2/longshen/Checkpoints/musecoco/transformers/'
    pytorch_dump_folder_path = jpath(huggingface_resource_root, 'temp')
    create_dir_if_not_exist(pytorch_dump_folder_path)

    dump_vocab_file_from_fairseq(
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
        pytorch_dump_folder_path=pytorch_dump_folder_path
    )

    vocab_fp = jpath(pytorch_dump_folder_path, "vocab.json")
    tk_save_dir = jpath(huggingface_resource_root, '1b/tokenizer')
    
    validate_tokenizer(
        vocab_fp=vocab_fp, 
        tk_save_dir=tk_save_dir, 
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )

    validate_tokenizer_loading_local(
        tk_dir=tk_save_dir,
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )

def get_dict_from_fairseq(
        # fairseq_ckpt_fp="/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large.pt",
        # dict_fp='/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/convert_original_model/large/3-dict-3241.txt',
        # args_fp='/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/convert_original_model/large/large_args.json',
        fairseq_ckpt_fp,
        dict_fp,
        args_fp
):
    musecoco, ori_dict = load_fairseq_checkpoint(fairseq_ckpt_fp, dict_fp, args_fp)
    del musecoco
    return ori_dict

def dump_vocab_file_from_fairseq(
        fairseq_ckpt_fp,
        dict_fp,
        args_fp,
        pytorch_dump_folder_path,
):
    # Original tokenizer
    musecoco, ori_dict = load_fairseq_checkpoint(fairseq_ckpt_fp, dict_fp, args_fp)

    hf_dict = ori_dict.indices
    vocab_size = len(hf_dict) # 3245
    vocab_file = jpath(pytorch_dump_folder_path, "vocab.json")
    print(f"Generating {vocab_file}")

    import json
    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(hf_dict, ensure_ascii=False, indent=4))


def validate_tokenizer(vocab_fp, tk_save_dir, fairseq_ckpt_fp, dict_fp, args_fp):
    '''
    Initialize HF tokenzier with vocab file, check correctness
    Then save the tokenzer by save_pretrained()
    '''
    # Hugging Face tokenizer
    tk = MuseCocoTokenizer(
        vocab_file=vocab_fp,
        merges_file=None,
    )  # Also 3245 vocab size. Good job.
    tokens_hf = tk(SAMPLE_TEXT, add_special_tokens=True, return_tensors='pt')['input_ids']

    # Ensure the tokenizer match with each other
    ori_dict = get_dict_from_fairseq(
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )
    tokens_fs = ori_dict.encode_line(SAMPLE_TEXT).long().unsqueeze(0)

    assert torch.eq(tokens_fs, tokens_hf).all()

    detok = tk.batch_decode(tokens_hf)
    assert detok == [SAMPLE_TEXT + ' </s>']

    # Save to local file
    save_dir = tk_save_dir
    tk.save_pretrained(save_dir)
    
def validate_tokenizer_loading_local(tk_dir, fairseq_ckpt_fp, dict_fp, args_fp):
    '''
    Initialize HF tokenzier with from_pretrained, check correctness
    '''
    # Hugging Face tokenizer
    tk = MuseCocoTokenizer.from_pretrained(tk_dir)
    tokens_hf = tk(SAMPLE_TEXT, add_special_tokens=True, return_tensors='pt')['input_ids']

    # Ensure the tokenizer match with each other
    ori_dict = get_dict_from_fairseq(
        fairseq_ckpt_fp=fairseq_ckpt_fp,
        dict_fp=dict_fp,
        args_fp=args_fp,
    )
    tokens_fs = ori_dict.encode_line(SAMPLE_TEXT).long().unsqueeze(0)

    assert torch.eq(tokens_fs, tokens_hf).all()

    detok = tk.batch_decode(tokens_hf)
    assert detok == [SAMPLE_TEXT + ' </s>']


if __name__ == '__main__':
    main()