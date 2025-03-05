import os
import sys
sys.path.append('..')

from hf_musecoco.modeling_musecoco import MuseCocoLMHeadModel
from hf_musecoco.tokenization_musecoco import MuseCocoTokenizer

model_hf_fp = '/data1/longshen/Results/musecoco_data/pretrained_models/1b/model'
tk_fp = '/data1/longshen/Results/musecoco_data/pretrained_models/1b/tokenizer'
model_hf = MuseCocoLMHeadModel.from_pretrained(model_hf_fp)
tk = MuseCocoTokenizer.from_pretrained(tk_fp)

SAMPLE_TEXT = "s-9 o-0 t-26 i-52 p-77 d-24 v-20 p-62 d-24 v-20 o-12 t-26 i-52 p-64 d-12 v-20 o-36 t-26 i-52 p-69 d-12 v-20 p-65 d-12 v-20 b-1"
tokens_hf = tk(SAMPLE_TEXT, return_tensors="pt")['input_ids']
output_hf = model_hf(tokens_hf)  # logits

# Note: hugging face version only support generating for one sample at a time (no batch generation)
# Note2: 实现的时候为了方便，positional encoding的计算方式和fairseq有所不同。 
# huggingface version没有给prefix计算positional encoding，但是原本的fairseq版本是给prefix计算positional encoding的。
# 但可以保证除此之外，huggingface version和fairseq version的forward pass完全一致

# To convert midi <-> tokens:
from midi_utils.utils_midi import RemiTokenizer
midi_tok = RemiTokenizer()
midi_tok.midi_to_remi(...)
midi_tok.remi_to_midi(...)

# Original musecoco
midi_to_remi(
    midi_path, 
    normalize_pitch=True, 
    return_pitch_shift=False, 
    return_key=False,
    reorder_by_inst=False, # notes are strictly sorted by positions
    include_ts=True,
    include_tempo=True,
    include_velocity=True,
)

# Finetune的时候time signature, tempo, velocity可以不需要，根据你的task决定
# 另外，如果task和multitrack有关，可以试一试reorder_by_inst=True，
# 这样可以保证每个每个小节内每个instrument的音符是连贯的，可能会有用

# 上次我投AAAI的用法
midi_to_remi(
    midi_path, 
    normalize_pitch=True, 
    return_pitch_shift=False, 
    return_key=False,
    reorder_by_inst=True,
    include_ts=False,
    include_tempo=False,
    include_velocity=False,
)
