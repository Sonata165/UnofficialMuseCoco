'''
Validate the conversion of the model
'''
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import MuseCocoLMHeadModel, MuseCocoTokenizer
from transformers.models.musecoco.convert_original_model.fairseq_utils import load_fairseq_checkpoint

from typing import List, Tuple

torch.backends.cuda.matmul.allow_tf32 = True

def main():
    # test_generate()
    test_generate_from_conditions()

def compare_results():
    model_hf_fp = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_hf_pt'
    tk_fp = '/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/tokenizers/ori_large'
    model_hf = MuseCocoLMHeadModel.from_pretrained(model_hf_fp)
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    model_fs, ori_dict = load_fairseq_checkpoint()
    model_fs.eval()

    SAMPLE_TEXT = "s-9 o-0 t-26 i-52 p-77 d-24 v-20 p-62 d-24 v-20 o-12 t-26 i-52 p-64 d-12 v-20 o-36 t-26 i-52 p-69 d-12 v-20 p-65 d-12 v-20 b-1"
    tokens_hf = tk(SAMPLE_TEXT, return_tensors="pt")['input_ids']

    out_fs = model_fs.generate(tokens_hf)

    output_fs = model_fs(tokens_hf, sep_pos=None)[0]
    output_hf = model_hf(tokens_hf)  # logits

def test_generate():
    model_fp = '/data2/longshen/Checkpoints/musecoco/transformers/1b/model'
    tk_fp = '/data2/longshen/Checkpoints/musecoco/transformers/1b/tokenizer'
    batch_size = 1
    n_batch = 3

    model = MuseCocoLMHeadModel.from_pretrained(
        model_fp,
        # torch_dtype=torch.float16, # may be less accurate
    )
    model.cuda()

    # # Model quantization
    # quantization_config = QuantoConfig(weights="int8")
    # model = MuseCocoLMHeadModel.from_pretrained(
    #     model_fp,
    #     device_map="auto", 
    #     quantization_config=quantization_config,
    # )

    tk = MuseCocoTokenizer.from_pretrained(tk_fp)

    inp = "s-9 o-0 t-26 i-52 p-77 d-24 v-20 p-62 d-24 v-20 o-12 t-26 i-52 p-64 d-12 v-20 o-36 t-26 i-52 p-69 d-12 v-20 p-65 d-12 v-20 b-1"
    batched_inp = [inp for i in range(batch_size)]
    tokens_hf = tk(batched_inp, return_tensors="pt")['input_ids'].cuda() # [1, len]
    inp = torch.cat([tokens_hf[:,-1:], tokens_hf[:,:-1]], dim=1)

    import time
    from tqdm import tqdm

    t0 = time.time()

    generate_kwargs = {
        'min_length': 500,
        'max_length': 700,
        'use_cache': True,
    }

    with torch.no_grad():
        for i in tqdm(range(n_batch)):
            out = model.generate(
                inp,
                pad_token_id=tk.pad_token_id,
                **generate_kwargs
            )
            out_str = tk.batch_decode(out)
            print(out_str)
    
    '''
    Time overhead:
    (no KV cache)
    with fast_transformers: n_batch = 10, 10 batches: 109.09s, about 1s / sample
    with python implemented attention: with n_batch = 10, 25s / batch, 2.5s / sample
    1b model: 
        raw inference
            batch=1: 26.12s / sample, 
            batch=10: 20s/sample
        + tf32
            fast attn
                batch=1: 15s / sample
                batch=5: 10s / sample
                batch=10: 8.9 / sample
                + kv cache
                    batch=1: 18s / sample
                    batch=5: 7s / sample
                    batch=10: 5.6s / sample
            (py attn) 
                batch=1: 30s / sample
                
            py attn v2
                just cannot finish
    '''

    print(time.time()-t0)
    
    

def test_generate_from_conditions():
    inp = MuseCocoInput(
        instruments=['piano'],
        rhythm_danceability='low',
        rhythm_intensity='mid',
        artist=1,
        genre=1,
        bar=16,
        time_signature=(4,4),
        key='major',
        tempo='fast',
        pitch_range=2,
        emotion=None,
        time=None
    )
    inp_seq = inp.to_seq()
    inp_str = ' '.join(inp_seq)
    print(inp_seq)

    # Load model and tokenizer
    model_fp = '/data2/longshen/Checkpoints/musecoco/transformers/1b/model'
    tk_fp = '/data2/longshen/Checkpoints/musecoco/transformers/1b/tokenizer'
    model = MuseCocoLMHeadModel.from_pretrained(
        model_fp,
        # torch_dtype=torch.float16, # output trash if trained in fp32
    )
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)
    model.cuda()

    # Tokenize input
    tokens_hf = tk(inp_str, return_tensors="pt")['input_ids'].cuda() # [1, len]
    inp = torch.cat([tokens_hf[:,-1:], tokens_hf[:,:-1]], dim=1)

    generate_kwargs = {
        # min_length=500,
        'max_length': 700, # 2000
        'use_cache': True,
        # 'do_sample':True,
    }

    out = model.generate(
        inp,
        pad_token_id=tk.pad_token_id,
        **generate_kwargs
    )
    
    out_str = tk.batch_decode(out)
    print(out_str)

class MuseCocoInput:

    key_order = ['I1s2', 'R1', 'R3', 'S2s1', 'S4', 'B1s1', 'TS1s1', 'K1', 'T1s1', 'P4', 'EM1', 'TM1'],

    def __init__(
            self,
            instruments: List[str],
            rhythm_danceability: str,
            rhythm_intensity: str,
            artist: int,
            genre: int,
            bar: int,
            time_signature: Tuple[int, int],
            key: str,
            tempo: str,
            pitch_range: int,
            emotion,
            time,
        ):
        """Construct an input object containing all possible input conditions for MuseCoco

        Args:
            instruments (List[str]): list of instrument want to show up
            rhythm_danceability (str): 'high' | 'low' | 'unk'
            rhythm_intensity (str): 'high' | 'mid' | 'low'
            artist (int): 0~16
            genre (int): 0~21
            bar (int): 1~16
            time_signature (Tuple[int, int]): (4,4) | (2,4) | (3,4) | (1,4) | (6,8) | (3,8)
            key (str): 'major' | 'minor' | 'unk'
            tempo (str): 'slow' | 'mid' | 'fast'
            pitch_range (int): n_octaves, 0~11
            emotion (_type_): _description_
            time (_type_): _description_
        """        
        self.instruments = instruments
        self.rhythm_danceability = rhythm_danceability
        self.rhythm_intensity = rhythm_intensity
        self.artist = artist
        self.genre = genre
        self.bar = bar
        self.time_signature = time_signature
        self.key = key
        self.tempo = tempo
        self.pitch_range = pitch_range
        self.emotion = emotion
        self.time = time

        self.inst_name_to_inst_id = {
            'piano':0,
            'keyboard':1,
            'percussion':2,
            'organ':3,
            'guitar':4,
            'bass':5,
            'violin':6,
            'viola':7,
            'cello':8,
            'harp':9,
            'strings':10,
            'voice':11,
            'trumpet':12,
            'trombone':13,
            'tuba':14,
            'horn':15,
            'brass':16,
            'sax':17,
            'oboe':18,
            'bassoon':19,
            'clarinet':20,
            'piccolo':21,
            'flute':22,
            'pipe':23,
            'synthesizer':24,
            'ethnic instrument':25,
            'sound effect':26,
            'drum':27,
        }
        
        # Validity check
        for inst in instruments:
            assert inst in self.inst_name_to_inst_id
        assert rhythm_danceability in ['high', 'low', 'unk']
        assert rhythm_intensity in ['high', 'mid', 'low']
        assert artist in range(17)
        assert genre in range(22)
        assert bar in range(1, 17)
        assert time_signature in [(4,4), (2,4), (3,4), (1,4), (6,8), (3,8)]
        assert key in ['major', 'minor', 'unk']
        assert tempo in ['slow', 'mid', 'fast']
        assert pitch_range in range(12)

    def to_seq(self, add_sep=True) -> str:
        """Convert the specified condition to a input string for MuseCoco feat-to-music model.

        Returns:
            str: input condition sequence
        """        
        '''
        'I1s2', 'R1', 'R3', 'S2s1', 'S4', 'B1s1', 'TS1s1', 'K1', 'T1s1', 'P4', 'EM1', 'TM1'],
        '''
        ret = []

        # I1s2: instruments
        inst_ids = [self.inst_name_to_inst_id[inst] for inst in self.instruments]
        # tok_inst = ['I1s2_{}_0'.format(i) for i in inst_ids]
        tok_inst = []
        for inst in range(28):
            if inst in inst_ids:
                tok_inst.append('I1s2_{}_0'.format(inst))
            else:
                tok_inst.append('I1s2_{}_1'.format(inst))
        ret.extend(tok_inst)

        # R1: Rhythm danceability
        if self.rhythm_danceability == 'high':
            danceable_tok = 'R1_0'
        elif self.rhythm_danceability == 'low':
            danceable_tok = 'R1_1'
        elif self.rhythm_danceability == 'unk':
            danceable_tok = 'R1_2'
        ret.append(danceable_tok)

        # R3: Rhythm intensity
        if self.rhythm_danceability == 'high':
            intensity_tok = 'R3_2'
        elif self.rhythm_danceability == 'mid':
            intensity_tok = 'R3_1'
        elif self.rhythm_danceability == 'low':
            intensity_tok = 'R3_0'
        ret.append(intensity_tok)

        # S2s1: Artist
        artist_tok = 'S2s1_{}'.format(self.artist)
        ret.append(artist_tok)

        # S4: Genre
        # genre_tok = 'S4_{}_0'.format(self.genre)
        # ret.append(genre_tok)
        genre_tok = []
        for i in range(22):
            if i == self.genre:
                genre_tok.append('S4_{}_0'.format(self.genre))
            else:
                genre_tok.append('S4_{}_1'.format(i))
        ret.extend(genre_tok)

        # B1s1: Num of bars (range)
        if self.bar in range(1, 5):
            bar_id = 0
        elif self.bar in range(5, 9):
            bar_id = 1
        elif self.bar in range(9, 13):
            bar_id = 2
        elif self.bar in range(13, 17):
            bar_id = 3
        
        # # DEBUG: Bar num debug
        # bar_id = 4

        # bar: 4-> 3bars   3 -> 2bars

        bar_tok = 'B1s1_{}'.format(bar_id)
        ret.append(bar_tok)
        
        # TS1s1: Time signature
        ts_map = {
            (4,4): 0,
            (2,4): 1,
            (3,4): 2,
            (1,4): 3,
            (6,8): 4,
            (3,8): 5,
        }
        ts_id = ts_map[self.time_signature]
        ts_tok = 'TS1s1_{}'.format(ts_id)
        ret.append(ts_tok)

        # K1: Major or minor
        key_map = {
            'major': 0,
            'minor': 1,
            'unk': 2,
        }
        key_id = key_map[self.key]
        key_tok = 'K1_{}'.format(key_id)
        ret.append(key_tok)

        # T1s1: Tempo
        tempo_map = {
            'slow': 0,
            'mid': 1,
            'fast': 2,
        }
        tempo_id = tempo_map[self.tempo]
        tempo_tok = 'T1s1_{}'.format(tempo_id)
        ret.append(tempo_tok)

        # P4: Pitch range
        pitch_range_tok = 'P4_{}'.format(self.pitch_range)
        ret.append(pitch_range_tok)

        # EM1: Emotion (but don't know the mapping)
        if self.emotion is not None:
            raise NotImplementedError
        
        # TM1: Output duration in seconds (deprecated)
        if self.time is not None:
            raise NotImplementedError

        # DEBUG: time token
        ret.append('TM1_4')

        # Add the '<sep>' token
        if add_sep:
            ret.append('<sep>')

        return ret

def test_feat_to_inp_seq_converter():
    pass


def test_feat_to_music_gen():
    '''
    Test the ability to generate music from musical descriptors.
    '''
    use_cuda = torch.cuda.is_available()

    # Load model and tokenizer
    model_hf_fp = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/large_hf_pt'
    tk_fp = '/home/longshen/work/StyleTrans/modules/transformers/src/transformers/models/musecoco/tokenizers/ori_large'
    out_dir = '/home/longshen/work/StyleTrans/modules/musecoco/checkpoints/hf_out'

    max_positions = 500

    os.makedirs(out_dir, exist_ok=True)

    # test_command = np.load("../Text2Music_data/v2.1_20230218/full_0218_filter_by_5866/infer_command_balanced.npy",
    #                        allow_pickle=True).item()
    # test_command = np.load(args.ctrl_command_path, allow_pickle=True).item()

    # test_command = json.load(open(args.ctrl_command_path, "r"))
    # test_command -> test_input_list
    if args.use_gold_labels:
        with open(args.save_root + "/Using_gold_labels!.txt", "w") as check_input:
            pass
    else:
        with open(args.save_root + "/Using_pred_labels!.txt", "w") as check_input:
            pass
    test_input_list = pickle.load(open(args.ctrl_command_path, "rb"))
    if args.start is None:
        args.start = 0
        args.end = len(test_input_list)
    else:
        args.start = min(max(args.start, 0), len(test_input_list))
        args.end = min(max(args.end, 0), len(test_input_list))

    gen_command_list = []
    # Inputs are added multiple times, depending on how many times we want to generate for each attribute input
    for gen_time_id in range(args.need_num):
        for sample_id in range(args.start, args.end): # For each sample in testing
            # Construct attribute dict, whose keys are aspects of attributes, values are one hot vectors
            if args.use_gold_labels:
                attr_dict = test_input_list[sample_id]["gold_labels"]
            else:
                attr_dict = test_input_list[sample_id]["pred_labels"]
            # Convert attribute vectors to a list of tokens
            attribute_tokens = convert_vector_to_token(attr_dict)
            # for key in key_order:
            #     if key not in pred_labels.keys():
            #         continue
            #     if key in key_has_NA and pred_labels[key][-1] == 1:
            #         continue
            #     for j in range(len(pred_labels[key])):
            #         if pred_labels[key][j] == 1:
            #             attribute_tokens.append(f"{key}_{j}")

            test_input_list[sample_id]["infer_command_tokens"] = attribute_tokens
            full_attr_info = test_input_list[sample_id]
            gen_command_list.append(
                [attribute_tokens, f"{sample_id}", gen_time_id, full_attr_info]
            )

    num_of_samples = len(gen_command_list)
    steps = num_of_samples // args.batch_size
    # We can infer that items in each bach is different samples (music).
    print(
        f"Starts to generate {args.start} to {args.end} of {len(gen_command_list)} samples in {steps + 1} batch steps!"
    )

    for batch_step in range(steps + 1):
        infer_list = gen_command_list[
                     batch_step * args.batch_size: (batch_step + 1) * args.batch_size
                     ]
        infer_command_token = [g[0] for g in infer_list]
        # assert infer_command.shape[1] == 133, f"error feature dim for {gen_key}!"
        if len(infer_list) == 0:
            continue
        # with open(save_root + f"/{command_index}/text_description.txt", "w") as text_output:
        #     text_output.write(text_description[command_index])

        if os.path.exists(
                save_root + f"/{infer_list[-1][1]}/remi/{infer_list[-1][2]}.txt"
        ):
            print(f"Skip the {batch_step}-th batch since has been generated!")
            continue

        # start_tokens = [f""]
        start_tokens = []
        sep_pos = []
        for attribute_prefix in infer_command_token:
            start_tokens.append(" ".join(attribute_prefix) + " <sep>")
            sep_pos.append(
                len(attribute_prefix)
            )  # notice that <sep> pos is len(attribute_prefix) in this sequence
        sep_pos = np.array(sep_pos)
        for inputs in [start_tokens]:  # "" for none prefix input
            results = []
            for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                bsz = batch.src_tokens.size(0)
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                constraints = batch.constraints

                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()
                    if constraints is not None:
                        constraints = constraints.cuda()

                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                        "sep_pos": sep_pos,
                    },
                }
                translate_start_time = time.time()
                translations = task.inference_step(
                    generator, models, sample, constraints=constraints
                )
                translate_time = time.time() - translate_start_time
                total_translate_time += translate_time
                list_constraints = [[] for _ in range(bsz)]
                if args.constraints:
                    list_constraints = [unpack_constraints(c) for c in constraints]

                for sample_id, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[sample_id], tgt_dict.pad())
                    constraints = list_constraints[sample_id]
                    results.append(
                        (
                            start_id + id,
                            src_tokens_i,
                            hypos,
                            {
                                "constraints": constraints,
                                "time": translate_time / len(translations),
                                "translation_shape": len(translations),
                            },
                        )
                    )

            # sort output to match input order
            for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                # Process top predictions
                for hypo in hypos[: min(len(hypos), args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

                    os.makedirs(save_root + f"/{infer_list[id_][1]}", exist_ok=True)
                    if not os.path.exists(
                            save_root + f"/{infer_list[id_][1]}/infer_command.json"
                    ):
                        with open(
                                save_root + f"/{infer_list[id_][1]}/infer_command.json", "w"
                        ) as f:
                            json.dump(infer_list[id_][-1], f)
                    save_id = infer_list[id_][2]

                    os.makedirs(
                        save_root + f"/{infer_list[id_][1]}/remi", exist_ok=True
                    )
                    with open(
                            save_root + f"/{infer_list[id_][1]}/remi/{save_id}.txt", "w"
                    ) as f:
                        f.write(hypo_str)
                    remi_token = hypo_str.split(" ")[sep_pos[id_] + 1:]
                    print(
                        f"batch:{batch_step} save_id:{save_id} over with length {len(hypo_str.split(' '))}; "
                        f"Average translation time:{info['time']} seconds; Remi seq length: {len(remi_token)}; Batch size:{args.batch_size}; \
                          Translation shape:{info['translation_shape']}."
                    )
                    os.makedirs(
                        save_root + f"/{infer_list[id_][1]}/midi", exist_ok=True
                    )
                    try:
                        midi_obj = midi_decoder.decode_from_token_str_list(remi_token)
                        midi_obj.dump(
                            save_root + f"/{infer_list[id_][1]}/midi/{save_id}.mid"
                        )
                    except:
                        print(traceback.format_exc())
                        pass

if __name__ == '__main__':
    main()