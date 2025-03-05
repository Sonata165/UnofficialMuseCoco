'''
Get statistics of the data
'''
import os
import sys

sys.path.append('..')

from utils import jpath, save_json, create_dir_if_not_exist, read_json, ls
from utils_midi import remi_utils
from utils_instrument.inst_map import InstMapUtil
from tqdm import tqdm
import pretty_midi

def main():
    # convert_inst_id_to_name()
    observe_multiple_same_type_instruments()

def procedures():
    get_instrument_statistics()
    convert_inst_id_to_name()

def observe_multiple_same_type_instruments():
    '''
    Observe the frequency of there are multiple instrument tracks that contains same type of instruments
    '''
    train_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/train'
    tracks = ls(train_dir)

    res = {
        'has_inst_collision': 0,
        'no_inst_collision': 0,
    }

    inst_util = InstMapUtil()
    for track in tqdm(tracks):
        track_dir = jpath(train_dir, track)
        midi_fp = jpath(track_dir, 'all_src.mid')
        midi = pretty_midi.PrettyMIDI(midi_fp)
        insts = midi.instruments

        # Get all instrument program ids
        inst_programs = [inst.program for inst in insts]
        
        # Determine if there are multiple same type instruments
        inst_types = [inst_util.slakh_get_inst_name_from_prettymidi_inst(inst) for inst in insts]
        # Remove 'unsupported' instruments
        inst_types = [inst for inst in inst_types if inst != 'unsupported']

        # Observe if there are multiple same type instruments
        has_inst_collision = False
        for inst_type in inst_types:
            if inst_types.count(inst_type) > 1:
                has_inst_collision = True
                break

        if has_inst_collision:
            res['has_inst_collision'] += 1
        else:
            res['no_inst_collision'] += 1
    
    res_dir = jpath(train_dir, 'statistics')
    create_dir_if_not_exist(res_dir)
    save_fp = jpath(res_dir, 'inst_collision_cnt.json')
    save_json(res, save_fp)


def convert_inst_id_to_name():
    '''
    Convert instrument id to instrument name
    '''
    inst_util = InstMapUtil()
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/slakh_segmented_2bar_sss'
    for split in ['valid', 'test', 'train']:
        statistics_fp = jpath(dataset_dir, 'statistics', split, 'n_note_of_track.json')
        statistics = read_json(statistics_fp)
        new_statistics = {}
        for inst_tok, n_note in statistics.items():
            if n_note == 0:
                continue
            slakh_id, inst_name = inst_util.slakh_from_midi_program_get_id_and_inst(int(inst_tok.split('-')[1]))
            new_statistics[inst_name] = (n_note, inst_tok)

        # Sort the new_statistics by n_note
        new_statistics = dict(sorted(new_statistics.items(), key=lambda item: item[1][0], reverse=True))

        save_fn = 'n_note_of_track_inst_name.json'
        save_fp = jpath(dataset_dir, 'statistics', split, save_fn)
        save_json(new_statistics, save_fp)


def get_instrument_statistics():
    '''
    Observe all notes in the dataset
    Count the instrument distribution
    '''
    dataset_dir = '/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/slakh_segmented_2bar_sss'
    save_dir = 'statistics'
    save_fp = jpath(dataset_dir, save_dir)

    splits = ['valid', 'test', 'train']
    for split in splits:
        split_fp = jpath(dataset_dir, '{}.txt'.format(split))
        with open(split_fp, 'r') as f:
            data = f.readlines()
        data = [l.strip().split(' ') for l in data]
        
        note_cnt_of_track = {}
        for inst_id in range(129):
            note_cnt_of_track['i-{}'.format(inst_id)] = 0

        for remi_seq in tqdm(data):
            pitch_of_track = remi_utils.from_remi_get_pitch_seq_per_track(remi_seq)
            for track, pitch_seq in pitch_of_track.items():
                if track not in note_cnt_of_track:
                    note_cnt_of_track[track] = 0
                note_cnt_of_track[track] += len(pitch_seq)
        print(note_cnt_of_track)

        save_dir = jpath(save_fp, split)
        create_dir_if_not_exist(save_dir)
        n_note_of_track_fp = jpath(save_dir, 'n_note_of_track.json')
        save_json(note_cnt_of_track, n_note_of_track_fp)

        

if __name__ == '__main__':
    main()