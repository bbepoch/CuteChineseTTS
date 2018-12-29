################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.converter import separate_phoneme_tone
from cute.common.parameter import parameter_manager
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import pickle
import json
import os


def obtain_phoneme_set(unique_ph_dur_list):
    """
    obtain (phoneme, duration) map, (pure phone, index) map and (tone, index) map
    :param unique_ph_dur_list: a list of tuples, each tuple' pattern is (phoneme, duration mean, duration variance)
    :return:
        a dict contains the desired three maps
    """
    ph_tone_dur_dict = OrderedDict()
    tmp_ph_dict, tmp_tone_dict = dict(), dict()
    pure_ph_dict, pure_tone_dict = OrderedDict(), OrderedDict()
    unique_ph_dur_list.sort()
    for i, k in enumerate(unique_ph_dur_list):
        # k => (phoneme, duration mean, duration variance)
        ph_tone_dur_dict[k[0]] = (i, k[1], k[2])
        ph, tone = separate_phoneme_tone(k[0])
        tmp_ph_dict[ph] = 0
        tmp_tone_dict[tone] = 0
    for i, k in enumerate(sorted(tmp_ph_dict.keys())):
        # k => pure phone
        pure_ph_dict[k] = i
    for i, k in enumerate(sorted(tmp_tone_dict.keys())):
        # k => tone
        pure_tone_dict[k] = i
    return {'phone_set': pure_ph_dict, 'tone_set': pure_tone_dict, 'duration': ph_tone_dur_dict}


def parse_interval_files(interval_file_list):
    """
    extract useful information (e.g. phonemes and their durations) from interval files
    :param interval_file_list: a list of interval file path
    :return:
        ph_dur_list_all: a list of list of (phoneme, duration) tuples
        ph_dur_map: a dict of (phoneme, duration list)
    """
    ph_dur_list_all = []
    ph_dur_map = defaultdict(list)
    for interval_file in interval_file_list:
        assert os.path.exists(interval_file), interval_file
        with open(interval_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = lines[lines.index('<exists>')+4:]
            st, et, num = float(lines[0]), float(lines[1]), int(lines[2])
            # print(st, et, num, len(lines))
            assert (num + 1) * 3 == len(lines)
            # add file name first
            ph_dur_list = [interval_file.split('/')[-1].split('.')[0]]
            for idx in range(1, num+1):
                st = round(float(lines[3 * idx]) * 1000)
                et = round(float(lines[3 * idx + 1]) * 1000)
                fm = round(et / 5) - round(st / 5)
                ph = lines[3 * idx + 2].replace('\"', '')
                # print(st, et, ph, fm)
                ph_dur_list.append((ph, fm))
                ph_dur_map[ph].append(fm)
            print(ph_dur_list)
            ph_dur_list_all.append(ph_dur_list)
    return ph_dur_list_all, ph_dur_map


def main():
    intervals_folder = parameter_manager.intervals_folder
    phoneme_dur_path = parameter_manager.phoneme_dur_path
    phoneme_set_path = parameter_manager.phoneme_set_path

    interval_file_list = [os.path.join(intervals_folder, '%06d.interval' % (i+1)) for i in range(10000)]

    ph_dur_list_all, ph_dur_map = parse_interval_files(interval_file_list=interval_file_list)
    unique_ph_dur_list = [(k, np.array(v).mean(), np.array(v).std()) for k, v in ph_dur_map.items()]
    phoneme_set = obtain_phoneme_set(unique_ph_dur_list=unique_ph_dur_list)

    with open(phoneme_dur_path, 'wb') as f:
        pickle.dump(ph_dur_list_all, f)

    with open(phoneme_set_path, 'w') as f:
        json.dump(phoneme_set, f, indent=4)


if __name__ == '__main__':
    main()
