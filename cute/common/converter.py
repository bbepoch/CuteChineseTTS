################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.mapper import pinyin_mapper
from cute.common.mapper import ph_dur_mapper
import numpy as np


def separate_phoneme_tone(pinyin_str):
    """
    separate phoneme and tone from given pinyin string. tone is set to 0 if there is not any
    :param pinyin_str: pinyin string
    :return:
        phoneme(str) and tone(int)
    example:
        print(separate_phoneme_tone('wan3')) => ('wan', 3)
        print(separate_phoneme_tone('ai2')) => ('ai', 2)
        print(separate_phoneme_tone('h')) => ('h', 0)
    """
    if pinyin_str[-1].isdigit():
        phoneme = pinyin_str[0:-1]
        tone = int(pinyin_str[-1])
    else:
        phoneme = pinyin_str
        tone = 0
    return phoneme, tone


def pinyin_to_phoneme_list(pinyin_str, pinyin_parser):
    """
    convert input pinyin string to a list of phoneme strings
    :param pinyin_str: pinyin string
    :param pinyin_parser: a map from pinyin to phoneme pair: (consonant, vowel)
    :return:
        a list of phoneme strings
    example:
        input: 'ka2 er2 pu3 pei2 wai4 sun1 , wan2 hua2 ti1'
        output:['k','a2','er2','p','u3','p','ei2','uai4','s','uen1','sp','uan2','h','ua2','t','i1']
    """
    phoneme_list = []
    for py in pinyin_str.split():
        if py in [',', '.']:
            phoneme_list.append('sp1')
        else:
            ph, tone = separate_phoneme_tone(py)  # sun1 => sun, 1
            assert ph in pinyin_parser, '%s not found.' % ph
            k1, k2 = pinyin_parser[ph]  # sun => s, uen
            if k2 == '':
                phoneme_list.append(k1 + str(tone))
            else:
                phoneme_list.append(k1)
                phoneme_list.append(k2 + str(tone))
    return phoneme_list


def eval_phoneme_duration(phoneme_list, phoneme_dur_parser):
    """
    obtain duration for each phoneme in the phoneme_list
    :param phoneme_list: a list of phoneme strings
    :param phoneme_dur_parser: a map from phoneme to its duration
    :return:
        a list of (phoneme, duration) pairs
    example:
        input: ['k', 'a2', 'er2', 'p', 'u3', 'p', 'ei2', 'uai4']
        output:[('k', 18), ('a2', 32), ('er2', 40), ('p', 19), ('u3', 29), ('p', 19), ('ei2', 25), ('uai4', 38)]
    """
    phoneme_duration_list = []
    for ph in phoneme_list:
        assert ph in phoneme_dur_parser['duration'], '%s not found.' % ph
        ph_dur = (ph, int(phoneme_dur_parser['duration'][ph][1]))
        phoneme_duration_list.append(ph_dur)
    return phoneme_duration_list


def phoneme_to_feature_matrix(phoneme_dur_list, phoneme_dur_parser, feat_type='two_hot', min_val=0.01, max_val=0.99):
    """
    convert a list of (phoneme, duration) pairs into a feature matrix for nn input
    :param phoneme_dur_list: a list of (phoneme, duration) pairs
    :param phoneme_dur_parser: maps containing phoneme to index and tone to index
    :param feat_type: how to construct the feature: 'one_hot' or 'two_hot'
    :param min_val: minimum for a one-hot vector
    :param max_val: maximum for a one-hot vector
    :return:
        feature matrix of shape (num_frame, 260) for one-hot or (num_frame, 93) for two-hot
    """
    assert feat_type in ['one_hot', 'two_hot']
    phoneme_list = []
    for ph in phoneme_dur_list:
        for k in range(ph[1]):
            phoneme_list.append(ph[0])
    num_frame = len(phoneme_list)
    # construct feature matrix
    if feat_type == 'one_hot':
        dim_feats = len(phoneme_dur_parser['duration'])
        lab_feats = np.ones(shape=(num_frame, dim_feats), dtype=np.float32) * min_val
        for k in range(num_frame):
            lab_feats[k][phoneme_dur_parser['duration'][phoneme_list[k]][0]] = max_val
    else:
        dim_feats = len(phoneme_dur_parser['phone_set'])+len(phoneme_dur_parser['tone_set'])
        lab_feats = np.ones(shape=(num_frame, dim_feats), dtype=np.float32) * min_val
        for k in range(num_frame):
            ph, tone = separate_phoneme_tone(phoneme_list[k])
            lab_feats[k][phoneme_dur_parser['phone_set'][ph]] = max_val
            lab_feats[k][len(phoneme_dur_parser['phone_set'])+phoneme_dur_parser['tone_set'][str(tone)]] = max_val
    return lab_feats


def pinyin_to_feature_matrix(pinyin_str, show=False):
    phoneme_list = pinyin_to_phoneme_list(pinyin_str=pinyin_str, pinyin_parser=pinyin_mapper)
    phoneme_dur_list = eval_phoneme_duration(phoneme_list=phoneme_list, phoneme_dur_parser=ph_dur_mapper)
    feature_matrix = phoneme_to_feature_matrix(phoneme_dur_list=phoneme_dur_list, phoneme_dur_parser=ph_dur_mapper)
    if show:
        print(phoneme_list)
        print(phoneme_dur_list)
        print(feature_matrix.shape)
    return feature_matrix


def test():
    input_pinyin = 'ka2 er2 pu3 pei2 wai4 sun1 , wan2 hua2 ti1'
    pinyin_to_feature_matrix(pinyin_str=input_pinyin, show=True)


if __name__ == '__main__':
    test()
