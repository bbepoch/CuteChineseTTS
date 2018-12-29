################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.converter import phoneme_to_feature_matrix
from cute.common.parameter import parameter_manager
import pickle
import json
import os


def generate_lab_file(phoneme_set_path, phoneme_dur_path, lab_files_folder):
    """
    generate binary label files from phoneme duration files
    :param phoneme_set_path: file containing phoneme set
    :param phoneme_dur_path: file containing phoneme durations
    :param lab_files_folder: folder to save output binary label files
    :return: none
    """
    assert os.path.exists(phoneme_set_path), phoneme_set_path
    assert os.path.exists(phoneme_dur_path), phoneme_dur_path
    assert os.path.exists(lab_files_folder), lab_files_folder
    ph_set = json.load(open(phoneme_set_path, 'r'))
    ph_dur_list = pickle.load(open(phoneme_dur_path, 'rb'))
    for i in range(len(ph_dur_list)):
        file_name = os.path.join(lab_files_folder, ph_dur_list[i][0]+'.lab')
        lab = phoneme_to_feature_matrix(phoneme_dur_list=ph_dur_list[i][1:], phoneme_dur_parser=ph_set)
        lab.tofile(file_name)
        print(file_name, lab.shape)


def main():
    phoneme_set_path = parameter_manager.phoneme_set_path
    phoneme_dur_path = parameter_manager.phoneme_dur_path
    lab_files_folder = parameter_manager.lab_files_folder
    generate_lab_file(phoneme_set_path=phoneme_set_path, phoneme_dur_path=phoneme_dur_path,
                      lab_files_folder=lab_files_folder)


if __name__ == '__main__':
    main()
