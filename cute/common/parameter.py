################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


import os


class ParameterManager:
    def __init__(self):
        # home folder
        self.home_path = os.environ['HOME']
        # root for baker open sourced data
        self.data_root_path = os.path.join(self.home_path, 'DATA/audio/baker/')
        # speech tools path: world and sptk
        # maybe you should download and compile them first
        self.speech_tools_root_folder = os.path.join(self.home_path, 'program/speech-tools/')
        self.num_acos_to_norm = 4000
        self.num_train = 3200
        self.num_valid = 798

        # folder containing interval files
        self.intervals_folder = os.path.join(self.data_root_path, 'PhoneLabeling/')
        # file generated from all interval files containing phoneme set
        # used to quantize lab features
        self.phoneme_set_path = os.path.join(self.data_root_path, 'ph_set.json')
        # file generated from all interval files containing phoneme durations
        # used to generate training lab files
        self.phoneme_dur_path = os.path.join(self.data_root_path, 'ph_dur.pkl')
        # folder containing quantized binary lab files for training
        # would be created automatically
        self.lab_files_folder = os.path.join(self.data_root_path, 'Lab/')

        # folder containing wave files
        self.wav_audio_folder = os.path.join(self.data_root_path, '48k/wav/')
        # folder containing extracted acoustic features and training logs
        self.aco_feats_folder = os.path.join(self.data_root_path, '48k/features/')
        self.norm_file_folder = os.path.join(self.data_root_path, '48k/features/norm/')
        self.norm_file_sheets = os.path.join(self.data_root_path, '48k/features/norm/norm_file_sheets.txt')
        self.train_log_folder = os.path.join(self.data_root_path, '48k/features/log/')
        self.saved_ckp_prefix = os.path.join(self.data_root_path, '48k/features/log/model_')
        self.training_los_log = os.path.join(self.data_root_path, '48k/features/log/training.log')

        # world and sptk 's binaries folder
        self.bin_world_path = os.path.join(self.speech_tools_root_folder, 'bin/WORLD/')
        self.bin_sp_tk_path = os.path.join(self.speech_tools_root_folder, 'bin/SPTK-3.9/')

        if not os.path.exists(self.lab_files_folder):
            os.mkdir(self.lab_files_folder)

        if not os.path.exists(self.aco_feats_folder):
            os.mkdir(self.aco_feats_folder)

        if not os.path.exists(self.train_log_folder):
            os.mkdir(self.train_log_folder)


parameter_manager = ParameterManager()
