################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.parameter import parameter_manager
import os


# speech tools root folder: world and sptk
# maybe you should download and compile them first
speech_tools_dir = parameter_manager.speech_tools_root_folder
# folder containing wave files
wav_dir = parameter_manager.wav_audio_folder
# folder to save extracted acoustic features
out_dir = parameter_manager.aco_feats_folder
assert os.path.exists(speech_tools_dir), speech_tools_dir
assert os.path.exists(wav_dir), wav_dir
assert os.path.exists(out_dir), out_dir


command = 'bash feat_extractor.sh %s %s %s' % (speech_tools_dir[0:-1], wav_dir[0:-1], out_dir[0:-1])
os.system(command=command)
