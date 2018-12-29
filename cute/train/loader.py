################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################
# Source: Modified from open source code
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


import numpy as np
import random


class DataLoader(object):
    def __init__(self, file_list, dim_x, dim_y, batch_size=4, shuffle=False):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        assert len(file_list) > 0
        self.file_list = file_list
        self.list_size = len(self.file_list)
        self.file_index = 0
        self.end_reading = False
        if shuffle:
            random.seed(123678)
            random.shuffle(self.file_list)

    def __iter__(self):
        return self

    def reset(self):
        self.file_index = 0
        self.end_reading = False

    def is_finish(self):
        return self.end_reading

    def load_one_batch(self):
        x_list, y_list = [], []
        for i in range(self.batch_size):
            feat = np.fromfile(self.file_list[self.file_index], dtype=np.float32).reshape(-1, self.dim_x+self.dim_y)
            x_data = feat[:, 0:self.dim_x]
            y_data = feat[:, self.dim_x:self.dim_x+self.dim_y]
            x_list.append(x_data)
            y_list.append(y_data)
            self.file_index += 1
        if self.file_index + self.batch_size > self.list_size:
            self.file_index = 0
            self.end_reading = True
        return x_list, y_list
